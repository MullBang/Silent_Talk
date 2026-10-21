"""Frozen-checkpoint recording probe: 18 new and 18 reused rows per model.

This evaluator opens only six selected student input caches. Step12 training
identity is verified from frozen metadata and complete checkpoint state; no
teacher cache, audio, optimizer update, or new checkpoint selection is needed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR_PATH = ROOT / 'experiments/multisentence_step12_20261007/evaluate.py'
PRIOR_SHA = '895c9e8e71807b239ec288d7846feaf7484eeaf4f4d2f896ba6a7fe53da664dd'
if hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest() != PRIOR_SHA:
    raise ValueError('Frozen Step12 evaluator changed')
spec = importlib.util.spec_from_file_location('step13_frozen_step12_eval', PRIOR_PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
decoder, original, runner, core = prior.decoder, prior.original, prior.runner, prior.core
require, bind, verify = prior.require, prior.bind, prior.verify
CONDITIONS = ('normal', 'static', 'shuffled')
SAME_GROUPS = {'same6': 6, 'same_old4': 4, 'same_new2': 2}
DIFFERENT_GROUPS = {'different6': 6, 'different_old4': 4, 'different_new2': 2}
ROLE = 'same_recording_holdout'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evaluation_code():
    refs = [bind(__file__), bind(BASE / 'test_evaluate.py'), *prior.evaluation_code()]
    return list({ref['path']: ref for ref in refs}.values())


def evaluation_contract():
    return {'schema': 'lipnet-step13-recording-probe-evaluation-contract-v1',
            'checkpoint_update': 3600, 'conditions': list(CONDITIONS),
            'same_groups': SAME_GROUPS, 'different_groups': DIFFERENT_GROUPS,
            'new_inference_rows_per_model': 18, 'reused_inference_rows_per_model': 18,
            'new_inference_rows_total': 72, 'reused_inference_rows_total': 72,
            'reused_records': 'Exact unmodified Step12 final row values; only aggregation changes',
            'new_role': ROLE, 'original_source_split': 'train',
            'decoder': prior.evaluation_contract()['decoder'],
            'primary_metric': prior.evaluation_contract()['primary_metric'],
            'ctc': prior.evaluation_contract()['ctc'],
            'failure_policy': prior.evaluation_contract()['failure_policy'],
            'posterior_evidence': prior.evaluation_contract()['posterior_evidence'],
            'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
            'teacher_feature_caches_loaded': 0, 'student_input_caches_loaded': 6,
            'scope': 'Shared speakers; same versus different recording with different sentence contents. Not a causal recording-only experiment or untouched test.'}


def validate_same_clips(clips, groups):
    require(set(clips) == set(groups['same6']) and len(clips) == 6, 'Changed same-recording cohort')
    for name, size in SAME_GROUPS.items():
        require(len(groups[name]) == len(set(groups[name])) == size, 'Changed group: ' + name)
    require(set(groups['same_old4']).isdisjoint(groups['same_new2'])
            and set(groups['same_old4'] + groups['same_new2']) == set(clips), 'Invalid same-recording strata')
    speakers = set()
    for sid, clip in clips.items():
        require(sid == clip['id'] and clip['split'] == clip['diagnostic_role'] == ROLE
                and clip['source_split'] == 'train', 'Wrong same-recording diagnostic/source role')
        require('teacher_features' not in clip and 'audio' not in clip, 'Forbidden inference modality')
        pixels, tokens = clip['pixels'], clip['tokens']
        require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
                and pixels.shape[1:] == (96, 96, 3) and len(pixels) > 0, 'Invalid student pixels')
        require(isinstance(tokens, list) and tokens and all(type(t) is int and 0 < t < 69 for t in tokens)
                and tokens == decoder.tokenizer.encode_text(clip['text'])
                and len(pixels) >= core.ctc_min_input_len(tokens), 'Invalid CTC target')
        speakers.add(clip['speaker_id'])
    require(len(speakers) == 6, 'Expected one utterance per selected speaker')


def evaluate_clips(model, clips, groups, device, posteriors):
    validate_same_clips(clips, groups)
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Student must be frozen eval')
    before = runner.nested_hash(model.state_dict())
    values = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            records = []
            for position, sid in enumerate(groups['same6']):
                clip = clips[sid]
                started = time.perf_counter()
                decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                ctc, diagnostic, error, temporal, key = None, None, None, None, None
                try:
                    pixels = clip['pixels']
                    temporal = original.condition_metadata(pixels, condition, sid)
                    lp = model(core.prepare_student(core.perturb_pixels(pixels, condition, sid), device))
                    require(lp.shape == (len(pixels), 1, 69) and torch.isfinite(lp).all().item(), 'Malformed/nonfinite model output')
                    lp_cpu = lp.float().cpu()
                    array = lp_cpu[:, 0].numpy().copy()
                    candidate = decoder.decode_prediction(array)
                    diagnostic = prior.prior.output_diagnostics(array, candidate, clip['text'])
                    ctc = float(F.ctc_loss(lp_cpu, torch.tensor(clip['tokens'], dtype=torch.long),
                                torch.tensor([len(pixels)]), torch.tensor([len(clip['tokens'])]),
                                blank=0, reduction='mean', zero_infinity=False))
                    require(np.isfinite(ctc) and ctc >= -1e-6, 'Nonfinite/negative CTC evaluation loss')
                    decoded = candidate
                    key = f'u3600_{condition}_{position}'
                    require(key not in posteriors, 'Duplicate posterior key')
                    posteriors[key] = array
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                    ctc, diagnostic, key = None, None, None
                token_edits = decoder.diagnostics.edit_distance(clip['tokens'], decoded['collapsed_ids'])
                stratum = 'same_old4' if sid in groups['same_old4'] else 'same_new2'
                records.append({'id': sid, 'speaker_id': clip['speaker_id'], 'source_stem': clip['source_stem'],
                    'split': ROLE, 'source_split': 'train', 'diagnostic_role': ROLE,
                    'group': stratum, 'condition': condition, 'reference': clip['text'], 'prediction': decoded['prediction'],
                    'status': 'ok' if error is None else 'failed', 'error': error, 'frames': len(clip['pixels']),
                    'temporal': temporal, 'seconds': time.perf_counter() - started,
                    'raw_argmax_ids': decoded['raw_argmax_ids'], 'collapsed_ids': decoded['collapsed_ids'],
                    'target_token_ids': clip['tokens'], 'replacement_characters': decoded['replacement_characters'],
                    'strict_label_decoder_diagnostic': decoded['strict_replay'],
                    'metrics': core.counts(clip['text'], decoded['prediction']), 'ctc_loss_mean': ctc,
                    'token_metrics': {'token_edits': token_edits, 'target_tokens': len(clip['tokens']),
                                      'exact': clip['tokens'] == decoded['collapsed_ids']},
                    'output_diagnostics': diagnostic, 'log_posteriors_key': key})
            values[condition] = {'records': records, 'groups': {
                name: prior.prior.summarize_records([row for row in records if row['id'] in groups[name]]) for name in SAME_GROUPS}}
    after = runner.nested_hash(model.state_dict())
    require(before == after, 'Evaluation changed student state')
    return {'conditions': values, 'model_state_before': before, 'model_state_after': after, 'model_state_unchanged': True}


def validate_reused_rows(source, plan, checkpoint, selection, archive):
    """Validate and return exact old rows; do not rewrite their original roles."""
    require(source['schema'] == 'lipnet-step12-multisentence-evaluation-v1'
            and source['status'] == 'completed' and source['seed'] == plan['seed']
            and source['arm'] == checkpoint['arm']
            and source['plan_sha256'] == plan['plan_sha256'], 'Wrong reused evaluation provenance')
    require(source['evaluation_data_sha256'] == plan['evaluation_data']['data_sha256']
            and source['data_sha256'] == plan['training_data']['data_sha256'], 'Wrong reused evaluation data')
    require(source['initial_state_hashes_verified'] == plan['initial_state_hashes_expected'][source['arm']], 'Wrong reused initial state')
    finals = [item for item in source['milestones'] if item['completed_updates'] == 3600]
    require(len(finals) == 1, 'Expected one original final checkpoint evaluation')
    final = finals[0]
    require(final['checkpoint'] == checkpoint['reference'] and final['model_state_unchanged']
            and final['model_state_before'] == final['model_state_after'] == checkpoint['reference']['state_hashes']['model'],
            'Reused checkpoint/model state mismatch')
    groups = selection['groups']
    for name, size in DIFFERENT_GROUPS.items():
        require(len(groups[name]) == len(set(groups[name])) == size, 'Changed different-recording group')
    require(set(groups['different_old4']).isdisjoint(groups['different_new2'])
            and set(groups['different6']) == set(groups['different_old4'] + groups['different_new2']), 'Invalid reused strata')
    values = {}
    for condition in CONDITIONS:
        original_rows = final['conditions'][condition]['records']
        require(len(original_rows) == len({row['id'] for row in original_rows}) == 32, 'Changed original row cardinality')
        lookup = {row['id']: row for row in original_rows}
        records = [copy.deepcopy(lookup[sid]) for sid in groups['different6']]
        for row in records:
            sid = row['id']
            bound = plan['evaluation_data']['clips'][sid]
            require(sid in plan['diagnostic_groups']['hold8'] and row['split'] == row['diagnostic_role'] == 'internal_holdout'
                    and row['source_split'] == 'train' and row['condition'] == condition, 'Wrong reused role/condition')
            require(row['speaker_id'] == bound['speaker_id'] and row['source_stem'] == bound['source_stem']
                    and row['reference'] == bound['text'] and row['frames'] == bound['frames']
                    and runner.nested_hash(row['target_token_ids']) == bound['tokens_sha256'], 'Changed reused row identity')
            require(row['metrics'] == core.counts(row['reference'], row['prediction']), 'Changed reused metrics')
            if row['status'] == 'ok':
                lp = archive[row['log_posteriors_key']]
                require(lp.dtype == np.float32 and lp.shape == (row['frames'], 69) and np.isfinite(lp).all(), 'Invalid reused posterior')
                decoded = decoder.decode_prediction(lp)
                require(all(row[key] == decoded[key] for key in ('prediction', 'raw_argmax_ids', 'collapsed_ids', 'replacement_characters')),
                        'Reused posterior decoding mismatch')
                diagnostics = prior.prior.output_diagnostics(lp, decoded, row['reference'])
                require(row['output_diagnostics'] == diagnostics, 'Reused output diagnostics mismatch')
            else:
                require(row['status'] == 'failed' and row['prediction'] == '' and row['ctc_loss_mean'] is None
                        and row['log_posteriors_key'] is None, 'Invalid failed reused row')
        values[condition] = {'records': records, 'groups': {
            name: prior.prior.summarize_records([row for row in records if row['id'] in groups[name]]) for name in DIFFERENT_GROUPS},
            'reused': True}
    return values


def validate_checkpoint(model_spec, plan):
    """Read complete CPU checkpoint and its original plan without teacher data."""
    for name in ('plan', 'run_result', 'prior_evaluation', 'prior_posteriors'):
        verify(model_spec[name])
    for ref in plan['execution_code']:
        verify(ref)
    ref = model_spec['checkpoint']
    require(ref['completed_updates'] == 3600 and Path(ref['sidecar']).resolve() == Path(ref['path']).resolve().with_suffix('.pt.json'),
            'Wrong final checkpoint reference')
    require(bind(ref['path'])['sha256'] == ref['sha256'] and bind(ref['sidecar'])['sha256'] == ref['sidecar_sha256'], 'Checkpoint SHA changed')
    run = read(model_spec['run_result']['path'])
    require(run['status'] == 'completed' and run['seed'] == model_spec['seed'] and run['arm'] == model_spec['arm']
            and run['completed_updates'] == 3600 and run['start_update'] == 0 and run['restored_from'] is None
            and run['checkpoints'][-1] == ref and run['plan_file'] == model_spec['plan'], 'Changed original training completion')
    payload = runner.load_checkpoint(Path(ref['path']))
    validation = {'plan_sha256': plan['plan_sha256'], **plan['training_data']}
    expected = prior.validate_checkpoint_payload(plan, validation, model_spec['arm'], payload, 3600,
                                                 plan['initial_state_hashes_expected'][model_spec['arm']])
    require(ref['state_hashes'] == payload['state_hashes'] and ref['identity_sha256'] == runner.nested_hash(expected)
            and run['updates'] == payload['history'], 'Checkpoint full state/history/reference differs')
    return payload


def evaluate_one(seed, arm, device='cuda:0'):
    import protocol
    policy = protocol.verify_frozen()
    model_spec = next(item for item in policy['models'] if item['seed'] == seed and item['arm'] == arm)
    frozen_code = {ref['path']: ref for ref in policy['execution_code']}
    for ref in evaluation_code():
        require(frozen_code.get(ref['path']) == ref, 'Evaluation code not frozen')
    loaded = protocol.load_inputs(policy['selection'])
    require(loaded['evaluation_inputs'] == policy['evaluation_inputs'], 'Changed selected pixel inputs')
    clips, selection = loaded['clips'], policy['selection']
    validate_same_clips(clips, selection['groups'])
    plan = read(model_spec['plan']['path'])
    payload = validate_checkpoint(model_spec, plan)
    source = read(model_spec['prior_evaluation']['path'])
    require(source['log_posteriors_archive'] == model_spec['prior_posteriors'], 'Wrong reused posterior archive binding')
    with np.load(model_spec['prior_posteriors']['path'], allow_pickle=False) as archive:
        reused = validate_reused_rows(source, plan, {'arm': arm, 'reference': model_spec['checkpoint']}, selection, archive)
    core.setup_runtime()
    selected_device = torch.device(device)
    runtime = runner._runtime(selected_device)
    require(runtime == plan['runtime'] == policy['runtime'], 'Evaluation runtime changed')
    output = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
    archive_path = output.with_suffix('.posteriors.npz')
    contract_path = output.with_suffix('.contract.json')
    require(not output.exists() and not archive_path.exists() and not contract_path.exists(), 'Use new immutable evaluation paths')
    preflight_path = BASE / 'results' / 'independent_preflight.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'passed' and preflight['experiment_plan'] == bind(BASE / 'diagnostic_plan.json'),
            'A passed independent preflight for this frozen policy is required before inference')
    contract = {'schema': 'lipnet-step13-before-inference-contract-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'policy': bind(BASE / 'diagnostic_plan.json'), 'seed': seed, 'arm': arm,
                'checkpoint': model_spec['checkpoint'], 'evaluation_inputs': policy['evaluation_inputs'],
                'evaluation_code': evaluation_code(), 'runtime': runtime,
                'independent_preflight': bind(preflight_path)}
    runner._save_json(contract_path, contract)
    outer_rng = core.capture_rng()
    began = time.perf_counter()
    arrays = {}
    try:
        model, adapter, optimizer = core.make_objects(seed, arm, selected_device)
        actual_initial = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
        require(actual_initial == plan['initial_state_hashes_expected'][arm], 'Reconstructed initialization differs')
        del adapter, optimizer
        model.load_state_dict(payload['state']['model'], strict=True)
        model.eval().requires_grad_(False)
        require(runner.nested_hash(model.state_dict()) == payload['state_hashes']['model'], 'Loaded model state mismatch')
        result = evaluate_clips(model, clips, selection['groups'], selected_device, arrays)
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
    finally:
        core.restore_rng(outer_rng)
    require(runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer_rng), 'Evaluation changed caller RNG')
    protocol.verify_frozen()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    runner._atomic_bytes(archive_path, buffer.getvalue())
    conditions = {name: {'same': result['conditions'][name], 'different': {
        **reused[name], 'source_evaluation': model_spec['prior_evaluation'],
        'source_log_posteriors_archive': model_spec['prior_posteriors']}} for name in CONDITIONS}
    value = {'schema': 'lipnet-step13-recording-probe-evaluation-v1', 'status': 'completed',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'arm': arm,
             'checkpoint': model_spec['checkpoint'], 'original_plan': model_spec['plan'],
             'original_run_result': model_spec['run_result'], 'experiment_plan': bind(BASE / 'diagnostic_plan.json'),
             'prior_evaluation': model_spec['prior_evaluation'], 'prior_posteriors': model_spec['prior_posteriors'],
             'before_inference_contract': bind(contract_path), 'initial_state_hashes_verified': actual_initial,
             'evaluation_inputs': policy['evaluation_inputs'], 'groups': selection['groups'], 'conditions': conditions,
             'new_inference_rows': 18, 'reused_inference_rows': 18, 'record_count': 36,
             'log_posteriors_archive': bind(archive_path), 'posterior_arrays': len(arrays),
             'runtime': runtime, 'evaluation_code': evaluation_code(), 'evaluation_contract': evaluation_contract(),
             'elapsed_seconds': time.perf_counter() - began,
             **{key: result[key] for key in ('model_state_before', 'model_state_after', 'model_state_unchanged')},
             'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
             'teacher_feature_caches_loaded': 0, 'student_input_caches_loaded': 6,
             'dev_inputs_loaded': False, 'original_held27_inputs_loaded': False, 'test_inputs_loaded': False,
             'service_weights_updated': False, 'caller_rng_restored_exactly': True,
             'unseen_speaker_generalization_estimate': False}
    runner._save_json(output, value)
    return bind(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['evaluate-all', 'evaluate'])
    parser.add_argument('--seed', type=int, choices=[42, 43])
    parser.add_argument('--arm', choices=['A_ctc', 'B_feature'])
    args = parser.parse_args()
    if args.action == 'evaluate-all':
        for seed in (42, 43):
            for arm in ('A_ctc', 'B_feature'):
                print(json.dumps(evaluate_one(seed, arm), ensure_ascii=False), flush=True)
    else:
        require(args.seed is not None and args.arm is not None, 'Seed and arm required')
        print(json.dumps(evaluate_one(args.seed, args.arm), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
