"""Predeclared 824-row evaluation of the new paired diversity experiment.

All current train/dev/probe records came from historical training pools. The
new dev16 is recording-disjoint within this experiment, not an untouched test.
"""
from __future__ import annotations

import argparse
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

import deterministic_pool as pooling

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR_PATH = ROOT / 'experiments/memorization_step11_20261007/evaluate.py'
PRIOR_SHA = '95b30c6d0d21dced9622506c62bbdad428e53e373dc6f163af0bca913bdc589d'
if hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest() != PRIOR_SHA:
    raise ValueError('Frozen diagnostic helpers changed')
spec = importlib.util.spec_from_file_location('step16_frozen_diagnostic_helpers', PRIOR_PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
decoder, original, runner, core = prior.decoder, prior.original, prior.runner, prior.core
require, bind, verify = prior.require, prior.bind, prior.verify
UPDATES = (1200, 2400, 3000, 3600)
CONDITIONS = ('normal', 'static', 'shuffled')
ALL_CHECKPOINTS = (600, 1200, 1800, 2400, 3000, 3600)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evaluation_code():
    refs = [bind(__file__), bind(BASE / 'test_evaluate.py'), bind(BASE / 'deterministic_pool.py'),
            *prior.evaluation_code()]
    return list({ref['path']: ref for ref in refs}.values())


def evaluation_contract():
    return {'schema': 'lipnet-step16-diversity-evaluation-contract-v1',
            'normal_dev_updates': list(UPDATES), 'normal_train_updates': [2400, 3600],
            'historical_probe_normal_updates': [3600], 'static_shuffled_dev_updates': [3600],
            'expected_rows_per_cohort': {'train24': 158, 'train72': 254}, 'expected_rows_total': 824,
            'normal_row_order': 'native train IDs when planned, then dev16, then historical probe14 at final only',
            'strata_are_reaggregation_not_extra_inference': True,
            'primary': 'Fixed final3600 dev16 pooled CER, compared between fresh train24 and train72 within each seed',
            'decoder': prior.evaluation_contract()['decoder'],
            'primary_metric': prior.evaluation_contract()['primary_metric'],
            'ctc': prior.evaluation_contract()['ctc'],
            'posterior_evidence': prior.evaluation_contract()['posterior_evidence'],
            'failure_policy': prior.evaluation_contract()['failure_policy'],
            'execution_revision': 'step16r1_deterministic_framewise_spatial_pooling',
            'strict_deterministic_algorithms': True,
            'pooling_implementation': pooling.pooling_contract(),
            'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
            'limits': ['Current dev16 uses known speakers and disjoint recordings but belonged to older training pools.',
                       'Probe14 is a repeatedly inspected diagnostic; its current unseen6 speaker rows are not an untouched test.',
                       'Only B_feature is compared; no A versus B generalization estimate.',
                       'Temporal sensitivity alone does not prove semantic lipreading.']}


def evaluation_inputs(data, plan):
    """Construct pixel-only evaluation inputs with explicit current roles."""
    train_ids, dev_ids, probe_ids = plan['train_ids'], plan['dev_ids'], plan['probe_ids']
    common = plan['common_train24_ids']
    expected_train = 24 if plan['cohort'] == 'train24' else 72
    require(len(train_ids) == len(set(train_ids)) == expected_train and len(common) == len(set(common)) == 24
            and set(common) <= set(train_ids), 'Changed native/common training cohort')
    require(len(dev_ids) == len(set(dev_ids)) == 16 and len(probe_ids) == len(set(probe_ids)) == 14,
            'Changed development/probe denominator')
    require(set(train_ids).isdisjoint(dev_ids + probe_ids) and set(dev_ids).isdisjoint(probe_ids), 'Overlapping experiment roles')
    pools = {'train': data['train'], 'internal_dev': data['dev'], 'historical_probe': data['probe']}
    ids_by_role = {'train': train_ids, 'internal_dev': dev_ids, 'historical_probe': probe_ids}
    require(all(set(ids) <= set(pools[role]) for role, ids in ids_by_role.items()), 'Selected input is missing from its role pool')
    speakers = {data['train'][sid]['speaker_id'] for sid in train_ids}
    require(len(speakers) == 8, 'Expected eight current training speakers')
    clips = {}
    for role, ids in ids_by_role.items():
        for sid in ids:
            source = pools[role][sid]
            require('audio' not in source, 'Audio input is forbidden')
            require(role == 'train' or 'teacher_features' not in source, 'Held inputs must not contain teacher features')
            clip = {key: source[key] for key in ('id', 'speaker_id', 'source_stem', 'text', 'pixels', 'tokens')}
            clip.update(split=role, diagnostic_role=role, source_split='train',
                        seen_in_current_training=clip['speaker_id'] in speakers)
            require(sid == clip['id'], 'Changed input key/ID')
            validate_clip(clip)
            clips[sid] = clip
    require(all(clips[sid]['seen_in_current_training'] for sid in dev_ids), 'Development speaker is outside current training')
    known = [sid for sid in probe_ids if clips[sid]['seen_in_current_training']]
    unseen = [sid for sid in probe_ids if not clips[sid]['seen_in_current_training']]
    require(known == plan['probe_known_ids'] and unseen == plan['probe_unseen_ids'] and len(known) == 8 and len(unseen) == 6,
            'Changed current known/unseen probe strata')
    return clips


def validate_clip(clip):
    require(clip['split'] == clip['diagnostic_role'] and clip['split'] in ('train', 'internal_dev', 'historical_probe')
            and clip['source_split'] == 'train', 'Wrong current or historical source role')
    require('teacher_features' not in clip and 'audio' not in clip, 'Forbidden evaluation model modality')
    pixels, tokens = clip['pixels'], clip['tokens']
    require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
            and pixels.shape[1:] == (96, 96, 3) and len(pixels) > 0, 'Malformed student pixels')
    require(isinstance(tokens, list) and tokens and all(type(t) is int and 1 <= t < 69 for t in tokens)
            and tokens == decoder.tokenizer.encode_text(clip['text'])
            and len(pixels) >= core.ctc_min_input_len(tokens), 'Invalid/infeasible CTC target')


def input_bindings(clips):
    result = {}
    for sid in sorted(clips):
        clip = clips[sid]
        validate_clip(clip)
        result[sid] = {key: clip[key] for key in ('id', 'speaker_id', 'source_stem', 'text', 'split',
                                                'diagnostic_role', 'source_split', 'seen_in_current_training')}
        result[sid].update(frames=len(clip['pixels']), pixels_sha256=runner.nested_hash(clip['pixels']),
                           tokens_sha256=runner.nested_hash(clip['tokens']))
    return {'data_sha256': runner.nested_hash(result), 'clips': result}


def condition_groups(plan, update, condition):
    require(update in UPDATES and condition in CONDITIONS, 'Unplanned evaluation position/condition')
    if condition != 'normal':
        require(update == 3600, 'Temporal conditions only at fixed final checkpoint')
        return list(plan['dev_ids']), {'dev16': list(plan['dev_ids'])}
    ids, groups = [], {}
    if update in (2400, 3600):
        ids.extend(plan['train_ids'])
        groups.update(train_native=list(plan['train_ids']), train_common24=list(plan['common_train24_ids']))
    ids.extend(plan['dev_ids'])
    groups['dev16'] = list(plan['dev_ids'])
    if update == 3600:
        ids.extend(plan['probe_ids'])
        groups.update(historical_probe14=list(plan['probe_ids']), probe_known8=list(plan['probe_known_ids']),
                      probe_unseen6=list(plan['probe_unseen_ids']))
    require(len(ids) == len(set(ids)), 'A cohort was inferred more than once within a condition')
    return ids, groups


def evaluate_milestone(model, clips, plan, update, device, posteriors):
    require(update in UPDATES, 'Unplanned milestone')
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Student must be frozen eval')
    before = runner.nested_hash(model.state_dict())
    for condition in CONDITIONS if update == 3600 else ('normal',):
        ids, _ = condition_groups(plan, update, condition)
        for position, sid in enumerate(ids):
            validate_clip(clips[sid])
            require(f'u{update:04d}_{condition}_{position}' not in posteriors, 'Duplicate saved posterior key')
    values = {}
    with torch.inference_mode():
        for condition in CONDITIONS if update == 3600 else ('normal',):
            ids, groups = condition_groups(plan, update, condition)
            records = []
            for position, sid in enumerate(ids):
                clip = clips[sid]
                validate_clip(clip)
                role = 'train' if sid in plan['train_ids'] else 'internal_dev' if sid in plan['dev_ids'] else 'historical_probe'
                require(clip['split'] == role, 'Wrong evaluated current role')
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
                    diagnostic = prior.output_diagnostics(array, candidate, clip['text'])
                    ctc = float(F.ctc_loss(lp_cpu, torch.tensor(clip['tokens'], dtype=torch.long),
                                torch.tensor([len(pixels)]), torch.tensor([len(clip['tokens'])]),
                                blank=0, reduction='mean', zero_infinity=False))
                    require(np.isfinite(ctc) and ctc >= -1e-6, 'Nonfinite/negative CTC evaluation loss')
                    decoded = candidate
                    key = f'u{update:04d}_{condition}_{position}'
                    require(key not in posteriors, 'Duplicate saved posterior key')
                    posteriors[key] = array
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                    ctc, diagnostic, key = None, None, None
                token_edits = decoder.diagnostics.edit_distance(clip['tokens'], decoded['collapsed_ids'])
                group = 'train_native' if role == 'train' else 'dev16' if role == 'internal_dev' else 'probe_known8' if clip['seen_in_current_training'] else 'probe_unseen6'
                records.append({'id': sid, 'speaker_id': clip['speaker_id'], 'source_stem': clip['source_stem'],
                    'split': role, 'source_split': 'train', 'diagnostic_role': role,
                    'seen_in_current_training': clip['seen_in_current_training'],
                    'group': group, 'condition': condition, 'reference': clip['text'], 'prediction': decoded['prediction'],
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
                name: prior.summarize_records([row for row in records if row['id'] in group_ids]) for name, group_ids in groups.items()},
                'per_speaker': {name: {speaker: prior.summarize_records([row for row in records if row['id'] in group_ids and row['speaker_id'] == speaker])
                    for speaker in sorted({clips[sid]['speaker_id'] for sid in group_ids})} for name, group_ids in groups.items()}}
    after = runner.nested_hash(model.state_dict())
    require(before == after, 'Evaluation changed student state')
    return {'completed_updates': update, 'conditions': values, 'model_state_before': before,
            'model_state_after': after, 'model_state_unchanged': True}


def validate_checkpoint_refs(checkpoints):
    require([ref['completed_updates'] for ref in checkpoints] == list(ALL_CHECKPOINTS), 'Incomplete/unordered checkpoint set')
    for ref in checkpoints:
        require(Path(ref['sidecar']).resolve() == Path(ref['path']).resolve().with_suffix('.pt.json')
                and bind(ref['path'])['sha256'] == ref['sha256'] and bind(ref['sidecar'])['sha256'] == ref['sidecar_sha256'],
                'Checkpoint/sidecar binding changed')


def validate_training_result(value, plan, plan_ref, policy_ref):
    require(value['schema'] == 'lipnet-step16-training-run-result-v1' and value['status'] == 'completed'
            and value['run_kind'] == 'main' and value['seed'] == plan['seed']
            and value['cohort'] == plan['cohort'] and value['arm'] == plan['arm'] == 'B_feature',
            'Wrong completed production run')
    require(value['plan_file'] == plan_ref and value['experiment_plan'] == policy_ref == plan['experiment_plan']
            and value['plan_sha256'] == plan['plan_sha256'] == runner.canonical_plan_hash(plan), 'Training contract changed')
    require(value['start_update'] == 0 and value['completed_updates'] == 3600
            and len(value['updates']) == value['new_successful_optimizer_updates'] == 3600
            and value['training_input_ids'] == plan['train_ids'] == plan['ids'], 'Incomplete or changed native training scope')
    require(value['initial_state_hashes'] == plan['initial_state_hashes_expected']
            and value['runtime'] == plan['runtime'] and value['resume_count'] in (0, 1), 'Changed run initialization/runtime')
    require(all(value[name] is False for name in ('dev_inputs_loaded', 'probe_inputs_loaded', 'audio_used',
                                                'teacher_model_loaded', 'service_weights_updated')), 'Training modality/role violation')
    validate_checkpoint_refs(value['checkpoints'])
    require(value['checkpoint'] == value['final_checkpoint'] == value['checkpoints'][-1], 'Wrong fixed final checkpoint')


def validated_model_states(plan, run, engine):
    """Verify all six full states before any inference; retain model tensors only."""
    states, evidence = {}, []
    for ref in run['checkpoints']:
        update = ref['completed_updates']
        payload = engine.load_checkpoint(Path(ref['path']))
        identity = engine.validate_checkpoint_payload(plan, payload, update)
        require(ref['identity_sha256'] == run['identity_sha256'] == runner.nested_hash(identity)
                and ref['state_hashes'] == payload['state_hashes'], 'Checkpoint identity or full-state reference changed')
        require(payload['history'] == run['updates'][:update], 'Checkpoint is not the recorded training prefix')
        require(payload['initial_state_hashes'] == run['initial_state_hashes'], 'Checkpoint initialization differs from run')
        if update == 3600:
            require(payload['state_hashes'] == run['final_state_hashes'], 'Wrong final training state')
        evidence.append({'checkpoint': ref, 'identity_sha256': runner.nested_hash(identity),
                         'state_hashes_verified': payload['state_hashes'], 'history_updates_verified': update,
                         'next_update': payload['next_update'], 'next_learning_rate': payload['next_learning_rate'],
                         'resume_count': payload['resume_count'], 'restricted_cpu_load': True})
        if update in UPDATES:
            states[update] = payload['state']['model']
        del payload
    return states, evidence


def make_evaluation_objects(seed, device):
    """Use the same parameter-free spatial pooling as revision-r1 training."""
    model, adapter, optimizer = core.make_objects(seed, 'B_feature', device)
    before = runner.nested_hash(model.state_dict())
    pooling.configure_model(model)
    require(runner.nested_hash(model.state_dict()) == before, 'Pooling configuration changed model parameters/buffers')
    return model, adapter, optimizer


def evaluate_one(seed, cohort, device='cuda:0'):
    import protocol
    import engine
    policy = protocol.verify_frozen()
    require(seed in (42, 43) and cohort in ('train24', 'train72'), 'Unplanned evaluation run')
    policy_ref, plan_ref = bind(protocol.POLICY), bind(protocol.plan_path(seed, cohort))
    plan = protocol.run_plan(seed, cohort)
    frozen_code = {ref['path']: ref for ref in policy['execution_code']}
    for ref in evaluation_code():
        require(frozen_code.get(ref['path']) == ref, 'Evaluation code not frozen')
    require(policy['evaluation_contract'] == evaluation_contract(), 'Frozen evaluation contract differs')
    result_path = protocol.run_path(seed, cohort) / 'run_result.json'
    run_ref, run = bind(result_path), read(result_path)
    validate_training_result(run, plan, plan_ref, policy_ref)
    for ref in run['segments']:
        verify(ref)
    loaded = protocol.load_data('evaluation')
    require(loaded['counts'] == {'student_input_npz_decoded': 102, 'training_feature_npz_decoded': 0,
                                'dev_or_probe_teacher_features_opened': 0, 'teacher_pixel_arrays_decoded': 0,
                                'raw_media_opened': 0}, 'Evaluation loader scope changed')
    clips = evaluation_inputs(loaded, plan)
    inputs = input_bindings(clips)
    for sid in plan['train_ids']:
        bound = plan['training_data']['clips'][sid]
        require(all(inputs['clips'][sid][key] == bound[key] for key in ('id', 'frames', 'pixels_sha256', 'tokens_sha256')),
                'Native training evaluation pixels/tokens differ from original training contract')
    states, checkpoint_evidence = validated_model_states(plan, run, engine)
    pooling.setup_runtime(core)
    require(torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled(),
            'Revision-r1 evaluation requires strict deterministic algorithms')
    selected_device = torch.device(device)
    runtime = engine.runtime_for(selected_device)
    require(runtime == plan['runtime'] == policy['runtime'], 'Evaluation runtime changed')
    output = BASE / 'results' / f'evaluation_seed{seed}_{cohort}.json'
    archive_path, contract_path = output.with_suffix('.posteriors.npz'), output.with_suffix('.contract.json')
    require(not output.exists() and not archive_path.exists() and not contract_path.exists(), 'Use new immutable evaluation paths')
    preflight_path = BASE / 'results/independent_preflight.json'
    preflight = read(preflight_path)
    smoke_path = BASE / 'results/smoke_preflight.json'
    smoke = read(smoke_path)
    require(preflight['status'] == smoke['status'] == 'passed'
            and preflight['experiment_plan'] == smoke['experiment_plan'] == policy_ref,
            'Passed independent preflight and GPU smoke for this policy are required before inference')
    contract = {'schema': 'lipnet-step16-before-inference-contract-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'experiment_plan': policy_ref, 'policy': policy_ref, 'seed': seed, 'cohort': cohort, 'arm': 'B_feature',
                'original_plan': plan_ref, 'original_run_result': run_ref, 'checkpoints': run['checkpoints'],
                'evaluation_inputs': inputs, 'evaluation_code': evaluation_code(), 'runtime': runtime,
                'independent_preflight': bind(preflight_path), 'gpu_smoke_preflight': bind(smoke_path),
                'checkpoint_validations': checkpoint_evidence, 'input_load_counts': loaded['counts']}
    runner._save_json(contract_path, contract)
    outer_rng = core.capture_rng()
    began, arrays, milestones = time.perf_counter(), {}, []
    try:
        model, adapter, optimizer = make_evaluation_objects(seed, selected_device)
        actual_initial = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
        require(actual_initial == plan['initial_state_hashes_expected'], 'Reconstructed initialization differs')
        del adapter, optimizer
        for ref in run['checkpoints']:
            update = ref['completed_updates']
            if update not in UPDATES:
                continue
            model.load_state_dict(states.pop(update), strict=True)
            model.eval().requires_grad_(False)
            require(runner.nested_hash(model.state_dict()) == ref['state_hashes']['model'], 'Loaded model state mismatch')
            value = evaluate_milestone(model, clips, plan, update, selected_device, arrays)
            value['checkpoint'] = ref
            milestones.append(value)
            print(json.dumps({'seed': seed, 'cohort': cohort, 'evaluated_update': update,
                              'rows': sum(len(v['records']) for v in value['conditions'].values())}), flush=True)
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
    finally:
        core.restore_rng(outer_rng)
    require(runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer_rng), 'Evaluation changed caller RNG')
    require(input_bindings(clips) == inputs, 'Evaluation changed input pixels/targets/metadata')
    protocol.verify_frozen()
    require(bind(result_path) == run_ref, 'Original run result changed during evaluation')
    validate_checkpoint_refs(run['checkpoints'])
    rows = [row for milestone in milestones for values in milestone['conditions'].values() for row in values['records']]
    expected = evaluation_contract()['expected_rows_per_cohort'][cohort]
    require(len(rows) == expected and len(arrays) == sum(row['status'] == 'ok' for row in rows), 'Evaluation row/posterior accounting differs')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    runner._atomic_bytes(archive_path, buffer.getvalue())
    value = {'schema': 'lipnet-step16-diversity-evaluation-v1', 'status': 'completed',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'cohort': cohort, 'arm': 'B_feature',
             'experiment_plan': policy_ref, 'original_plan': plan_ref, 'original_run_result': run_ref,
             'plan_sha256': plan['plan_sha256'], 'training_data_sha256': plan['training_data']['data_sha256'],
             'before_inference_contract': bind(contract_path), 'checkpoint_validations': checkpoint_evidence,
             'initial_state_hashes_verified': actual_initial, 'evaluation_inputs': inputs,
             'milestones': milestones, 'record_count': len(rows), 'new_inference_rows': len(rows), 'reused_inference_rows': 0,
             'failed_rows': sum(row['status'] == 'failed' for row in rows), 'posterior_arrays': len(arrays),
             'log_posteriors_archive': bind(archive_path), 'runtime': runtime, 'evaluation_code': evaluation_code(),
             'evaluation_contract': evaluation_contract(), 'input_load_counts': loaded['counts'],
             'student_inputs_inferred': len(clips), 'student_inputs_loaded_but_not_inferred': 102 - len(clips),
             'elapsed_seconds': time.perf_counter() - began, 'optimization_steps': 0, 'teacher_forward': False,
             'audio_used': False, 'teacher_feature_caches_loaded': 0, 'student_input_caches_loaded': 102,
             'service_weights_updated': False, 'caller_rng_restored_exactly': True, 'evaluation_inputs_unchanged': True,
             'unseen_speaker_generalization_estimate': False}
    runner._save_json(output, value)
    return bind(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['evaluate-all', 'evaluate'])
    parser.add_argument('--seed', type=int, choices=[42, 43])
    parser.add_argument('--cohort', choices=['train24', 'train72'])
    args = parser.parse_args()
    if args.action == 'evaluate-all':
        for seed in (42, 43):
            for cohort in ('train24', 'train72'):
                print(json.dumps(evaluate_one(seed, cohort), ensure_ascii=False), flush=True)
    else:
        require(args.seed is not None and args.cohort is not None, 'Seed and cohort required')
        print(json.dumps(evaluate_one(args.seed, args.cohort), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

