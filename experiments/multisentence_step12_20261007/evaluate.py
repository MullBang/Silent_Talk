"""Fixed train24 and recording-disjoint internal-hold8 diagnostics.

All original records remain globally training records. The eight held records
are excluded from this fresh experiment's optimization, not an untouched test.
Saved FP32 log probabilities permit independent CPU CTC and metric checking.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR_PATH = ROOT / 'experiments/memorization_step11_20261007/evaluate.py'
PRIOR_SHA = '95b30c6d0d21dced9622506c62bbdad428e53e373dc6f163af0bca913bdc589d'
if hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest() != PRIOR_SHA:
    raise ValueError('Frozen Step11 evaluator helper changed')
spec = importlib.util.spec_from_file_location('step12_frozen_step11_eval_helpers', PRIOR_PATH)
if spec is None or spec.loader is None:
    raise ImportError(str(PRIOR_PATH))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
decoder, original, runner, core = prior.decoder, prior.original, prior.runner, prior.core
require, bind, verify = runner.require, prior.bind, prior.verify
MILESTONES = (0, 900, 1800, 2700, 3600)
CONDITIONS = ('normal', 'static', 'shuffled')
GROUP_SIZES = {'train24': 24, 'hold8': 8, 'train_old12': 12, 'train_new12': 12, 'hold_old4': 4, 'hold_new4': 4}


def evaluation_code():
    """Bind this new evaluator and every reused frozen helper source."""
    refs = [bind(Path(__file__)), bind(BASE / 'test_evaluate.py'), *prior.evaluation_code()]
    return list({ref['path']: ref for ref in refs}.values())


def evaluation_contract():
    return {'schema': 'lipnet-step12-train-hold-evaluation-contract-v1',
            'milestones': list(MILESTONES), 'normal_at_all_milestones': True,
            'static_and_shuffled_at_update': 3600, 'records_per_run': 224,
            'groups': list(GROUP_SIZES), 'headline_groups': ['train24', 'hold8'],
            'record_order': 'train24 then hold8; each row inferred once per condition, strata are reaggregation',
            'primary_training_diagnostic': 'Final normal train24 exact sentences and pooled CER; fixed final checkpoint, no early stop',
            'primary_holdout_diagnostic': 'Final normal hold8 pooled CER and exact sentences, separately from training; no checkpoint selection',
            'decoder': prior.evaluation_contract()['decoder'],
            'primary_metric': prior.evaluation_contract()['primary_metric'],
            'secondary_metrics': prior.evaluation_contract()['secondary_metrics'],
            'ctc': prior.evaluation_contract()['ctc'],
            'posterior_evidence': prior.evaluation_contract()['posterior_evidence'],
            'failure_policy': prior.evaluation_contract()['failure_policy'],
            'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
            'original_source_split': 'train for all32; hold8 split/diagnostic_role is internal_holdout only in this experiment',
            'internal_holdout_inputs_loaded': True, 'original_held27_inputs_loaded': False,
            'limits': ['Three fitting sentences per speaker and one held sentence per speaker are a small selected diagnostic',
                       'Held source recordings are disjoint from fitting recordings but speakers are shared',
                       'Held records were globally training candidates and may have been used in prior experiments; not an untouched final test',
                       'New-source labels remain provisional, not human-approved',
                       'Temporal perturbation sensitivity does not by itself establish semantic lipreading or webcam generalization']}


def validate_checkpoint_payload(plan, validation, arm, payload, expected_update, initial_hashes):
    """Validate all train24 checkpoint provenance without fitting to held data."""
    require(expected_update in MILESTONES[1:], 'Unplanned checkpoint milestone')
    require(plan['updates'] == 3600 and plan['checkpoint_interval'] == 900 and plan['batch_size'] == 2,
            'Changed diagnostic budget')
    require(payload.get('schema') == runner.SCHEMA and payload['completed_updates'] == expected_update,
            'Checkpoint position/schema mismatch')
    require(validation['plan_sha256'] == runner.canonical_plan_hash(plan)
            and plan.get('plan_sha256') == validation['plan_sha256'], 'Plan digest mismatch')
    expected = {'plan_sha256': validation['plan_sha256'], 'seed': plan['seed'], 'arm': arm,
                'total_updates': 3600, 'batch_size': 2, 'schedule_sha256': runner.nested_hash(plan['schedule']),
                'data_sha256': validation['data_sha256'], 'clip_bindings': validation['clips'],
                'code_bindings': runner._code_bindings(), 'runtime': plan['runtime'],
                'optimization': prior.optimization_contract(arm)}
    require(payload['identity'] == expected, 'Checkpoint identity differs from plan/data/code/arm/runtime')
    require(payload['initial_state_hashes'] == initial_hashes, 'Wrong fresh initialization')
    history = payload['history']
    require(len(history) == expected_update and all(row['update'] == i + 1 and row['ids'] == plan['schedule'][i]
            for i, row in enumerate(history)), 'Checkpoint history/schedule mismatch')
    hashes = runner.state_hashes(payload['state'])
    require(hashes == payload['state_hashes'] == history[-1]['state_hashes'], 'Checkpoint full state mismatch')
    return expected


def evaluation_binding(clip):
    """Bind pixel-only input plus explicit source and experimental split roles."""
    sid, pixels, tokens = clip['id'], clip['pixels'], clip['tokens']
    require(clip['split'] == clip['diagnostic_role'] and clip['split'] in ('train', 'internal_holdout')
            and clip['source_split'] == 'train', 'Wrong diagnostic/source split')
    require('teacher_features' not in clip and 'audio' not in clip, 'Forbidden evaluation modality or teacher target')
    require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8 and pixels.ndim == 4
            and pixels.shape[1:] == (96, 96, 3) and len(pixels) > 0, 'Invalid evaluation pixels')
    require(isinstance(tokens, list) and tokens and all(type(t) is int and 1 <= t < 69 for t in tokens)
            and len(pixels) >= core.ctc_min_input_len(tokens), 'Invalid or infeasible evaluation CTC target')
    require(tokens == decoder.tokenizer.encode_text(clip['text']), 'Transcript-token mismatch')
    return {'id': sid, 'split': clip['split'], 'source_split': clip['source_split'],
            'diagnostic_role': clip['diagnostic_role'], 'speaker_id': clip['speaker_id'],
            'source_stem': clip['source_stem'], 'text': clip['text'], 'frames': len(pixels),
            'pixels_sha256': runner.nested_hash(pixels), 'tokens_sha256': runner.nested_hash(tokens)}


def evaluation_bindings(clips):
    require(clips and all(sid == clip['id'] for sid, clip in clips.items()), 'Changed evaluation IDs')
    bindings = {sid: evaluation_binding(clips[sid]) for sid in sorted(clips)}
    return {'data_sha256': runner.nested_hash(bindings), 'clips': bindings}


def validate_groups(plan, train_clips, evaluation_clips):
    groups = plan['diagnostic_groups']
    require(set(groups) == set(GROUP_SIZES), 'Changed diagnostic groups')
    for name, size in GROUP_SIZES.items():
        require(len(groups[name]) == len(set(groups[name])) == size, 'Wrong group cardinality: ' + name)
    train_ids, hold_ids = groups['train24'], groups['hold8']
    require(set(train_ids).isdisjoint(hold_ids) and set(train_clips) == set(train_ids)
            and set(evaluation_clips) == set(train_ids + hold_ids), 'Changed train/hold membership')
    for parent, left, right in (('train24', 'train_old12', 'train_new12'), ('hold8', 'hold_old4', 'hold_new4')):
        require(set(groups[left]).isdisjoint(groups[right]) and set(groups[parent]) == set(groups[left] + groups[right]),
                'Strata do not partition their parent')
    for sid, clip in evaluation_clips.items():
        expected_role = 'train' if sid in train_clips else 'internal_holdout'
        require(clip['split'] == clip['diagnostic_role'] == expected_role and clip['source_split'] == 'train',
                'Training/hold diagnostic role mismatch')
        evaluation_binding(clip)
    for sid, clip in train_clips.items():
        other = evaluation_clips[sid]
        require(clip['split'] == clip['diagnostic_role'] == 'train' and clip['source_split'] == 'train', 'Nontraining optimized input')
        require({key: clip[key] for key in ('id', 'speaker_id', 'source_stem', 'text', 'tokens')}
                == {key: other[key] for key in ('id', 'speaker_id', 'source_stem', 'text', 'tokens')}
                and runner.nested_hash(clip['pixels']) == runner.nested_hash(other['pixels']),
                'Training and evaluation copies disagree')
    train_speakers = Counter(evaluation_clips[sid]['speaker_id'] for sid in train_ids)
    hold_speakers = Counter(evaluation_clips[sid]['speaker_id'] for sid in hold_ids)
    require(len(train_speakers) == len(hold_speakers) == 8 and set(train_speakers) == set(hold_speakers)
            and set(train_speakers.values()) == {3} and set(hold_speakers.values()) == {1},
            'Expected three train and one hold sentence for each of eight speakers')
    train_sources = {evaluation_clips[sid]['source_stem'] for sid in train_ids}
    require(train_sources.isdisjoint(evaluation_clips[sid]['source_stem'] for sid in hold_ids), 'Held recording overlaps training')
    for speaker in train_speakers:
        require(len({evaluation_clips[sid]['source_stem'] for sid in train_ids if evaluation_clips[sid]['speaker_id'] == speaker}) == 1,
                'Training examples must use the selected source recording')
    return groups


def evaluate_clips(model, clips, groups, conditions, device, update, posteriors):
    """Infer all32 exactly once per condition and aggregate train/hold separately."""
    ids = groups['train24'] + groups['hold8']
    require(len(ids) == len(set(ids)) == 32 and set(ids) == set(clips), 'Changed evaluation cohort')
    require(conditions in (('normal',), CONDITIONS), 'Unplanned conditions')
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Student must be frozen eval')
    before = runner.nested_hash(model.state_dict())
    values = {}
    with torch.inference_mode():
        for condition in conditions:
            records = []
            for position, sid in enumerate(ids):
                clip = clips[sid]
                # Reject role or teacher/audio contamination outside failure accounting.
                require('teacher_features' not in clip and 'audio' not in clip, 'Forbidden evaluation model input')
                expected_role = 'train' if sid in groups['train24'] else 'internal_holdout'
                require(clip['split'] == clip['diagnostic_role'] == expected_role and clip['source_split'] == 'train',
                        'Evaluation role changed')
                started = time.perf_counter()
                decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                ctc, diagnostic, error, temporal, posterior_key = None, None, None, None, None
                try:
                    pixels = clip['pixels']
                    temporal = original.condition_metadata(pixels, condition, sid)
                    lp = model(core.prepare_student(core.perturb_pixels(pixels, condition, sid), device))
                    require(lp.shape == (len(pixels), 1, 69) and torch.isfinite(lp).all().item(), 'Malformed/nonfinite model output')
                    lp_cpu = lp.float().cpu()
                    lp_array = lp_cpu[:, 0].numpy().copy()
                    candidate = decoder.decode_prediction(lp_array)
                    diagnostic = prior.output_diagnostics(lp_array, candidate, clip['text'])
                    ctc = float(F.ctc_loss(lp_cpu, torch.tensor(clip['tokens'], dtype=torch.long),
                                torch.tensor([len(pixels)]), torch.tensor([len(clip['tokens'])]),
                                blank=0, reduction='mean', zero_infinity=False))
                    require(np.isfinite(ctc) and ctc >= -1e-6, 'Nonfinite/negative CTC evaluation loss')
                    decoded = candidate
                    posterior_key = f'u{update:04d}_{condition}_{position}'
                    require(posterior_key not in posteriors, 'Duplicate posterior archive key')
                    posteriors[posterior_key] = lp_array
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    decoded = {'prediction': '', 'raw_argmax_ids': [], 'collapsed_ids': [], 'replacement_characters': 0, 'strict_replay': None}
                    ctc, diagnostic, posterior_key = None, None, None
                token_edits = decoder.diagnostics.edit_distance(clip['tokens'], decoded['collapsed_ids'])
                stratum = next(name for name in ('train_old12', 'train_new12', 'hold_old4', 'hold_new4') if sid in groups[name])
                records.append({'id': sid, 'speaker_id': clip['speaker_id'], 'source_stem': clip['source_stem'],
                    'split': expected_role, 'source_split': 'train', 'diagnostic_role': expected_role,
                    'group': stratum, 'condition': condition, 'reference': clip['text'], 'prediction': decoded['prediction'],
                    'status': 'ok' if error is None else 'failed', 'error': error, 'frames': len(clip['pixels']),
                    'temporal': temporal, 'seconds': time.perf_counter() - started,
                    'raw_argmax_ids': decoded['raw_argmax_ids'], 'collapsed_ids': decoded['collapsed_ids'],
                    'target_token_ids': clip['tokens'], 'replacement_characters': decoded['replacement_characters'],
                    'strict_label_decoder_diagnostic': decoded['strict_replay'],
                    'metrics': core.counts(clip['text'], decoded['prediction']), 'ctc_loss_mean': ctc,
                    'token_metrics': {'token_edits': token_edits, 'target_tokens': len(clip['tokens']),
                                      'exact': clip['tokens'] == decoded['collapsed_ids']},
                    'output_diagnostics': diagnostic, 'log_posteriors_key': posterior_key})
            values[condition] = {'records': records, 'groups': {
                name: prior.summarize_records([row for row in records if row['id'] in group_ids]) for name, group_ids in groups.items()}}
    after = runner.nested_hash(model.state_dict())
    require(before == after, 'Evaluation changed student state')
    return {'completed_updates': update, 'conditions': values, 'model_state_before': before,
            'model_state_after': after, 'model_state_unchanged': True}


def validate_checkpoint_refs(checkpoints):
    require([item['completed_updates'] for item in checkpoints] == list(MILESTONES[1:]),
            'Missing, duplicate, reordered or extra checkpoint milestone')
    for item in checkpoints:
        path = Path(item['path']).resolve()
        require(Path(item['sidecar']).resolve() == path.with_suffix('.pt.json'), 'Wrong checkpoint sidecar path')
        require(bind(path)['sha256'] == item['sha256'] and bind(item['sidecar'])['sha256'] == item['sidecar_sha256'],
                'Checkpoint reference SHA mismatch')


def evaluate_run(plan, loaded, arm, checkpoint_refs, initial_state_hashes_expected, output,
                 device='cuda:0', experiment_plan_ref=None):
    """Evaluate saved milestones only after a fixed-budget run has completed."""
    output = Path(output).resolve()
    archive_path = output.with_suffix('.posteriors.npz')
    require(output.is_relative_to(BASE / 'results') and not output.exists() and not archive_path.exists(),
            'Use a new immutable Step12 results path')
    require(plan['updates'] == 3600 and plan['checkpoint_interval'] == 900 and plan['seed'] in (42, 43), 'Wrong diagnostic recipe')
    train_clips, eval_clips = loaded['train_clips'], loaded['evaluation_clips']
    groups = validate_groups(plan, train_clips, eval_clips)
    validation = runner.validate_plan(plan, train_clips, arm)
    require(plan['training_data'] == {key: validation[key] for key in ('data_sha256', 'clips')}, 'Changed train input bytes')
    eval_validation = evaluation_bindings(eval_clips)
    require(plan['evaluation_data'] == eval_validation, 'Changed evaluation input bytes/metadata')
    require(plan['initial_state_hashes_expected'][arm] == initial_state_hashes_expected, 'Unbound initial state expectation')
    frozen = {ref['path']: ref for ref in plan['execution_code']}
    for ref in evaluation_code():
        require(frozen.get(ref['path']) == ref, 'Evaluator code was not frozen before training')
    for ref in plan['execution_code']:
        verify(ref)
    if experiment_plan_ref is not None:
        verify(experiment_plan_ref)
        require(plan['experiment_plan'] == experiment_plan_ref, 'Wrong experiment policy reference')
    validate_checkpoint_refs(checkpoint_refs)
    core.setup_runtime()
    selected_device = torch.device(device)
    runtime = runner._runtime(selected_device)
    require(runtime == plan['runtime'], 'Evaluation runtime changed')
    outer_rng = core.capture_rng()
    began = time.perf_counter()
    posterior_arrays, results = {}, []
    try:
        model, adapter, optimizer = core.make_objects(plan['seed'], arm, selected_device)
        actual_initial = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
        require(actual_initial == initial_state_hashes_expected, 'Reconstructed initialization differs')
        model.eval().requires_grad_(False)
        initial = evaluate_clips(model, eval_clips, groups, ('normal',), selected_device, 0, posterior_arrays)
        initial['checkpoint'] = None
        results.append(initial)
        del adapter, optimizer
        for checkpoint in checkpoint_refs:
            update = checkpoint['completed_updates']
            payload = runner.load_checkpoint(Path(checkpoint['path']))
            validate_checkpoint_payload(plan, validation, arm, payload, update, initial_state_hashes_expected)
            require(checkpoint['state_hashes'] == payload['state_hashes'], 'Checkpoint reference state mismatch')
            model.load_state_dict(payload['state']['model'], strict=True)
            require(runner.nested_hash(model.state_dict()) == payload['state_hashes']['model'], 'Loaded student state mismatch')
            result = evaluate_clips(model, eval_clips, groups, CONDITIONS if update == 3600 else ('normal',),
                                    selected_device, update, posterior_arrays)
            result['checkpoint'] = checkpoint
            results.append(result)
            del payload
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
    finally:
        core.restore_rng(outer_rng)
    require(runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer_rng), 'Evaluation changed caller RNG')
    row_count = sum(len(condition['records']) for item in results for condition in item['conditions'].values())
    require(row_count == 224, 'Incomplete milestone evaluation')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **posterior_arrays)
    runner._atomic_bytes(archive_path, buffer.getvalue())
    value = {'schema': 'lipnet-step12-multisentence-evaluation-v1', 'status': 'completed',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': plan['seed'], 'arm': arm,
             'plan_sha256': validation['plan_sha256'], 'data_sha256': validation['data_sha256'],
             'evaluation_data_sha256': eval_validation['data_sha256'],
             'experiment_plan': experiment_plan_ref, 'initial_state_hashes_verified': actual_initial,
             'ids': groups['train24'] + groups['hold8'], 'diagnostic_groups': groups, 'evaluation_groups': groups, 'milestones': results,
             'record_count': row_count, 'log_posteriors_archive': bind(archive_path),
             'posterior_arrays': len(posterior_arrays), 'runtime': runtime, 'evaluation_code': evaluation_code(),
             'evaluation_contract': evaluation_contract(), 'elapsed_seconds': time.perf_counter() - began,
             'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
             'dev_inputs_loaded': False, 'internal_holdout_inputs_loaded': True, 'original_held27_inputs_loaded': False,
             'test_inputs_loaded': False, 'holdout_teacher_features_present_in_evaluation_inputs': False,
             'holdout_teacher_features_used_for_inference': False,
             'service_weights_updated': False, 'caller_rng_restored_exactly': True}
    runner._save_json(output, value)
    return bind(output)
