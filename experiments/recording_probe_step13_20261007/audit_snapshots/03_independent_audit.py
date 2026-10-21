"""Independent, CPU-only provenance and numerical audit of train memorization.

Only the immutable earlier independent auditor is imported. No production
training, model, protocol, selection or evaluation implementation is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
NUMERICAL = ROOT / 'experiments/local_training_step05_20261007/verify_step05.py'
NUMERICAL_SHA = 'a4462634845cf1a888abe5f0fae7cbe8b554a6edba271006b6d14c68833e5d76'
if hashlib.sha256(NUMERICAL.read_bytes()).hexdigest() != NUMERICAL_SHA:
    raise ValueError('Independent numerical ancestor changed')
spec = importlib.util.spec_from_file_location('step11_independent_numerical', NUMERICAL)
num = importlib.util.module_from_spec(spec)
spec.loader.exec_module(num)
same, require = num.same, num.require
read, bind, verify, sha = num.read_json, num.binding, num.verify_binding, num.sha
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
MILESTONES = (0, 300, 600, 900, 1200)
HANDOFF = ROOT / 'experiments/new_inputs_step09_20261007/candidate_handoff.json'
HANDOFF_SHA = '8135c7900ad5ad244e98ddb490fad2a05db115b221edba957cb030603a224ae6'
POLICY_SHA = '0b9b5f2a20b5b42b67cfff9d06b52ec53fa11c6ee8f7e30afdaecdd0c7b34037'
PLAN_SHAS = {42: '528f26d74dfa9bc7e8341a4745593a0a242c1b37cea70caf51c5a93cd9bed054',
             43: 'f5e437f0cc3099a23cea0f75c35d1271d0cccada305de41d5fb77cfca25d5fe5'}


def collapse(raw: list[int], frames: int) -> list[int]:
    """Collapse adjacent frame repeats before removing the zero blank."""
    require(type(frames) is int and frames > 0 and isinstance(raw, list) and len(raw) == frames,
            'Missing frame-level argmax IDs')
    require(all(type(t) is int and 0 <= t <= 68 for t in raw), 'Invalid argmax token')
    return [t for i, t in enumerate(raw) if t != 0 and (i == 0 or t != raw[i - 1])]


def compose(ids: list[int]) -> dict:
    """Independent Unicode syllable arithmetic preserving malformed jamos."""
    require(isinstance(ids, list) and all(type(t) is int and 1 <= t <= 68 for t in ids),
            'Invalid collapsed token')
    out, index, first_error = [], 0, None
    while index < len(ids):
        token = ids[index]
        if token == 68:
            out.append(' ')
            index += 1
        elif 1 <= token <= 19 and index + 1 < len(ids) and 20 <= ids[index + 1] <= 40:
            tail = ids[index + 2] - 40 if index + 2 < len(ids) and 41 <= ids[index + 2] <= 67 else 0
            out.append(chr(0xAC00 + ((token - 1) * 21 + ids[index + 1] - 20) * 28 + tail))
            index += 3 if tail else 2
        else:
            if first_error is None:
                grapheme = chr(0x1100 + token - 1) if token <= 19 else chr(0x1161 + token - 20) if token <= 40 else chr(0x11A8 + token - 41)
                first_error = f'ValueError: 잘못된 자모 순서/토큰 (위치 {index}): {grapheme!r}'
            out.append('\ufffd')
            index += 1
    prediction = ''.join(out)
    return {'prediction': prediction, 'replacement_characters': prediction.count('\ufffd'),
            'strict': {'status': 'ok' if first_error is None else 'failed',
                       'prediction': prediction if first_error is None else None, 'error': first_error}}


def selection_ids(entries: list[dict], rows: dict) -> tuple[list[str], dict]:
    """Independently apply the fixed speaker hash and lower frame median rule."""
    require(len({row['id'] for row in entries}) == len(entries), 'Duplicate source ID')
    groups = {}
    for role, group in (('old_train', 'old4'), ('new_provisional_train', 'new4')):
        eligible = [entry for entry in entries if entry['role'] == role]
        speakers = sorted({entry['speaker_id'] for entry in eligible}, key=lambda speaker: (
            hashlib.sha256(('step11-memorization-v1:' + role + ':' + speaker).encode()).hexdigest(), speaker))
        require(len(speakers) >= 4, 'Insufficient eligible speakers')
        selected = []
        for speaker in speakers[:4]:
            candidates = sorted([entry['id'] for entry in eligible if entry['speaker_id'] == speaker],
                                key=lambda sid: (rows[sid]['frames'], sid))
            selected.append(candidates[(len(candidates) - 1) // 2])
        groups[group] = selected
    ids = groups['old4'] + groups['new4']
    require(len(ids) == len(set(ids)) == 8, 'Diagnostic IDs not eight unique items')
    require(len({rows[sid]['speaker_id'] for sid in ids}) == 8, 'Diagnostic speakers not distinct')
    return ids, groups


def source_metadata() -> tuple[dict, dict, dict]:
    """Read only frozen metadata for selection; this never opens held/dev payloads."""
    handoff = read(HANDOFF, HANDOFF_SHA)
    rows = {}
    for generation in ('old', 'new'):
        reference = handoff['artifacts'][generation + '_data_plan']
        verify(reference)
        for row in read(reference['path'])['records']:
            require(row['id'] not in rows, 'Duplicate logical source ID')
            rows[row['id']] = row
    same(len(rows), 632, 'Original logical source count')
    entries = {entry['id']: entry for entry in handoff['records']}
    same(set(entries), set(rows), 'Handoff and logical metadata IDs')
    return handoff, rows, entries


def selected_payloads(ids: list[str], rows: dict, entries: dict) -> tuple[dict, list[dict]]:
    """Read eight bound inputs and eight feature payloads only, on CPU."""
    bindings, artifacts = {}, []
    for sid in sorted(ids):
        entry, row = entries[sid], rows[sid]
        require(entry['role'] in ('old_train', 'new_provisional_train'), 'Forbidden payload role')
        same(row['split'], 'train', 'Selected original split')
        require(entry['teacher_feature'] is not None, 'Missing selected teacher feature')
        for reference in (entry['input'], entry['teacher_feature']):
            verify(reference)
            artifacts.append(reference)
        with np.load(entry['input']['path'], allow_pickle=False) as saved:
            pixels = saved['student_rgb']
            tokens = saved['label_tokens']
            metadata = json.loads(str(saved['metadata_json'].item()))
        same(metadata['record'], row, 'Saved logical input record')
        same(pixels.shape, (row['frames'], 96, 96, 3), 'Student input shape')
        same(pixels.dtype, np.dtype('uint8'), 'Student input precision')
        same(tokens.dtype, np.dtype('int64'), 'Target token precision')
        same(tokens.tolist(), row['token_ids'], 'Target token identity')
        with np.load(entry['teacher_feature']['path'], allow_pickle=False) as saved:
            features = saved['features']
        same(features.shape, (row['frames'], 512), 'Teacher feature shape')
        same(features.dtype, np.dtype('float32'), 'Teacher feature precision')
        require(np.isfinite(features).all() and (np.linalg.norm(features, axis=1) > 1e-8).all(),
                'Invalid teacher feature values')
        bindings[sid] = {'id': sid, 'frames': row['frames'], 'pixels_sha256': num.nested_hash(pixels),
                         'tokens_sha256': num.nested_hash(tokens.tolist()),
                         'teacher_sha256': num.nested_hash(features)}
    return bindings, artifacts


def assert_near(actual: float, expected: float, message: str, tolerance: float = 1e-6) -> None:
    """Compare finite independently calculated scalar statistics."""
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and abs(actual - expected) <= tolerance, message)


def repeated_runs(text: str) -> tuple[int, int]:
    """Count nonspace runs; a whitespace boundary terminates a run."""
    longest, runs, length, last = 0, 0, 0, None
    for char in unicodedata.normalize('NFC', text).strip() + ' ':
        if not char.isspace() and char == last:
            length += 1
        else:
            longest = max(longest, length)
            runs += int(length >= 5)
            length = 0 if char.isspace() else 1
            last = None if char.isspace() else char
    return longest, runs


def audit_training(plan: dict, seed: int, arm: str, rows: dict, clip_bindings: dict) -> dict:
    """Check all 1,200 update records and each complete milestone checkpoint."""
    run_path = BASE / 'runs' / f'seed{seed}' / arm / 'continuous'
    result = read(run_path / 'run_result.json')
    contract = read(run_path / 'segment_contract.json')
    identity = contract['identity']
    same(result['status'], 'completed', 'Unfinished diagnostic')
    same(result['seed'], seed, 'Training seed')
    same(result['arm'], arm, 'Training arm')
    same(result['plan_file'], bind(BASE / 'plans' / f'seed{seed}.json'), 'Run plan binding')
    same(result['experiment_plan'], bind(BASE / 'diagnostic_plan.json'), 'Run policy binding')
    same(result['start_update'], 0, 'Run did not start fresh')
    same(result['completed_updates'], 1200, 'Run budget')
    same(result['restored_from'], None, 'Unexpected recovery')
    same(result['restore_verification'], None, 'Unexpected restore state')
    same(result['initial_state_hashes'], plan['initial_state_hashes_expected'][arm], 'Seeded initialization')
    same(contract['initial_state_hashes'], result['initial_state_hashes'], 'Contract initialization')
    for key, expected in [('seed', seed), ('arm', arm), ('total_updates', 1200), ('batch_size', 2),
                          ('plan_sha256', num.canonical_hash(plan)),
                          ('data_sha256', num.nested_hash(clip_bindings)),
                          ('clip_bindings', clip_bindings), ('schedule_sha256', num.nested_hash(plan['schedule'])),
                          ('runtime', plan['runtime'])]:
        same(identity[key], expected, 'Training identity ' + key)
    same(result['identity_sha256'], num.nested_hash(identity), 'Identity digest')
    same(result['plan_sha256'], identity['plan_sha256'], 'Run plan digest')
    same(result['runtime'], identity['runtime'], 'Run runtime')
    same(contract['start_update'], 0, 'Contract fresh position')
    same(contract['stop_update'], 1200, 'Contract final position')
    same(contract['resume_checkpoint'], None, 'Contract recovery')
    same(contract['resume_checkpoint_sha256'], None, 'Contract recovery binding')
    same(identity['optimization'], {'optimizer': 'Adam', 'lr': .001, 'betas': [.9, .999],
         'eps': 1e-8, 'weight_decay': 0., 'gradient_clip': 1., 'accumulation_clips': 2,
         'lambda': 0. if arm == 'A_ctc' else 1., 'ctc_device': 'cpu', 'precision': 'float32'},
         'Optimization contract')
    for reference in identity['code_bindings']:
        verify(reference)
    for key in ('dev_inputs_loaded', 'held_inputs_loaded', 'test_inputs_loaded', 'audio_used',
                'teacher_model_loaded', 'service_weights_updated'):
        same(result[key], False, 'Forbidden run scope ' + key)
    history = result['updates']
    accounting = num.validate_history(history, plan, arm, rows)
    same(accounting['per_clip_presentations'], {sid: 300 for sid in plan['ids']}, 'Exactly 300 presentations each')
    same([item['completed_updates'] for item in result['checkpoints']], [300, 600, 900, 1200],
         'Checkpoint milestone count')
    same(result['checkpoint'], result['checkpoints'][-1], 'Final checkpoint selection')
    same(result['final_state_hashes'], history[-1]['state_hashes'], 'Final history state')
    for checkpoint in result['checkpoints']:
        payload = num.load_checkpoint(checkpoint)
        completed = payload['completed_updates']
        same(payload['identity'], identity, 'Checkpoint run identity')
        same(payload['initial_state_hashes'], result['initial_state_hashes'], 'Checkpoint initial state')
        same(payload['history'], history[:completed], 'Checkpoint full history prefix')
        same(payload['state_hashes'], history[completed - 1]['state_hashes'], 'Checkpoint final history state')
        num.validate_optimizer(payload['state'], arm, completed)
        same(num.tensor_hash(payload['state']['model']), history[completed - 1]['model_sha256'], 'Model tensor hash')
        same(num.tensor_hash(payload['state']['adapter']), history[completed - 1]['adapter_sha256'], 'Adapter tensor hash')
        if arm == 'A_ctc':
            same(payload['state_hashes']['adapter'], result['initial_state_hashes']['adapter'], 'CTC-only adapter changed')
    segment = read(run_path / 'segment_result.json')
    for key, value in segment.items():
        same(result[key], value, 'Run result changed segment ' + key)
    return {'run_result': bind(run_path / 'run_result.json'), 'checkpoints_verified': 4,
            'processed_frames': sum(sum(record['frames']) for record in history),
            'elapsed_seconds': result['elapsed_seconds'], 'accounting': accounting,
            'initial_state_hashes': result['initial_state_hashes'],
            'checkpoint_bindings': result['checkpoints']}


def ctc_mean_numpy(log_probabilities: np.ndarray, target: list[int]) -> float:
    """Float64 CTC forward dynamic program, independent of torch.ctc_loss."""
    require(log_probabilities.ndim == 2 and log_probabilities.shape[1] == 69
            and len(log_probabilities) > 0 and np.isfinite(log_probabilities).all(), 'Invalid CTC input')
    require(target and all(type(token) is int and 1 <= token <= 68 for token in target), 'Invalid CTC target')
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    alpha = np.full(len(labels), -np.inf, dtype=np.float64)
    alpha[0] = log_probabilities[0, 0]
    alpha[1] = log_probabilities[0, target[0]]
    skip = np.zeros(len(labels), dtype=bool)
    skip[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for frame in log_probabilities[1:]:
        adjacent = np.r_[-np.inf, alpha[:-1]]
        jumped = np.r_[-np.inf, -np.inf, alpha[:-2]]
        jumped[~skip] = -np.inf
        alpha = np.logaddexp(np.logaddexp(alpha, adjacent), jumped) + frame[labels]
    return float(-np.logaddexp(alpha[-1], alpha[-2]) / len(target))


def row_diagnostics(record: dict, lp: np.ndarray) -> dict:
    """Validate saved frame probabilities and independently derive output statistics."""
    frames = record['frames']
    same(lp.shape, (frames, 69), 'Log posterior dimensions')
    same(lp.dtype, np.dtype('float32'), 'Log posterior precision')
    require(np.isfinite(lp).all() and (lp <= 1e-5).all(), 'Invalid saved log probabilities')
    maximum = lp.astype(np.float64).max(axis=1)
    logsumexp = maximum + np.log(np.exp(lp.astype(np.float64) - maximum[:, None]).sum(axis=1))
    norm_error = float(np.abs(logsumexp).max())
    require(norm_error <= 1e-4, 'Saved probabilities not normalized')
    raw = np.argmax(lp, axis=1).tolist()
    same(record['raw_argmax_ids'], raw, 'Raw argmax disagrees with saved log probabilities')
    collapsed = collapse(raw, frames)
    same(record['collapsed_ids'], collapsed, 'CTC collapse order')
    decoded = compose(collapsed)
    same(record['prediction'], decoded['prediction'], 'Independent tolerant composition')
    same(record['replacement_characters'], decoded['replacement_characters'], 'Malformed token accounting')
    same(record['strict_label_decoder_diagnostic'], decoded['strict'], 'Strict decoder diagnostic')
    independent_ctc = ctc_mean_numpy(lp, record['target_token_ids'])
    assert_near(record['ctc_loss_mean'], independent_ctc, 'Independent CTC forward loss', 1e-4)
    prediction = unicodedata.normalize('NFC', record['prediction']).strip()
    reference = unicodedata.normalize('NFC', record['reference']).strip()
    longest, runs = repeated_runs(prediction)
    histogram = lambda values: {str(key): value for key, value in Counter(values).items()}
    expected = {'frames': frames, 'blank_argmax_frames': raw.count(0),
                'blank_argmax_fraction': raw.count(0) / frames,
                'raw_token_histogram': histogram(raw), 'collapsed_token_histogram': histogram(collapsed),
                'character_histogram': histogram(prediction),
                'nonspace_character_histogram': histogram(char for char in prediction if not char.isspace()),
                'prediction_characters': len(prediction), 'reference_characters': len(reference),
                'prediction_nonspace_characters': sum(not char.isspace() for char in prediction),
                'reference_nonspace_characters': sum(not char.isspace() for char in reference),
                'longest_nonspace_same_character_run': longest, 'same_character_runs_ge5': runs}
    actual = record['output_diagnostics']
    for key, value in expected.items():
        same(actual[key], value, 'Output diagnostic ' + key)
    blank = np.exp(lp[:, 0].astype(np.float64))
    require(np.allclose(actual['blank_posterior_per_frame'], blank, rtol=0, atol=2e-7),
            'Blank probabilities disagree with saved log probabilities')
    assert_near(actual['mean_blank_posterior'], float(blank.mean()), 'Mean blank posterior', 2e-7)
    assert_near(actual['max_abs_logsumexp_error'], norm_error, 'Probability normalization diagnostic', 1e-6)
    return {'ctc_loss': independent_ctc, 'ctc_abs_difference': abs(independent_ctc - record['ctc_loss_mean'])}


def aggregate_condition(records: list[dict]) -> dict:
    """Independently recompute all integer and pooled scalar report fields."""
    successful = [row for row in records if row['status'] == 'ok']
    diagnostic = [row['output_diagnostics'] for row in successful]
    frames = sum(row['frames'] for row in diagnostic)
    blank_frames = sum(row['blank_argmax_frames'] for row in diagnostic)
    blank_sum = sum(sum(row['blank_posterior_per_frame']) for row in diagnostic)
    target_tokens = sum(len(row['target_token_ids']) for row in records)
    token_edits = sum(num.edit_distance(row['target_token_ids'], row['collapsed_ids']) for row in records)
    histograms = {}
    for key in ('raw_token_histogram', 'collapsed_token_histogram', 'character_histogram', 'nonspace_character_histogram'):
        merged = Counter()
        for row in diagnostic:
            merged.update(row[key])
        histograms[key] = dict(merged)
    return {'aggregate': num.aggregate_metrics(records), 'diagnostics': {
        'successful_inferences': len(successful), 'failed_inferences': len(records) - len(successful),
        'ctc_loss_count': len(successful),
        'ctc_loss_mean_over_successful_clips': float(np.mean([row['ctc_loss_mean'] for row in successful])) if successful else None,
        'token_edits': token_edits, 'target_tokens': target_tokens,
        'token_error_percent': 100 * token_edits / target_tokens,
        'token_exact_sentences': sum(row['target_token_ids'] == row['collapsed_ids'] for row in records),
        'frames': frames, 'blank_argmax_frames': blank_frames,
        'blank_argmax_fraction': blank_frames / frames if frames else None,
        'blank_posterior_sum': blank_sum, 'mean_blank_posterior': blank_sum / frames if frames else None,
        'prediction_characters': sum(row['prediction_characters'] for row in diagnostic),
        'reference_characters': sum(num.metric_counts(row['reference'], row['prediction'])['reference_characters'] for row in records),
        'replacement_characters': sum(row['prediction'].count('\ufffd') for row in records),
        'sentences_with_same_character_run_ge5': sum(repeated_runs(row['prediction'])[1] > 0 for row in successful),
        'longest_nonspace_same_character_run': max((repeated_runs(row['prediction'])[0] for row in successful), default=0),
        **histograms}}


def audit_evaluation(plan: dict, seed: int, arm: str, rows: dict, run: dict) -> dict:
    """Audit 56 saved inferences, their checkpoint identity and posterior evidence."""
    path = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
    result = read(path)
    for key, expected in [('schema', 'lipnet-step11-train-memorization-evaluation-v1'),
                          ('status', 'completed'), ('seed', seed), ('arm', arm),
                          ('plan_sha256', num.canonical_hash(plan)), ('data_sha256', plan['training_data']['data_sha256']),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')),
                          ('initial_state_hashes_verified', plan['initial_state_hashes_expected'][arm]),
                          ('ids', plan['ids']), ('diagnostic_groups', plan['diagnostic_groups']),
                          ('runtime', plan['runtime']), ('optimization_steps', 0), ('caller_rng_restored_exactly', True)]:
        same(result[key], expected, 'Evaluation contract ' + key)
    for key in ('teacher_forward', 'audio_used', 'dev_inputs_loaded', 'held_inputs_loaded',
                'test_inputs_loaded', 'service_weights_updated'):
        same(result[key], False, 'Forbidden evaluation scope ' + key)
    frozen_code = {reference['path']: reference for reference in plan['execution_code']}
    for reference in result['evaluation_code']:
        same(frozen_code[reference['path']], reference, 'Evaluator was not frozen')
        verify(reference)
    same([item['completed_updates'] for item in result['milestones']], list(MILESTONES), 'Milestone positions')
    archive_ref = result['log_posteriors_archive']
    verify(archive_ref)
    groups = {'all8': plan['ids'], **plan['diagnostic_groups']}
    counters, keys, max_ctc_error, condition_count = Counter(), set(), 0., 0
    milestone_results = {}
    with np.load(archive_ref['path'], allow_pickle=False) as archive:
        for milestone in result['milestones']:
            update = milestone['completed_updates']
            expected_checkpoint = None if update == 0 else run['checkpoint_bindings'][update // 300 - 1]
            same(milestone['checkpoint'], expected_checkpoint, 'Evaluation chose different checkpoint')
            expected_state = plan['initial_state_hashes_expected'][arm]['model'] if update == 0 else expected_checkpoint['state_hashes']['model']
            same(milestone['model_state_before'], expected_state, 'Evaluated model identity')
            same(milestone['model_state_after'], expected_state, 'Evaluation modified student')
            same(milestone['model_state_unchanged'], True, 'Evaluation state preservation')
            conditions = ['normal', 'static', 'shuffled'] if update == 1200 else ['normal']
            same(list(milestone['conditions']), conditions, 'Temporal condition scope')
            milestone_results[str(update)] = {}
            for condition, value in milestone['conditions'].items():
                condition_count += 1
                records = value['records']
                same([record['id'] for record in records], plan['ids'], 'Evaluation row order/scope')
                for position, record in enumerate(records):
                    row = rows[record['id']]
                    counters['records'] += 1
                    for key, expected in [('speaker_id', row['speaker_id']), ('source_stem', row['source_stem']),
                                          ('split', 'train'), ('source_split', 'train'), ('reference', row['text']),
                                          ('frames', row['frames']), ('condition', condition),
                                          ('target_token_ids', row['token_ids']),
                                          ('group', 'old4' if row['id'] in groups['old4'] else 'new4')]:
                        same(record[key], expected, 'Evaluation row identity ' + key)
                    require(record['status'] in ('ok', 'failed'), 'Unknown inference status')
                    if record['status'] == 'ok':
                        same(record['error'], None, 'Successful inference error')
                        key = f'u{update:04d}_{condition}_{position}'
                        same(record['log_posteriors_key'], key, 'Posterior archive key')
                        require(key not in keys, 'Duplicate posterior evidence')
                        keys.add(key)
                        posterior = row_diagnostics(record, archive[key])
                        max_ctc_error = max(max_ctc_error, posterior['ctc_abs_difference'])
                        counters['posterior_frames'] += record['frames']
                    else:
                        counters['failed'] += 1
                        require(isinstance(record['error'], str) and record['error'], 'Missing failure reason')
                        for key, expected in [('prediction', ''), ('raw_argmax_ids', []), ('collapsed_ids', []),
                                              ('log_posteriors_key', None), ('output_diagnostics', None),
                                              ('ctc_loss_mean', None), ('replacement_characters', 0),
                                              ('strict_label_decoder_diagnostic', None)]:
                            same(record[key], expected, 'Failed inference fabricated output ' + key)
                    if record['temporal'] is not None:
                        temporal = record['temporal']
                        positions = num.temporal_order(row['frames'], condition, row['id'])
                        same(temporal['source_positions'], positions, 'Temporal manipulation')
                        same(temporal['source_positions_sha256'], num.nested_hash(positions), 'Temporal position hash')
                        same(temporal['frames'], row['frames'], 'Temporal frame count')
                        same(temporal['static_position'], row['frames'] // 2 if condition == 'static' else None, 'Static frame')
                    else:
                        same(record['status'], 'failed', 'Successful inference lacks temporal evidence')
                    require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid inference wall time')
                    same(record['metrics'], num.metric_counts(row['text'], record['prediction']), 'Independent character/word edits')
                    same(record['token_metrics'], {'token_edits': num.edit_distance(row['token_ids'], record['collapsed_ids']),
                         'target_tokens': len(row['token_ids']), 'exact': row['token_ids'] == record['collapsed_ids']}, 'Independent token edits')
                same(set(value['groups']), set(groups), 'Evaluation aggregate groups')
                for group, ids in groups.items():
                    expected = aggregate_condition([record for record in records if record['id'] in ids])
                    same(value['groups'][group], expected, 'Independent aggregate ' + group)
                milestone_results[str(update)][condition] = value['groups']
        same(set(archive.files), keys, 'Missing/extra posterior arrays')
    same(result['record_count'], counters['records'], 'Evaluation record count')
    same(counters['records'], 56, 'Fixed evaluation rows')
    same(result['posterior_arrays'], len(keys), 'Posterior count')
    same(condition_count, 7, 'Milestone/condition count')
    return {'evaluation': bind(path), 'record_count': counters['records'], 'failed_inputs': counters['failed'],
            'posterior_arrays': len(keys), 'posterior_frames': counters['posterior_frames'],
            'independent_ctc_max_abs_difference': max_ctc_error, 'milestones': milestone_results,
            'final_success_8_of_8': milestone_results['1200']['normal']['all8']['aggregate']['exact_sentences'] == 8}


def audit_contract() -> tuple[dict, dict, dict, list[dict], dict]:
    """Verify frozen plans, selection, runtime pairing and source preservation."""
    policy_path = BASE / 'diagnostic_plan.json'
    policy = read(policy_path, POLICY_SHA)
    frozen = read(BASE / 'plans/freeze_manifest.json')
    same(frozen['schema'], 'lipnet-step11-freeze-manifest-v1', 'Freeze schema')
    same(frozen['status'], 'frozen', 'Freeze state')
    same(frozen['experiment_plan'], bind(policy_path), 'Frozen policy hash')
    same(frozen['plans'], [bind(BASE / 'plans' / f'seed{seed}.json') for seed in SEEDS], 'Frozen plan hashes')
    same(frozen['optimizer_updates_planned'], 4800, 'Frozen update budget')
    same(frozen['optimizer_updates_executed'], 0, 'Freeze must precede optimization')
    for key, expected in [('schema', 'lipnet-step11-memorization-protocol-v1'), ('status', 'frozen_before_training'),
                          ('seeds', list(SEEDS)), ('arms', list(ARMS)), ('updates_per_run', 1200), ('batch_size', 2),
                          ('checkpoint_interval', 300), ('evaluation_milestones', list(MILESTONES)),
                          ('normal_train_rows_per_run', 40), ('final_temporal_conditions', ['normal', 'static', 'shuffled']),
                          ('all_inference_rows_planned', 224), ('optimizer_updates_planned', 4800),
                          ('presentations_per_run', 2400), ('presentations_per_clip_per_run', 300),
                          ('fixed_budget_without_early_stop_or_extension', True), ('generalization_claim_allowed', False),
                          ('development_or_test_evaluation', False), ('new_human_approvals_added', 0),
                          ('audio_used', False), ('teacher_model_loaded', False), ('service_weights_updated', False)]:
        same(policy[key], expected, 'Frozen policy ' + key)
    handoff, rows, entries = source_metadata()
    ids, groups = selection_ids(list(entries.values()), rows)
    selection = policy['selection']
    same(selection['ids'], ids, 'Independent deterministic clip selection')
    same(selection['diagnostic_groups'], groups, 'Independent diagnostic groups')
    same(selection['eligible_training_ids'], handoff['provisional_train_ids'], 'Eligible train metadata')
    excluded = handoff['held_new_train_ids'] + handoff['development_groups']['old71_same_training_speakers'] + handoff['development_groups']['new24_all_original_items']
    same(selection['excluded_ids'], excluded, 'Held/development exclusions')
    require(not set(ids) & set(excluded), 'Selected held/development ID')
    same(selection['selection_source_handoff'], bind(HANDOFF), 'Selection source anchor')
    same(policy['selection_source_handoff'], bind(HANDOFF), 'Policy source anchor')
    same(selection['selection_uses_transcripts_or_model_results'], False, 'Outcome-based selection')
    same(selection['new_human_approvals_added'], 0, 'Fabricated human approval')
    same(selection['generalization_estimate'], False, 'Train score described as generalization')
    same([record['id'] for record in selection['records']], ids, 'Selection record order')
    for record in selection['records']:
        sid, row = record['id'], rows[record['id']]
        entry = entries[sid]
        minimum = len(row['token_ids']) + sum(a == b for a, b in zip(row['token_ids'], row['token_ids'][1:]))
        require(row['frames'] >= minimum, 'Selected CTC target infeasible')
        for key, expected in [('frames', row['frames']), ('tokens', row['token_ids']), ('text', row['text']),
                              ('role', entry['role']), ('generation', entry['generation']), ('speaker_id', row['speaker_id']),
                              ('source_stem', row['source_stem']), ('logical_record_sha256', entry['logical_record_sha256']),
                              ('input', entry['input']), ('teacher_feature', entry['teacher_feature']),
                              ('ctc_min_input_frames', minimum), ('ctc_slack', row['frames'] - minimum)]:
            same(record[key], expected, 'Selection logical record ' + key)
    clip_bindings, payload_refs = selected_payloads(ids, rows, entries)
    expected_npz = [entries[sid][kind] for sid in handoff['provisional_train_ids'] for kind in ('input', 'teacher_feature')]
    num.validate_npz_binding_scope(policy['data_bindings'], expected_npz)
    # All hashes are checked after the allowed NPZ scope has been established.
    artifacts = policy['execution_code'] + policy['data_bindings'] + [bind(HANDOFF), bind(policy_path),
                  bind(BASE / 'plans/freeze_manifest.json')] + frozen['plans']
    preservation = policy['preservation']
    weights = ROOT / 'backend/models/weights'
    same(preservation['weights_directory'], str(weights), 'Service weights directory')
    same(preservation['weights_inventory'], [str(path) for path in sorted((p for p in weights.rglob('*') if p.is_file()), key=str)],
         'Service weights inventory changed')
    for path in preservation['absent_paths']:
        path = num.local(path)
        require(path.is_relative_to(weights) and not path.exists(), 'Previously absent service weight appeared')
    artifacts += preservation['artifacts']
    for reference in artifacts:
        verify(reference)
    plans = {}
    for seed in SEEDS:
        plan = read(BASE / 'plans' / f'seed{seed}.json', PLAN_SHAS[seed])
        for key, expected in [('schema', 'lipnet-step11-memorization-training-plan-v1'), ('seed', seed),
                              ('arms', list(ARMS)), ('updates', 1200), ('batch_size', 2), ('checkpoint_interval', 300),
                              ('ids', ids), ('diagnostic_groups', groups), ('experiment_plan', bind(policy_path)),
                              ('selection_source_handoff', bind(HANDOFF)), ('service_weights_updated', False),
                              ('training_data', {'data_sha256': num.nested_hash(clip_bindings), 'clips': clip_bindings}),
                              ('schedule', num.independent_schedule(ids, seed, 1200))]:
            same(plan[key], expected, 'Independent execution plan ' + key)
        for key in ('runtime', 'execution_code', 'data_bindings', 'optimizer', 'loss'):
            same(plan[key], policy[key], 'Plan-policy identity ' + key)
        same(plan['plan_sha256'], num.canonical_hash(plan), 'Canonical plan hash')
        initial = plan['initial_state_hashes_expected']
        same(initial, policy['initial_state_hashes_expected'][str(seed)], 'Initial state binding')
        for component in ('model', 'adapter', 'rng'):
            same(initial['A_ctc'][component], initial['B_feature'][component], 'Paired initial ' + component)
        require(initial['A_ctc']['optimizer'] != initial['B_feature']['optimizer'], 'Different optimizer scopes not represented')
        plans[seed] = plan
    return policy, plans, rows, artifacts, clip_bindings


def audit_summary(runs: dict, evaluations: dict) -> dict:
    """Compare every report number to independently checked raw artifacts."""
    path = BASE / 'results/comparison.json'
    summary = read(path)
    same(summary['schema'], 'lipnet-step11-memorization-comparison-v1', 'Comparison schema')
    same(summary['status'], 'completed', 'Comparison incomplete')
    same(summary['experiment_plan'], bind(BASE / 'diagnostic_plan.json'), 'Comparison policy')
    verify(summary['generator'])
    expected_keys = [(seed, arm) for seed in SEEDS for arm in ARMS]
    same([(row['seed'], row['arm']) for row in summary['models']], expected_keys, 'Comparison model omissions/order')
    expected_artifacts = []
    for row in summary['models']:
        key = row['seed'], row['arm']
        run, evaluation = runs[key], evaluations[key]
        same(row['evaluation'], evaluation['evaluation'], 'Comparison evaluation binding')
        same(row['run_result'], run['run_result'], 'Comparison run binding')
        same(row['success_8_of_8_exact_at_fixed_final'], evaluation['final_success_8_of_8'], 'Comparison success rule')
        expected_final = evaluation['milestones']['1200']
        same(row['final_normal_exact_sentences'], expected_final['normal']['all8']['aggregate']['exact_sentences'], 'Final exact count')
        same(row['final_conditions'], expected_final, 'Comparison all final conditions')
        same(row['normal_curve'], [{'update': update, 'groups': evaluation['milestones'][str(update)]['normal']}
                                  for update in MILESTONES], 'Comparison all normal milestones')
        original = read(run['run_result']['path'])
        history = original['updates']
        training = row['training']
        for name, expected in [('updates', 1200), ('presentations', 2400),
                               ('exposure_per_clip', run['accounting']['per_clip_presentations']),
                               ('processed_frames', run['processed_frames']), ('elapsed_seconds', run['elapsed_seconds']),
                               ('max_cuda_allocated_bytes', run['accounting']['peak_allocated_bytes']),
                               ('max_cuda_reserved_bytes', run['accounting']['peak_reserved_bytes']),
                               ('gradient_norm_after_clipping_max', max(r['gradient_norm_after_clipping'] for r in history)),
                               ('initial_state_hashes', run['initial_state_hashes']),
                               ('final_state_hashes', original['final_state_hashes'])]:
            same(training[name], expected, 'Comparison training accounting ' + name)
        for name, subset in [('first100', history[:100]), ('last100', history[-100:])]:
            for metric in ('loss', 'ctc_loss', 'feature_loss', 'gradient_norm_before_clipping'):
                assert_near(training['loss'][name][metric], math.fsum(r[metric] for r in subset) / 100,
                            'Comparison loss average', 1e-12)
        expected_artifacts.extend([evaluation['evaluation'], run['run_result'],
                                   read(evaluation['evaluation']['path'])['log_posteriors_archive']])
    same(summary['artifacts'], expected_artifacts, 'Comparison artifact references')
    for reference in expected_artifacts:
        verify(reference)
    for key, expected in [('new_optimizer_updates', 4800), ('inference_rows', 224),
                          ('training_elapsed_seconds_sum', sum(run['elapsed_seconds'] for run in runs.values())),
                          ('processed_frames_sum', sum(run['processed_frames'] for run in runs.values())),
                          ('fixed_final_success_runs', sum(value['final_success_8_of_8'] for value in evaluations.values())),
                          ('failures', sum(value['failed_inputs'] for value in evaluations.values())),
                          ('generalization_estimate', False), ('service_weights_updated', False)]:
        same(summary[key], expected, 'Comparison total ' + key)
    return bind(path)


def main() -> None:
    """Write one immutable audit after all four runs and all fixed evaluations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'results/independent_audit.json')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite existing audit')
    started = time.perf_counter()
    audit_sources = [bind(Path(__file__)), bind(BASE / 'test_independent_audit.py'), bind(NUMERICAL)]
    num.torch.set_num_threads(2)
    policy, plans, rows, artifacts, clips = audit_contract()
    artifacts += audit_sources
    if args.preflight:
        for reference in artifacts:
            verify(reference)
        result = {'schema': 'lipnet-step11-independent-preflight-v1', 'status': 'passed',
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'audit_sources': audit_sources,
                  'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'plans': [bind(BASE / 'plans' / f'seed{seed}.json') for seed in SEEDS],
                  'selection_ids': policy['selection']['ids'], 'diagnostic_groups': policy['selection']['diagnostic_groups'],
                  'student_frames': sum(clip['frames'] for clip in clips.values()), 'data_sha256': num.nested_hash(clips),
                  'source_bindings_verified': len(artifacts), 'decoded_input_npz': 8, 'decoded_teacher_feature_npz': 8,
                  'development_held_test_payloads_opened': 0, 'raw_media_opened': 0,
                  'selection_and_fixed_schedule_reconstructed': True, 'service_weights_preserved_at_preflight': True,
                  'scope': 'Selection, schedules and source hashes only; no completed-run or inference claim.',
                  'elapsed_seconds': time.perf_counter() - started}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'status': result['status'], 'output': bind(args.output)}))
        return
    runs, evaluations = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            runs[seed, arm] = audit_training(plans[seed], seed, arm, rows, clips)
            evaluations[seed, arm] = audit_evaluation(plans[seed], seed, arm, rows, runs[seed, arm])
    for seed in SEEDS:
        a, b = runs[seed, 'A_ctc'], runs[seed, 'B_feature']
        same(a['processed_frames'], b['processed_frames'], 'Paired frame presentations')
        for component in ('model', 'adapter', 'rng'):
            same(a['initial_state_hashes'][component], b['initial_state_hashes'][component], 'Paired fresh run state')
        ae = read(evaluations[seed, 'A_ctc']['evaluation']['path'])['milestones'][0]['conditions']['normal']['records']
        be = read(evaluations[seed, 'B_feature']['evaluation']['path'])['milestones'][0]['conditions']['normal']['records']
        for ar, br in zip(ae, be, strict=True):
            for field in ('raw_argmax_ids', 'collapsed_ids', 'prediction', 'ctc_loss_mean', 'output_diagnostics', 'metrics', 'token_metrics'):
                same(ar[field], br[field], 'Paired initial model inference')
    summary = audit_summary(runs, evaluations)
    for reference in artifacts:
        verify(reference)
    result = {'schema': 'lipnet-step11-independent-audit-v1', 'status': 'passed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
              'test_source': bind(BASE / 'test_independent_audit.py'), 'numerical_ancestor': bind(NUMERICAL),
              'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'comparison': summary,
              'selection_reconstructed_independently': True, 'paired_initialization_and_schedule_verified': True,
              'source_and_code_hashes_unchanged_before_and_after_audit': True,
              'source_bindings_checked': len(artifacts), 'decoded_input_npz': 8, 'decoded_teacher_feature_npz': 8,
              'development_held_test_payloads_opened': 0, 'raw_media_opened': 0,
              'inference_rows_independently_checked': sum(value['record_count'] for value in evaluations.values()),
              'saved_posterior_arrays_checked': sum(value['posterior_arrays'] for value in evaluations.values()),
              'saved_posterior_frames_checked': sum(value['posterior_frames'] for value in evaluations.values()),
              'ctc_independent_float64_forward_max_abs_difference': max(value['independent_ctc_max_abs_difference'] for value in evaluations.values()),
              'complete_checkpoints_independently_checked': sum(value['checkpoints_verified'] for value in runs.values()),
              'optimizer_updates_verified': 4800, 'utterance_presentations_verified': 9600,
              'processed_frames_verified': sum(value['processed_frames'] for value in runs.values()),
              'failures': sum(value['failed_inputs'] for value in evaluations.values()),
              'success_runs_at_fixed_final': sum(value['final_success_8_of_8'] for value in evaluations.values()),
              'per_run': [{'seed': seed, 'arm': arm, 'training': runs[seed, arm], 'evaluation': evaluations[seed, arm]}
                          for seed in SEEDS for arm in ARMS],
              'elapsed_seconds': time.perf_counter() - started, 'service_weights_preserved': True,
              'generalization_estimate': False,
              'limitations': ['Saved posterior arithmetic is verified; this audit does not rerun the GPU model.',
                              'The loader hashed and validated510 permitted train caches; only selected8 were scheduled.',
                              'Unchanged hashes detect mutation and corruption, not author authenticity.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': bind(args.output),
                      'rows': result['inference_rows_independently_checked'],
                      'checkpoints': result['complete_checkpoints_independently_checked'],
                      'ctc_max_difference': result['ctc_independent_float64_forward_max_abs_difference']}))


if __name__ == '__main__':
    main()
