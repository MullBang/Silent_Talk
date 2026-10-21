"""Independent CPU-only audit of the same-recording held-utterance probe.

No production protocol, evaluator, model or training implementation is imported.
The hash-pinned earlier independent auditor supplies numerical arithmetic only.
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

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / 'experiments/multisentence_step12_20261007'
ANCESTOR = PRIOR / 'independent_audit.py'
ANCESTOR_SHA = 'f84e36237b5d5ffb0287a4ecffa820a96067ff5ed4c7bd5c1e14170e42573a00'
if hashlib.sha256(ANCESTOR.read_bytes()).hexdigest() != ANCESTOR_SHA:
    raise ValueError('Independent numerical ancestor changed')
spec = importlib.util.spec_from_file_location('step13_independent_ancestor', ANCESTOR)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
old, num = previous.old, previous.num
same, require = num.same, num.require
read, bind, verify, sha = num.read_json, num.binding, num.verify_binding, num.sha
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
CONDITIONS = ('normal', 'static', 'shuffled')
SPEAKERS = ('C002', 'C032', 'C282', 'C293', 'C675', 'C321')
SAME_GROUPS = ('same6', 'same_old4', 'same_new2')
DIFFERENT_GROUPS = ('different6', 'different_old4', 'different_new2')
PRIOR_SUMMARY_SHA = '7bdce41e511d012a9590e229fbe67e09f31c4f82466547821fcfe9622144ce99'


def native_positions(row: dict) -> list[int]:
    """Reconstruct source frame selection from adjusted times and source FPS."""
    require(type(row['frames']) is int and row['frames'] > 0, 'Invalid frame count')
    require(all(math.isfinite(row[k]) for k in ('start_time', 'end_time', 'video_fps')),
            'Nonfinite source timing')
    require(0 <= row['start_time'] < row['end_time'] and row['video_fps'] > 0, 'Invalid source interval')
    return [round((row['start_time'] + position / 25) * row['video_fps']) for position in range(row['frames'])]


def overlap(a: dict, b: dict) -> bool:
    """Reject both temporal overlap and source-frame reuse within one recording."""
    if a['source_stem'] != b['source_stem']:
        return False
    interval = max(a['start_time'], b['start_time']) < min(a['end_time'], b['end_time'])
    return interval or bool(set(native_positions(a)) & set(native_positions(b)))


def rank(sid: str) -> tuple[str, str]:
    return hashlib.sha256(('step13-same-recording-v1:' + sid).encode()).hexdigest(), sid


def independent_selection(prior: dict, handoff: dict, rows: dict, entries: dict,
                          provenance: dict) -> dict:
    """Select one previously unoptimized same-recording utterance per speaker."""
    train, held = prior['selection']['train_ids'], prior['selection']['holdout_ids']
    anchors = prior['selection']['retained_step11_ids']
    excluded = set(train + held)
    eligible = handoff['provisional_train_ids']
    require(len(train) == len(set(train)) == 24 and len(held) == len(set(held)) == 8,
            'Prior cohorts must retain 24 train and 8 internal holdout clips')
    require(not set(train) & set(held), 'Prior train/held overlap')
    require(len(eligible) == len(set(eligible)) == 510, 'Historical eligible pool changed')
    used_text = {previous.normalized_text(rows[sid]['text']) for sid in excluded}
    selected, different, details, events = [], [], [], []
    for anchor in anchors:
        row = rows[anchor]
        pool = sorted([sid for sid in eligible if sid not in excluded
                       and rows[sid]['speaker_id'] == row['speaker_id']
                       and rows[sid]['source_stem'] == row['source_stem']], key=rank)
        chosen = None
        for sid in pool:
            candidate, entry = rows[sid], entries[sid]
            reasons = []
            if entry['role'] not in ('old_train', 'new_provisional_train') or candidate['split'] != 'train':
                reasons.append('forbidden_original_role')
            if not previous.ctc_feasible(candidate):
                reasons.append('invalid_or_infeasible_ctc')
            target = previous.normalized_text(candidate['text'])
            if not target:
                reasons.append('empty_normalized_target')
            if target in used_text:
                reasons.append('duplicate_normalized_target')
            if provenance[sid] != provenance[anchor]:
                reasons.append('source_hash_mismatch')
            if any(overlap(candidate, rows[tid]) for tid in train):
                reasons.append('training_interval_or_frame_overlap')
            status = 'not_needed' if chosen is not None else 'skipped' if reasons else 'selected'
            events.append({'id': sid, 'speaker_id': row['speaker_id'], 'status': status, 'reasons': reasons})
            if status == 'selected':
                chosen = sid
                used_text.add(target)
        details.append({'speaker_id': row['speaker_id'], 'candidate_count': len(pool), 'selected_id': chosen})
        if chosen is not None:
            selected.append(chosen)
            partner = [sid for sid in held if rows[sid]['speaker_id'] == row['speaker_id']]
            same(len(partner), 1, 'Missing unique different-recording partner')
            different.extend(partner)
    groups = {'same6': selected, 'same_old4': [sid for sid in selected if entries[sid]['generation'] == 'old'],
              'same_new2': [sid for sid in selected if entries[sid]['generation'] == 'new'],
              'different6': different, 'different_old4': [sid for sid in different if entries[sid]['generation'] == 'old'],
              'different_new2': [sid for sid in different if entries[sid]['generation'] == 'new']}
    return {'same_ids': selected, 'different_ids': different, 'groups': groups,
            'speaker_selections': details, 'candidate_events': events}


def validate_selected_scope(selection: dict, prior: dict, rows: dict, entries: dict, provenance: dict) -> None:
    """Check pairing, no train exposure, distinct targets and source separation."""
    same_ids, different_ids = selection['same_ids'], selection['different_ids']
    same(len(same_ids), 6, 'Same-recording cohort size')
    same(len(set(same_ids + different_ids)), 12, 'Unique probe IDs')
    same([rows[sid]['speaker_id'] for sid in same_ids], list(SPEAKERS), 'Probe speaker order')
    same([rows[sid]['speaker_id'] for sid in different_ids], list(SPEAKERS), 'Paired speaker order')
    train, held = prior['selection']['train_ids'], prior['selection']['holdout_ids']
    require(set(same_ids).isdisjoint(train + held), 'New probe exposed during Step12 training/evaluation')
    require(set(different_ids).issubset(held), 'Different-recording records were not original Step12 holdouts')
    targets = [previous.normalized_text(rows[sid]['text']) for sid in train + held + same_ids]
    same(len(set(targets)), len(targets), 'Duplicate normalized transcription')
    for sid, did in zip(same_ids, different_ids, strict=True):
        row, other = rows[sid], rows[did]
        anchors = [tid for tid in train if rows[tid]['speaker_id'] == row['speaker_id']]
        same(len(anchors), 3, 'Training examples per selected speaker')
        for tid in anchors:
            same(row['source_stem'], rows[tid]['source_stem'], 'Different source stem in same cohort')
            same(provenance[sid], provenance[tid], 'Same-recording source provenance differs')
            require(not overlap(row, rows[tid]), 'Source interval/frame overlaps training data')
        require(other['source_stem'] != row['source_stem'], 'Different recording has same source stem')
        require(provenance[did]['video_sha256'] != provenance[sid]['video_sha256'], 'Different recording has same video')
        require(provenance[did]['label_sha256'] != provenance[sid]['label_sha256'], 'Different recording has same label')
        require(previous.ctc_feasible(row), 'Probe CTC target infeasible')
        require(entries[sid]['role'] in ('old_train', 'new_provisional_train'), 'Probe outside historical train510')


def independent_inputs(ids: list[str], rows: dict, entries: dict) -> tuple[dict, list[dict]]:
    """Decode only six RGB/token caches; no teacher or original media payloads."""
    values, references = {}, []
    for sid in ids:
        entry, row = entries[sid], rows[sid]
        verify(entry['input'])
        references.append(entry['input'])
        with np.load(entry['input']['path'], allow_pickle=False) as saved:
            pixels, tokens = saved['student_rgb'], saved['label_tokens']
            positions = saved['student_source_indices']
            metadata = json.loads(str(saved['metadata_json'].item()))
        same(metadata['record'], row, 'Saved logical input identity')
        same(pixels.shape, (row['frames'], 96, 96, 3), 'Student RGB shape')
        same(pixels.dtype, np.dtype('uint8'), 'Student RGB dtype')
        same(tokens.dtype, np.dtype('int64'), 'Token dtype')
        same(tokens.tolist(), row['token_ids'], 'Saved target tokens')
        same(positions.tolist(), native_positions(row), 'Saved native source frame mapping')
        values[sid] = {'pixels_sha256': num.nested_hash(pixels), 'tokens_sha256': num.nested_hash(tokens.tolist()),
                       'frames': row['frames']}
    return values, references


def audit_contract(expected_policy_sha: str | None = None) -> tuple[dict, dict, list[dict], dict]:
    """Reconstruct the frozen selection directly from historical source metadata."""
    policy_path = BASE / 'diagnostic_plan.json'
    policy = read(policy_path, expected_policy_sha)
    frozen = read(BASE / 'plans/freeze_manifest.json')
    for key, value in [('schema', 'lipnet-step13-freeze-manifest-v1'), ('status', 'frozen'),
                       ('experiment_plan', bind(policy_path)), ('optimizer_updates_executed', 0),
                       ('new_inference_rows_executed', 0)]:
        same(frozen[key], value, 'Freeze manifest ' + key)
    expected = {'schema': 'lipnet-step13-recording-protocol-v1', 'status': 'frozen_before_inference',
                'new_inference_rows_planned': 72, 'reused_inference_rows_planned': 72,
                'optimizer_updates_planned': 0, 'fixed_checkpoint_update': 3600,
                'checkpoint_selection_uses_evaluation_results': False, 'all_models_and_conditions_reported': True,
                'new_human_approvals_added': 0, 'audio_used': False, 'teacher_model_loaded': False,
                'teacher_feature_npz_opened': 0, 'service_weights_updated': False,
                'untouched_test_claim_allowed': False, 'unseen_speaker_or_webcam_claim_allowed': False}
    for key, value in expected.items():
        same(policy[key], value, 'Frozen protocol ' + key)
    summary = read(PRIOR / 'results/step12_summary.json', PRIOR_SUMMARY_SHA)
    same(summary['status'], 'completed_multisentence_internal_holdout_diagnostic', 'Prior completion status')
    prior_policy = read(PRIOR / 'diagnostic_plan.json', previous.POLICY_SHA)
    same(policy['prior_step12_policy'], bind(PRIOR / 'diagnostic_plan.json'), 'Prior policy anchor')
    same(policy['prior_step12_summary'], bind(PRIOR / 'results/step12_summary.json'), 'Prior completion anchor')
    same(policy['prior_step12_audit'], summary['artifacts']['results/independent_audit.json'], 'Prior audit anchor')
    for reference in summary['artifacts'].values():
        verify(reference)
    same(read(policy['prior_step12_audit']['path'])['status'], 'passed', 'Prior audit status')
    handoff, rows, entries, provenance, metadata_refs = previous.source_metadata()
    selected = independent_selection(prior_policy, handoff, rows, entries, provenance)
    selection = policy['selection']
    validate_selected_scope(selected, prior_policy, rows, entries, provenance)
    for key in ('same_ids', 'different_ids', 'groups'):
        same(selection[key], selected[key], 'Independent selection ' + key)
    for actual, reconstructed in zip(selection['speaker_selections'], selected['speaker_selections'], strict=True):
        for key, value in reconstructed.items():
            same(actual[key], value, 'Independent speaker selection ' + key)
        sid = actual['selected_id']
        if actual['speaker_id'] in ('C058', 'C658'):
            same(actual['candidate_count'], 0, 'Unavailable speaker has candidates')
            same(sid, None, 'Unavailable speaker was selected')
    same(len(selection['candidate_events']), len(selected['candidate_events']), 'Selection event count')
    for actual, reconstructed in zip(selection['candidate_events'], selected['candidate_events'], strict=True):
        for key in ('id', 'speaker_id', 'status'):
            same(actual[key], reconstructed[key], 'Independent candidate event ' + key)
        same(actual['rank_sha256'], rank(actual['id'])[0], 'Candidate ranking digest')
        same(bool(actual['reasons']), bool(reconstructed['reasons']), 'Candidate exclusion reason presence')
    excluded = handoff['held_new_train_ids'] + handoff['development_groups']['old71_same_training_speakers'] + handoff['development_groups']['new24_all_original_items']
    expected_selection = {'schema': 'lipnet-step13-recording-selection-v1',
        'ids': selected['same_ids'] + selected['different_ids'], 'eligible_training_ids': handoff['provisional_train_ids'],
        'excluded_step12_ids': prior_policy['selection']['train_ids'] + prior_policy['selection']['holdout_ids'],
        'historical_excluded_ids': excluded, 'selection_salt': 'step13-same-recording-v1:',
        'prediction_or_loss_used_for_selection': False, 'same_recording_clips_excluded_from_step12_training': True,
        'same_recording_time_and_native_frames_disjoint_from_training': True, 'source_separation_checked': ['source_stem', 'video_sha256', 'label_sha256'],
        'historically_used_by_prior_models': True, 'untouched_test_set': False, 'held_out_speakers': False,
        'new_human_approvals_added': 0, 'different_recording_transcripts_and_contexts_not_matched': True,
        'source_handoff': bind(previous.HANDOFF), 'prior_step12_policy': bind(PRIOR / 'diagnostic_plan.json')}
    for key, value in expected_selection.items():
        same(selection[key], value, 'Selection scope ' + key)
    anchors = {rows[sid]['speaker_id']: sid for sid in prior_policy['selection']['retained_step11_ids']}
    for pair, sid, did in zip(selection['pairs'], selected['same_ids'], selected['different_ids'], strict=True):
        speaker = rows[sid]['speaker_id']
        same(pair, {'speaker_id': speaker, 'generation': entries[sid]['generation'], 'same_id': sid,
                    'different_id': did, 'train_ids': [tid for tid in prior_policy['selection']['train_ids'] if rows[tid]['speaker_id'] == speaker],
                    'anchor_id': anchors[speaker]}, 'Pair identity')
    same([item['id'] for item in selection['records']], selection['ids'], 'Selected metadata record order')
    for record in selection['records']:
        sid, row = record['id'], rows[record['id']]
        entry = entries[sid]
        for key in ('speaker_id', 'source_stem', 'text', 'frames', 'start_time', 'end_time', 'video_fps'):
            same(record[key], row[key], 'Selected metadata ' + key)
        expected_record = {'generation': entry['generation'], 'original_role': entry['role'], 'original_split': 'train',
            'diagnostic_role': 'same_recording_holdout' if sid in selected['same_ids'] else 'different_recording_holdout',
            'normalized_target': previous.normalized_text(row['text']), 'tokens': row['token_ids'],
            'student_source_indices': native_positions(row), 'logical_record_sha256': entry['logical_record_sha256'],
            'input': entry['input'], **provenance[sid]}
        minimum = len(row['token_ids']) + sum(a == b for a, b in zip(row['token_ids'], row['token_ids'][1:]))
        expected_record.update(ctc_min_input_frames=minimum, ctc_slack=row['frames'] - minimum)
        for key, value in expected_record.items():
            same(record[key], value, 'Selected provenance ' + key)
    values, input_refs = independent_inputs(selected['same_ids'], rows, entries)
    bound = {sid: {'id': sid, 'split': 'same_recording_holdout', 'source_split': 'train',
                    'diagnostic_role': 'same_recording_holdout', 'speaker_id': rows[sid]['speaker_id'],
                    'source_stem': rows[sid]['source_stem'], 'text': rows[sid]['text'], **values[sid]}
             for sid in sorted(values)}
    same(policy['evaluation_inputs'], {'data_sha256': num.nested_hash(bound), 'clips': bound}, 'Student input tensor bindings')
    num.validate_npz_binding_scope(policy['input_bindings'], input_refs)
    same(policy['loading_scope'], {'student_input_npz_decoded': 6, 'student_frames_decoded': sum(rows[sid]['frames'] for sid in selected['same_ids']),
         'paired_npz_whole_file_hashed': 6, 'paired_teacher_compressed_bytes_read_for_file_hash': True,
         'teacher_pixel_arrays_decoded': 0, 'teacher_feature_npz_opened': 0, 'different_recording_input_npz_opened': 0,
         'training_input_npz_opened': 0, 'original_development_or_held27_or_test_payloads_opened': 0, 'source_media_opened': 0}, 'Honest loading scope')
    baseline = policy['speaker_anchor_baseline']
    expected_anchors = {speaker: {'id': anchors[speaker], 'text': rows[anchors[speaker]]['text']} for speaker in SPEAKERS}
    for key, value in [('kind', 'known_speaker_fixed_training_sentence'), ('baseline_rows', 12), ('neural_inference_rows', 0),
                       ('choice_uses_evaluation_targets', False), ('known_speaker_metadata_required', True),
                       ('evaluation_ids', selection['ids']), ('anchor_ids_by_speaker', {s: anchors[s] for s in SPEAKERS}), ('anchors', expected_anchors)]:
        same(baseline[key], value, 'Known-speaker baseline contract ' + key)
    prior_comparison = read(summary['artifacts']['results/comparison.json']['path'])
    same([(model['seed'], model['arm']) for model in policy['models']], [(seed, arm) for seed in SEEDS for arm in ARMS], 'Four fixed model order')
    references = [bind(policy_path), bind(BASE / 'plans/freeze_manifest.json'), *metadata_refs, *input_refs,
                  *summary['artifacts'].values(), *policy['execution_code'], *policy['preservation']['artifacts']]
    for model, prior_model in zip(policy['models'], prior_comparison['models'], strict=True):
        same(model['run_result'], prior_model['run_result'], 'Original run binding')
        same(model['prior_evaluation'], prior_model['evaluation'], 'Original evaluation binding')
        source = read(model['prior_evaluation']['path'])
        same(model['prior_posteriors'], source['log_posteriors_archive'], 'Original posterior binding')
        same(model['checkpoint'], source['milestones'][-1]['checkpoint'], 'Fixed final checkpoint binding')
        references.extend([model[key] for key in ('run_result', 'prior_evaluation', 'prior_posteriors', 'plan')])
        references.extend([{'path': model['checkpoint']['path'], 'sha256': model['checkpoint']['sha256']},
                           {'path': model['checkpoint']['sidecar'], 'sha256': model['checkpoint']['sidecar_sha256']}])
    for reference in references:
        verify(reference)
    previous.verify_preservation(policy['preservation'])
    return policy, rows, references, selected


def require_zero_exposure(history: list[dict], schedule: list, ids: list[str]) -> None:
    """Check actual executed clip IDs as well as the planned schedule."""
    require(set(ids).isdisjoint(sid for batch in schedule for sid in batch), 'Probe ID appears in training schedule')
    require(set(ids).isdisjoint(sid for update in history for sid in update['ids']), 'Probe ID appears in executed history')


def audit_checkpoint(model: dict, ids: list[str], rows: dict) -> dict:
    """Restricted CPU load checks all state and all 3,600 executed update records."""
    seed, arm = model['seed'], model['arm']
    require(seed in SEEDS and arm in ARMS, 'Unexpected model identity')
    same(model['plan']['path'], str(PRIOR / 'plans' / f'seed{seed}.json'), 'Original seed plan path')
    same(model['plan']['sha256'], previous.PLAN_SHAS[seed], 'Original plan anchor')
    for name in ('plan', 'run_result', 'prior_evaluation', 'prior_posteriors'):
        verify(model[name])
    plan, run = read(model['plan']['path']), read(model['run_result']['path'])
    same(plan['plan_sha256'], num.canonical_hash(plan), 'Original canonical plan digest')
    same(plan['schedule'], num.independent_schedule(plan['ids'], seed, 3600), 'Original training schedule')
    same(run['seed'], seed, 'Run seed')
    same(run['arm'], arm, 'Run arm')
    same(run['completed_updates'], 3600, 'Final update position')
    same(run['checkpoint'], model['checkpoint'], 'Final checkpoint selection')
    same(run['checkpoints'][-1], model['checkpoint'], 'Fixed final checkpoint')
    require_zero_exposure(run['updates'], plan['schedule'], ids)
    accounting = num.validate_history(run['updates'], plan, arm, rows)
    same(accounting['per_clip_presentations'], {sid: 300 for sid in plan['ids']}, 'Training exposure per clip')
    checkpoint = num.load_checkpoint(model['checkpoint'])
    same(checkpoint['history'], run['updates'], 'Full checkpoint history')
    same(checkpoint['initial_state_hashes'], plan['initial_state_hashes_expected'][arm], 'Checkpoint initialization')
    same(checkpoint['state_hashes'], run['final_state_hashes'], 'Final checkpoint state')
    same(checkpoint['state_hashes'], run['updates'][-1]['state_hashes'], 'Final update state')
    same(checkpoint['identity']['seed'], seed, 'Checkpoint seed')
    same(checkpoint['identity']['arm'], arm, 'Checkpoint arm')
    same(checkpoint['identity']['clip_bindings'], plan['training_data']['clips'], 'Training input scope')
    same(checkpoint['identity']['plan_sha256'], num.canonical_hash(plan), 'Checkpoint plan identity')
    same(checkpoint['identity']['schedule_sha256'], num.nested_hash(plan['schedule']), 'Checkpoint schedule identity')
    require(set(checkpoint['identity']['clip_bindings']).isdisjoint(ids), 'Probe tensors in checkpoint optimization scope')
    num.validate_optimizer(checkpoint['state'], arm, 3600)
    return {'seed': seed, 'arm': arm, 'checkpoint': model['checkpoint'], 'updates_verified': 3600,
            'probe_optimization_presentations': 0, 'model_state_hash': checkpoint['state_hashes']['model']}


def validate_row(record: dict, row: dict, condition: str, posterior: np.ndarray | None) -> dict:
    """Validate row identities and recalculate decoding, edits, diagnostics and CTC."""
    for key, expected in [('speaker_id', row['speaker_id']), ('source_stem', row['source_stem']),
                          ('source_split', 'train'), ('reference', row['text']), ('frames', row['frames']),
                          ('condition', condition), ('target_token_ids', row['token_ids'])]:
        same(record[key], expected, 'Inference row identity ' + key)
    require(record['status'] in ('ok', 'failed'), 'Unknown inference status')
    difference = 0.
    if record['status'] == 'ok':
        same(record['error'], None, 'Successful inference has error')
        require(posterior is not None, 'Missing posterior evidence')
        difference = old.row_diagnostics(record, posterior)['ctc_abs_difference']
    else:
        require(isinstance(record['error'], str) and record['error'], 'Missing failure reason')
        for key, expected in [('prediction', ''), ('raw_argmax_ids', []), ('collapsed_ids', []),
                              ('log_posteriors_key', None), ('output_diagnostics', None), ('ctc_loss_mean', None),
                              ('replacement_characters', 0), ('strict_label_decoder_diagnostic', None)]:
            same(record[key], expected, 'Failure fabricated result ' + key)
    if record['temporal'] is None:
        same(record['status'], 'failed', 'Successful inference lacks temporal evidence')
    else:
        temporal, positions = record['temporal'], num.temporal_order(row['frames'], condition, row['id'])
        for key, expected in [('source_positions', positions), ('source_positions_sha256', num.nested_hash(positions)),
                              ('frames', row['frames']), ('static_position', row['frames'] // 2 if condition == 'static' else None),
                              ('shuffle_policy', 'sha256(step07-order-v1: + sample_id), first8 little endian, numpy.default_rng.permutation'
                               if condition == 'shuffled' else None)]:
            same(temporal[key], expected, 'Temporal manipulation ' + key)
    require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid inference elapsed time')
    same(record['metrics'], num.metric_counts(row['text'], record['prediction']), 'Character and word edits')
    same(record['token_metrics'], {'token_edits': num.edit_distance(row['token_ids'], record['collapsed_ids']),
                                  'target_tokens': len(row['token_ids']), 'exact': row['token_ids'] == record['collapsed_ids']},
         'Token edits')
    return {'ctc_abs_difference': difference, 'failed': int(record['status'] == 'failed'),
            'posterior_frames': row['frames'] if record['status'] == 'ok' else 0}


def validate_group_aggregates(value: dict, groups: dict, names: tuple) -> None:
    same(list(value['groups']), list(names), 'Group order/scope differs')
    for group in names:
        records = [record for record in value['records'] if record['id'] in groups[group]]
        same([record['id'] for record in records], groups[group], 'Group membership/order differs')
        same(value['groups'][group], old.aggregate_condition(records), 'Independent group aggregate ' + group)


def validate_reused_records(records: list[dict], original: list[dict], ids: list[str]) -> None:
    by_id = {row['id']: row for row in original}
    require(len(by_id) == len(original), 'Duplicate original result row')
    same(records, [by_id[sid] for sid in ids], 'Reused Step12 rows changed')


def verify_frozen_sources(references: list[dict], execution_code: list[dict]) -> None:
    require(len({item['path'] for item in references}) == len(references), 'Duplicate evaluator dependency')
    frozen = {item['path']: item for item in execution_code}
    require(references, 'No evaluator provenance')
    paths = [BASE / 'evaluate.py', BASE / 'test_evaluate.py']
    expected = [frozen.get(str(path)) for path in paths] + previous.expected_evaluation_sources(frozen)
    require(all(item is not None for item in expected), 'Missing frozen evaluator dependency')
    same(references, expected, 'Incomplete or reordered evaluator dependencies')
    for item in references:
        same(item, frozen.get(item['path']), 'Evaluator dependency not frozen')
        verify(item)


def audit_evaluation(policy: dict, model: dict, rows: dict, checkpoint: dict, preflight_ref: dict) -> dict:
    """Check 18 new plus 18 exactly reused inference rows per fixed final model."""
    seed, arm = model['seed'], model['arm']
    path = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
    result_ref = bind(path)
    result, selection = read(path, result_ref['sha256']), policy['selection']
    expected = {'schema': 'lipnet-step13-recording-probe-evaluation-v1', 'status': 'completed',
                'seed': seed, 'arm': arm, 'checkpoint': model['checkpoint'], 'original_plan': model['plan'],
                'original_run_result': model['run_result'], 'experiment_plan': bind(BASE / 'diagnostic_plan.json'),
                'prior_evaluation': model['prior_evaluation'], 'prior_posteriors': model['prior_posteriors'],
                'evaluation_inputs': policy['evaluation_inputs'], 'groups': selection['groups'],
                'new_inference_rows': 18, 'reused_inference_rows': 18, 'record_count': 36,
                'runtime': policy['runtime'], 'optimization_steps': 0, 'teacher_forward': False, 'audio_used': False,
                'teacher_feature_caches_loaded': 0, 'student_input_caches_loaded': 6,
                'dev_inputs_loaded': False, 'original_held27_inputs_loaded': False, 'test_inputs_loaded': False,
                'service_weights_updated': False, 'caller_rng_restored_exactly': True,
                'unseen_speaker_generalization_estimate': False, 'model_state_unchanged': True,
                'model_state_before': checkpoint['model_state_hash'], 'model_state_after': checkpoint['model_state_hash']}
    for key, value in expected.items():
        same(result[key], value, 'Evaluation identity/scope ' + key)
    plan = read(model['plan']['path'])
    same(result['initial_state_hashes_verified'], plan['initial_state_hashes_expected'][arm], 'Evaluator initialization')
    verify_frozen_sources(result['evaluation_code'], policy['execution_code'])
    same(result['evaluation_contract'], policy['evaluation_contract'], 'Evaluator contract differs')
    verify(result['before_inference_contract'])
    contract = read(result['before_inference_contract']['path'])
    for key, value in [('schema', 'lipnet-step13-before-inference-contract-v1'), ('seed', seed), ('arm', arm),
                       ('policy', bind(BASE / 'diagnostic_plan.json')), ('checkpoint', model['checkpoint']),
                       ('evaluation_inputs', policy['evaluation_inputs']), ('evaluation_code', result['evaluation_code']),
                       ('runtime', policy['runtime']), ('independent_preflight', preflight_ref)]:
        same(contract[key], value, 'Before-inference contract ' + key)
    preflight = read(preflight_ref['path'])
    require(datetime.fromisoformat(preflight['created_utc']) <= datetime.fromisoformat(contract['created_utc'])
            <= datetime.fromisoformat(result['created_utc']), 'Preflight/inference chronology differs')
    require(Path(preflight_ref['path']).stat().st_mtime <= Path(result['before_inference_contract']['path']).stat().st_mtime,
            'Preflight file written after inference contract')
    previous_result = read(model['prior_evaluation']['path'])
    same(previous_result['log_posteriors_archive'], model['prior_posteriors'], 'Prior posterior archive changed')
    final = previous_result['milestones'][-1]
    same(final['completed_updates'], 3600, 'Reused row checkpoint position')
    same(final['checkpoint'], model['checkpoint'], 'Reused row model differs')
    same(list(result['conditions']), list(CONDITIONS), 'Temporal condition scope')
    archive_ref = result['log_posteriors_archive']
    verify(archive_ref)
    counters, keys, maximum = Counter(), set(), 0.
    with np.load(archive_ref['path'], allow_pickle=False) as archive, np.load(model['prior_posteriors']['path'], allow_pickle=False) as previous_archive:
        for condition in CONDITIONS:
            blocks = result['conditions'][condition]
            same(list(blocks), ['same', 'different'], 'Recording condition blocks')
            same([record['id'] for record in blocks['same']['records']], selection['same_ids'], 'Same-recording row order')
            validate_reused_records(blocks['different']['records'], final['conditions'][condition]['records'], selection['different_ids'])
            for name, expected in [('reused', True), ('source_evaluation', model['prior_evaluation']),
                                   ('source_log_posteriors_archive', model['prior_posteriors'])]:
                same(blocks['different'][name], expected, 'Reused evidence binding ' + name)
            for side in ('same', 'different'):
                for index, record in enumerate(blocks[side]['records']):
                    row = rows[record['id']]
                    if side == 'same':
                        for key in ('split', 'diagnostic_role'):
                            same(record[key], 'same_recording_holdout', 'Probe role differs')
                        expected_group = 'same_old4' if record['id'] in selection['groups']['same_old4'] else 'same_new2'
                        same(record['group'], expected_group, 'Probe stratum differs')
                        if record['status'] == 'ok':
                            key = f'u3600_{condition}_{index}'
                            same(record['log_posteriors_key'], key, 'New posterior key differs')
                            require(key not in keys, 'Duplicate new posterior key')
                            keys.add(key)
                        source = archive
                    else:
                        source = previous_archive
                    posterior = source[record['log_posteriors_key']] if record['status'] == 'ok' else None
                    diagnostic = validate_row(record, row, condition, posterior)
                    maximum = max(maximum, diagnostic['ctc_abs_difference'])
                    counters[side + '_rows'] += 1
                    counters[side + '_posterior_frames'] += diagnostic['posterior_frames']
                    counters[side + '_failures'] += diagnostic['failed']
                validate_group_aggregates(blocks[side], selection['groups'], SAME_GROUPS if side == 'same' else DIFFERENT_GROUPS)
        same(set(archive.files), keys, 'Missing/extra new posterior arrays')
    same(result['posterior_arrays'], len(keys), 'New posterior count differs')
    same(counters['same_rows'], 18, 'New inference row count')
    same(counters['different_rows'], 18, 'Reused inference row count')
    verify(model['prior_evaluation'])
    verify(model['prior_posteriors'])
    verify(result_ref)
    verify(archive_ref)
    return {'seed': seed, 'arm': arm, 'evaluation': result_ref, 'posterior_archive': archive_ref,
            'counters': dict(counters), 'new_posterior_arrays': len(keys), 'maximum_ctc_abs_difference': maximum,
            'groups': {condition: {side: result['conditions'][condition][side]['groups'] for side in ('same', 'different')}
                       for condition in CONDITIONS}}


def write_json(path: Path, value: dict) -> dict:
    require(not path.exists() and path.resolve().is_relative_to(BASE), 'Use a new local audit path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def baseline_result(policy: dict) -> dict:
    """Recalculate all 12 fixed-speaker anchor outputs and their six aggregates."""
    selected, contract = policy['selection'], policy['speaker_anchor_baseline']
    rows = {row['id']: row for row in selected['records']}
    records = []
    for cohort in ('same', 'different'):
        for sid in selected[cohort + '_ids']:
            row = rows[sid]
            anchor = contract['anchors'][row['speaker_id']]
            same(anchor['id'], contract['anchor_ids_by_speaker'][row['speaker_id']], 'Baseline anchor identity')
            require(anchor['id'] not in rows, 'Baseline chosen from evaluation targets')
            records.append({'id': sid, 'speaker_id': row['speaker_id'], 'cohort': cohort,
                            'anchor_id': anchor['id'], 'reference': row['text'], 'prediction': anchor['text'],
                            'status': 'ok', 'metrics': num.metric_counts(row['text'], anchor['text'])})
    return {'kind': 'known_speaker_fixed_training_sentence', 'records': records,
            'groups': {group: num.aggregate_metrics([row for row in records if row['id'] in selected['groups'][group]])
                       for group in SAME_GROUPS + DIFFERENT_GROUPS},
            'baseline_rows': 12, 'neural_inference_rows': 0, 'requires_known_metadata_speaker_id': True,
            'deployable_unseen_speaker_recognizer': False}


def model_comparison(evaluation: dict, source: dict, selection: dict) -> dict:
    """Recalculate paired speaker results and descriptive recording differences."""
    conditions = {condition: {**evaluation['conditions'][condition]['same']['groups'],
                              **evaluation['conditions'][condition]['different']['groups']} for condition in CONDITIONS}
    changes, pairs = {}, {}
    for condition in CONDITIONS:
        changes[condition] = {}
        for suffix in ('6', '_old4', '_new2'):
            a = conditions[condition]['same' + suffix]['aggregate']
            b = conditions[condition]['different' + suffix]['aggregate']
            changes[condition][suffix] = {'cer_percentage_points': b['cer_percent'] - a['cer_percent'],
                                          'exact_sentences': b['exact_sentences'] - a['exact_sentences']}
        first = {row['id']: row for row in evaluation['conditions'][condition]['same']['records']}
        second = {row['id']: row for row in evaluation['conditions'][condition]['different']['records']}
        pairs[condition] = []
        for pair in selection['pairs']:
            a, b = first[pair['same_id']], second[pair['different_id']]
            ca = 100 * a['metrics']['character_edits'] / a['metrics']['reference_characters']
            cb = 100 * b['metrics']['character_edits'] / b['metrics']['reference_characters']
            pairs[condition].append({'speaker_id': pair['speaker_id'], 'same_id': a['id'], 'different_id': b['id'],
                'same_cer_percent': ca, 'different_cer_percent': cb, 'different_minus_same_cer_percentage_points': cb - ca,
                'same_exact': a['metrics']['exact'], 'different_exact': b['metrics']['exact']})
    return {'seed': source['seed'], 'arm': source['arm'], 'evaluation': source['evaluation'],
            'checkpoint': evaluation['checkpoint'], 'final_conditions': conditions, 'different_minus_same': changes,
            'speaker_pairs': pairs, 'new_inference_rows': 18, 'reused_inference_rows': 18,
            'failed_new_inferences': sum(record['status'] != 'ok' for condition in CONDITIONS for record in evaluation['conditions'][condition]['same']['records']),
            'failed_reused_inferences': sum(record['status'] != 'ok' for condition in CONDITIONS for record in evaluation['conditions'][condition]['different']['records'])}


def paired_comparisons(models: list[dict]) -> tuple[list[dict], dict]:
    require([(model['seed'], model['arm']) for model in models] == [(s, a) for s in SEEDS for a in ARMS], 'Incomplete model pairs')
    groups = SAME_GROUPS + DIFFERENT_GROUPS
    paired = []
    for seed in SEEDS:
        a, b = [model for model in models if model['seed'] == seed]
        paired.append({'seed': seed, 'B_minus_A': {condition: {group: {
            'cer_percentage_points': b['final_conditions'][condition][group]['aggregate']['cer_percent'] - a['final_conditions'][condition][group]['aggregate']['cer_percent'],
            'exact_sentences': b['final_conditions'][condition][group]['aggregate']['exact_sentences'] - a['final_conditions'][condition][group]['aggregate']['exact_sentences']}
            for group in groups} for condition in CONDITIONS}})
    # Each arm has exactly two runs; arithmetic means remain descriptive.
    means = {arm: {condition: {group: {metric: math.fsum(model['final_conditions'][condition][group]['aggregate'][metric]
                    for model in models if model['arm'] == arm) / 2
                    for metric in ('cer_percent', 'wer_percent', 'exact_sentences')}
                for group in groups} for condition in CONDITIONS} for arm in ARMS}
    return paired, means


def audit_summary(policy: dict, evaluations: list[dict]) -> dict:
    path = BASE / 'results/comparison.json'
    summary_ref = bind(path)
    summary = read(path, summary_ref['sha256'])
    models = [model_comparison(read(source['evaluation']['path']), source, policy['selection']) for source in evaluations]
    paired, means = paired_comparisons(models)
    expected = {'schema': 'lipnet-step13-recording-probe-comparison-v1', 'status': 'completed',
        'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'generator': bind(BASE / 'summarize.py'),
        'models': models, 'speaker_anchor_baseline': baseline_result(policy), 'paired_seed_comparisons': paired,
        'two_seed_arithmetic_means': means, 'new_inference_rows': 72, 'reused_inference_rows': 72,
        'baseline_rows': 12, 'new_optimizer_updates': 0,
        'failed_new_inferences': sum(model['failed_new_inferences'] for model in models),
        'failed_reused_inferences': sum(model['failed_reused_inferences'] for model in models),
        'matched_speakers': list(SPEAKERS), 'service_weights_updated': False, 'new_human_approvals_added': 0,
        'unseen_speaker_generalization_estimate': False, 'untouched_test_estimate': False,
        'pure_recording_environment_causal_effect': False}
    for key, value in expected.items():
        same(summary[key], value, 'Independent comparison ' + key)
    references = []
    for model in policy['models']:
        path = BASE / 'results' / f"evaluation_seed{model['seed']}_{model['arm']}.json"
        evaluation = read(path)
        references.extend([bind(path), evaluation['log_posteriors_archive'], model['prior_evaluation'], model['prior_posteriors']])
    same(summary['artifacts'], references, 'Complete comparison evidence')
    for reference in references:
        verify(reference)
    verify(summary_ref)
    return summary_ref


def source_snapshots(sources: list[dict]) -> dict:
    """Preserve the exact preflight audit source versions before any inference."""
    directory = BASE / 'audit_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for index, reference in enumerate(sources):
        path = directory / f"{index:02d}_{Path(reference['path']).name}"
        require(not path.exists(), 'Preflight source snapshot exists')
        with path.open('xb') as stream:
            stream.write(Path(reference['path']).read_bytes())
        snapshot = bind(path)
        same(snapshot['sha256'], reference['sha256'], 'Preflight source changed while saving')
        snapshots.append({'original': reference, 'snapshot': snapshot})
    return write_json(directory / 'preflight_source_snapshot.json', {'sources': snapshots})


def verify_preflight() -> dict:
    reference = bind(BASE / 'results/independent_preflight.json')
    preflight = read(reference['path'])
    for key, expected in [('schema', 'lipnet-step13-independent-preflight-v1'), ('status', 'passed'),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')),
                          ('new_inference_rows_executed', 0), ('complete_checkpoints_independently_checked', 4)]:
        same(preflight[key], expected, 'Independent preflight ' + key)
    verify(preflight['source_snapshot_manifest'])
    snapshots = read(preflight['source_snapshot_manifest']['path'])['sources']
    same([entry['original'] for entry in snapshots], preflight['audit_sources'], 'Preflight snapshot source scope')
    for entry in snapshots:
        verify(entry['snapshot'])
        same(entry['snapshot']['sha256'], entry['original']['sha256'], 'Snapshot content differs')
        require(Path(entry['snapshot']['path']).resolve().is_relative_to(BASE / 'audit_snapshots'), 'Snapshot outside audit directory')
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--expected-policy-sha')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or BASE / 'results' / ('independent_preflight.json' if args.preflight else 'independent_audit.json')
    require(not output.exists(), 'Refusing to replace existing audit')
    if args.preflight:
        require(not list((BASE / 'results').glob('evaluation_seed*')), 'Preflight must precede all inference outputs/contracts')
    started = time.perf_counter()
    num.torch.set_num_threads(2)
    policy, rows, references, selection = audit_contract(args.expected_policy_sha)
    sources = [bind(Path(__file__)), bind(BASE / 'test_independent_audit.py'), bind(ANCESTOR), bind(previous.ANCESTOR), bind(old.NUMERICAL)]
    checkpoints = [audit_checkpoint(model, selection['same_ids'] + selection['different_ids'], rows) for model in policy['models']]
    common = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'audit_sources': sources,
              'generator': bind(Path(__file__)), 'test_source': bind(BASE / 'test_independent_audit.py'),
              'same_ids': selection['same_ids'], 'different_ids': selection['different_ids'],
              'matched_speakers': list(SPEAKERS), 'selection_events_independently_checked': len(selection['candidate_events']),
              'new_optimizer_updates': 0, 'probe_optimization_presentations': 0,
              'complete_checkpoints_independently_checked': 4, 'historical_optimizer_updates_verified': 14400,
              'checkpoints': checkpoints, 'student_input_npz_decoded': 6,
              'student_frames_decoded': sum(rows[sid]['frames'] for sid in selection['same_ids']),
              'teacher_pixel_arrays_decoded': 0, 'teacher_feature_npz_opened': 0,
              'different_recording_input_npz_opened': 0, 'training_input_npz_opened': 0,
              'historical_development_held27_test_payloads_opened': 0, 'raw_media_opened': 0,
              'paired_teacher_compressed_bytes_read_for_whole_file_hash': True,
              'service_weights_preserved': True, 'unseen_speaker_generalization_estimate': False,
              'pure_recording_environment_causal_effect': False}
    if args.preflight:
        result = {'schema': 'lipnet-step13-independent-preflight-v1', **common,
                  'new_inference_rows_executed': 0, 'source_snapshot_manifest': source_snapshots(sources),
                  'scope': 'Pre-inference selection, input hashes, actual prior training exposure and fixed checkpoint audit.'}
    else:
        preflight = verify_preflight()
        evaluations = [audit_evaluation(policy, model, rows, checkpoint, preflight)
                       for model, checkpoint in zip(policy['models'], checkpoints, strict=True)]
        comparison = audit_summary(policy, evaluations)
        result = {'schema': 'lipnet-step13-independent-audit-v1', **common,
                  'comparison': comparison, 'independent_preflight': preflight,
                  'preflight_source_snapshots_verified': True, 'preflight_precedes_all_inference_contracts': True,
                  'new_neural_inference_rows_independently_checked': sum(item['counters']['same_rows'] for item in evaluations),
                  'reused_neural_inference_rows_independently_checked': sum(item['counters']['different_rows'] for item in evaluations),
                  'known_speaker_baseline_rows_independently_checked': 12,
                  'new_posterior_arrays_checked': sum(item['new_posterior_arrays'] for item in evaluations),
                  'new_posterior_frames_checked': sum(item['counters']['same_posterior_frames'] for item in evaluations),
                  'reused_posterior_frames_checked': sum(item['counters']['different_posterior_frames'] for item in evaluations),
                  'ctc_independent_float64_forward_max_abs_difference': max(item['maximum_ctc_abs_difference'] for item in evaluations),
                  'new_inference_failures': sum(item['counters'].get('same_failures', 0) for item in evaluations),
                  'reused_inference_failures': sum(item['counters'].get('different_failures', 0) for item in evaluations),
                  'evaluations': evaluations,
                  'limitations': ['Saved posterior arithmetic and bound checkpoint state are audited without rerunning GPU inference.',
                                 'Both cohorts came from historical training candidates; this is not an untouched test set.',
                                 'Matched speakers do not match utterance difficulty or isolate recording environment causally.']}
    for reference in references + sources:
        verify(reference)
    previous.verify_preservation(policy['preservation'])
    result.update(source_and_code_hashes_unchanged_before_and_after_audit=True,
                  source_bindings_verified=len(references), elapsed_seconds=time.perf_counter() - started)
    print(json.dumps({'status': 'passed', 'output': write_json(output, result)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

