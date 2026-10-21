"""Independent CPU audit of the fresh-model train24/internal-hold8 diagnostic.

Imports only hash-pinned independent numerical audit code. No production
loader, selection, model, training or evaluation implementation is imported.
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
ANCESTOR = ROOT / 'experiments/memorization_step11_20261007/independent_audit.py'
ANCESTOR_SHA = '2f241a622d267f4e6839c4bfc47f04a968ecea41936500cf8e0dc6dc98bbb799'
if hashlib.sha256(ANCESTOR.read_bytes()).hexdigest() != ANCESTOR_SHA:
    raise ValueError('Independent numerical ancestor changed')
spec = importlib.util.spec_from_file_location('step12_independent_ancestor', ANCESTOR)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
num = old.num
same, require = num.same, num.require
read, bind, verify, sha = num.read_json, num.binding, num.verify_binding, num.sha
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
MILESTONES = (0, 900, 1800, 2700, 3600)
GROUPS = ('train24', 'hold8', 'train_old12', 'train_new12', 'hold_old4', 'hold_new4')
HANDOFF, HANDOFF_SHA = old.HANDOFF, old.HANDOFF_SHA
STEP11_POLICY = ROOT / 'experiments/memorization_step11_20261007/diagnostic_plan.json'
STEP11_POLICY_SHA = old.POLICY_SHA
POLICY_SHA = '2fc10937d9c9a879e62839cac57bbe9337319dff80a3bdf71b5f344f1d701063'
PLAN_SHAS = {42: 'cb6f4d87d74292594b90e72309ba29c423003845c5f2de99f2850b735ba6b7ef',
             43: '99709b4fb83d8a2af47e325e60f1cbc421b2e86747537513d628eb70317d4473'}
PREFLIGHT_SHA = '02c97c6c077a4a9f75520b2fdf19605f88fc9c3bad124de27e6212d1ab47fdd2'


def normalized_text(text: str) -> str:
    """NFC normalization then removal of all Unicode whitespace for leakage checks."""
    return ''.join(unicodedata.normalize('NFC', text).split())


def ctc_feasible(row: dict) -> bool:
    """Check positive frames and unambiguous nonblank targets before selection."""
    tokens = row['token_ids']
    return (isinstance(tokens, list) and bool(tokens)
            and all(type(token) is int and 1 <= token <= 68 for token in tokens)
            and type(row['frames']) is int
            and row['frames'] >= len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])))


def digest_rank(sid: str, purpose: str) -> tuple[str, str]:
    """Return the prescribed opaque-ID ordering, without outcome information."""
    require(purpose in ('train', 'holdout'), 'Unknown selection purpose')
    return hashlib.sha256(('step12-multisentence-v1:' + purpose + ':' + sid).encode()).hexdigest(), sid


def independent_selection(anchors: list[str], eligible: list[str], rows: dict, entries: dict,
                          provenance: dict, event_sink: list | None = None) -> tuple[list[str], list[str], dict]:
    """Reconstruct retained anchors, same-recording additions and held recordings."""
    require(len(anchors) == len(set(anchors)) == 8, 'Eight unique anchors required')
    require(len(eligible) == len(set(eligible)) and set(anchors).issubset(eligible), 'Eligible source IDs changed')
    speakers = [rows[sid]['speaker_id'] for sid in anchors]
    require(len(set(speakers)) == 8, 'Anchor speakers must be distinct')
    used_text = set()
    for sid in anchors:
        require(entries[sid]['role'] in ('old_train', 'new_provisional_train')
                and rows[sid]['split'] == 'train' and ctc_feasible(rows[sid]), 'Invalid retained anchor')
        text = normalized_text(rows[sid]['text'])
        require(text and text not in used_text, 'Duplicate/empty retained anchor target')
        used_text.add(text)
    train = []
    for anchor in anchors:
        row = rows[anchor]
        pool = [sid for sid in eligible if sid != anchor and rows[sid]['speaker_id'] == row['speaker_id']
                and rows[sid]['source_stem'] == row['source_stem']]
        additions = []
        for sid in sorted(pool, key=lambda value: digest_rank(value, 'train')):
            candidate = rows[sid]
            tokens = candidate['token_ids']
            reasons = (['invalid_ctc_target'] if not tokens or not all(type(t) is int and 1 <= t <= 68 for t in tokens)
                       else [] if ctc_feasible(candidate) else ['ctc_infeasible'])
            if not normalized_text(candidate['text']):
                reasons.append('empty_normalized_target')
            if normalized_text(candidate['text']) in used_text:
                reasons.append('duplicate_normalized_target')
            if provenance[sid] != provenance[anchor]:
                reasons.append('same_stem_source_hash_mismatch')
            status = 'not_needed' if len(additions) == 2 else 'skipped' if reasons else 'selected'
            if event_sink is not None:
                event_sink.append({'scope': 'train', 'anchor_id': anchor, 'id': sid,
                                   'rank_sha256': digest_rank(sid, 'train')[0], 'status': status, 'reasons': reasons})
            if status == 'selected':
                additions.append(sid)
                used_text.add(normalized_text(candidate['text']))
        require(len(additions) == 2, 'Insufficient fixed-speaker same-recording train additions')
        train.extend([anchor, *additions])
    train_stems = {rows[sid]['source_stem'] for sid in train}
    train_video = {provenance[sid]['video_sha256'] for sid in train}
    train_label = {provenance[sid]['label_sha256'] for sid in train}
    hold = []
    for anchor in anchors:
        row = rows[anchor]
        pool = [sid for sid in eligible if rows[sid]['speaker_id'] == row['speaker_id']
                and rows[sid]['source_stem'] != row['source_stem']]
        selected = None
        for sid in sorted(pool, key=lambda value: digest_rank(value, 'holdout')):
            candidate = rows[sid]
            tokens = candidate['token_ids']
            reasons = (['invalid_ctc_target'] if not tokens or not all(type(t) is int and 1 <= t <= 68 for t in tokens)
                       else [] if ctc_feasible(candidate) else ['ctc_infeasible'])
            if not normalized_text(candidate['text']):
                reasons.append('empty_normalized_target')
            if normalized_text(candidate['text']) in used_text:
                reasons.append('duplicate_normalized_target')
            if candidate['source_stem'] in train_stems:
                reasons.append('training_source_stem_overlap')
            if provenance[sid]['video_sha256'] in train_video:
                reasons.append('training_video_sha_overlap')
            if provenance[sid]['label_sha256'] in train_label:
                reasons.append('training_label_sha_overlap')
            status = 'not_needed' if selected is not None else 'skipped' if reasons else 'selected'
            if event_sink is not None:
                event_sink.append({'scope': 'holdout', 'anchor_id': anchor, 'id': sid,
                                   'rank_sha256': digest_rank(sid, 'holdout')[0], 'status': status, 'reasons': reasons})
            if status == 'selected':
                selected = sid
                used_text.add(normalized_text(candidate['text']))
        require(selected is not None, 'Insufficient fixed-speaker different-recording holdout')
        hold.append(selected)
    require(len(set(train)) == 24 and len(set(hold)) == 8 and set(train).isdisjoint(hold), 'Train/hold ID leakage')
    require(len({normalized_text(rows[sid]['text']) for sid in train + hold}) == 32, 'Normalized target leakage')
    groups = {'train24': train, 'hold8': hold,
              'train_old12': [sid for sid in train if entries[sid]['generation'] == 'old'],
              'train_new12': [sid for sid in train if entries[sid]['generation'] == 'new'],
              'hold_old4': [sid for sid in hold if entries[sid]['generation'] == 'old'],
              'hold_new4': [sid for sid in hold if entries[sid]['generation'] == 'new']}
    same([len(groups[key]) for key in GROUPS], [24, 8, 12, 12, 4, 4], 'Diagnostic subgroup counts')
    return train, hold, groups


def source_metadata() -> tuple[dict, dict, dict, dict, list[dict]]:
    """Read frozen JSON metadata only; no raw source or original held/dev payload."""
    handoff, rows, entries = old.source_metadata()
    provenance, refs = {}, [bind(HANDOFF), bind(STEP11_POLICY)]
    for generation in ('old', 'new'):
        reference = handoff['artifacts'][generation + '_inputs']
        verify(reference)
        refs.append(reference)
        for item in read(reference['path'])['records']:
            members = item['source_verification']['members']
            values = {'video_sha256': members['video']['sha256'], 'label_sha256': members['label']['sha256']}
            require(all(isinstance(digest, str) and len(digest) == 64
                        and all(char in '0123456789abcdef' for char in digest) for digest in values.values()),
                    'Invalid source provenance digest')
            provenance[item['id']] = values
    same(set(provenance), set(rows), 'Missing/extra source provenance')
    return handoff, rows, entries, provenance, refs


def selected_payloads(train: list[str], hold: list[str], rows: dict, entries: dict) -> tuple[dict, dict, list[dict]]:
    """Read32 input NPZ and24 teacher NPZ only; never open held teacher tensors."""
    training, evaluation, references = {}, {}, []
    for sid in sorted(train + hold):
        entry, row = entries[sid], rows[sid]
        require(entry['role'] in ('old_train', 'new_provisional_train') and row['split'] == 'train', 'Forbidden original source role')
        verify(entry['input'])
        references.append(entry['input'])
        with np.load(entry['input']['path'], allow_pickle=False) as saved:
            pixels, tokens = saved['student_rgb'], saved['label_tokens']
            metadata = json.loads(str(saved['metadata_json'].item()))
        same(metadata['record'], row, 'Saved logical record')
        same(pixels.shape, (row['frames'], 96, 96, 3), 'Input dimensions')
        same(pixels.dtype, np.dtype('uint8'), 'Input precision')
        same(tokens.dtype, np.dtype('int64'), 'Target precision')
        same(tokens.tolist(), row['token_ids'], 'Target tokens')
        plain = {'id': sid, 'frames': row['frames'], 'pixels_sha256': num.nested_hash(pixels),
                 'tokens_sha256': num.nested_hash(tokens.tolist())}
        role = 'train' if sid in train else 'internal_holdout'
        evaluation[sid] = {**plain, 'split': role, 'source_split': 'train', 'diagnostic_role': role,
                           'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'], 'text': row['text']}
        if sid in train:
            verify(entry['teacher_feature'])
            references.append(entry['teacher_feature'])
            with np.load(entry['teacher_feature']['path'], allow_pickle=False) as saved:
                feature = saved['features']
            same(feature.shape, (row['frames'], 512), 'Feature dimensions')
            same(feature.dtype, np.dtype('float32'), 'Feature precision')
            require(np.isfinite(feature).all() and (np.linalg.norm(feature, axis=1) > 1e-8).all(), 'Invalid feature values')
            training[sid] = {**plain, 'teacher_sha256': num.nested_hash(feature)}
    return training, evaluation, references


def verify_preservation(preservation: dict) -> None:
    """Check existing and absent service weights, configuration and prior results."""
    weights = ROOT / 'backend/models/weights'
    same(preservation['weights_directory'], str(weights), 'Service weights directory')
    same(preservation['weights_inventory'], [str(path) for path in sorted((p for p in weights.rglob('*') if p.is_file()), key=str)],
         'Service weights inventory changed')
    for path in preservation['absent_paths']:
        path = num.local(path)
        require(path.is_relative_to(weights) and not path.exists(), 'Previously absent service weight appeared')
    for reference in preservation['artifacts']:
        verify(reference)


def audit_contract(expected_policy_sha: str | None = None) -> tuple[dict, dict, dict, list[dict], dict, dict]:
    """Independently verify frozen selection, schedule, tensor scope and preservation."""
    policy_path = BASE / 'diagnostic_plan.json'
    policy = read(policy_path, expected_policy_sha)
    frozen = read(BASE / 'plans/freeze_manifest.json')
    same(frozen['schema'], 'lipnet-step12-freeze-manifest-v1', 'Freeze schema')
    same(frozen['status'], 'frozen', 'Freeze status')
    same(frozen['experiment_plan'], bind(policy_path), 'Frozen policy binding')
    same(frozen['plans'], [bind(BASE / 'plans' / f'seed{seed}.json') for seed in SEEDS], 'Frozen seed plans')
    same(frozen['optimizer_updates_planned'], 14400, 'Freeze total updates')
    same(frozen['optimizer_updates_executed'], 0, 'Freeze before updates')
    for key, expected in [('schema', 'lipnet-step12-multisentence-protocol-v1'), ('status', 'frozen_before_training'),
                          ('seeds', list(SEEDS)), ('arms', list(ARMS)), ('updates_per_run', 3600), ('batch_size', 2),
                          ('checkpoint_interval', 900), ('evaluation_milestones', list(MILESTONES)),
                          ('all_inference_rows_planned', 896), ('optimizer_updates_planned', 14400),
                          ('presentations_per_run', 7200), ('presentations_per_train_clip_per_run', 300),
                          ('holdout_used_for_stopping_tuning_or_selection', False),
                          ('fixed_budget_without_early_stop_or_extension', True),
                          ('unseen_speaker_or_webcam_claim_allowed', False), ('untouched_test_claim_allowed', False),
                          ('historical_development_or_test_evaluation', False), ('new_human_approvals_added', 0),
                          ('audio_used', False), ('teacher_model_loaded', False), ('service_weights_updated', False)]:
        same(policy[key], expected, 'Frozen policy ' + key)
    handoff, rows, entries, provenance, source_refs = source_metadata()
    prior = read(STEP11_POLICY, STEP11_POLICY_SHA)
    events = []
    train, hold, groups = independent_selection(prior['selection']['ids'], handoff['provisional_train_ids'],
                                               rows, entries, provenance, events)
    selection = policy['selection']
    baseline = policy['speaker_anchor_baseline']
    anchors_by_speaker = {rows[sid]['speaker_id']: sid for sid in prior['selection']['ids']}
    for key, expected in [('kind', 'known_speaker_fixed_training_sentence'),
                          ('anchor_ids_by_speaker', anchors_by_speaker), ('evaluation_ids', hold),
                          ('choice_uses_holdout_targets', False), ('enabled', True),
                          ('training_or_model_inference', False), ('requires_known_speaker_metadata', True),
                          ('anchor_selection_uses_holdout_targets', False),
                          ('pairs', [{'speaker_id': rows[sid]['speaker_id'], 'holdout_id': sid,
                                      'anchor_id': anchors_by_speaker[rows[sid]['speaker_id']]} for sid in hold])]:
        same(baseline[key], expected, 'Known-speaker baseline policy ' + key)
    excluded = handoff['held_new_train_ids'] + handoff['development_groups']['old71_same_training_speakers'] + handoff['development_groups']['new24_all_original_items']
    for key, expected in [('schema', 'lipnet-step12-train24-hold8-selection-v1'), ('train_ids', train),
                          ('holdout_ids', hold), ('ids', train + hold), ('retained_step11_ids', prior['selection']['ids']),
                          ('evaluation_groups', groups), ('eligible_training_ids', handoff['provisional_train_ids']),
                          ('historical_excluded_ids', excluded), ('selection_events', events),
                          ('source_handoff', bind(HANDOFF)), ('prior_step11_policy', bind(STEP11_POLICY)),
                          ('selection_salt', 'step12-multisentence-v1:'), ('held_out_speakers', False),
                          ('held_out_recordings', True), ('untouched_test_set', False), ('historically_used_by_prior_models', True),
                          ('prediction_or_loss_used_for_selection', False), ('new_human_approvals_added', 0),
                          ('source_audio_sha_checked', False), ('source_separation_checked', ['source_stem', 'video_sha256', 'label_sha256'])]:
        same(selection[key], expected, 'Independent selection ' + key)
    same([record['id'] for record in selection['records']], train + hold, 'Selected record order')
    for record in selection['records']:
        sid, row = record['id'], rows[record['id']]
        entry = entries[sid]
        minimum = len(row['token_ids']) + sum(a == b for a, b in zip(row['token_ids'], row['token_ids'][1:]))
        for key, expected in [('frames', row['frames']), ('tokens', row['token_ids']), ('text', row['text']),
                              ('normalized_target', normalized_text(row['text'])), ('original_role', entry['role']),
                              ('generation', entry['generation']), ('speaker_id', row['speaker_id']), ('original_split', 'train'),
                              ('diagnostic_role', 'train' if sid in train else 'internal_holdout'),
                              ('retained_step11_anchor', sid in prior['selection']['ids']),
                              ('source_stem', row['source_stem']), ('logical_record_sha256', entry['logical_record_sha256']),
                              ('input', entry['input']), ('teacher_feature', entry['teacher_feature'] if sid in train else None),
                              ('ctc_min_input_frames', minimum), ('ctc_slack', row['frames'] - minimum),
                              ('video_sha256', provenance[sid]['video_sha256']), ('label_sha256', provenance[sid]['label_sha256'])]:
            same(record[key], expected, 'Selected logical record ' + key)
    training_data, evaluation_data, selected_refs = selected_payloads(train, hold, rows, entries)
    evaluation_bound = {'data_sha256': num.nested_hash(evaluation_data), 'clips': evaluation_data}
    same(policy['evaluation_data'], evaluation_bound, 'Independent evaluation tensor scope')
    expected_npz = [entries[sid][kind] for sid in handoff['provisional_train_ids'] for kind in ('input', 'teacher_feature')]
    num.validate_npz_binding_scope(policy['data_bindings'], expected_npz)
    same(policy['loading_scope'], {'train_clips': 24, 'holdout_clips': 8, 'evaluation_clips': 32, 'speakers': 8,
         'train_frames': sum(rows[sid]['frames'] for sid in train), 'holdout_frames': sum(rows[sid]['frames'] for sid in hold),
         'original_training_payloads_validated': 510, 'original_teacher_targets_validated': 510,
         'internal_holdout_teacher_targets_read_for_validation_only': 8, 'internal_holdout_teacher_targets_returned': 0,
         'historical_development_payloads': 0, 'historical_held27_payloads': 0, 'test_payloads': 0, 'source_media_payloads': 0},
         'Honest loader scope')
    references = policy['execution_code'] + policy['data_bindings'] + source_refs + selected_refs + [bind(policy_path),
                  bind(BASE / 'plans/freeze_manifest.json')] + frozen['plans'] + policy['preservation']['artifacts']
    verify_preservation(policy['preservation'])
    for reference in references:
        verify(reference)
    plans = {}
    for seed in SEEDS:
        plan = read(BASE / 'plans' / f'seed{seed}.json', PLAN_SHAS[seed])
        for key, expected in [('schema', 'lipnet-step12-multisentence-training-plan-v1'), ('seed', seed),
                              ('arms', list(ARMS)), ('updates', 3600), ('batch_size', 2), ('checkpoint_interval', 900),
                              ('ids', train), ('train_ids', train), ('holdout_ids', hold), ('diagnostic_groups', groups),
                              ('evaluation_groups', groups), ('experiment_plan', bind(policy_path)), ('service_weights_updated', False),
                              ('training_data', {'data_sha256': num.nested_hash(training_data), 'clips': training_data}),
                              ('schedule', num.independent_schedule(train, seed, 3600))]:
            same(plan[key], expected, 'Independent execution plan ' + key)
        same(Counter(sid for batch in plan['schedule'] for sid in batch), Counter({sid: 300 for sid in train}), 'Exactly300 train exposures each')
        require(set(plan['training_data']['clips']).isdisjoint(hold), 'Holdout entered engine tensor binding')
        for key in ('runtime', 'execution_code', 'data_bindings', 'optimizer', 'loss', 'evaluation_data'):
            same(plan[key], policy[key], 'Plan-policy identity ' + key)
        same(plan['plan_sha256'], num.canonical_hash(plan), 'Canonical plan hash')
        initial = plan['initial_state_hashes_expected']
        same(initial, policy['initial_state_hashes_expected'][str(seed)], 'Initial state binding')
        for component in ('model', 'adapter', 'rng'):
            same(initial['A_ctc'][component], initial['B_feature'][component], 'Paired initial ' + component)
        require(initial['A_ctc']['optimizer'] != initial['B_feature']['optimizer'], 'Different optimizer scopes not represented')
        plans[seed] = plan
    return policy, plans, rows, references, training_data, evaluation_data


def audit_training(plan: dict, seed: int, arm: str, rows: dict, clip_bindings: dict) -> dict:
    """Check all3,600 real updates and each complete milestone checkpoint on CPU."""
    run_path = BASE / 'runs' / f'seed{seed}' / arm / 'continuous'
    result = read(run_path / 'run_result.json')
    contract = read(run_path / 'segment_contract.json')
    identity = contract['identity']
    for key, expected in [('status', 'completed'), ('seed', seed), ('arm', arm),
                          ('plan_file', bind(BASE / 'plans' / f'seed{seed}.json')),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')), ('start_update', 0),
                          ('completed_updates', 3600), ('restored_from', None), ('restore_verification', None),
                          ('initial_state_hashes', plan['initial_state_hashes_expected'][arm]),
                          ('internal_holdout_inputs_validated', True),
                          ('internal_holdout_teacher_features_validated_not_used', True),
                          ('internal_holdout_optimization_presentations', 0)]:
        same(result[key], expected, 'Training result ' + key)
    same(contract['initial_state_hashes'], result['initial_state_hashes'], 'Contract initialization')
    for key, expected in [('seed', seed), ('arm', arm), ('total_updates', 3600), ('batch_size', 2),
                          ('plan_sha256', num.canonical_hash(plan)), ('data_sha256', num.nested_hash(clip_bindings)),
                          ('clip_bindings', clip_bindings), ('schedule_sha256', num.nested_hash(plan['schedule'])),
                          ('runtime', plan['runtime'])]:
        same(identity[key], expected, 'Training identity ' + key)
    require(set(identity['clip_bindings']).isdisjoint(plan['holdout_ids']), 'Holdout tensor entered engine')
    same(result['identity_sha256'], num.nested_hash(identity), 'Identity digest')
    same(result['plan_sha256'], identity['plan_sha256'], 'Run plan digest')
    same(result['runtime'], identity['runtime'], 'Run runtime')
    for key, expected in [('start_update', 0), ('stop_update', 3600), ('resume_checkpoint', None), ('resume_checkpoint_sha256', None)]:
        same(contract[key], expected, 'Contract field ' + key)
    same(identity['optimization'], {'optimizer': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
         'weight_decay': 0., 'gradient_clip': 1., 'accumulation_clips': 2, 'lambda': 0. if arm == 'A_ctc' else 1.,
         'ctc_device': 'cpu', 'precision': 'float32'}, 'Optimization contract')
    for reference in identity['code_bindings']:
        verify(reference)
    for key in ('dev_inputs_loaded', 'original_held27_inputs_loaded', 'test_inputs_loaded', 'audio_used',
                'teacher_model_loaded', 'service_weights_updated'):
        same(result[key], False, 'Forbidden run scope ' + key)
    history = result['updates']
    accounting = num.validate_history(history, plan, arm, rows)
    same(accounting['per_clip_presentations'], {sid: 300 for sid in plan['ids']}, 'Exactly300 train presentations each')
    require(set(accounting['per_clip_presentations']).isdisjoint(plan['holdout_ids']), 'Holdout optimized')
    same([item['completed_updates'] for item in result['checkpoints']], [900, 1800, 2700, 3600], 'Checkpoint milestones')
    same(result['checkpoint'], result['checkpoints'][-1], 'Final checkpoint selection')
    same(result['final_state_hashes'], history[-1]['state_hashes'], 'Final history state')
    for checkpoint in result['checkpoints']:
        payload = num.load_checkpoint(checkpoint)
        completed = payload['completed_updates']
        same(payload['identity'], identity, 'Checkpoint identity')
        same(payload['initial_state_hashes'], result['initial_state_hashes'], 'Checkpoint initial state')
        same(payload['history'], history[:completed], 'Checkpoint full history prefix')
        same(payload['state_hashes'], history[completed - 1]['state_hashes'], 'Checkpoint final history state')
        num.validate_optimizer(payload['state'], arm, completed)
        same(num.tensor_hash(payload['state']['model']), history[completed - 1]['model_sha256'], 'Model tensor hash')
        same(num.tensor_hash(payload['state']['adapter']), history[completed - 1]['adapter_sha256'], 'Adapter tensor hash')
        if arm == 'A_ctc':
            same(payload['state_hashes']['adapter'], result['initial_state_hashes']['adapter'], 'CTC-only adapter changed')
    for key, value in read(run_path / 'segment_result.json').items():
        same(result[key], value, 'Run result changed original segment ' + key)
    preflight_time = datetime.fromisoformat(read(BASE / 'results/independent_preflight.json', PREFLIGHT_SHA)['created_utc']).timestamp()
    require(preflight_time <= (run_path / 'segment_contract.json').stat().st_mtime, 'Independent preflight followed run output creation')
    return {'run_result': bind(run_path / 'run_result.json'), 'checkpoints_verified': 4,
            'processed_frames': sum(sum(record['frames']) for record in history),
            'elapsed_seconds': result['elapsed_seconds'], 'accounting': accounting,
            'initial_state_hashes': result['initial_state_hashes'], 'checkpoint_bindings': result['checkpoints'],
            'internal_holdout_optimization_presentations': 0}


def expected_evaluation_sources(frozen_code: dict) -> list[dict]:
    """Require the complete known evaluator dependency set, not a subset claim."""
    paths = [BASE / 'evaluate.py', BASE / 'test_evaluate.py',
             ROOT / 'experiments/memorization_step11_20261007/evaluate.py',
             ROOT / 'experiments/memorization_step11_20261007/test_evaluate.py',
             ROOT / 'experiments/local_training_step05_20261007/evaluate_prediction_decode.py',
             ROOT / 'experiments/local_training_step05_20261007/evaluate.py',
             ROOT / 'experiments/olkavs_official/diagnostic_utils.py',
             ROOT / 'experiments/olkavs_official/evaluation_metrics.py',
             ROOT / 'experiments/local_training_step03_20261007/runner.py',
             ROOT / 'experiments/feature_transfer_step07/train_compare.py',
             ROOT / 'backend/models/baseline.py', ROOT / 'backend/models/__init__.py',
             ROOT / 'backend/models/tokenizer.py']
    require(all(str(path) in frozen_code for path in paths), 'Missing frozen evaluator dependency')
    return [frozen_code[str(path)] for path in paths]


def verify_evaluation_sources(references: list[dict], frozen_code: dict) -> None:
    """Reject missing, duplicated, reordered or substituted evaluator provenance."""
    same(references, expected_evaluation_sources(frozen_code), 'Incomplete evaluator dependency evidence')
    for reference in references:
        verify(reference)


def audit_evaluation(plan: dict, seed: int, arm: str, rows: dict, run: dict) -> dict:
    """Independently verify224 saved neural inferences, without rerunning a model."""
    path = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
    result = read(path)
    ids = plan['train_ids'] + plan['holdout_ids']
    groups = plan['evaluation_groups']
    for key, expected in [('schema', 'lipnet-step12-multisentence-evaluation-v1'), ('status', 'completed'),
                          ('seed', seed), ('arm', arm), ('plan_sha256', num.canonical_hash(plan)),
                          ('data_sha256', plan['training_data']['data_sha256']),
                          ('evaluation_data_sha256', plan['evaluation_data']['data_sha256']),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')),
                          ('initial_state_hashes_verified', plan['initial_state_hashes_expected'][arm]),
                          ('ids', ids), ('diagnostic_groups', groups), ('evaluation_groups', groups),
                          ('runtime', plan['runtime']), ('optimization_steps', 0),
                          ('caller_rng_restored_exactly', True), ('internal_holdout_inputs_loaded', True)]:
        same(result[key], expected, 'Evaluation contract ' + key)
    for key in ('teacher_forward', 'audio_used', 'dev_inputs_loaded', 'original_held27_inputs_loaded',
                'test_inputs_loaded', 'service_weights_updated', 'holdout_teacher_features_present_in_evaluation_inputs',
                'holdout_teacher_features_used_for_inference'):
        same(result[key], False, 'Forbidden evaluation scope ' + key)
    frozen_code = {reference['path']: reference for reference in plan['execution_code']}
    verify_evaluation_sources(result['evaluation_code'], frozen_code)
    same(result['evaluation_contract'], read(BASE / 'diagnostic_plan.json', POLICY_SHA)['evaluation_contract'], 'Evaluation contract changed')
    same([item['completed_updates'] for item in result['milestones']], list(MILESTONES), 'Milestone positions')
    archive_ref = result['log_posteriors_archive']
    verify(archive_ref)
    counters, keys, max_ctc_error, condition_count = Counter(), set(), 0., 0
    milestone_results = {}
    with np.load(archive_ref['path'], allow_pickle=False) as archive:
        for milestone in result['milestones']:
            update = milestone['completed_updates']
            expected_checkpoint = None if update == 0 else run['checkpoint_bindings'][update // 900 - 1]
            same(milestone['checkpoint'], expected_checkpoint, 'Evaluation chose different checkpoint')
            expected_state = plan['initial_state_hashes_expected'][arm]['model'] if update == 0 else expected_checkpoint['state_hashes']['model']
            same(milestone['model_state_before'], expected_state, 'Evaluated model identity')
            same(milestone['model_state_after'], expected_state, 'Evaluation modified student')
            same(milestone['model_state_unchanged'], True, 'Evaluation state preservation')
            conditions = ['normal', 'static', 'shuffled'] if update == 3600 else ['normal']
            same(list(milestone['conditions']), conditions, 'Temporal condition scope')
            milestone_results[str(update)] = {}
            for condition, value in milestone['conditions'].items():
                condition_count += 1
                records = value['records']
                same([record['id'] for record in records], ids, 'Evaluation row order/scope')
                for position, record in enumerate(records):
                    row = rows[record['id']]
                    role = 'train' if row['id'] in plan['train_ids'] else 'internal_holdout'
                    stratum = next(group for group in GROUPS[2:] if row['id'] in groups[group])
                    counters['records'] += 1
                    counters[role + '_records'] += 1
                    for key, expected in [('speaker_id', row['speaker_id']), ('source_stem', row['source_stem']),
                                          ('split', role), ('diagnostic_role', role), ('source_split', 'train'),
                                          ('reference', row['text']), ('frames', row['frames']), ('condition', condition),
                                          ('target_token_ids', row['token_ids']), ('group', stratum)]:
                        same(record[key], expected, 'Evaluation row identity ' + key)
                    require(record['status'] in ('ok', 'failed'), 'Unknown inference status')
                    if record['status'] == 'ok':
                        same(record['error'], None, 'Successful inference error')
                        key = f'u{update:04d}_{condition}_{position}'
                        same(record['log_posteriors_key'], key, 'Posterior archive key')
                        require(key not in keys, 'Duplicate posterior evidence')
                        keys.add(key)
                        posterior = old.row_diagnostics(record, archive[key])
                        max_ctc_error = max(max_ctc_error, posterior['ctc_abs_difference'])
                        counters['posterior_frames'] += record['frames']
                    else:
                        counters['failed'] += 1
                        require(isinstance(record['error'], str) and record['error'], 'Missing failure reason')
                        for key, expected in [('prediction', ''), ('raw_argmax_ids', []), ('collapsed_ids', []),
                                              ('log_posteriors_key', None), ('output_diagnostics', None), ('ctc_loss_mean', None),
                                              ('replacement_characters', 0), ('strict_label_decoder_diagnostic', None)]:
                            same(record[key], expected, 'Failed inference fabricated output ' + key)
                    if record['temporal'] is not None:
                        temporal = record['temporal']
                        positions = num.temporal_order(row['frames'], condition, row['id'])
                        same(temporal['source_positions'], positions, 'Temporal manipulation')
                        same(temporal['source_positions_sha256'], num.nested_hash(positions), 'Temporal position hash')
                        same(temporal['frames'], row['frames'], 'Temporal frame count')
                        same(temporal['static_position'], row['frames'] // 2 if condition == 'static' else None, 'Static frame')
                        same(temporal['shuffle_policy'], 'sha256(step07-order-v1: + sample_id), first8 little endian, numpy.default_rng.permutation'
                             if condition == 'shuffled' else None, 'Temporal policy description')
                    else:
                        same(record['status'], 'failed', 'Successful inference lacks temporal evidence')
                    require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid inference wall time')
                    same(record['metrics'], num.metric_counts(row['text'], record['prediction']), 'Independent character/word edits')
                    same(record['token_metrics'], {'token_edits': num.edit_distance(row['token_ids'], record['collapsed_ids']),
                         'target_tokens': len(row['token_ids']), 'exact': row['token_ids'] == record['collapsed_ids']}, 'Independent token edits')
                same(set(value['groups']), set(groups), 'Evaluation aggregate groups')
                for group, members in groups.items():
                    expected = old.aggregate_condition([record for record in records if record['id'] in members])
                    same(value['groups'][group], expected, 'Independent aggregate ' + group)
                milestone_results[str(update)][condition] = value['groups']
        same(set(archive.files), keys, 'Missing/extra posterior arrays')
    same(result['record_count'], counters['records'], 'Evaluation record count')
    same(counters['records'], 224, 'Fixed evaluation rows')
    same(counters['train_records'], 168, 'Fixed train rows')
    same(counters['internal_holdout_records'], 56, 'Fixed holdout rows')
    same(result['posterior_arrays'], len(keys), 'Posterior count')
    same(condition_count, 7, 'Milestone/condition count')
    return {'evaluation': bind(path), 'record_count': counters['records'], 'failed_inputs': counters['failed'],
            'train_records': counters['train_records'], 'internal_holdout_records': counters['internal_holdout_records'],
            'posterior_arrays': len(keys), 'posterior_frames': counters['posterior_frames'],
            'independent_ctc_max_abs_difference': max_ctc_error, 'milestones': milestone_results,
            'final_success_24_of_24': milestone_results['3600']['normal']['train24']['aggregate']['exact_sentences'] == 24}


def independent_baseline(policy: dict) -> dict:
    """Construct the predeclared speaker-anchor text output and count edits."""
    selection, contract = policy['selection'], policy['speaker_anchor_baseline']
    rows = {record['id']: record for record in selection['records']}
    same(contract['evaluation_ids'], selection['holdout_ids'], 'Baseline holdout cohort')
    same(contract['choice_uses_holdout_targets'], False, 'Target-chosen baseline prohibited')
    same(contract['kind'], 'known_speaker_fixed_training_sentence', 'Baseline type')
    records = []
    for sid in selection['holdout_ids']:
        held = rows[sid]
        anchor_id = contract['anchor_ids_by_speaker'][held['speaker_id']]
        require(anchor_id in selection['train_ids'] and anchor_id in selection['retained_step11_ids'],
                'Baseline selected an unretained/nontraining anchor')
        anchor = rows[anchor_id]
        same(anchor['speaker_id'], held['speaker_id'], 'Baseline speaker identity')
        records.append({'id': sid, 'speaker_id': held['speaker_id'], 'anchor_id': anchor_id,
                        'status': 'ok', 'reference': held['text'], 'prediction': anchor['text'],
                        'metrics': num.metric_counts(held['text'], anchor['text'])})
    groups = selection['evaluation_groups']
    return {'kind': contract['kind'], 'records': records,
            'groups': {group: num.aggregate_metrics([record for record in records if record['id'] in groups[group]])
                       for group in ('hold8', 'hold_old4', 'hold_new4')},
            'neural_inference_rows': 0, 'requires_known_metadata_speaker_id': True,
            'deployable_unseen_speaker_recognizer': False}


def independent_comparisons(evaluations: dict) -> tuple[list[dict], dict]:
    """Calculate paired differences and arithmetic means from audited aggregates."""
    paired, means = [], {}
    for seed in SEEDS:
        a, b = evaluations[seed, 'A_ctc']['milestones']['3600'], evaluations[seed, 'B_feature']['milestones']['3600']
        difference = {}
        for condition in ('normal', 'static', 'shuffled'):
            difference[condition] = {}
            for group in GROUPS:
                left, right = a[condition][group]['aggregate'], b[condition][group]['aggregate']
                difference[condition][group] = {'cer_percentage_points': right['cer_percent'] - left['cer_percent'],
                                                'exact_sentences': right['exact_sentences'] - left['exact_sentences']}
        paired.append({'seed': seed, 'B_minus_A': difference})
    for arm in ARMS:
        means[arm] = {condition: {group: {
            metric: math.fsum(evaluations[seed, arm]['milestones']['3600'][condition][group]['aggregate'][metric]
                              for seed in SEEDS) / len(SEEDS)
            for metric in ('cer_percent', 'wer_percent', 'exact_sentences')}
            for group in GROUPS} for condition in ('normal', 'static', 'shuffled')}
    return paired, means


def audit_summary(policy: dict, runs: dict, evaluations: dict) -> dict:
    """Compare all summary values against independently checked artifacts."""
    path = BASE / 'results/comparison.json'
    summary = read(path)
    same(summary['schema'], 'lipnet-step12-multisentence-comparison-v1', 'Comparison schema')
    same(summary['status'], 'completed', 'Comparison incomplete')
    same(summary['experiment_plan'], bind(BASE / 'diagnostic_plan.json'), 'Comparison policy')
    verify(summary['generator'])
    same([(row['seed'], row['arm']) for row in summary['models']], [(s, a) for s in SEEDS for a in ARMS],
         'Comparison model omissions/order')
    expected_artifacts = []
    for model in summary['models']:
        key = model['seed'], model['arm']
        run, evaluation = runs[key], evaluations[key]
        for name, expected in [('evaluation', evaluation['evaluation']), ('run_result', run['run_result']),
                               ('success_24_of_24_exact_at_fixed_final', evaluation['final_success_24_of_24']),
                               ('failed_inferences', evaluation['failed_inputs']),
                               ('final_normal_train_exact_sentences', evaluation['milestones']['3600']['normal']['train24']['aggregate']['exact_sentences']),
                               ('final_conditions', evaluation['milestones']['3600']),
                               ('normal_curve', [{'update': u, 'groups': evaluation['milestones'][str(u)]['normal']} for u in MILESTONES])]:
            same(model[name], expected, 'Comparison model ' + name)
        original = read(run['run_result']['path'])
        history, training = original['updates'], model['training']
        for name, expected in [('updates', 3600), ('presentations', 7200),
                               ('exposure_per_clip', run['accounting']['per_clip_presentations']),
                               ('processed_frames', run['processed_frames']), ('elapsed_seconds', run['elapsed_seconds']),
                               ('max_cuda_allocated_bytes', run['accounting']['peak_allocated_bytes']),
                               ('max_cuda_reserved_bytes', run['accounting']['peak_reserved_bytes']),
                               ('gradient_norm_after_clipping_max', max(r['gradient_norm_after_clipping'] for r in history)),
                               ('initial_state_hashes', run['initial_state_hashes']), ('final_state_hashes', original['final_state_hashes'])]:
            same(training[name], expected, 'Comparison training accounting ' + name)
        for name, subset in [('first100', history[:100]), ('last100', history[-100:])]:
            for metric in ('loss', 'ctc_loss', 'feature_loss', 'gradient_norm_before_clipping'):
                old.assert_near(training['loss'][name][metric], math.fsum(r[metric] for r in subset) / 100,
                                'Comparison loss average', 1e-12)
        expected_artifacts.extend([evaluation['evaluation'], run['run_result'], read(evaluation['evaluation']['path'])['log_posteriors_archive']])
    same(summary['artifacts'], expected_artifacts, 'Comparison artifact references')
    for reference in expected_artifacts:
        verify(reference)
    for key, expected in [('new_optimizer_updates', 14400), ('inference_rows', 896),
                          ('training_elapsed_seconds_sum', sum(run['elapsed_seconds'] for run in runs.values())),
                          ('processed_frames_sum', sum(run['processed_frames'] for run in runs.values())),
                          ('fixed_final_success_runs', sum(value['final_success_24_of_24'] for value in evaluations.values())),
                          ('failures', sum(value['failed_inputs'] for value in evaluations.values())),
                          ('unseen_speaker_generalization_estimate', False), ('service_weights_updated', False)]:
        same(summary[key], expected, 'Comparison total ' + key)
    same(summary['speaker_anchor_baseline'], independent_baseline(policy), 'Independent speaker-anchor baseline')
    paired, means = independent_comparisons(evaluations)
    same(summary['paired_seed_comparisons'], paired, 'Independent paired differences')
    same(summary['two_seed_arithmetic_means'], means, 'Independent two-seed means')
    return bind(path)


def verify_preflight() -> list[dict]:
    """Validate pretraining audit and the exact saved versions of its source files."""
    reference = {'path': str(BASE / 'results/independent_preflight.json'), 'sha256': PREFLIGHT_SHA}
    preflight = read(reference['path'], reference['sha256'])
    same(preflight['status'], 'passed', 'Independent preflight did not pass')
    same(preflight['experiment_plan'], bind(BASE / 'diagnostic_plan.json'), 'Preflight policy changed')
    same(preflight['plans'], [bind(BASE / 'plans' / f'seed{seed}.json') for seed in SEEDS], 'Preflight seed plans changed')
    snapshot_path = BASE / 'audit_snapshots/preflight_source_snapshot.json'
    snapshots = read(snapshot_path)
    references = [reference, bind(snapshot_path)]
    for original in preflight['audit_sources']:
        if sha(original['path']) == original['sha256']:
            verify(original)
            references.append(original)
        else:
            matches = [item for item in snapshots if item['original'] == original]
            same(len(matches), 1, 'Changed preflight source has no exact snapshot')
            item = matches[0]
            require(num.local(item['snapshot']).is_relative_to(BASE / 'audit_snapshots'), 'Nonlocal source snapshot')
            same(item['sha256'], original['sha256'], 'Snapshot source digest differs')
            saved = {'path': item['snapshot'], 'sha256': item['sha256']}
            verify(saved)
            references.append(saved)
    return references


def final_audit(policy: dict, plans: dict, rows: dict, references: list[dict], training_data: dict,
                evaluation_data: dict, started: float) -> dict:
    """Verify all four fresh runs and all fixed trajectories and numeric summaries."""
    references += verify_preflight()
    runs, evaluations = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            runs[seed, arm] = audit_training(plans[seed], seed, arm, rows, training_data)
            evaluations[seed, arm] = audit_evaluation(plans[seed], seed, arm, rows, runs[seed, arm])
    completion = read(BASE / 'results/training_complete.json')
    same(completion['schema'], 'lipnet-step12-training-complete-v1', 'Training completion schema')
    same(completion['status'], 'completed', 'Training completion status')
    same(completion['experiment_plan'], bind(BASE / 'diagnostic_plan.json'), 'Training completion policy')
    same(completion['runs'], [runs[seed, arm]['run_result'] for seed in SEEDS for arm in ARMS], 'Completed run set')
    same(completion['new_optimizer_updates'], 14400, 'Completion update accounting')
    same(completion['service_weights_updated'], False, 'Completion service scope')
    for seed in SEEDS:
        a, b = runs[seed, 'A_ctc'], runs[seed, 'B_feature']
        same(a['processed_frames'], b['processed_frames'], 'Paired frame presentations')
        for component in ('model', 'adapter', 'rng'):
            same(a['initial_state_hashes'][component], b['initial_state_hashes'][component], 'Paired fresh run state')
        ae = read(evaluations[seed, 'A_ctc']['evaluation']['path'])['milestones'][0]['conditions']['normal']['records']
        be = read(evaluations[seed, 'B_feature']['evaluation']['path'])['milestones'][0]['conditions']['normal']['records']
        for ar, br in zip(ae, be, strict=True):
            for field in ('raw_argmax_ids', 'collapsed_ids', 'prediction', 'ctc_loss_mean', 'output_diagnostics', 'metrics', 'token_metrics'):
                same(ar[field], br[field], 'Paired initial inference')
    summary = audit_summary(policy, runs, evaluations)
    for reference in references:
        verify(reference)
    verify_preservation(policy['preservation'])
    return {'schema': 'lipnet-step12-independent-audit-v1', 'status': 'passed',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
            'test_source': bind(BASE / 'test_independent_audit.py'), 'numerical_ancestor': bind(ANCESTOR),
            'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'comparison': summary,
            'independent_preflight': bind(BASE / 'results/independent_preflight.json'),
            'preflight_source_snapshots_verified': True, 'preflight_precedes_all_run_contract_files': True,
            'selection_decisions_independently_checked': len(policy['selection']['selection_events']),
            'paired_initialization_and_schedule_verified': True,
            'source_and_code_hashes_unchanged_before_and_after_audit': True,
            'source_bindings_checked': len(references), 'input_npz_decoded': 32, 'teacher_npz_decoded': 24,
            'auditor_internal_holdout_teacher_npz_decoded': 0,
            'production_validation_reads_historical_train510_teacher_features': True,
            'historical_development_held27_test_payloads_opened': 0, 'raw_media_opened': 0,
            'internal_holdout_optimization_presentations': 0,
            'neural_inference_rows_independently_checked': sum(value['record_count'] for value in evaluations.values()),
            'known_speaker_baseline_rows_independently_checked': 8,
            'saved_posterior_arrays_checked': sum(value['posterior_arrays'] for value in evaluations.values()),
            'saved_posterior_frames_checked': sum(value['posterior_frames'] for value in evaluations.values()),
            'ctc_independent_float64_forward_max_abs_difference': max(value['independent_ctc_max_abs_difference'] for value in evaluations.values()),
            'complete_checkpoints_independently_checked': sum(value['checkpoints_verified'] for value in runs.values()),
            'optimizer_updates_verified': 14400, 'utterance_presentations_verified': 28800,
            'processed_frames_verified': sum(value['processed_frames'] for value in runs.values()),
            'failures': sum(value['failed_inputs'] for value in evaluations.values()),
            'success_runs_at_fixed_final': sum(value['final_success_24_of_24'] for value in evaluations.values()),
            'per_run': [{'seed': seed, 'arm': arm, 'training': runs[seed, arm], 'evaluation': evaluations[seed, arm]}
                        for seed in SEEDS for arm in ARMS],
            'elapsed_seconds': time.perf_counter() - started, 'service_weights_preserved': True,
            'unseen_speaker_generalization_estimate': False,
            'limitations': ['Saved posterior arithmetic is verified; the audit does not rerun the GPU model.',
                            'The32 selected clips came from historical train510; internal hold8 is specific to these fresh models.',
                            'Historical loader validates510 teacher caches; onlytrain24 targets enter optimization.',
                            'Unchanged hashes detect mutation and corruption, not author authenticity.']}


def write_json(path: Path, value: dict) -> dict:
    """Save new immutable audit evidence; never overwrite a previous result."""
    require(not path.exists() and path.resolve().is_relative_to(BASE), 'Use a new local audit path')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def main() -> None:
    """Run independent preflight or final numerical audit without GPU/model imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-policy-sha', default=POLICY_SHA)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    require(args.expected_policy_sha == POLICY_SHA, 'Unexpected diagnostic policy anchor')
    output = args.output or BASE / 'results' / ('independent_preflight.json' if args.preflight else 'independent_audit.json')
    require(not output.exists(), 'Refusing to replace existing audit')
    started = time.perf_counter()
    audit_sources = [bind(Path(__file__)), bind(BASE / 'test_independent_audit.py'), bind(ANCESTOR), bind(old.NUMERICAL)]
    num.torch.set_num_threads(2)
    policy, plans, rows, references, training_data, evaluation_data = audit_contract(args.expected_policy_sha)
    references += audit_sources
    if not args.preflight:
        result = final_audit(policy, plans, rows, references, training_data, evaluation_data, started)
        print(json.dumps({'status': 'passed', 'output': write_json(output, result),
                          'neural_rows': result['neural_inference_rows_independently_checked'],
                          'ctc_max_difference': result['ctc_independent_float64_forward_max_abs_difference']}))
        return
    for reference in references:
        verify(reference)
    verify_preservation(policy['preservation'])
    result = {'schema': 'lipnet-step12-independent-preflight-v1', 'status': 'passed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'audit_sources': audit_sources,
              'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'plans': [bind(BASE / 'plans' / f'seed{seed}.json') for seed in SEEDS],
              'train_ids': policy['selection']['train_ids'], 'holdout_ids': policy['selection']['holdout_ids'],
              'train_frames': sum(item['frames'] for item in training_data.values()),
              'holdout_frames': sum(evaluation_data[sid]['frames'] for sid in policy['selection']['holdout_ids']),
              'training_data_sha256': num.nested_hash(training_data), 'evaluation_data_sha256': num.nested_hash(evaluation_data),
              'source_bindings_verified': len(references), 'input_npz_decoded': 32, 'teacher_npz_decoded': 24,
              'auditor_internal_holdout_teacher_npz_decoded': 0,
              'production_validation_reads_historical_train510_teacher_features': True,
              'internal_holdout_optimization_presentations': 0, 'historical_development_held27_test_payloads_opened': 0,
              'raw_media_opened': 0, 'selection_events_independently_checked': len(policy['selection']['selection_events']),
              'service_weights_preserved': True, 'source_and_code_hashes_unchanged_before_and_after': True,
              'elapsed_seconds': time.perf_counter() - started,
              'scope': 'Selection, schedules and source hashes only; no completed-run or inference claim.'}
    print(json.dumps({'status': 'passed', 'output': write_json(output, result)}))


if __name__ == '__main__':
    main()
