"""Frozen inference-only same-recording diagnostic; no optimization or teacher reads.

Only six selected paired input archives are hashed and their student arrays decoded.
Hashing a paired archive reads compressed bytes, including teacher-pixel bytes; no
teacher-pixel array or separate teacher-feature archive is decoded or used.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / 'experiments/multisentence_step12_20261007'
POLICY = BASE / 'diagnostic_plan.json'
FROZEN = BASE / 'plans/freeze_manifest.json'
SALT = 'step13-same-recording-v1:'
SPEAKERS = ('C002', 'C032', 'C282', 'C293', 'C058', 'C658', 'C675', 'C321')
EXPECTED_AVAILABLE = ('C002', 'C032', 'C282', 'C293', 'C675', 'C321')
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
STEP12_SUMMARY_SHA = '7bdce41e511d012a9590e229fbe67e09f31c4f82466547821fcfe9622144ce99'
STEP12_PROTOCOL_SHA = 'b5363fe649dc336b898c61d2fba70f9b92d581367cb1db24e31b7f77758507bd'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == ref, 'Changed/nonlocal evidence: ' + str(path))
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


require(sha(PRIOR / 'protocol.py') == STEP12_PROTOCOL_SHA, 'Frozen Step12 protocol changed')
require(sha(PRIOR / 'results/step12_summary.json') == STEP12_SUMMARY_SHA, 'Step12 completion evidence changed')
_summary = read(PRIOR / 'results/step12_summary.json')
require(_summary['status'] == 'completed_multisentence_internal_holdout_diagnostic', 'Step12 incomplete')
for _ref in _summary['artifacts'].values():
    verify(_ref)
_policy_ref = _summary['artifacts']['diagnostic_plan.json']
_audit_ref = _summary['artifacts']['results/independent_audit.json']
require(read(_audit_ref['path'])['status'] == 'passed', 'Step12 audit failed')
_prior_policy = read(_policy_ref['path'])
for _ref in _prior_policy['execution_code']:
    verify(_ref)
prior = module('step13_frozen_step12_protocol', PRIOR / 'protocol.py')
runner, core, mixed = prior.runner, prior.core, prior.mixed
normalized_target, source_hashes, ctc_reasons = prior.normalized_target, prior.source_hashes, prior.ctc_reasons


def selection_hash(sid):
    return hashlib.sha256((SALT + sid).encode()).hexdigest()


def source_indices(row):
    require(type(row['frames']) is int and row['frames'] > 0 and
            all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in ('start_time', 'end_time', 'video_fps')) and
            row['video_fps'] > 0 and 0 <= row['start_time'] < row['end_time'], 'Invalid timing metadata')
    return [round((row['start_time'] + k / 25) * row['video_fps']) for k in range(row['frames'])]


def overlap_reasons(row, training_rows):
    indices = set(source_indices(row))
    reasons = []
    for other in training_rows:
        if row['source_stem'] != other['source_stem']:
            continue
        if max(row['start_time'], other['start_time']) < min(row['end_time'], other['end_time']):
            reasons.append('training_time_interval_overlap:' + other['id'])
        if indices.intersection(source_indices(other)):
            reasons.append('training_native_frame_overlap:' + other['id'])
    return reasons


def select_from_bundle(bundle, previous_selection):
    rows, entries, groups = bundle['rows'], bundle['entries'], bundle['groups']
    eligible = list(groups['expanded510'])
    excluded = list(groups['held27'] + groups['old71'] + groups['new24'])
    require(len(set(eligible)) == len(eligible) and not set(eligible).intersection(excluded), 'Eligible/excluded contamination')
    train_ids, hold_ids = list(previous_selection['train_ids']), list(previous_selection['holdout_ids'])
    reserved_ids = train_ids + hold_ids
    require(len(train_ids) == 24 and len(hold_ids) == 8 and len(set(reserved_ids)) == 32 and set(reserved_ids) <= set(eligible), 'Wrong previous32 scope')
    speakers = previous_selection['speaker_selections']
    require(tuple(item['speaker_id'] for item in speakers) == SPEAKERS, 'Prior speaker order changed')
    for sid in eligible:
        row, entry = rows[sid], entries[sid]
        require(row['split'] == entry['original_split'] == 'train' and entry['role'] in ('old_train', 'new_provisional_train') and
                entry['generation'] == ('old' if entry['role'] == 'old_train' else 'new') and
                entry['provisional_training_candidate'] is True and row['speaker_id'] == entry['speaker_id'] and
                row['frames'] == entry['student_frames'], 'Ineligible historical training record')
    old_records = {row['id']: row for row in previous_selection['records']}
    for sid in reserved_ids:
        old, row = old_records[sid], rows[sid]
        require(old['logical_record_sha256'] == entries[sid]['logical_record_sha256'] and
                all(old[key] == row[key] for key in ('speaker_id', 'source_stem', 'text', 'frames')) and
                old['tokens'] == row['token_ids'], 'Previous selected metadata changed')
    texts = {normalized_target(rows[sid]['text']) for sid in reserved_ids}
    require(len(texts) == 32 and '' not in texts, 'Previous32 normalized targets not distinct')
    same_ids, different_ids, pairs, events, selections = [], [], [], [], []
    for item in speakers:
        speaker, anchor = item['speaker_id'], item['retained_anchor_id']
        training = list(item['train_ids'])
        require(len(training) == 3 and set(training) <= set(train_ids) and item['holdout_id'] in hold_ids and
                all(rows[sid]['speaker_id'] == speaker for sid in training + [item['holdout_id']]) and
                rows[anchor]['source_stem'] == item['training_source_stem'], 'Previous speaker membership changed')
        reference_hashes = source_hashes(bundle, anchor)
        require(all(source_hashes(bundle, sid) == reference_hashes and rows[sid]['source_stem'] == item['training_source_stem'] for sid in training),
                'Previous training source identity changed')
        pool = sorted((sid for sid in eligible if sid not in reserved_ids and rows[sid]['speaker_id'] == speaker and
                       rows[sid]['source_stem'] == item['training_source_stem']), key=lambda sid: (selection_hash(sid), sid))
        chosen = None
        for sid in pool:
            row = rows[sid]
            reasons = ctc_reasons(row)
            text = normalized_target(row['text'])
            if not text:
                reasons.append('empty_normalized_target')
            if text in texts:
                reasons.append('duplicate_normalized_target')
            if source_hashes(bundle, sid) != reference_hashes:
                reasons.append('same_stem_source_hash_mismatch')
            reasons.extend(overlap_reasons(row, [rows[tid] for tid in train_ids]))
            status = 'not_needed' if chosen is not None else 'skipped' if reasons else 'selected'
            events.append({'speaker_id': speaker, 'id': sid, 'rank_sha256': selection_hash(sid), 'status': status, 'reasons': reasons})
            if status == 'selected':
                chosen = sid
                texts.add(text)
        selections.append({'speaker_id': speaker, 'generation': entries[anchor]['generation'], 'retained_anchor_id': anchor,
                           'train_ids': training, 'training_source_stem': item['training_source_stem'],
                           'candidate_ids_ranked': pool, 'candidate_count': len(pool), 'selected_id': chosen,
                           'unavailable_reason': 'no_remaining_same_recording_candidate' if not pool else 'all_candidates_ineligible' if chosen is None else None})
        if chosen is None:
            continue
        different = item['holdout_id']
        require(rows[different]['source_stem'] != rows[chosen]['source_stem'] and
                all(source_hashes(bundle, different)[key] != reference_hashes[key] for key in reference_hashes), 'Different-recording identity overlaps')
        same_ids.append(chosen)
        different_ids.append(different)
        pairs.append({'speaker_id': speaker, 'generation': entries[chosen]['generation'], 'same_id': chosen,
                      'different_id': different, 'train_ids': training, 'anchor_id': anchor})
    require(tuple(pair['speaker_id'] for pair in pairs) == EXPECTED_AVAILABLE and
            all(item['candidate_count'] == 0 for item in selections if item['speaker_id'] not in EXPECTED_AVAILABLE), 'Available speaker scope changed')
    records = []
    for sid in same_ids + different_ids:
        row, entry = rows[sid], entries[sid]
        tokens = row['token_ids']
        minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        records.append({'id': sid, 'generation': entry['generation'], 'original_role': entry['role'], 'original_split': 'train',
                        'diagnostic_role': 'same_recording_holdout' if sid in same_ids else 'different_recording_holdout',
                        'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'], 'text': row['text'],
                        'normalized_target': normalized_target(row['text']), 'frames': row['frames'], 'tokens': list(tokens),
                        'start_time': row['start_time'], 'end_time': row['end_time'], 'video_fps': row['video_fps'],
                        'student_source_indices': source_indices(row), 'ctc_min_input_frames': minimum, 'ctc_slack': row['frames'] - minimum,
                        'logical_record_sha256': entry['logical_record_sha256'], 'input': entry['input'], **source_hashes(bundle, sid)})
    eval_groups = {scope + '6': ids for scope, ids in (('same', same_ids), ('different', different_ids))}
    for scope, ids in (('same', same_ids), ('different', different_ids)):
        for generation, count in (('old', 4), ('new', 2)):
            eval_groups[scope + '_' + generation + str(count)] = [sid for sid in ids if entries[sid]['generation'] == generation]
    require({key: len(ids) for key, ids in eval_groups.items()} == {'same6': 6, 'different6': 6, 'same_old4': 4, 'same_new2': 2, 'different_old4': 4, 'different_new2': 2}, 'Source-stratum counts changed')
    return {'schema': 'lipnet-step13-recording-selection-v1', 'same_ids': same_ids, 'different_ids': different_ids,
            'ids': same_ids + different_ids, 'pairs': pairs, 'records': records, 'groups': eval_groups,
            'speaker_selections': selections, 'candidate_events': events, 'eligible_training_ids': eligible,
            'excluded_step12_ids': reserved_ids, 'historical_excluded_ids': excluded,
            'selection_salt': SALT, 'normalization': 'NFC then remove all Unicode whitespace using str.split',
            'prediction_or_loss_used_for_selection': False, 'same_recording_clips_excluded_from_step12_training': True,
            'same_recording_time_and_native_frames_disjoint_from_training': True,
            'source_separation_checked': ['source_stem', 'video_sha256', 'label_sha256'],
            'historically_used_by_prior_models': True, 'untouched_test_set': False, 'held_out_speakers': False,
            'new_human_approvals_added': 0, 'different_recording_transcripts_and_contexts_not_matched': True}


def selection_metadata():
    bundle = mixed.verify_handoff()
    value = select_from_bundle(bundle, _prior_policy['selection'])
    require(len(value['eligible_training_ids']) == 510 and len(value['historical_excluded_ids']) == 122, 'Historical cohort changed')
    value.update(source_handoff=bundle['handoff_ref'], prior_step12_policy=_policy_ref)
    return value


def evaluation_binding(clip):
    require(clip['split'] == clip['diagnostic_role'] == 'same_recording_holdout' and clip['source_split'] == 'train' and
            'teacher_features' not in clip, 'Invalid inference clip scope')
    return {**{key: clip[key] for key in ('id', 'split', 'source_split', 'diagnostic_role', 'speaker_id', 'source_stem', 'text')},
            'frames': len(clip['pixels']), 'pixels_sha256': runner.nested_hash(clip['pixels']), 'tokens_sha256': runner.nested_hash(clip['tokens'])}


def evaluation_bindings(clips):
    bindings = {sid: evaluation_binding(clips[sid]) for sid in sorted(clips)}
    require(all(sid == value['id'] for sid, value in bindings.items()), 'Clip key/ID mismatch')
    return {'data_sha256': runner.nested_hash(bindings), 'clips': bindings}


def load_inputs(selection=None):
    bundle = mixed.verify_handoff()
    actual = select_from_bundle(bundle, _prior_policy['selection'])
    actual.update(source_handoff=bundle['handoff_ref'], prior_step12_policy=_policy_ref)
    if selection is not None:
        require(selection == actual, 'Selection changed before input load')
    checked = mixed.old_loader().LocalBindings()
    clips = {}
    for sid in actual['same_ids']:
        row, entry = bundle['rows'][sid], bundle['entries'][sid]
        item = bundle['inputs'][entry['generation']][sid]
        path = checked.verify(entry['input'])
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == mixed.PAIRED_KEYS, 'Unexpected paired input keys')
            arrays = {key: saved[key] for key in ('student_rgb', 'student_source_indices', 'label_tokens')}
            metadata = json.loads(str(saved['metadata_json'].item()))
        if entry['generation'] == 'old':
            mixed.old_loader().validate_student(row, item, arrays, metadata, 'train')
        else:
            mixed.validate_new_student(row, item, arrays, metadata, bundle, 'train')
        require(metadata['source_verification'] == item['source_verification'], 'Student source provenance changed')
        require(arrays['student_source_indices'].tolist() == source_indices(row), 'Native input frame mapping differs')
        clips[sid] = {'id': sid, 'split': 'same_recording_holdout', 'diagnostic_role': 'same_recording_holdout',
                      'source_split': 'train', 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                      'text': row['text'], 'tokens': list(row['token_ids']), 'pixels': arrays['student_rgb']}
    return {'clips': clips, 'same_clips': clips, 'selection': actual, 'input_bindings': checked.finish(),
            'evaluation_inputs': evaluation_bindings(clips), 'groups': actual['groups'],
            'counts': {'student_input_npz_decoded': 6, 'student_frames_decoded': sum(len(c['pixels']) for c in clips.values()),
                       'paired_npz_whole_file_hashed': 6, 'paired_teacher_compressed_bytes_read_for_file_hash': True,
                       'teacher_pixel_arrays_decoded': 0, 'teacher_feature_npz_opened': 0,
                       'different_recording_input_npz_opened': 0, 'training_input_npz_opened': 0,
                       'original_development_or_held27_or_test_payloads_opened': 0, 'source_media_opened': 0}}


load_clips = load_inputs


def model_bindings():
    comparison = read(verify(_summary['artifacts']['results/comparison.json']))
    models = []
    for seed in SEEDS:
        for arm in ARMS:
            item = next(row for row in comparison['models'] if row['seed'] == seed and row['arm'] == arm)
            result = read(verify(item['run_result']))
            evaluation = read(verify(item['evaluation']))
            require(result['status'] == 'completed' and result['completed_updates'] == 3600 and
                    result['seed'] == evaluation['seed'] == seed and result['arm'] == evaluation['arm'] == arm, 'Incomplete/wrong prior model')
            checkpoints = [ref for ref in result['checkpoints'] if ref['completed_updates'] == 3600]
            require(len(checkpoints) == 1, 'Fixed final checkpoint absent/ambiguous')
            checkpoint = checkpoints[0]
            verify({key: checkpoint[key] for key in ('path', 'sha256')})
            verify({'path': checkpoint['sidecar'], 'sha256': checkpoint['sidecar_sha256']})
            verify(evaluation['log_posteriors_archive'])
            verify(result['plan_file'])
            require(result['experiment_plan'] == _policy_ref and result['plan_file'] == _summary['artifacts'][f'plans/seed{seed}.json'], 'Prior plan binding changed')
            models.append({'seed': seed, 'arm': arm, 'run_result': item['run_result'], 'checkpoint': checkpoint,
                           'prior_evaluation': item['evaluation'], 'prior_posteriors': evaluation['log_posteriors_archive'], 'plan': result['plan_file']})
    return models


def preservation_contract():
    value = prior.preservation_contract()
    value['artifacts'].extend([bind(PRIOR / 'results/step12_summary.json'), _policy_ref, _audit_ref])
    return value


def verify_preservation(value):
    prior.verify_preservation(value)


def code_bindings(extra_code_paths=()):
    evaluation = module('step13_evaluation_code', BASE / 'evaluate.py')
    paths = {BASE / name for name in ('protocol.py', 'test_protocol.py', 'evaluate.py', 'test_evaluate.py')}
    paths.update(Path(path).resolve() for path in extra_code_paths)
    paths.update(Path(ref['path']) for ref in _prior_policy['execution_code'])
    paths.update(Path(ref['path']) for ref in evaluation.evaluation_code())
    paths.add(PRIOR / 'protocol.py')
    require(all(path.is_relative_to(ROOT) and path.suffix in {'.py', '.txt'} for path in paths), 'Nonlocal execution code')
    return [bind(path) for path in sorted(paths, key=str)]


def freeze(extra_code_paths=()):
    require(not POLICY.exists() and not FROZEN.exists(), 'Step13 contract already exists')
    loaded = load_inputs()
    core.setup_runtime()
    require(runner.torch.cuda.is_available(), 'CUDA required for declared inference runtime')
    selection = loaded['selection']
    baseline = {'kind': 'known_speaker_fixed_training_sentence', 'baseline_rows': 12, 'neural_inference_rows': 0,
                'choice_uses_evaluation_targets': False, 'known_speaker_metadata_required': True,
                'anchor_ids_by_speaker': {pair['speaker_id']: pair['anchor_id'] for pair in selection['pairs']},
                'evaluation_ids': selection['ids'], 'prediction_rule': 'Output the retained Step11 training-anchor transcript for the known speaker'}
    previous_records = {row['id']: row for row in _prior_policy['selection']['records']}
    baseline['anchors'] = {pair['speaker_id']: {'id': pair['anchor_id'], 'text': previous_records[pair['anchor_id']]['text']}
                           for pair in selection['pairs']}
    evaluation = module('step13_evaluation_contract', BASE / 'evaluate.py')
    policy = {'schema': 'lipnet-step13-recording-protocol-v1', 'status': 'frozen_before_inference',
              'purpose': 'Same-speaker unseen utterances from training recording versus reused different-recording holdout; descriptive matched-speaker diagnostic',
              'selection': selection, 'models': model_bindings(), 'prior_step12_summary': bind(PRIOR / 'results/step12_summary.json'),
              'prior_step12_policy': _policy_ref, 'prior_step12_audit': _audit_ref,
              'evaluation_inputs': loaded['evaluation_inputs'], 'input_bindings': loaded['input_bindings'],
              'evaluation_contract': evaluation.evaluation_contract(), 'speaker_anchor_baseline': baseline,
              'execution_code': code_bindings(extra_code_paths), 'preservation': preservation_contract(),
              'runtime': runner._runtime(runner.torch.device('cuda:0')), 'loading_scope': loaded['counts'],
              'new_inference_rows_planned': 72, 'reused_inference_rows_planned': 72, 'optimizer_updates_planned': 0,
              'fixed_checkpoint_update': 3600, 'checkpoint_selection_uses_evaluation_results': False,
              'all_models_and_conditions_reported': True, 'new_human_approvals_added': 0,
              'audio_used': False, 'teacher_model_loaded': False, 'teacher_feature_npz_opened': 0,
              'service_weights_updated': False, 'untouched_test_claim_allowed': False, 'unseen_speaker_or_webcam_claim_allowed': False}
    verify_preservation(policy['preservation'])
    runner._save_json(POLICY, policy)
    manifest = {'schema': 'lipnet-step13-freeze-manifest-v1', 'status': 'frozen', 'experiment_plan': bind(POLICY),
                'optimizer_updates_executed': 0, 'new_inference_rows_executed': 0}
    runner._save_json(FROZEN, manifest)
    return manifest


def verify_frozen():
    manifest = read(FROZEN)
    require(manifest['schema'] == 'lipnet-step13-freeze-manifest-v1' and manifest['status'] == 'frozen' and
            manifest['experiment_plan'] == bind(POLICY), 'Freeze manifest changed')
    policy = read(verify(manifest['experiment_plan']))
    require(policy['selection'] == selection_metadata(), 'Deterministic selection changed')
    require(policy['models'] == model_bindings(), 'Prior model bindings changed')
    require(policy['optimizer_updates_planned'] == 0 and policy['new_inference_rows_planned'] == policy['reused_inference_rows_planned'] == 72 and
            policy['fixed_checkpoint_update'] == 3600 and not policy['service_weights_updated'], 'Inference-only budget changed')
    for ref in policy['execution_code']:
        verify(ref)
    verify_preservation(policy['preservation'])
    core.setup_runtime()
    require(policy['runtime'] == runner._runtime(runner.torch.device('cuda:0')), 'Runtime changed')
    return policy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('selection', 'freeze', 'verify'))
    parser.add_argument('--extra-code', action='append', default=[])
    args = parser.parse_args()
    result = selection_metadata() if args.command == 'selection' else freeze(args.extra_code) if args.command == 'freeze' else verify_frozen()
    print(json.dumps(result, ensure_ascii=False, indent=2))
