"""Bind verified caches into a provisional handoff; never train or rewrite inputs."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXP = BASE.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify(ref):
    require(bind(ref['path']) == {'path': ref['path'], 'sha256': ref['sha256']}, 'Changed file: ' + ref['path'])


def row_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    output = BASE / 'candidate_handoff.json'
    summary_path = BASE / 'results/step09_summary.json'
    require(not output.exists() and not summary_path.exists(), 'Completion artifacts already exist')
    paths = {
        'old_data_plan': EXP / 'feature_transfer_step07/data_plan.json',
        'old_handoff': EXP / 'local_dataset_step04_20261007/full_data_plan.json',
        'old_inputs': EXP / 'local_dataset_step04_20261007/inputs/manifest.json',
        'old_features': EXP / 'local_dataset_step04_20261007/teacher_results/train_manifest.json',
        'new_data_plan': BASE / 'data_plan.json',
        'selection_audit': BASE / 'selection_audit.json',
        'input_execution_plan': BASE / 'input_execution_plan.json',
        'new_inputs': BASE / 'inputs/manifest.json',
        'input_audit': BASE / 'results/input_audit.json',
        'teacher_execution_plan': BASE / 'teacher_execution_plan.json',
        'new_features': BASE / 'results/teacher_manifest.json',
        'feature_audit': BASE / 'results/feature_audit.json',
    }
    refs = {key: bind(path) for key, path in paths.items()}
    docs = {key: read(path) for key, path in paths.items()}
    old, new = docs['old_data_plan'], docs['new_data_plan']
    require(refs['old_data_plan']['sha256'] == 'c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30', 'Changed old plan')
    require(refs['old_handoff']['sha256'] == '3003209c04109a01f9737593b3645ffe7aef28e16530d84e53ba0caca77fca4b', 'Changed old handoff')
    require(refs['new_data_plan']['sha256'] == '251d28ee44ea934ec2db7925821b0821dd4487f9e2f64c219f86ee87d5529585', 'Changed new plan')
    previous = docs['old_handoff']
    require(previous['artifacts']['inputs'] == refs['old_inputs'] and previous['artifacts']['features'] == refs['old_features'], 'Changed old manifests')
    for ref in previous['artifacts'].values():
        verify(ref)
    for key in ('selection_audit', 'input_audit', 'feature_audit'):
        require(docs[key]['status'] == 'passed' and docs[key]['data_plan'] == refs['new_data_plan'], 'Audit not passed/bound: ' + key)
        verify(docs[key]['code'])
    require(docs['input_audit']['input_manifest'] == refs['new_inputs'], 'Input audit mismatch')
    require(docs['input_audit']['execution_plan'] == refs['input_execution_plan'] == docs['new_inputs']['execution_plan'], 'Input execution mismatch')
    require(docs['feature_audit']['teacher_manifest'] == refs['new_features'], 'Feature audit mismatch')
    require(docs['feature_audit']['teacher_plan'] == refs['teacher_execution_plan'], 'Feature audit plan mismatch')
    require(docs['new_features']['teacher_plan'] == refs['teacher_execution_plan'], 'Teacher plan mismatch')
    require(docs['new_features']['input_audit'] == refs['input_audit'], 'Teacher did not use audited inputs')
    for key in ('input_execution_plan', 'teacher_execution_plan'):
        plan = docs[key]
        verify(plan['data_plan'])
        require(plan['data_plan'] == refs['new_data_plan'], 'Changed execution data plan')
        for ref in (plan['code'].values() if isinstance(plan['code'], dict) else plan['code']):
            verify(ref)
    require(docs['new_inputs']['status'] == 'passed' and docs['new_features']['status'] == 'completed', 'Incomplete preparation')
    records = []
    for generation, logical, input_doc, feature_doc in (
        ('old', old, docs['old_inputs'], docs['old_features']),
        ('new', new, docs['new_inputs'], docs['new_features']),
    ):
        inputs = {r['id']: r for r in input_doc['records']}
        features = {r['id']: r for r in feature_doc['records']}
        require(len(inputs) == len(input_doc['records']) and len(features) == len(feature_doc['records']), 'Duplicate manifest identity')
        require(len(inputs) == len(logical['records']) and set(inputs) == {r['id'] for r in logical['records']}, 'Input inventory mismatch')
        expected_feature_ids = {r['id'] for r in logical['records'] if (r['split'] == 'train' if generation == 'old' else r['teacher_candidate'])}
        require(set(features) == expected_feature_ids, 'Extra or missing feature identity')
        for row in logical['records']:
            sid = row['id']
            role = ('old_train' if row['split'] == 'train' else 'old_development') if generation == 'old' else (
                'new_provisional_train' if row['teacher_candidate'] else 'new_development' if row['split'] == 'val' else 'new_train_hold')
            training_candidate = role in ('old_train', 'new_provisional_train')
            require((sid in features) == training_candidate, 'Feature/role mismatch: ' + sid)
            item = inputs[sid]
            inp = {'path': item['npz_path'], 'sha256': item['npz_sha256']}
            feat = features[sid]['feature_file'] if training_candidate else None
            verify(inp)
            if feat:
                verify(feat)
            require(item['student_frames'] == row['frames'], 'Frame total mismatch')
            records.append({'id': sid, 'role': role, 'original_split': row['split'],
                            'generation': generation, 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                            'student_frames': row['frames'], 'logical_data_plan': refs[generation + '_data_plan'],
                            'logical_record_sha256': row_hash(row), 'input_manifest': refs[generation + '_inputs'],
                            'input': inp, 'teacher_feature_manifest': refs[generation + '_features'] if feat else None,
                            'teacher_feature': feat, 'provisional_training_candidate': training_candidate,
                            'new_human_approval_added': False})
    require(len(records) == len({r['id'] for r in records}) == 632, 'Duplicate or missing sample')
    roles = Counter(r['role'] for r in records)
    require(roles == {'old_train': 441, 'new_provisional_train': 69, 'old_development': 71, 'new_development': 24, 'new_train_hold': 27}, 'Roles changed')
    ids = lambda role: [r['id'] for r in records if r['role'] == role]
    train = ids('old_train') + ids('new_provisional_train')
    train_speakers = {r['speaker_id'] for r in records if r['provisional_training_candidate']}
    new_dev_speakers = {r['speaker_id'] for r in records if r['role'] == 'new_development'}
    require(len(train_speakers) == 20 and len(new_dev_speakers) == 3 and not train_speakers & new_dev_speakers, 'New development speaker leakage')
    overlap = new['candidate_vs_dev_text_overlap']
    require(len(overlap) == 1, 'Changed duplicate text inventory')
    duplicate_dev_ids = {r['val_id'] for r in overlap}
    clean_dev = [sid for sid in ids('new_development') if sid not in duplicate_dev_ids]
    require(len(clean_dev) == 23, 'Changed supplemental development cohort')
    require(not any(p.is_file() for p in (BASE / 'temporary').rglob('*')), 'Temporary raw payload remains')
    no_activity = {'student_optimization_steps_this_step': 0, 'teacher_optimization_steps_this_step': 0,
                   'service_weights_changed': False, 'new_performance_measurement': False,
                   'new_human_approvals': 0, 'training_ready': False}
    counts = {'all_referenced_records': 632, 'student_frames_all': sum(r['student_frames'] for r in records),
              'provisional_training_records': 510, 'provisional_training_speakers': 20,
              'provisional_training_frames': sum(r['student_frames'] for r in records if r['provisional_training_candidate']),
              'roles': dict(roles)}
    require(counts['student_frames_all'] == 73528 and counts['provisional_training_frames'] == 58974, 'Combined frame count')
    handoff = {'schema': 'lipnet-step09-provisional-candidate-handoff-v1',
               'status': 'verified_data_available_pending_new_training_execution_plan',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
               'artifacts': refs, 'counts': counts, 'records': records,
               'provisional_train_ids': train, 'control_old_train_ids': ids('old_train'),
               'development_groups': {'old71_same_training_speakers': ids('old_development'),
                                      'new24_all_original_items': ids('new_development'),
                                      'new23_supplement_excluding_training_text_equal_without_spaces': clean_dev},
               'held_new_train_ids': ids('new_train_hold'), 'text_overlap': overlap,
               'pending_listening': new['policy']['pending_listening'],
               'next_step_requirements': [
                   'Use a new mixed-schema loader/entrypoint; previous fixed441/71 runner is not directly compatible.',
                   'Retain the original512 dictionaries/NPZ/features by verified references; do not recopy or overwrite them.',
                   'Freeze provisional label policy, checkpoints, control/expanded cohorts, schedule, decoder and execution environment before optimization.',
                   'Use old441 vs expanded510 with matched starting states and compute budget to examine the effect of new speakers; do not mix data-size and method effects.',
                   'Keep old71 and full new24 separate; also report supplemental new23 excluding the one repeated transcript.',
                   'Use error-preserving prediction composition, saved raw token IDs and frame-order controls; report exact matches and CER/WER.',
                   'Do not use held27 or development examples for gradients, teacher targets or schedule selection.',
                   'Unanswered local listening questions remain pending; later answers require versioned evidence, not retroactive approval.',
               ], 'limitations': new['limits'] + [
                   'Step6 observed strong dotted texture in C630/C646/C658/C675 samples (32 new clips, including22 candidates); pixel equality does not certify visual quality or its cause.',
                   'Old71 has been used in prior experiments, and new24 has undergone inspection; neither is an untouched final test.',
                   'Teacher public-pretraining overlap remains unverified.',
                   'training_ready=false preserves the prior human-review status; this handoff proposes an explicitly provisional experiment, not certified ground truth.',
               ], **no_activity}
    summary = {'schema': 'lipnet-step09-summary-v1', 'status': 'completed_provisional_input_and_teacher_feature_preparation',
               'created_utc': handoff['created_utc'], 'generator': handoff['generator'], 'artifacts': refs,
               'new_records': 120, 'new_student_frames': 15109, 'new_teacher_feature_records': 69,
               'new_teacher_aligned_frames': 8455, 'held_new_train_records': 27, 'new_development_records': 24,
               'new_teacher_native_frames': docs['new_features']['native_frames'],
               'input_npz_bytes': docs['new_inputs']['input_npz_bytes'],
               'feature_npz_bytes': docs['new_features']['feature_file_bytes'],
               'input_seconds': docs['new_inputs']['elapsed_seconds_this_run'],
               'teacher_seconds': docs['new_features']['elapsed_seconds'],
               'copied_raw_payload_bytes': docs['new_inputs']['copied_payload_bytes_this_run'],
               'temporary_payload_files_remaining': 0,
               'old_cached_input_files_rehashed': 512, 'old_cached_feature_files_rehashed': 441,
               'pending_listening': handoff['pending_listening'], 'combined_candidate_counts': counts,
               **no_activity}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(handoff, stream, ensure_ascii=False, indent=2); stream.write('\n')
    summary['candidate_handoff'] = bind(output)
    with summary_path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('artifacts', 'pending_listening')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
