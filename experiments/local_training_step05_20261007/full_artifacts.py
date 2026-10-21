"""Read the immutable Step4 full cohort without models, source drives or writes.

``verify_handoff()`` validates JSON/code evidence but opens no input/feature NPZ.
``load_full('train')`` retains RGB uint8 and FP32 targets for exactly 441 clips.
``load_full('dev')`` retains only RGB uint8 and labels for exactly 71 clips;
it never opens a teacher-feature NPZ. Historical D:/G: paths are descriptions.
All integrity failures abort; inference failures belong to the evaluator.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP4 = ROOT / 'experiments/local_dataset_step04_20261007'
PILOT = ROOT / 'experiments/local_pilot_step02_20261007'
HANDOFF = {'path': str(STEP4 / 'full_data_plan.json'), 'sha256': '3003209c04109a01f9737593b3645ffe7aef28e16530d84e53ba0caca77fca4b'}
SUMMARY = {'path': str(STEP4 / 'results/step04_summary.json'), 'sha256': 'dc0005af778507b8f2a7d1379bd984026e1728fda331dc0921afd0dec259bcad'}
PARENT_SHA = 'c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30'
SOURCE_SHA = 'd3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9'
STATE_SHA = '08fda31cd2a8ed5c1fe021da7d547229189d47d6f9df99e82cc276524ca19316'
STUDENT_KEYS = {'student_rgb', 'student_source_indices', 'label_tokens', 'metadata_json'}
TRAIN_KEYS = STUDENT_KEYS | {'teacher_bgr', 'teacher_source_indices', 'student_teacher_rows'}
FEATURE_KEYS = {'features', 'student_source_indices', 'teacher_source_indices', 'gather_indices', 'metadata_json'}


def sha256(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same(actual, expected, reason: str) -> None:
    if actual != expected:
        raise ValueError(reason)


class LocalBindings:
    """Workspace-only file identity/hash checks cached within one read transaction."""
    def __init__(self, root: Path = ROOT):
        self.root = root.resolve()
        self.items = {}
        self.stats = {}

    def path(self, value: str | Path) -> Path:
        path = Path(value).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f'Only existing workspace files may be read: {value}')
        return path

    @staticmethod
    def identity(path: Path) -> tuple:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def verify(self, ref: dict) -> Path:
        path = self.path(ref['path'])
        key, wanted = str(path), ref['sha256']
        identity = self.identity(path)
        if key in self.items:
            if self.items[key]['sha256'] != wanted or self.stats[key] != identity:
                raise ValueError(f'Conflicting/changed binding: {path}')
            return path
        if len(wanted) != 64 or sha256(path) != wanted:
            raise ValueError(f'Frozen file hash changed: {path}')
        if self.identity(path) != identity:
            raise ValueError(f'File changed while hashing: {path}')
        self.items[key] = {'path': key, 'sha256': wanted}
        self.stats[key] = identity
        return path

    def read(self, ref: dict) -> dict:
        raw = self.verify(ref).read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('JSON changed while reading')
        return json.loads(raw)

    def finish(self) -> list[dict]:
        for key, identity in self.stats.items():
            if self.identity(Path(key)) != identity:
                raise ValueError(f'Artifact changed during load: {key}')
        return list(self.items.values())


def verify_refs(value, checked: LocalBindings) -> None:
    """Check explicit references, excluding per-record NPZ references at this stage."""
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            if Path(value['path']).suffix.lower() == '.npz':
                raise ValueError('NPZ encountered in metadata-only evidence verification')
            checked.verify(value)
        else:
            for key, child in value.items():
                if key != 'records':
                    verify_refs(child, checked)
    elif isinstance(value, list):
        for child in value:
            verify_refs(child, checked)


def validate_cohort(parent: dict, source: dict, handoff: dict, docs: dict) -> tuple[list[str], list[str]]:
    rows = parent['records']
    ids = [r['id'] for r in rows]
    train = [r['id'] for r in rows if r['split'] == 'train']
    dev = [r['id'] for r in rows if r['split'] == 'dev']
    if len(ids) != 512 or len(set(ids)) != 512 or len(train) != 441 or len(dev) != 71:
        raise ValueError('Require unchanged full512 with train441/dev71')
    same(source['records'], rows, 'Local source rows differ from logical parent')
    same(source['parent_plan']['sha256'], PARENT_SHA, 'Local source parent SHA changed')
    same(handoff['train_ids'], train, 'Handoff train order changed')
    same(handoff['development_ids'], dev, 'Handoff dev order changed')
    same(docs['recipe']['train_ids'], train, 'Recipe train cohort changed')
    same(docs['recipe']['development_ids'], dev, 'Recipe dev cohort changed')
    inputs, features = docs['inputs'], docs['features']
    same(inputs['ids'], ids, 'Input ID order changed')
    same([r['id'] for r in inputs['records']], ids, 'Input entry order changed')
    same(inputs['record_count'], 512, 'Incomplete full inputs')
    same(features['ids'], train, 'Feature IDs must exactly match train441')
    same([r['id'] for r in features['records']], train, 'Feature entry order changed')
    same(features['record_count'], 441, 'Incomplete full features')
    if any(r['split'] != 'train' for r in features['records']):
        raise ValueError('Development target contamination')
    for row, entry in zip(rows, inputs['records'], strict=True):
        same(entry['split'], row['split'], 'Input split changed')
        same(entry['source_stem'], row['source_stem'], 'Input source changed')
        if row['split'] == 'dev':
            same(entry['teacher_frames'], 0, 'Development input must be student-only')
    pilot_ids = parent['preparation']['pilot_ids']
    same(docs['pilot_inputs']['ids'], pilot_ids, 'Original pilot inputs replaced')
    same(docs['pilot_features']['ids'], pilot_ids, 'Original pilot features replaced')
    if len(pilot_ids) != 8 or not set(pilot_ids).issubset(train):
        raise ValueError('Original eight must be unchanged training references')
    by_input = {r['id']: r for r in inputs['records']}
    by_feature = {r['id']: r for r in features['records']}
    for entry in docs['pilot_inputs']['records']:
        same(by_input[entry['id']], entry, 'Original8 input reference changed')
    for entry in docs['pilot_features']['records']:
        same(by_feature[entry['id']], entry, 'Original8 feature reference changed')
    if {r['source_stem'] for r in rows if r['split'] == 'train'} & {r['source_stem'] for r in rows if r['split'] == 'dev'}:
        raise ValueError('Train/development source overlap')
    for schedule, seed in zip(handoff['training_schedules'], [42, 43], strict=True):
        same(schedule['seed'], seed, 'Schedule seed changed')
        same(schedule['arms'], ['A_ctc', 'B_feature'], 'Schedule arms changed')
        batches = schedule['schedule']
        if len(batches) != 3200 or any(len(batch) != 2 for batch in batches):
            raise ValueError('Full schedule must have3200 pairs')
        counts = Counter(sid for batch in batches for sid in batch)
        if set(counts) != set(train) or set(counts.values()) != {14, 15}:
            raise ValueError('Schedule includes dev/test or lacks training coverage')
        same(dict(sorted(counts.items())), schedule['per_clip_uses'], 'Schedule coverage evidence differs')
    return train, dev


def _handoff(checked: LocalBindings) -> dict:
    summary, handoff = checked.read(SUMMARY), checked.read(HANDOFF)
    same(summary['status'], 'completed', 'Step4 must be completed')
    same(summary['full_data_plan'], HANDOFF, 'Step4 summary does not bind handoff')
    same(handoff['schema'], 'lipnet-step04-full-data-handoff-v1', 'Wrong handoff schema')
    same(handoff['status'], 'data_and_schedules_frozen_pending_step05_execution_binding', 'Wrong handoff state')
    same(handoff['artifacts'], summary['artifacts'], 'Step4 evidence differs')
    same(handoff['parent_plan']['sha256'], PARENT_SHA, 'Logical parent changed')
    same(handoff['source_plan']['sha256'], SOURCE_SHA, 'Local source plan changed')
    refs = dict(handoff['artifacts'])
    docs = {key: checked.read(ref) for key, ref in refs.items()}
    parent, source = checked.read(handoff['parent_plan']), checked.read(handoff['source_plan'])
    refs['input_plan'] = docs['inputs']['preparation_plan']
    teacher = docs['teacher_plan']
    for key, teacher_key in [('pilot_inputs', 'pilot_inputs'), ('pilot_features', 'pilot_features'), ('pilot_teacher_plan', 'pilot_teacher_plan')]:
        refs[key] = teacher[teacher_key]
    for key in ['input_plan', 'pilot_inputs', 'pilot_features', 'pilot_teacher_plan']:
        docs[key] = checked.read(refs[key])
    for value in [summary, handoff, parent, source, *docs.values()]:
        verify_refs(value, checked)
    train, dev = validate_cohort(parent, source, handoff, docs)
    inputs, features = docs['inputs'], docs['features']
    same(inputs['status'], 'passed', 'Input preparation not passed')
    same(inputs['phase'], 'all', 'Input scope must be full')
    same(features['status'], 'completed', 'Feature extraction incomplete')
    same(features['phase'], 'train', 'Teacher targets must be train-only')
    for key, ref in [('data_plan', handoff['parent_plan']), ('source_plan', handoff['source_plan']),
                     ('input_manifest', refs['inputs']), ('input_audit', refs['input_audit']), ('comparison_recipe', refs['recipe'])]:
        same(teacher[key], ref, 'Teacher plan evidence mismatch: ' + key)
    for key in ['input_manifest', 'input_audit', 'pilot_features', 'pilot_teacher_plan', 'comparison_recipe', 'runtime']:
        same(features[key], teacher[key], 'Feature extraction provenance differs: ' + key)
    same(features['teacher_plan'], refs['teacher_plan'], 'Full feature teacher plan mismatch')
    same(teacher['runtime'], docs['pilot_teacher_plan']['runtime'], 'Pilot/full teacher runtime changed')
    same(teacher['checkpoint'], docs['pilot_teacher_plan']['checkpoint'], 'Pilot/full teacher checkpoint changed')
    for key in ['state_sha256_before', 'state_sha256_after']:
        same(features[key], STATE_SHA, 'Teacher visual state changed')
    same(teacher['expected_state_sha256'], STATE_SHA, 'Teacher expected state changed')
    for key in ['audio_used', 'dev_input_npz_opened', 'dev_teacher_targets_created', 'decoder_instantiated']:
        same(features[key], False, 'Invalid teacher execution scope: ' + key)
    for key in ['student_optimization_steps', 'teacher_optimization_steps']:
        same(features[key], 0, 'Preparation performed optimization')
    audit, review = docs['input_audit'], docs['feature_audit']
    for key, expected in [('status', 'passed'), ('scope', 'all512'), ('record_count', 512),
                          ('manifest', refs['inputs']), ('plan', handoff['parent_plan']), ('source_plan', handoff['source_plan'])]:
        same(audit[key], expected, 'Full input audit mismatch: ' + key)
    same([r['id'] for r in audit['records']], inputs['ids'], 'Input audit coverage/order differs')
    for key, expected in [('status', 'passed'), ('record_count', 441), ('manifest', refs['features']),
                          ('input_manifest', refs['inputs']), ('teacher_plan', refs['teacher_plan']),
                          ('input_audit', refs['input_audit']), ('development_target_count', 0)]:
        same(review[key], expected, 'Full feature audit mismatch: ' + key)
    same([r['id'] for r in review['records']], train, 'Feature audit coverage/order differs')
    same(docs['visual_review']['status'], 'passed_sampled_review', 'Visual review incomplete')
    same(docs['visual_review']['input_manifest'], refs['inputs'], 'Visual review points elsewhere')
    return {'documents': docs, 'refs': refs, 'parent': parent, 'source': source,
            'handoff': handoff, 'summary': summary, 'bindings': checked.finish(), 'train_ids': train, 'dev_ids': dev}


def verify_handoff() -> dict:
    """Return checked documents/refs, parent/source JSON and handoff/summary JSON."""
    return _handoff(LocalBindings())


def validate_student(row: dict, entry: dict, arrays: dict, metadata: dict, scope: str) -> None:
    same(row['split'], scope, 'Wrong split selected for loader')
    same(entry['split'], scope, 'Input entry split differs')
    same(metadata['record'], row, 'Student transcript/timing/source changed')
    same(metadata['id'], row['id'], 'Student ID differs')
    same(metadata['plan_sha256'], PARENT_SHA, 'Student parent binding differs')
    pixels = arrays['student_rgb']
    if pixels.dtype != np.uint8 or pixels.shape != (row['frames'], 96, 96, 3):
        raise ValueError('Student must remain T x96 x96 x3 uint8 RGB')
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    same(digest, metadata['student_rgb_sha256'], 'Embedded student pixel digest differs')
    same(digest, entry['student_rgb_sha256'], 'Manifest student pixel digest differs')
    tokens = row['token_ids']
    if (not tokens or any(type(t) is not int or not 1 <= t < 69 for t in tokens)
            or len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])) > row['frames']):
        raise ValueError('Invalid or CTC-infeasible frozen tokens')
    expected = np.asarray([round((row['start_time'] + k / 25) * row['video_fps']) for k in range(row['frames'])], np.int64)
    for key, wanted in [('student_source_indices', expected), ('label_tokens', np.asarray(tokens, np.int64))]:
        if arrays[key].dtype != np.int64 or not np.array_equal(arrays[key], wanted):
            raise ValueError('Student frame/token contract differs: ' + key)
    if expected.min() < 0 or expected.max() >= metadata['source_frame_count']:
        raise ValueError('Student native indices outside physical source')
    if scope == 'dev':
        same(entry['teacher_frames'], 0, 'Development teacher frames forbidden')
        same(set(metadata), {'schema', 'id', 'record', 'plan_sha256', 'source_frame_count', 'student_rgb_sha256',
                            'student_trace', 'source_verification', 'preparation_seconds', 'local_full_preparation'}, 'Development metadata must remain student-only')
        same(metadata['schema'], 'local-tar-step04-dev-input-v1', 'Development schema differs')


def validate_target(row: dict, input_entry: dict, entry: dict, arrays: dict,
                    target: dict, meta: dict, plan: dict, plan_ref: dict, reused: bool) -> None:
    same(row['split'], 'train', 'Development cannot consume teacher targets')
    same({**meta, 'feature_file': entry['feature_file'], 'status': 'completed'}, entry, 'Embedded feature provenance differs from immutable manifest')
    for key, wanted in [('id', row['id']), ('speaker_id', row['speaker_id']), ('source_stem', row['source_stem']),
                        ('split', 'train'), ('teacher_plan_sha256', plan_ref['sha256']), ('input_record', input_entry),
                        ('data_record_sha256', canonical_sha(row)), ('state_sha256_before', STATE_SHA), ('state_sha256_after', STATE_SHA),
                        ('aligned_feature_shape', [row['frames'], 512]), ('feature_dtype', 'float32'), ('optimization_steps', 0),
                        ('trainable_parameters', 0), ('audio_used', False), ('decoder_instantiated', False), ('transcript_used_for_teacher_forward', False)]:
        same(meta[key], wanted, 'Teacher contract differs: ' + key)
    for key in ['data_plan', 'source_plan', 'input_audit', 'checkpoint', 'runtime']:
        same(meta[key], plan[key], 'Teacher evidence differs: ' + key)
    same(meta['input_manifest'], plan['pilot_manifest'] if reused else plan['input_manifest'], 'Mixed pilot/full provenance changed')
    value = target['features']
    if value.dtype != np.float32 or value.shape != (row['frames'], 512) or not np.isfinite(value).all():
        raise ValueError('Teacher target must be finite T x512 FP32')
    if np.any(np.linalg.norm(value.astype(np.float64), axis=1) <= 1e-8) or np.all(value == value[0]):
        raise ValueError('Teacher target has zero rows or temporal collapse')
    same(hashlib.sha256(value.tobytes()).hexdigest(), meta['feature_values_sha256'], 'Teacher target values changed')
    for key, data_key in [('student_source_indices', 'student_source_indices'), ('teacher_source_indices', 'teacher_source_indices'), ('gather_indices', 'student_teacher_rows')]:
        if target[key].dtype != np.int64 or arrays[data_key].dtype != np.int64 or not np.array_equal(target[key], arrays[data_key]):
            raise ValueError('Target/input source-frame correspondence differs: ' + key)
    dense, gather = target['teacher_source_indices'], target['gather_indices']
    if dense.ndim != 1 or len(dense) < 1 or not np.array_equal(dense, np.arange(dense[0], dense[-1] + 1, dtype=np.int64)):
        raise ValueError('Teacher native source indices are not dense')
    if gather.shape != (row['frames'],) or np.any(gather < 0) or np.any(gather >= len(dense)):
        raise ValueError('Teacher gather outside native interval')
    if not np.array_equal(dense[gather], arrays['student_source_indices']):
        raise ValueError('Teacher features map to another physical frame')
    same(meta['native_feature_shape'], [1, len(dense), 512], 'Native feature shape differs')


def load_full(scope: str = 'train') -> dict:
    """Return raw uint8 clips, train-only saved FP32 targets, and all bindings.

    The returned ``clips`` mapping is in parent order. Each clip contains id,
    split, source_stem, speaker_id, text, pixels, tokens; train also contains
    teacher_features. The caller normalizes one clip at a time on the GPU.
    """
    if scope not in {'train', 'dev'}:
        raise ValueError("scope must be exactly 'train' or 'dev'")
    checked = LocalBindings()
    bundle = _handoff(checked)
    docs, refs = bundle['documents'], bundle['refs']
    rows = {r['id']: r for r in bundle['parent']['records']}
    ids = bundle['train_ids'] if scope == 'train' else bundle['dev_ids']
    input_rows = {r['id']: r for r in docs['inputs']['records']}
    feature_rows = {r['id']: r for r in docs['features']['records']}
    input_reviews = {r['id']: r for r in docs['input_audit']['records']}
    feature_reviews = {r['id']: r for r in docs['feature_audit']['records']}
    source_rows = {r['stem']: r for r in bundle['source']['sources']}
    pilot_ids = set(docs['pilot_inputs']['ids'])
    clips = {}
    for sid in ids:
        row, entry, audit = rows[sid], input_rows[sid], input_reviews[sid]
        reused = sid in pilot_ids
        home = PILOT if reused else STEP4
        input_ref = {'path': entry['npz_path'], 'sha256': entry['npz_sha256']}
        path = checked.verify(input_ref)
        same(path, home / 'inputs/records' / (sid + '.npz'), 'Input path differs from original provenance')
        sidecar = home / 'inputs/metadata' / (sid + '.json')
        same(checked.read({'path': str(sidecar), 'sha256': audit['metadata_sha256']}), entry, 'Input sidecar differs')
        same(audit['npz_sha256'], input_ref['sha256'], 'Input audit refers to another NPZ')
        with np.load(path, allow_pickle=False) as saved:
            same(set(saved.files), TRAIN_KEYS if scope == 'train' else STUDENT_KEYS, 'Unexpected input payload keys')
            keys = ['student_rgb', 'student_source_indices', 'label_tokens']
            if scope == 'train':
                keys += ['teacher_source_indices', 'student_teacher_rows']
            arrays = {key: saved[key] for key in keys}
            metadata = json.loads(str(saved['metadata_json'].item()))
        validate_student(row, entry, arrays, metadata, scope)
        original = docs['pilot_inputs'] if reused else docs['inputs']
        same(metadata['local_pilot_preparation' if reused else 'local_full_preparation'], original['preparation_runner'], 'Preparation producer binding differs')
        provenance = metadata['source_verification']
        source = source_rows[row['source_stem']]
        for key, value in [('transport', 'audited-local-tar-v1'), ('source_plan_path', bundle['handoff']['source_plan']['path']),
                           ('source_plan_sha256', SOURCE_SHA), ('parent_plan_sha256', PARENT_SHA), ('source_stem', row['source_stem']),
                           ('members', {k: v for k, v in source['members'].items() if k != 'audio'}),
                           ('audio_used_as_model_input', False), ('transport_code', original['transport_code'])]:
            same(provenance[key], value, 'D transport evidence differs: ' + key)
        same(entry['source_verification'], provenance, 'Input provenance differs from manifest')
        clip = {'id': sid, 'split': scope, 'source_stem': row['source_stem'], 'speaker_id': row['speaker_id'],
                'text': row['text'], 'pixels': arrays['student_rgb'], 'tokens': list(row['token_ids'])}
        if scope == 'train':
            feature = feature_rows[sid]
            target_path = checked.verify(feature['feature_file'])
            same(target_path, home / 'features' / (sid + '.npz'), 'Feature path changes original pilot/full provenance')
            review = feature_reviews[sid]
            same(review['feature_file'], feature['feature_file'], 'Feature audit refers to another target')
            same(review['input_file'], input_ref, 'Feature audit refers to another input')
            feature_sidecar = home / 'teacher_results/records' / (sid + '.json')
            same(checked.read({'path': str(feature_sidecar), 'sha256': review['sidecar_sha256']}), feature, 'Feature sidecar differs')
            with np.load(target_path, allow_pickle=False) as saved:
                same(set(saved.files), FEATURE_KEYS, 'Unexpected target payload keys')
                target = {key: saved[key] for key in FEATURE_KEYS - {'metadata_json'}}
                meta = json.loads(str(saved['metadata_json'].item()))
            plan_name = 'pilot_teacher_plan' if reused else 'teacher_plan'
            validate_target(row, entry, feature, arrays, target, meta, docs[plan_name], refs[plan_name], reused)
            clip['teacher_features'] = target['features']
        clips[sid] = clip
    total_frames = sum(len(c['pixels']) for c in clips.values())
    same(total_frames, 50519 if scope == 'train' else 7900, 'Loaded frame total differs')
    counts = {'clips': len(ids), 'student_frames': total_frames, 'train': len(ids) if scope == 'train' else 0,
              'dev': len(ids) if scope == 'dev' else 0, 'test': 0, 'speakers': len({c['speaker_id'] for c in clips.values()}),
              'feature_width': 512 if scope == 'train' else 0,
              'pixel_bytes': sum(c['pixels'].nbytes for c in clips.values()),
              'feature_bytes': sum(c['teacher_features'].nbytes for c in clips.values()) if scope == 'train' else 0}
    return {'clips': clips, 'ids': ids, 'bindings': checked.finish(), 'parent_plan': bundle['handoff']['parent_plan'],
            'source_plan': bundle['handoff']['source_plan'], 'input_manifest': refs['inputs'], 'feature_manifest': refs['features'],
            'teacher_plan': refs['teacher_plan'], 'handoff': dict(HANDOFF), 'step04_summary': dict(SUMMARY), 'counts': counts, 'scope': scope}
