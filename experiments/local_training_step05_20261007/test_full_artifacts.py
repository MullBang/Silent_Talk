"""Integrity/split/provenance rejection tests for the full-cohort reader."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('step05_full_artifacts_under_test', Path(__file__).with_name('full_artifacts.py'))
art = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(art)


@pytest.fixture(scope='module')
def evidence():
    return art.verify_handoff()


@pytest.fixture(scope='module')
def one(evidence):
    docs = evidence['documents']
    sid = evidence['train_ids'][0]
    row = next(r for r in evidence['parent']['records'] if r['id'] == sid)
    entry = next(r for r in docs['inputs']['records'] if r['id'] == sid)
    feature = next(r for r in docs['features']['records'] if r['id'] == sid)
    with np.load(entry['npz_path'], allow_pickle=False) as z:
        arrays = {key: z[key] for key in ['student_rgb', 'student_source_indices', 'teacher_source_indices', 'student_teacher_rows', 'label_tokens']}
        metadata = json.loads(str(z['metadata_json'].item()))
    with np.load(feature['feature_file']['path'], allow_pickle=False) as z:
        target = {key: z[key] for key in art.FEATURE_KEYS - {'metadata_json'}}
        meta = json.loads(str(z['metadata_json'].item()))
    return row, entry, feature, arrays, metadata, target, meta


def test_completed_handoff_and_mixed_provenance(evidence):
    assert len(evidence['train_ids']) == 441
    assert len(evidence['dev_ids']) == 71
    assert len(evidence['documents']['pilot_features']['records']) == 8
    assert not any(Path(ref['path']).suffix == '.npz' for ref in evidence['bindings'])


@pytest.mark.parametrize('mutation', ['dev_target', 'reordered_train', 'duplicate_input', 'modified_reused_input', 'modified_reused_feature', 'dev_in_schedule'])
def test_rejects_cohort_or_reuse_changes(evidence, mutation):
    docs = copy.deepcopy(evidence['documents'])
    handoff = copy.deepcopy(evidence['handoff'])
    if mutation == 'dev_target':
        docs['features']['records'][0]['split'] = 'dev'
    elif mutation == 'reordered_train':
        handoff['train_ids'].reverse()
    elif mutation == 'duplicate_input':
        docs['inputs']['records'][1] = docs['inputs']['records'][0]
    elif mutation == 'modified_reused_input':
        sid = docs['pilot_inputs']['ids'][0]
        next(r for r in docs['inputs']['records'] if r['id'] == sid)['npz_path'] = 'copy.npz'
    elif mutation == 'modified_reused_feature':
        sid = docs['pilot_features']['ids'][0]
        next(r for r in docs['features']['records'] if r['id'] == sid)['teacher_plan_sha256'] = '0' * 64
    else:
        handoff['training_schedules'][0]['schedule'][0][0] = evidence['dev_ids'][0]
    with pytest.raises(ValueError):
        art.validate_cohort(evidence['parent'], evidence['source'], handoff, docs)


def test_local_binding_rejects_tamper(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('{"a":1}', encoding='utf-8')
    ref = {'path': str(path), 'sha256': art.sha256(path)}
    reader = art.LocalBindings(tmp_path)
    assert reader.read(ref) == {'a': 1}
    path.write_text('{"a":2}', encoding='utf-8')
    with pytest.raises(ValueError):
        reader.verify(ref)


def test_local_binding_rejects_outside_workspace(tmp_path):
    reader = art.LocalBindings(tmp_path / 'inside')
    path = tmp_path / 'outside.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        reader.read({'path': str(path), 'sha256': art.sha256(path)})


def test_valid_student(one):
    row, entry, _, arrays, metadata, _, _ = one
    art.validate_student(row, entry, arrays, metadata, 'train')


@pytest.mark.parametrize('mutation', ['pixels', 'float_pixels', 'shape', 'label', 'index', 'wrong_split', 'logical_text'])
def test_rejects_student_corruption(one, mutation):
    row, entry, _, arrays, metadata, _, _ = copy.deepcopy(one)
    if mutation == 'pixels':
        arrays['student_rgb'][0, 0, 0, 0] ^= 1
    elif mutation == 'float_pixels':
        arrays['student_rgb'] = arrays['student_rgb'].astype(np.float32)
    elif mutation == 'shape':
        arrays['student_rgb'] = arrays['student_rgb'][:-1]
    elif mutation == 'label':
        arrays['label_tokens'][0] = 0
    elif mutation == 'index':
        arrays['student_source_indices'][0] += 1
    elif mutation == 'wrong_split':
        entry['split'] = 'dev'
    else:
        metadata['record']['text'] += 'x'
    with pytest.raises(ValueError):
        art.validate_student(row, entry, arrays, metadata, 'train')


def test_valid_new_target(one, evidence):
    row, entry, feature, arrays, _, target, meta = one
    art.validate_target(row, entry, feature, arrays, target, meta, evidence['documents']['teacher_plan'], evidence['refs']['teacher_plan'], False)


@pytest.mark.parametrize('mutation', ['nan', 'zero', 'float64', 'width', 'wrong_gather', 'wrong_input', 'pretend_reused'])
def test_rejects_target_corruption(one, evidence, mutation):
    row, entry, feature, arrays, _, target, meta = copy.deepcopy(one)
    reused = False
    if mutation == 'nan':
        target['features'][0, 0] = np.nan
    elif mutation == 'zero':
        target['features'][0] = 0
    elif mutation == 'float64':
        target['features'] = target['features'].astype(np.float64)
    elif mutation == 'width':
        target['features'] = target['features'][:, :-1]
    elif mutation == 'wrong_gather':
        target['gather_indices'][0] += 1
    elif mutation == 'wrong_input':
        meta['input_record']['npz_sha256'] = '0' * 64
    else:
        reused = True
    plan_name = 'pilot_teacher_plan' if reused else 'teacher_plan'
    with pytest.raises(ValueError):
        art.validate_target(row, entry, feature, arrays, target, meta, evidence['documents'][plan_name], evidence['refs'][plan_name], reused)


def test_invalid_scope_rejected_without_io(monkeypatch):
    monkeypatch.setattr(art, '_handoff', lambda _: pytest.fail('Invalid scope reached evidence I/O'))
    with pytest.raises(ValueError):
        art.load_full('all')


def test_all_dev_inputs_never_open_target_npz(monkeypatch):
    original = np.load
    opened = []
    def guarded(path, *args, **kwargs):
        assert 'features' not in Path(path).parts
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(art.np, 'load', guarded)
    loaded = art.load_full('dev')
    assert len(opened) == len(loaded['ids']) == 71
    assert loaded['counts']['student_frames'] == 7900
    assert loaded['counts']['feature_bytes'] == 0
    assert all(c['pixels'].dtype == np.uint8 and 'teacher_features' not in c for c in loaded['clips'].values())
