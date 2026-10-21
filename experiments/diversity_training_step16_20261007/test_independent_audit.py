"""CPU-only negative and numerical tests of the independent Step16 audit."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).with_name('independent_audit.py')
spec = importlib.util.spec_from_file_location('step16_audit_test_subject', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


MAIN = [{'start_update': 1, 'end_update': 2400, 'lr': .001},
        {'start_update': 2401, 'end_update': 3000, 'lr': .0003},
        {'start_update': 3001, 'end_update': 3600, 'lr': .0001}]


@pytest.mark.parametrize('update,expected', [(1,.001),(2400,.001),(2401,.0003),(3000,.0003),(3001,.0001),(3600,.0001)])
def test_global_learning_rate_boundaries(update, expected):
    assert audit.learning_rate(update, MAIN) == expected


@pytest.mark.parametrize('update', [0, -1, 3601, 2.0, True])
def test_invalid_global_learning_rate_position_rejected(update):
    with pytest.raises(ValueError):
        audit.learning_rate(update, MAIN)


def test_learning_rate_overlap_rejected():
    with pytest.raises(ValueError):
        audit.learning_rate(2400, [*MAIN, {'start_update': 2400, 'end_update': 2400, 'lr': .01}])


def fake_student():
    row = {'id': 'student', 'frames': 3, 'token_ids': [1,2], 'start_time': 1., 'end_time': 1.12, 'video_fps': 25.}
    entry = {'logical_data_plan': {'sha256': 'generation-sha'}}
    pixels = np.arange(3*96*96*3, dtype=np.uint8).reshape(3,96,96,3)
    raw = hashlib.sha256(pixels.tobytes()).hexdigest()
    item = {'source_verification': {'members': {'video': {'sha256': 'video'}}}, 'student_rgb_sha256': raw}
    arrays = {'student_rgb': pixels, 'label_tokens': np.array([1,2], dtype=np.int64),
              'student_source_indices': np.array([25,26,27], dtype=np.int64)}
    embedded = {'id': 'student', 'record': copy.deepcopy(row), 'plan_sha256': 'generation-sha',
                'source_verification': item['source_verification'], 'student_rgb_sha256': raw, 'source_frame_count': 100}
    return row, entry, item, arrays, embedded


def test_generation_aware_student_binding():
    values = fake_student()
    result = audit.validate_student(*values)
    assert result['tokens_sha256'] == audit.num.nested_hash([1,2])
    assert result['pixels_sha256'] == audit.num.nested_hash(values[3]['student_rgb'])


@pytest.mark.parametrize('fault', ['pixels','generation','source','indices','index_dtype','tokens','physical_length','shape'])
def test_student_corruption_rejected(fault):
    row, entry, item, arrays, embedded = fake_student()
    if fault == 'pixels': arrays['student_rgb'][0,0,0,0] ^= 1
    elif fault == 'generation': embedded['plan_sha256'] = 'wrong'
    elif fault == 'source': embedded['source_verification'] = {}
    elif fault == 'indices': arrays['student_source_indices'][0] += 1
    elif fault == 'index_dtype': arrays['student_source_indices'] = arrays['student_source_indices'].astype(np.int32)
    elif fault == 'tokens': arrays['label_tokens'][0] = 0
    elif fault == 'physical_length': embedded['source_frame_count'] = 27
    elif fault == 'shape': arrays['student_rgb'] = arrays['student_rgb'][:2]
    with pytest.raises(ValueError): audit.validate_student(row, entry, item, arrays, embedded)


def test_repeated_label_ctc_infeasibility_rejected():
    row, entry, item, arrays, embedded = fake_student()
    row['token_ids'] = [1,1,1]
    embedded['record'] = copy.deepcopy(row)
    arrays['label_tokens'] = np.array(row['token_ids'], dtype=np.int64)
    with pytest.raises(ValueError): audit.validate_student(row, entry, item, arrays, embedded)


def plan(cohort):
    train = ['t'+str(i) for i in range(24 if cohort == 'train24' else 72)]
    return {'cohort': cohort, 'train_ids': train, 'common_train24_ids': train[:24],
            'dev_ids': ['d'+str(i) for i in range(16)], 'probe_ids': ['p'+str(i) for i in range(14)],
            'probe_known_ids': ['p'+str(i) for i in range(8)], 'probe_unseen_ids': ['p'+str(i) for i in range(8,14)]}


@pytest.mark.parametrize('cohort,count', [('train24',158),('train72',254)])
def test_full_fixed_evaluation_count(cohort,count):
    value = plan(cohort)
    actual = 0
    for update in audit.EVAL_UPDATES:
        for condition in ('normal','static','shuffled') if update == 3600 else ('normal',):
            ids, groups = audit.evaluation_groups(value, update, condition)
            actual += len(ids)
            assert len(ids) == len(set(ids))
            assert len(groups['dev16']) == 16
    assert actual == count


def test_common24_is_reaggregation_not_forward_duplicate():
    value = plan('train72')
    ids, groups = audit.evaluation_groups(value, 2400, 'normal')
    assert len(ids) == 88 and len(groups['train_common24']) == 24
    assert all(ids.count(sid) == 1 for sid in groups['train_common24'])


@pytest.mark.parametrize('update,condition', [(0,'normal'), (1200,'static'), (3000,'shuffled'), (3600,'audio')])
def test_unplanned_evaluation_rejected(update,condition):
    with pytest.raises(ValueError): audit.evaluation_groups(plan('train24'), update, condition)


def test_duplicate_forward_input_rejected():
    value = plan('train24'); value['dev_ids'][0] = value['train_ids'][0]
    with pytest.raises(ValueError): audit.evaluation_groups(value,2400,'normal')


def comparison_for_state(left, right):
    # Construct a single-array replay record without calling a production comparator.
    a,b = left['model']['weight'],right['model']['weight']
    delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
    absolute = float(delta.max(initial=0)); exact = np.array_equal(a,b)
    item = {'path':'model.weight','category':'model','shape':list(a.shape),'dtype':str(a.dtype),
            'bitwise_equal':bool(exact),'passed':True,'max_absolute_difference':absolute,
            'max_relative_difference_denominator_floor_1e12':float((delta/np.maximum(np.abs(a.astype(np.float64)),1e-12)).max(initial=0)),
            'atol':1e-6,'rtol':1e-5}
    return {'passed':True,'bitwise_equal':bool(exact),'arrays_checked':1,'arrays_with_bitwise_difference':int(not exact),
            'maximum_absolute_difference':absolute,'records':[item],'failures':[]}


def test_replay_within_predeclared_tolerance():
    a = {'model': {'weight':np.array([1.],np.float32)}, 'rng': {'cursor': 42}}
    b = copy.deepcopy(a); b['model']['weight'][0] += 1e-6
    result = audit.compare_replay_state(a,b,comparison_for_state(a,b))
    assert not result['bitwise_equal'] and result['arrays_checked'] == 1


@pytest.mark.parametrize('fault', ['values','rng','dtype','extra_path','reported_max','relaxed_tolerance'])
def test_replay_corruption_rejected(fault):
    a = {'model': {'weight':np.array([1.],np.float32)}, 'rng': {'cursor':42}}
    b = copy.deepcopy(a)
    evidence = comparison_for_state(a,b)
    if fault == 'values': b['model']['weight'][0] += .1
    elif fault == 'rng': b['rng']['cursor'] += 1
    elif fault == 'dtype': b['model']['weight'] = b['model']['weight'].astype(np.float64)
    elif fault == 'extra_path': evidence['records'].append({**evidence['records'][0], 'path':'extra'})
    elif fault == 'reported_max': evidence['maximum_absolute_difference'] = 1.
    elif fault == 'relaxed_tolerance': evidence['records'][0]['atol'] = 1.
    with pytest.raises(ValueError): audit.compare_replay_state(a,b,evidence)


def test_optimizer_learning_rate_checked_before_legacy_buffers(monkeypatch):
    calls=[]
    monkeypatch.setattr(audit.num, 'validate_optimizer', lambda state,arm,updates:calls.append((state,arm,updates)))
    state={'optimizer':{'param_groups':[{'lr':.0001}]}}
    audit.validate_optimizer(state,3600,.0001)
    assert state['optimizer']['param_groups'][0]['lr'] == .0001
    assert calls[0][0]['optimizer']['param_groups'][0]['lr'] == .001
    with pytest.raises(ValueError): audit.validate_optimizer(state,3600,.001)


def test_history_lr_cannot_reset_after_resume(monkeypatch):
    monkeypatch.setattr(audit.num, 'validate_history', lambda *args: {'updates':1})
    with pytest.raises(ValueError):
        audit.validate_history([{'update':2401,'learning_rate':.001}], {'lr_segments':MAIN}, {})


def test_no_production_imports_in_auditor_source():
    source=PATH.read_text(encoding='utf-8')
    assert 'import engine' not in source and 'import protocol' not in source and 'import evaluate' not in source
    assert all('independent_audit.py' in str(path) for path,_ in audit.HELPERS)


def test_previous_selection_and_schedule_still_reconstruct_without_models():
    bundle=audit.data15.source_metadata()
    data=audit.read(audit.PREVIOUS/'data_plan.json',audit.EXPECTED15['data_plan.json'])
    chosen=audit.data15.independent_selection(bundle)
    for key in ('train24_ids','train72_ids','dev16_ids'):
        assert chosen[key]==data['selection'][key]
    audit.data15.verify_selected_separation(data['selection'],bundle)
    assert audit.data15.audit_design(data,bundle,audit.EXPECTED15['training_design.json'])[1]==2959600
