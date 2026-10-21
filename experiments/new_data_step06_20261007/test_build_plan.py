"""New120 cohort/provenance guards, using existing local metadata only."""
import copy
import importlib.util
import json
from pathlib import Path, PureWindowsPath

import pytest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('step06_source_builder_test', BASE / 'build_plan.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture(scope='module')
def evidence():
    selection = b.read_json(b.SELECTION)
    protected = b.read_json(b.ROOT / 'results/step18-intake/run01/protected_reference.json')
    roles = b.read_json(b.AUDIT / 'archive_results/readiness_role_map.json')
    return selection, protected, roles


def test_fixed120_roles_and_all_development_speaker_recordings(evidence):
    result = b.validate_selection(*evidence)
    assert result['new_development_speakers'] == ['C040', 'C059', 'C769']
    assert result['new_development_all_recordings_protected'] == 36
    assert result['original_v4_validation']['clips'] == 20
    assert result['original_v4_final_test']['clips'] == 50


@pytest.mark.parametrize('mutation', ['split', 'duplicate', 'source_text', 'time', 'ctc_min', 'old_speaker', 'old_transcript', 'role', 'unselected_id', 'dev_unselected_train'])
def test_selection_contamination_or_contract_changes_fail(evidence, mutation):
    selection, protected, roles = copy.deepcopy(evidence)
    row = selection['records'][0]
    if mutation == 'split':
        row['split'] = 'val'
    elif mutation == 'duplicate':
        selection['records'][1] = copy.deepcopy(row)
    elif mutation == 'source_text':
        row['source_text'] = '다른 내용'
    elif mutation == 'time':
        row['start_time'] = row['end_time'] + 1
    elif mutation == 'ctc_min':
        row['ctc_min_frames'] += 1
    elif mutation == 'old_speaker':
        protected['guards']['blocked_existing_speaker_ids'].append(row['speaker_id'])
    elif mutation == 'old_transcript':
        protected['guards']['blocked_existing_normalized_texts'].append(row['text'])
    else:
        stem = PureWindowsPath(row['source_video']['path']).stem
        role = next(r for r in roles['records'] if r['source_stem'] == stem)
        if mutation == 'role':
            role['speaker_role'] = 'new_pilot_development'
        elif mutation == 'unselected_id':
            role['stage18_pilot_selected_clip_ids'][0] = 'unselected'
        else:
            role = next(r for r in roles['records'] if r['speaker_id'] == 'C040' and r['readiness_role'].endswith('unselected_recording'))
            role['speaker_role'] = 'new_pilot_train'
    with pytest.raises(b.PlanError):
        b.validate_selection(selection, protected, roles)


def test_ambiguous_member_identity_rejected():
    with pytest.raises(b.PlanError):
        b.unique([{'name': 'same'}, {'name': 'same'}], 'name')


def test_workspace_evidence_rejects_historical_drive_reads():
    with pytest.raises(b.PlanError):
        b.local('G:/untrusted/history.json')
    with pytest.raises(b.PlanError):
        b.local('D:/source/archive.tar')


def test_role_and_tar_member_must_agree():
    with pytest.raises(b.PlanError):
        b.validate_role_member({'archive_path': 'a.tar', 'archive_member': 'x.mp4'},
                               {'archive_path': 'b.tar', 'member_name': 'x.mp4'})


def test_actual_bound_plan_reads_no_archive_or_G_payload(monkeypatch, evidence):
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert PureWindowsPath(str(path)).drive.upper() not in {'D:', 'G:'}, 'Original media payload was opened'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    monkeypatch.setattr(b.transport, 'materialize_source', lambda *args, **kwargs: pytest.fail('No materialization in planning'))
    plan = b.collect_plan()
    assert plan['records'] == evidence[0]['records']
    assert plan['counts']['train'] == 96 and plan['counts']['val'] == 24
    assert len(plan['sources']) == 30 and len(plan['archives']) == 6
    assert len(plan['historical_G_size_differences']) == 28
    assert plan['storage']['bytes_by_role'] == {'video': 9016121942, 'audio': 1767892510, 'label': 89264746}
    assert plan['storage']['max_one_recording_combined_bytes'] == 439572369
    for source in plan['sources']:
        assert source['training_ready'] is False
        assert source['old59_time_offset_applied'] is False and source['verified_new_time_offset_seconds'] is None
        assert source['video_frame_count'] > 0 and source['video_duration_seconds'] > 0
        assert len(source['record_ids']) == 4
        assert set(source['members']) == {'video', 'label', 'audio'}
