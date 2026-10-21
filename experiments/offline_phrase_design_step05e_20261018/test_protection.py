"""Meaningful leakage/canonical identity checks without media or network calls."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('protection', Path(__file__).with_name('prepare_inventory.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_angle_group():
    """Camera angle variants stay in one recording group."""
    assert p.recording_key('lip_J_1_F_02_C004_A_011') == p.recording_key('lip_J_1_F_02_C004_B_011')
    assert p.recording_key('lip_J_1_F_02_C004_A_011') != p.recording_key('lip_J_1_F_02_C004_A_012')


def test_malformed_stem():
    """Malformed identifiers must not silently collide."""
    with pytest.raises(ValueError):
        p.recording_key('C004_011')


@pytest.fixture
def row_scope():
    """Minimal metadata demonstrates both old and new text protection."""
    return ({'id': 'x', 'source_stem': 'lip_J_1_F_02_C004_A_011', 'text': '이전 전사',
             'video_sha256': 'video', 'label_sha256': 'label'},
            {'protected_parent_ids': [], 'protected_recording_keys': [], 'blocked_source_stem': [],
             'blocked_video_sha256': [], 'blocked_label_sha256': [], 'protected_full_text_keys': [],
             'protected_phrase_text_keys': ['너무많이막혀'], 'held_short_alignment_parent_ids': []})


def test_canonical_parent_contains_dev_phrase(row_scope):
    """A protected phrase embedded in a longer canonical parent is blocked."""
    row, scope = row_scope
    assert 'contains_current_dev_phrase' in p.reasons_for(row, '오늘은 너무 많이 막혀 답답해', scope)


def test_historical_text_stays_protected_after_correction(row_scope):
    """Correction does not erase an old transcript exclusion."""
    row, scope = row_scope
    scope['protected_full_text_keys'] = ['이전전사']
    assert 'protected_historical_or_canonical_full_transcript' in p.reasons_for(row, '새 전사', scope)


def test_angle_alias_and_oov_hold(row_scope):
    """Angle variants and acoustic alignment holds remain excluded."""
    row, scope = row_scope
    scope['protected_recording_keys'] = [p.recording_key('lip_J_1_F_02_C004_B_011')]
    scope['held_short_alignment_parent_ids'] = ['x']
    assert set(p.reasons_for(row, row['text'], scope)) == {'protected_angle_independent_recording', 'held_acoustic_OOV_short_alignment'}


def test_unblocked_candidate(row_scope):
    """Unrelated old/canonical text does not acquire a fabricated exclusion."""
    row, scope = row_scope
    assert p.reasons_for(row, '읽을 수 있는 문장', scope) == []
