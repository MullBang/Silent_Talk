"""Independent read-only preservation and source-contract audit for stage5e."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S1 = ROOT / 'experiments/offline_curriculum_step01_20261012'
S2G = ROOT / 'experiments/offline_phrase_review_step02g_20261013'
S3 = ROOT / 'experiments/offline_dataset_step03_20261013'
S5D = ROOT / 'experiments/offline_unseen_phrase_step05d_20261018'


def read(path):
    path = Path(path)
    assert path.suffix == '.json' and path.is_relative_to(ROOT)
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    path = Path(path)
    assert path.is_relative_to(ROOT)
    assert path.suffix.lower() not in {'.wav', '.mp4', '.tar', '.zip', '.mp3'}
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def binding(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def check(row):
    path = Path(row['path'])
    if row.get('exists') is False:
        assert not path.exists(), str(path)
        return
    assert path.is_file() and digest(path) == row['sha256'], str(path)
    if 'bytes' in row:
        assert path.stat().st_size == row['bytes'], str(path)


def textkey(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def anglekey(stem):
    assert re.fullmatch(r'lip_J_\d+_[FM]_\d{2}_C\d{3}_[ABC]_\d{3}', stem), stem
    parts = stem.split('_')
    return '_'.join(parts[:-2] + parts[-1:])


def inspect():
    manifests = []
    for folder, count in [
        ('offline_dataset_step03_20261013', 114),
        ('offline_training_step04_20261013', 101),
        ('offline_comparison_step05_20261013', 1238),
        ('offline_diagnostic_step05b_20261018', 45),
        ('offline_phrase_only_step05c_20261018', 280),
        ('offline_unseen_phrase_step05d_20261018', 82),
    ]:
        path = ROOT / 'experiments' / folder / 'results/completion_manifest.json'
        rows = read(path)['artifacts']
        assert len(rows) == len({r['path'] for r in rows}) == count
        for row in rows:
            check(row)
        manifests.append({**binding(path), 'verified_artifact_entries': count})
    assert sum(r['verified_artifact_entries'] for r in manifests) == 1860
    scope_path = S3 / 'results/scope_snapshot.json'
    scope = read(scope_path)['artifacts']
    assert len(scope) == 10
    for row in scope:
        check(row)
    first_manifest_path = S1 / 'results/completion_manifest.json'
    first_manifest = read(first_manifest_path)
    assert len(first_manifest['artifacts']) == 10
    for row in first_manifest['artifacts']:
        check(row)
    first = read(S1 / 'results/summary.json')
    for path, sha in first['source_files'].items():
        check({'path': path, 'sha256': sha})
    inventory = read(S1 / 'results/candidate_inventory.json')
    old_rows = read(S1 / 'results/analysis_sentences.json')
    old = {r['id']: r for r in old_rows}
    assert len(old_rows) == len(old) == 293
    assert len(inventory) == 510
    assert {r['id'] for r in inventory if r['stage1_analysis_eligible']} == set(old)
    assert all(not r['stage1_human_approval_added'] and not r['training_ready_for_new_word_curriculum'] and r['word_boundaries'] is None for r in old_rows)
    assert len({r['speaker_id'] for r in old_rows}) == 8
    assert len({r['source_stem'] for r in old_rows}) == 35
    for r in old_rows:
        anglekey(r['source_stem'])
    assert {k: len(first[k]) for k in ['protected_historical_ids', 'protected_probe_ids', 'protected_dev16_ids', 'pending_listening_ids']} == {
        'protected_historical_ids': 122, 'protected_probe_ids': 14, 'protected_dev16_ids': 16, 'pending_listening_ids': 2}
    protected_ids = set().union(*(first[k] for k in ['protected_historical_ids', 'protected_probe_ids', 'protected_dev16_ids', 'pending_listening_ids']))
    assert not set(old) & protected_ids
    split_path = S3 / 'split/split_plan.json'
    split = read(split_path)
    for r in split['inputs'].values():
        check(r)
    reviewed_path = S2G / 'results/verified_phrase_handoff.json'
    reviewed = read(reviewed_path)
    canonical = {r['parent_id']: r for r in reviewed['records']}
    parents = {r['parent_id']: r for r in split['records']}
    assert len(canonical) == len(parents) == 15
    assert set(canonical) == set(parents) <= set(old)
    assert reviewed['held_short_alignment'][0]['parent_id'] == 'lip_J_1_F_03_C293_A_012_019'
    assert len(reviewed['held_short_alignment']) == 1
    held = reviewed['held_short_alignment'][0]
    assert held['phrase'] is None and held['training_ready'] is False and held['parent_id'] not in canonical
    assert held['parent_id'] in old
    changes = []
    for sid, r in canonical.items():
        s = parents[sid]
        assert r['parent_text'] == s['parent_text']
        assert r['text'] == s['phrase']['text']
        assert s['historical_parent_text'] == old[sid]['text']
        assert s['angle_independent_recording_key'] == anglekey(s['source_stem'])
        assert all(s[k] == old[sid][k] for k in ['source_stem', 'video_sha256', 'label_sha256'])
        assert r['audio_boundary_confirmed'] and r['visual_no_obvious_cut_reported']
        assert r['training_ready'] is False and r['exact_visual_boundary_confirmed'] is False
        if old[sid]['text'] != r['parent_text']:
            changes.append({'parent_id': sid, 'historical_text': old[sid]['text'], 'canonical_parent_text': r['parent_text']})
    cache = read(S3 / 'results/cache_handoff.json')
    assert len(cache['records']) == 30 and sum(r['boundary_followup_flag'] for r in cache['records']) == 2
    assert all(r['provisional'] and not r['training_ready'] for r in cache['records'])
    current_dev = []
    for sid in split['dev_parent_ids']:
        s = parents[sid]
        assert s['split'] == 'dev'
        current_dev.append({k: s[k] for k in ['parent_id', 'source_stem', 'angle_independent_recording_key', 'video_sha256', 'label_sha256', 'historical_parent_text', 'parent_text']} | {'phrase_text': s['phrase']['text'], 'full_text_keys': sorted({textkey(s['historical_parent_text']), textkey(s['parent_text'])}), 'phrase_text_key': textkey(s['phrase']['text'])})
    assert len(current_dev) == 3
    latest = read(S5D / 'results/next_step_handoff.json')
    assert set(latest['current_three_dev_parents_remain_protected']) == {r['parent_id'] for r in current_dev}
    assert latest['prior5c_strict_fit_gate_passed'] is False
    assert latest['next_stage'] == '5e' and latest['next_status'] == 'outline_only_not_executed'
    for r in latest['existing_inventory_sources']:
        check(r)
    return {
        'schema': 'offline-stage5e-independent-source-preflight-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'frozen_manifests': manifests, 'verified_frozen_artifact_entries': 1860,
        'stage1_manifest': binding(first_manifest_path), 'stage1_manifest_artifact_entries': 10,
        'scope': {'binding': binding(scope_path), 'verified_artifact_entries': 10},
        'sources': [binding(S1 / 'results' / n) for n in ['summary.json', 'candidate_inventory.json', 'analysis_sentences.json']] + [binding(split_path), binding(reviewed_path), binding(S5D / 'results/next_step_handoff.json')],
        'candidate_analysis_rows': 293, 'candidate_analysis_speakers': 8, 'candidate_analysis_recordings': 35,
        'preserved_protected_id_counts': {k: len(first[k]) for k in ['protected_historical_ids', 'protected_probe_ids', 'protected_dev16_ids', 'pending_listening_ids']},
        'canonical_parent_sources_agree': 15, 'canonical_parent_text_changes_vs_stage1': changes,
        'held_short_alignment': held, 'current_dev_protection_metadata': current_dev,
        'recommendations': [
            'Keep inherited historical122 ID/full-transcript and probe14/dev16 recording/source/full-transcript protection scopes explicit.',
            'Reapply protections to both original and reviewed canonical text before counting new candidates.',
            'Block current dev3 whole recordings using angle-independent keys and source hashes, plus historical/canonical full texts.',
            'Protect complete dev3 phrase text against containment in any proposed training phrase or parent text after NFC/whitespace removal.',
            'Use stage2g reviewed15 parent text, cross-checked against stage3, without changing older source files.',
            'Canonical parent evidence does not grant boundary approval to new phrase candidates.',
            'Retain held acoustic-OOV parent and prior pending cases; observed69-token coverage does not override their holds.'
        ],
        'source_provisional_flags_preserved': True, 'boundary_followup_flags_preserved': 2,
        'producer_imports': 0, 'NPZ_arrays_loaded': 0, 'checkpoint_tensor_loads': 0,
        'raw_media_reads': 0, 'audio_reads': 0, 'network_forward_calls': 0, 'optimizer_updates': 0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='source_preflight.json')
    args = parser.parse_args()
    output = (HERE / args.output).resolve()
    assert output.parent == HERE and not output.exists()
    result = inspect()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: result[k] for k in ['status', 'verified_frozen_artifact_entries', 'stage1_manifest_artifact_entries', 'candidate_analysis_rows', 'canonical_parent_sources_agree']}))
