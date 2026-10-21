"""Build a metadata-only, canonical, protected subset of the old293 sentences."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unicodedata
from collections import Counter
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S1 = ROOT / 'experiments/offline_curriculum_step01_20261012/results'
S2G = ROOT / 'experiments/offline_phrase_review_step02g_20261013/results/verified_phrase_handoff.json'
S3 = ROOT / 'experiments/offline_dataset_step03_20261013'
S5D = ROOT / 'experiments/offline_unseen_phrase_step05d_20261018'
PARENT = ROOT / 'experiments/diversity_step15_20261007/data_plan.json'


def ref(path: Path) -> dict:
    """Bind exact local bytes without opening media formats or model tensors."""
    try:
        path = path.resolve()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind {path}') from exc


def read(path: Path, sources: list[dict] | None = None) -> Any:
    """Read only workspace JSON, optionally recording its exact source hash."""
    try:
        assert path.resolve().is_relative_to(ROOT) and path.suffix == '.json'
        if sources is not None:
            sources.append(ref(path))
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'Cannot read metadata {path}') from exc


def save(path: Path, value: Any) -> None:
    """Create separate evidence and refuse overwriting previous versions."""
    try:
        assert path.resolve().is_relative_to(HERE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'Cannot create {path}') from exc


def text_key(text: str) -> str:
    """Use NFC and remove whitespace for conservative duplicate protection."""
    return ''.join(unicodedata.normalize('NFC', text).split())


def recording_key(stem: str) -> str:
    """Remove only the explicit angle field; preserve participant and session."""
    try:
        match = re.fullmatch(r'(.+_C\d+)_[A-Z]_(\d{3})', stem)
        if match is None:
            raise ValueError('Unrecognized recording identity')
        return match.group(1) + '_' + match.group(2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid recording stem {stem!r}') from exc


def reasons_for(row: dict, canonical: str, scope: dict) -> list[str]:
    """Preserve old protection and conservatively exclude current dev phrases."""
    try:
        reasons = []
        if row['id'] in scope['protected_parent_ids']:
            reasons.append('protected_parent_id')
        if recording_key(row['source_stem']) in scope['protected_recording_keys']:
            reasons.append('protected_angle_independent_recording')
        for field in ['source_stem', 'video_sha256', 'label_sha256']:
            if row[field] in scope['blocked_' + field]:
                reasons.append('protected_' + field)
        keys = {text_key(row['text']), text_key(canonical)}
        if keys.intersection(scope['protected_full_text_keys']):
            reasons.append('protected_historical_or_canonical_full_transcript')
        if any(phrase in key for key in keys for phrase in scope['protected_phrase_text_keys']):
            reasons.append('contains_current_dev_phrase')
        if row['id'] in scope['held_short_alignment_parent_ids']:
            reasons.append('held_acoustic_OOV_short_alignment')
        return reasons
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete protection metadata') from exc


def build() -> None:
    """Create canonical statistics inputs without producing or approving clips."""
    try:
        sources = []
        for manifest_path in [S1 / 'completion_manifest.json', S5D / 'results/completion_manifest.json']:
            manifest = read(manifest_path, sources)
            for item in manifest['artifacts']:
                assert ref(Path(item['path']))['sha256'] == item['sha256']
        original = read(S1 / 'analysis_sentences.json', sources)
        summary = read(S1 / 'summary.json', sources)
        old_inventory = read(S1 / 'candidate_inventory.json', sources)
        old_parent = read(PARENT, sources)
        review = read(S2G, sources)
        split = read(S3 / 'split/split_plan.json', sources)
        dev_handoff = read(S5D / 'results/next_step_handoff.json', sources)
        assert len(original) == 293 and len({r['id'] for r in original}) == 293
        assert {r['id'] for r in original} == {r['id'] for r in old_inventory if r['stage1_analysis_eligible']}
        historical = old_parent['pool_profile']
        assert set(summary['protected_historical_ids']) == set(historical['historical_excluded_ids'])
        assert set(summary['protected_probe_ids']) == set(historical['prior_probe_ids'])
        canon = {r['parent_id']: r for r in review['records']}
        split_rows = {r['parent_id']: r for r in split['records']}
        assert len(canon) == 15 and set(canon) == set(split_rows)
        for sid, row in canon.items():
            assert row['parent_text'] == split_rows[sid]['parent_text']
            assert recording_key(row['recording_group']) == split_rows[sid]['angle_independent_recording_key']
        dev_ids = dev_handoff['current_three_dev_parents_remain_protected']
        assert set(dev_ids) == set(split['dev_parent_ids']) and len(dev_ids) == 3
        dev = [split_rows[sid] for sid in dev_ids]
        old_dev = [r for r in old_parent['selection']['records'] if r['id'] in summary['protected_dev16_ids']]
        assert len(old_dev) == 16
        blocked_stems = set(historical['blocked_recording_stems']) | {r['source_stem'] for r in old_dev + dev}
        blocked_targets = set(historical['blocked_normalized_targets']) | {text_key(r['text']) for r in old_dev}
        for row in dev:
            blocked_targets.update([text_key(row['parent_text']), text_key(row['historical_parent_text'])])
        scope = {
            'schema': 'offline-stage5e-protected-scope-v1', 'source_bindings': sources.copy(),
            'historical_scope': 'Historical122 IDs/full transcripts only; probe14 and dev16 recordings/hashes/full transcripts; current dev3 complete recordings/hashes/old+canonical parent text and contained phrase texts.',
            'protected_parent_ids': sorted(set(summary['protected_historical_ids'] + summary['protected_probe_ids'] + summary['protected_dev16_ids'] + summary['pending_listening_ids'] + dev_ids)),
            'protected_recording_keys': sorted({recording_key(stem) for stem in blocked_stems}),
            'blocked_source_stem': sorted(blocked_stems),
            'blocked_video_sha256': sorted(set(historical['blocked_video_sha256']) | {r['video_sha256'] for r in old_dev + dev}),
            'blocked_label_sha256': sorted(set(historical['blocked_label_sha256']) | {r['label_sha256'] for r in old_dev + dev}),
            'protected_full_text_keys': sorted(blocked_targets),
            'protected_phrase_text_keys': sorted({text_key(r['phrase']['text']) for r in dev}),
            'protected_candidate_text_keys': sorted(blocked_targets | {text_key(r['phrase']['text']) for r in dev}),
            'held_short_alignment_parent_ids': sorted(r['parent_id'] for r in review['held_short_alignment']),
            'pending_listening_ids': summary['pending_listening_ids'],
            'current_dev_parent_ids': dev_ids, 'canonical_parent_evidence': ref(S2G),
            'source_training_ready': False, 'new_approval_added': False,
            'conservative_substring_rule': 'For current dev3 phrase keys only, block parent when key occurs anywhere in old or canonical no-space full text. Also block phrase candidates containing them.',
        }
        tokenizer_path = ROOT / 'backend/models/tokenizer.py'
        expected = next(r for r in read(ROOT / 'experiments/offline_training_step04_20261013/smoke_plan.json')['bindings'] if Path(r['path']).resolve() == tokenizer_path.resolve())
        assert ref(tokenizer_path)['sha256'] == expected['sha256']
        spec = importlib.util.spec_from_file_location('stage5e_tokenizer', tokenizer_path)
        tokenizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tokenizer)
        assert tokenizer.NUM_CLASSES == 69
        sources.append(ref(tokenizer_path))
        accepted, decisions, overlays = [], [], []
        for old in original:
            sid = old['id']
            reviewed = canon.get(sid)
            text = reviewed['parent_text'] if reviewed else old['text']
            assert tokenizer.decode_labels(tokenizer.encode_text(text)) == text
            reasons = reasons_for(old, text, scope)
            decision = {'parent_id': sid, 'speaker_id': old['speaker_id'], 'recording_key': recording_key(old['source_stem']),
                        'eligible': not reasons, 'exclusion_reasons': reasons, 'historical_text': old['text'], 'canonical_text': text,
                        'canonical_overlay_applied': reviewed is not None, 'text_changed': text != old['text']}
            decisions.append(decision)
            if reviewed:
                overlays.append({**decision, 'source': ref(S2G)})
            if reasons:
                continue
            accepted.append({**old, 'parent_id': sid, 'recording_key': recording_key(old['source_stem']),
                'historical_text': old['text'], 'historical_token_ids': old['token_ids'],
                'text': text, 'token_ids': tokenizer.encode_text(text), 'eojeols': text.split(),
                'source_approval_status': {'parent_transcript_confirmed': reviewed is not None,
                    'parent_evidence': ref(S2G) if reviewed else None,
                    'previous_reviewed_phrase': reviewed if reviewed else None,
                    'new_phrase_boundary_confirmed': False, 'exact_visual_boundary_certified': False,
                    'new_human_approval_added': False, 'training_ready': False},
                'current5c_training_parent': sid in split['train_parent_ids'],
                'historical_pool': 'Previously considered old training candidate, not untouched final test data.',
                'word_boundaries': None, 'new_phrase_frames': None, 'metadata_only': True,
                'training_ready': False, 'source_provisional': True})
        save(HERE / 'data/protected_scope.json', scope)
        save(HERE / 'data/eligible_sentences.json', {'schema': 'offline-stage5e-canonical-eligible-sentences-v1',
            'source_bindings': sources, 'protected_scope': ref(HERE / 'data/protected_scope.json'), 'records': accepted,
            'metadata_only': True, 'candidate_not_clip': True, 'new_human_approvals': 0, 'training_ready': False})
        save(HERE / 'data/parent_decisions.json', {'schema': 'offline-stage5e-parent-decisions-v1', 'records': decisions})
        save(HERE / 'data/canonical_overlays.json', {'schema': 'offline-stage5e-canonical-evidence-v1', 'records': overlays,
                                                  'source_labels_modified': False, 'new_human_approvals': 0})
        counts = {'initial_sentences': len(original), 'eligible_sentences': len(accepted),
                  'excluded_sentences': len(original) - len(accepted),
                  'eligible_speakers': len({r['speaker_id'] for r in accepted}),
                  'eligible_recordings': len({r['recording_key'] for r in accepted}),
                  'reason_counts_overlapping': dict(Counter(reason for r in decisions for reason in r['exclusion_reasons'])),
                  'canonical_parent_evidence_records': len(overlays), 'canonical_changed_texts': sum(r['text_changed'] for r in overlays),
                  'eligible_confirmed_parent_transcripts': sum(r['source_approval_status']['parent_transcript_confirmed'] for r in accepted),
                  'new_clips': 0, 'new_neural_calls': 0, 'new_training_updates': 0, 'raw_media_reads': 0,
                  'downloads': 0, 'service_changes': 0}
        save(HERE / 'results/protection_summary.json', {'status': 'completed_metadata_protection', **counts,
            'eligible': ref(HERE / 'data/eligible_sentences.json'), 'scope': ref(HERE / 'data/protected_scope.json')})
        print(json.dumps(counts, ensure_ascii=True))
    except Exception as exc:
        raise RuntimeError('Metadata protection preparation failed; no automatic widening is allowed') from exc


if __name__ == '__main__':
    build()
