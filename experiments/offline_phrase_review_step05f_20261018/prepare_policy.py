"""Bind the exact first eight review parents under a separate stage5f policy."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = ROOT / 'experiments/offline_phrase_design_step05e_20261018'


def binding(path: Path) -> dict[str, Any]:
    """Bind a file by exact size and digest without interpreting media."""
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind {path}') from exc


def load(path: Path) -> Any:
    """Read declared local JSON evidence."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read {path}') from exc


def save(path: Path, value: Any) -> None:
    """Create new evidence, refusing overwrite or out-of-stage paths."""
    try:
        if not path.resolve().is_relative_to(BASE):
            raise ValueError('Output outside this stage')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot create {path}') from exc


def main() -> None:
    """Prepare bounded source extraction and review; do not run models."""
    try:
        manifest_path = PREVIOUS / 'results/completion_manifest.json'
        assert binding(manifest_path)['sha256'] == 'c0f641cbf8d9acd70b8a9eac11013a68bcc12b69b3760cabb2b4b4e2bc629b8a'
        manifest = load(manifest_path)
        for item in manifest['artifacts']:
            assert binding(Path(item['path'])) == item
        handoff_path = PREVIOUS / 'results/next_step_handoff.json'
        handoff = load(handoff_path)
        queue_path = Path(handoff['queue']['path'])
        eligible_path = PREVIOUS / 'data/eligible_sentences.json'
        queue = load(queue_path)
        parents = {row['parent_id']: row for row in load(eligible_path)['records']}
        rows = queue['records'][:8]
        assert [row['parent_id'] for row in rows] == handoff['first_batch']['parent_ids']
        assert [row['candidate_id'] for row in rows] == handoff['first_batch']['candidate_ids']
        assert {row['proposed_role'] for row in rows} == {'proposed_train'}
        assert len({row['speaker_id'] for row in rows}) == 8
        enriched = []
        for row in rows:
            assert not row['parent_transcript_confirmed'] and not row['training_ready']
            parent = parents[row['parent_id']]
            enriched.append({'review': row, 'parent': parent})
        save(BASE / 'selected_inputs.json', {
            'schema': 'offline-stage5f-fixed-eight-inputs-v1',
            'sources': [binding(queue_path), binding(eligible_path), binding(handoff_path)],
            'records': enriched, 'new_human_approvals': 0, 'training_ready': False})
        policy = {
            'schema': 'offline-stage5f-execution-policy-v1', 'client_date': '2026-10-18',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'stage': '5f', 'status': 'fixed_first_eight_review_preparation_authorized',
            'authorization': 'User requested the next stage after the completed5e handoff.',
            'bindings': [binding(manifest_path), binding(handoff_path), binding(queue_path),
                         binding(BASE / 'selected_inputs.json'), binding(Path(__file__))],
            'record_ids': [row['parent_id'] for row in rows],
            'candidate_ids': [row['candidate_id'] for row in rows],
            'review_ids': [row['review_id'] for row in rows],
            'allowed': ['Read exact declared TAR video/audio/label members for these eight parents only.',
                        'Verify source identity, local WAV/MP4 clock correspondence and source metadata.',
                        'Create bounded full-parent and context PCM, review-only original-grid ROI/full videos and traces.',
                        'Prepare a local read-only review page and record real user responses with their exact scope.',
                        'Inspect acoustic vocabulary and prepare a separate source-bound alignment plan.'],
            'source_archive_policy': 'Sequential bounded member reads; no full-archive unpacking or source edits.',
            'audio': {'original_clock': 'original label WAV', 'output_rate': 16000,
                      'sample_policy': 'Exact native PCM slice; rational half-even integer sample endpoints; resample separately.',
                      'parent_context_seconds_each_side': 0.35,
                      'context_neighbor_intersections_must_be_reported': True},
            'video': {'fps': 25, 'roi_rgb_size': [96, 96], 'margin': 0.20,
                      'grid': 'Recording-anchored exact rational D grid; nearest actual presentation timestamp, earlier tie.',
                      'offset': 'Original label WAV clock plus measured recording offset exactly once.',
                      'review_only': True, 'training_cache_creation': False},
            'acoustic_forward_in_this_initial_policy': False,
            'alignment_followup': 'Run only under a separate explicit media/model/code-bound plan; provisional proposals never mean approval.',
            'approval_policy': 'Actual answers only; parent transcript, parent audio boundaries, phrase audio boundaries and qualitative visual boundaries are separate scopes.',
            'parent_approval_implies_phrase_approval': False,
            'unanswered_is_pending': True, 'actual_split': None, 'training_ready': False,
            'forbidden': ['Q09-Q20 or substituted parents', 'new visual model inference',
                          'student training or checkpoint replacement', 'automatic cache promotion',
                          'new download or cloud upload', 'whole TAR extraction',
                          'original label modification', 'automatic stage6 sentence expansion'],
            'source_provisional': True, 'old_oov_and_pending_flags_preserved': True,
            'old_dev3_and_reserved_probe7_recordings_preserved': True,
            'current_source_split_changes': 0, 'service_changes': 0,
        }
        save(BASE / 'execution_policy.json', policy)
        print(json.dumps(binding(BASE / 'execution_policy.json'), ensure_ascii=True))
    except Exception as exc:
        raise RuntimeError('Stage5f scope binding failed; do not widen selection') from exc


if __name__ == '__main__':
    main()
