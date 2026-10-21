"""Freeze the completed technical preparation and pending question snapshot."""
from __future__ import annotations

from prepare_policy import BASE, ROOT, binding, load, save


def main() -> None:
    """Hash final files while excluding transient server output and caches."""
    try:
        assert load(BASE / 'audit/final_report_audit.json')['status'] == 'passed'
        summary = load(BASE / 'results/stage5f_summary.json')
        assert summary['phrase_audio_boundary_approvals'] == 0
        assert summary['phrase_visual_boundary_approvals'] == 0
        assert summary['training_ready'] is False
        ignored = {'__pycache__', '.pytest_cache', 'runtime'}
        destination = BASE / 'results/completion_manifest.json'
        files = [p for p in BASE.rglob('*') if p.is_file()
                 and p != destination and not ignored.intersection(p.relative_to(BASE).parts)
                 and p.suffix != '.pyc']
        files.append(ROOT / 'docs/녹화형_단계학습_2026-10-12/05f_첫8문장_전사확인과_구절정렬.md')
        artifacts = [binding(p) for p in sorted(files, key=lambda p: str(p))]
        assert len(artifacts) == len({r['path'] for r in artifacts})
        save(destination, {
            'schema': 'offline-stage5f-completion-manifest-v1',
            'status': summary['status'], 'artifacts': artifacts,
            'artifact_count': len(artifacts),
            'excluded_transient_directories': sorted(ignored),
            'frozen_scope': 'Technical preparation and original nine parent answers; future phrase answers must be saved as separate revisions.',
            'human_phrase_audio_approvals': 0, 'human_phrase_visual_approvals': 0,
            'training_ready': False,
        })
        for expected in artifacts:
            from pathlib import Path
            assert binding(Path(expected['path'])) == expected
        print(f'Frozen and verified {len(artifacts)} artifact entries; phrase review remains pending')
    except Exception as exc:
        raise RuntimeError('Stage5f freeze failed; do not report final preservation success') from exc


if __name__ == '__main__':
    main()
