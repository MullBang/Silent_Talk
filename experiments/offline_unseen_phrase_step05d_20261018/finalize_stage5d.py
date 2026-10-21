"""Freeze the completed18-evaluation probe and a metadata-only next-step outline."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from prepare import read, ref, save

REPORT = ROOT / 'docs/녹화형_단계학습_2026-10-12/05d_미학습_구절_고정모델_평가.md'


def main() -> None:
    """Create the handoff and verify every frozen artifact byte-for-byte."""
    try:
        audits = ['source_preflight', 'preflight_plan_audit', 'execution_audit', 'evaluation_audit',
                  'aggregate_audit', 'final_source_preservation', 'report_audit', 'analysis_audit', 'coverage_audit']
        for name in audits:
            assert read(HERE / f'audit/{name}.json')['status'] == 'passed', name
        summary = read(HERE / 'results/stage5d_summary.json')
        assert summary['primary']['exact_observations'] == 0
        assert summary['neural_forward_calls'] == 18 and summary['optimizer_updates'] == 0
        inventory = ROOT / 'experiments/offline_curriculum_step01_20261012/results'
        handoff = {
            'schema': 'offline-stage5d-next-step-handoff-v1', 'client_date': '2026-10-18',
            'status': 'fixed_probe_completed_next_data_design_inventory_proposed',
            'summary': ref(HERE / 'results/stage5d_summary.json'), 'report': ref(REPORT),
            'saved_output_analysis': ref(HERE / 'analysis/result_interpretation.json'),
            'coverage_analysis': ref(HERE / 'analysis/coverage_analysis.json'),
            'primary': summary['primary'], 'prior5c_strict_fit_gate_passed': False,
            'next_stage': '5e', 'next_status': 'outline_only_not_executed',
            'next_question': 'Can existing eligible metadata support many phrases per speaker and repeated expressions across speakers/recordings while retaining protected development scopes?',
            'next_scope': 'Metadata inventory, diversity/coverage statistics and a review/split design only. No new neural evaluation, training, source-media extraction or download in that preparation step.',
            'existing_inventory_sources': [ref(inventory / name) for name in ['summary.json', 'candidate_inventory.json', 'analysis_sentences.json']],
            'current_evaluation_plan': ref(HERE / 'evaluation_plan.json'),
            'current_three_dev_parents_remain_protected': [r['parent_id'] for r in read(HERE / 'evaluation_plan.json')['items']],
            'proposed_tasks': [
                'Load the frozen prior inventory and reapply historical protected recordings/transcripts/held/pending flags, plus all current dev3 parent recording and transcript protections.',
                'Count sentence-derived eojeol and short-phrase occurrences by speaker/recording, including repeated expressions, speaker-specific expressions, length and token/syllable/word coverage.',
                'Assess feasibility of multiple different phrases per speaker and repeated expressions across different speakers/recordings; report actual counts rather than assuming a target quota is available.',
                'Separate same-expression/different-recording evaluation from held-out-new-phrase evaluation; do not call either an untouched final test if reused in design.',
                'Prepare a small prioritized transcription/alignment/boundary review queue with exact source IDs and evidence requirements. Candidate text counts are not approved or extracted clips.',
                'Keep the69-token CTC task and all current model/data artifacts unchanged. Any later extraction, training budget or model change needs a separate declared design.',
            ],
            'future_training_data_status': 'More diverse verified clips are needed for a broader training comparison. Existing metadata may provide them; availability and boundaries still require review.',
            'immediate_additional_download_needed': False, 'immediate_colab_needed': False,
            'if_inventory_insufficient': 'Report the missing phrase/speaker/recording coverage and required data to the user before seeking additional source data.',
            'actual_next_stage_work': {'inventory_runs': 0, 'new_neural_calls': 0, 'new_optimizer_updates': 0, 'new_clips': 0},
            'source_training_ready': False, 'source_provisional': True,
            'automatic_sentence_expansion': False, 'automatic_more_phrase_training': False,
            'new_speaker_or_webcam_performance_established': False, 'service_changes': 0,
        }
        save(HERE / 'results/next_step_handoff.json', handoff)
        manifest_path = HERE / 'results/completion_manifest.json'
        files = sorted(p for p in HERE.rglob('*') if p.is_file() and p != manifest_path
                       and not {'__pycache__', '.pytest_cache'}.intersection(p.parts)
                       and p.suffix not in {'.pyc', '.partial'})
        assert not list(HERE.rglob('*.partial'))
        files.append(REPORT)
        artifacts = [ref(p) for p in files]
        manifest = {
            'schema': 'offline-stage5d-completion-snapshot-v1', 'client_date': '2026-10-18',
            'status': summary['status'], 'artifacts': artifacts, 'artifact_count': len(artifacts),
            'excluded': ['this manifest', '__pycache__', '.pytest_cache', '*.pyc', 'mutable repository AGENTS.md'],
            'frozen_production_and_results': True, 'changes_require_separate_version': True,
            'evaluation_rows': 18, 'evaluation_frames': 576, 'optimizer_updates': 0,
            'source_training_ready': False, 'automatic_sentence_expansion': False,
            'next5e_not_executed': True,
        }
        save(manifest_path, manifest)
        assert all(ref(Path(item['path'])) == item for item in read(manifest_path)['artifacts'])
        print({'status': 'frozen_and_verified', 'artifact_count': len(artifacts), 'manifest': ref(manifest_path)})
    except Exception as exc:
        raise RuntimeError('Completion snapshot failed; preserve all earlier artifacts') from exc


if __name__ == '__main__':
    main()
