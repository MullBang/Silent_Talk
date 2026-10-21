"""Summarize actual fixed-model probe execution without any neural work."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare import read, ref, save


def main() -> None:
    """Create traceable resource and scientific summaries from completed evidence."""
    try:
        execution = read(HERE / 'results/execution_summary.json')
        aggregate = read(HERE / 'results/aggregate_metrics.json')
        assert execution['status'] == aggregate['status'] == 'completed'
        assert execution['neural_forward_calls'] == aggregate['forward_calls'] == aggregate['evaluation_count'] == 18
        assert execution['evaluation_frames'] == aggregate['frames'] == 576
        assert aggregate['failures'] == execution['failed_rows'] == 0
        for name in ['source_preflight', 'preflight_plan_audit', 'execution_audit', 'evaluation_audit',
                     'aggregate_audit', 'final_source_preservation']:
            assert read(HERE / f'audit/{name}.json')['status'] == 'passed'
        evaluations = [r['evaluation'] for r in execution['runs']]
        resource = {
            'schema': 'offline-stage5d-resource-accounting-v1', 'status': 'completed',
            'neural_forward_calls': 18, 'evaluation_frames': 576,
            'unique_phrase_clips': 3, 'unique_phrase_frames': 96,
            'optimizer_updates': 0, 'optimizer_objects_created': 0, 'new_checkpoints': 0,
            'main_checkpoint_tensor_loads': 2, 'checkpoint_load_count_scope': 'Inference process only; independent audits may also read the saved CPU state.',
            'evaluation_failures': 0, 'posterior_files': len(list((HERE / 'runs').glob('**/*.npz'))),
            'posterior_bytes': sum(p.stat().st_size for p in (HERE / 'runs').glob('**/*.npz')),
            'execution_seconds': execution['seconds'], 'evaluation_seconds_sum': sum(r['seconds'] for r in evaluations),
            'maximum_allocated_cuda_bytes': max(r['peak_allocated_cuda_bytes'] for r in evaluations),
            'maximum_reserved_cuda_bytes': max(r['peak_reserved_cuda_bytes'] for r in evaluations),
            'timing_scope': 'Cached96x96 phrase probe, not full-webcam-video processing; preparation/imports/audits excluded from execution_seconds.',
            'memory_scope': 'PyTorch allocator peaks, not total device memory.',
            'train_forward_calls': 0, 'full_sentence_forward_calls': 0, 'source_media_reads': 0,
            'audio_inputs': 0, 'teacher_calls': 0, 'downloads': 0, 'service_changes': 0,
        }
        assert resource['posterior_files'] == 18
        save(HERE / 'results/resource_accounting.json', resource)
        summary = {
            'schema': 'offline-stage5d-summary-v1', 'client_date': '2026-10-18',
            'status': 'completed_fixed_model_reused_dev_phrase_probe',
            'primary': aggregate['primary'], 'all_conditions': aggregate['diagnostics'],
            'neural_forward_calls': 18, 'optimizer_updates': 0, 'new_checkpoints': 0,
            'unique_dev_phrase_clips': 3, 'known_speakers': 3, 'new_speaker_or_webcam_evaluation': False,
            'exploratory_reused_development_probe': True, 'generalization_established': False,
            'new_in_this_model_training': True, 'never_used_in_project_history': False,
            'source_training_ready': False, 'source_provisional': True, 'prior5c_strict_fit_gate_passed': False,
            'automatic_sentence_expansion': False, 'service_changes': 0,
            'execution': ref(HERE / 'results/execution_summary.json'),
            'aggregate': ref(HERE / 'results/aggregate_metrics.json'),
            'resources': ref(HERE / 'results/resource_accounting.json'),
            'limits': ['Only three distinct utterances; six normal model observations are not six independent utterances.',
                       'This reused known-speaker dev set informed prior experiments/design; not an independent final test.',
                       'Train12 perfect fit did not transfer to these dev3; no unique cause or universal model incapability is established.',
                       'Temporal controls retain identity, appearance, duration and/or frame content; they do not isolate a single causal shortcut.'],
        }
        save(HERE / 'results/stage5d_summary.json', summary)
        print({'status': summary['status'], 'mean_CER': aggregate['primary']['mean_CER'],
               'exact_observations': aggregate['primary']['exact_observations']})
    except Exception as exc:
        raise RuntimeError('Cannot summarize incomplete probe evidence') from exc


if __name__ == '__main__':
    main()
