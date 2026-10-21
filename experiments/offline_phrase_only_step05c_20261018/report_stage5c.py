"""Summarize actual work after fixed training and all independent audits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    execution = read(HERE / 'results/execution_summary.json')
    aggregate = read(HERE / 'results/aggregate_metrics.json')
    assert execution['status'] == aggregate['status'] == 'completed'
    assert execution['main_optimizer_updates'] == 3360
    assert aggregate['evaluation_count'] == aggregate['forward_calls'] == 144
    assert aggregate['frames'] == 5772 and aggregate['failures'] == 0
    run_paths = [Path(item['path']) for item in execution['runs']]
    runs, all_rows, evaluations = [], [], []
    for path in run_paths:
        result = read(path)
        history = [json.loads(line) for line in (path.parent / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['update'] for r in history] == list(range(1, 1681))
        assert all(r['kind'] == 'phrase' for r in history)
        replay = read(path.parent / 'prior420_reproduction.json')
        assert replay['status'] == 'passed' and replay['initial_exact'] and replay['state_exact'] and replay['loss_trajectory_exact']
        evals = [read(item['summary']['path']) for item in result['evaluations']]
        assert len(evals) == 4 and sum(e['count'] for e in evals) == 72
        runs.append({'seed': result['identity']['seed'], 'updates': len(history),
                     'actual_frames': sum(r['true_frames'] for r in history),
                     'dense_frames': sum(r['dense_frames'] for r in history),
                     'target_tokens': sum(r['target_tokens'] for r in history),
                     'optimizer_seconds': sum(r['seconds'] for r in history),
                     'elapsed_seconds': result['elapsed_seconds'],
                     'maximum_training_cuda_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in history),
                     'maximum_training_cuda_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in history),
                     'prior420_exact': True, 'result': ref(path), 'prior420_evidence': ref(path.parent / 'prior420_reproduction.json')})
        all_rows.extend(history); evaluations.extend(evals)
    preflight = read(HERE / 'results/preflight_summary.json')
    preflight_rows = []
    for path in sorted((HERE / 'preflight').glob('**/progress.jsonl')):
        preflight_rows.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    assert len(preflight_rows) == preflight['physical_optimizer_updates'] == 8
    resources = {
        'schema': 'offline-stage5c-resource-accounting-v1', 'status': 'completed',
        'main_optimizer_updates': len(all_rows), 'preflight_optimizer_updates': len(preflight_rows),
        'total_physical_optimizer_updates': len(all_rows) + len(preflight_rows),
        'main_neural_forward_calls': len(all_rows), 'evaluation_neural_forward_calls': aggregate['forward_calls'],
        'preflight_neural_forward_calls': 8, 'total_neural_forward_calls': len(all_rows) + 8 + aggregate['forward_calls'],
        'main_actual_frames': sum(r['true_frames'] for r in all_rows),
        'main_dense_frames': sum(r['dense_frames'] for r in all_rows),
        'main_padding_frames': sum(r['padding_frames'] for r in all_rows),
        'main_target_tokens': sum(r['target_tokens'] for r in all_rows),
        'preflight_actual_frames': sum(r['true_frames'] for r in preflight_rows),
        'preflight_dense_frames': sum(r['dense_frames'] for r in preflight_rows),
        'evaluation_frames': aggregate['frames'],
        'main_optimizer_seconds': sum(r['seconds'] for r in all_rows),
        'main_execution_wall_seconds': execution['elapsed_seconds'],
        'per_run_segment_seconds_sum': sum(r['elapsed_seconds'] for r in runs),
        'evaluation_seconds_sum': sum(r['seconds'] for r in evaluations),
        'maximum_training_cuda_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in all_rows),
        'maximum_training_cuda_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in all_rows),
        'maximum_evaluation_cuda_allocated_bytes': max(r['peak_allocated_cuda_bytes'] for r in evaluations),
        'maximum_evaluation_cuda_reserved_bytes': max(r['peak_reserved_cuda_bytes'] for r in evaluations),
        'main_checkpoints': len(list((HERE / 'runs').glob('**/*.pt'))),
        'preflight_checkpoints': len(list((HERE / 'preflight').glob('**/*.pt'))),
        'checkpoint_bytes': sum(p.stat().st_size for p in HERE.glob('**/*.pt')),
        'posterior_files': len(list((HERE / 'runs').glob('**/*.npz'))),
        'posterior_bytes': sum(p.stat().st_size for p in (HERE / 'runs').glob('**/*.npz')),
        'updates_with_zero_gradient_tensor': sum(r['nonzero_gradient_tensors'] != r['parameter_tensors'] for r in all_rows),
        'evaluation_failures': aggregate['failures'], 'dev_forward_calls': 0, 'full_sentence_forward_calls': 0,
        'new_downloads': 0, 'source_audio_video_reads': 0, 'teacher_calls': 0, 'service_changes': 0,
        'timing_scope': 'Main execution only; excludes code preparation, tests, preflight and final independent audits.',
        'memory_scope': 'PyTorch allocator peaks, not total device/system memory.',
    }
    assert resources['main_actual_frames'] == 269360 and resources['main_dense_frames'] == 315946
    assert resources['main_target_tokens'] == 152320 and resources['main_checkpoints'] == 8
    assert resources['preflight_checkpoints'] == 8 and resources['posterior_files'] == 144
    save(HERE / 'results/resource_accounting.json', resources)
    normal = [r for r in aggregate['diagnostics'] if r['condition'] == 'normal']
    temporal = [r for r in aggregate['diagnostics'] if r['completed_update'] == 1680]
    summary = {
        'schema': 'offline-stage5c-completion-summary-v1', 'client_date': '2026-10-18',
        'status': 'completed_fixed_phrase_only_learnability_diagnostic',
        'main_runs': runs, 'normal_trajectory': normal, 'final_temporal_controls': temporal,
        'fit_gate': aggregate['fit_gate'],
        'training_parents': 12, 'training_unique_phrase_texts': 11, 'historical_speakers': 8,
        'data_train_only': True, 'new_sentence_generalization_measured': False,
        'source_training_ready': False, 'source_provisional': True, 'precise_visual_boundary_approval': False,
        'train_phrase_boundary_flag_count': 1, 'original_all30_boundary_flag_count': 2,
        'resource_accounting': ref(HERE / 'results/resource_accounting.json'),
        'execution': ref(HERE / 'results/execution_summary.json'),
        'aggregate': ref(HERE / 'results/aggregate_metrics.json'),
        'automatic_sentence_expansion': False, 'service_changes': 0,
        'limitations': [
            'This is reproduction of the same12 train phrases, not unseen sentences/speakers or webcam accuracy.',
            'The arbitrary gate requires both seeds to match all12 at both1260 and1680; no best checkpoint substitution.',
            'The fixed extra phrase exposure changes both target regime and training amount relative to previous B; not a general causal proof.',
            'Normal versus temporal controls diagnoses sensitivity; perfect fit alone does not prove semantic lip understanding.',
        ],
    }
    save(HERE / 'results/stage5c_summary.json', summary)
    print(json.dumps({'status': summary['status'], 'fit_gate': aggregate['fit_gate']['passed'],
                      'resources': resources}, ensure_ascii=True))


if __name__ == '__main__':
    main()
