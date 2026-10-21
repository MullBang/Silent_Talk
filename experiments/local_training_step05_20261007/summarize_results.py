"""Read-only aggregation/plotting of the four fixed Step5 final experiments.

No model is constructed or run. Final checkpoint histories are read on CPU with
the restricted weights-only loader. Saved development predictions are rescored
independently; every condition keeps the complete fixed71 denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS = (42, 43)
ARMS = ('A_ctc', 'B_feature')
CONDITIONS = ('normal', 'static', 'shuffled')
HANDOFF_SHA = '3003209c04109a01f9737593b3645ffe7aef28e16530d84e53ba0caca77fca4b'


def local(path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Analysis may only read/write workspace artifacts')
    return resolved


def bind(path) -> dict:
    path = local(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def read(ref: dict) -> dict:
    path = local(ref['path'])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('Analysis evidence changed: ' + str(path))
    return json.loads(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distance(left, right) -> int:
    """Independent Levenshtein count, for NFC characters and whitespace words."""
    previous = list(range(len(right) + 1))
    for i, first in enumerate(left, 1):
        current = [i]
        for j, second in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (first != second)))
        previous = current
    return previous[-1]


def score_records(records: list[dict]) -> dict:
    require(records and len({r['id'] for r in records}) == len(records), 'Empty/duplicate evaluation records')
    fields = ['character_edits', 'reference_characters', 'word_edits', 'reference_words',
              'normalized_character_edits', 'normalized_reference_characters']
    totals = dict.fromkeys(fields, 0)
    exact = normalized_exact = failed = 0
    for row in records:
        ref = unicodedata.normalize('NFC', row['reference']).strip()
        pred = unicodedata.normalize('NFC', row['prediction']).strip()
        nref, npred = ' '.join(ref.split()), ' '.join(pred.split())
        measured = {'character_edits': distance(ref, pred), 'reference_characters': len(ref),
                    'word_edits': distance(ref.split(), pred.split()), 'reference_words': len(ref.split()),
                    'exact': ref == pred, 'normalized_character_edits': distance(nref, npred),
                    'normalized_reference_characters': len(nref), 'normalized_exact': nref == npred}
        require(measured == row['metrics'], 'Saved per-record metric differs from independently counted text')
        require(row['status'] in {'ok', 'failed'}, 'Unknown inference status')
        if row['status'] == 'failed':
            require(row['prediction'] == '', 'A failed input must retain its row with empty prediction')
            failed += 1
        for key in fields:
            totals[key] += measured[key]
        exact += measured['exact']
        normalized_exact += measured['normalized_exact']
    require(totals['reference_characters'] > 0 and totals['reference_words'] > 0, 'Empty reference denominator')
    return {**totals, 'clips': len(records), 'failed_inputs': failed, 'exact_sentences': exact,
            'normalized_exact_sentences': normalized_exact,
            'cer_percent': 100 * totals['character_edits'] / totals['reference_characters'],
            'wer_percent': 100 * totals['word_edits'] / totals['reference_words'],
            'normalized_cer_percent': 100 * totals['normalized_character_edits'] / totals['normalized_reference_characters']}


def paired_metrics(evaluations: dict) -> dict:
    """Use both prespecified seeds equally; lower CER/WER is better, no selection."""
    require(set(evaluations) == {(s, a) for s in SEEDS for a in ARMS}, 'Require exactly the four fixed runs')
    per_seed = []
    for seed in SEEDS:
        conditions = {}
        for condition in CONDITIONS:
            a = evaluations[(seed, 'A_ctc')]['conditions'][condition]['aggregate']
            b = evaluations[(seed, 'B_feature')]['conditions'][condition]['aggregate']
            for field in ['clips', 'reference_characters', 'reference_words', 'normalized_reference_characters']:
                require(a[field] == b[field], 'Unequal paired evaluation denominator: ' + field)
            conditions[condition] = {'A_ctc': a, 'B_feature': b,
                'B_minus_A_cer_percentage_points': b['cer_percent'] - a['cer_percent'],
                'B_minus_A_wer_percentage_points': b['wer_percent'] - a['wer_percent'],
                'B_minus_A_exact_sentences': b['exact_sentences'] - a['exact_sentences']}
        ablations = {}
        for arm in ARMS:
            values = evaluations[(seed, arm)]['conditions']
            baseline = values['normal']['aggregate']['cer_percent']
            ablations[arm] = {condition + '_minus_normal_cer_percentage_points': values[condition]['aggregate']['cer_percent'] - baseline
                               for condition in ('static', 'shuffled')}
        per_seed.append({'seed': seed, 'conditions': conditions, 'ablation_gaps': ablations})
    means = {}
    for condition in CONDITIONS:
        means[condition] = {}
        for arm in ARMS:
            means[condition][arm] = {field: statistics.mean(evaluations[(seed, arm)]['conditions'][condition]['aggregate'][field] for seed in SEEDS)
                                     for field in ['cer_percent', 'wer_percent', 'normalized_cer_percent', 'exact_sentences', 'failed_inputs']}
        for field in ['B_minus_A_cer_percentage_points', 'B_minus_A_wer_percentage_points', 'B_minus_A_exact_sentences']:
            means[condition][field] = statistics.mean(row['conditions'][condition][field] for row in per_seed)
    mean_gaps = {arm: {condition + '_minus_normal_cer_percentage_points': statistics.mean(row['ablation_gaps'][arm][condition + '_minus_normal_cer_percentage_points'] for row in per_seed)
                        for condition in ('static', 'shuffled')} for arm in ARMS}
    return {'per_seed': per_seed, 'arithmetic_mean_of_two_seeds': means, 'mean_ablation_gaps': mean_gaps,
            'difference_direction': 'B minus A CER/WER percentage points: negative favors B; positive favors A',
            'ablation_direction': 'Ablated minus normal CER percentage points: positive means removing/changing temporal information increased errors',
            'selection': 'Both prespecified seeds42/43 and fixed update3200 retained; no best seed/checkpoint selection'}


def validate_history(history: list[dict], plan: dict, arm: str) -> dict:
    require(len(history) == 3200 and len(plan['schedule']) == 3200, 'Incomplete final update history')
    finite = ['ctc_loss', 'feature_loss', 'loss', 'gradient_norm_before_clipping', 'gradient_norm_after_clipping', 'seconds_optimizer_update']
    for index, row in enumerate(history):
        require(row['update'] == index + 1 and row['ids'] == plan['schedule'][index], 'History differs from full frozen schedule')
        require(all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in finite), 'Nonfinite update statistic')
        require(row['optimizer_updated'] is True and row['student_changed'] is True, 'Skipped/non-updating student step')
        require(row['feature_hook_count_before'] == row['feature_hook_count_after'] == 0, 'Leaked feature hook')
        require(row['seconds_optimizer_update'] > 0 and row['gradient_norm_after_clipping'] <= 1.0001, 'Invalid update timing/clipping')
        require(math.isclose(row['loss'], row['ctc_loss'] + row['feature_loss'], rel_tol=1e-12, abs_tol=1e-12), 'Combined loss differs')
        require(row['lambda'] == (1. if arm == 'B_feature' else 0.), 'Wrong arm loss weight')
        if arm == 'A_ctc':
            require(row['feature_loss'] == 0 and row['adapter_changed'] is False, 'A unexpectedly trained adapter')
        else:
            require(row['feature_loss'] >= 0 and row['adapter_changed'] is True, 'B adapter did not update')
    return {'updates': len(history), 'finite_updates': len(history), 'clip_presentations': sum(len(r['ids']) for r in history),
            'optimizer_update_seconds': sum(r['seconds_optimizer_update'] for r in history),
            'mean_optimizer_update_seconds': statistics.mean(r['seconds_optimizer_update'] for r in history),
            'median_optimizer_update_seconds': statistics.median(r['seconds_optimizer_update'] for r in history),
            'peak_cuda_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in history),
            'peak_cuda_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in history),
            'first100_mean_ctc_loss': statistics.mean(r['ctc_loss'] for r in history[:100]),
            'last100_mean_ctc_loss': statistics.mean(r['ctc_loss'] for r in history[-100:]),
            'last100_mean_feature_loss': statistics.mean(r['feature_loss'] for r in history[-100:]),
            'final_ctc_loss': history[-1]['ctc_loss'], 'final_feature_loss': history[-1]['feature_loss']}


def plots(histories: dict, paired: dict, output: Path) -> list[dict]:
    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / 'training_losses.png', output / 'development_cer.png']
    require(not any(path.exists() for path in destinations), 'Refusing to replace existing scientific figures')
    colors = {'A_ctc': '#2367a3', 'B_feature': '#c65f1a'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, seed in enumerate(SEEDS):
        for arm in ARMS:
            history = histories[(seed, arm)]
            # Nonoverlapping100-update means show all3200 updates equally.
            xs = list(range(50, 3200, 100))
            ys = [statistics.mean(r['ctc_loss'] for r in history[i:i+100]) for i in range(0, 3200, 100)]
            axes[0, column].plot(xs, ys, label=arm, color=colors[arm], linewidth=1.8)
        feature = histories[(seed, 'B_feature')]
        ys = [statistics.mean(r['feature_loss'] for r in feature[i:i+100]) for i in range(0, 3200, 100)]
        axes[1, column].plot(xs, ys, color=colors['B_feature'], label='B feature loss', linewidth=1.8)
        axes[0, column].set_title(f'Seed {seed}')
        axes[0, column].legend(frameon=False)
        axes[1, column].set_xlabel('Optimizer update')
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('CTC loss (100-update mean)')
    axes[1, 0].set_ylabel('Feature loss (100-update mean)')
    fig.suptitle('Fixed paired training: comparable CTC and separate feature loss')
    fig.tight_layout()
    fig.savefig(destinations[0], dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, item in zip(axes, paired['per_seed'], strict=True):
        for index, arm in enumerate(ARMS):
            xs = [k + (-.18 if index == 0 else .18) for k in range(3)]
            values = [item['conditions'][condition][arm]['cer_percent'] for condition in CONDITIONS]
            bars = axis.bar(xs, values, width=.34, label=arm, color=colors[arm])
            axis.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
        axis.set_xticks(range(3), CONDITIONS)
        axis.set_title(f"Seed {item['seed']}: fixed final update3200")
        axis.set_ylim(bottom=0)
        axis.margins(y=.17)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel('Pooled NFC CER (%) - lower is better')
    fig.suptitle('Development71: same speakers, held-out recordings')
    fig.tight_layout()
    fig.savefig(destinations[1], dpi=180, bbox_inches='tight')
    plt.close(fig)
    return [bind(path) for path in destinations]


def summarize(completion_path: Path, output: Path, plot_dir: Path, make_plots=True) -> dict:
    require(not output.exists(), 'Refusing to overwrite comparison evidence')
    completion_ref = bind(completion_path)
    completion = read(completion_ref)
    require(completion['status'] == 'completed' and completion['optimizer_updates'] == 12800, 'Training is incomplete')
    require(len(completion['run_results']) == 4, 'Require all four frozen runs')
    # CPU-only checkpoint deserialization; no model imports/construction/CUDA calls.
    import torch
    runs, evaluations, histories, verified = {}, {}, {}, [completion_ref]
    checkpoint_refs, final_refs = {}, []
    for run_ref in completion['run_results']:
        run = read(run_ref)
        key = (run['seed'], run['arm'])
        require(key in {(s, a) for s in SEEDS for a in ARMS} and key not in runs, 'Unknown/duplicate trained pair')
        require(run['status'] == 'completed' and run['completed_updates'] == 3200, 'Incomplete final result')
        plan = read(run['plan_file'])
        handoff = read(plan['full_data_plan'])
        require(plan['full_data_plan']['sha256'] == HANDOFF_SHA, 'Changed full-data handoff')
        parent = read(handoff['parent_plan'])
        dev_rows = [row for row in parent['records'] if row['split'] == 'dev']
        require(plan['seed'] == key[0] and plan['development_ids'] == [r['id'] for r in dev_rows], 'Wrong plan/dev order')
        info = run['checkpoint']
        final_ref = {'path': info['path'], 'sha256': info['sha256']}
        require(bind(info['path']) == final_ref and local(info['path']).stat().st_size == info['bytes'], 'Final checkpoint changed')
        require(bind(info['sidecar'])['sha256'] == info['sidecar_sha256'], 'Final checkpoint sidecar changed')
        with local(info['path']).open('rb') as stream:
            payload = torch.load(stream, map_location='cpu', weights_only=True)
        require(payload['completed_updates'] == 3200 and payload['state_hashes'] == run['final_state_hashes'], 'Final checkpoint/result mismatch')
        history = payload['history']
        require(run['updates'] == history[run['start_update']:], 'Run result suffix does not match full checkpoint history')
        require(payload['identity']['plan_sha256'] == plan['plan_sha256'] == run['plan_sha256'], 'Checkpoint plan identity differs')
        require(payload['identity']['seed'] == key[0] and payload['identity']['arm'] == key[1], 'Checkpoint pair identity differs')
        stats = validate_history(history, plan, key[1])
        histories[key] = history
        del payload
        for record in run['checkpoints']:
            ref = {'path': record['path'], 'sha256': record['sha256']}
            require(bind(ref['path']) == ref and local(ref['path']).stat().st_size == record['bytes'], 'Stored checkpoint bytes changed')
            checkpoint_refs[ref['path']] = record
        stats.update({'seed': key[0], 'arm': key[1], 'run_result': run_ref, 'checkpoint': final_ref,
                      'start_update_of_final_process': run['start_update'],
                      'final_process_elapsed_seconds': run['elapsed_seconds'],
                      'complete_run_elapsed_seconds': run['elapsed_seconds'] if run['start_update'] == 0 else None,
                      'checkpoint_files_in_final_segment': len(run['checkpoints']),
                      'checkpoint_bytes_in_final_segment': sum(r['bytes'] for r in run['checkpoints'])})
        runs[key] = stats
        final_refs.append(final_ref)
        eval_path = BASE / 'results' / f'evaluation_seed{key[0]}_{key[1]}.json'
        eval_ref = bind(eval_path)
        evaluated = read(eval_ref)
        require(evaluated['status'] == 'completed' and (evaluated['seed'], evaluated['arm']) == key, 'Incomplete/wrong evaluation pair')
        require(evaluated['run_result'] == run_ref and evaluated['checkpoint'] == info and evaluated['plan'] == run['plan_file'], 'Evaluation does not use fixed final checkpoint')
        require(evaluated['completed_updates'] == 3200 and evaluated['record_count'] == 71, 'Wrong evaluation checkpoint/cohort')
        require(evaluated['ids'] == plan['development_ids'] and set(evaluated['conditions']) == set(CONDITIONS), 'Evaluation order/conditions changed')
        require(evaluated['model_state_unchanged'] is True and evaluated['model_state_before'] == evaluated['model_state_after'] == run['final_state_hashes']['model'], 'Evaluation changed/differs from final model')
        require(all(evaluated[k] is False for k in ['audio_read', 'original_media_read', 'teacher_targets_read', 'teacher_executed', 'optimizer_executed']), 'Evaluation used forbidden input/optimization')
        for condition in CONDITIONS:
            block = evaluated['conditions'][condition]
            require([r['id'] for r in block['records']] == plan['development_ids'], 'Evaluation dropped or reordered records')
            for row, frozen in zip(block['records'], dev_rows, strict=True):
                require(row['reference'] == frozen['text'] and row['speaker_id'] == frozen['speaker_id'] and row['source_stem'] == frozen['source_stem'], 'Evaluation transcript/source changed')
                require(row['split'] == 'dev' and row['condition'] == condition, 'Evaluation scope differs')
            measured = score_records(block['records'])
            require(measured == block['aggregate'], 'Saved aggregate differs from independent recount')
        stats['evaluation_seconds'] = evaluated['elapsed_seconds']
        stats['evaluation_record_seconds_mean'] = statistics.mean(row['seconds'] for block in evaluated['conditions'].values() for row in block['records'])
        evaluations[key] = evaluated
        verified.extend([run_ref, run['plan_file'], eval_ref, plan['full_data_plan'], handoff['parent_plan']])
    paired = paired_metrics(evaluations)
    figure_refs = plots(histories, paired, plot_dir) if make_plots else []
    total_elapsed = [row['complete_run_elapsed_seconds'] for row in runs.values()]
    result = {'schema': 'lipnet-step05-fixed-comparison-summary-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'summary_code': bind(Path(__file__)),
              'tests': bind(BASE / 'test_summarize_results.py'), 'training_complete': completion_ref,
              'verified_evidence': verified, 'figures': figure_refs, **paired,
              'training_runs': [runs[key] for key in sorted(runs)],
              'training_totals': {'optimizer_updates': sum(r['updates'] for r in runs.values()),
                  'finite_updates': sum(r['finite_updates'] for r in runs.values()),
                  'clip_presentations': sum(r['clip_presentations'] for r in runs.values()),
                  'optimizer_update_seconds': sum(r['optimizer_update_seconds'] for r in runs.values()),
                  'complete_run_elapsed_seconds': sum(total_elapsed) if all(v is not None for v in total_elapsed) else None,
                  'peak_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in runs.values()),
                  'peak_cuda_reserved_bytes': max(r['peak_cuda_reserved_bytes'] for r in runs.values()),
                  'referenced_checkpoint_files': len(checkpoint_refs),
                  'referenced_checkpoint_bytes': sum(r['bytes'] for r in checkpoint_refs.values()),
                  'evaluation_seconds': sum(r['evaluation_seconds'] for r in runs.values())},
              'timing_scope': {'optimizer_update_seconds': 'Synchronized per-update clip normalization/transfers, forward, CPU CTC, backward, clipping and optimizer; excludes per-update state hashing, checkpoint I/O and startup/data validation.',
                  'complete_run_elapsed_seconds': 'Sum of four continuous execute() wall times, including verification/loading and checkpointing; excludes outer scheduler startup and final evaluation. Null if a recovery makes whole-trajectory wall time unavailable.',
                  'evaluation_seconds': 'Final evaluation loop across71 clips x3 conditions; excludes data loading/checkpoint verification/model construction. Not webcam end-to-end latency.',
                  'evaluation_record_seconds_mean': 'Mean saved per-record evaluation duration including preprocessing/inference/decoding; not an interactive webcam latency benchmark.'},
              'limitations': ['Development71 has the same8 speakers as training with disjoint recordings; not an untouched test.',
                  'Two fixed seeds and this limited cohort do not establish unseen-speaker or webcam performance.',
                  'The frozen greedy decoder sends predicted Jamo through strict decode_labels; invalid predicted sequences raise errors and are scored as empty predictions under the original failure policy. These are the original uncorrected evaluation results, including all failed rows; they are not a tolerant-decoder re-evaluation.',
                  'CER can exceed100%; do not convert1-CER into recognition accuracy.',
                  'Temporal ablations are diagnostic controls, not proof that a model understands speech.',
                  'This summary recounts predictions and reads checkpoint history; it never trains, evaluates or updates service weights.'],
              'models_constructed': False, 'inference_executed': False, 'optimizer_executed': False, 'source_media_read': False}
    for ref in verified:
        require(bind(ref['path']) == ref, 'Evidence changed while summarizing')
    output = local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-complete', type=Path, default=BASE / 'results/training_complete.json')
    parser.add_argument('--output', type=Path, default=BASE / 'results/comparison.json')
    parser.add_argument('--plot-dir', type=Path, default=BASE / 'results/figures')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    result = summarize(args.training_complete, args.output, args.plot_dir, not args.no_plots)
    print(json.dumps({'status': result['status'], 'comparison': bind(args.output), 'training_totals': result['training_totals'],
                      'mean_metrics': result['arithmetic_mean_of_two_seeds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
