"""Summarize separately labeled post-hoc decoder diagnostics without inference.

The frozen comparison is preserved. This report does not interpret a changed
decoder score as improved model weights. U+FFFD remains in all predicted text,
lengths, output frequencies and independently recounted recognition errors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUTPUT = BASE / 'results/decode_correction'
ORIGINAL_COMPARISON_SHA = '9a18036ebf5a35409ca9649afe9832e0424e112329303abdb1471b095a8f1a84'
PAIRS = ((42, 'A_ctc'), (42, 'B_feature'), (43, 'A_ctc'), (43, 'B_feature'))
CONDITIONS = ('normal', 'static', 'shuffled')
INVALID_JAMO_PREFIX = 'ValueError: 잘못된 자모 순서/토큰'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# These modules contain only read-only pure analysis helpers on import.
summary_helpers = module_at('step05_posthoc_summary_helpers', BASE / 'summarize_results.py')
prediction_helpers = module_at('step05_posthoc_prediction_helpers', BASE / 'analyze_predictions.py')
bind, read, require, local = summary_helpers.bind, summary_helpers.read, summary_helpers.require, summary_helpers.local


def collapse_ids(ids: list[int]) -> list[int]:
    require(isinstance(ids, list) and ids and all(type(x) is int and 0 <= x < 69 for x in ids), 'Malformed saved argmax IDs')
    collapsed = []
    previous = None
    for value in ids:
        if value != previous and value != 0:
            collapsed.append(value)
        previous = value
    return collapsed


def validate_record(row: dict, old: dict, tokenizer) -> None:
    """Check correction eligibility, exact old-row connection and token retention."""
    for key in ['id', 'speaker_id', 'source_stem', 'split', 'condition', 'reference']:
        require(row[key] == old[key], 'Post-hoc record identity/reference differs: ' + key)
    require(row['original'] == {key: old[key] for key in ['status', 'prediction', 'error', 'metrics']}, 'Embedded original record differs')
    eligible = old['status'] == 'ok' or str(old.get('error', '')).startswith(INVALID_JAMO_PREFIX)
    if not eligible:
        require(row['status'] == 'failed' and row['prediction'] == '', 'Unrelated old failure was silently corrected')
        return
    require(row['status'] == 'ok' and row['error'] is None, 'Eligible decoder replay failed')
    require(row['temporal'] == old['temporal'] and row['frames'] == old['frames'], 'Post-hoc temporal input differs')
    require(row['strict_replay_matches_original'] is True, 'Strict original output/error was not reproduced')
    raw, collapsed = row['raw_argmax_ids'], row['collapsed_ids']
    require(len(raw) == row['frames'] and collapsed == collapse_ids(raw), 'CTC token sequence was changed or deleted')
    composed = tokenizer.compose_jamos([tokenizer.IDX_TO_GRAPHEME[index] for index in collapsed], strict=False)
    require(row['prediction'] == composed, 'Post-hoc text does not preserve the existing tolerant composition')
    require(row['replacement_characters'] == row['prediction'].count('\ufffd'), 'Replacement-character evidence differs')
    if old['status'] == 'ok':
        require(row['prediction'] == old['prediction'] and row['original_success_prediction_match'] is True,
                'Originally valid prediction was changed')
        require(row['correction'] == 'unchanged_valid_prediction', 'Wrong correction label for valid output')
        require(row['strict_replay'] == {'status': 'ok', 'prediction': old['prediction'], 'error': None}, 'Original successful replay differs')
    else:
        require(row['replacement_characters'] > 0 and row['original_success_prediction_match'] is None, 'Invalid jamo must remain an error character')
        require(row['correction'] == 'invalid_predicted_jamo_retained_as_replacement_character', 'Wrong invalid-output correction label')
        require(row['strict_replay'] == {'status': 'failed', 'prediction': None, 'error': old['error']}, 'Original invalid-jamo error differs')


def diagnostics(value: dict) -> dict:
    """Adapt the already validated schema only in memory to reuse pure counts."""
    adapted = {**value, 'schema': 'lipnet-step05-final-development-evaluation-v1'}
    result = prediction_helpers.analyze_evaluation(adapted)
    for condition in CONDITIONS:
        records = value['conditions'][condition]['records']
        count = sum(row['prediction'].count('\ufffd') for row in records)
        sentences = sum('\ufffd' in row['prediction'] for row in records)
        result['conditions'][condition].update({'replacement_characters': count,
            'sentences_with_replacement_characters': sentences,
            'mean_replacement_characters_per_sentence': count / len(records),
            'replacement_characters_preserved_in_lengths_frequencies_and_error_metrics': True})
    return result


def compare_original(evaluations: dict, originals: dict) -> dict:
    per_seed = []
    for seed, arm in PAIRS:
        conditions = {}
        for condition in CONDITIONS:
            previous = originals[(seed, arm)]['conditions'][condition]['aggregate']
            current = evaluations[(seed, arm)]['conditions'][condition]['aggregate']
            conditions[condition] = {'original': previous, 'posthoc_decoder_correction': current,
                'posthoc_minus_original_cer_percentage_points': current['cer_percent'] - previous['cer_percent'],
                'posthoc_minus_original_wer_percentage_points': current['wer_percent'] - previous['wer_percent'],
                'posthoc_minus_original_exact_sentences': current['exact_sentences'] - previous['exact_sentences']}
        per_seed.append({'seed': seed, 'arm': arm, 'conditions': conditions})
    means = {}
    for arm in ('A_ctc', 'B_feature'):
        selected = [entry for entry in per_seed if entry['arm'] == arm]
        means[arm] = {condition: {field: statistics.mean(entry['conditions'][condition][field] for entry in selected)
                                 for field in ['posthoc_minus_original_cer_percentage_points', 'posthoc_minus_original_wer_percentage_points',
                                               'posthoc_minus_original_exact_sentences']} for condition in CONDITIONS}
    return {'per_seed_and_arm': per_seed, 'mean_changes_across_two_seeds': means,
            'interpretation': 'These differences result from post-hoc prediction decoding; model weights, training and selected checkpoints did not change.'}


def plot(paired: dict, path: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = local(path)
    require(not path.exists(), 'Refusing to overwrite corrected diagnostic figure')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    colors = {'A_ctc': '#2367a3', 'B_feature': '#c65f1a'}
    for axis, seed in zip(axes, paired['per_seed'], strict=True):
        for index, arm in enumerate(('A_ctc', 'B_feature')):
            positions = [k + (-.18 if index == 0 else .18) for k in range(3)]
            values = [seed['conditions'][condition][arm]['cer_percent'] for condition in CONDITIONS]
            bars = axis.bar(positions, values, width=.34, color=colors[arm], label=arm)
            axis.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=3, fontsize=9)
        axis.set_xticks(range(3), CONDITIONS)
        axis.set_title(f"Seed {seed['seed']} | unchanged update3200 weights")
        axis.set_ylim(bottom=0)
        axis.margins(y=.2)
        axis.grid(axis='y', alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, loc='lower left')
    axes[0].set_ylabel('Pooled NFC CER (%) - lower is better')
    fig.suptitle('Post-hoc prediction decoder correction', fontsize=15)
    fig.text(.5, .035, 'U+FFFD replacement characters are retained as recognition errors.\nOriginal evaluation remains preserved; this is not a model improvement.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return bind(path)


def summarize(output: Path, figure: Path) -> dict:
    require(not output.exists(), 'Refusing to overwrite post-hoc analysis')
    original_ref = {'path': str(BASE / 'results/comparison.json'), 'sha256': ORIGINAL_COMPARISON_SHA}
    original_comparison = read(original_ref)
    summary_ref = bind(OUTPUT / 'summary.json')
    completion = read(summary_ref)
    require(completion['schema'] == 'lipnet-step05-posthoc-prediction-decode-summary-v1'
            and completion['status'] == 'completed' and completion['analysis_type'] == 'posthoc_decoder_correction', 'Post-hoc evaluation incomplete')
    require(completion['official_frozen_results_replaced'] is False and completion['new_optimizer_updates'] == 0, 'Original comparison was replaced or training changed')
    plan_ref = completion['posthoc_plan']
    plan = read(plan_ref)
    require(plan['schema'] == 'lipnet-step05-posthoc-prediction-decode-plan-v1'
            and plan['status'] == 'frozen_before_inference' and plan['new_optimizer_updates'] == 0
            and plan['original_protocol_and_results_preserved'] is True, 'Post-hoc plan invariant differs')
    require([(r['seed'], r['arm']) for r in plan['models']] == list(PAIRS), 'Post-hoc model order changed')
    for ref in plan['code_bindings']:
        require(bind(ref['path']) == ref, 'Post-hoc decoding code differs')
    tokenizer_ref = next(ref for ref in plan['code_bindings'] if Path(ref['path']).resolve() == ROOT / 'backend/models/tokenizer.py')
    tokenizer = module_at('step05_posthoc_bound_tokenizer_for_summary', Path(tokenizer_ref['path']))
    require(len(completion['evaluations']) == 4, 'Require four corrected evaluations')
    original_references = {ref['path']: ref for ref in original_comparison['verified_evidence']}
    original_by_pair = {(row['seed'], row['arm']): row for row in original_comparison['training_runs']}
    evaluations, originals, files = {}, {}, [original_ref, summary_ref, plan_ref]
    for entry, ref in zip(plan['models'], completion['evaluations'], strict=True):
        key = entry['seed'], entry['arm']
        require(local(entry['output']) == local(ref['path']), 'Post-hoc output reference differs')
        require(entry['original_evaluation'] == original_references.get(entry['original_evaluation']['path']), 'Post-hoc original evaluation is not the preserved original comparison')
        previous, current = read(entry['original_evaluation']), read(ref)
        prediction_helpers.validate_evaluation(previous)
        require((current['seed'], current['arm']) == key and current['schema'] == 'lipnet-step05-posthoc-prediction-decode-evaluation-v1'
                and current['status'] == 'completed' and current['analysis_type'] == 'posthoc_decoder_correction', 'Wrong corrected evaluation')
        require(current['posthoc_plan'] == plan_ref and current['original_evaluation'] == entry['original_evaluation'], 'Wrong post-hoc source/plan')
        for name in ['ids', 'completed_updates', 'record_count', 'plan', 'run_result', 'checkpoint', 'full_data_plan', 'runtime', 'model_state_before', 'model_state_after']:
            require(current[name] == previous[name], 'Corrected evaluation changed immutable evidence: ' + name)
        require(current['run_result'] == original_by_pair[key]['run_result'], 'Changed trained model result')
        require(current['official_frozen_results_replaced'] is False and current['new_optimizer_updates'] == 0
                and current['model_state_unchanged'] is True, 'Corrected evaluation changed original experiment')
        require(all(current[name] is False for name in ['audio_read', 'teacher_targets_read', 'teacher_executed', 'service_weights_updated']), 'Changed diagnostic scope')
        require(set(current['conditions']) == set(CONDITIONS), 'Missing post-hoc condition')
        for condition in CONDITIONS:
            old_block, block = previous['conditions'][condition], current['conditions'][condition]
            require([r['id'] for r in block['records']] == previous['ids'], 'Changed post-hoc row order/denominator')
            for row, old in zip(block['records'], old_block['records'], strict=True):
                validate_record(row, old, tokenizer)
            require(summary_helpers.score_records(block['records']) == block['aggregate'], 'Post-hoc text metrics differ from independent counts')
            require(summary_helpers.score_records(old_block['records']) == old_block['aggregate'], 'Original text metrics differ')
            require(block['original_aggregate'] == old_block['aggregate'], 'Original aggregate changed')
            require(block['posthoc_minus_original_cer_pp'] == block['aggregate']['cer_percent'] - old_block['aggregate']['cer_percent'], 'Wrong original/post-hoc CER difference')
            counts = {'corrected_invalid_jamo_failures': sum(r['correction'] == 'invalid_predicted_jamo_retained_as_replacement_character' for r in block['records']),
                      'unchanged_original_successes': sum(r['original_success_prediction_match'] is True for r in block['records']),
                      'sentences_with_replacement_characters': sum('\ufffd' in r['prediction'] for r in block['records']),
                      'replacement_characters': sum(r['prediction'].count('\ufffd') for r in block['records'])}
            for name, value in counts.items():
                require(block[name] == value, 'Wrong correction/replacement character count: ' + name)
        evaluations[key], originals[key] = current, previous
        files.extend([entry['original_evaluation'], ref])
    original_pairs = summary_helpers.paired_metrics(originals)
    for name in ['per_seed', 'arithmetic_mean_of_two_seeds', 'mean_ablation_gaps']:
        require(original_pairs[name] == original_comparison[name], 'Original preserved comparison arithmetic differs')
    expected_models = [{'seed': key[0], 'arm': key[1], 'conditions': {condition: {name: values[name] for name in (
        'aggregate', 'original_aggregate', 'posthoc_minus_original_cer_pp', 'corrected_invalid_jamo_failures',
        'unchanged_original_successes', 'sentences_with_replacement_characters', 'replacement_characters')}
        for condition, values in evaluations[key]['conditions'].items()}} for key in PAIRS]
    require(completion['models'] == expected_models, 'Post-hoc completion summary differs from four evaluation records')
    paired = summary_helpers.paired_metrics(evaluations)
    image_ref = plot(paired, figure)
    result = {'schema': 'lipnet-step05-posthoc-decoder-comparison-analysis-v1', 'status': 'completed',
        'analysis_type': 'posthoc_decoder_correction', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'code': bind(Path(__file__)), 'tests': bind(BASE / 'test_summarize_decode_correction.py'),
        'pure_analysis_helpers': [bind(BASE / 'summarize_results.py'), bind(BASE / 'analyze_predictions.py')],
        'evidence': files, 'posthoc_summary': summary_ref, 'original_comparison': original_ref,
        'figure': image_ref, **paired, 'posthoc_versus_original': compare_original(evaluations, originals),
        'prediction_diagnostics': [diagnostics(evaluations[key]) for key in PAIRS],
        'replacement_policy': 'Each malformed predicted jamo is retained as U+FFFD; never stripped from CER, lengths, frequencies or examples.',
        'example_policy': 'First five fixed development rows, never selected by outcome.',
        'original_results_preserved': True, 'new_optimizer_updates': 0, 'models_constructed': False, 'inference_executed': False,
        'interpretation_limits': ['Correction settings were chosen after observing original evaluation failures; this is explicitly a post-hoc decoder diagnostic.',
            'Original strict-decoder comparison remains preserved and is not replaced by these scores.',
            'The same four final checkpoints are reused; score changes do not represent further training or improved model weights.',
            'Original argmax token paths were not stored; exact token-path identity with original inference cannot be recovered retroactively.',
            'Both prespecified seeds are included, and all71 rows remain in every denominator. CER can exceed100% and is not converted to accuracy.',
            'This same-speaker development cohort does not establish unseen-speaker or webcam performance.']}
    for ref in files + plan['code_bindings']:
        require(bind(ref['path']) == ref, 'Evidence changed during readonly aggregation')
    output = local(output)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'comparison.json')
    parser.add_argument('--figure', type=Path, default=OUTPUT / 'development_cer.png')
    args = parser.parse_args()
    result = summarize(args.output, args.figure)
    print(json.dumps({'status': result['status'], 'comparison': bind(args.output),
                      'mean_metrics': result['arithmetic_mean_of_two_seeds'], 'mean_gaps': result['mean_ablation_gaps'],
                      'original_comparison_preserved': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
