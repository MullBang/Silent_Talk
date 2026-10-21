"""Describe oracle candidate ranks without turning them into recognition accuracy."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
CONDITIONS = ('normal', 'static', 'shuffled')
GROUPS = ('same6', 'same_old4', 'same_new2', 'different6', 'different_old4', 'different_new2')
SCORE_KEYS = ('total_log_probability', 'mean_log_probability_per_target_token')
EPSILON = 1e-8


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def numeric_summary(values):
    selected = [value for value in values if value is not None]
    require(all(math.isfinite(value) for value in selected), 'Nonfinite summarized score')
    return {'count': len(selected), 'unavailable': len(values) - len(selected),
            'mean': statistics.mean(selected) if selected else None,
            'median': statistics.median(selected) if selected else None,
            'minimum': min(selected) if selected else None, 'maximum': max(selected) if selected else None}


def margin_summary(values):
    return {**numeric_summary(values),
            'positive_count': sum(value is not None and value > EPSILON for value in values),
            'tied_count': sum(value is not None and abs(value) <= EPSILON for value in values),
            'negative_count': sum(value is not None and value < -EPSILON for value in values)}


def aggregate_rows(rows):
    """Failures remain in the declared row denominator; finite means say their scope."""
    require(rows and len({row['source_key'] for row in rows}) == len(rows), 'Empty or duplicate row set')
    require(all(row['status'] in ('ok', 'failed') and len(row['candidates']) == 25 for row in rows), 'Wrong score row scope')
    successful = [row for row in rows if row['status'] == 'ok']
    rankings = {}
    for key in SCORE_KEYS:
        values = [row['rankings'][key] for row in successful]
        require(all(1 <= value['reference_rank'] <= 25 for value in values), 'Invalid reference rank')
        rankings[key] = {
            'reference_top_including_ties_count': sum(value['reference_rank'] == 1 for value in values),
            'reference_unique_top_count': sum(value['reference_rank'] == 1 and len(value['top_candidate_ids']) == 1 for value in values),
            'reference_rank_histogram': dict(sorted(Counter(str(value['reference_rank']) for value in values).items(), key=lambda item: int(item[0]))),
            'reference_rank_over_scored_rows': numeric_summary([value['reference_rank'] for value in values]),
            'reference_minus_best_training_over_scored_rows': margin_summary([value['reference_minus_best_training'] for value in values]),
            'same_minus_other_training_over_scored_rows': margin_summary([value['same_minus_other_training_score'] for value in values]),
            'best_training_relation_counts': {relation: sum(value['best_training_relation'] == relation for value in values)
                                              for relation in ('same_speaker', 'other_speaker', 'mixed_tie', None)},
            'feasible_candidates_over_scored_rows': numeric_summary([value['feasible_candidates'] for value in values])}
        # JSON object keys should describe missing training comparators explicitly.
        rankings[key]['best_training_relation_counts']['unavailable'] = rankings[key]['best_training_relation_counts'].pop(None)
    return {'rows_total': len(rows), 'rows_scored': len(successful), 'rows_failed': len(rows) - len(successful),
            'candidate_pairs': len(rows) * 25,
            'feasible_candidate_pairs': sum(c['status'] == 'ok' for row in rows for c in row['candidates']),
            'infeasible_candidate_pairs': sum(c['status'] == 'infeasible' for row in rows for c in row['candidates']),
            'source_failed_candidate_pairs': sum(c['status'] == 'source_failed' for row in rows for c in row['candidates']),
            'rankings': rankings,
            'reference_minus_greedy_total_log_probability_over_scored_rows': margin_summary([
                row['reference_minus_greedy_total_log_probability'] for row in successful]),
            'empty_greedy_sequences': sum(row['greedy_score']['target_tokens'] == 0 for row in successful),
            'ranks_are_recognition_accuracy': False}


def source_metrics(sources):
    """Report unchanged greedy transcript metrics on every original source row."""
    metrics = [source['source_record']['metrics'] for source in sources]
    chars = sum(row['reference_characters'] for row in metrics)
    words = sum(row['reference_words'] for row in metrics)
    character_edits = sum(row['character_edits'] for row in metrics)
    word_edits = sum(row['word_edits'] for row in metrics)
    return {'rows': len(metrics), 'character_edits': character_edits, 'reference_characters': chars,
            'word_edits': word_edits, 'reference_words': words,
            'exact_sentences': sum(row['exact'] for row in metrics),
            'cer_percent': 100 * character_edits / chars, 'wer_percent': 100 * word_edits / words,
            'source_predictions_unchanged': True}


def validate_scores(value, policy):
    require(value['schema'] == 'lipnet-step14-posterior-candidate-scores-v1' and value['status'] == 'completed', 'Wrong score schema')
    require(value['source_rows'] == 144 and value['candidate_pairs'] == 3600 and value['auxiliary_greedy_pairs'] == 144
            and value['total_ctc_pairs'] == 3744, 'Changed score budget')
    require(value['new_model_inferences'] == value['new_optimizer_updates'] == value['checkpoint_payloads_loaded'] ==
            value['visual_input_caches_loaded'] == 0 and not value['audio_used'] and not value['teacher_used']
            and not value['service_weights_updated'], 'Analysis scope changed')
    require(value['reference_inclusive_candidate_ranks_are_recognition_accuracy'] is False and
            value['token_length_normalized_scores_are_actual_log_probabilities'] is False, 'Misinterpreted score semantics')
    require([row['source_key'] for row in value['rows']] == [source['key'] for source in policy['sources']],
            'Missing, reordered or selected source row')
    for row, source in zip(value['rows'], policy['sources'], strict=True):
        require(all(row[key] == source[key] for key in ('seed', 'arm', 'condition', 'cohort', 'id', 'speaker_id',
                   'frames', 'source_record_sha256', 'source_evaluation', 'posterior_archive', 'posterior_key')),
                'Changed row provenance')
        require(row['candidate_pairs'] == 25 and row['auxiliary_greedy_pairs'] == 1 and len(row['candidates']) == 25,
                'Incomplete candidate row')
    require(value['failed_source_rows'] == sum(row['status'] != 'ok' for row in value['rows']), 'Failure count differs')


def make_comparison(value, policy):
    validate_scores(value, policy)
    models = []
    for seed in SEEDS:
        for arm in ARMS:
            conditions = {}
            for condition in CONDITIONS:
                groups = {}
                for group in GROUPS:
                    ids = policy['groups'][group]
                    selected = [row for row in value['rows'] if row['seed'] == seed and row['arm'] == arm
                                and row['condition'] == condition and row['id'] in ids]
                    sources = [source for source in policy['sources'] if source['seed'] == seed and source['arm'] == arm
                               and source['condition'] == condition and source['id'] in ids]
                    require(len(selected) == len(sources) == len(ids), 'Matched-group denominator changed')
                    groups[group] = {**aggregate_rows(selected), 'original_greedy_metrics': source_metrics(sources)}
                conditions[condition] = groups
            models.append({'seed': seed, 'arm': arm, 'conditions': conditions})
    # Each record here is a model/clip/condition combination, not a distinct person.
    totals = {condition: {cohort: aggregate_rows([row for row in value['rows'] if row['condition'] == condition
                                                and row['cohort'] == cohort]) for cohort in ('same', 'different')}
              for condition in CONDITIONS}
    return {'models': models, 'across_models_condition_cohort_counts': totals,
            'source_rows': 144, 'main_candidate_pairs': 3600, 'auxiliary_greedy_pairs': 144, 'total_ctc_pairs': 3744,
            'failed_source_rows': value['failed_source_rows'], 'infeasible_candidate_pairs': value['infeasible_candidate_pairs'],
            'source_failed_candidate_pairs': value['source_failed_candidate_pairs'],
            'new_model_inferences': 0, 'new_optimizer_updates': 0, 'source_predictions_changed': False,
            'service_weights_updated': False, 'reference_candidate_is_oracle': True, 'ranking_is_accuracy': False,
            'length_normalization_is_heuristic': True, 'new_human_approvals_added': 0,
            'unseen_speaker_generalization_estimate': False, 'pure_recording_environment_causal_effect': False}


def main():
    import protocol
    policy = protocol.verify_frozen()
    path = BASE / 'results/candidate_scores.json'
    reference = bind(path)
    value = read(path)
    require(value['experiment_plan'] == bind(BASE / 'diagnostic_plan.json'), 'Scores use another plan')
    result = {'schema': 'lipnet-step14-candidate-comparison-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': value['experiment_plan'],
              'candidate_scores': reference, 'generator': bind(__file__), **make_comparison(value, policy),
              'limitations': ['Each reference is supplied as an oracle candidate among 24 training utterances.',
                             'Candidate rank is not open-vocabulary recognition accuracy or a deployable decoder.',
                             'Per-token normalization is a heuristic and can change which candidate is preferred.',
                             'Speaker groups have 3 same-speaker and 21 other-speaker training candidates, with unequal lengths.',
                             'Reference outranking greedy would not prove the decoder is the sole problem.',
                             'All cohorts were historical training candidates and new-source labels remain provisional.']}
    require(bind(path) == reference, 'Scores changed during aggregation')
    protocol.verify_frozen()
    output = BASE / 'results/comparison.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(output)))


if __name__ == '__main__':
    main()
