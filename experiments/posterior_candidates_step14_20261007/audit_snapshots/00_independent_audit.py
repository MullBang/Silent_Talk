"""Independent NumPy/stdlib audit of oracle candidate CTC scores.

No production implementation, PyTorch, model or checkpoint is imported/loaded.
All scores are reconstructed from the unchanged saved FP32 posterior arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / 'experiments/recording_probe_step13_20261007'
STEP12 = ROOT / 'experiments/multisentence_step12_20261007'
STEP13_SUMMARY_SHA = 'bb32f1461b8a86f51c9a3e9121c9ce81fb48f62da4b41308a7b42b4c75f66c0d'
STEP13_POLICY_SHA = '9fb25b0a352241b119e0ac92093d7164bd00440024e889d584c1d6d9f269ac16'
STEP12_POLICY_SHA = '2fc10937d9c9a879e62839cac57bbe9337319dff80a3bdf71b5f344f1d701063'
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
CONDITIONS = ('normal', 'static', 'shuffled')
GROUPS = ('same6', 'same_old4', 'same_new2', 'different6', 'different_old4', 'different_new2')
EPSILON = 1e-8
SCORE_KEYS = ('total_log_probability', 'mean_log_probability_per_target_token')


def require(value, message):
    if not value:
        raise ValueError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Only workspace files are permitted')
    return path


def sha(path):
    with local(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    return {'path': str(local(path)), 'sha256': sha(path)}


def verify(reference):
    same(sha(reference['path']), reference['sha256'], 'Changed artifact: ' + reference['path'])
    return local(reference['path'])


def read(path, expected_sha=None):
    content = local(path).read_bytes()
    if expected_sha is not None:
        same(hashlib.sha256(content).hexdigest(), expected_sha, 'Changed JSON artifact')
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(content, parse_constant=invalid)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def normalized_text(value):
    return ''.join(unicodedata.normalize('NFC', value).split())


def minimum_frames(tokens):
    require(isinstance(tokens, list) and tokens and all(type(token) is int and 1 <= token <= 68 for token in tokens),
            'Target must contain nonblank integer tokens')
    return len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))


def posterior_valid(array, frames):
    same(array.dtype, np.dtype('float32'), 'Saved posterior precision')
    same(array.shape, (frames, 69), 'Saved posterior dimensions')
    require(np.isfinite(array).all() and (array <= 1e-5).all(), 'Invalid log probabilities')
    values = array.astype(np.float64)
    maximum = values.max(axis=1)
    error = float(np.max(np.abs(maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1)))))
    require(error <= 1e-5, 'Saved probabilities are not normalized')
    return error


def ctc_log_probability(log_probabilities, target):
    """Independent forward recursion with blank states and repeated-label guards."""
    require(isinstance(log_probabilities, np.ndarray) and log_probabilities.ndim == 2
            and log_probabilities.shape[1] == 69 and len(log_probabilities) > 0
            and np.isfinite(log_probabilities).all(), 'Invalid CTC matrix')
    minimum = minimum_frames(target)
    if len(log_probabilities) < minimum:
        return None
    labels = np.zeros(2 * len(target) + 1, dtype=np.int64)
    labels[1::2] = target
    alpha = np.full(labels.shape, -np.inf, dtype=np.float64)
    alpha[0] = float(log_probabilities[0, 0])
    alpha[1] = float(log_probabilities[0, target[0]])
    jumping = np.zeros(len(labels), dtype=bool)
    jumping[2:] = (labels[2:] != 0) & (labels[2:] != labels[:-2])
    for frame in log_probabilities[1:]:
        prior = alpha.copy()
        one = np.r_[-np.inf, prior[:-1]]
        two = np.r_[-np.inf, -np.inf, prior[:-2]]
        two[~jumping] = -np.inf
        alpha = np.logaddexp(np.logaddexp(prior, one), two) + frame[labels].astype(np.float64)
    result = float(np.logaddexp(alpha[-1], alpha[-2]))
    require(math.isfinite(result), 'Feasible finite posterior yielded nonfinite CTC')
    return result


def collapse(raw):
    return [token for index, token in enumerate(raw) if token and (index == 0 or token != raw[index - 1])]


def compose(tokens):
    output, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if token == 68:
            output.append(' ')
            index += 1
        elif 1 <= token <= 19 and index + 1 < len(tokens) and 20 <= tokens[index + 1] <= 40:
            tail = tokens[index + 2] - 40 if index + 2 < len(tokens) and 41 <= tokens[index + 2] <= 67 else 0
            output.append(chr(0xAC00 + ((token - 1) * 21 + tokens[index + 1] - 20) * 28 + tail))
            index += 3 if tail else 2
        else:
            output.append('\ufffd')
            index += 1
    return ''.join(output)


def distance(first, second):
    previous = list(range(len(second) + 1))
    for index, left in enumerate(first, 1):
        current = [index]
        for position, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[position] + 1, previous[position - 1] + (left != right)))
        previous = current
    return previous[-1]


def validate_source_posterior(source, array):
    """Verify the original transcript survives exactly, without running a network."""
    error = posterior_valid(array, source['frames'])
    row = source['source_record']
    raw = array.argmax(axis=1).tolist()
    tokens = collapse(raw)
    same(raw, row['raw_argmax_ids'], 'Source argmax output differs')
    same(tokens, row['collapsed_ids'], 'Source CTC collapse differs')
    same(compose(tokens), row['prediction'], 'Source tolerant transcript differs')
    return error


def near(actual, expected, message, tolerance=1e-7):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected), message + ': nonfinite/non-numeric value')
    require(abs(actual - expected) <= tolerance, message + ': numeric difference exceeds tolerance')


def near_tree(actual, expected, message, tolerance=1e-7):
    if isinstance(expected, dict):
        same(set(actual), set(expected), message + ': field set')
        for key in expected:
            near_tree(actual[key], expected[key], message + '.' + key, tolerance)
    elif isinstance(expected, list):
        same(len(actual), len(expected), message + ': list length')
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            near_tree(left, right, message + f'[{index}]', tolerance)
    elif type(expected) is float:
        near(actual, expected, message, tolerance)
    else:
        same(actual, expected, message)


def verify_preservation(preservation):
    weights = ROOT / 'backend/models/weights'
    same(preservation['weights_directory'], str(weights), 'Service weights directory')
    same(preservation['weights_inventory'], [str(path) for path in sorted((p for p in weights.rglob('*') if p.is_file()), key=str)],
         'Service weights inventory changed')
    for path in preservation['absent_paths']:
        require(local(path).is_relative_to(weights) and not local(path).exists(), 'Previously absent service weight appeared')
    for reference in preservation['artifacts']:
        verify(reference)


def source_metadata():
    """Trace immutable source rows to Step13 and Step12 without loading models."""
    summary = read(PRIOR / 'results/step13_summary.json', STEP13_SUMMARY_SHA)
    same(summary['status'], 'completed_matched_speaker_recording_probe', 'Prior stage incomplete')
    for reference in summary['artifacts'].values():
        verify(reference)
    previous = read(PRIOR / 'diagnostic_plan.json', STEP13_POLICY_SHA)
    original = read(STEP12 / 'diagnostic_plan.json', STEP12_POLICY_SHA)
    comparison_ref = summary['artifacts']['results/comparison.json']
    comparison = read(comparison_ref['path'], comparison_ref['sha256'])
    original_rows = {row['id']: row for row in original['selection']['records']}
    selected_rows = {row['id']: row for row in previous['selection']['records']}
    training = [original_rows[sid] for sid in original['selection']['train_ids']]
    sources, references = [], [bind(PRIOR / 'results/step13_summary.json'), *summary['artifacts'].values(),
                               bind(PRIOR / 'diagnostic_plan.json'), bind(STEP12 / 'diagnostic_plan.json')]
    models = []
    same([(m['seed'], m['arm']) for m in previous['models']], [(seed, arm) for seed in SEEDS for arm in ARMS], 'Original model order')
    same([(m['seed'], m['arm']) for m in comparison['models']], [(seed, arm) for seed in SEEDS for arm in ARMS], 'Original comparison model order')
    for model, analysis in zip(previous['models'], comparison['models'], strict=True):
        evaluation_ref = analysis['evaluation']
        evaluation = read(evaluation_ref['path'], evaluation_ref['sha256'])
        for key in ('seed', 'arm', 'checkpoint'):
            same(evaluation[key], model[key], 'Source model identity ' + key)
        same(evaluation['experiment_plan'], bind(PRIOR / 'diagnostic_plan.json'), 'Source evaluation experiment')
        same(evaluation['prior_evaluation'], model['prior_evaluation'], 'Reused evaluation binding')
        same(evaluation['prior_posteriors'], model['prior_posteriors'], 'Reused posterior binding')
        original_evaluation = read(model['prior_evaluation']['path'], model['prior_evaluation']['sha256'])
        same(original_evaluation['log_posteriors_archive'], model['prior_posteriors'], 'Original posterior archive')
        final = original_evaluation['milestones'][-1]
        same(final['checkpoint'], model['checkpoint'], 'Original result checkpoint identity')
        same(final['completed_updates'], 3600, 'Fixed checkpoint update')
        checkpoint_ref = {'path': model['checkpoint']['path'], 'sha256': model['checkpoint']['sha256']}
        sidecar = {'path': model['checkpoint']['sidecar'], 'sha256': model['checkpoint']['sidecar_sha256']}
        # Byte hashing confirms preservation; there is no checkpoint deserialization.
        references.extend([evaluation_ref, evaluation['log_posteriors_archive'], model['prior_evaluation'],
                           model['prior_posteriors'], checkpoint_ref, sidecar])
        models.append({'seed': model['seed'], 'arm': model['arm'], 'step13_evaluation': evaluation_ref,
            'step13_posteriors': evaluation['log_posteriors_archive'], 'step12_evaluation': model['prior_evaluation'],
            'step12_posteriors': model['prior_posteriors'], 'checkpoint': model['checkpoint']})
        for condition in CONDITIONS:
            for cohort in ('same', 'different'):
                value = evaluation['conditions'][condition][cohort]
                ids = previous['selection'][cohort + '_ids']
                same([row['id'] for row in value['records']], ids, 'Original row order')
                if cohort == 'different':
                    original_by_id = {row['id']: row for row in final['conditions'][condition]['records']}
                    same(value['records'], [original_by_id[sid] for sid in ids], 'Reused source row mutation')
                    same(value['source_evaluation'], model['prior_evaluation'], 'Different-source evaluation')
                    same(value['source_log_posteriors_archive'], model['prior_posteriors'], 'Different-source posterior')
                for row in value['records']:
                    metadata = selected_rows[row['id']]
                    for key, expected in [('reference', metadata['text']), ('target_token_ids', metadata['tokens']),
                                          ('frames', metadata['frames']), ('speaker_id', metadata['speaker_id']),
                                          ('condition', condition), ('source_split', 'train'), ('status', 'ok')]:
                        same(row[key], expected, 'Original inference identity ' + key)
                    source = {'key': f"seed{model['seed']}/{model['arm']}/{condition}/{cohort}/{row['id']}",
                        'seed': model['seed'], 'arm': model['arm'], 'condition': condition, 'cohort': cohort,
                        'id': row['id'], 'text': row['reference'], 'tokens': row['target_token_ids'], 'frames': row['frames'],
                        'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'], 'generation': metadata['generation'],
                        'original_split': 'train', 'original_role': metadata['original_role'], 'source_record': row,
                        'source_record_sha256': canonical_sha(row), 'source_evaluation': evaluation_ref,
                        'posterior_archive': evaluation['log_posteriors_archive'] if cohort == 'same' else model['prior_posteriors'],
                        'posterior_key': row['log_posteriors_key']}
                    sources.append(source)
    for reference in references:
        verify(reference)
    return {'step13': previous, 'step12': original, 'training': training, 'sources': sources,
            'models': models, 'references': references, 'summary': summary}


def score_value(array, tokens, allow_empty=False):
    if allow_empty and not tokens:
        return {'target_tokens': 0, 'minimum_frames': 0, 'status': 'ok',
                'total_log_probability': float(array[:, 0].astype(np.float64).sum()),
                'mean_log_probability_per_target_token': None}
    minimum = minimum_frames(tokens)
    total = ctc_log_probability(array, tokens)
    return {'target_tokens': len(tokens), 'minimum_frames': minimum,
            'status': 'infeasible' if total is None else 'ok', 'total_log_probability': total,
            'mean_log_probability_per_target_token': None if total is None else total / len(tokens)}


def candidates_for(source, training):
    require(len(training) == len({row['id'] for row in training}) == 24, 'Expected 24 unique training candidates')
    require(source['id'] not in {row['id'] for row in training}, 'Reference is a training candidate')
    require(normalized_text(source['text']) not in {normalized_text(row['text']) for row in training}, 'Reference duplicates training transcript')
    require(sum(row['speaker_id'] == source['speaker_id'] for row in training) == 3, 'Expected same-speaker3 and other-speaker21')
    output = [{key: source[key] for key in ('id', 'text', 'tokens', 'speaker_id')}]
    output[0].update(kind='reference', same_speaker=True)
    for candidate in sorted(training, key=lambda value: value['id']):
        minimum_frames(candidate['tokens'])
        output.append({**{key: candidate[key] for key in ('id', 'text', 'tokens', 'speaker_id')},
                       'kind': 'training', 'same_speaker': candidate['speaker_id'] == source['speaker_id']})
    return output


def independent_ranking(candidates, score_key):
    require(score_key in SCORE_KEYS, 'Unknown scoring axis')
    require(len({row['id'] for row in candidates}) == len(candidates), 'Duplicate ranked candidate IDs')
    references = [row for row in candidates if row['kind'] == 'reference']
    require(len(references) == 1 and references[0]['status'] == 'ok', 'Exactly one feasible reference required')
    require(all(row['status'] in ('ok', 'infeasible') for row in candidates), 'Unsupported candidate status')
    feasible = [row for row in candidates if row['status'] == 'ok']
    require(all(type(row[score_key]) in (int, float) and math.isfinite(row[score_key]) for row in feasible), 'Nonfinite candidate score')
    reference = references[0][score_key]
    train = [row for row in feasible if row['kind'] == 'training']
    def highest(values):
        if not values:
            return None, []
        value = max(row[score_key] for row in values)
        return value, sorted(row['id'] for row in values if abs(row[score_key] - value) <= EPSILON)
    top, top_ids = highest(feasible)
    best, best_ids = highest(train)
    own, own_ids = highest([row for row in train if row['same_speaker']])
    other, other_ids = highest([row for row in train if not row['same_speaker']])
    matching = {row['same_speaker'] for row in train if row['id'] in best_ids}
    relation = ('mixed_tie' if matching == {True, False} else 'same_speaker' if matching == {True}
                else 'other_speaker' if matching == {False} else None)
    return {'reference_rank': 1 + sum(row[score_key] > reference + EPSILON for row in feasible),
        'reference_tie_ids': sorted(row['id'] for row in feasible if abs(row[score_key] - reference) <= EPSILON),
        'reference_score': reference, 'top_score': top, 'top_candidate_ids': top_ids,
        'best_training_score': best, 'best_training_ids': best_ids,
        'reference_minus_best_training': None if best is None else reference - best,
        'best_same_speaker_training_score': own, 'best_same_speaker_training_ids': own_ids,
        'best_other_speaker_training_score': other, 'best_other_speaker_training_ids': other_ids,
        'same_minus_other_training_score': None if own is None or other is None else own - other,
        'best_training_relation': relation,
        'ordered_candidate_ids': [row['id'] for row in sorted(feasible, key=lambda row: (-row[score_key], row['id']))],
        'infeasible_candidate_ids': sorted(row['id'] for row in candidates if row['status'] == 'infeasible'),
        'feasible_candidates': len(feasible), 'infeasible_candidates': len(candidates) - len(feasible)}


def audit_contract(expected_policy_sha=None):
    policy_path = BASE / 'diagnostic_plan.json'
    policy = read(policy_path, expected_policy_sha)
    frozen = read(BASE / 'plans/freeze_manifest.json')
    for key, value in [('schema', 'lipnet-step14-freeze-manifest-v1'), ('status', 'frozen'),
                       ('experiment_plan', bind(policy_path)), ('main_candidate_pairs_executed', 0),
                       ('auxiliary_greedy_pairs_executed', 0), ('new_neural_inference_rows', 0)]:
        same(frozen[key], value, 'Freeze manifest ' + key)
    for key, value in [('schema', 'lipnet-step14-posterior-candidates-protocol-v1'), ('status', 'frozen_before_scoring'),
                       ('main_candidate_pairs_planned', 3600), ('auxiliary_greedy_pairs_planned', 144),
                       ('total_score_pairs_planned', 3744), ('new_neural_inference_rows', 0), ('new_optimizer_updates', 0),
                       ('service_weights_updated', False), ('ranking_is_accuracy', False),
                       ('reference_candidate_is_oracle', True), ('new_human_approvals_added', 0)]:
        same(policy[key], value, 'Frozen diagnostic scope ' + key)
    metadata = source_metadata()
    same(policy['sources'], metadata['sources'], 'Independent 144 source rows')
    same(policy['models'], metadata['models'], 'Four original model/posterior bindings')
    for key in ('groups', 'pairs'):
        same(policy[key], metadata['step13']['selection'][key], 'Original matched speaker ' + key)
    fields = ('id', 'speaker_id', 'source_stem', 'text', 'tokens', 'generation', 'original_split', 'original_role', 'diagnostic_role')
    training = [{key: row[key] for key in fields} for row in metadata['training']]
    same(policy['training_candidates'], training, 'Unchanged training candidate metadata')
    same(len({normalized_text(row['text']) for row in training}), 24, 'Duplicate training text')
    same(len(metadata['sources']), 144, 'Source row count')
    same(len({row['key'] for row in metadata['sources']}), 144, 'Unique source row count')
    for source in metadata['sources']:
        require(source['frames'] >= minimum_frames(source['tokens']), 'Reference is CTC-infeasible')
        candidates_for(source, training)
    for key, expected in [('prior_step13_summary', bind(PRIOR / 'results/step13_summary.json')),
                          ('prior_step13_policy', bind(PRIOR / 'diagnostic_plan.json')),
                          ('prior_step13_comparison', metadata['summary']['artifacts']['results/comparison.json']),
                          ('prior_step13_audit', metadata['summary']['artifacts']['results/independent_audit.json']),
                          ('prior_step12_policy', bind(STEP12 / 'diagnostic_plan.json'))]:
        same(policy[key], expected, 'Prior artifact anchor ' + key)
    same(policy['scope'], {'source_rows': 144, 'unique_evaluation_clips': 12, 'training_candidates': 24,
        'same_recording_source_rows': 72, 'different_recording_source_rows': 72,
        'all_original_source_splits': 'train', 'historically_seen_by_older_models_possible': True,
        'new_human_approvals_added': 0, 'untouched_test_set': False, 'unseen_speakers': False}, 'Honest historical scope')
    contract = policy['scoring_contract']
    for key, value in [('schema', 'lipnet-step14-candidate-scoring-contract-v1'), ('device', 'cpu'),
                       ('calculation_dtype', 'float64'), ('blank_id', 0), ('classes', 69), ('ctc_reduction', 'none'),
                       ('higher_score_is_better', True), ('candidate_count_per_source', 25),
                       ('candidate_ranking_axes', list(SCORE_KEYS)), ('rank_tolerance', EPSILON),
                       ('main_candidate_pairs', 3600), ('auxiliary_greedy_pairs', 144), ('total_score_pairs', 3744),
                       ('all_conditions_reported', list(CONDITIONS)), ('reference_in_candidate_set_is_oracle', True),
                       ('rank_is_recognition_accuracy', False), ('deployable_candidate_recognizer', False),
                       ('new_neural_inference_rows', 0), ('optimizer_updates', 0), ('new_audio_or_video_read', False)]:
        same(contract[key], value, 'Scoring contract ' + key)
    same(contract['auxiliary_greedy']['excluded_from_candidate_ranks'], True, 'Auxiliary greedy leaked into candidate ranks')
    same(policy['loading_scope'], {'saved_posterior_archives_opened': 8, 'saved_posterior_arrays_decoded': 144,
        'saved_posterior_frames_decoded': 17640, 'student_input_npz_opened': 0, 'teacher_feature_npz_opened': 0,
        'raw_audio_or_video_opened': 0, 'checkpoint_tensors_deserialized': 0, 'checkpoint_files_hashed': 4,
        'new_neural_inference_rows': 0, 'optimizer_updates': 0}, 'Posterior-only input scope')
    references = metadata['references'] + policy['execution_code'] + policy['source_bindings'] + policy['preservation']['artifacts']
    references += [bind(policy_path), bind(BASE / 'plans/freeze_manifest.json')]
    for reference in references:
        verify(reference)
    verify_preservation(metadata['step13']['preservation'])
    verify_preservation(policy['preservation'])
    arrays, maximum_error = {}, 0.
    archive_sources = {}
    for source in metadata['sources']:
        archive_sources.setdefault(source['posterior_archive']['path'], []).append(source)
    same(len(archive_sources), 8, 'Posterior archive count')
    for path, sources in archive_sources.items():
        verify(sources[0]['posterior_archive'])
        with np.load(path, allow_pickle=False) as archive:
            keys = [source['posterior_key'] for source in sources]
            same(len(keys), len(set(keys)), 'Duplicate posterior evidence key within archive')
            for source in sources:
                array = archive[source['posterior_key']]
                maximum_error = max(maximum_error, validate_source_posterior(source, array))
                arrays[source['key']] = array
        verify(sources[0]['posterior_archive'])
    return policy, arrays, references, maximum_error


def verify_scoring_sources(references, execution_code):
    paths = [BASE / 'scoring.py', BASE / 'test_scoring.py', ROOT / 'backend/models/tokenizer.py',
             ROOT / 'experiments/olkavs_official/diagnostic_utils.py', ROOT / 'experiments/olkavs_official/evaluation_metrics.py']
    frozen = {row['path']: row for row in execution_code}
    require(all(str(path) in frozen for path in paths), 'Incomplete frozen scoring dependencies')
    same(references, [frozen[str(path)] for path in paths], 'Missing/reordered scoring dependency evidence')
    for reference in references:
        verify(reference)


def audit_scored_row(actual, source, training, array):
    expected = {key: source[key] for key in ('seed', 'arm', 'condition', 'cohort', 'id', 'speaker_id', 'frames',
                                           'source_record_sha256', 'source_evaluation', 'posterior_archive', 'posterior_key')}
    expected.update(source_key=source['key'], status='ok', error=None, candidate_pairs=25, auxiliary_greedy_pairs=1)
    for key, value in expected.items():
        same(actual[key], value, 'Scored source identity ' + key)
    candidates, maximum = [], 0.
    for candidate, scored in zip(candidates_for(source, training), actual['candidates'], strict=True):
        value = score_value(array, candidate['tokens'])
        expected_score = {**candidate, **value, 'probability_is_zero': value['status'] == 'infeasible'}
        near_tree(scored, expected_score, 'Independent candidate score')
        if value['status'] == 'ok':
            maximum = max(maximum, abs(scored['total_log_probability'] - value['total_log_probability']))
        candidates.append(expected_score)
    same(len(candidates), 25, 'Candidate count')
    near_tree(actual['rankings'], {key: independent_ranking(candidates, key) for key in SCORE_KEYS}, 'Independent candidate rankings')
    tokens = source['source_record']['collapsed_ids']
    greedy = {'tokens': tokens, **score_value(array, tokens, allow_empty=True), 'probability_is_zero': False}
    same(greedy['status'], 'ok', 'Actual greedy path cannot be infeasible')
    near_tree(actual['greedy_score'], greedy, 'Independent greedy CTC score')
    maximum = max(maximum, abs(actual['greedy_score']['total_log_probability'] - greedy['total_log_probability']))
    margin = candidates[0]['total_log_probability'] - greedy['total_log_probability']
    near(actual['reference_minus_greedy_total_log_probability'], margin, 'Reference/greedy total-score margin')
    validation = actual['source_validation']
    original = source['source_record']
    for key, value in [('posterior_shape', [source['frames'], 69]), ('posterior_saved_dtype', 'float32'),
                       ('raw_argmax_ids', original['raw_argmax_ids']), ('collapsed_ids', original['collapsed_ids']),
                       ('prediction', original['prediction']), ('replacement_characters', original['replacement_characters']),
                       ('metrics', original['metrics']), ('token_metrics', original['token_metrics']),
                       ('source_ctc_loss_mean_float32', original['ctc_loss_mean']), ('raw_collapsed_text_and_metrics_unchanged', True)]:
        same(validation[key], value, 'Preserved original decoding evidence ' + key)
    near(validation['normalization_max_abs_logsumexp'], posterior_valid(array, source['frames']), 'Posterior normalization evidence', 1e-12)
    expected_ctc = -candidates[0]['mean_log_probability_per_target_token']
    near(validation['reference_float64_mean_ctc'], expected_ctc, 'Reference mean CTC64')
    near(validation['reference_float64_minus_saved_float32_mean_ctc'], expected_ctc - original['ctc_loss_mean'], 'CTC precision difference')
    tolerance = 1e-5 + 1e-6 * abs(original['ctc_loss_mean'])
    near(validation['recomputed_ctc_loss_mean_float32'], original['ctc_loss_mean'], 'Recomputed source FP32 CTC', tolerance)
    near(expected_ctc, original['ctc_loss_mean'], 'Independent reference CTC versus saved FP32', tolerance)
    return {'candidate_pairs': 25, 'auxiliary_greedy_pairs': 1,
            'infeasible_candidate_pairs': sum(row['status'] == 'infeasible' for row in candidates),
            'maximum_total_score_abs_difference': maximum,
            'maximum_reference_ctc32_abs_difference': abs(expected_ctc - original['ctc_loss_mean'])}


def audit_scores(policy, arrays, preflight_ref):
    path = BASE / 'results/candidate_scores.json'
    reference = bind(path)
    result = read(path, reference['sha256'])
    for key, value in [('schema', 'lipnet-step14-posterior-candidate-scores-v1'), ('status', 'completed'),
                       ('experiment_plan', bind(BASE / 'diagnostic_plan.json')), ('scoring_contract', policy['scoring_contract']),
                       ('source_rows', 144), ('candidate_pairs', 3600), ('auxiliary_greedy_pairs', 144), ('total_ctc_pairs', 3744),
                       ('failed_source_rows', 0), ('source_failed_candidate_pairs', 0), ('new_model_inferences', 0),
                       ('new_optimizer_updates', 0), ('checkpoint_payloads_loaded', 0), ('visual_input_caches_loaded', 0),
                       ('audio_used', False), ('teacher_used', False), ('service_weights_updated', False), ('cpu_rng_unchanged', True),
                       ('reference_inclusive_candidate_ranks_are_recognition_accuracy', False),
                       ('token_length_normalized_scores_are_actual_log_probabilities', False)]:
        same(result[key], value, 'Scoring result scope ' + key)
    verify_scoring_sources(result['scoring_code'], policy['execution_code'])
    verify(result['before_scoring_contract'])
    contract = read(result['before_scoring_contract']['path'])
    for key, value in [('schema', 'lipnet-step14-before-scoring-contract-v1'), ('experiment_plan', bind(BASE / 'diagnostic_plan.json')),
                       ('independent_preflight', preflight_ref), ('scoring_code', result['scoring_code']),
                       ('new_model_inferences', 0), ('new_optimizer_updates', 0)]:
        same(contract[key], value, 'Before-scoring contract ' + key)
    preflight = read(preflight_ref['path'])
    require(datetime.fromisoformat(preflight['created_utc']) <= datetime.fromisoformat(contract['created_utc'])
            <= datetime.fromisoformat(result['created_utc']), 'Preflight/scoring chronology differs')
    require(Path(preflight_ref['path']).stat().st_mtime <= Path(result['before_scoring_contract']['path']).stat().st_mtime,
            'Preflight written after scoring contract')
    same([row['source_key'] for row in result['rows']], [source['key'] for source in policy['sources']], 'All fixed source rows/order')
    checked = [audit_scored_row(row, source, policy['training_candidates'], arrays[source['key']])
               for row, source in zip(result['rows'], policy['sources'], strict=True)]
    infeasible = sum(row['infeasible_candidate_pairs'] for row in checked)
    same(result['infeasible_candidate_pairs'], infeasible, 'Infeasible pair accounting')
    verify(reference)
    return result, {'scores': reference, 'source_rows_checked': 144, 'candidate_pairs_checked': 3600,
        'auxiliary_greedy_pairs_checked': 144, 'total_ctc_pairs_checked': 3744, 'infeasible_candidate_pairs_checked': infeasible,
        'maximum_total_score_abs_difference': max(row['maximum_total_score_abs_difference'] for row in checked),
        'maximum_reference_ctc32_abs_difference': max(row['maximum_reference_ctc32_abs_difference'] for row in checked)}


def numeric_summary(values, margins=False):
    finite = sorted(value for value in values if value is not None)
    require(all(math.isfinite(value) for value in finite), 'Nonfinite summarized value')
    count = len(finite)
    result = {'count': count, 'unavailable': len(values) - count,
              'mean': math.fsum(finite) / count if count else None,
              'median': (finite[count // 2] if count % 2 else (finite[count // 2 - 1] + finite[count // 2]) / 2) if count else None,
              'minimum': finite[0] if count else None, 'maximum': finite[-1] if count else None}
    if margins:
        result.update(positive_count=sum(value > EPSILON for value in finite), tied_count=sum(abs(value) <= EPSILON for value in finite),
                      negative_count=sum(value < -EPSILON for value in finite))
    return result


def aggregate_scored_rows(rows):
    require(rows and len({row['source_key'] for row in rows}) == len(rows), 'Invalid aggregate source denominator')
    successful = [row for row in rows if row['status'] == 'ok']
    axes = {}
    for axis in SCORE_KEYS:
        values = [row['rankings'][axis] for row in successful]
        histogram = Counter(str(value['reference_rank']) for value in values)
        relations = Counter(value['best_training_relation'] for value in values)
        axes[axis] = {
            'reference_top_including_ties_count': sum(value['reference_rank'] == 1 for value in values),
            'reference_unique_top_count': sum(value['reference_rank'] == 1 and len(value['top_candidate_ids']) == 1 for value in values),
            'reference_rank_histogram': dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
            'reference_rank_over_scored_rows': numeric_summary([value['reference_rank'] for value in values]),
            'reference_minus_best_training_over_scored_rows': numeric_summary([value['reference_minus_best_training'] for value in values], True),
            'same_minus_other_training_over_scored_rows': numeric_summary([value['same_minus_other_training_score'] for value in values], True),
            'best_training_relation_counts': {'same_speaker': relations['same_speaker'], 'other_speaker': relations['other_speaker'],
                                             'mixed_tie': relations['mixed_tie'], 'unavailable': relations[None]},
            'feasible_candidates_over_scored_rows': numeric_summary([value['feasible_candidates'] for value in values])}
    statuses = Counter(candidate['status'] for row in rows for candidate in row['candidates'])
    return {'rows_total': len(rows), 'rows_scored': len(successful), 'rows_failed': len(rows) - len(successful),
        'candidate_pairs': 25 * len(rows), 'feasible_candidate_pairs': statuses['ok'],
        'infeasible_candidate_pairs': statuses['infeasible'], 'source_failed_candidate_pairs': statuses['source_failed'],
        'rankings': axes, 'reference_minus_greedy_total_log_probability_over_scored_rows': numeric_summary(
            [row['reference_minus_greedy_total_log_probability'] for row in successful], True),
        'empty_greedy_sequences': sum(row['greedy_score']['target_tokens'] == 0 for row in successful),
        'ranks_are_recognition_accuracy': False}


def source_metrics(sources):
    rows = [source['source_record']['metrics'] for source in sources]
    sums = {key: sum(row[key] for row in rows) for key in ('character_edits', 'reference_characters', 'word_edits', 'reference_words')}
    return {'rows': len(rows), **sums, 'exact_sentences': sum(row['exact'] for row in rows),
            'cer_percent': 100 * sums['character_edits'] / sums['reference_characters'],
            'wer_percent': 100 * sums['word_edits'] / sums['reference_words'], 'source_predictions_unchanged': True}


def audit_summary(policy, scores):
    reference = bind(BASE / 'results/comparison.json')
    result = read(reference['path'], reference['sha256'])
    for key, expected in [('schema', 'lipnet-step14-candidate-comparison-v1'), ('status', 'completed'),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')), ('candidate_scores', bind(BASE / 'results/candidate_scores.json')),
                          ('generator', bind(BASE / 'summarize.py')), ('source_rows', 144), ('main_candidate_pairs', 3600),
                          ('auxiliary_greedy_pairs', 144), ('total_ctc_pairs', 3744), ('failed_source_rows', 0),
                          ('infeasible_candidate_pairs', scores['infeasible_candidate_pairs']), ('source_failed_candidate_pairs', 0),
                          ('new_model_inferences', 0), ('new_optimizer_updates', 0), ('source_predictions_changed', False),
                          ('service_weights_updated', False), ('reference_candidate_is_oracle', True), ('ranking_is_accuracy', False),
                          ('length_normalization_is_heuristic', True), ('new_human_approvals_added', 0),
                          ('unseen_speaker_generalization_estimate', False), ('pure_recording_environment_causal_effect', False)]:
        same(result[key], expected, 'Independent comparison identity/scope ' + key)
    models = []
    for seed in SEEDS:
        for arm in ARMS:
            conditions = {}
            for condition in CONDITIONS:
                groups = {}
                for group in GROUPS:
                    ids = policy['groups'][group]
                    selected = [row for row in scores['rows'] if (row['seed'], row['arm'], row['condition']) == (seed, arm, condition) and row['id'] in ids]
                    sources = [row for row in policy['sources'] if (row['seed'], row['arm'], row['condition']) == (seed, arm, condition) and row['id'] in ids]
                    same(len(selected), len(ids), 'Independent group score denominator')
                    same(len(sources), len(ids), 'Independent group original metric denominator')
                    groups[group] = {**aggregate_scored_rows(selected), 'original_greedy_metrics': source_metrics(sources)}
                conditions[condition] = groups
            models.append({'seed': seed, 'arm': arm, 'conditions': conditions})
    near_tree(result['models'], models, 'Independent four-model score summaries')
    totals = {condition: {cohort: aggregate_scored_rows([row for row in scores['rows'] if row['condition'] == condition and row['cohort'] == cohort])
                         for cohort in ('same', 'different')} for condition in CONDITIONS}
    near_tree(result['across_models_condition_cohort_counts'], totals, 'Across-model combination counts')
    verify(reference)
    return reference


def save_new(path, value):
    path = local(path)
    require(path.is_relative_to(BASE) and not path.exists(), 'Refusing to overwrite audit artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def source_snapshots(sources):
    directory = BASE / 'audit_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, source in enumerate(sources):
        path = directory / f"{index:02d}_{Path(source['path']).name}"
        require(not path.exists(), 'Source snapshot already exists')
        with path.open('xb') as stream:
            stream.write(Path(source['path']).read_bytes())
        reference = bind(path)
        same(reference['sha256'], source['sha256'], 'Source changed during snapshot')
        entries.append({'original': source, 'snapshot': reference})
    return save_new(directory / 'preflight_source_snapshot.json', {'sources': entries})


def verify_preflight():
    reference = bind(BASE / 'results/independent_preflight.json')
    value = read(reference['path'])
    for key, expected in [('schema', 'lipnet-step14-independent-preflight-v1'), ('status', 'passed'),
                          ('experiment_plan', bind(BASE / 'diagnostic_plan.json')), ('main_candidate_pairs_executed', 0),
                          ('auxiliary_greedy_pairs_executed', 0), ('saved_posterior_arrays_decoded', 144)]:
        same(value[key], expected, 'Independent preflight ' + key)
    verify(value['source_snapshot_manifest'])
    snapshots = read(value['source_snapshot_manifest']['path'])['sources']
    same([item['original'] for item in snapshots], value['audit_sources'], 'Preflight source snapshot scope')
    for item in snapshots:
        verify(item['snapshot'])
        same(item['snapshot']['sha256'], item['original']['sha256'], 'Snapshot contents changed')
        require(local(item['snapshot']['path']).is_relative_to(BASE / 'audit_snapshots'), 'Invalid source snapshot path')
    return reference


def main():
    import sys
    require('torch' not in sys.modules, 'Independent audit must not import PyTorch')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--expected-policy-sha')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or BASE / 'results' / ('independent_preflight.json' if args.preflight else 'independent_audit.json')
    require(not output.exists(), 'Refusing to overwrite existing audit')
    if args.preflight:
        require(not (BASE / 'results/scoring.contract.json').exists() and not (BASE / 'results/candidate_scores.json').exists(),
                'Preflight must precede scoring')
    started = time.perf_counter()
    policy, arrays, references, max_normalization_error = audit_contract(args.expected_policy_sha)
    sources = [bind(Path(__file__)), bind(BASE / 'test_independent_audit.py')]
    common = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'generator': bind(Path(__file__)),
        'test_source': bind(BASE / 'test_independent_audit.py'), 'audit_sources': sources,
        'source_rows_independently_reconstructed': 144, 'training_candidates_independently_reconstructed': 24,
        'original_greedy_transcripts_and_metrics_preserved': True,
        'saved_posterior_archives_opened': 8, 'saved_posterior_arrays_decoded': 144,
        'saved_posterior_frames_decoded': sum(len(array) for array in arrays.values()),
        'posterior_normalization_max_abs_error': max_normalization_error,
        'new_model_inferences': 0, 'new_optimizer_updates': 0, 'checkpoint_tensors_deserialized': 0,
        'checkpoint_files_hashed': 4, 'student_input_npz_opened': 0, 'teacher_feature_npz_opened': 0,
        'raw_audio_or_video_opened': 0, 'pytorch_imported': False, 'service_weights_preserved': True,
        'reference_inclusive_ranks_are_recognition_accuracy': False,
        'token_length_normalized_scores_are_actual_log_probabilities': False,
        'unseen_speaker_generalization_estimate': False}
    if args.preflight:
        result = {'schema': 'lipnet-step14-independent-preflight-v1', **common,
            'main_candidate_pairs_executed': 0, 'auxiliary_greedy_pairs_executed': 0,
            'source_snapshot_manifest': source_snapshots(sources),
            'scope': 'Pre-scoring immutable source, candidate and posterior integrity audit; no candidate scores computed.'}
    else:
        preflight = verify_preflight()
        scores, checked = audit_scores(policy, arrays, preflight)
        comparison = audit_summary(policy, scores)
        result = {'schema': 'lipnet-step14-independent-audit-v1', **common,
            'comparison': comparison, 'candidate_scores': checked['scores'], 'independent_preflight': preflight,
            'preflight_source_snapshots_verified': True, 'preflight_precedes_scoring_contract': True,
            'source_rows_independently_checked': checked['source_rows_checked'],
            'main_candidate_pairs_independently_checked': checked['candidate_pairs_checked'],
            'auxiliary_greedy_pairs_independently_checked': checked['auxiliary_greedy_pairs_checked'],
            'total_ctc_pairs_independently_checked': checked['total_ctc_pairs_checked'],
            'infeasible_candidate_pairs_independently_checked': checked['infeasible_candidate_pairs_checked'],
            'independent_float64_ctc_max_abs_total_score_difference': checked['maximum_total_score_abs_difference'],
            'independent_float64_ctc_max_abs_saved_float32_mean_difference': checked['maximum_reference_ctc32_abs_difference'],
            'failed_source_rows': 0, 'source_failed_candidate_pairs': 0,
            'limitations': ['Known correct references are included among candidates; these ranks are oracle diagnostics.',
                           'Twenty-four remembered training candidates do not cover arbitrary Korean utterances.',
                           'Saved probabilities are audited without a new model forward or checkpoint deserialization.']}
    for reference in references + sources:
        verify(reference)
    verify_preservation(policy['preservation'])
    require('torch' not in sys.modules, 'Independent audit imported PyTorch')
    result.update(source_and_code_hashes_unchanged_before_and_after_audit=True, source_bindings_verified=len(references),
                  elapsed_seconds=time.perf_counter() - started)
    print(json.dumps({'status': 'passed', 'output': save_new(output, result)}, ensure_ascii=False))


if __name__ == '__main__':
    main()


