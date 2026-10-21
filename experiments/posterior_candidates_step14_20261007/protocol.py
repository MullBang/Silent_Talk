"""CPU-only saved-posterior candidate analysis contract; no model/pixel loading."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
from importlib.metadata import version as distribution_version
import json
import math
import os
from pathlib import Path
import platform
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / 'experiments/recording_probe_step13_20261007'
POLICY = BASE / 'diagnostic_plan.json'
FROZEN = BASE / 'plans/freeze_manifest.json'
STEP13_SUMMARY_SHA = 'bb32f1461b8a86f51c9a3e9121c9ce81fb48f62da4b41308a7b42b4c75f66c0d'
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
CONDITIONS, COHORTS = ('normal', 'static', 'shuffled'), ('same', 'different')
PURE_FILES = {
    ROOT / 'backend/models/tokenizer.py': 'c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714',
    ROOT / 'experiments/olkavs_official/diagnostic_utils.py': '3bac7121c89a8cfbae12b97dc4cfbdc8147ca44eccc2eb9d0af5c69f234d1ef9',
    ROOT / 'experiments/olkavs_official/evaluation_metrics.py': 'eb7fc30b04e08143d6b7f455f1ad7918681c42d395f4da58b1ced774a502b4b3',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == ref, 'Changed/nonlocal evidence: ' + str(path))
    return path


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalized_target(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def save_json(path, value):
    path = Path(path)
    require(not path.exists(), 'Immutable output already exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw)


def validate_tokens(tokens, allow_empty=False):
    require(isinstance(tokens, list) and (tokens or allow_empty) and
            all(type(token) is int and 1 <= token < 69 for token in tokens), 'Invalid canonical CTC tokens')
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def source_key(seed, arm, condition, cohort, sid):
    require(seed in SEEDS and arm in ARMS and condition in CONDITIONS and cohort in COHORTS, 'Unknown source dimension')
    return f'seed{seed}/{arm}/{condition}/{cohort}/{sid}'


def scoring_contract():
    return {'schema': 'lipnet-step14-candidate-scoring-contract-v1', 'device': 'cpu', 'calculation_dtype': 'float64',
            'input_dtype': 'float32 saved normalized log posteriors', 'blank_id': 0, 'classes': 69,
            'ctc_reduction': 'none', 'ctc_score': 'total_log_probability = -CTC negative_log_likelihood',
            'normalized_score': 'total_log_probability / target_token_count', 'higher_score_is_better': True,
            'candidate_count_per_source': 25, 'candidate_order': 'reference first, then training candidates sorted by stable candidate ID',
            'candidate_ranking_axes': ['total_log_probability', 'mean_log_probability_per_target_token'],
            'saved_log_probability_normalization_tolerance': 1e-5,
            'source_reference_ctc_replay_absolute_tolerance': 1e-5,
            'source_reference_ctc_replay_relative_tolerance': 1e-6,
            'rank_tolerance': 1e-8, 'reference_rank_rule': '1 + count(candidate_score > reference_score + 1e-8)',
            'top_tie_rule': 'all feasible candidates with score >= maximum_score - 1e-8; sort by stable candidate ID',
            'infeasible_rule': 'T < token_count + adjacent_repeated_token_count: status=infeasible and scores=null; means probability0/log_probability=-infinity',
            'best_training_margin': 'reference score minus maximum feasible training candidate score on the same axis',
            'speaker_preference': 'Report training-only best/ties among same-speaker3 versus other-speaker21 candidates separately from reference',
            'main_candidate_pairs': 3600, 'auxiliary_greedy_pairs': 144, 'total_score_pairs': 3744,
            'auxiliary_greedy': {'tokens': 'exact original source_record.collapsed_ids before Hangul composition',
                'excluded_from_candidate_ranks': True, 'score': 'CTC total_log_probability of greedy collapsed token sequence',
                'empty_sequence_rule': 'total_log_probability=sum(frame blank log probability); normalized score=null because token count0',
                'margin': 'reference total_log_probability minus greedy-sequence total_log_probability',
                'interpretation': 'diagnostic search discrepancy; not accuracy or proof a decoder change solves recognition'},
            'all_conditions_reported': list(CONDITIONS), 'reference_in_candidate_set_is_oracle': True,
            'rank_is_recognition_accuracy': False, 'deployable_candidate_recognizer': False,
            'new_neural_inference_rows': 0, 'optimizer_updates': 0, 'new_audio_or_video_read': False}


def validate_training_candidates(records, ids):
    require(len(ids) == len(set(ids)) == 24, 'Expected distinct train24')
    by_id = {row['id']: row for row in records}
    require(len(by_id) == len(records) and set(ids) <= set(by_id), 'Incomplete/duplicate training metadata')
    candidates = []
    for sid in ids:
        row = by_id[sid]
        require(row['original_split'] == 'train' and row['diagnostic_role'] == 'train' and
                row['original_role'] in ('old_train', 'new_provisional_train') and row['generation'] in ('old', 'new'), 'Nontraining candidate')
        validate_tokens(row['tokens'])
        require(normalized_target(row['text']), 'Empty candidate transcript')
        candidates.append({key: copy.deepcopy(row[key]) for key in ('id', 'speaker_id', 'source_stem', 'text', 'tokens', 'generation', 'original_split', 'original_role', 'diagnostic_role')})
    require(len({normalized_target(row['text']) for row in candidates}) == 24, 'Duplicate normalized training targets')
    require(set(Counter(row['speaker_id'] for row in candidates).values()) == {3}, 'Wrong training speaker balance')
    return candidates


def candidates_for(source, training_candidates):
    require(len(training_candidates) == len({row['id'] for row in training_candidates}) == 24, 'Expected train24 candidates')
    validate_tokens(source['tokens'])
    require(source['id'] not in {row['id'] for row in training_candidates} and
            normalized_target(source['text']) not in {normalized_target(row['text']) for row in training_candidates}, 'Reference leaked into train candidates')
    require(sum(row['speaker_id'] == source['speaker_id'] for row in training_candidates) == 3, 'Expected same-speaker3 training candidates')
    reference = {key: copy.deepcopy(source[key]) for key in ('id', 'text', 'tokens', 'speaker_id')}
    result = [{**reference, 'kind': 'reference', 'same_speaker': True}]
    for row in sorted(training_candidates, key=lambda item: item['id']):
        validate_tokens(row['tokens'])
        result.append({**{key: copy.deepcopy(row[key]) for key in ('id', 'text', 'tokens', 'speaker_id')},
                       'kind': 'training', 'same_speaker': row['speaker_id'] == source['speaker_id']})
    return result


def validate_source_record(row, selected, condition, cohort):
    expected_role = 'same_recording_holdout' if cohort == 'same' else 'internal_holdout'
    require(row['id'] == selected['id'] and row['speaker_id'] == selected['speaker_id'] and
            row['source_stem'] == selected['source_stem'] and row['reference'] == selected['text'] and
            row['target_token_ids'] == selected['tokens'] and row['frames'] == selected['frames'], 'Source identity differs from fixed selection')
    require(row['status'] == 'ok' and row['error'] is None and row['condition'] == condition and
            row['split'] == row['diagnostic_role'] == expected_role and row['source_split'] == 'train', 'Wrong source role/status/condition')
    require(type(row['frames']) is int and row['frames'] >= validate_tokens(row['target_token_ids']), 'Infeasible reference source')
    validate_tokens(row['collapsed_ids'], allow_empty=True)
    require(len(row['raw_argmax_ids']) == row['frames'] and
            all(type(t) is int and 0 <= t < 69 for t in row['raw_argmax_ids']), 'Invalid greedy path')
    collapsed = []
    for index, token in enumerate(row['raw_argmax_ids']):
        if token and (index == 0 or token != row['raw_argmax_ids'][index - 1]):
            collapsed.append(token)
    require(collapsed == row['collapsed_ids'] and row['log_posteriors_key'], 'Greedy CTC collapse/posterior key changed')


def build_metadata():
    summary_ref = {'path': str(PRIOR / 'results/step13_summary.json'), 'sha256': STEP13_SUMMARY_SHA}
    summary = read(verify(summary_ref))
    require(summary['status'] == 'completed_matched_speaker_recording_probe' and summary['new_inference_failures'] == summary['reused_inference_failures'] == 0,
            'Step13 incomplete/failed')
    for ref in summary['artifacts'].values():
        verify(ref)
    policy_ref, comparison_ref = summary['artifacts']['diagnostic_plan.json'], summary['artifacts']['results/comparison.json']
    audit_ref = summary['artifacts']['results/independent_audit.json']
    previous, comparison, audit = read(policy_ref['path']), read(comparison_ref['path']), read(audit_ref['path'])
    require(audit['status'] == 'passed' and audit['experiment_plan'] == comparison['experiment_plan'] == policy_ref and
            audit['comparison'] == comparison_ref, 'Step13 audit/comparison mismatch')
    for ref in previous['execution_code']:
        verify(ref)
    step12_ref = previous['prior_step12_policy']
    step12 = read(verify(step12_ref))
    training = validate_training_candidates(step12['selection']['records'], step12['selection']['train_ids'])
    selection = previous['selection']
    require({key: len(value) for key, value in selection['groups'].items()} ==
            {'same6': 6, 'different6': 6, 'same_old4': 4, 'same_new2': 2, 'different_old4': 4, 'different_new2': 2}, 'Changed fixed groups')
    selected = {row['id']: row for row in selection['records']}
    sources, models, bindings = [], [], [summary_ref, policy_ref, comparison_ref, audit_ref, step12_ref]
    for seed in SEEDS:
        for arm in ARMS:
            item = next(row for row in comparison['models'] if row['seed'] == seed and row['arm'] == arm)
            model = next(row for row in previous['models'] if row['seed'] == seed and row['arm'] == arm)
            evaluation_ref = item['evaluation']
            evaluation = read(verify(evaluation_ref))
            require(evaluation['seed'] == seed and evaluation['arm'] == arm and evaluation['status'] == 'completed' and
                    evaluation['experiment_plan'] == policy_ref and evaluation['groups'] == selection['groups'] and
                    evaluation['checkpoint'] == model['checkpoint'] == item['checkpoint'], 'Wrong model/evaluation binding')
            require(evaluation['prior_evaluation'] == model['prior_evaluation'] and evaluation['prior_posteriors'] == model['prior_posteriors'], 'Wrong reused source binding')
            original = read(verify(model['prior_evaluation']))
            original_final = next(row for row in original['milestones'] if row['completed_updates'] == 3600)
            checkpoint = model['checkpoint']
            for ref in ({key: checkpoint[key] for key in ('path', 'sha256')},
                        {'path': checkpoint['sidecar'], 'sha256': checkpoint['sidecar_sha256']},
                        evaluation['log_posteriors_archive'], model['prior_posteriors']):
                verify(ref)
                bindings.append(ref)
            models.append({'seed': seed, 'arm': arm, 'step13_evaluation': evaluation_ref,
                           'step13_posteriors': evaluation['log_posteriors_archive'], 'step12_evaluation': model['prior_evaluation'],
                           'step12_posteriors': model['prior_posteriors'], 'checkpoint': checkpoint})
            bindings.extend([evaluation_ref, model['prior_evaluation']])
            for condition in CONDITIONS:
                for cohort in COHORTS:
                    group = cohort + '6'
                    rows = evaluation['conditions'][condition][cohort]['records']
                    require([row['id'] for row in rows] == selection['groups'][group], 'Changed source row order/count')
                    if cohort == 'different':
                        old = {row['id']: row for row in original_final['conditions'][condition]['records']}
                        require(all(row == old[row['id']] for row in rows), 'Reused original row altered')
                    posterior_ref = evaluation['log_posteriors_archive'] if cohort == 'same' else evaluation['prior_posteriors']
                    for row in rows:
                        chosen = selected[row['id']]
                        validate_source_record(row, chosen, condition, cohort)
                        source = {'key': source_key(seed, arm, condition, cohort, row['id']), 'seed': seed, 'arm': arm,
                                  'condition': condition, 'cohort': cohort, 'id': row['id'], 'speaker_id': row['speaker_id'],
                                  'source_stem': row['source_stem'], 'text': row['reference'], 'tokens': row['target_token_ids'],
                                  'frames': row['frames'], 'generation': chosen['generation'], 'original_split': 'train',
                                  'original_role': chosen['original_role'], 'source_record': copy.deepcopy(row),
                                  'source_record_sha256': canonical_sha(row), 'source_evaluation': evaluation_ref,
                                  'posterior_archive': posterior_ref, 'posterior_key': row['log_posteriors_key']}
                        candidates_for(source, training)
                        sources.append(source)
    require(len(sources) == len({row['key'] for row in sources}) == 144 and
            Counter((row['seed'], row['arm'], row['condition'], row['cohort']) for row in sources) ==
            Counter({(seed, arm, condition, cohort): 6 for seed in SEEDS for arm in ARMS for condition in CONDITIONS for cohort in COHORTS}), 'Incomplete source144 coverage')
    return {'sources': sources, 'training_candidates': training, 'groups': copy.deepcopy(selection['groups']),
            'pairs': copy.deepcopy(selection['pairs']), 'models': models, 'prior_step13_summary': summary_ref,
            'prior_step13_policy': policy_ref, 'prior_step13_comparison': comparison_ref, 'prior_step13_audit': audit_ref,
            'prior_step12_policy': step12_ref, 'source_bindings': list({ref['path']: ref for ref in bindings}.values()),
            'scope': {'source_rows': 144, 'unique_evaluation_clips': 12, 'training_candidates': 24,
                      'same_recording_source_rows': 72, 'different_recording_source_rows': 72,
                      'all_original_source_splits': 'train', 'historically_seen_by_older_models_possible': True,
                      'new_human_approvals_added': 0, 'untouched_test_set': False, 'unseen_speakers': False}}


source_metadata = build_metadata


def validate_log_probs(array, source):
    require(isinstance(array, np.ndarray) and array.dtype == np.float32 and array.shape == (source['frames'], 69) and
            np.isfinite(array).all(), 'Invalid saved log posterior shape/dtype/finiteness')
    values = array.astype(np.float64)
    maximum = values.max(axis=1)
    log_mass = maximum + np.log(np.exp(values - maximum[:, None]).sum(axis=1))
    require(np.max(np.abs(log_mass)) <= 1e-5 and np.max(array) <= 1e-5, 'Unnormalized saved log posterior')
    require(array.argmax(axis=1).tolist() == source['source_record']['raw_argmax_ids'], 'Saved posterior greedy path differs')


def load_sources(policy=None):
    metadata = build_metadata()
    if policy is not None:
        require(all(policy[key] == value for key, value in metadata.items()), 'Frozen source metadata changed')
    grouped = {}
    for row in metadata['sources']:
        grouped.setdefault(row['posterior_archive']['path'], []).append(row)
    loaded = {}
    for path, rows in grouped.items():
        require(len({row['posterior_key'] for row in rows}) == len(rows), 'Reused posterior key within source scope')
        verify(rows[0]['posterior_archive'])
        with np.load(path, allow_pickle=False) as archive:
            for row in rows:
                require(row['posterior_key'] in archive.files, 'Saved posterior key missing')
                array = archive[row['posterior_key']]
                validate_log_probs(array, row)
                array.setflags(write=False)
                loaded[row['key']] = {**row, 'log_probs': array}
    sources = [loaded[row['key']] for row in metadata['sources']]
    return {'sources': sources, 'rows': sources, 'training_candidates': metadata['training_candidates'], 'metadata': metadata,
            'input_bindings': [rows[0]['posterior_archive'] for rows in grouped.values()],
            'counts': {'saved_posterior_archives_opened': len(grouped), 'saved_posterior_arrays_decoded': len(sources),
                       'saved_posterior_frames_decoded': sum(row['frames'] for row in sources),
                       'student_input_npz_opened': 0, 'teacher_feature_npz_opened': 0, 'raw_audio_or_video_opened': 0,
                       'checkpoint_tensors_deserialized': 0, 'checkpoint_files_hashed': 4,
                       'new_neural_inference_rows': 0, 'optimizer_updates': 0}}


def preservation_contract(metadata):
    previous = read(metadata['prior_step13_policy']['path'])
    value = copy.deepcopy(previous['preservation'])
    value['artifacts'].extend(metadata['source_bindings'])
    value['artifacts'].extend(previous['execution_code'])
    value['artifacts'] = list({ref['path']: ref for ref in value['artifacts']}.values())
    return value


def verify_preservation(value):
    for ref in value['artifacts']:
        verify(ref)
    require(sorted(str(path.resolve()) for path in Path(value['weights_directory']).glob('*') if path.is_file()) ==
            sorted(value['weights_inventory']), 'Service weight inventory changed')
    require(all(not Path(path).exists() for path in value['absent_paths']), 'Previously absent service weight created')


def runtime():
    return {'python': platform.python_version(), 'numpy': np.__version__, 'torch': distribution_version('torch'),
            'torch_version_source': 'installed distribution metadata without importing torch', 'device': 'cpu', 'ctc_dtype': 'float64',
            'torch_cpu_threads': 4,
            'gpu_query_or_model_required': False}


def code_bindings(extra_code_paths=()):
    paths = {BASE / name for name in ('protocol.py', 'test_protocol.py', 'scoring.py', 'test_scoring.py')}
    for path, expected_sha in PURE_FILES.items():
        require(sha(path) == expected_sha, 'Frozen pure scoring helper changed: ' + str(path))
        paths.add(path)
    paths.update(Path(path).resolve() for path in extra_code_paths)
    require(all(path.is_relative_to(ROOT) and path.suffix in {'.py', '.txt'} for path in paths), 'Nonlocal execution code')
    return [bind(path) for path in sorted(paths, key=str)]


def freeze(extra_code_paths=()):
    require(not POLICY.exists() and not FROZEN.exists(), 'Step14 contract already exists')
    loaded = load_sources()
    metadata = loaded['metadata']
    policy = {'schema': 'lipnet-step14-posterior-candidates-protocol-v1', 'status': 'frozen_before_scoring',
              'created_utc': datetime.now(timezone.utc).isoformat(), **metadata,
              'scoring_contract': scoring_contract(), 'execution_code': code_bindings(extra_code_paths),
              'preservation': preservation_contract(metadata), 'runtime': runtime(), 'loading_scope': loaded['counts'],
              'main_candidate_pairs_planned': 3600, 'auxiliary_greedy_pairs_planned': 144, 'total_score_pairs_planned': 3744,
              'new_neural_inference_rows': 0, 'new_optimizer_updates': 0, 'service_weights_updated': False,
              'ranking_is_accuracy': False, 'reference_candidate_is_oracle': True, 'new_human_approvals_added': 0}
    verify_preservation(policy['preservation'])
    save_json(POLICY, policy)
    manifest = {'schema': 'lipnet-step14-freeze-manifest-v1', 'status': 'frozen', 'experiment_plan': bind(POLICY),
                'main_candidate_pairs_executed': 0, 'auxiliary_greedy_pairs_executed': 0, 'new_neural_inference_rows': 0}
    save_json(FROZEN, manifest)
    return manifest


def verify_frozen():
    manifest = read(FROZEN)
    require(manifest['schema'] == 'lipnet-step14-freeze-manifest-v1' and manifest['status'] == 'frozen' and
            manifest['experiment_plan'] == bind(POLICY), 'Freeze manifest changed')
    policy = read(verify(manifest['experiment_plan']))
    metadata = build_metadata()
    require(all(policy[key] == value for key, value in metadata.items()), 'Frozen source metadata changed')
    require(policy['scoring_contract'] == scoring_contract() and policy['runtime'] == runtime(), 'Scoring contract/runtime changed')
    require(policy['main_candidate_pairs_planned'] == 3600 and policy['auxiliary_greedy_pairs_planned'] == 144 and
            policy['total_score_pairs_planned'] == 3744 and policy['new_neural_inference_rows'] == policy['new_optimizer_updates'] == 0,
            'Frozen score budget changed')
    for ref in policy['execution_code']:
        verify(ref)
    verify_preservation(policy['preservation'])
    return policy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('metadata', 'freeze', 'verify'))
    parser.add_argument('--extra-code', action='append', default=[])
    args = parser.parse_args()
    value = build_metadata() if args.command == 'metadata' else freeze(args.extra_code) if args.command == 'freeze' else verify_frozen()
    print(json.dumps(value, ensure_ascii=False, indent=2))
