"""Summarize matched-speaker recording probes without pooling the two cohorts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
CONDITIONS = ('normal', 'static', 'shuffled')
GROUPS = ('same6', 'same_old4', 'same_new2', 'different6', 'different_old4', 'different_new2')
PRIOR_SUMMARY_SHA = '9cb8b269549d8e94aba75b5ce745b61ec3a0f0afbb68af4c27564691f7074869'


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


def metrics_module():
    path = ROOT / 'experiments/multisentence_step12_20261007/summarize.py'
    require(bind(path)['sha256'] == PRIOR_SUMMARY_SHA, 'Frozen metric loader changed')
    spec = importlib.util.spec_from_file_location('step13_metrics_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics_module()


def validate_evaluation(value, seed, arm, selection):
    require(value['schema'] == 'lipnet-step13-recording-probe-evaluation-v1' and value['status'] == 'completed'
            and value['seed'] == seed and value['arm'] == arm, 'Wrong evaluation identity')
    require(value['new_inference_rows'] == value['reused_inference_rows'] == 18
            and value['record_count'] == 36, 'New/reused inference accounting changed')
    require(value['optimization_steps'] == 0 and not value['audio_used'] and not value['service_weights_updated'],
            'Evaluation changed training/service scope')
    require(set(value['conditions']) == set(CONDITIONS), 'Missing or selected temporal condition')
    metadata = {row['id']: row for row in selection['records']}
    for condition in CONDITIONS:
        item = value['conditions'][condition]
        require(set(item) == {'same', 'different'}, 'Mixed recording cohorts')
        for kind in ('same', 'different'):
            ids = selection[kind + '_ids']
            branch = item[kind]
            require(len(ids) == len(set(ids)) == 6 and [r['id'] for r in branch['records']] == ids,
                    'Missing, duplicated or reordered matched-speaker row')
            wanted = [g for g in GROUPS if g.startswith(kind)]
            require(set(branch['groups']) == set(wanted), 'Wrong recording/source strata')
            for row in branch['records']:
                source = metadata[row['id']]
                require(row['speaker_id'] == source['speaker_id'] and row['reference'] == source['text']
                        and row['condition'] == condition and row['source_split'] == 'train', 'Changed row provenance')
            for group in wanted:
                require(branch['groups'][group]['aggregate']['clips'] == len(selection['groups'][group]),
                        'Wrong metric denominator')
    require(set(selection['same_ids']).isdisjoint(selection['different_ids']), 'Cohorts overlap')


def baseline(policy):
    contract, selection = policy['speaker_anchor_baseline'], policy['selection']
    require(contract['choice_uses_evaluation_targets'] is False and contract['baseline_rows'] == 12
            and contract['neural_inference_rows'] == 0, 'Changed fixed training-sentence baseline')
    metrics, records = metrics_module(), []
    rows = {row['id']: row for row in selection['records']}
    for kind in ('same', 'different'):
        for sid in selection[kind + '_ids']:
            row = rows[sid]
            anchor = contract['anchors'][row['speaker_id']]
            require(anchor['id'] == contract['anchor_ids_by_speaker'][row['speaker_id']]
                    and anchor['id'] not in rows, 'Baseline anchor is not a separate training sentence')
            records.append({'id': sid, 'speaker_id': row['speaker_id'], 'cohort': kind,
                            'anchor_id': anchor['id'], 'reference': row['text'], 'prediction': anchor['text'],
                            'status': 'ok', 'metrics': metrics.counts(row['text'], anchor['text'])})
    return {'kind': 'known_speaker_fixed_training_sentence', 'records': records,
            'groups': {g: metrics.aggregate([r for r in records if r['id'] in selection['groups'][g]]) for g in GROUPS},
            'baseline_rows': 12, 'neural_inference_rows': 0, 'requires_known_metadata_speaker_id': True,
            'deployable_unseen_speaker_recognizer': False}


def model_summary(value, reference, selection):
    validate_evaluation(value, value['seed'], value['arm'], selection)
    conditions = {c: {**value['conditions'][c]['same']['groups'], **value['conditions'][c]['different']['groups']}
                  for c in CONDITIONS}
    paired, changes = {}, {}
    for condition in CONDITIONS:
        source = value['conditions'][condition]
        same = {r['id']: r for r in source['same']['records']}
        different = {r['id']: r for r in source['different']['records']}
        paired[condition] = []
        for pair in selection['pairs']:
            a, b = same[pair['same_id']], different[pair['different_id']]
            require(a['speaker_id'] == b['speaker_id'] == pair['speaker_id'], 'Unmatched speaker comparison')
            sa, sb = a['metrics'], b['metrics']
            # Per-clip CER is descriptive. Pooled corpus CER is in final_conditions.
            cer_a = 100 * sa['character_edits'] / sa['reference_characters']
            cer_b = 100 * sb['character_edits'] / sb['reference_characters']
            paired[condition].append({'speaker_id': pair['speaker_id'], 'same_id': a['id'], 'different_id': b['id'],
                'same_cer_percent': cer_a, 'different_cer_percent': cer_b,
                'different_minus_same_cer_percentage_points': cer_b - cer_a,
                'same_exact': sa['exact'], 'different_exact': sb['exact']})
        changes[condition] = {suffix: {
            'cer_percentage_points': conditions[condition]['different' + suffix]['aggregate']['cer_percent'] -
                                     conditions[condition]['same' + suffix]['aggregate']['cer_percent'],
            'exact_sentences': conditions[condition]['different' + suffix]['aggregate']['exact_sentences'] -
                               conditions[condition]['same' + suffix]['aggregate']['exact_sentences']}
            for suffix in ('6', '_old4', '_new2')}
    failures = {kind: sum(r['status'] != 'ok' for c in CONDITIONS for r in value['conditions'][c][kind]['records'])
                for kind in ('same', 'different')}
    return {'seed': value['seed'], 'arm': value['arm'], 'evaluation': reference, 'checkpoint': value['checkpoint'],
            'final_conditions': conditions, 'different_minus_same': changes, 'speaker_pairs': paired,
            'new_inference_rows': 18, 'reused_inference_rows': 18,
            'failed_new_inferences': failures['same'], 'failed_reused_inferences': failures['different']}


def comparisons(models):
    require([(m['seed'], m['arm']) for m in models] == [(s, a) for s in SEEDS for a in ARMS], 'Missing model pair')
    paired = []
    for seed in SEEDS:
        a, b = [m for m in models if m['seed'] == seed]
        paired.append({'seed': seed, 'B_minus_A': {c: {g: {
            'cer_percentage_points': b['final_conditions'][c][g]['aggregate']['cer_percent'] -
                                     a['final_conditions'][c][g]['aggregate']['cer_percent'],
            'exact_sentences': b['final_conditions'][c][g]['aggregate']['exact_sentences'] -
                               a['final_conditions'][c][g]['aggregate']['exact_sentences']}
            for g in GROUPS} for c in CONDITIONS}})
    means = {arm: {c: {g: {key: statistics.mean(m['final_conditions'][c][g]['aggregate'][key]
                                              for m in models if m['arm'] == arm)
                           for key in ('cer_percent', 'wer_percent', 'exact_sentences')}
                      for g in GROUPS} for c in CONDITIONS} for arm in ARMS}
    return paired, means


def main():
    import protocol
    policy = protocol.verify_frozen()
    policy_ref = bind(BASE / 'diagnostic_plan.json')
    models, artifacts = [], []
    for source in policy['models']:
        seed, arm = source['seed'], source['arm']
        path = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
        value = read(path)
        validate_evaluation(value, seed, arm, policy['selection'])
        require(value['experiment_plan'] == policy_ref and value['checkpoint'] == source['checkpoint'],
                'Changed evaluation experiment/checkpoint')
        require(value['prior_evaluation'] == source['prior_evaluation'] and value['prior_posteriors'] == source['prior_posteriors'],
                'Changed reused evaluation provenance')
        models.append(model_summary(value, bind(path), policy['selection']))
        artifacts.extend([bind(path), value['log_posteriors_archive'], value['prior_evaluation'], value['prior_posteriors']])
    paired, means = comparisons(models)
    result = {'schema': 'lipnet-step13-recording-probe-comparison-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': policy_ref,
              'generator': bind(Path(__file__)), 'artifacts': artifacts, 'models': models,
              'speaker_anchor_baseline': baseline(policy), 'paired_seed_comparisons': paired,
              'two_seed_arithmetic_means': means, 'new_inference_rows': 72, 'reused_inference_rows': 72,
              'baseline_rows': 12, 'new_optimizer_updates': 0,
              'failed_new_inferences': sum(m['failed_new_inferences'] for m in models),
              'failed_reused_inferences': sum(m['failed_reused_inferences'] for m in models),
              'matched_speakers': [p['speaker_id'] for p in policy['selection']['pairs']],
              'service_weights_updated': False, 'new_human_approvals_added': 0,
              'unseen_speaker_generalization_estimate': False, 'untouched_test_estimate': False,
              'pure_recording_environment_causal_effect': False,
              'limitations': ['Six matched known speakers only; old4/new2 cohorts remain descriptive.',
                             'Same/different recordings also contain different utterances, durations and difficulty.',
                             'Provisional new-source labels retain their prior approval status.',
                             'Prior experimental models may have used these historical training candidates.',
                             'Temporal controls cannot establish correct semantic lipreading by themselves.']}
    for ref in artifacts:
        require(bind(ref['path']) == ref, 'Evidence changed during summary')
    protocol.verify_frozen()
    path = BASE / 'results/comparison.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(path)))


if __name__ == '__main__':
    main()
