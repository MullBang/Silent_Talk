"""Describe every fixed diagnostic run without selecting a winning checkpoint."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
GROUPS = ('train24', 'hold8', 'train_old12', 'train_new12', 'hold_old4', 'hold_new4')
MILESTONES = (0, 900, 1800, 2700, 3600)


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
    """Use the same frozen text metric definitions without importing a model."""
    home = ROOT / 'experiments/olkavs_official'
    expected = {'evaluation_metrics.py': 'eb7fc30b04e08143d6b7f455f1ad7918681c42d395f4da58b1ced774a502b4b3',
                'diagnostic_utils.py': '3bac7121c89a8cfbae12b97dc4cfbdc8147ca44eccc2eb9d0af5c69f234d1ef9'}
    for name, digest in expected.items():
        require(bind(home / name)['sha256'] == digest, 'Frozen metric helper changed')
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    module = importlib.import_module('evaluation_metrics')
    require(Path(module.__file__).resolve() == home / 'evaluation_metrics.py', 'Wrong metric module')
    return module


def speaker_baseline(policy):
    """Output each known speaker's predeclared training anchor for internal hold8."""
    contract = policy['speaker_anchor_baseline']
    require(contract['kind'] == 'known_speaker_fixed_training_sentence' and
            contract['choice_uses_holdout_targets'] is False, 'Wrong baseline rule')
    selection = policy['selection']
    rows = {r['id']: r for r in selection['records']}
    require(contract['evaluation_ids'] == selection['holdout_ids'], 'Wrong baseline cohort')
    metrics = metrics_module()
    records = []
    for sid in contract['evaluation_ids']:
        held = rows[sid]
        anchor_id = contract['anchor_ids_by_speaker'][held['speaker_id']]
        require(anchor_id in selection['train_ids'] and anchor_id not in selection['holdout_ids'],
                'Baseline anchor is not training-only')
        anchor = rows[anchor_id]
        require(anchor['speaker_id'] == held['speaker_id'], 'Wrong anchor speaker')
        records.append({'id': sid, 'speaker_id': held['speaker_id'], 'anchor_id': anchor_id,
                        'status': 'ok', 'reference': held['text'], 'prediction': anchor['text'],
                        'metrics': metrics.counts(held['text'], anchor['text'])})
    groups = selection['evaluation_groups']
    return {'kind': contract['kind'], 'records': records,
            'groups': {name: metrics.aggregate([r for r in records if r['id'] in groups[name]])
                       for name in ('hold8', 'hold_old4', 'hold_new4')},
            'neural_inference_rows': 0, 'requires_known_metadata_speaker_id': True,
            'deployable_unseen_speaker_recognizer': False}


def comparisons(models):
    """Pair A/B by seed and retain all training/holdout strata and temporal controls."""
    require([(r['seed'], r['arm']) for r in models] == [(s, a) for s in SEEDS for a in ARMS],
            'Missing or reordered model pair')
    paired, means = [], {}
    for seed in SEEDS:
        a, b = [r for r in models if r['seed'] == seed]
        paired.append({'seed': seed, 'B_minus_A': {
            condition: {group: {
                'cer_percentage_points': b['final_conditions'][condition][group]['aggregate']['cer_percent'] -
                                         a['final_conditions'][condition][group]['aggregate']['cer_percent'],
                'exact_sentences': b['final_conditions'][condition][group]['aggregate']['exact_sentences'] -
                                   a['final_conditions'][condition][group]['aggregate']['exact_sentences']}
                for group in GROUPS} for condition in ('normal', 'static', 'shuffled')}})
    for arm in ARMS:
        selected = [r for r in models if r['arm'] == arm]
        means[arm] = {condition: {group: {
            key: statistics.mean(m['final_conditions'][condition][group]['aggregate'][key] for m in selected)
            for key in ('cer_percent', 'wer_percent', 'exact_sentences')}
            for group in GROUPS} for condition in ('normal', 'static', 'shuffled')}
    return paired, means


def validate_evaluation(value, seed, arm):
    """Reject incomplete trajectories and substitutions, including missing failures."""
    require(value['schema'] == 'lipnet-step12-multisentence-evaluation-v1'
            and value['status'] == 'completed' and value['seed'] == seed and value['arm'] == arm,
            'Wrong evaluation identity')
    require(len(value['ids']) == len(set(value['ids'])) == 32 and value['record_count'] == 224,
            'Wrong evaluation scope')
    groups = value['diagnostic_groups']
    require(len(groups['train24']) == 24 and len(groups['hold8']) == 8 and
            set(groups['train24']).isdisjoint(groups['hold8']) and
            value['ids'] == groups['train24'] + groups['hold8'], 'Training/holdout cohort mismatch')
    require([r['completed_updates'] for r in value['milestones']] == list(MILESTONES),
            'Missing/selected checkpoint')
    for milestone in value['milestones']:
        expected = {'normal', 'static', 'shuffled'} if milestone['completed_updates'] == 3600 else {'normal'}
        require(set(milestone['conditions']) == expected, 'Wrong temporal conditions')
        for condition in milestone['conditions'].values():
            require([r['id'] for r in condition['records']] == value['ids'], 'Missing/reordered inference')
            require(set(condition['groups']) == set(GROUPS), 'Wrong diagnostic groups')
            require(all(r['source_split'] == 'train' and r['split'] == r['diagnostic_role'] == ('train' if r['id'] in value['diagnostic_groups']['train24'] else 'internal_holdout') for r in condition['records']), 'Wrong training/holdout role')
    require(value['optimization_steps'] == 0 and not value['dev_inputs_loaded']
            and not value['original_held27_inputs_loaded'] and not value['test_inputs_loaded']
            and not value['service_weights_updated'], 'Diagnostic scope violation')


def summarize_training(value):
    """Account for real optimizer work and the saved last100 losses."""
    require(value['status'] == 'completed' and value['start_update'] == 0
            and value['completed_updates'] == 3600 and value['restored_from'] is None,
            'Expected a complete continuous3600 run')
    rows = value['updates']
    require(len(rows) == 3600 and [r['update'] for r in rows] == list(range(1, 3601)), 'Incomplete optimizer history')
    require(all(r['optimizer_updated'] and r['student_changed'] for r in rows), 'Skipped update')
    exposure = Counter(sid for row in rows for sid in row['ids'])
    require(len(exposure) == 24 and set(exposure.values()) == {300}, 'Unequal clip exposure')
    loss = {}
    for name, selected in (('first100', rows[:100]), ('last100', rows[-100:])):
        loss[name] = {key: statistics.mean(row[key] for row in selected)
                      for key in ('loss', 'ctc_loss', 'feature_loss', 'gradient_norm_before_clipping')}
        require(all(math.isfinite(x) for x in loss[name].values()), 'Nonfinite training summary')
    return {'updates': 3600, 'presentations': sum(exposure.values()), 'exposure_per_clip': dict(sorted(exposure.items())),
            'processed_frames': sum(sum(row['frames']) for row in rows),
            'elapsed_seconds': value['elapsed_seconds'],
            'max_cuda_allocated_bytes': max(row['cuda_peak_allocated_bytes'] for row in rows),
            'max_cuda_reserved_bytes': max(row['cuda_peak_reserved_bytes'] for row in rows),
            'gradient_norm_after_clipping_max': max(row['gradient_norm_after_clipping'] for row in rows),
            'loss': loss, 'initial_state_hashes': value['initial_state_hashes'],
            'final_state_hashes': value['final_state_hashes']}


def main():
    """Write an immutable all-runs numerical summary after inference."""
    policy_ref = bind(BASE / 'diagnostic_plan.json')
    policy = read(policy_ref['path'])
    models, refs = [], []
    for seed in SEEDS:
        for arm in ARMS:
            ep = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
            rp = BASE / 'runs' / f'seed{seed}' / arm / 'continuous' / 'run_result.json'
            evaluation, run = read(ep), read(rp)
            validate_evaluation(evaluation, seed, arm)
            require(evaluation['experiment_plan'] == run['experiment_plan'] == policy_ref, 'Policy mismatch')
            require(evaluation['initial_state_hashes_verified'] == run['initial_state_hashes'], 'Initial state mismatch')
            require(run['seed'] == seed and run['arm'] == arm, 'Wrong training run')
            require(evaluation['milestones'][-1]['checkpoint'] == run['checkpoint'], 'Wrong final model')
            final = evaluation['milestones'][-1]['conditions']
            normal_records = [r for r in final['normal']['records'] if r['diagnostic_role'] == 'train']
            exact = sum(r['status'] == 'ok' and r['metrics']['exact'] for r in normal_records)
            failures = sum(row['status'] != 'ok' for m in evaluation['milestones']
                           for condition in m['conditions'].values() for row in condition['records'])
            models.append({'seed': seed, 'arm': arm, 'evaluation': bind(ep), 'run_result': bind(rp),
                'failed_inferences': failures,
                'success_24_of_24_exact_at_fixed_final': exact == 24, 'final_normal_train_exact_sentences': exact,
                'normal_curve': [{'update': m['completed_updates'], 'groups': m['conditions']['normal']['groups']}
                                 for m in evaluation['milestones']],
                'final_conditions': {c: item['groups'] for c, item in final.items()},
                'training': summarize_training(run)})
            refs.extend([bind(ep), bind(rp), evaluation['log_posteriors_archive']])
    for seed in SEEDS:
        a, b = [r for r in models if r['seed'] == seed]
        require(all(a['training']['initial_state_hashes'][k] == b['training']['initial_state_hashes'][k]
                    for k in ('model', 'adapter', 'rng')), 'Unpaired initialization')
        require(a['training']['processed_frames'] == b['training']['processed_frames'], 'Unequal paired frame exposure')
    paired, means = comparisons(models)
    result = {'schema': 'lipnet-step12-multisentence-comparison-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': policy_ref,
              'generator': bind(Path(__file__)), 'artifacts': refs, 'models': models,
              'speaker_anchor_baseline': speaker_baseline(policy),
              'paired_seed_comparisons': paired, 'two_seed_arithmetic_means': means,
              'new_optimizer_updates': sum(m['training']['updates'] for m in models),
              'inference_rows': 896, 'training_elapsed_seconds_sum': sum(m['training']['elapsed_seconds'] for m in models),
              'processed_frames_sum': sum(m['training']['processed_frames'] for m in models),
              'fixed_final_success_runs': sum(m['success_24_of_24_exact_at_fixed_final'] for m in models),
              'failures': sum(m['failed_inferences'] for m in models),
              'unseen_speaker_generalization_estimate': False, 'service_weights_updated': False,
              'limitations': ['Train24 and same-speaker different-recording hold8; not unseen-speaker/webcam recognition.',
                             'Old/new strata are descriptive: source and speaker are not experimentally separable.',
                             'New16 labels retain provisional status; no new transcript approvals.',
                             'Failure at3600 updates cannot establish architectural impossibility; no adaptive extension.']}
    output = BASE / 'results' / 'comparison.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(output)))


if __name__ == '__main__':
    main()
