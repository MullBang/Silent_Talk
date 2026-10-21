"""Describe every fixed diagnostic run without selecting a winning checkpoint."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')
GROUPS = ('all8', 'old4', 'new4')
MILESTONES = (0, 300, 600, 900, 1200)


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


def validate_evaluation(value, seed, arm):
    """Reject incomplete trajectories and substitutions, including missing failures."""
    require(value['schema'] == 'lipnet-step11-train-memorization-evaluation-v1'
            and value['status'] == 'completed' and value['seed'] == seed and value['arm'] == arm,
            'Wrong evaluation identity')
    require(len(value['ids']) == len(set(value['ids'])) == 8 and value['record_count'] == 56,
            'Wrong evaluation scope')
    require([r['completed_updates'] for r in value['milestones']] == list(MILESTONES),
            'Missing/selected checkpoint')
    for milestone in value['milestones']:
        expected = {'normal', 'static', 'shuffled'} if milestone['completed_updates'] == 1200 else {'normal'}
        require(set(milestone['conditions']) == expected, 'Wrong temporal conditions')
        for condition in milestone['conditions'].values():
            require([r['id'] for r in condition['records']] == value['ids'], 'Missing/reordered inference')
            require(set(condition['groups']) == set(GROUPS), 'Wrong diagnostic groups')
            require(all(r['split'] == 'train' for r in condition['records']), 'Nontraining result')
    require(value['optimization_steps'] == 0 and not value['dev_inputs_loaded']
            and not value['held_inputs_loaded'] and not value['test_inputs_loaded']
            and not value['service_weights_updated'], 'Diagnostic scope violation')


def summarize_training(value):
    """Account for real optimizer work and the saved last100 losses."""
    require(value['status'] == 'completed' and value['start_update'] == 0
            and value['completed_updates'] == 1200 and value['restored_from'] is None,
            'Expected a complete continuous1200 run')
    rows = value['updates']
    require(len(rows) == 1200 and [r['update'] for r in rows] == list(range(1, 1201)), 'Incomplete optimizer history')
    require(all(r['optimizer_updated'] and r['student_changed'] for r in rows), 'Skipped update')
    exposure = Counter(sid for row in rows for sid in row['ids'])
    require(len(exposure) == 8 and set(exposure.values()) == {300}, 'Unequal clip exposure')
    loss = {}
    for name, selected in (('first100', rows[:100]), ('last100', rows[-100:])):
        loss[name] = {key: statistics.mean(row[key] for row in selected)
                      for key in ('loss', 'ctc_loss', 'feature_loss', 'gradient_norm_before_clipping')}
        require(all(math.isfinite(x) for x in loss[name].values()), 'Nonfinite training summary')
    return {'updates': 1200, 'presentations': sum(exposure.values()), 'exposure_per_clip': dict(sorted(exposure.items())),
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
            normal_records = final['normal']['records']
            exact = sum(r['status'] == 'ok' and r['metrics']['exact'] for r in normal_records)
            failures = sum(row['status'] != 'ok' for m in evaluation['milestones']
                           for condition in m['conditions'].values() for row in condition['records'])
            models.append({'seed': seed, 'arm': arm, 'evaluation': bind(ep), 'run_result': bind(rp),
                'failed_inferences': failures,
                'success_8_of_8_exact_at_fixed_final': exact == 8, 'final_normal_exact_sentences': exact,
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
    result = {'schema': 'lipnet-step11-memorization-comparison-v1', 'status': 'completed',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': policy_ref,
              'generator': bind(Path(__file__)), 'artifacts': refs, 'models': models,
              'new_optimizer_updates': sum(m['training']['updates'] for m in models),
              'inference_rows': 224, 'training_elapsed_seconds_sum': sum(m['training']['elapsed_seconds'] for m in models),
              'processed_frames_sum': sum(m['training']['processed_frames'] for m in models),
              'fixed_final_success_runs': sum(m['success_8_of_8_exact_at_fixed_final'] for m in models),
              'failures': sum(m['failed_inferences'] for m in models),
              'generalization_estimate': False, 'service_weights_updated': False,
              'limitations': ['Training-only8 clips; memorization cannot establish unseen-speaker/webcam recognition.',
                             'Old/new strata are descriptive: source and speaker are not experimentally separable.',
                             'New4 labels retain provisional status; no new transcript approvals.',
                             'Failure at1200 updates cannot establish architectural impossibility; no adaptive extension.']}
    output = BASE / 'results' / 'comparison.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(output)))


if __name__ == '__main__':
    main()
