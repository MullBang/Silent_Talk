"""Run the predeclared multiple-utterance diagnostic without modifying prior artifacts."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import protocol

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
runner, core = protocol.runner, protocol.core
require, bind, read = runner.require, protocol.bind, protocol.read
SEEDS, ARMS = (42, 43), ('A_ctc', 'B_feature')


def run_path(seed: int, arm: str) -> Path:
    """Return the isolated output location for a permitted diagnostic run."""
    require(seed in SEEDS and arm in ARMS, 'Unknown run')
    return BASE / 'runs' / f'seed{seed}' / arm / 'continuous'


def validate_loaded(plan: dict, loaded: dict, arm: str) -> dict:
    """Check exact train pixels, tokens, features and complete scheduled scope."""
    require(not set(loaded['train_clips']) & set(loaded['holdout_ids']), 'Internal holdout entered optimizer scope')
    actual = runner.validate_plan(plan, loaded['train_clips'], arm)
    require(plan['training_data'] == {k: actual[k] for k in ('data_sha256', 'clips')},
            'Diagnostic training bytes changed')
    return actual


def execute(seed: int, arm: str) -> dict:
    """Train a fresh model for the complete fixed budget, saving every milestone."""
    started = time.perf_counter()
    require(shutil.disk_usage(BASE).free >= 5 * 1024**3, 'Less than5GiB disk reserve')
    policy = protocol.verify_frozen()
    plan = read(protocol.plan_path(seed))
    protocol.check_plan(plan, policy)
    loaded = protocol.load_clips()
    validate_loaded(plan, loaded, arm)
    expected = plan['initial_state_hashes_expected'][arm]
    objects = core.make_objects(seed, arm, 'cuda:0')
    require(runner.state_hashes(core.runtime_state(*objects)) == expected,
            'Fresh seeded initialization changed')
    del objects
    gc.collect()
    runner.torch.cuda.empty_cache()
    output = run_path(seed, arm)
    result = runner.run_segment(plan, loaded['train_clips'], arm, output, device='cuda:0')
    require(result['initial_state_hashes'] == expected, 'Run initialization mismatch')
    require(result['start_update'] == 0 and result['completed_updates'] == 3600
            and result['restored_from'] is None, 'Incomplete fresh diagnostic run')
    result.update(status='completed', seed=seed, plan_file=bind(protocol.plan_path(seed)),
                  experiment_plan=bind(BASE / 'diagnostic_plan.json'),
                  elapsed_seconds=time.perf_counter() - started,
                  dev_inputs_loaded=False, original_held27_inputs_loaded=False, test_inputs_loaded=False,
                  audio_used=False, teacher_model_loaded=False, service_weights_updated=False,
                  internal_holdout_inputs_validated=True,
                  internal_holdout_teacher_features_validated_not_used=True,
                  internal_holdout_optimization_presentations=0,
                  scope='Train24 and recording-disjoint internalhold8; original source split is train, not unseen-speaker test.')
    runner._save_json(output / 'run_result.json', result)
    return bind(output / 'run_result.json')


def evaluate_one(seed: int, arm: str) -> dict:
    """Evaluate saved milestones after optimization has finished."""
    import evaluate
    policy = protocol.verify_frozen()
    plan = read(protocol.plan_path(seed))
    protocol.check_plan(plan, policy)
    result_path = run_path(seed, arm) / 'run_result.json'
    result = read(result_path)
    require(result['status'] == 'completed' and result['seed'] == seed and result['arm'] == arm,
            'Wrong diagnostic result')
    require(result['plan_file'] == bind(protocol.plan_path(seed)) and
            result['experiment_plan'] == bind(BASE / 'diagnostic_plan.json'), 'Changed run binding')
    loaded = protocol.load_clips()
    validate_loaded(plan, loaded, arm)
    output = BASE / 'results' / f'evaluation_seed{seed}_{arm}.json'
    return evaluate.evaluate_run(
        plan, loaded, arm, result['checkpoints'],
        plan['initial_state_hashes_expected'][arm], output,
        device='cuda:0', experiment_plan_ref=bind(BASE / 'diagnostic_plan.json'))


def run_all() -> dict:
    """Run four bounded jobs serially so only one job uses the GPU at a time."""
    protocol.verify_frozen()
    logs = BASE / 'logs'
    logs.mkdir(exist_ok=True)
    completed = []
    for seed in SEEDS:
        for arm in ARMS:
            path = run_path(seed, arm) / 'run_result.json'
            require(not path.exists(), 'Use completed records; do not silently rerun training')
            with (logs / f'seed{seed}_{arm}.log').open('x', encoding='utf-8') as log:
                command = [sys.executable, str(Path(__file__).resolve()), 'run', '--seed', str(seed), '--arm', arm]
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, check=False)
            require(process.returncode == 0, f'Run failed: seed{seed}/{arm}; inspect preserved log')
            completed.append(bind(path))
            print(json.dumps({'completed': f'seed{seed}/{arm}', 'run_result': completed[-1]}), flush=True)
    result = {'schema': 'lipnet-step12-training-complete-v1', 'status': 'completed',
              'experiment_plan': bind(BASE / 'diagnostic_plan.json'), 'runs': completed,
              'new_optimizer_updates': 14400, 'service_weights_updated': False}
    runner._save_json(BASE / 'results' / 'training_complete.json', result)
    return result


def main() -> None:
    """Expose separate freeze, training and evaluation operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'run-all', 'evaluate', 'evaluate-all'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.command == 'freeze':
        value = protocol.freeze(extra_code_paths=[BASE / 'test_training.py', BASE / 'test_evaluate.py'])
    elif args.command == 'run':
        value = execute(args.seed, args.arm)
    elif args.command == 'run-all':
        value = run_all()
    elif args.command == 'evaluate':
        value = evaluate_one(args.seed, args.arm)
    else:
        value = [evaluate_one(seed, arm) for seed in SEEDS for arm in ARMS]
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
