"""Serial Step16 main jobs and separately counted six-update GPU resume smoke."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import engine
import protocol

BASE, ROOT = protocol.BASE, protocol.ROOT
runner, core, require = engine.runner, engine.core, engine.require
bind, read = protocol.bind, protocol.read
SEEDS, COHORTS = (42, 43), ('train24', 'train72')
SMOKE_RESULT = BASE / 'results/smoke_preflight.json'


def utc_now():
    return datetime.now(timezone.utc)


def elapsed_since(value):
    return (utc_now() - datetime.fromisoformat(value)).total_seconds()


def run_path(seed, cohort):
    require(seed in SEEDS and cohort in COHORTS, 'Unknown run')
    return protocol.run_path(seed, cohort)


def result_path(seed, cohort):
    return run_path(seed, cohort) / 'run_result.json'


def validate_loaded(plan, loaded):
    clips = {sid: loaded['train'][sid] for sid in plan['ids']}
    require(not set(clips).intersection(plan['dev_ids'] + plan['probe_ids']) and
            not loaded.get('dev') and not loaded.get('probe'), 'Training loaded evaluation pixels')
    engine.validate_plan(plan, clips)
    return clips


def _job_clock():
    path = BASE / 'results/main_jobs_clock.json'
    if not path.exists():
        runner._save_json(path, {'schema': 'step16-main-job-clock-v1', 'created_utc': utc_now().isoformat(),
                                'time_budget_seconds': 9000, 'experiment_plan': bind(protocol.POLICY)})
    value = read(path)
    require(value['time_budget_seconds'] == 9000 and value['experiment_plan'] == bind(protocol.POLICY), 'Job clock changed')
    return value


def _run_clock(seed, cohort, resume):
    path = run_path(seed, cohort).parent / 'run_clock.json'
    if not resume:
        require(not path.exists(), 'Refusing to restart a prior run')
        runner._save_json(path, {'schema': 'step16-run-clock-v1', 'created_utc': utc_now().isoformat(),
                                'seed': seed, 'cohort': cohort, 'time_budget_seconds': 2100,
                                'plan': bind(protocol.plan_path(seed, cohort))})
    value = read(path)
    require(value['seed'] == seed and value['cohort'] == cohort and value['time_budget_seconds'] == 2100 and
            value['plan'] == bind(protocol.plan_path(seed, cohort)), 'Run clock changed')
    return value


def _saved_refs(directory):
    refs = []
    for sidecar in sorted(Path(directory).glob('update_*.pt.json')):
        info = read(sidecar)
        refs.append({**info, 'sidecar': str(sidecar.resolve()), 'sidecar_sha256': engine.file_sha(sidecar)})
    return refs


def _resume_origin(directory, plan):
    result = Path(directory) / 'segment_result.json'
    if result.exists():
        prior = read(result)
        require(prior['status'] in ('failed', 'interrupted'), 'Only failed/interrupted jobs may resume; time budgets cannot extend')
        return prior['checkpoint'], prior
    contract = read(Path(directory) / 'segment_contract.json')
    require(contract['identity'] == engine.make_identity(plan), 'Interrupted segment contract changed')
    refs = _saved_refs(directory)
    require(refs, 'No complete checkpoint exists for interrupted run')
    latest = max(refs, key=lambda value: value['completed_updates'])
    progress_path = Path(directory) / 'progress.jsonl'
    rows = []
    if progress_path.exists():
        for line in progress_path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            rows.append(row)
    require(all(row['update'] == index + 1 for index, row in enumerate(rows)), 'Interrupted progress ledger is not contiguous')
    return latest, {'status': 'abruptly_interrupted', 'completed_updates': latest['completed_updates'],
                    'recorded_updates_after_last_checkpoint_discarded': max(0, len(rows) - latest['completed_updates']),
                    'unrecorded_partial_update_attempt_possible': True, 'progress_records': len(rows)}


def execute(seed, cohort, resume=False):
    require(shutil.disk_usage(BASE).free >= 5 * 1024**3, 'Less than5GiB reserve')
    policy = protocol.verify_frozen()
    preflight_path = BASE / 'results/independent_preflight.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'passed' and preflight['experiment_plan'] == bind(protocol.POLICY), 'Passed frozen-plan preflight required')
    smoke = read(SMOKE_RESULT)
    require(smoke['status'] == 'passed' and smoke['experiment_plan'] == bind(protocol.POLICY), 'GPU resume smoke required')
    plan = protocol.run_plan(seed, cohort)
    require(plan['experiment_plan'] == bind(protocol.POLICY) and Path(plan['actual_output_directory']) == run_path(seed, cohort), 'Actual run plan path differs')
    require(not result_path(seed, cohort).exists(), 'Completed run result already exists')
    global_clock, clock = _job_clock(), _run_clock(seed, cohort, resume)
    loaded = protocol.load_data(mode='train')
    clips = validate_loaded(plan, loaded)
    remaining = min(2100 - elapsed_since(clock['created_utc']), 9000 - elapsed_since(global_clock['created_utc']))
    require(remaining > 0, 'Original wall-clock training budget exhausted')
    initial_dir = run_path(seed, cohort)
    output = initial_dir.parent / 'resumed' if resume else initial_dir
    origin, interrupted = _resume_origin(initial_dir, plan) if resume else (None, None)
    if resume:
        require(not output.exists(), 'Only one resume is allowed')
    result = engine.run_segment(plan, clips, output, resume_checkpoint=Path(origin['path']) if origin else None,
                                remaining_time_seconds=remaining,
                                prior_elapsed_seconds=max(0., elapsed_since(clock['created_utc'])) if resume else None)
    runner._save_json(output / 'attempt_result.json', {'schema': 'step16-training-attempt-v1',
        'plan_file': bind(protocol.plan_path(seed, cohort)), 'experiment_plan': bind(protocol.POLICY),
        'segment_result': bind(output / 'segment_result.json'), 'interrupted_origin': interrupted,
        'student_input_npz_decoded': loaded['counts']['student_input_npz_decoded'],
        'training_feature_npz_decoded': loaded['counts']['training_feature_npz_decoded'],
        'dev_probe_pixels_loaded': False, 'resume': resume})
    if result['status'] != 'completed':
        return {'status': result['status'], 'segment_result': bind(output / 'segment_result.json')}
    payload = engine.load_checkpoint(result['checkpoint']['path'])
    engine.validate_checkpoint_payload(plan, payload, 3600)
    all_refs = _saved_refs(initial_dir) + (_saved_refs(output) if resume else [])
    milestones = [ref for ref in all_refs if ref['completed_updates'] in range(600, 3601, 600)]
    require([ref['completed_updates'] for ref in sorted(milestones, key=lambda value: value['completed_updates'])] == list(range(600, 3601, 600)), 'Missing/duplicated scheduled checkpoint')
    segments = [bind(path / 'segment_result.json') for path in [initial_dir, output] if (path / 'segment_result.json').exists()]
    segments = list({ref['path']: ref for ref in segments}.values())
    failed_attempts = sum(read(ref['path'])['failed_or_unrecorded_update_attempts'] for ref in segments)
    value = {**result, 'schema': 'lipnet-step16-training-run-result-v1', 'updates': payload['history'],
             'start_update': 0, 'completed_updates': 3600, 'plan_file': bind(protocol.plan_path(seed, cohort)),
             'experiment_plan': bind(protocol.POLICY), 'checkpoints': sorted(milestones, key=lambda item: item['completed_updates']),
             'recovery_checkpoints': [ref for ref in all_refs if ref not in milestones], 'segments': segments,
             'interrupted_origin': interrupted, 'final_checkpoint': result['checkpoint'],
             'wall_seconds_including_resume': elapsed_since(clock['created_utc']),
             'new_successful_optimizer_updates': 3600,
             'failed_or_unrecorded_update_attempts_total': failed_attempts,
             'actual_optimizer_step_count_fully_known': not failed_attempts and (interrupted or {}).get('status') != 'abruptly_interrupted',
             'discarded_recorded_updates_from_abrupt_interruption': (interrupted or {}).get('recorded_updates_after_last_checkpoint_discarded', 0),
             'dev_inputs_loaded': False, 'probe_inputs_loaded': False, 'audio_used': False, 'teacher_model_loaded': False,
             'service_weights_updated': False, 'training_input_ids': plan['ids']}
    runner._save_json(result_path(seed, cohort), value)
    protocol.verify_frozen()
    return {'status': 'completed', 'run_result': bind(result_path(seed, cohort))}


def smoke_plan(plan, clips):
    ids = list(plan['schedule'][0])
    small = {sid: clips[sid] for sid in ids}
    result = {'schema': 'lipnet-step16-smoke-plan-v1', 'seed': plan['seed'], 'cohort': 'smoke_first_pair', 'arm': 'B_feature',
              'run_kind': 'smoke', 'updates': 6, 'batch_size': 2, 'checkpoint_interval': 2, 'ids': ids,
              'schedule': [ids[:] if index % 2 == 0 else ids[::-1] for index in range(6)],
              'lr_segments': copy.deepcopy(engine.SMOKE_LR), 'time_budget_seconds': 120,
              'initial_state_hashes_expected': plan['initial_state_hashes_expected'], 'runtime': plan['runtime'],
              'execution_code': plan['execution_code'], 'experiment_plan': plan['experiment_plan'],
              'production_run_plan': bind(protocol.plan_path(plan['seed'], plan['cohort'])), 'smoke_contract': engine.smoke_contract()}
    binding = runner.validate_plan(result, small, 'B_feature')
    result['training_data'] = {key: binding[key] for key in ('data_sha256', 'clips')}
    result['plan_sha256'] = runner.canonical_plan_hash(result)
    return result, small


def smoke():
    policy = protocol.verify_frozen()
    require(not SMOKE_RESULT.exists(), 'Smoke evidence already exists; do not rerun selectively')
    loaded = protocol.load_data(mode='train')
    plan = protocol.run_plan(42, 'train24')
    clips = validate_loaded(plan, loaded)
    smoke_value, small = smoke_plan(plan, clips)
    directory = BASE / 'smoke'
    require(not directory.exists(), 'Smoke output already exists')
    directory.mkdir()
    runner._save_json(directory / 'plan.json', smoke_value)
    compare_path = ROOT / 'experiments/local_training_step03_20261007/compare_replay.py'
    require(engine.file_sha(compare_path) == '0f278fb6d207cc65404c56413794af1439f1a35c9eb03cb43860eb10938293cb', 'Replay comparator changed')
    spec = importlib.util.spec_from_file_location('step16_frozen_replay_comparator', compare_path)
    comparator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparator)
    outer = core.capture_rng()
    try:
        continuous = engine.run_segment(smoke_value, small, directory / 'continuous')
        split = engine.run_segment(smoke_value, small, directory / 'split', stop_update=2)
        resumed = engine.run_segment(smoke_value, small, directory / 'resumed', resume_checkpoint=Path(split['checkpoint']['path']))
    finally:
        core.restore_rng(outer)
        gc.collect()
        engine.torch.cuda.empty_cache()
    left, right = engine.load_checkpoint(continuous['checkpoint']['path']), engine.load_checkpoint(resumed['checkpoint']['path'])
    engine.validate_checkpoint_payload(smoke_value, left, 6)
    engine.validate_checkpoint_payload(smoke_value, right, 6)
    state = comparator.compare_state(left['state'], right['state'], engine.REPLAY_TOLERANCES)
    metric_checks = [comparator.compare_metrics(left['history'][index], right['history'][index], engine.REPLAY_TOLERANCES) for index in range(2, 6)]
    passed = state['passed'] and all(item['passed'] for item in metric_checks) and resumed['restore_verification']['exact']
    value = {'schema': 'lipnet-step16-gpu-smoke-preflight-v1', 'status': 'passed' if passed else 'failed',
             'created_utc': utc_now().isoformat(),
             'experiment_plan': bind(protocol.POLICY), 'smoke_plan': bind(directory / 'plan.json'),
             'contract': engine.smoke_contract(), 'segments': [bind(directory / name / 'segment_result.json') for name in ('continuous', 'split', 'resumed')],
             'continuous_checkpoint': continuous['checkpoint'], 'split_checkpoint': split['checkpoint'], 'resumed_checkpoint': resumed['checkpoint'],
             'restore_verification': resumed['restore_verification'], 'state_comparison': state, 'metrics_comparisons': metric_checks,
             'learning_rates': [row['learning_rate'] for row in left['history']],
             'smoke_optimizer_updates_executed': sum(len(item['updates']) for item in (continuous, split, resumed)),
             'main_optimizer_updates_executed': 0, 'training_clip_ids': list(small),
             'dev_or_probe_pixels_loaded': False, 'teacher_forward_calls': 0, 'service_weights_updated': False,
             'caller_rng_restored_exactly': runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer)}
    runner._save_json(SMOKE_RESULT, value)
    require(passed and value['smoke_optimizer_updates_executed'] == 12 and value['caller_rng_restored_exactly'], 'GPU smoke failed; main training prohibited')
    return value


def run_all():
    protocol.verify_frozen()
    completed = []
    logs = BASE / 'logs'
    logs.mkdir(exist_ok=True)
    _job_clock()
    for seed in SEEDS:
        for cohort in COHORTS:
            require(not result_path(seed, cohort).exists(), 'Refusing to rerun completed experiment')
            for attempt in range(2):
                name = f'seed{seed}_{cohort}' + ('_resume' if attempt else '')
                with (logs / (name + '.log')).open('x', encoding='utf-8') as log:
                    command = [sys.executable, str(Path(__file__).resolve()), 'run', '--seed', str(seed), '--cohort', cohort]
                    if attempt:
                        command.append('--resume')
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}, check=False)
                if process.returncode == 0:
                    break
                segment_path = run_path(seed, cohort) / 'segment_result.json'
                require(attempt == 0 and segment_path.exists() and read(segment_path)['status'] in ('failed', 'interrupted'),
                        'Training stopped; inspect preserved failure/time-budget evidence')
            require(process.returncode == 0 and result_path(seed, cohort).exists(), 'Single resume failed; no automatic retry')
            completed.append(bind(result_path(seed, cohort)))
            print(json.dumps({'completed': name, 'result': completed[-1]}), flush=True)
    results = [read(ref['path']) for ref in completed]
    value = {'schema': 'lipnet-step16-training-complete-v1', 'status': 'completed', 'experiment_plan': bind(protocol.POLICY),
             'runs': completed, 'new_main_optimizer_updates': sum(len(result['updates']) for result in results),
             'separate_smoke_optimizer_updates': read(SMOKE_RESULT)['smoke_optimizer_updates_executed'],
             'failed_or_unrecorded_update_attempts_total': sum(result['failed_or_unrecorded_update_attempts_total'] for result in results),
             'actual_optimizer_step_count_fully_known': all(result['actual_optimizer_step_count_fully_known'] for result in results),
             'discarded_recorded_updates_from_abrupt_interruption': sum(result['discarded_recorded_updates_from_abrupt_interruption'] for result in results),
             'service_weights_updated': False}
    require(value['new_main_optimizer_updates'] == 14400, 'Main update accounting changed')
    runner._save_json(BASE / 'results/training_complete.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('smoke', 'run', 'run-all'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--cohort', choices=COHORTS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'smoke':
        value = smoke()
    elif args.command == 'run-all':
        value = run_all()
    else:
        require(args.seed is not None and args.cohort is not None, 'Run identity required')
        value = execute(args.seed, args.cohort, args.resume)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    if value.get('status') not in ('completed', 'passed'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
