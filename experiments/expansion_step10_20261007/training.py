"""Fresh expanded-data runs with byte-verified previous controls and no service writes."""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / 'experiments/local_training_step05_20261007'
POLICY = BASE / 'comparison_plan.json'
FROZEN = BASE / 'plans/freeze_manifest.json'
SEEDS = (42, 43)
VARIANTS = ('control_old441', 'expanded510')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


for relative, expected in {
    'experiments/local_training_step03_20261007/runner.py': '71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693',
    'experiments/feature_transfer_step07/train_compare.py': 'eb65e48cf6d4d35825feee1c3d0bd885952357c50c011049944358081790c62a',
}.items():
    if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
        raise ValueError('Changed frozen training engine: ' + relative)
runner = module('step10_frozen_runner', ROOT / 'experiments/local_training_step03_20261007/runner.py')
core, require = runner.core, runner.require
bind, read = core.bind, core.read


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == {'path': ref['path'], 'sha256': ref['sha256']}, 'Changed/nonlocal evidence: ' + str(path))


def bound(ref):
    verify(ref)
    return read(ref['path'])


def plan_path(seed):
    require(seed in SEEDS, 'Unknown seed')
    return BASE / 'plans' / f'seed{seed}.json'


def control(seed):
    audit_ref = {'path': str(OLD / 'results/independent_audit.json'),
                 'sha256': '821c722f43320ab672cd3ae773305b970a5d841908bde88b5729562e1771f982'}
    historical = bound(audit_ref)
    matches = [r for r in historical['runs'] if r['seed'] == seed and r['arm'] == 'B_feature']
    require(historical['status'] == 'passed' and len(matches) == 1, 'Missing historical control audit')
    accepted = matches[0]
    path = OLD / f'runs/seed{seed}/B_feature/continuous/run_result.json'
    result, plan = read(path), read(OLD / f'plans/seed{seed}.json')
    require(accepted['run_result'] == bind(path) and accepted['plan'] == bind(OLD / f'plans/seed{seed}.json')
            and accepted['checkpoint'] == result['checkpoint'], 'Control no longer equals the audited historical run')
    require(result['status'] == 'completed' and result['arm'] == 'B_feature' and result['seed'] == seed, 'Wrong old control')
    require(result['plan_file'] == bind(OLD / f'plans/seed{seed}.json'), 'Changed control plan')
    for ref in plan['execution_code']:
        verify(ref)
    for key in ('full_data_plan', 'comparison_recipe', 'step03_replay_audit'):
        verify(plan[key])
    segment = OLD / f'runs/seed{seed}/B_feature/continuous/segment_contract.json'
    require(read(segment)['initial_state_hashes'] == result['initial_state_hashes'], 'Control initial state mismatch')
    checkpoint = runner.load_checkpoint(Path(result['checkpoint']['path']))
    validator = module('step10_prior_final_validator', OLD / 'evaluate.py')
    validator.validate_final_payload(plan, result, checkpoint)
    require(checkpoint['initial_state_hashes'] == result['initial_state_hashes'] and result['start_update'] == 0 and result['restored_from'] is None, 'Control not fresh')
    for key, hash_key in (('path', 'sha256'), ('sidecar', 'sidecar_sha256')):
        require(bind(Path(result['checkpoint'][key]))['sha256'] == result['checkpoint'][hash_key], 'Control checkpoint binding')
    return {'historical_audit': audit_ref, 'plan_file': bind(OLD / f'plans/seed{seed}.json'), 'run_result': bind(path),
            'segment_contract': bind(segment), 'checkpoint': bind(Path(result['checkpoint']['path'])),
            'checkpoint_sidecar': bind(Path(result['checkpoint']['sidecar'])),
            'initial_state_hashes': result['initial_state_hashes'], 'final_state_hashes': result['final_state_hashes'],
            'runtime': plan['runtime'], 'historical_optimizer_updates': 3200,
            'historical_processed_frames': sum(sum(row['frames']) for row in result['updates'])}


def code_bindings():
    import evaluation
    paths = [BASE / name for name in ('training.py', 'test_training.py', 'mixed_artifacts.py', 'test_mixed_artifacts.py')]
    paths += [Path(r['path']) for r in runner._code_bindings()]
    paths += [OLD / 'full_artifacts.py', OLD / 'evaluate.py']
    paths += [Path(r['path']) for r in evaluation.evaluation_code()]
    return [bind(p) for p in sorted(set(paths), key=str)]


def check_contract(plan, policy):
    seed = plan['seed']
    require(seed in SEEDS and plan['schema'] == 'lipnet-step10-expanded-training-plan-v1', 'Plan identity')
    require(plan['updates'] == 3200 and plan['batch_size'] == 2 and plan['checkpoint_interval'] == 100, 'Budget changed')
    require(plan['variant'] == 'expanded510' and plan['arm'] == 'B_feature', 'Method changed')
    require(plan['ids'] == policy['provisional_train_ids'] and len(set(plan['ids'])) == 510, 'Candidate cohort changed')
    require(plan['schedule'] == core.fixed_schedule(plan['ids'], seed, 3200, 2), 'Schedule changed')
    require(set(s for batch in plan['schedule'] for s in batch) == set(plan['ids']), 'Incomplete training coverage')
    excluded = set(policy['held_train_ids']) | set(policy['development_groups']['old71_same_training_speakers']) | set(policy['development_groups']['new24_all_original_items'])
    require(not set(plan['ids']) & excluded, 'Forbidden training data')
    require(plan['runtime'] == policy['runtime'] and plan['initial_state_hashes_expected'] == policy['controls'][str(seed)]['initial_state_hashes'], 'Initialization/runtime changed')
    require(plan['optimizer'] == policy['optimizer'] and plan['loss'] == policy['loss'], 'Optimizer/loss changed')
    require(plan['experiment_plan'] == bind(POLICY) and plan['plan_sha256'] == runner.canonical_plan_hash(plan), 'Plan binding changed')


def freeze():
    import mixed_artifacts
    import evaluation
    require(not POLICY.exists() and not FROZEN.exists() and not any(plan_path(s).exists() for s in SEEDS), 'Preserve existing freeze artifacts')
    core.setup_runtime()
    require(runner.torch.cuda.is_available(), 'CUDA required')
    loaded = mixed_artifacts.load_training('expanded510')
    handoff = bound(loaded['handoff'])
    loader_validation = read(BASE / 'results/mixed_loader_validation.json')
    require(loader_validation['status'] == 'passed' and loader_validation['handoff'] == loaded['handoff'], 'Loader validation not passed')
    verify(loader_validation['loader']); verify(loader_validation['tests'])
    controls, checks = {}, {}
    runtime = runner._runtime(runner.torch.device('cuda:0'))
    for seed in SEEDS:
        entry = control(seed)
        old_plan = bound(entry['plan_file'])
        require(runtime == entry['runtime'], 'Runtime no longer matches reused controls')
        old_clips = {sid: loaded['clips'][sid] for sid in handoff['control_old_train_ids']}
        actual = runner.validate_plan(old_plan, old_clips, 'B_feature')
        require(old_plan['training_data'] == {k: actual[k] for k in ('data_sha256', 'clips')}, 'Old mixed-loader tensors differ')
        objects = core.make_objects(seed, 'B_feature', 'cuda:0')
        initial = runner.state_hashes(core.runtime_state(*objects))
        require(initial == entry['initial_state_hashes'], 'Fresh initialization differs from reused control')
        del objects
        gc.collect(); runner.torch.cuda.empty_cache()
        checks[str(seed)] = {'old441_data_sha256': actual['data_sha256'], 'old441_tensor_bindings_exact': True,
                             'fresh_initial_state_hashes': initial, 'fresh_initial_state_exact': True,
                             'student_updates_executed': 0}
        controls[str(seed)] = entry
    recipe = bound(bound(controls['42']['plan_file'])['comparison_recipe'])
    policy = {'schema': 'lipnet-step10-expansion-comparison-plan-v1',
              'status': 'frozen_before_expanded_training_and_new_evaluation', 'handoff': loaded['handoff'],
              'mixed_loader_validation': bind(BASE / 'results/mixed_loader_validation.json'),
              'seeds': list(SEEDS), 'variants': list(VARIANTS), 'method': 'B_feature',
              'updates_per_model': 3200, 'clips_per_update': 2, 'new_optimizer_updates_planned': 6400,
              'reused_historical_optimizer_updates': 6400, 'controls': controls, 'reuse_gates': checks,
              'runtime': runtime, 'optimizer': recipe['optimizer'], 'loss': recipe['loss'],
              'provisional_train_ids': handoff['provisional_train_ids'], 'control_train_ids': handoff['control_old_train_ids'],
              'development_groups': handoff['development_groups'], 'held_train_ids': handoff['held_new_train_ids'],
              'evaluation_contract': evaluation.evaluation_contract(), 'execution_code': code_bindings(),
              'budget_interpretation': 'Equal3200 optimizer updates and6400 utterance presentations per seed; actual frames, epochs, FLOPs and time are not identical.',
              'initialization': 'Fresh seeded student and adapter matching historical controls; never continue learned weights.',
              'primary_checkpoint': 'Fixed update3200; no development-based selection or early stopping.',
              'provisional_label_policy': 'Use the frozen Step9 candidate69 and target tokens with ASR-selection bias and unconfirmed full transcript/boundary status retained.',
              'pending_listening': handoff['pending_listening'], 'limitations': handoff['limitations'],
              'new_full_transcript_approvals': 0, 'service_weights_updated': False, 'audio_used': False,
              'training_ready_flags_promoted': False}
    runner._save_json(POLICY, policy)
    plans = []
    for seed in SEEDS:
        plan = {'schema': 'lipnet-step10-expanded-training-plan-v1', 'seed': seed, 'variant': 'expanded510',
                'arm': 'B_feature', 'updates': 3200, 'batch_size': 2, 'checkpoint_interval': 100,
                'ids': loaded['ids'], 'schedule': core.fixed_schedule(loaded['ids'], seed, 3200, 2),
                'experiment_plan': bind(POLICY), 'handoff': loaded['handoff'], 'data_bindings': loaded['bindings'],
                'runtime': runtime, 'optimizer': policy['optimizer'], 'loss': policy['loss'],
                'initial_state_hashes_expected': controls[str(seed)]['initial_state_hashes'],
                'development_groups': policy['development_groups'], 'service_weights_updated': False}
        actual = runner.validate_plan(plan, loaded['clips'], 'B_feature')
        plan['training_data'] = {k: actual[k] for k in ('data_sha256', 'clips')}
        plan['plan_sha256'] = runner.canonical_plan_hash(plan)
        check_contract(plan, policy)
        runner._save_json(plan_path(seed), plan)
        plans.append(bind(plan_path(seed)))
    result = {'schema': 'lipnet-step10-freeze-manifest-v1', 'status': 'frozen', 'experiment_plan': bind(POLICY),
              'plans': plans, 'new_optimizer_updates_planned': 6400, 'optimizer_updates_executed': 0}
    runner._save_json(FROZEN, result)
    return result


def verify_frozen():
    manifest = read(FROZEN)
    require(manifest['status'] == 'frozen' and manifest['experiment_plan'] == bind(POLICY), 'Frozen manifest mismatch')
    policy = bound(manifest['experiment_plan'])
    require(policy['execution_code'] == code_bindings(), 'Execution code changed')
    verify(policy['handoff'])
    verify(policy['mixed_loader_validation'])
    for ref in manifest['plans']:
        plan = bound(ref); check_contract(plan, policy)
    require({Path(r['path']).name for r in manifest['plans']} == {'seed42.json', 'seed43.json'}, 'Frozen seeds differ')
    core.setup_runtime()
    require(policy['runtime'] == runner._runtime(runner.torch.device('cuda:0')), 'Runtime changed')
    return policy


def execute(seed, resume=None, recovery_name=None):
    import mixed_artifacts
    require(shutil.disk_usage(BASE).free >= 10 * 1024**3, 'Less than10GiB reserve')
    started = time.perf_counter()
    policy = verify_frozen()
    plan = read(plan_path(seed))
    loaded = mixed_artifacts.load_training('expanded510')
    actual = runner.validate_plan(plan, loaded['clips'], 'B_feature')
    require(loaded['bindings'] == plan['data_bindings'] and plan['training_data'] == {k: actual[k] for k in ('data_sha256', 'clips')}, 'Training bytes changed')
    # Verify before the first optimizer update, not only after the run.
    objects = core.make_objects(seed, 'B_feature', 'cuda:0')
    require(runner.state_hashes(core.runtime_state(*objects)) == plan['initial_state_hashes_expected'], 'Fresh initial state changed')
    del objects; gc.collect(); runner.torch.cuda.empty_cache()
    home = BASE / 'runs' / f'seed{seed}'
    if resume is None:
        require(recovery_name is None, 'Recovery name needs checkpoint')
        output = home / 'continuous'
    else:
        resume = Path(resume).resolve()
        require(resume.is_relative_to(home) and resume.suffix == '.pt' and recovery_name and re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', recovery_name), 'Invalid recovery path/name')
        output = home / 'recoveries' / recovery_name
    result = runner.run_segment(plan, loaded['clips'], 'B_feature', output, stop_update=3200, resume_checkpoint=resume, device='cuda:0')
    require(result['initial_state_hashes'] == plan['initial_state_hashes_expected'], 'Run initialization mismatch')
    result.update(status='completed', seed=seed, variant='expanded510', plan_file=bind(plan_path(seed)),
                  experiment_plan=bind(POLICY), elapsed_seconds=time.perf_counter() - started,
                  dev_inputs_loaded=False, held_inputs_loaded=False, audio_used=False,
                  teacher_model_loaded=False, service_weights_updated=False)
    runner._save_json(output / 'run_result.json', result)
    return bind(output / 'run_result.json')


def evaluation_target(variant, seed):
    require(variant in VARIANTS and seed in SEEDS, 'Unknown evaluation target')
    policy = verify_frozen()
    expected = policy['controls'][str(seed)]['initial_state_hashes']
    if variant == 'control_old441':
        entry = policy['controls'][str(seed)]
        require(control(seed) == entry, 'Historical control changed')
        plan_ref, result_ref = entry['plan_file'], entry['run_result']
    else:
        candidates = [p for p in [BASE / f'runs/seed{seed}/continuous/run_result.json', *sorted((BASE / f'runs/seed{seed}/recoveries').glob('*/run_result.json'))] if p.exists()]
        require(len(candidates) == 1, 'Missing/ambiguous completed expansion run')
        result_ref, plan_ref = bind(candidates[0]), bind(plan_path(seed))
    result, plan = bound(result_ref), bound(plan_ref)
    require(result['status'] == 'completed' and result['plan_file'] == plan_ref and result['arm'] == 'B_feature', 'Wrong completed result')
    require(result['initial_state_hashes'] == expected, 'Initial state differs')
    payload = runner.load_checkpoint(Path(result['checkpoint']['path']))
    validator = module('step10_target_final_validator', OLD / 'evaluate.py')
    validator.validate_final_payload(plan, result, payload)
    require(payload['initial_state_hashes'] == expected, 'Checkpoint initial state differs')
    return {'plan': plan, 'result': result, 'experiment_plan': bind(POLICY), 'plan_file': plan_ref,
            'run_result': result_ref, 'initial_state_hashes_expected': expected}


def run_all():
    verify_frozen()
    logs = BASE / 'logs'; logs.mkdir(exist_ok=True)
    completed = []
    for seed in SEEDS:
        home = BASE / f'runs/seed{seed}/continuous'
        if (home / 'run_result.json').exists():
            completed.append(evaluation_target('expanded510', seed)['run_result']); continue
        require(not home.exists() or not any(home.iterdir()), 'Interrupted run: explicitly resume its complete checkpoint')
        log = logs / f'seed{seed}.log'
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--seed', str(seed)], cwd=ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for line in child.stdout:
                stream.write(line); stream.flush()
                try:
                    row = json.loads(line)
                except ValueError:
                    print(line.rstrip(), flush=True); continue
                if isinstance(row, dict) and row.get('update', 0) % 100 == 0:
                    print(json.dumps({'seed': seed, **row}), flush=True)
            require(child.wait() == 0, 'Training stopped; inspect ' + str(log))
        completed.append(evaluation_target('expanded510', seed)['run_result'])
    result = {'schema': 'lipnet-step10-training-complete-v1', 'status': 'completed', 'experiment_plan': bind(POLICY),
              'run_results': completed, 'new_optimizer_updates': 6400, 'reused_historical_optimizer_updates': 6400,
              'service_weights_updated': False}
    runner._save_json(BASE / 'results/training_complete.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true'); mode.add_argument('--all', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--resume', type=Path); parser.add_argument('--recovery-name')
    args = parser.parse_args()
    if args.freeze: value = freeze()
    elif args.all: value = run_all()
    else:
        if args.seed is None: parser.error('--seed is required')
        value = execute(args.seed, args.resume, args.recovery_name)
    print(json.dumps(value), flush=True)
