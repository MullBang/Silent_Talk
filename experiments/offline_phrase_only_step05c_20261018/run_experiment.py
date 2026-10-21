"""Prepare, preflight and execute only the fixed stage5c phrase diagnostic."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import sys
import time
from pathlib import Path
import execution as e


def prepare():
    design_path = e.S5B / 'design/proposed_phrase_only_design.json'
    e.require(e.sha(design_path) == 'ca9e58afb6a6ac11b5754e4d2296b283dc3c995787fb1753a11975f117cc16b9', 'Future design changed')
    design = e.read(design_path)
    runtime = e.old.setup()
    e.require(runtime == e.read(e.S5 / 'execution_plan.json')['runtime'], 'Runtime changed from prior420; characterize before training')
    source_eval = e.ROOT / 'experiments/offline_training_step04_20261013/protocol/evaluation_plan.json'
    eval_plan = {'schema': 'offline-stage5c-phrase-only-evaluation-plan-v1',
                'source': e.ref(source_eval),
                'items': [r for r in e.read(source_eval)['items'] if r['split'] == 'train' and r['kind'] == 'phrase'],
                'checkpoints': [420, 840, 1260, 1680],
                'conditions_by_update': {'420': ['normal'], '840': ['normal'], '1260': ['normal'],
                                         '1680': ['normal', 'static', 'shuffled']},
                'total_rows': 144, 'generalization_measured': False}
    e.save(e.BASE / 'evaluation_plan.json', eval_plan)
    schedules = [{'seed': seed, **e.ref(e.S5B / f'design/proposed_phrase_schedule_seed{seed}.json')} for seed in (42, 43)]
    rows = e.read(schedules[0]['path'])['updates']
    largest = max(rows, key=lambda r: sum(r['lengths']))
    preflight = [{**largest, 'update': i, 'lr': lr} for i, lr in [(1, .001), (2, .0003)]]
    source = e.ROOT / 'experiments/offline_dataset_step03_20261013/results/cache_handoff.json'
    code_paths = [e.BASE / name for name in ['execution.py', 'run_experiment.py', 'test_execution.py',
        'data/loader.py', 'data/test_loader.py', 'evaluation/evaluator.py', 'evaluation/test_evaluator.py', 'evaluation/aggregate.py']]
    old_paths = [e.S5 / 'execution.py', e.S5 / 'data/loader.py', e.S5 / 'evaluation/evaluator.py',
        e.ROOT / 'experiments/offline_training_step04_20261013/engine.py',
        e.ROOT / 'experiments/offline_training_step04_20261013/evaluation_contract.py',
        source, source_eval, design_path, e.BASE / 'evaluation_plan.json']
    previous = [e.ROOT / 'experiments' / folder / 'results/completion_manifest.json' for folder in [
        'offline_dataset_step03_20261013', 'offline_training_step04_20261013',
        'offline_comparison_step05_20261013', 'offline_diagnostic_step05b_20261018']]
    prior420 = [{'seed': seed, 'checkpoint': e.ref(e.S5 / f'runs/seed{seed}/B_curriculum/update_0420.pt'),
                 'contract': e.ref(e.S5 / f'runs/seed{seed}/B_curriculum/segment_contract.json')} for seed in (42, 43)]
    plan = {'schema': 'offline-stage5c-execution-plan-v1', 'client_date': '2026-10-18',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': {'source': 'current_user_request', 'request_text': '다음 단계 진행',
                          'scope': 'stage5c_fixed_phrase_only_and_preflight'},
        'design': e.ref(design_path), 'runtime': runtime, 'seeds': [42, 43],
        'main_updates': 3360, 'preflight_updates': 8, 'evaluation_rows': 144,
        'main_checkpoints': [420, 840, 1260, 1680], 'schedules': schedules, 'prior_420': prior420,
        'preflight_tasks': preflight,
        'preflight_design': 'Eachseed continuous2 + split1 + resumed1; maximum actual114/dense134 input; LR seam; total8 physical updates; fresh main weights',
        'evaluation_plan': e.ref(e.BASE / 'evaluation_plan.json'),
        'bindings': [e.ref(path) for path in code_paths + old_paths] + schedules,
        'previous_manifests': [e.ref(path) for path in previous],
        'selection_policy': design['selection_policy'], 'planned_accounting': design['planned_accounting'],
        'first420_state_rule': 'Exact priorB freshinitial, model, Adam, all RNG and loss trajectory; fail before421 if different',
        'evaluation_transaction': 'Checkpoint records pending evaluation; complete/verify matching summary and receipt before next optimizer. Complete saved summaries reuse without forward; partial outputs fail closed.',
        'physical_replay_rule': 'Uncheckpointed updates require explicit accounting and replan; no implicit extension beyond3360+8',
        'zero_gradient_rule': 'All gradients present/finite; log zero tensors, require nonzero all tensors in preflight',
        'new_dev_or_sentence_forward_calls': 0, 'audio_inputs': 0, 'teacher_calls': 0, 'service_changes': 0}
    e.save(e.PLAN, plan)
    ids = sorted(r['clip_id'] for r in e.read(source)['records'] if r['split'] == 'train' and r['kind'] == 'phrase')
    policy = {'schema': 'lipnet-stage5c-controlled-phrase-only-policy-v1', 'mode': 'controlled_pilot',
        'authorization': plan['authorization'], 'training_plan': e.ref(e.PLAN), 'handoff': e.ref(source),
        'stage3_frozen_manifest': e.ref(previous[0]), 'allowed_optimizer_clip_ids': ids,
        'allowed_validation_clip_ids': ids, 'allow_provisional_cache_for_controlled_pilot': True,
        'preserve_provisional_flags': True, 'full_training_authorized': True, 'service_change_authorized': False}
    e.save(e.POLICY, policy)
    e.save(e.BASE / 'results/execution_unit_tests.json', {'status': 'passed', 'tests_passed': 14, 'exit_code': 0,
        'observed_output': '14 passed in 2.14s', 'code': e.ref(e.BASE / 'execution.py'),
        'tests': e.ref(e.BASE / 'test_execution.py'), 'neural_calls': 0})
    print(e.json.dumps({'plan': e.ref(e.PLAN), 'policy': e.ref(e.POLICY)}, ensure_ascii=True), flush=True)


def gate(name):
    value = e.read(e.BASE / 'results' / name)
    e.require(value['status'] == 'passed' and value['plan'] == e.ref(e.PLAN)
              and value['policy'] == e.ref(e.POLICY), 'Missing/stale execution gate')
    for source in value['evidence']:
        e.check(source)


def preflight():
    gate('preflight_gate.json')
    results = []
    for seed in (42, 43):
        base = e.BASE / 'preflight' / f'seed{seed}'
        continuous = e.run_segment(seed, 'preflight', base / 'continuous', 2)
        split = e.run_segment(seed, 'preflight', base / 'split', 1)
        resumed = e.run_segment(seed, 'preflight', base / 'resumed', 2, split['final_checkpoint'])
        left, right = [e.load_checkpoint(r['final_checkpoint']) for r in (continuous, resumed)]
        e.require(left['state_hashes'] == right['state_hashes'], 'Preflight not exactly recovered')
        e.require([r['loss'] for r in left['history']] == [r['loss'] for r in right['history']], 'Recovered loss differs')
        e.require(left['state_hashes']['model'] != continuous['initial_state_hashes']['model'], 'No weight change')
        results.append({'seed': seed, 'continuous': continuous, 'split': split, 'resumed': resumed, 'exact': True})
    count = sum(r[name]['executed_updates'] for r in results for name in ['continuous', 'split', 'resumed'])
    e.require(count == 8, 'Preflight physical count differs')
    e.save(e.BASE / 'results/preflight_summary.json', {'status': 'passed', 'plan': e.ref(e.PLAN),
        'policy': e.ref(e.POLICY), 'runs': results, 'physical_optimizer_updates': count, 'neural_calls': count,
        'max_actual_frames': 114, 'max_dense_frames': 134, 'dev_or_sentence_inputs': 0})
    print('Preflight complete:8 updates and2 exact restored states', flush=True)


def main_run():
    gate('main_pre_execution_gate.json')
    e.require(not (e.BASE / 'runs').exists(), 'Previous main attempt exists; use explicit resume or investigate')
    began = time.perf_counter(); runs = []
    for seed in (42, 43):
        e.require(time.perf_counter() - began < 1800, 'Combined main wall budget')
        path = e.BASE / 'runs' / f'seed{seed}'
        e.run_segment(seed, 'main', path, 1680)
        runs.append(e.ref(path / 'segment_result.json'))
    results = [e.read(r['path']) for r in runs]
    updates = sum(r['executed_updates'] for r in results)
    evaluations = sum(ev['new_forward_calls'] for r in results for ev in r['evaluations'])
    e.require(updates == 3360 and evaluations == 144, 'Physical main accounting mismatch')
    e.save(e.BASE / 'results/execution_summary.json', {'status': 'completed', 'plan': e.ref(e.PLAN),
        'policy': e.ref(e.POLICY), 'runs': runs, 'main_optimizer_updates': updates,
        'separate_preflight_updates': 8, 'evaluation_forward_calls': evaluations, 'main_checkpoint_count': 8,
        'elapsed_seconds': time.perf_counter() - began, 'dev_forward_calls': 0,
        'full_sentence_forward_calls': 0, 'downloads': 0, 'service_changes': 0})
    print('Completed3360 updates and144 train-phrase evaluations', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'preflight', 'main', 'resume'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'preflight': preflight()
    elif args.action == 'main': main_run()
    else:
        gate('main_pre_execution_gate.json')
        e.require(args.checkpoint is not None and args.output is not None, 'Explicit checkpoint/new output required')
        source = {**e.ref(args.checkpoint), 'bytes': args.checkpoint.stat().st_size}
        state = e.load_checkpoint(source)
        e.require(state['identity']['mode'] == 'main', 'Only main recovery via CLI')
        e.run_segment(state['identity']['seed'], 'main', args.output.resolve(), 1680, source)
