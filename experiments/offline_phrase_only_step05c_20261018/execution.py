"""Separate phrase-only scope with pending-evaluation recovery and exact420 gate."""
from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
S5 = ROOT / 'experiments/offline_comparison_step05_20261013'
S5B = ROOT / 'experiments/offline_diagnostic_step05b_20261018'
PLAN = BASE / 'execution_plan.json'
POLICY = BASE / 'controlled_pilot_policy.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(item):
    require(sha(item['path']) == item['sha256'], 'Changed bound file: ' + item['path'])


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), 'Must create new in-scope evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    partial.rename(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Missing local module')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


require(sha(S5 / 'execution.py') == '623e4a21d38bb8d32bc90e8949cada112fe19fd70d24c6734c9b5ccfadebce93', 'Frozen arithmetic changed')
prior = module(S5 / 'execution.py', 'stage5c_frozen_arithmetic')
old = prior.old
torch, np = old.torch, old.np


def verify_plan(plan):
    require(plan['schema'] == 'offline-stage5c-execution-plan-v1', 'Wrong scope')
    require(plan['main_updates'] == 3360 and plan['preflight_updates'] == 8 and plan['evaluation_rows'] == 144, 'Budget changed')
    for item in plan['bindings']:
        check(item)
    for item in plan['previous_manifests']:
        check(item)
        for artifact in read(item['path'])['artifacts']:
            check(artifact)


def tasks_for(plan, seed, mode):
    require(seed in (42, 43), 'Unexpected seed')
    if mode == 'preflight':
        return plan['preflight_tasks']
    require(mode == 'main', 'Wrong mode')
    item = next(source for source in plan['schedules'] if source['seed'] == seed)
    check(item)
    return read(item['path'])['updates']


def load_checkpoint(source):
    path = Path(source['path']).resolve()
    require(path.is_relative_to(BASE), 'Checkpoint outside current experiment')
    check(source)
    return old.runner._unpack_safe(torch.load(path, map_location='cpu', weights_only=True))


def validate_checkpoint(payload, identity, tasks):
    require(payload['schema'] == 'offline-stage5c-complete-checkpoint-v1' and payload['identity'] == identity, 'Wrong checkpoint identity')
    done = payload['completed_updates']
    require(type(done) is int and 0 < done <= len(tasks) and len(payload['history']) == done, 'Wrong cursor/history')
    require(payload['state_hashes'] == old.hashes(payload['state']), 'State hash mismatch')
    require(payload['next_task'] == (tasks[done] if done < len(tasks) else None), 'Wrong next task/LR')
    for index, row in enumerate(payload['history']):
        task = tasks[index]
        require(row['update'] == index + 1 and row['clip_ids'] == task['clip_ids']
                and row['kind'] == task['kind'] == 'phrase' and row['learning_rate'] == task['lr']
                and row['optimizer_updated'] is True, 'Changed input/LR/history')
    opt = payload['state']['optimizer']
    require(opt['state'] and all(float(value['step']) == done for value in opt['state'].values()), 'Wrong Adam step')
    require(all(group['lr'] == tasks[done - 1]['lr'] for group in opt['param_groups']), 'Wrong saved LR')
    due = identity['mode'] == 'main' and done in (420, 840, 1260, 1680)
    require(payload['evaluation_due'] is due, 'Pending evaluation lost')
    if due:
        directory = Path(payload['evaluation_dir']).resolve()
        require(directory.is_relative_to(BASE) and directory.name == f'update_{done:04d}', 'Wrong evaluation destination')


def write_checkpoint(path, payload, identity, tasks):
    require(path.resolve().is_relative_to(BASE) and not path.exists(), 'Checkpoint exists/outside scope')
    partial = path.with_suffix('.pt.partial')
    with partial.open('xb') as stream:
        torch.save(old.runner._pack_safe(payload), stream)
        stream.flush(); os.fsync(stream.fileno())
    partial.rename(path)
    source = {**ref(path), 'bytes': path.stat().st_size}
    validate_checkpoint(load_checkpoint(source), identity, tasks)
    return source


def assert_receipt(receipt, checkpoint, state_hashes):
    require(receipt['status'] == 'completed' and receipt['checkpoint'] == checkpoint, 'Wrong evaluation transaction')
    require(receipt['training_state_before'] == receipt['training_state_after'] == state_hashes, 'Evaluation mutated state')
    check(receipt['evaluation']['summary'])
    require(receipt['evaluation']['failures'] == 0, 'Failed evaluation cannot authorize continuation')


def finish_evaluation(model, optimizer, dataset, payload, checkpoint):
    """A completed checkpoint never permits optimization past missing evaluation."""
    require(payload['evaluation_due'], 'No pending evaluation at this checkpoint')
    output = Path(payload['evaluation_dir'])
    receipt_path = output.parent / (output.name + '_completed.json')
    identity = {**payload['identity'], 'completed_update': payload['completed_updates'], 'checkpoint': checkpoint}
    evaluator = module(BASE / 'evaluation/evaluator.py', 'stage5c_evaluator')
    before = old.hashes(old.state(model, optimizer))
    require(before == payload['state_hashes'], 'Evaluation starts from wrong state')
    # Completed summaries are verified/reused; incomplete existing directories fail closed.
    result = evaluator.evaluate(model, dataset, identity, output)
    after = old.hashes(old.state(model, optimizer))
    require(before == after, 'Evaluation changed model/Adam/RNG')
    require(result['failures'] == 0, 'Failed rows retained: stop before optimization')
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert_receipt(receipt, checkpoint, before)
        require(receipt['evaluation']['summary'] == result['summary'], 'Changed completed evaluation')
    else:
        save(receipt_path, {'status': 'completed', 'checkpoint': checkpoint, 'evaluation': result,
                            'training_state_before': before, 'training_state_after': after})
    return result


def compare_420(plan, seed, current, output):
    source = next(item for item in plan['prior_420'] if item['seed'] == seed)
    check(source['checkpoint']); check(source['contract'])
    previous = old.runner._unpack_safe(torch.load(source['checkpoint']['path'], map_location='cpu', weights_only=True))
    contract = read(source['contract']['path'])
    evidence = {'seed': seed, 'source_checkpoint': source['checkpoint'],
                'initial_exact': current['initial_state_hashes'] == contract['initial_state_hashes'],
                'state_exact': current['state_hashes'] == old.hashes(previous['state']) == previous['state_hashes'],
                'loss_trajectory_exact': [r['loss'] for r in current['history']] == [r['loss'] for r in previous['history']],
                'old_hashes': previous['state_hashes'], 'new_hashes': current['state_hashes']}
    evidence['status'] = 'passed' if all(evidence[key] for key in ['initial_exact', 'state_exact', 'loss_trajectory_exact']) else 'failed'
    save(output / 'prior420_reproduction.json', evidence)
    require(evidence['status'] == 'passed', 'Prior420 is not exactly reproduced; preserve and investigate')


def run_segment(seed, mode, output, stop, resume=None):
    started = time.perf_counter()
    plan = read(PLAN); verify_plan(plan)
    require(old.setup() == plan['runtime'], 'Numerical runtime changed')
    tasks = tasks_for(plan, seed, mode)
    require(0 < stop <= len(tasks) and (mode != 'main' or stop in (420, 840, 1260, 1680)), 'Wrong stop')
    require(output.resolve().is_relative_to(BASE) and not output.exists(), 'New segment directory required')
    require(shutil.disk_usage(BASE).free >= 4 * 1024**3, 'Disk budget')
    require(torch.cuda.mem_get_info()[0] >= 2 * 1024**3, 'GPU free memory budget')
    loader = module(BASE / 'data/loader.py', 'stage5c_phrase_loader')
    dataset = loader.PhraseDataset(PLAN, POLICY, expected_policy_sha256=sha(POLICY))
    ids = sorted({clip_id for task in tasks for clip_id in task['clip_ids']})
    items = {clip_id: dataset.load_clip(clip_id, purpose='optimizer') for clip_id in ids}
    model, optimizer = old.objects(seed)
    initial = old.hashes(old.state(model, optimizer))
    if mode == 'main':
        expected = next(item for item in plan['prior_420'] if item['seed'] == seed)
        require(initial == read(expected['contract']['path'])['initial_state_hashes'], 'Fresh initial state differs')
    identity = {'plan_sha256': sha(PLAN), 'policy_sha256': sha(POLICY), 'runtime': plan['runtime'],
                'mode': mode, 'seed': seed, 'arm': 'P_phrase',
                'tasks_sha256': hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
    history, checkpoints, evaluations = [], [], []
    cumulative = 0.
    previous = None
    if resume:
        previous = load_checkpoint(resume)
        validate_checkpoint(previous, identity, tasks)
        require(previous['initial_state_hashes'] == initial and previous['completed_updates'] <= stop, 'Wrong resume origin')
        progress = Path(resume['path']).parent / 'progress.jsonl'
        physical_tail = [readrow for readrow in (json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines())
                         if readrow['update'] > previous['completed_updates']]
        require(not physical_tail, 'Uncheckpointed physical updates require explicit replay accounting/replan')
        old.restore(previous['state'], model, optimizer)
        require(old.hashes(old.state(model, optimizer)) == previous['state_hashes'], 'Restore not exact')
        history = list(previous['history']); cumulative = previous['elapsed_seconds']
    output.mkdir(parents=True)
    save(output / 'segment_contract.json', {'identity': identity, 'initial_state_hashes': initial,
        'start_update': len(history), 'stop_update': stop, 'resume': resume,
        'loaded_optimizer_clips': [items[key]['audit'] for key in ids]})
    executed = []
    try:
        if previous is not None and previous['evaluation_due']:
            if previous['completed_updates'] == 420:
                compare_420(plan, seed, previous, output)
            evaluations.append(finish_evaluation(model, optimizer, dataset, previous, resume))
        for index in range(len(history), stop):
            require(cumulative + time.perf_counter() - started < 900, 'Per-seed wall budget exceeded')
            task = tasks[index]
            batch = loader.collate_dense([items[key] for key in task['clip_ids']], purpose='optimizer')
            require(batch['lengths'].tolist() == task['lengths'], 'Actual lengths differ')
            row = {'update': index + 1, 'kind': 'phrase', **prior.update(model, optimizer, batch, task['lr'])}
            history.append(row); executed.append(row)
            with (output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
            require(row['cuda_peak_reserved_bytes'] <= 6 * 1024**3, 'CUDA reservation exceeded')
            if mode == 'preflight':
                require(row['nonzero_gradient_tensors'] == row['parameter_tensors'], 'No genuine gradient coverage')
            if mode == 'preflight' or index + 1 in (420, 840, 1260, 1680):
                state = old.state(model, optimizer)
                due = mode == 'main'
                payload = {'schema': 'offline-stage5c-complete-checkpoint-v1', 'identity': identity,
                    'completed_updates': index + 1, 'state': state, 'state_hashes': old.hashes(state),
                    'initial_state_hashes': initial, 'history': list(history),
                    'next_task': tasks[index + 1] if index + 1 < len(tasks) else None,
                    'elapsed_seconds': cumulative + time.perf_counter() - started,
                    'evaluation_due': due, 'evaluation_dir': str(output / 'evaluations' / f'update_{index + 1:04d}') if due else None}
                checkpoint = write_checkpoint(output / f'update_{index + 1:04d}.pt', payload, identity, tasks)
                checkpoints.append(checkpoint)
                if due:
                    if index + 1 == 420:
                        compare_420(plan, seed, payload, output)
                    evaluations.append(finish_evaluation(model, optimizer, dataset, payload, checkpoint))
            if mode == 'preflight' or (index + 1) % 120 == 0:
                print(json.dumps({'seed': seed, 'mode': mode, 'update': index + 1, 'loss': row['loss'],
                                  'seconds': round(time.perf_counter() - started, 2)}), flush=True)
        result = {'status': 'completed', 'identity': identity, 'initial_state_hashes': initial,
                  'completed_updates': stop, 'executed_updates': len(executed), 'resume': resume,
                  'history_count': len(history), 'checkpoints': checkpoints, 'evaluations': evaluations,
                  'elapsed_seconds': time.perf_counter() - started,
                  'final_checkpoint': checkpoints[-1] if checkpoints else resume,
                  'dev_forward_calls': 0, 'full_sentence_forward_calls': 0, 'service_changes': 0}
        save(output / 'segment_result.json', result)
        return result
    except Exception as exc:
        save(output / 'failure.json', {'error': repr(exc), 'completed_updates': len(history),
            'physical_updates_this_segment': len(executed), 'last_checkpoint': checkpoints[-1] if checkpoints else resume})
        raise
    finally:
        del model, optimizer, items, dataset
        gc.collect(); torch.cuda.empty_cache()
