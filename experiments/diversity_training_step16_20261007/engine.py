"""Fixed global-update LR schedule with complete-state checkpoints and one resume."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RUNNER_PATH = ROOT / 'experiments/local_training_step03_20261007/runner.py'
RUNNER_SHA = '71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693'
CORE_SHA = 'eb65e48cf6d4d35825feee1c3d0bd885952357c50c011049944358081790c62a'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if file_sha(RUNNER_PATH) != RUNNER_SHA or file_sha(ROOT / 'experiments/feature_transfer_step07/train_compare.py') != CORE_SHA:
    raise ValueError('Frozen execution primitives changed')
_spec = importlib.util.spec_from_file_location('step16_frozen_runner', RUNNER_PATH)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
core, torch, np = runner.core, runner.torch, runner.np
require, load_checkpoint = runner.require, runner.load_checkpoint
runtime_for = runner._runtime
ENGINE_SCHEMA = 'step16_lr_training_checkpoint_v1'
MAIN_LR = [{'start_update': 1, 'end_update': 2400, 'lr': .001},
           {'start_update': 2401, 'end_update': 3000, 'lr': .0003},
           {'start_update': 3001, 'end_update': 3600, 'lr': .0001}]
SMOKE_LR = [{'start_update': 1, 'end_update': 2, 'lr': .001},
            {'start_update': 3, 'end_update': 4, 'lr': .0003},
            {'start_update': 5, 'end_update': 6, 'lr': .0001}]
REPLAY_TOLERANCES = {'model': {'atol': 1e-6, 'rtol': 1e-5}, 'adapter': {'atol': 1e-6, 'rtol': 1e-5},
                     'exp_avg': {'atol': 1e-7, 'rtol': 1e-4}, 'exp_avg_sq': {'atol': 1e-10, 'rtol': 1e-4},
                     'loss': {'atol': 1e-5, 'rtol': 1e-5}, 'gradient_norm': {'atol': 1e-4, 'rtol': 1e-4}}


def smoke_contract():
    return {'updates_per_trajectory': 6, 'continuous_updates': 6, 'split_first_updates': 2,
            'split_resumed_updates': 4, 'total_executed_optimizer_updates': 12, 'lr_segments': SMOKE_LR,
            'training_clips': 'first pair of frozen seed42 train24 schedule', 'training_data_only': True,
            'main_optimizer_updates': 0, 'initial_update': 0, 'fake_history_allowed': False,
            'restore_hashes_required_exact': True, 'next_update_replay_tolerances': REPLAY_TOLERANCES,
            'tolerances_may_not_be_relaxed_after_observing_results': True}


def execution_code():
    paths = {Path(ref['path']) for ref in runner._code_bindings()}
    paths.update(BASE / name for name in ('engine.py', 'training.py', 'test_training.py'))
    paths.add(ROOT / 'experiments/local_training_step03_20261007/compare_replay.py')
    return [{'path': str(path.resolve()), 'sha256': file_sha(path)} for path in sorted(paths, key=str)]


def validate_lr_segments(segments, updates):
    require(isinstance(segments, list) and segments, 'Missing LR segments')
    next_start = 1
    for segment in segments:
        require(set(segment) == {'start_update', 'end_update', 'lr'} and
                type(segment['start_update']) is int and type(segment['end_update']) is int and
                segment['start_update'] == next_start and segment['end_update'] >= next_start and
                isinstance(segment['lr'], (int, float)) and math.isfinite(segment['lr']) and segment['lr'] > 0,
                'Noncontiguous/invalid LR segments')
        next_start = segment['end_update'] + 1
    require(next_start == updates + 1, 'LR segments do not cover whole budget')


def learning_rate(plan, update):
    require(type(update) is int and 1 <= update <= plan['updates'], 'LR requested outside global update budget')
    segments = plan['lr_segments']
    validate_lr_segments(segments, plan['updates'])
    return next(float(row['lr']) for row in segments if row['start_update'] <= update <= row['end_update'])


def validate_plan(plan, clips):
    require(plan['arm'] == 'B_feature' and plan['run_kind'] in ('main', 'smoke'), 'Wrong training arm/kind')
    validate_lr_segments(plan['lr_segments'], plan['updates'])
    if plan['run_kind'] == 'main':
        require(plan['updates'] == 3600 and plan['checkpoint_interval'] == 600 and plan['lr_segments'] == MAIN_LR and
                plan['cohort'] in ('train24', 'train72') and plan['time_budget_seconds'] == 2100, 'Main training budget changed')
    else:
        require(plan['updates'] == 6 and plan['checkpoint_interval'] == 2 and plan['lr_segments'] == SMOKE_LR and
                plan['time_budget_seconds'] == 120, 'Smoke training budget changed')
    require(plan['initial_state_hashes_expected'] and plan['runtime'], 'Unbound initialization/runtime')
    data = runner.validate_plan(plan, clips, 'B_feature')
    require(plan['training_data'] == {key: data[key] for key in ('data_sha256', 'clips')}, 'Training input bytes changed')
    bound = {ref['path']: ref for ref in plan['execution_code']}
    for ref in execution_code():
        require(bound.get(ref['path']) == ref, 'Unbound/changed execution code')
    return data


def make_identity(plan, data=None, runtime=None):
    data = data or {'plan_sha256': runner.canonical_plan_hash(plan), **plan['training_data']}
    require(data['plan_sha256'] == plan['plan_sha256'] == runner.canonical_plan_hash(plan), 'Plan digest changed')
    return {'schema': 'step16-lr-training-identity-v1', 'plan_sha256': data['plan_sha256'],
            'run_kind': plan['run_kind'], 'seed': plan['seed'], 'cohort': plan['cohort'], 'arm': 'B_feature',
            'total_updates': plan['updates'], 'batch_size': 2, 'schedule_sha256': runner.nested_hash(plan['schedule']),
            'data_sha256': data['data_sha256'], 'clip_bindings': data['clips'], 'lr_segments': plan['lr_segments'],
            'code_bindings': execution_code(), 'runtime': runtime or plan['runtime'],
            'optimization': {'optimizer': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
                             'gradient_clip': 1., 'accumulation_clips': 2, 'lambda': 1., 'ctc_device': 'cpu', 'precision': 'float32'}}


def validate_checkpoint_payload(plan, payload, expected_update=None):
    require(payload['schema'] == runner.SCHEMA and payload['engine_schema'] == ENGINE_SCHEMA, 'Wrong checkpoint schema')
    expected = make_identity(plan)
    require(payload['identity'] == expected, 'Checkpoint identity differs from frozen run')
    completed = payload['completed_updates']
    require(type(completed) is int and 0 <= completed <= plan['updates'] and
            (expected_update is None or completed == expected_update), 'Wrong checkpoint update')
    require(payload['initial_state_hashes'] == plan['initial_state_hashes_expected'], 'Checkpoint initialization changed')
    require(payload['state_hashes'] == runner.state_hashes(payload['state']), 'Checkpoint full-state digest mismatch')
    require(len(payload['history']) == completed, 'Checkpoint history length mismatch')
    for index, row in enumerate(payload['history'], 1):
        require(row['update'] == index and row['ids'] == plan['schedule'][index - 1] and
                row['learning_rate'] == learning_rate(plan, index) and row['optimizer_updated'] is True,
                'Checkpoint history/schedule/LR mismatch')
    require(not completed or payload['history'][-1]['state_hashes'] == payload['state_hashes'], 'Last recorded state differs')
    saved_lr = learning_rate(plan, max(1, completed))
    require(all(group['lr'] == saved_lr for group in payload['state']['optimizer']['param_groups']), 'Saved optimizer LR differs')
    require(payload['next_update'] == completed + 1 and payload['next_learning_rate'] ==
            (learning_rate(plan, completed + 1) if completed < plan['updates'] else None), 'Next global update/LR differs')
    require(payload['resume_count'] in (0, 1) and math.isfinite(payload['elapsed_training_seconds']) and
            payload['elapsed_training_seconds'] >= 0, 'Invalid resume/time ledger')
    return expected


def run_segment(plan, clips, output, stop_update=None, resume_checkpoint=None, device='cuda:0',
                remaining_time_seconds=None, prior_elapsed_seconds=None):
    began_segment = time.perf_counter()
    data = validate_plan(plan, clips)
    stop = plan['updates'] if stop_update is None else stop_update
    require(type(stop) is int and 1 <= stop <= plan['updates'], 'Invalid stop update')
    selected_device = torch.device(device)
    require(selected_device.type in ('cpu', 'cuda') and (selected_device.type != 'cuda' or torch.cuda.is_available()), 'Unavailable device')
    core.setup_runtime()
    runtime = runtime_for(selected_device)
    require(runtime == plan['runtime'], 'Runtime differs from frozen plan')
    identity = make_identity(plan, data, runtime)
    saved = load_checkpoint(resume_checkpoint) if resume_checkpoint is not None else None
    if saved is not None:
        validate_checkpoint_payload(plan, saved)
        require(saved['completed_updates'] < stop and saved['resume_count'] == 0, 'Resume must advance; at most one resume')
    start = saved['completed_updates'] if saved else 0
    resume_count = saved['resume_count'] + 1 if saved else 0
    prior_elapsed = saved['elapsed_training_seconds'] if saved else 0.
    if prior_elapsed_seconds is not None:
        require(math.isfinite(prior_elapsed_seconds) and prior_elapsed_seconds >= prior_elapsed, 'Resume cannot erase consumed time')
        prior_elapsed = prior_elapsed_seconds
    remaining = plan['time_budget_seconds'] - prior_elapsed
    if remaining_time_seconds is not None:
        require(math.isfinite(remaining_time_seconds) and remaining_time_seconds >= 0, 'Invalid remaining time')
        remaining = min(remaining, remaining_time_seconds)
    require(remaining > 0, 'Original time budget exhausted before segment')
    output = Path(output).resolve()
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), 'Output segment must be empty')
    output.mkdir(parents=True, exist_ok=True)
    model, adapter, optimizer = core.make_objects(plan['seed'], 'B_feature', selected_device)
    initial = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
    require(initial == plan['initial_state_hashes_expected'], 'Fresh initialization differs')
    history, restore = [], None
    if saved is not None:
        core.restore_state(saved['state'], model, adapter, optimizer)
        restored = runner.state_hashes(core.runtime_state(model, adapter, optimizer))
        require(restored == saved['state_hashes'], 'Exact restore failed')
        restore = {'exact': True, 'expected_hashes': saved['state_hashes'], 'actual_hashes': restored,
                   'next_update': saved['next_update'], 'next_learning_rate': saved['next_learning_rate']}
        history = list(saved['history'])
    runner._save_json(output / 'segment_contract.json', {'schema': ENGINE_SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'identity': identity, 'start_update': start, 'stop_update': stop, 'initial_state_hashes': initial,
        'resume_checkpoint': str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None,
        'resume_checkpoint_sha256': file_sha(resume_checkpoint) if resume_checkpoint else None,
        'resume_count': resume_count, 'prior_elapsed_seconds': prior_elapsed, 'remaining_time_seconds': remaining})
    state = core.runtime_state(model, adapter, optimizer)
    hashes = runner.state_hashes(state)
    checkpoints, current, status, error, attempted = [], [], 'completed' if stop == plan['updates'] else 'paused_at_declared_update', None, 0
    checkpoint_updates = set()

    def save_current(reason):
        completed = len(history)
        if completed in checkpoint_updates:
            return
        payload = {'schema': runner.SCHEMA, 'engine_schema': ENGINE_SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'identity': identity, 'completed_updates': completed, 'initial_state_hashes': initial,
                   'state_hashes': hashes, 'state': state, 'history': history,
                   'next_update': completed + 1, 'next_learning_rate': learning_rate(plan, completed + 1) if completed < plan['updates'] else None,
                   'resume_count': resume_count, 'elapsed_training_seconds': prior_elapsed + time.perf_counter() - began_segment,
                   'checkpoint_reason': reason}
        checkpoints.append(runner.save_checkpoint(output, payload))
        checkpoint_updates.add(completed)

    for index in range(start, stop):
        if time.perf_counter() - began_segment >= remaining:
            status = 'time_budget_exhausted'
            break
        batch = [clips[sid] for sid in plan['schedule'][index]]
        lr = learning_rate(plan, index + 1)
        for group in optimizer.param_groups:
            group['lr'] = lr
        before_model, before_adapter = core.tensor_hash(model.state_dict()), core.tensor_hash(adapter.state_dict())
        hook_count = len(model.projection[2]._forward_hooks)
        if selected_device.type == 'cuda':
            torch.cuda.synchronize(selected_device)
            torch.cuda.reset_peak_memory_stats(selected_device)
        started_update = time.perf_counter()
        attempted += 1
        try:
            try:
                metrics = core.optimizer_update(model, adapter, optimizer, batch, 'B_feature', selected_device)
            finally:
                require(len(model.projection[2]._forward_hooks) == hook_count, 'Feature hook leaked')
            if selected_device.type == 'cuda':
                torch.cuda.synchronize(selected_device)
            seconds = time.perf_counter() - started_update
            peak_allocated = torch.cuda.max_memory_allocated(selected_device) if selected_device.type == 'cuda' else 0
            peak_reserved = torch.cuda.max_memory_reserved(selected_device) if selected_device.type == 'cuda' else 0
            student_grad, adapter_grad = runner._gradient_summary(model.parameters()), runner._gradient_summary(adapter.parameters())
            after_model, after_adapter = core.tensor_hash(model.state_dict()), core.tensor_hash(adapter.state_dict())
            require(before_model != after_model and before_adapter != after_adapter and
                    student_grad['parameter_tensors_with_nonzero_gradient'] and adapter_grad['parameter_tensors_with_nonzero_gradient'], 'Missing model/adapter update')
            norm = math.hypot(student_grad['norm_after_clipping'], adapter_grad['norm_after_clipping'])
            require(norm <= 1.0001, 'Gradient clipping bound exceeded')
            next_state = core.runtime_state(model, adapter, optimizer)
            next_hashes = runner.state_hashes(next_state)
            row = {**metrics, 'update': index + 1, 'learning_rate': lr, 'seconds_optimizer_update': seconds,
                   'frames': [len(clip['pixels']) for clip in batch], 'student_gradients': student_grad, 'adapter_gradients': adapter_grad,
                   'gradient_norm_after_clipping': norm, 'student_changed': True, 'adapter_changed': True,
                   'feature_hook_count_before': hook_count, 'feature_hook_count_after': len(model.projection[2]._forward_hooks),
                   'model_sha256': after_model, 'adapter_sha256': after_adapter, 'cuda_peak_allocated_bytes': peak_allocated,
                   'cuda_peak_reserved_bytes': peak_reserved, 'state_hashes': next_hashes}
            state, hashes = next_state, next_hashes
            history.append(row)
            current.append(row)
            with (output / 'progress.jsonl').open('a', encoding='utf-8') as progress:
                progress.write(json.dumps({'update': row['update'], 'learning_rate': lr, 'loss': row['loss'],
                                           'state_hashes': hashes}, allow_nan=False) + '\n')
                progress.flush()
            if (index + 1) % plan['checkpoint_interval'] == 0 or index + 1 == stop:
                save_current('scheduled' if (index + 1) % plan['checkpoint_interval'] == 0 else 'declared_segment_end')
            print(json.dumps({'seed': plan['seed'], 'cohort': plan['cohort'], 'update': index + 1,
                              'learning_rate': lr, 'loss': row['loss'], 'seconds': seconds}), flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            error = {'type': type(exc).__name__, 'message': str(exc), 'attempted_global_update': index + 1,
                     'complete_updates_preserved': len(history), 'unsaved_partial_update_state_discarded': len(history) < index + 1}
            break
    save_current('final' if status == 'completed' else status)
    require(checkpoints, 'No complete state preserved')
    result = {'schema': 'lipnet-step16-training-segment-result-v1', 'status': status, 'seed': plan['seed'], 'cohort': plan['cohort'],
              'arm': 'B_feature', 'run_kind': plan['run_kind'], 'plan_sha256': data['plan_sha256'],
              'identity_sha256': runner.nested_hash(identity), 'start_update': start, 'completed_updates': len(history),
              'initial_state_hashes': initial, 'restored_from': str(Path(resume_checkpoint).resolve()) if resume_checkpoint else None,
              'restore_verification': restore, 'resume_count': resume_count, 'updates': current,
              'complete_history_updates': len(history), 'attempted_updates_this_segment': attempted,
              'failed_or_unrecorded_update_attempts': attempted - len(current), 'error': error,
              'final_state_hashes': hashes, 'checkpoint': checkpoints[-1], 'checkpoints': checkpoints, 'runtime': runtime,
              'elapsed_seconds_this_segment': time.perf_counter() - began_segment,
              'elapsed_training_seconds': prior_elapsed + time.perf_counter() - began_segment,
              'original_time_budget_seconds': plan['time_budget_seconds'], 'service_weights_updated': False}
    runner._save_json(output / 'segment_result.json', result)
    return result
