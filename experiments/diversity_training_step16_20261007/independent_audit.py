"""Independent CPU evidence audit of the fixed four-run diversity experiment.

Only hash-pinned independent auditors are imported. Production selection,
training, model and evaluation modules are never imported or executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = ROOT / 'experiments/diversity_step15_20261007'
POLICY = BASE / 'experiment_plan.json'
SEEDS, COHORTS = (42, 43), ('train24', 'train72')
CHECKPOINTS = (600, 1200, 1800, 2400, 3000, 3600)
EVAL_UPDATES = (1200, 2400, 3000, 3600)
EXPECTED15 = {'data_plan.json': '64f5cf399a7ddd5af00b858d75b72de1cad300be247e01c6597e7e787fc2eee1',
              'training_design.json': 'af14b6bd13171ad2137a4484611b70511531a09e1837133518a420ed90b53f3e',
              'results/step15_summary.json': '51525be973c428e8b3650abf375476e746f8512e791380662ab886637d944f65',
              'results/independent_audit.json': '4a1c864af0b5763e56fd9f85450b5a0a4eb7adc04c5794684c4eed72f43ed941'}
HELPERS = [
    (PREVIOUS / 'independent_audit.py', '598a5997d6c20b97a4504ef2cbe6d97b26a4a7bd4b81fe09a769073067236557'),
    (ROOT / 'experiments/memorization_step11_20261007/independent_audit.py', '2f241a622d267f4e6839c4bfc47f04a968ecea41936500cf8e0dc6dc98bbb799'),
]


def helper(name, path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Changed independent helper: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


data15 = helper('step16_independent_data15', *HELPERS[0])
old = helper('step16_independent_numerical11', *HELPERS[1])
num = old.num
read, bind, verify, sha = num.read_json, num.binding, num.verify_binding, num.sha
same, require = num.same, num.require


def save_new(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), 'Refusing to overwrite audit evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def near(actual, expected, label, tolerance=1e-10):
    require(isinstance(actual, (float, int)) and math.isfinite(actual)
            and abs(actual - expected) <= tolerance, label)


def learning_rate(update, segments):
    require(type(update) is int and update >= 1, 'Invalid global update')
    selected = [part['lr'] for part in segments if part['start_update'] <= update <= part['end_update']]
    same(len(selected), 1, 'Learning-rate schedule has a gap or overlap')
    return selected[0]


def validate_student(row, entry, item, arrays, embedded):
    """Generation-aware student cache verification, never decoding teacher pixels."""
    same(embedded['id'], row['id'], 'Student cache ID')
    same(embedded['record'], row, 'Student embedded original row')
    same(embedded['plan_sha256'], entry['logical_data_plan']['sha256'], 'Student original data generation')
    same(embedded['source_verification'], item['source_verification'], 'Student source provenance')
    pixels, tokens, indices = (arrays[key] for key in ('student_rgb', 'label_tokens', 'student_source_indices'))
    same(pixels.dtype, np.dtype('uint8'), 'Student pixel dtype')
    same(pixels.shape, (row['frames'], 96, 96, 3), 'Student pixel shape')
    raw = hashlib.sha256(pixels.tobytes()).hexdigest()
    same(raw, embedded['student_rgb_sha256'], 'Student embedded pixel hash')
    same(raw, item['student_rgb_sha256'], 'Student manifest pixel hash')
    for name, array, expected in [('tokens', tokens, row['token_ids']), ('indices', indices, data15.source_indices(row))]:
        same(array.dtype, np.dtype('int64'), 'Student '+name+' dtype')
        same(array.tolist(), expected, 'Student '+name+' values')
    require(row['frames'] >= data15.ctc_minimum(row['token_ids']), 'CTC-infeasible input')
    require(min(indices) >= 0 and max(indices) < embedded['source_frame_count'], 'Student outside original recording')
    return {'id': row['id'], 'frames': row['frames'], 'pixels_sha256': num.nested_hash(pixels),
            'tokens_sha256': num.nested_hash(row['token_ids']), 'student_raw_sha256': raw}


def audit_inputs():
    for name, expected in EXPECTED15.items():
        read(PREVIOUS / name, expected)
    data = read(PREVIOUS / 'data_plan.json')
    design = read(PREVIOUS / 'training_design.json')
    same(read(PREVIOUS / 'results/independent_audit.json')['status'], 'passed', 'Previous independent audit')
    bundle = data15.source_metadata()
    chosen = data15.independent_selection(bundle)
    same(data['pool_profile'], chosen['inventory'], 'Independent historical candidate inventory')
    for key in ('train24_ids', 'train72_ids', 'dev16_ids', 'speaker_selections'):
        same(data['selection'][key], chosen[key], 'Independent selected scope '+key)
    data15.verify_selected_separation(data['selection'], bundle)
    data15.audit_design(data, bundle, EXPECTED15['training_design.json'])
    ids = chosen['train72_ids'] + chosen['dev16_ids'] + bundle['probes']
    same(len(set(ids)), 102, 'Unique current input scope')
    selected = {row['id']: row for row in data['selection']['records']}
    cache_refs, training, evaluation, values = [], {}, {}, {}
    feature_results = []
    for sid in ids:
        row, entry, item = bundle['rows'][sid], bundle['entries'][sid], bundle['inputs'][sid]
        if sid in selected:
            same(selected[sid]['logical_record'], row, 'Selected logical original row')
            same(selected[sid]['source_entry'], entry, 'Selected handoff entry')
        verify(entry['input'])
        cache_refs.append(entry['input'])
        is_train = sid in chosen['train72_ids']
        keys = ['student_rgb', 'student_source_indices', 'label_tokens']
        if is_train:
            keys += ['teacher_source_indices', 'student_teacher_rows']
        with np.load(entry['input']['path'], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in keys}
            embedded = json.loads(str(saved['metadata_json'].item()))
        checked = validate_student(row, entry, item, arrays, embedded)
        values[sid] = arrays['student_rgb']
        role = 'train' if is_train else 'internal_dev' if sid in chosen['dev16_ids'] else 'historical_probe'
        evaluation[sid] = {key: row[key] for key in ('id', 'speaker_id', 'source_stem', 'text')}
        evaluation[sid].update(split=role, diagnostic_role=role, source_split='train',
            seen_in_current_training=row['speaker_id'] in data15.SPEAKERS,
            frames=row['frames'], pixels_sha256=checked['pixels_sha256'], tokens_sha256=checked['tokens_sha256'])
        if is_train:
            verify(entry['teacher_feature'])
            with np.load(entry['teacher_feature']['path'], allow_pickle=False) as saved:
                features = {key: saved[key] for key in saved.files if key != 'metadata_json'}
                embedded = json.loads(str(saved['metadata_json'].item()))
            feature_results.append(data15.validate_feature_arrays(row, item, bundle['features'][sid], arrays, features, embedded))
            training[sid] = {key: checked[key] for key in ('id', 'frames', 'pixels_sha256', 'tokens_sha256')}
            training[sid]['teacher_sha256'] = num.nested_hash(features['features'])
    cache_refs += [bundle['entries'][sid]['teacher_feature'] for sid in chosen['train72_ids']]
    same(len(cache_refs), 174, 'Selected input/feature binding count')
    same(len({ref['path'] for ref in cache_refs}), 174, 'Duplicate cache path')
    same(len({row['pixels_sha256'] for row in evaluation.values()}), 102, 'Duplicate input pixel payload')
    data15.old.verify_preservation(data['preservation'])
    return {'data': data, 'design': design, 'bundle': bundle, 'pixels': values,
            'training': training, 'evaluation': evaluation, 'cache_refs': cache_refs,
            'feature_checks': feature_results}


def audit_policy(inputs, expected_sha=None):
    manifest = read(BASE / 'plans/freeze_manifest.json')
    policy = read(POLICY, expected_sha or manifest['experiment_plan']['sha256'])
    same(manifest['experiment_plan'], bind(POLICY), 'Freeze experiment binding')
    same(manifest['schema'], 'lipnet-step16-freeze-v1', 'Freeze schema')
    same(manifest['status'], 'frozen', 'Freeze status')
    same(manifest['main_optimizer_updates_executed'], 0, 'Main training preceded freeze')
    same(manifest['smoke_optimizer_updates_executed'], 0, 'Smoke preceded freeze')
    for key, name in [('step15_data_plan', 'data_plan.json'), ('step15_training_design', 'training_design.json'),
                      ('step15_summary', 'results/step15_summary.json'), ('step15_audit', 'results/independent_audit.json')]:
        same(policy[key], bind(PREVIOUS / name), 'Previous stage binding')
    for key, expected in [('schema', 'lipnet-step16-execution-protocol-v1'), ('status', 'frozen_before_smoke_and_main_training'),
                          ('new_main_optimizer_updates_planned', 14400), ('new_inference_rows_planned', 824),
                          ('smoke_updates_are_additional_and_separate', True), ('service_weights_updated', False),
                          ('new_human_approvals_added', 0), ('teacher_forward_calls', 0), ('raw_media_opened', 0)]:
        same(policy[key], expected, 'Execution policy '+key)
    same(policy['runtime'], inputs['design']['runtime_reference_not_rechecked_on_gpu_in_step15'], 'Actual runtime vs fixed design')
    same(policy['preservation'], inputs['data']['preservation'], 'Service/source preservation scope')
    same(policy['data_bindings'], inputs['cache_refs'], 'Frozen exact input scope/order')
    evaluation = policy['evaluation_contract']
    for key, wanted in [('schema', 'lipnet-step16-diversity-evaluation-contract-v1'),
        ('normal_dev_updates', list(EVAL_UPDATES)), ('normal_train_updates', [2400,3600]),
        ('historical_probe_normal_updates', [3600]), ('static_shuffled_dev_updates', [3600]),
        ('expected_rows_per_cohort', {'train24':158,'train72':254}), ('expected_rows_total', 824),
        ('strata_are_reaggregation_not_extra_inference', True), ('optimization_steps',0), ('teacher_forward',False), ('audio_used',False)]:
        same(evaluation[key], wanted, 'Frozen evaluation contract '+key)
    for ref in policy['execution_code']:
        verify(ref)
    code_paths = {str(Path(ref['path']).resolve()) for ref in policy['execution_code']}
    require(str(Path(__file__).resolve()) in code_paths and str((BASE/'test_independent_audit.py').resolve()) in code_paths,
            'Independent audit code/tests were not frozen')
    plans = {}
    same(len(manifest['run_plans']), 4, 'Four run plans required')
    for spec, ref in zip(inputs['design']['runs'], manifest['run_plans'], strict=True):
        verify(ref)
        plan = read(ref['path'])
        for key, value in spec.items():
            same(plan[key], value, 'Frozen inherited design '+key)
        seed, cohort = spec['seed'], spec['cohort']
        same(Path(ref['path']).resolve(), BASE/'plans'/f'seed{seed}_{cohort}.json', 'Run plan path/order')
        for key, expected in [('ids', spec['train_ids']), ('batch_size', 2), ('checkpoint_interval', 600),
                              ('runtime', policy['runtime']), ('execution_code', policy['execution_code']),
                              ('experiment_plan', bind(POLICY)), ('data_bindings', inputs['cache_refs']),
                              ('lr_segments', spec['learning_rate_segments']), ('scope', 'production'),
                              ('run_kind', 'main'), ('time_budget_seconds', 2100), ('service_weights_updated', False),
                              ('resource_budget', inputs['design']['budget']),
                              ('dev_ids', inputs['design']['cohorts']['dev16']),
                              ('common_train24_ids', inputs['design']['cohorts']['train24']),
                              ('probe_ids', inputs['bundle']['probes'])]:
            same(plan[key], expected, 'Run execution binding '+key)
        for part, name in [('known', 'current_known_speakers'), ('unseen', 'current_unseen_speakers')]:
            same(plan['probe_'+part+'_ids'], inputs['design']['evaluation']['historical_probe14_groups'][name], 'Probe speaker scope')
        clips = {sid: inputs['training'][sid] for sid in sorted(spec['train_ids'])}
        same(plan['training_data'], {'data_sha256': num.nested_hash(clips), 'clips': clips}, 'Exact training arrays binding')
        same(plan['plan_sha256'], num.canonical_hash(plan), 'Canonical run plan hash')
        same(policy['initial_state_hashes_expected'][str(seed)], spec['initial_state_hashes_expected'], 'Actual fresh initialization vs fixed reference')
        same(Path(plan['actual_output_directory']).resolve(), BASE/'runs'/f'seed{seed}'/cohort/'continuous', 'Main output location')
        plans[seed, cohort] = plan
    require(policy['initial_state_hashes_expected']['42']['model'] != policy['initial_state_hashes_expected']['43']['model'], 'Different seeds share initialization')
    return policy, plans


def validate_optimizer(state, updates, lr):
    """Reuse independent tensor/buffer checks after separately checking saved LR."""
    saved = state['optimizer']['param_groups'][0]['lr']
    same(saved, lr, 'Saved Adam LR does not match completed global update')
    # A shallow non-mutating copy supplies the legacy checker's fixed reference LR.
    copied = {**state, 'optimizer': {**state['optimizer'], 'param_groups': [
        {**state['optimizer']['param_groups'][0], 'lr': .001}]}}
    num.validate_optimizer(copied, 'B_feature', updates)


def validate_history(history, plan, rows):
    summary = num.validate_history(history, {**plan, 'updates': len(history)}, 'B_feature', rows)
    for record in history:
        same(record['learning_rate'], learning_rate(record['update'], plan['lr_segments']), 'Actual per-update learning rate')
    return summary


def evaluation_groups(plan, update, condition):
    require(update in EVAL_UPDATES and condition in ('normal', 'static', 'shuffled'), 'Unplanned evaluation')
    if condition != 'normal':
        same(update, 3600, 'Temporal evaluation only at final checkpoint')
        return plan['dev_ids'], {'dev16': plan['dev_ids']}
    groups, ids = {}, []
    if update in (2400, 3600):
        groups.update(train_native=plan['train_ids'], train_common24=plan['common_train24_ids'])
        ids += plan['train_ids']
    groups['dev16'] = plan['dev_ids']
    ids += plan['dev_ids']
    if update == 3600:
        groups.update(historical_probe14=plan['probe_ids'], probe_known8=plan['probe_known_ids'], probe_unseen6=plan['probe_unseen_ids'])
        ids += plan['probe_ids']
    same(len(ids), len(set(ids)), 'Duplicate forward rows')
    return ids, groups


def validate_record(record, lp, source, pixels, condition, key):
    for name in ('id', 'speaker_id', 'source_stem', 'split', 'source_split', 'diagnostic_role', 'seen_in_current_training', 'frames'):
        same(record[name], source[name], 'Evaluation source role '+name)
    same(record['reference'], source['text'], 'Evaluation reference')
    same(record['condition'], condition, 'Evaluation condition')
    group = 'train_native' if source['split'] == 'train' else 'dev16' if source['split'] == 'internal_dev' else 'probe_known8' if source['seen_in_current_training'] else 'probe_unseen6'
    same(record['group'], group, 'Evaluation primary group')
    same(record['metrics'], num.metric_counts(source['text'], record['prediction']), 'Evaluation text metrics')
    tokens = record['target_token_ids']
    same(num.nested_hash(tokens), source['tokens_sha256'], 'Evaluation target token binding')
    same(record['token_metrics'], {'token_edits': num.edit_distance(tokens, record['collapsed_ids']),
         'target_tokens': len(tokens), 'exact': tokens == record['collapsed_ids']}, 'Evaluation token metrics')
    require(math.isfinite(record['seconds']) and record['seconds'] >= 0, 'Invalid inference duration')
    if record['status'] == 'failed':
        require(isinstance(record['error'], str) and bool(record['error']), 'Missing failed inference explanation')
        for name, value in [('prediction', ''), ('raw_argmax_ids', []), ('collapsed_ids', []), ('log_posteriors_key', None),
                            ('ctc_loss_mean', None), ('output_diagnostics', None)]:
            same(record[name], value, 'Failure denominator/empty prediction contract')
        require(lp is None, 'Failed row has posterior')
        return {'ctc_abs_difference': 0., 'failed': True}
    same(record['status'], 'ok', 'Unknown inference status')
    same(record['error'], None, 'Successful row has error')
    same(record['log_posteriors_key'], key, 'Posterior key/row order')
    order = num.temporal_order(source['frames'], condition, source['id'])
    temporal = record['temporal']
    same(temporal['source_positions'], order, 'Temporal permutation')
    same(temporal['source_positions_sha256'], num.nested_hash(order), 'Temporal permutation hash')
    same(temporal['frames'], source['frames'], 'Temporal frame count')
    same(temporal['static_position'], source['frames']//2 if condition == 'static' else None, 'Static source frame')
    result = old.row_diagnostics(record, lp)
    result['failed'] = False
    return result


def expected_identity(plan):
    paths = [ROOT / 'experiments/local_training_step03_20261007/runner.py',
             ROOT / 'experiments/feature_transfer_step07/train_compare.py',
             ROOT / 'backend/models/baseline.py', ROOT / 'backend/models/__init__.py',
             ROOT / 'backend/models/tokenizer.py', ROOT / 'experiments/local_training_step03_20261007/compare_replay.py',
             *(BASE / name for name in ('engine.py', 'training.py', 'test_training.py'))]
    return {'schema': 'step16-lr-training-identity-v1', 'plan_sha256': plan['plan_sha256'],
        'run_kind': plan['run_kind'], 'seed': plan['seed'], 'cohort': plan['cohort'], 'arm': 'B_feature',
        'total_updates': plan['updates'], 'batch_size': 2, 'schedule_sha256': num.nested_hash(plan['schedule']),
        'data_sha256': plan['training_data']['data_sha256'], 'clip_bindings': plan['training_data']['clips'],
        'lr_segments': plan['lr_segments'], 'code_bindings': [bind(path) for path in sorted(paths, key=str)],
        'runtime': plan['runtime'], 'optimization': {'optimizer': 'Adam', 'betas': [.9, .999], 'eps': 1e-8,
        'weight_decay': 0., 'gradient_clip': 1., 'accumulation_clips': 2, 'lambda': 1., 'ctc_device': 'cpu', 'precision': 'float32'}}


def validate_payload(payload, plan, completed, rows):
    same(payload['schema'], 'local-paired-training-checkpoint-v1', 'Complete checkpoint schema')
    same(payload['engine_schema'], 'step16_lr_training_checkpoint_v1', 'Learning-rate checkpoint schema')
    same(payload['identity'], expected_identity(plan), 'Complete checkpoint identity')
    same(payload['completed_updates'], completed, 'Complete checkpoint position')
    same(payload['initial_state_hashes'], plan['initial_state_hashes_expected'], 'Checkpoint fresh initialization')
    same(len(payload['history']), completed, 'Complete history length')
    same(payload['next_update'], completed + 1, 'Next global update')
    same(payload['next_learning_rate'], learning_rate(completed+1, plan['lr_segments']) if completed < plan['updates'] else None,
         'Next global learning rate')
    require(type(payload['resume_count']) is int and payload['resume_count'] in (0, 1), 'Resume exceeded one')
    require(math.isfinite(payload['elapsed_training_seconds']) and 0 <= payload['elapsed_training_seconds'], 'Invalid elapsed budget')
    same(payload['state_hashes'], num.state_hashes(payload['state']), 'Recomputed complete checkpoint state')
    if completed:
        same(payload['history'][-1]['state_hashes'], payload['state_hashes'], 'Final history/checkpoint state')
        same(payload['history'][-1]['model_sha256'], num.tensor_hash(payload['state']['model']), 'Student tensor hash')
        same(payload['history'][-1]['adapter_sha256'], num.tensor_hash(payload['state']['adapter']), 'Adapter tensor hash')
        validate_optimizer(payload['state'], completed, learning_rate(completed, plan['lr_segments']))
        result = validate_history(payload['history'], plan, rows)
    else:
        same(payload['state_hashes'], plan['initial_state_hashes_expected'], 'Zero-update checkpoint changed state')
        result = {'updates': 0, 'clip_presentations': 0}
    return result


def audit_segment(result_ref, plan, rows, checkpoint_cache=None):
    """Check actual per-segment histories, contracts, state snapshots and progress."""
    result = read(verify(result_ref))
    directory = Path(result_ref['path']).parent
    contract_ref = bind(directory / 'segment_contract.json')
    contract = read(contract_ref['path'])
    identity = expected_identity(plan)
    same(contract['identity'], identity, 'Segment before-training identity')
    same(contract['schema'], 'step16_lr_training_checkpoint_v1', 'Segment contract schema')
    same(contract['initial_state_hashes'], plan['initial_state_hashes_expected'], 'Segment initial state')
    for key, expected in [('schema', 'lipnet-step16-training-segment-result-v1'), ('seed', plan['seed']),
                          ('cohort', plan['cohort']), ('arm', 'B_feature'), ('run_kind', plan['run_kind']),
                          ('plan_sha256', plan['plan_sha256']), ('identity_sha256', num.nested_hash(identity)),
                          ('initial_state_hashes', plan['initial_state_hashes_expected']), ('runtime', plan['runtime']),
                          ('original_time_budget_seconds', plan['time_budget_seconds']), ('service_weights_updated', False)]:
        same(result[key], expected, 'Segment result '+key)
    start, completed = result['start_update'], result['completed_updates']
    same(contract['start_update'], start, 'Segment start position')
    require(0 <= start <= completed <= contract['stop_update'] <= plan['updates'], 'Segment position range')
    same(result['complete_history_updates'], completed, 'Segment cumulative updates')
    same(len(result['updates']), completed-start, 'Segment applied updates')
    same(result['attempted_updates_this_segment'], len(result['updates']) + result['failed_or_unrecorded_update_attempts'], 'Segment attempted update accounting')
    require(result['failed_or_unrecorded_update_attempts'] in (0, 1), 'Multiple failed update attempts in segment')
    same(result['resume_count'], contract['resume_count'], 'Resume count differs')
    require(math.isfinite(result['elapsed_seconds_this_segment']) and result['elapsed_seconds_this_segment'] >= 0, 'Segment duration')
    require(0 < contract['remaining_time_seconds'] <= plan['time_budget_seconds'] - contract['prior_elapsed_seconds'] + 1e-8,
            'Resume expanded original budget')
    require(result['elapsed_training_seconds'] >= contract['prior_elapsed_seconds'] + result['elapsed_seconds_this_segment'] - .1,
            'Lost elapsed execution time')
    cache = checkpoint_cache if checkpoint_cache is not None else {}
    snapshots = []
    previous_history = None
    for info in result['checkpoints']:
        key = info['path']
        payload = cache.get(key)
        if payload is None:
            payload = num.load_checkpoint(info)
        validate_payload(payload, plan, info['completed_updates'], rows)
        require(start <= payload['completed_updates'] <= completed, 'Checkpoint outside segment')
        if previous_history is not None:
            same(payload['history'][:len(previous_history)], previous_history, 'Checkpoint history prefix changed')
        previous_history = payload['history']
        snapshots.append({'checkpoint': info, 'completed_updates': payload['completed_updates'], 'state_hashes': payload['state_hashes']})
        # Retain hashes/history only; loading all 24 tensor states simultaneously is unnecessary.
    same(result['checkpoint'], result['checkpoints'][-1], 'Segment final checkpoint binding')
    same(payload['completed_updates'], completed, 'Final segment checkpoint position')
    same(result['final_state_hashes'], payload['state_hashes'], 'Segment final complete state')
    same(result['updates'], payload['history'][start:], 'Segment history suffix')
    if start == 0:
        same(contract['resume_checkpoint'], None, 'Fresh run imported checkpoint')
        same(result['restored_from'], None, 'Fresh result imported checkpoint')
        same(result['restore_verification'], None, 'Unexpected fresh restore evidence')
        same(result['resume_count'], 0, 'Fresh segment resume count')
    else:
        require(contract['resume_checkpoint'] and contract['resume_checkpoint_sha256'], 'Missing recovery checkpoint binding')
        same(sha(contract['resume_checkpoint']), contract['resume_checkpoint_sha256'], 'Recovery source changed')
        same(result['restored_from'], contract['resume_checkpoint'], 'Recovery source disagreement')
        resume_side = Path(contract['resume_checkpoint']).with_suffix('.pt.json')
        source_info = {**read(resume_side), 'sidecar': str(resume_side), 'sidecar_sha256': sha(resume_side)}
        source = num.load_checkpoint(source_info)
        validate_payload(source, plan, start, rows)
        same(source['resume_count'], 0, 'Second recovery disallowed')
        same(payload['history'][:start], source['history'], 'Recovery erased prior history')
        same(result['restore_verification'], {'exact': True, 'expected_hashes': source['state_hashes'],
             'actual_hashes': source['state_hashes'], 'next_update': start+1,
             'next_learning_rate': learning_rate(start+1, plan['lr_segments'])}, 'Exact restoration proof')
        same(result['resume_count'], 1, 'Resumed count')
        require(contract['prior_elapsed_seconds'] >= source['elapsed_training_seconds'], 'Recovery erased consumed time')
    progress = directory / 'progress.jsonl'
    progress_rows = [json.loads(line) for line in progress.read_text(encoding='utf-8').splitlines()] if progress.exists() else []
    same(progress_rows, [{key: row[key] for key in ('update', 'learning_rate', 'loss', 'state_hashes')} for row in result['updates']], 'Actual progress log')
    if result['status'] == 'completed':
        same(completed, plan['updates'], 'Premature completion')
        same(result['error'], None, 'Completed run has error')
        same(result['failed_or_unrecorded_update_attempts'], 0, 'Completed run discarded update attempt')
    return {'result': result, 'contract': contract, 'contract_ref': contract_ref,
            'final_payload': payload, 'checkpoints': snapshots}


REPLAY_TOLERANCES = {'model': {'atol': 1e-6, 'rtol': 1e-5}, 'adapter': {'atol': 1e-6, 'rtol': 1e-5},
    'exp_avg': {'atol': 1e-7, 'rtol': 1e-4}, 'exp_avg_sq': {'atol': 1e-10, 'rtol': 1e-4},
    'loss': {'atol': 1e-5, 'rtol': 1e-5}, 'gradient_norm': {'atol': 1e-4, 'rtol': 1e-4}}


def compare_replay_state(reference, actual, saved):
    """Independent full-state walk; RNG/scalars exact, named FP arrays bounded."""
    records = {item['path']: item for item in saved['records']}
    same(len(records), len(saved['records']), 'Duplicate replay comparison path')
    visited, differences, maximum = set(), 0, 0.

    def visit(left, right, path):
        nonlocal differences, maximum
        same(type(left), type(right), 'Replay state type '+path)
        if isinstance(left, dict):
            same(set(left), set(right), 'Replay state keys '+path)
            for name in left:
                visit(left[name], right[name], (path+'.' if path else '')+str(name))
        elif isinstance(left, (list, tuple)):
            same(len(left), len(right), 'Replay state length '+path)
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, path+'.'+str(index))
        elif isinstance(left, (np.ndarray, num.torch.Tensor)):
            a = left.detach().cpu().numpy() if isinstance(left, num.torch.Tensor) else left
            b = right.detach().cpu().numpy() if isinstance(right, num.torch.Tensor) else right
            same(a.shape, b.shape, 'Replay array shape '+path)
            same(a.dtype, b.dtype, 'Replay array dtype '+path)
            category = ('model' if path.startswith('model.') else 'adapter' if path.startswith('adapter.')
                else 'exp_avg_sq' if path.endswith('.exp_avg_sq') else 'exp_avg' if path.endswith('.exp_avg') else 'exact')
            tol = REPLAY_TOLERANCES.get(category, {'atol': 0., 'rtol': 0.})
            exact = bool(np.array_equal(a, b))
            differences += not exact
            if np.issubdtype(a.dtype, np.floating):
                require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite replay array')
                delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
                require(np.all(delta <= tol['atol']+tol['rtol']*np.abs(a.astype(np.float64))), 'Replay tolerance exceeded '+path)
                absolute = float(delta.max(initial=0))
                relative = float((delta/np.maximum(np.abs(a.astype(np.float64)), 1e-12)).max(initial=0))
            else:
                require(exact, 'Nonfloating replay state differs '+path)
                absolute = relative = 0.
            maximum = max(maximum, absolute)
            item = {'path': path, 'category': category, 'shape': list(a.shape), 'dtype': str(a.dtype),
                    'bitwise_equal': exact, 'passed': True, 'max_absolute_difference': absolute,
                    'max_relative_difference_denominator_floor_1e12': relative, **tol}
            same(records[path], item, 'Independent replay comparison row '+path)
            visited.add(path)
        else:
            same(left, right, 'Replay scalar/RNG state '+path)

    visit(reference, actual, '')
    same(visited, set(records), 'Replay compared complete array scope')
    for key, value in [('passed', True), ('bitwise_equal', differences == 0), ('arrays_checked', len(visited)),
                       ('arrays_with_bitwise_difference', differences), ('maximum_absolute_difference', maximum), ('failures', [])]:
        same(saved[key], value, 'Independent replay summary '+key)
    return {'arrays_checked': len(visited), 'maximum_absolute_difference': maximum, 'bitwise_equal': differences == 0}


def audit_smoke(plans, inputs):
    path = BASE/'results/smoke_preflight.json'
    evidence = read(path)
    plan = read(verify(evidence['smoke_plan']))
    main = plans[42, 'train24']
    ids = main['schedule'][0]
    segments = [{'start_update': 1, 'end_update': 2, 'lr': .001}, {'start_update': 3, 'end_update': 4, 'lr': .0003},
                {'start_update': 5, 'end_update': 6, 'lr': .0001}]
    for key, wanted in [('schema', 'lipnet-step16-smoke-plan-v1'), ('seed', 42), ('cohort', 'smoke_first_pair'),
        ('arm', 'B_feature'), ('run_kind', 'smoke'), ('updates', 6), ('batch_size', 2), ('checkpoint_interval', 2),
        ('ids', ids), ('schedule', [ids[:] if i%2 == 0 else ids[::-1] for i in range(6)]), ('lr_segments', segments),
        ('time_budget_seconds', 120), ('initial_state_hashes_expected', main['initial_state_hashes_expected']),
        ('runtime', main['runtime']), ('execution_code', main['execution_code']), ('experiment_plan', bind(POLICY)),
        ('production_run_plan', bind(BASE/'plans/seed42_train24.json'))]:
        same(plan[key], wanted, 'Smoke scope '+key)
    clips = {sid: inputs['training'][sid] for sid in sorted(ids)}
    same(plan['training_data'], {'data_sha256': num.nested_hash(clips), 'clips': clips}, 'Smoke input pixels/features')
    same(plan['plan_sha256'], num.canonical_hash(plan), 'Smoke canonical plan')
    for key, wanted in [('schema', 'lipnet-step16-gpu-smoke-preflight-v1'), ('status', 'passed'),
        ('experiment_plan', bind(POLICY)), ('smoke_optimizer_updates_executed', 12), ('main_optimizer_updates_executed', 0),
        ('training_clip_ids', ids), ('dev_or_probe_pixels_loaded', False), ('teacher_forward_calls', 0),
        ('service_weights_updated', False), ('caller_rng_restored_exactly', True),
        ('learning_rates', [.001, .001, .0003, .0003, .0001, .0001])]:
        same(evidence[key], wanted, 'Smoke evidence '+key)
    same(evidence['contract'], plan['smoke_contract'], 'Smoke before-execution contract')
    expected_contract = {'updates_per_trajectory':6,'continuous_updates':6,'split_first_updates':2,
        'split_resumed_updates':4,'total_executed_optimizer_updates':12,'lr_segments':segments,
        'training_clips':'first pair of frozen seed42 train24 schedule','training_data_only':True,
        'main_optimizer_updates':0,'initial_update':0,'fake_history_allowed':False,
        'restore_hashes_required_exact':True,'next_update_replay_tolerances':REPLAY_TOLERANCES,
        'tolerances_may_not_be_relaxed_after_observing_results':True}
    same(evidence['contract'], expected_contract, 'Independent fixed GPU smoke rule')
    same(evidence['contract'], read(POLICY)['smoke_contract'], 'GPU smoke rule frozen before execution')
    same(len(evidence['segments']), 3, 'Smoke continuous/split/resumed count')
    audited = []
    for name, ref in zip(('continuous', 'split', 'resumed'), evidence['segments'], strict=True):
        same(Path(ref['path']).resolve(), BASE/'smoke'/name/'segment_result.json', 'Smoke segment path/order')
        audited.append(audit_segment(ref, plan, inputs['bundle']['rows']))
    continuous, split, resumed = audited
    same([x['result']['start_update'] for x in audited], [0, 0, 2], 'Smoke real initial cursors')
    same([x['result']['completed_updates'] for x in audited], [6, 2, 6], 'Smoke real terminal cursors')
    same([x['result']['status'] for x in audited], ['completed','paused_at_declared_update','completed'], 'GPU smoke complete trajectories')
    same(sum(x['result']['failed_or_unrecorded_update_attempts'] for x in audited), 0, 'GPU smoke discarded update attempts')
    same(sum(len(x['result']['updates']) for x in audited), 12, 'Actual separate smoke optimizer updates')
    for name, item in zip(('continuous', 'split', 'resumed'), audited, strict=True):
        same(evidence[name+'_checkpoint'], item['result']['checkpoint'], 'Smoke checkpoint binding')
    same(evidence['restore_verification'], resumed['result']['restore_verification'], 'Smoke exact restoration evidence')
    replay = compare_replay_state(continuous['final_payload']['state'], resumed['final_payload']['state'], evidence['state_comparison'])
    same(len(evidence['metrics_comparisons']), 4, 'Both LR transitions replayed')
    for i, comparison in enumerate(evidence['metrics_comparisons'], 2):
        left, right = continuous['final_payload']['history'][i], resumed['final_payload']['history'][i]
        same(comparison['passed'], True, 'Smoke metric replay failed')
        for key in ('ids', 'lambda', 'optimizer_updated', 'learning_rate'):
            same(left[key], right[key], 'Smoke replay history '+key)
        for key in ('ctc_loss', 'feature_loss', 'loss', 'gradient_norm_before_clipping'):
            tol = REPLAY_TOLERANCES['gradient_norm' if key.startswith('gradient') else 'loss']
            a, b = float(left[key]), float(right[key])
            require(abs(a-b) <= tol['atol']+tol['rtol']*abs(a), 'Smoke replay metric exceeds tolerance')
            same(comparison['records'][key], {'reference': a, 'replayed': b, 'absolute_difference': abs(a-b), **tol, 'passed': True},
                 'Independent smoke metric replay')
    return {'smoke_evidence': bind(path), 'smoke_optimizer_updates': 12, 'replay': replay,
            'smoke_checkpoint_payloads_verified': sum(len(item['checkpoints']) for item in audited)}


def preflight(inputs, policy, plans):
    require(not (BASE/'results/main_jobs_clock.json').exists(), 'Main training clock preceded independent preflight')
    require(not (BASE/'runs').exists() or not any(path.is_file() for path in (BASE/'runs').rglob('*')), 'Main training files preceded independent preflight')
    require(not list((BASE/'results').glob('evaluation_*.json')), 'Inference preceded independent preflight')
    smoke = audit_smoke(plans, inputs)
    snapshot_dir = BASE/'results/preflight_code_snapshot'
    require(not snapshot_dir.exists(), 'Preflight snapshot already exists')
    snapshot_dir.mkdir(parents=True)
    snapshots = []
    refs = policy['execution_code'] + [bind(path) for path, _ in HELPERS]
    for index, ref in enumerate(refs):
        verify(ref)
        destination = snapshot_dir / (f'{index:03d}_'+Path(ref['path']).name)
        shutil.copyfile(ref['path'], destination)
        same(sha(destination), ref['sha256'], 'Preflight source copy')
        snapshots.append({'source': ref, 'snapshot': bind(destination)})
    result = {'schema': 'lipnet-step16-independent-preflight-v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': bind(POLICY),
        'run_plans': read(BASE/'plans/freeze_manifest.json')['run_plans'],
        'main_optimizer_updates_executed': 0, 'new_evaluation_inferences': 0, **smoke,
        'student_input_npz_decoded': 102, 'training_teacher_feature_npz_decoded': 72,
        'development_or_probe_teacher_features_opened': 0, 'teacher_pixel_arrays_decoded': 0,
        'paired_teacher_pixel_compressed_bytes_read_for_input_hash': True,
        'raw_media_opened': 0, 'teacher_forward_calls': 0, 'service_weights_preserved': True,
        'selected_data_and_schedules_independently_reconstructed': True,
        'exact_training_array_bindings_verified': True, 'source_code_snapshots': snapshots,
        'generator': bind(__file__), 'test_source': bind(BASE/'test_independent_audit.py'),
        'no_production_model_training_or_evaluation_modules_imported': True,
        'historical_trained_checkpoint_payloads_loaded': 0, 'audit_model_forward_calls': 0}
    return result


def audit_run(plan, inputs, preflight_time):
    directory = Path(plan['actual_output_directory'])
    reference = bind(directory/'run_result.json')
    result = read(reference['path'])
    for key, expected in [('schema', 'lipnet-step16-training-run-result-v1'), ('status', 'completed'),
        ('seed', plan['seed']), ('cohort', plan['cohort']), ('arm', 'B_feature'), ('run_kind', 'main'),
        ('start_update', 0), ('completed_updates', 3600), ('new_successful_optimizer_updates', 3600),
        ('plan_file', bind(BASE/'plans'/f"seed{plan['seed']}_{plan['cohort']}.json")),
        ('experiment_plan', bind(POLICY)), ('training_input_ids', plan['ids']),
        ('initial_state_hashes', plan['initial_state_hashes_expected']), ('runtime', plan['runtime']),
        ('dev_inputs_loaded', False), ('probe_inputs_loaded', False), ('audio_used', False),
        ('teacher_model_loaded', False), ('service_weights_updated', False),
        ('discarded_recorded_updates_from_abrupt_interruption', 0), ('actual_optimizer_step_count_fully_known', True),
        ('failed_or_unrecorded_update_attempts_total', 0)]:
        same(result[key], expected, 'Main run '+key)
    require(0 < result['wall_seconds_including_resume'] <= 2100, 'Main wall-clock budget exceeded')
    same([ref['completed_updates'] for ref in result['checkpoints']], list(CHECKPOINTS), 'Six scheduled checkpoints')
    require(1 <= len(result['segments']) <= 2, 'Missing/excess main segments')
    clock = read(directory.parent/'run_clock.json')
    same(clock['plan'], result['plan_file'], 'Run clock plan')
    same(clock['time_budget_seconds'], 2100, 'Run clock budget')
    require(datetime.fromisoformat(clock['created_utc']) > preflight_time, 'Training began before independent preflight')
    last, checks = None, []
    for index, ref in enumerate(result['segments']):
        expected_dir = directory if index == 0 else directory.parent/'resumed'
        same(Path(ref['path']).resolve(), expected_dir/'segment_result.json', 'Run segment path/order')
        audited = audit_segment(ref, plan, inputs['bundle']['rows'])
        require(datetime.fromisoformat(audited['contract']['created_utc']) > preflight_time, 'Segment predates preflight')
        attempt = read(expected_dir/'attempt_result.json')
        for key, value in [('plan_file', result['plan_file']), ('experiment_plan', bind(POLICY)), ('segment_result', ref),
                           ('student_input_npz_decoded', 72), ('training_feature_npz_decoded', 72),
                           ('dev_probe_pixels_loaded', False), ('resume', index > 0)]:
            same(attempt[key], value, 'Actual training loader scope '+key)
        if last:
            same(audited['result']['start_update'], last['result']['completed_updates'], 'Resume segment boundary')
            require(last['result']['status'] in ('failed', 'interrupted'), 'Unpermitted outcome-dependent resume')
        checks.extend(audited['checkpoints'])
        last = audited
    same(last['result']['status'], 'completed', 'Last main segment incomplete')
    payload = last['final_payload']
    same(payload['history'], result['updates'], 'Main full history vs final checkpoint')
    same(payload['state_hashes'], result['final_state_hashes'], 'Main final state')
    same(result['checkpoint'], result['final_checkpoint'], 'Final checkpoint aliases')
    same(result['checkpoint'], last['result']['checkpoint'], 'Final segment checkpoint alias')
    same(result['resume_count'], len(result['segments'])-1, 'Main recovery count')
    same(sum(len(read(ref['path'])['updates']) for ref in result['segments']), 3600, 'Actual segment successful updates')
    full_refs = [item['checkpoint'] for item in checks]
    scheduled = sorted((info for info in full_refs if info['completed_updates'] in CHECKPOINTS), key=lambda item:item['completed_updates'])
    same(scheduled, result['checkpoints'], 'All scheduled checkpoint scope')
    same([info for info in full_refs if info not in scheduled], result['recovery_checkpoints'], 'Recovery checkpoint scope')
    summary = validate_history(result['updates'], plan, inputs['bundle']['rows'])
    same(summary['per_clip_presentations'], {sid: 7200//len(plan['ids']) for sid in sorted(plan['ids'])}, 'Actual per-clip exposure count')
    same(sum(sum(row['frames']) for row in result['updates']), plan['student_frames_presented'], 'Actual frame presentation budget')
    checkpoint_states = {item['completed_updates']: item['state_hashes'] for item in checks if item['completed_updates'] in CHECKPOINTS}
    return {'run_result': reference, 'seed': plan['seed'], 'cohort': plan['cohort'], 'updates': 3600,
            'checkpoint_count': 6, 'checkpoint_states': checkpoint_states, 'result': result,
            'history_summary': summary, 'resume_count': result['resume_count'],
            'student_frames_presented': plan['student_frames_presented']}


def audit_evaluation(plan, training, inputs, policy):
    path = BASE/'results'/f"evaluation_seed{plan['seed']}_{plan['cohort']}.json"
    result = read(path)
    wanted_ids = plan['train_ids']+plan['dev_ids']+plan['probe_ids']
    clips = {sid: inputs['evaluation'][sid] for sid in sorted(wanted_ids)}
    bound_inputs = {'clips': clips, 'data_sha256': num.nested_hash(clips)}
    for key, expected in [('schema', 'lipnet-step16-diversity-evaluation-v1'), ('status', 'completed'),
        ('seed', plan['seed']), ('cohort', plan['cohort']), ('arm', 'B_feature'), ('experiment_plan', bind(POLICY)),
        ('original_plan', bind(BASE/'plans'/f"seed{plan['seed']}_{plan['cohort']}.json")),
        ('original_run_result', training['run_result']), ('plan_sha256', plan['plan_sha256']),
        ('training_data_sha256', plan['training_data']['data_sha256']), ('initial_state_hashes_verified', plan['initial_state_hashes_expected']),
        ('evaluation_inputs', bound_inputs), ('runtime', policy['runtime']), ('evaluation_contract', policy['evaluation_contract']),
        ('optimization_steps', 0), ('reused_inference_rows', 0), ('teacher_forward', False), ('audio_used', False),
        ('teacher_feature_caches_loaded', 0), ('student_input_caches_loaded', 102), ('student_inputs_inferred', len(wanted_ids)),
        ('student_inputs_loaded_but_not_inferred', 102-len(wanted_ids)), ('service_weights_updated', False),
        ('caller_rng_restored_exactly', True), ('evaluation_inputs_unchanged', True), ('unseen_speaker_generalization_estimate', False),
        ('input_load_counts', {'student_input_npz_decoded': 102, 'training_feature_npz_decoded': 0,
          'dev_or_probe_teacher_features_opened': 0, 'teacher_pixel_arrays_decoded': 0, 'raw_media_opened': 0})]:
        same(result[key], expected, 'Evaluation identity/scope '+key)
    contract = read(verify(result['before_inference_contract']))
    same(contract['schema'], 'lipnet-step16-before-inference-contract-v1', 'Evaluation before-inference contract schema')
    same(contract['policy'], bind(POLICY), 'Evaluation contract policy alias')
    for key in ('seed', 'cohort', 'arm', 'experiment_plan', 'original_plan', 'original_run_result',
                'evaluation_inputs', 'runtime', 'evaluation_code', 'checkpoint_validations', 'input_load_counts'):
        same(contract[key], result[key], 'Evaluation before-inference contract '+key)
    same(contract['independent_preflight'], bind(BASE/'results/independent_preflight.json'), 'Evaluation preflight binding')
    same(contract['gpu_smoke_preflight'], bind(BASE/'results/smoke_preflight.json'), 'Evaluation smoke binding')
    same(contract['checkpoints'], training['result']['checkpoints'], 'Evaluation complete checkpoint scope')
    require(datetime.fromisoformat(contract['created_utc']) < datetime.fromisoformat(result['created_utc']), 'Inference contract saved after result')
    for ref in result['evaluation_code']:
        verify(ref)
        require(ref in policy['execution_code'], 'Evaluation code not frozen')
    same(len(result['checkpoint_validations']), 6, 'Six checkpoints validated before inference')
    for evidence, info in zip(result['checkpoint_validations'], training['result']['checkpoints'], strict=True):
        update = info['completed_updates']
        same(evidence['checkpoint'], info, 'Evaluation checkpoint validation binding')
        same(evidence['state_hashes_verified'], training['checkpoint_states'][update], 'Evaluation checked full state')
        same(evidence['history_updates_verified'], update, 'Evaluation checkpoint history prefix')
        same(evidence['identity_sha256'], training['result']['identity_sha256'], 'Evaluation checkpoint identity')
        same(evidence['next_update'], update+1, 'Evaluation checkpoint next cursor')
        same(evidence['next_learning_rate'], learning_rate(update+1, plan['lr_segments']) if update<3600 else None, 'Evaluation checkpoint next LR')
        same(evidence['restricted_cpu_load'], True, 'Unrestricted checkpoint deserialization')
    same([item['completed_updates'] for item in result['milestones']], list(EVAL_UPDATES), 'Fixed evaluated updates')
    archive_ref = result['log_posteriors_archive']
    verify(archive_ref)
    keys, count, failures, frames, max_error = set(), 0, 0, 0, 0.
    with np.load(archive_ref['path'], allow_pickle=False) as archive:
        for milestone in result['milestones']:
            update = milestone['completed_updates']
            info = next(ref for ref in training['result']['checkpoints'] if ref['completed_updates'] == update)
            same(milestone['checkpoint'], info, 'Evaluated checkpoint binding')
            for key in ('model_state_before', 'model_state_after'):
                same(milestone[key], training['checkpoint_states'][update]['model'], 'Evaluation model state before/after')
            same(milestone['model_state_unchanged'], True, 'Evaluation mutated model')
            same(set(milestone['conditions']), {'normal', 'static', 'shuffled'} if update == 3600 else {'normal'}, 'Fixed conditions')
            for condition, value in milestone['conditions'].items():
                ids, groups = evaluation_groups(plan, update, condition)
                records = value['records']
                same([row['id'] for row in records], ids, 'Evaluation fixed row order/denominator')
                count += len(records)
                for position, record in enumerate(records):
                    key = f'u{update:04d}_{condition}_{position}'
                    successful = record['status'] == 'ok'
                    if successful:
                        require(key not in keys, 'Duplicate posterior evidence')
                        keys.add(key)
                        frames += record['frames']
                    else:
                        failures += 1
                    sid = record['id']
                    checked = validate_record(record, archive[key] if successful else None, clips[sid], inputs['pixels'][sid], condition, key)
                    max_error = max(max_error, checked['ctc_abs_difference'])
                same(set(value['groups']), set(groups), 'Evaluation aggregate scope')
                same(set(value['per_speaker']), set(groups), 'Per-speaker aggregate scope')
                for name, group_ids in groups.items():
                    selected = [row for row in records if row['id'] in group_ids]
                    same(value['groups'][name], old.aggregate_condition(selected), 'Independent group metrics '+name)
                    speakers = sorted({row['speaker_id'] for row in selected})
                    same(set(value['per_speaker'][name]), set(speakers), 'Per-speaker denominator')
                    for speaker in speakers:
                        same(value['per_speaker'][name][speaker], old.aggregate_condition([row for row in selected if row['speaker_id'] == speaker]),
                             'Independent per-speaker metrics')
        same(set(archive.files), keys, 'Missing or additional saved posteriors')
    for key, expected in [('record_count', count), ('new_inference_rows', count), ('failed_rows', failures), ('posterior_arrays', len(keys))]:
        same(result[key], expected, 'Evaluation row accounting '+key)
    same(count, 2*len(plan['train_ids'])+110, 'Predeclared evaluation count')
    return {'evaluation': bind(path), 'result': result, 'seed': plan['seed'], 'cohort': plan['cohort'],
            'record_count': count, 'failed_inference_rows': failures, 'posterior_arrays': len(keys),
            'posterior_frames': frames, 'max_ctc_abs_difference': max_error}


def audit_comparison(evaluations, inputs):
    reference = bind(BASE/'results/comparison.json')
    result = read(reference['path'])
    for key, wanted in [('schema', 'lipnet-step16-diversity-comparison-v1'), ('status', 'completed'),
        ('experiment_plan', bind(POLICY)), ('evaluation_runs', [item['evaluation'] for item in evaluations]),
        ('generator', bind(BASE/'summarize.py')), ('inference_rows', 824), ('final_checkpoint_selected', 3600),
        ('used_best_intermediate_checkpoint', False), ('new_main_optimizer_updates', 14400),
        ('same_updates_not_same_frames_or_wall_time', True), ('service_weights_updated', False),
        ('unseen_speaker_or_untouched_test_claim', False)]:
        same(result[key], wanted, 'Comparison identity/scope '+key)
    models = []
    for item in evaluations:
        value = item['result']
        final = value['milestones'][-1]
        models.append({'seed': item['seed'], 'cohort': item['cohort'], 'final_update': 3600,
            'final_normal': final['conditions']['normal']['groups'],
            'final_dev_conditions': {condition: final['conditions'][condition]['groups']['dev16'] for condition in ('normal', 'static', 'shuffled')},
            'final_normal_per_speaker': final['conditions']['normal']['per_speaker'],
            'dev_learning_curve': [{'update': m['completed_updates'], 'metrics': m['conditions']['normal']['groups']['dev16']} for m in value['milestones']],
            'training_learning_curve': [{'update': m['completed_updates'], 'train_native': m['conditions']['normal']['groups']['train_native'],
                'train_common24': m['conditions']['normal']['groups']['train_common24']} for m in value['milestones'] if m['completed_updates'] in (2400, 3600)]})
    same(result['models'], models, 'Comparison all final and intermediate metrics')
    paired = []
    for seed in SEEDS:
        a, b = [next(model for model in models if model['seed']==seed and model['cohort']==cohort)['final_normal']['dev16']['aggregate'] for cohort in COHORTS]
        paired.append({'seed': seed, 'train24_final_dev_cer': a['cer_percent'], 'train72_final_dev_cer': b['cer_percent'],
            'delta_cer_percentage_points': b['cer_percent']-a['cer_percent'], 'train24_final_dev_wer': a['wer_percent'],
            'train72_final_dev_wer': b['wer_percent'], 'train24_final_dev_exact': a['exact_sentences'], 'train72_final_dev_exact': b['exact_sentences']})
    same(result['paired_final_dev16'], paired, 'Independent paired final-development comparison')
    failures = sum(item['failed_inference_rows'] for item in evaluations)
    same(result['failed_inference_rows'], failures, 'Comparison failure denominator')
    deltas = [row['delta_cer_percentage_points'] for row in paired]
    mean = sum(deltas)/2
    gate = {'seed_deltas_cer_percentage_points': deltas, 'mean_delta_cer_percentage_points': mean,
        'both_seeds_improved': all(delta<0 for delta in deltas), 'mean_improvement_at_least_5_percentage_points': mean<=-5.,
        'failed_inferences': failures, 'passed': failures==0 and all(delta<0 for delta in deltas) and mean<=-5.,
        'statistical_significance_established': False, 'webcam_readiness_established': False}
    same(result['predeclared_followup_expansion_gate'], gate, 'Predeclared research gate')
    baseline = result['known_speaker_baseline']
    anchors = {}
    rows = inputs['bundle']['rows']
    for sid in inputs['design']['cohorts']['train24']:
        anchors.setdefault(rows[sid]['speaker_id'], sid)
    records = []
    for sid in inputs['design']['cohorts']['dev16']:
        anchor = anchors[rows[sid]['speaker_id']]
        ref = data15.unicodedata.normalize('NFC', rows[sid]['text']).strip()
        pred = data15.unicodedata.normalize('NFC', rows[anchor]['text']).strip()
        metrics = num.metric_counts(ref,pred)
        records.append({'id': sid, 'anchor_id': anchor, 'reference': ref, 'prediction': pred, 'status': 'ok',
            'metrics': {key:metrics[key] for key in ('character_edits','reference_characters','word_edits','reference_words','exact')}})
    same(baseline['records'], records, 'Known-speaker fixed sentence baseline')
    aggregate = num.aggregate_metrics(records)
    same(baseline['aggregate'], {key:aggregate[key] for key in ('character_edits','reference_characters','word_edits','reference_words',
        'clips','failed_inputs','cer_percent','wer_percent','exact_sentences')}, 'Baseline text metric aggregation')
    for key, wanted in [('kind', 'known_speaker_fixed_control_training_sentence'), ('uses_dev_targets_to_choose_anchor', False),
                        ('is_deployable_lipreading_decoder', False), ('neural_inference_rows', 0)]:
        same(baseline[key], wanted, 'Baseline interpretation '+key)
    return reference, paired, gate


def final_audit(inputs, policy, plans):
    preflight_path = BASE/'results/independent_preflight.json'
    before = read(preflight_path)
    same(before['status'], 'passed', 'Missing passed preflight')
    same(before['experiment_plan'], bind(POLICY), 'Preflight policy changed')
    same(before['main_optimizer_updates_executed'], 0, 'Preflight occurred after main training')
    same(before['run_plans'], read(BASE/'plans/freeze_manifest.json')['run_plans'], 'Preflight run plans changed')
    same(before['smoke_evidence'], bind(BASE/'results/smoke_preflight.json'), 'Smoke changed after preflight')
    same([pair['source'] for pair in before['source_code_snapshots']], policy['execution_code']+[bind(path) for path,_ in HELPERS],
         'Complete frozen source-code snapshot scope')
    for pair in before['source_code_snapshots']:
        verify(pair['source']); verify(pair['snapshot'])
        same(pair['source']['sha256'], pair['snapshot']['sha256'], 'Preflight source snapshot mismatch')
    before_time = datetime.fromisoformat(before['created_utc'])
    clock = read(BASE/'results/main_jobs_clock.json')
    same(clock['experiment_plan'], bind(POLICY), 'Global job clock binding')
    same(clock['time_budget_seconds'], 9000, 'Global job time budget')
    require(datetime.fromisoformat(clock['created_utc']) > before_time, 'Global jobs began before independent preflight')
    runs, evaluations = [], []
    for plan in plans.values():
        run = audit_run(plan, inputs, before_time)
        runs.append(run)
        evaluations.append(audit_evaluation(plan, run, inputs, policy))
    completed = read(BASE/'results/training_complete.json')
    for key, wanted in [('schema', 'lipnet-step16-training-complete-v1'), ('status', 'completed'),
        ('experiment_plan', bind(POLICY)), ('runs', [run['run_result'] for run in runs]),
        ('new_main_optimizer_updates', 14400), ('separate_smoke_optimizer_updates', 12),
        ('discarded_recorded_updates_from_abrupt_interruption', 0), ('service_weights_updated', False),
        ('failed_or_unrecorded_update_attempts_total', 0), ('actual_optimizer_step_count_fully_known', True)]:
        same(completed[key], wanted, 'Completed training inventory '+key)
    comparison_ref, paired, gate = audit_comparison(evaluations, inputs)
    data15.old.verify_preservation(policy['preservation'])
    return {'schema': 'lipnet-step16-independent-audit-v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_plan': bind(POLICY),
        'independent_preflight': bind(preflight_path), 'comparison': comparison_ref,
        'main_optimizer_updates': sum(run['updates'] for run in runs), 'smoke_optimizer_updates': 12,
        'discarded_recorded_optimizer_updates': 0, 'checkpoints_verified': sum(run['checkpoint_count'] for run in runs),
        'inference_rows': sum(item['record_count'] for item in evaluations),
        'failed_inference_rows': sum(item['failed_inference_rows'] for item in evaluations),
        'posterior_arrays': sum(item['posterior_arrays'] for item in evaluations),
        'posterior_frames': sum(item['posterior_frames'] for item in evaluations),
        'max_ctc_abs_difference': max(item['max_ctc_abs_difference'] for item in evaluations),
        'training_runs': [{key:value for key,value in run.items() if key not in ('result',)} for run in runs],
        'evaluation_runs': [{key:value for key,value in item.items() if key!='result'} for item in evaluations],
        'paired_final_dev16': paired, 'predeclared_followup_expansion_gate': gate,
        'student_input_npz_decoded': 102, 'training_teacher_feature_npz_decoded': 72,
        'development_or_probe_teacher_features_opened': 0, 'teacher_pixel_arrays_decoded': 0,
        'paired_teacher_pixel_compressed_bytes_read_for_input_hash': True,
        'raw_media_opened': 0, 'teacher_forward_calls': 0, 'audit_model_forward_calls': 0,
        'historical_trained_checkpoint_payloads_loaded': 0, 'service_weights_preserved': True,
        'source_code_snapshots_preserved': True, 'new_human_approvals_added': 0,
        'generator': bind(__file__), 'test_source': bind(BASE/'test_independent_audit.py'),
        'no_production_model_training_or_evaluation_modules_imported': True,
        'limitations': ['This audit verifies recorded training/output evidence; it is not an independent GPU training rerun.',
          'The known-speaker development set and repeatedly inspected historical probes are not untouched tests.',
          'The practical expansion gate is neither statistical significance nor a deployment approval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--expected-policy-sha')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or BASE/'results'/('independent_preflight.json' if args.preflight else 'independent_audit.json')
    require(not output.exists(), 'Audit output already exists')
    started = time.perf_counter()
    own_sources = [bind(__file__), bind(BASE/'test_independent_audit.py'), *[bind(path) for path, _ in HELPERS]]
    inputs = audit_inputs()
    policy, plans = audit_policy(inputs, args.expected_policy_sha)
    result = preflight(inputs, policy, plans) if args.preflight else final_audit(inputs, policy, plans)
    for ref in own_sources + policy['execution_code'] + inputs['cache_refs']:
        verify(ref)
    data15.old.verify_preservation(policy['preservation'])
    same(bind(POLICY), result['experiment_plan'], 'Policy changed during audit')
    result.update(elapsed_seconds=time.perf_counter()-started, audit_sources=own_sources,
                  all_current_source_bindings_unchanged_before_and_after_audit=True)
    reference = save_new(output, result)
    print(json.dumps({'status': 'passed', 'audit': reference}, ensure_ascii=False))


if __name__ == '__main__':
    main()



