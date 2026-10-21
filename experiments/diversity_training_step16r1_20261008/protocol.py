"""Bind the Step15 design to current inputs, runtime and fresh student states."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = ROOT / 'experiments/diversity_step15_20261007'
POLICY = BASE / 'experiment_plan.json'
MANIFEST = BASE / 'plans/freeze_manifest.json'
SEEDS, COHORTS = (42, 43), ('train24', 'train72')
EXPECTED = {'data_plan.json': '64f5cf399a7ddd5af00b858d75b72de1cad300be247e01c6597e7e787fc2eee1',
            'training_design.json': 'af14b6bd13171ad2137a4484611b70511531a09e1837133518a420ed90b53f3e',
            'results/step15_summary.json': '51525be973c428e8b3650abf375476e746f8512e791380662ab886637d944f65',
            'results/independent_audit.json': '4a1c864af0b5763e56fd9f85450b5a0a4eb7adc04c5794684c4eed72f43ed941'}
CORE_SHA = 'eb65e48cf6d4d35825feee1c3d0bd885952357c50c011049944358081790c62a'
RUNNER_SHA = '71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693'


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


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == ref, 'Changed or nonlocal evidence: '+str(path))
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


require(bind(ROOT / 'experiments/local_training_step03_20261007/runner.py')['sha256'] == RUNNER_SHA and
        bind(ROOT / 'experiments/feature_transfer_step07/train_compare.py')['sha256'] == CORE_SHA,
        'Frozen training primitives changed')
legacy = module('step16_legacy_checkpoint_helpers', ROOT / 'experiments/local_training_step03_20261007/runner.py')
runner, core = legacy, legacy.core
# These pure metadata helpers are themselves recorded in Step15's source bindings.
_previous = read(PREVIOUS / 'data_plan.json')
require(bind(PREVIOUS / 'data_plan.json')['sha256'] == EXPECTED['data_plan.json'], 'Step15 data changed')
for _ref in _previous['execution_code']:
    verify(_ref)
data15 = module('step16_frozen_data15', PREVIOUS / 'data_plan.py')
design15 = module('step16_frozen_design15', PREVIOUS / 'experiment_plan.py')


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def metadata():
    for name, expected in EXPECTED.items():
        require(bind(PREVIOUS / name)['sha256'] == expected, 'Step15 artifact changed: '+name)
    data = data15.verify_plan()
    design = read(PREVIOUS / 'training_design.json')
    design15.validate_plan(design, data, read(design15.PRIOR))
    require(read(PREVIOUS / 'results/independent_audit.json')['status'] == 'passed', 'Step15 audit incomplete')
    bundle = data15.source_metadata()
    groups = design['cohorts']
    probes = design['evaluation']['historical_probe14_ids']
    require(set(groups['train72']).isdisjoint(groups['dev16']+probes), 'Training/evaluation overlap')
    records = {row['id']: row for row in data['selection']['records']}
    for sid in probes:
        row, entry = bundle['rows'][sid], bundle['entries'][sid]
        records[sid] = {'id': sid, 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                        'generation': entry['generation'], 'role': 'historical_probe', 'text': row['text'],
                        'tokens': row['token_ids'], 'frames': row['frames'], 'logical_record': row,
                        'source_entry': entry, 'input': entry['input'], 'teacher_feature': None,
                        'new_human_approval_added': False}
    require(len(records) == 102, 'Expected72+16+14 unique input clips')
    return {'data': data, 'design': design, 'records': records, 'bundle': bundle}


def validate_student(record, pixels, tokens, indices, embedded):
    row = record['logical_record']
    require(pixels.dtype == np.uint8 and pixels.shape == (record['frames'], 96, 96, 3), 'Student shape/dtype mismatch')
    require(tokens.dtype == indices.dtype == np.int64 and tokens.tolist() == record['tokens'], 'Student token mismatch')
    require(tokens.ndim == 1 and len(tokens) > 0 and np.all((tokens >= 1) & (tokens < 69)), 'Invalid nonblank targets')
    require(embedded['record'] == row and embedded['id'] == record['id'], 'Embedded input provenance mismatch')
    expected = [round((row['start_time']+i/25)*row['video_fps']) for i in range(record['frames'])]
    require(indices.tolist() == expected and hashlib.sha256(pixels.tobytes()).hexdigest() == embedded['student_rgb_sha256'],
            'Student frame alignment or pixels changed')
    require(record['frames'] >= len(tokens)+sum(a == b for a,b in zip(tokens, tokens[1:])), 'CTC target infeasible')


def load_data(mode='all'):
    require(mode in ('all', 'train', 'evaluation'), 'Unknown data-loading scope')
    meta = metadata()
    groups = meta['design']['cohorts']
    scope = {'train': groups['train72'], 'dev': groups['dev16'],
             'probe': meta['design']['evaluation']['historical_probe14_ids']}
    result = {'train': {}, 'dev': {}, 'probe': {}, 'metadata': meta['data'], 'design': meta['design']}
    for group, ids in scope.items():
        if mode == 'train' and group != 'train':
            continue
        for sid in ids:
            record = meta['records'][sid]
            verify(record['input'])
            with np.load(record['input']['path'], allow_pickle=False) as saved:
                pixels, tokens, indices = saved['student_rgb'], saved['label_tokens'], saved['student_source_indices']
                embedded = json.loads(str(saved['metadata_json'].item()))
            validate_student(record, pixels, tokens, indices, embedded)
            role = 'train' if group == 'train' else 'dev' if group == 'dev' else 'historical_probe'
            clip = {'id': sid, 'split': role, 'diagnostic_role': role, 'source_split': 'train', 'original_split': 'train',
                    'generation': record['generation'], 'speaker_id': record['speaker_id'], 'source_stem': record['source_stem'],
                    'text': record['text'], 'tokens': tokens.tolist(), 'pixels': pixels, 'input': record['input']}
            if group == 'train' and mode != 'evaluation':
                verify(record['teacher_feature'])
                with np.load(record['teacher_feature']['path'], allow_pickle=False) as saved:
                    feature, aligned = saved['features'], saved['student_source_indices']
                    feature_meta = json.loads(str(saved['metadata_json'].item()))
                require(feature.dtype == np.float32 and feature.shape == (len(pixels), 512) and
                        np.isfinite(feature).all() and (np.linalg.norm(feature,axis=1)>1e-8).all(), 'Invalid training features')
                require(np.array_equal(aligned, indices) and feature_meta['id'] == sid and
                        hashlib.sha256(feature.tobytes()).hexdigest() == feature_meta['feature_values_sha256'],
                        'Feature alignment or values changed')
                clip['teacher_features'] = feature
            result[group][sid] = clip
    result['counts'] = {'student_input_npz_decoded': sum(len(result[group]) for group in scope),
                        'training_feature_npz_decoded': len(result['train']) if mode != 'evaluation' else 0,
                        'dev_or_probe_teacher_features_opened': 0, 'teacher_pixel_arrays_decoded': 0,
                        'raw_media_opened': 0}
    return result


def data_bindings(meta):
    refs = [record['input'] for record in meta['records'].values()]
    refs += [record['teacher_feature'] for record in meta['records'].values() if record['role'] == 'train']
    require(len(refs) == 174 and len({ref['path'] for ref in refs}) == 174, 'Changed bound cache scope')
    for ref in refs:
        verify(ref)
    return refs


def plan_path(seed, cohort):
    require(seed in SEEDS and cohort in COHORTS, 'Unknown planned run')
    return BASE / 'plans' / f'seed{seed}_{cohort}.json'


def run_path(seed, cohort):
    return BASE / 'runs' / f'seed{seed}' / cohort / 'continuous'


def run_plan(seed, cohort):
    return read(plan_path(seed, cohort))


def plan_template(spec, policy, clips):
    design = read(policy['step15_training_design']['path'])
    plan = {**copy.deepcopy(spec), 'batch_size': 2, 'checkpoint_interval': 600,
            'ids': spec['train_ids'], 'lr_segments': spec['learning_rate_segments'],
            'common_train24_ids': design['cohorts']['train24'], 'dev_ids': design['cohorts']['dev16'],
            'probe_ids': design['evaluation']['historical_probe14_ids'],
            'probe_known_ids': design['evaluation']['historical_probe14_groups']['current_known_speakers'],
            'probe_unseen_ids': design['evaluation']['historical_probe14_groups']['current_unseen_speakers'],
            'runtime': policy['runtime'], 'execution_code': policy['execution_code'],
            'experiment_plan': bind(POLICY), 'data_bindings': policy['data_bindings'],
            'initial_state_hashes_expected': policy['initial_state_hashes_expected'][str(spec['seed'])],
            'actual_output_directory': str(run_path(spec['seed'], spec['cohort'])),
            'resource_budget': design['budget'], 'scope': 'production', 'service_weights_updated': False}
    plan.update(run_kind='main', time_budget_seconds=2100)
    binding = legacy.validate_plan(plan, clips, 'B_feature')
    plan['training_data'] = {key: binding[key] for key in ('data_sha256', 'clips')}
    plan['plan_sha256'] = legacy.canonical_plan_hash(plan)
    return plan


def freeze(extra_code_paths=()):
    require(not POLICY.exists() and not MANIFEST.exists(), 'Step16 already frozen')
    meta = metadata()
    import deterministic_pool, engine
    deterministic_pool.setup_runtime(core)
    require(torch.cuda.is_available(), 'CUDA required for the predeclared experiment')
    runtime = engine.runtime_for(torch.device('cuda:0'))
    expected_runtime = copy.deepcopy(meta['design']['runtime_reference_not_rechecked_on_gpu_in_step15'])
    expected_runtime['deterministic_algorithms'] = True
    expected_runtime['strict_deterministic_algorithms'] = True
    expected_runtime['deterministic_warn_only'] = False
    require(runtime == expected_runtime, 'GPU runtime differs from amended reference')
    initial = {}
    for seed in SEEDS:
        objects = core.make_objects(seed, 'B_feature', 'cuda:0')
        deterministic_pool.configure_model(objects[0])
        initial[str(seed)] = legacy.state_hashes(core.runtime_state(*objects))
        expected = next(run['initial_state_hashes_expected'] for run in meta['design']['runs'] if run['seed'] == seed)
        require(initial[str(seed)] == expected, 'Fresh initial state differs from Step15 design')
        del objects
    gc.collect()
    torch.cuda.empty_cache()
    import engine, evaluate
    paths = {Path(ref['path']) for ref in legacy._code_bindings()+evaluate.evaluation_code()+engine.execution_code()}
    paths.update({Path(__file__), BASE/'test_protocol.py', *(Path(path).resolve() for path in extra_code_paths)})
    codes = [bind(path) for path in sorted(paths,key=str)]
    policy = {'schema': 'lipnet-step16-execution-protocol-v1', 'status': 'frozen_before_smoke_and_main_training',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'step15_data_plan': bind(PREVIOUS/'data_plan.json'),
              'step15_training_design': bind(PREVIOUS/'training_design.json'), 'step15_summary': bind(PREVIOUS/'results/step15_summary.json'),
              'step15_audit': bind(PREVIOUS/'results/independent_audit.json'), 'runtime': runtime,
              'initial_state_hashes_expected': initial, 'execution_code': codes, 'data_bindings': data_bindings(meta),
              'preservation': meta['data']['preservation'], 'evaluation_contract': evaluate.evaluation_contract(),
              'smoke_contract': engine.smoke_contract(),
              'new_main_optimizer_updates_planned': 14400, 'new_inference_rows_planned': 824,
              'smoke_updates_are_additional_and_separate': True, 'service_weights_updated': False,
              'new_human_approvals_added': 0, 'teacher_forward_calls': 0, 'raw_media_opened': 0}
    previous = ROOT/'experiments/diversity_training_step16_20261007'
    prior_policy = bind(previous/'experiment_plan.json')
    prior_smoke = bind(previous/'results/smoke_preflight.json')
    require(prior_policy['sha256']=='4868b5145457629c6baf326e6ad886f268f7783954cd188ebaf53d913c309742', 'Original failed experiment changed')
    prior = read(prior_smoke['path'])
    require(prior['status']=='failed' and prior['smoke_optimizer_updates_executed']==12 and
            prior['main_optimizer_updates_executed']==0, 'Original failure evidence differs')
    policy['amendment'] = {'reason':'deterministic_spatial_pooling_after_failed_gpu_replay',
                           'previous_policy':prior_policy, 'previous_smoke':prior_smoke,
                           'prior_optimizer_updates':12, 'numerical_tolerances_relaxed':False,
                           'mathematical_pooling_unchanged':True}
    pool_verification = BASE/'results/deterministic_pool_verification.json'
    require(read(pool_verification)['status']=='passed', 'Deterministic pooling not verified')
    policy['pooling_contract'] = deterministic_pool.pooling_contract()
    policy['pooling_verification'] = bind(pool_verification)
    save_new(POLICY, policy)
    loaded = load_data('train')
    refs = []
    for spec in meta['design']['runs']:
        clips = {sid: loaded['train'][sid] for sid in spec['train_ids']}
        refs.append(save_new(plan_path(spec['seed'],spec['cohort']), plan_template(spec, policy, clips)))
    manifest = {'schema': 'lipnet-step16-freeze-v1', 'status': 'frozen', 'experiment_plan': bind(POLICY), 'run_plans': refs,
                'main_optimizer_updates_executed': 0, 'smoke_optimizer_updates_executed': 0}
    save_new(MANIFEST, manifest)
    return manifest


def verify_frozen():
    manifest = read(MANIFEST)
    require(manifest['schema'] == 'lipnet-step16-freeze-v1' and manifest['status'] == 'frozen', 'Invalid freeze')
    policy = read(verify(manifest['experiment_plan']))
    require(policy['schema'] == 'lipnet-step16-execution-protocol-v1', 'Unexpected Step16 policy')
    for ref in policy['execution_code']+policy['data_bindings']+manifest['run_plans']:
        verify(ref)
    for key in ('step15_data_plan','step15_training_design','step15_summary','step15_audit'):
        verify(policy[key])
    for key in ('previous_policy','previous_smoke'):
        verify(policy['amendment'][key])
    verify(policy['pooling_verification'])
    data15.verify_preservation(policy['preservation'])
    require(len(manifest['run_plans']) == 4 and policy['new_main_optimizer_updates_planned'] == 14400 and
            policy['new_inference_rows_planned'] == 824, 'Execution budget changed')
    return policy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze','verify'))
    parser.add_argument('--extra-code',action='append',default=[])
    args = parser.parse_args()
    result = freeze(args.extra_code) if args.command == 'freeze' else verify_frozen()
    print(json.dumps({'status': result.get('status'),'experiment_plan': bind(POLICY)},ensure_ascii=False))
