"""Immutable, train-only eight-utterance memorization comparison contract.

The validated Step10 loader opens the existing 510 permitted training caches;
only the deterministically selected eight are retained or scheduled. No raw
media, development/held payload, teacher forward, or service update is used.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
POLICY = BASE / 'diagnostic_plan.json'
FROZEN = BASE / 'plans/freeze_manifest.json'
SEEDS = (42, 43)
ARMS = ('A_ctc', 'B_feature')
UPDATES = 1200
CHECKPOINT_INTERVAL = 300
MILESTONES = (0, 300, 600, 900, 1200)
ROLE_GENERATIONS = {'old_train': 'old', 'new_provisional_train': 'new'}
SELECTION_SALT = 'step11-memorization-v1:'
ANCHORS = {
    'experiments/local_training_step03_20261007/runner.py': '71ef1e815ce21a0e52b2aff37257f570c4b07fa81531de9f38e1681c695e8693',
    'experiments/feature_transfer_step07/train_compare.py': 'eb65e48cf6d4d35825feee1c3d0bd885952357c50c011049944358081790c62a',
    'experiments/expansion_step10_20261007/mixed_artifacts.py': 'aef1a1cc7d2921934746105065de94a339bc1223795e2c80c10983f03f796f44',
    'experiments/expansion_step10_20261007/comparison_plan.json': '93a90fefc66a1152cfa2f9b3a0e00de3a90f506ae5ef0c70c2dbf70713cd1bed',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == ref,
            'Changed/nonlocal evidence: ' + str(path))
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


for relative, expected in ANCHORS.items():
    require(sha(ROOT / relative) == expected, 'Frozen prerequisite changed: ' + relative)
# Verify model/tokenizer dependencies before importing the frozen engine.
for _ref in read(ROOT / 'experiments/expansion_step10_20261007/comparison_plan.json')['execution_code']:
    verify(_ref)
runner = module('step11_frozen_runner', ROOT / 'experiments/local_training_step03_20261007/runner.py')
core = runner.core
mixed = module('step11_frozen_mixed_artifacts', ROOT / 'experiments/expansion_step10_20261007/mixed_artifacts.py')


def select_from_bundle(bundle):
    """Select using only role, speaker, frame count and ID, never predictions."""
    entries, rows, groups = bundle['entries'], bundle['rows'], bundle['groups']
    eligible = groups['expanded510']
    excluded = groups['held27'] + groups['old71'] + groups['new24']
    require(len(set(eligible)) == len(eligible) and len(set(excluded)) == len(excluded),
            'Duplicate eligible/excluded metadata')
    require(not set(eligible) & set(excluded), 'Development/held contamination')
    for sid in eligible:
        entry, row = entries[sid], rows[sid]
        require(entry['role'] in ROLE_GENERATIONS and
                entry['generation'] == ROLE_GENERATIONS[entry['role']] and
                entry['original_split'] == row['split'] == 'train' and
                entry['provisional_training_candidate'] is True,
                'Nontraining role among eligible inputs')
        require(entry['speaker_id'] == row['speaker_id'] and
                entry['student_frames'] == row['frames'], 'Logical metadata mismatch')
    rankings, selected, diagnostic_groups = {}, [], {}
    for role, generation in ROLE_GENERATIONS.items():
        candidates = [sid for sid in eligible if entries[sid]['role'] == role]
        speakers = sorted({entries[sid]['speaker_id'] for sid in candidates})
        ranking = sorted(({'speaker_id': speaker,
                           'sha256': hashlib.sha256((SELECTION_SALT + role + ':' + speaker).encode()).hexdigest()}
                          for speaker in speakers), key=lambda item: (item['sha256'], item['speaker_id']))
        require(len(ranking) >= 4, 'Fewer than four eligible speakers')
        rankings[role] = ranking
        chosen = []
        for rank, item in enumerate(ranking[:4], 1):
            speaker = item['speaker_id']
            ids = sorted((sid for sid in candidates if entries[sid]['speaker_id'] == speaker),
                         key=lambda sid: (rows[sid]['frames'], sid))
            median_index = (len(ids) - 1) // 2
            sid = ids[median_index]
            row, entry = rows[sid], entries[sid]
            tokens = row['token_ids']
            require(tokens and all(type(token) is int and 1 <= token < 69 for token in tokens),
                    'Invalid selected CTC target')
            minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
            require(type(row['frames']) is int and row['frames'] >= minimum,
                    'Selected target is CTC infeasible; do not silently replace it')
            selected.append({'id': sid, 'generation': generation, 'role': role,
                             'speaker_id': speaker, 'source_stem': row['source_stem'],
                             'frames': row['frames'], 'tokens': list(tokens), 'text': row['text'],
                             'ctc_min_input_frames': minimum, 'ctc_slack': row['frames'] - minimum,
                             'selection_rank': rank, 'speaker_selection_sha256': item['sha256'],
                             'speaker_candidate_count': len(ids), 'lower_median_index': median_index,
                             'speaker_candidates_by_frames_then_id': ids,
                             'logical_record_sha256': entry['logical_record_sha256'],
                             'input': entry['input'], 'teacher_feature': entry['teacher_feature']})
            chosen.append(sid)
        diagnostic_groups[generation + '4'] = chosen
    ids = [record['id'] for record in selected]
    require(len(set(ids)) == 8 and len({record['speaker_id'] for record in selected}) == 8,
            'Eight distinct clips/speakers required')
    return {'schema': 'lipnet-step11-train-only-selection-v1', 'ids': ids,
            'diagnostic_groups': diagnostic_groups, 'records': selected,
            'eligible_speaker_ranking': rankings, 'eligible_training_ids': list(eligible),
            'excluded_ids': excluded, 'selection_salt': SELECTION_SALT,
            'speaker_rule': 'first four ascending SHA256(salt + role + colon + speaker), tie speaker ID',
            'clip_rule': 'lower median index (n-1)//2 after ascending (frames,id) within selected speaker',
            'selection_uses_transcripts_or_model_results': False,
            'new_human_approvals_added': 0, 'generalization_estimate': False}


def selection_metadata():
    """Verify frozen metadata without opening any pixel/feature payload."""
    bundle = mixed.verify_handoff()
    result = select_from_bundle(bundle)
    require(len(result['eligible_training_ids']) == 510 and len(result['excluded_ids']) == 122,
            'Frozen source cohort size changed')
    result['selection_source_handoff'] = bundle['handoff_ref']
    return result


def load_clips():
    """Validate all permitted train caches, retaining the selected eight only."""
    selection = selection_metadata()
    loaded = mixed.load_training('expanded510')
    require(loaded['ids'] == selection['eligible_training_ids'] and
            loaded['held_training_ids_opened'] == [] and loaded['source_media_opened'] is False,
            'Frozen loader scope differs')
    clips = {sid: loaded['clips'][sid] for sid in selection['ids']}
    for record in selection['records']:
        clip = clips[record['id']]
        require(clip['split'] == clip['source_split'] == 'train' and
                clip['speaker_id'] == record['speaker_id'] and clip['text'] == record['text'] and
                clip['tokens'] == record['tokens'] and len(clip['pixels']) == record['frames'],
                'Selected loaded clip differs from metadata')
    return {'clips': clips, 'ids': list(clips), 'selection': selection,
            'diagnostic_groups': selection['diagnostic_groups'], 'data_bindings': loaded['bindings'],
            'bindings': loaded['bindings'], 'handoff': loaded['handoff'],
            'counts': {'clips': 8, 'speakers': 8, 'old': 4, 'new': 4,
                       'student_frames': sum(len(clip['pixels']) for clip in clips.values()),
                       'validated_training_payloads': 510, 'development_payloads': 0,
                       'held_payloads': 0, 'raw_media': 0}}


def plan_path(seed):
    require(seed in SEEDS, 'Unknown seed')
    return BASE / 'plans' / f'seed{seed}.json'


def code_bindings(extra_code_paths=()):
    evaluation = module('step11_bound_evaluator_code', BASE / 'evaluate.py')
    paths = {BASE / name for name in ('protocol.py', 'test_protocol.py', 'training.py', 'evaluate.py')}
    paths.update(Path(path).resolve() for path in extra_code_paths)
    paths.update(Path(ref['path']) for ref in evaluation.evaluation_code())
    paths.update(Path(ref['path']) for ref in runner._code_bindings())
    paths.update(ROOT / relative for relative in ANCHORS if relative.endswith('.py'))
    paths.add(Path(mixed.OLD_LOADER['path']))
    require(all(path.is_relative_to(ROOT) and path.suffix in {'.py', '.txt'} for path in paths),
            'Code bindings must be local source files')
    return [bind(path) for path in sorted(paths, key=str)]


def preservation_contract():
    weights = ROOT / 'backend/models/weights'
    inventory = sorted((path for path in weights.rglob('*') if path.is_file()), key=str)
    summary = ROOT / 'experiments/expansion_step10_20261007/results/step10_summary.json'
    require(sha(summary) == '5d051d18c2a2466657003d2b0a497d3a9e07275a9332a99a43d90ee85168f1a8',
            'Step10 completion evidence changed')
    intended = weights / 'baseline_jamo_v2.pt'
    return {'artifacts': [bind(path) for path in [ROOT / 'backend/config.py', summary, *inventory]],
            'weights_directory': str(weights), 'weights_inventory': [str(path) for path in inventory],
            'absent_paths': [] if intended.exists() else [str(intended)]}


def verify_preservation(contract):
    weights = ROOT / 'backend/models/weights'
    require(contract['weights_directory'] == str(weights) and
            [str(path) for path in sorted((path for path in weights.rglob('*') if path.is_file()), key=str)] ==
            contract['weights_inventory'], 'Service weight inventory changed')
    for ref in contract['artifacts']:
        verify(ref)
    for path in contract['absent_paths']:
        require(Path(path).is_relative_to(weights) and not Path(path).exists(), 'Previously absent service weight created')


def check_plan(plan, policy):
    require(plan['schema'] == 'lipnet-step11-memorization-training-plan-v1' and plan['seed'] in SEEDS,
            'Wrong diagnostic plan identity')
    require(plan['arms'] == list(ARMS) and plan['updates'] == UPDATES and plan['batch_size'] == 2 and
            plan['checkpoint_interval'] == CHECKPOINT_INTERVAL, 'Arm/budget changed')
    require(plan['ids'] == policy['selection']['ids'] and
            plan['diagnostic_groups'] == policy['selection']['diagnostic_groups'], 'Selected cohort changed')
    require(len(set(plan['ids'])) == 8 and not set(plan['ids']) & set(policy['selection']['excluded_ids']),
            'Held/development contamination')
    require(plan['schedule'] == core.fixed_schedule(plan['ids'], plan['seed'], UPDATES, 2),
            'Fixed paired schedule changed')
    require(Counter(sid for batch in plan['schedule'] for sid in batch) == Counter({sid: 300 for sid in plan['ids']}),
            'Per-clip exposure budget changed')
    require(plan['runtime'] == policy['runtime'] and plan['execution_code'] == policy['execution_code'] and
            plan['data_bindings'] == policy['data_bindings'], 'Runtime/code/data evidence changed')
    require(plan['optimizer'] == policy['optimizer'] and plan['loss'] == policy['loss'], 'Optimization policy changed')
    require(plan['initial_state_hashes_expected'] == policy['initial_state_hashes_expected'][str(plan['seed'])],
            'Initial state contract changed')
    initial = plan['initial_state_hashes_expected']
    require(set(initial) == set(ARMS) and all(initial['A_ctc'][key] == initial['B_feature'][key]
                                           for key in ('model', 'adapter', 'rng')),
            'Unpaired student/adapter/RNG initialization')
    require(initial['A_ctc']['optimizer'] != initial['B_feature']['optimizer'],
            'A/B optimizer parameter scope should differ')
    require(plan['experiment_plan'] == bind(POLICY) and plan['plan_sha256'] == runner.canonical_plan_hash(plan),
            'Plan policy/hash binding changed')


def freeze(extra_code_paths=()):
    """Create immutable protocol/plans before the first update; never overwrite."""
    require(not POLICY.exists() and not FROZEN.exists() and not any(plan_path(seed).exists() for seed in SEEDS),
            'Diagnostic contract already exists')
    core.setup_runtime()
    require(runner.torch.cuda.is_available(), 'CUDA required by this local diagnostic')
    loaded = load_clips()
    runtime = runner._runtime(runner.torch.device('cuda:0'))
    initial = {}
    for seed in SEEDS:
        initial[str(seed)] = {}
        for arm in ARMS:
            objects = core.make_objects(seed, arm, 'cuda:0')
            initial[str(seed)][arm] = runner.state_hashes(core.runtime_state(*objects))
            del objects
        require(all(initial[str(seed)]['A_ctc'][key] == initial[str(seed)]['B_feature'][key]
                    for key in ('model', 'adapter', 'rng')), 'Fresh paired initialization differs')
    gc.collect()
    runner.torch.cuda.empty_cache()
    policy = {'schema': 'lipnet-step11-memorization-protocol-v1', 'status': 'frozen_before_training',
              'purpose': 'train-only small-cohort memorization and paired CTC versus feature-loss diagnostic',
              'selection': loaded['selection'], 'selection_source_handoff': loaded['handoff'],
              'seeds': list(SEEDS), 'arms': list(ARMS), 'updates_per_run': UPDATES, 'batch_size': 2,
              'checkpoint_interval': CHECKPOINT_INTERVAL, 'evaluation_milestones': list(MILESTONES),
              'normal_train_rows_per_run': 40, 'final_temporal_conditions': ['normal', 'static', 'shuffled'],
              'evaluation_contract': module('step11_bound_evaluation_contract', BASE / 'evaluate.py').evaluation_contract(),
              'all_inference_rows_planned': 224, 'optimizer_updates_planned': 4800,
              'presentations_per_run': 2400, 'presentations_per_clip_per_run': 300,
              'success_rule': 'final normal greedy prediction equals target exactly on all eight train utterances',
              'success_strata': {'old4': '4/4 exact', 'new4': '4/4 exact'},
              'fixed_budget_without_early_stop_or_extension': True,
              'optimizer': {'name': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
                            'weight_decay': 0., 'gradient_clip': 1., 'precision': 'float32', 'ctc_device': 'cpu'},
              'loss': {'A_ctc': {'ctc': 1., 'feature': 0.}, 'B_feature': {'ctc': 1., 'feature': 1.}},
              'runtime': runtime, 'initial_state_hashes_expected': initial,
              'execution_code': code_bindings(extra_code_paths), 'data_bindings': loaded['data_bindings'],
              'preservation': preservation_contract(),
              'loading_scope': loaded['counts'], 'generalization_claim_allowed': False,
              'development_or_test_evaluation': False, 'new_human_approvals_added': 0,
              'audio_used': False, 'teacher_model_loaded': False, 'service_weights_updated': False}
    runner._save_json(POLICY, policy)
    plan_refs = []
    for seed in SEEDS:
        plan = {'schema': 'lipnet-step11-memorization-training-plan-v1', 'seed': seed,
                'arms': list(ARMS), 'updates': UPDATES, 'batch_size': 2, 'checkpoint_interval': CHECKPOINT_INTERVAL,
                'ids': loaded['ids'], 'diagnostic_groups': loaded['diagnostic_groups'],
                'schedule': core.fixed_schedule(loaded['ids'], seed, UPDATES, 2),
                'experiment_plan': bind(POLICY), 'selection_source_handoff': loaded['handoff'],
                'runtime': runtime, 'execution_code': policy['execution_code'], 'data_bindings': loaded['data_bindings'],
                'optimizer': policy['optimizer'], 'loss': policy['loss'],
                'initial_state_hashes_expected': initial[str(seed)], 'service_weights_updated': False}
        data = [runner.validate_plan(plan, loaded['clips'], arm) for arm in ARMS]
        require(data[0] == data[1], 'Paired arm tensor bindings differ')
        plan['training_data'] = {key: data[0][key] for key in ('data_sha256', 'clips')}
        plan['plan_sha256'] = runner.canonical_plan_hash(plan)
        check_plan(plan, policy)
        runner._save_json(plan_path(seed), plan)
        plan_refs.append(bind(plan_path(seed)))
    result = {'schema': 'lipnet-step11-freeze-manifest-v1', 'status': 'frozen',
              'experiment_plan': bind(POLICY), 'plans': plan_refs,
              'optimizer_updates_planned': 4800, 'optimizer_updates_executed': 0}
    runner._save_json(FROZEN, result)
    return result


def verify_frozen():
    manifest = read(FROZEN)
    require(manifest['schema'] == 'lipnet-step11-freeze-manifest-v1' and manifest['status'] == 'frozen' and
            manifest['experiment_plan'] == bind(POLICY), 'Freeze manifest changed')
    policy = read(verify(manifest['experiment_plan']))
    require(policy['selection'] == selection_metadata(), 'Deterministic selection changed')
    verify_preservation(policy['preservation'])
    for ref in policy['execution_code']:
        verify(ref)
    require(manifest['plans'] == [bind(plan_path(seed)) for seed in SEEDS], 'Frozen seed plans changed')
    for ref in manifest['plans']:
        check_plan(read(verify(ref)), policy)
    core.setup_runtime()
    require(policy['runtime'] == runner._runtime(runner.torch.device('cuda:0')), 'Runtime changed')
    return policy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('selection', 'freeze', 'verify'))
    parser.add_argument('--extra-code', action='append', default=[])
    args = parser.parse_args()
    result = (selection_metadata() if args.command == 'selection' else
              freeze(args.extra_code) if args.command == 'freeze' else verify_frozen())
    print(json.dumps(result, ensure_ascii=False, indent=2))
