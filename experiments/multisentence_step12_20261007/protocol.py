"""Frozen multi-sentence diagnostic with 24 train and 8 recording-held clips.

All candidates were historically in the original510 training pool. The eight
held clips are internal holdouts for freshly initialized Step12 models, never
an untouched test set. Frozen full-loader validation may read their teacher
arrays; no holdout teacher target is returned or sent to the training runner.
"""
from __future__ import annotations
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import unicodedata

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / 'experiments/memorization_step11_20261007'
POLICY = BASE / 'diagnostic_plan.json'
FROZEN = BASE / 'plans/freeze_manifest.json'
SEEDS = (42, 43)
ARMS = ('A_ctc', 'B_feature')
UPDATES, CHECKPOINT_INTERVAL = 3600, 900
MILESTONES = (0, 900, 1800, 2700, 3600)
SALT = 'step12-multisentence-v1:'
STEP11_SUMMARY_SHA = 'c656b3fb40ba1e1eb609babc03dc9b55c8038a27163736709b210cd7b1db9f8f'


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
    require(path.is_relative_to(ROOT) and bind(path) == ref, 'Changed/nonlocal evidence: ' + str(path))
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


require(sha(PRIOR / 'protocol.py') == 'a4b4f213c7b399932081086841b71f09a4d5f570c5106c90facd797618f47b43',
        'Frozen Step11 protocol changed')
require(sha(PRIOR / 'results/step11_summary.json') == STEP11_SUMMARY_SHA, 'Step11 completion evidence changed')
_prior_summary = read(PRIOR / 'results/step11_summary.json')
_prior_policy_ref = _prior_summary['artifacts']['diagnostic_plan.json']
verify(_prior_policy_ref)
for _ref in read(_prior_policy_ref['path'])['execution_code']:
    verify(_ref)
prior = module('step12_frozen_step11_protocol', PRIOR / 'protocol.py')
runner, core, mixed = prior.runner, prior.core, prior.mixed


def normalized_target(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def selection_hash(scope, sid):
    require(scope in ('train', 'holdout'), 'Unknown selection scope')
    return hashlib.sha256((SALT + scope + ':' + sid).encode()).hexdigest()


def ctc_reasons(row):
    tokens = row['token_ids']
    if not tokens or not all(type(token) is int and 1 <= token < 69 for token in tokens):
        return ['invalid_ctc_target']
    minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
    return [] if type(row['frames']) is int and row['frames'] >= minimum else ['ctc_infeasible']


def source_hashes(bundle, sid):
    entry = bundle['entries'][sid]
    members = bundle['inputs'][entry['generation']][sid]['source_verification']['members']
    result = {kind + '_sha256': members[kind]['sha256'] for kind in ('video', 'label')}
    require(all(re.fullmatch(r'[0-9a-f]{64}', value) for value in result.values()), 'Invalid source member SHA')
    return result


def select_from_bundle(bundle, prior_selection):
    """Select metadata deterministically, reserving all eight previous anchors."""
    rows, entries, groups = bundle['rows'], bundle['entries'], bundle['groups']
    eligible = list(groups['expanded510'])
    excluded = groups['held27'] + groups['old71'] + groups['new24']
    require(len(set(eligible)) == len(eligible) and not set(eligible) & set(excluded), 'Eligible/excluded contamination')
    anchor_ids = list(prior_selection['ids'])
    require(len(set(anchor_ids)) == 8 and set(anchor_ids) <= set(eligible), 'Wrong retained anchor scope')
    require(len({rows[sid]['speaker_id'] for sid in anchor_ids}) == 8, 'Anchor speakers not distinct')
    for sid in eligible:
        row, entry = rows[sid], entries[sid]
        require(row['split'] == entry['original_split'] == 'train' and
                entry['role'] in ('old_train', 'new_provisional_train') and
                entry['generation'] == ('old' if entry['role'] == 'old_train' else 'new') and
                entry['provisional_training_candidate'] is True and
                row['speaker_id'] == entry['speaker_id'] and row['frames'] == entry['student_frames'],
                'Ineligible original training record')
    prior_records = {record['id']: record for record in prior_selection['records']}
    require(set(prior_records) == set(anchor_ids), 'Prior anchor metadata incomplete')
    selected_texts, train_ids, hold_ids, events, speaker_selections = set(), [], [], [], []
    for sid in anchor_ids:
        row = rows[sid]
        require(not ctc_reasons(row) and normalized_target(row['text']), 'Invalid retained anchor')
        require(normalized_target(row['text']) not in selected_texts, 'Duplicate retained anchor target')
        require(prior_records[sid]['source_stem'] == row['source_stem'] and
                prior_records[sid]['speaker_id'] == row['speaker_id'] and
                prior_records[sid]['logical_record_sha256'] == entries[sid]['logical_record_sha256'],
                'Retained anchor changed')
        source_hashes(bundle, sid)
        selected_texts.add(normalized_target(row['text']))
    for anchor in anchor_ids:
        source, speaker = rows[anchor]['source_stem'], rows[anchor]['speaker_id']
        anchor_hashes = source_hashes(bundle, anchor)
        pool = sorted((sid for sid in eligible if sid != anchor and rows[sid]['speaker_id'] == speaker and
                       rows[sid]['source_stem'] == source), key=lambda sid: (selection_hash('train', sid), sid))
        additions = []
        for sid in pool:
            reasons = ctc_reasons(rows[sid])
            if not normalized_target(rows[sid]['text']):
                reasons.append('empty_normalized_target')
            if normalized_target(rows[sid]['text']) in selected_texts:
                reasons.append('duplicate_normalized_target')
            if source_hashes(bundle, sid) != anchor_hashes:
                reasons.append('same_stem_source_hash_mismatch')
            status = 'not_needed' if len(additions) == 2 else 'skipped' if reasons else 'selected'
            events.append({'scope': 'train', 'anchor_id': anchor, 'id': sid,
                           'rank_sha256': selection_hash('train', sid), 'status': status, 'reasons': reasons})
            if status == 'selected':
                additions.append(sid)
                selected_texts.add(normalized_target(rows[sid]['text']))
        require(len(additions) == 2, 'Cannot select two same-recording additions: ' + speaker)
        train_ids.extend([anchor, *additions])
        speaker_selections.append({'speaker_id': speaker, 'generation': entries[anchor]['generation'],
                                   'retained_anchor_id': anchor, 'training_source_stem': source,
                                   'train_ids': [anchor, *additions], 'train_candidate_ids_ranked': pool})
    train_stems = {rows[sid]['source_stem'] for sid in train_ids}
    train_video = {source_hashes(bundle, sid)['video_sha256'] for sid in train_ids}
    train_labels = {source_hashes(bundle, sid)['label_sha256'] for sid in train_ids}
    for item in speaker_selections:
        anchor, speaker = item['retained_anchor_id'], item['speaker_id']
        pool = sorted((sid for sid in eligible if rows[sid]['speaker_id'] == speaker and
                       rows[sid]['source_stem'] != rows[anchor]['source_stem']),
                      key=lambda sid: (selection_hash('holdout', sid), sid))
        chosen = None
        for sid in pool:
            hashes, row = source_hashes(bundle, sid), rows[sid]
            reasons = ctc_reasons(row)
            if not normalized_target(row['text']):
                reasons.append('empty_normalized_target')
            if normalized_target(row['text']) in selected_texts:
                reasons.append('duplicate_normalized_target')
            if row['source_stem'] in train_stems:
                reasons.append('training_source_stem_overlap')
            if hashes['video_sha256'] in train_video:
                reasons.append('training_video_sha_overlap')
            if hashes['label_sha256'] in train_labels:
                reasons.append('training_label_sha_overlap')
            status = 'not_needed' if chosen is not None else 'skipped' if reasons else 'selected'
            events.append({'scope': 'holdout', 'anchor_id': anchor, 'id': sid,
                           'rank_sha256': selection_hash('holdout', sid), 'status': status, 'reasons': reasons})
            if status == 'selected':
                chosen = sid
                selected_texts.add(normalized_target(row['text']))
        require(chosen is not None, 'Cannot select source-disjoint holdout: ' + speaker)
        hold_ids.append(chosen)
        item.update(holdout_id=chosen, holdout_source_stem=rows[chosen]['source_stem'], holdout_candidate_ids_ranked=pool)
    all_ids = train_ids + hold_ids
    require(len(set(all_ids)) == 32 and len(selected_texts) == 32, 'Expected32 unique clips and normalized targets')
    require(train_stems.isdisjoint(rows[sid]['source_stem'] for sid in hold_ids) and
            train_video.isdisjoint(source_hashes(bundle, sid)['video_sha256'] for sid in hold_ids) and
            train_labels.isdisjoint(source_hashes(bundle, sid)['label_sha256'] for sid in hold_ids),
            'Training/holdout source collision')
    records = []
    for sid in all_ids:
        row, entry = rows[sid], entries[sid]
        tokens = row['token_ids']
        minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        records.append({'id': sid, 'generation': entry['generation'], 'original_role': entry['role'],
                        'original_split': 'train', 'diagnostic_role': 'train' if sid in train_ids else 'internal_holdout',
                        'retained_step11_anchor': sid in anchor_ids, 'speaker_id': row['speaker_id'],
                        'source_stem': row['source_stem'], 'frames': row['frames'], 'tokens': list(tokens),
                        'text': row['text'], 'normalized_target': normalized_target(row['text']),
                        'ctc_min_input_frames': minimum, 'ctc_slack': row['frames'] - minimum,
                        'logical_record_sha256': entry['logical_record_sha256'], 'input': entry['input'],
                        'teacher_feature': entry['teacher_feature'] if sid in train_ids else None,
                        **source_hashes(bundle, sid)})
    evaluation_groups = {'train24': train_ids, 'hold8': hold_ids,
                         **{scope + '_' + generation + str(count): [sid for sid in ids if entries[sid]['generation'] == generation]
                            for scope, ids, count in (('train', train_ids, 12), ('hold', hold_ids, 4))
                            for generation in ('old', 'new')}}
    require({key: len(value) for key, value in evaluation_groups.items()} ==
            {'train24': 24, 'hold8': 8, 'train_old12': 12, 'train_new12': 12, 'hold_old4': 4, 'hold_new4': 4},
            'Source-stratum counts changed')
    return {'schema': 'lipnet-step12-train24-hold8-selection-v1', 'train_ids': train_ids, 'holdout_ids': hold_ids,
            'ids': all_ids, 'retained_step11_ids': anchor_ids, 'records': records,
            'evaluation_groups': evaluation_groups, 'speaker_selections': speaker_selections,
            'selection_events': events, 'eligible_training_ids': eligible, 'historical_excluded_ids': excluded,
            'selection_salt': SALT, 'normalization': 'NFC then remove all Unicode whitespace using str.split',
            'held_out_speakers': False, 'held_out_recordings': True, 'untouched_test_set': False,
            'historically_used_by_prior_models': True, 'prediction_or_loss_used_for_selection': False,
            'new_human_approvals_added': 0, 'source_audio_sha_checked': False,
            'source_separation_checked': ['source_stem', 'video_sha256', 'label_sha256']}


def selection_metadata():
    bundle = mixed.verify_handoff()
    selected = select_from_bundle(bundle, read(_prior_policy_ref['path'])['selection'])
    require(len(selected['eligible_training_ids']) == 510 and len(selected['historical_excluded_ids']) == 122,
            'Historical source cohort changed')
    selected['source_handoff'] = bundle['handoff_ref']
    selected['prior_step11_policy'] = _prior_policy_ref
    return selected


def evaluation_binding(clip):
    require(clip['split'] == clip['diagnostic_role'] and clip['source_split'] == 'train' and
            clip['diagnostic_role'] in ('train', 'internal_holdout') and 'teacher_features' not in clip,
            'Invalid evaluation clip role or leaked teacher target')
    return {**{key: clip[key] for key in ('id', 'split', 'source_split', 'diagnostic_role', 'speaker_id', 'source_stem', 'text')},
            'frames': len(clip['pixels']), 'pixels_sha256': runner.nested_hash(clip['pixels']),
            'tokens_sha256': runner.nested_hash(clip['tokens'])}


def evaluation_bindings(clips):
    bindings = {sid: evaluation_binding(clips[sid]) for sid in sorted(clips)}
    require(all(sid == value['id'] for sid, value in bindings.items()), 'Evaluation clip ID mismatch')
    return {'data_sha256': runner.nested_hash(bindings), 'clips': bindings}


def load_clips():
    selection = selection_metadata()
    loaded = mixed.load_training('expanded510')
    require(loaded['ids'] == selection['eligible_training_ids'] and loaded['held_training_ids_opened'] == [] and
            loaded['source_media_opened'] is False, 'Historical frozen loader scope changed')
    train = {sid: dict(loaded['clips'][sid], diagnostic_role='train') for sid in selection['train_ids']}
    evaluation = {}
    for sid in selection['ids']:
        role = 'train' if sid in train else 'internal_holdout'
        evaluation[sid] = {**{key: value for key, value in loaded['clips'][sid].items() if key != 'teacher_features'},
                           'split': role, 'source_split': 'train', 'diagnostic_role': role}
    for record in selection['records']:
        clip = evaluation[record['id']]
        require(clip['tokens'] == record['tokens'] and clip['text'] == record['text'] and
                len(clip['pixels']) == record['frames'] and clip['speaker_id'] == record['speaker_id'] and
                clip['source_stem'] == record['source_stem'], 'Loaded selected cache differs from metadata')
    return {'train_clips': train, 'training_clips': train, 'clips': train, 'evaluation_clips': evaluation,
            'holdout_clips': {sid: evaluation[sid] for sid in selection['holdout_ids']},
            'train_ids': selection['train_ids'], 'holdout_ids': selection['holdout_ids'], 'ids': selection['train_ids'],
            'selection': selection, 'evaluation_groups': selection['evaluation_groups'],
            'diagnostic_groups': selection['evaluation_groups'], 'data_bindings': loaded['bindings'],
            'counts': {'train_clips': 24, 'holdout_clips': 8, 'evaluation_clips': 32, 'speakers': 8,
                       'train_frames': sum(len(clip['pixels']) for clip in train.values()),
                       'holdout_frames': sum(len(evaluation[sid]['pixels']) for sid in selection['holdout_ids']),
                       'original_training_payloads_validated': 510, 'original_teacher_targets_validated': 510,
                       'internal_holdout_teacher_targets_read_for_validation_only': 8,
                       'internal_holdout_teacher_targets_returned': 0, 'historical_development_payloads': 0,
                       'historical_held27_payloads': 0, 'test_payloads': 0, 'source_media_payloads': 0}}


def plan_path(seed):
    require(seed in SEEDS, 'Unknown seed')
    return BASE / 'plans' / f'seed{seed}.json'


def code_bindings(extra_code_paths=()):
    evaluation = module('step12_evaluation_code', BASE / 'evaluate.py')
    paths = {BASE / name for name in ('protocol.py', 'test_protocol.py', 'training.py', 'evaluate.py')}
    paths.update(Path(path).resolve() for path in extra_code_paths)
    paths.update(Path(ref['path']) for ref in evaluation.evaluation_code())
    paths.update(Path(ref['path']) for ref in read(_prior_policy_ref['path'])['execution_code'])
    paths.add(PRIOR / 'protocol.py')
    require(all(path.is_relative_to(ROOT) and path.suffix in {'.py', '.txt'} for path in paths), 'Nonlocal code source')
    return [bind(path) for path in sorted(paths, key=str)]


def preservation_contract():
    value = prior.preservation_contract()
    value['artifacts'].append(bind(PRIOR / 'results/step11_summary.json'))
    return value


def verify_preservation(contract):
    prior.verify_preservation(contract)


def check_plan(plan, policy):
    require(plan['schema'] == 'lipnet-step12-multisentence-training-plan-v1' and plan['seed'] in SEEDS and
            plan['arms'] == list(ARMS), 'Wrong Step12 plan identity')
    require(plan['updates'] == UPDATES and plan['batch_size'] == 2 and plan['checkpoint_interval'] == CHECKPOINT_INTERVAL,
            'Fixed diagnostic budget changed')
    selection = policy['selection']
    require(plan['ids'] == plan['train_ids'] == selection['train_ids'] and plan['holdout_ids'] == selection['holdout_ids'] and
            plan['diagnostic_groups'] == plan['evaluation_groups'] == selection['evaluation_groups'], 'Selected scopes changed')
    require(len(set(plan['ids'])) == 24 and not set(plan['ids']) &
            (set(plan['holdout_ids']) | set(selection['historical_excluded_ids'])), 'Holdout/development training contamination')
    require(plan['schedule'] == core.fixed_schedule(plan['ids'], plan['seed'], UPDATES, 2), 'Paired schedule changed')
    require(Counter(sid for batch in plan['schedule'] for sid in batch) == Counter({sid: 300 for sid in plan['ids']}),
            'Train exposure budget changed')
    for key in ('runtime', 'execution_code', 'data_bindings', 'optimizer', 'loss', 'evaluation_data'):
        require(plan[key] == policy[key], 'Changed policy field: ' + key)
    initial = plan['initial_state_hashes_expected']
    require(initial == policy['initial_state_hashes_expected'][str(plan['seed'])] and set(initial) == set(ARMS) and
            all(initial['A_ctc'][key] == initial['B_feature'][key] for key in ('model', 'adapter', 'rng')) and
            initial['A_ctc']['optimizer'] != initial['B_feature']['optimizer'], 'Unpaired fresh initialization')
    require(plan['experiment_plan'] == bind(POLICY) and plan['plan_sha256'] == runner.canonical_plan_hash(plan),
            'Plan policy/hash binding changed')


def freeze(extra_code_paths=()):
    require(not POLICY.exists() and not FROZEN.exists() and not any(plan_path(seed).exists() for seed in SEEDS),
            'Step12 contract already exists')
    core.setup_runtime()
    require(runner.torch.cuda.is_available(), 'CUDA required')
    loaded = load_clips()
    runtime = runner._runtime(runner.torch.device('cuda:0'))
    initial = {}
    for seed in SEEDS:
        initial[str(seed)] = {}
        for arm in ARMS:
            objects = core.make_objects(seed, arm, 'cuda:0')
            initial[str(seed)][arm] = runner.state_hashes(core.runtime_state(*objects))
            del objects
    gc.collect()
    runner.torch.cuda.empty_cache()
    policy = {'schema': 'lipnet-step12-multisentence-protocol-v1', 'status': 'frozen_before_training',
              'purpose': 'Multi-sentence same-speaker memorization and different-recording internal holdout diagnostic',
              'selection': loaded['selection'], 'prior_step11_summary': bind(PRIOR / 'results/step11_summary.json'),
              'seeds': list(SEEDS), 'arms': list(ARMS), 'updates_per_run': UPDATES, 'batch_size': 2,
              'checkpoint_interval': CHECKPOINT_INTERVAL, 'evaluation_milestones': list(MILESTONES),
              'evaluation_contract': module('step12_evaluation_contract', BASE / 'evaluate.py').evaluation_contract(),
              'evaluation_data': evaluation_bindings(loaded['evaluation_clips']),
              'all_inference_rows_planned': 896, 'optimizer_updates_planned': 14400,
              'presentations_per_run': 7200, 'presentations_per_train_clip_per_run': 300,
              'success_rule': 'final normal greedy prediction exactly matches all24 training targets; hold8 reported descriptively',
              'speaker_anchor_baseline': {
                  'kind': 'known_speaker_fixed_training_sentence',
                  'anchor_ids_by_speaker': {item['speaker_id']: item['retained_anchor_id'] for item in loaded['selection']['speaker_selections']},
                  'evaluation_ids': loaded['holdout_ids'], 'choice_uses_holdout_targets': False,
                  'enabled': True, 'training_or_model_inference': False, 'requires_known_speaker_metadata': True,
                  'prediction_rule': 'For each internal holdout speaker, output that speaker retained Step11 training-anchor transcript',
                  'anchor_selection_uses_holdout_targets': False,
                  'purpose': 'Descriptive control for transcript recall keyed by speaker identity, not a deployable lipreading baseline',
                  'metrics': ['pooled CER including internal spaces', 'pooled WER', 'exact sentences'],
                  'pairs': [{'speaker_id': item['speaker_id'], 'holdout_id': item['holdout_id'],
                             'anchor_id': item['retained_anchor_id']} for item in loaded['selection']['speaker_selections']]},
              'holdout_used_for_stopping_tuning_or_selection': False, 'fixed_budget_without_early_stop_or_extension': True,
              'optimizer': {'name': 'Adam', 'lr': .001, 'betas': [.9, .999], 'eps': 1e-8,
                            'weight_decay': 0., 'gradient_clip': 1., 'precision': 'float32', 'ctc_device': 'cpu'},
              'loss': {'A_ctc': {'ctc': 1., 'feature': 0.}, 'B_feature': {'ctc': 1., 'feature': 1.}},
              'runtime': runtime, 'initial_state_hashes_expected': initial,
              'execution_code': code_bindings(extra_code_paths), 'data_bindings': loaded['data_bindings'],
              'preservation': preservation_contract(), 'loading_scope': loaded['counts'],
              'unseen_speaker_or_webcam_claim_allowed': False, 'untouched_test_claim_allowed': False,
              'historical_development_or_test_evaluation': False, 'new_human_approvals_added': 0,
              'audio_used': False, 'teacher_model_loaded': False, 'service_weights_updated': False}
    runner._save_json(POLICY, policy)
    refs = []
    for seed in SEEDS:
        plan = {'schema': 'lipnet-step12-multisentence-training-plan-v1', 'seed': seed, 'arms': list(ARMS),
                'updates': UPDATES, 'batch_size': 2, 'checkpoint_interval': CHECKPOINT_INTERVAL,
                'ids': loaded['train_ids'], 'train_ids': loaded['train_ids'], 'holdout_ids': loaded['holdout_ids'],
                'diagnostic_groups': loaded['evaluation_groups'], 'evaluation_groups': loaded['evaluation_groups'],
                'schedule': core.fixed_schedule(loaded['train_ids'], seed, UPDATES, 2),
                'experiment_plan': bind(POLICY), 'initial_state_hashes_expected': initial[str(seed)],
                'service_weights_updated': False,
                **{key: policy[key] for key in ('runtime', 'execution_code', 'data_bindings', 'optimizer', 'loss', 'evaluation_data')}}
        data = [runner.validate_plan(plan, loaded['train_clips'], arm) for arm in ARMS]
        require(data[0] == data[1], 'A/B input bindings differ')
        plan['training_data'] = {key: data[0][key] for key in ('data_sha256', 'clips')}
        plan['plan_sha256'] = runner.canonical_plan_hash(plan)
        check_plan(plan, policy)
        runner._save_json(plan_path(seed), plan)
        refs.append(bind(plan_path(seed)))
    result = {'schema': 'lipnet-step12-freeze-manifest-v1', 'status': 'frozen', 'experiment_plan': bind(POLICY),
              'plans': refs, 'optimizer_updates_planned': 14400, 'optimizer_updates_executed': 0}
    runner._save_json(FROZEN, result)
    return result


def verify_frozen():
    manifest = read(FROZEN)
    require(manifest['schema'] == 'lipnet-step12-freeze-manifest-v1' and manifest['status'] == 'frozen' and
            manifest['experiment_plan'] == bind(POLICY), 'Freeze manifest changed')
    policy = read(verify(manifest['experiment_plan']))
    require(policy['selection'] == selection_metadata(), 'Deterministic selection changed')
    verify_preservation(policy['preservation'])
    for ref in policy['execution_code']:
        verify(ref)
    require(manifest['plans'] == [bind(plan_path(seed)) for seed in SEEDS], 'Frozen plans changed')
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
    value = selection_metadata() if args.command == 'selection' else freeze(args.extra_code) if args.command == 'freeze' else verify_frozen()
    print(json.dumps(value, ensure_ascii=False, indent=2))
