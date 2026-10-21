"""Prepare a fresh old-eight-speaker 24/72 training pair and recording-held dev16.

This module reads JSON provenance and hashes selected cache files only. It does
not decode NPZ/media, construct models, create teacher targets, or train.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PLAN = BASE / 'data_plan.json'
SPEAKERS = ('C002', 'C004', 'C017', 'C032', 'C282', 'C293', 'C297', 'C312')
SALT = 'step15-diversity-v1:'
STEP13 = ROOT / 'experiments/recording_probe_step13_20261007'
STEP14 = ROOT / 'experiments/posterior_candidates_step14_20261007'
STEP13_SUMMARY_SHA = 'bb32f1461b8a86f51c9a3e9121c9ce81fb48f62da4b41308a7b42b4c75f66c0d'
STEP14_SUMMARY_SHA = '4f0977e5328b6be26a548bed920820a5bd08b272ef59d404106bab339f937a9a'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verify(ref):
    path = Path(ref['path']).resolve()
    require(path.is_relative_to(ROOT) and bind(path) == ref, 'Changed/nonlocal evidence: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_ref(ref):
    require(Path(ref['path']).suffix == '.json', 'Metadata source must be JSON')
    return read(verify(ref))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalized_target(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def ranking(scope, identity):
    require(scope in ('dev_source', 'train_source', 'dev_clip', 'train_clip'), 'Unknown selection rank scope')
    return hashlib.sha256((SALT + scope + ':' + identity).encode()).hexdigest()


def unique(rows):
    lookup = {row['id']: row for row in rows}
    require(len(lookup) == len(rows), 'Duplicate source metadata ID')
    return lookup


def source_metadata():
    refs = {'step13_summary': {'path': str(STEP13 / 'results/step13_summary.json'), 'sha256': STEP13_SUMMARY_SHA},
            'step14_summary': {'path': str(STEP14 / 'results/step14_summary.json'), 'sha256': STEP14_SUMMARY_SHA}}
    summary13, summary14 = read_ref(refs['step13_summary']), read_ref(refs['step14_summary'])
    require(summary13['status'] == 'completed_matched_speaker_recording_probe' and
            summary14['status'] == 'completed_posterior_candidate_diagnostic', 'Prior stage incomplete')
    refs.update(step13_policy=summary13['artifacts']['diagnostic_plan.json'],
                step14_policy=summary14['artifacts']['diagnostic_plan.json'],
                step14_audit=summary14['artifacts']['results/independent_audit.json'])
    policy13, policy14, audit14 = read_ref(refs['step13_policy']), read_ref(refs['step14_policy']), read_ref(refs['step14_audit'])
    require(audit14['status'] == 'passed' and audit14['experiment_plan'] == refs['step14_policy'], 'Step14 audit unbound/failed')
    refs['step12_policy'] = policy13['prior_step12_policy']
    policy12 = read_ref(refs['step12_policy'])
    refs['handoff'] = policy13['selection']['source_handoff']
    handoff = read_ref(refs['handoff'])
    require(handoff['schema'] == 'lipnet-step09-provisional-candidate-handoff-v1' and
            handoff['new_human_approvals'] == 0, 'Handoff identity/approval status changed')
    rows, inputs, features = {}, {}, {}
    for generation in ('old', 'new'):
        for kind in ('data_plan', 'inputs', 'features'):
            name = generation + '_' + kind
            refs[name] = handoff['artifacts'][name]
        logical = read_ref(refs[generation + '_data_plan'])
        input_doc, feature_doc = read_ref(refs[generation + '_inputs']), read_ref(refs[generation + '_features'])
        for sid, row in unique(logical['records']).items():
            require(sid not in rows, 'Duplicate generation ID')
            rows[sid] = row
        inputs[generation] = unique(input_doc['records'])
        features[generation] = unique(feature_doc['records'])
    entries = unique(handoff['records'])
    require(set(rows) == set(entries) and len(rows) == 632, 'Historical632 inventory changed')
    pool = list(handoff['provisional_train_ids'])
    require(len(pool) == len(set(pool)) == 510 and len(handoff['held_new_train_ids']) == 27, 'Historical510/held27 changed')
    for sid, row in rows.items():
        entry = entries[sid]
        generation = entry['generation']
        item = inputs[generation][sid]
        require(entry['logical_record_sha256'] == canonical_sha(row) and entry['speaker_id'] == row['speaker_id'] and
                entry['source_stem'] == row['source_stem'] and entry['original_split'] == row['split'] and
                entry['student_frames'] == row['frames'], 'Logical provenance changed')
        require(entry['input'] == {'path': item['npz_path'], 'sha256': item['npz_sha256']} and
                entry['new_human_approval_added'] is False, 'Input/approval provenance changed')
        candidate = sid in pool
        require(entry['provisional_training_candidate'] is candidate and
                (sid in features[generation]) is candidate, 'Candidate/teacher scope mismatch')
        if candidate:
            require(row['split'] == 'train' and entry['role'] in ('old_train', 'new_provisional_train') and
                    entry['teacher_feature'] == features[generation][sid]['feature_file'], 'Training provenance changed')
    require(Counter(entries[sid]['role'] for sid in pool) == {'old_train': 441, 'new_provisional_train': 69}, 'Training role counts changed')
    historical_excluded = [sid for sid in rows if sid not in set(pool)]
    require(Counter(entries[sid]['role'] for sid in historical_excluded) ==
            {'old_development': 71, 'new_development': 24, 'new_train_hold': 27}, 'Excluded role counts changed')
    probe_ids = list(dict.fromkeys(policy13['selection']['ids'] + policy12['selection']['holdout_ids']))
    require(len(probe_ids) == 14 and set(probe_ids) <= set(pool), 'Prior probe14 changed')
    require(set(handoff['pending_listening']) <= set(handoff['held_new_train_ids']) and len(handoff['pending_listening']) == 2,
            'Pending human review scope changed')
    prior_code = list({ref['path']: ref for ref in policy13['execution_code'] + policy14['execution_code']}.values())
    for ref in prior_code:
        verify(ref)
    return {'rows': rows, 'entries': entries, 'inputs': inputs, 'features': features, 'pool_ids': pool,
            'historical_excluded_ids': historical_excluded, 'prior_probe_ids': probe_ids,
            'pending_listening': list(handoff['pending_listening']), 'source_refs': refs,
            'prior_preservation': policy14['preservation'], 'prior_execution_code': prior_code}


def source_hashes(bundle, sid):
    entry = bundle['entries'][sid]
    members = bundle['inputs'][entry['generation']][sid]['source_verification']['members']
    result = {key + '_sha256': members[key]['sha256'] for key in ('video', 'label')}
    require(all(re.fullmatch('[0-9a-f]{64}', value) for value in result.values()), 'Invalid source SHA')
    return result


def native_indices(row):
    require(type(row['frames']) is int and row['frames'] > 0 and
            all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in ('start_time', 'end_time', 'video_fps')) and
            0 <= row['start_time'] < row['end_time'] and row['video_fps'] > 0, 'Invalid timing metadata')
    return [round((row['start_time'] + k / 25) * row['video_fps']) for k in range(row['frames'])]


def ctc_reasons(row):
    tokens = row['token_ids']
    if not tokens or not all(type(token) is int and 1 <= token < 69 for token in tokens):
        return ['invalid_ctc_target']
    minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
    return [] if type(row['frames']) is int and row['frames'] >= minimum else ['ctc_infeasible']


def overlap_reasons(row, selected_rows):
    reasons, indices = [], set(native_indices(row))
    for selected in selected_rows:
        if row['source_stem'] != selected['source_stem']:
            continue
        if max(row['start_time'], selected['start_time']) < min(row['end_time'], selected['end_time']):
            reasons.append('selected_time_overlap:' + selected['id'])
        if indices.intersection(native_indices(selected)):
            reasons.append('selected_native_frame_overlap:' + selected['id'])
    return reasons


def candidate_inventory(bundle):
    rows, entries, pool = bundle['rows'], bundle['entries'], bundle['pool_ids']
    probe = bundle['prior_probe_ids']
    blocked_stems = {rows[sid]['source_stem'] for sid in probe}
    blocked_video = {source_hashes(bundle, sid)['video_sha256'] for sid in probe}
    blocked_label = {source_hashes(bundle, sid)['label_sha256'] for sid in probe}
    blocked_texts = {normalized_target(rows[sid]['text']) for sid in probe + bundle['historical_excluded_ids']}
    require(len(set(pool)) == len(pool) and not set(pool).intersection(bundle['historical_excluded_ids']), 'Pool/excluded contamination')
    result, source_identities = [], {}
    for sid in pool:
        row, entry = rows[sid], entries[sid]
        require(row['split'] == entry['original_split'] == 'train' and entry['role'] in ('old_train', 'new_provisional_train') and
                entry['provisional_training_candidate'] is True, 'Forbidden role in original pool')
        hashes = source_hashes(bundle, sid)
        identity = {'speaker_id': row['speaker_id'], **hashes}
        require(source_identities.setdefault(row['source_stem'], identity) == identity, 'Same source stem has inconsistent identity')
        reasons = ctc_reasons(row)
        if entry['generation'] != 'old' or row['speaker_id'] not in SPEAKERS:
            reasons.append('outside_old8_experiment')
        if row['source_stem'] in blocked_stems:
            reasons.append('prior_probe_recording_stem')
        if hashes['video_sha256'] in blocked_video:
            reasons.append('prior_probe_video_sha')
        if hashes['label_sha256'] in blocked_label:
            reasons.append('prior_probe_label_sha')
        text = normalized_target(row['text'])
        if not text:
            reasons.append('empty_normalized_target')
        if text in blocked_texts:
            reasons.append('prior_probe_or_excluded_normalized_target')
        native_indices(row)
        result.append({'id': sid, 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                       'generation': entry['generation'], 'frames': row['frames'], 'normalized_target': text,
                       'eligible': not reasons, 'reasons': reasons, **hashes})
    profile = []
    for speaker in sorted({row['speaker_id'] for row in result}):
        items = [row for row in result if row['speaker_id'] == speaker]
        available = [row for row in items if row['eligible']]
        profile.append({'speaker_id': speaker, 'generation': items[0]['generation'], 'original_count': len(items),
                        'original_recordings': dict(sorted(Counter(row['source_stem'] for row in items).items())),
                        'experiment_eligible_count': len(available),
                        'experiment_eligible_recordings': dict(sorted(Counter(row['source_stem'] for row in available).items()))})
    return {'records': result, 'speaker_profiles': profile, 'original_pool_count': len(pool),
            'experiment_eligible_count': sum(row['eligible'] for row in result),
            'prior_probe_ids': list(probe), 'blocked_recording_stems': sorted(blocked_stems),
            'blocked_video_sha256': sorted(blocked_video), 'blocked_label_sha256': sorted(blocked_label),
            'blocked_normalized_targets': sorted(blocked_texts), 'historical_excluded_ids': list(bundle['historical_excluded_ids']),
            'pending_listening_ids': list(bundle.get('pending_listening', []))}


def choose_clips(bundle, ids, count, selected_ids, scope, context):
    rows = bundle['rows']
    chosen, events = [], []
    reserved = [rows[sid] for sid in selected_ids]
    texts = {normalized_target(row['text']) for row in reserved}
    for sid in sorted(ids, key=lambda value: (ranking(scope + '_clip', value), value)):
        row = rows[sid]
        reasons = []
        if normalized_target(row['text']) in texts:
            reasons.append('duplicate_selected_normalized_target')
        reasons.extend(overlap_reasons(row, reserved))
        status = 'not_needed' if len(chosen) == count else 'skipped' if reasons else 'selected'
        events.append({'kind': 'clip', 'scope': scope, 'context': context, 'id': sid,
                       'rank_sha256': ranking(scope + '_clip', sid), 'status': status, 'reasons': reasons})
        if status == 'selected':
            chosen.append(sid)
            reserved.append(row)
            texts.add(normalized_target(row['text']))
    return chosen, events


def select_from_bundle(bundle):
    inventory = candidate_inventory(bundle)
    rows = bundle['rows']
    eligible = [item['id'] for item in inventory['records'] if item['eligible']]
    accepted, events, speakers = [], [], []
    train24, train72, dev16 = [], [], []
    for speaker in SPEAKERS:
        ids = [sid for sid in eligible if rows[sid]['speaker_id'] == speaker]
        stems = sorted({rows[sid]['source_stem'] for sid in ids}, key=lambda value: (ranking('dev_source', value), value))
        by_stem = {stem: [sid for sid in ids if rows[sid]['source_stem'] == stem] for stem in stems}
        chosen = None
        for dev_stem in stems:
            context = speaker + '/dev=' + dev_stem
            dev, attempts = choose_clips(bundle, by_stem[dev_stem], 2, accepted, 'dev', context)
            events.extend(attempts)
            if len(dev) < 2:
                events.append({'kind': 'dev_source', 'speaker_id': speaker, 'source_stem': dev_stem,
                               'rank_sha256': ranking('dev_source', dev_stem), 'status': 'rejected', 'reasons': ['fewer_than2_valid_dev_clips']})
                continue
            train_stems = sorted((stem for stem in stems if stem != dev_stem), key=lambda value: (ranking('train_source', value), value))
            for primary in train_stems:
                primary_ids, attempts = choose_clips(bundle, by_stem[primary], 6, accepted + dev, 'train', context + '/primary=' + primary)
                events.extend(attempts)
                if len(primary_ids) < 6:
                    events.append({'kind': 'primary_source', 'speaker_id': speaker, 'source_stem': primary,
                                   'dev_source_stem': dev_stem, 'status': 'rejected', 'reasons': ['fewer_than6_valid_primary_clips']})
                    continue
                for secondary in train_stems:
                    if secondary == primary:
                        continue
                    secondary_ids, attempts = choose_clips(bundle, by_stem[secondary], 3, accepted + dev + primary_ids, 'train', context + '/primary=' + primary + '/secondary=' + secondary)
                    events.extend(attempts)
                    if len(secondary_ids) < 3:
                        events.append({'kind': 'secondary_source', 'speaker_id': speaker, 'source_stem': secondary,
                                       'dev_source_stem': dev_stem, 'primary_source_stem': primary,
                                       'status': 'rejected', 'reasons': ['fewer_than3_valid_secondary_clips']})
                        continue
                    hashes = [source_hashes(bundle, source_ids[0]) for source_ids in (dev, primary_ids, secondary_ids)]
                    if any(len({value[key] for value in hashes}) != 3 for key in ('video_sha256', 'label_sha256')):
                        events.append({'kind': 'source_combination', 'speaker_id': speaker, 'dev_source_stem': dev_stem,
                                       'primary_source_stem': primary, 'secondary_source_stem': secondary,
                                       'status': 'rejected', 'reasons': ['source_sha_not_disjoint']})
                        continue
                    chosen = {'speaker_id': speaker, 'dev_source_stem': dev_stem, 'primary_source_stem': primary,
                              'secondary_source_stem': secondary, 'dev_ids': dev, 'primary_ids': primary_ids,
                              'secondary_ids': secondary_ids, 'train24_ids': primary_ids[:2] + secondary_ids[:1],
                              'train72_ids': primary_ids + secondary_ids,
                              'ranked_dev_sources': stems, 'ranked_training_sources': train_stems}
                    break
                if chosen:
                    break
            if chosen:
                events.append({'kind': 'dev_source', 'speaker_id': speaker, 'source_stem': dev_stem,
                               'rank_sha256': ranking('dev_source', dev_stem), 'status': 'accepted',
                               'reasons': ['dev2_primary6_secondary3_complete']})
                break
            events.append({'kind': 'dev_source', 'speaker_id': speaker, 'source_stem': dev_stem,
                           'rank_sha256': ranking('dev_source', dev_stem), 'status': 'rejected',
                           'reasons': ['no_disjoint_primary6_secondary3_combination']})
        require(chosen is not None, 'Cannot satisfy fixed dev2/train6+3 for speaker: ' + speaker)
        speakers.append(chosen)
        train24.extend(chosen['train24_ids'])
        train72.extend(chosen['train72_ids'])
        dev16.extend(chosen['dev_ids'])
        accepted.extend(chosen['train72_ids'] + chosen['dev_ids'])
    require(len(set(train24)) == 24 and len(set(train72)) == 72 and len(set(dev16)) == 16 and
            set(train24) <= set(train72) and not set(train72).intersection(dev16), 'Final scope counts changed')
    require(len({normalized_target(rows[sid]['text']) for sid in accepted}) == 88, 'Final target collision')
    all_stems = [item[key] for item in speakers for key in ('dev_source_stem', 'primary_source_stem', 'secondary_source_stem')]
    require(len(set(all_stems)) == 24, 'Recording identity used by multiple speakers')
    train_video = {source_hashes(bundle, sid)['video_sha256'] for sid in train72}
    dev_video = {source_hashes(bundle, sid)['video_sha256'] for sid in dev16}
    train_label = {source_hashes(bundle, sid)['label_sha256'] for sid in train72}
    dev_label = {source_hashes(bundle, sid)['label_sha256'] for sid in dev16}
    require(train_video.isdisjoint(dev_video) and train_label.isdisjoint(dev_label), 'Global training/development source SHA overlap')
    records = []
    for sid in train72 + dev16:
        row, entry = rows[sid], bundle['entries'][sid]
        tokens = row['token_ids']
        records.append({'id': sid, 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                        'generation': entry['generation'], 'original_split': row['split'], 'original_role': entry['role'],
                        'role': 'train' if sid in train72 else 'development', 'in_control24': sid in train24,
                        'text': row['text'], 'tokens': tokens, 'frames': row['frames'],
                        'start_time': row['start_time'], 'end_time': row['end_time'], 'video_fps': row['video_fps'],
                        'student_source_indices': native_indices(row), 'ctc_min_frames': len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:])),
                        'normalized_target': normalized_target(row['text']), 'logical_record_sha256': entry['logical_record_sha256'],
                        'logical_record': copy.deepcopy(row), 'source_entry': copy.deepcopy(entry),
                        'input': entry['input'], 'teacher_feature': entry['teacher_feature'] if sid in train72 else None,
                        'new_human_approval_added': False, **source_hashes(bundle, sid)})
    return {'pool_profile': inventory, 'selection': {'train24_ids': train24, 'train72_ids': train72, 'dev16_ids': dev16,
            'records': records, 'speaker_selections': speakers, 'events': events, 'selection_salt': SALT,
            'speaker_order': list(SPEAKERS), 'control_is_subset_of_expanded': True,
            'events_describe_tentative_attempts_final_membership_is_in_id_lists': True,
            'same_speakers_and_training_recordings_across_conditions': True, 'train_recording_ratio_both_conditions': '2:1',
            'new_human_approvals_added': 0, 'development_speakers_unseen': False, 'development_recordings_disjoint': True,
            'prior_evaluation14_recordings_disjoint': True, 'prediction_or_loss_used_for_selection': False,
            'student_frame_overlap_allowed': False, 'historically_training_records': True,
            'untouched_test_set': False, 'model_generalization_not_yet_measured': True,
            'statistics': {'train24_frames': sum(rows[sid]['frames'] for sid in train24),
                           'train72_frames': sum(rows[sid]['frames'] for sid in train72),
                           'dev16_frames': sum(rows[sid]['frames'] for sid in dev16),
                           'train24_recordings': 16, 'train72_recordings': 16, 'dev16_recordings': 8,
                           'train24_per_speaker': 3, 'train72_per_speaker': 9, 'dev16_per_speaker': 2}}}


def build_metadata():
    bundle = source_metadata()
    return {'source_refs': bundle['source_refs'], **select_from_bundle(bundle)}


def hash_selected_caches(selection):
    records = selection['records']
    input_refs = [row['input'] for row in records]
    teacher_refs = [row['teacher_feature'] for row in records if row['role'] == 'train']
    require(len(input_refs) == len({ref['path'] for ref in input_refs}) == 88 and
            len(teacher_refs) == len({ref['path'] for ref in teacher_refs}) == 72 and
            all(row['teacher_feature'] is None for row in records if row['role'] == 'development'), 'Cache scope changed')
    for ref in input_refs + teacher_refs:
        require(Path(ref['path']).suffix == '.npz', 'Expected selected cache NPZ')
        verify(ref)
    return {'inputs': input_refs, 'teacher_features': teacher_refs,
            'input_files_hashed': 88, 'teacher_feature_files_hashed': 72,
            'input_bytes_hashed': sum(Path(ref['path']).stat().st_size for ref in input_refs),
            'teacher_feature_bytes_hashed': sum(Path(ref['path']).stat().st_size for ref in teacher_refs),
            'npz_arrays_decoded': 0, 'development_teacher_feature_files_opened': 0,
            'paired_teacher_pixel_compressed_bytes_read_for_input_file_hash': True,
            'raw_media_or_archive_files_opened': 0}


def preservation_contract(bundle):
    contract = copy.deepcopy(bundle['prior_preservation'])
    # Prior trained tensors/posteriors remain bound by the immutable Step14
    # plan; this preparation phase has no reason to read their payload bytes.
    service_directory = Path(contract['weights_directory']).resolve()
    prior_payloads = [ref for ref in contract['artifacts']
                      if Path(ref['path']).suffix in ('.pt', '.npz') and
                      not Path(ref['path']).resolve().is_relative_to(service_directory)]
    contract['artifacts'] = [ref for ref in contract['artifacts'] if ref not in prior_payloads]
    contract['prior_payload_bindings_preserved_without_rehashing'] = prior_payloads
    contract['historical_trained_checkpoint_or_posterior_payloads_rehashed'] = 0
    contract['artifacts'].extend(bundle['source_refs'].values())
    contract['artifacts'].extend(bundle['prior_execution_code'])
    contract['artifacts'] = list({ref['path']: ref for ref in contract['artifacts']}.values())
    return contract


def verify_preservation(contract):
    for ref in contract['artifacts']:
        verify(ref)
    require(sorted(str(path.resolve()) for path in Path(contract['weights_directory']).glob('*') if path.is_file()) ==
            sorted(contract['weights_inventory']), 'Service weights inventory changed')
    require(all(not Path(path).exists() for path in contract['absent_paths']), 'Previously absent service weight created')


def build(extra_code_paths=()):
    require(not PLAN.exists(), 'Immutable Step15 data plan already exists')
    bundle = source_metadata()
    metadata = {'source_refs': bundle['source_refs'], **select_from_bundle(bundle)}
    caches = hash_selected_caches(metadata['selection'])
    paths = {Path(__file__).resolve(), BASE / 'test_data_plan.py', *(Path(path).resolve() for path in extra_code_paths)}
    require(all(path.is_relative_to(ROOT) and path.suffix == '.py' for path in paths), 'Nonlocal data-plan code')
    value = {'schema': 'lipnet-step15-diversity-data-plan-v1', 'status': 'prepared_pending_training_protocol_and_execution',
             'created_utc': datetime.now(timezone.utc).isoformat(), **metadata,
             'selected_cache_bindings': caches, 'execution_code': [bind(path) for path in sorted(paths, key=str)],
             'preservation': preservation_contract(bundle), 'new_training_updates': 0, 'new_inference_rows': 0,
             'new_teacher_forward_calls': 0, 'source_video_or_audio_read': False, 'service_weights_updated': False,
             'fresh_training_required': True, 'prior_step12_metrics_not_directly_comparable': True,
             'scope': 'Fresh paired old-eight-speaker 24/72 experiment; same speakers, same training recordings, 2:1 recording ratio; development16 is same-speaker recording-disjoint, not unseen-speaker test.'}
    verify_preservation(value['preservation'])
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(PLAN)


def verify_plan():
    value = read(PLAN)
    require(value['schema'] == 'lipnet-step15-diversity-data-plan-v1' and
            value['status'] == 'prepared_pending_training_protocol_and_execution', 'Wrong Step15 data-plan identity')
    expected = build_metadata()
    require(all(value[key] == part for key, part in expected.items()), 'Deterministic data selection changed')
    for ref in value['execution_code']:
        verify(ref)
    require(value['selected_cache_bindings'] == hash_selected_caches(value['selection']), 'Selected cache bindings changed')
    verify_preservation(value['preservation'])
    require(value['new_training_updates'] == value['new_inference_rows'] == value['new_teacher_forward_calls'] == 0 and
            value['service_weights_updated'] is False, 'Preparation scope changed')
    return value


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('metadata', 'build', 'verify'))
    parser.add_argument('--extra-code', action='append', default=[])
    args = parser.parse_args()
    result = build_metadata() if args.command == 'metadata' else build(args.extra_code) if args.command == 'build' else verify_plan()
    print(json.dumps(result, ensure_ascii=False, indent=2))
