"""Independent NumPy-only audit of a future, unexecuted diversity comparison.

Reconstruct selection and schedules from immutable historical JSON; decode only
the selected 88 student inputs and 72 training feature caches. No model, torch,
raw media, historical checkpoint or development feature cache is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import unicodedata

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ANCESTOR = ROOT / 'experiments/posterior_candidates_step14_20261007/independent_audit.py'
ANCESTOR_SHA = 'bc06aa5650fad64de68838bb788fcf1b28331ab33f5396aff171a25801435b4b'
if hashlib.sha256(ANCESTOR.read_bytes()).hexdigest() != ANCESTOR_SHA:
    raise ValueError('Independent pure-JSON helper ancestor changed')
spec = importlib.util.spec_from_file_location('step15_independent_json_helpers', ANCESTOR)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read, bind, sha, verify = old.read, old.bind, old.sha, old.verify
require, same, near = old.require, old.same, old.near
canonical_sha, normalized_text = old.canonical_sha, old.normalized_text
HANDOFF = ROOT / 'experiments/new_inputs_step09_20261007/candidate_handoff.json'
HANDOFF_SHA = '8135c7900ad5ad244e98ddb490fad2a05db115b221edba957cb030603a224ae6'
STEP12 = ROOT / 'experiments/multisentence_step12_20261007'
STEP13 = ROOT / 'experiments/recording_probe_step13_20261007'
STEP14 = ROOT / 'experiments/posterior_candidates_step14_20261007'
STEP14_SUMMARY_SHA = '4f0977e5328b6be26a548bed920820a5bd08b272ef59d404106bab339f937a9a'
SPEAKERS = ('C002', 'C004', 'C017', 'C032', 'C282', 'C293', 'C297', 'C312')
SEEDS, COHORTS = (42, 43), ('train24', 'train72')


def source_indices(row):
    require(type(row['frames']) is int and row['frames'] > 0, 'Invalid frame count')
    require(all(math.isfinite(row[key]) for key in ('start_time', 'end_time', 'video_fps'))
            and 0 <= row['start_time'] < row['end_time'] and row['video_fps'] > 0, 'Invalid source timing')
    return [round((row['start_time'] + index / 25) * row['video_fps']) for index in range(row['frames'])]


def ctc_minimum(tokens):
    require(isinstance(tokens, list) and tokens and all(type(token) is int and 1 <= token <= 68 for token in tokens), 'Invalid target tokens')
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def overlap(first, second):
    if first['source_stem'] != second['source_stem']:
        return False
    return (max(first['start_time'], second['start_time']) < min(first['end_time'], second['end_time'])
            or bool(set(source_indices(first)) & set(source_indices(second))))


def source_metadata():
    handoff = read(HANDOFF, HANDOFF_SHA)
    rows, inputs, features, provenance, references = {}, {}, {}, {}, [bind(HANDOFF)]
    entries = {row['id']: row for row in handoff['records']}
    same(len(entries), len(handoff['records']), 'Duplicate handoff record')
    for generation in ('old', 'new'):
        for suffix in ('data_plan', 'inputs', 'features'):
            reference = handoff['artifacts'][generation + '_' + suffix]
            verify(reference)
            references.append(reference)
        logical = read(handoff['artifacts'][generation + '_data_plan']['path'])
        source_inputs = read(handoff['artifacts'][generation + '_inputs']['path'])
        source_features = read(handoff['artifacts'][generation + '_features']['path'])
        rows.update({row['id']: row for row in logical['records']})
        inputs.update({row['id']: row for row in source_inputs['records']})
        features.update({row['id']: row for row in source_features['records']})
        for row in source_inputs['records']:
            members = row['source_verification']['members']
            provenance[row['id']] = {name + '_sha256': members[name]['sha256'] for name in ('video', 'label')}
    same(set(rows), set(entries), 'Logical/handoff source scope')
    same(set(inputs), set(entries), 'Input manifest/handoff scope')
    pool = handoff['provisional_train_ids']
    same(len(pool), 510, 'Original candidate population')
    same(Counter(entries[sid]['role'] for sid in pool), Counter({'old_train': 441, 'new_provisional_train': 69}), 'Original pool roles')
    for sid, entry in entries.items():
        row = rows[sid]
        same(canonical_sha(row), entry['logical_record_sha256'], 'Logical source row digest')
        same(row['speaker_id'], entry['speaker_id'], 'Handoff speaker')
        same(row['source_stem'], entry['source_stem'], 'Handoff recording')
        same(row['split'], entry['original_split'], 'Handoff original split')
        same(row['frames'], entry['student_frames'], 'Handoff input length')
        item = inputs[sid]
        same(entry['input'], {'path': item['npz_path'], 'sha256': item['npz_sha256']}, 'Input cache reference')
        if sid in pool:
            same(entry['teacher_feature'], features[sid]['feature_file'], 'Training feature cache reference')
    prior12 = read(STEP12 / 'diagnostic_plan.json', old.STEP12_POLICY_SHA)
    prior13 = read(STEP13 / 'diagnostic_plan.json', old.STEP13_POLICY_SHA)
    prior14 = read(STEP14 / 'results/step14_summary.json', STEP14_SUMMARY_SHA)
    same(prior14['status'], 'completed_posterior_candidate_diagnostic', 'Previous stage status')
    references.extend([bind(STEP12 / 'diagnostic_plan.json'), bind(STEP13 / 'diagnostic_plan.json'), bind(STEP14 / 'results/step14_summary.json')])
    probes = list(dict.fromkeys(prior13['selection']['ids'] + prior12['selection']['holdout_ids']))
    same(len(probes), 14, 'Unique historical probe IDs')
    excluded = [sid for sid in rows if sid not in set(pool)]
    same(len(set(excluded)), 122, 'Historical excluded pool')
    return {'handoff': handoff, 'rows': rows, 'entries': entries, 'inputs': inputs, 'features': features,
            'provenance': provenance, 'pool': pool, 'probes': probes, 'excluded': excluded,
            'prior12': prior12, 'prior13': prior13, 'references': references}


def ranking(scope, identity):
    require(scope in ('dev_source', 'train_source', 'dev_clip', 'train_clip'), 'Unknown deterministic ranking scope')
    return hashlib.sha256(('step15-diversity-v1:' + scope + ':' + identity).encode()).hexdigest()


def candidate_inventory(bundle):
    rows, entries, hashes = bundle['rows'], bundle['entries'], bundle['provenance']
    probes = bundle['probes']
    stems = {rows[sid]['source_stem'] for sid in probes}
    videos = {hashes[sid]['video_sha256'] for sid in probes}
    labels = {hashes[sid]['label_sha256'] for sid in probes}
    texts = {normalized_text(rows[sid]['text']) for sid in probes + bundle['excluded']}
    items, identities = [], {}
    for sid in bundle['pool']:
        row, entry = rows[sid], entries[sid]
        if row['source_stem'] in identities:
            same(hashes[sid], identities[row['source_stem']], 'One recording stem has different source hashes')
        else:
            identities[row['source_stem']] = hashes[sid]
        require(row['split'] == entry['original_split'] == 'train' and entry['role'] in ('old_train', 'new_provisional_train')
                and entry['provisional_training_candidate'] is True, 'Forbidden role in training pool')
        tokens = row['token_ids']
        reasons = (['invalid_ctc_target'] if not tokens or not all(type(token) is int and 1 <= token <= 68 for token in tokens)
                   else ['ctc_infeasible'] if type(row['frames']) is not int or row['frames'] < ctc_minimum(tokens) else [])
        if entry['generation'] != 'old' or row['speaker_id'] not in SPEAKERS:
            reasons.append('outside_old8_experiment')
        if row['source_stem'] in stems:
            reasons.append('prior_probe_recording_stem')
        if hashes[sid]['video_sha256'] in videos:
            reasons.append('prior_probe_video_sha')
        if hashes[sid]['label_sha256'] in labels:
            reasons.append('prior_probe_label_sha')
        text = normalized_text(row['text'])
        if not text:
            reasons.append('empty_normalized_target')
        if text in texts:
            reasons.append('prior_probe_or_excluded_normalized_target')
        source_indices(row)
        items.append({'id': sid, 'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                      'generation': entry['generation'], 'frames': row['frames'], 'normalized_target': text,
                      'eligible': not reasons, 'reasons': reasons, **hashes[sid]})
    profiles = []
    for speaker in sorted({item['speaker_id'] for item in items}):
        original = [item for item in items if item['speaker_id'] == speaker]
        eligible = [item for item in original if item['eligible']]
        profiles.append({'speaker_id': speaker, 'generation': original[0]['generation'], 'original_count': len(original),
            'original_recordings': dict(sorted(Counter(item['source_stem'] for item in original).items())),
            'experiment_eligible_count': len(eligible),
            'experiment_eligible_recordings': dict(sorted(Counter(item['source_stem'] for item in eligible).items()))})
    return {'records': items, 'speaker_profiles': profiles, 'original_pool_count': len(items),
        'experiment_eligible_count': sum(item['eligible'] for item in items), 'prior_probe_ids': probes,
        'blocked_recording_stems': sorted(stems), 'blocked_video_sha256': sorted(videos), 'blocked_label_sha256': sorted(labels),
        'blocked_normalized_targets': sorted(texts), 'historical_excluded_ids': bundle['excluded'],
        'pending_listening_ids': bundle['handoff']['pending_listening']}


def select_clips(rows, candidates, count, already, purpose):
    selected, trace = [], []
    reserved = list(already)
    texts = {normalized_text(rows[sid]['text']) for sid in reserved}
    for sid in sorted(candidates, key=lambda value: (ranking(purpose + '_clip', value), value)):
        row = rows[sid]
        reasons = []
        if normalized_text(row['text']) in texts:
            reasons.append('duplicate_selected_normalized_target')
        for previous_id in reserved:
            previous_row = rows[previous_id]
            if row['source_stem'] != previous_row['source_stem']:
                continue
            if max(row['start_time'], previous_row['start_time']) < min(row['end_time'], previous_row['end_time']):
                reasons.append('selected_time_overlap:' + previous_id)
            if set(source_indices(row)) & set(source_indices(previous_row)):
                reasons.append('selected_native_frame_overlap:' + previous_id)
        status = 'not_needed' if len(selected) == count else 'skipped' if reasons else 'selected'
        trace.append({'kind': 'clip', 'scope': purpose, 'id': sid, 'rank_sha256': ranking(purpose + '_clip', sid), 'status': status, 'reasons': reasons})
        if status == 'selected':
            selected.append(sid)
            reserved.append(sid)
            texts.add(normalized_text(row['text']))
    return selected, trace


def independent_selection(bundle):
    inventory = candidate_inventory(bundle)
    rows, provenance = bundle['rows'], bundle['provenance']
    eligible = [item['id'] for item in inventory['records'] if item['eligible']]
    accepted, trace, chosen_speakers = [], [], []
    small, large, development = [], [], []
    for speaker in SPEAKERS:
        ids = [sid for sid in eligible if rows[sid]['speaker_id'] == speaker]
        recordings = {rows[sid]['source_stem'] for sid in ids}
        dev_order = sorted(recordings, key=lambda stem: (ranking('dev_source', stem), stem))
        chosen = None
        for dev_stem in dev_order:
            dev, events = select_clips(rows, [sid for sid in ids if rows[sid]['source_stem'] == dev_stem], 2, accepted, 'dev')
            trace.extend(events)
            if len(dev) != 2:
                trace.append({'kind': 'dev_source', 'speaker_id': speaker, 'source_stem': dev_stem, 'status': 'rejected', 'reasons': ['fewer_than2_valid_dev_clips']})
                continue
            train_order = sorted(recordings - {dev_stem}, key=lambda stem: (ranking('train_source', stem), stem))
            for primary in train_order:
                first, events = select_clips(rows, [sid for sid in ids if rows[sid]['source_stem'] == primary], 6, accepted + dev, 'train')
                trace.extend(events)
                if len(first) != 6:
                    trace.append({'kind': 'primary_source', 'speaker_id': speaker, 'source_stem': primary, 'status': 'rejected', 'reasons': ['fewer_than6_valid_primary_clips']})
                    continue
                for secondary in train_order:
                    if secondary == primary:
                        continue
                    second, events = select_clips(rows, [sid for sid in ids if rows[sid]['source_stem'] == secondary], 3, accepted + dev + first, 'train')
                    trace.extend(events)
                    if len(second) != 3:
                        trace.append({'kind': 'secondary_source', 'speaker_id': speaker, 'source_stem': secondary, 'status': 'rejected', 'reasons': ['fewer_than3_valid_secondary_clips']})
                        continue
                    if any(len({provenance[group[0]][key] for group in (dev, first, second)}) != 3 for key in ('video_sha256', 'label_sha256')):
                        trace.append({'kind': 'source_combination', 'speaker_id': speaker, 'status': 'rejected', 'reasons': ['source_sha_not_disjoint']})
                        continue
                    chosen = {'speaker_id': speaker, 'dev_source_stem': dev_stem, 'primary_source_stem': primary,
                        'secondary_source_stem': secondary, 'dev_ids': dev, 'primary_ids': first, 'secondary_ids': second,
                        'train24_ids': first[:2] + second[:1], 'train72_ids': first + second,
                        'ranked_dev_sources': dev_order, 'ranked_training_sources': train_order}
                    break
                if chosen is not None:
                    break
            trace.append({'kind': 'dev_source', 'speaker_id': speaker, 'source_stem': dev_stem,
                          'status': 'accepted' if chosen else 'rejected',
                          'reasons': ['dev2_primary6_secondary3_complete'] if chosen else ['no_disjoint_primary6_secondary3_combination']})
            if chosen is not None:
                break
        require(chosen is not None, 'No valid dev2/train6+3 configuration for speaker')
        chosen_speakers.append(chosen)
        small.extend(chosen['train24_ids'])
        large.extend(chosen['train72_ids'])
        development.extend(chosen['dev_ids'])
        accepted.extend(chosen['train72_ids'] + chosen['dev_ids'])
    return {'inventory': inventory, 'train24_ids': small, 'train72_ids': large, 'dev16_ids': development,
            'speaker_selections': chosen_speakers, 'events': trace}


def validate_student_arrays(row, item, arrays, metadata):
    same(metadata['record'], row, 'Embedded logical student row')
    same(metadata['id'], row['id'], 'Embedded student ID')
    same(metadata['plan_sha256'], 'c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30', 'Original data plan digest')
    same(metadata['source_verification'], item['source_verification'], 'Embedded source verification')
    pixels = arrays['student_rgb']
    same(pixels.dtype, np.dtype('uint8'), 'Student pixel precision')
    same(pixels.shape, (row['frames'], 96, 96, 3), 'Student pixel shape')
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    same(digest, metadata['student_rgb_sha256'], 'Embedded student pixel digest')
    same(digest, item['student_rgb_sha256'], 'Manifest student pixel digest')
    require(row['frames'] >= ctc_minimum(row['token_ids']), 'CTC-infeasible selected transcript')
    for name, expected in [('student_source_indices', source_indices(row)), ('label_tokens', row['token_ids'])]:
        same(arrays[name].dtype, np.dtype('int64'), 'Student index/token precision')
        same(arrays[name].tolist(), expected, 'Student index/token values')
    require(min(source_indices(row)) >= 0 and max(source_indices(row)) < metadata['source_frame_count'], 'Source indices outside physical video')
    return {'id': row['id'], 'frames': row['frames'], 'student_pixels_sha256': digest,
            'target_tokens_sha256': canonical_sha(row['token_ids']), 'source_indices_sha256': canonical_sha(source_indices(row))}


def validate_feature_arrays(row, item, feature_entry, student_arrays, arrays, metadata):
    same({**metadata, 'feature_file': feature_entry['feature_file'], 'status': 'completed'}, feature_entry,
         'Embedded feature metadata differs from immutable manifest')
    for key, expected in [('id', row['id']), ('speaker_id', row['speaker_id']), ('source_stem', row['source_stem']),
                          ('split', 'train'), ('input_record', item), ('data_record_sha256', canonical_sha(row)),
                          ('aligned_feature_shape', [row['frames'], 512]), ('feature_dtype', 'float32'),
                          ('optimization_steps', 0), ('trainable_parameters', 0), ('audio_used', False),
                          ('decoder_instantiated', False), ('transcript_used_for_teacher_forward', False)]:
        same(metadata[key], expected, 'Frozen teacher provenance ' + key)
    same(metadata['state_sha256_before'], metadata['state_sha256_after'], 'Teacher feature extraction changed teacher state')
    feature = arrays['features']
    same(feature.dtype, np.dtype('float32'), 'Feature precision')
    same(feature.shape, (row['frames'], 512), 'Feature shape')
    require(np.isfinite(feature).all(), 'Nonfinite feature target')
    values = feature.astype(np.float64)
    require((np.linalg.norm(values, axis=1) > 1e-8).all() and not np.all(feature == feature[0]), 'Collapsed feature target')
    digest = hashlib.sha256(feature.tobytes()).hexdigest()
    same(digest, metadata['feature_values_sha256'], 'Feature target digest')
    for name, expected_key in [('student_source_indices', 'student_source_indices'), ('teacher_source_indices', 'teacher_source_indices'),
                               ('gather_indices', 'student_teacher_rows')]:
        same(arrays[name].dtype, np.dtype('int64'), 'Feature index precision')
        require(np.array_equal(arrays[name], student_arrays[expected_key]), 'Feature/input native frame alignment')
    dense, gather = arrays['teacher_source_indices'], arrays['gather_indices']
    require(dense.ndim == 1 and len(dense) > 0 and np.array_equal(dense, np.arange(dense[0], dense[-1] + 1, dtype=np.int64)), 'Teacher native window must be dense')
    require(gather.shape == (row['frames'],) and np.all(gather >= 0) and np.all(gather < len(dense)), 'Teacher gather outside window')
    same(dense[gather].tolist(), source_indices(row), 'Features come from different original frames')
    same(metadata['native_feature_shape'], [1, len(dense), 512], 'Teacher native feature shape')
    return {'id': row['id'], 'frames': row['frames'], 'feature_values_sha256': digest,
            'minimum_row_l2_norm': float(np.linalg.norm(values, axis=1).min())}


def verify_selected_separation(selection, bundle):
    small, large, dev = (selection[key] for key in ('train24_ids', 'train72_ids', 'dev16_ids'))
    same(len(set(small)), 24, 'Control sample size')
    same(len(set(large)), 72, 'Expanded sample size')
    same(len(set(dev)), 16, 'Development sample size')
    require(set(small) < set(large) and set(large).isdisjoint(dev), 'Invalid nested cohorts')
    rows, hashes = bundle['rows'], bundle['provenance']
    ids = large + dev
    same(len({normalized_text(rows[sid]['text']) for sid in ids}), 88, 'Duplicate selected normalized transcript')
    blocked = {normalized_text(rows[sid]['text']) for sid in bundle['probes'] + bundle['excluded']}
    require(not {normalized_text(rows[sid]['text']) for sid in ids} & blocked, 'Selected transcript leaks reserved/excluded target')
    for name in ('source_stem', 'video_sha256', 'label_sha256'):
        value = lambda sid: rows[sid][name] if name == 'source_stem' else hashes[sid][name]
        require({value(sid) for sid in large}.isdisjoint(value(sid) for sid in dev), 'Train/dev recording overlap ' + name)
        require({value(sid) for sid in ids}.isdisjoint(value(sid) for sid in bundle['probes']), 'Prior probe recording overlap ' + name)
    for index, sid in enumerate(ids):
        entry = bundle['entries'][sid]
        require(sid in bundle['pool'] and entry['generation'] == 'old' and entry['role'] == 'old_train'
                and entry['new_human_approval_added'] is False, 'Forbidden original role/approval')
        for other in ids[index + 1:]:
            require(not overlap(rows[sid], rows[other]), 'Selected intervals/native frames overlap')
    for key, wanted in [('train24_ids', 3), ('train72_ids', 9), ('dev16_ids', 2)]:
        same(Counter(rows[sid]['speaker_id'] for sid in selection[key]), Counter({speaker: wanted for speaker in SPEAKERS}), 'Unbalanced speaker cohort')


def audit_data(expected_sha=None):
    path = BASE / 'data_plan.json'
    reference = bind(path)
    data = read(path, expected_sha or reference['sha256'])
    for key, expected in [('schema', 'lipnet-step15-diversity-data-plan-v1'), ('status', 'prepared_pending_training_protocol_and_execution'),
                          ('new_training_updates', 0), ('new_inference_rows', 0), ('new_teacher_forward_calls', 0),
                          ('source_video_or_audio_read', False), ('service_weights_updated', False), ('fresh_training_required', True),
                          ('prior_step12_metrics_not_directly_comparable', True)]:
        same(data[key], expected, 'Data-plan identity/scope ' + key)
    bundle = source_metadata()
    expected_refs = {'handoff': bind(HANDOFF), 'step12_policy': bind(STEP12 / 'diagnostic_plan.json'),
        'step13_policy': bind(STEP13 / 'diagnostic_plan.json'), 'step13_summary': bind(STEP13 / 'results/step13_summary.json'),
        'step14_summary': bind(STEP14 / 'results/step14_summary.json')}
    prior14_summary = read(STEP14 / 'results/step14_summary.json', STEP14_SUMMARY_SHA)
    expected_refs.update(step14_policy=prior14_summary['artifacts']['diagnostic_plan.json'],
                         step14_audit=prior14_summary['artifacts']['results/independent_audit.json'])
    for generation in ('old', 'new'):
        for kind in ('data_plan', 'inputs', 'features'):
            expected_refs[generation + '_' + kind] = bundle['handoff']['artifacts'][generation + '_' + kind]
    same(data['source_refs'], expected_refs, 'Source metadata reference scope')
    computed = independent_selection(bundle)
    same(data['pool_profile'], computed['inventory'], 'Independent candidate inventory')
    selection, rows = data['selection'], bundle['rows']
    for key in ('train24_ids', 'train72_ids', 'dev16_ids', 'speaker_selections'):
        same(selection[key], computed[key], 'Independent selection ' + key)
    verify_selected_separation(selection, bundle)
    same(len(selection['events']), len(computed['events']), 'Selection event count')
    for actual, reconstructed in zip(selection['events'], computed['events'], strict=True):
        for key, value in reconstructed.items():
            same(actual[key], value, 'Independent selection event ' + key)
    for key, expected in [('selection_salt', 'step15-diversity-v1:'), ('speaker_order', list(SPEAKERS)),
                          ('control_is_subset_of_expanded', True), ('same_speakers_and_training_recordings_across_conditions', True),
                          ('train_recording_ratio_both_conditions', '2:1'), ('new_human_approvals_added', 0),
                          ('development_speakers_unseen', False), ('development_recordings_disjoint', True),
                          ('prior_evaluation14_recordings_disjoint', True), ('prediction_or_loss_used_for_selection', False),
                          ('student_frame_overlap_allowed', False), ('historically_training_records', True),
                          ('untouched_test_set', False), ('model_generalization_not_yet_measured', True)]:
        same(selection[key], expected, 'Selection meaning/scope ' + key)
    ids = selection['train72_ids'] + selection['dev16_ids']
    same([record['id'] for record in selection['records']], ids, 'Selected record order')
    for selected in selection['records']:
        sid, row = selected['id'], rows[selected['id']]
        entry = bundle['entries'][sid]
        expected = {'id': sid, 'generation': 'old', 'original_split': 'train', 'original_role': 'old_train',
            'role': 'train' if sid in selection['train72_ids'] else 'development', 'in_control24': sid in selection['train24_ids'],
            'tokens': row['token_ids'], 'student_source_indices': source_indices(row), 'ctc_min_frames': ctc_minimum(row['token_ids']),
            'normalized_target': normalized_text(row['text']), 'logical_record_sha256': entry['logical_record_sha256'],
            'logical_record': row, 'source_entry': entry, 'input': entry['input'],
            'teacher_feature': entry['teacher_feature'] if sid in selection['train72_ids'] else None,
            'new_human_approval_added': False, **bundle['provenance'][sid]}
        expected.update({key: row[key] for key in ('speaker_id', 'source_stem', 'text', 'frames', 'start_time', 'end_time', 'video_fps')})
        same(selected, expected, 'Selected logical record/provenance')
    frames = {group: sum(rows[sid]['frames'] for sid in selection[group + '_ids']) for group in (*COHORTS, 'dev16')}
    same(selection['statistics'], {**{name + '_frames': value for name, value in frames.items()},
        'train24_recordings': 16, 'train72_recordings': 16, 'dev16_recordings': 8,
        'train24_per_speaker': 3, 'train72_per_speaker': 9, 'dev16_per_speaker': 2}, 'Selection statistics')
    references = bundle['references'] + list(data['source_refs'].values()) + data['execution_code'] + [reference]
    for ref in references:
        verify(ref)
    caches = data['selected_cache_bindings']
    expected_inputs = [bundle['entries'][sid]['input'] for sid in ids]
    expected_features = [bundle['entries'][sid]['teacher_feature'] for sid in selection['train72_ids']]
    same(caches['inputs'], expected_inputs, 'Selected input cache references')
    same(caches['teacher_features'], expected_features, 'Train-only feature cache references')
    same(len({ref['path'] for ref in expected_inputs}), 88, 'Duplicate input cache path')
    same(len({ref['path'] for ref in expected_features}), 72, 'Duplicate feature cache path')
    for key, expected in [('input_files_hashed', 88), ('teacher_feature_files_hashed', 72), ('npz_arrays_decoded', 0),
                          ('development_teacher_feature_files_opened', 0), ('paired_teacher_pixel_compressed_bytes_read_for_input_file_hash', True),
                          ('raw_media_or_archive_files_opened', 0)]:
        same(caches[key], expected, 'Preparation cache scope ' + key)
    student_results, feature_results = [], []
    for sid in ids:
        row, entry, item = rows[sid], bundle['entries'][sid], bundle['inputs'][sid]
        verify(entry['input'])
        with np.load(entry['input']['path'], allow_pickle=False) as saved:
            same(set(saved.files), {'student_rgb', 'student_source_indices', 'label_tokens', 'teacher_bgr',
                                   'teacher_source_indices', 'student_teacher_rows', 'metadata_json'}, 'Historical paired cache keys')
            keys = ['student_rgb', 'student_source_indices', 'label_tokens']
            if sid in selection['train72_ids']:
                keys.extend(['teacher_source_indices', 'student_teacher_rows'])
            arrays = {key: saved[key] for key in keys}
            metadata = json.loads(str(saved['metadata_json'].item()))
        student_results.append(validate_student_arrays(row, item, arrays, metadata))
        if sid in selection['train72_ids']:
            verify(entry['teacher_feature'])
            with np.load(entry['teacher_feature']['path'], allow_pickle=False) as saved:
                same(set(saved.files), {'features', 'student_source_indices', 'teacher_source_indices', 'gather_indices', 'metadata_json'}, 'Feature cache keys')
                features = {key: saved[key] for key in saved.files if key != 'metadata_json'}
                metadata = json.loads(str(saved['metadata_json'].item()))
            feature_results.append(validate_feature_arrays(row, item, bundle['features'][sid], arrays, features, metadata))
    same(len({row['student_pixels_sha256'] for row in student_results}), 88, 'Duplicate selected student pixel payload')
    same(len({row['feature_values_sha256'] for row in feature_results}), 72, 'Duplicate training feature payload')
    same(caches['input_bytes_hashed'], sum(Path(ref['path']).stat().st_size for ref in expected_inputs), 'Selected input byte accounting')
    same(caches['teacher_feature_bytes_hashed'], sum(Path(ref['path']).stat().st_size for ref in expected_features), 'Selected feature byte accounting')
    references.extend(expected_inputs + expected_features)
    return data, bundle, references, {'student': student_results, 'teacher_features': feature_results, 'frames': frames,
                                    'selection_event_count': len(computed['events']), 'eligible_count': computed['inventory']['experiment_eligible_count']}


def schedule_for(ids, rows, seed):
    require(seed in SEEDS, 'Unknown experiment seed')
    grouped = {speaker: [sid for sid in ids if rows[sid]['speaker_id'] == speaker] for speaker in SPEAKERS}
    count = len(ids) // 8
    require(len(ids) in (24, 72) and len(set(ids)) == len(ids) and all(len(group) == count for group in grouped.values()), 'Unbalanced schedule population')
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    stream = []
    for round_number in range(900):
        order = sorted(SPEAKERS, key=lambda speaker: (digest(f'step15-schedule-v1:speakers:{seed}:{round_number}:{speaker}'), speaker))
        epoch, index = divmod(round_number, count)
        for speaker in order:
            choices = sorted(grouped[speaker], key=lambda sid: (digest(f'step15-schedule-v1:clips:{seed}:{speaker}:{epoch}:{sid}'), sid))
            stream.append(choices[index])
    return [stream[index:index + 2] for index in range(0, len(stream), 2)]


def learning_rate(update):
    require(type(update) is int and 1 <= update <= 3600, 'One-based update outside budget')
    if update > 3000:
        return .0001
    if update > 2400:
        return .0003
    return .001


def corpus_profile(rows, groups):
    profiles, vocab = {}, {}
    for group, ids in groups.items():
        text = [unicodedata.normalize('NFC', rows[sid]['text']) for sid in ids]
        tokens = [token for sid in ids for token in rows[sid]['tokens']]
        words = [word for sentence in text for word in sentence.split()]
        syllables = {char for sentence in text for char in sentence if 0xAC00 <= ord(char) <= 0xD7A3}
        vocab[group] = {'tokens': set(tokens), 'words': set(words), 'syllables': syllables}
        profiles[group] = {'utterances': len(ids), 'frames': sum(rows[sid]['frames'] for sid in ids),
            'unique_token_ids': sorted(set(tokens)), 'unique_token_classes': len(set(tokens)),
            'hangul_syllable_types': len(syllables), 'whitespace_word_types': len(set(words)), 'whitespace_word_occurrences': len(words)}
    return {'groups': profiles,
        'dev16_values_absent_from_training': {group: {kind: sorted(vocab['dev16'][kind] - vocab[group][kind]) for kind in ('tokens', 'words', 'syllables')}
                                            for group in COHORTS},
        'post_selection_description_only': True, 'used_to_select_or_tune': False,
        'words_are_surface_forms_not_dictionary_or_morphological_units': True,
        'unseen_word_does_not_imply_impossible_jamo_ctc_output': True}


def audit_design(data, bundle, expected_sha=None):
    path = BASE / 'training_design.json'
    reference = bind(path)
    design = read(path, expected_sha or reference['sha256'])
    selected = data['selection']
    rows = {row['id']: row for row in selected['records']}
    groups = {cohort: selected[cohort + '_ids'] for cohort in (*COHORTS, 'dev16')}
    for key, expected in [('schema', 'lipnet-step15-diversity-training-design-v1'), ('status', 'prepared_not_trained'),
                          ('data_plan', bind(BASE / 'data_plan.json')), ('prior_initialization_and_runtime_reference', bind(STEP12 / 'diagnostic_plan.json')),
                          ('seeds', list(SEEDS)), ('arm', 'B_feature'), ('cohorts', groups),
                          ('post_selection_corpus_profile', corpus_profile(rows, groups)),
                          ('runtime_reference_not_rechecked_on_gpu_in_step15', bundle['prior12']['runtime']),
                          ('optimizer_updates_executed', 0), ('new_model_inferences', 0), ('teacher_forwards', 0),
                          ('service_weights_updated', False), ('downloads', 0), ('new_human_approvals_added', 0)]:
        same(design[key], expected, 'Training design identity/scope ' + key)
    same(design['generator_sources'], [bind(BASE / 'experiment_plan.py'), bind(BASE / 'test_experiment_plan.py')], 'Training design generator source')
    same(design['speaker_sampling'], {'salt': 'step15-schedule-v1:', 'speakers_per_round': 8, 'rounds': 900,
        'paired_speaker_order': True, 'different_speakers_per_update': True, 'primary_to_secondary_recording_ratio': '2:1 in each cohort'}, 'Speaker schedule rule')
    same(design['optimizer'], {'name': 'Adam', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
        'gradient_clip': 1., 'precision': 'FP32 GPU student and FP32 CPU CTC'}, 'Optimizer rule')
    same(design['loss'], {'ctc': 1., 'feature': 1., 'feature_type': 'cached visual teacher features; no teacher forward'}, 'Feature loss rule')
    model = design['model']
    for key, expected in [('architecture', 'lipnet-compact-v1'), ('fresh_initialization_each_run', True),
                          ('resume_prior_trained_model', False), ('same_initial_state_within_seed', True), ('no_audio_at_inference', True)]:
        same(model[key], expected, 'Model design ' + key)
    same([(run['seed'], run['cohort']) for run in design['runs']], [(seed, cohort) for seed in SEEDS for cohort in COHORTS], 'Four planned run order')
    schedules, total_frames = {}, 0
    for run in design['runs']:
        seed, cohort = run['seed'], run['cohort']
        ids = groups[cohort]
        schedule = schedule_for(ids, rows, seed)
        frames = sum(rows[sid]['frames'] for batch in schedule for sid in batch)
        total_frames += frames
        expected = {'run_id': f'seed{seed}_{cohort}', 'seed': seed, 'cohort': cohort, 'arm': 'B_feature',
            'initial_state_hashes_expected': bundle['prior12']['initial_state_hashes_expected'][str(seed)]['B_feature'],
            'train_ids': ids, 'schedule': schedule, 'updates': 3600, 'clips_per_update': 2, 'presentations': 7200,
            'presentations_per_clip': 7200 // len(ids), 'presentations_per_speaker': 900, 'student_frames_presented': frames,
            'learning_rate_segments': [{'start_update': 1, 'end_update': 2400, 'lr': .001},
                                       {'start_update': 2401, 'end_update': 3000, 'lr': .0003},
                                       {'start_update': 3001, 'end_update': 3600, 'lr': .0001}],
            'checkpoint_updates': [600, 1200, 1800, 2400, 3000, 3600], 'normal_train_evaluation_updates': [2400, 3600],
            'normal_dev_evaluation_updates': [1200, 2400, 3000, 3600], 'static_shuffled_dev_evaluation_updates': [3600],
            'historical_probe_normal_evaluation_updates': [3600], 'final_comparison_update': 3600,
            'planned_output_directory': str(BASE / 'future_runs' / f'seed{seed}' / cohort)}
        same(run, expected, 'Independent complete run design')
        exposures = Counter(sid for batch in schedule for sid in batch)
        same(exposures, Counter({sid: 7200 // len(ids) for sid in ids}), 'Exact per-clip exposure count')
        require(set(exposures).isdisjoint(groups['dev16'] + bundle['probes'] + bundle['excluded']), 'Development/reserved clip in future schedule')
        speakers = [[rows[sid]['speaker_id'] for sid in batch] for batch in schedule]
        require(all(len(set(batch)) == 2 for batch in speakers), 'Same speaker appears twice in one optimizer update')
        schedules[seed, cohort] = speakers
    for seed in SEEDS:
        same(schedules[seed, 'train24'], schedules[seed, 'train72'], 'Speaker order differs across cohort sizes')
    evaluation = design['evaluation']
    same(evaluation['followup_expansion_gate'], {'both_seed_deltas_strictly_below_zero': True,
        'mean_delta_at_most_percentage_points': -5., 'completed_runs': 4, 'inference_failures': 0,
        'statistical_significance_claim': False, 'meaning': 'A practical predeclared signal for further research, not webcam readiness.'}, 'Predeclared follow-up criterion')
    same(evaluation['historical_probe14_ids'], bundle['probes'], 'Historical probe scope/order')
    expected_probe_groups = {'current_known_speakers': [sid for sid in bundle['probes'] if bundle['rows'][sid]['speaker_id'] in SPEAKERS],
                             'current_unseen_speakers': [sid for sid in bundle['probes'] if bundle['rows'][sid]['speaker_id'] not in SPEAKERS]}
    same(evaluation['historical_probe14_groups'], expected_probe_groups, 'Historical probe speaker roles')
    same(evaluation['dev_used_for_training_early_stopping_or_model_choice'], False, 'Development entered training/model selection')
    budget = design['budget']
    expected_budget = {'runs': 4, 'updates_per_run': 3600, 'total_updates': 14400, 'total_presentations': 28800,
        'student_frames_presented': total_frames, 'checkpoint_count': 24, 'neural_evaluation_rows_planned': 824,
        'evaluation_rows_breakdown': {'train_normal': 384, 'dev_normal': 256, 'dev_static_shuffled': 128, 'historical_probe_normal': 56},
        'max_minutes_per_run_including_resume': 35, 'max_minutes_all_jobs': 150, 'minimum_free_disk_gib_before_training': 5,
        'same_updates_not_same_frames_flops_or_wall_time': True, 'automatic_extension': False, 'original_datasets_or_checkpoints_deleted': False}
    same(budget, expected_budget, 'Fixed future execution/evaluation budget')
    require(design['step16_execution_requirements'] and len(design['step16_execution_requirements']) >= 7, 'Missing execution implementation requirements')
    same(design['interruptions']['restart_from_scratch_for_better_result'], False, 'Outcome-dependent rerun allowed')
    verify(reference)
    return reference, total_frames


def save_new(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), 'Refusing to overwrite audit artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def main():
    import sys
    require('torch' not in sys.modules, 'Independent preparation audit cannot import torch')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-data-sha')
    parser.add_argument('--expected-design-sha')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or BASE / 'results/independent_audit.json'
    require(not output.exists(), 'Refusing to overwrite existing audit')
    require(not list(BASE.rglob('*.pt')), 'A model checkpoint was created during preparation-only stage')
    future = BASE / 'future_runs'
    require(not future.exists() or not any(path.is_file() for path in future.rglob('*')), 'Unexpected executed run files during preparation')
    started = time.perf_counter()
    sources = [bind(Path(__file__)), bind(BASE / 'test_independent_audit.py'), bind(ANCESTOR)]
    data, bundle, references, verification = audit_data(args.expected_data_sha)
    design_ref, planned_frames = audit_design(data, bundle, args.expected_design_sha)
    preservation = data['preservation']
    old.verify_preservation(preservation)
    for ref in references + sources:
        verify(ref)
    verify(design_ref)
    require('torch' not in sys.modules, 'Independent audit imported torch')
    result = {'schema': 'lipnet-step15-independent-audit-v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
        'test_source': bind(BASE / 'test_independent_audit.py'), 'audit_sources': sources,
        'data_plan': bind(BASE / 'data_plan.json'), 'training_design': design_ref,
        'metadata_records_reconstructed': 632, 'historical_training_pool': 510,
        'eligible_old_clips': verification['eligible_count'],
        'selection_events_independently_checked': verification['selection_event_count'],
        'student_input_npz_decoded': 88, 'training_teacher_feature_npz_decoded': 72,
        'development_teacher_feature_npz_opened': 0, 'teacher_pixel_arrays_decoded': 0,
        'student_frames_by_cohort': verification['frames'],
        'paired_teacher_pixel_compressed_bytes_read_for_input_hash': True,
        'student_cache_verification': verification['student'], 'training_feature_verification': verification['teacher_features'],
        'planned_runs': 4, 'planned_optimizer_updates': 14400, 'planned_presentations': 28800,
        'planned_student_frames': planned_frames, 'planned_neural_evaluation_rows': 824,
        'planned_lr_boundaries_verified': [1, 2400, 2401, 3000, 3001, 3600],
        'paired_speaker_schedules_independently_verified': True, 'training_control_is_nested_subset': True,
        'dev16_recordings_disjoint_from_training_and_prior_probes': True,
        'selected_normalized_transcripts_disjoint_from_historical_dev_hold_and_probes': True,
        'selected_source_intervals_and_frames_disjoint': True,
        'optimizer_updates_executed': 0, 'new_model_inferences': 0, 'teacher_forwards': 0,
        'checkpoint_tensors_loaded': 0, 'historical_model_checkpoint_payloads_rehashed': 0,
        'raw_media_opened': 0, 'pytorch_imported': False, 'new_human_approvals_added': 0,
        'service_weights_preserved': True, 'source_and_code_hashes_unchanged_before_and_after_audit': True,
        'runtime_and_model_initialization_checked_on_gpu': False, 'execution_implementation_still_required': True,
        'unseen_speaker_test_claim': False, 'model_improvement_established': False,
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': ['Preparation and cache integrity do not establish improved recognition.',
                       'The planned comparison uses equal update counts, with 300 versus 100 exposures per clip and unequal frame budgets.',
                       'All selected records came from historical old training data; development16 is same-speaker recording-disjoint development.',
                       'GPU runtime, fresh initial-state reproduction, LR scheduling and checkpoint-resume behavior must be checked before Step16 training.']}
    print(json.dumps({'status': 'passed', 'output': save_new(output, result)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

