"""Independent read-only verification of new120 source and review artifacts.

No source producer, tokenizer, alignment producer, model, or training code is
imported. Archive payloads are never opened. Saved review PCM supplies the signal
evidence; original D: archives are checked with stat only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import statistics
import time
import wave

import numpy as np
from PIL import Image
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
AUDIT = ROOT / 'experiments/data_integrity_20261002'
SELECTION = ROOT / 'results/step18-intake/run01/selection.json'
SELECTION_SHA = 'abadd58b0fdf15197c1ced38b01033b464e87f439b8ae2bd43405098be07cc9c'
SOURCE_PLAN = BASE / 'source_plan.json'
SOURCE_SHA = '5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929'
MEDIA_PLAN = BASE / 'media_plan.json'
MEDIA_SHA = '1e866dc03e70161eabc16b5fe5b0afc88eae10eefcafbc9e06718691091aee3d'
OVERALL_SHA = 'ca411651e6f50e66a7eb44ebfaad48c7c3be7dc5bd10ea714044eaed3ba9e625'
INPUT_RATE, GRID_RATE = 8000, 48000
GATES = {'minimum_correlation': .99, 'minimum_far_peak_ratio': 1.5,
         'minimum_filtered_rms': 1e-4, 'minimum_search_boundary_clearance_seconds': .5,
         'maximum_offset_range_exclusive_seconds': .02}


def require(value, message):
    if not value:
        raise ValueError(message)


def same(value, expected, message):
    require(value == expected, message)


def close(value, expected, message, atol=1e-8):
    require(isinstance(value, (int, float)) and math.isfinite(value)
            and math.isfinite(expected) and abs(value - expected) <= atol, message)


def local(path):
    result = Path(path).resolve()
    require(result.is_relative_to(ROOT), 'Only workspace evidence files may be opened')
    return result


def sha(path):
    with local(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path):
    return {'path': str(local(path)), 'sha256': sha(path)}


def verify(ref, expected=None):
    path = local(ref['path'])
    if expected is not None:
        same(path, local(expected), 'Unexpected evidence path')
    same(sha(path), ref['sha256'], 'Changed evidence: ' + str(path))
    return path


def read(path, expected_sha=None):
    data = local(path).read_bytes()
    if expected_sha is not None:
        same(hashlib.sha256(data).hexdigest(), expected_sha, 'Changed JSON evidence')
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(data.decode('utf-8-sig'), parse_constant=invalid)


def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, 'Ambiguous duplicate: ' + key)
        result[row[key]] = row
    return result


def validate_roles(plan, selection, role_map, protected):
    same(plan['records'], selection['records'], 'Original selection records changed')
    rows = plan['records']
    same(len(rows), 120, 'Wrong selected denominator')
    same(Counter(r['split'] for r in rows), {'train': 96, 'val': 24}, 'Changed selected roles')
    same(len({r['id'] for r in rows}), 120, 'Duplicated selected ID')
    same(len({r['text'] for r in rows}), 120, 'Duplicated selected text')
    roles = unique(role_map['records'], 'source_stem')
    guard = protected['guards']
    for field, name in (('id', 'blocked_existing_sample_ids'), ('speaker_id', 'blocked_existing_speaker_ids'),
                        ('recording_id', 'blocked_existing_recording_ids'), ('text', 'blocked_existing_normalized_texts')):
        require(not {r[field] for r in rows} & set(guard[name]), 'Protected role/content overlap: ' + field)
    for key, values in guard.items():
        digest = hashlib.sha256(json.dumps(sorted(values), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        same(digest, protected['guard_sha256'][key], 'Changed protected guard digest')
    same({r['speaker_id'] for r in rows if r['split'] == 'val'}, {'C040', 'C059', 'C769'}, 'Changed new development speakers')
    for source in plan['sources']:
        selected = [r for r in rows if r['id'].rsplit('_', 1)[0] == source['stem']]
        same(len(selected), 4, 'Wrong fixed per-recording quota')
        same(source['record_ids'], [r['id'] for r in selected], 'Changed selected source order/IDs')
        same({r['split'] for r in selected}, {source['split']}, 'Mixed source split')
        same({r['speaker_id'] for r in selected}, {source['speaker_id']}, 'Mixed source speaker')
        role = roles[source['stem']]
        same(source['readiness_role'], role['readiness_role'], 'Changed source readiness role')
        same(role['stage18_pilot_split'], source['split'], 'Changed role-map split')
        same(set(role['stage18_pilot_selected_clip_ids']), set(source['record_ids']), 'Role-map IDs differ')
        same(role['original_v4_split'], None, 'Existing cohort entered new120')
        same(role['protected_reference_cohorts'], [], 'Protected recording entered new120')
        same(role['stage7_planned_split'], None, 'Old Step5 split entered new120')
        same(source['training_ready'], False, 'Premature training approval')
        same(source['old59_time_offset_applied'], False, 'Old recording timing reused')
        same(source['verified_new_time_offset_seconds'], None, 'Source plan invents new timing')
    new_dev = [r for r in role_map['records'] if r['speaker_id'] in {'C040', 'C059', 'C769'}]
    same(len(new_dev), 36, 'Incomplete speaker-wide development protection')
    require(all(r['speaker_role'] == 'new_pilot_development' for r in new_dev), 'New development speaker reassigned')
    same(role_map['original_v4_contract']['val'], {'speaker_ids': ['C025', 'C294'], 'clips': 20}, 'Old validation changed')
    same(role_map['original_v4_contract']['test'], {'speaker_ids': ['C138', 'C281', 'C334', 'C428', 'C558'], 'clips': 50}, 'Final test changed')


def audit_source_plan():
    plan = read(SOURCE_PLAN, SOURCE_SHA)
    selection = read(SELECTION, SELECTION_SHA)
    same(plan['schema'], 'lipnet-new120-audited-tar-source-plan-v1', 'Wrong source plan schema')
    same(plan['selection'], bind(SELECTION), 'Wrong selected parent')
    same(plan['parent_plan'], plan['selection'], 'Different source parent')
    require(len(plan['bindings']) == len({r['path'] for r in plan['bindings']}), 'Duplicate source evidence')
    for ref in plan['bindings'] + plan['code']:
        verify(ref)
    overall = read(AUDIT / 'overall_summary.json', OVERALL_SHA)
    same(plan['overall_audit'], bind(AUDIT / 'overall_summary.json'), 'Wrong original integrity report')
    same(overall['evidence_consistency_pass'], True, 'Original integrity did not pass')
    evidence = {name: read(AUDIT / name, digest) for name, digest in overall['evidence_sha256'].items()}
    role_map = evidence['archive_results/readiness_role_map.json']
    protected_ref = role_map['evidence']['protected_reference']
    protected = read(verify(protected_ref), protected_ref['sha256'])
    validate_roles(plan, selection, role_map, protected)
    index = defaultdict(list)
    for member in evidence['archive_results/member_index.json']['members']:
        if member['is_file']:
            index[PurePosixPath(member['member_name']).name].append(member)
    archives = unique(evidence['archive_results/summary.json']['archives'], 'archive_path')
    labels = unique(read(AUDIT / 'label_results/label_records.json'), 'stem')
    decoder_refs = unique(evidence['archive_results/media_crosschecks.json']['decoder_result_evidence'], 'path')
    decode_plan = unique(evidence['decode_results/plan.json']['records'], 'name')
    fronts = unique(overall['frontal_sources'], 'video')
    impact = unique([r for r in evidence['label_results/selection_impact.json']['records']
                     if r['group'] == 'new_stage18_pilot120'], 'id')
    same(set(impact), {r['id'] for r in selection['records']}, 'Nominal selected-range evidence incomplete')
    seen, totals, lengths, used_archives, changes = [], Counter(), [], {}, []
    same(len(plan['sources']), 30, 'Wrong source count')
    same(len(unique(plan['sources'], 'stem')), 30, 'Duplicate source recording')
    for source in plan['sources']:
        stem = source['stem']
        selected = [r for r in selection['records'] if r['id'].rsplit('_', 1)[0] == stem]
        seen.extend(source['record_ids'])
        label = labels[stem]
        same(label['errors'], [], 'Selected label has technical errors')
        same(fronts[stem + '.mp4']['status'], 'decode_and_metadata_pass', 'Selected video has technical errors')
        for row in selected:
            checked = impact[row['id']]
            for key in ('errors', 'record_level_errors', 'selected_sentence_issues'):
                same(checked[key], [], 'Selected nominal-range errors')
            same(checked['start_time'], row['start_time'], 'Nominal start differs')
            same(checked['end_time'], row['end_time'], 'Nominal end differs')
            same(checked['label_sha256_matches_frozen_selection'], True, 'Selection label mismatch')
        copied = read(verify(source['label_copy']))[0]
        annotations = unique(copied['Sentence_info'], 'ID')
        for row in selected:
            annotation = annotations[row['sentence_id']]
            for key, original_key in (('sentence_text', 'source_text'), ('start_time', 'start_time'), ('end_time', 'end_time')):
                same(annotation[key], row[original_key], 'Original selected label changed')
        same(set(source['members']), {'video', 'audio', 'label'}, 'Incomplete member roles')
        for role, extension in (('video', '.mp4'), ('audio', '.wav'), ('label', '.json')):
            name = stem + extension
            same(len(index[name]), 1, 'Ambiguous or absent TAR member')
            indexed, member = index[name][0], source['members'][role]
            structure = archives[indexed['archive_path']]
            snapshot = structure['after']
            same(structure['before'], snapshot, 'Archive changed during original audit')
            same(structure['status'], 'structure_pass', 'Invalid original TAR structure')
            same(PureWindowsPath(member['archive_path']).drive.upper(), 'D:', 'Non-D source')
            require(indexed['is_file'] and not indexed['is_dir'] and indexed['type'] in ('0', '\x00', ''), 'Nonregular member')
            require('..' not in PurePosixPath(indexed['member_name']).parts and not indexed['member_name'].startswith('/'), 'Unsafe TAR member')
            for key, expected in (('archive_path', indexed['archive_path']), ('member_name', indexed['member_name']),
                    ('offset', indexed['offset_data']), ('size', indexed['size']), ('archive_size', snapshot['size']),
                    ('archive_mtime_ns', snapshot['mtime_ns'])):
                same(member[key], expected, 'TAR member reference differs: ' + key)
            require(member['offset'] >= 512 and member['offset'] % 512 == 0 and member['size'] > 0
                    and member['offset'] + member['size'] <= member['archive_size'], 'Invalid member bounds')
            if role == 'label':
                same(member['sha256'], label['sha256'], 'Audited label SHA differs')
                require(all(r['label_sha256'] == member['sha256'] for r in selected), 'Selected label SHA differs')
                same(source['label_copy']['sha256'], member['sha256'], 'Local label copy differs')
            else:
                ref = source['decode_evidence'][role]
                same(ref, decoder_refs[ref['path']], 'Decoder evidence unanchored')
                decoded = read(verify(ref))
                for key, value in decode_plan[name].items():
                    same(decoded[key], value, 'Audited decoder plan changed')
                same(decoded['member_sha256'], member['sha256'], 'Decoded payload SHA differs')
                same(decoded['status'], 'full_decode_pass', 'Source decode incomplete')
                same(decoded['returncode'], 0, 'Source decode failed')
                for key, expected in (('archive', member['archive_path']), ('member', member['member_name']),
                                      ('offset', member['offset']), ('size', member['size']), ('archive_stat', snapshot)):
                    same(decoded[key], expected, 'Decoder/TAR source binding differs')
                if role == 'video':
                    same(source['video_metadata'], decoded['video_metadata'], 'Video metadata differs')
                    same(source['video_fps'], decoded['video_metadata']['fps'], 'Video FPS differs')
                    same(source['video_frame_count'], decoded['video_metadata']['frame_count'], 'Video frame count differs')
            totals[role] += member['size']
            used_archives[member['archive_path']] = {'path': member['archive_path'], **snapshot}
        same(source['video_duration_seconds'], source['video_frame_count'] / source['video_fps'], 'Physical duration differs')
        lengths.append(sum(m['size'] for m in source['members'].values()))
        old_sizes = {r['source_video']['size_bytes'] for r in selected}
        same(len(old_sizes), 1, 'Mixed historical source sizes')
        old_size = old_sizes.pop()
        same(source['historical_G_size_bytes'], old_size, 'Historical source metadata rewritten')
        same(source['historical_G_size_matches_audited_D'], old_size == source['members']['video']['size'], 'Incorrect old/new size comparison')
        if old_size != source['members']['video']['size']:
            changes.append({'stem': stem, 'historical_G_size_bytes': old_size, 'audited_D_size_bytes': source['members']['video']['size']})
    same(len(seen), 120, 'Missing source record coverage')
    same(set(seen), {r['id'] for r in selection['records']}, 'Extra or missing source selected ID')
    same(plan['historical_G_size_differences'], changes, 'Historical truncation differences not preserved')
    same(plan['archives'], [used_archives[k] for k in sorted(used_archives)], 'Wrong physical archive inventory')
    same(len(plan['archives']), 6, 'Wrong archive count')
    for archive in plan['archives']:
        state = Path(archive['path']).stat()
        same((state.st_size, state.st_mtime_ns), (archive['size'], archive['mtime_ns']), 'Current D archive stat changed')
    same(plan['storage']['bytes_by_role'], dict(totals), 'Incorrect source byte totals')
    same(plan['storage']['total_mapped_member_bytes'], sum(totals.values()), 'Incorrect total source bytes')
    same(plan['storage']['max_one_recording_combined_bytes'], max(lengths), 'Incorrect temporary source bound')
    same(plan['scope']['training_ready'], False, 'Source plan prematurely approves training')
    return plan


def pcm(spec, rate, seconds, expected_start=None):
    path = verify(spec)
    require(path.is_relative_to(BASE), 'Saved review audio outside new Step6')
    if expected_start is not None:
        close(spec['start_seconds'], expected_start, 'PCM start differs', 1e-10)
    with wave.open(str(path), 'rb') as stream:
        same((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()),
             (rate, 1, 2, 'NONE'), 'Wrong saved PCM format')
        frames = stream.getnframes()
        same(frames, round(seconds * rate), 'Truncated or padded PCM sample count')
        raw = stream.readframes(frames)
        same(len(raw), frames * 2, 'Truncated saved PCM bytes')
    value = np.frombuffer(raw, dtype='<i2').astype(np.float64) / 32768.
    require(np.isfinite(value).all(), 'Nonfinite saved PCM')
    return value


def independent_score(reference, search):
    """Independent full FFT numerator and sliding FFT moments, direct peak check."""
    require(reference.ndim == search.ndim == 1 and len(search) > len(reference) > 100, 'Wrong PCM shape')
    require(np.isfinite(reference).all() and np.isfinite(search).all(), 'Nonfinite signal')
    filt = butter(4, [100, 3400], btype='bandpass', fs=INPUT_RATE, output='sos')
    first = resample_poly(sosfiltfilt(filt, reference), 6, 1, window=('kaiser', 5.), padtype='constant')
    second = resample_poly(sosfiltfilt(filt, search), 6, 1, window=('kaiser', 5.), padtype='constant')
    first -= first.mean()
    count = len(first)
    ones = np.ones(count, dtype=np.float64)
    sums = fftconvolve(second, ones, mode='valid')
    squares = fftconvolve(second * second, ones, mode='valid')
    energy = np.maximum(squares - sums * sums / count, 0.)
    reference_energy = float(np.dot(first, first))
    denominator = np.sqrt(reference_energy * energy)
    numerator = fftconvolve(second, first[::-1], mode='valid')
    curve = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    require(np.isfinite(curve).all(), 'Nonfinite independent score')
    absolute = np.abs(curve)
    best = int(np.argmax(absolute))
    alternate = absolute.copy()
    exclusion = GRID_RATE // 4
    alternate[max(0, best-exclusion):min(len(alternate), best+exclusion+1)] = -1.
    runner_up = int(np.argmax(alternate))
    def direct(position):
        centered = second[position:position+count] - second[position:position+count].mean()
        denom = float(np.linalg.norm(first) * np.linalg.norm(centered))
        return 0. if denom <= 1e-12 else float(np.dot(first, centered) / denom)
    peak, other = direct(best), direct(runner_up)
    close(peak, float(curve[best]), 'Independent FFT/direct peak differs')
    close(other, float(curve[runner_up]), 'Independent FFT/direct runner-up differs')
    return {'evaluation_grid_rate': GRID_RATE, 'global_candidate_count': len(curve),
            'best_start_sample_in_search': best, 'best_start_seconds_in_search': best / GRID_RATE,
            'signed_correlation': peak, 'absolute_correlation': abs(peak),
            'runner_up_exclusion_seconds': .25, 'runner_up_start_sample_in_search': runner_up,
            'runner_up_start_seconds_in_search': runner_up / GRID_RATE,
            'runner_up_absolute_correlation': abs(other), 'peak_margin': abs(peak) - abs(other),
            'peak_ratio': abs(peak) / max(abs(other), 1e-12),
            'reference_rms_filtered_interpolated': math.sqrt(reference_energy / count),
            'matched_search_rms_filtered_interpolated_centered': float(np.std(second[best:best+count])),
            'search_boundary_clearance_seconds': min(best, len(second)-count-best) / GRID_RATE,
            'direct_signed_correlations': {'best': peak, 'far_runner_up': other}}


def gates(score):
    return {'correlation': score['absolute_correlation'] >= .99,
            'unique_far_peak': score['peak_ratio'] >= 1.5,
            'reference_energy': score['reference_rms_filtered_interpolated'] >= 1e-4,
            'matched_energy': score['matched_search_rms_filtered_interpolated_centered'] >= 1e-4,
            'boundary_clearance': score['search_boundary_clearance_seconds'] >= .5}


def validate_score(saved, actual):
    for key, value in actual.items():
        if isinstance(value, dict):
            for name, scalar in value.items():
                close(saved[key][name], scalar, 'Independent direct-score mismatch: ' + name)
        elif type(value) is int:
            same(saved[key], value, 'Independent sample-grid position differs: ' + key)
        else:
            close(saved[key], value, 'Independent signal score differs: ' + key)
    expected = gates(actual)
    same(saved['strict_gates'], expected, 'Signal gates differ')
    same(saved['strict_anchor_pass'], all(expected.values()), 'Signal gate outcome differs')


def validate_pcm(spec, source, role, start, seconds, rate):
    same(spec['source_role'], role, 'Wrong PCM source role')
    same(spec['source_sha256'], source['members'][role]['sha256'], 'Wrong PCM source payload SHA')
    same(spec['clock'], 'original_label_wav' if role == 'audio' else 'mp4', 'Wrong audio clock')
    close(spec['start_seconds'], start, 'Wrong PCM start', 1e-10)
    close(spec['end_seconds'], start + seconds, 'Wrong PCM end', 1e-10)
    close(spec['requested_seconds'], seconds, 'Wrong requested PCM interval', 1e-10)
    same(spec['sample_rate'], rate, 'Wrong PCM sample rate')
    same(spec['channels'], 1, 'PCM must be mono')
    same(spec['sample_width_bytes'], 2, 'PCM must use signed16 samples')
    same(spec['compression'], 'NONE', 'Compressed WAV forbidden')
    same(spec['frames'], round(seconds * rate), 'Incorrect sample count metadata')
    same(spec['duration_seconds'], spec['frames'] / rate, 'Incorrect physical PCM duration')
    same(local(spec['path']).stat().st_size, spec['bytes'], 'Incorrect PCM byte size')
    command = spec['command']
    require('-ss' in command and '-t' in command and '-i' in command and '-n' in command, 'PCM extraction command incomplete')
    close(float(command[command.index('-ss') + 1]), start, 'Recorded FFmpeg seek differs', 1e-9)
    close(float(command[command.index('-t') + 1]), seconds, 'Recorded FFmpeg duration differs', 1e-9)
    original_path = local(command[command.index('-i') + 1])
    require(original_path.is_relative_to(BASE / 'temporary'), 'PCM was not read from bounded temporary member')
    same(original_path.name, source['stem'] + ('.wav' if role == 'audio' else '.mp4'), 'PCM temporary source name differs')
    same(local(command[-1]), local(spec['path']), 'PCM command output path differs')
    return pcm(spec, rate, seconds, start)


def validate_pair(pair, source, reference_start, search_start, search_seconds):
    reference = validate_pcm(pair['reference'], source, 'audio', reference_start, 6., INPUT_RATE)
    search = validate_pcm(pair['mp4_search'], source, 'video', search_start, search_seconds, INPUT_RATE)
    actual = independent_score(reference, search)
    validate_score(pair['score'], actual)
    offset = search_start + actual['best_start_seconds_in_search'] - reference_start
    close(pair['score']['offset_mp4_minus_original_seconds'], offset, 'Wrong WAV-to-MP4 offset sign/origin', 1e-10)
    return actual, offset


def global_mapping(anchors, source):
    require(len(anchors) <= 3, 'Extra global alignment anchors')
    offsets, checked = [], []
    for anchor, reference_start in zip(anchors, (30., 150., 270.)):
        score, offset = validate_pair(anchor, source, reference_start, reference_start - 15., 36.)
        offsets.append(offset)
        checked.append(score)
    if len(offsets) < 3:
        return None, checked
    passed = all(all(gates(score).values()) for score in checked)
    median = round(statistics.median(offsets) * GRID_RATE) / GRID_RATE
    spread = max(offsets) - min(offsets)
    return {'median_offset_seconds': median, 'offset_range_seconds': spread,
            'all_anchors_strict': passed, 'constant_mapping_supported': passed and spread < .02}, checked


def validate_timing(value, record, offset, source):
    start, end = record['start_time'] + offset, record['end_time'] + offset
    count = round((record['end_time'] - record['start_time']) * 25)
    same(count, record['expected_frames'], 'Changed selected sentence duration')
    require(0 <= start < end <= source['video_duration_seconds'] + 1e-9, 'Corrected sentence outside physical video')
    times = [start + k / 25 for k in range(count)]
    indices = [round(t * source['video_fps']) for t in times]
    require(indices[0] >= 0 and indices[-1] < source['video_frame_count'], 'Corrected frame indices outside physical video')
    require(all(a < b for a, b in zip(indices, indices[1:])), 'Frame order not strictly increasing')
    require(count >= record['ctc_min_frames'] > 0, 'CTC path impossible')
    for key, expected in (('corrected_start_time', start), ('corrected_end_time', end), ('student_frames', count),
            ('ctc_min_frames', record['ctc_min_frames']), ('ctc_feasible', True), ('actual_video_fps', source['video_fps']),
            ('native_frame_indices', indices), ('target_times_seconds', times)):
        same(value[key], expected, 'Independent temporal trace differs: ' + key)
    return indices, times


def pixels(spec, shape):
    path = verify(spec)
    require(path.is_relative_to(BASE / 'results/media') and path.suffix == '.png', 'Wrong review PNG location')
    with Image.open(path) as image:
        same(image.mode, 'RGB', 'Unexpected review PNG mode')
        rgb = np.array(image)
    same(list(rgb.shape), shape, 'Review PNG dimensions differ')
    same(str(rgb.dtype), 'uint8', 'Wrong image type')
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    same(spec['pixels'], {'sha256': hashlib.sha256(bgr.tobytes()).hexdigest(), 'shape': shape,
                         'dtype': 'uint8', 'color_order': 'BGR'}, 'Saved PNG pixel hash differs')
    return bgr


def shape_digest(value, shape):
    same(value['shape'], shape, 'Recorded array shape differs')
    same(value['dtype'], 'uint8', 'Recorded array dtype differs')
    same(value['color_order'], 'BGR', 'Recorded color order differs')
    require(isinstance(value['sha256'], str) and len(value['sha256']) == 64
            and set(value['sha256']) <= set('0123456789abcdef'), 'Invalid raw-frame array digest')


def validate_previews(clip, record, source, indices, times, label):
    width, height = source['video_metadata']['width'], source['video_metadata']['height']
    lips = label['Bounding_box_info']['Lip_bounding_box']['xtl_ytl_xbr_ybr']
    faces = label['Bounding_box_info']['Face_bounding_box']['xtl_ytl_xbr_ybr']
    same(len(lips), source['video_frame_count'], 'Lip annotation/frame cardinality differs')
    same(len(faces), source['video_frame_count'], 'Face annotation/frame cardinality differs')
    require(len(clip['roi_samples']) <= 3, 'Unplanned ROI samples')
    expected = list(zip(('first', 'middle', 'last'), (0, len(indices) // 2, len(indices) - 1)))
    for sample, (name, student_position) in zip(clip['roi_samples'], expected):
        index = indices[student_position]
        same(sample['position'], name, 'ROI order changed')
        same(sample['student_index'], student_position, 'ROI student position differs')
        same(sample['native_frame_index'], index, 'ROI native source index differs')
        same(sample['annotation_box_index'], index, 'ROI annotation index differs')
        same(sample['target_time_seconds'], times[student_position], 'ROI target time differs')
        close(sample['decoded_pts_seconds'], index / source['video_fps'], 'ROI PTS/CFR mismatch', .001)
        require(abs(sample['decoded_pts_seconds'] - times[student_position]) <= .5 / source['video_fps'] + 1e-6,
                'ROI PTS differs from target time')
        box = list(map(float, lips[index]))
        same(sample['bbox_yxyx'], box, 'ROI lip box differs from immutable label')
        same(sample['bbox_xyxy'], [box[1], box[0], box[3], box[2]], 'Wrong yxyx conversion')
        y1, x1, y2, x2 = np.asarray(box, dtype=np.float32).tolist()
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'Invalid lip box geometry')
        bw, bh = x2-x1, y2-y1
        expected_crop = [max(0, min(math.floor(x1-.2*bw), width-1)), max(0, min(math.floor(y1-.2*bh), height-1)),
                         min(width, math.ceil(x2+.2*bw)), min(height, math.ceil(y2+.2*bh))]
        same(sample['crop_xyxy'], expected_crop, 'Preview crop rectangle differs')
        pixels(sample['image'], [96, 96, 3])
        same(sample['roi_array'], sample['image']['pixels'], 'ROI image and array digest differ')
        shape_digest(sample['raw_frame_array'], [height, width, 3])
    context = clip['face_context']
    if context is not None:
        middle = len(indices) // 2
        index = indices[middle]
        same(context['native_frame_index'], index, 'Context frame index differs')
        close(context['decoded_pts_seconds'], index / source['video_fps'], 'Context PTS/CFR mismatch', .001)
        box = list(map(float, faces[index]))
        same(context['face_bbox_yxyx'], box, 'Face box differs from immutable label')
        same(context['face_bbox_xyxy'], [box[1], box[0], box[3], box[2]], 'Wrong face coordinate conversion')
        shape_digest(context['raw_frame_array'], [height, width, 3])
        samples = [s for s in clip['roi_samples'] if s['position'] == 'middle']
        same(len(samples), 1, 'Context without middle ROI')
        same(context['raw_frame_array'], samples[0]['raw_frame_array'], 'Context/ROI refer to different raw frames')
        pixels(context['image'], [270, 480, 3])


def track_duration(metadata, handler):
    tracks = [track for track in metadata['tracks'] if track['handler'] == handler]
    same(len(tracks), 1, 'Ambiguous MP4 media clock')
    track = tracks[0]
    bounds = [metadata['movie_clock']['duration_seconds'], track['media_clock']['duration_seconds']]
    if track.get('edit_list'):
        require(all(e['media_rate_integer'] == 1 and e['media_rate_fraction'] == 0 for e in track['edit_list']),
                'MP4 edit list changes playback rate')
        bounds.append(sum(e['movie_duration_seconds'] for e in track['edit_list']))
    require(all(math.isfinite(n) and n > 0 for n in bounds), 'Invalid MP4 clock duration')
    return min(bounds)


def validate_source_metadata(validation, source, label, audio_check):
    if not validation:
        return
    for key, expected in (('original_members_payload_sha256_verified', True), ('original_and_cached_label_exact', True),
            ('selected_sentences_verified', 4), ('all_lip_and_face_boxes_valid', True),
            ('bbox_count', source['video_frame_count']), ('actual_video_fps', source['video_fps']),
            ('video_frame_count', source['video_frame_count']), ('width', source['video_metadata']['width']),
            ('height', source['video_metadata']['height']), ('old59_offsets_used', False)):
        same(validation[key], expected, 'Source verification metadata differs: ' + key)
    for key in ('Lip_bounding_box', 'Face_bounding_box'):
        boxes = np.asarray(label['Bounding_box_info'][key]['xtl_ytl_xbr_ybr'], dtype=np.float64)
        same(boxes.shape, (source['video_frame_count'], 4), 'Annotation frame count differs')
        require(np.isfinite(boxes).all(), 'Nonfinite annotation')
        y1, x1, y2, x2 = boxes.T
        require(np.all((y1 >= 0) & (x1 >= 0) & (y1 < y2) & (x1 < x2)
                       & (y2 <= validation['height']) & (x2 <= validation['width'])), 'Invalid yxyx annotation geometry')
    wav = validation['original_wav_header']
    same(wav, {'sample_rate': 48000, 'channels': 2, 'sample_width_bytes': 2,
               'frames': audio_check['expected_sample_frames'], 'compression': 'NONE',
               'duration_seconds': audio_check['expected_sample_frames'] / 48000}, 'Original WAV header differs from audited PCM samples')
    metadata = validation['mp4_metadata']
    same(validation['mp4_audio_duration_bound_seconds'], track_duration(metadata, 'soun'), 'MP4 audio bounds differ')
    same(validation['video_duration_bound_seconds'], min(track_duration(metadata, 'vide'), source['video_duration_seconds']),
         'MP4 video bounds differ')


def validate_video_preview(spec, row, offset, source):
    if spec is None:
        return
    path = verify(spec)
    require(path.is_relative_to(BASE / 'results/media') and path.suffix == '.mp4' and path.stat().st_size > 0,
            'Invalid review video artifact')
    same(spec['source_role'], 'video', 'Wrong preview source role')
    same(spec['source_sha256'], source['members']['video']['sha256'], 'Wrong preview source identity')
    same(spec['clock'], 'mp4', 'Wrong preview clock')
    same(spec['review_only'], True, 'Preview incorrectly promoted to input cache')
    close(spec['start_seconds'], row['start_time'] + offset, 'Wrong preview start', 1e-10)
    close(spec['end_seconds'], row['end_time'] + offset, 'Wrong preview end', 1e-10)
    close(spec['requested_seconds'], row['end_time'] - row['start_time'], 'Wrong preview duration', 1e-10)
    same(local(spec['command'][-1]), path, 'Preview command output differs')


def validate_clip(clip, row, source, item, mapping, label):
    for key, expected in (('id', row['id']), ('split', row['split']), ('speaker_id', row['speaker_id']),
            ('source_stem', source['stem']), ('reference', row['source_text']), ('text', row['text']),
            ('sentence_id', row['sentence_id']), ('start_time', row['start_time']), ('end_time', row['end_time']),
            ('label_sha256', row['label_sha256']), ('training_ready', False),
            ('human_transcript_review', 'not_reviewed'), ('human_visual_review', 'not_reviewed'),
            ('source_contact_sheet', item['contact_sheet'])):
        same(clip[key], expected, 'Selected clip identity or approval changed: ' + key)
    require(clip['status'] in ('prepared', 'quarantined'), 'Unknown clip outcome')
    require(clip['timing_status'] in ('verified', 'not_verified', 'invalidated_source_integrity'), 'Unknown timing outcome')
    supported = mapping is not None and mapping['constant_mapping_supported']
    local_result = clip['local_check']
    local_verified = False
    if local_result is not None:
        require(supported, 'Local mapping exists without supported global mapping')
        offset = mapping['median_offset_seconds']
        midpoint_start = (row['start_time'] + row['end_time']) / 2 - 3
        actual, local_offset = validate_pair(local_result, source, midpoint_start, midpoint_start + offset - 1, 8.)
        same(local_result['source_offset_seconds'], offset, 'Local check uses another global mapping')
        residual = local_offset - offset
        close(local_result['offset_residual_seconds'], residual, 'Local residual differs', 1e-10)
        same(local_result['maximum_absolute_residual_seconds'], .001, 'Local residual threshold relaxed')
        same(local_result['residual_gate_pass'], abs(residual) <= .001, 'Local residual gate differs')
        local_verified = all(gates(actual).values()) and abs(residual) <= .001
        if clip['timing_status'] == 'invalidated_source_integrity':
            same(local_result['waveform_gate_pass_before_source_invalidation'], local_verified, 'Pre-invalidation numerical decision changed')
            same(local_result['invalidated_by_source_error'], True, 'Missing source invalidation evidence')
            same(local_result['verified'], False, 'Invalidated source retains local approval')
        else:
            same(local_result['verified'], local_verified, 'Local combined gate differs')
    if clip.get('timing') is not None:
        require(supported and local_verified, 'Corrected frame trace without valid global/local numerical mapping')
        indices, times = validate_timing(clip['timing'], row, mapping['median_offset_seconds'], source)
        validate_previews(clip, row, source, indices, times, label)
    else:
        same(clip['roi_samples'], [], 'ROI images without corrected frame trace')
        same(clip['face_context'], None, 'Face context without corrected frame trace')
    if clip['timing_status'] == 'verified':
        require(supported and local_verified and clip.get('timing') is not None, 'Timing approved without complete evidence')
    for name, role, start, seconds in (
            ('audio', 'audio', row['start_time'], row['end_time'] - row['start_time']),
            ('context_audio', 'audio', row['start_time']-.5, row['end_time']-row['start_time']+1),
            ('mp4_audio', 'video', row['start_time'] + (mapping['median_offset_seconds'] if supported else 0.), row['end_time']-row['start_time'])):
        if clip[name] is not None:
            require(supported and local_verified, 'Review audio without valid mapping evidence')
            validate_pcm(clip[name], source, role, start, seconds, 16000)
    if clip['preview_video'] is not None:
        require(supported, 'Video preview without valid mapping')
        validate_video_preview(clip['preview_video'], row, mapping['median_offset_seconds'], source)
    if item['status'] == 'quarantined':
        same(clip['status'], 'quarantined', 'Source integrity failure did not quarantine all clips')
        same(clip['artifacts_ready'], False, 'Source integrity failure retains artifact approval')
        require(clip['timing_status'] != 'verified', 'Source integrity failure retains timing approval')
    if clip['status'] == 'prepared':
        same(clip['artifacts_ready'], True, 'Prepared row missing artifact approval')
        same(clip['timing_status'], 'verified', 'Prepared row lacks timing verification')
        require(all(clip[name] is not None for name in ('audio', 'context_audio', 'mp4_audio', 'face_context'))
                and len(clip['roi_samples']) == 3, 'Prepared row missing required artifacts')
    else:
        same(clip['artifacts_ready'], False, 'Quarantined row retains artifact approval')
        require(isinstance(clip.get('quarantine_reason'), str) and clip['quarantine_reason'], 'Quarantine reason missing')
    return {'id': row['id'], 'split': row['split'], 'timing_status': clip['timing_status'], 'status': clip['status'],
            'local_pair_checked': local_result is not None, 'artifacts_ready': clip['artifacts_ready']}


def artifact_inventory(item):
    folder = local(item['artifact_directory'])
    require(folder.is_relative_to(BASE / 'results/media' / item['source_stem']), 'Artifact directory outside source scope')
    expected = [bind(path) for path in sorted(folder.rglob('*')) if path.is_file()]
    same(item['artifacts_all'], expected, 'Artifact added, removed, or modified')
    indexed = {ref['path']: ref['sha256'] for ref in expected}
    same(len(indexed), len(expected), 'Duplicate artifact inventory')
    def walk(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                path = local(value['path'])
                if path.is_relative_to(BASE / 'results/media'):
                    require(path.is_relative_to(folder), 'Artifact reference crosses source folders')
                    same(indexed.get(str(path)), value['sha256'], 'Referenced artifact absent from inventory')
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(item)
    return {'files': len(expected), 'bytes': sum(local(ref['path']).stat().st_size for ref in expected)}


def validate_media_plan(source_plan):
    plan = read(MEDIA_PLAN, MEDIA_SHA)
    same(plan['schema'], 'lipnet-new120-media-plan-v1', 'Wrong prospective media plan')
    same(plan['frozen_before_first_new_media_extraction'], True, 'Prospective freeze missing')
    same(plan['source_plan'], bind(SOURCE_PLAN), 'Media plan references different new120 sources')
    same(plan['source_order'], [s['stem'] for s in source_plan['sources']], 'Media source order differs')
    same(plan['ids'], [r['id'] for r in source_plan['records']], 'Media sentence order differs')
    for ref in plan['code'] + [plan['ffmpeg'], plan['environment']['executable']]:
        verify(ref)
    expected = {'global_anchors_seconds': [30., 150., 270.], 'global_reference_seconds': 6.,
        'global_search_radius_seconds': 15., 'global_search_seconds': 36.,
        'local_reference_seconds': 6., 'local_search_radius_seconds': 1., 'local_search_seconds': 8.,
        'pcm_rate': 8000, 'interpolation_factor': 6, 'evaluation_grid_rate': 48000,
        'minimum_correlation': .99, 'minimum_far_peak_ratio': 1.5, 'minimum_filtered_rms': 1e-4,
        'minimum_search_boundary_clearance_seconds': .5, 'maximum_global_offset_range_exclusive_seconds': .02,
        'maximum_local_residual_inclusive_seconds': .001, 'audio_output_rate': 16000, 'audio_channels': 1,
        'audio_sample_width_bytes': 2, 'context_radius_seconds': .5, 'student_fps': 25,
        'roi_margin_each_side': .2, 'roi_size': [96, 96], 'roi_samples': ['first', 'middle', 'last'],
        'preview_height': 360, 'pts_cfr_tolerance_seconds': .001}
    for key, value in expected.items():
        same(plan['rules'][key], value, 'Prospective media rule differs: ' + key)
    for key in ('G_access', 'original_writes', 'label_edits', 'old59_offsets_used', 'model_execution', 'ASR', 'training', 'training_ready'):
        same(plan['scope'][key], False, 'Unexpected preparation scope: ' + key)
    return plan


def audit(output):
    began = time.perf_counter()
    source_plan = audit_source_plan()
    plan = validate_media_plan(source_plan)
    plan_ref = bind(MEDIA_PLAN)
    manifest_path = BASE / 'results/media_manifest.json'
    manifest_ref, manifest = bind(manifest_path), read(manifest_path)
    for key, expected in (('schema', 'lipnet-new120-media-manifest-v1'), ('status', 'completed_all_planned_sources'),
            ('source_plan', bind(SOURCE_PLAN)), ('media_plan', plan_ref), ('training_ready', False),
            ('human_transcript_review', 'not_reviewed'), ('model_execution', False), ('temporary_originals_cleaned', True)):
        same(manifest[key], expected, 'Incomplete or changed media manifest: ' + key)
    same(manifest['limits'], plan['limits'], 'Preparation limitations omitted')
    expected_sources = [bind(BASE / 'results/sources' / (s['stem'] + '.json')) for s in source_plan['sources']]
    same(manifest['sources'], expected_sources, 'Source outcome coverage or order differs')
    progress = read(BASE / 'results/media_progress.json')
    same(progress, {'media_plan': plan_ref, 'sources': expected_sources}, 'Completed progress differs from source outcomes')
    metadata_checks = unique(read(AUDIT / 'archive_results/media_crosschecks.json')['records'], 'name')
    rows = {r['id']: r for r in source_plan['records']}
    results, by_id, checked_rows, artifact_files, artifact_bytes, global_pairs, local_pairs = [], {}, [], 0, 0, 0, 0
    for source, ref in zip(source_plan['sources'], expected_sources, strict=True):
        item = read(verify(ref))
        for key, expected in (('schema', 'lipnet-new120-media-source-v1'), ('source_stem', source['stem']),
                ('speaker_id', source['speaker_id']), ('split', source['split']), ('source_plan', bind(SOURCE_PLAN)),
                ('media_plan', plan_ref), ('training_ready', False), ('human_transcript_review', 'not_reviewed'),
                ('temporary_originals_cleaned', True)):
            same(item[key], expected, 'Source outcome binding differs: ' + key)
        require(item['status'] in ('prepared', 'partial_quarantine', 'quarantined'), 'Unknown source status')
        same([c['id'] for c in item['clips']], source['record_ids'], 'Source outcome changed selected rows')
        inventory = artifact_inventory(item)
        artifact_files += inventory['files']
        artifact_bytes += inventory['bytes']
        label = read(source['label_copy']['path'], source['label_copy']['sha256'])[0]
        validate_source_metadata(item['validation'], source, label, metadata_checks[source['stem'] + '.wav'])
        mapping, checked = global_mapping(item['anchors'], source)
        if mapping is None:
            same(item['offset_summary'], None, 'Incomplete global anchors have a summary')
            same(item['status'], 'quarantined', 'Incomplete mapping not quarantined')
        else:
            for key, value in mapping.items():
                if isinstance(value, bool):
                    same(item['offset_summary'][key], value, 'Global offset decision differs')
                else:
                    close(item['offset_summary'][key], value, 'Global offset summary differs', 1e-10)
            if not mapping['constant_mapping_supported']:
                same(item['status'], 'quarantined', 'Failed global mapping not quarantined')
        pixels(item['contact_sheet'], [840, 720, 3])
        source_rows = []
        for clip in item['clips']:
            require(clip['id'] not in by_id, 'Duplicate final selected record')
            by_id[clip['id']] = clip
            checked_row = validate_clip(clip, rows[clip['id']], source, item, mapping, label)
            source_rows.append(checked_row)
            local_pairs += checked_row['local_pair_checked']
        global_pairs += len(checked)
        checked_rows.extend(source_rows)
        if item['status'] != 'quarantined':
            same(item['status'], 'prepared' if all(c['status'] == 'prepared' for c in item['clips']) else 'partial_quarantine',
                 'Source status inconsistent with row outcomes')
        else:
            require(isinstance(item.get('quarantine_reason'), str) and item['quarantine_reason'], 'Source quarantine lacks reason')
        results.append({'source_stem': source['stem'], 'artifact': ref, 'status': item['status'],
                        'offset_summary': mapping, 'global_pairs_checked': len(checked), 'clips': source_rows,
                        'artifact_files': inventory['files'], 'artifact_bytes': inventory['bytes']})
        print(f"[independent Step6 audit] {source['stem']}: {len(checked)} global pairs, {len(source_rows)} selected rows checked", flush=True)
    same(set(by_id), set(rows), 'Selected manifest lost IDs')
    expected_clips = [by_id[r['id']] for r in source_plan['records']]
    same(manifest['clips'], expected_clips, 'Manifest altered or reordered source outcomes')
    expected_counts = {'sources': 30, 'clips': 120, 'train': 96, 'val': 24,
        'artifacts_ready': sum(c['artifacts_ready'] for c in expected_clips),
        'timing_verified': sum(c['timing_status'] == 'verified' for c in expected_clips),
        'status': dict(Counter(c['status'] for c in expected_clips))}
    same(manifest['counts'], expected_counts, 'Incorrect final preparation counts')
    require(not any((BASE / 'temporary').rglob('*')), 'Temporary original source files/directories remain')
    for ref in [manifest_ref, plan_ref, *expected_sources]:
        verify(ref)
    result = {'schema': 'lipnet-new120-independent-media-audit-v1', 'status': 'passed',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'code': bind(Path(__file__)),
        'tests': bind(BASE / 'test_verify_step06.py'), 'source_plan': bind(SOURCE_PLAN),
        'media_plan': plan_ref, 'media_manifest': manifest_ref, 'counts': expected_counts,
        'source_results': results, 'global_waveform_pairs_checked': global_pairs,
        'local_waveform_pairs_checked': local_pairs, 'artifact_files_checked': artifact_files,
        'artifact_bytes_checked': artifact_bytes, 'training_ready': False,
        'source_payloads_read': False, 'source_archive_stats_checked': True,
        'producer_or_model_imported': False, 'asr_executed': False, 'model_training_performed': False,
        'human_listening_certified': False, 'human_visual_review_certified': False,
        'elapsed_seconds': time.perf_counter()-began,
        'limits': ['Independent saved PCM correlation confirms recorded WAV-to-MP4 clock evidence; it does not prove lip synchrony or transcript correctness.',
                   'Saved PNG pixels and their immutable-label frame/box traces are checked. Original source frames are not reread, so raw-frame pixel identity cannot be independently reconstructed.',
                   'Encoded video previews are byte-hashed and their declared source/window checked, not independently decoded by this audit.',
                   'The full120 cohort, quarantine outcomes and train96/val24 roles are preserved; this audit never grants training or human listening approval.']}
    output = local(output)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'results/independent_media_audit.json')
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'output': str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    main()


