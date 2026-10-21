"""Bind the frozen new120 readiness selection to audited D: TAR members.

Reads only workspace JSON/code/label copies and stats six archives. No archive
payload, G: source, media decoder, model or training path is opened/executed.
The original120 records, including historical G: metadata, remain unchanged.
No timing offset from the old59 recordings is applied to the new30 recordings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SELECTION = ROOT / 'results/step18-intake/run01/selection.json'
SELECTION_SHA = 'abadd58b0fdf15197c1ced38b01033b464e87f439b8ae2bd43405098be07cc9c'
AUDIT = ROOT / 'experiments/data_integrity_20261002'
OVERALL_SHA = 'ca411651e6f50e66a7eb44ebfaad48c7c3be7dc5bd10ea714044eaed3ba9e625'
OLD_BUILD = ROOT / 'experiments/local_source_step01_20261007/build_plan.py'
OLD_BUILD_SHA = '601948e4ebf9479da3c684dc31b6e6a2255108db7c9c127384953495763c1bd5'
TAR_SOURCE = ROOT / 'experiments/local_source_step01_20261007/tar_source.py'
TAR_SHA = '05f72840576046683d6b5a3c1121a94b7fe2099fd0d85596c30da511801effb2'
TOKENIZER = ROOT / 'backend/models/tokenizer.py'
TOKENIZER_SHA = 'c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714'
SCHEMA = 'lipnet-new120-audited-tar-source-plan-v1'
PACKAGES = ('TS2', 'TS20', 'TS200')


class PlanError(ValueError):
    pass


def require(value, message):
    if not value:
        raise PlanError(message)


def sha256(path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def local(path) -> Path:
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), f'Evidence must be a workspace file: {path}')
    return path


def read_json(path):
    return json.loads(local(path).read_text(encoding='utf-8-sig'))


def load_module(name, path, expected):
    path = local(path)
    require(sha256(path) == expected, 'Frozen helper changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load_module('new120_frozen_source_binding_primitives', OLD_BUILD, OLD_BUILD_SHA)
transport = load_module('new120_frozen_bounded_tar_reader', TAR_SOURCE, TAR_SHA)
tokenizer = load_module('new120_frozen_text_contract', TOKENIZER, TOKENIZER_SHA)


def unique(rows, key):
    result = {}
    for row in rows:
        require(row[key] not in result, f'Ambiguous duplicate {key}: {row[key]}')
        result[row[key]] = row
    return result


def validate_selection(selection: dict, protected: dict, role_map: dict) -> dict:
    """Preserve readiness roles, speaker holdout and every prior exclusion."""
    rows = selection['records']
    require(selection['schema'] == 'lipnet-new-source-readiness-v1', 'Unexpected selection schema')
    require(len(rows) == len({r['id'] for r in rows}) == 120, 'Require all120 unique selection rows')
    require(Counter(r['split'] for r in rows) == {'train': 96, 'val': 24}, 'Frozen readiness split changed')
    require(len({r['speaker_id'] for r in rows}) == 15 and len({r['recording_id'] for r in rows}) == 30, 'Frozen15 speakers/30 recordings changed')
    require(set(selection['speaker_mapping']) == {r['speaker_id'] for r in rows}, 'Speaker role mapping incomplete')
    groups = defaultdict(list)
    for row in rows:
        require(row['split'] == selection['speaker_mapping'][row['speaker_id']], 'Speaker leakage between readiness roles')
        require(row['source_partition'] == 'Training' and row['angle'] == 'A' and row['package'] in PACKAGES, 'Unexpected partition/angle/package')
        stem = PureWindowsPath(row['source_video']['path']).stem
        require(row['id'] == f"{stem}_{row['sentence_id']:03d}" and row['utterance_id'] == f"{row['recording_id']}:{row['sentence_id']:03d}", 'Source/sentence identity changed')
        require(PureWindowsPath(row['label_path']).name == stem + '.json' and PureWindowsPath(row['source_label_path']).name == stem + '.json', 'Label/source association differs')
        require(row['source_video']['speaker_id'] == row['speaker_id'] and row['source_video']['recording_id'] == row['recording_id']
                and row['source_video']['angle'] == 'A' and row['source_video']['package'] == row['package'], 'Historical source identity differs')
        start, end = row['start_time'], row['end_time']
        require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end, 'Invalid unchanged nominal interval')
        require(round((end - start) * 25) == row['expected_frames'] and row['expected_frames'] <= 200, 'Nominal25fps length differs')
        tokens = tokenizer.encode_text(row['source_text'])
        require(tokenizer.normalize_text(row['source_text']) == row['text'], 'Frozen transcript normalization differs')
        minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        require(minimum == row['ctc_min_frames'] <= row['expected_frames'], 'CTC minimum or feasibility differs')
        groups[stem].append(row)
    require(len(groups) == 30 and all(len(group) == 4 for group in groups.values()), 'Require four selected utterances per recording')
    require(Counter(r['speaker_id'] for r in rows) == {s: 8 for s in selection['speaker_mapping']}, 'Require eight selected utterances per speaker')
    for package in PACKAGES:
        require(Counter(r['split'] for r in rows if r['package'] == package) == {'train': 32, 'val': 8}, 'Package96/24 quota changed')
    ids, speakers, recordings, texts = ({r[field] for r in rows} for field in ['id', 'speaker_id', 'recording_id', 'text'])
    require(len(texts) == 120, 'Selected normalized transcripts are not unique')
    guard = protected['guards']
    for actual, name in [(ids, 'blocked_existing_sample_ids'), (speakers, 'blocked_existing_speaker_ids'),
                         (recordings, 'blocked_existing_recording_ids'), (texts, 'blocked_existing_normalized_texts')]:
        require(not actual.intersection(guard[name]), 'Protected old cohort overlap: ' + name)
    for name, values in guard.items():
        digest = hashlib.sha256(json.dumps(sorted(values), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        require(digest == protected['guard_sha256'][name], 'Protected guard digest differs')
    role_index = unique(role_map['records'], 'source_stem')
    for stem, group in groups.items():
        role = role_index[stem]
        split, speaker = group[0]['split'], group[0]['speaker_id']
        require({r['split'] for r in group} == {split} and {r['speaker_id'] for r in group} == {speaker}, 'Recording crosses speaker/split')
        expected = 'new_pilot_train' if split == 'train' else 'new_pilot_development'
        require(role['speaker_id'] == speaker and role['recording_id'] == group[0]['recording_id']
                and role['speaker_role'] == expected and role['readiness_role'] == expected + '_selected_recording'
                and role['stage18_pilot_split'] == split, 'Audited readiness role mismatch')
        require(set(role['stage18_pilot_selected_clip_ids']) == {r['id'] for r in group}, 'Audited selected utterance scope differs')
        require(role['original_v4_split'] is None and not role['protected_reference_cohorts']
                and role['stage7_planned_split'] is None and not role['stage7_planned_selected_clip_ids']
                and role['stage17_recording_selected'] is False, 'Old/test selection entered new readiness scope')
        require(role['all_sentences_in_recording_approved_for_training'] is False
                and role['integrity_audit_changes_split_or_authorizes_training'] is False, 'File integrity cannot authorize all utterances')
    dev_speakers = {r['speaker_id'] for r in rows if r['split'] == 'val'}
    require(dev_speakers == {'C040', 'C059', 'C769'}, 'Frozen new development speakers changed')
    all_dev_recordings = [r for r in role_map['records'] if r['speaker_id'] in dev_speakers]
    require(len(all_dev_recordings) == 36 and all(r['speaker_role'] == 'new_pilot_development' for r in all_dev_recordings), 'Development speaker-wide guard differs')
    original = role_map['original_v4_contract']
    require(original['val'] == {'speaker_ids': ['C025', 'C294'], 'clips': 20}
            and original['test'] == {'speaker_ids': ['C138', 'C281', 'C334', 'C428', 'C558'], 'clips': 50}, 'Existing held-out roles changed')
    return {'ids_unique': 120, 'normalized_transcripts_unique': 120, 'existing_speaker_overlap': 0,
            'existing_recording_overlap': 0, 'existing_sample_overlap': 0, 'existing_transcript_overlap': 0,
            'new_train_speakers': sorted(speakers - dev_speakers), 'new_development_speakers': sorted(dev_speakers),
            'new_development_all_recordings_protected': 36, 'original_v4_validation': deepcopy(original['val']),
            'original_v4_final_test': deepcopy(original['test']), 'all_new_sources_readiness_only': True}


def validate_role_member(row, member):
    require(row['archive_path'] == member['archive_path'] and row['archive_member'] == member['member_name'], 'Readiness role and video TAR member differ')


def collect_plan(selection_path: Path = SELECTION, audit_root: Path = AUDIT) -> dict:
    bindings = {}
    def file_ref(path, expected=None):
        path = local(path)
        digest = sha256(path)
        require(expected is None or digest == expected, f'Frozen evidence changed: {path}')
        require(str(path) not in bindings or bindings[str(path)] == digest, 'Evidence changed during collection')
        bindings[str(path)] = digest
        return {'path': str(path), 'sha256': digest}
    def load(path, expected=None):
        ref = file_ref(path, expected)
        raw = Path(ref['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Evidence changed while reading')
        return json.loads(raw.decode('utf-8-sig'))

    selection_path, audit_root = local(selection_path), local(audit_root)
    selection = load(selection_path, SELECTION_SHA)
    for name, digest in selection['frozen_files'].items():
        file_ref(ROOT / name, digest)
    overall = load(audit_root / 'overall_summary.json', OVERALL_SHA)
    final = load(audit_root / 'archive_results/final_validation.json')
    require(final['overall_summary_sha256'] == OVERALL_SHA and final['status'] == 'evidence_consistency_pass'
            and not final['failures'] and final['checks_count'] == final['passed_count'], 'Final integrity evidence did not pass')
    require(overall['evidence_consistency_pass'] is True and not overall['source_changes_since_inventory']
            and not overall['protected_input_changes'] and not overall['selected_label_hash_mismatches'], 'Inconsistent October2 integrity evidence')
    evidence = {}
    for name, digest in overall['evidence_sha256'].items():
        path = (audit_root / name).resolve()
        require(path.is_relative_to(audit_root), 'Audit reference escapes evidence directory')
        evidence[name] = load(path, digest)
    role_map = evidence['archive_results/readiness_role_map.json']
    for ref in role_map['evidence'].values():
        file_ref(ref['path'], ref['sha256'])
    require(role_map['evidence']['stage18_selection'] == {'path': str(selection_path), 'sha256': SELECTION_SHA}, 'Role map anchors another selection')
    protected_ref = role_map['evidence']['protected_reference']
    protected = load(protected_ref['path'], protected_ref['sha256'])
    for ref in protected['parents'].values():
        file_ref(ref['path'], ref['sha256'])
    role_checks = validate_selection(selection, protected, role_map)
    structures = unique(evidence['archive_results/summary.json']['archives'], 'archive_path')
    index = defaultdict(list)
    for entry in evidence['archive_results/member_index.json']['members']:
        if entry['is_file']:
            index[PurePosixPath(entry['member_name']).name].append(entry)
    decoder_refs = unique(evidence['archive_results/media_crosschecks.json']['decoder_result_evidence'], 'path')
    decode_plan = unique(evidence['decode_results/plan.json']['records'], 'name')
    label_ref = file_ref(audit_root / 'label_results/label_records.json')
    labels = unique(load(label_ref['path'], label_ref['sha256']), 'stem')
    fronts = unique(overall['frontal_sources'], 'video')
    roles = unique(role_map['records'], 'source_stem')
    impact = evidence['label_results/selection_impact.json']
    require(impact['groups']['new_stage18_pilot120'] == {'selected_sentences': 120, 'sentences_with_errors': 0, 'frozen_label_hash_mismatches': 0}, 'Selection impact review failed')
    impact_rows = unique([r for r in impact['records'] if r['group'] == 'new_stage18_pilot120'], 'id')
    require(set(impact_rows) == {r['id'] for r in selection['records']}, 'Audited selected120 coverage differs')
    grouped = defaultdict(list)
    for row in selection['records']:
        grouped[PureWindowsPath(row['source_video']['path']).stem].append(row)
    sources, archives, changed_sizes = [], {}, []
    for stem, rows in sorted(grouped.items()):
        first, label, front, role = rows[0], labels[stem], fronts[stem + '.mp4'], roles[stem]
        require(not label['errors'] and front['status'] == 'decode_and_metadata_pass'
                and not front['metadata_errors'] and not front['sentence_issues'] and not front['evidence_issues'], 'Unready selected technical source')
        identity = label['detail']['identity']
        require(identity == {'speaker': first['speaker_id'], 'angle': 'A', 'recording': first['recording_id'], 'frontal_stem': stem}, 'Label identity differs')
        require(all(row['label_sha256'] == label['sha256'] == front['label_sha256'] for row in rows), 'Frozen selection/D label SHA differs')
        # The pre-existing local label copy is complete and hash-equal to the
        # audited D member; source video size in historical G metadata is not.
        local_label = load(first['label_path'], label['sha256'])
        require(isinstance(local_label, list) and len(local_label) == 1, 'Unexpected official label container')
        annotated = local_label[0]
        require(annotated['Video_info']['video_Name'] == stem + '.mp4'
                and annotated['Audio_info']['Audio_Name'] == stem + '.wav', 'Label media association differs')
        sentences = unique(annotated['Sentence_info'], 'ID')
        for row in rows:
            sentence, checked = sentences[row['sentence_id']], impact_rows[row['id']]
            require(sentence['sentence_text'] == row['source_text'] and sentence['start_time'] == row['start_time']
                    and sentence['end_time'] == row['end_time'], 'Frozen nominal text/timing differs from D-identical label')
            require(checked['source_stem'] == stem and checked['start_time'] == row['start_time']
                    and checked['end_time'] == row['end_time'] and checked['label_sha256_matches_frozen_selection'] is True
                    and not checked['errors'] and not checked['record_level_errors'] and not checked['selected_sentence_issues'], 'Selected nominal interval integrity review differs')
        members, decoded, decoded_refs = {}, {}, {}
        for kind, extension in [('video', '.mp4'), ('label', '.json'), ('audio', '.wav')]:
            name = stem + extension
            require(len(index[name]) == 1, f'Missing/ambiguous TAR member: {name}')
            item = index[name][0]
            old.validate_member(item, name)
            archive = structures[item['archive_path']]
            require(PureWindowsPath(item['archive_path']).drive.upper() == 'D:', 'Only D TAR references allowed')
            expected_package = first['package'] if kind != 'label' else first['package'].replace('TS', 'TL', 1)
            require(PureWindowsPath(item['archive_path']).name == expected_package + '.tar'
                    and PurePosixPath(item['member_name']).parts[0] == expected_package, 'Wrong package TAR association')
            require(archive['status'] == 'structure_pass' and archive['before'] == archive['after'], 'TAR structure/stat evidence failed')
            snapshot = archive['after']
            if kind == 'label':
                require(label['archive'] == item['archive_path'] and label['member'] == item['member_name'] and label['size'] == item['size'], 'D label membership differs')
                digest = label['sha256']
            else:
                decoder_path = audit_root / 'decode_results/records' / (name + '.json')
                require(str(decoder_path) in decoder_refs, 'Decoder record is not anchored')
                ref = decoder_refs[str(decoder_path)]
                result = load(decoder_path, ref['sha256'])
                require(all(result[k] == value for k, value in decode_plan[name].items()), 'Decoder plan mismatch')
                require(result['status'] == 'full_decode_pass' and result['returncode'] == 0 and result['progress']['progress'] == 'end', 'Member decode incomplete')
                require(result['archive'] == item['archive_path'] and result['member'] == item['member_name']
                        and result['offset'] == item['offset_data'] and result['size'] == item['size'] and result['archive_stat'] == snapshot, 'Decoded TAR membership differs')
                digest = result['member_sha256']
                decoded[kind], decoded_refs[kind] = result, file_ref(decoder_path, ref['sha256'])
            members[kind] = {'archive_path': item['archive_path'], 'archive_size': snapshot['size'], 'archive_mtime_ns': snapshot['mtime_ns'],
                             'member_name': item['member_name'], 'offset': item['offset_data'], 'size': item['size'], 'sha256': digest}
            # This frozen validator checks only the reference, never reads TAR.
            transport._validate_member(stem, kind, members[kind])
            archives[item['archive_path']] = {'path': item['archive_path'], **snapshot}
        validate_role_member(role, members['video'])
        require(front['archive'] == members['video']['archive_path'] and front['member'] == members['video']['member_name']
                and front['offset'] == members['video']['offset'] and front['bytes'] == members['video']['size']
                and front['video_sha256'] == members['video']['sha256'], 'Consolidated video evidence differs')
        metadata = decoded['video']['video_metadata']
        require(all(row['end_time'] <= metadata['frame_count'] / metadata['fps'] + 1e-6 for row in rows), 'Nominal interval exceeds physical D video')
        g_sizes = {r['source_video']['size_bytes'] for r in rows}
        require(len(g_sizes) == 1, 'Inconsistent historical G sizes within recording')
        previous_size = next(iter(g_sizes))
        if previous_size != members['video']['size']:
            changed_sizes.append({'stem': stem, 'historical_G_size_bytes': previous_size, 'audited_D_size_bytes': members['video']['size']})
        sources.append({'stem': stem, 'speaker_id': first['speaker_id'], 'split': first['split'], 'recording_id': first['recording_id'],
            'package': first['package'], 'record_ids': [r['id'] for r in rows], 'members': members,
            'video_fps': metadata['fps'], 'video_frame_count': metadata['frame_count'],
            'video_duration_seconds': metadata['frame_count'] / metadata['fps'],
            'video_metadata': deepcopy(metadata), 'label_fps': annotated['Video_info']['FPS'],
            'decode_evidence': decoded_refs, 'label_copy': file_ref(first['label_path'], label['sha256']),
            'readiness_role': role['readiness_role'], 'historical_G_size_bytes': previous_size,
            'historical_G_size_matches_audited_D': previous_size == members['video']['size'],
            'nominal_label_times_preserved': True, 'old59_time_offset_applied': False, 'verified_new_time_offset_seconds': None,
            'spoken_transcript_alignment_reviewed': False, 'training_ready': False})
    archive_list = [archives[name] for name in sorted(archives)]
    require({PureWindowsPath(ref['path']).name for ref in archive_list} == {'TS2.tar', 'TS20.tar', 'TS200.tar', 'TL2.tar', 'TL20.tar', 'TL200.tar'}, 'Wrong six-source TAR scope')
    old.stat_archives(archive_list)
    for path, digest in bindings.items():
        require(sha256(path) == digest, 'Evidence changed during plan construction')
    lengths = [sum(member['size'] for member in source['members'].values()) for source in sources]
    return {'schema': SCHEMA, 'status': 'technical_sources_bound_pending_new_timing_and_transcript_review',
        'selection': {'path': str(selection_path), 'sha256': SELECTION_SHA}, 'parent_plan': {'path': str(selection_path), 'sha256': SELECTION_SHA},
        'records': deepcopy(selection['records']), 'sources': sources, 'archives': archive_list,
        'audit_root': str(audit_root), 'overall_audit': {'path': str(audit_root / 'overall_summary.json'), 'sha256': OVERALL_SHA},
        'bindings': [{'path': path, 'sha256': digest} for path, digest in sorted(bindings.items())],
        'code': [{'path': str(Path(__file__).resolve()), 'sha256': sha256(__file__)},
                 {'path': str(BASE / 'test_build_plan.py'), 'sha256': sha256(BASE / 'test_build_plan.py')},
                 {'path': str(OLD_BUILD), 'sha256': OLD_BUILD_SHA}, {'path': str(TAR_SOURCE), 'sha256': TAR_SHA},
                 {'path': str(TOKENIZER), 'sha256': TOKENIZER_SHA}],
        'counts': {'sentences': 120, 'train': 96, 'val': 24, 'test': 0, 'speakers': 15, 'recordings': 30,
                   'archives': 6, 'video_members': 30, 'label_members': 30, 'audio_members': 30},
        'role_checks': role_checks, 'historical_G_size_differences': changed_sizes,
        'storage': {'strategy': 'one_recording_at_a_time_workspace_temporary_files', 'max_one_recording_combined_bytes': max(lengths),
                    'total_mapped_member_bytes': sum(lengths), 'bytes_by_role': {kind: sum(s['members'][kind]['size'] for s in sources) for kind in ['video', 'audio', 'label']},
                    'minimum_free_after_materialization_bytes': 10 * 1024**3, 'full_archive_extraction': False, 'persistent_original_copy': False},
        'scope': {'original_selection_records_unchanged': True, 'source_payloads_read': False, 'archive_stats_rechecked': True,
                  'G_drive_accessed': False, 'training': False, 'model_inference': False, 'new_time_offsets_applied': False,
                  'old59_offsets_reused': False, 'audio_model_input': False, 'final_test_payloads_read': False,
                  'all_recording_utterances_approved': False, 'training_ready': False,
                  'next_gate': 'Independently verify new video/audio timing and actual selected transcript correspondence before preparing training inputs.',
                  'limits': ['No vendor checksum; local hashes establish audited member identity only.',
                             'Current size/mtime checks must be followed by member SHA verification when materializing.',
                             'Historical G video sizes may be truncated and are preserved only as original selection metadata; no G payload was read.',
                             'Nominal sentence times and technical integrity do not establish audio/video alignment, actual spoken transcript or lipreading accuracy.',
                             'Audio is available solely for correspondence/timing review and is not a visual model input.',
                             'Passing integrity does not change speaker roles or admit all sentences from a selected recording.']}}


def load_plan(path, expected_sha256):
    require(sha256(local(path)) == expected_sha256, 'New120 source plan SHA differs')
    plan = read_json(path)
    require(plan['schema'] == SCHEMA, 'Wrong new120 plan schema')
    require(plan == collect_plan(Path(plan['selection']['path']), Path(plan['audit_root'])), 'Source plan differs from current bound evidence')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'source_plan.json')
    parser.add_argument('--verify-sha256')
    args = parser.parse_args()
    output = local(args.output)
    if args.verify_sha256:
        plan = load_plan(output, args.verify_sha256)
    else:
        plan = collect_plan()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        load_plan(output, sha256(output))
    print(json.dumps({'status': plan['status'], 'source_plan': {'path': str(output), 'sha256': sha256(output)},
                      'counts': plan['counts'], 'storage': plan['storage'], 'historical_G_size_differences': len(plan['historical_G_size_differences'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
