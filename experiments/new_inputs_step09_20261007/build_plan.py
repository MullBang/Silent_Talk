"""Freeze new120 visual preparation and a provisional, original-train-only feature subset."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build():
    refs = {
        'source_plan': ('experiments/new_data_step06_20261007/source_plan.json', '5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929'),
        'media_manifest': ('experiments/new_data_step06_20261007/results/media_manifest.json', 'fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8'),
        'label_targets': ('experiments/transcript_step07_20261007/results/label_targets.json', 'e791f567ded1860e5baa5935232eb1d8bc32326a27b34e66f5392472143a18f6'),
        'step08_summary': ('experiments/review_step08_20261007/results/step08_summary.json', 'cb4a0e03a3422e48d624984c5bf2310cf9632cf68b4144d701f88d5bc420f5bd'),
        'review_data': ('experiments/transcript_step07_20261007/review/review_data.json', '7efa45a3154ce78f9c11da717966406eb9ce3aef883fc939496b5a5ecfbbaa42'),
    }
    bindings = {}
    for key, (path, digest) in refs.items():
        bindings[key] = bind(ROOT / path)
        require(bindings[key]['sha256'] == digest, 'Changed ' + key)
    docs = {key: read(spec['path']) for key, spec in bindings.items()}
    source, media, labels, review = (docs[k] for k in ('source_plan', 'media_manifest', 'label_targets', 'review_data'))
    ids = [r['id'] for r in source['records']]
    require(len(set(ids)) == 120 and ids == labels['ids'] == [r['id'] for r in media['clips']] ==
            [r['id'] for r in review['records']], 'Cohort identity/order mismatch')
    sources = {s['stem']: s for s in source['sources']}
    records = []
    for src, clip, target, compared in zip(source['records'], media['clips'], labels['records'], review['records']):
        require(src['split'] == clip['split'] == target['split'] == compared['split'], 'Changed role')
        require(src['source_text'] == clip['reference'] == target['source_text'] == compared['reference'], 'Changed original text')
        require(src['label_sha256'] == clip['label_sha256'] == target['label_sha256'], 'Changed original label')
        require(clip['status'] == 'prepared' and clip['timing_status'] == 'verified', 'Unverified input timing')
        require(compared['canonical_target'] == target['canonical_target'] and target['ctc_feasible'], 'Unusable target')
        timing = clip['timing']
        start, end = timing['corrected_start_time'], timing['corrected_end_time']
        offset = start - src['start_time']
        require(math.isclose(end - src['end_time'], offset, abs_tol=1e-12), 'Nonconstant time correction')
        require(timing['student_frames'] == target['student_frames'] == src['expected_frames'], 'Changed frame count')
        tokens = target['canonical_tokens']
        ctc = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        require(ctc == target['ctc_min_frames'] and ctc <= timing['student_frames'], 'CTC infeasible')
        selected = src['split'] == 'train' and compared['comparison_group'] == 'both_supported'
        stem = clip['source_stem']
        require(sources[stem]['split'] == src['split'] and src['id'] in sources[stem]['record_ids'], 'Source role mismatch')
        row = {'id': src['id'], 'split': src['split'], 'speaker_id': src['speaker_id'],
               'source_stem': stem, 'sentence_id': src['sentence_id'], 'package': src['package'],
               'source_text': src['source_text'], 'text': target['canonical_target'], 'token_ids': tokens,
               'label_path': src['label_path'], 'label_sha256': src['label_sha256'],
               'original_start_time': src['start_time'], 'original_end_time': src['end_time'],
               'start_time': start, 'end_time': end, 'offset_seconds': offset,
               'video_fps': timing['actual_video_fps'], 'source_frame_count': sources[stem]['video_frame_count'],
               'frames': timing['student_frames'], 'ctc_min_frames': ctc,
               'student_native_frame_indices': timing['native_frame_indices'],
               'comparison_group': compared['comparison_group'], 'teacher_candidate': selected,
               'preparation_role': 'provisional_train_candidate' if selected else 'development_all_unfiltered' if src['split'] == 'val' else 'train_hold',
               'full_transcript_human_approved': False, 'human_boundary_approved': False,
               'training_ready': False}
        records.append(row)
    counts = Counter(r['preparation_role'] for r in records)
    require(counts == {'provisional_train_candidate': 69, 'train_hold': 27, 'development_all_unfiltered': 24}, 'Changed selection')
    candidate_ids = [r['id'] for r in records if r['teacher_candidate']]
    dev_ids = [r['id'] for r in records if r['split'] == 'val']
    train_speakers = {r['speaker_id'] for r in records if r['split'] == 'train'}
    dev_speakers = {r['speaker_id'] for r in records if r['split'] == 'val'}
    require(len(train_speakers) == 12 and len(dev_speakers) == 3 and not train_speakers & dev_speakers, 'Speaker leakage')
    overlap = [{'train_id': a['id'], 'val_id': b['id'], 'train_text': a['text'], 'val_text': b['text']}
               for a in records if a['teacher_candidate'] for b in records if b['split'] == 'val'
               and a['text'].replace(' ', '') == b['text'].replace(' ', '')]
    require(len(overlap) == 1, 'Text overlap inventory changed')
    return {'schema': 'lipnet-step09-provisional-data-plan-v1',
            'status': 'frozen_before_new_input_or_feature_preparation', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'generator': bind(Path(__file__)), 'inputs': bindings,
            'original512_plan': bind(ROOT / 'experiments/feature_transfer_step07/data_plan.json'),
            'records': records, 'ids': ids, 'sources': source['sources'],
            'teacher_candidate_ids': candidate_ids, 'held_train_ids': [r['id'] for r in records if r['preparation_role'] == 'train_hold'],
            'development_ids': dev_ids,
            'counts': {'all': 120, 'train': 96, 'val': 24, 'teacher_candidate': 69, 'train_hold': 27,
                       'student_frames': sum(r['frames'] for r in records),
                       'candidate_student_frames': sum(r['frames'] for r in records if r['teacher_candidate']),
                       'dev_student_frames': sum(r['frames'] for r in records if r['split'] == 'val')},
            'policy': {'purpose': 'Prepare reversible visual inputs and frozen teacher features, not student training.',
                       'candidate_selection': 'Original split=train AND Step7 comparison_group=both_supported (NFC/punctuation-normalized, spaces ignored).',
                       'development_selection': 'All original24 regardless of ASR agreement; separate full and transcript-overlap-excluded results in a future evaluation.',
                       'boundaries': 'Reuse measured Step6 WAV-to-video offsets and exact native indices; do not use ASR word timestamps.',
                       'pending_listening': docs['step08_summary']['pending_scoped_questions'],
                       'paired_storage': 'Only candidate69 persist teacher pixels; val24 and heldtrain27 persist student arrays only.',
                       'teacher_context': 'Two real native frames per side, matching earlier teacher/student preparation.',
                       'source_transport': 'Bounded original D: video+label TAR members one recording at a time; no G: or preview MP4 input.',
                       'execution_binding': 'Separate input/teacher execution plans must freeze code, runtime and exact preceding output/audit hashes before execution.'},
            'candidate_vs_dev_text_overlap': overlap,
            'limits': ['ASR agreement from the same model is provisional evidence, not independent verification or human approval.',
                       'Selection excludes some numeric expressions, disfluencies and difficult utterances; it is biased toward consistent ASR outputs.',
                       'All24 new development items keep uncertain transcript status; they are not certified ground truth.',
                       'New development speakers are disjoint, but one candidate/dev text is duplicated.',
                       'Old development71 shares speakers with old train441; do not call combined development95 entirely speaker-disjoint.',
                       'Input/feature availability is separate from training inclusion; original frozen training_ready flags remain false.'],
            'training_ready': False, 'student_training_executed': False, 'service_weights_changed': False}


if __name__ == '__main__':
    plan = build()
    path = BASE / 'data_plan.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': bind(path), 'counts': plan['counts']}, ensure_ascii=False))
