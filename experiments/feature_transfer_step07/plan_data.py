"""Freeze the previously proposed recording split and new, isolated inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from models.tokenizer import encode_text, normalize_text, ctc_min_input_len, tokenizer_metadata


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path | str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def freeze() -> None:
    previous = ROOT/'experiments/training_alignment_step06'
    inventory_path = previous/'training_inventory.json'
    mapping_path = previous/'results/audited_time_mapping_candidates.json'
    require(sha(inventory_path) == '0c0409d1df00841841b45cf83a447fea1c11dcff685744dc63c4b05f708e36f1', 'Inventory changed')
    require(sha(mapping_path) == '0ab3803902c590e076625b5be7af88febbdc92601b7cee18f7be509064ef7ee9', 'Audited mapping changed')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    offsets = {s['source_stem']: s for s in mapping['sources']}
    require(len(offsets) == 59 and not mapping['unresolved_source_stems'], 'Require all59 supported mappings')
    require(sha(mapping['summary']['path']) == mapping['summary']['sha256'], 'Summary changed')
    dev_sources = set(inventory['screen8_source_stems'])
    records = []
    for source in inventory['sources']:
        stem = source['source_stem']
        candidate = offsets[stem]
        offset = candidate['offset_seconds']
        require(sha(candidate['refined_result']['path']) == candidate['refined_result']['sha256'], 'Source mapping changed')
        for row in source['training_records']:
            tokens = encode_text(row['text'])
            require(normalize_text(row['source_text']) == row['text'], 'Text normalization changed')
            require(ctc_min_input_len(tokens) <= row['frames'], 'CTC infeasible')
            records.append({
                'id': row['id'], 'source_stem': stem, 'speaker_id': source['speaker_id'],
                'split': 'dev' if stem in dev_sources else 'train', 'sentence_id': row['sentence_id'],
                'video_path': source['video']['path'], 'video_size_bytes': source['video']['size_bytes'],
                'video_mtime_ns': source['video']['mtime_ns'], 'video_fps': row['video_fps'],
                'label_path': source['label']['path'], 'label_sha256': source['expected_label_sha256'],
                'original_start_time': row['start_time'], 'original_end_time': row['end_time'],
                'start_time': row['start_time']+offset, 'end_time': row['end_time']+offset,
                'offset_seconds': offset, 'frames': row['frames'], 'text': row['text'],
                'source_text': row['source_text'], 'token_ids': tokens,
                'legacy_cache_path': str(ROOT/'data_cache/datasets/aihub_timed_v4'/row['cache_path']),
                'legacy_cache_sha256': row['cache_sha256'],
            })
    records.sort(key=lambda r: (r['source_stem'], r['original_start_time'], r['id']))
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'dev']
    require(len(records) == len({r['id'] for r in records}) == 512 and len(train) == 441 and len(dev) == 71, 'Fixed split changed')
    require(not {r['source_stem'] for r in train} & {r['source_stem'] for r in dev}, 'Recording leakage')
    require(not {r['text'] for r in train} & {r['text'] for r in dev}, 'Exact transcript overlap')
    speakers = sorted({r['speaker_id'] for r in records})
    pilot = [min((r for r in train if r['speaker_id'] == speaker), key=lambda r: (-r['frames'], r['id']))['id'] for speaker in speakers]
    reproduction = [min(r['id'] for r in train if r['speaker_id'] == speaker) for speaker in ['C002','C004','C282','C312']]
    code = [BASE/'prepare_inputs.py', ROOT/'experiments/legacy_recheck_step05/prepare_inputs.py',
            ROOT/'scripts/prepare_data.py', ROOT/'backend/preprocessing/roi_extractor.py',
            ROOT/'backend/models/tokenizer.py', ROOT/'backend/models/cache.py', ROOT/'backend/config.py']
    value = {
        'schema': 'feature-transfer-step07-data-plan-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_before_new_extraction': True, 'planner': bind(Path(__file__)),
        'bindings': [bind(inventory_path), bind(mapping_path), bind(previous/'results/summary.json'),
                     bind(previous/'results/independent_refined_audit.json'), bind(ROOT/'data_cache/datasets/aihub_timed_v4/manifest.json')],
        'records': records, 'tokenizer': tokenizer_metadata(),
        'split': {'train_clips':441,'train_recordings':51,'dev_clips':71,'dev_recordings':8,'speakers':speakers,
                  'dev_source_stems': sorted(dev_sources),
                  'rule':'First lexicographic recording per speaker from audited59, proposed in stage06 before new outcomes',
                  'prior_exposure':'All512 were used in earlier project training; this is development, not an untouched final test. New students start from identical fresh random initialization.',
                  'teacher_pretraining_overlap':'Unknown; cannot claim teacher-unseen evaluation'},
        'preparation': {'output_directory': str(BASE/'inputs'), 'reproduction_ids': reproduction, 'pilot_ids': pilot,
            'teacher_context_frames': 2, 'environment': {'opencv': cv2.__version__, 'numpy': np.__version__},
            'code': [bind(p) for p in code],
            'student':'Original unchanged bbox helper:RGB96x96,25fps,per-side margin0.20,round((corrected_start+k/25)*actualfps),uint8',
            'teacher':'BGR96x96,provided bbox margin0,native dense frame indices; union of ceil-time interval and student indices,extended2frames each side,clip to physical media bounds without synthesized frames',
            'teacher_normalization':'Whole prepared native context window FP32 scalar mean then sample standard deviation(correction=1); follows published scalar normalization, with explicitly extended context window',
            'mapping':'Exact original integer source frame identity, not length-ratio interpolation; no out-of-range clamping',
            'reproduction_gate':'Four fixed original-time inputs must reproduce every original cached pixel and label; stop on any difference',
        },
        'storage': {'workspace_free_bytes_at_freeze': shutil.disk_usage(ROOT).free,
                    'minimum_free_bytes': 10*1024**3, 'original_video_copy':False,
                    'cloud_cache_caveat':'Drive may hydrate more than requested source ranges'},
        'scope': {'read_only_originals':True,'new_dataset_download':False,'audio_input':False,
                  'test50_payload':False,'old_val20_payload':False,'service_change':False},
    }
    target = BASE/'data_plan.json'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'data_plan':bind(target),'train':441,'dev':71,'pilot_ids':pilot,'reproduction_ids':reproduction}, ensure_ascii=False))


if __name__ == '__main__':
    freeze()
