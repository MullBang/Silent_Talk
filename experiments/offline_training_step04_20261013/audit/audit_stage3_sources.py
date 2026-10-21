"""Independent read-only stage-3 evidence check; never imports production code."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S3 = ROOT / 'experiments/offline_dataset_step03_20261013'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def tokens(text):
    result = []
    for c in unicodedata.normalize('NFC', text):
        if c == ' ':
            result.append(68)
        elif 0xAC00 <= ord(c) <= 0xD7A3:
            syllable = ord(c) - 0xAC00
            result.extend([1 + syllable // 588, 20 + (syllable % 588) // 28])
            if syllable % 28:
                result.append(40 + syllable % 28)
        else:
            raise AssertionError(('Unexpected canonical character', c))
    return result

def inspect():
    complete = read(S3 / 'results/completion_manifest.json')
    assert len(complete['artifacts']) == 114
    for row in complete['artifacts']:
        p = Path(row['path'])
        assert p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], str(p)
    scope = read(S3 / 'results/scope_snapshot.json')
    for row in scope['artifacts']:
        p = Path(row['path'])
        assert p.exists() == row['exists'], str(p)
        if row['exists']:
            assert digest(p) == row['sha256'], str(p)
    handoff = read(S3 / 'results/cache_handoff.json')
    assert handoff['training_ready'] is False
    assert handoff['new_original_pixels_human_approved'] is False
    rows = handoff['records']
    assert len(rows) == 30
    parents, clips, pixel_counts = defaultdict(list), [], Counter()
    for row in rows:
        assert digest(row['npz_path']) == row['npz_sha256']
        assert digest(row['metadata_path']) == row['metadata_sha256']
        m = read(row['metadata_path'])
        y = tokens(row['text'])
        minimum = len(y) + sum(a == b for a,b in zip(y,y[1:]))
        assert m['label'] == y and m['ctc_min_frames'] == minimum <= m['T']
        assert row['training_ready'] is False and row['provisional'] is True
        assert row['exact_visual_boundary_certified'] is False
        assert m['training_ready'] is False and m['provisional'] is True
        assert m['new_original_pixels_human_approved'] is False
        assert m['audio_included'] is False and m['raw_padding_frames'] == 0
        with np.load(row['npz_path'], allow_pickle=False) as z:
            assert set(z.files) == {'frames_rgb','label','length','metadata_json','schema','training_ready','provisional'}
            x = z['frames_rgb']
            assert x.dtype == np.uint8 and x.shape == (row['T'],96,96,3)
            assert z['label'].dtype == np.int64 and z['label'].tolist() == y
            assert z['length'].shape == () and int(z['length']) == len(x)
            assert json.loads(str(z['metadata_json'].item())) == m
            assert bool(z['training_ready'].item()) is False and bool(z['provisional'].item()) is True
            assert hashlib.sha256(x.tobytes()).hexdigest() == m['rgb_uint8_sha256']
        for key in ('clip_id','parent_id','kind','split','text','T'):
            assert row[key] == m[key], key
        parents[row['parent_id']].append(row)
        pixel_counts[row['kind']] += row['T']
        clips.append({k:row[k] for k in ('clip_id','parent_id','kind','split','text','T','ctc_min_frames','boundary_followup_flag','display_neighbor_intersections','exact_visual_boundary_certified')})
    assert len(parents) == 15
    for pair in parents.values():
        assert {r['kind'] for r in pair} == {'phrase','full_sentence'}
        assert len({r['split'] for r in pair}) == 1
    split = Counter(pair[0]['split'] for pair in parents.values())
    assert split == {'train':12,'dev':3}
    assert pixel_counts == {'phrase':577,'full_sentence':1446}
    by_text = defaultdict(set)
    for row in rows:
        by_text[''.join(unicodedata.normalize('NFC',row['text']).split())].add(row['split'])
    assert all(len(v) == 1 for v in by_text.values())
    held = 'lip_J_1_F_03_C293_A_012_019'
    assert held not in parents
    flags = [r for r in rows if r['boundary_followup_flag']]
    assert len(flags) == 2 and {r['parent_id'] for r in flags} == {'lip_J_1_F_02_C002_A_012_035'}
    assert all(r['display_neighbor_intersections'][0]['overlap']['numerator'] == 1 and
               r['display_neighbor_intersections'][0]['overlap']['denominator'] == 375 for r in flags)
    return {'schema':'stage4-independent-stage3-source-audit-v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
            'audit_code_sha256':digest(__file__),
            'completion_manifest_sha256':digest(S3/'results/completion_manifest.json'),
            'stage3_frozen_artifacts_unchanged':114,
            'scope_artifacts_checked':len(scope['artifacts']),
            'baseline_service_weights_already_absent':True,
            'parents':15,'clips':30,'parent_splits':dict(split),
            'frames_by_role':dict(pixel_counts),'total_frames':sum(pixel_counts.values()),
            'boundary_flagged_clips':2,'held_oov_parent_excluded':held,
            'source_training_ready_preserved':False,'exact_visual_boundary_certified':False,
            'scope':'Independent source hashes, NPZ pixels/metadata, 69-token arithmetic, paired split and preserved flags; no original TAR, neural model, or training execution.',
            'boundary_interpretation':'Display support overlaps next label by 1/375 second. Prior actual native-frame support audit found zero overlap; this flag does not prove audible or visible contamination.',
            'records':clips}

if __name__ == '__main__':
    output = HERE / 'stage3_source_preflight.json'
    result = inspect()
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'},ensure_ascii=False))
