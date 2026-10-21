"""Check labeled neighboring intervals in saved context audio, without ASR alignment."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / 'new_data_step06_20261007'
PREVIOUS = BASE.parent / 'transcript_step07_20261007'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def interval(start, end):
    require(not isinstance(start, bool) and not isinstance(end, bool), 'Boolean interval')
    require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end, 'Invalid interval')
    return float(start), float(end)


def overlap(a, b):
    interval(*a)
    interval(*b)
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def main():
    specs = {
        'source_plan': (OLD / 'source_plan.json', '5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929'),
        'media_manifest': (OLD / 'results/media_manifest.json', 'fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8'),
        'review_data': (PREVIOUS / 'review/review_data.json', '7efa45a3154ce78f9c11da717966406eb9ce3aef883fc939496b5a5ecfbbaa42'),
    }
    for name, (path, digest) in specs.items():
        require(bind(path)['sha256'] == digest, 'Changed ' + name)
    source, media, review = (read(specs[k][0]) for k in specs)
    require([r['id'] for r in source['records']] == [r['id'] for r in media['clips']] ==
            [r['id'] for r in review['records']], 'Different records/order')
    label_docs, label_refs, rows = {}, {}, []
    for src, clip, displayed in zip(source['records'], media['clips'], review['records']):
        path = Path(src['label_path'])
        if str(path) not in label_docs:
            ref = bind(path)
            require(ref['sha256'] == src['label_sha256'], 'Original label changed')
            document = read(path)
            require(isinstance(document, list) and len(document) == 1, 'Unsupported label schema')
            sentences = document[0]['Sentence_info']
            require(len({s['ID'] for s in sentences}) == len(sentences), 'Duplicate sentence id')
            for sentence in sentences:
                interval(sentence['start_time'], sentence['end_time'])
            label_docs[str(path)] = sentences
            label_refs[str(path)] = ref
        require(label_refs[str(path)]['sha256'] == src['label_sha256'] == clip['label_sha256'], 'Label binding mismatch')
        sentences = label_docs[str(path)]
        match = [s for s in sentences if s['ID'] == src['sentence_id']]
        require(len(match) == 1, 'Missing sentence')
        sentence = match[0]
        require(sentence['sentence_text'] == src['source_text'] == clip['reference'] == displayed['reference'], 'Text mismatch')
        target = interval(src['start_time'], src['end_time'])
        require(target == interval(sentence['start_time'], sentence['end_time']) == interval(clip['start_time'], clip['end_time']), 'Clock mismatch')
        context = clip['context_audio']
        require(context['clock'] == 'original_label_wav', 'Wrong context clock')
        bounds = interval(context['start_seconds'], context['end_seconds'])
        require(bounds[0] <= target[0] and bounds[1] >= target[1], 'Context must contain target')
        neighbors = []
        for other in sentences:
            if other['ID'] == sentence['ID']:
                continue
            other_bounds = interval(other['start_time'], other['end_time'])
            duration = overlap(bounds, other_bounds)
            if duration:
                target_overlap = overlap(target, other_bounds)
                side = 'before' if other_bounds[1] <= target[0] else 'after' if other_bounds[0] >= target[1] else 'overlaps_target_label'
                neighbors.append({'sentence_id': other['ID'], 'text': other['sentence_text'],
                                  'start_time': other_bounds[0], 'end_time': other_bounds[1], 'side': side,
                                  'overlap_with_context_seconds': duration, 'overlap_with_target_seconds': target_overlap})
        rows.append({'id': src['id'], 'split': src['split'], 'speaker_id': src['speaker_id'],
                     'label': label_refs[str(path)], 'target_start': target[0], 'target_end': target[1],
                     'context_start': bounds[0], 'context_end': bounds[1], 'context_audio': context,
                     'comparison_group': displayed['comparison_group'], 'neighbors': neighbors,
                     'actual_neighbor_speech_confirmed': False, 'label_boundary_modified': False})
    require(len(rows) == 120 and Counter(r['split'] for r in rows) == {'train': 96, 'val': 24}, 'Changed cohort')
    result = {'schema': 'lipnet-step08-context-neighbors-v1', 'status': 'completed_metadata_only',
              'code': bind(Path(__file__)), 'inputs': {k: bind(v[0]) for k, v in specs.items()},
              'label_files': list(label_refs.values()), 'records': rows,
              'counts': {'records': 120, 'label_files': len(label_refs),
                         'context_with_labeled_neighbor': sum(bool(r['neighbors']) for r in rows),
                         'priority_context_with_labeled_neighbor': sum(bool(r['neighbors']) and r['comparison_group'] != 'both_supported' for r in rows),
                         'target_with_labeled_overlap': sum(any(n['overlap_with_target_seconds'] > 0 for n in r['neighbors']) for r in rows)},
              'limits': ['Metadata overlap is not confirmation of audible neighboring speech.',
                         'No audio timing or labels are changed; ASR word times are not used.',
                         'No full-human approval or training inclusion is inferred.']}
    output = BASE / 'context_neighbors.json'
    require(not output.exists(), 'Preserve previous result')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
