"""Describe whether shared phrase surfaces occur in different complete transcripts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

BASE=Path(__file__).resolve().parent


def key(text):return ''.join(unicodedata.normalize('NFC',text).split())


def ref(path):
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def main():
    eligible_path=BASE.parent/'data/eligible_sentences.json';frequency_path=BASE/'frequencies.json'
    eligible=json.loads(eligible_path.read_text(encoding='utf-8'))['records']
    frequencies=json.loads(frequency_path.read_text(encoding='utf-8'))['phrases']
    parents={r['parent_id']:r for r in eligible}
    shared=[r for r in frequencies if r['speaker_count']>=2]
    records=[]
    for surface in shared:
        contexts=defaultdict(list)
        for parent_id in surface['parent_ids']:
            parent=parents[parent_id];contexts[key(parent['text'])].append(parent)
        context_rows=[]
        for transcript_key,group in sorted(contexts.items()):
            context_rows.append({'full_transcript_key':transcript_key,
                'surface_variants':sorted({p['text'] for p in group}),
                'parent_ids':sorted(p['parent_id'] for p in group),
                'speaker_ids':sorted({p['speaker_id'] for p in group}),
                'recording_keys':sorted({p['recording_key'] for p in group})})
        records.append({'text':surface['text'],'n_eojeols':surface['n_eojeols'],
            'occurrences':surface['occurrences'],'parent_count':surface['sentence_count'],
            'recording_count':surface['recording_count'],'speaker_count':surface['speaker_count'],
            'distinct_full_parent_transcript_count':len(contexts),
            'all_occurrences_share_one_full_parent_transcript':len(contexts)==1,
            'contexts':context_rows})
    records.sort(key=lambda r:(-r['speaker_count'],-r['distinct_full_parent_transcript_count'],-r['recording_count'],r['text']))
    varied=[r for r in records if r['distinct_full_parent_transcript_count']>=2]
    same=[r for r in records if r['distinct_full_parent_transcript_count']==1]
    examples=varied[:8]+same[:2]
    if len(examples)<10:examples+=varied[8:8+10-len(examples)]
    output={'schema':'offline-stage5e-shared-phrase-parent-context-diversity-v1',
        'sources':{'eligible':ref(eligible_path),'frequencies':ref(frequency_path),'producer':ref(Path(__file__))},
        'matching_rule':'Full canonical parent transcript NFC with all whitespace removed; no semantic or inflection equivalence',
        'shared_phrase_surface_count':len(shared),
        'shared_phrase_surfaces_with_at_least_two_distinct_parent_transcripts':len(varied),
        'shared_phrase_surfaces_with_one_repeated_full_parent_transcript':len(same),
        'descriptive_representatives':examples,'records':records,
        'limitations':['Transcript diversity is descriptive metadata, not statistical independence or verified visual learnability.',
            'Overlapping phrases, shared recordings, related topics and repeated scripts can remain dependent.',
            'The examples are illustrations, not a fixed training vocabulary or an approved review queue.'],
        'new_media_reads':0,'new_model_calls':0,'new_training_updates':0,'new_boundary_approvals':0}
    target=BASE/'shared_phrase_context_diversity.json'
    if target.exists():raise ValueError('Existing supplemental output is immutable')
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['shared_phrase_surface_count','shared_phrase_surfaces_with_at_least_two_distinct_parent_transcripts',
        'shared_phrase_surfaces_with_one_repeated_full_parent_transcript']},ensure_ascii=False))
    print(json.dumps([{k:r[k] for k in ['text','speaker_count','parent_count','distinct_full_parent_transcript_count']} for r in examples],ensure_ascii=False))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
