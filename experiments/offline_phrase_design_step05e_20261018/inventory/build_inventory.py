"""Count transcript-only 1-4-eojeol spans; never infer media times or clip readiness."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unicodedata

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
TOKENIZER=ROOT/'backend/models/tokenizer.py'
TOKENIZER_SHA='c73b1c393a739cfd0de03d9bd310e1a6d6339949d2add7121f98794fd7020714'
VOCAB_SHA='3ab81291a1892a4aece2d41bce634ff54663c4b93e3d56aef81f6783c822ff62'


def require(condition,message):
    if not condition:raise ValueError(message)


def ref(path):
    raw=path.read_bytes()
    return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def text_key(text):
    require(isinstance(text,str),'Text must be a string')
    return ''.join(unicodedata.normalize('NFC',text).split())


def load_tokenizer():
    require(ref(TOKENIZER)['sha256']==TOKENIZER_SHA,'Frozen tokenizer source changed')
    spec=importlib.util.spec_from_file_location('stage5e_inventory_tokenizer',TOKENIZER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.NUM_CLASSES==69 and module.VOCAB_SHA256==VOCAB_SHA,'Wrong tokenizer vocabulary')
    return module


def read_bound_json(path,expected):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.suffix.lower()=='.json','Only workspace JSON metadata inputs allowed')
    require(isinstance(expected,str) and re.fullmatch(r'[0-9a-f]{64}',expected),'An external SHA256 is required')
    artifact=ref(path);require(artifact['sha256']==expected,'Input metadata hash mismatch')
    return json.loads(path.read_text(encoding='utf-8')),artifact


def validate_records(records,protected,tokenizer):
    require(isinstance(records,list) and records,'Expected nonempty eligible record list')
    ids=set();parents=set()
    phrase_keys=protected['protected_phrase_text_keys']
    require(isinstance(phrase_keys,list) and len(phrase_keys)==len(set(phrase_keys))
            and all(isinstance(k,str) and k and text_key(k)==k for k in phrase_keys),'Invalid protected phrase keys')
    p_ids=set(protected.get('protected_parent_ids',[]))
    p_recordings=set(protected.get('protected_recording_keys',[]))
    p_texts=set(protected.get('protected_full_text_keys',[]))
    for row in records:
        require(isinstance(row,dict),'Record must be a dictionary')
        for name in ('id','parent_id','speaker_id','source_stem','recording_key','text'):
            require(isinstance(row[name],str) and row[name].strip(),'Missing record identity or canonical text')
        require(row['id'] not in ids and row['parent_id'] not in parents,'Duplicate source/parent ID')
        ids.add(row['id']);parents.add(row['parent_id'])
        require(row['parent_id'] not in p_ids and row['recording_key'] not in p_recordings
                and text_key(row['text']) not in p_texts,'Protected parent/recording/full transcript entered inventory')
        require(row['text']==tokenizer.normalize_text(row['text']) and row['text'],
                'Canonical transcript must already match frozen normalization; do not silently rewrite')
        require(isinstance(row['token_ids'],list) and all(type(t) is int and 0<t<69 for t in row['token_ids'])
                and row['token_ids']==tokenizer.encode_text(row['text']),'Parent canonical token IDs differ')
        require(type(row['frames']) is int and row['frames']>0,'Parent frames must be a positive reference count')
        require(isinstance(row['source_approval_status'],(dict,str)),'Source approval provenance must be retained')
    candidate_keys=protected.get('protected_candidate_text_keys',phrase_keys)
    require(isinstance(candidate_keys,list) and len(candidate_keys)==len(set(candidate_keys))
            and all(isinstance(k,str) and k and text_key(k)==k for k in candidate_keys),'Invalid protected candidate keys')
    require(set(candidate_keys)>=set(phrase_keys),'Candidate protection cannot omit development phrases')
    if 'protected_candidate_text_keys' in protected:
        require(set(candidate_keys)>=p_texts,'Candidate protection cannot omit protected full transcripts')
    return set(candidate_keys)


def make_candidates(records,protected,tokenizer):
    protected_keys=validate_records(records,protected,tokenizer)
    dev_phrase_keys=set(protected['protected_phrase_text_keys'])
    protected_full_keys=set(protected.get('protected_full_text_keys',[]))
    candidates=[];excluded=[]
    for row in records:
        words=row['text'].split()
        for n in (1,2,3,4):
            for start in range(len(words)-n+1):
                end=start+n;text=' '.join(words[start:end]);key=text_key(text)
                candidate_id=f"{row['parent_id']}__eojeol_{start:03d}_{end:03d}"
                dev_substrings=sorted(k for k in dev_phrase_keys if k in key)
                if key in protected_keys or dev_substrings:
                    excluded.append({'candidate_id':candidate_id,'parent_id':row['parent_id'],
                        'word_start':start,'word_end':end,'n_eojeols':n,'text':text,
                        'reason':'exact_NFC_whitespace_removed_protected_text_match' if key in protected_keys
                                 else 'contains_current_dev_phrase_NFC_whitespace_removed',
                        'exact_protected_text_match':key in protected_keys,
                        'contained_current_dev_phrase_keys':dev_substrings})
                    continue
                tokens=tokenizer.encode_text(text)
                candidate={'candidate_id':candidate_id,'source_record_id':row['id'],'parent_id':row['parent_id'],
                    'speaker_id':row['speaker_id'],'recording_key':row['recording_key'],'source_stem':row['source_stem'],
                    'word_start':start,'word_end':end,'span_indexing':'zero_based_half_open_whitespace_eojeols',
                    'text':text,'n_eojeols':n,'token_ids':tokens,'label_length':len(tokens),
                    'ctc_min_frames':tokenizer.ctc_min_input_len(tokens),
                    'syllable_length':sum('가'<=char<='힣' for char in text),'text_key_for_protection':key,
                    'contained_protected_full_text_keys_without_exact_match':sorted(k for k in protected_full_keys if k in key and k!=key),
                    'parent_reference_frames':row['frames'],'parent_frames_are_not_candidate_frames':True,
                    'source_approval_status':copy.deepcopy(row['source_approval_status']),
                    'start_seconds':None,'end_seconds':None,'frame_count':None,'ctc_feasible':None,
                    'metadata_only':True,'training_ready':False,'new_human_approval_added':False,
                    'candidate_boundary_confirmed':False,'candidate_status':'transcript_span_requires_audio_visual_alignment_review'}
                candidates.append(candidate)
    require(len(candidates)==len({r['candidate_id'] for r in candidates}),'Candidate ID collision')
    return candidates,excluded


def frequencies(candidates):
    grouped=defaultdict(list)
    for row in candidates:grouped[row['text']].append(row)
    result=[]
    for text,rows in grouped.items():
        parents=sorted({r['parent_id'] for r in rows});recordings=sorted({r['recording_key'] for r in rows})
        speakers=sorted({r['speaker_id'] for r in rows})
        by_speaker=[]
        for speaker in speakers:
            selected=[r for r in rows if r['speaker_id']==speaker]
            by_speaker.append({'speaker_id':speaker,'occurrences':len(selected),
                'sentence_count':len({r['parent_id'] for r in selected}),
                'recording_count':len({r['recording_key'] for r in selected}),
                'parent_ids':sorted({r['parent_id'] for r in selected}),
                'recording_keys':sorted({r['recording_key'] for r in selected})})
        result.append({'text':text,'n_eojeols':len(text.split()),'occurrences':len(rows),
            'sentence_count':len(parents),'recording_count':len(recordings),'speaker_count':len(speakers),
            'candidate_ids':sorted(r['candidate_id'] for r in rows),'parent_ids':parents,
            'recording_keys':recordings,'speaker_ids':speakers,'per_speaker':by_speaker})
    return sorted(result,key=lambda r:(-r['speaker_count'],-r['recording_count'],-r['sentence_count'],-r['occurrences'],r['text']))


def numeric_profile(values):
    return {'count':len(values),'min':min(values) if values else None,'max':max(values) if values else None,
        'sum':sum(values),'mean':sum(values)/len(values) if values else None,
        'histogram':{str(k):v for k,v in sorted(Counter(values).items())}}


def coverage(candidates,tokenizer):
    tokens=Counter(t for row in candidates for t in row['token_ids'])
    syllables=Counter(ch for row in candidates for ch in row['text'] if '가'<=ch<='힣')
    words=Counter(word for row in candidates for word in row['text'].split())
    return {'candidate_occurrences':len(candidates),'unique_surface_texts':len({r['text'] for r in candidates}),
        'n_eojeols':numeric_profile([r['n_eojeols'] for r in candidates]),
        'syllable_lengths':numeric_profile([r['syllable_length'] for r in candidates]),
        'label_lengths':numeric_profile([r['label_length'] for r in candidates]),
        'ctc_minimum_lengths':numeric_profile([r['ctc_min_frames'] for r in candidates]),
        'nonblank_token_vocabulary_size':68,'observed_token_type_count_including_space':len(tokens),
        'observed_token_ids':sorted(tokens),'missing_nonblank_token_ids':[t for t in range(1,69) if t not in tokens],
        'token_occurrences':[{'token_id':t,'grapheme':tokenizer.IDX_TO_GRAPHEME[t],'occurrences':tokens[t]} for t in sorted(tokens)],
        'unique_hangul_syllable_count':len(syllables),
        'syllable_occurrences':[{'syllable':s,'occurrences':n} for s,n in sorted(syllables.items(),key=lambda x:(-x[1],x[0]))],
        'unique_eojeol_surface_count':len(words),
        'eojeol_occurrences':[{'text':w,'occurrences':n} for w,n in sorted(words.items(),key=lambda x:(-x[1],x[0]))],
        'overlapping_phrase_spans_repeat_tokens_and_words':True,
        'no_duration_or_candidate_frame_count_inferred':True,'CTC_length_feasibility_measured':False}


def full_transcript_duplicates(records):
    grouped=defaultdict(list)
    for row in records:grouped[text_key(row['text'])].append(row)
    groups=[]
    for key,rows in grouped.items():
        if len(rows)<2:continue
        groups.append({'whitespace_removed_NFC_text_key':key,'surface_variants':sorted({r['text'] for r in rows}),
            'parent_count':len(rows),'parent_ids':sorted(r['parent_id'] for r in rows),
            'recording_count':len({r['recording_key'] for r in rows}),
            'recording_keys':sorted({r['recording_key'] for r in rows}),
            'speaker_count':len({r['speaker_id'] for r in rows}),'speaker_ids':sorted({r['speaker_id'] for r in rows})})
    groups.sort(key=lambda r:(-r['parent_count'],r['whitespace_removed_NFC_text_key']))
    return {'matching_rule':'NFC plus removal of all whitespace; punctuation is not additionally removed',
        'duplicate_group_count':len(groups),'parents_in_duplicate_groups':sum(r['parent_count'] for r in groups),
        'duplicate_excess_parents':sum(r['parent_count']-1 for r in groups),'groups':groups}


def summarize(records,candidates,excluded,word_freq,phrase_freq):
    phrases=[r for r in candidates if r['n_eojeols']>1]
    speakers=[]
    for speaker in sorted({r['speaker_id'] for r in records}):
        parents=[r for r in records if r['speaker_id']==speaker]
        spans=[r for r in phrases if r['speaker_id']==speaker]
        common=[r for r in phrase_freq if speaker in r['speaker_ids'] and r['speaker_count']>=2]
        speakers.append({'speaker_id':speaker,'parent_count':len(parents),
            'recording_count':len({r['recording_key'] for r in parents}),
            'unique_full_transcript_count':len({text_key(r['text']) for r in parents}),
            'phrase_occurrences':len(spans),'unique_phrase_surfaces':len({r['text'] for r in spans}),
            'shared_phrase_surfaces_with_other_speakers':len(common),
            'recording_keys':sorted({r['recording_key'] for r in parents})})
    stats={}
    for name,items in [('eojeols',word_freq),('phrases',phrase_freq)]:
        stats[name]={'surface_count':len(items),'occurrences':sum(r['occurrences'] for r in items),
            'types_by_min_speakers':{str(n):sum(r['speaker_count']>=n for r in items) for n in (2,3,4,5,8)},
            'types_by_min_recordings':{str(n):sum(r['recording_count']>=n for r in items) for n in (2,3,4)},
            'surface_speaker_count_distribution':{str(k):v for k,v in sorted(Counter(r['speaker_count'] for r in items).items())},
            'surface_occurrence_distribution':{str(k):v for k,v in sorted(Counter(r['occurrences'] for r in items).items())},
            'singleton_surface_count':sum(r['occurrences']==1 for r in items)}
    return {'schema':'offline-stage5e-transcript-span-inventory-summary-v1','status':'completed_metadata_inventory_only',
        'parent_count':len(records),'speaker_count':len(speakers),
        'all_angle_recording_count':len({r['recording_key'] for r in records}),
        'source_stem_count':len({r['source_stem'] for r in records}),
        'candidate_count':len(candidates),'single_eojeol_candidate_count':sum(r['n_eojeols']==1 for r in candidates),
        'phrase_candidate_count':len(phrases),'candidate_count_by_n_eojeols':{str(n):sum(r['n_eojeols']==n for r in candidates) for n in (1,2,3,4)},
        'excluded_protected_phrase_spans':len(excluded),'frequency_counts':stats,'per_speaker':speakers,
        'excluded_exact_protected_text_spans':sum(r['exact_protected_text_match'] for r in excluded),
        'excluded_current_dev_phrase_substring_spans':sum(bool(r['contained_current_dev_phrase_keys']) for r in excluded),
        'allowed_candidates_with_protected_full_text_substrings':sum(bool(r['contained_protected_full_text_keys_without_exact_match']) for r in candidates),
        'frequency_surface_policy':'Exact canonical surface text with single spaces; inflection and word-boundary variants remain distinct',
        'recording_policy':'Provided all-angle recording_key; source_stem is provenance and does not increase recording independence',
        'source_approval_promoted':False,'metadata_candidates_are_not_training_clips':True,
        'new_phrase_boundary_confirmations':0,'new_neural_calls':0,'new_optimizer_updates':0,
        'new_media_reads':0,'new_downloads':0,'model_output_contract':'Unchanged open text 69-token CTC; no phrase-classification labels',
        'limitations':['Contiguous overlapping text spans are dependent observations, not independent training examples.',
            'Frequency across recordings/speakers is metadata evidence only; every chosen new span still needs boundaries and review.',
            'Parent full-sentence frame counts cannot determine the duration, sample count or CTC feasibility of a substring.']}


def build(records,protected,tokenizer):
    candidates,excluded=make_candidates(records,protected,tokenizer)
    words=[r for r in candidates if r['n_eojeols']==1];phrases=[r for r in candidates if r['n_eojeols']>1]
    word_freq,phrase_freq=frequencies(words),frequencies(phrases)
    return {'eojeol_candidates':{'schema':'offline-stage5e-eojeol-candidates-v1','records':words},
        'phrase_candidates':{'schema':'offline-stage5e-phrase-candidates-v1','records':phrases},
        'excluded_candidates':{'schema':'offline-stage5e-protected-phrase-exclusions-v1','records':excluded},
        'frequencies':{'schema':'offline-stage5e-surface-frequency-v1','eojeols':word_freq,'phrases':phrase_freq},
        'coverage':{'schema':'offline-stage5e-character-token-coverage-v1','eojeols':coverage(words,tokenizer),
                    'phrases':coverage(phrases,tokenizer)},
        'full_transcript_duplicates':full_transcript_duplicates(records),
        'protected_full_text_substrings':{'schema':'offline-stage5e-protected-full-text-substring-observations-v1',
            'rule':'Nonexact contained protected full transcripts are descriptive flags; current dev phrase substrings are always excluded',
            'records':[{key:r[key] for key in ('candidate_id','parent_id','text','contained_protected_full_text_keys_without_exact_match')}
                       for r in candidates if r['contained_protected_full_text_keys_without_exact_match']]},
        'summary':summarize(records,candidates,excluded,word_freq,phrase_freq)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eligible',type=Path,required=True);parser.add_argument('--eligible-sha',required=True)
    parser.add_argument('--protected',type=Path,required=True);parser.add_argument('--protected-sha',required=True)
    args=parser.parse_args()
    eligible,eligible_ref=read_bound_json(args.eligible,args.eligible_sha)
    protected,protected_ref=read_bound_json(args.protected,args.protected_sha)
    result=build(eligible['records'],protected,load_tokenizer())
    outputs={name:BASE/(name+'.json') for name in result}
    require(not any(path.exists() for path in outputs.values()),'Existing inventory artifacts are immutable')
    refs={'eligible':eligible_ref,'protected':protected_ref,'tokenizer':ref(TOKENIZER),'producer':ref(Path(__file__))}
    for name,value in result.items():
        value['sources']=refs;value['created_utc']=datetime.now(timezone.utc).isoformat()
    require(ref(args.eligible)['sha256']==eligible_ref['sha256'] and ref(args.protected)['sha256']==protected_ref['sha256'],
            'Source metadata changed during inventory')
    for name,value in result.items():outputs[name].write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result['summary'][k] for k in ['status','parent_count','all_angle_recording_count','candidate_count',
        'single_eojeol_candidate_count','phrase_candidate_count','excluded_protected_phrase_spans']}))


if __name__=='__main__':main()
