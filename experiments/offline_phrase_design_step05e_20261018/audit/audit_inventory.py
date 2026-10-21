"""Independent Unicode arithmetic and exhaustive transcript-span inventory audit."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import unicodedata
from audit_sources import ROOT, BASE, HERE, read, binding, check, textkey


def encode(text):
    result=[]
    for char in text:
        if char == ' ':
            result.append(68)
            continue
        n=ord(char)-0xAC00
        assert 0<=n<11172,repr(char)
        lead,tail=divmod(n,588)
        vowel,final=divmod(tail,28)
        result.extend([lead+1,vowel+20])
        if final:result.append(final+40)
    return result


def grapheme(t):
    if 1<=t<=19:return chr(0x1100+t-1)
    if 20<=t<=40:return chr(0x1161+t-20)
    if 41<=t<=67:return chr(0x11A8+t-41)
    assert t==68
    return ' '


def profile(values):
    counts=Counter(values)
    return {'count':len(values),'min':min(values) if values else None,'max':max(values) if values else None,
            'sum':sum(values),'mean':sum(values)/len(values) if values else None,
            'histogram':{str(k):counts[k] for k in sorted(counts)}}


def freq_expect(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['text']].append(r)
    result=[]
    for text,rs in groups.items():
        speakers=sorted({r['speaker_id'] for r in rs})
        by_speaker=[]
        for speaker in speakers:
            rr=[r for r in rs if r['speaker_id']==speaker]
            by_speaker.append({'speaker_id':speaker,'occurrences':len(rr),'sentence_count':len({r['parent_id'] for r in rr}),
                'recording_count':len({r['recording_key'] for r in rr}),'parent_ids':sorted({r['parent_id'] for r in rr}),
                'recording_keys':sorted({r['recording_key'] for r in rr})})
        result.append({'text':text,'n_eojeols':len(text.split()),'occurrences':len(rs),
            'sentence_count':len({r['parent_id'] for r in rs}),'recording_count':len({r['recording_key'] for r in rs}),
            'speaker_count':len(speakers),'candidate_ids':sorted(r['candidate_id'] for r in rs),
            'parent_ids':sorted({r['parent_id'] for r in rs}),'recording_keys':sorted({r['recording_key'] for r in rs}),
            'speaker_ids':speakers,'per_speaker':by_speaker})
    return sorted(result,key=lambda r:(-r['speaker_count'],-r['recording_count'],-r['sentence_count'],-r['occurrences'],r['text']))


def audit():
    folder=BASE/'inventory'
    docs={n:read(folder/(n+'.json')) for n in ['eojeol_candidates','phrase_candidates','excluded_candidates','frequencies','coverage','full_transcript_duplicates','protected_full_text_substrings','summary']}
    sources=docs['summary']['sources']
    for r in sources.values():check(r)
    for doc in docs.values():assert doc['sources']==sources
    parents=read(sources['eligible']['path'])['records']
    protected=read(sources['protected']['path'])
    assert len(parents)==266
    expected={1:[],2:[],3:[],4:[]};excluded=[]
    for parent in parents:
        words=parent['text'].split()
        assert encode(parent['text'])==parent['token_ids']
        for start in range(len(words)):
            for stop in range(start+1,min(start+4,len(words))+1):
                text=' '.join(words[start:stop]);n=stop-start
                cid=f"{parent['parent_id']}__eojeol_{start:03d}_{stop:03d}"
                protected_exact=textkey(text) in protected['protected_candidate_text_keys']
                contained_dev=sorted(k for k in protected['protected_phrase_text_keys'] if k in textkey(text))
                if protected_exact or contained_dev:
                    excluded.append({'candidate_id':cid,'parent_id':parent['parent_id'],'word_start':start,'word_end':stop,
                        'n_eojeols':n,'text':text,'reason':'exact_NFC_whitespace_removed_protected_text_match' if protected_exact else 'contains_current_dev_phrase_NFC_whitespace_removed',
                        'exact_protected_text_match':protected_exact,'contained_current_dev_phrase_keys':contained_dev})
                    continue
                tokens=encode(text)
                expected[n].append({'candidate_id':cid,'source_record_id':parent['id'],'parent_id':parent['parent_id'],
                    'speaker_id':parent['speaker_id'],'recording_key':parent['recording_key'],'source_stem':parent['source_stem'],
                    'word_start':start,'word_end':stop,'span_indexing':'zero_based_half_open_whitespace_eojeols','text':text,
                    'n_eojeols':n,'token_ids':tokens,'label_length':len(tokens),
                    'ctc_min_frames':len(tokens)+sum(a==b for a,b in zip(tokens,tokens[1:])),
                    'syllable_length':sum(0xAC00<=ord(c)<=0xD7A3 for c in text),'text_key_for_protection':textkey(text),
                    'contained_protected_full_text_keys_without_exact_match':sorted(k for k in protected['protected_full_text_keys'] if k in textkey(text) and k!=textkey(text)),
                    'parent_reference_frames':parent['frames'],'parent_frames_are_not_candidate_frames':True,
                    'source_approval_status':parent['source_approval_status'],'start_seconds':None,'end_seconds':None,
                    'frame_count':None,'ctc_feasible':None,'metadata_only':True,'training_ready':False,'new_human_approval_added':False,
                    'candidate_boundary_confirmed':False,'candidate_status':'transcript_span_requires_audio_visual_alignment_review'})
    expected_rows=expected[1]+expected[2]+expected[3]+expected[4]
    keyed=lambda rows:{r['candidate_id']:r for r in rows}
    observed=docs['eojeol_candidates']['records']+docs['phrase_candidates']['records']
    assert len(keyed(observed))==len(observed)
    assert keyed(observed)==keyed(expected_rows),'exhaustive candidate mismatch'
    assert keyed(docs['excluded_candidates']['records'])==keyed(excluded)
    subrows=[{k:r[k] for k in ['candidate_id','parent_id','text','contained_protected_full_text_keys_without_exact_match']}
             for r in observed if r['contained_protected_full_text_keys_without_exact_match']]
    assert keyed(docs['protected_full_text_substrings']['records'])==keyed(subrows)
    for r in observed:
        assert all(k not in r for k in ['npz_path','start_sample_16k','end_sample_16k'])
        assert not any(k in textkey(r['text']) for k in protected['protected_phrase_text_keys'])
    words=[r for r in observed if r['n_eojeols']==1]
    phrases=[r for r in observed if r['n_eojeols']>1]
    frequencies={name:freq_expect(rows) for name,rows in [('eojeols',words),('phrases',phrases)]}
    for name,rows in [('eojeols',words),('phrases',phrases)]:
        assert docs['frequencies'][name]==frequencies[name]
        got=docs['coverage'][name]
        tokens=Counter(t for r in rows for t in r['token_ids'])
        syllables=Counter(c for r in rows for c in r['text'] if 0xAC00<=ord(c)<=0xD7A3)
        eojeols=Counter(w for r in rows for w in r['text'].split())
        exp={'candidate_occurrences':len(rows),'unique_surface_texts':len({r['text'] for r in rows}),
            'n_eojeols':profile([r['n_eojeols'] for r in rows]),'syllable_lengths':profile([r['syllable_length'] for r in rows]),
            'label_lengths':profile([r['label_length'] for r in rows]),'ctc_minimum_lengths':profile([r['ctc_min_frames'] for r in rows]),
            'nonblank_token_vocabulary_size':68,'observed_token_type_count_including_space':len(tokens),
            'observed_token_ids':sorted(tokens),'missing_nonblank_token_ids':[t for t in range(1,69) if t not in tokens],
            'token_occurrences':[{'token_id':t,'grapheme':grapheme(t),'occurrences':tokens[t]} for t in sorted(tokens)],
            'unique_hangul_syllable_count':len(syllables),'syllable_occurrences':[{'syllable':s,'occurrences':v} for s,v in sorted(syllables.items(),key=lambda x:(-x[1],x[0]))],
            'unique_eojeol_surface_count':len(eojeols),'eojeol_occurrences':[{'text':s,'occurrences':v} for s,v in sorted(eojeols.items(),key=lambda x:(-x[1],x[0]))],
            'overlapping_phrase_spans_repeat_tokens_and_words':True,'no_duration_or_candidate_frame_count_inferred':True,'CTC_length_feasibility_measured':False}
        assert got==exp,f'coverage {name}'
    group=defaultdict(list)
    for r in parents:group[textkey(r['text'])].append(r)
    duplicate_groups=[]
    for key,rs in group.items():
        if len(rs)<2:continue
        duplicate_groups.append({'whitespace_removed_NFC_text_key':key,'surface_variants':sorted({r['text'] for r in rs}),
            'parent_count':len(rs),'parent_ids':sorted(r['parent_id'] for r in rs),'recording_count':len({r['recording_key'] for r in rs}),
            'recording_keys':sorted({r['recording_key'] for r in rs}),'speaker_count':len({r['speaker_id'] for r in rs}),
            'speaker_ids':sorted({r['speaker_id'] for r in rs})})
    duplicate_groups.sort(key=lambda r:(-r['parent_count'],r['whitespace_removed_NFC_text_key']))
    dup=docs['full_transcript_duplicates']
    assert dup['groups']==duplicate_groups
    assert dup['duplicate_group_count']==len(duplicate_groups)
    assert dup['parents_in_duplicate_groups']==sum(r['parent_count'] for r in duplicate_groups)
    assert dup['duplicate_excess_parents']==sum(r['parent_count']-1 for r in duplicate_groups)
    summary=docs['summary']
    for key,val in {'parent_count':len(parents),'speaker_count':len({r['speaker_id'] for r in parents}),
        'all_angle_recording_count':len({r['recording_key'] for r in parents}),'source_stem_count':len({r['source_stem'] for r in parents}),
        'candidate_count':len(observed),'single_eojeol_candidate_count':len(words),'phrase_candidate_count':len(phrases),
        'candidate_count_by_n_eojeols':{str(n):len(expected[n]) for n in range(1,5)},'excluded_protected_phrase_spans':len(excluded),
        'excluded_exact_protected_text_spans':sum(r['exact_protected_text_match'] for r in excluded),
        'excluded_current_dev_phrase_substring_spans':sum(bool(r['contained_current_dev_phrase_keys']) for r in excluded),
        'allowed_candidates_with_protected_full_text_substrings':len(subrows)}.items():assert summary[key]==val,key
    for name,freqs in frequencies.items():
        exp={'surface_count':len(freqs),'occurrences':sum(r['occurrences'] for r in freqs),
            'types_by_min_speakers':{str(n):sum(r['speaker_count']>=n for r in freqs) for n in (2,3,4,5,8)},
            'types_by_min_recordings':{str(n):sum(r['recording_count']>=n for r in freqs) for n in (2,3,4)},
            'surface_speaker_count_distribution':{str(k):v for k,v in sorted(Counter(r['speaker_count'] for r in freqs).items())},
            'surface_occurrence_distribution':{str(k):v for k,v in sorted(Counter(r['occurrences'] for r in freqs).items())},
            'singleton_surface_count':sum(r['occurrences']==1 for r in freqs)}
        assert summary['frequency_counts'][name]==exp,name
    for got in summary['per_speaker']:
        speaker=got['speaker_id'];ps=[r for r in parents if r['speaker_id']==speaker];rs=[r for r in phrases if r['speaker_id']==speaker]
        exp={'speaker_id':speaker,'parent_count':len(ps),'recording_count':len({r['recording_key'] for r in ps}),
            'unique_full_transcript_count':len({textkey(r['text']) for r in ps}),'phrase_occurrences':len(rs),
            'unique_phrase_surfaces':len({r['text'] for r in rs}),
            'shared_phrase_surfaces_with_other_speakers':sum(speaker in r['speaker_ids'] and r['speaker_count']>=2 for r in frequencies['phrases']),
            'recording_keys':sorted({r['recording_key'] for r in ps})}
        assert got==exp,speaker
    assert len(summary['per_speaker'])==len({r['speaker_id'] for r in parents})
    return {'schema':'offline-stage5e-independent-inventory-audit-v1','status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'sources':sources,'artifacts':{n:binding(folder/(n+'.json')) for n in docs},
        'parent_count':len(parents),'exhaustively_verified_candidates':len(observed),'eojeol_candidates':len(words),
        'phrase_candidates':len(phrases),'excluded_candidates':len(excluded),'frequency_counts':summary['frequency_counts'],
        'complete_transcript_duplicate_groups':len(duplicate_groups),'all_candidate_token_ids_independently_encoded':True,
        'per_speaker_and_coverage_exact':True,'timing_frames_and_ctc_feasibility_remain_unknown':True,
        'producer_imports':0,'network_forward_calls':0,'optimizer_updates':0,'raw_media_reads':0,'NPZ_arrays_loaded':0,'checkpoint_tensor_loads':0}


if __name__=='__main__':
    output=HERE/'inventory_audit.json'
    assert not output.exists()
    result=audit()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','parent_count','exhaustively_verified_candidates','eojeol_candidates','phrase_candidates']}))
