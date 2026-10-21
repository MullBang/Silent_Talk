"""Check shared-surface context counts directly from individual candidates."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from audit_sources import BASE, HERE, read, check, binding, textkey


def inspect():
    path=BASE/'inventory/shared_phrase_context_diversity.json';output=read(path)
    for r in output['sources'].values():check(r)
    parents={r['parent_id']:r for r in read(BASE/'data/eligible_sentences.json')['records']}
    groups=defaultdict(list)
    for r in read(BASE/'inventory/phrase_candidates.json')['records']:groups[r['text']].append(r)
    rows=[]
    for text,rs in groups.items():
        speakers={r['speaker_id'] for r in rs}
        if len(speakers)<2:continue
        selected={r['parent_id'] for r in rs};contexts=defaultdict(set)
        for sid in selected:contexts[textkey(parents[sid]['text'])].add(sid)
        context_rows=[]
        for k,ids in sorted(contexts.items()):
            context_rows.append({'full_transcript_key':k,'surface_variants':sorted({parents[s]['text'] for s in ids}),
                'parent_ids':sorted(ids),'speaker_ids':sorted({parents[s]['speaker_id'] for s in ids}),
                'recording_keys':sorted({parents[s]['recording_key'] for s in ids})})
        rows.append({'text':text,'n_eojeols':len(text.split()),'occurrences':len(rs),'parent_count':len(selected),
            'recording_count':len({r['recording_key'] for r in rs}),'speaker_count':len(speakers),
            'distinct_full_parent_transcript_count':len(contexts),'all_occurrences_share_one_full_parent_transcript':len(contexts)==1,
            'contexts':context_rows})
    rows.sort(key=lambda r:(-r['speaker_count'],-r['distinct_full_parent_transcript_count'],-r['recording_count'],r['text']))
    assert rows==output['records']
    varied=[r for r in rows if r['distinct_full_parent_transcript_count']>=2]
    same=[r for r in rows if r['distinct_full_parent_transcript_count']==1]
    assert len(rows)==output['shared_phrase_surface_count']==88
    assert len(varied)==output['shared_phrase_surfaces_with_at_least_two_distinct_parent_transcripts']==65
    assert len(same)==output['shared_phrase_surfaces_with_one_repeated_full_parent_transcript']==23
    assert output['descriptive_representatives']==varied[:8]+same[:2]
    for k in ['new_media_reads','new_model_calls','new_training_updates','new_boundary_approvals']:assert output[k]==0
    return {'schema':'offline-stage5e-independent-shared-context-audit-v1','status':'passed',
        'created_utc':datetime.now(timezone.utc).isoformat(),'artifact':binding(path),
        'independent_inputs':[binding(BASE/'data/eligible_sentences.json'),binding(BASE/'inventory/phrase_candidates.json')],
        'all88_context_groups_and10_representatives_exact':True,'shared_surfaces':88,'different_parent_texts':65,'one_parent_text':23,
        'producer_imports':0,'new_model_calls':0,'new_training_updates':0,'raw_media_reads':0,'NPZ_arrays_loaded':0,'checkpoint_tensor_loads':0}


if __name__=='__main__':
    path=HERE/'shared_context_audit.json';assert not path.exists()
    result=inspect();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','shared_surfaces','different_parent_texts','one_parent_text']}))
