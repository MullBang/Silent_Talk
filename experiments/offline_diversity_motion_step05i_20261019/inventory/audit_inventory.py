"""Read-only source inventory; no neural modules, media extraction or training."""
import hashlib
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def bind(p):
    p = Path(p).resolve()
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def key(t):
    return ''.join(t.split())

def stats(rs):
    words=[w for r in rs for w in r['text'].split()]
    syllables=[c for r in rs for c in r['text'] if '\uac00' <= c <= '\ud7a3']
    return dict(clips=len(rs), parents=len({r['parent_id'] for r in rs}),
        recordings=len({r['recording_group'] for r in rs}),speakers=len({r['speaker_id'] for r in rs}),
        unique_texts=len({r['text'] for r in rs}),unique_no_space_texts=len({key(r['text']) for r in rs}),
        frames=sum(r['T'] for r in rs),eojeol_occurrences=len(words),unique_eojeols=len(set(words)),
        syllable_occurrences=len(syllables),unique_syllables=len(set(syllables)),
        speaker_text_counts={s:len({r['text'] for r in rs if r['speaker_id']==s}) for s in sorted({r['speaker_id'] for r in rs})})

def main():
    p5h=ROOT/'experiments/offline_confirmed_learning_step05h_20261019/data_plan.json'
    p3=ROOT/'experiments/offline_dataset_step03_20261013/results/cache_handoff.json'
    pp=ROOT/'experiments/offline_phrase_design_step05e_20261018/data/protected_scope.json'
    pr=ROOT/'experiments/offline_phrase_design_step05e_20261018/design/role_proposal.json'
    pq=ROOT/'experiments/offline_phrase_design_step05e_20261018/design/review_queue.json'
    phrases=load(p5h)['records']
    sentences=[r for r in load(p3)['records'] if r['kind']=='full_sentence']
    assert len(phrases)==17 and len(sentences)==15
    protected=load(pp); reserved=load(pr)['reserved_recordings']
    held=[r for r in load(pq)['records'] if r['review_id'] in ('Q03','Q04','Q05','Q06','Q07','Q08')]
    records=[]
    for source in phrases+sentences:
        r=dict(source);m=load(r['metadata_path'])
        for pathk,shak in [('npz_path','npz_sha256'),('metadata_path','metadata_sha256')]:
            assert bind(r[pathk])['sha256']==r[shak]
        assert m['text']==r['text'] and m['T']==r['T'] and m['parent_id']==r['parent_id']
        assert m['ctc_feasible'] and r['ctc_min_frames']<=r['T']
        assert m['training_ready'] is False and m['provisional'] is True
        assert m['audio_included'] is False
        if r['kind']=='full_sentence':
            e=m['parent_audio_review_provenance']
            assert e['parent_transcript_confirmed'] is True and e['parent_audio_boundary_confirmed'] is True
            assert e['canonical_parent_text']==r['text']
            eb=bind(e['evidence_path'])
            r['parent_audio_review_provenance']=e
            r['parent_audio_evidence_binding']=eb
            r['full_sentence_visual_boundary_approved']=False
            r['approval_scope_note']='Full parent transcript/audio boundary confirmed independently. Phrase visual no-obvious-cut answer does not transfer to full sentence. Original D-grid/PTS and sampled spatial QC are technical checks, not precise articulation certification.'
        r['source_kind']='existing_stage3_sentence' if r['kind']=='full_sentence' else 'existing_stage5h_phrase'
        r['source_members']=m['source_members']
        r['recording_key']=re.sub(r'_[A-Z]_(\d+)$',r'_\1',r['recording_group'])
        r['cohort']=('train_sentence12' if r['split']=='train' else 'dev_sentence3') if r['kind']=='full_sentence' else ('train_phrase14' if r['split']=='train' else 'dev_phrase3')
        records.append(r)
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='dev']
    train14=[r for r in train if r['kind']=='phrase'];train12=[r for r in train if r['kind']=='full_sentence']
    assert len(train)==26 and len(dev)==6 and len(train14)==14 and len(train12)==12
    failures=[]
    heldparents={r['parent_id'] for r in held}
    heldtexts={key(r['text']) for r in held}
    reservedkeys={r['recording_key'] for r in reserved}
    reservedstems={s for r in reserved for s in r['blocked_source_stems']}
    reservedvideo={s for r in reserved for s in r['blocked_video_sha256']}
    reservedlabel={s for r in reserved for s in r['blocked_label_sha256']}
    tests={}
    for attr in ['parent_id','recording_group','recording_key']:
        tests['train_dev_'+attr+'_intersection']=sorted({r[attr] for r in train}&{r[attr] for r in dev})
    for kind in ['video','label']:
        tests['train_dev_'+kind+'_sha_intersection']=sorted({r['source_members'][kind]['sha256'] for r in train}&{r['source_members'][kind]['sha256'] for r in dev})
    tests['train_dev_exact_no_space_text_intersection']=sorted({key(r['text']) for r in train}&{key(r['text']) for r in dev})
    tests['held_parent_intersection']=sorted({r['parent_id'] for r in records}&heldparents)
    tests['held_exact_text_intersection']=sorted({key(r['text']) for r in train}&heldtexts)
    tests['protected_parent_intersection']=sorted({r['parent_id'] for r in train}&set(protected['protected_parent_ids']))
    tests['protected_recording_intersection']=sorted({r['recording_key'] for r in train}&set(protected['protected_recording_keys']))
    tests['protected_stem_intersection']=sorted({r['recording_group'] for r in train}&set(protected['blocked_source_stem']))
    tests['protected_full_text_intersection']=sorted({key(r['text']) for r in train}&set(protected['protected_full_text_keys']))
    tests['protected_phrase_substring_hits']=[dict(clip_id=r['clip_id'],protected_key=k) for r in train for k in protected['protected_phrase_text_keys'] if k in key(r['text']) or key(r['text']) in k]
    tests['reserved_recording_intersection']=sorted({r['recording_key'] for r in train}&reservedkeys)
    tests['reserved_stem_intersection']=sorted({r['recording_group'] for r in train}&reservedstems)
    for kind,values in [('video',reservedvideo),('label',reservedlabel)]:
        tests['protected_'+kind+'_sha_intersection']=sorted({r['source_members'][kind]['sha256'] for r in train}&set(protected['blocked_'+kind+'_sha256']))
        tests['reserved_'+kind+'_sha_intersection']=sorted({r['source_members'][kind]['sha256'] for r in train}&values)
    failures=[k for k,v in tests.items() if v]
    assert not failures, failures
    cohorts={c:[r['clip_id'] for r in records if r['cohort']==c] for c in ['train_phrase14','train_sentence12','dev_phrase3','dev_sentence3']}
    out=dict(schema='offline-stage5i-confirmed-expansion-inventory-v1',status='passed',
        source_bindings=[bind(p) for p in [p5h,p3,pp,pr,pq]],records=records,cohorts=cohorts,
        comparison_arms=dict(A14=cohorts['train_phrase14'],B26=cohorts['train_phrase14']+cohorts['train_sentence12'],C26_static=cohorts['train_phrase14']+cohorts['train_sentence12']),
        held_records=[{k:r[k] for k in ['review_id','parent_id','recording_key','text']} for r in held],
        protection_checks=tests,statistics=dict(A14=stats(train14),added_sentence12=stats(train12),B26=stats(train),dev_phrase3=stats([r for r in dev if r['kind']=='phrase']),dev_sentence3=stats([r for r in dev if r['kind']=='full_sentence'])),
        source_training_ready=False,source_approval_changed=False,new_raw_reads=0,new_media_or_cache_creation=0,new_downloads=0,neural_calls=0,
        caveats=['26 clips remain 14 parents, 13 recordings and 8 speakers; sentence/phrase clips overlap within each training parent and are not independent examples.',
            'Full sentence transcriptions and audio bounds are confirmed; full sentence visual-boundary human approval is not inferred from phrase answers.',
            'The known C002_A_012_035 display-neighbor warning survives on phrase and full sentence. Prior actual PTS audit found no sampled neighbor overlap; no new human certification.',
            'Q03-Q08 held parent IDs are excluded entirely. Their recordings may contain previously confirmed different train parents; this does not confer approval to held parents.',
            'Existing dev3 phrase and corresponding sentence3 remain same 3 internal reused parents/speakers, not six independent samples or a final test.',
            'Historical automatic/ASR-approved sentence pools are not relabeled human-confirmed by this audit; no such additions selected.',
            'Full sentence expansion changes context/length/frame exposure as well as vocabulary. A14-B26 is not a pure expression-count causal test.',
            'Training static control on same B26 schedule is a baseline sensitivity experiment; normal-vs-static/shuffle evaluation alone cannot certify articulation understanding.'])
    target=OUT/'verified_expansion.json'
    assert not target.exists()
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=out['status'],statistics=out['statistics'],records=len(records),path=str(target)),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
