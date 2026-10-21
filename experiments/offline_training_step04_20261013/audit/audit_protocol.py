"""Independent schedule, role, leakage and compute-budget arithmetic."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PROTO=HERE.parent/'protocol'
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect():
    p=load(PROTO/'comparison_protocol.json')
    bindings=load(PROTO/'data_bindings.json')
    source=load(ROOT/'experiments/offline_dataset_step03_20261013/results/cache_handoff.json')
    rows={r['clip_id']:r for r in bindings['records']}
    assert len(rows)==30
    for original in source['records']:
        current=rows[original['clip_id']]
        assert all(current[k]==v for k,v in original.items())
        assert current['label']==load(current['metadata_path'])['label']
    for path,expected in p['source_hashes'].items(): assert sha(path)==expected,path
    for ref in [p['data'],p['accounting'],p['evaluation']]+p['schedule_files']:
        assert sha(ref['path'])==ref['sha256']
    train=sorted({r['parent_id'] for r in rows.values() if r['split']=='train'})
    dev={r['parent_id'] for r in rows.values() if r['split']=='dev'}
    assert len(train)==12 and len(dev)==3 and not set(train)&dev
    accounting=load(PROTO/'schedule_accounting.json')
    planned={r['run_id']:r for r in accounting['runs']}
    totals_all=Counter()
    results=[]
    for seed in [42,43]:
        schedule=load(PROTO/f'schedule_seed{seed}.json')
        assert schedule['seed']==seed and len(schedule['updates'])==1260
        rng=random.Random(seed)
        order=[]
        for round_i in range(1,211):
            local=list(train); rng.shuffle(local)
            order.extend((round_i,local[k:k+2]) for k in range(0,12,2))
        accum={arm:Counter() for arm in ['A_sentence','B_curriculum']}
        exposure={arm:defaultdict(Counter) for arm in accum}
        phases={arm:defaultdict(Counter) for arm in accum}
        for i,(row,(rnd,ids)) in enumerate(zip(schedule['updates'],order),1):
            assert row['update']==i and row['round_index']==rnd and row['parent_ids']==ids
            assert row['lr']==(.001 if i<=840 else .0003 if i<=1050 else .0001)
            for arm,part in row['arms'].items():
                kind='phrase' if arm=='B_curriculum' and i<=420 else 'full_sentence'
                expected=[f'{parent}__{kind}' for parent in ids]
                assert part['kind']==kind and part['clip_ids']==expected
                clips=[rows[c] for c in expected]
                assert all(c['split']=='train' for c in clips)
                lengths=[c['T'] for c in clips]
                targets=[c['label_length'] for c in clips]
                assert part['lengths']==lengths and part['target_lengths']==targets
                actual=sum(lengths); dense=2*max(lengths); pad=dense-actual
                assert (part['actual_frames'],part['dense_processed_frames'],part['padding_frames'])==(actual,dense,pad)
                counters={'updates':1,'clip_presentations':2,'actual_frames':actual,'dense_processed_frames':dense,'padding_frames':pad,'target_tokens':sum(targets)}
                accum[arm].update(counters)
                phases[arm]['early_1_420' if i<=420 else 'late_421_1260'].update(counters)
                for parent in ids: exposure[arm][parent][kind]+=1
        for arm in accum:
            run_id=f'{arm}__seed{seed}'; declared=planned[run_id]
            assert dict(accum[arm])==declared['totals']
            assert {k:dict(v) for k,v in phases[arm].items()}==declared['phases']
            for parent in train:
                ex=exposure[arm][parent]
                expected={'phrase':70,'full_sentence':140} if arm=='B_curriculum' else {'full_sentence':210}
                assert dict(ex)==expected
                assert declared['per_parent_exposures'][parent]=={'phrase':ex['phrase'],'full_sentence':ex['full_sentence'],'total':210}
            totals_all.update(accum[arm])
            results.append({'run_id':run_id,'totals':dict(accum[arm]),'per_parent_exposures':dict(expected)})
    for key,value in totals_all.items():
        assert accounting['totals_all_four_runs'][key]==value,(key,value)
    evaluation=load(PROTO/'evaluation_plan.json')
    assert len(evaluation['items'])==30
    assert {e['clip_id'] for e in evaluation['items']}==set(rows)
    primary=[]
    for item in evaluation['items']:
        r=rows[item['clip_id']]
        for key in ['parent_id','split','kind','T','text']: assert item[key]==r[key]
        assert item['static_frame_index']==r['T']//2
        assert sorted(item['shuffled_frame_indices'])==list(range(r['T']))
        q=list(range(r['T'])); random.Random(item['shuffle_seed']).shuffle(q)
        assert q==item['shuffled_frame_indices']
        assert item['primary_at_final_normal_only']==(r['split']=='dev' and r['kind']=='full_sentence')
        if item['primary_at_final_normal_only']:primary.append(r)
    assert len(primary)==3
    assert sum(len(r['text'].strip()) for r in primary)==evaluation['primary_final_reference_characters']==74
    assert sum(len(r['text'].split()) for r in primary)==evaluation['primary_final_reference_words']==23
    assert evaluation['total_evaluations_four_runs']==30*3*3*4==1080
    assert evaluation['total_evaluation_frames']==sum(r['T'] for r in rows.values())*3*3*4==72828
    assert p['main_training_updates_executed_here']==0 and p['training_ready'] is False
    assert p['model']['audio_input'] is False and p['model']['teacher_model_or_features_used'] is False
    return {'schema':'stage4-independent-protocol-audit-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','audit_code_sha256':sha(__file__),'protocol_sha256':sha(PROTO/'comparison_protocol.json'),
        'source_hashes_checked':len(p['source_hashes']),'rows_checked':30,'schedule_rows_checked':2520,
        'arm_rows_checked':5040,'runs':results,'totals_all_four_runs':dict(totals_all),
        'evaluation_items':30,'planned_evaluation_forwards':1080,'planned_evaluation_frames':72828,
        'primary_dev_sentence_count':3,'main_training_executed_by_audit':0,'production_imports':0,
        'limitations':['Same optimizer update and parent exposure budgets; actual frames, padded frames, target tokens and wall time are not matched.',
                       'B changes both unit duration/content and stage order; does not isolate ordering alone.',
                       '3 internal dev parents from known speakers; no practical accuracy or unseen-speaker claim.',
                       'Source provisional, two boundary flags and held acoustic-OOV parent remain unchanged.']}

if __name__=='__main__':
    result=inspect()
    with (HERE/'protocol_audit.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(result,ensure_ascii=False))
