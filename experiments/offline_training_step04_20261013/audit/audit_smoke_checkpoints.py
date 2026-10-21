"""Independently inspect saved smoke tensors and ledgers; no model imports/forward."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=HERE.parents[2]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unpack(x):
    if isinstance(x,dict):
        if x.get('__step03_numpy_array_v1__') is True:
            assert set(x)=={'__step03_numpy_array_v1__','dtype','shape','bytes'}
            dtype=np.dtype(x['dtype']); assert not dtype.hasobject
            b=x['bytes']; assert b.dtype==torch.uint8 and b.ndim==1
            assert b.numel()==math.prod(x['shape'])*dtype.itemsize
            return np.frombuffer(b.numpy().tobytes(),dtype=dtype).reshape(x['shape']).copy()
        return {k:unpack(v) for k,v in x.items()}
    if isinstance(x,list):return [unpack(v) for v in x]
    if isinstance(x,tuple):return tuple(unpack(v) for v in x)
    return x

def fingerprint(x):
    """Implement declared state serialization independently, without importing runner."""
    h=hashlib.sha256()
    def put(b):h.update(struct.pack('<Q',len(b)));h.update(b)
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous()
            for b in [b'torch',str(a.dtype).encode(),str(tuple(a.shape)).encode(),a.reshape(-1).view(torch.uint8).numpy().tobytes()]:put(b)
        elif isinstance(v,np.ndarray):
            assert not v.dtype.hasobject
            for b in [b'numpy',v.dtype.str.encode(),str(v.shape).encode(),np.ascontiguousarray(v).tobytes()]:put(b)
        elif isinstance(v,dict):
            put(b'dict')
            for k in sorted(v,key=lambda q:(type(q).__name__,repr(q))):visit(k);visit(v[k])
            put(b'end_dict')
        elif isinstance(v,(tuple,list)):
            put(type(v).__name__.encode())
            for q in v:visit(q)
            put(b'end_sequence')
        elif v is None:put(b'none')
        elif isinstance(v,bool):put(b'bool');put(str(v).encode())
        elif isinstance(v,int):put(b'int');put(str(v).encode())
        elif isinstance(v,float):assert math.isfinite(v);put(b'float');put(struct.pack('<d',v))
        elif isinstance(v,str):put(b'str');put(v.encode())
        else:raise AssertionError(type(v))
    visit(x);return h.hexdigest()

def state_hashes(state):
    return {**{k:fingerprint(state[k]) for k in ['model','optimizer','rng']},'complete':fingerprint(state)}

def compare(left,right,tolerances):
    values=[]
    def visit(a,b,path):
        assert type(a)==type(b),(path,type(a),type(b))
        if isinstance(a,torch.Tensor):
            assert a.shape==b.shape and a.dtype==b.dtype,path
            exact=torch.equal(a,b)
            if a.is_floating_point() and not path.endswith('/step'):
                key='optimizer_exp_avg_sq' if path.endswith('/exp_avg_sq') else 'optimizer_exp_avg' if path.endswith('/exp_avg') else 'model'
                d=tolerances[key]
                err=(a-b).abs(); ok=bool(torch.all(err<=d['atol']+d['rtol']*b.abs()))
                assert ok,path
                values.append({'path':path,'exact':exact,'max_abs':float(err.max()),'bound':key})
            else:assert exact,path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:visit(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b),path
            for i,(q,r) in enumerate(zip(a,b)):visit(q,r,path+'/'+str(i))
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
        else:assert a==b,path
    visit(left['model'],right['model'],'model')
    visit(left['optimizer'],right['optimizer'],'optimizer')
    assert fingerprint(left['rng'])==fingerprint(right['rng'])
    return {'tensor_count':len(values),'all_exact':all(v['exact'] for v in values),
            'max_abs_difference':max(v['max_abs'] for v in values),'rng_exact':True}

def inspect():
    plan=read(BASE/'smoke_plan.json'); policy=read(BASE/'controlled_pilot_policy.json')
    summary=read(BASE/'results/smoke_summary.json')
    assert summary['status']=='passed'
    assert plan['updates']==6 and plan['planned_optimizer_updates']==48
    assert plan['planned_main_comparison_updates']==0
    for b in plan['bindings']:assert sha(b['path'])==b['sha256']
    sources=read(policy['handoff']['path']);rows={r['clip_id']:r for r in sources['records']}
    assert sha(policy['handoff']['path'])==policy['handoff']['sha256']
    assert policy['training_plan']['sha256']==sha(BASE/'smoke_plan.json')
    totals=Counter();checked=[];comparisons=[];init={};all_files=[]
    for run in summary['runs']:
        seed,arm=run['seed'],run['arm'];schedule=plan['schedules'][arm]
        states={}
        for segment,start,stop in [('continuous',0,6),('split',0,2),('resumed',2,6)]:
            record=run[segment];ident=record['identity']
            assert ident=={'plan_sha256':sha(BASE/'smoke_plan.json'),'policy_sha256':sha(BASE/'controlled_pilot_policy.json'),'seed':seed,'arm':arm,'runtime':plan['runtime']}
            folder=BASE/'smoke'/f'seed{seed}'/arm/segment
            assert read(folder/'segment_result.json')==record
            log=[json.loads(s) for s in (folder/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            assert log==record['executed_updates'] and len(log)==stop-start
            assert record['completed_updates']==stop
            assert record['dev_neural_inputs']==record['teacher_calls']==record['audio_inputs']==0
            if seed not in init:init[seed]=record['initial_state_hashes']
            assert init[seed]==record['initial_state_hashes']
            for i,row in enumerate(log,start+1):
                task=schedule[i-1]
                assert row['update']==i and row['kind']==task['kind'] and row['clip_ids']==task['clip_ids']
                assert row['learning_rate']==task['lr'] and row['optimizer_updated'] is True
                lengths=[rows[c]['T'] for c in row['clip_ids']]
                assert all(rows[c]['split']=='train' for c in row['clip_ids'])
                assert row['lengths']==lengths and row['true_frames']==sum(lengths)
                assert row['dense_frames']==2*max(lengths) and row['padding_frames']==2*max(lengths)-sum(lengths)
                assert row['nonzero_gradient_tensors']==row['parameter_tensors']>0
                assert 0 < row['gradient_norm_after_clip']<=1.0001
                assert all(math.isfinite(row[k]) for k in ['loss','gradient_norm_before_clip','gradient_norm_after_clip','seconds'])
                assert row['cuda_peak_allocated_bytes']>0 and row['cuda_peak_reserved_bytes']>=row['cuda_peak_allocated_bytes']
                totals.update({'optimizer_updates':1,'actual_frames':sum(lengths),'dense_frames':2*max(lengths),'padding_frames':2*max(lengths)-sum(lengths)})
            for ref in record['checkpoints']:
                assert sha(ref['path'])==ref['sha256'] and Path(ref['path']).stat().st_size==ref['bytes']
                c=unpack(torch.load(ref['path'],map_location='cpu',weights_only=True))
                done=c['completed_updates']
                assert c['identity']==ident and c['initial_state_hashes']==init[seed]
                assert c['state_hashes']==state_hashes(c['state'])==c['history'][-1]['state_hashes']
                assert c['schema']=='offline-stage4-complete-checkpoint-v1'
                assert len(c['history'])==done and done in [2,6]
                assert c['next_schedule_row']==(schedule[done] if done<6 else None)
                assert set(c['state'])=={'model','optimizer','rng'}
                opt=c['state']['optimizer'];assert len(opt['param_groups'])==1
                group=opt['param_groups'][0]
                assert group['lr']==schedule[done-1]['lr'] and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
                assert set(group['params'])==set(opt['state']) and len(opt['state'])==len(c['state']['model'])
                for value in c['state']['model'].values():assert value.dtype==torch.float32 and torch.isfinite(value).all()
                for value in opt['state'].values():
                    assert float(value['step'])==done
                    assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                    assert (value['exp_avg_sq']>=0).all()
                for i,row in enumerate(c['history'],1):
                    task=schedule[i-1]
                    assert row['update']==i and row['clip_ids']==task['clip_ids'] and row['kind']==task['kind'] and row['learning_rate']==task['lr']
                if done==stop:states[segment]=c
                checked.append({'path':ref['path'],'sha256':ref['sha256'],'completed_updates':done,'state_hashes':c['state_hashes'],'Adam_parameter_states':len(opt['state']),'RNG_keys':list(c['state']['rng'])})
            if segment=='resumed':
                assert record['restore']['exact'] is True
                assert record['restore']['restored_hashes']==states['split']['state_hashes']
                assert record['restore']['next_schedule_row']==schedule[2]
                assert states['resumed']['history'][:2]==states['split']['history']
        assert states['continuous']['history'][:2]==states['split']['history'][:2] or all(
            a['state_hashes']==b['state_hashes'] for a,b in zip(states['continuous']['history'][:2],states['split']['history'][:2]))
        diff=compare(states['continuous']['state'],states['resumed']['state'],plan['tolerances_fixed_before_execution'])
        for a,b in zip(states['continuous']['history'],states['resumed']['history']):
            for key,tolkey in [('loss','loss'),('gradient_norm_before_clip','gradient_norm')]:
                tolerance=plan['tolerances_fixed_before_execution'][tolkey]
                assert abs(a[key]-b[key])<=tolerance['atol']+tolerance['rtol']*abs(b[key])
        comparisons.append({'seed':seed,'arm':arm,**diff})
    assert len(checked)==16 and totals['optimizer_updates']==48==summary['actual_smoke_optimizer_updates']
    assert totals['actual_frames']==summary['actual_true_training_frames_including_replays']
    assert totals['dense_frames']==summary['actual_dense_training_frames_including_padding_and_replays']
    assert summary['main_comparison_updates']==summary['dev_neural_inputs']==summary['audio_inputs']==summary['teacher_calls']==summary['service_changes']==0
    frozen=read(ROOT/'experiments/offline_dataset_step03_20261013/results/completion_manifest.json')
    assert all(sha(r['path'])==r['sha256'] for r in frozen['artifacts'])
    return {'schema':'stage4-independent-saved-smoke-audit-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','audit_code_sha256':sha(__file__),'plan_sha256':sha(BASE/'smoke_plan.json'),
        'summary_sha256':sha(BASE/'results/smoke_summary.json'),'checkpoints':checked,'comparisons':comparisons,
        'actual_execution_totals':dict(totals),'paired_initial_state_hashes':init,'stage3_frozen_unchanged':len(frozen['artifacts']),
        'audit_neural_forwards':0,'audit_optimizer_updates':0,'production_imports':0,
        'scope':'Restricted CPU tensor deserialization, independent state digests/finite Adam counters and full-state replay comparisons, schedule/log/physical update accounting. No model execution by this auditor.',
        'limitations':['These smoke weights are recovery-test artifacts, not recognition-performance or deployment evidence.','Source provisional and qualitative-review scope remain unchanged.']}

if __name__=='__main__':
    torch.set_num_threads(2)
    report=inspect()
    with (HERE/'smoke_checkpoint_audit.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checkpoints','paired_initial_state_hashes']},ensure_ascii=False))
