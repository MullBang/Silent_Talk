"""Exercise the new loader against independently computed RGB/token references."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect():
    path=BASE/'data/loader.py'
    spec=importlib.util.spec_from_file_location('new_loader_under_test',path)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    policy_path=BASE/'controlled_pilot_policy.json';policy=read(policy_path)
    handoff_path=Path(policy['handoff']['path']);handoff=read(handoff_path)
    args=(handoff_path,sha(handoff_path),policy_path,sha(policy_path))
    rows={r['clip_id']:r for r in handoff['records']}
    datasets={s:loader.ControlledPilotDataset(*args,split=s) for s in ['train','dev']}
    assert len(datasets['train'])==24 and len(datasets['dev'])==6
    checked=[];valid={};batches=[];failures=[]
    def reject(name,call):
        try:call()
        except (loader.DataContractError,ValueError,KeyError,TypeError) as e:
            failures.append({'case':name,'rejected':True,'exception':type(e).__name__,'message':str(e)})
        else:raise AssertionError('Accepted invalid case: '+name)
    for split,data in datasets.items():
        purpose='optimizer' if split=='train' else 'validation'
        for ident in data.clip_ids:
            item=data.load_clip(ident,purpose=purpose);r=rows[ident]
            with np.load(r['npz_path'],allow_pickle=False) as z:
                rgb=z['frames_rgb'];expected=np.empty((3,len(rgb),96,96),dtype=np.float32)
                for channel,(mean,std) in enumerate(zip([.485,.456,.406],[.229,.224,.225])):
                    expected[channel]=(rgb[:,:,:,channel].astype(np.float32)/np.float32(255)-np.float32(mean))/np.float32(std)
                assert np.array_equal(item['input'],expected)
                assert np.array_equal(item['labels'],z['label'])
                assert item['length']==int(z['length'])
            assert item['input'].flags.c_contiguous and not item['input'].flags.writeable
            assert item['labels'].dtype==np.int64 and not item['labels'].flags.writeable
            assert item['split']==split and item['purpose']==purpose and item['clip_id']==ident
            for key in ['boundary_followup_flag','display_neighbor_intersections','exact_visual_boundary_certified']:
                assert item['audit'][key]==r[key],key
            assert item['audit']['source_training_ready'] is False and item['audit']['source_provisional'] is True
            assert item['audit']['approval_changed'] is False
            checked.append({'clip_id':ident,'split':split,'length':item['length'],'normalized_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),'boundary_followup_flag':r['boundary_followup_flag']})
            valid[ident]=item
        for i in range(0,len(data.clip_ids),2):
            items=[valid[c] for c in data.clip_ids[i:i+2]]
            b=loader.collate(items,purpose=purpose)
            assert b['lengths'].tolist()==[x['length'] for x in items]
            assert b['label_lengths'].tolist()==[len(x['labels']) for x in items]
            assert np.array_equal(b['labels'],np.concatenate([x['labels'] for x in items]))
            for j,x in enumerate(items):
                t=x['length'];assert np.array_equal(b['inputs'][j,:,:t],x['input'])
                for k in range(t,b['inputs'].shape[2]):assert np.array_equal(b['inputs'][j,:,k],x['input'][:,-1])
            loader.validate_batch(b)
            batches.append({'ids':b['clip_ids'],'true_frames':sum(b['lengths']).item(),'dense_frames':2*int(max(b['lengths']))})
    train_ids=datasets['train'].clip_ids;dev_id=datasets['dev'].clip_ids[0]
    reject('dev_for_optimizer',lambda:datasets['dev'].load_clip(dev_id,purpose='optimizer'))
    reject('train_loader_cannot_read_dev',lambda:datasets['train'].load_clip(dev_id))
    reject('unknown_ID',lambda:datasets['train'].load_clip('unknown'))
    reject('invalid_item_index_bool',lambda:datasets['train'].load_item(True))
    reject('mixed_train_dev_collate',lambda:loader.collate([valid[train_ids[0]],valid[dev_id]],purpose='optimizer'))
    reject('wrong_handoff_external_sha',lambda:loader.ControlledPilotDataset(handoff_path,'0'*64,policy_path,sha(policy_path)))
    reject('wrong_policy_external_sha',lambda:loader.ControlledPilotDataset(handoff_path,sha(handoff_path),policy_path,'0'*64))
    sample=valid[train_ids[0]]
    for case in ['blank_target','float64_pixels','NaN_pixels','zero_length','changed_identity','approval_promotion']:
        v=copy.deepcopy(sample)
        if case=='blank_target':v['labels'][0]=0
        if case=='float64_pixels':v['input']=v['input'].astype(np.float64)
        if case=='NaN_pixels':v['input'][0,0,0,0]=np.nan
        if case=='zero_length':v['length']=0
        if case=='changed_identity':v['clip_id']=dev_id
        if case=='approval_promotion':v['audit']['source_training_ready']=True
        reject(case,lambda v=v:loader.collate([v],purpose='optimizer'))
    pair=[valid[train_ids[0]],valid[train_ids[1]]]
    good=loader.collate(pair,purpose='optimizer')
    assert good['lengths'][0]!=good['lengths'][1]
    for case in ['zero_batch_padding','changed_batch_length','changed_batch_label','wrong_padding_mask','batch_dev_split']:
        b=copy.deepcopy(good)
        if case=='zero_batch_padding':
            j=int(np.argmin(b['lengths']));b['inputs'][j,:,b['lengths'][j]:]=0
        if case=='changed_batch_length':b['lengths'][0]-=1
        if case=='changed_batch_label':b['labels'][0]=68
        if case=='wrong_padding_mask':b['padding_mask'][:]=False
        if case=='batch_dev_split':b['splits'][0]='dev'
        reject(case,lambda b=b:loader.validate_batch(b))
    with tempfile.TemporaryDirectory(prefix='negative_policy_',dir=HERE) as temp:
        for case in ['full_training_authorized','service_change_authorized','preserve_provisional_flags']:
            q=copy.deepcopy(policy);q[case]=not q[case]
            p=Path(temp)/(case+'.json');p.write_text(json.dumps(q,ensure_ascii=False),encoding='utf-8')
            reject(case,lambda p=p:loader.ControlledPilotDataset(handoff_path,sha(handoff_path),p,sha(p)))
    assert 'torch' not in sys.modules
    return {'schema':'stage4-independent-loader-behavior-audit-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed','audit_code_sha256':sha(__file__),'loader_sha256':sha(path),'policy_sha256':sha(policy_path),
        'handoff_sha256':sha(handoff_path),'normalized_clips_checked':30,'collated_batches_checked':len(batches),
        'negative_cases_rejected':len(failures),'records':checked,'batches':batches,'negative_cases':failures,
        'neural_imports_or_execution':0,'optimizer_updates':0,'original_media_reads':0,
        'scope':'The new loader was imported as the unit under test. NumPy expected normalization and padding were computed separately directly from the frozen NPZ payloads. Original files were never mutated.'}

if __name__=='__main__':
    result=inspect()
    with (HERE/'loader_behavior_audit.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','batches','negative_cases']},ensure_ascii=False))
