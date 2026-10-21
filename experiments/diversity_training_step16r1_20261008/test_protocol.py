"""Check input roles, frame alignment and frozen-byte guards before GPU work."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('step16_protocol_test_subject', Path(__file__).with_name('protocol.py'))
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def fixture():
    pixels = np.arange(4*96*96*3,dtype=np.uint8).reshape(4,96,96,3)
    tokens = np.array([1,2],dtype=np.int64)
    indices = np.arange(4,dtype=np.int64)
    row = {'id':'sample','frames':4,'start_time':0.,'end_time':.16,'video_fps':25.}
    record = {'id':'sample','frames':4,'tokens':[1,2],'logical_record':row}
    embedded = {'id':'sample','record':row,'student_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
    return record,pixels,tokens,indices,embedded


def test_valid_input_requires_real_frame_mapping():
    subject.validate_student(*fixture())


@pytest.mark.parametrize('change',['pixels','shape','dtype','tokens','blank','indices','embedded','ctc'])
def test_input_corruption_is_rejected(change):
    record,pixels,tokens,indices,embedded=fixture()
    if change=='pixels': pixels[0,0,0,0]^=1
    elif change=='shape': pixels=pixels[:3]
    elif change=='dtype': pixels=pixels.astype(np.float32)
    elif change=='tokens': tokens[0]=3
    elif change=='blank': tokens[0]=0;record['tokens'][0]=0
    elif change=='indices': indices[-1]=4
    elif change=='embedded': embedded['id']='other'
    else: tokens=np.array([1,1,1],np.int64);record['tokens']=[1,1,1]
    with pytest.raises(ValueError): subject.validate_student(record,pixels,tokens,indices,embedded)


def mocked_data(tmp_path,monkeypatch):
    records={}
    for group in ('train','dev','probe'):
        record,pixels,tokens,indices,embedded=fixture()
        record.update(id=group,role='train' if group=='train' else group,speaker_id='s',source_stem=group,
                      generation='old',text='가',logical_record={**record['logical_record'],'id':group})
        embedded.update(id=group,record=record['logical_record'])
        path=tmp_path/(group+'.npz')
        np.savez(path,student_rgb=pixels,label_tokens=tokens,student_source_indices=indices,
                 metadata_json=json.dumps(embedded))
        record['input']=subject.bind(path)
        record['teacher_feature']=None
        if group=='train':
            feature=np.ones((4,512),dtype=np.float32)
            fp=tmp_path/'feature.npz'
            np.savez(fp,features=feature,student_source_indices=indices,
                     metadata_json=json.dumps({'id':group,'feature_values_sha256':hashlib.sha256(feature.tobytes()).hexdigest()}))
            record['teacher_feature']=subject.bind(fp)
        records[group]=record
    meta={'data':{},'design':{'cohorts':{'train72':['train'],'dev16':['dev']},
                             'evaluation':{'historical_probe14_ids':['probe']}},'records':records}
    monkeypatch.setattr(subject,'metadata',lambda:meta)
    opened=[]
    def verify(ref):
        opened.append(ref['path'])
        assert subject.bind(ref['path'])==ref
        return Path(ref['path'])
    monkeypatch.setattr(subject,'verify',verify)
    return opened


def test_training_loader_never_opens_development_or_probe(tmp_path,monkeypatch):
    opened=mocked_data(tmp_path,monkeypatch)
    loaded=subject.load_data('train')
    assert list(loaded['train'])==['train'] and loaded['dev']==loaded['probe']=={}
    assert {Path(p).name for p in opened}=={'train.npz','feature.npz'}
    assert loaded['counts']['training_feature_npz_decoded']==1


def test_evaluation_loader_never_opens_teacher_features(tmp_path,monkeypatch):
    opened=mocked_data(tmp_path,monkeypatch)
    loaded=subject.load_data('evaluation')
    assert {Path(p).name for p in opened}=={'train.npz','dev.npz','probe.npz'}
    assert loaded['counts']['training_feature_npz_decoded']==0
    assert all('teacher_features' not in c for group in ('train','dev','probe') for c in loaded[group].values())
    assert loaded['probe']['probe']['source_split']=='train'
    assert loaded['dev']['dev']['split']=='dev'


def test_all_mode_still_limits_features_to_training(tmp_path,monkeypatch):
    mocked_data(tmp_path,monkeypatch)
    loaded=subject.load_data('all')
    assert loaded['counts']['student_input_npz_decoded']==3
    assert 'teacher_features' in loaded['train']['train']
    assert 'teacher_features' not in loaded['dev']['dev']


def test_unknown_scope_rejected():
    with pytest.raises(ValueError): subject.load_data('test')


def test_nonlocal_or_modified_binding_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(subject,'ROOT',tmp_path)
    path=tmp_path/'bound.txt';path.write_text('before')
    ref=subject.bind(path)
    assert subject.verify(ref)==path.resolve()
    path.write_text('after')
    with pytest.raises(ValueError): subject.verify(ref)


def test_unknown_run_rejected():
    with pytest.raises(ValueError): subject.plan_path(44,'train72')
    with pytest.raises(ValueError): subject.plan_path(42,'train510')
