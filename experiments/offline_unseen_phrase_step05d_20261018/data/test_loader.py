"""Evaluation-only loader tests; no network, model, optimizer or source mutation."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from loader import Dataset, DataContractError, array_sha256, file_sha256

BASE=Path(__file__).resolve().parent
CONFIG={}
AUDITS=[]


class EvaluationLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset=Dataset(CONFIG['plan'],CONFIG['policy'],CONFIG['policy_sha'])
        cls.sample=cls.dataset.get(cls.dataset.clip_ids[0])

    def test_three_dev_phrases_exact_normalization_labels_and_flags(self):
        AUDITS.clear()
        for clip_id in self.dataset.clip_ids:
            item=self.dataset.get(clip_id)
            with np.load(self.dataset._rows[clip_id]['npz_path'],allow_pickle=False) as packed:
                raw=packed['frames_rgb']; expected=np.empty((3,len(raw),96,96),dtype=np.float32)
                for c,(mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                    expected[c]=(raw[:,:,:,c].astype(np.float32)/np.float32(255)-np.float32(mean))/np.float32(std)
                np.testing.assert_array_equal(item['input'],expected)
                np.testing.assert_array_equal(item['labels'],packed['label'])
            self.assertEqual((item['split'],item['kind'],item['purpose']),('dev','phrase','evaluation'))
            self.assertEqual(item['T'],item['length'])
            self.assertFalse(item['input'].flags.writeable); self.assertFalse(item['labels'].flags.writeable)
            self.assertFalse(item['audit']['source_training_ready']);self.assertTrue(item['audit']['source_provisional'])
            self.assertFalse(item['audit']['exact_visual_boundary_certified'])
            self.assertFalse(item['audit']['approval_changed'])
            AUDITS.append(item['audit'])
        self.assertEqual([a['source_T'] for a in AUDITS],[31,23,42])
        self.assertEqual(sum(a['source_T'] for a in AUDITS),96)
        self.assertEqual(sum(a['boundary_followup_flag'] for a in AUDITS),0)

    def test_train_and_full_sentence_rejected_before_npz_load(self):
        excluded=[sid for sid in self.dataset._rows if sid not in self.dataset.clip_ids]
        self.assertEqual(len(excluded),27)
        with mock.patch('loader.np.load',side_effect=AssertionError('Excluded pixels read')):
            for sid in excluded:
                with self.assertRaises(DataContractError): self.dataset.get(sid)
                with self.assertRaises(DataContractError): self.dataset.load_clip(sid)

    def test_optimizer_or_validation_alias_is_forbidden_for_every_clip(self):
        with mock.patch('loader.np.load',side_effect=AssertionError('Unauthorized pixels read')):
            for sid in self.dataset._rows:
                for purpose in ('optimizer','validation','train','test'):
                    with self.assertRaises(DataContractError): self.dataset.get(sid,purpose=purpose)
                    with self.assertRaises(DataContractError): self.dataset.load_clip(sid,purpose=purpose)

    def test_get_and_load_clip_are_same_array_interface(self):
        for sid in self.dataset.clip_ids:
            a,b=self.dataset.get(sid),self.dataset.load_clip(sid,purpose='evaluation')
            np.testing.assert_array_equal(a['input'],b['input']);np.testing.assert_array_equal(a['labels'],b['labels'])
            self.assertEqual(a['audit'],b['audit'])

    def test_no_python_numpy_or_torch_cpu_rng_consumption(self):
        import torch
        py,np_state,cpu=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        dataset=Dataset(CONFIG['plan'],CONFIG['policy'],CONFIG['policy_sha'])
        for sid in dataset.clip_ids: dataset.get(sid)
        self.assertEqual(random.getstate(),py)
        later=np.random.get_state();self.assertEqual(later[0],np_state[0]);np.testing.assert_array_equal(later[1],np_state[1]);self.assertEqual(later[2:],np_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),cpu))

    def _bad_policy(self,change):
        with tempfile.TemporaryDirectory(prefix='negative-policy-',dir=BASE) as temp:
            value=copy.deepcopy(self.dataset.policy);change(value)
            path=Path(temp)/'policy.json';path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(DataContractError): Dataset(CONFIG['plan'],path,file_sha256(path))

    def test_policy_cannot_allow_optimization_or_change_approval_scope(self):
        self._bad_policy(lambda p:p.update(optimizer_updates_authorized=True))
        self._bad_policy(lambda p:p['allowed_optimizer_clip_ids'].append(self.dataset.clip_ids[0]))
        self._bad_policy(lambda p:p.update(service_change_authorized=True))
        for key in ('allow_provisional_cache_for_controlled_evaluation','preserve_provisional_flags'):
            self._bad_policy(lambda p,k=key:p.update({k:False}))
        self._bad_policy(lambda p:p['authorization'].update(scope='any_evaluation'))
        self._bad_policy(lambda p:p.update(schema='lipnet-stage5c-controlled-phrase-only-policy-v1'))
        self._bad_policy(lambda p:p.update(mode='controlled_pilot'))

    def test_allowed_evaluation_ids_cannot_be_added_removed_or_duplicated(self):
        excluded=next(sid for sid in self.dataset._rows if sid not in self.dataset.clip_ids)
        self._bad_policy(lambda p:p['allowed_evaluation_clip_ids'].append(excluded))
        self._bad_policy(lambda p:p['allowed_evaluation_clip_ids'].pop())
        self._bad_policy(lambda p:p['allowed_evaluation_clip_ids'].append(p['allowed_evaluation_clip_ids'][0]))

    def test_hashes_and_requested_plan_path_are_bound(self):
        for key in ('evaluation_plan','handoff','stage3_frozen_manifest'):
            self._bad_policy(lambda p,k=key:p[k].update(sha256='0'*64))
        self._bad_policy(lambda p:p['evaluation_plan'].update(path=str(BASE/'other.json')))
        for digest in ('0'*64,None,''):
            with self.assertRaises(DataContractError): Dataset(CONFIG['plan'],CONFIG['policy'],digest)

    def test_bound_plan_cannot_change_order_text_or_lengths(self):
        mutations=[lambda p:p['items'].reverse(),lambda p:p['items'][0].update(text='changed'),
                   lambda p:p['items'][0].update(T=32),lambda p:p['items'][0].update(split='train'),
                   lambda p:p.update(schema='other')]
        for mutate in mutations:
            with tempfile.TemporaryDirectory(prefix='negative-plan-',dir=BASE) as temp:
                plan=copy.deepcopy(self.dataset.evaluation_plan);mutate(plan)
                planpath=Path(temp)/'plan.json';planpath.write_text(json.dumps(plan),encoding='utf-8')
                policy=copy.deepcopy(self.dataset.policy);policy['evaluation_plan']={'path':str(planpath),'sha256':file_sha256(planpath)}
                policypath=Path(temp)/'policy.json';policypath.write_text(json.dumps(policy),encoding='utf-8')
                with self.assertRaises(DataContractError): Dataset(planpath,policypath,file_sha256(policypath))

    def test_invalid_indices_and_unknown_id(self):
        for index in (-1,3,True):
            with self.assertRaises(DataContractError):self.dataset.load_item(index)
        with self.assertRaises(DataContractError):self.dataset.get('unknown')

    def _bad_npz(self,change,match,rehash=True):
        sid=self.sample['clip_id'];original=self.dataset._rows[sid]
        with tempfile.TemporaryDirectory(prefix='negative-npz-',dir=BASE) as temp:
            with np.load(original['npz_path'],allow_pickle=False) as packed: arrays={k:packed[k].copy() for k in packed.files}
            meta=copy.deepcopy(self.dataset._metadata[sid]);change(arrays,meta)
            arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False))
            npz,sidecar=Path(temp)/'invalid.npz',Path(temp)/'invalid.json'
            np.savez_compressed(npz,**arrays);sidecar.write_text(json.dumps(meta),encoding='utf-8')
            clone=copy.copy(self.dataset);clone._rows=dict(clone._rows);clone._metadata=dict(clone._metadata)
            row=dict(original);row.update(npz_path=str(npz),npz_bytes=npz.stat().st_size,metadata_path=str(sidecar),metadata_sha256=file_sha256(sidecar))
            if rehash:row['npz_sha256']=file_sha256(npz)
            clone._rows[sid],clone._metadata[sid]=row,meta
            with self.assertRaisesRegex(DataContractError,match):clone.get(sid)

    def test_npz_hash_mismatch_is_rejected(self):
        self._bad_npz(lambda a,m:a.update(length=np.asarray(99,dtype=np.int64)),'NPZ SHA mismatch',rehash=False)

    def test_audio_extra_payload_is_rejected_with_matching_file_hash(self):
        self._bad_npz(lambda a,m:a.update(audio=np.zeros(5)),'Unexpected NPZ payload')

    def test_zero_frame_is_rejected_with_matching_pixel_hash(self):
        def change(a,m):
            a['frames_rgb'][0]=0;m['rgb_uint8_sha256']=array_sha256(a['frames_rgb'])
        self._bad_npz(change,'Raw zero/filler')

    def test_npz_length_label_dtype_and_approval_promotions_rejected(self):
        self._bad_npz(lambda a,m:a.update(length=np.asarray(99,dtype=np.int64)),'actual length')
        self._bad_npz(lambda a,m:a.update(label=a['label'].astype(np.int32)),'labels differ')
        self._bad_npz(lambda a,m:a.update(training_ready=np.asarray(True)),'source approval')
        self._bad_npz(lambda a,m:a.update(provisional=np.asarray(False)),'provisional flag')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--policy-sha',required=True);parser.add_argument('--result',type=Path,default=BASE/'loader_test_results.json')
    args=parser.parse_args();CONFIG.update(plan=args.plan,policy=args.policy,policy_sha=args.policy_sha)
    if not args.result.resolve().is_relative_to(BASE.resolve()) or args.result.exists():raise ValueError('New output under data required')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationLoaderTests))
    report={'schema':'lipnet-stage5d-loader-test-report-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':[{'test':str(t),'traceback':s} for t,s in result.failures],
        'errors':[{'test':str(t),'traceback':s} for t,s in result.errors],
        'policy_sha256':args.policy_sha,'plan_sha256':file_sha256(args.plan),'loader_sha256':file_sha256(BASE/'loader.py'),
        'test_code_sha256':file_sha256(Path(__file__)),'loaded_clip_audits':AUDITS,
        'selected_distinct_clips':len(AUDITS),'selected_distinct_frames':sum(a['source_T'] for a in AUDITS),
        'excluded_train_or_full_sentence_npz_arrays_loaded':0,'new_original_media_reads':0,
        'new_model_calls':0,'new_training_updates':0,'source_mutations':0}
    args.result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
