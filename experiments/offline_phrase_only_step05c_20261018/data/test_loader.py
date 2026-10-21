"""Stage5c source-preservation, exact-input and phrase-scope tests; no model calls."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from loader import PhraseDataset, DataContractError, array_sha256, collate_dense, file_sha256, validate_batch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
CONFIG = {}
AUDITS = []


class PhraseLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PhraseDataset(CONFIG['plan'], CONFIG['policy'], expected_policy_sha256=CONFIG['policy_sha'])
        cls.short = cls.dataset.load_clip(min(cls.dataset.clip_ids, key=lambda x: cls.dataset._rows[x]['T']))
        cls.long = cls.dataset.load_clip(max(cls.dataset.clip_ids, key=lambda x: cls.dataset._rows[x]['T']))

    def test_exact_twelve_phrase_inputs_and_independent_normalization(self):
        AUDITS.clear()
        for clip_id in self.dataset.clip_ids:
            item = self.dataset.load_clip(clip_id)
            with np.load(self.dataset._rows[clip_id]['npz_path'], allow_pickle=False) as packed:
                raw = packed['frames_rgb']
                expected = np.empty((3,len(raw),96,96), dtype=np.float32)
                for c, (mean,std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                    expected[c] = (raw[:,:,:,c].astype(np.float32)/np.float32(255)-np.float32(mean))/np.float32(std)
                np.testing.assert_array_equal(item['input'], expected)
                np.testing.assert_array_equal(item['labels'], packed['label'])
            self.assertEqual((item['split'],item['kind']), ('train','phrase'))
            self.assertFalse(item['input'].flags.writeable)
            self.assertFalse(item['labels'].flags.writeable)
            AUDITS.append(item['audit'])
        self.assertEqual(len(AUDITS),12)
        self.assertEqual(sum(a['source_T'] for a in AUDITS),481)
        self.assertEqual(sum(a['boundary_followup_flag'] for a in AUDITS),1)

    def test_dev_and_full_sentence_rejected_before_npz_load(self):
        rejected = [r['clip_id'] for r in self.dataset.handoff['records'] if r['clip_id'] not in self.dataset.clip_ids]
        self.assertEqual(len(rejected),18)
        with mock.patch('loader.np.load', side_effect=AssertionError('Excluded pixel read')):
            for clip_id in rejected:
                for purpose in ('optimizer','validation'):
                    with self.assertRaises(DataContractError):
                        self.dataset.load_clip(clip_id,purpose=purpose)

    def test_validation_is_same_train_phrase_pixels_and_distinct_purpose(self):
        item = self.dataset.load_clip(self.short['clip_id'],purpose='validation')
        np.testing.assert_array_equal(item['input'],self.short['input'])
        self.assertEqual(collate_dense([item],purpose='validation')['splits'],['train'])
        self.assertEqual(len(item['audit']['allowlisted_clip_ids_for_purpose']),12)
        with self.assertRaises(DataContractError): collate_dense([item],purpose='optimizer')

    def test_no_numpy_python_or_torch_cpu_rng_consumption(self):
        import torch
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        ds = PhraseDataset(CONFIG['plan'],CONFIG['policy'])
        collate_dense([ds.load_item(0),ds.load_item(1)])
        self.assertEqual(random.getstate(),py_state)
        later = np.random.get_state()
        self.assertEqual(later[0],np_state[0]); np.testing.assert_array_equal(later[1],np_state[1]); self.assertEqual(later[2:],np_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))

    def test_exact_dense_batch_matches_frozen_stage5_primitive(self):
        path = ROOT/'experiments/offline_comparison_step05_20261013/data/loader.py'
        spec = importlib.util.spec_from_file_location('stage5_prior_collation_only',path)
        prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
        # No old-policy constructor: only the unchanged mathematical primitive.
        items = [self.short,self.long]
        actual, reference = collate_dense(items), prior.collate(items)
        for key in ('inputs','lengths','labels','label_lengths','padding_mask'):
            np.testing.assert_array_equal(actual[key],reference[key])
        self.assertEqual(actual['clip_ids'],reference['clip_ids'])
        for t in range(self.short['length'],self.long['length']):
            np.testing.assert_array_equal(actual['inputs'][0,:,t],self.short['input'][:,-1])

    def _bad_policy(self, change):
        with tempfile.TemporaryDirectory(prefix='negative-policy-',dir=BASE) as temp:
            value=copy.deepcopy(self.dataset.policy); change(value)
            path=Path(temp)/'policy.json'; path.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaises(DataContractError): PhraseDataset(CONFIG['plan'],path)

    def test_policy_scope_and_approval_gates(self):
        for name in ('allow_provisional_cache_for_controlled_pilot','preserve_provisional_flags','full_training_authorized'):
            self._bad_policy(lambda p,k=name:p.update({k:False}))
        self._bad_policy(lambda p:p.update(service_change_authorized=True))
        self._bad_policy(lambda p:p.update(schema='lipnet-stage5-controlled-pilot-data-policy-v1'))
        self._bad_policy(lambda p:p['authorization'].update(scope='any_training'))

    def test_allowlists_exact_and_cannot_include_sentence_or_dev(self):
        excluded=next(x for x in self.dataset._rows if x not in self.dataset.clip_ids)
        for key in ('allowed_optimizer_clip_ids','allowed_validation_clip_ids'):
            self._bad_policy(lambda p,k=key:p[k].append(excluded))
            self._bad_policy(lambda p,k=key:p[k].pop())
            self._bad_policy(lambda p,k=key:p[k].append(p[k][0]))

    def test_metadata_hash_and_path_bindings(self):
        for key in ('training_plan','handoff','stage3_frozen_manifest'):
            self._bad_policy(lambda p,k=key:p[k].update(sha256='0'*64))
        self._bad_policy(lambda p:p['training_plan'].update(path=str(BASE/'other.json')))
        with self.assertRaises(DataContractError): PhraseDataset(CONFIG['plan'],CONFIG['policy'],expected_policy_sha256='0'*64)

    def test_unknown_index_and_purpose(self):
        for index in (-1,12,True):
            with self.assertRaises(DataContractError): self.dataset.load_item(index)
        with self.assertRaises(DataContractError): self.dataset.load_clip(self.short['clip_id'],purpose='test')
        with self.assertRaises(DataContractError): collate_dense([])

    def test_mutated_item_identity_kind_split_and_pixels_rejected(self):
        for key,value in [('clip_id',self.long['clip_id']),('kind','full_sentence'),('split','dev'),('length',24)]:
            item=dict(self.short); item[key]=value
            with self.assertRaises(DataContractError): collate_dense([item])
        item=dict(self.short); item['input']=item['input'].copy(); item['input'][0,0,0,0]+=.1
        with self.assertRaises(DataContractError): collate_dense([item])

    def test_batch_padding_mask_lengths_and_scope_rejected(self):
        batch=collate_dense([self.short,self.long]); batch['inputs'][0,:,self.short['length']:]=0
        with self.assertRaisesRegex(DataContractError,'Padding must repeat'): validate_batch(batch)
        for field in ('lengths','padding_mask'):
            batch=collate_dense([self.short,self.long])
            if field == 'lengths': batch[field][0]+=1
            else: batch[field][:]=False
            with self.assertRaises(DataContractError): validate_batch(batch)
        batch=collate_dense([self.short]); batch['splits'][0]='dev'
        with self.assertRaises(DataContractError): validate_batch(batch)
        batch=collate_dense([self.short]); batch['audit']['source_clips'][0]['source_kind']='full_sentence'
        with self.assertRaises(DataContractError): validate_batch(batch)

    def test_invalid_labels_nan_and_boundary_promotion_rejected(self):
        for labels in (np.asarray([0],dtype=np.int64),np.asarray([True]),np.asarray([1.0]),np.ones(self.short['length'],dtype=np.int64)):
            item=dict(self.short); item['labels']=labels; item['audit']=dict(item['audit']); item['audit']['labels_sha256']=array_sha256(labels)
            with self.assertRaises(DataContractError): collate_dense([item])
        item=dict(self.short); item['input']=item['input'].copy(); item['input'][0,0,0,0]=np.nan
        with self.assertRaises(DataContractError): collate_dense([item])
        item=copy.deepcopy(self.short); item['audit']['exact_visual_boundary_certified']=True
        with self.assertRaises(DataContractError): collate_dense([item])

    def _bad_npz(self,change,match,rehash=True):
        sid=self.short['clip_id']; original=self.dataset._rows[sid]
        with tempfile.TemporaryDirectory(prefix='negative-npz-',dir=BASE) as temp:
            with np.load(original['npz_path'],allow_pickle=False) as packed:
                arrays={key:packed[key].copy() for key in packed.files}
            meta=copy.deepcopy(self.dataset._metadata[sid]); change(arrays,meta)
            arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False))
            npz,jsonpath=Path(temp)/'invalid.npz',Path(temp)/'invalid.json'
            np.savez_compressed(npz,**arrays); jsonpath.write_text(json.dumps(meta),encoding='utf-8')
            clone=copy.copy(self.dataset); clone._rows=dict(clone._rows); clone._metadata=dict(clone._metadata)
            row=dict(original); row.update(npz_path=str(npz),npz_bytes=npz.stat().st_size,metadata_path=str(jsonpath),metadata_sha256=file_sha256(jsonpath))
            if rehash: row['npz_sha256']=file_sha256(npz)
            clone._rows[sid],clone._metadata[sid]=row,meta
            with self.assertRaisesRegex(DataContractError,match): clone.load_clip(sid)

    def test_reject_corrupt_npz_hash(self):
        self._bad_npz(lambda a,m:a.update(length=np.asarray(99,dtype=np.int64)),'NPZ SHA mismatch',rehash=False)

    def test_reject_audio_payload_with_matching_hash(self):
        self._bad_npz(lambda a,m:a.update(audio=np.zeros(5)),'Unexpected NPZ payload')

    def test_reject_zero_frame_even_when_rehashed(self):
        def change(a,m):
            a['frames_rgb'][0]=0; m['rgb_uint8_sha256']=array_sha256(a['frames_rgb'])
        self._bad_npz(change,'Raw zero/filler')

    def test_reject_npz_length_label_dtype_and_promotion(self):
        self._bad_npz(lambda a,m:a.update(length=np.asarray(99,dtype=np.int64)),'actual length')
        self._bad_npz(lambda a,m:a.update(label=a['label'].astype(np.int32)),'labels differ')
        self._bad_npz(lambda a,m:a.update(training_ready=np.asarray(True)),'source approval')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--policy-sha',required=True)
    parser.add_argument('--result',type=Path,default=BASE/'loader_test_results.json')
    args=parser.parse_args(); CONFIG.update(plan=args.plan,policy=args.policy,policy_sha=args.policy_sha)
    if not args.result.resolve().is_relative_to(BASE.resolve()) or args.result.exists(): raise ValueError('New output under data required')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhraseLoaderTests))
    report={'schema':'lipnet-stage5c-loader-test-report-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':[{'test':str(t),'traceback':s} for t,s in result.failures],
        'errors':[{'test':str(t),'traceback':s} for t,s in result.errors],
        'policy_sha256':args.policy_sha,'plan_sha256':file_sha256(args.plan),'loader_sha256':file_sha256(BASE/'loader.py'),
        'test_code_sha256':file_sha256(Path(__file__)),'loaded_clip_audits':AUDITS,
        'excluded_dev_or_full_sentence_npz_arrays_loaded':0,'new_original_media_reads':0,'new_model_calls':0,'new_training_updates':0}
    args.result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful(): raise SystemExit(1)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
