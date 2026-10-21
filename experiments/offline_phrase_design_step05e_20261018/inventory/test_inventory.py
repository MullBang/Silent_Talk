"""Small exact transcript fixtures test span indices, exclusions and dependence counts."""
from __future__ import annotations

import copy
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata
import unittest

import build_inventory as inventory

BASE=Path(__file__).resolve().parent
TOK=inventory.load_tokenizer()


def record(ident,speaker,recording,text,stem=None):
    return {'id':ident,'parent_id':ident,'speaker_id':speaker,'recording_key':recording,
        'source_stem':stem or recording+'_angleF','text':text,'frames':999,
        'token_ids':TOK.encode_text(text),'source_approval_status':{'parent_transcript_confirmed':False,'training_ready':False}}


def fixture():
    return [record('r1','s1','rec1','나는 오늘 나는 오늘 기뻐'),
        record('r2','s1','rec2','나는 내일 정말 기뻐'),
        record('r3','s2','rec3','나는 오늘 정말 기뻐'),
        record('r4','s2','rec3','나는 오늘 정말 기뻐','rec3_angleR')]


class InventoryTests(unittest.TestCase):
    def test_all_contiguous_spans_and_half_open_indices(self):
        rows=[record('r1','s1','rec1','나는 오늘 정말 기뻐')]
        candidates,excluded=inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)
        self.assertEqual(len(candidates),10);self.assertEqual(excluded,[])
        self.assertEqual({n:sum(c['n_eojeols']==n for c in candidates) for n in (1,2,3,4)},{1:4,2:3,3:2,4:1})
        for item in candidates:
            self.assertEqual(item['text'],' '.join(rows[0]['text'].split()[item['word_start']:item['word_end']]))
            self.assertEqual(item['n_eojeols'],item['word_end']-item['word_start'])
            self.assertEqual(item['candidate_id'],f"r1__eojeol_{item['word_start']:03d}_{item['word_end']:03d}")

    def test_repeated_occurrences_do_not_equal_independent_sentences(self):
        candidates,_=inventory.make_candidates(fixture(),{'protected_phrase_text_keys':[]},TOK)
        table=inventory.frequencies([r for r in candidates if r['n_eojeols']>1])
        row=next(r for r in table if r['text']=='나는 오늘')
        self.assertEqual((row['occurrences'],row['sentence_count'],row['recording_count'],row['speaker_count']),(4,3,2,2))
        self.assertEqual(row['speaker_ids'],['s1','s2'])
        self.assertEqual(row['per_speaker'][0]['occurrences'],2)

    def test_multiple_angles_do_not_increase_recording_count(self):
        results=inventory.build(fixture(),{'protected_phrase_text_keys':[]},TOK)
        self.assertEqual(results['summary']['all_angle_recording_count'],3)
        self.assertEqual(results['summary']['source_stem_count'],4)
        row=next(r for r in results['frequencies']['phrases'] if r['text']=='정말 기뻐')
        self.assertEqual((row['occurrences'],row['recording_count']),(3,2))

    def test_current_dev_phrase_exact_and_substrings_are_both_excluded(self):
        rows=[record('r1','s1','rec1','너무 많이 막혀서 정말 너무 많이 막혀')]
        candidates,excluded=inventory.make_candidates(rows,{'protected_phrase_text_keys':['너무많이막혀']},TOK)
        self.assertEqual(len(excluded),4)
        self.assertEqual(sum(r['exact_protected_text_match'] for r in excluded),1)
        self.assertTrue(any(r['text']=='너무 많이 막혀서' for r in excluded))
        self.assertFalse(any('너무많이막혀' in inventory.text_key(r['text']) for r in candidates))

    def test_text_key_normalizes_nfc_and_all_whitespace(self):
        self.assertEqual(inventory.text_key(unicodedata.normalize('NFD','너무')+'\t 많이\n막혀'),'너무많이막혀')
        self.assertNotEqual(inventory.text_key('너무 많이 막혀요'),'너무많이막혀')

    def test_candidate_also_protects_whole_heldout_text_embedded_in_longer_parent(self):
        rows=[record('r1','s1','rec1','나는 오늘 정말 기뻐')]
        scope={'protected_phrase_text_keys':[],'protected_full_text_keys':['나는오늘'],
               'protected_candidate_text_keys':['나는오늘']}
        candidates,excluded=inventory.make_candidates(rows,scope,TOK)
        self.assertEqual(len(excluded),1);self.assertEqual(excluded[0]['text'],'나는 오늘')
        self.assertFalse(any(r['text']=='나는 오늘' for r in candidates))
        longer=next(r for r in candidates if r['text']=='나는 오늘 정말')
        self.assertEqual(longer['contained_protected_full_text_keys_without_exact_match'],['나는오늘'])
        scope['protected_candidate_text_keys']=[]
        with self.assertRaisesRegex(ValueError,'omit protected full'):inventory.make_candidates(rows,scope,TOK)

    def test_protected_parent_recording_and_full_text_fail_closed(self):
        rows=fixture()
        for extra in [{'protected_parent_ids':['r1']},{'protected_recording_keys':['rec1']},
                      {'protected_full_text_keys':['나는오늘나는오늘기뻐']}]:
            with self.assertRaisesRegex(ValueError,'Protected parent'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[],**extra},TOK)

    def test_parent_text_cannot_silently_change_normalization(self):
        rows=fixture();rows[0]['text']='나는  오늘'
        with self.assertRaisesRegex(ValueError,'Canonical transcript'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)
        rows=fixture();rows[0]['text']='나는 오늘.';rows[0]['token_ids']=TOK.encode_text(rows[0]['text'])
        with self.assertRaisesRegex(ValueError,'Canonical transcript'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)

    def test_parent_token_ids_cannot_drift_or_use_bool(self):
        for changed in ([1],[True]):
            rows=fixture();rows[0]['token_ids']=changed
            with self.assertRaisesRegex(ValueError,'token IDs'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)

    def test_duplicate_source_or_parent_id_is_rejected(self):
        rows=fixture();rows[1]['parent_id']='r1'
        with self.assertRaisesRegex(ValueError,'Duplicate'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)
        rows=fixture();rows[1]['id']='r1'
        with self.assertRaisesRegex(ValueError,'Duplicate'):inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)

    def test_candidate_boundaries_and_ctc_feasibility_remain_unknown(self):
        candidates,_=inventory.make_candidates(fixture(),{'protected_phrase_text_keys':[]},TOK)
        for item in candidates:
            self.assertEqual(item['parent_reference_frames'],999)
            for key in ('start_seconds','end_seconds','frame_count','ctc_feasible'):self.assertIsNone(item[key])
            self.assertFalse(item['training_ready']);self.assertTrue(item['metadata_only'])
            self.assertFalse(item['candidate_boundary_confirmed']);self.assertFalse(item['new_human_approval_added'])

    def test_token_ids_use_same_69_ctc_and_exact_syllable_length(self):
        rows=[record('r1','s1','rec1','못 꿀 것 같아')]
        candidates,_=inventory.make_candidates(rows,{'protected_phrase_text_keys':[]},TOK)
        whole=next(r for r in candidates if r['n_eojeols']==4)
        # Initial hand-written fixture used two wrong positions: initial ㅁ is7,
        # final ㅌ is65 in the frozen position-specific69 dictionary.
        self.assertEqual(whole['token_ids'],[7,28,59,68,2,33,48,68,1,24,59,68,1,20,65,12,20])
        self.assertEqual(whole['syllable_length'],5)
        self.assertEqual(whole['ctc_min_frames'],17)
        self.assertNotIn(0,whole['token_ids'])

    def test_empty_candidate_subgroup_has_defined_zero_coverage(self):
        result=inventory.coverage([],TOK)
        self.assertEqual(result['candidate_occurrences'],0)
        self.assertEqual(result['missing_nonblank_token_ids'],list(range(1,69)))
        self.assertIsNone(result['syllable_lengths']['min'])
        self.assertEqual(result['eojeol_occurrences'],[])

    def test_coverage_does_not_count_blank_as_seen_token(self):
        candidates,_=inventory.make_candidates([record('r1','s1','rec1','가 나')],{'protected_phrase_text_keys':[]},TOK)
        c=inventory.coverage(candidates,TOK)
        self.assertNotIn(0,c['observed_token_ids']);self.assertNotIn(0,c['missing_nonblank_token_ids'])
        self.assertIn(68,c['observed_token_ids']);self.assertEqual(c['unique_hangul_syllable_count'],2)
        self.assertEqual(c['unique_eojeol_surface_count'],2)

    def test_full_transcript_duplicate_groups_preserve_parent_recording_speaker_counts(self):
        d=inventory.full_transcript_duplicates(fixture())
        self.assertEqual((d['duplicate_group_count'],d['parents_in_duplicate_groups'],d['duplicate_excess_parents']),(1,2,1))
        self.assertEqual((d['groups'][0]['recording_count'],d['groups'][0]['speaker_count']),(1,1))

    def test_source_records_and_approval_status_are_not_mutated_or_aliased(self):
        rows=fixture();before=copy.deepcopy(rows)
        result=inventory.build(rows,{'protected_phrase_text_keys':[]},TOK)
        self.assertEqual(rows,before)
        result['phrase_candidates']['records'][0]['source_approval_status']['training_ready']=True
        self.assertEqual(rows,before)
        self.assertIn('69-token CTC',result['summary']['model_output_contract'])


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--result',type=Path,default=BASE/'unit_test_results_r3.json')
    output=parser.parse_args().result.resolve()
    if output.exists() or not output.is_relative_to(BASE):raise ValueError('New test output under inventory required')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InventoryTests))
    report={'schema':'offline-stage5e-inventory-tests-v1','passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'failures':[{'test':str(t),'traceback':s} for t,s in result.failures],
        'errors':[{'test':str(t),'traceback':s} for t,s in result.errors],
        'producer':inventory.ref(Path(inventory.__file__)),'test_code':inventory.ref(Path(__file__)),
        'new_model_calls':0,'new_media_reads':0,'new_training_updates':0}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)
