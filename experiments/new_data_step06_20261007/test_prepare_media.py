"""Synthetic media contract tests. No dataset archive or model access.

Set LIPNET_STEP06_FFMPEG_TESTS=1 to include the real FFmpeg synthetic-WAV
fractional interval test. All other tests run without launching FFmpeg.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new120_prepare_media_tests',HERE/'prepare_media.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def wav(path, seconds=8., rate=8000, channels=1, values=None):
    if values is None:
        values=np.zeros(round(seconds*rate)*channels,dtype='<i2')
    with wave.open(str(path),'wb') as stream:
        stream.setframerate(rate); stream.setnchannels(channels); stream.setsampwidth(2)
        stream.writeframes(np.asarray(values,dtype='<i2').tobytes())


def row():
    return {'id':'source_001','speaker_id':'C001','split':'train','source_text':'hello',
            'text':'hello','sentence_id':1,'start_time':20.,'end_time':22.,
            'expected_frames':50,'ctc_min_frames':5,'label_sha256':'a'*64}


def source():
    return {'stem':'source','speaker_id':'C001','split':'train','label_fps':30,
            'video_metadata':{'width':192,'height':108},'members':{'audio':{'sha256':'b'*64},'video':{'sha256':'c'*64}}}


class Tokenizer:
    @staticmethod
    def encode_text(text):
        return [1,2,3,4,5]
    @staticmethod
    def normalize_text(text):
        return text


def label():
    return [{'Video_info':{'video_Name':'source.mp4','FPS':30},'Audio_info':{'Audio_Name':'source.wav'},
             'speaker_info':{'speaker_ID':'C001'},'Video_env':{'Angle':'A'},
             'Sentence_info':[{'ID':1,'sentence_text':'hello','start_time':20.,'end_time':22.}]}]


class TimingTests(unittest.TestCase):
    def test_global_offsets_sign_and_median_without_old_offsets(self):
        anchors=[{'score':{'offset_mp4_minus_original_seconds':v,'strict_anchor_pass':True}} for v in [-4.,-4.000125,-3.999875]]
        got=m.offset_summary(anchors)
        self.assertEqual(got['median_offset_seconds'],-4.)
        self.assertTrue(got['constant_mapping_supported'])

    def test_global_spread_is_exclusive_and_failure_preserved(self):
        a=[{'score':{'offset_mp4_minus_original_seconds':v,'strict_anchor_pass':True}} for v in [0.,.01,.02]]
        self.assertFalse(m.offset_summary(a)['constant_mapping_supported'])
        a[2]['score']['offset_mp4_minus_original_seconds']=.019
        a[1]['score']['strict_anchor_pass']=False
        self.assertFalse(m.offset_summary(a)['constant_mapping_supported'])

    def test_global_requires_all_three(self):
        with self.assertRaisesRegex(ValueError,'three'):
            m.offset_summary([])

    def test_local_residual_boundary_and_strict_signal_gate(self):
        score={'offset_mp4_minus_original_seconds':.001,'strict_anchor_pass':True}
        self.assertTrue(m.local_check_gate(score,0.)['verified'])
        score['offset_mp4_minus_original_seconds']=.00100001
        self.assertFalse(m.local_check_gate(score,0.)['verified'])
        score.update(offset_mp4_minus_original_seconds=0.,strict_anchor_pass=False)
        self.assertFalse(m.local_check_gate(score,0.)['verified'])

    def test_windows_are_never_clamped(self):
        for args in [(-.001,6.,300.),(295.,6.,300.),(1.,0.,300.),(float('nan'),6.,300.)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                m.in_bounds(*args)
        m.in_bounds(294.,6.,300.)

    def test_actual_fps_mapping_and_ctc_preserve_nominal_times(self):
        r=row(); before=copy.deepcopy(r)
        got=m.target_indices(r,-4.,30000/1001,9000,300.3)
        self.assertEqual(got['native_frame_indices'][0],480)
        self.assertEqual(got['native_frame_indices'][-1],538)
        self.assertEqual(got['student_frames'],50)
        self.assertEqual(got['corrected_start_time'],16.)
        self.assertEqual(r,before)

    def test_ctc_impossible_and_physical_end_rejected(self):
        r=row(); r['ctc_min_frames']=51
        with self.assertRaisesRegex(ValueError,'CTC'):
            m.target_indices(r,0.,30.,9000,300.)
        with self.assertRaisesRegex(ValueError,'outside'):
            m.target_indices(row(),-21.,30.,9000,300.)
        r=row(); r.update(start_time=0.,end_time=.04,expected_frames=1,ctc_min_frames=1)
        with self.assertRaisesRegex(ValueError,'Native frame'):
            m.target_indices(r,.025,30.,1,1.)

    def test_late_integrity_failure_withdraws_verified_flags(self):
        c=m.initial_clip(row(),source())
        c.update(status='prepared',artifacts_ready=True,timing_status='verified',local_check={'verified':True,'score':{'absolute_correlation':1.}})
        m.invalidate_source_clips([c],'late archive stat mismatch')
        self.assertFalse(c['artifacts_ready']); self.assertFalse(c['local_check']['verified'])
        self.assertEqual(c['timing_status'],'invalidated_source_integrity')
        self.assertTrue(c['local_check']['waveform_gate_pass_before_source_invalidation'])
        self.assertEqual(c['local_check']['score']['absolute_correlation'],1.)


class WaveformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scoring=m.module('new120_test_unchanged_scoring',m.REFINE,m.REFINE_SHA)

    def test_full_global_search_recovers_negative_polarity_and_known_clock_offset(self):
        # The reference is physically copied at17s in a36s search. WAV30 vs
        # MP4 search15 therefore gives offset +2s. Sign inversion is allowed.
        rng=np.random.default_rng(915)
        search=rng.normal(0,.08,36*8000)
        reference=-search[17*8000:23*8000].copy()
        score=self.scoring.global_score(reference,search)
        self.assertTrue(score['strict_anchor_pass'])
        self.assertAlmostEqual(score['best_start_seconds_in_search'],17.,places=6)
        self.assertLess(score['signed_correlation'],-.999)
        self.assertAlmostEqual(15+score['best_start_seconds_in_search']-30,2.,places=6)

    def test_local_unique_shift_recovered_on_same_interpolated_grid(self):
        rng=np.random.default_rng(916)
        search=rng.normal(0,.08,8*8000)
        reference=search[8000:7*8000].copy()
        score=self.scoring.global_score(reference,search)
        self.assertTrue(score['strict_anchor_pass'])
        self.assertEqual(score['best_start_seconds_in_search'],1.)
        self.assertEqual(score['evaluation_grid_rate'],48000)

    def test_silence_cannot_pass_correlation_or_energy_gates(self):
        score=self.scoring.global_score(np.zeros(6*8000),np.zeros(8*8000))
        self.assertFalse(score['strict_anchor_pass'])
        self.assertFalse(score['strict_gates']['reference_energy'])

    def test_repeated_periodic_audio_fails_peak_uniqueness(self):
        t=np.arange(8*8000)/8000
        search=.1*np.sin(2*np.pi*440*t)
        score=self.scoring.global_score(search[8000:7*8000],search)
        self.assertFalse(score['strict_gates']['unique_far_peak'])


class LabelAndRoiTests(unittest.TestCase):
    def test_exact_sentence_and_speaker_link(self):
        self.assertEqual(m.validate_label(label(),source(),[row()],Tokenizer)['speaker_info']['speaker_ID'],'C001')
        for key,value in [('sentence_text','changed'),('start_time',20.001)]:
            doc=label(); doc[0]['Sentence_info'][0][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'text/time'):
                m.validate_label(doc,source(),[row()],Tokenizer)
        doc=label(); doc[0]['speaker_info']['speaker_ID']='C002'
        with self.assertRaisesRegex(ValueError,'Speaker'):
            m.validate_label(doc,source(),[row()],Tokenizer)

    def test_duplicate_sentence_id_cannot_silently_overwrite(self):
        doc=label(); doc[0]['Sentence_info']*=2
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            m.validate_label(doc,source(),[row()],Tokenizer)

    def test_yxyx_geometry_is_converted_before_width_height_bounds(self):
        doc={'Bounding_box_info':{'Lip_bounding_box':{'xtl_ytl_xbr_ybr':[[10,150,40,190]]}}}
        got=m.boxes_yxyx(doc,'Lip_bounding_box',1,192,108)
        self.assertEqual(got.tolist(),[[10,150,40,190]])
        with self.assertRaisesRegex(ValueError,'count'):
            m.boxes_yxyx(doc,'Lip_bounding_box',2,192,108)
        doc['Bounding_box_info']['Lip_bounding_box']['xtl_ytl_xbr_ybr'][0][2]=109
        with self.assertRaisesRegex(ValueError,'geometry'):
            m.boxes_yxyx(doc,'Lip_bounding_box',1,192,108)

    def test_preview_crop_has_twenty_percent_each_side_and_bgr_pixels(self):
        frame=np.zeros((108,192,3),dtype=np.uint8); frame[:,:,0]=77
        roi,rectangle=m.preview_crop(frame,[20,40,50,90])
        self.assertEqual(rectangle,[30,14,100,56]); self.assertEqual(roi.shape,(96,96,3))
        self.assertTrue(np.all(roi[:,:,0]==77)); self.assertTrue(np.all(roi[:,:,2]==0))
        _,edge=m.preview_crop(frame,[0,0,10,10])
        self.assertEqual(edge,[0,0,12,12])

    def test_mismatched_frame_pts_rejected(self):
        class Capture:
            def set(self,*args): return True
            def read(self): return True,np.zeros((108,192,3),dtype=np.uint8)
            def get(self,prop): return 1000. if prop==m.cv2.CAP_PROP_POS_MSEC else 1.
        targets={'student_frames':3,'native_frame_indices':[0,1,2],'actual_video_fps':30.,'target_times_seconds':[0.,1/30,2/30]}
        with self.assertRaisesRegex(ValueError,'PTS/index'):
            m.roi_previews(Capture(),source(),row(),targets,np.array([[20,40,50,90]]*3),np.array([[0,0,100,190]]*3),HERE)

    def test_duration_bound_rejects_nonunit_edit_speed(self):
        metadata={'movie_clock':{'duration_seconds':300},'tracks':[{'handler':'soun','media_clock':{'duration_seconds':301},'edit_list':[{'media_rate_integer':2,'media_rate_fraction':0,'movie_duration_seconds':299}]}]}
        with self.assertRaisesRegex(ValueError,'Non-unit'):
            m.duration_bound(metadata,'soun')
        metadata['tracks'][0]['edit_list'][0]['media_rate_integer']=1
        self.assertEqual(m.duration_bound(metadata,'soun'),299)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='media-synthetic-',dir=HERE)
        self.folder=Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()

    def test_pcm_header_and_hash_are_checked_before_scoring(self):
        path=self.folder/'a.wav'; wav(path)
        spec={**m.bind(path),'frames':64000}
        self.assertEqual(m.pcm(spec).shape,(64000,))
        spec['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'changed'):
            m.pcm(spec)
        wav(path,rate=16000)
        spec={**m.bind(path),'frames':128000}
        with self.assertRaisesRegex(ValueError,'mono8k'):
            m.pcm(spec)

    def test_truncated_pcm_frame_count_rejected(self):
        path=self.folder/'a.wav'; wav(path,seconds=5.)
        with self.assertRaisesRegex(ValueError,'count'):
            m.pcm({**m.bind(path),'frames':48000})

    def test_extraction_must_emit_exact_length(self):
        output=self.folder/'audio.wav'
        def fake_ffmpeg(*args,**kwargs):
            wav(output,seconds=.999,rate=16000)
            return type('Result',(),{'returncode':0,'stderr':''})()
        with patch.object(m.subprocess,'run',fake_ffmpeg),self.assertRaisesRegex(ValueError,'Incomplete'):
            m.excerpt(self.folder/'synthetic.wav',{'sha256':'a'*64},'audio',0.,1.,2.,output,16000,'fake')

    def test_resume_rejects_pixel_tamper_wrong_role_and_bad_timing_approval(self):
        result_root=self.folder/'results'; artifact=result_root/'media/source/attempt'; artifact.mkdir(parents=True)
        image=artifact/'sample.bin'; image.write_bytes(b'pixels')
        source_plan=self.folder/'source_plan.json'; source_plan.write_text('{}')
        c=m.initial_clip(row(),source())
        item={'schema':'lipnet-new120-media-source-v1','media_plan':{'path':'plan','sha256':'x'},
              'source_plan':m.bind(source_plan),'source_stem':'source','speaker_id':'C001','split':'train',
              'training_ready':False,'human_transcript_review':'not_reviewed','status':'quarantined',
              'clips':[c],'artifact_directory':str(artifact),'artifacts_all':m.artifact_list(artifact)}
        with patch.object(m,'RESULTS',result_root),patch.object(m,'SOURCE_PLAN',source_plan):
            m.verify_saved_result(item,source(),[row()],item['media_plan'])
            bad=copy.deepcopy(item); bad['clips'][0]['split']='val'
            with self.assertRaisesRegex(ValueError,'roles'):
                m.verify_saved_result(bad,source(),[row()],item['media_plan'])
            bad=copy.deepcopy(item); bad['clips'][0]['timing_status']='verified'
            with self.assertRaisesRegex(ValueError,'Quarantined source'):
                m.verify_saved_result(bad,source(),[row()],item['media_plan'])
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact'):
                m.verify_saved_result(item,source(),[row()],item['media_plan'])

    def test_final_initial_rows_preserve_order_and_remain_unapproved(self):
        a=row(); b={**row(),'id':'source_004','sentence_id':4}
        clips=[m.initial_clip(r,source()) for r in [b,a]]
        self.assertEqual([c['id'] for c in clips],['source_004','source_001'])
        self.assertTrue(all(c['training_ready'] is False and c['human_transcript_review']=='not_reviewed' for c in clips))

    @unittest.skipUnless(os.environ.get('LIPNET_STEP06_FFMPEG_TESTS')=='1','Optional real FFmpeg synthetic test')
    def test_real_ffmpeg_fractional_48k_stereo_to_16k_and_8k(self):
        rng=np.random.default_rng(918)
        original=self.folder/'synthetic.wav'
        wav(original,seconds=12.,rate=48000,channels=2,values=rng.integers(-2000,2001,12*48000*2,dtype=np.int16))
        # Fractional48k-label boundaries representative of the frozen dataset.
        for index,(start,seconds,rate) in enumerate([(1.125333333333333,3.178666666666667,16000),
                                                    (2.3106666666666666,2.922666666666667,16000),
                                                    (3.251333333333333,6.,8000)]):
            info=m.excerpt(original,{'sha256':m.sha(original)},'audio',start,seconds,12.,self.folder/f'excerpt{index}.wav',rate,str(m.FFMPEG))
            self.assertEqual(info['frames'],round(seconds*rate))
            self.assertEqual(info['channels'],1); self.assertEqual(info['sample_width_bytes'],2)


if __name__=='__main__':
    unittest.main(verbosity=2)
