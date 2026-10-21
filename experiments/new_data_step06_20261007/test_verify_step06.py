"""Tamper and numerical tests for the independent new120 evidence audit."""
from __future__ import annotations

import ast
import copy
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

import verify_step06 as subject


def analytic_pair(delay, polarity=1., dc=0., reference_seconds=6., search_seconds=8.):
    rng = np.random.default_rng(6017)
    frequencies = rng.uniform(150, 3200, 83)
    phases = rng.uniform(-np.pi, np.pi, len(frequencies))
    amplitudes = rng.uniform(.3, 1, len(frequencies))
    def signal(times):
        out = np.zeros_like(times)
        for frequency, phase, amplitude in zip(frequencies, phases, amplitudes):
            out += amplitude * np.sin(2 * np.pi * frequency * times + phase)
        return out / len(frequencies)
    return (signal(np.arange(round(reference_seconds * 8000)) / 8000 + delay),
            polarity * signal(np.arange(round(search_seconds * 8000)) / 8000) + dc)


class SourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = subject.read(subject.SOURCE_PLAN, subject.SOURCE_SHA)
        cls.selection = subject.read(subject.SELECTION, subject.SELECTION_SHA)
        cls.role_map = subject.read(subject.AUDIT / 'archive_results/readiness_role_map.json')
        cls.protected = subject.read(cls.role_map['evidence']['protected_reference']['path'])

    def check(self, plan=None, selection=None, role_map=None, protected=None):
        subject.validate_roles(plan or self.plan, selection or self.selection,
                               role_map or self.role_map, protected or self.protected)

    def test_actual_complete_source_plan_without_archive_payload_access(self):
        original = Path.open
        def protected_open(path, *args, **kwargs):
            if PureWindowsPath(str(path)).drive.upper() in ('D:', 'G:'):
                raise AssertionError('Audit attempted source payload access')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', protected_open):
            actual = subject.audit_source_plan()
        self.assertEqual(actual['records'], self.selection['records'])
        self.assertEqual(len(actual['sources']), 30)

    def test_source_selection_exact_unchanged(self):
        changed = copy.deepcopy(self.plan)
        changed['records'][0]['start_time'] += .1
        with self.assertRaises(ValueError):
            self.check(plan=changed)

    def test_development_role_swap_rejected(self):
        changed = copy.deepcopy(self.plan)
        changed['sources'][0]['split'] = 'train'
        with self.assertRaises(ValueError):
            self.check(plan=changed)

    def test_missing_duplicate_or_reordered_source_ids_rejected(self):
        for ids in (self.plan['sources'][0]['record_ids'][:3],
                    [self.plan['sources'][0]['record_ids'][0]] * 4,
                    list(reversed(self.plan['sources'][0]['record_ids']))):
            changed = copy.deepcopy(self.plan)
            changed['sources'][0]['record_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.check(plan=changed)

    def test_old_offsets_and_training_approval_rejected(self):
        for key, value in (('old59_time_offset_applied', True), ('verified_new_time_offset_seconds', .02), ('training_ready', True)):
            changed = copy.deepcopy(self.plan)
            changed['sources'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(plan=changed)

    def test_unselected_development_recording_cannot_become_training(self):
        changed = copy.deepcopy(self.role_map)
        record = next(r for r in changed['records'] if r['speaker_id'] == 'C040'
                      and r['readiness_role'].endswith('_unselected_recording'))
        record['speaker_role'] = 'new_pilot_train'
        with self.assertRaises(ValueError):
            self.check(role_map=changed)

    def test_final_test_or_old_validation_reassignment_rejected(self):
        for split in ('test', 'val'):
            changed = copy.deepcopy(self.role_map)
            changed['original_v4_contract'][split]['clips'] = 0
            with self.subTest(split=split), self.assertRaises(ValueError):
                self.check(role_map=changed)

    def test_guard_digest_tamper_rejected(self):
        changed = copy.deepcopy(self.protected)
        changed['guards']['blocked_existing_sample_ids'].pop()
        with self.assertRaises(ValueError):
            self.check(protected=changed)

    def test_no_producer_or_model_imports(self):
        tree = ast.parse(Path(subject.__file__).read_text(encoding='utf-8'))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(value and any(word in value for word in ('build_plan', 'prepare_media', 'tokenizer', 'torch', 'refine_alignment'))
                             for value in imports))


class SignalEvidenceTests(unittest.TestCase):
    def test_fractional_local_delay_and_all_gates(self):
        expected = 1 + 1024 / 48000
        actual = subject.independent_score(*analytic_pair(expected))
        self.assertEqual(actual['best_start_sample_in_search'], 48000 + 1024)
        self.assertEqual(actual['global_candidate_count'], 96001)
        self.assertGreater(actual['absolute_correlation'], .995)
        self.assertTrue(all(subject.gates(actual).values()))

    def test_polarity_dc_do_not_change_offset(self):
        expected = 1 + 1337 / 48000
        actual = subject.independent_score(*analytic_pair(expected, -1.75, .3))
        self.assertAlmostEqual(actual['best_start_seconds_in_search'], expected, places=12)
        self.assertLess(actual['signed_correlation'], -.995)
        self.assertTrue(all(subject.gates(actual).values()))

    def test_silence_and_boundary_not_supported(self):
        silence = subject.independent_score(np.zeros(6 * 8000), np.zeros(8 * 8000))
        self.assertFalse(subject.gates(silence)['reference_energy'])
        self.assertFalse(subject.gates(silence)['matched_energy'])
        boundary = subject.independent_score(*analytic_pair(.1 + 1 / 48000))
        self.assertFalse(subject.gates(boundary)['boundary_clearance'])

    def test_repeated_signal_is_ambiguous(self):
        tile = np.random.default_rng(6077).normal(size=8000)
        search = np.tile(tile, 8)
        reference = search[8000:4 * 8000].copy()
        result = subject.independent_score(reference, search)
        self.assertGreater(result['absolute_correlation'], .99)
        self.assertFalse(subject.gates(result)['unique_far_peak'])

    def test_nonfinite_signal_rejected(self):
        with self.assertRaises(ValueError):
            subject.independent_score(np.full(1000, np.nan), np.zeros(2000))

    def test_saved_score_tampering_rejected(self):
        result = subject.independent_score(*analytic_pair(1.))
        saved = {**result, 'strict_gates': subject.gates(result), 'strict_anchor_pass': True}
        subject.validate_score(saved, result)
        for key, bad in (('best_start_sample_in_search', 0), ('signed_correlation', 0.),
                         ('strict_anchor_pass', False), ('strict_gates', {})):
            altered = copy.deepcopy(saved)
            altered[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_score(altered, result)


class PCMTests(unittest.TestCase):
    def test_saved_pcm_hash_length_rate_and_origin(self):
        with tempfile.TemporaryDirectory(prefix='independent-test-', dir=subject.BASE) as folder:
            path = Path(folder) / 'audio.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                stream.writeframes(np.arange(8000, dtype='<i2').tobytes())
            spec = {**subject.bind(path), 'start_seconds': 30.}
            values = subject.pcm(spec, 8000, 1., 30.)
            self.assertEqual(len(values), 8000)
            self.assertEqual(values[-1], 7999 / 32768)
            for rate, seconds, start in ((16000, 1., 30.), (8000, 2., 30.), (8000, 1., 31.)):
                with self.subTest(rate=rate, seconds=seconds), self.assertRaises(ValueError):
                    subject.pcm(spec, rate, seconds, start)
            bad = {**spec, 'sha256': '0' * 64}
            with self.assertRaises(ValueError):
                subject.pcm(bad, 8000, 1.)


class SavedMediaTamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = subject.read(subject.SOURCE_PLAN)
        cls.source = cls.plan['sources'][0]
        cls.item = subject.read(subject.BASE / 'results/sources' / (cls.source['stem'] + '.json'))
        cls.clip = cls.item['clips'][0]
        cls.row = next(r for r in cls.plan['records'] if r['id'] == cls.clip['id'])
        cls.mapping = cls.item['offset_summary']
        cls.label = subject.read(cls.source['label_copy']['path'])[0]

    def test_actual_png_pixels_and_wrong_dimensions_or_hash(self):
        spec = self.clip['roi_samples'][0]['image']
        self.assertEqual(subject.pixels(spec, [96, 96, 3]).shape, (96, 96, 3))
        with self.assertRaises(ValueError):
            subject.pixels(spec, [95, 96, 3])
        bad = copy.deepcopy(spec)
        bad['pixels']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            subject.pixels(bad, [96, 96, 3])

    def test_clock_source_sample_count_and_window_tamper(self):
        spec = self.clip['audio']
        args = self.source, 'audio', self.row['start_time'], self.row['end_time']-self.row['start_time'], 16000
        for key, wrong in (('clock', 'mp4'), ('source_sha256', '0'*64), ('frames', 0),
                           ('requested_seconds', 6.), ('start_seconds', spec['start_seconds']+.5)):
            bad = copy.deepcopy(spec)
            bad[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_pcm(bad, *args)

    def test_timing_indices_offset_fps_and_ctc_tamper(self):
        original = self.clip['timing']
        offset = self.mapping['median_offset_seconds']
        subject.validate_timing(original, self.row, offset, self.source)
        changes = [('actual_video_fps', 30.), ('native_frame_indices', original['native_frame_indices'][::-1]),
                   ('student_frames', original['student_frames']-1), ('ctc_min_frames', 1),
                   ('corrected_start_time', self.row['start_time'])]
        for key, wrong in changes:
            bad = copy.deepcopy(original)
            bad[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_timing(bad, self.row, offset, self.source)

    def test_preview_box_native_index_and_pixel_provenance_tamper(self):
        indices, times = subject.validate_timing(self.clip['timing'], self.row,
            self.mapping['median_offset_seconds'], self.source)
        subject.validate_previews(self.clip, self.row, self.source, indices, times, self.label)
        for key, wrong in (('student_index', 3), ('native_frame_index', 0), ('bbox_yxyx', [0., 0., 1., 1.]),
                           ('crop_xyxy', [0, 0, 96, 96]), ('decoded_pts_seconds', 0.)):
            bad = copy.deepcopy(self.clip)
            bad['roi_samples'][0][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_previews(bad, self.row, self.source, indices, times, self.label)

    def test_artifact_inventory_omission_or_cross_source_rejected(self):
        bad = copy.deepcopy(self.item)
        bad['artifacts_all'].pop()
        with self.assertRaises(ValueError):
            subject.artifact_inventory(bad)
        bad = copy.deepcopy(self.item)
        bad['clips'][0]['audio']['path'] = str(subject.BASE/'results/media/other/label_audio.wav')
        with self.assertRaises(ValueError):
            subject.artifact_inventory(bad)

    def test_prepared_label_and_local_decision_cannot_change(self):
        # Signal math has separate known-delay tests; isolate decision/evidence guards here.
        score = {k: v for k, v in self.clip['local_check']['score'].items()
                 if k not in ('strict_gates', 'strict_anchor_pass', 'offset_mp4_minus_original_seconds')}
        for change in ({'text': '변경'}, {'training_ready': True}, {'human_transcript_review': 'approved'},
                       {'split': 'train'}, {'timing_status': 'not_verified'}):
            bad = copy.deepcopy(self.clip)
            bad.update(change)
            with self.subTest(change=change), patch.object(subject, 'independent_score', return_value=score), self.assertRaises(ValueError):
                subject.validate_clip(bad, self.row, self.source, self.item, self.mapping, self.label)
        bad = copy.deepcopy(self.clip)
        bad['local_check']['verified'] = False
        with patch.object(subject, 'independent_score', return_value=score), self.assertRaises(ValueError):
            subject.validate_clip(bad, self.row, self.source, self.item, self.mapping, self.label)

    def test_late_source_failure_requires_withdrawal_of_timing_approval(self):
        score = {k: v for k, v in self.clip['local_check']['score'].items()
                 if k not in ('strict_gates', 'strict_anchor_pass', 'offset_mp4_minus_original_seconds')}
        item = copy.deepcopy(self.item)
        item['status'] = 'quarantined'
        with patch.object(subject, 'independent_score', return_value=score), self.assertRaises(ValueError):
            subject.validate_clip(self.clip, self.row, self.source, item, self.mapping, self.label)
        clip = copy.deepcopy(self.clip)
        clip.update(status='quarantined', artifacts_ready=False, timing_status='invalidated_source_integrity', quarantine_reason='late source change')
        clip['local_check'].update(verified=False, waveform_gate_pass_before_source_invalidation=True, invalidated_by_source_error=True)
        with patch.object(subject, 'independent_score', return_value=score):
            result = subject.validate_clip(clip, self.row, self.source, item, self.mapping, self.label)
        self.assertFalse(result['artifacts_ready'])

    def test_actual_frozen_media_plan_binds_current_producer_without_import(self):
        value = subject.validate_media_plan(self.plan)
        self.assertEqual(value['source_plan']['sha256'], subject.SOURCE_SHA)
        self.assertFalse(value['scope']['training_ready'])


if __name__ == '__main__':
    unittest.main()
