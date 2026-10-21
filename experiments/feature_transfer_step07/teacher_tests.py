"""Synthetic teacher-contract tests; no source video or public checkpoint access."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from teacher_features import (STATE_PREFIX, exact_gather_indices, load_front_state,
                              normalize_teacher, read_prepared_npz, save_new_npz,
                              sha256, state_sha256, validate_front_output)


class TeacherContractTests(unittest.TestCase):
    def test_exact_original_frame_mapping_preserves_repeats(self):
        teacher = np.arange(90, 109, dtype=np.int64)
        student = np.array([92, 93, 93, 96, 99, 106], dtype=np.int64)
        gather = exact_gather_indices(teacher, student)
        np.testing.assert_array_equal(gather, [2, 3, 3, 6, 9, 16])
        np.testing.assert_array_equal(teacher[gather], student)

    def test_mapping_refuses_clamp_holes_and_wrong_types(self):
        teacher = np.arange(90, 99, dtype=np.int64)
        for student in (np.array([89, 91], dtype=np.int64), np.array([93, 99], dtype=np.int64),
                        np.array([93, 92], dtype=np.int64), np.array([92.0]),
                        np.array([], dtype=np.int64)):
            with self.subTest(student=student):
                with self.assertRaises(ValueError):
                    exact_gather_indices(teacher, student)
        with self.assertRaises(ValueError):
            exact_gather_indices(np.array([90, 91, 93], dtype=np.int64), np.array([91], dtype=np.int64))

    def test_bgr_global_sample_std_without_input_mutation(self):
        pixels = np.zeros((3, 96, 96, 3), dtype=np.uint8)
        pixels[..., 0], pixels[..., 1], pixels[..., 2] = 7, 59, 191
        pixels[2] += 11
        original = pixels.copy()
        tensor, report = normalize_teacher(pixels)
        expected = pixels.astype(np.float64)
        expected = (expected - expected.mean()) / expected.std(ddof=1)
        self.assertEqual(tuple(tensor.shape), (1, 3, 3, 96, 96))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor.numpy()[0].transpose(1, 2, 3, 0), expected, atol=2e-6)
        np.testing.assert_array_equal(pixels, original)
        self.assertAlmostEqual(report["normalized_std"], 1.0, places=6)

    def test_zero_variance_and_non_uint8_are_rejected(self):
        for pixels in (np.zeros((2, 96, 96, 3), np.uint8),
                       np.zeros((2, 96, 96, 3), np.float32),
                       np.zeros((0, 96, 96, 3), np.uint8)):
            with self.assertRaises(ValueError):
                normalize_teacher(pixels)

    def test_feature_gather_validates_length_dtype_and_finite(self):
        output = torch.arange(7 * 512, dtype=torch.float32).reshape(1, 7, 512)
        gather = np.array([1, 1, 3, 6], dtype=np.int64)
        selected = validate_front_output(output, 7, gather)
        np.testing.assert_array_equal(selected, output.numpy()[0, [1, 1, 3, 6]])
        for bad, frames in ((output, 6), (output.double(), 7), (output[:, :, :511], 7)):
            with self.assertRaises(ValueError):
                validate_front_output(bad, frames, gather)
        output[0, 2, 3] = torch.nan
        with self.assertRaises(ValueError):
            validate_front_output(output, 7, gather)
        with self.assertRaises(ValueError):
            validate_front_output(torch.zeros((1, 7, 512), dtype=torch.float32), 7, gather)

    @staticmethod
    def tiny_front():
        return torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))

    def test_strict_front_only_load_and_freeze(self):
        source, target = self.tiny_front(), self.tiny_front()
        state = {STATE_PREFIX + k: v.clone() for k, v in source.state_dict().items()}
        state["decoder.unrelated.weight"] = torch.ones((11, 19))
        report = load_front_state(target, state)
        self.assertEqual(report["other_state_entries_loaded"], 0)
        self.assertEqual(state_sha256(target), state_sha256(source))
        self.assertFalse(target.training)
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))

    def test_strict_front_load_rejects_missing_unexpected_shape_dtype(self):
        base = {STATE_PREFIX + k: v.clone() for k, v in self.tiny_front().state_dict().items()}
        mutations = []
        missing = dict(base)
        del missing[STATE_PREFIX + "0.weight"]
        mutations.append(missing)
        unexpected = dict(base)
        unexpected[STATE_PREFIX + "invented"] = torch.ones(1)
        mutations.append(unexpected)
        shape = dict(base)
        shape[STATE_PREFIX + "0.weight"] = torch.ones(3, 3)
        mutations.append(shape)
        dtype = dict(base)
        dtype[STATE_PREFIX + "0.weight"] = dtype[STATE_PREFIX + "0.weight"].double()
        mutations.append(dtype)
        for state in mutations:
            with self.assertRaises(ValueError):
                load_front_state(self.tiny_front(), state)

    def test_state_hash_includes_batchnorm_buffers(self):
        front = self.tiny_front().eval()
        before = state_sha256(front)
        front[1].num_batches_tracked += 1
        self.assertNotEqual(state_sha256(front), before)
        before = state_sha256(front)
        front[1].running_mean[0] += 0.125
        self.assertNotEqual(state_sha256(front), before)

    def test_npz_integrity_and_distinct_geometry_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.npz"
            evidence = save_new_npz(path, teacher_bgr=np.ones((7, 96, 96, 3), np.uint8),
                                    student_rgb=np.zeros((3, 96, 96, 3), np.uint8),
                                    teacher_source_indices=np.arange(100, 107, dtype=np.int64),
                                    student_source_indices=np.array([102, 103, 105], dtype=np.int64))
            result = read_prepared_npz(path, evidence["sha256"])
            np.testing.assert_array_equal(result["gather_indices"], [2, 3, 5])
            self.assertEqual(sha256(path), evidence["sha256"])
            with self.assertRaises(FileExistsError):
                save_new_npz(path, other=np.zeros(1))
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                read_prepared_npz(path, evidence["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
