"""Independent synthetic checks for the paired stage07 training contract.

No project clips, pretrained checkpoints, Drive source, or development labels
are opened. Tests use CPU tensors and a small stand-in with the same live-hook
interface as the compact student. Run with a torch-enabled Python environment.
"""
from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_compare as subject


class TinyStudent(nn.Module):
    """Retain temporal length and projection[2] while making tests inexpensive."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(3, 256), nn.LayerNorm(256),
                                        nn.ReLU(), nn.Dropout(dropout))
        self.fc = nn.Linear(256, 69)

    def forward(self, value: torch.Tensor, lengths=None) -> torch.Tensor:
        del lengths
        features = value.mean(dim=(-1, -2)).transpose(1, 2)
        features = self.projection(features)
        return self.fc(features).log_softmax(-1).transpose(0, 1).contiguous()


def objects(seed: int = 17, dropout: float = 0.1, arm: str = "B_feature"):
    subject.seed_all(seed)
    model, adapter = TinyStudent(dropout), nn.Linear(256, 512)
    params = list(model.parameters())
    if arm == "B_feature":
        params += list(adapter.parameters())
    return model, adapter, torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999),
                                           eps=1e-8, weight_decay=0)


def clips() -> list[dict]:
    generator = np.random.default_rng(20260924)
    return [{"id": f"synthetic_{length}",
             "pixels": generator.integers(0, 256, (length, 96, 96, 3), dtype=np.uint8),
             "tokens": [1, 2],
             "teacher_features": generator.normal(size=(length, 512)).astype(np.float32)}
            for length in (6, 9)]


def assert_nested_equal(test: unittest.TestCase, a, b, path: str = "root") -> None:
    if isinstance(a, torch.Tensor):
        test.assertIsInstance(b, torch.Tensor, path)
        test.assertEqual(a.dtype, b.dtype, path)
        test.assertTrue(torch.equal(a, b), path)
    elif isinstance(a, np.ndarray):
        test.assertTrue(np.array_equal(a, b), path)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys(), path)
        for key in a:
            assert_nested_equal(test, a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(type(a), type(b), path)
        test.assertEqual(len(a), len(b), path)
        for index, (left, right) in enumerate(zip(a, b)):
            assert_nested_equal(test, left, right, f"{path}[{index}]")
    else:
        test.assertEqual(a, b, path)


class TrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_normalized_distance_has_known_scale_and_live_gradient(self):
        adapter = nn.Linear(256, 512, bias=False)
        with torch.no_grad():
            adapter.weight.zero_()
            adapter.weight[:256].copy_(torch.eye(256))
        h = torch.zeros(1, 3, 256, requires_grad=True)
        with torch.no_grad():
            h[..., 0] = 1
        same = torch.zeros(1, 3, 512)
        same[..., 0] = 4
        orthogonal = torch.zeros_like(same)
        orthogonal[..., 1] = 2
        self.assertAlmostEqual(float(subject.frame_distillation_loss(h, same, adapter).detach()), 0, places=6)
        self.assertAlmostEqual(float(subject.frame_distillation_loss(h, -same, adapter).detach()), 4, places=6)
        loss = subject.frame_distillation_loss(h, orthogonal, adapter)
        self.assertAlmostEqual(float(loss.detach()), 2, places=6)
        loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertGreater(float(h.grad.abs().sum()), 0)
        self.assertGreater(float(adapter.weight.grad.abs().sum()), 0)

    def test_teacher_requires_grad_is_rejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            subject.frame_distillation_loss(torch.randn(1, 4, 256, requires_grad=True),
                                            torch.randn(1, 4, 512, requires_grad=True),
                                            nn.Linear(256, 512))

    def test_zero_or_nonfinite_teacher_is_rejected(self):
        for target in (torch.zeros(1, 4, 512), torch.full((1, 4, 512), float("nan"))):
            with self.assertRaises((ValueError, RuntimeError)):
                subject.frame_distillation_loss(torch.randn(1, 4, 256, requires_grad=True),
                                                target, nn.Linear(256, 512))

    def test_distillation_matches_independent_formula(self):
        subject.seed_all(44)
        h, target, adapter = torch.randn(1, 7, 256), torch.randn(1, 7, 512), nn.Linear(256, 512)
        projection = adapter(h)
        student = projection / projection.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        teacher = target / target.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        expected = (student - teacher).square().sum(-1).mean()
        self.assertTrue(torch.allclose(subject.frame_distillation_loss(h, target, adapter),
                                      expected, rtol=1e-6, atol=1e-7))

    def test_student_normalization_rgb_and_temporal_shape(self):
        pixels = np.zeros((2, 96, 96, 3), dtype=np.uint8)
        pixels[0] = [0, 127, 255]
        pixels[1] = [255, 64, 0]
        result = subject.prepare_student(pixels, torch.device("cpu"))
        self.assertEqual(tuple(result.shape), (1, 3, 2, 96, 96))
        expected = (torch.tensor([[0, 127, 255], [255, 64, 0]], dtype=torch.float32) / 255
                    - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        self.assertTrue(torch.allclose(result[0, :, :, 0, 0].T, expected, atol=1e-6, rtol=0))
        self.assertEqual(result.dtype, torch.float32)

    def test_pair_initialization_and_next_random_values_match(self):
        a, pa, _ = subject.make_objects(42, "A_ctc", torch.device("cpu"))
        ar = torch.rand(16)
        b, pb, _ = subject.make_objects(42, "B_feature", torch.device("cpu"))
        br = torch.rand(16)
        assert_nested_equal(self, a.state_dict(), b.state_dict())
        assert_nested_equal(self, pa.state_dict(), pb.state_dict())
        self.assertTrue(torch.equal(ar, br))

    def test_unequal_length_accumulation_equals_mean_of_clip_losses(self):
        batch = clips()
        model, adapter, optimizer = objects(dropout=0, arm="A_ctc")
        independent, _, independent_opt = objects(dropout=0, arm="A_ctc")
        independent_opt.zero_grad(set_to_none=True)
        loss = torch.zeros(())
        for record in batch:
            x = torch.from_numpy(record["pixels"].astype(np.float32) / 255).permute(3, 0, 1, 2)[None]
            x = (x - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None, None]) / torch.tensor(
                [0.229, 0.224, 0.225])[None, :, None, None, None]
            y = torch.as_tensor(record["tokens"], dtype=torch.long)
            loss = loss + F.ctc_loss(independent(x), y, [len(record["pixels"])], [len(y)],
                                     blank=0, reduction="mean", zero_infinity=False) / len(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(independent.parameters(), 1.0, error_if_nonfinite=True)
        independent_opt.step()
        subject.optimizer_update(model, adapter, optimizer, batch, "A_ctc", torch.device("cpu"))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, independent.state_dict()[name], atol=2e-7, rtol=2e-6), name)
        self.assertEqual(len(model.projection[2]._forward_hooks), 0)

    def test_rng_restore_covers_python_numpy_and_torch(self):
        subject.seed_all(781)
        state = subject.capture_rng()
        expected = (random.random(), np.random.random(5), torch.rand(5))
        subject.restore_rng(state)
        actual = (random.random(), np.random.random(5), torch.rand(5))
        assert_nested_equal(self, expected, actual)

    def test_runtime_resume_reproduces_next_update_and_restores_adapter(self):
        batch = clips()
        model, adapter, opt = objects()
        subject.optimizer_update(model, adapter, opt, batch, "B_feature", torch.device("cpu"))
        saved = subject.runtime_state(model, adapter, opt)
        saved_snapshot = copy.deepcopy(saved)
        subject.optimizer_update(model, adapter, opt, batch[::-1], "B_feature", torch.device("cpu"))
        expected = subject.runtime_state(model, adapter, opt)
        assert_nested_equal(self, saved, saved_snapshot)
        other, other_adapter, other_opt = objects(seed=55)
        subject.restore_state(saved, other, other_adapter, other_opt)
        subject.optimizer_update(other, other_adapter, other_opt, batch[::-1], "B_feature", torch.device("cpu"))
        actual = subject.runtime_state(other, other_adapter, other_opt)
        assert_nested_equal(self, expected, actual)
        self.assertEqual(len(model.projection[2]._forward_hooks), 0)
        self.assertEqual(len(other.projection[2]._forward_hooks), 0)

    def test_failed_update_removes_hook_and_does_not_change_parameters(self):
        model, adapter, opt = objects()
        before = {"model": copy.deepcopy(model.state_dict()), "adapter": copy.deepcopy(adapter.state_dict())}
        batch = clips()
        batch[1]["teacher_features"] = np.zeros_like(batch[1]["teacher_features"])
        with self.assertRaises((ValueError, RuntimeError)):
            subject.optimizer_update(model, adapter, opt, batch, "B_feature", torch.device("cpu"))
        assert_nested_equal(self, before["model"], model.state_dict())
        assert_nested_equal(self, before["adapter"], adapter.state_dict())
        self.assertEqual(len(model.projection[2]._forward_hooks), 0)
        self.assertEqual(len(opt.state), 0)

    def test_controls_preserve_source_pixels_and_have_fixed_order(self):
        pixels = np.arange(17, dtype=np.uint8)[:, None, None, None] * np.ones((1, 96, 96, 3), np.uint8)
        saved = pixels.copy()
        static = subject.perturb_pixels(pixels, "static", "synthetic")
        shuffled = subject.perturb_pixels(pixels, "shuffled", "synthetic")
        shuffled_again = subject.perturb_pixels(pixels, "shuffled", "synthetic")
        normal = subject.perturb_pixels(pixels, "normal", "synthetic")
        self.assertTrue(np.array_equal(normal, pixels))
        self.assertTrue(np.array_equal(pixels, saved))
        self.assertTrue(np.array_equal(static, np.repeat(pixels[8:9], 17, axis=0)))
        self.assertTrue(np.array_equal(shuffled, shuffled_again))
        self.assertEqual(sorted(shuffled[:, 0, 0, 0].tolist()), list(range(17)))
        self.assertFalse(np.array_equal(shuffled, pixels))


if __name__ == "__main__":
    unittest.main(verbosity=2)
