"""CPU contract tests for fixed-budget execution and on-disk restoration."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

import runner as subject


class TinyStudent(nn.Module):
    """Exercise live projection gradients and dropout at inexpensive CPU cost."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(3, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.1))
        self.fc = nn.Linear(256, 69)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(pixels.mean(dim=(-1, -2)).transpose(1, 2))
        return self.fc(hidden).log_softmax(-1).transpose(0, 1).contiguous()


def tiny_objects(seed: int, arm: str, device: torch.device) -> tuple:
    """Use the production optimizer/initialization ordering with a tiny student."""
    subject.core.seed_all(seed)
    model, adapter = TinyStudent().to(device), nn.Linear(256, 512).to(device)
    parameters = list(model.parameters()) + (list(adapter.parameters()) if arm == "B_feature" else [])
    optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    return model, adapter, optimizer


def synthetic_clips() -> dict:
    """Supply unequal-length clips with fixed nonzero teacher targets."""
    generator = np.random.default_rng(31)
    return {f"clip_{i}": {"id": f"clip_{i}", "split": "train",
            "pixels": generator.integers(0, 256, (length, 96, 96, 3), dtype=np.uint8),
            "tokens": [1, 2], "teacher_features": generator.normal(size=(length, 512)).astype(np.float32)}
            for i, length in enumerate((4, 6, 7, 9))}


class RunnerTests(unittest.TestCase):
    """Check actual serialization, replay and fail-closed provenance behavior."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="step03_runner_test_")
        self.base = Path(self.temp.name)
        self.clips = synthetic_clips()
        self.plan = {"seed": 42, "updates": 4, "batch_size": 2, "checkpoint_interval": 1,
                     "schedule": [["clip_0", "clip_1"], ["clip_2", "clip_3"],
                                  ["clip_1", "clip_3"], ["clip_2", "clip_0"]]}
        self.plan["plan_sha256"] = subject.canonical_plan_hash(self.plan)
        self.patch = mock.patch.object(subject.core, "make_objects", side_effect=tiny_objects)
        self.patch.start()
        self.output_buffer = io.StringIO()
        self.redirect = redirect_stdout(self.output_buffer)
        self.redirect.__enter__()

    def tearDown(self) -> None:
        self.redirect.__exit__(None, None, None)
        self.patch.stop()
        self.temp.cleanup()

    def run_arm(self, arm: str = "B_feature", name: str = "original", **kwargs) -> dict:
        """Execute the real runner and on-disk checkpoints using CPU stand-ins."""
        return subject.run_segment(self.plan, self.clips, arm, self.base / name, device="cpu", **kwargs)

    def test_saved_resume_reproduces_next_update_for_each_arm(self) -> None:
        for arm in subject.core.ARMS:
            with self.subTest(arm=arm):
                original = self.run_arm(arm, arm + "_original")
                resume = self.run_arm(arm, arm + "_resume",
                                      resume_checkpoint=self.base / (arm + "_original") / "update_0003.pt")
                self.assertEqual(resume["start_update"], 3)
                self.assertEqual([row["update"] for row in resume["updates"]], [4])
                self.assertTrue(resume["restore_verification"]["exact"])
                self.assertEqual(original["final_state_hashes"], resume["final_state_hashes"])
                for key in ("loss", "ctc_loss", "feature_loss", "gradient_norm_before_clipping"):
                    self.assertEqual(original["updates"][-1][key], resume["updates"][0][key])
                restored = subject.load_checkpoint(Path(resume["checkpoint"]["path"]))
                self.assertEqual(len(restored["history"]), 4)
                self.assertEqual(restored["completed_updates"], 4)

    def test_pair_initialization_and_arm_specific_gradient_flow(self) -> None:
        a = self.run_arm("A_ctc", "A", stop_update=1)
        b = self.run_arm("B_feature", "B", stop_update=1)
        for component in ("model", "adapter", "rng"):
            self.assertEqual(a["initial_state_hashes"][component], b["initial_state_hashes"][component])
        self.assertNotEqual(a["initial_state_hashes"]["optimizer"], b["initial_state_hashes"]["optimizer"])
        self.assertEqual(a["updates"][0]["feature_loss"], 0.)
        self.assertFalse(a["updates"][0]["adapter_changed"])
        self.assertTrue(b["updates"][0]["adapter_changed"])
        for result in (a, b):
            row = result["updates"][0]
            self.assertTrue(row["student_changed"])
            self.assertLessEqual(row["gradient_norm_after_clipping"], 1.0001)
            self.assertEqual(row["feature_hook_count_before"], row["feature_hook_count_after"])

    def test_rng_serialization_uses_restricted_loader_and_exact_types(self) -> None:
        result = self.run_arm(stop_update=1)
        path = Path(result["checkpoint"]["path"])
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self.assertTrue(raw["state"]["rng"]["numpy"][1][subject.NUMPY_MARKER])
        decoded = subject.load_checkpoint(path)
        self.assertIsInstance(decoded["state"]["rng"]["numpy"][1], np.ndarray)
        self.assertEqual(decoded["state"]["rng"]["numpy"][1].dtype, np.uint32)
        self.assertEqual(subject.state_hashes(decoded["state"]), decoded["state_hashes"])

    def test_checkpoint_byte_tamper_rejected_before_torch_load(self) -> None:
        result = self.run_arm(stop_update=1)
        path = Path(result["checkpoint"]["path"])
        with path.open("r+b") as stream:
            stream.seek(100)
            old = stream.read(1)
            stream.seek(100)
            stream.write(bytes([old[0] ^ 1]))
        with mock.patch.object(torch, "load") as loader:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                subject.load_checkpoint(path)
            loader.assert_not_called()

    def test_internal_state_digest_detects_inconsistent_checkpoint(self) -> None:
        result = self.run_arm(stop_update=1)
        payload = subject.load_checkpoint(Path(result["checkpoint"]["path"]))
        payload["state"]["adapter"]["bias"][0] += 1
        new_output = self.base / "inconsistent"
        new_output.mkdir()
        info = subject.save_checkpoint(new_output, payload)
        with self.assertRaisesRegex(ValueError, "state digest mismatch"):
            subject.load_checkpoint(Path(info["path"]))

    def test_resume_rejects_changed_pixels_and_does_not_create_output(self) -> None:
        result = self.run_arm(stop_update=1)
        self.clips["clip_0"]["pixels"][0, 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
            self.run_arm(name="changed", resume_checkpoint=Path(result["checkpoint"]["path"]))
        self.assertFalse((self.base / "changed").exists())

    def test_resume_rejects_other_arm_and_changed_plan(self) -> None:
        result = self.run_arm(stop_update=1)
        checkpoint = Path(result["checkpoint"]["path"])
        with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
            self.run_arm("A_ctc", "other_arm", resume_checkpoint=checkpoint)
        self.plan["seed"] = 43
        self.plan["plan_sha256"] = subject.canonical_plan_hash(self.plan)
        with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
            self.run_arm(name="other_plan", resume_checkpoint=checkpoint)

    def test_unknown_schedule_or_nontrain_clip_fails_before_output(self) -> None:
        self.plan["schedule"][0][0] = "missing_clip"
        with self.assertRaisesRegex(ValueError, "Unknown or malformed"):
            self.run_arm()
        self.assertFalse((self.base / "original").exists())
        self.plan["schedule"][0][0] = "clip_0"
        self.clips["clip_3"]["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "Non-training clip"):
            self.run_arm()

    def test_refuses_stale_plan_digest_and_duplicate_output(self) -> None:
        self.plan["seed"] = 43
        with self.assertRaisesRegex(ValueError, "Plan SHA-256 mismatch"):
            self.run_arm()
        self.plan["seed"] = 42
        result = self.run_arm(stop_update=1)
        saved_bytes = Path(result["checkpoint"]["path"]).read_bytes()
        with self.assertRaisesRegex(ValueError, "Output must be empty"):
            self.run_arm(stop_update=1)
        self.assertEqual(Path(result["checkpoint"]["path"]).read_bytes(), saved_bytes)

    def test_saved_cursor_cannot_be_repeated_or_moved_backwards(self) -> None:
        result = self.run_arm(stop_update=2)
        checkpoint = Path(result["checkpoint"]["path"])
        for stop in (1, 2):
            with self.subTest(stop=stop), self.assertRaisesRegex(ValueError, "must advance"):
                self.run_arm(name="backwards", stop_update=stop, resume_checkpoint=checkpoint)
        for stop in (0, 5):
            with self.subTest(stop=stop), self.assertRaisesRegex(ValueError, "Invalid stop"):
                self.run_arm(name="bad_stop", stop_update=stop)

    def test_checkpoint_interval_saves_boundaries_and_segment_end(self) -> None:
        self.plan["checkpoint_interval"] = 3
        self.plan["plan_sha256"] = subject.canonical_plan_hash(self.plan)
        result = self.run_arm()
        self.assertEqual([row["completed_updates"] for row in result["checkpoints"]], [3, 4])
        self.assertFalse((self.base / "original" / "update_0001.pt").exists())
        self.assertTrue((self.base / "original" / "segment_result.json").is_file())

    def test_failure_removes_feature_hook_and_publishes_no_checkpoint(self) -> None:
        model, adapter, optimizer = tiny_objects(42, "B_feature", torch.device("cpu"))
        original_forward = model.forward
        calls = 0

        def fail_second_clip(value: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interrupted forward")
            return original_forward(value)

        with mock.patch.object(subject.core, "make_objects", return_value=(model, adapter, optimizer)), \
                mock.patch.object(model, "forward", side_effect=fail_second_clip):
            with self.assertRaisesRegex(RuntimeError, "interrupted forward"):
                self.run_arm(stop_update=1)
        self.assertEqual(len(model.projection[2]._forward_hooks), 0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(list((self.base / "original").glob("*.pt")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
