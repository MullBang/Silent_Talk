"""Behavioral tests for immutable logical records across temporary transport."""
from __future__ import annotations

import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapter import LocalSourceAdapter


class AdapterTests(unittest.TestCase):
    """No real dataset, original Drive path or model is opened by these tests."""

    def setUp(self) -> None:
        self.row = {"id": "sample_001", "source_stem": "sample", "split": "train",
                    "text": "원본", "start_time": 1.2, "end_time": 2.3,
                    "video_path": "G:/never-open.mp4", "label_path": "G:/never-open.json",
                    "video_size_bytes": 123, "video_mtime_ns": 456}
        self.adapter = object.__new__(LocalSourceAdapter)
        self.adapter.plan_path = Path("test-source-plan.json")
        self.adapter.plan_sha256 = "d" * 64
        self.adapter.parent = {"sha256": "p" * 64}
        self.adapter.rows = {self.row["id"]: copy.deepcopy(self.row)}
        self.adapter.sources = {"sample": {"stem": "sample", "members": {
            role: {"sha256": role} for role in ("video", "label", "audio")}}}
        self.adapter.temp_root = Path("unused-test-temp")
        self.adapter.code = [{"sha256": "code"}]
        self.seen_roles = None
        self.temp_path = None
        self.code_patch = patch("adapter.code_bindings", return_value=self.adapter.code)
        self.materializer_patch = patch("adapter.materialize_source", side_effect=self.materialize)
        self.code_patch.start()
        self.materializer_patch.start()
        self.addCleanup(self.code_patch.stop)
        self.addCleanup(self.materializer_patch.stop)

    @contextmanager
    def materialize(self, source: dict, root: Path):
        self.seen_roles = set(source["members"])
        with tempfile.TemporaryDirectory() as directory:
            self.temp_path = Path(directory)
            paths = {role: self.temp_path / role for role in source["members"]}
            for path in paths.values():
                path.write_bytes(b"test payload")
            yield paths

    def producer(self, rows: list[dict], parent_sha: str, *, reproduction: bool):
        self.assertEqual(parent_sha, "p" * 64)
        self.assertTrue(Path(rows[0]["video_path"]).is_file())
        self.assertNotEqual(rows[0]["video_mtime_ns"], 456)
        self.assertEqual(rows[0]["video_size_bytes"], 12)
        self.assertEqual(rows[0]["start_time"], 1.2)
        metadata = {"id": rows[0]["id"], "record": rows[0],
                    "source_verification": {"video_mtime_ns": rows[0]["video_mtime_ns"]}}
        return [(rows[0], metadata if reproduction else {"meta": metadata, "arrays": {}})]

    def test_restores_logical_record_and_keeps_truthful_transport(self) -> None:
        self.adapter.producer = SimpleNamespace(prepare_source=self.producer)
        original = copy.deepcopy(self.row)
        output = self.adapter.prepare_source([self.row], reproduction=False)
        self.assertEqual(self.row, original)
        self.assertEqual(output[0][0], original)
        metadata = output[0][1]["meta"]
        self.assertEqual(metadata["record"], original)
        self.assertEqual(self.seen_roles, {"video", "label"})
        self.assertEqual(metadata["source_verification"]["transport"], "audited-local-tar-v1")
        self.assertNotEqual(metadata["source_verification"]["temporary_payload_verification"]["video_mtime_ns"], 456)
        self.assertFalse(self.temp_path.exists())

    def test_audio_is_materialized_only_when_requested(self) -> None:
        self.adapter.producer = SimpleNamespace(prepare_source=self.producer)
        output = self.adapter.prepare_source([self.row], reproduction=True, include_audio=True)
        self.assertEqual(self.seen_roles, {"video", "label", "audio"})
        self.assertFalse(output[0][1]["source_verification"]["audio_used_as_model_input"])
        self.assertEqual(set(self.adapter.sources["sample"]["members"]), self.seen_roles)

    def test_rejects_changed_text_time_or_split_before_reading(self) -> None:
        for key, value in (("text", "변조"), ("start_time", 1.3), ("split", "dev")):
            with self.subTest(key=key):
                changed = dict(self.row, **{key: value})
                with self.assertRaises(RuntimeError):
                    self.adapter.prepare_source([changed], reproduction=True)
        self.assertIsNone(self.seen_roles)

    def test_rejects_duplicate_and_multiple_source_batches(self) -> None:
        with self.assertRaises(RuntimeError):
            self.adapter.prepare_source([self.row, self.row], reproduction=True)
        other = dict(self.row, id="other_001", source_stem="other")
        self.adapter.rows[other["id"]] = other
        with self.assertRaises(RuntimeError):
            self.adapter.prepare_source([self.row, other], reproduction=True)
        self.assertIsNone(self.seen_roles)

    def test_producer_failure_cleans_temporary_source(self) -> None:
        def fail(*args, **kwargs):
            raise ValueError("Simulated decode failure")
        self.adapter.producer = SimpleNamespace(prepare_source=fail)
        with self.assertRaises(RuntimeError):
            self.adapter.prepare_source([self.row], reproduction=True)
        self.assertFalse(self.temp_path.exists())

    def test_rejects_changed_code_after_preparation(self) -> None:
        self.adapter.producer = SimpleNamespace(prepare_source=self.producer)
        with patch("adapter.code_bindings", return_value=[]):
            with self.assertRaises(RuntimeError):
                self.adapter.prepare_source([self.row], reproduction=True)
        self.assertFalse(self.temp_path.exists())


if __name__ == "__main__":
    unittest.main()
