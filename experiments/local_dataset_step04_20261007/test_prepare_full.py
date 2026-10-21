"""Focused storage, provenance and resume checks for the full input stage."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_full as full


def example(split: str = "dev") -> tuple[dict, dict, object, dict]:
    """Make synthetic paired pixels and a truthful transport envelope."""
    try:
        from adapter import load_producer
        producer = load_producer()
        row = {"id": "synthetic_001", "source_stem": "synthetic", "split": split,
               "frames": 2, "video_fps": 25.0, "start_time": 0.0, "end_time": .08,
               "token_ids": [1], "label_sha256": "a" * 64}
        runner = {"path": "synthetic.py", "sha256": "b" * 64}
        members = {"video": {"size": 20}, "label": {"sha256": row["label_sha256"]}, "audio": {"size": 10}}
        adapter = SimpleNamespace(producer=producer, sources={"synthetic": {"members": members}},
            parent={"sha256": "parent"}, plan_path=Path("source.json"), plan_sha256="source", code=[])
        provenance = {"transport": "audited-local-tar-v1", "source_plan_path": "source.json",
            "source_plan_sha256": "source", "parent_plan_sha256": "parent", "source_stem": "synthetic",
            "members": {k: members[k] for k in ("video", "label")}, "transport_code": [],
            "audio_used_as_model_input": False, "temporary_paths_persisted": False,
            "archive_stats_checked_before_and_after": True, "member_hashes_checked_on_materialization": True,
            "temporary_payload_verification": {"video_size_bytes": 20, "video_mtime_ns": 1,
                "label_sha256_before_and_after": row["label_sha256"], "source_video_unchanged": True}}
        contract = producer.interval_contract(0.0, .08, 25.0, 10)
        student = np.ones((2, 96, 96, 3), dtype=np.uint8)
        teacher = np.ones((len(contract["teacher_indices"]), 96, 96, 3), dtype=np.uint8)
        traces = [{"source_frame_index": int(i), "source_bgr_pixel_sha256": str(i)} for i in contract["teacher_indices"]]
        meta = {"schema": "feature-transfer-step07-input-v1", "id": row["id"], "record": row,
            "plan_sha256": "parent", "source_frame_count": 10,
            "student_rgb_sha256": hashlib.sha256(student.tobytes()).hexdigest(),
            "teacher_bgr_sha256": hashlib.sha256(teacher.tobytes()).hexdigest(),
            "student_trace": {"selected": traces[:2]}, "teacher_trace": {"selected": traces},
            "teacher_context": {}, "preparation_seconds": 1.0, "source_verification": provenance,
            "local_full_preparation": runner}
        arrays = {"student_rgb": student, "teacher_bgr": teacher,
            "student_source_indices": contract["student_indices"], "teacher_source_indices": contract["teacher_indices"],
            "student_teacher_rows": contract["student_teacher_rows"], "label_tokens": np.array([1], dtype=np.int64)}
        return row, {"arrays": arrays, "meta": meta}, adapter, runner
    except Exception:
        raise


def test_dev_persists_only_student_and_resumes(tmp_path: Path) -> None:
    """Evaluation data contain no teacher storage and preserve exact resumed bytes."""
    try:
        row, value, adapter, runner = example()
        saved = full.write_dev(tmp_path, row, full.dev_value(row, value), adapter.producer)
        with np.load(saved["npz_path"], allow_pickle=False) as payload:
            assert set(payload.files) == full.DEV_ARRAYS | {"metadata_json"}
            assert set(json.loads(str(payload["metadata_json"].item()))) == full.DEV_META
        assert saved["teacher_frames"] == 0 and "teacher_context" not in saved
        assert full.saved_record(tmp_path, row, adapter, runner) == saved
    except Exception:
        raise


def test_train_retains_paired_schema_and_resumes(tmp_path: Path) -> None:
    """New training arrays remain readable by the unchanged teacher primitive."""
    try:
        row, value, adapter, runner = example("train")
        entry = adapter.producer.write_record(tmp_path, row, value)
        assert entry["teacher_frames"] == 4
        assert full.saved_record(tmp_path, row, adapter, runner) == entry
        with pytest.raises(RuntimeError):
            full.dev_value(row, value)
    except Exception:
        raise


def test_pilot_reference_is_exact_and_never_rewritten(tmp_path: Path) -> None:
    """An accepted pilot dictionary/byte hash is mandatory when reused."""
    try:
        row, value, adapter, runner = example("train")
        value["meta"]["local_pilot_preparation"] = value["meta"].pop("local_full_preparation")
        entry = adapter.producer.write_record(tmp_path, row, value)
        before = Path(entry["npz_path"]).read_bytes()
        assert full.saved_record(tmp_path, row, adapter, runner, entry) == entry
        assert Path(entry["npz_path"]).read_bytes() == before
        changed = dict(entry, student_frames=3)
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, runner, changed)
        (tmp_path / "metadata" / (row["id"] + ".json")).unlink()
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, runner, entry)
        assert not (tmp_path / "metadata" / (row["id"] + ".json")).exists()
    except Exception:
        raise


@pytest.mark.parametrize("kind", ["teacher_array", "pixels", "record", "indices", "labels"])
def test_dev_rejects_corruption(tmp_path: Path, kind: str) -> None:
    """Meaningful array/identity corruption cannot pass a resumed development read."""
    try:
        row, value, adapter, _ = example()
        value = full.dev_value(row, value)
        if kind == "teacher_array":
            value["arrays"]["teacher_bgr"] = np.zeros((1,), dtype=np.uint8)
        elif kind == "pixels":
            value["arrays"]["student_rgb"][0, 0, 0, 0] = 255
        elif kind == "record":
            value["meta"]["id"] = "other"
        elif kind == "indices":
            value["arrays"]["student_source_indices"][0] = 9
        else:
            value["arrays"]["label_tokens"][0] = 2
        path = tmp_path / "bad.npz"
        np.savez_compressed(path, **value["arrays"], metadata_json=json.dumps(value["meta"]))
        with pytest.raises(RuntimeError):
            full.verify_dev(path, row, "parent", adapter.producer)
    except Exception:
        raise


def test_resume_rejects_changed_runner_and_source(tmp_path: Path) -> None:
    """Pixel validity alone cannot override changed transport or program bindings."""
    try:
        row, value, adapter, runner = example()
        full.write_dev(tmp_path, row, full.dev_value(row, value), adapter.producer)
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, dict(runner, sha256="changed"))
        adapter.sources["synthetic"]["members"]["video"]["size"] = 21
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, runner)
    except Exception:
        raise


def test_recover_missing_new_sidecar_but_not_pilot_sidecar(tmp_path: Path) -> None:
    """Resume heals interrupted new publication, while accepted pilots stay immutable."""
    try:
        row, value, adapter, runner = example()
        entry = full.write_dev(tmp_path, row, full.dev_value(row, value), adapter.producer)
        sidecar = tmp_path / "metadata" / (row["id"] + ".json")
        sidecar.unlink()
        assert full.saved_record(tmp_path, row, adapter, runner) == entry
        changed = dict(entry, student_frames=3)
        sidecar.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, runner)
    except Exception:
        raise


def test_missing_accepted_pilot_is_not_recreated(tmp_path: Path) -> None:
    """A missing accepted input is a hard failure, not permission to rebuild it."""
    try:
        row, _, adapter, runner = example("train")
        assert full.saved_record(tmp_path, row, adapter, runner) is None
        with pytest.raises(RuntimeError):
            full.saved_record(tmp_path, row, adapter, runner, {"id": row["id"]})
    except Exception:
        raise
