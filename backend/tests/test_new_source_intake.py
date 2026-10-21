"""Stage18 intake boundaries with synthetic files only; no cloud or model access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import intake_new_sources as intake
from models.dataset_manifest import parse_identity, sha256_file
from models.tokenizer import encode_text, tokenizer_metadata


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def source_pair(base, speaker, *, role="Training", angle="A", video=1, sentences=None):
    stem = f"lip_J_1_F_02_{speaker}_{angle}_{video:03d}"
    partition = base / ("1.Training" if role == "Training" else "2.Validation")
    vp = partition / ("TS1" if role == "Training" else "VS1") / (stem + ".mp4")
    lp = partition / ("TL1" if role == "Training" else "VL1") / (stem + ".json")
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_bytes(b"synthetic-video-bytes-never-decoded")
    value = {"speaker_info": {"speaker_ID": speaker}, "Video_env": {"Angle": angle},
             "Video_info": {"video_Name": vp.name, "FPS": 30},
             "Bounding_box_info": {"Lip_bounding_box": {"xtl_ytl_xbr_ybr": [[0, 0, 8, 8]] * 30}},
             "Sentence_info": sentences or [{"ID": 1, "sentence_text": "가", "start_time": 0., "end_time": .4}]}
    save_json(lp, value)
    return vp, lp, value


@pytest.fixture
def inventory_case(tmp_path, monkeypatch):
    cloud, out = tmp_path / "source", tmp_path / "work" / "results/step18-intake/run01"
    new = source_pair(cloud, "C011")
    old = source_pair(cloud, "C001")
    other_angle = source_pair(cloud, "C011", angle="B")
    official = source_pair(cloud, "C099", role="Validation")
    audio = cloud / "1.Training" / "TS1" / "audio.wav"
    audio.write_bytes(b"audio payload must not be opened")
    parent = tmp_path / "work" / "parent.json"
    old_id = "lip_J_1_F_02_C001_A_001_001"
    save_json(parent, {"records": [{"id": old_id, **parse_identity(old_id), "split": "train", "text": "나"}]})
    monkeypatch.setattr(intake, "PARENT", parent)
    monkeypatch.setattr(intake, "ROOT", tmp_path / "work")
    return SimpleNamespace(cloud=cloud, out=out, new=new, old=old, other_angle=other_angle,
                           official=official, audio=audio, parent=parent)


@pytest.mark.parametrize("prefix,part", [("TS|VS", "TS1"), ("TS|VS", "VS12"), ("TL|VL", "TL6"), ("TL|VL", "VL2")])
def test_package_alternatives_include_numeric_suffix(prefix, part):
    assert intake.package(Path("root") / part / "file.mp4", prefix) == part


@pytest.mark.parametrize("parts", [("TS",), ("TSx",), ("TS1", "VS2"), ("nestedTS1",)])
def test_package_rejects_missing_or_ambiguous_match(parts):
    with pytest.raises(ValueError, match="package"):
        intake.package(Path(*parts) / "file.mp4", "TS|VS")


def test_json_writer_never_overwrites_existing_artifact(tmp_path):
    path = tmp_path / "artifact.json"
    intake.write_json(path, {"original": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        intake.write_json(path, {"replacement": True})
    assert path.read_bytes() == original


def test_inventory_uses_file_metadata_without_reading_any_source_payload(inventory_case, monkeypatch):
    case = inventory_case
    real_open = Path.open
    opened = []
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.cloud):
            opened.append(path)
            raise AssertionError("Inventory opened source payload")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    intake.inventory(case.cloud, case.out)
    result = intake.read_json(case.out / "inventory.json")
    assert not opened and result["payloads_read"] is False
    assert result["partitions"]["Training"]["summary"]["videos"] == 3
    assert result["partitions"]["Training"]["summary"]["audio"] == 1
    assert result["partitions"]["Validation"]["summary"]["videos"] == 1


@pytest.mark.parametrize("fault", ["missing_pair", "duplicate", "package"])
def test_inventory_rejects_unpaired_duplicate_or_mismatched_package(inventory_case, fault):
    case = inventory_case
    if fault == "missing_pair":
        case.new[1].unlink()
    elif fault == "duplicate":
        copy_path = case.new[0].parent / "nested" / case.new[0].name
        copy_path.parent.mkdir()
        copy_path.write_bytes(case.new[0].read_bytes())
    else:
        moved = case.new[1].parent.parent / "TL2" / case.new[1].name
        moved.parent.mkdir()
        case.new[1].rename(moved)
    with pytest.raises(ValueError, match="pair|Duplicate|Package"):
        intake.inventory(case.cloud, case.out)
    assert not (case.out / "inventory.json").exists()


def test_index_reads_only_new_frontal_training_labels_and_never_source_writes(inventory_case, monkeypatch):
    case = inventory_case
    intake.inventory(case.cloud, case.out)
    real_open = Path.open
    reads = []
    def guarded(path, mode="r", *args, **kwargs):
        if path.is_relative_to(case.cloud):
            assert mode in ("r", "rb"), "Source write attempted"
            assert path == case.new[1], "Excluded source payload opened"
            reads.append(path)
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    intake.index(case.out)
    result = intake.read_json(case.out / "source_index.json")
    assert reads == [case.new[1]]
    assert {r["speaker_id"] for r in result["candidates"]} == {"C011"}
    assert result["excluded_existing_speakers"] == ["C001"]
    assert result["official_validation_speakers_reserved"] == ["C099"]
    local = Path(result["sources"][0]["local_label"])
    assert local.read_bytes() == json.dumps(case.new[2], ensure_ascii=False).encode("utf-8")
    assert sha256_file(local) == result["sources"][0]["label_sha256"]


def test_index_rejects_changed_label_metadata_before_read(inventory_case, monkeypatch):
    case = inventory_case
    intake.inventory(case.cloud, case.out)
    case.new[1].write_bytes(case.new[1].read_bytes() + b" ")
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path == case.new[1]:
            raise AssertionError("Changed label was read before metadata gate")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="Label metadata changed"):
        intake.index(case.out)
    assert not (case.out / "source_index.json").exists()


@pytest.mark.parametrize("fault", ["string_id", "duplicate_id", "speaker", "angle", "name", "fps", "boxes", "multi_label"])
def test_index_invalid_source_cannot_produce_completed_index(inventory_case, fault):
    case = inventory_case
    value = copy.deepcopy(case.new[2])
    if fault == "string_id":
        value["Sentence_info"][0]["ID"] = "1"
    elif fault == "duplicate_id":
        value["Sentence_info"].append(copy.deepcopy(value["Sentence_info"][0]))
    elif fault == "speaker":
        value["speaker_info"]["speaker_ID"] = "C099"
    elif fault == "angle":
        value["Video_env"]["Angle"] = "B"
    elif fault == "name":
        value["Video_info"]["video_Name"] = "different.mp4"
    elif fault == "fps":
        value["Video_info"]["FPS"] = 0
    elif fault == "boxes":
        value["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"] = []
    else:
        value = [value, copy.deepcopy(value)]
    save_json(case.new[1], value)
    intake.inventory(case.cloud, case.out)
    with pytest.raises(ValueError):
        intake.index(case.out)
    assert not (case.out / "source_index.json").exists()


def test_new_training_speaker_overlap_official_validation_fails_before_payload_read(inventory_case, monkeypatch):
    case = inventory_case
    source_pair(case.cloud, "C011", role="Validation")
    intake.inventory(case.cloud, case.out)
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.cloud):
            raise AssertionError("Overlap was not rejected before source read")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="overlap official Validation"):
        intake.index(case.out)
    assert not (case.out / "source_index.json").exists()


def test_frozen_hash_change_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    path = tmp_path / "frozen.json"
    path.write_bytes(b"original")
    document = {"frozen_files": {"frozen.json": sha256_file(path)}}
    intake.verify_frozen(document)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        intake.verify_frozen(document)


def test_cli_rejects_output_outside_stage18_before_any_command(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["intake", "prepare", "--out", str(tmp_path / "old-dataset")])
    monkeypatch.setattr(intake, "prepare", lambda _: pytest.fail("Unsafe output reached prepare"))
    with pytest.raises(ValueError, match="Stage18 results directory required"):
        intake.main()


@pytest.fixture
def prepared_case(inventory_case, monkeypatch):
    """Mock only deterministic selector and video decoding; preserve real I/O checks."""
    import cv2
    import prepare_data
    case = inventory_case
    source_pair(case.cloud, "C012")
    intake.inventory(case.cloud, case.out)
    intake.index(case.out)
    save_json(case.out / "protected_reference.json", {"status": "synthetic metadata only"})
    previous = intake.ROOT / "previous"
    monkeypatch.setattr(intake, "PREVIOUS", previous)
    save_json(previous / "plan.json", {"cohorts": {"train60": [], "val20": []}, "manifest_sha256": sha256_file(case.parent)})
    save_json(previous / "completed.json", {"status": "synthetic"})
    save_json(previous / "quality_checks.json", {"status": "synthetic"})
    save_json(intake.ROOT / "docs/단계별_개선/17_검증결과.json", {"status": "synthetic"})
    for name in ("scripts/intake_new_sources.py", "scripts/intake_selection.py", "scripts/prepare_data.py",
                 "backend/models/cache.py", "backend/models/tokenizer.py", "backend/models/dataset_manifest.py",
                 "backend/preprocessing/roi_extractor.py", "backend/preprocessing/resampler.py", "backend/config.py"):
        path = intake.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic frozen source fixture\n", encoding="utf-8")
    def selector(candidates, old_records, protected):
        rows = copy.deepcopy(candidates)
        for row in rows:
            row["split"] = "train" if row["speaker_id"] == "C011" else "val"
        return {"records": rows, "speaker_mapping": {"C011": "train", "C012": "val"},
                "counts": {"train": 1, "val": 1}}
    monkeypatch.setitem(sys.modules, "intake_selection", SimpleNamespace(select_intake=selector))
    intake.plan(case.out)
    case.document = intake.read_json(case.out / "selection.json")
    case.calls = []
    case.fault = None
    case.after_prepare = None
    class Capture:
        def isOpened(self):
            return True
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_COUNT: 30.,
                    cv2.CAP_PROP_FRAME_WIDTH: 96., cv2.CAP_PROP_FRAME_HEIGHT: 96.}[prop]
        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    monkeypatch.setattr(intake.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    def preparer(video, label, out, **kwargs):
        case.calls.append((video, label, copy.deepcopy(kwargs)))
        assert kwargs["roi_mode"] == "bbox" and kwargs["max_frames"] == 200
        assert kwargs["bbox_coordinate_order"] == "yxyx"
        raw = intake.read_json(Path(label))
        for sentence in raw["Sentence_info"]:
            if sentence["ID"] not in kwargs["sentence_ids"]:
                continue
            sid = sentence["ID"]
            values = {"frames": np.full((10, 96, 96, 3), 55, dtype=np.uint8),
                      "label": np.asarray(encode_text(sentence["sentence_text"]), dtype=np.int64),
                      "text": sentence["sentence_text"], "source_text": sentence["sentence_text"],
                      **kwargs["provenance"][sid], **tokenizer_metadata(),
                      "temporal_method": intake.TEMPORAL_METHOD, "video_fps": 30., "label_fps": 30.}
            if case.fault == "missing_timing":
                values.pop("temporal_method")
            elif case.fault == "fps":
                values["video_fps"] = 29.97
            elif case.fault == "provenance":
                values["speaker_id"] = "C999"
            elif case.fault == "frames":
                values["frames"] = values["frames"][:-1]
            elif case.fault == "encoded_label":
                values["label"] = np.asarray(encode_text("다"))
            path = Path(out) / f"{Path(video).stem}_{sid:03d}.npz"
            if case.fault != "missing_file":
                np.savez_compressed(path, **values)
        if case.after_prepare:
            case.after_prepare()
        if case.fault == "interrupted":
            raise RuntimeError("Synthetic interrupted preparation")
        return 0 if case.fault == "short_count" else len(kwargs["sentence_ids"])
    monkeypatch.setattr(prepare_data, "prepare_video", preparer)
    return case


def test_prepare_outputs_verified_cache_receipts_and_pending_visual_status(prepared_case, monkeypatch):
    case = prepared_case
    source_bytes = {Path(r["source_video"]["path"]): Path(r["source_video"]["path"]).read_bytes()
                    for r in case.document["records"]}
    frozen = dict(case.document["frozen_files"])
    real_open = Path.open
    reads = []
    def guarded(path, mode="r", *args, **kwargs):
        if path.is_relative_to(case.cloud):
            assert mode == "rb" and path in source_bytes, "Unselected or writable source access"
            reads.append(path)
        return real_open(path, mode, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(Path, "open", guarded)
        intake.prepare(case.out)
    assert set(reads) == set(source_bytes)
    result = intake.read_json(case.out / "intake_manifest.json")
    assert result["status"] == "prepared_pending_visual_review"
    assert len(result["records"]) == len(case.calls) == len(result["source_receipts"]) == 2
    assert result["scope"]["training"] is result["scope"]["model_forward"] is False
    assert result["selection_sha256"] == sha256_file(case.out / "selection.json")
    for row in result["records"]:
        assert sha256_file(case.out / row["cache_path"]) == row["cache_sha256"]
        assert row["video_fps"] == row["label_fps"] == 30.
        assert row["frames"] == 10
    for receipt in result["source_receipts"]:
        assert sha256_file(receipt["local_path"]) == receipt["sha256"]
        assert (case.out / "receipts" / (Path(receipt["local_path"]).stem + ".json")).is_file()
    for path, value in source_bytes.items():
        assert path.read_bytes() == value
    for relative, digest in frozen.items():
        assert sha256_file(intake.ROOT / relative) == digest


@pytest.mark.parametrize("fault", ["short_count", "interrupted", "missing_file", "missing_timing", "fps", "provenance", "frames", "encoded_label"])
def test_partial_or_invalid_preparation_cannot_publish_manifest(prepared_case, fault):
    case = prepared_case
    case.fault = fault
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError, KeyError)):
        intake.prepare(case.out)
    assert not (case.out / "intake_manifest.json").exists()
    assert len(case.calls) == 1
    before = {p: p.read_bytes() for p in (case.out / "clips").glob("*.npz")}
    case.fault = None
    with pytest.raises(FileExistsError):
        intake.prepare(case.out)
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("fault", ["source_path", "split", "id", "time", "scope", "copy_budget", "label_path", "schema", "tokenizer", "missing_frozen"])
def test_selection_tampering_fails_before_any_source_or_cache_access(prepared_case, monkeypatch, fault):
    case = prepared_case
    value = copy.deepcopy(case.document)
    if fault == "source_path":
        value["records"][0]["source_video"]["path"] = str(case.official[0])
    elif fault == "split":
        value["records"][0]["split"] = "test"
    elif fault == "id":
        value["records"][0]["id"] = "../escape"
    elif fault == "time":
        value["records"][0]["start_time"] = .1
    elif fault == "scope":
        value["scope"]["test_payload_access"] = True
    elif fault == "copy_budget":
        value["video_bytes_to_copy"] += 1
    elif fault == "label_path":
        value["records"][0]["label_path"] = str(case.official[1])
    elif fault == "schema":
        value["schema"] = "wrong-schema"
    elif fault == "tokenizer":
        value["tokenizer"] = {}
    else:
        value["frozen_files"] = {}
    save_json(case.out / "selection.json", value)
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.cloud) or path.suffix == ".npz":
            raise AssertionError("Rejected selection opened payload")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError):
        intake.prepare(case.out)
    assert not case.calls and not (case.out / "clips").exists()
    assert not (case.out / "intake_manifest.json").exists()


@pytest.mark.parametrize("fault", ["video", "label", "frozen"])
def test_changed_inputs_cannot_reach_preparer(prepared_case, monkeypatch, fault):
    case = prepared_case
    row = case.document["records"][0]
    path = Path(row["source_video"]["path"] if fault == "video" else row["label_path"] if fault == "label" else case.parent)
    path.write_bytes(path.read_bytes() + b" ")
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.cloud):
            raise AssertionError("Changed input reached source read")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="changed|changed|Changed"):
        intake.prepare(case.out)
    assert not case.calls and not (case.out / "intake_manifest.json").exists()


def test_selection_change_during_preparation_cannot_publish_manifest(prepared_case):
    case = prepared_case
    changed = False
    def mutate_once():
        nonlocal changed
        if not changed:
            path = case.out / "selection.json"
            path.write_bytes(path.read_bytes() + b" ")
            changed = True
    case.after_prepare = mutate_once
    with pytest.raises(ValueError, match="Selection changed during preparation"):
        intake.prepare(case.out)
    assert case.calls and not (case.out / "intake_manifest.json").exists()


def test_completed_manifest_cannot_be_overwritten(prepared_case):
    case = prepared_case
    intake.prepare(case.out)
    path = case.out / "intake_manifest.json"
    original = path.read_bytes()
    calls = len(case.calls)
    with pytest.raises(ValueError, match="already completed"):
        intake.prepare(case.out)
    assert path.read_bytes() == original and len(case.calls) == calls


def test_frozen_path_cannot_escape_workspace_before_open(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(intake, "ROOT", work)
    outside = tmp_path / "not-allowed.bin"
    outside.write_bytes(b"must not be opened")
    document = {"frozen_files": {"../not-allowed.bin": "wrong"}}
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.resolve() == outside.resolve():
            raise AssertionError("Escaped artifact was opened")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError):
        intake.verify_frozen(document)


def test_inadequate_disk_budget_stops_before_source_read(prepared_case, monkeypatch):
    case = prepared_case
    monkeypatch.setattr(intake.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    real_open = Path.open
    def guarded(path, *args, **kwargs):
        if path.is_relative_to(case.cloud):
            raise AssertionError("Source read despite inadequate local space")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="Insufficient free disk"):
        intake.prepare(case.out)
    assert not case.calls and not (case.out / "intake_manifest.json").exists()
