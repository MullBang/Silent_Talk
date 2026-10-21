"""화자/발화 누수 방지와 실제 학습·평가 집합 선택을 검사한다."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from models.dataset_manifest import (
    SCHEMA, assign_speakers, load_manifest, parse_identity, sha256_file, validate_manifest,
)
from models.tokenizer import encode_text, tokenizer_metadata
from build_speaker_dataset import select_pilot
import train
import test_model as evaluation


@pytest.fixture
def dataset(tmp_path):
    rows = []
    for i, split in enumerate(("train", "val", "test"), 1):
        sample_id = f"lip_J_1_F_02_C{i:03d}_A_001_001"
        row = {"id": sample_id, **parse_identity(sample_id), "split": split,
               "source_partition": "Validation" if split == "test" else "Training",
               "text": "값이 없다", "source_text": "값이 없다.", "frames": 30,
               "cache_path": f"{sample_id}.npz", "roi_method": "aihub-bbox-yxyx-v2", "label_sha256": "label-test"}
        keys = ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition", "roi_method", "label_sha256")
        path = tmp_path / row["cache_path"]
        np.savez_compressed(path, frames=np.full((30, 96, 96, 3), i, dtype=np.uint8),
                            label=np.asarray(encode_text(row["text"])), text=row["text"],
                            source_text=row["source_text"], **{k: row[k] for k in keys}, **tokenizer_metadata())
        row["cache_sha256"] = sha256_file(path)
        rows.append(row)
    document = {"schema": SCHEMA, "tokenizer": tokenizer_metadata(),
                "policy": {"roi_method": "aihub-bbox-yxyx-v2"}, "records": rows}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path, document


def test_speaker_assignment_is_reproducible_and_preserves_official_validation():
    training = {f"C{i:03d}" for i in range(10)}
    parts = {"Training": training, "Validation": {"C100", "C101"}}
    result = assign_speakers(parts, 0.2, 42)
    assert result == assign_speakers(parts, 0.2, 42)
    assert sum(v == "train" for v in result.values()) == 8
    assert sum(v == "val" for v in result.values()) == 2
    assert result["C100"] == result["C101"] == "test"


@pytest.mark.parametrize("parts", [
    {"Training": {"C001"}},
    {"Training": {"C001", "C002"}, "Validation": {"C001"}},
    {"Training": {"C001"}, "Validation": {"C002"}},
])
def test_insufficient_or_overlapping_speakers_cannot_create_three_sets(parts):
    with pytest.raises(ValueError):
        assign_speakers(parts, 0.2, 42)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        assign_speakers({}, fraction, 42)


def test_camera_angles_share_one_utterance_key():
    a = parse_identity("lip_J_1_F_02_C002_A_001_007.npz")
    b = parse_identity("lip_J_1_F_02_C002_D_001_007.npz")
    assert a["utterance_id"] == b["utterance_id"]
    assert a["angle"] != b["angle"]


def test_valid_manifest_and_shared_text_are_reported_separately(dataset):
    path, _ = dataset
    _, summary = load_manifest(path)
    assert summary["speaker_overlap"] == summary["utterance_overlap"] == 0
    assert summary["normalized_text_overlap"]["train_test"] == 1


def test_cross_angle_same_speaker_leak_is_rejected(dataset):
    path, doc = dataset
    row = copy.deepcopy(doc["records"][0])
    row["id"] = row["id"].replace("_A_", "_B_")
    row["angle"], row["split"] = "B", "val"
    doc["records"].append(row)
    with pytest.raises(ValueError, match="누수"):
        validate_manifest(doc, path.parent)


def test_forged_speaker_cannot_hide_filename_identity(dataset):
    path, doc = dataset
    doc["records"][1]["speaker_id"] = "C999"
    with pytest.raises(ValueError, match="speaker_id 불일치"):
        validate_manifest(doc, path.parent)


def test_duplicate_clip_id_rejected(dataset):
    path, doc = dataset
    doc["records"].append(copy.deepcopy(doc["records"][0]))
    with pytest.raises(ValueError, match="중복 ID"):
        validate_manifest(doc, path.parent)


def test_official_validation_cannot_become_training(dataset):
    path, doc = dataset
    doc["records"][2]["split"] = "train"
    with pytest.raises(ValueError, match="공식 구획"):
        validate_manifest(doc, path.parent)


def test_tampered_cache_detected(dataset):
    path, doc = dataset
    cached = path.parent / doc["records"][0]["cache_path"]
    with cached.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        load_manifest(path)


def test_missing_cache_detected(dataset):
    path, doc = dataset
    (path.parent / doc["records"][0]["cache_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        load_manifest(path)


def test_cache_path_cannot_escape_dataset(dataset):
    path, doc = dataset
    doc["records"][0]["cache_path"] = "../outside.npz"
    with pytest.raises(ValueError, match="폴더 밖"):
        validate_manifest(doc, path.parent)


def test_wrong_roi_version_cannot_be_used(dataset):
    path, doc = dataset
    doc["policy"]["roi_method"] = "aihub-bbox-v1"
    with pytest.raises(ValueError, match="전처리 버전"):
        validate_manifest(doc, path.parent)


def test_training_and_evaluation_use_only_their_fixed_sets(dataset, monkeypatch):
    path, doc = dataset
    args = SimpleNamespace(smoke_data=None, manifest=str(path), max_frames=0, check_data=True)
    training, validation, provenance = train.training_files(args)
    testing, summary = evaluation.evaluation_files(SimpleNamespace(smoke_data=None, manifest=str(path), split="test"))
    assert [Path(p).stem for p in training] == [doc["records"][0]["id"]]
    assert [Path(p).stem for p in validation] == [doc["records"][1]["id"]]
    assert [Path(p).stem for p in testing] == [doc["records"][2]["id"]]
    assert not (set(training) | set(validation)) & set(testing)
    evaluation.validate_evaluation_provenance(provenance, summary)
    monkeypatch.setattr(train, "LipNetBaseline", lambda: pytest.fail("check-data는 모델을 만들면 안 됨"))
    train.train(args)


@pytest.mark.parametrize("provenance", [None, {"mode": "smoke_only"}, {"mode": "speaker_disjoint", "manifest_sha256": "wrong"}])
def test_wrong_training_history_cannot_claim_fixed_test_score(provenance):
    with pytest.raises(ValueError, match="학습 분할"):
        evaluation.validate_evaluation_provenance(provenance, {"manifest_sha256": "correct"})


def test_training_does_not_refilter_validation_or_test(dataset):
    path, _ = dataset
    with pytest.raises(ValueError, match="다시 필터링"):
        train.training_files(SimpleNamespace(smoke_data=None, manifest=str(path), max_frames=10))


def test_selection_order_independent_and_balanced_across_recordings():
    rows = [{"id": f"{s}-{v}-{u}", "speaker_id": s, "recording_id": f"{s}-{v}"}
            for s in ("C001", "C002") for v in range(3) for u in range(6)]
    speakers = {"C001": "train", "C002": "val"}
    result = select_pilot(rows, speakers, 10, 2, 42)
    assert result == select_pilot(list(reversed(rows)), speakers, 10, 2, 42)
    for speaker in speakers:
        selected = [r for r in result if r["speaker_id"] == speaker]
        assert len(selected) == 10
        assert len({r["recording_id"] for r in selected}) == 2
