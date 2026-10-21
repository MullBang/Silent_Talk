"""시간축 재생성의 표본 보존·부분 재개·출처 검증·TAR 단일 추출을 검사한다."""

import copy
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
import rebuild_timed_dataset as rebuild
from models.dataset_manifest import SCHEMA, load_manifest, parse_identity, sha256_file
from models.tokenizer import ctc_min_input_len, encode_text, tokenizer_metadata


def save_cache(path, row, *, timed=False, overrides=None):
    values = {"frames": np.full((10, 96, 96, 3), 17 if timed else 1, dtype=np.uint8),
              "label": np.asarray(encode_text(row["text"])), "text": row["text"],
              "source_text": row["source_text"], **tokenizer_metadata(),
              **{k: row[k] for k in rebuild.PROVENANCE}}
    if timed:
        values.update(temporal_method=rebuild.TEMPORAL_METHOD, video_fps=30000 / 1001, label_fps=30.)
    values.update(overrides or {})
    np.savez_compressed(path, **values)


@pytest.fixture
def dataset(tmp_path):
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    rows = []
    for speaker, split, texts in [("C001", "train", ["가", "나"]),
                                  ("C002", "val", ["다"]), ("C003", "test", ["라"])]:
        video = tmp_path / f"lip_J_1_F_02_{speaker}_A_001.mp4"
        video.write_bytes(b"video-source-for-mocked-preparer")
        label_path = video.with_suffix(".json")
        label = {"speaker_info": {"speaker_ID": speaker}, "Video_env": {"Angle": "A"},
                 "Video_info": {"video_Name": video.name, "FPS": 30},
                 "Sentence_info": [{"ID": i, "sentence_text": text,
                                    "start_time": 0., "end_time": .4}
                                   for i, text in enumerate(texts, 1)]}
        label_path.write_text(json.dumps(label), encoding="utf-8")
        for i, text in enumerate(texts, 1):
            sample_id = f"{video.stem}_{i:03d}"
            row = {"id": sample_id, **parse_identity(sample_id), "sentence_id": i,
                   "split": split, "text": text, "source_text": text,
                   "start_time": 0., "end_time": .4, "expected_frames": 10,
                   "ctc_min_frames": ctc_min_input_len(encode_text(text)),
                   "roi_method": "aihub-bbox-yxyx-v2",
                   "source_partition": "Validation" if split == "test" else "Training",
                   "label_path": str(label_path), "label_sha256": sha256_file(label_path),
                   "source_video": {"path": str(video), "size_bytes": video.stat().st_size},
                   "cache_path": sample_id + ".npz", "frames": 10}
            save_cache(parent_dir / row["cache_path"], row)
            row["cache_sha256"] = sha256_file(parent_dir / row["cache_path"])
            rows.append(row)
    doc = {"schema": SCHEMA, "tokenizer": tokenizer_metadata(), "records": rows,
           "policy": {"roi_method": "aihub-bbox-yxyx-v2", "max_frames": 200,
                      "require_disjoint_text": True, "frozen_parent_manifest": "older.json",
                      "frozen_parent_sha256": "older-hash"}}
    parent = parent_dir / "manifest.json"
    parent.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "timed"
    return parent, out, doc


def planned(dataset):
    parent, out, doc = dataset
    rebuild.plan(SimpleNamespace(parent=str(parent), out=str(out)))
    return SimpleNamespace(selection=str(out / "selection.json"), resume=False)


def fake_preparer(doc, calls, *, interrupt=False, omit_temporal=False):
    def prepare(_video, label_path, out, **kwargs):
        ids = kwargs["sentence_ids"]
        calls.append((Path(label_path).stem, list(ids)))
        for row in doc["records"]:
            if row["label_path"] == label_path and row["sentence_id"] in ids:
                assert kwargs["provenance"][row["sentence_id"]] == {
                    k: row[k] for k in rebuild.PROVENANCE}
                save_cache(Path(out) / f"{row['id']}.npz", row, timed=not omit_temporal)
                if interrupt and len(calls) == 1:
                    raise RuntimeError("simulated interruption after first saved clip")
        return len(ids)
    return prepare


def test_full_rebuild_preserves_parent_and_all_splits_including_test(dataset, monkeypatch):
    parent, out, doc = dataset
    before = {p.name: sha256_file(p) for p in parent.parent.iterdir()}
    args = planned(dataset)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    assert "frozen_parent_manifest" not in selection["policy"]
    assert selection["policy"]["temporal_parent_sha256"] == before["manifest.json"]
    assert all("video_fps" not in r and "cache_path" not in r for r in selection["records"])
    calls = []
    monkeypatch.setattr(rebuild, "_load_preparer", lambda: fake_preparer(doc, calls))
    rebuild.prepare(args)
    result, summary = load_manifest(out / "manifest.json")
    rebuild.assert_parent_preserved(result, doc)
    assert [r["split"] for r in result["records"]] == ["train", "train", "val", "test"]
    assert all(r["video_fps"] == 30000 / 1001 and r["label_fps"] == 30 for r in result["records"])
    assert all(new["cache_sha256"] != old["cache_sha256"]
               for new, old in zip(result["records"], doc["records"]))
    assert {p.name: sha256_file(p) for p in parent.parent.iterdir()} == before
    report = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert not report["test_inference_performed"] and not report["test_visual_review_performed"]
    assert report["rebuilt_clips"] == 4 and summary["splits"]["test"]["clips"] == 1
    with pytest.raises(FileExistsError, match="완성된 데이터셋"):
        rebuild.prepare(args)


def test_partial_failure_can_resume_without_rewriting_completed_cache(dataset, monkeypatch):
    _, out, doc = dataset
    args = planned(dataset)
    calls = []
    monkeypatch.setattr(rebuild, "_load_preparer", lambda: fake_preparer(doc, calls, interrupt=True))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        rebuild.prepare(args)
    assert not (out / "manifest.json").exists() and not (out / "prepare.lock").exists()
    first = out / "clips" / f"{doc['records'][0]['id']}.npz"
    before = (sha256_file(first), first.stat().st_mtime_ns)
    with pytest.raises(FileExistsError, match="resume"):
        rebuild.prepare(args)
    args.resume = True
    rebuild.prepare(args)
    assert calls[0][1] == [1, 2] and calls[1][1] == [2]
    assert before == (sha256_file(first), first.stat().st_mtime_ns)
    load_manifest(out / "manifest.json")


def test_resume_rejects_tampered_completed_file(dataset, monkeypatch):
    _, out, doc = dataset
    args = planned(dataset)
    monkeypatch.setattr(rebuild, "_load_preparer", lambda: fake_preparer(doc, [], interrupt=True))
    with pytest.raises(RuntimeError):
        rebuild.prepare(args)
    path = out / "clips" / f"{doc['records'][0]['id']}.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    args.resume = True
    with pytest.raises(ValueError, match="SHA-256/메타데이터 변경"):
        rebuild.prepare(args)
    assert not (out / "manifest.json").exists()


def test_resume_rejects_unrecorded_cache_instead_of_assuming_completed(dataset):
    _, out, doc = dataset
    args = planned(dataset)
    (out / "clips").mkdir()
    row = doc["records"][0]
    save_cache(out / "clips" / f"{row['id']}.npz", row, timed=True)
    args.resume = True
    with pytest.raises(ValueError, match="기록 불일치"):
        rebuild.prepare(args)


@pytest.mark.parametrize("change", ["parent", "selection"])
def test_changed_manifest_or_plan_is_rejected_before_extraction(dataset, monkeypatch, change):
    parent, _, _ = dataset
    args = planned(dataset)
    path = parent if change == "parent" else Path(args.selection)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    monkeypatch.setattr(rebuild, "_load_preparer", lambda: pytest.fail("must reject before extraction"))
    with pytest.raises(ValueError, match="SHA-256"):
        rebuild.prepare(args)


@pytest.mark.parametrize("key,value", [("text", "변경"), ("start_time", .1),
                                      ("end_time", .3), ("speaker_id", "C999"),
                                      ("split", "train"), ("source_partition", "Training")])
def test_temporal_rebuild_cannot_modify_heldout_identity_or_label(dataset, key, value):
    _, _, parent = dataset
    result = copy.deepcopy(parent)
    for row in result["records"]:
        row["temporal_method"] = rebuild.TEMPORAL_METHOD
    result["records"][-1][key] = value
    with pytest.raises(ValueError, match="부모 표본"):
        rebuild.assert_parent_preserved(result, parent)


def test_temporal_rebuild_cannot_add_or_remove_ids(dataset):
    _, _, parent = dataset
    result = copy.deepcopy(parent)
    result["records"].pop()
    with pytest.raises(ValueError, match="ID"):
        rebuild.assert_parent_preserved(result, parent)


def test_legacy_fallback_cache_cannot_finish_timed_dataset(dataset, monkeypatch):
    _, out, doc = dataset
    args = planned(dataset)
    monkeypatch.setattr(rebuild, "_load_preparer", lambda: fake_preparer(doc, [], omit_temporal=True))
    with pytest.raises(ValueError, match="시간 메타데이터"):
        rebuild.prepare(args)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("fps", [0., -1., float("nan"), float("inf"), [30.]])
def test_invalid_fps_metadata_is_rejected(dataset, fps):
    _, out, doc = dataset
    (out / "clips").mkdir(parents=True)
    row = {**rebuild.original_fields(doc["records"][0]), "temporal_method": rebuild.TEMPORAL_METHOD}
    save_cache(out / "clips" / f"{row['id']}.npz", row, timed=True, overrides={"video_fps": fps})
    with pytest.raises(ValueError, match="FPS|메타데이터"):
        rebuild.cached_row(row, out)


def test_plan_requires_empty_output_separate_from_parent(dataset):
    parent, out, _ = dataset
    with pytest.raises(ValueError, match="별도 폴더"):
        rebuild.plan(SimpleNamespace(parent=str(parent), out=str(parent.parent / "nested")))
    out.mkdir()
    (out / "user-file.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="빈 데이터셋"):
        rebuild.plan(SimpleNamespace(parent=str(parent), out=str(out)))
    assert (out / "user-file.txt").read_text(encoding="utf-8") == "preserve"


def test_tar_reads_only_requested_member_into_fixed_local_filename(tmp_path):
    path = tmp_path / "source.tar"
    name = "lip_J_1_F_02_C001_A_001.mp4"
    # extractfile의 바이트만 고정 경로로 쓰므로 멤버 경로가 디스크 경로가 되지 않는다.
    member_name = "../../" + name
    with tarfile.open(path, "w") as archive:
        for member, content in [(member_name, b"chosen"), ("other.mp4", b"unselected")]:
            entry = tarfile.TarInfo(member)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    archives = {}
    try:
        extracted = rebuild._source_video({"path": str(path), "archive_member": member_name,
                                           "size_bytes": 6}, str(tmp_path / Path(name).with_suffix(".json")),
                                          temporary, archives)
        assert extracted == temporary / name and extracted.read_bytes() == b"chosen"
        assert list(temporary.iterdir()) == [extracted]
        assert not (tmp_path / name).exists()
    finally:
        for archive in archives.values():
            archive.close()
