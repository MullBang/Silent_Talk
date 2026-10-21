"""확대 시 부모 표본 보존, 텍스트 누수 차단, 결정적 균형 선정을 검증한다."""

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
from build_speaker_dataset import build_index, prepare_selection, require_current_timing
from expand_speaker_dataset import PROVENANCE, assert_parent_preserved, cached_row, check_source_labels, plan, prepare, select_expansion, without_cache
from models.dataset_manifest import SCHEMA, TEMPORAL_METHOD, parse_identity, sha256_file, validate_manifest
from models.tokenizer import ctc_min_input_len, encode_text, tokenizer_metadata


def sample(speaker, video, sentence, text, split="train"):
    sample_id = f"lip_J_1_F_02_{speaker}_A_{video:03d}_{sentence:03d}"
    return {"id": sample_id, **parse_identity(sample_id), "sentence_id": sentence, "split": split,
            "source_partition": "Validation" if split == "test" else "Training",
            "source_text": text, "text": text, "expected_frames": 50,
            "ctc_min_frames": ctc_min_input_len(encode_text(text)), "roi_method": "aihub-bbox-yxyx-v2",
            "start_time": 0., "end_time": 2., "label_sha256": "label",
            "temporal_method": TEMPORAL_METHOD}


@pytest.fixture
def expansion():
    rows = [sample("C001", 1, 1, "가"), sample("C002", 1, 1, "나"),
            sample("C003", 1, 1, "다", "val"), sample("C004", 1, 1, "라", "test")]
    for row in rows:
        row.update(cache_path=row["id"] + ".npz", cache_sha256=row["id"], frames=50,
                   video_fps=30000 / 1001, label_fps=30.0)
    parent = {"schema": SCHEMA, "tokenizer": tokenizer_metadata(),
              "policy": {"roi_method": "aihub-bbox-yxyx-v2", "max_frames": 200,
                         "temporal_method": TEMPORAL_METHOD}, "records": rows}
    candidates = [sample(speaker, video, sentence, chr(0xAC10 + offset * 20 + video * 5 + sentence))
                  for offset, speaker in enumerate(("C001", "C002"))
                  for video in (1, 2, 3) for sentence in (2, 3, 4)]
    return parent, candidates


def test_selection_is_deterministic_balanced_and_keeps_every_parent(expansion):
    parent, candidates = expansion
    before = copy.deepcopy(parent)
    rows, audit = select_expansion(parent, candidates, 4, 42)
    assert rows == select_expansion(parent, list(reversed(candidates)), 4, 42)[0]
    assert parent == before
    assert audit["new_clips"] == 6
    assert audit["train_per_speaker"] == {"C001": 4, "C002": 4}
    assert_parent_preserved({"records": rows}, parent)
    for speaker in ("C001", "C002"):
        assert len({r["recording_id"] for r in rows if r["speaker_id"] == speaker}) == 3


def test_heldout_text_and_speaker_cannot_enter_training(expansion):
    parent, candidates = expansion
    forbidden = [sample("C001", 4, 1, "다"), sample("C002", 4, 1, "라"),
                 sample("C003", 4, 1, "마")]
    rows, audit = select_expansion(parent, candidates + forbidden, 4, 42)
    assert not {r["id"] for r in forbidden} & {r["id"] for r in rows}
    assert audit["candidate_filter_counts"]["matches_parent_text_or_utterance"] == 2
    assert audit["candidate_filter_counts"]["outside_fixed_train_speakers"] == 1


def test_duplicate_new_transcripts_are_not_selected_twice(expansion):
    parent, candidates = expansion
    for r in candidates:
        r["text"] = r["source_text"] = "같다"
    with pytest.raises(ValueError, match="미달"):
        select_expansion(parent, candidates, 2, 42)


@pytest.mark.parametrize("change", ["missing", "text", "split", "hash", "added_val"])
def test_frozen_parent_guard_rejects_changes(expansion, change):
    parent, _ = expansion
    document = copy.deepcopy(parent)
    row = document["records"][2]
    if change == "missing":
        document["records"].pop()
    elif change == "text":
        row["text"] = "변경"
    elif change == "split":
        row["split"] = "train"
    elif change == "hash":
        row["cache_sha256"] = "changed"
    else:
        document["records"].append(sample("C003", 2, 1, "마", "val"))
    with pytest.raises(ValueError, match="부모|고정 분할"):
        assert_parent_preserved(document, parent, check_hash=True)


def test_opt_in_text_guard_does_not_change_legacy_manifest_policy(expansion, tmp_path):
    parent, _ = expansion
    parent["records"][2]["text"] = parent["records"][2]["source_text"] = "가"
    assert validate_manifest(parent, tmp_path)["normalized_text_overlap"]["train_val"] == 1
    parent["policy"]["require_disjoint_text"] = True
    with pytest.raises(ValueError, match="정규화 정답 누수"):
        validate_manifest(parent, tmp_path)


def test_source_label_time_is_rechecked_even_when_hash_matches(tmp_path):
    row = sample("C001", 1, 1, "가")
    path = tmp_path / "lip_J_1_F_02_C001_A_001.json"
    label = {"speaker_info": {"speaker_ID": "C001"}, "Video_env": {"Angle": "A"},
             "Video_info": {"video_Name": path.stem + ".mp4"},
             "Sentence_info": [{"ID": 1, "sentence_text": "가", "start_time": 0., "end_time": 2.}]}
    path.write_text(json.dumps([label]), encoding="utf-8")
    row.update(label_path=str(path), label_sha256=sha256_file(path))
    check_source_labels([row])
    row["start_time"] = 0.5
    with pytest.raises(ValueError, match="실제 라벨"):
        check_source_labels([row])


def test_interrupted_preparation_resumes_without_rewriting_parent_or_complete_clips(expansion, tmp_path, monkeypatch):
    import prepare_data
    parent, _ = expansion
    old_dir, out = tmp_path / "parent", tmp_path / "expanded"
    old_dir.mkdir()
    out.mkdir()

    def save(path, row):
        np.savez_compressed(path, frames=np.ones((50, 96, 96, 3), dtype=np.uint8),
                            label=np.asarray(encode_text(row["text"])), text=row["text"],
                            source_text=row["source_text"], **{k: row[k] for k in PROVENANCE},
                            temporal_method=TEMPORAL_METHOD, video_fps=30000 / 1001, label_fps=30.0,
                            **tokenizer_metadata())

    for row in parent["records"]:
        path = old_dir / row["cache_path"]
        save(path, row)
        row["cache_sha256"] = sha256_file(path)
    parent_path = old_dir / "manifest.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    video = tmp_path / "lip_J_1_F_02_C001_A_002.mp4"
    video.write_bytes(b"fake-video-for-mocked-preprocessing")
    label_path = video.with_suffix(".json")
    new = [sample("C001", 2, 2, "마"), sample("C001", 2, 3, "바")]
    label = {"speaker_info": {"speaker_ID": "C001"}, "Video_env": {"Angle": "A"},
             "Video_info": {"video_Name": video.name}, "Sentence_info": [
                 {"ID": r["sentence_id"], "sentence_text": r["source_text"], "start_time": 0., "end_time": 2.}
                 for r in new]}
    label_path.write_text(json.dumps(label), encoding="utf-8")
    for row in new:
        row.update(label_path=str(label_path), label_sha256=sha256_file(label_path),
                   source_video={"path": str(video), "size_bytes": video.stat().st_size})
    selection = copy.deepcopy(parent)
    selection["policy"].update(frozen_parent_manifest=str(parent_path), frozen_parent_sha256=sha256_file(parent_path),
                               require_disjoint_text=True)
    selection["records"] = [without_cache(r) for r in parent["records"]] + new
    selection_path = out / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    calls = []

    def interrupted_video(_video, _label, destination, **kwargs):
        ids = kwargs["sentence_ids"]
        calls.append(ids)
        for row in new:
            if row["sentence_id"] in ids:
                save(Path(destination) / f"{row['id']}.npz", row)
                if len(calls) == 1:
                    raise RuntimeError("simulated interruption")
        return len(ids)

    monkeypatch.setattr(prepare_data, "prepare_video", interrupted_video)
    args = SimpleNamespace(selection=str(selection_path), resume=False)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare(args)
    assert not (out / "manifest.json").exists()
    first = out / "clips" / f"{new[0]['id']}.npz"
    before = (sha256_file(first), first.stat().st_mtime_ns)
    args.resume = True
    prepare(args)
    result = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert calls == [[2, 3], [3]]
    assert before == (sha256_file(first), first.stat().st_mtime_ns)
    assert_parent_preserved(result, parent, check_hash=True)
    assert all(r["video_fps"] == 30000 / 1001 and r["label_fps"] == 30.0 for r in result["records"])
    assert sha256_file(parent_path) == selection["policy"]["frozen_parent_sha256"]
    with pytest.raises(FileExistsError, match="완성된 데이터셋"):
        prepare(args)


def test_roi_inspection_writes_json_and_leaves_visual_approval_pending(expansion, tmp_path):
    from inspect_dataset_rois import inspect
    parent, _ = expansion
    for row in parent["records"]:
        path = tmp_path / row["cache_path"]
        np.savez_compressed(path, frames=np.ones((50, 96, 96, 3), dtype=np.uint8),
                            label=np.asarray(encode_text(row["text"])), text=row["text"],
                            source_text=row["source_text"], **{k: row[k] for k in PROVENANCE},
                            temporal_method=TEMPORAL_METHOD, video_fps=30000 / 1001, label_fps=30.0,
                            **tokenizer_metadata())
        row["cache_sha256"] = sha256_file(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(parent), encoding="utf-8")
    inspect(manifest)
    report = json.loads((tmp_path / "visual_qa/qa_inventory.json").read_text(encoding="utf-8"))
    assert report["thumbnail_count"] == 12
    assert report["automatic_flag_count"] == 4
    assert report["manual_review_status"] == "pending"
    assert isinstance(report["metrics"][0]["ctc_slack_frames"], int)
    assert (tmp_path / "visual_qa/roi_samples_01.png").is_file()


def test_legacy_source_index_can_select_only_new_timed_clips(expansion):
    """구 색인은 원본 선택 정보로만 사용하고, 부모 캐시의 출처는 그대로 둔다."""
    parent, candidates = expansion
    for row in candidates:
        row.pop("temporal_method")
        row["video_fps"] = 30.0  # 구 정보가 있더라도 새 전처리 FPS로 쓰면 안 된다.
    before = copy.deepcopy(candidates)
    rows, _ = select_expansion(parent, candidates, 4, 42)
    old_ids = {row["id"] for row in parent["records"]}
    assert candidates == before
    assert all(row["temporal_method"] == TEMPORAL_METHOD for row in rows)
    assert all("video_fps" not in row for row in rows if row["id"] not in old_ids)
    assert_parent_preserved({"records": rows}, parent)


def _legacy(document):
    """기존 명세를 읽는 경로와 새 캐시 쓰기 경로의 차이를 검증할 구 버전."""
    result = copy.deepcopy(document)
    result["policy"].pop("temporal_method", None)
    for row in result["records"]:
        for key in ("temporal_method", "video_fps", "label_fps"):
            row.pop(key, None)
    return result


def test_legacy_manifest_remains_readable_but_cannot_be_expansion_parent(expansion, tmp_path):
    parent, candidates = expansion
    legacy = _legacy(parent)
    assert validate_manifest(legacy, tmp_path)["splits"]["train"]["clips"] == 2
    with pytest.raises(ValueError, match="rebuild_timed_dataset"):
        select_expansion(legacy, candidates, 4, 42)


@pytest.mark.parametrize("command", ["index_prepare", "expansion_prepare", "expansion_plan"])
def test_legacy_preparation_and_plan_fail_before_any_cache_write(expansion, tmp_path, command):
    parent, _ = expansion
    legacy = _legacy(parent)
    legacy["records"] = [without_cache(row) for row in legacy["records"]]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    args = SimpleNamespace(selection=str(path), resume=True, parent=str(path),
                           index=str(tmp_path / "does_not_exist.json"), out=str(tmp_path / "new"),
                           per_speaker=4, seed=42)
    action = {"index_prepare": prepare_selection, "expansion_prepare": prepare,
              "expansion_plan": plan}[command]
    with pytest.raises(ValueError, match="rebuild_timed_dataset"):
        action(args)
    assert not (tmp_path / "clips").exists()
    assert not (tmp_path / "new").exists()


def test_new_policy_does_not_permit_legacy_row_mixing(expansion):
    parent, _ = expansion
    parent["records"][0].pop("temporal_method")
    with pytest.raises(ValueError, match="rebuild_timed_dataset"):
        require_current_timing(parent)


def test_partial_legacy_cache_cannot_be_reused_as_timed_data(tmp_path):
    row = sample("C001", 1, 1, "가")
    (tmp_path / "clips").mkdir()
    path = tmp_path / "clips" / f"{row['id']}.npz"
    np.savez_compressed(path, frames=np.ones((50, 96, 96, 3), dtype=np.uint8),
                        label=np.asarray(encode_text(row["text"])), text=row["text"],
                        source_text=row["source_text"], **{k: row[k] for k in PROVENANCE},
                        **tokenizer_metadata())
    before = sha256_file(path)
    with pytest.raises(ValueError, match="구 시간축 캐시 혼합"):
        cached_row(row, tmp_path)
    assert sha256_file(path) == before


def test_new_index_marks_timing_and_defers_actual_video_bounds_to_preparation(tmp_path):
    """색인 단계의 라벨 30 FPS 추정으로 실제 29.97 FPS 표본을 제거하지 않는다."""
    source = tmp_path / "source"
    for directory, speaker in [("1.Training", "C001"), ("1.Training", "C002"),
                               ("2.Validation", "C003")]:
        folder = source / directory
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"lip_J_1_F_02_{speaker}_A_001"
        (folder / f"{stem}.mp4").write_bytes(b"index-only-placeholder")
        # 30 FPS라면 마지막 bbox를 넘지만 29.97 FPS라면 아직 범위 안일 수 있다.
        boxes = [[0, 0, 1, 1]] * 9053
        label = {"speaker_info": {"speaker_ID": speaker}, "Video_env": {"Angle": "A"},
                 "Video_info": {"video_Name": f"{stem}.mp4", "FPS": 30},
                 "Bounding_box_info": {"Lip_bounding_box": {"xtl_ytl_xbr_ybr": boxes}},
                 "Sentence_info": [{"ID": 1, "sentence_text": "가", "start_time": 300., "end_time": 302.}]}
        (folder / f"{stem}.json").write_text(json.dumps(label), encoding="utf-8")
    out = tmp_path / "indexed"
    build_index(SimpleNamespace(source_root=str(source), out=str(out), max_frames=200,
                                val_fraction=0.5, seed=42, clips_per_speaker=1, videos_per_speaker=1))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((out / "source_index.json").read_text(encoding="utf-8"))
    assert selection["policy"]["temporal_method"] == TEMPORAL_METHOD
    assert len(selection["records"]) == 3
    assert all(row["temporal_method"] == TEMPORAL_METHOD for row in index["eligible_records"])
    assert all("video_fps" not in row for row in selection["records"])
    assert not (out / "manifest.json").exists()


def test_pilot_preparation_records_fps_from_completed_cache(expansion, tmp_path, monkeypatch):
    """pilot 경로도 라벨 FPS를 복사하지 않고 NPZ의 실제 영상 FPS를 명세에 적는다."""
    import prepare_data
    parent, _ = expansion
    selection = copy.deepcopy(parent)
    selection["records"] = [without_cache(row) for row in parent["records"]]
    by_label = {}
    for row in selection["records"]:
        row.pop("video_fps")
        row.pop("label_fps")
        label = tmp_path / f"{row['id'].rsplit('_', 1)[0]}.json"
        video = label.with_suffix(".mp4")
        label.write_text("{}", encoding="utf-8")
        video.write_bytes(b"mocked-video")
        row.update(label_path=str(label), label_sha256=sha256_file(label),
                   source_video={"path": str(video), "size_bytes": video.stat().st_size})
        by_label[str(label)] = row
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")

    def save_prepared(_video, label_path, destination, **kwargs):
        row = by_label[label_path]
        assert kwargs["sentence_ids"] == [row["sentence_id"]]
        np.savez_compressed(Path(destination) / f"{row['id']}.npz",
                            frames=np.ones((50, 96, 96, 3), dtype=np.uint8),
                            label=np.asarray(encode_text(row["text"])), text=row["text"],
                            source_text=row["source_text"], **{k: row[k] for k in PROVENANCE},
                            temporal_method=TEMPORAL_METHOD, video_fps=30000 / 1001, label_fps=30.0,
                            **tokenizer_metadata())
        return 1

    monkeypatch.setattr(prepare_data, "prepare_video", save_prepared)
    prepare_selection(SimpleNamespace(selection=str(path)))
    result = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert all(row["video_fps"] == 30000 / 1001 for row in result["records"])
    assert all(row["label_fps"] == 30.0 for row in result["records"])
    assert all(row["temporal_method"] == TEMPORAL_METHOD for row in result["records"])
    validate_manifest(result, tmp_path, verify_files=True)
