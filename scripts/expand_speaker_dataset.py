"""기존 화자/검증/평가 표본을 고정한 채 학습 문장만 새 데이터 버전으로 확대한다."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.dataset_manifest import load_manifest, sha256_file, validate_manifest
from models.tokenizer import ctc_min_input_len, encode_text, normalize_text
from build_speaker_dataset import cache_timing_metadata, require_current_timing, stable_key, write_json

PROVENANCE = ("speaker_id", "utterance_id", "recording_id", "angle",
              "source_partition", "roi_method", "label_sha256")
CACHE_KEYS = ("cache_path", "cache_sha256", "frames")


def without_cache(row: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in row.items() if k not in CACHE_KEYS}


def select_expansion(parent: dict, candidates: list[dict], per_speaker: int, seed: int) -> tuple[list, dict]:
    """부모 표본 전부 보존, 새 문장만 화자별/녹화별 균형 선정. 입력 순서와 무관하다."""
    require_current_timing(parent)
    rows = [without_cache(r) for r in parent["records"]]
    speakers = sorted({r["speaker_id"] for r in rows if r["split"] == "train"})
    ids = {r["id"] for r in rows}
    used_text = {r["text"] for r in rows}
    used_utterance = {r["utterance_id"] for r in rows}
    counts = Counter(r["speaker_id"] for r in rows if r["split"] == "train")
    recording_counts = Counter(r["recording_id"] for r in rows if r["split"] == "train")
    if per_speaker <= 0 or any(counts[s] > per_speaker for s in speakers):
        raise ValueError("화자별 목표는 기존 학습 문장 수 이상이어야 합니다.")
    groups = defaultdict(lambda: defaultdict(list))
    rejected = Counter()
    seen = set()
    for r in sorted(candidates, key=lambda r: (stable_key(seed, r["id"]), r["id"])):
        if r["id"] in seen:
            raise ValueError(f"색인 중복 ID: {r['id']}")
        seen.add(r["id"])
        if r["speaker_id"] not in speakers:
            rejected["outside_fixed_train_speakers"] += 1
        elif r["id"] in ids:
            rejected["already_in_parent"] += 1
        elif (r["source_partition"] != "Training" or r["angle"] != "A"
              or r["roi_method"] != parent["policy"]["roi_method"]
              or not r["ctc_min_frames"] <= r["expected_frames"] <= parent["policy"]["max_frames"]):
            rejected["outside_training_policy"] += 1
        elif r["text"] in used_text or r["utterance_id"] in used_utterance:
            rejected["matches_parent_text_or_utterance"] += 1
        else:
            groups[r["speaker_id"]][r["recording_id"]].append(r)
    # 화자를 번갈아 처리하고, 현재 채택 수가 가장 적은 녹화부터 선택한다.
    while any(counts[s] < per_speaker for s in speakers):
        for speaker in speakers:
            if counts[speaker] == per_speaker:
                continue
            videos = groups[speaker]
            while videos:
                video = min(videos, key=lambda v: (recording_counts[v], stable_key(seed, v), v))
                r = videos[video].pop(0)
                if not videos[video]:
                    del videos[video]
                if r["text"] in used_text or r["utterance_id"] in used_utterance:
                    rejected["duplicate_new_selection"] += 1
                    continue
                added = without_cache(r)
                # 예전 원본 색인도 선택 정보로 재사용하지만 새 클립은 현재 방식으로 준비한다.
                # 영상 FPS는 라벨/구 색인에서 추정하지 않고 완성 NPZ에서 다시 취득한다.
                added.pop("video_fps", None)
                added.pop("label_fps", None)
                rows.append({**added, "split": "train",
                             "temporal_method": parent["policy"]["temporal_method"]})
                counts[speaker] += 1
                recording_counts[video] += 1
                used_text.add(r["text"])
                used_utterance.add(r["utterance_id"])
                break
            else:
                raise ValueError(f"{speaker}: 중복 제외 후 목표 {per_speaker}개에 미달")
    rows.sort(key=lambda r: r["id"])
    return rows, {"target_train_clips": per_speaker * len(speakers),
                  "new_clips": len(rows) - len(parent["records"]),
                  "reused_clips": len(parent["records"]), "train_per_speaker": dict(counts),
                  "candidate_filter_counts": dict(rejected),
                  "unused_eligible_candidates": sum(len(v) for g in groups.values() for v in g.values()),
                  "selection_seed": seed}


def assert_parent_preserved(document: dict, parent: dict, check_hash: bool = False) -> None:
    """동일 화자뿐 아니라 모든 부모 표본의 정답/출처 및 선택적으로 캐시 해시도 고정한다."""
    current = {r["id"]: r for r in document["records"]}
    original = {r["id"]: r for r in parent["records"]}
    train_speakers = {r["speaker_id"] for r in original.values() if r["split"] == "train"}
    for sample_id, row in original.items():
        actual = current.get(sample_id)
        if actual is None or without_cache(actual) != without_cache(row):
            raise ValueError(f"부모 표본 변경 또는 누락: {sample_id}")
        if check_hash and any(actual[k] != row[k] for k in ("cache_sha256", "frames")):
            raise ValueError(f"부모 캐시 변경: {sample_id}")
    for sample_id, row in current.items():
        if sample_id not in original and (row["split"] != "train" or row["speaker_id"] not in train_speakers):
            raise ValueError(f"고정 분할 밖의 추가 표본: {sample_id}")


def plan(args) -> None:
    parent_path, index_path, out = Path(args.parent).resolve(), Path(args.index).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"새 빈 데이터셋 폴더가 필요합니다: {out}")
    parent, _ = load_manifest(parent_path, verify_files=False)
    require_current_timing(parent)
    parent, _ = load_manifest(parent_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows, audit = select_expansion(parent, index["eligible_records"], args.per_speaker, args.seed)
    policy = {k: v for k, v in parent["policy"].items() if k not in ("clips_per_speaker", "videos_per_speaker")}
    policy.update(seed=args.seed, train_clips_per_speaker=args.per_speaker,
                  selection="preserve_parent_then_balance_recordings", require_disjoint_text=True,
                  frozen_parent_manifest=str(parent_path), frozen_parent_sha256=sha256_file(parent_path),
                  source_index=str(index_path), source_index_sha256=sha256_file(index_path))
    document = {"schema": parent["schema"], "tokenizer": parent["tokenizer"], "policy": policy, "records": rows}
    assert_parent_preserved(document, parent)
    audit["selection_summary"] = validate_manifest(document, out)
    write_json(out / "selection.json", document)
    write_json(out / "selection_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def check_source_labels(records: list[dict]) -> None:
    """색인의 시간/정답/화자 정보를 실제 JSON과 다시 대조한다."""
    path = Path(records[0]["label_path"])
    digest = sha256_file(path)
    label = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(label, list):
        if len(label) != 1:
            raise ValueError("단일 녹화 라벨이 아닙니다.")
        label = label[0]
    sentences = {int(s["ID"]): s for s in label["Sentence_info"]}
    if len(sentences) != len(label["Sentence_info"]):
        raise ValueError("라벨 문장 ID 중복")
    for r in records:
        s = sentences[r["sentence_id"]]
        if (digest != r["label_sha256"] or label["speaker_info"]["speaker_ID"] != r["speaker_id"]
                or label["Video_env"]["Angle"] != r["angle"]
                or label["Video_info"]["video_Name"] != path.stem + ".mp4"
                or s["sentence_text"] != r["source_text"] or normalize_text(s["sentence_text"]) != r["text"]
                or float(s["start_time"]) != r["start_time"] or float(s["end_time"]) != r["end_time"]
                or round((r["end_time"] - r["start_time"]) * 25) != r["expected_frames"]
                or ctc_min_input_len(encode_text(r["text"])) != r["ctc_min_frames"]):
            raise ValueError(f"실제 라벨과 선정 명세 불일치: {r['id']}")


def cached_row(row: dict, out: Path) -> dict:
    import numpy as np
    from models.cache import load_clip
    path = out / "clips" / f"{row['id']}.npz"
    clip = load_clip(path)
    if (clip["text"] != row["text"] or clip["source_text"] != row["source_text"]
            or len(clip["frames"]) != row["expected_frames"]):
        raise ValueError(f"전처리 결과와 명세 불일치: {row['id']}")
    with np.load(path, allow_pickle=False) as cache:
        if any(str(cache[k]) != str(row[k]) for k in PROVENANCE):
            raise ValueError(f"캐시 출처 불일치: {row['id']}")
    return {**row, "cache_path": path.relative_to(out).as_posix(),
            "cache_sha256": sha256_file(path), "frames": len(clip["frames"]),
            **cache_timing_metadata(path)}


def prepare(args) -> None:
    import cv2
    from prepare_data import prepare_video
    cv2.setNumThreads(2)
    started = time.perf_counter()
    selection = Path(args.selection).resolve()
    out = selection.parent
    document, _ = load_manifest(selection, verify_files=False)
    require_current_timing(document)
    if (out / "manifest.json").exists():
        raise FileExistsError("완성된 데이터셋은 변경하지 않습니다.")
    parent_path = Path(document["policy"]["frozen_parent_manifest"])
    if sha256_file(parent_path) != document["policy"]["frozen_parent_sha256"]:
        raise ValueError("부모 명세가 변경되었습니다.")
    parent, _ = load_manifest(parent_path)
    require_current_timing(parent)
    assert_parent_preserved(document, parent)
    parent_rows = {r["id"]: r for r in parent["records"]}
    cache = out / "clips"
    if cache.exists() and any(cache.iterdir()) and not args.resume:
        raise FileExistsError("부분 캐시가 있습니다. 검사 후 이어 준비하려면 --resume을 지정하세요.")
    cache.mkdir(exist_ok=True)
    expected_paths = {f"{r['id']}.npz" for r in document["records"]}
    if any(p.name not in expected_paths for p in cache.iterdir()):
        raise ValueError("선정 명세에 없는 캐시 파일이 있습니다.")
    groups = defaultdict(list)
    result = copy.deepcopy(document)
    result["selection_sha256"] = sha256_file(selection)
    result["records"] = []
    for row in document["records"]:
        if row["id"] in parent_rows:
            old = parent_rows[row["id"]]
            destination = cache / f"{row['id']}.npz"
            if not destination.exists():
                with (parent_path.parent / old["cache_path"]).open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            ready = cached_row(row, out)
            if ready["cache_sha256"] != old["cache_sha256"]:
                raise ValueError(f"부모 캐시 복사본 불일치: {row['id']}")
            result["records"].append(ready)
        else:
            groups[row["label_path"]].append(row)
    print(f"[보존] 기존 {len(parent_rows)}개 캐시의 해시 일치", flush=True)
    archives = {}
    try:
        for number, (label_path, records) in enumerate(sorted(groups.items()), 1):
            check_source_labels(records)
            missing = [r for r in records if not (cache / f"{r['id']}.npz").exists()]
            print(f"[준비 {number}/{len(groups)}] {Path(label_path).stem}: 새 {len(missing)}개", flush=True)
            if missing:
                source = records[0]["source_video"]
                if any(r["source_video"] != source for r in records):
                    raise ValueError("같은 녹화의 원본 영상 정보 불일치")
                with tempfile.TemporaryDirectory(prefix="source_", dir=out) as temporary:
                    if "archive_member" in source:
                        archive = archives.get(source["path"])
                        if archive is None:
                            archive = tarfile.open(source["path"], "r:")
                            archives[source["path"]] = archive
                        member = archive.getmember(source["archive_member"])
                        if not member.isfile() or member.size != source["size_bytes"]:
                            raise ValueError("TAR 원본 영상 크기/형식 불일치")
                        video = Path(temporary) / (Path(label_path).stem + ".mp4")
                        with archive.extractfile(member) as src, video.open("xb") as dst:
                            shutil.copyfileobj(src, dst, 1024 * 1024)
                    else:
                        video = Path(source["path"])
                        if video.stat().st_size != source["size_bytes"]:
                            raise ValueError("원본 영상 크기 불일치")
                    count = prepare_video(str(video), label_path, str(cache),
                                          max_frames=document["policy"]["max_frames"], roi_mode="bbox",
                                          sentence_ids=[r["sentence_id"] for r in missing],
                                          provenance={r["sentence_id"]: {k: r[k] for k in PROVENANCE} for r in missing})
                    if count != len(missing):
                        raise RuntimeError("일부 문장 전처리 실패. 완성 명세를 생성하지 않습니다.")
            result["records"].extend(cached_row(r, out) for r in records)
            print(f"[진행] {len(result['records'])}/{len(document['records'])}개, {time.perf_counter()-started:.1f}초", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    result["records"].sort(key=lambda r: r["id"])
    assert_parent_preserved(result, parent, check_hash=True)
    summary = validate_manifest(result, out, verify_files=True)
    if sha256_file(parent_path) != document["policy"]["frozen_parent_sha256"]:
        raise ValueError("작업 도중 부모 명세 변경")
    write_json(out / "manifest.json", result)
    summary.update(manifest_sha256=sha256_file(out / "manifest.json"),
                   parent_sha256=sha256_file(parent_path), preserved_parent_clips=len(parent_rows),
                   cache_bytes=sum(p.stat().st_size for p in cache.glob("*.npz")),
                   preparation_seconds=time.perf_counter() - started)
    write_json(out / "validation_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--parent", default="data_cache/datasets/aihub_timed_v4/manifest.json")
    p.add_argument("--index", default="data_cache/datasets/aihub_pilot_v2/source_index.json")
    p.add_argument("--out", default="data_cache/datasets/aihub_expanded_timed_next")
    p.add_argument("--per-speaker", type=int, default=125)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", default="data_cache/datasets/aihub_expanded_timed_next/selection.json")
    p.add_argument("--resume", action="store_true", help="이미 준비된 클립도 모두 재검증한 뒤 이어 준비")
    args = parser.parse_args()
    (plan if args.command == "plan" else prepare)(args)


if __name__ == "__main__":
    main()
