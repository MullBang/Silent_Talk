"""고정 표본의 영상 시간축만 교정하여 별도 데이터 버전으로 재생성한다.

train/val/test의 ID·정답·시각·화자·분할은 부모와 같아야 한다. test도 공통
전처리와 무결성 검사만 수행하며 이 스크립트는 추론/시각 검수를 하지 않는다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.cache import load_clip
from models.dataset_manifest import load_manifest, sha256_file, validate_manifest
from expand_speaker_dataset import PROVENANCE, check_source_labels

TEMPORAL_METHOD = "video-cfr-timestamps-v1"
REBUILT_KEYS = {"cache_path", "cache_sha256", "frames", "temporal_method", "video_fps", "label_fps"}
PROGRESS_SCHEMA = "lipnet-timing-rebuild-progress-v1"


def original_fields(row: dict) -> dict:
    return {k: copy.deepcopy(v) for k, v in row.items() if k not in REBUILT_KEYS}


def assert_parent_preserved(document: dict, parent: dict) -> None:
    """캐시와 새 시간 메타데이터 외에는 추가·삭제·변경을 허용하지 않는다."""
    current = {r["id"]: r for r in document["records"]}
    old = {r["id"]: r for r in parent["records"]}
    if (len(current) != len(document["records"]) or len(old) != len(parent["records"])
            or current.keys() != old.keys()):
        raise ValueError("부모 표본 ID의 중복·추가·누락")
    for sample_id, row in old.items():
        actual = current[sample_id]
        if original_fields(actual) != original_fields(row):
            raise ValueError(f"부모 표본의 정답/시간/화자/분할/출처 변경: {sample_id}")
        if actual.get("temporal_method") != TEMPORAL_METHOD:
            raise ValueError(f"시간 전처리 버전 불일치: {sample_id}")
        if "frames" in actual and actual["frames"] != row["frames"]:
            raise ValueError(f"부모 표본의 출력 길이 변경: {sample_id}")


def _check_output_location(out: Path, parent_path: Path) -> None:
    if out.is_relative_to(parent_path.parent) or parent_path.parent.is_relative_to(out):
        raise ValueError("출력은 부모 데이터셋과 겹치지 않는 별도 폴더여야 합니다.")


def _write_new_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _save_progress(out: Path, progress: dict) -> None:
    temporary = out / "progress.next.json"
    _write_new_json(temporary, progress)
    temporary.replace(out / "progress.json")


def plan(args) -> None:
    parent_path, out = Path(args.parent).resolve(), Path(args.out).resolve()
    _check_output_location(out, parent_path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"새 빈 데이터셋 폴더가 필요합니다: {out}")
    parent, summary = load_manifest(parent_path)
    rows = [{**original_fields(r), "temporal_method": TEMPORAL_METHOD} for r in parent["records"]]
    policy = {k: copy.deepcopy(v) for k, v in parent["policy"].items()
              if not k.startswith(("frozen_parent_", "temporal_parent_"))}
    policy.update(temporal_method=TEMPORAL_METHOD,
                  temporal_parent_manifest=str(parent_path),
                  temporal_parent_sha256=summary["manifest_sha256"],
                  selection="preserve_all_parent_samples_rebuild_video_timing")
    document = {"schema": parent["schema"], "tokenizer": copy.deepcopy(parent["tokenizer"]),
                "policy": policy, "records": rows}
    assert_parent_preserved(document, parent)
    validate_manifest(document, out, verify_files=False)
    out.mkdir(parents=True, exist_ok=True)
    _write_new_json(out / "selection.json", document)
    _write_new_json(out / "progress.json", {
        "schema": PROGRESS_SCHEMA, "selection_sha256": sha256_file(out / "selection.json"),
        "temporal_parent_sha256": summary["manifest_sha256"], "completed": {},
    })
    print(f"[계획] 부모의 {len(rows)}개 표본과 분할 고정, 출력 {out}", flush=True)


def cached_row(row: dict, out: Path) -> dict:
    path = out / "clips" / f"{row['id']}.npz"
    if not path.resolve().is_relative_to((out / "clips").resolve()):
        raise ValueError("캐시 경로가 출력 폴더 밖입니다.")
    clip = load_clip(path)
    if (clip["text"] != row["text"] or clip["source_text"] != row["source_text"]
            or len(clip["frames"]) != row["expected_frames"]):
        raise ValueError(f"재생성 캐시와 명세 불일치: {row['id']}")
    with np.load(path, allow_pickle=False) as data:
        for key in (*PROVENANCE, "temporal_method"):
            if key not in data or data[key].shape != () or str(data[key]) != str(row[key]):
                raise ValueError(f"캐시 출처/시간 메타데이터 불일치: {row['id']} {key}")
        rates = {}
        for key in ("video_fps", "label_fps"):
            if key not in data or data[key].shape != ():
                raise ValueError(f"캐시 시간 메타데이터 누락: {row['id']} {key}")
            value = float(data[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"유효하지 않은 FPS: {row['id']} {key}")
            rates[key] = value
    return {**row, **rates, "cache_path": path.relative_to(out).as_posix(),
            "cache_sha256": sha256_file(path), "frames": len(clip["frames"])}


def _receipt(row: dict) -> dict:
    return {k: row[k] for k in REBUILT_KEYS}


def _load_preparer():
    import cv2
    from prepare_data import TEMPORAL_METHOD as implemented_method, prepare_video
    if implemented_method != TEMPORAL_METHOD:
        raise ValueError("전처리 구현의 시간 버전이 다릅니다.")
    cv2.setNumThreads(2)
    return prepare_video


def _source_video(source: dict, label_path: str, temporary: Path, archives: dict) -> Path:
    """TAR 전체 대신 검증된 영상 한 개의 바이트만 고정 임시 경로에 복사한다."""
    expected_name = Path(label_path).stem + ".mp4"
    if "archive_member" in source:
        archive = archives.get(source["path"])
        if archive is None:
            archive = tarfile.open(source["path"], "r:")
            archives[source["path"]] = archive
        member = archive.getmember(source["archive_member"])
        if (not member.isfile() or member.size != source["size_bytes"]
                or Path(member.name).name != expected_name):
            raise ValueError("TAR 원본 영상 이름/크기/형식 불일치")
        video = temporary / expected_name
        with archive.extractfile(member) as src, video.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
        return video
    video = Path(source["path"])
    if video.name != expected_name or video.stat().st_size != source["size_bytes"]:
        raise ValueError("원본 영상 이름/크기 불일치")
    return video


def prepare(args) -> None:
    started = time.perf_counter()
    selection = Path(args.selection).resolve()
    out = selection.parent
    if (out / "manifest.json").exists():
        raise FileExistsError("완성된 데이터셋은 변경하지 않습니다.")
    document, _ = load_manifest(selection, verify_files=False)
    policy = document["policy"]
    parent_path = Path(policy["temporal_parent_manifest"]).resolve()
    _check_output_location(out, parent_path)
    if sha256_file(parent_path) != policy["temporal_parent_sha256"]:
        raise ValueError("부모 명세 SHA-256이 변경되었습니다.")
    parent, _ = load_manifest(parent_path)
    assert_parent_preserved(document, parent)
    progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    if (progress.get("schema") != PROGRESS_SCHEMA
            or progress.get("selection_sha256") != sha256_file(selection)
            or progress.get("temporal_parent_sha256") != policy["temporal_parent_sha256"]):
        raise ValueError("계획/부모/진행 기록 SHA-256 불일치")
    cache = out / "clips"
    if cache.resolve() != cache or (cache.exists() and not cache.is_dir()):
        raise ValueError("clips는 데이터셋 내부의 실제 폴더여야 합니다.")
    cache.mkdir(exist_ok=True)
    rows = {r["id"]: r for r in document["records"]}
    existing = {p.stem: p for p in cache.iterdir()}
    if any(p.name != f"{sample_id}.npz" or not p.is_file() or sample_id not in rows
           for sample_id, p in existing.items()):
        raise ValueError("계획에 없는 캐시 파일/폴더가 있습니다.")
    completed = progress.get("completed")
    if not isinstance(completed, dict) or existing.keys() != completed.keys():
        raise ValueError("캐시와 진행 해시 기록 불일치. 기록 없는 부분 파일은 새 출력에서 재생성하세요.")
    if existing and not args.resume:
        raise FileExistsError("부분 캐시가 있습니다. 모두 재검증하려면 --resume을 지정하세요.")
    for sample_id in existing:
        if _receipt(cached_row(rows[sample_id], out)) != completed[sample_id]:
            raise ValueError(f"기존 부분 캐시 SHA-256/메타데이터 변경: {sample_id}")
    prepare_video = _load_preparer()
    groups = defaultdict(list)
    for row in document["records"]:
        groups[row["label_path"]].append(row)
    lock = out / "prepare.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Timing rebuild is running; do not start another preparation process.\n")
    archives = {}
    try:
        for number, (label_path, records) in enumerate(sorted(groups.items()), 1):
            check_source_labels(records)
            source = records[0]["source_video"]
            if any(r["source_video"] != source for r in records):
                raise ValueError("같은 녹화의 원본 영상 정보 불일치")
            missing = [r for r in records if r["id"] not in completed]
            print(f"[시간 교정 {number}/{len(groups)}] {Path(label_path).stem}: {len(missing)}개", flush=True)
            if missing:
                with tempfile.TemporaryDirectory(prefix="source_", dir=out) as temporary:
                    video = _source_video(source, label_path, Path(temporary), archives)
                    # 일반 예외/부분 성공에도 정상적으로 저장된 파일은 해시를 봉인한다.
                    # 강제 종료 중 기록되지 않은 파일은 resume 시 임의 수용하지 않는다.
                    try:
                        count = prepare_video(str(video), label_path, str(cache),
                                              max_frames=policy["max_frames"], roi_mode="bbox",
                                              sentence_ids=[r["sentence_id"] for r in missing],
                                              provenance={r["sentence_id"]: {k: r[k] for k in PROVENANCE}
                                                          for r in missing})
                    finally:
                        for row in missing:
                            if (cache / f"{row['id']}.npz").exists():
                                completed[row["id"]] = _receipt(cached_row(row, out))
                        _save_progress(out, progress)
                    if count != len(missing):
                        raise RuntimeError("일부 문장 재생성 실패. 완성 명세를 생성하지 않습니다.")
            print(f"[진행] {len(completed)}/{len(rows)}개, {time.perf_counter()-started:.1f}초", flush=True)
        result = copy.deepcopy(document)
        result["selection_sha256"] = sha256_file(selection)
        result["records"] = [cached_row(r, out) for r in document["records"]]
        assert_parent_preserved(result, parent)
        if any(_receipt(r) != completed[r["id"]] for r in result["records"]):
            raise ValueError("작업 도중 캐시가 변경되었습니다.")
        summary = validate_manifest(result, out, verify_files=True)
        if (sha256_file(parent_path) != policy["temporal_parent_sha256"]
                or sha256_file(selection) != progress["selection_sha256"]):
            raise ValueError("작업 도중 부모/선정 명세 변경")
        _write_new_json(out / "manifest.json", result)
        summary.update(manifest_sha256=sha256_file(out / "manifest.json"),
                       temporal_parent_sha256=policy["temporal_parent_sha256"],
                       temporal_method=TEMPORAL_METHOD, rebuilt_clips=len(rows),
                       preserved_sample_ids_texts_times_speakers_splits=True,
                       test_inference_performed=False, test_visual_review_performed=False,
                       cache_bytes=sum(p.stat().st_size for p in cache.iterdir()),
                       preparation_seconds=time.perf_counter() - started)
        _write_new_json(out / "validation_report.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        for archive in archives.values():
            archive.close()
        lock.unlink()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--parent", default="data_cache/datasets/aihub_expanded_v3/manifest.json")
    p.add_argument("--out", default="data_cache/datasets/aihub_timed_v4")
    p = commands.add_parser("prepare")
    p.add_argument("--selection", default="data_cache/datasets/aihub_timed_v4/selection.json")
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (plan if args.command == "plan" else prepare)(args)


if __name__ == "__main__":
    main()
