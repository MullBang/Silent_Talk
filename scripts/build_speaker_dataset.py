"""AI Hub 로컬 원본을 색인하고 화자 분리 파일럿을 준비한다. 원본은 읽기 전용.

index: 이미 풀린 MP4/JSON 및 비압축 TAR 내부 MP4를 읽어 정면 자료를 선정한다.
prepare: 선정된 TAR 영상만 임시로 풀어 라벨 bbox ROI를 만들고 즉시 정리한다.
validate: manifest와 모든 캐시의 출처/해시/라벨/분할을 다시 검사한다.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.dataset_manifest import (  # noqa: E402
    SCHEMA, TEMPORAL_METHOD, assign_speakers, load_manifest, parse_identity, sha256_file, validate_manifest,
)
from models.tokenizer import (  # noqa: E402
    ctc_min_input_len, encode_text, normalize_text, tokenizer_metadata,
)


def write_json(path: Path, value: dict) -> None:
    """기존 산출물을 덮어쓰지 않고 UTF-8 JSON으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require_current_timing(document: dict) -> None:
    """새 전처리/확대에서 구 시간축 캐시를 섞지 못하게 한다. 읽기 검증은 별개다."""
    if (document.get("policy", {}).get("temporal_method") != TEMPORAL_METHOD
            or any(row.get("temporal_method") != TEMPORAL_METHOD for row in document["records"])):
        raise ValueError(
            "구 시간축 명세로 새 캐시를 준비하거나 확대할 수 없습니다. "
            "scripts/rebuild_timed_dataset.py로 별도 데이터 버전을 먼저 재생성하세요."
        )


def cache_timing_metadata(path: Path) -> dict:
    """완성 NPZ에 실제 기록된 영상/라벨 FPS를 읽고 현재 시간축 버전을 확인한다."""
    import numpy as np
    with np.load(path, allow_pickle=False) as cached:
        if "temporal_method" not in cached or str(cached["temporal_method"]) != TEMPORAL_METHOD:
            raise ValueError(f"{path.name}: 구 시간축 캐시 혼합. rebuild_timed_dataset.py로 재생성하세요.")
        result = {"temporal_method": TEMPORAL_METHOD}
        for key in ("video_fps", "label_fps"):
            if key not in cached:
                raise ValueError(f"{path.name}: 캐시 {key}가 없습니다.")
            value = float(cached[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path.name}: 캐시 {key}가 유효하지 않습니다.")
            result[key] = value
    return result


def video_inventory(partition: Path) -> dict[str, dict]:
    """같은 이름의 이미 풀린 영상은 재사용하며 TAR 전체를 풀지 않는다."""
    sources = {}
    for path in sorted(partition.rglob("*.mp4")):
        if path.stem in sources:
            raise ValueError(f"중복 원본 영상 이름: {path.stem}")
        sources[path.stem] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
    for archive in sorted(partition.rglob("*.tar")):
        if not archive.name.startswith(("TS", "VS")):
            continue
        with tarfile.open(archive, "r:") as stream:
            for member in stream:
                if not member.isfile() or not member.name.lower().endswith(".mp4"):
                    continue
                stem = Path(member.name).stem
                if stem in sources:
                    if sources[stem]["size_bytes"] != member.size:
                        raise ValueError(f"동일 이름 원본 크기 불일치: {stem}")
                    if "archive_member" in sources[stem]:
                        raise ValueError(f"서로 다른 TAR에 중복된 영상: {stem}")
                    continue
                sources[stem] = {"path": str(archive.resolve()), "archive_member": member.name,
                                 "size_bytes": member.size}
    return sources


def stable_key(seed: int, value: str) -> str:
    """Python 해시 랜덤화/디렉터리 열거 순서와 무관한 선정 순서."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def select_pilot(rows: list[dict], speakers: dict[str, str], clips: int,
                 videos: int, seed: int) -> list[dict]:
    """화자별 같은 수의 문장을 여러 녹화에 분산하고 선정 결과를 고정한다."""
    if clips < videos or videos < 1:
        raise ValueError("화자당 문장 수는 영상 수 이상, 영상 수는 1 이상이어야 합니다.")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["speaker_id"]][row["recording_id"]].append(row)
    selected = []
    for speaker in sorted(speakers):
        candidates = grouped[speaker]
        per_video = math.ceil(clips / videos)
        usable = [k for k, v in candidates.items() if len(v) >= per_video]
        usable.sort(key=lambda k: stable_key(seed, k))
        if len(usable) < videos:
            raise ValueError(f"{speaker}: 영상 {videos}개에서 {clips}문장을 선정할 수 없습니다.")
        remaining = clips
        for index, key in enumerate(usable[:videos]):
            count = min(per_video, remaining - (videos - index - 1))
            pool = sorted(candidates[key], key=lambda r: stable_key(seed, r["id"]))
            for row in pool[:count]:
                selected.append({**row, "split": speakers[speaker]})
            remaining -= count
    return sorted(selected, key=lambda r: r["id"])


def build_index(args: argparse.Namespace) -> None:
    """로컬 자료의 가용성·정답 품질을 확인한 뒤 원본 선정 명세를 만든다."""
    root, out = Path(args.source_root).resolve(), Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"빈 새 출력 폴더가 필요합니다: {out}")
    rows, excluded, inventory, demographics = [], [], {}, {}
    partitions = {}
    for folder, name in (("1.Training", "Training"), ("2.Validation", "Validation")):
        partition = root / folder
        if not partition.is_dir():
            raise FileNotFoundError(partition)
        sources = video_inventory(partition)
        labels = {}
        for path in sorted(partition.rglob("*.json")):
            if path.stem in labels:
                raise ValueError(f"중복 라벨 이름: {path.stem}")
            labels[path.stem] = path
        frontal = {k: v for k, v in sources.items() if parse_identity(k)["angle"] == "A"}
        missing = sorted(set(frontal) - set(labels))
        if missing:
            raise ValueError(f"{name}: 정면 영상의 JSON 라벨이 없습니다: {missing[:5]}")
        source_speakers = {parse_identity(k)["speaker_id"] for k in sources}
        partitions[name] = source_speakers
        inventory[name] = {"video_count_all_angles": len(sources), "frontal_video_count": len(frontal),
                           "label_file_count": len(labels), "speakers": sorted(source_speakers),
                           "videos_in_tar_only": sum("archive_member" in s for s in sources.values())}
        for stem, source in sorted(frontal.items()):
            path = labels[stem]
            raw = path.read_bytes()
            data = json.loads(raw)
            if isinstance(data, list):
                if len(data) != 1:
                    raise ValueError(f"단일 녹화 라벨이 아닙니다: {path}")
                data = data[0]
            identity = parse_identity(stem)
            speaker_info = data["speaker_info"]
            if (speaker_info["speaker_ID"] != identity["speaker_id"]
                    or data["Video_env"]["Angle"] != "A"
                    or data["Video_info"]["video_Name"] != f"{stem}.mp4"):
                raise ValueError(f"파일명과 라벨 출처 불일치: {path}")
            demographics[identity["speaker_id"]] = speaker_info
            fps = float(data["Video_info"]["FPS"])
            boxes = data.get("Bounding_box_info", {}).get("Lip_bounding_box", {}).get("xtl_ytl_xbr_ybr", [])
            seen = set()
            for sentence in data["Sentence_info"]:
                sentence_id = int(sentence["ID"])
                sample_id = f"{stem}_{sentence_id:03d}"
                if sentence_id in seen:
                    raise ValueError(f"문장 ID 중복: {sample_id}")
                seen.add(sentence_id)
                try:
                    text = normalize_text(sentence["sentence_text"])
                    if not text:
                        raise ValueError("빈 정답")
                    start, end = float(sentence["start_time"]), float(sentence["end_time"])
                    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                        raise ValueError("잘못된 문장 시간")
                    frames = round((end - start) * 25)
                    ctc_min = ctc_min_input_len(encode_text(text))
                    if frames < ctc_min:
                        raise ValueError("CTC 최소 프레임 미달")
                    if frames > args.max_frames:
                        raise ValueError("파일럿 최대 프레임 초과")
                    # 라벨 FPS는 실제 영상 FPS와 다를 수 있어 bbox 끝 위치 추정에 쓰지 않는다.
                    # 실제 시각/영상 길이/전체 bbox 범위는 prepare_video에서 확정 검증한다.
                    if not math.isfinite(fps) or fps <= 0 or not boxes:
                        raise ValueError("유효한 라벨 FPS 또는 bbox가 없음")
                except ValueError as exc:
                    excluded.append({"id": sample_id, "reason": str(exc), "source_text": sentence["sentence_text"]})
                    continue
                rows.append({"id": sample_id, **parse_identity(sample_id), "sentence_id": sentence_id,
                             "source_partition": name, "source_video": source, "label_path": str(path),
                             "label_sha256": hashlib.sha256(raw).hexdigest(),
                             "source_text": sentence["sentence_text"], "text": text,
                             "start_time": start, "end_time": end, "expected_frames": frames,
                             "ctc_min_frames": ctc_min, "roi_method": "aihub-bbox-yxyx-v2",
                             "temporal_method": TEMPORAL_METHOD})
        print(f"[색인] {name}: {len(source_speakers)}명 / 정면 {len(frontal)}개", flush=True)
    mapping = assign_speakers(partitions, args.val_fraction, args.seed)
    selected = select_pilot(rows, mapping, args.clips_per_speaker, args.videos_per_speaker, args.seed)
    document = {"schema": SCHEMA, "tokenizer": tokenizer_metadata(),
                "policy": {"seed": args.seed, "official_validation_role": "local_frozen_test",
                           "training_val_speaker_fraction": args.val_fraction,
                           "angle": "A", "clips_per_speaker": args.clips_per_speaker,
                           "videos_per_speaker": args.videos_per_speaker, "max_frames": args.max_frames,
                           "split_unit": "speaker_id", "roi_method": "aihub-bbox-yxyx-v2",
                           "temporal_method": TEMPORAL_METHOD},
                "records": selected}
    summary = validate_manifest(document, out)
    write_json(out / "source_index.json", {"source_root": str(root), "inventory": inventory,
               "speaker_metadata": demographics, "eligible_records": rows, "excluded_records": excluded})
    write_json(out / "selection.json", document)
    write_json(out / "inventory_summary.json", {"inventory": inventory, "speaker_metadata": demographics,
               "eligible_frontal_sentences": len(rows), "excluded_frontal_sentences": len(excluded),
               "selection": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_selection(args: argparse.Namespace) -> None:
    """선정 문장을 실제 캐시로 변환하고 완성된 명세에만 manifest.json을 부여한다."""
    from models.cache import load_clip
    from prepare_data import prepare_video
    path = Path(args.selection).resolve()
    document, _ = load_manifest(path, verify_files=False)
    require_current_timing(document)
    out = path.parent
    cache = out / "clips"
    if (out / "manifest.json").exists() or (cache.exists() and any(cache.iterdir())):
        raise FileExistsError("이미 준비된 캐시가 있습니다. 새 빈 데이터셋 폴더를 사용하세요.")
    cache.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in document["records"]:
        groups[row["label_path"]].append(row)
    result = copy.deepcopy(document)
    result["selection_sha256"] = sha256_file(path)
    result["records"] = []
    archives = {}
    try:
        for index, (label_path, records) in enumerate(groups.items(), 1):
            source = records[0]["source_video"]
            if sha256_file(label_path) != records[0]["label_sha256"]:
                raise ValueError(f"선정 이후 라벨 변경: {label_path}")
            print(f"[준비 {index}/{len(groups)}] {Path(label_path).stem}", flush=True)
            with tempfile.TemporaryDirectory(prefix="source_", dir=out) as temporary:
                if "archive_member" in source:
                    archive = archives.get(source["path"])
                    if archive is None:
                        archive = tarfile.open(source["path"], "r:")
                        archives[source["path"]] = archive
                    member = archive.getmember(source["archive_member"])
                    if not member.isfile() or member.size != source["size_bytes"]:
                        raise ValueError("TAR 영상 크기 또는 형식이 바뀌었습니다.")
                    # TAR 경로를 extract하지 않고 검증된 basename으로 한 영상만 복사한다.
                    video = Path(temporary) / (Path(label_path).stem + ".mp4")
                    with archive.extractfile(member) as src, video.open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                else:
                    video = Path(source["path"])
                    if video.stat().st_size != source["size_bytes"]:
                        raise ValueError("선정 이후 원본 영상 크기 변경")
                keys = ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition", "roi_method", "label_sha256")
                provenance = {r["sentence_id"]: {k: r[k] for k in keys} for r in records}
                count = prepare_video(str(video), label_path, str(cache),
                                      max_frames=document["policy"]["max_frames"], roi_mode="bbox",
                                      sentence_ids=[r["sentence_id"] for r in records], provenance=provenance)
                if count != len(records):
                    raise RuntimeError("일부 선정 문장의 전처리에 실패했습니다. 완성 명세는 생성하지 않습니다.")
            for row in records:
                clip_path = cache / f"{row['id']}.npz"
                clip = load_clip(clip_path)
                if len(clip["frames"]) != row["expected_frames"]:
                    raise ValueError(f"{row['id']}: 영상 길이 부족 또는 리샘플 길이 불일치")
                result["records"].append({**row, "cache_path": clip_path.relative_to(out).as_posix(),
                                          "cache_sha256": sha256_file(clip_path), "frames": len(clip["frames"]),
                                          **cache_timing_metadata(clip_path)})
    finally:
        for archive in archives.values():
            archive.close()
    result["records"].sort(key=lambda r: r["id"])
    summary = validate_manifest(result, out, verify_files=True)
    write_json(out / "manifest.json", result)
    summary["manifest_sha256"] = sha256_file(out / "manifest.json")
    summary["cache_bytes"] = sum(p.stat().st_size for p in cache.glob("*.npz"))
    write_json(out / "validation_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    """색인/준비/검증 명령을 구분해 장시간 학습을 자동 시작하지 않는다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index")
    index.add_argument("--source-root", required=True, help="1.Training과 2.Validation을 포함하는 폴더")
    index.add_argument("--out", default="data_cache/datasets/aihub_pilot_timed_next")
    index.add_argument("--seed", type=int, default=42)
    index.add_argument("--val-fraction", type=float, default=0.2)
    index.add_argument("--clips-per-speaker", type=int, default=10)
    index.add_argument("--videos-per-speaker", type=int, default=2)
    index.add_argument("--max-frames", type=int, default=200)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--selection", default="data_cache/datasets/aihub_pilot_timed_next/selection.json")
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", default="data_cache/datasets/aihub_timed_v4/manifest.json")
    args = parser.parse_args()
    if args.command == "index":
        build_index(args)
    elif args.command == "prepare":
        prepare_selection(args)
    else:
        _, summary = load_manifest(args.manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
