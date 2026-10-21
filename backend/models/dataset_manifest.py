"""화자 분리 명세와 캐시의 정합을 검증하는 학습/평가 공용 계약."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np

from .cache import load_clip
from .tokenizer import normalize_text, validate_tokenizer_metadata

SCHEMA = "lipnet-speaker-split-v1"
ROI_METHOD = "aihub-bbox-yxyx-v2"
TEMPORAL_METHOD = "video-cfr-timestamps-v1"
SPLITS = ("train", "val", "test")
DEFAULT_MANIFEST = "data_cache/datasets/aihub_timed_v4/manifest.json"
_NAME = re.compile(r"^(lip_[A-Z]+_\d+_[FM]_\d+_([A-Z]\d+))_([A-Z])_(\d+)(?:_(\d+))?$")


def parse_identity(name: str) -> dict:
    """각도를 제외한 동일 녹화/발화 키를 파일명에서 구성한다."""
    match = _NAME.fullmatch(Path(name).stem)
    if not match:
        raise ValueError(f"AI Hub 파일명 형식 불일치: {name}")
    prefix, speaker, angle, recording, sentence = match.groups()
    recording_id = f"{prefix}_{int(recording):03d}"
    return {"speaker_id": speaker, "angle": angle, "recording_id": recording_id,
            "utterance_id": f"{recording_id}:{int(sentence):03d}" if sentence else None}


def sha256_file(path: str | Path) -> str:
    """대용량 파일도 일정 메모리로 해시한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_speakers(partitions: dict[str, set[str]], val_fraction: float, seed: int) -> dict[str, str]:
    """공식 Validation을 test로 보존하고 Training 화자만 train/val로 나눈다."""
    if not 0 < val_fraction < 1 or not math.isfinite(val_fraction):
        raise ValueError("val_fraction은 0과 1 사이여야 합니다.")
    training = set(partitions.get("Training", set()))
    testing = set(partitions.get("Validation", set()))
    if training & testing:
        raise ValueError("공식 Training/Validation에 동일 화자가 있습니다.")
    if len(training) < 2 or not testing:
        raise ValueError("화자 분리에 Training 2명 이상과 별도 Validation 1명 이상이 필요합니다.")
    speakers = sorted(training)
    random.Random(seed).shuffle(speakers)
    count = min(len(speakers) - 1, max(1, round(len(speakers) * val_fraction)))
    val = set(speakers[:count])
    return {**{s: "val" if s in val else "train" for s in training},
            **{s: "test" for s in testing}}


def validate_manifest(document: dict, base: Path, verify_files: bool = False) -> dict:
    """중복·화자/발화 누수·공식 구획 혼입 및 선택적으로 캐시 전체를 검사한다."""
    if document.get("schema") != SCHEMA:
        raise ValueError("지원하지 않는 데이터 분할 명세입니다.")
    validate_tokenizer_metadata(document.get("tokenizer", {}), "manifest")
    if document.get("policy", {}).get("roi_method") != ROI_METHOD:
        raise ValueError("검증되지 않은 ROI 전처리 버전입니다. 교정된 yxyx 명세가 필요합니다.")
    temporal_method = document["policy"].get("temporal_method")
    if temporal_method not in (None, TEMPORAL_METHOD):
        raise ValueError("지원하지 않는 시간축 전처리 버전입니다.")
    records = document.get("records", [])
    if not records:
        raise ValueError("데이터 명세가 비었습니다.")
    speakers, utterances, hashes, paths, ids = {}, {}, {}, set(), set()
    counts = {s: {"clips": 0, "speakers": set(), "texts": set(), "frames": 0} for s in SPLITS}
    for row in records:
        split, sample_id = row["split"], row["id"]
        if split not in SPLITS or sample_id in ids:
            raise ValueError(f"잘못된 분할 또는 중복 ID: {sample_id}")
        ids.add(sample_id)
        if row.get("roi_method") != ROI_METHOD:
            raise ValueError(f"{sample_id}: ROI 전처리 버전 불일치")
        if row.get("temporal_method") != temporal_method:
            raise ValueError(f"{sample_id}: 시간축 전처리 버전 불일치")
        identity = parse_identity(sample_id)
        for key in ("speaker_id", "angle", "recording_id", "utterance_id"):
            if row.get(key) != identity[key] or identity[key] is None:
                raise ValueError(f"{sample_id}: {key} 불일치")
        if row.get("source_partition") != ("Validation" if split == "test" else "Training"):
            raise ValueError(f"{sample_id}: 공식 구획 보존 규칙 위반")
        for values, key in ((speakers, "speaker_id"), (utterances, "utterance_id")):
            value = row[key]
            if value in values and values[value] != split:
                raise ValueError(f"분할 사이 {key} 누수: {value}")
            values[value] = split
        if not row["text"] or normalize_text(row["source_text"]) != row["text"]:
            raise ValueError(f"{sample_id}: 정답 불일치")
        info = counts[split]
        info["clips"] += 1
        info["speakers"].add(row["speaker_id"])
        info["texts"].add(row["text"])
        info["frames"] += row.get("frames", row.get("expected_frames", 0))
        if "cache_path" not in row:
            if verify_files:
                raise ValueError(f"{sample_id}: 아직 준비되지 않은 원본 명세입니다.")
            continue
        if temporal_method:
            for key in ("video_fps", "label_fps"):
                value = row.get(key)
                if (not isinstance(value, (float, int)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{sample_id}: 유효한 {key} 기록이 필요합니다.")
        relative = Path(row["cache_path"])
        path = (base / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(base.resolve()) or path in paths:
            raise ValueError(f"{sample_id}: 중복 또는 명세 폴더 밖의 캐시 경로")
        paths.add(path)
        if path.stem != sample_id:
            raise ValueError(f"{sample_id}: 캐시 파일명 불일치")
        digest = row["cache_sha256"]
        if digest in hashes and hashes[digest] != split:
            raise ValueError("분할 사이 동일 캐시 해시 누수")
        hashes[digest] = split
        if verify_files:
            if sha256_file(path) != digest:
                raise ValueError(f"{sample_id}: 캐시 SHA-256 불일치")
            clip = load_clip(path)
            if (clip["text"] != row["text"] or clip["source_text"] != row["source_text"]
                    or len(clip["frames"]) != row["frames"]):
                raise ValueError(f"{sample_id}: 명세와 캐시 내용 불일치")
            with np.load(path, allow_pickle=False) as cached:
                if ("temporal_method" in cached) != bool(temporal_method):
                    raise ValueError(f"{sample_id}: 캐시 시간축 출처 누락/혼합")
                if temporal_method:
                    if str(cached["temporal_method"]) != temporal_method:
                        raise ValueError(f"{sample_id}: 캐시 시간축 버전 불일치")
                    for key in ("video_fps", "label_fps"):
                        if key not in cached or float(cached[key]) != row[key]:
                            raise ValueError(f"{sample_id}: 캐시 {key} 불일치")
                for key in ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition", "roi_method", "label_sha256"):
                    if key not in cached or str(cached[key]) != str(row[key]):
                        raise ValueError(f"{sample_id}: 캐시 출처 {key} 불일치")
    if any(not counts[s]["clips"] for s in SPLITS):
        raise ValueError("train/val/test에 각각 실제 클립이 필요합니다.")
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}_{b}"] = len(counts[a]["texts"] & counts[b]["texts"])
    if document["policy"].get("require_disjoint_text") and any(overlaps.values()):
        raise ValueError("분할 사이 동일 정규화 정답 누수")
    return {"splits": {s: {"clips": v["clips"], "speaker_count": len(v["speakers"]),
                           "speaker_ids": sorted(v["speakers"]), "unique_texts": len(v["texts"]),
                           "duration_seconds": v["frames"] / 25} for s, v in counts.items()},
            "speaker_overlap": 0, "utterance_overlap": 0,
            "normalized_text_overlap": overlaps}


def load_manifest(path: str | Path, verify_files: bool = True) -> tuple[dict, dict]:
    """명세를 검증하고 고정 파일 해시와 함께 반환한다."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_manifest(document, path.parent, verify_files)
    summary["manifest_sha256"] = sha256_file(path)
    return document, summary


def split_files(document: dict, manifest_path: str | Path, split: str) -> list[str]:
    """정해진 집합만 반환한다. 파일 순서로 분할을 재생성하지 않는다."""
    if split not in SPLITS:
        raise ValueError(f"지원하지 않는 분할: {split}")
    base = Path(manifest_path).resolve().parent
    return [str(base / r["cache_path"]) for r in document["records"] if r["split"] == split]
