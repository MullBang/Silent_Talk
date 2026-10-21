"""Measure stage11 input-pixel controls without loading a model or importing torch.

These descriptive pixel distances are not speech information or meaningful
phonetic differences. Only the planned train control union and validation clips
are opened; their overlapping reporting cohorts remain separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]

from config import NORMALIZE_MEAN, NORMALIZE_STD
from models.cache import load_clip
from models.dataset_manifest import TEMPORAL_METHOD, load_manifest, sha256_file
from temporal_diagnostics import transform_frames

MEAN = np.asarray(NORMALIZE_MEAN, dtype=np.float32)
STD = np.asarray(NORMALIZE_STD, dtype=np.float32)
CONDITIONS = [("normal", 42), ("static_middle", 42), ("shuffle", 42),
              ("shuffle", 137), ("shuffle", 2026), ("reversed", 42), ("constant_gray", 42)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def length_bin(frames: int) -> str:
    require(1 <= frames <= 200, "1..200 frames required")
    start = ((frames - 1) // 50) * 50 + 1
    return f"{start:03d}-{start + 49:03d}"


def verified_cohorts(records: list[dict], planned: dict) -> dict:
    """Independently reproduce the predeclared metadata-only cohort selection."""
    train = [row for row in records if row["split"] == "train"]
    groups = defaultdict(list)
    for row in train:
        groups[(row["speaker_id"], length_bin(row["frames"]))].append(row)
    selected, strata = [], []
    for (speaker, bucket), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: hashlib.sha256(
            f"step11-cohort-42:{row['id']}".encode("utf-8")).hexdigest())
        chosen = [row["id"] for row in ordered[:2]]
        selected.extend(chosen)
        strata.append({"speaker": speaker, "length_bin": bucket,
                       "available": len(rows), "ids": chosen})
    short = [row["id"] for row in sorted(train, key=lambda row: (row["frames"], row["id"]))[:16]]
    actual = {"train_stratified": sorted(selected), "short16": short,
              "train_controls_union": sorted(set(selected) | set(short)), "strata": strata,
              "val": [row["id"] for row in records if row["split"] == "val"]}
    require(actual == planned, "Plan cohort selection does not match manifest")
    require({name: len(actual[name]) for name in ("val", "train_stratified", "short16", "train_controls_union")}
            == {"val": 20, "train_stratified": 60, "short16": 16, "train_controls_union": 73},
            "Unexpected stage11 cohort counts")
    require(not set(actual["val"]) & set(actual["train_controls_union"]), "Train/val overlap")
    return actual


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def pixel_stats(values: np.ndarray) -> dict:
    """Equal weighting of all T,H,W,C entries within each clip."""
    x = values.astype(np.float64, copy=False)
    magnitude = rms(x)
    centered = rms(x - x.mean(axis=0, keepdims=True))
    adjacent = rms(np.diff(x, axis=0)) if len(x) > 1 else 0.0
    return {"rms": magnitude, "temporal_centered_rms": centered,
            "temporal_centered_relative_rms": centered / max(magnitude, 1e-12),
            "adjacent_rms": adjacent, "adjacent_relative_rms": adjacent / max(magnitude, 1e-12)}


def pixel_distance(reference: np.ndarray, changed: np.ndarray) -> dict:
    require(reference.shape == changed.shape, "Pixel array shapes differ")
    a, b = reference.astype(np.float64, copy=False), changed.astype(np.float64, copy=False)
    difference = b - a
    magnitude = rms(difference)
    return {"mae": float(np.abs(difference).mean()), "rms_difference": magnitude,
            "relative_rms_difference": magnitude / max(rms(a), 1e-12)}


def normalized(frames: np.ndarray) -> np.ndarray:
    """Same float32 channelwise arithmetic as scripts/test_model.py::_to_tensor."""
    return (frames.astype(np.float32) / 255.0 - MEAN) / STD


def describe(values: list[float]) -> dict:
    require(bool(values), "Cannot summarize an empty cohort")
    array = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(array).all()), "Nonfinite pixel metric")
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "minimum": float(array.min()), "maximum": float(array.max())}


def summarize(rows: list[dict]) -> dict:
    result = {"clips": len(rows), "ids": [row["id"] for row in rows],
              "total_frames": sum(row["frames"] for row in rows),
              "changed_frame_fraction": describe([row["changed_frame_fraction"] for row in rows]),
              "unchanged_clip_count": sum(row["changed_frame_count"] == 0 for row in rows)}
    for space in ("raw_01", "imagenet_normalized"):
        result[space] = {}
        for kind in ("statistics", "comparison_to_normal"):
            result[space][kind] = {
                name: describe([row[space][kind][name] for row in rows])
                for name in rows[0][space][kind]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/step11-temporal/run01")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run = (ROOT / args.run).resolve()
    require(run.is_relative_to(ROOT / "results"), "Run must be inside workspace results")
    output = run / "input_pixels.json"
    require(not output.exists(), "Refusing to overwrite input_pixels.json")
    plan_path = run / "plan.json"
    plan_hash = sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    require(plan["stage"] == 11 and plan["training_performed"] is False
            and plan["test_inference_performed"] is False, "Wrong diagnosis plan")
    expected_conditions = [{"name": f"shuffle_{seed}" if name == "shuffle" else name,
                            "transform": name, "seed": seed} for name, seed in CONDITIONS]
    require(plan["conditions"] == expected_conditions, "Unexpected control plan")
    manifest_path = Path(plan["manifest"]).resolve()
    require(manifest_path == ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json", "Unexpected manifest")
    require(sha256_file(manifest_path) == plan["manifest_sha256"], "Manifest hash mismatch")
    # verify_files=False checks metadata, including test metadata, without opening any test cache.
    document, manifest_summary = load_manifest(manifest_path, verify_files=False)
    require(document["policy"]["temporal_method"] == TEMPORAL_METHOD, "Corrected timestamps required")
    require(manifest_summary["manifest_sha256"] == plan["manifest_sha256"], "Loaded manifest mismatch")
    cohorts = verified_cohorts(document["records"], plan["cohorts"])
    sources_before = {name: sha256_file(ROOT / name) for name in plan["sources"]}
    require(sources_before == plan["sources"], "Frozen diagnosis source hash mismatch")
    script_hash = sha256_file(Path(__file__))
    by_id = {row["id"]: row for row in document["records"]}
    selected_ids = cohorts["train_controls_union"] + cohorts["val"]
    require(len(selected_ids) == len(set(selected_ids)) == 93, "Expected exactly93 unique clips")
    rows, cache_hashes = [], {}
    for index, sample_id in enumerate(selected_ids):
        source = by_id[sample_id]
        require(source["split"] in ("train", "val"), "Test cache access forbidden")
        cache_path = (manifest_path.parent / source["cache_path"]).resolve()
        require(cache_path.is_relative_to(manifest_path.parent), "Cache outside manifest folder")
        digest = sha256_file(cache_path)
        require(digest == source["cache_sha256"], f"Cache hash mismatch: {sample_id}")
        item = load_clip(cache_path)
        frames = item["frames"]
        require(item["text"] == source["text"] and item["source_text"] == source["source_text"]
                and len(frames) == source["frames"], f"Cache contents mismatch: {sample_id}")
        with np.load(cache_path, allow_pickle=False) as cache:
            for field in ("speaker_id", "utterance_id", "recording_id", "angle", "source_partition",
                          "roi_method", "label_sha256", "temporal_method", "video_fps", "label_fps"):
                require(str(cache[field]) == str(source[field]), f"Cache metadata mismatch: {sample_id}:{field}")
        before_pixels_hash = hashlib.sha256(frames.tobytes()).hexdigest()
        raw_normal = frames.astype(np.float64) / 255.0
        norm_normal = normalized(frames)
        for spec in expected_conditions:
            changed, metadata = transform_frames(frames, spec["transform"], sample_id=sample_id, seed=spec["seed"])
            require(changed.shape == frames.shape and changed.dtype == frames.dtype
                    and not np.shares_memory(changed, frames), "Input transform contract mismatch")
            changed_count = int(np.any(changed != frames, axis=(1, 2, 3)).sum())
            require(changed_count == metadata["pixel_changed_frame_count"], "Changed frame metadata mismatch")
            row = {"id": sample_id, "split": source["split"], "speaker_id": source["speaker_id"],
                   "frames": len(frames), "shape": list(frames.shape), "length_bin": length_bin(len(frames)),
                   "stratified": sample_id in cohorts["train_stratified"], "short16": sample_id in cohorts["short16"],
                   "condition": spec["name"], "transform": metadata, "cache_sha256": digest,
                   "changed_frame_count": changed_count, "changed_frame_fraction": changed_count / len(frames)}
            for space, reference, values in (
                    ("raw_01", raw_normal, changed.astype(np.float64) / 255.0),
                    ("imagenet_normalized", norm_normal, normalized(changed))):
                row[space] = {"statistics": pixel_stats(values),
                              "comparison_to_normal": pixel_distance(reference, values)}
            rows.append(row)
        require(hashlib.sha256(frames.tobytes()).hexdigest() == before_pixels_hash, "Input mutated in memory")
        require(sha256_file(cache_path) == digest, "Input cache changed while measuring")
        cache_hashes[sample_id] = {"path": str(cache_path), "sha256": digest, "split": source["split"]}
        if (index + 1) % 10 == 0 or index + 1 == len(selected_ids):
            print(json.dumps({"pixel_clips": index + 1, "total": 93, "rows": len(rows)}), flush=True)
    require(len(rows) == 651, "Expected651 pixel rows")
    groups = {}
    for cohort in ("val", "train_stratified", "short16"):
        ids = set(cohorts[cohort])
        for spec in expected_conditions:
            chosen = [row for row in rows if row["id"] in ids and row["condition"] == spec["name"]]
            require(len(chosen) == len(ids), "Missing cohort condition")
            groups[f"{cohort}/{spec['name']}"] = summarize(chosen)
    sources_after = {name: sha256_file(ROOT / name) for name in plan["sources"]}
    require(sources_after == sources_before, "Frozen source changed while measuring")
    require(sha256_file(plan_path) == plan_hash and sha256_file(manifest_path) == plan["manifest_sha256"],
            "Plan/manifest changed while measuring")
    require(sha256_file(Path(__file__)) == script_hash, "Pixel script changed while measuring")
    require("torch" not in sys.modules, "Pixel inspection must not import torch")
    result = {
        "schema": "lipnet-temporal-input-pixels-v1", "status": "completed", "stage": 11,
        "created_at": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()),
        "script_sha256": script_hash, "plan": str(plan_path), "plan_sha256": plan_hash,
        "manifest": str(manifest_path), "manifest_sha256": plan["manifest_sha256"],
        "sources_before": sources_before, "sources_after": sources_after,
        "source_hashes_unchanged": True, "model_inference_performed": False, "training_performed": False,
        "test_cache_opened": False, "torch_imported": False, "numpy_version": np.__version__,
        "clips": len(selected_ids), "rows_count": len(rows), "conditions": expected_conditions,
        "cohorts": cohorts, "cache_hashes": cache_hashes,
        "normalization": {"mean": list(NORMALIZE_MEAN), "std": list(NORMALIZE_STD),
                          "formula": "(uint8.astype(float32)/255.0 - mean_float32) / std_float32",
                          "source": "scripts/test_model.py::_to_tensor; channel order RGB",
                          "statistics_accumulator_dtype": "float64"},
        "definitions": {
            "raw_01": "uint8 RGB converted to float64 and divided by255",
            "mae": "mean(abs(control-normal)) across all T,H,W,C entries in each clip",
            "rms_difference": "sqrt(mean((control-normal)^2)) across all T,H,W,C entries",
            "relative_rms_difference": "rms_difference / max(RMS(normal),1e-12), separately in each pixel space",
            "rms": "sqrt(mean(x^2)) across all T,H,W,C entries",
            "temporal_centered_rms": "sqrt(mean((x-mean_T(x))^2)); per-pixel/channel temporal mean",
            "adjacent_rms": "sqrt(mean((x[t+1]-x[t])^2));0 for T1",
            "temporal_centered_relative_rms": "temporal_centered_rms / max(RMS(control),1e-12)",
            "adjacent_relative_rms": "adjacent_rms / max(RMS(control),1e-12)",
            "changed_frame_fraction": "frames with at least one RGB byte different from normal at the same time position divided by T",
            "group_weighting": "arithmetic mean of per-clip scalar metrics; equal clip weights, not pooled pixels/frames",
            "shuffle_repeats": "42,137,2026 are separate conditions on the same clips; no sample-size multiplication",
        },
        "limits": [
            "Pixel distances describe numerical input changes, not information content or useful phonetic differences.",
            "Only planned train control union73 and val20 were opened; these are not whole-train1000 pixel statistics.",
            "Train stratified60 and short16 overlap in3 clips; their results must not be added as independent cohorts.",
            "All controls preserve T; length and static appearance can still affect outputs.",
            "Same-position distances for shuffle/reverse include relocation of the same frames; they do not measure lost content.",
            "Feature and pixel spaces have different dimensions/scales; compare conditions within one stage, not information loss across stages.",
            "Changed-frame fraction detects any byte difference, not the magnitude or perceptual importance of change.",
        ],
        "groups": groups, "rows": rows,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "output": str(output), "sha256": sha256_file(output),
                      "clips": len(selected_ids), "rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
