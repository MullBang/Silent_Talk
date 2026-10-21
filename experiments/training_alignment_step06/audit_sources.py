"""Read bounded audio excerpts of the frozen59 training recordings; never train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import wave

import numpy as np
import scipy

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PRIOR = ROOT / "experiments/olkavs_official"
sys.path[:0] = [str(PRIOR)]
from align_step04_audio import correlate_windows, read_pcm
from inspect_timeline_step04 import inspect

ANCHORS = [30.0, 150.0, 270.0]
RATE, SECONDS, RADIUS = 8000, 6.0, 15.0
FFMPEG = PRIOR / "audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def require(ok: bool, reason: str) -> None:
    """Keep failures explicit rather than relaxing a frozen gate."""
    if not ok:
        raise ValueError(reason)


def sha(path: Path | str) -> str:
    """Hash only evidence files, not complete cloud source media."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path | str) -> dict:
    """Read a local frozen metadata document."""
    return json.loads(Path(path).read_text(encoding="utf8"))


def bind(path: Path) -> dict:
    """Bind local evidence to immutable content."""
    return {"path": str(path.resolve()), "sha256": sha(path)}


def save(path: Path, data: dict, *, replace: bool = False) -> None:
    """Write unique evidence or atomically update this run's progress manifest."""
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf8")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf8") as stream:
            stream.write(text)


def freeze() -> None:
    """Fix cohort, signal gates, storage bounds and execution order before media reads."""
    inventory_path = BASE / "training_inventory.json"
    inventory = read(inventory_path)
    for item in inventory["evidence_files"]:
        require(sha(item["path"]) == item["sha256"], "Inventory provenance changed")
    sources = inventory["sources"]
    require(len(sources) == 59 and len({s["source_stem"] for s in sources}) == 59, "Expected distinct59 training recordings")
    require(len({s["speaker_id"] for s in sources}) == 8, "Expected8 training speakers")
    require(sum(len(s["training_records"]) for s in sources) == 512 and
            all(r["split"] == "train" for s in sources for r in s["training_records"]), "Training512-only scope required")
    selected = [s for s in sources if s["phase"] == "screen8"]
    require(len(selected) == 8 and len({s["speaker_id"] for s in selected}) == 8, "Expected one fixed screen per speaker")
    for speaker in sorted({s["speaker_id"] for s in sources}):
        require(next(s for s in selected if s["speaker_id"] == speaker)["source_stem"] ==
                min(s["source_stem"] for s in sources if s["speaker_id"] == speaker), "Screen selection differs")
    order = [s["source_stem"] for phase in ["screen8", "remaining51"]
             for s in sorted(sources, key=lambda s: s["source_stem"]) if s["phase"] == phase]
    require(len(order) == 59, "Unplanned phase")
    expected_media_bytes = sum(s.get(k, {}).get("size_bytes", 0) if s.get(k) else 0
                               for s in sources for k in ["video", "audio"])
    free = shutil.disk_usage(ROOT).free
    require(free > expected_media_bytes + 10 * 1024**3, "Insufficient headroom for possible Drive caching")
    plan = {"schema": "training-step06-waveform-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_media_read": True, "inventory": bind(inventory_path), "source_order": order,
            "source_count": 59, "speaker_count": 8, "screen_count": 8,
            "code": [bind(Path(__file__)), bind(PRIOR / "align_step04_audio.py"), bind(PRIOR / "inspect_timeline_step04.py")],
            "ffmpeg": bind(FFMPEG), "ffmpeg_version": subprocess.check_output([str(FFMPEG), "-version"], text=True).splitlines()[0],
            "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
            "method": {"reference_clock": "original WAV", "search_clock": "MP4 first audio track",
                       "anchors_seconds": ANCHORS, "reference_seconds": SECONDS, "search_radius_seconds": RADIUS,
                       "search_seconds": SECONDS + 2*RADIUS, "rate": RATE, "channels": 1, "sample_width_bytes": 2,
                       "filter": "4th-order Butterworth 100-3400Hz zero-phase",
                       "score": "absolute centered normalized cross-correlation; preserve signed value",
                       "minimum_correlation": .99, "minimum_far_peak_ratio": 1.5,
                       "runner_up_exclusion_seconds": .25, "minimum_filtered_rms": 1e-4,
                       "minimum_search_boundary_clearance_seconds": .5, "maximum_offset_range_exclusive_seconds": .02,
                       "offset_formula": "matched_MP4_start - original_WAV_anchor", "rounding": "median to1/8000s sample grid",
                       "significant_offset_reporting_threshold_seconds": .1,
                       "short_source": "quarantine if any fixed window is out of range; no adaptive anchors",
                       "failed_source": "retain reason; never infer offset from another recording or duration difference",
                       "source_timeout_seconds_per_excerpt": 90},
            "storage": {"available_workspace_volume_bytes_at_freeze": free,
                        "maximum_inventoried_media_bytes_subject_to_Drive_caching": expected_media_bytes,
                        "planned_PCM_payload_bytes": 59 * 3 * 42 * RATE * 2,
                        "original_copy_created": False, "Drive_cache_bytes_measured": False},
            "scope": {"original_writes": False, "label_edits": False, "training": False,
                      "model_inference": False, "test_payload": False, "ASR": False,
                      "model_outputs_for_offsets": False, "new_dataset_download": False},
            "limits": ["Audio clock correspondence does not prove transcript correctness or lip-audio synchronization.",
                       "Three fixed anchors cannot rule out every internal edit between anchors.",
                       "File stat plus sparse payload hashes do not validate every byte of the full media.",
                       "Google Drive may hydrate more source bytes than requested; no full original copies are created."]}
    save(BASE / "plan.json", plan)
    print(json.dumps({"plan": str(BASE / "plan.json"), "sha256": sha(BASE / "plan.json"), "storage": plan["storage"]}, ensure_ascii=False), flush=True)


def stat_source(spec: dict | None) -> dict:
    """Check a unique inventory match still exists without reading its payload."""
    require(spec is not None and "path" in spec, "Missing or ambiguous inventory match")
    path = Path(spec["path"])
    current = path.stat()
    require(path.is_file() and current.st_size > 0, "Missing/empty source")
    require(current.st_size == spec["size_bytes"], "Source size changed since inventory")
    if "mtime_ns" in spec:
        require(current.st_mtime_ns == spec["mtime_ns"], "Source mtime changed since inventory")
    return {"path": str(path), "size_bytes": current.st_size, "mtime_ns": current.st_mtime_ns}


def excerpt(source: Path, start: float, seconds: float, output: Path, ffmpeg: str) -> dict:
    """Decode a bounded audio-only range and require exact expected PCM length."""
    require(not output.exists(), "Refuse to overwrite PCM evidence")
    command = [ffmpeg, "-nostdin", "-v", "error", "-ss", f"{start:.9f}", "-i", str(source),
               "-t", f"{seconds:.9f}", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(RATE),
               "-c:a", "pcm_s16le", "-n", str(output)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=90)
    require(process.returncode == 0, process.stderr.strip() or "FFmpeg failed")
    with wave.open(str(output), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (RATE, 1, 2, "NONE"), "Unexpected PCM format")
        require(stream.getnframes() == round(seconds*RATE), "Truncated PCM excerpt")
        frames = stream.getnframes()
    return {**bind(output), "source": str(source), "start_seconds": start, "requested_seconds": seconds,
            "frames": frames, "sample_rate": RATE, "bytes": output.stat().st_size, "command": command}


def strict_score(reference: dict, search: dict) -> dict:
    """Apply frozen gates to the unchanged stage04 signal estimator."""
    a, b = read_pcm(reference["path"]), read_pcm(search["path"])
    require(a.ndim == b.ndim == 1 and np.isfinite(a).all() and np.isfinite(b).all() and len(b) > len(a) > 100, "Invalid signals")
    score = correlate_windows(a, b)
    clearance = min(score["best_start_sample_in_search"], len(b)-len(a)-score["best_start_sample_in_search"]) / RATE
    gates = {"correlation": score["absolute_correlation"] >= .99, "unique_far_peak": score["peak_ratio"] >= 1.5,
             "reference_energy": score["reference_rms_filtered"] >= 1e-4,
             "matched_energy": score["matched_search_rms_filtered_centered"] >= 1e-4, "boundary_clearance": clearance >= .5}
    matched = search["start_seconds"] + score["best_start_seconds_in_search"]
    score.update({"matched_mp4_start_seconds": matched,
                  "offset_mp4_minus_original_seconds": matched-reference["start_seconds"],
                  "search_boundary_clearance_seconds": clearance, "strict_gates": gates,
                  "strict_anchor_pass": all(gates.values())})
    return score


def one_source(source: dict, plan: dict, output: Path) -> dict:
    """Inspect metadata/labels and compare three independent waveform anchors."""
    started = time.perf_counter()
    item = {"source_stem": source["source_stem"], "speaker_id": source["speaker_id"], "phase": source["phase"],
            "status": "running", "anchors": [], "source_metadata": {}, "branch_sample_ids": source["branch_sample_ids"]}
    try:
        for kind in ["video", "audio", "label"]:
            item["source_metadata"][kind] = stat_source(source.get(kind))
        video, audio, label = (Path(item["source_metadata"][kind]["path"]) for kind in ["video", "audio", "label"])
        require(label.stat().st_size < 16*1024*1024, "Unexpected label size")
        require(sha(label) == source["expected_label_sha256"], "Original training label hash differs")
        doc = json.loads(label.read_text(encoding="utf-8-sig"))
        require(isinstance(doc, list) and len(doc) == 1, "Unexpected label schema")
        doc = doc[0]
        require(Path(doc["Audio_info"]["Audio_Name"]).name == audio.name and
                Path(doc["Video_info"]["video_Name"]).name == video.name, "Label refers to different audio/video")
        annotated = {int(r["ID"]): r for r in doc["Sentence_info"]}
        for record in source["training_records"]:
            target = annotated[int(record["sentence_id"])]
            require(target["sentence_text"] == record["source_text"] and
                    abs(target["start_time"]-record["start_time"]) < 1e-9 and abs(target["end_time"]-record["end_time"]) < 1e-9,
                    "Selected training record differs from original label")
        metadata = inspect(video)
        video_tracks = [t for t in metadata["tracks"] if t["handler"] == "vide"]
        audio_tracks = [t for t in metadata["tracks"] if t["handler"] == "soun"]
        require(len(video_tracks) == len(audio_tracks) == 1, "Need one video and one audio track")
        def conservative_duration(track: dict) -> float:
            durations = [track["media_clock"]["duration_seconds"], metadata["movie_clock"]["duration_seconds"]]
            edits = track.get("edit_list", [])
            if edits:
                require(all(e["media_rate_integer"] == 1 and e["media_rate_fraction"] == 0 for e in edits), "Non-unit track edit rate")
                durations.append(sum(e["movie_duration_seconds"] for e in edits))
            return min(durations)
        video_duration = conservative_duration(video_tracks[0])
        audio_duration = conservative_duration(audio_tracks[0])
        item["duration_bound_definition"] = "minimum of track mdhd, movie mvhd and rate1 edit-list presentation durations; later crop decode must still validate actual PTS"
        item["mp4_metadata"] = metadata
        with wave.open(str(audio), "rb") as stream:
            header = {"channels": stream.getnchannels(), "sample_width_bytes": stream.getsampwidth(),
                      "sample_rate": stream.getframerate(), "frame_count": stream.getnframes(),
                      "duration_seconds": stream.getnframes()/stream.getframerate(), "compression": stream.getcomptype()}
        item["original_wav_header"] = header
        require(header["sample_rate"] == 48000 and header["compression"] == "NONE", "Unexpected original WAV format")
        item["label_metadata"] = {"audio_info": doc["Audio_info"], "video_info": doc["Video_info"],
                                  "label_sha256": source["expected_label_sha256"], "selected_training_records_verified": len(source["training_records"]),
                                  "sentence_count": len(annotated), "last_label_end_seconds": max(r["end_time"] for r in annotated.values()),
                                  "label_clock_assumption": "Audio_Name matches original WAV; relative audio offset does not independently validate spoken transcripts"}
        require(all(t + SECONDS <= header["duration_seconds"] and t-RADIUS >= 0 and
                    t+RADIUS+SECONDS <= audio_duration for t in ANCHORS), "Source shorter than fixed anchor/search windows")
        folder = output / "pcm" / source["source_stem"]
        folder.mkdir()
        for anchor in ANCHORS:
            reference = excerpt(audio, anchor, SECONDS, folder / f"t{int(anchor):03}_reference.wav", plan["ffmpeg"]["path"])
            search = excerpt(video, anchor-RADIUS, SECONDS+2*RADIUS, folder / f"t{int(anchor):03}_search.wav", plan["ffmpeg"]["path"])
            score = strict_score(reference, search)
            item["anchors"].append({"reference": reference, "mp4_search": search, "score": score})
        offsets = [r["score"]["offset_mp4_minus_original_seconds"] for r in item["anchors"]]
        offset = round(statistics.median(offsets)*RATE)/RATE
        strict = all(r["score"]["strict_anchor_pass"] for r in item["anchors"]) and max(offsets)-min(offsets) < .02
        for kind in ["video", "audio", "label"]:
            require(stat_source(source[kind]) == item["source_metadata"][kind], "Source changed while inspecting")
        require(sha(label) == source["expected_label_sha256"], "Label changed while inspecting")
        item["summary"] = {"median_offset_seconds": offset, "offset_range_seconds": max(offsets)-min(offsets),
                           "minimum_correlation": min(r["score"]["absolute_correlation"] for r in item["anchors"]),
                           "minimum_far_peak_ratio": min(r["score"]["peak_ratio"] for r in item["anchors"]),
                           "all_anchors_strict": all(r["score"]["strict_anchor_pass"] for r in item["anchors"]),
                           "constant_mapping_supported": strict, "abs_offset_at_least_100ms": abs(offset) >= .1,
                           "video_duration_seconds": video_duration,
                           "wav_minus_video_duration_seconds": header["duration_seconds"]-video_duration}
        item["status"] = "supported" if strict else "quarantined"
        if not strict:
            item["reason"] = "At least one strict anchor gate or constant-offset range failed"
        else:
            affected = []
            for record in source["training_records"]:
                start, end = record["start_time"]+offset, record["end_time"]+offset
                fps, n = record["video_fps"], record["frames"]
                changed = sum(round((record["start_time"]+k/25)*fps) != round((start+k/25)*fps) for k in range(n))
                affected.append({"id": record["id"], "video_start_time": start, "video_end_time": end,
                                 "fully_within_video_duration": start >= 0 and end <= video_duration,
                                 "target_frames": n, "changed_selected_frame_indices": changed})
            item["training_frame_selection_impact"] = affected
    except Exception as exc:
        item["status"] = "quarantined"
        item["reason"] = f"{type(exc).__name__}: {exc}"
    item["seconds"] = time.perf_counter()-started
    return item


def run(phase: str) -> None:
    """Audit the first8 or remaining51 once, preserving each source outcome."""
    plan = read(BASE / "plan.json")
    for item in [plan["inventory"], plan["ffmpeg"], *plan["code"]]:
        require(sha(item["path"]) == item["sha256"], "Frozen evidence/code changed")
    require(plan["environment"] == {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}, "Numerical runtime changed")
    inventory = read(plan["inventory"]["path"])
    by_stem = {s["source_stem"]: s for s in inventory["sources"]}
    output = BASE / "results"
    output.mkdir(exist_ok=True)
    (output / "pcm").mkdir(exist_ok=True)
    (output / "sources").mkdir(exist_ok=True)
    if phase == "remaining51":
        require(all((output / "sources" / f"{stem}.json").exists() for stem in plan["source_order"][:8]), "Screen8 must finish before expansion")
    if not (output / "availability.json").exists():
        availability = []
        for stem in plan["source_order"]:
            row = {"source_stem": stem, "files": {}}
            for kind in ["video", "audio", "label"]:
                try:
                    row["files"][kind] = {"status": "available", **stat_source(by_stem[stem].get(kind))}
                except Exception as exc:
                    row["files"][kind] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            availability.append(row)
        save(output / "availability.json", {"schema": "training-step06-availability-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                                            "plan_sha256": sha(BASE / "plan.json"), "sources": availability})
    for stem in plan["source_order"]:
        source = by_stem[stem]
        if source["phase"] != phase:
            continue
        target = output / "sources" / f"{stem}.json"
        if target.exists():
            previous = read(target)
            require(previous["plan_sha256"] == sha(BASE / "plan.json"), "Existing source result has another plan")
            continue
        require(shutil.disk_usage(ROOT).free >= 10*1024**3, "Workspace headroom below10GiB; stop before more cloud reads")
        value = one_source(source, plan, output)
        value["plan_sha256"] = sha(BASE / "plan.json")
        save(target, value)
        completed = [read(output / "sources" / f"{s}.json") for s in plan["source_order"]
                     if (output / "sources" / f"{s}.json").exists()]
        progress = {"schema": "training-step06-progress-v1", "status": "completed" if len(completed) == 59 else "running",
                    "plan_sha256": sha(BASE / "plan.json"), "completed_sources": len(completed),
                    "supported_sources": sum(s["status"] == "supported" for s in completed),
                    "quarantined_sources": sum(s["status"] == "quarantined" for s in completed),
                    "source_results": [bind(output / "sources" / f"{s['source_stem']}.json") for s in completed],
                    "workspace_free_bytes": shutil.disk_usage(ROOT).free, "scope": plan["scope"]}
        save(output / "progress.json", progress, replace=True)
        print(json.dumps({"completed": len(completed), "stem": stem, "status": value["status"],
                          "summary": value.get("summary"), "reason": value.get("reason"), "seconds": value["seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--phase", choices=["screen8", "remaining51"])
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.phase:
        run(args.phase)
    else:
        parser.error("Choose --freeze or --phase")
