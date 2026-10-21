"""Independently audit saved Step 06 PCM and bookkeeping; no cloud/model reads.

The producer correlator is NOT imported. Numerators and both sliding moments
are recomputed with FFT convolution (producer uses cumulative-sum moments),
and correlations at the winning and runner-up positions use direct Pearson.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RATE = 8000


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError(reason)


def local(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    require(resolved.is_relative_to(ROOT), f"Refuse non-workspace read: {resolved}")
    return resolved


def sha(path: str | Path) -> str:
    with local(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: str | Path) -> dict:
    return json.loads(local(path).read_text(encoding="utf8"))


def close(actual: float, expected: float, field: str, tolerance: float = 1e-7) -> None:
    require(math.isfinite(actual) and math.isfinite(expected), f"Nonfinite {field}")
    require(abs(actual-expected) <= tolerance, f"{field}: {actual} != {expected}")


def check_binding(spec: dict) -> None:
    require(sha(spec["path"]) == spec["sha256"], f"Binding changed: {spec['path']}")


def pcm(spec: dict, duration: float, source: str, start: float) -> np.ndarray:
    check_binding(spec)
    path = local(spec["path"])
    require(path.is_relative_to(BASE / "results/pcm"), "PCM outside audit evidence directory")
    require(spec["source"] == source, "PCM source metadata differs")
    close(spec["start_seconds"], start, "excerpt start")
    close(spec["requested_seconds"], duration, "excerpt duration")
    command = spec["command"]
    require(command[command.index("-i")+1] == source, "FFmpeg source differs")
    close(float(command[command.index("-ss")+1]), start, "FFmpeg seek")
    close(float(command[command.index("-t")+1]), duration, "FFmpeg duration")
    require(command[command.index("-map")+1] == "0:a:0", "Unexpected source track")
    require(command[command.index("-ac")+1] == "1", "Unexpected downmix")
    require(command[command.index("-ar")+1] == str(RATE), "Unexpected resampling rate")
    require(Path(command[-1]).resolve() == path, "FFmpeg output differs")
    require(spec["bytes"] == path.stat().st_size, "PCM byte count differs")
    with wave.open(str(path), "rb") as stream:
        require((stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) ==
                (RATE, 1, 2, "NONE"), "PCM format differs")
        n = stream.getnframes()
        require(n == spec["frames"] == round(duration*RATE), "PCM sample count differs")
        values = np.frombuffer(stream.readframes(n), dtype="<i2").astype(np.float64) / 32768.0
    require(len(values) == n, "PCM payload is shorter than WAV header")
    return values


def independent_score(reference: np.ndarray, search: np.ndarray) -> dict:
    require(reference.ndim == search.ndim == 1 and len(search) > len(reference) > 100, "Invalid PCM shape")
    require(np.isfinite(reference).all() and np.isfinite(search).all(), "Nonfinite PCM")
    sos = butter(4, [100, 3400], btype="bandpass", fs=RATE, output="sos")
    a = sosfiltfilt(sos, reference)
    b = sosfiltfilt(sos, search)
    a -= a.mean()
    n = len(a)
    ones = np.ones(n, dtype=np.float64)
    local_sum = fftconvolve(b, ones, mode="valid")
    local_square = fftconvolve(b*b, ones, mode="valid")
    local_energy = np.maximum(local_square-local_sum*local_sum/n, 0.0)
    a_energy = float(np.dot(a, a))
    denominator = np.sqrt(a_energy*local_energy)
    numerators = fftconvolve(b, a[::-1], mode="valid")
    curve = np.divide(numerators, denominator, out=np.zeros_like(numerators), where=denominator > 1e-12)
    strength = np.abs(curve)
    best = int(np.argmax(strength))
    alternatives = strength.copy()
    exclusion = RATE//4
    alternatives[max(0, best-exclusion):min(len(alternatives), best+exclusion+1)] = -1.0
    second = int(np.argmax(alternatives))

    def direct(index: int) -> float:
        window = b[index:index+n]
        centered = window-window.mean()
        den = float(np.linalg.norm(a)*np.linalg.norm(centered))
        return 0.0 if den <= 1e-12 else float(np.dot(a, centered)/den)

    direct_best, direct_second = direct(best), direct(second)
    close(float(curve[best]), direct_best, "independent winner direct Pearson")
    close(float(curve[second]), direct_second, "independent runner-up direct Pearson")
    clearance = min(best, len(b)-n-best) / RATE
    return {"best_start_sample_in_search": best, "best_start_seconds_in_search": best/RATE,
            "signed_correlation": direct_best, "absolute_correlation": abs(direct_best),
            "runner_up_exclusion_seconds": .25, "runner_up_start_seconds_in_search": second/RATE,
            "runner_up_absolute_correlation": abs(direct_second),
            "peak_margin": abs(direct_best)-abs(direct_second),
            "peak_ratio": abs(direct_best)/max(abs(direct_second), 1e-12),
            "reference_rms_filtered": math.sqrt(a_energy/n),
            "matched_search_rms_filtered_centered": float(np.std(b[best:best+n])),
            "search_boundary_clearance_seconds": clearance,
            "direct_Pearson_at_winner": direct_best, "direct_Pearson_at_runner_up": direct_second}


def anchor_audit(anchor: dict, expected_anchor: float, source: dict, method: dict) -> dict:
    reference = pcm(anchor["reference"], 6.0, source["audio"]["path"], expected_anchor)
    search = pcm(anchor["mp4_search"], 36.0, source["video"]["path"], expected_anchor-15.0)
    score = independent_score(reference, search)
    saved = anchor["score"]
    for key, value in score.items():
        if not key.startswith("direct_"):
            close(value, saved[key], key)
    gates = {"correlation": score["absolute_correlation"] >= method["minimum_correlation"],
             "unique_far_peak": score["peak_ratio"] >= method["minimum_far_peak_ratio"],
             "reference_energy": score["reference_rms_filtered"] >= method["minimum_filtered_rms"],
             "matched_energy": score["matched_search_rms_filtered_centered"] >= method["minimum_filtered_rms"],
             "boundary_clearance": score["search_boundary_clearance_seconds"] >= method["minimum_search_boundary_clearance_seconds"]}
    require(gates == saved["strict_gates"], "Strict anchor gate differs")
    require(all(gates.values()) == saved["strict_anchor_pass"], "Anchor disposition differs")
    matched = expected_anchor-15.0 + score["best_start_seconds_in_search"]
    offset = matched-expected_anchor
    close(matched, saved["matched_mp4_start_seconds"], "matched timestamp", 1e-10)
    close(offset, saved["offset_mp4_minus_original_seconds"], "source offset", 1e-10)
    return {"anchor_seconds": expected_anchor, **score, "strict_gates": gates,
            "strict_anchor_pass": all(gates.values()), "offset_seconds": offset,
            "reference_sha256": anchor["reference"]["sha256"], "search_sha256": anchor["mp4_search"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="Audit a completed immutable prefix; save separate partial report")
    args = parser.parse_args()
    started = time.perf_counter()
    plan_path, progress_path = BASE/"plan.json", BASE/"results/progress.json"
    plan, progress = read(plan_path), read(progress_path)
    plan_sha, progress_sha = sha(plan_path), sha(progress_path)
    require(progress["plan_sha256"] == plan_sha, "Progress belongs to another plan")
    for binding in [plan["inventory"], plan["ffmpeg"], *plan["code"]]:
        check_binding(binding)
    inventory = read(plan["inventory"]["path"])
    for binding in inventory["evidence_files"]:
        check_binding(binding)
    require(plan["method"]["anchors_seconds"] == [30.0,150.0,270.0], "Anchor policy changed")
    for key, expected in {"reference_seconds":6.0,"search_seconds":36.0,"search_radius_seconds":15.0,
                          "rate":8000,"minimum_correlation":.99,"minimum_far_peak_ratio":1.5,
                          "minimum_filtered_rms":1e-4,"minimum_search_boundary_clearance_seconds":.5,
                          "maximum_offset_range_exclusive_seconds":.02}.items():
        close(plan["method"][key], expected, "Frozen method "+key, 1e-12)
    sources = inventory["sources"]
    by_stem = {s["source_stem"]: s for s in sources}
    require(len(sources) == len(by_stem) == 59, "Cohort is not 59 unique recordings")
    require(len({s["speaker_id"] for s in sources}) == 8, "Cohort is not eight speakers")
    expected_order = [s["source_stem"] for phase in ["screen8", "remaining51"]
                      for s in sorted(sources, key=lambda x:x["source_stem"]) if s["phase"] == phase]
    require(expected_order == plan["source_order"], "Frozen source order differs")
    records = {r["id"]: r for s in sources for r in s["training_records"]}
    require(len(records) == sum(len(s["training_records"]) for s in sources) == 512, "Expected 512 unique training clips")
    require(all(r["split"] == "train" and r["source_partition"] == "Training" for r in records.values()), "Non-training clip in cohort")
    a_ids, b_ids = (set(inventory["branch_sample_ids"][b]) for b in ["A_keep256", "B_expand512"])
    require(len(a_ids) == 256 and len(b_ids) == 512 and a_ids < b_ids and set(records) == b_ids, "Branch cohort mismatch")
    for source in sources:
        require(set(r["id"] for r in source["training_records"]) == set(source["branch_sample_ids"]["B_expand512"]), "Source B records mismatch")
        for branch, ids in [("A_keep256", a_ids), ("B_expand512", b_ids)]:
            require(set(source["branch_sample_ids"][branch]) == {r["id"] for r in source["training_records"] if r["id"] in ids}, "Source branch membership mismatch")
    bindings = progress["source_results"]
    require(len(bindings) == progress["completed_sources"], "Progress source count mismatch")
    if not args.partial:
        require(progress["status"] == "completed" and len(bindings) == 59, "Full audit needs all 59 outcomes")
    completed_stems, rows, outcomes, evidence_paths, clip_outcomes = [], [], [], set(), {}
    for binding in bindings:
        check_binding(binding)
        item = read(binding["path"])
        stem = item["source_stem"]
        completed_stems.append(stem)
        source = by_stem[stem]
        require(item["plan_sha256"] == plan_sha and item["speaker_id"] == source["speaker_id"] and item["phase"] == source["phase"], "Source provenance mismatch")
        require(item["branch_sample_ids"] == source["branch_sample_ids"], "Source branch IDs changed")
        require(item["status"] in {"supported", "quarantined"}, "Unfinished source outcome")
        for kind, metadata in item["source_metadata"].items():
            expected = source[kind]
            require(metadata["path"] == expected["path"] and metadata["size_bytes"] == expected["size_bytes"], "Recorded source stat differs from inventory")
            if "mtime_ns" in expected:
                require(metadata["mtime_ns"] == expected["mtime_ns"], "Recorded source mtime differs")
        if "label_metadata" in item:
            labels = item["label_metadata"]
            require(labels["label_sha256"] == source["expected_label_sha256"] and labels["selected_training_records_verified"] == len(source["training_records"]), "Label verification metadata mismatch")
        if "original_wav_header" in item:
            wav = item["original_wav_header"]
            close(wav["duration_seconds"], wav["frame_count"]/wav["sample_rate"], "WAV header duration")
        if "summary" in item:
            meta = item["mp4_metadata"]
            track = next(t for t in meta["tracks"] if t["handler"] == "vide")
            durations = [track["media_clock"]["duration_seconds"], meta["movie_clock"]["duration_seconds"]]
            if track.get("edit_list"):
                require(all(e["media_rate_integer"]==1 and e["media_rate_fraction"]==0 for e in track["edit_list"]), "Nonunit video edit rate")
                durations.append(sum(e["movie_duration_seconds"] for e in track["edit_list"]))
            close(item["summary"]["video_duration_seconds"], min(durations), "Conservative video presentation bound")
            close(item["summary"]["wav_minus_video_duration_seconds"], item["original_wav_header"]["duration_seconds"]-min(durations), "Duration difference")
        checked = []
        for index, anchor in enumerate(item["anchors"]):
            require(index < 3, "Unplanned extra anchor")
            checked.append(anchor_audit(anchor, plan["method"]["anchors_seconds"][index], source, plan["method"]))
            evidence_paths.update(str(local(anchor[k]["path"])) for k in ["reference", "mp4_search"])
        if "summary" in item:
            require(len(checked) == 3, "Summary needs all three anchors")
            offsets = [v["offset_seconds"] for v in checked]
            median = round(statistics.median(offsets)*RATE)/RATE
            spread = max(offsets)-min(offsets)
            strict = all(v["strict_anchor_pass"] for v in checked)
            supported = strict and spread < plan["method"]["maximum_offset_range_exclusive_seconds"]
            summary = item["summary"]
            close(summary["median_offset_seconds"], median, "Source median", 1e-10)
            close(summary["offset_range_seconds"], spread, "Source offset range", 1e-10)
            close(summary["minimum_correlation"], min(v["absolute_correlation"] for v in checked), "Source min correlation")
            close(summary["minimum_far_peak_ratio"], min(v["peak_ratio"] for v in checked), "Source min peak ratio")
            require(summary["all_anchors_strict"] == strict and summary["constant_mapping_supported"] == supported, "Source aggregate gates mismatch")
            require(summary["abs_offset_at_least_100ms"] == (abs(median) >= .1), "Significant-offset flag mismatch")
            require((item["status"] == "supported") == supported, "Source disposition mismatch")
        if item["status"] == "supported":
            require(len(checked) == 3 and "summary" in item, "Supported source missing evidence")
            impacts = {r["id"]:r for r in item["training_frame_selection_impact"]}
            require(set(impacts) == set(source["branch_sample_ids"]["B_expand512"]), "Impact clip coverage mismatch")
            for record in source["training_records"]:
                impact = impacts[record["id"]]
                new_start = record["start_time"] + median
                new_end = record["end_time"] + median
                count = record["frames"]
                old_indices = [int(round((record["start_time"] + k/25)*record["video_fps"])) for k in range(count)]
                new_indices = [int(round((new_start + k/25)*record["video_fps"])) for k in range(count)]
                changed = sum(a != b for a,b in zip(old_indices, new_indices))
                bounded = new_start >= 0 and new_end <= summary["video_duration_seconds"]
                close(impact["video_start_time"], new_start, "Mapped training start")
                close(impact["video_end_time"], new_end, "Mapped training end")
                require(impact["target_frames"] == count and impact["changed_selected_frame_indices"] == changed and impact["fully_within_video_duration"] == bounded, "Frame selection impact differs")
                clip_outcomes[record["id"]] = {"source_stem":stem, "status":"supported", "offset_seconds":median,
                                               "significant_offset":abs(median)>=.1, "target_frames":count,
                                               "changed_selected_frame_indices":changed, "within_video_duration":bounded}
        else:
            require(bool(item.get("reason")), "Quarantine missing reason")
            require("training_frame_selection_impact" not in item, "Quarantined source has applied candidate impact")
            for record in source["training_records"]:
                clip_outcomes[record["id"]] = {"source_stem":stem, "status":"quarantined"}
        rows.append({"source_stem":stem, "speaker_id":source["speaker_id"], "status":item["status"],
                     "source_result_sha256":binding["sha256"], "anchors":checked,
                     "summary":item.get("summary"), "reason":item.get("reason")})
        outcomes.append(item["status"])
        print(json.dumps({"independently_audited_sources":len(rows), "anchors":sum(len(r["anchors"]) for r in rows)}), flush=True)
    require(completed_stems == plan["source_order"][:len(completed_stems)], "Results are not the frozen cohort prefix")
    require(Counter(outcomes)["supported"] == progress["supported_sources"] and Counter(outcomes)["quarantined"] == progress["quarantined_sources"], "Progress aggregate mismatch")
    require(sha(plan_path) == plan_sha, "Plan changed during audit")
    if not args.partial:
        require(sha(progress_path) == progress_sha, "Final progress changed during audit")
    branches = {}
    for branch, ids in [("A_keep256",a_ids), ("B_expand512",b_ids)]:
        selected = [clip_outcomes[i] for i in ids if i in clip_outcomes]
        supported_clips = [v for v in selected if v["status"] == "supported"]
        branches[branch] = {"total_selected_clips":len(ids), "reviewed_clips":len(selected),
                            "supported_mapping_clips":len(supported_clips),
                            "quarantined_clips":sum(v["status"]=="quarantined" for v in selected),
                            "unreviewed_clips":len(ids)-len(selected),
                            "clips_on_at_least_100ms_offset_sources":sum(v["significant_offset"] for v in supported_clips),
                            "clips_with_changed_frame_indices":sum(v["changed_selected_frame_indices"]>0 for v in supported_clips),
                            "changed_selected_frame_indices":sum(v["changed_selected_frame_indices"] for v in supported_clips),
                            "supported_clip_target_frames":sum(v["target_frames"] for v in supported_clips),
                            "mapped_clip_out_of_bounds_ids":sorted(i for i in ids if i in clip_outcomes and clip_outcomes[i]["status"]=="supported" and not clip_outcomes[i]["within_video_duration"])}
    registered_pcm_bytes = sum(Path(path).stat().st_size for path in evidence_paths)
    all_pcm = {str(p.resolve()) for p in (BASE/"results/pcm").rglob("*.wav")}
    result = {"schema":"training-step06-independent-audit-v1", "status":"passed",
              "scope":"completed_prefix_only" if args.partial else "all59_source_outcomes",
              "created_utc":datetime.now(timezone.utc).isoformat(), "auditor_sha256":sha(Path(__file__)),
              "plan_sha256":plan_sha, "progress_snapshot_sha256":progress_sha, "inventory_sha256":plan["inventory"]["sha256"],
              "method":"Independent FFT convolution for correlation and both sliding moments; direct centered Pearson at winner and distant runner-up for every available anchor; producer functions not imported",
              "verified":{"cohort_sources":59, "cohort_speakers":8, "cohort_unique_clips":512,
                          "source_outcomes":len(rows), "supported_sources":outcomes.count("supported"),
                          "quarantined_sources":outcomes.count("quarantined"), "independent_full_search_anchors":sum(len(r["anchors"]) for r in rows),
                          "PCM_hash_format_length_and_command_count":len(evidence_paths), "registered_PCM_bytes":registered_pcm_bytes,
                          "unregistered_PCM_files":sorted(all_pcm-evidence_paths)},
              "branch_impact":branches, "source_audits":rows,
              "scope_guards":{"cloud_source_read":False,"models_loaded":False,"test_payload_read":False,"new_PCM_extraction":False,"original_edits":False},
              "limits":["Audit validates saved excerpts and arithmetic, not full cloud media integrity or a fresh cloud reread.",
                        "Passed audit may include correctly quarantined recordings; only supported source outcomes support a constant candidate clock mapping.",
                        "Audio-clock mapping does not establish lip/audio synchrony, transcript truth, or absence of local edits between three anchors.",
                        "Frame selection impact is index arithmetic at existing25fps sampling; no crop, input cache or model is rebuilt."],
              "seconds":time.perf_counter()-started}
    output = BASE/"results"/("independent_audit_partial.json" if args.partial else "independent_audit.json")
    with output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"path":str(output),"sha256":sha(output),"verified":result["verified"],"branch_impact":branches},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
