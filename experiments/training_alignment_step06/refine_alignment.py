"""Separately planned precision follow-up using saved local 8kHz PCM only.

The original8kHz audit and quarantine decisions remain immutable. This follow-up
was proposed AFTER a two-source fractional-grid diagnostic; it was not part of
the original prospective plan. It retains every original numerical gate and
searches the entire saved36-second search window on a finer interpolated grid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import wave

import numpy as np
import scipy
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PLAN_PATH = BASE / "refinement_plan.json"
RATE, UP, FINE_RATE = 8000, 6, 48000
ORIGINAL_SIGNAL_GATE_FAILURE = "At least one strict anchor gate or constant-offset range failed"


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def local(path):
    candidate = Path(path)
    require(candidate.drive.lower() in ("", ROOT.drive.lower()), "External-drive read forbidden")
    resolved = candidate.resolve()
    require(resolved.is_relative_to(ROOT), f"Non-workspace read forbidden: {resolved}")
    return resolved


def sha(path):
    with local(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(local(path).read_text(encoding="utf-8"))


def bind(path):
    return {"path": str(local(path)), "sha256": sha(path)}


def save(path, value, *, replace=False):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)


def freeze():
    original = read(BASE / "plan.json")
    inventory = read(original["inventory"]["path"])
    require(sha(original["inventory"]["path"]) == original["inventory"]["sha256"], "Inventory changed")
    require(len(original["source_order"]) == len(set(original["source_order"])) == 59, "Need all59 sources")
    require(set(original["source_order"]) == {s["source_stem"] for s in inventory["sources"]}, "Source cohort changed")
    require(original["method"]["minimum_correlation"] == .99 and
            original["method"]["minimum_far_peak_ratio"] == 1.5 and
            original["method"]["minimum_filtered_rms"] == 1e-4 and
            original["method"]["minimum_search_boundary_clearance_seconds"] == .5 and
            original["method"]["maximum_offset_range_exclusive_seconds"] == .02, "Unexpected original gates")
    probe = read(BASE / "results/fractional_probe.json")
    require(probe["scope"]["posthoc_diagnostic_only"] and probe["scope"]["sources"] == 2,
            "Expected documented two-source posthoc trigger")
    value = {"schema": "training-step06-refinement-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "planning_sequence": "This is a separately frozen follow-up AFTER original8kHz screening and a two-source first-anchor posthoc fractional-delay probe. It is not retroactively claimed as the original plan.",
        "reason": "The probe supported loss of correlation from a125us delay grid: preserving the .99 threshold, investigate a20.833us interpolated delay grid over the entire original search window.",
        "original_plan": bind(BASE / "plan.json"), "inventory": original["inventory"],
        "trigger_probe": bind(BASE / "results/fractional_probe.json"),
        "trigger_probe_code": bind(BASE / "probe_fractional_lag.py"), "code": bind(Path(__file__)),
        "source_order": original["source_order"], "source_count": 59, "anchors_seconds": [30., 150., 270.],
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "method": {"input_rate": RATE, "interpolation_factor": UP, "evaluation_grid_rate": FINE_RATE,
            "filter": "4th-order Butterworth100-3400Hz zero-phase at8kHz on each complete saved excerpt",
            "interpolation": "scipy.signal.resample_poly(x,6,1,window=('kaiser',5.0),padtype='constant') after filtering on both signals",
            "search": "Global valid normalized FFT cross-correlation across the full saved36-second search,6-second reference; every1/48000s shift, not a local peak-only search",
            "score": "absolute centered Pearson; signed value retained; sliding moments from cumulative sums; FFT numerator",
            "reference_seconds": 6., "search_seconds": 36., "search_radius_seconds": 15.,
            "minimum_correlation": .99, "minimum_far_peak_ratio": 1.5,
            "runner_up_exclusion_seconds": .25, "minimum_filtered_rms": 1e-4,
            "minimum_search_boundary_clearance_seconds": .5, "maximum_offset_range_exclusive_seconds": .02,
            "offset_formula": "MP4 search start + best grid start - original WAV reference start",
            "recording_offset": "median rounded to1/48000s grid", "significant_offset_reporting_threshold_seconds": .1,
            "channel_policy": "Use existing8kHz mono PCM exactly; no original WAV or MP4 reads",
            "source_policy": "Every completed source with three complete matching anchors is scored regardless of original signal-gate status. Missing, truncated, wrong-hash or source-integrity error evidence remains quarantined. Pending originals are skipped until a later run; no output overwrite.",
            "numerical_checks": "Direct centered-dot-product Pearson at best and far runner-up verifies FFT Pearson within1e-8"},
        "scope": {"original_results_modified": False, "original_sources_read": False, "test_payload": False,
            "model_loaded": False, "model_outputs_read": False, "training": False, "labels_modified": False,
            "thresholds_lowered": False, "all_sources_included_without_result_selection": True},
        "limits": ["Interpolating existing8kHz samples does not create new48kHz observations.",
            "Original8kHz decisions must be reported alongside this separate follow-up.",
            "Three anchors support an audio time mapping, not transcript correctness, lip synchronization, or every internal edit.",
            "No model-quality improvement is established by this timing-only analysis."]}
    save(PLAN_PATH, value)
    print(json.dumps({"refinement_plan": str(PLAN_PATH), "sha256": sha(PLAN_PATH), "sources": 59}, ensure_ascii=False), flush=True)


def pcm(spec, expected_seconds):
    path = local(spec["path"])
    require(path.is_relative_to((BASE/"results/pcm").resolve()), "Unexpected local PCM location")
    require(sha(path) == spec["sha256"], "PCM hash changed")
    require(spec["sample_rate"] == RATE and spec["requested_seconds"] == expected_seconds,
            "Unexpected saved excerpt metadata")
    with wave.open(str(path), "rb") as handle:
        require((handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) ==
                (RATE, 1, 2, "NONE"), "Unexpected PCM format")
        require(handle.getnframes() == round(expected_seconds*RATE) == spec["frames"], "Truncated PCM")
        value = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64)/32768
    require(len(value) == round(expected_seconds*RATE) and np.isfinite(value).all(), "Invalid PCM values")
    return value


def global_score(reference, search):
    """Global finer-grid score; no coarse-score peak or outcome is consulted."""
    filt = butter(4, [100, 3400], btype="bandpass", fs=RATE, output="sos")
    reference = resample_poly(sosfiltfilt(filt, reference), UP, 1, window=("kaiser", 5.), padtype="constant")
    search = resample_poly(sosfiltfilt(filt, search), UP, 1, window=("kaiser", 5.), padtype="constant")
    reference = reference-reference.mean()
    n = len(reference)
    sums = np.r_[0., np.cumsum(search)]
    squares = np.r_[0., np.cumsum(search*search)]
    local_sums = sums[n:]-sums[:-n]
    centered_energy = np.maximum(squares[n:]-squares[:-n]-local_sums*local_sums/n, 0.)
    ref_energy = float(np.dot(reference, reference))
    denominator = np.sqrt(ref_energy*centered_energy)
    numerator = correlate(search, reference, mode="valid", method="fft")
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    strength = np.abs(scores)
    require(np.isfinite(strength).all(), "Nonfinite correlation")
    best = int(np.argmax(strength))
    alternative = strength.copy()
    exclusion = round(.25*FINE_RATE)
    alternative[max(0, best-exclusion):min(len(alternative), best+exclusion+1)] = -1
    second = int(np.argmax(alternative))
    direct = {}
    for name, index in (("best", best), ("far_runner_up", second)):
        matched = search[index:index+n]
        matched = matched-matched.mean()
        energy = float(np.dot(matched, matched))
        check = float(np.dot(reference, matched)/np.sqrt(ref_energy*energy)) if ref_energy*energy > 1e-24 else 0.
        require(abs(check-float(scores[index])) <= 1e-8, "Direct/FFT Pearson mismatch")
        direct[name] = check
    clearance = min(best, len(search)-n-best)/FINE_RATE
    ref_rms = float(np.sqrt(ref_energy/n))
    matched_rms = float(np.sqrt(centered_energy[best]/n))
    peak_ratio = float(strength[best]/max(strength[second], 1e-12))
    gates = {"correlation": bool(strength[best] >= .99), "unique_far_peak": peak_ratio >= 1.5,
             "reference_energy": ref_rms >= 1e-4, "matched_energy": matched_rms >= 1e-4,
             "boundary_clearance": clearance >= .5}
    return {"evaluation_grid_rate": FINE_RATE, "global_candidate_count": len(scores),
        "best_start_sample_in_search": best, "best_start_seconds_in_search": best/FINE_RATE,
        "signed_correlation": float(scores[best]), "absolute_correlation": float(strength[best]),
        "runner_up_exclusion_seconds": .25, "runner_up_start_sample_in_search": second,
        "runner_up_start_seconds_in_search": second/FINE_RATE,
        "runner_up_absolute_correlation": float(strength[second]),
        "peak_margin": float(strength[best]-strength[second]), "peak_ratio": peak_ratio,
        "reference_rms_filtered_interpolated": ref_rms, "matched_search_rms_filtered_interpolated_centered": matched_rms,
        "search_boundary_clearance_seconds": clearance, "direct_signed_correlations": direct,
        "strict_gates": gates, "strict_anchor_pass": all(gates.values())}


def refine_source(original_path, source, plan):
    started = time.perf_counter()
    original = read(original_path)
    original_binding = bind(original_path)
    item = {"schema": "training-step06-refined-source-v1", "source_stem": source["source_stem"],
        "speaker_id": source["speaker_id"], "phase": source["phase"], "status": "quarantined",
        "original_result": original_binding, "original_status": original.get("status"),
        "original_reason": original.get("reason"), "original_summary": original.get("summary"),
        "refinement_plan_sha256": sha(PLAN_PATH), "anchors": [], "branch_sample_ids": source["branch_sample_ids"]}
    try:
        require(original["plan_sha256"] == plan["original_plan"]["sha256"], "Wrong original plan")
        require(original["source_stem"] == source["source_stem"], "Source identity changed")
        anchors = original.get("anchors", [])
        require(len(anchors) == 3, "Incomplete original anchors remain quarantined")
        for anchor, expected in zip(anchors, plan["anchors_seconds"]):
            reference, search = anchor["reference"], anchor["mp4_search"]
            require(reference["start_seconds"] == expected and search["start_seconds"] == expected-15,
                    "Unexpected anchor clock")
            score = global_score(pcm(reference, 6.), pcm(search, 36.))
            matched = search["start_seconds"] + score["best_start_seconds_in_search"]
            score["matched_mp4_start_seconds"] = matched
            score["offset_mp4_minus_original_seconds"] = matched-reference["start_seconds"]
            item["anchors"].append({"reference": {k: reference[k] for k in ("path", "sha256", "start_seconds", "frames")},
                "mp4_search": {k: search[k] for k in ("path", "sha256", "start_seconds", "frames")}, "score": score})
        offsets = [anchor["score"]["offset_mp4_minus_original_seconds"] for anchor in item["anchors"]]
        offset = round(statistics.median(offsets)*FINE_RATE)/FINE_RATE
        spread = max(offsets)-min(offsets)
        original_integrity = (original.get("summary") is not None and original["status"] in ("supported", "quarantined")
                              and original.get("reason") in (None, ORIGINAL_SIGNAL_GATE_FAILURE))
        strict = original_integrity and all(anchor["score"]["strict_anchor_pass"] for anchor in item["anchors"]) and spread < .02
        item["summary"] = {"median_offset_seconds": offset, "offset_range_seconds": spread,
            "minimum_correlation": min(a["score"]["absolute_correlation"] for a in item["anchors"]),
            "minimum_far_peak_ratio": min(a["score"]["peak_ratio"] for a in item["anchors"]),
            "all_anchors_strict": all(a["score"]["strict_anchor_pass"] for a in item["anchors"]),
            "original_integrity_checks_completed": original_integrity,
            "constant_mapping_supported": bool(strict), "abs_offset_at_least_100ms": abs(offset) >= .1,
            "video_duration_seconds": original.get("summary", {}).get("video_duration_seconds")}
        item["status"] = "supported" if strict else "quarantined"
        if not strict:
            item["reason"] = "Original integrity error retained" if not original_integrity else "Refined strict anchor or offset-range gate failed"
        else:
            duration = original["summary"]["video_duration_seconds"]
            affected = []
            for record in source["training_records"]:
                start, end = record["start_time"]+offset, record["end_time"]+offset
                fps, n = record["video_fps"], record["frames"]
                changed = sum(round((record["start_time"]+k/25)*fps) != round((start+k/25)*fps) for k in range(n))
                affected.append({"id": record["id"], "video_start_time": start, "video_end_time": end,
                    "fully_within_video_duration": start >= 0 and end <= duration, "target_frames": n,
                    "changed_selected_frame_indices": changed})
            item["training_frame_selection_impact"] = affected
        require(sha(original_path) == original_binding["sha256"], "Original result changed during refinement")
    except Exception as exc:
        item["status"] = "quarantined"
        item["reason"] = f"{type(exc).__name__}: {exc}"
        item.pop("training_frame_selection_impact", None)
        if "summary" in item:
            item["summary"]["constant_mapping_supported"] = False
            item["summary"]["original_integrity_checks_completed"] = False
    item["seconds"] = time.perf_counter()-started
    return item


def run_available():
    plan = read(PLAN_PATH)
    for name in ("original_plan", "inventory", "trigger_probe", "trigger_probe_code", "code"):
        require(sha(plan[name]["path"]) == plan[name]["sha256"], f"Frozen {name} changed")
    require(plan["environment"] == {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
            "Refinement numerical runtime changed")
    sources = {s["source_stem"]: s for s in read(plan["inventory"]["path"])["sources"]}
    output = BASE/"results/refined_sources"
    output.mkdir(exist_ok=True)
    produced = 0
    for stem in plan["source_order"]:
        target, original = output/f"{stem}.json", BASE/f"results/sources/{stem}.json"
        if target.exists():
            prior = read(target)
            require(prior["refinement_plan_sha256"] == sha(PLAN_PATH), "Existing refined result has wrong plan")
            require(sha(prior["original_result"]["path"]) == prior["original_result"]["sha256"], "Existing original evidence changed")
            continue
        if not original.exists():
            continue
        try:
            saved = read(original)
        except json.JSONDecodeError:
            continue  # The original audit may still be atomically finishing its JSON write.
        if saved.get("status") not in ("supported", "quarantined"):
            continue
        result = refine_source(original, sources[stem], plan)
        save(target, result)
        produced += 1
        print(json.dumps({"source_stem": stem, "status": result["status"], "original_status": result["original_status"],
            "summary": result.get("summary"), "reason": result.get("reason"), "seconds": result["seconds"]}, ensure_ascii=False), flush=True)
    rows = [read(output/f"{stem}.json") for stem in plan["source_order"] if (output/f"{stem}.json").exists()]
    progress = {"schema": "training-step06-refinement-progress-v1", "status": "completed" if len(rows) == 59 else "partial_waiting_for_originals",
        "refinement_plan_sha256": sha(PLAN_PATH), "completed_sources": len(rows), "newly_produced_sources": produced,
        "supported_sources": sum(row["status"] == "supported" for row in rows),
        "quarantined_sources": sum(row["status"] == "quarantined" for row in rows),
        "source_results": [bind(output/f"{row['source_stem']}.json") for row in rows],
        "pending_source_stems": [stem for stem in plan["source_order"] if not (output/f"{stem}.json").exists()]}
    save(BASE/"results/refinement_progress.json", progress, replace=True)
    print(json.dumps({key: value for key, value in progress.items() if key not in ("source_results", "pending_source_stems")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run-available", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run_available()
