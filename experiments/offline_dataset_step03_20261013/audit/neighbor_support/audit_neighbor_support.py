"""Audit saved timestamp metadata only; no TAR, audio, video or NPZ access."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
STAGE = BASE.parents[1]
EXTRACTION = STAGE / "extraction"


def load(path: Path) -> tuple[dict, dict]:
    """Read a JSON artifact and bind its exact bytes."""
    if path.suffix != ".json":
        raise ValueError("Only JSON metadata can be read")
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def f(value: dict) -> Fraction:
    """Read exact rational fields; the floating seconds are display values."""
    assert type(value["numerator"]) is int and type(value["denominator"]) is int
    result = Fraction(value["numerator"], value["denominator"])
    assert abs(float(result) - value["seconds"]) < 1e-9
    return result


def r(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator, "seconds": float(value)}


def overlap(start: Fraction, end: Fraction, left: Fraction, right: Fraction) -> Fraction:
    return max(Fraction(0), min(end, right) - max(start, left))


def main() -> None:
    plan, plan_ref = load(EXTRACTION / "extraction_plan.json")
    summary, summary_ref = load(EXTRACTION / "results/extraction_summary.json")
    assert summary["status"] == "provisional_caches_created_root_QC_pending"
    assert summary["cache_count"] == 30 and len(summary["records"]) == 15
    assert summary["plan_sha256"] == plan_ref["sha256"]
    planned = {record["parent_id"]: record for record in plan["records"]}
    bindings = [plan_ref, summary_ref]
    rows = []
    for parent in summary["records"]:
        entry = planned[parent["parent_id"]]
        history = entry["historical_record"]
        offset = f(entry["frozen_offset"])
        context = history["context_audio"]
        # Context.wav sample zero maps to this exact native 48kHz source sample.
        context_start = Fraction(context["source_first_sample"], context["source_sample_rate"]) + offset
        context_end = context_start + Fraction(context["samples"], context["sample_rate"])
        native_context_end = context_start + Fraction(context["source_samples_read"], context["source_sample_rate"])
        assert context["source_sample_origin_verified"] is True
        for cache_ref in parent["clips"]:
            cache, binding = load(Path(cache_ref["metadata_path"]))
            assert binding["sha256"] == cache_ref["metadata_sha256"]
            bindings.append(binding)
            assert cache["parent_id"] == parent["parent_id"]
            assert cache["plan_sha256"] == plan_ref["sha256"]
            assert cache["training_ready"] is False and cache["provisional"] is True
            assert cache["clock_mapping"]["exact_CFR_table_and_decoded_brackets_agree"] is True
            period = f(cache["clock_mapping"]["native_period"])
            origin = f(cache["clock_mapping"]["recording_origin"])
            start, end = f(cache["semantic_video_start"]), f(cache["semantic_video_end"])
            assert start == context_start + Fraction(cache["start_sample_16k"], 16000)
            assert end == context_start + Fraction(cache["end_sample_16k"], 16000)
            display_start, display_end = f(cache["grid"]["display_start"]), f(cache["grid"]["display_end"])
            assert context_start <= display_start < display_end <= context_end
            traces = cache["trace"]
            assert len(traces) == cache["T"] == cache_ref["T"]
            pts = []
            grid_times = []
            max_decoded_delta = 0.0
            for index, trace in enumerate(traces):
                point, target = f(trace["actual_pts"]), f(trace["grid_time"])
                assert point == origin + trace["native_frame_index"] * period
                assert target == display_start + Fraction(index, 25)
                assert trace["grid_index"] == cache["grid"]["first"] + index
                delta = abs(trace["decoded_pts_seconds"] - float(point))
                assert delta <= 1e-6
                max_decoded_delta = max(max_decoded_delta, delta)
                bracket = [(f(item["actual_pts"]), item["native_frame_index"])
                           for item in trace["selection_bracket"]]
                assert min(bracket)[0] <= target <= max(bracket)[0]
                chosen = min(bracket, key=lambda value: (abs(value[0]-target), value[0]))
                assert chosen == (point, trace["native_frame_index"])
                assert context_start <= point < context_end
                assert point + period <= context_end
                pts.append(point)
                grid_times.append(target)
            assert all(left < right for left, right in zip(pts, pts[1:]))
            assert display_end == display_start + Fraction(len(traces), 25)
            neighbors = []
            expected_display_neighbors = {}
            for neighbor in history["context_neighbor_intersections"]:
                # Retain source decimal label times; no invented sample correction.
                left = Fraction(str(neighbor["start_time"])) + offset
                right = Fraction(str(neighbor["end_time"])) + offset
                intersects_display = overlap(display_start, display_end, left, right)
                sample_indices = [i for i, point in enumerate(pts) if left <= point < right]
                bin_indices = [i for i, target in enumerate(grid_times)
                               if overlap(target, target+Fraction(1, 25), left, right) > 0]
                cadence_indices = [i for i, point in enumerate(pts)
                                   if overlap(point, point+period, left, right) > 0]
                if intersects_display:
                    expected_display_neighbors[str(neighbor["sentence_id"])] = intersects_display
                neighbors.append({"sentence_id": neighbor["sentence_id"],
                    "source_text": neighbor["source_text"],
                    "neighbor_video_start_from_label": r(left), "neighbor_video_end_from_label": r(right),
                    "display_interval_overlap": r(intersects_display),
                    "display_bins_with_positive_label_overlap": bin_indices,
                    "selected_native_PTS_points_inside_label_interval": sample_indices,
                    "selected_native_frame_indices_inside_label_interval": [traces[i]["native_frame_index"] for i in sample_indices],
                    "native_cadence_intervals_overlapping_label": cadence_indices,
                    "cadence_note": "[native PTS, PTS+native period) is a source presentation-cadence interval, not the 25fps display bin, shutter exposure, or proof of phonetic content."})
            actual_display_neighbors = {str(item["sentence_id"]): f(item["overlap"])
                                        for item in cache["neighbor_label_intersections"]}
            assert actual_display_neighbors == expected_display_neighbors
            rows.append({"parent_id": cache["parent_id"], "kind": cache["kind"],
                "clip_id": cache["clip_id"], "split": cache["split"], "T": len(pts),
                "metadata_sha256": binding["sha256"], "semantic_start": r(start), "semantic_end": r(end),
                "display_start": r(display_start), "display_end": r(display_end),
                "context_sample_coverage_start": r(context_start), "context_sample_coverage_end": r(context_end),
                "native_context_source_coverage_end": r(native_context_end),
                "all_display_bins_inside_parent_review_context": True,
                "all_selected_native_PTS_inside_parent_review_context": True,
                "all_native_cadence_intervals_inside_parent_review_context": True,
                "first_selected_native_PTS": r(pts[0]), "last_selected_native_PTS": r(pts[-1]),
                "selected_PTS_before_semantic_start_positions": [i for i, point in enumerate(pts) if point < start],
                "selected_PTS_at_or_after_semantic_end_positions": [i for i, point in enumerate(pts) if point >= end],
                "selected_PTS_before_display_start_positions": [i for i, point in enumerate(pts) if point < display_start],
                "last_native_PTS_to_semantic_end": r(end-pts[-1]),
                "max_native_PTS_distance_from_display_grid_time": r(max(abs(p-t) for p,t in zip(pts,grid_times))),
                "max_saved_decoder_pts_error_seconds": max_decoded_delta,
                "neighbors": neighbors, "saved_display_neighbor_intersections_reproduced": True,
                "new_visual_or_audio_approval": False, "training_ready": False})
    assert len(rows) == 30
    assert sum(row["T"] for row in rows) == summary["total_rgb_frames"]
    display_rows = [row["clip_id"] for row in rows if any(n["display_interval_overlap"]["numerator"] > 0 for n in row["neighbors"])]
    point_rows = [row["clip_id"] for row in rows if any(n["selected_native_PTS_points_inside_label_interval"] for n in row["neighbors"])]
    cadence_rows = [row["clip_id"] for row in rows if any(n["native_cadence_intervals_overlapping_label"] for n in row["neighbors"])]
    result = {"schema": "lipnet-stage3-saved-PTS-neighbor-support-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed_metadata_consistency_audit",
        "scope": "Saved exact clock/trace/neighbor JSON only. No new media, NPZ, audio or TAR reads; not a perceptual or phonetic-boundary approval.",
        "generator": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "inputs": bindings, "parents": 15, "clips": 30, "total_saved_trace_frames": sum(row["T"] for row in rows),
        "display_interval_neighbor_overlap_clips": display_rows,
        "selected_native_PTS_inside_neighbor_label_clips": point_rows,
        "native_cadence_interval_neighbor_overlap_clips": cadence_rows,
        "clips_with_native_PTS_before_display_start": [row["clip_id"] for row in rows if row["selected_PTS_before_display_start_positions"]],
        "clips_with_native_PTS_before_semantic_start": [row["clip_id"] for row in rows if row["selected_PTS_before_semantic_start_positions"]],
        "clips_with_native_PTS_at_or_after_semantic_end": [row["clip_id"] for row in rows if row["selected_PTS_at_or_after_semantic_end_positions"]],
        "all_selected_points_and_display_bins_inside_review_context": True,
        "all_nearest_PTS_brackets_and_earlier_ties_reproduced": True,
        "all_saved_display_neighbor_intersections_reproduced": True,
        "boundary_interpretation": [
            "Display support may overlap a neighboring sentence while no selected source PTS lies in its label interval.",
            "A native PTS is a timestamped image sample. [PTS,PTS+period) describes native presentation cadence, not lip phoneme content or exposure.",
            "No point overlap cannot prove absence of neighboring articulation; label timing and visual articulation can differ.",
            "All examined supports lie within the earlier reviewed context, so its recorded neighbor list covers the intervals tested. This does not validate annotation accuracy.",
            "These checks do not turn qualitative phrase review into exact frame-boundary approval or independently approve new pixels."],
        "records": rows, "new_original_media_reads": 0, "new_NPZ_reads": 0,
        "new_model_calls": 0, "training_ready": False}
    target = BASE / "neighbor_support_audit.json"
    if target.exists():
        raise FileExistsError("Do not overwrite an audit result")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                      "frames": result["total_saved_trace_frames"], "display": display_rows,
                      "points": point_rows, "native_cadence": cadence_rows}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        print(f"Neighbor support audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
