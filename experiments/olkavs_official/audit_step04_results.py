"""Independent aligned step04 audit: local derived files only, no model forward.

Run in .venv-olkavs without -O. Audits 100 prepared inputs and 200 outputs.
Original cloud media, model weights and test payloads are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

from summarize_legacy_step03 import independent_counts

BASE = Path(__file__).resolve().parent
ARMS = ["annotation_bbox", "rectangle_control", "square", "context_square", "stable_context_square"]
METHODS = ["corrected_hybrid", "ctc_greedy"]


def local(path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError(f"Audit accepts experiment-local derived paths only: {path}")
    return path


def sha(path) -> str:
    return hashlib.sha256(local(path).read_bytes()).hexdigest()


def read_json(path) -> dict:
    return json.loads(local(path).read_text(encoding="utf8"))


def anchor(item: dict) -> dict:
    assert sha(item["path"]) == item["sha256"]
    return read_json(item["path"])


def audit_inputs(manifest_path: Path, plan_path: Path) -> tuple[dict, dict, dict, dict]:
    manifest, plan = read_json(manifest_path), read_json(plan_path)
    sources = read_json(BASE / "step03_source_plan.json")
    old_plan = read_json(BASE / "step03_eval_plan.json")
    previous = anchor(plan["parent_manifest"])
    anchor(plan["parent_results"])
    assert sha(BASE / "step03_eval_plan.json") == plan["parent_plan_sha256"]
    assert sha(BASE / "step03_source_plan.json") == plan["source_plan_sha256"] == manifest["source_plan_sha256"]
    assert sha(plan_path) == manifest["evaluation_plan"]["sha256"]
    assert manifest["status"] == "prepared" and manifest["record_count"] == 100
    assert plan["arms"] == manifest["arms"] == ARMS and plan["methods"] == METHODS
    assert plan["cohort_size"] == 20 and plan["source_ids"] == old_plan["source_ids"]
    for key in ("source_plan_sha256", "reference_profile_sha256", "corrected_search_sha256",
                "checkpoint_sha256", "common_input", "hybrid", "methods"):
        assert plan[key] == old_plan[key]
    assert sha(BASE / "reference_profile_96_v1.json") == plan["reference_profile_sha256"] == manifest["reference_profile_sha256"]
    assert sha(BASE / "corrected_search.py") == plan["corrected_search_sha256"]
    assert sha(BASE / "prepare_step04_aligned_inputs.py") == manifest["preparation_script_sha256"]
    assert sha(BASE / "roi_geometry_step04.py") == plan["geometry_script_sha256"] == manifest["geometry_script_sha256"]
    assert sha(BASE / "automatic_roi.py") == plan["parent_detector_script_sha256"] == manifest["automatic_roi_script_sha256"]
    assert sha(BASE / "requirements-lock-roi.txt") == manifest["roi_environment_requirements_sha256"]
    assert manifest["alignment_sidecar"] == plan["alignment_sidecar"]
    alignment = anchor(plan["alignment_sidecar"])
    assert alignment["status"] == "validated" and len(alignment["sources"]) == 4
    for evidence in alignment["evidence_files"]:
        assert sha(evidence["path"]) == evidence["sha256"]
    offsets = {r["source_name"]: r["label_to_video_offset_seconds"] for r in alignment["sources"]}
    assert len(offsets) == 4 and all(math.isfinite(x) for x in offsets.values())
    assert not any(manifest[k] for k in ("training_run", "model_run", "audio_used", "audio_read", "test_payloads_read"))
    refs = {r["id"]: r for r in sources["records"]}
    inputs = {(r["id"], r["arm"]): r for r in manifest["records"]}
    assert len(refs) == 20 and set(refs) == set(plan["source_ids"])
    assert len(inputs) == len(manifest["records"]) == 100
    assert set(inputs) == {(sid, arm) for sid in refs for arm in ARMS}
    frames, statuses, detected = Counter(), Counter(), Counter()
    max_timestamp_error = 0.0
    trace_count = geometry_count = 0
    for sid in plan["source_ids"]:
        rows = [inputs[sid, arm] for arm in ARMS]
        assert len({r["source_selected_frames_sha256"] for r in rows}) == 1
        source_name = sid.rsplit("_", 1)[0] + ".mp4"
        source = previous["sources"][source_name]
        fps, offset = source["native_fps"], offsets[source_name]
        width, height = source["width"], source["height"]
        first = math.ceil((refs[sid]["start_time"] + offset) * fps - 1e-9)
        stop = math.ceil((refs[sid]["end_time"] + offset) * fps - 1e-9)
        assert 0 <= first < stop <= source["reported_frames"]
        read_trace = None
        for row in rows:
            arm = row["arm"]
            assert row["split"] == refs[sid]["split"] == "val"
            assert row["text"] == refs[sid]["text"] and row["speaker_id"] == refs[sid]["speaker_id"]
            assert row["source_name"] == source_name and row["native_fps"] == fps
            assert row["label_to_video_offset_seconds"] == offset
            assert row["aligned_video_start_time"] == refs[sid]["start_time"] + offset
            assert row["aligned_video_end_time"] == refs[sid]["end_time"] + offset
            assert row["temporal_method"] == "native-cfr-audio-aligned-v1"
            assert (row["start_frame"], row["stop_frame_exclusive"]) == (first, stop)
            assert row["expected_frames"] == stop - first
            assert row["source_label_bbox_used"] == (arm == "annotation_bbox")
            trace = anchor(row["trace"])
            assert trace["id"] == sid and trace["arm"] == arm
            assert trace["annotation_boxes_used"] == (arm == "annotation_bbox")
            assert trace["alignment_sidecar"] == plan["alignment_sidecar"]
            if read_trace is None:
                read_trace = trace["source_read"]
                assert len(read_trace) == stop - first
                for frame_index, entry in enumerate(read_trace, first):
                    assert entry["source_frame_index"] == entry["annotation_box_index"] == frame_index
                    assert entry["decoder_position_after_read"] == frame_index + 1
                    error = abs(entry["decoder_timestamp_seconds"] - frame_index / fps)
                    assert error < 1e-5
                    max_timestamp_error = max(max_timestamp_error, error)
                    detected[entry["automatic_frame_record"]["status"]] += 1
            else:
                assert trace["source_read"] == read_trace
            trace_count += 1
            prior_state = None
            for i, geometry in enumerate(trace["geometry"]):
                assert geometry["source_frame_index"] == first + i
                if arm != "annotation_bbox":
                    assert geometry["raw_bbox_xyxy"] == read_trace[i]["automatic_xyxy"]
                box = geometry["raw_bbox_xyxy"]
                if arm in ARMS[2:]:
                    x1, y1, x2, y2 = box
                    factor = 1.0 if arm == "square" else 1.5
                    state = np.array([(x1+x2)/2, (y1+y2)/2, max(x2-x1, y2-y1)*factor])
                    np.testing.assert_allclose(geometry["pre_smoothing_center_xy"], state[:2], atol=1e-12, rtol=0)
                    assert geometry["pre_smoothing_side"] == state[2]
                    if arm == "stable_context_square" and prior_state is not None:
                        state = 0.4 * state + 0.6 * prior_state
                    prior_state = state.copy()
                    np.testing.assert_allclose(geometry["post_smoothing_center_xy"], state[:2], atol=1e-12, rtol=0)
                    assert geometry["post_smoothing_side"] == state[2]
                    n = math.ceil(state[2]); left = math.floor(state[0]-n/2); top = math.floor(state[1]-n/2)
                    requested = [left, top, left+n, top+n]
                    left = min(max(left, 0), width-n); top = min(max(top, 0), height-n)
                    final = [left, top, left+n, top+n]
                    assert n <= min(width, height)
                    assert geometry["requested_square_xyxy"] == requested
                    assert geometry["final_square_xyxy"] == final
                    assert geometry["boundary_shifted"] == (requested != final)
                    geometry_count += 1
                else:
                    assert geometry["final_rectangle_xyxy"] == box
            assert sha(row["path"]) == row["sha256"]
            statuses[row["status"]] += 1
            if row["status"] != "ok":
                assert not row["valid_for_inference"] and row["partial_artifact_not_used_for_inference"]
                continue
            assert row["valid_for_inference"] and row["frames"] == stop-first == len(trace["geometry"])
            capture, parts = cv2.VideoCapture(str(local(row["path"]))), []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    parts.append(frame)
            finally:
                capture.release()
            pixels = np.stack(parts)
            assert list(pixels.shape) == row["shape"] == [stop-first, 96, 96, 3]
            assert pixels.dtype == np.uint8
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["decoded_bgr_stream_sha256"]
            assert abs(float(pixels.mean()) - row["mean_before_normalization"]) < 1e-8
            assert abs(float(pixels.std(ddof=1)) - row["std_before_normalization_ddof1"]) < 1e-8
            frames[arm] += len(pixels)
    summary = {"status": "passed", "manifest_sha256": sha(manifest_path),
               "records": len(inputs), "utterances": len(refs), "source_files": len(offsets),
               "decoded_frames_by_arm": dict(frames), "input_statuses": dict(statuses),
               "automatic_statuses": dict(detected), "trace_count": trace_count,
               "square_geometry_frames_independently_recalculated": geometry_count,
               "max_abs_decoder_minus_cfr_seconds": max_timestamp_error,
               "all_encoded_and_successful_decoded_hashes_verified": True,
               "all_corrected_bounds_recomputed": True, "all_trace_positions_checked": True,
               "original_cloud_sources_not_reread": True, "opencv_version": cv2.__version__}
    return summary, manifest, plan, refs


def independent_aggregate(rows: list[tuple[dict, dict]]) -> dict:
    values = [independent_counts(r["reference"], out["prediction"]) for r, out in rows]
    keys = ("character_edits", "reference_characters", "word_edits", "reference_words",
            "normalized_character_edits", "normalized_reference_characters")
    result = {k: sum(v[k] for v in values) for k in keys}
    result.update(clips=len(rows), failed_inputs=sum(out["status"] != "ok" for _, out in rows),
                  exact_sentences=sum(v["exact"] for v in values),
                  normalized_exact_sentences=sum(v["normalized_exact"] for v in values))
    result.update(cer_percent=100*result["character_edits"]/result["reference_characters"],
                  wer_percent=100*result["word_edits"]/result["reference_words"],
                  normalized_cer_percent=100*result["normalized_character_edits"]/result["normalized_reference_characters"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=BASE / "results/step04.json")
    parser.add_argument("--manifest", type=Path, default=BASE / "inputs_step04_aligned/manifest.json")
    parser.add_argument("--plan", type=Path, default=BASE / "step04_aligned_eval_plan.json")
    parser.add_argument("--output", type=Path, default=BASE / "results/step04_metrics_audit.json")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Do not disable audit assertions with Python -O")
    if local(args.output).exists():
        raise FileExistsError("Preserve previous audits; choose another --output")
    result = read_json(args.result)
    assert result["status"] == "completed", "Wait for completed inference before audit"
    input_audit, manifest, plan, refs = audit_inputs(args.manifest, args.plan)
    assert result["plan"] == plan and result["plan_sha256"] == sha(args.plan)
    assert result["input_manifest_sha256"] == sha(args.manifest)
    assert result["source_plan_sha256"] == plan["source_plan_sha256"]
    assert result["alignment_sidecar"] == plan["alignment_sidecar"]
    for key, filename in (("runner_sha256", "evaluate_step04.py"), ("metrics_code_sha256", "evaluation_metrics.py"),
                          ("read_tensor_code_sha256", "diagnose_step02.py"), ("geometry_script_sha256", "roi_geometry_step04.py")):
        assert result[key] == sha(BASE / filename)
    assert not result["audio_used"] and not result["training_run"] and not result["test50_used"]
    sys.path.insert(0, str(BASE / "upstream"))
    from avsr.utils.korean_funcs import grp2char
    with (BASE / "upstream/vocabulary/kor_grapheme.csv").open(encoding="utf-8-sig", newline="") as stream:
        vocab = {int(r["id"]): r["char"] for r in csv.DictReader(stream)}
    assert len(vocab) == 72
    inputs = {(r["id"], r["arm"]): r for r in manifest["records"]}
    records = result["records"]
    assert len(records) == len({(r["id"], r["arm"]) for r in records}) == 100
    assert [(r["id"], r["arm"]) for r in records] == [(sid, arm) for arm in ARMS for sid in plan["source_ids"]]
    feature_count = metric_count = string_count = ctc_count = 0
    search_depth, search_caps, detailed = [], [], []
    output_status = Counter()
    for record in records:
        inp = inputs[record["id"], record["arm"]]
        ref = refs[record["id"]]
        assert record["reference"] == inp["text"] == ref["text"]
        assert record["frames"] == inp["frames"] and record["speaker_id"] == ref["speaker_id"]
        assert record["input_status"] == inp["status"]
        if record["input_status"] == "ok" and "input_sha256" in record:
            assert record["input_sha256"] == inp["sha256"]
        if "feature_file" in record:
            assert sha(record["feature_file"]) == record["feature_file_sha256"]
            with np.load(local(record["feature_file"]), allow_pickle=False) as saved:
                features = saved["features"]
                assert list(features.shape) == record["feature_shape"][1:] == [inp["frames"], 256]
                assert features.dtype == np.float32 and np.isfinite(features).all()
                if "ctc_logits" in saved:
                    logits = saved["ctc_logits"]
                    assert logits.shape == (inp["frames"], 72) and logits.dtype == np.float32 and np.isfinite(logits).all()
                    path = logits.argmax(-1).tolist()
                    assert path == record["ctc_frame_path"]
                    assert record["ctc_stats"]["token_counts"] == {str(k): v for k, v in Counter(path).items()}
                    assert record["ctc_stats"]["blank_fraction"] == path.count(3)/len(path)
                    assert record["ctc_stats"]["first_eos_frame"] == (path.index(2) if 2 in path else None)
                    ctc = [token for token, _ in groupby(path) if token != 3]
                    if record["outputs"]["ctc_greedy"]["status"] == "ok":
                        assert ctc == record["outputs"]["ctc_greedy"]["token_ids"]
                        ctc_count += 1
            feature_count += 1
        for method in METHODS:
            out = record["outputs"][method]
            metric_count += 1; output_status[out["status"]] += 1
            computed = independent_counts(record["reference"], out["prediction"])
            assert computed == out["metrics"], (record["id"], record["arm"], method)
            if out["status"] != "ok":
                assert out["prediction"] == ""
            else:
                tokens = out["token_ids"]
                assert all(type(t) == int and 0 <= t < 72 for t in tokens)
                prefix = tokens[:tokens.index(2)] if 2 in tokens else tokens
                graphemes = "".join(vocab[t] for t in prefix if t != 3)
                assert graphemes == out["graphemes"] and grp2char(graphemes).strip() == out["prediction"]
                assert out["eos_index"] == (tokens.index(2) if 2 in tokens else None)
                string_count += 1
                if method == "corrected_hybrid":
                    assert tokens[-1] == 2 and tokens.count(2) == 1 and not set(tokens) & {0, 1}
                    assert len(tokens) <= out["max_search_length"] <= 199
                    assert out["search_cap_reached"] == (out["max_search_length"] >= 199)
                    search_depth.append(out["max_search_length"])
                    if out["search_cap_reached"]:
                        search_caps.append({"id": record["id"], "arm": record["arm"]})
            detailed.append({"id": record["id"], "arm": record["arm"], "method": method,
                             "status": out["status"], "counts": computed})
    aggregates = {}
    for arm in ARMS:
        selected = [r for r in records if r["arm"] == arm]
        for method in METHODS:
            key = f"{arm}/{method}"
            rows = [(r, r["outputs"][method]) for r in selected]
            combined = independent_aggregate(rows)
            for k, v in combined.items():
                assert result["aggregate"][key][k] == v, (key, k)
            assert (combined["clips"], combined["reference_characters"], combined["reference_words"]) == (20, 667, 189)
            combined["per_speaker"] = {}
            for speaker in sorted({r["speaker_id"] for r in selected}):
                speaker_result = independent_aggregate([(r, o) for r, o in rows if r["speaker_id"] == speaker])
                assert speaker_result == result["aggregate"][key]["per_speaker"][speaker]
                combined["per_speaker"][speaker] = speaker_result
            times = [r["feature_timing"]["seconds"] + o["decoder_timing"]["seconds"] for r, o in rows if o["status"] == "ok"]
            timings = {"n": len(times), "min_seconds": min(times), "median_seconds": statistics.median(times),
                       "max_seconds": max(times), "mean_seconds": statistics.mean(times)} if times else None
            assert timings == result["aggregate"][key]["model_timing"]
            combined["model_timing"] = timings
            aggregates[key] = combined
    assert metric_count == 200
    paired = [{"id": sid, "hybrid_character_edits": {r["arm"]: r["outputs"]["corrected_hybrid"]["metrics"]["character_edits"]
              for r in records if r["id"] == sid}} for sid in plan["source_ids"]]
    audit = {"schema": "olkavs-step04-independent-metrics-audit-v1", "status": "passed",
             "created_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(__file__),
             "result_path": str(local(args.result)), "result_sha256": sha(args.result),
             "plan_sha256": sha(args.plan), "input_audit": input_audit,
             "verified": {"plan_code_alignment_hash_chain": True, "fixed20_ids_five_arms": True,
                          "output_metric_records": metric_count, "independent_full_matrix_levenshtein": True,
                          "per_speaker_and_pooled_aggregates": True, "timing_statistics": True,
                          "feature_files_hash_finite_shape": feature_count,
                          "ctc_logits_argmax_collapse_reproduced": ctc_count,
                          "token_dictionary_string_reconstruction": string_count,
                          "model_not_rerun": True, "cloud_sources_not_reread": True,
                          "checkpoint_and_test50_payload_not_read": True},
             "output_status_counts": dict(output_status),
             "search": {"minimum_max_search_length": min(search_depth) if search_depth else None,
                        "maximum_max_search_length": max(search_depth) if search_depth else None,
                        "cap_reached": search_caps},
             "aggregate": aggregates, "paired_hybrid_character_edits": paired,
             "per_output_independent_counts": detailed,
             "limits": ["Development20, not untouched final test; public pretraining overlap unknown.",
                        "Waveform offsets map WAV to MP4 audio, not independent transcript or lip/audio synchrony certification.",
                        "Source frame hashes and extraction traces checked; original cloud pixels not independently reread.",
                        "Geometry arithmetic checked from saved traces; detector and native-pixel crop extraction not rerun.",
                        "Hangul composition uses frozen official text helper; edit distance and CTC collapse independently recomputed.",
                        "Saved model tensors checked without rerunning model forward; timing is not live end-to-end latency."]}
    local(args.output).write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"status": "passed", "output": str(local(args.output)), "inputs": 100,
                      "outputs": metric_count, "frames_by_arm": input_audit["decoded_frames_by_arm"],
                      "search": audit["search"], "metrics": {k: {x: v[x] for x in ("cer_percent", "wer_percent", "exact_sentences", "failed_inputs")}
                      for k, v in aggregates.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
