"""Bounded Q01/Q02 original-grid extraction for the explicitly requested pilot.

Old review evidence remains unchanged. Qualitative no-clipping judgments are not
exact articulation certificates. The ambiguous six items are never extracted.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
EXP = ROOT / "experiments"
OLD = EXP / "offline_dataset_step03_20261013/extraction"
F = EXP / "offline_phrase_review_step05f_20261018"
G = EXP / "offline_phrase_boundary_step05g_20261018"
E = EXP / "offline_phrase_design_step05e_20261018"
sys.path.insert(0, str(OLD))
import prepare_and_extract as extract
from exact_clock import fraction, grid_window, nearest, rational, require


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), "Unsafe/existing output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def text_key(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFC", text) if c.isalnum())


def prepare():
    summary_path = G / "results/stage5g_summary_r8.json"
    role_path = E / "design/role_proposal.json"
    protect_path = E / "data/protected_scope.json"
    source_path = F / "media_prepare/source_plan.json"
    parent_path = F / "effective_parent_inputs.json"
    old_handoff_path = EXP / "offline_dataset_step03_20261013/results/cache_handoff.json"
    summary, roles, protection, sources, parents, old_hand = map(read, [summary_path, role_path, protect_path, source_path, parent_path, old_handoff_path])
    paths = [Path(__file__), summary_path, role_path, protect_path, source_path, parent_path, old_handoff_path,
             OLD / "prepare_and_extract.py", OLD / "exact_clock.py", extract.TOKENIZER, extract.TAR_HELPER, extract.ROI_HELPER]
    codec = extract.load_module(extract.TOKENIZER, "stage5h_tokenizer")
    tar = extract.load_module(extract.TAR_HELPER, "stage5h_tar_plan")
    require(codec.NUM_CLASSES == 69, "Unexpected vocabulary")
    selected = [r for r in summary["records"] if r["review_id"] in ("Q01", "Q02")]
    require([r["review_id"] for r in selected] == ["Q01", "Q02"], "Scope mismatch")
    entries, leakage = [], []
    for rank, review in enumerate(selected, 1):
        qid, pid = review["review_id"], review["parent_id"]
        source = next(r for r in sources["records"] if r["review_id"] == qid)
        queue = source["queue_record"]
        parent = next(r for r in parents["records"] if r["review_id"] == qid)
        old_path = Path(parent["media_record"]["path"])
        require(bind(old_path) == parent["media_record"], "Parent source record changed")
        old = read(old_path)
        require(review["human_phrase_audio_boundary_approved"] and review["qualitative_visual_no_obvious_clipping"], "Review insufficient")
        require(parent["parent_transcript_confirmed"] and parent["parent_audio_qualitatively_confirmed"], "Parent is not confirmed")
        require(review["candidate_id"] in roles["proposed_train_candidate_ids"] and queue["proposed_role"] == "proposed_train", "Not proposed train")
        require(review["text"] == queue["text"] == parent["phrase_text"], "Phrase differs")
        for key in ("audio_confirmation", "visual_answer_evidence", "selected_audio"):
            b = review[key]
            require(bind(b["path"]) == b, "Review evidence changed")
            paths.append(Path(b["path"]))
        visual = read(review["visual_answer_evidence"]["path"])
        require(visual["record"]["review_id"] == qid and visual["record"]["candidate_id"] == review["candidate_id"] and visual["record"]["text"] == review["text"], "Visual response identity changed")
        for video_role in ("phrase_video", "context_video"):
            if video_role in visual["record"]:
                b = visual["record"][video_role]
                require(bind(b["path"]) == b, "Viewed preview changed")
                paths.append(Path(b["path"]))
        require(pid not in protection["protected_parent_ids"] + protection["held_short_alignment_parent_ids"] + protection["pending_listening_ids"], "Protected parent")
        require(queue["recording_key"] not in protection["protected_recording_keys"] and old["source_stem"] not in protection["blocked_source_stem"], "Protected recording")
        for role in ("video", "label"):
            require(old["source_members"][role]["sha256"] not in protection["blocked_" + role + "_sha256"], "Protected source bytes")
        require(text_key(review["text"]) not in protection["protected_candidate_text_keys"], "Protected phrase text")
        require(all(text_key(t) not in protection["protected_full_text_keys"] for t in (queue["parent_text"], queue["parent_source_text"], queue["parent_historical_text"])), "Protected parent transcript")
        for reserved in roles["reserved_recordings"]:
            require(queue["recording_key"] != reserved["recording_key"] and old["source_stem"] not in reserved["blocked_source_stems"], "Reserved probe recording")
            require(all(old["source_members"][role]["sha256"] not in reserved["blocked_" + role + "_sha256"] for role in ("video", "label")), "Reserved source bytes")
        train_overlaps = [r["clip_id"] for r in old_hand["records"] if r["kind"] == "phrase" and r["split"] == "train" and r["recording_group"] == old["source_stem"]]
        require(not any(r["parent_id"] == pid for r in old_hand["records"]), "Selected new parent already exists")
        devs = [r for r in old_hand["records"] if r["kind"] == "phrase" and r["split"] == "dev"]
        require(all(r["recording_group"] != old["source_stem"] and text_key(r["text"]) != text_key(review["text"]) for r in devs), "Current dev collision")
        leakage.append({"review_id": qid, "protected_collision": False, "reserved_probe_collision": False, "current_dev_collision": False,
                        "existing_train_same_recording_clip_ids": train_overlaps, "same_train_recording_is_not_independent_recording_gain": bool(train_overlaps)})
        if qid == "Q01":
            media_path = G / "revisions/Q01_end_parent/record.json"
            audio = read(media_path)["media"]["phrase"]
        else:
            media_path = F / "phrase_media_r2/Q02/record.json"
            audio = read(media_path)["phrase_audio"]
        require(audio["sha256"] == review["selected_audio"]["sha256"], "Chosen audio differs")
        a, b = audio["source_original_wav_first_sample_48k"], audio["source_original_wav_end_sample_48k"]
        require(parent["original_recording_first_sample_48k"] <= a < b <= parent["original_recording_end_sample_48k"], "Outside confirmed parent")
        offset = Fraction(old["offset_native_samples"], 48000)
        require(old["offset_native_samples"] == parent["video_offset_native_samples"], "Offset differs")
        start, end = Fraction(a, 48000) + offset, Fraction(b, 48000) + offset
        require((b-a) == audio["samples"]*3, "PCM sample length differs")
        grid = grid_window(start, end, fraction(old["video"]["recording_origin"]), fraction(old["video"]["recording_origin"])+old["video"]["native_frame_count"]*fraction(old["video"]["native_period"]))
        labels = codec.encode_text(review["text"])
        require(codec.decode_labels(labels) == review["text"] and len(grid["times"]) >= codec.ctc_min_input_len(labels), "CTC/normalization failed")
        trace_map = {r["display_grid_index"]: r for r in old["video"]["trace"]}
        refs = [{"role": "same_original_grid_review_trace", "preview_frame_index": i-min(trace_map), "trace": trace_map[i]} for i in range(grid["first"], grid["stop"])]
        clip = {"clip_id": pid+"__phrase", "kind": "phrase", "text": review["text"], "label": labels, "label_length": len(labels),
                "ctc_min_frames": codec.ctc_min_input_len(labels), "start_sample_16k": audio["source_context_first_sample_16k"],
                "end_sample_16k": audio["source_context_end_sample_16k"], "semantic_video_start": rational(start), "semantic_video_end": rational(end),
                "split": "train", "provisional": True, "training_ready": False, "new_original_pixels_human_approved": False,
                "precise_articulatory_boundary_approved": False, "bounded_experiment_eligible_after_technical_QC": True,
                "audio_is_annotation_only": True, "review_id": qid}
        members = {role: old["source_members"][role] for role in ("video", "label")}
        for role, member in members.items():
            require(Path(member["archive_path"]).drive.upper() == "D:", "Only existing D archives")
            tar._validate_member(old["source_stem"], role, member)
        entries.append({"rank": rank, "parent_id": pid, "speaker_id": old["speaker_id"], "recording_group": old["source_stem"],
                        "source_members": members, "historical_record": old, "review_provenance": review, "parent_audio_review_provenance": parent,
                        "context_origin": rational(Fraction(old["context_audio"]["source_first_sample"], 48000)), "frozen_offset": rational(offset),
                        "clips": [clip], "review_references": refs, "expected_T": len(grid["times"])})
        paths.extend([old_path, media_path])
    old_plan = read(OLD / "extraction_plan.json")
    paths.append(OLD / "extraction_plan.json")
    peak = max(sum(m["size"] for m in e["source_members"].values()) for e in entries)
    require(shutil.disk_usage(BASE).free >= peak + extract.RESERVE, "Insufficient local reserve")
    plan = {"schema": "offline-stage5h-confirmed-two-original-cache-plan-v1", "client_date": "2026-10-19",
            "user_authorization": "검수는 필요한 만큼만 하고 애매한 자료는 보류한 뒤, 확인된 자료로 학습과 평가를 진행해보자",
            "scope": "Only Q01/Q02 as bounded provisional train candidates; Q03-Q08 held; existing evidence is not upgraded.",
            "records": entries, "leakage_checks": leakage, "source_hashes": {str(p.resolve()): sha(p) for p in paths},
            "normalization": old_plan["normalization"], "tokenizer": old_plan["tokenizer"],
            "planned_member_count": 4, "planned_payload_bytes": sum(m["size"] for e in entries for m in e["source_members"].values()),
            "maximum_member_temp_bytes": peak, "expected_total_frames": sum(e["expected_T"] for e in entries),
            "audio_members": 0, "model_calls": 0, "training_updates": 0, "global_training_ready": False,
            "new_original_pixel_approval": False, "qualitative_review_only": True}
    save(BASE / "extraction_plan.json", plan)
    return {"expected_total_frames": plan["expected_total_frames"], "payload_bytes": plan["planned_payload_bytes"], "leakage": leakage}


def execute():
    plan = read(BASE / "extraction_plan.json")
    for path, digest in plan["source_hashes"].items():
        require(sha(path) == digest, "Bound source changed: " + path)
    extract.BASE, extract.PLAN = BASE, BASE / "extraction_plan.json"
    io = {"member_read_attempts": 0, "payload_bytes_read": 0, "members_sha_verified": 0}
    tar = extract.load_module(extract.TAR_HELPER, "stage5h_tar_runtime")
    cropper = extract.load_module(extract.ROI_HELPER, "stage5h_roi_runtime")
    records, results = [], []
    started = time.perf_counter()
    try:
        for entry in plan["records"]:
            print("START", entry["parent_id"], flush=True)
            with extract.materialize(entry, tar, io, BASE / "io_events.jsonl") as paths:
                result = extract.decode_record(paths, entry, plan, cropper)
            require(len(result["clips"]) == 1 and result["clips"][0]["T"] == entry["expected_T"], "Unexpected cache count")
            row = result["clips"][0]
            metadata = read(row["metadata_path"])
            require(not metadata["neighbor_label_intersections"], "Neighbor label display overlap requires further review")
            require(len(result["historical_raw_comparisons"]) == row["T"], "Incomplete original source pixel comparison")
            row.update({"parent_id": entry["parent_id"], "rank": entry["rank"], "speaker_id": entry["speaker_id"], "recording_group": entry["recording_group"],
                        "text": metadata["text"], "label_length": metadata["label_length"], "ctc_min_frames": metadata["ctc_min_frames"],
                        "cache_integrity_audit_passed": True, "technical_sample_QC_passed": True, "exact_visual_boundary_certified": False,
                        "technical_QC_scope": "All selected original pixels/ROI/bbox/PTS match prior review-source hashes; human judgment remains qualitative.",
                        "bounded_experiment_eligible": True, "display_neighbor_intersections": [], "boundary_followup_flag": False,
                        "review_id": entry["clips"][0]["review_id"]})
            records.append(row)
            results.append(result)
            print("READY", row["clip_id"], row["T"], flush=True)
        require(io["payload_bytes_read"] == plan["planned_payload_bytes"] and io["members_sha_verified"] == 4, "I/O mismatch")
        require(not list((BASE / "_tmp").iterdir()), "Temporary originals remain")
        for path, digest in plan["source_hashes"].items():
            require(sha(path) == digest, "Bound evidence changed during extraction")
        result = {"schema": "offline-stage5h-confirmed-cache-handoff-v1", "status": "two_original_grid_caches_ready_for_bounded_pilot",
                  "plan": bind(BASE / "extraction_plan.json"), "records": records, "extraction_results": results, "io": io,
                  "maximum_simultaneous_temporary_bytes": plan["maximum_member_temp_bytes"], "temporary_originals_remaining": 0,
                  "cache_bytes": sum(r["npz_bytes"] for r in records), "total_frames": sum(r["T"] for r in records),
                  "elapsed_seconds": time.perf_counter()-started, "audio_source_reads": 0, "downloads": 0, "model_calls": 0,
                  "training_updates": 0, "global_training_ready": False, "new_human_approvals": 0,
                  "held_review_ids": ["Q03", "Q04", "Q05", "Q06", "Q07", "Q08"]}
        save(BASE / "cache_handoff.json", result)
        return {k: result[k] for k in ("status", "io", "total_frames", "cache_bytes", "elapsed_seconds")}
    except Exception as exc:
        save(BASE / ("failure_"+str(time.time_ns())+".json"), {"error": str(exc), "traceback": traceback.format_exc(), "io": io, "completed_records": records})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute() if args.execute else prepare(), ensure_ascii=False, indent=2))
