"""Cross-check completed decoder records against local archive-structure results.

Does not open D:/G:, run FFmpeg, load models, or modify the live decoder script.
Final report requires every planned decode result to exist. WAV final output time
must agree with PCM data length within one source sample plus one microsecond.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def wav_time_check(structure, decoded):
    fmt = structure["format"]
    if not fmt or fmt["format_tag"] != 1:
        return {"status": "unsupported_non_pcm_wave"}
    data_bytes = sum(chunk["size"] for chunk in structure["chunks"] if chunk["type"] == "data")
    align, rate = fmt["block_align"], fmt["sample_rate"]
    if align <= 0 or rate <= 0 or data_bytes % align:
        return {"status": "invalid_pcm_shape"}
    samples = data_bytes // align
    expected_us = Fraction(samples * 1_000_000, rate)
    actual_us = int(decoded["progress"]["out_time_us"])
    difference = Fraction(actual_us) - expected_us
    tolerance = Fraction(1_000_000, rate) + 1
    return {
        "status": "duration_matches_pcm_samples" if abs(difference) <= tolerance else "decoded_pcm_duration_mismatch",
        "pcm_data_bytes": data_bytes, "block_align": align, "sample_rate": rate,
        "expected_sample_frames": samples, "expected_duration_us": float(expected_us),
        "decoded_out_time_us": actual_us, "difference_us": float(difference),
        "tolerance_us": float(tolerance),
    }


def self_test():
    structure = {"format": {"format_tag": 1, "block_align": 4, "sample_rate": 48000},
                 "chunks": [{"type": "data", "size": 192004}]}
    good = {"progress": {"out_time_us": "1000021"}}
    assert wav_time_check(structure, good)["status"] == "duration_matches_pcm_samples"
    bad = {"progress": {"out_time_us": "100000"}}
    assert wav_time_check(structure, bad)["status"] == "decoded_pcm_duration_mismatch"
    structure["chunks"][0]["size"] += 1
    assert wav_time_check(structure, good)["status"] == "invalid_pcm_shape"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.test_only:
        self_test()
        print("synthetic_media_crosschecks_pass", flush=True)
    if args.test_only:
        return
    plan_path = BASE / "decode_results/plan.json"
    plan = read(plan_path)
    structures = {}
    structure_files = sorted((BASE / "archive_results").glob("TS*_structure.json"))
    for path in structure_files:
        for row in read(path)["media"]:
            key = (row["archive_path"], row["member_name"])
            if key in structures:
                raise ValueError("duplicate structural member key")
            structures[key] = row
    rows, evidence = [], []
    for selected in plan["records"]:
        path = BASE / "decode_results/records" / (selected["name"] + ".json")
        if not path.is_file():
            raise ValueError("Decode incomplete; missing " + selected["name"])
        decoded = read(path)
        if any(decoded.get(k) != value for k, value in selected.items()):
            raise ValueError("Saved decoder record differs from plan: " + selected["name"])
        structure = structures[(selected["archive"], selected["member"])]
        if structure["size"] != selected["size"] or structure["offset_data"] != selected["offset"]:
            raise ValueError("Member extent differs between structural and decoder plans")
        result = {"name": selected["name"], "decode_status": decoded["status"],
                  "structure_status": structure["status"], "decoder_warning_present": bool(decoded.get("decoder_messages"))}
        if decoded["status"] != "full_decode_pass" or structure["status"] != "structural_screen_pass":
            result["status"] = "upstream_check_failed"
        elif not re.fullmatch(r"[0-9a-f]{64}", decoded.get("member_sha256", "")):
            result["status"] = "missing_local_member_fingerprint"
        elif decoded.get("returncode") != 0 or decoded.get("progress", {}).get("progress") != "end":
            result["status"] = "invalid_decoder_completion"
        elif PurePosixPath(selected["member"]).suffix.lower() == ".wav":
            result.update(wav_time_check(structure, decoded))
        else:
            expected = decoded["video_metadata"]["frame_count"]
            actual = int(decoded["progress"].get("frame", 0))
            duplicated = int(decoded["progress"].get("dup_frames", 0))
            dropped = int(decoded["progress"].get("drop_frames", 0))
            result.update({"metadata_frame_count": expected, "decoded_frames": actual,
                           "duplicate_output_frames": duplicated, "dropped_output_frames": dropped,
                           "status": "video_frames_match_no_output_duplication_or_drop"
                           if expected == actual and actual > 0 and duplicated == dropped == 0
                           else "video_frame_crosscheck_failed"})
        rows.append(result)
        evidence.append(fingerprint(path))
    good = {"duration_matches_pcm_samples", "video_frames_match_no_output_duplication_or_drop"}
    failures = [r for r in rows if r["status"] not in good]
    report = {
        "schema": "decoded-media-structure-crosschecks-v1", "date": "2026-10-02",
        "scope": "Local completed audit JSON only. No original source access and no training/inference.",
        "script": fingerprint(Path(__file__)), "plan": fingerprint(plan_path),
        "structure_reports": [fingerprint(path) for path in structure_files],
        "decoder_result_evidence": evidence,
        "summary": {"planned_records": len(plan["records"]), "checked_records": len(rows),
                    "status_counts": dict(Counter(r["status"] for r in rows)),
                    "failures": len(failures),
                    "wav_max_abs_duration_difference_us": max((abs(r["difference_us"]) for r in rows if "difference_us" in r), default=None)},
        "failures": failures, "records": rows,
        "limitations": ["Decode/sample count consistency does not establish original vendor byte identity, spoken-transcript correctness, audiovisual timing, ROI correctness, or model performance.",
                        "No full TAR payload hash or official checksum comparison is performed by this cross-check."],
    }
    output = BASE / "archive_results/media_crosschecks.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
