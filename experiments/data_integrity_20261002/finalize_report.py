"""Consolidate completed integrity checks without changing source datasets."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def load(relative):
    return json.loads((BASE / relative).read_text(encoding="utf-8"))


def main():
    archive = load("archive_results/summary.json")
    labels = load("label_results/summary.json")
    selected = load("label_results/selection_impact.json")
    label_rows = {r["stem"]: r for r in load("label_results/label_records.json")}
    decode = load("decode_results/summary.json")
    crosschecks = load("archive_results/media_crosschecks.json")
    copies = load("copy_results/summary.json")
    recovery = load("archive_results/old_new_comparison.json")
    roles = load("archive_results/readiness_role_map.json")
    protected_input_changes = [row["path"] for row in roles["evidence"].values()
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]]
    plan = load("decode_results/plan.json")
    decoded = {p.name.removesuffix(".json"): json.loads(p.read_text("utf-8"))
               for p in (BASE / "decode_results/records").glob("*.json")}
    wave_layout_notices, other_decoder_messages = [], []
    for name, row in decoded.items():
        message = row.get("decoder_messages", "").strip()
        if not message:
            continue
        if name.endswith(".wav") and re.fullmatch(r"\[aist#0:0/pcm_s16le @ [0-9a-fA-F]+\] Guessed Channel Layout: stereo", message):
            wave_layout_notices.append(name)
        else:
            other_decoder_messages.append({"name": name, "message": message})
    copy_rows = load("copy_results/details.json")
    if len(decoded) != len(plan["records"]) or len(decoded) != 600 or len(copy_rows) != 360:
        raise ValueError("The requested decode/copy audit has not completed")
    if not copies["full_inventory_covered"] or copies["checked_count"] != len(copy_rows) or copies["status_counts"] != dict(Counter(r["status"] for r in copy_rows)):
        raise ValueError("Copy evidence coverage/count mismatch")
    if crosschecks["summary"]["checked_records"] != 600:
        raise ValueError("Crosscheck evidence is incomplete")
    changes = []
    for row in archive["archives"]:
        stat = Path(row["archive_path"]).stat()
        if {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns} != row["before"]:
            changes.append(row["archive_path"])
    loose_changes = []
    for row in copy_rows:
        stat = Path(row["loose_file"]).stat()
        if stat.st_size != row["loose_after"]["size"] or stat.st_mtime_ns != row["loose_after"]["mtime_ns"]:
            loose_changes.append(row["loose_file"])
    copied = {r["name"]: r for r in copy_rows}
    cross_rows = {r["name"]: r for r in crosschecks["records"]}
    # Decoder stores the fingerprint under member_sha256; copy audit uses
    # archive_member_sha256. Compare both independent reads for MP4 AND WAV.
    shared_hash_checks = [{"name": name, "match": bool(row.get("member_sha256")) and row["member_sha256"] == copied[name].get("archive_member_sha256")}
                          for name, row in decoded.items() if name in copied]
    source_rows = []
    for record in plan["records"]:
        if not record["name"].endswith(".mp4"):
            continue
        source = decoded[record["name"]]
        label = label_rows[Path(record["name"]).stem]
        evidence_issues = []
        if record["archive"] in changes:
            evidence_issues.append("source_changed_since_inventory")
        if cross_rows[record["name"]]["status"] != "video_frames_match_no_output_duplication_or_drop":
            evidence_issues.append("media_crosscheck_failed")
        expected_frames = label["detail"]["video_metadata"]["sample_count"]
        if source.get("video_metadata", {}).get("frame_count") != expected_frames:
            evidence_issues.append("independent_moov_vs_decode_frame_count_mismatch")
        copy = copied.get(record["name"])
        if copy and (copy["status"] != "match" or copy.get("archive_member_sha256") != source.get("member_sha256")):
            evidence_issues.append("copy_vs_decoder_evidence_disagrees")
        source_rows.append({"video": record["name"], "archive": record["archive"], "member": record["member"],
            "offset": record["offset"], "bytes": record["size"], "video_sha256": source.get("member_sha256"),
            "decode_status": source["status"], "label_sha256": label["sha256"],
            "metadata_errors": label["errors"], "metadata_warnings": label["warnings"],
            "sentence_issues": label["detail"].get("sentence_issues", []), "evidence_issues": evidence_issues,
            "status": "decode_and_metadata_pass" if source["status"] == "full_decode_pass" and not label["errors"] and not evidence_issues else "review_required",
            "scope": "Technical integrity only; preserve role map and independently verify transcript/timing before new training."})
    target_groups = {}
    for group in selected["groups"]:
        group_rows = [r for r in selected["records"] if r["group"] == group]
        names = sorted({r["source_stem"] + ".mp4" for r in group_rows})
        target_groups[group] = {**selected["groups"][group], "recordings": len(names),
            "decode_status_counts": dict(Counter(decoded[n]["status"] for n in names)),
            "note": "Selected nominal sentence metadata passed does not establish spoken-text correctness or new audiovisual alignment."}
    changed_labels = sum(v["frozen_label_hash_mismatches"] for v in selected["groups"].values())
    evidence_paths = ["archive_results/summary.json", "archive_results/member_index.json",
        "archive_results/old_new_comparison.json", "archive_results/readiness_role_map.json",
        "archive_results/media_crosschecks.json", "archive_results/priority89_local_crosscheck.json", "decode_results/summary.json", "decode_results/plan.json",
        "copy_results/summary.json", "copy_results/details.json", "label_results/summary.json", "label_results/selection_impact.json",
        "label_results/sentence_time_exclusions.json", "label_results/execution_provenance.json", "copy_results/final_stat_verification.json"]
    evidence = {p: hashlib.sha256((BASE / p).read_bytes()).hexdigest() for p in evidence_paths}
    summary = {"schema": "lipnet-data-integrity-consolidated-v1", "audit_date": "2026-10-02",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "finalized_utc": datetime.now(timezone.utc).isoformat(), "source_changes_since_inventory": changes,
        "loose_changes_after_hash_check": loose_changes,
        "protected_input_changes": protected_input_changes,
        "archive_count": archive["archive_count"], "archive_bytes": archive["archive_bytes"],
        "archive_status_counts": archive["archive_status_counts"], "container_checks": archive["media_status_counts"],
        "decode_counts": decode["counts"], "decode_by_extension": decode["by_extension"],
        "wav_stereo_layout_notice_count": len(wave_layout_notices), "other_decoder_messages": other_decoder_messages,
        "copy_counts": dict(Counter(r["status"] for r in copy_rows)), "copy_count": len(copy_rows),
        "labels": labels, "selected_groups": target_groups, "selected_label_hash_mismatches": changed_labels,
        "frontal_source_status_counts": dict(Counter(r["status"] for r in source_rows)),
        "frontal_sources": source_rows, "recovery_comparison": recovery["summary"],
        "role_map_summary": roles["summary"], "media_crosschecks": crosschecks,
        "evidence_sha256": evidence, "training_performed": False,
        "shared_copy_decode_fingerprints": shared_hash_checks,
        "evidence_consistency_pass": not other_decoder_messages and not protected_input_changes and not changes and not loose_changes and copies["all_match"] and all(r["match"] for r in shared_hash_checks) and crosschecks["summary"]["failures"] == 0 and not any(r["evidence_issues"] for r in source_rows),
        "limits": ["No publisher checksum: not a byte-for-byte authenticity claim against AIHub servers.",
                   "1,200 nonfrontal MP4s received container/metadata validation, not full payload decoding.",
                   "Whole-archive payload hashes were not calculated. Selected member and loose-copy hashes are in per-file evidence.",
                   "No exhaustive listening, lip synchronization, crop-content or transcript truth verification.",
                   "Previous G: copies were compared using historical evidence; they were not repaired or modified."]}
    destination = BASE / "overall_summary.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_groups": target_groups, "decode": decode["counts"], "copies": summary["copy_counts"],
                      "frontal": summary["frontal_source_status_counts"], "source_changes": changes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
