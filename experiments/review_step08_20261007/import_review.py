"""Validate a Step7 review export and record separate transcript/boundary evidence.

JSON hashes bind a review to its targets and media; they do not authenticate a
human reviewer. A caller may record separately established user provenance. This
tool never edits labels, splits, prior experiments, or training readiness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent
STEP7 = BASE.parent / "transcript_step07_20261007"
STEP6 = BASE.parent / "new_data_step06_20261007"
SUMMARY_SHA = "ec7456b3a5fc6a7a4837241d40f01958cdfbecc43d4a17b2ada00edde2ced9ae"
MAX_INPUT_BYTES = 2 * 1024 * 1024
EXPORT_SCHEMA = "lipnet-step07-human-listening-review-v1"
TOP_KEYS = {"schema", "created_utc", "reviewer", "label_targets", "context_results",
            "media_manifest_sha256", "records"}
ROW_KEYS = {"id", "canonical_target_sha256", "transcript_decision", "boundary_decision",
            "notes", "reviewed_at", "review_method"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_json(raw):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=no_duplicate_keys,
                      parse_constant=invalid_constant)


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha_file(path)}


def read_bound(spec, expected_path):
    expected_path = Path(expected_path).resolve()
    require(isinstance(spec, dict) and set(spec) == {"path", "sha256"}, "Invalid binding")
    require(isinstance(spec["path"], str) and
            str(expected_path) == spec["path"], "Unexpected bound path")
    require(isinstance(spec["sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]), "Invalid bound SHA256")
    raw = expected_path.read_bytes()
    require(sha_bytes(raw) == spec["sha256"], f"Upstream hash mismatch: {expected_path.name}")
    return parse_json(raw)


def load_upstream():
    """Only read fixed project files whose hashes chain to the frozen summary."""
    summary_ref = {"path": str((STEP7 / "results/step07_summary.json").resolve()),
                   "sha256": SUMMARY_SHA}
    summary = read_bound(summary_ref, STEP7 / "results/step07_summary.json")
    labels_ref = summary["artifacts"]["results/label_targets.json"]
    context_ref = summary["artifacts"]["results/context_asr_results.json"]
    labels = read_bound(labels_ref, STEP7 / "results/label_targets.json")
    context = read_bound(context_ref, STEP7 / "results/context_asr_results.json")
    package_ref = summary["artifacts"]["results/review_package.json"]
    package = read_bound(package_ref, STEP7 / "results/review_package.json")
    data = read_bound(package["data"], STEP7 / "review/review_data.json")
    old_summary = read_bound(summary["previous_step_summary"], STEP6 / "results/step06_summary.json")
    media_ref = old_summary["evidence"]["results/media_manifest.json"]
    read_bound(media_ref, STEP6 / "results/media_manifest.json")
    require(data["bindings"]["transcript_step07_20261007/results/label_targets.json"] == labels_ref,
            "Review label binding differs")
    require(data["bindings"]["transcript_step07_20261007/results/context_asr_results.json"] == context_ref,
            "Review context binding differs")
    require(data["media_sha256"] == media_ref["sha256"], "Review media binding differs")
    ids = [r["id"] for r in labels["records"]]
    require(len(ids) == len(set(ids)) == 120, "Upstream cohort must contain 120 unique IDs")
    require(ids == labels["ids"] == context["ids"] == [r["id"] for r in data["records"]],
            "Upstream cohort ordering differs")
    require(Counter(r["split"] for r in labels["records"]) == {"train": 96, "val": 24},
            "Upstream train/dev roles changed")
    speakers = {role: {r["speaker_id"] for r in labels["records"] if r["split"] == role}
                for role in ("train", "val")}
    require(len(speakers["train"]) == 12 and len(speakers["val"]) == 3 and
            speakers["train"].isdisjoint(speakers["val"]), "Upstream speaker roles changed")
    for target, review, ctx in zip(labels["records"], data["records"], context["records"]):
        require(target["id"] == ctx["id"] and target["split"] == review["split"] == ctx["split"],
                "Upstream record roles differ")
        require(isinstance(target["canonical_target"], str), "Canonical target unavailable")
        require(target["canonical_target"] == review["canonical_target"] and
                sha_bytes(target["canonical_target"].encode("utf-8")) == review["canonical_target_sha256"],
                "Canonical target hash differs")
    return {"summary": summary_ref, "label_targets": labels_ref, "context_results": context_ref,
            "media_manifest": media_ref, "review_data": package["data"],
            "records": labels["records"]}


def utc_timestamp(value, name):
    require(isinstance(value, str) and 1 <= len(value) <= 50, f"Invalid {name}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {name}") from exc
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{name} needs a timezone")
    return value


def validate_export(payload, upstream):
    require(isinstance(payload, dict) and set(payload) == TOP_KEYS, "Unexpected or missing export fields")
    require(payload["schema"] == EXPORT_SCHEMA, "Unsupported review schema")
    require(payload["reviewer"] == "user", "Unexpected reviewer declaration")
    utc_timestamp(payload["created_utc"], "created_utc")
    for key in ("label_targets", "context_results"):
        require(isinstance(payload[key], dict) and set(payload[key]) == {"path", "sha256"} and
                payload[key] == upstream[key], f"Export {key} binding mismatch")
    require(payload["media_manifest_sha256"] == upstream["media_manifest"]["sha256"],
            "Export media binding mismatch")
    records = payload["records"]
    require(isinstance(records, list) and len(records) <= 120, "Invalid review record list")
    targets = {r["id"]: r for r in upstream["records"]}
    accepted = {}
    for row in records:
        require(isinstance(row, dict) and set(row) == ROW_KEYS, "Unexpected or missing review fields")
        identity = row["id"]
        require(isinstance(identity, str) and identity in targets, "Unknown review ID")
        require(identity not in accepted, f"Duplicate review ID: {identity}")
        target_hash = sha_bytes(targets[identity]["canonical_target"].encode("utf-8"))
        require(row["canonical_target_sha256"] == target_hash, f"Canonical hash mismatch: {identity}")
        require(row["transcript_decision"] is None or
                (isinstance(row["transcript_decision"], str) and
                 row["transcript_decision"] in {"match", "mismatch", "uncertain"}), "Invalid transcript decision")
        require(row["boundary_decision"] is None or
                (isinstance(row["boundary_decision"], str) and
                 row["boundary_decision"] in {"complete", "cut_suspected", "uncertain"}), "Invalid boundary decision")
        require(isinstance(row["notes"], str) and len(row["notes"]) <= 16000, "Invalid review notes")
        require(row["transcript_decision"] is not None or row["boundary_decision"] is not None or
                bool(row["notes"].strip()), "Empty review row")
        utc_timestamp(row["reviewed_at"], "reviewed_at")
        require(row["review_method"] == "user_interaction_in_local_review_page", "Unsupported review method")
        accepted[identity] = dict(row)
    return accepted


def make_receipt(upstream, records, source_ref, origin_context_note="", caller_confirms_user_submission=False):
    """Caller provenance is a separate assertion, never inferred from JSON fields."""
    require(type(caller_confirms_user_submission) is bool, "Invalid caller origin assertion")
    require(isinstance(origin_context_note, str) and len(origin_context_note) <= 16000, "Invalid origin note")
    require(not caller_confirms_user_submission or origin_context_note.strip(),
            "Confirmed user submission requires a separate caller context note")
    require(isinstance(records, dict) and all(k == r.get("id") for k, r in records.items()),
            "Review dictionary keys differ from record IDs")
    # Keep direct Python callers subject to the same validation as CLI callers.
    validate_export({"schema": EXPORT_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "reviewer": "user", "label_targets": upstream["label_targets"],
                     "context_results": upstream["context_results"],
                     "media_manifest_sha256": upstream["media_manifest"]["sha256"],
                     "records": list(records.values())}, upstream)
    rows = []
    for target in upstream["records"]:
        reviewed = records.get(target["id"])
        rows.append({"id": target["id"], "split": target["split"], "speaker_id": target["speaker_id"],
                     "canonical_target": target["canonical_target"],
                     "canonical_target_sha256": sha_bytes(target["canonical_target"].encode("utf-8")),
                     "review": reviewed,
                     "full_transcript_human_approved": bool(caller_confirms_user_submission and reviewed and
                                                            reviewed["transcript_decision"] == "match"),
                     "human_boundary_approved": bool(caller_confirms_user_submission and reviewed and
                                                     reviewed["boundary_decision"] == "complete"),
                     "training_ready": False})
    return {"schema": "lipnet-step08-review-import-receipt-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "importer": binding(Path(__file__)),
            "input": source_ref,
            "upstream": {k: upstream[k] for k in ("summary", "label_targets", "context_results",
                                                  "media_manifest", "review_data")},
            "provenance": {"caller_confirms_user_submission": caller_confirms_user_submission,
                           "origin_context_note": origin_context_note,
                           "human_origin_authenticated_by_json": False,
                           "meaning": "Hashes validate data binding; caller assertion separately describes provenance."},
            "counts": {"cohort": len(rows), "train": sum(r["split"] == "train" for r in rows),
                       "val": sum(r["split"] == "val" for r in rows), "imported_records": len(records),
                       "transcript_decisions": sum(r["transcript_decision"] is not None for r in records.values()),
                       "boundary_decisions": sum(r["boundary_decision"] is not None for r in records.values()),
                       "notes_only": sum(r["transcript_decision"] is None and r["boundary_decision"] is None
                                         for r in records.values()),
                       "full_transcript_human_approved": sum(r["full_transcript_human_approved"] for r in rows),
                       "human_boundary_approved": sum(r["human_boundary_approved"] for r in rows)},
            "records": rows, "training_ready": False, "labels_modified": False, "roles_modified": False,
            "automatic_label_corrections": 0, "models_trained": 0}


def write_new_result(path, payload):
    path = Path(path).resolve()
    require(path.is_relative_to((BASE / "results").resolve()) and path.is_relative_to(BASE.resolve()),
            "Outputs must stay in Step8 results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def import_file(input_path, origin_context_note="", caller_confirms_user_submission=False):
    path = Path(input_path).resolve(strict=True)
    require(path.is_file() and path.stat().st_size <= MAX_INPUT_BYTES, "Input is missing or too large")
    raw = path.read_bytes()
    require(len(raw) <= MAX_INPUT_BYTES, "Input is too large")
    payload = parse_json(raw)
    upstream = load_upstream()
    records = validate_export(payload, upstream)
    source_ref = {"path": str(path), "sha256": sha_bytes(raw), "bytes": len(raw),
                  "schema": payload["schema"], "created_utc": payload["created_utc"]}
    return make_receipt(upstream, records, source_ref, origin_context_note, caller_confirms_user_submission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight", help="Verify frozen upstream bindings without creating user decisions")
    pre.add_argument("--output", type=Path, default=BASE / "results/import_preflight.json")
    imp = sub.add_parser("import-review", help="Read one explicitly supplied Step7 JSON export")
    imp.add_argument("--input", type=Path, required=True)
    imp.add_argument("--origin-context-note", default="")
    imp.add_argument("--caller-confirms-user-submission", action="store_true",
                     help="Caller has separate user-origin evidence; this does not authenticate JSON")
    imp.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        upstream = load_upstream()
        result = {"schema": "lipnet-step08-review-import-preflight-v1", "status": "passed",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "importer": binding(Path(__file__)),
                  "upstream": {k: v for k, v in upstream.items() if k != "records"},
                  "counts": {"cohort": 120, "train": 96, "val": 24},
                  "user_records_created": 0, "training_ready": False, "models_trained": 0}
        output = args.output
    else:
        result = import_file(args.input, args.origin_context_note, args.caller_confirms_user_submission)
        output = args.output or BASE / "results" / ("review_receipt_" + result["input"]["sha256"][:16] + ".json")
    write_new_result(output, result)
    print(json.dumps({"output": str(Path(output).resolve()), "sha256": sha_file(output),
                      "counts": result["counts"], "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
