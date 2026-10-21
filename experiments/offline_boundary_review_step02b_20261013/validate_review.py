"""Validate an explicit review export without treating its claims as authentication.

This module only reads inputs by default.  An optional --output writes a new
validation receipt, never a training manifest or an approval.  A caller must
separately establish that the supplied file came from the actual human reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import Any

MANIFEST_SCHEMA = "lipnet-offline-boundary-review-manifest-v1"
EXPORT_SCHEMA = "lipnet-offline-boundary-human-review-v2"
TRANSCRIPT = {None, "match", "mismatch", "uncertain"}
SENTENCE_BOUNDARY = {None, "complete", "cut_suspected", "uncertain"}
AUDIO_BOUNDARY = {None, "acceptable", "cut_suspected", "uncertain"}
VISUAL_CONTEXT = {None, "acceptable", "needs_phrase", "uncertain"}
TOP_REQUIRED = {"schema", "reviewer", "review_manifest_sha256", "records", "training_ready"}
TOP_OPTIONAL = {"exported_at", "created_utc", "provenance_note"}
RECORD_KEYS = {
    "id", "manifest_sha256", "target_sha256", "transcript_decision",
    "sentence_boundary_decision", "notes", "segments", "reviewed_at",
}
SEGMENT_KEYS = {
    "segment_id", "text", "original_start_sample_16k", "original_end_sample_16k",
    "transcript_decision", "audio_boundary_decision", "visual_context_decision",
    "notes", "reviewed_at",
}


class ReviewValidationError(ValueError):
    """The submitted file or its binding to the manifest is invalid."""


def require(condition: bool, message: str) -> None:
    """Raise a stable validation error on a failed invariant."""
    try:
        if not condition:
            raise ReviewValidationError(message)
    except ReviewValidationError:
        raise


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON keys, including keys in nested objects."""
    try:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    except ReviewValidationError:
        raise


def reject_constant(value: str) -> None:
    """Reject nonstandard NaN/Infinity JSON constants."""
    try:
        raise ReviewValidationError(f"Non-finite JSON constant: {value}")
    except ReviewValidationError:
        raise


def reject_nonfinite(value: Any) -> None:
    """Also reject overflowing finite-looking literals such as 1e999."""
    try:
        if isinstance(value, float):
            require(math.isfinite(value), "Non-finite JSON number")
        elif isinstance(value, dict):
            for nested in value.values():
                reject_nonfinite(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_nonfinite(nested)
    except ReviewValidationError:
        raise


def load_json(path: Path) -> tuple[dict[str, Any], str]:
    """Parse strict UTF-8 JSON and hash the original bytes for exact binding."""
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object,
                           parse_constant=reject_constant)
        require(isinstance(value, dict), f"JSON root must be an object: {path}")
        reject_nonfinite(value)
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read strict JSON {path}: {exc}") from exc


def exact_keys(value: Any, required: set[str], optional: set[str], where: str) -> None:
    """Require the declared export schema and forbid unrecognized approval fields."""
    try:
        require(isinstance(value, dict), f"{where} must be an object")
        require(required <= value.keys(), f"{where}: missing fields {sorted(required - value.keys())}")
        require(value.keys() <= required | optional,
                f"{where}: unknown fields {sorted(value.keys() - required - optional)}")
    except ReviewValidationError:
        raise


def valid_hash(value: Any, where: str) -> None:
    """Require a lowercase SHA-256 hex digest."""
    try:
        require(isinstance(value, str) and len(value) == 64 and
                all(char in "0123456789abcdef" for char in value), f"{where}: invalid SHA-256")
    except ReviewValidationError:
        raise


def integer(value: Any, where: str, minimum: int = 0) -> int:
    """Require a JSON integer, explicitly excluding booleans."""
    try:
        require(type(value) is int and value >= minimum, f"{where}: invalid integer")
        return value
    except ReviewValidationError:
        raise


def timestamp(value: Any, where: str, required: bool = False) -> None:
    """Validate an ISO timestamp with explicit timezone, if one is supplied."""
    try:
        if value is None:
            require(not required, f"{where}: a recorded decision needs a timestamp")
            return
        require(isinstance(value, str), f"{where}: invalid timestamp type")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, f"{where}: timestamp timezone is required")
    except ValueError as exc:
        if isinstance(exc, ReviewValidationError):
            raise
        raise ReviewValidationError(f"{where}: invalid timestamp") from exc


def decision(value: Any, allowed: set[str | None], where: str) -> None:
    """Validate one submitted decision without assigning an approval."""
    try:
        require(value is None or isinstance(value, str), f"{where}: invalid decision type")
        require(value in allowed, f"{where}: unknown decision")
    except ReviewValidationError:
        raise


def manifest_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Check target identities and exact sample geometry of the trusted manifest."""
    try:
        require(manifest.get("schema") == MANIFEST_SCHEMA, "Unknown manifest schema")
        require(manifest.get("training_ready", False) is False, "Manifest must remain unapproved")
        sources = manifest.get("sources")
        require(isinstance(sources, dict) and bool(sources), "Manifest sources must be a nonempty object")
        for source_path, digest in sources.items():
            require(Path(source_path).is_absolute() or PureWindowsPath(source_path).is_absolute(),
                    "Manifest source paths must be absolute")
            valid_hash(digest, f"source {source_path}")
        records = manifest.get("records")
        require(isinstance(records, list), "Manifest records must be an array")
        index: dict[str, dict[str, Any]] = {}
        for record in records:
            require(isinstance(record, dict), "Manifest record must be an object")
            record_id = record.get("id")
            require(isinstance(record_id, str) and bool(record_id), "Manifest ID is required")
            require(record_id not in index, f"Duplicate manifest record ID: {record_id}")
            valid_hash(record.get("review_target_sha256"), f"{record_id} target")
            require(isinstance(record.get("text"), str) and bool(record["text"]),
                    f"{record_id}: target text is required")
            duration = record.get("context_duration_seconds")
            require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
                    f"{record_id}: invalid context duration")
            segments = record.get("segments")
            require(isinstance(segments, list), f"{record_id}: segments must be an array")
            segment_ids: set[str] = set()
            for segment in segments:
                require(isinstance(segment, dict), f"{record_id}: invalid manifest segment")
                segment_id = segment.get("segment_id")
                require(isinstance(segment_id, str) and bool(segment_id), "Segment ID is required")
                require(segment_id not in segment_ids, f"Duplicate manifest segment ID: {segment_id}")
                segment_ids.add(segment_id)
                require(segment.get("kind") in ("word", "phrase"), f"{segment_id}: unknown kind")
                require(isinstance(segment.get("text"), str) and bool(segment["text"]),
                        f"{segment_id}: empty target")
                start = integer(segment.get("start_sample_16k"), f"{segment_id} start")
                end = integer(segment.get("end_sample_16k"), f"{segment_id} end")
                require(end > start and end <= round(duration * 16000), f"{segment_id}: sample bounds")
                for key, sample in (("start_in_context", start), ("end_in_context", end)):
                    seconds = segment.get(key)
                    require(type(seconds) in (int, float) and math.isfinite(seconds) and
                            abs(seconds - sample / 16000) <= 1 / 16000,
                            f"{segment_id}: {key} does not match samples")
                frames = integer(segment.get("frames_25fps"), f"{segment_id} frames")
                minimum = integer(segment.get("ctc_min_frames"), f"{segment_id} CTC minimum", 1)
                require(frames == round(Fraction((end - start) * 25, 16000)),
                        f"{segment_id}: non-rational frame count")
                require(type(segment.get("ctc_feasible")) is bool and
                        segment["ctc_feasible"] == (frames >= minimum), f"{segment_id}: invalid CTC flag")
            index[record_id] = record
        return index
    except ReviewValidationError:
        raise


def validate(manifest_path: Path, submission_path: Path) -> dict[str, Any]:
    """Return a validation receipt; never infer the identity of a human reviewer."""
    try:
        manifest, manifest_sha = load_json(manifest_path)
        export, export_sha = load_json(submission_path)
        index = manifest_index(manifest)
        exact_keys(export, TOP_REQUIRED, TOP_OPTIONAL, "export")
        require(export["schema"] == EXPORT_SCHEMA, "Unknown export schema")
        require(export["reviewer"] == "user", "Export reviewer declaration must be user")
        require(export["training_ready"] is False, "Export cannot grant training approval")
        require(export["review_manifest_sha256"] == manifest_sha, "Manifest SHA mismatch")
        require(isinstance(export["records"], list), "Export records must be an array")
        for key in ("exported_at", "created_utc"):
            if key in export:
                timestamp(export[key], key)
        if "provenance_note" in export:
            require(isinstance(export["provenance_note"], str), "provenance_note must be text")
        seen: set[str] = set()
        sentence_positive = 0
        positive_segments = 0
        segment_decision_count = 0
        sentence_decision_count = 0
        provisional_candidates: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        for record in export["records"]:
            exact_keys(record, RECORD_KEYS, set(), "record")
            record_id = record["id"]
            require(isinstance(record_id, str) and record_id in index, "Unknown record ID")
            require(record_id not in seen, f"Duplicate submitted record ID: {record_id}")
            seen.add(record_id)
            trusted = index[record_id]
            require(record["manifest_sha256"] == manifest_sha, f"{record_id}: manifest SHA mismatch")
            require(record["target_sha256"] == trusted["review_target_sha256"],
                    f"{record_id}: target SHA mismatch")
            require(isinstance(record["notes"], str), f"{record_id}: notes must be text")
            decision(record["transcript_decision"], TRANSCRIPT, f"{record_id} transcript")
            decision(record["sentence_boundary_decision"], SENTENCE_BOUNDARY, f"{record_id} boundary")
            sentence_has_decision = any(record[key] is not None for key in
                                        ("transcript_decision", "sentence_boundary_decision"))
            timestamp(record["reviewed_at"], f"{record_id} reviewed_at", sentence_has_decision)
            sentence_decision_count += sum(record[key] is not None for key in
                                           ("transcript_decision", "sentence_boundary_decision"))
            sentence_good = (record["transcript_decision"] == "match" and
                             record["sentence_boundary_decision"] == "complete")
            sentence_positive += int(sentence_good)
            require(isinstance(record["segments"], dict), f"{record_id}: segments must be an object")
            segment_index = {row["segment_id"]: row for row in trusted["segments"]}
            for segment_id, submitted in record["segments"].items():
                require(segment_id in segment_index, f"{record_id}: unknown segment ID {segment_id}")
                exact_keys(submitted, SEGMENT_KEYS, set(), f"{record_id}/{segment_id}")
                require(submitted["segment_id"] == segment_id, "Segment key/ID mismatch")
                target = segment_index[segment_id]
                require(submitted["text"] == target["text"], f"{segment_id}: target text mismatch")
                for submitted_key, target_key in (("original_start_sample_16k", "start_sample_16k"),
                                                   ("original_end_sample_16k", "end_sample_16k")):
                    integer(submitted[submitted_key], f"{segment_id} {submitted_key}")
                    require(submitted[submitted_key] == target[target_key], f"{segment_id}: modified boundaries")
                require(isinstance(submitted["notes"], str), f"{segment_id}: notes must be text")
                decision(submitted["transcript_decision"], TRANSCRIPT, f"{segment_id} transcript")
                decision(submitted["audio_boundary_decision"], AUDIO_BOUNDARY, f"{segment_id} audio")
                decision(submitted["visual_context_decision"], VISUAL_CONTEXT, f"{segment_id} visual")
                keys = ("transcript_decision", "audio_boundary_decision", "visual_context_decision")
                timestamp(submitted["reviewed_at"], f"{segment_id} reviewed_at",
                          any(submitted[key] is not None for key in keys))
                segment_decision_count += sum(submitted[key] is not None for key in keys)
                positive = (submitted["transcript_decision"] == "match" and
                            submitted["audio_boundary_decision"] == "acceptable" and
                            submitted["visual_context_decision"] == "acceptable")
                if positive:
                    positive_segments += 1
                    reasons = []
                    if not sentence_good:
                        reasons.append("sentence_transcript_or_boundary_not_positively_reviewed")
                    if not target["ctc_feasible"]:
                        reasons.append("insufficient_visual_ctc_length")
                    item = {"record_id": record_id, "segment_id": segment_id,
                            "kind": target["kind"], "text": target["text"]}
                    if reasons:
                        blockers.append({**item, "reasons": reasons})
                    else:
                        provisional_candidates.append(item)
        return {
            "schema": "lipnet-offline-boundary-review-validation-v1",
            "status": "valid_submission_provenance_unverified",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_path": str(manifest_path.resolve()), "manifest_sha256": manifest_sha,
            "submission_path": str(submission_path.resolve()), "submission_sha256": export_sha,
            "reviewer_declaration": export["reviewer"],
            "actual_human_provenance_verified": False, "training_ready": False,
            "actual_human_approved_training_segments": 0,
            "submitted_sentence_records": len(seen), "manifest_sentence_records": len(index),
            "submitted_sentence_decision_fields": sentence_decision_count,
            "submitted_segment_decision_fields": segment_decision_count,
            "submitted_positive_complete_sentence_records": sentence_positive,
            "submitted_positive_segment_records": positive_segments,
            "provisional_candidates_if_actual_human_provenance_is_verified": provisional_candidates,
            "positive_segment_records_with_other_blockers": blockers,
            "limitations": [
                "A reviewer string, timestamp, manifest hash or positive decision is not identity authentication.",
                "Caller must independently establish actual user submission provenance before using decisions.",
                "This receipt grants no training approval and creates no training split or clips.",
                "Source hashes are manifest declarations; this validator does not rehash original media.",
                "Overlapping word and phrase proposals are not independent training examples.",
            ],
        }
    except ReviewValidationError:
        raise


def write_receipt(path: Path, receipt: dict[str, Any], input_paths: tuple[Path, ...]) -> None:
    """Exclusively create an explicit output path while preserving every input."""
    try:
        resolved = path.resolve()
        require(all(resolved != source.resolve() for source in input_paths), "Output cannot be an input path")
        serialized = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except OSError as exc:
        raise ReviewValidationError(f"Cannot create new receipt {path}: {exc}") from exc


def main() -> int:
    """Run read-only validation unless an explicit new output file is supplied."""
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--manifest", required=True, type=Path)
        parser.add_argument("--submission", required=True, type=Path)
        parser.add_argument("--output", type=Path, help="Explicit NEW receipt path; never overwrites")
        args = parser.parse_args()
        receipt = validate(args.manifest, args.submission)
        if args.output is not None:
            write_receipt(args.output, receipt, (args.manifest, args.submission))
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ReviewValidationError, OSError) as exc:
        print(f"Review validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
