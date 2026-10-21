"""Audit existing metadata for a word/phrase alignment pilot; never read media."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PARENT = ROOT / "experiments/diversity_step15_20261007/data_plan.json"
PARENT_SHA = "64f5cf399a7ddd5af00b858d75b72de1cad300be247e01c6597e7e787fc2eee1"
D_SOURCE = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"


def digest(value: Any) -> str:
    """Hash canonical JSON using the existing provenance contract."""
    try:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Cannot hash metadata") from exc


def read_json(path: Path, refs: dict[str, str], expected: str | None = None) -> Any:
    """Read a workspace JSON file and bind its exact bytes."""
    try:
        path = path.resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".json":
            raise ValueError("Only workspace JSON metadata is allowed")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError(f"Source changed: {path}")
        refs[str(path)] = sha
        return json.loads(raw)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify metadata: {path}") from exc


def text_key(text: str) -> str:
    """Remove whitespace only for exclusion matching, never for word counting."""
    try:
        return "".join(unicodedata.normalize("NFC", text).split())
    except (TypeError, AttributeError) as exc:
        raise ValueError("Expected a transcript string") from exc


def profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count whitespace-delimited surface forms, including repeated occurrences."""
    try:
        counts: Counter[str] = Counter()
        sentences: dict[str, set[str]] = defaultdict(set)
        speakers: dict[str, set[str]] = defaultdict(set)
        recordings: dict[str, set[str]] = defaultdict(set)
        for row in records:
            for word in row["text"].split():
                counts[word] += 1
                sentences[word].add(row["id"])
                speakers[word].add(row["speaker_id"])
                recordings[word].add(row["source_stem"])
        words = [{"surface": word, "occurrences": count,
                  "sentences": len(sentences[word]), "speakers": len(speakers[word]),
                  "recordings": len(recordings[word]), "sentence_ids": sorted(sentences[word]),
                  "speaker_ids": sorted(speakers[word])}
                 for word, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
        return {"sentence_count": len(records),
                "speaker_count": len({row["speaker_id"] for row in records}),
                "recording_count": len({row["source_stem"] for row in records}),
                "frames": sum(row["frames"] for row in records),
                "eojeol_occurrences": sum(counts.values()), "surface_types": len(counts),
                "singleton_types": sum(count == 1 for count in counts.values()),
                "types_by_min_speakers": {str(n): sum(len(s) >= n for s in speakers.values())
                                         for n in (2, 3, 4, 5)},
                "words": words}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid transcript records for frequency counting") from exc


def dev_reasons(row: dict[str, Any], dev: list[dict[str, Any]], text: str) -> list[str]:
    """Protect complete dev16 recordings and duplicate full transcripts."""
    try:
        reasons = ["dev16_" + key for key in ("source_stem", "video_sha256", "label_sha256")
                   if row[key] in {item[key] for item in dev}]
        if text_key(text) in {text_key(item["text"]) for item in dev}:
            reasons.append("dev16_full_transcript")
        return reasons
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing source identity for exclusion") from exc


def build() -> dict[str, Any]:
    """Build a reproducible metadata inventory and an unapproved alignment queue."""
    try:
        refs: dict[str, str] = {}
        parent = read_json(PARENT, refs, PARENT_SHA)
        source_docs = {name: read_json(Path(parent["source_refs"][name]["path"]), refs,
                                       parent["source_refs"][name]["sha256"])
                       for name in ("handoff", "old_data_plan", "new_data_plan")}
        handoff = source_docs["handoff"]
        logical_rows = source_docs["old_data_plan"]["records"] + source_docs["new_data_plan"]["records"]
        logical = {row["id"]: row for row in logical_rows}
        entries = {row["id"]: row for row in handoff["records"]}
        if len(logical) != 632 or len(logical_rows) != 632 or set(logical) != set(entries):
            raise ValueError("Unexpected or duplicated historical632 inventory")
        for sid, row in logical.items():
            entry = entries[sid]
            if (digest(row) != entry["logical_record_sha256"] or
                    entry["speaker_id"] != row["speaker_id"] or
                    entry["source_stem"] != row["source_stem"] or
                    entry["student_frames"] != row["frames"] or
                    entry["original_split"] != row["split"] or
                    entry["new_human_approval_added"] is not False):
                raise ValueError(f"Provenance mismatch: {sid}")
        d_source = read_json(D_SOURCE, refs)
        if d_source["parent_plan"] != parent["source_refs"]["old_data_plan"]:
            raise ValueError("D mapping does not bind the old data plan")
        sources = {row["stem"]: row for row in d_source["sources"]}
        dev_ids = set(parent["selection"]["dev16_ids"])
        dev = [row for row in parent["selection"]["records"] if row["id"] in dev_ids]
        if len(dev) != 16:
            raise ValueError("Unexpected dev16 scope")
        inventory, accepted = [], []
        for row in parent["pool_profile"]["records"]:
            sid = row["id"]
            original = logical[sid]
            if row["normalized_target"] != text_key(original["text"]):
                raise ValueError("Parent target mismatch")
            reasons = list(row["reasons"])
            if row["eligible"] != (not reasons):
                raise ValueError("Parent eligibility mismatch")
            extra = dev_reasons(row, dev, original["text"])
            reasons.extend(extra)
            inventory.append({"id": sid, "parent_eligible": row["eligible"],
                              "stage1_analysis_eligible": not reasons,
                              "exclusion_reasons": reasons})
            if reasons:
                continue
            if entries[sid]["role"] != "old_train" or row["generation"] != "old":
                raise ValueError("This first pilot is limited to the old candidate pool")
            source = sources[original["source_stem"]]
            for kind in ("video", "label"):
                if source["members"][kind]["sha256"] != row[kind + "_sha256"]:
                    raise ValueError("D mapping source identity mismatch")
            accepted.append({key: original[key] for key in
                             ("id", "speaker_id", "source_stem", "text", "source_text", "frames",
                              "original_start_time", "original_end_time", "start_time", "end_time",
                              "offset_seconds", "token_ids")} | {
                                  "logical_record_sha256": entries[sid]["logical_record_sha256"],
                                  "video_sha256": row["video_sha256"], "label_sha256": row["label_sha256"],
                                  "eojeols": original["text"].split(),
                                  "word_boundaries": None,
                                  "stage1_human_approval_added": False,
                                  "training_ready_for_new_word_curriculum": False})
        stats = profile(accepted)
        current72 = profile([logical[sid] for sid in parent["selection"]["train72_ids"]])
        # A modest review queue, not a vocabulary chosen for performance reporting.
        shortlist = [word for word in stats["words"]
                     if word["occurrences"] >= 8 and word["speakers"] >= 4 and
                     word["recordings"] >= 4 and re.fullmatch(r"[가-힣]{2,}", word["surface"])]
        targets = {word["surface"] for word in shortlist}
        queue = []
        for speaker in sorted({row["speaker_id"] for row in accepted}):
            candidates = [row for row in accepted if row["speaker_id"] == speaker and
                          50 <= row["frames"] <= 188 and
                          not re.search(r"[0-9A-Za-z()/]", row["source_text"]) and
                          targets.intersection(row["eojeols"])]
            candidates.sort(key=lambda row: (-len(targets.intersection(row["eojeols"])),
                                            row["frames"], row["id"]))
            used: set[str] = set()
            for row in candidates:
                if row["source_stem"] in used:
                    continue
                used.add(row["source_stem"])
                queue.append(row | {"review_priority": len(queue) + 1,
                                    "candidate_target_surfaces": sorted(targets.intersection(row["eojeols"])),
                                    "d_source_members": sources[row["source_stem"]]["members"],
                                    "queue_status": "proposed_for_alignment_review_only"})
                if len(used) == 2:
                    break
            if len(used) != 2:
                raise ValueError(f"Cannot choose two distinct recording examples for {speaker}")
        archive_stats = []
        for archive in d_source["archives"]:
            path = Path(archive["path"])
            try:
                actual = path.stat()
                archive_stats.append({"path": str(path), "exists": True,
                                      "size_matches": actual.st_size == archive["size"],
                                      "mtime_matches": actual.st_mtime_ns == archive["mtime_ns"],
                                      "content_verified_this_stage": False})
            except OSError as exc:
                archive_stats.append({"path": str(path), "exists": False,
                                      "error": str(exc), "content_verified_this_stage": False})
        old_dev_stems = {row["source_stem"] for sid, row in logical.items()
                         if entries[sid]["role"] == "old_development"}
        for path, before in refs.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != before:
                raise ValueError("Input metadata changed during analysis")
        summary = {
            "schema": "lipnet-offline-curriculum-stage1-v1", "date": "2026-10-12",
            "status": "completed_metadata_audit_alignment_pending",
            "target": "Record less than 30 seconds, then recognize Korean from video only",
            "source_files": refs, "source_metadata_unchanged": True,
            "logical_records_verified": len(logical),
            "counts": {"historical_candidate_pool": len(inventory),
                       "parent_eligible": sum(row["parent_eligible"] for row in inventory),
                       "excluded_dev16_recording_from_parent_eligible": sum(
                           row["parent_eligible"] and any(reason.startswith("dev16_") and
                           reason != "dev16_full_transcript" for reason in row["exclusion_reasons"])
                           for row in inventory),
                       "excluded_dev16_text_only_from_parent_eligible": sum(
                           row["parent_eligible"] and row["exclusion_reasons"] == ["dev16_full_transcript"]
                           for row in inventory),
                       "analysis_sentences": len(accepted), "proposed_alignment_sentences": len(queue),
                       "verified_word_clips": 0, "new_training_updates": 0, "new_human_approvals": 0},
            "analysis_pool": {key: value for key, value in stats.items() if key != "words"},
            "historical_train72": {key: value for key, value in current72.items() if key != "words"},
            "review_shortlist_rule": "occurrences>=8, speakers>=4, recordings>=4, at least 2 Hangul syllables",
            "review_shortlist_surface_count": len(shortlist),
            "alignment_queue_rule": "8 speakers x 2 distinct recordings; 50..188 frames; no digits/ASCII/alternative notation; most distinct shortlisted surfaces, then shortest frames, then ID",
            "alignment_queue_frames": sum(row["frames"] for row in queue),
            "alignment_queue_target_surfaces": sorted({word for row in queue for word in row["candidate_target_surfaces"]}),
            "protected_historical_ids": parent["pool_profile"]["historical_excluded_ids"],
            "protected_probe_ids": parent["pool_profile"]["prior_probe_ids"],
            "protected_dev16_ids": sorted(dev_ids),
            "pending_listening_ids": parent["pool_profile"]["pending_listening_ids"],
            "analysis_records_sharing_a_recording_with_olddev71": sum(
                row["source_stem"] in old_dev_stems for row in accepted),
            "limitations": ["Whitespace eojeols are surface forms, not morpheme-level dictionary words",
                            "No verified word boundaries or new transcript approval",
                            "Inherited olddev95/held27 protection is ID/full-transcript based, not whole-recording separation",
                            "Probe14 and dev16 are protected at recording identity and full-transcript level",
                            "This is a historical candidate pool, not an untouched final test",
                            "All-angle recording grouping must be verified before creating any new train/dev split",
                            "No archived media, cache arrays, or model weights were read"],
            "d_archive_stat_checks": archive_stats,
            "training_ready": False, "service_weights_changed": False,
        }
        return {"summary.json": summary, "candidate_inventory.json": inventory,
                "analysis_sentences.json": accepted, "eojeol_frequencies.json": stats["words"],
                "review_shortlist.json": shortlist, "alignment_review_queue.json": queue}
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("Metadata audit failed; no learning data should be generated") from exc


def main() -> None:
    """Write metadata outputs only after all source and exclusion checks pass."""
    try:
        results = build()
        output = BASE / "results"
        output.mkdir(exist_ok=True)
        for name, value in results.items():
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results["summary.json"]["analysis_pool"], ensure_ascii=True))
    except Exception as exc:
        raise SystemExit(f"Stage1 audit failed: {exc}") from exc


if __name__ == "__main__":
    main()
