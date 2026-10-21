"""Bind the frozen 512-sentence experiment to audited, read-only D: TAR members.

This module reads local JSON evidence and stats four TAR files. It does not read
G:, extract payloads, decode media, alter sentence records, or train a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments/feature_transfer_step07/data_plan.json"
AUDIT = ROOT / "experiments/data_integrity_20261002"
PARENT_SHA256 = "c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30"
OVERALL_SHA256 = "ca411651e6f50e66a7eb44ebfaad48c7c3be7dc5bd10ea714044eaed3ba9e625"
SCHEMA = "lipnet-audited-tar-source-plan-v1"


class PlanError(ValueError):
    """A frozen contract or its supporting evidence no longer agrees."""


def require(condition: bool, message: str) -> None:
    """Fail explicitly rather than relying on optimization-sensitive asserts."""
    if not condition:
        raise PlanError(message)


def sha256(path: Path | str) -> str:
    """Hash a local evidence file without invoking other applications."""
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError as exc:
        raise PlanError(f"Cannot hash evidence {path}: {exc}") from exc


def read_json(path: Path | str) -> Any:
    """Read UTF-8 evidence with a contextual failure message."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise PlanError(f"Cannot read evidence {path}: {exc}") from exc


def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    """Reject ambiguous duplicate names rather than choosing an arbitrary row."""
    result: dict[str, dict] = {}
    for row in rows:
        value = row[key]
        require(value not in result, f"Duplicate {key}: {value}")
        result[value] = row
    return result


def verify_binding(binding: dict) -> None:
    """Check a local input fingerprint; G: references are never opened."""
    path = Path(binding["path"])
    require(PureWindowsPath(str(path)).drive.upper() != "G:", "G: evidence access forbidden")
    require(sha256(path) == binding["sha256"], f"Evidence SHA mismatch: {path}")


def stat_archives(archives: list[dict]) -> None:
    """Require used TAR sizes and nanosecond mtimes to match the audited snapshot."""
    for archive in archives:
        path = Path(archive["path"])
        try:
            current = path.stat()
        except OSError as exc:
            raise PlanError(f"Cannot stat archive {path}: {exc}") from exc
        require(current.st_size == archive["size"] and current.st_mtime_ns == archive["mtime_ns"],
                f"Archive changed since audit: {path}")


def validate_member(member: dict, expected_name: str) -> None:
    """Reject unsafe, unbounded or non-file TAR references."""
    name = member["member_name"]
    parts = PurePosixPath(name).parts
    require("\\" not in name and ":" not in name and not name.startswith("/")
            and ".." not in parts and PurePosixPath(name).name == expected_name,
            f"Unsafe or wrong member name: {name}")
    require(member["is_file"] is True and member["is_dir"] is False,
            f"Member is not a regular file: {name}")
    require(member["type"] in ("0", "\x00", ""), f"Special TAR entry: {name}")
    require(isinstance(member["size"], int) and member["size"] > 0
            and isinstance(member["offset_data"], int) and member["offset_data"] >= 512
            and member["offset_data"] % 512 == 0, f"Invalid TAR bounds: {name}")


def collect_plan(parent_path: Path = PARENT, audit_root: Path = AUDIT) -> dict:
    """Build a deterministic source plan from pinned parent/audit evidence."""
    bindings: dict[str, str] = {}

    def bind(path: Path | str, expected: str | None = None) -> Any:
        """Fingerprint a JSON input and optionally enforce its upstream digest."""
        path = Path(path).resolve()
        require(PureWindowsPath(str(path)).drive.upper() != "G:", "G: evidence access forbidden")
        actual = sha256(path)
        require(expected is None or actual == expected, f"Evidence SHA mismatch: {path}")
        require(str(path) not in bindings or bindings[str(path)] == actual,
                f"Evidence changed while building: {path}")
        bindings[str(path)] = actual
        return read_json(path)

    parent_path = Path(parent_path).resolve()
    audit_root = Path(audit_root).resolve()
    parent = bind(parent_path, PARENT_SHA256)
    for item in parent["bindings"]:
        bind(item["path"], item["sha256"])
    overall = bind(audit_root / "overall_summary.json", OVERALL_SHA256)
    final = bind(audit_root / "archive_results/final_validation.json")
    require(final["overall_summary_sha256"] == OVERALL_SHA256
            and final["status"] == "evidence_consistency_pass"
            and not final["failures"] and final["checks_count"] == final["passed_count"],
            "Final audit validation did not pass")
    require(overall["evidence_consistency_pass"] is True
            and not overall["source_changes_since_inventory"]
            and not overall["protected_input_changes"]
            and not overall["selected_label_hash_mismatches"], "Overall audit is not consistent")
    evidence = {}
    for relative, digest in overall["evidence_sha256"].items():
        path = (audit_root / relative).resolve()
        require(path.is_relative_to(audit_root), "Audit evidence escapes its directory")
        evidence[relative] = bind(path, digest)
    index = evidence["archive_results/member_index.json"]["members"]
    structures = unique_index(evidence["archive_results/summary.json"]["archives"], "archive_path")
    media_checks = evidence["archive_results/media_crosschecks.json"]
    decoder_bindings = unique_index(media_checks["decoder_result_evidence"], "path")
    decode_plan = unique_index(evidence["decode_results/plan.json"]["records"], "name")
    labels = unique_index(bind(audit_root / "label_results/label_records.json"), "stem")
    frontal = unique_index(overall["frontal_sources"], "video")
    inventory_binding = [b for b in parent["bindings"]
                         if Path(b["path"]).name == "training_inventory.json"]
    require(len(inventory_binding) == 1, "Missing unique training inventory binding")
    inventory = unique_index(read_json(inventory_binding[0]["path"])["sources"], "source_stem")
    indexed: dict[str, list[dict]] = defaultdict(list)
    for row in index:
        if row["is_file"]:
            indexed[PurePosixPath(row["member_name"]).name].append(row)
    records = parent["records"]
    require(len(records) == 512 and len({r["id"] for r in records}) == 512,
            "Expected 512 unique frozen sentence records")
    require(Counter(r["split"] for r in records) == {"train": 441, "dev": 71},
            "Frozen train/dev counts changed")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["source_stem"]].append(record)
    require(len(grouped) == 59, "Expected 59 selected recordings")
    sources = []
    used_archives: dict[str, dict] = {}
    for stem, rows in sorted(grouped.items()):
        require(len({r["split"] for r in rows}) == 1 and len({r["speaker_id"] for r in rows}) == 1,
                f"Recording crosses split/speaker: {stem}")
        old = inventory[stem]
        label = labels[stem]
        front = frontal[stem + ".mp4"]
        require(old["angle"] == "A" and old["speaker_id"] == rows[0]["speaker_id"],
                f"Original source identity differs: {stem}")
        require(label["detail"]["identity"]["frontal_stem"] == stem
                and label["detail"]["identity"]["angle"] == "A"
                and label["detail"]["identity"]["speaker"] == rows[0]["speaker_id"],
                f"Label source identity differs: {stem}")
        require(not label["errors"] and front["status"] == "decode_and_metadata_pass"
                and not front["metadata_errors"] and not front["sentence_issues"]
                and not front["evidence_issues"], f"Unready selected source: {stem}")
        require(all(r["label_sha256"] == label["sha256"] == front["label_sha256"]
                    == old["expected_label_sha256"] for r in rows), f"Label SHA mismatch: {stem}")
        members = {}
        for role, extension in (("video", ".mp4"), ("label", ".json"), ("audio", ".wav")):
            name = stem + extension
            require(len(indexed[name]) == 1, f"Missing or ambiguous TAR member: {name}")
            item = indexed[name][0]
            validate_member(item, name)
            archive = structures[item["archive_path"]]
            require(PureWindowsPath(item["archive_path"]).drive.upper() == "D:",
                    f"Not a D: archive: {item['archive_path']}")
            require(archive["status"] == "structure_pass" and archive["before"] == archive["after"],
                    f"Archive audit failed or changed: {item['archive_path']}")
            snapshot = archive["after"]
            require(item["offset_data"] + item["size"] <= snapshot["size"], f"Member beyond archive: {name}")
            require(PureWindowsPath(old[role]["path"]).name == name
                    and old[role]["size_bytes"] == item["size"], f"Original {role} identity/size differs: {name}")
            if role == "label":
                require(label["archive"] == item["archive_path"] and label["member"] == item["member_name"]
                        and label["size"] == item["size"], f"Label index differs: {name}")
                digest = label["sha256"]
            else:
                decoder_path = (audit_root / "decode_results/records" / (name + ".json")).resolve()
                require(str(decoder_path) in decoder_bindings, f"No anchored decoder record: {name}")
                result = bind(decoder_path, decoder_bindings[str(decoder_path)]["sha256"])
                expected = decode_plan[name]
                require(all(result[key] == value for key, value in expected.items()), f"Decoder plan differs: {name}")
                require(result["status"] == "full_decode_pass" and result["returncode"] == 0
                        and result["progress"]["progress"] == "end", f"Decoder did not complete: {name}")
                require(result["archive"] == item["archive_path"] and result["member"] == item["member_name"]
                        and result["offset"] == item["offset_data"] and result["size"] == item["size"]
                        and result["archive_stat"] == snapshot, f"Decoder TAR reference differs: {name}")
                digest = result["member_sha256"]
                if role == "video":
                    require(front["archive"] == item["archive_path"] and front["member"] == item["member_name"]
                            and front["offset"] == item["offset_data"] and front["bytes"] == item["size"]
                            and front["video_sha256"] == digest, f"Consolidated video differs: {name}")
                    require(all(r["video_size_bytes"] == item["size"]
                                and math.isclose(r["video_fps"], result["video_metadata"]["fps"], abs_tol=1e-12, rel_tol=0)
                                for r in rows), f"Frozen video size/fps differs: {name}")
            require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"Invalid SHA: {name}")
            members[role] = {"archive_path": item["archive_path"], "archive_size": snapshot["size"],
                             "archive_mtime_ns": snapshot["mtime_ns"], "member_name": item["member_name"],
                             "offset": item["offset_data"], "size": item["size"], "sha256": digest}
            used_archives[item["archive_path"]] = {"path": item["archive_path"], **snapshot}
        sources.append({"stem": stem, "speaker_id": rows[0]["speaker_id"], "split": rows[0]["split"],
                        "record_ids": [r["id"] for r in rows], "video_fps": rows[0]["video_fps"], "members": members})
    require({PureWindowsPath(p).name for p in used_archives} == {"TL1.tar", "TL10.tar", "TS1.tar", "TS10.tar"},
            "Unexpected archive selection")
    archives = [used_archives[key] for key in sorted(used_archives)]
    stat_archives(archives)
    sizes = [sum(m["size"] for m in s["members"].values()) for s in sources]
    return {"schema": SCHEMA, "parent_plan": {"path": str(parent_path), "sha256": PARENT_SHA256},
            "audit_root": str(audit_root), "records": deepcopy(records), "sources": sources, "archives": archives,
            "bindings": [{"path": key, "sha256": value} for key, value in sorted(bindings.items())],
            "counts": {"sentences": 512, "train": 441, "dev": 71, "recordings": 59, "archives": 4,
                       "video_members": 59, "label_members": 59, "audio_members": 59},
            "storage": {"strategy": "one_recording_at_a_time_workspace_temporary_files",
                        "max_one_recording_combined_bytes": max(sizes), "total_mapped_member_bytes": sum(sizes),
                        "minimum_free_after_materialization_bytes": 10 * 1024 ** 3,
                        "full_archive_extraction": False, "persistent_original_copy": False},
            "scope": {"source_access": "D: TAR read-only; parent G: paths are historical identifiers only",
                      "original_records_unchanged": True, "new_training": False, "new120_included": False,
                      "audio_model_input": False, "final_test_or_oldval_used": False,
                      "limitations": ["No publisher checksum; audited member hashes are local fingerprints.",
                                      "Archive size/mtime checks do not replace member SHA verification when materializing.",
                                      "Matched metadata is not whole-recording G:/D: byte identity or transcript truth."]}}


def load_plan(path: Path | str, expected_sha256: str) -> dict:
    """Verify plan, immutable parent, bound evidence and current archive snapshots."""
    require(sha256(path) == expected_sha256, "D source plan SHA mismatch")
    plan = read_json(path)
    require(plan["schema"] == SCHEMA, "Unexpected source plan schema")
    for binding in plan["bindings"]:
        verify_binding(binding)
    require(plan["parent_plan"]["sha256"] == PARENT_SHA256, "Wrong parent SHA")
    rebuilt = collect_plan(Path(plan["parent_plan"]["path"]), Path(plan["audit_root"]))
    require(plan == rebuilt, "D source plan content differs from frozen source evidence")
    return plan


def write_new_plan(plan: dict, output: Path | str) -> str:
    """Create a new JSON file exclusively; never overwrite an accepted plan."""
    output = Path(output)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        return sha256(output)
    except OSError as exc:
        raise PlanError(f"Cannot create new plan {output}: {exc}") from exc


def main() -> None:
    """Create or validate the D source plan without reading any media payloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "d_source_plan.json")
    parser.add_argument("--verify-sha256")
    args = parser.parse_args()
    try:
        if args.verify_sha256:
            plan = load_plan(args.output, args.verify_sha256)
            digest = args.verify_sha256
        else:
            plan = collect_plan()
            digest = write_new_plan(plan, args.output)
            load_plan(args.output, digest)
        print(json.dumps({"path": str(args.output.resolve()), "sha256": digest,
                          "counts": plan["counts"], "storage": plan["storage"]}, ensure_ascii=False, indent=2))
    except (PlanError, KeyError, TypeError) as exc:
        parser.exit(1, f"Source plan rejected: {exc}\n")


if __name__ == "__main__":
    main()
