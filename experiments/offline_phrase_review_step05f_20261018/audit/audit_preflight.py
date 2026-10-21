"""Independently verify stage 5f's fixed queue and frozen provenance chain.

This auditor hashes existing artifacts but does not deserialize model weights or
load media. It writes only its own new audit result, leaving frozen producers
and prior outputs untouched.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import unicodedata
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S5E = ROOT / "experiments/offline_phrase_design_step05e_20261018"
EXPECTED_MANIFEST_SHA = "c0f641cbf8d9acd70b8a9eac11013a68bcc12b69b3760cabb2b4b4e2bc629b8a"


def read_json(path: Path) -> Any:
    """Read a repository JSON without executing any producer code."""
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"JSON outside repository: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def digest(path: Path) -> str:
    """Hash a file with bounded memory; weights remain opaque bytes."""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def check_binding(binding: dict[str, Any]) -> None:
    """Check byte identity or an explicitly recorded absent artifact."""
    path = Path(binding["path"])
    if binding.get("exists") is False:
        if path.exists():
            raise AssertionError(f"Previously absent file now exists: {path}")
        return
    if not path.is_file() or digest(path) != binding["sha256"]:
        raise AssertionError(f"Binding mismatch: {path}")
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise AssertionError(f"Byte size mismatch: {path}")


def bind(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def key(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())


def inspect() -> dict[str, Any]:
    manifest_path = S5E / "results/completion_manifest.json"
    if digest(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("Stage 5e manifest does not match assigned frozen identity")
    manifest = read_json(manifest_path)
    assert len(manifest["artifacts"]) == len({r["path"] for r in manifest["artifacts"]}) == 50
    for row in manifest["artifacts"]:
        check_binding(row)
    previous = read_json(S5E / "audit/final_source_preservation.json")
    manifest_results = [{**bind(manifest_path), "verified_entries": 50}]
    for source in previous["frozen_manifests"]:
        check_binding(source)
        frozen = read_json(Path(source["path"]))
        count = source["verified_artifact_entries"]
        assert len(frozen["artifacts"]) == len({r["path"] for r in frozen["artifacts"]}) == count
        for row in frozen["artifacts"]:
            check_binding(row)
        manifest_results.append({**bind(Path(source["path"])), "verified_entries": count})
    assert sum(r["verified_entries"] for r in manifest_results) == 1910
    first_manifest = previous["stage1_manifest"]
    check_binding(first_manifest)
    s1 = read_json(Path(first_manifest["path"]))
    assert len(s1["artifacts"]) == 10
    for row in s1["artifacts"]:
        check_binding(row)
    scope_binding = previous["scope"]["binding"]
    check_binding(scope_binding)
    original_scope = read_json(Path(scope_binding["path"]))["artifacts"]
    assert len(original_scope) == 10
    for row in original_scope:
        check_binding(row)

    handoff_path = S5E / "results/next_step_handoff.json"
    handoff = read_json(handoff_path)
    for name in ["summary", "report", "queue", "role_proposal", "protected_scope"]:
        check_binding(handoff[name])
    queue = read_json(Path(handoff["queue"]["path"]))
    rows = queue["records"]
    first = rows[:8]
    hfirst = handoff["first_batch"]
    assert len(rows) == len({r["parent_id"] for r in rows}) == 20
    assert [r["review_id"] for r in first] == hfirst["review_ids"] == [f"Q{i:02}" for i in range(1, 9)]
    assert [r["parent_id"] for r in first] == hfirst["parent_ids"]
    assert [r["candidate_id"] for r in first] == hfirst["candidate_ids"]
    assert len({r["speaker_id"] for r in first}) == 8
    protected = read_json(Path(handoff["protected_scope"]["path"]))
    checks = []
    for row in first:
        assert row["parent_id"] not in protected["protected_parent_ids"]
        assert row["parent_id"] not in protected["held_short_alignment_parent_ids"]
        assert row["parent_id"] not in protected["pending_listening_ids"]
        assert row["recording_key"] not in protected["protected_recording_keys"]
        assert row["recording_video_sha256"] not in protected["blocked_video_sha256"]
        assert row["recording_label_sha256"] not in protected["blocked_label_sha256"]
        assert key(row["parent_text"]) not in protected["protected_full_text_keys"]
        assert key(row["parent_historical_text"]) not in protected["protected_full_text_keys"]
        assert not any(p in key(row["parent_text"]) for p in protected["protected_phrase_text_keys"])
        assert key(row["text"]) not in protected["protected_candidate_text_keys"]
        assert " ".join(row["parent_text"].split()[row["word_start"]:row["word_end"]]) == row["text"]
        assert row["actual_split"] is None and row["training_ready"] is False
        assert row["parent_transcript_confirmed"] is False and row["candidate_boundary_confirmed"] is False
        assert all(row[k] is None for k in ["start_seconds", "end_seconds", "frame_count", "ctc_feasible"])
        assert not row["new_human_approval_added"]
        checks.append({k: row[k] for k in ["review_id", "parent_id", "candidate_id", "speaker_id", "recording_key", "text", "parent_text", "word_start", "word_end", "proposed_role"]})
    assert handoff["new_training_allowed_in_next_review_stage"] is False
    assert handoff["new_cache_auto_promotion"] is False
    return {
        "schema": "offline-stage5f-independent-source-preflight-v1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_manifests": manifest_results,
        "frozen_artifact_entries_verified": 1910,
        "stage1_additional_manifest": first_manifest,
        "stage1_artifact_entries_verified": 10,
        "service_and_prior_scope_binding": scope_binding,
        "service_and_prior_scope_entries_verified": 10,
        "handoff_binding": bind(handoff_path),
        "first_batch_identity_checks": checks,
        "fixed_first_batch_count": 8,
        "all_eight_parent_transcripts_unconfirmed": True,
        "all_eight_phrase_boundaries_unconfirmed": True,
        "actual_splits_unassigned": True,
        "producer_imports": 0,
        "media_reads": 0,
        "model_tensor_loads": 0,
        "network_forward_calls": 0,
        "optimizer_updates": 0,
        "limitations": [
            "This preflight checks metadata identities, not source media payload or correct speech boundaries.",
            "Parent, phrase audio, and qualitative visual confirmations must remain separate and require actual user evidence.",
            "Historical parent frame counts cannot certify candidate phrase CTC feasibility.",
        ],
    }


if __name__ == "__main__":
    result = inspect()
    output = HERE / "source_preflight.json"
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "frozen_entries": 1910, "first_batch": 8}))
