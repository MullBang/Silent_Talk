"""Integrate completed audits without rewriting provisional caches or past evidence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = BASE / "results"
REPORT = ROOT / "docs/녹화형_단계학습_2026-10-12/03_구절문장_분할과_원본캐시_준비.md"


def read(path: Path) -> Any:
    """Read explicit UTF-8 artifacts and surface missing/inconsistent evidence."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read completion evidence {path}") from exc


def binding(path: Path) -> dict[str, Any]:
    """Capture an immutable local artifact reference with content hash and size."""
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f"Cannot bind {path}") from exc


def save(path: Path, value: Any) -> None:
    """Create only a new completion artifact inside the stage results folder."""
    try:
        if path.resolve().parent != RESULTS.resolve() or path.exists():
            raise ValueError(f"Unsafe or existing completion output {path}")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot save {path}") from exc


def main() -> None:
    """Require all completed gates, summarize actual work, then freeze the artifacts."""
    try:
        audit_paths = [BASE / name for name in (
            "audit/source_time_audit.json", "audit/split_plan_audit.json",
            "audit/pre_extraction_plan_audit.json", "audit/saved_cache_audit.json",
            "audit/neighbor_support/neighbor_support_audit.json",
        )]
        for path in audit_paths:
            audit = read(path)
            if not audit["status"].startswith("passed") or audit.get("issues"):
                raise ValueError(f"Unpassed independent audit: {path}")
        visual = read(BASE / "qc/visual_observations.json")
        if visual["technical_spatial_QC"] != "passed_with_scope_limits" or visual["new_user_approvals"] != 0:
            raise ValueError("Missing scoped visual observations")
        panels = read(BASE / "qc/panel_manifest.json")
        for item in panels["images"]:
            if binding(Path(item["path"]))["sha256"] != item["sha256"]:
                raise ValueError("Reviewed panels changed")
        for item in read(RESULTS / "scope_snapshot.json")["artifacts"]:
            path = Path(item["path"])
            if item.get("exists", True):
                if binding(path)["sha256"] != item["sha256"]:
                    raise ValueError(f"Protected scope/service artifact changed: {path}")
            elif path.exists():
                raise ValueError("Previously absent default deployment checkpoint appeared")
        source = BASE / "extraction/results/extraction_summary.json"
        extracted = read(source)
        plan = read(BASE / "extraction/extraction_plan.json")
        if extracted["cache_count"] != 30 or extracted["temporary_originals_remaining"] != 0:
            raise ValueError("Incomplete original extraction")
        if list((BASE / "extraction/_tmp").iterdir()):
            raise ValueError("Temporary originals remain")
        if extracted["neural_calls"] or extracted["training_updates"]:
            raise ValueError("Unexpected neural execution")
        clips = []
        for parent in extracted["records"]:
            for clip in parent["clips"]:
                for prefix in ("npz", "metadata"):
                    if binding(Path(clip[prefix+"_path"]))["sha256"] != clip[prefix+"_sha256"]:
                        raise ValueError("Cache changed after extraction")
                meta = read(Path(clip["metadata_path"]))
                if meta["training_ready"] or not meta["provisional"] or not meta["ctc_feasible"]:
                    raise ValueError("Unexpected input qualification change")
                clips.append({**clip, "parent_id": parent["parent_id"], "rank": parent["rank"],
                    "speaker_id": meta["speaker_id"], "recording_group": meta["recording_group"],
                    "text": meta["text"], "label_length": meta["label_length"], "ctc_min_frames": meta["ctc_min_frames"],
                    "cache_integrity_audit_passed": True, "technical_sample_QC_passed": True,
                    "display_neighbor_intersections": meta["neighbor_label_intersections"],
                    "boundary_followup_flag": bool(meta["neighbor_label_intersections"]),
                    "boundary_followup_scope": "Preserve display-overlap traceability; native-PTS audit found no neighboring-label sample overlap. Not a newly required human reply or a phonetic certification.",
                    "exact_visual_boundary_certified": False})
        if Counter(c["split"] for c in clips) != {"train": 24, "dev": 6}:
            raise ValueError("Paired split counts changed")
        handoff = {"schema": "offline-stage3-paired-cache-handoff-v1",
            "status": "prepared_and_audited_for_stage4_loader_and_protocol_work",
            "records": clips, "parent_count": 15, "clip_count": 30,
            "cache_schema": plan["cache_schema"], "tokenizer": plan["tokenizer"], "normalization": plan["normalization"],
            "split_plan": binding(BASE / "split/split_plan.json"), "extraction": binding(source),
            "audits": [binding(p) for p in audit_paths],
            "technical_visual_evidence": binding(BASE / "qc/visual_observations.json"),
            "training_ready": False, "new_original_pixels_human_approved": False,
            "scope": "Technically audited small paired pilot; preserved provisional NPZ flags. Not a practical-scale, unseen-speaker or final-test dataset.",
            "next_gate": "Fix a separate variable-length loader and fresh-initialization comparison protocol; validate small backprop and state restoration in stage4 before stage5 comparison training.",
            "prohibited_automatic_actions": ["Reusing historically trained student weights for a clean new comparison", "Promoting unreviewed expansion candidates", "Changing frozen cache/label/approval metadata", "Launching full training or replacing service weights from this handoff alone"],
        }
        save(RESULTS / "cache_handoff.json", handoff)
        stats = {"parent_recordings": 15, "speakers": 8, "phrase_texts": 14,
            "train_parents": 12, "dev_parents": 3, "cache_count": len(clips),
            "total_RGB_frames_including_nested_reuse": sum(c["T"] for c in clips),
            "phrase_frames": sum(c["T"] for c in clips if c["kind"] == "phrase"),
            "sentence_frames": sum(c["T"] for c in clips if c["kind"] == "full_sentence"),
            "NPZ_bytes": sum(c["npz_bytes"] for c in clips),
            "phrase_audio_seconds": 22.4305, "parent_audio_seconds": 57.3671875,
            "paired_role_audio_seconds_including_overlap": 79.7976875,
            "CTC_feasible_clips": 30, "technical_QC_panels": 210,
            "source_payload_bytes_read": extracted["io"]["payload_bytes_read"],
            "maximum_temporary_original_bytes": plan["maximum_member_temp_bytes"],
            "temporary_originals_remaining": 0, "original_extraction_seconds": extracted["elapsed_seconds"],
            "additional_unreviewed_metadata_candidates": 254}
        summary = {"schema": "offline-stage3-preparation-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_small_paired_dataset_preparation_pilot",
            "completion_scope": "Verified15 short phrases and corresponding full15 sentences, fixed12/3 parent split, original D-grid caches, leakage/timing/CTC/pixel integrity and scoped technical QC. Does not complete large-scale293-parent preparation.",
            "stage3_pilot_complete": True, "large_scale_curriculum_dataset_complete": False,
            "stats": stats, "handoff": binding(RESULTS / "cache_handoff.json"),
            "report": binding(REPORT), "test_result": binding(BASE / "extraction/preflight_validation.json"),
            "audits": [binding(p) for p in audit_paths],
            "limitations": ["Only3 development parents from previously used speakers", "15 parent/phrase pairs share source frames", "Qualitative visual review is not precise phonetic-boundary certification", "Rank9 two roles have2.67ms display overlap with the next label but no selected native frame overlap", "The acoustic-OOV parent remains excluded from both paired arms", "254 extra metadata candidates are not approved clips"],
            "training_ready": False, "new_training_updates": 0, "new_neural_calls": 0,
            "new_original_downloads": 0, "service_changes": 0, "new_user_approvals": 0,
            "next_stage": "Stage4 model/loader/backprop-save-restore validation and fair direct-sentence versus phrase-to-sentence comparison protocol; no full comparison training in stage3."}
        save(RESULTS / "stage3_summary.json", summary)
        artifacts = []
        for path in sorted(BASE.rglob("*")):
            if not path.is_file() or any(part.startswith(("__pycache__", "_test", ".pytest", "_tmp")) for part in path.relative_to(BASE).parts):
                continue
            if path.suffix.lower() in (".py", ".json", ".jsonl", ".txt", ".npz", ".png"):
                artifacts.append(binding(path))
        artifacts.append(binding(REPORT))
        save(RESULTS / "completion_manifest.json", {
            "schema": "offline-stage3-completion-snapshot-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": summary["status"], "current_summary": str(RESULTS / "stage3_summary.json"),
            "current_handoff": str(RESULTS / "cache_handoff.json"), "artifacts": artifacts,
            "followup_policy": "Preserve these code/plans/cache/evidence/results. Follow-up work uses a new stage/version; no overwrite or automatic learning.",
            "training_ready": False, "stage3_pilot_complete": True})
        print(json.dumps({"status": summary["status"], "stats": stats, "frozen_artifacts": len(artifacts)}, ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError("Stage3 completion gates failed") from exc


if __name__ == "__main__":
    main()
