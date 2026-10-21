"""Link actual scoped replies into a fifteen-phrase pilot handoff without training approval."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from prepare_phrase_media import BASE, PREVIOUS, load, sha


def main() -> None:
    """Verify scoped visual bindings and preserve old snapshots before final handoff."""
    try:
        output = BASE / "results/verified_phrase_handoff.json"
        if output.exists():
            raise ValueError("Do not overwrite a completed handoff")
        pending_path = BASE / "results/pending_visual_question.json"
        pending = load(pending_path)
        for path, digest in pending["source_sha256"].items():
            if sha(Path(path)) != digest:
                raise ValueError(f"Question target changed: {path}")
        selected_path = BASE / "selected_phrase_manifest_r1.json"
        selected = load(selected_path)
        visual_path = BASE / "user_evidence/006_phrase_visual_answer.json"
        visual = load(visual_path)
        ui_path = BASE / "visual_review_r2/visual_review_manifest.json"
        ui = {r["rank"]: r for r in load(ui_path)["records"]}
        assert visual["covered_ranks"] == [r["rank"] for r in selected["records"]]
        assert visual["answer"] == "7개 모두 뚜렷한 잘림은 보이지 않음"
        checked_snapshots = []
        for manifest in [PREVIOUS / "results/completion_manifest.json", BASE / "visual_review_r2/completion_manifest.json"]:
            frozen = load(manifest)
            artifacts = frozen["artifacts"]
            if isinstance(artifacts, dict):
                artifacts = [{"path": path, "sha256": digest} for path, digest in artifacts.items()]
            if not isinstance(artifacts, list):
                raise TypeError("Unsupported completion artifact schema")
            for artifact in artifacts:
                path = Path(artifact["path"])
                if not path.is_absolute():
                    path = manifest.parent / path
                assert sha(path) == artifact["sha256"], path
            checked_snapshots.append({"path": str(manifest), "sha256": sha(manifest), "count": len(frozen["artifacts"])})
        old_dir = BASE.parent / "offline_boundary_finalize_step02d_20261013"
        old_plan_path = old_dir / "results/extraction_plan.json"
        old_summary_path = old_dir / "results/stage2d_summary_r2.json"
        old_visual_path = old_dir / "user_evidence/004_phrase_visual_answer.json"
        old_plan, old_summary = load(old_plan_path), load(old_summary_path)
        assert old_summary["prior_confirmed_phrase_audio_ranges"] == 8
        assert old_summary["visual_no_obvious_cut_reported_count"] == 8
        rows = []
        for r in old_plan["records"]:
            rows.append({"rank": r["rank"], "parent_id": r["parent_id"], "speaker_id": r["speaker_id"],
                         "recording_group": r["recording_group"], "parent_text": r["parent_text"], **r["phrase"],
                         "duration_seconds": r["duration"]["seconds"], "audio_boundary_confirmed": True,
                         "visual_no_obvious_cut_reported": True, "audio_scope_source": str(old_summary_path),
                         "visual_evidence_path": str(old_visual_path), "selection_source": str(old_plan_path),
                         "exact_visual_boundary_confirmed": False, "new_original_grid_approved": False,
                         "split": "unassigned", "training_ready": False})
        for r in selected["records"]:
            shown = ui[r["rank"]]
            for key in ("parent_id", "segment_id", "text", "start_sample_16k", "end_sample_16k"):
                assert r[key] == shown[key], (r["rank"], key)
            assert r["phrase_audio_approval"] is True
            assert sha(Path(r["audio_evidence_path"])) == r["audio_evidence_sha256"]
            rows.append({"rank": r["rank"], "parent_id": r["parent_id"], "speaker_id": r["speaker_id"],
                         "recording_group": r["parent_id"].rsplit("_", 1)[0], "parent_text": r["parent_text"],
                         "segment_id": r["segment_id"], "text": r["text"],
                         "start_sample_16k": r["start_sample_16k"], "end_sample_16k": r["end_sample_16k"],
                         "duration_seconds": r["duration_seconds"], "audio_boundary_confirmed": True,
                         "visual_no_obvious_cut_reported": True, "audio_evidence_path": r["audio_evidence_path"],
                         "actual_audio_question_id": r["actual_audio_reply"]["questionItemId"],
                         "visual_evidence_path": str(visual_path), "selection_source": r["selection_source"],
                         "exact_visual_boundary_confirmed": False, "new_original_grid_approved": False,
                         "split": "unassigned", "training_ready": False})
        rows.sort(key=lambda r:r["rank"])
        assert len(rows) == len({r["parent_id"] for r in rows}) == 15
        total = sum((Fraction(r["end_sample_16k"]-r["start_sample_16k"],16000) for r in rows), Fraction(0))
        duplicates = {}
        for r in rows:
            duplicates.setdefault(r["parent_text"], []).append(r["parent_id"])
        duplicate_groups = [ids for ids in duplicates.values() if len(ids)>1]
        result = {"schema": "offline-stage2g-reviewed-fifteen-phrase-handoff-v1",
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "representative_phrase_boundary_pilot_complete_with_one_short_alignment_hold",
                  "pilot_scope_complete": True, "stage2_complete": True,
                  "completion_scope": "Sixteen parent transcripts/audio boundaries and fifteen representative short phrases reviewed; one short alignment explicitly held. Does not mean all words/candidates are reviewed or all training gates pass.",
                  "records": rows, "held_short_alignment": selected["held"],
                  "parent_transcripts_and_audio_confirmed": 16, "phrases_audio_confirmed": 15,
                  "phrases_with_no_obvious_visual_cut_reported": 15,
                  "actual_aggregate_visual_replies": 2, "speakers": len({r["speaker_id"] for r in rows}),
                  "recordings": len({r["recording_group"] for r in rows}), "unique_phrase_texts": len({r["text"] for r in rows}),
                  "phrase_total_seconds": float(total), "phrase_total_sample_count_16k": int(total*16000),
                  "duplicate_parent_transcript_groups": duplicate_groups,
                  "sampling_design": "results/next_preparation_decisions_r2.json",
                  "new_sampling_implemented": False, "training_ready": False, "split": "unassigned",
                  "training_cache_count": 0, "new_training_updates": 0,
                  "checked_snapshots": checked_snapshots,
                  "source_sha256": {str(p):sha(p) for p in [selected_path, visual_path, pending_path, ui_path, old_plan_path, old_summary_path, old_visual_path, BASE / "results/selected_phrase_audio_audit.json", BASE / "results/next_preparation_decisions_r2.json", Path(__file__)]}}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"phrases":len(rows),"speakers":result["speakers"],"recordings":result["recordings"],"unique_phrase_texts":result["unique_phrase_texts"],"seconds":float(total),"short_alignment_held":1,"training_ready":False}))
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        raise RuntimeError("Final review handoff failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
