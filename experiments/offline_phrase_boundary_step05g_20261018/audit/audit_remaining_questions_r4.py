"""Verify remaining audio-question bindings and only the latest r3 snapshot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wave

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check four immutable original question WAVs without inferring replies."""
    stage = HERE.parent
    preserved_path = stage / "results/review_snapshot_manifest_r3.json"
    preserved = read_json(preserved_path)
    assert len(preserved["artifacts"]) == len({r["path"] for r in preserved["artifacts"]}) == 75
    for row in preserved["artifacts"]:
        check(row)
    reports = []
    for filename, ids in [("005_Q02_Q03.json", ["Q02", "Q03"]), ("006_Q07_Q08.json", ["Q07", "Q08"])]:
        path = stage / "question_batches" / filename
        batch = read_json(path)
        check(batch["source_question_manifest"])
        source = {r["review_id"]: r for r in read_json(Path(batch["source_question_manifest"]["path"]))["records"]}
        assert [r["review_id"] for r in batch["records"]] == ids
        for index, row in enumerate(batch["records"]):
            original = source[row["review_id"]]
            for key in ("review_id", "parent_id", "candidate_id", "text", "question"):
                assert row[key] == original[key]
            check(row["question_audio"])
            check(row["previous_candidate_record"])
            record = read_json(Path(row["previous_candidate_record"]["path"]))
            assert row["question_audio"] == bind(Path(record["phrase_audio"]["path"]))
            with wave.open(row["question_audio"]["path"], "rb") as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                assert wav.getnframes() == row["samples_16k"] == record["phrase_audio"]["samples"]
            assert row["duration_seconds"] == row["samples_16k"] / 16000
            assert row["actual_answer"] is None
            assert row["phrase_audio_boundary_approved"] is row["phrase_visual_boundary_approved"] is row["training_ready"] is False
            if filename.startswith("005"):
                assert row["same_original_audio_represented"] is True
                assert json.loads(row["reissued_unanswered_question_item_id"]) == ["request_user_input_async", "call_c6b912f45a194f748b5c9e8dc2e486de", index + 1]
        reports.append({"batch": bind(path), "review_ids": ids, "original_question_audio_bindings_exact": True})
    return {"schema": "offline-stage5g-independent-remaining-question-r4-audit-v1", "status": "passed",
            "records": reports, "r3_snapshot": bind(preserved_path), "r3_entries_preserved": 75,
            "older_frozen_sets_reexpanded": False, "new_approvals_inferred": 0,
            "auditor_schema_note": "Initial check assumed the r2 artifact_count field also existed in r3; corrected to count and deduplicate r3's actual artifacts array. No producer file changed.",
            "question_dispatch_or_reply_not_inferred_from_manifest": True,
            "TAR_reads": 0, "neural_calls": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "remaining_questions_r4_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "questions": 4, "r3_preserved": 75}))
