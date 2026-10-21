"""Verify the actual Q04 reply's exact question/audio and approval scope."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Inspect the new evidence only, preserving the original question snapshot."""
    path = HERE.parent / "user_evidence/003_Q04_original_audio_confirmed.json"
    evidence = read_json(path)
    check(evidence["question_batch"])
    check(evidence["question_audio"])
    batch = read_json(Path(evidence["question_batch"]["path"]))
    row = next(r for r in batch["records"] if r["review_id"] == "Q04")
    for key in ("review_id", "parent_id", "candidate_id", "question", "question_audio"):
        assert evidence[key] == row[key]
    assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_0db61fd52a5b4c349e274d8baa2a5d82", 0]
    assert evidence["answer"] == "온전히 들림·잘림·혼입 없음"
    assert evidence["phrase_audio_boundary_approved"] is True
    assert evidence["phrase_visual_boundary_approved"] is evidence["training_ready"] is False
    assert row["actual_answer"] is None and row["phrase_audio_boundary_approved"] is False
    return {"schema": "offline-stage5g-independent-Q04-confirmation-audit-v1", "status": "passed",
            "actual_answer": bind(path), "confirmed_scope": "Q04 original exact phrase audio only",
            "question_snapshot_pending_preserved": True, "visual_approval_added": False,
            "other_questions_inferred_approved": False, "TAR_reads": 0, "neural_calls": 0,
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q04_confirmation_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "review_id": "Q04"}))
