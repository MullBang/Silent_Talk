"""Check the exact pending Q05/Q06 revision question bindings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Verify that each question refers to its independently checked new WAV."""
    path = HERE.parent / "question_batches/003_Q05_Q06_revisions.json"
    batch = read_json(path)
    audit_path = HERE / "Q05_Q06_revision_audit.json"
    audits = {r["review_id"]: r for r in read_json(audit_path)["records"]}
    assert [r["review_id"] for r in batch["records"]] == ["Q05", "Q06"]
    for row in batch["records"]:
        check(row["revision"])
        check(row["question_audio"])
        assert row["revision"] == audits[row["review_id"]]["revision"]
        revision = read_json(Path(row["revision"]["path"]))
        assert row["text"] == revision["text"]
        assert row["question_audio"] == bind(Path(revision["media"]["phrase"]["path"]))
        assert row["text"] in row["question"]
        assert row["actual_answer"] is None
        assert row["phrase_audio_boundary_approved"] is row["phrase_visual_boundary_approved"] is row["training_ready"] is False
    assert "마지막 ‘어’까지" in batch["records"][0]["question"]
    assert "‘고’까지" in batch["records"][1]["question"] and "싶으면" in batch["records"][1]["question"]
    assert batch["new_revision_approvals"] == 0 and batch["training_ready"] is False
    return {"schema": "offline-stage5g-independent-Q05-Q06-revision-question-audit-v1", "status": "passed",
            "batch": bind(path), "revision_audit": bind(audit_path), "questions_verified": 2,
            "Q06_next_word_contamination_question_explicit": True, "new_approvals": 0,
            "scope": "Pre-answer snapshot only; no user response or visual approval inferred.", "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q05_Q06_question_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "questions": 2}))
