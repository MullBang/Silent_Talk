"""Bind the new Q04--Q06 questions to the immutable stage 5f candidates."""
from __future__ import annotations

import json
from pathlib import Path
import wave
from typing import Any

from source_preflight import HERE, bind, check, read_json


def inspect() -> dict[str, Any]:
    """Check three pending question/audio identities, without reading source TAR."""
    path = HERE.parent / "question_batches/002_Q04_Q06.json"
    batch = read_json(path)
    check(batch["source_question_manifest"])
    previous = {r["review_id"]: r for r in read_json(Path(batch["source_question_manifest"]["path"]))["records"]}
    assert [r["review_id"] for r in batch["records"]] == ["Q04", "Q05", "Q06"]
    verified = []
    for row in batch["records"]:
        source = previous[row["review_id"]]
        for key in ("review_id", "parent_id", "candidate_id", "text", "question"):
            assert row[key] == source[key]
        check(row["question_audio"])
        check(row["previous_candidate_record"])
        candidate = read_json(Path(row["previous_candidate_record"]["path"]))
        assert row["question_audio"] == bind(Path(candidate["phrase_audio"]["path"]))
        with wave.open(row["question_audio"]["path"], "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            assert wav.getnframes() == row["samples_16k"] == candidate["phrase_audio"]["samples"]
        assert row["duration_seconds"] == row["samples_16k"] / 16000
        assert row["actual_answer"] is None
        assert row["phrase_audio_boundary_approved"] is row["phrase_visual_boundary_approved"] is row["training_ready"] is False
        verified.append({"review_id": row["review_id"], "samples_16k": row["samples_16k"], "question_and_audio_binding_exact": True})
    assert batch["Q01_revised_audio_confirmed"] is True
    assert batch["Q02_Q03_answers"] == "pending"
    assert batch["new_human_approvals"] == 0 and batch["training_ready"] is False
    return {"schema": "offline-stage5g-independent-Q04-Q06-question-audit-v1", "status": "passed",
            "question_batch": bind(path), "verified": verified, "new_answers_in_batch_snapshot": 0,
            "Q02_Q03_pending_preserved": True, "visual_approvals_added": 0,
            "TAR_reads": 0, "neural_calls": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q04_Q06_question_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "questions": 3}))
