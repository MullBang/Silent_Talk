"""Independently verify the eight-call annotation plan before any model forward."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from audit_preflight import HERE, bind, check_binding, digest, read_json


def inspect() -> dict[str, Any]:
    base = HERE.parent
    path = base / "alignment/execution_plan_after_parent_review.json"
    assert digest(path) == "81ecc2acc85a6f7b5bf811e1126d4bf589ffbc3cf265f31a2bd65a690639e5ce"
    plan = read_json(path)
    for name in ["execution_policy", "fixed_queue", "effective_parents", "model_setup"]:
        check_binding(plan[name])
    for source in plan["execution_code"]:
        check_binding(source)
    policy = read_json(Path(plan["execution_policy"]["path"]))
    effective = read_json(Path(plan["effective_parents"]["path"]))["records"]
    answers_path = base / "user_evidence/parent_answers.json"
    answers = {r["evidence_id"]: r for r in read_json(answers_path)["records"]}
    for name in ["initial_policy_preserved", "effective_parents", "actual_parent_answers", "fixed_queue"]:
        check_binding(policy[name])
    assert policy["acoustic_alignment_neural_calls_allowed"] == 8
    assert policy["max_calls_per_parent"] == 1
    assert policy["no_automatic_boundary_extension"] and policy["no_context_audio_model_input"]
    assert policy["model_input"] == plan["model_input"] == "annotation_audio_only"
    assert plan["training_updates"] == policy["training_updates"] == policy["visual_model_calls"] == 0
    assert plan["actual_split"] is None and plan["training_ready"] is False
    assert len(plan["records"]) == len(effective) == 8
    assert [r["review_id"] for r in plan["records"]] == policy["review_ids"]
    assert [r["parent_id"] for r in plan["records"]] == policy["parent_ids"]
    for row, expected in zip(plan["records"], effective, strict=True):
        for name in ["review_id", "parent_id", "candidate_id", "speaker_id", "word_start", "word_end", "actual_answer_evidence_id", "actual_answer_question_item_id", "effective_revision_record"]:
            assert row[name] == expected[name]
        assert row["canonical_parent_text"] == expected["parent_text"]
        assert row["canonical_phrase_text"] == expected["phrase_text"]
        assert row["media_record_binding"] == expected["media_record"]
        assert row["parent_review_evidence"] == bind(answers_path)
        answer = answers[row["actual_answer_evidence_id"]]
        assert answer["parent_transcript_confirmed"] and answer["parent_audio_qualitatively_confirmed"]
        assert answer["parent_id"] == row["parent_id"]
        assert answer["question_item_id"] == row["actual_answer_question_item_id"]
        assert answer["question_audio"] == bind(Path(expected["audio"]["path"]))
        for key in ["path", "sha256", "bytes", "sample_rate"]:
            assert row["parent_audio"][key] == expected["audio"][key]
        assert row["parent_audio"]["sample_count"] == expected["audio"]["samples"]
        assert Fraction(**row["parent_audio"]["source_origin_seconds_exact"]) == Fraction(expected["original_recording_first_sample_48k"], 48000)
        assert Fraction(**row["video_offset_seconds_exact"]) == Fraction(expected["video_offset_native_samples"], 48000)
        assert row["human_parent_transcript_confirmed"] and row["human_parent_audio_boundary_confirmed"]
        assert not row["human_phrase_audio_boundary_approved"] and not row["human_phrase_visual_boundary_approved"]
        assert row["frame_count"] is None and row["ctc_feasible"] is None
        assert row["actual_split"] is None and row["training_ready"] is False
    assert sum(r["parent_audio"]["sample_count"] for r in plan["records"]) == policy["source_audio_samples_16k"] == 446549
    assert Fraction(**plan["records"][1]["parent_audio"]["source_origin_seconds_exact"]) == Fraction(35312, 125)
    assert not (base / "alignment/execution").exists()
    return {"schema": "offline-stage5f-independent-before-forward-plan-audit-v1", "status": "passed",
            "plan": bind(path), "policy": plan["execution_policy"],
            "actual_answer_audit": bind(HERE / "parent_answer_audit.json"),
            "allowed_annotation_forward_calls": 8, "model_calls_at_this_preflight": 0,
            "source_samples_16k": 446549, "exact_effective_audio_and_origins_verified": 8,
            "Q02_global_origin_seconds": 282.496,
            "phrase_audio_and_visual_approvals": 0, "training_updates": 0,
            "root_execution_dispatch_still_required": True, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "alignment_plan_preflight.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "allowed_calls": 8, "calls_before": 0}))
