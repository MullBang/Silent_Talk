"""Check final stage 5f reporting against the independently audited evidence.

This checks metadata and small bindings only. It does not repeat source-media,
model, frozen-history, or human-boundary verification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_preflight import HERE, ROOT, bind, check_binding, read_json


def inspect() -> dict[str, Any]:
    """Fail closed on a reporting mismatch and return an append-only audit."""
    stage = HERE.parent
    summary_path = stage / "results/stage5f_summary.json"
    handoff_path = stage / "results/next_step_handoff.json"
    summary = read_json(summary_path)
    handoff = read_json(handoff_path)
    effective = read_json(stage / "effective_parent_inputs.json")
    answers = read_json(stage / "user_evidence/parent_answers.json")
    alignment = read_json(stage / "alignment/execution/alignment_results.json")
    media = read_json(stage / "phrase_media_r2/manifest.json")
    questions = read_json(stage / "phrase_review_manifest.json")
    parent_audit = read_json(HERE / "parent_answer_audit.json")
    phrase_audit = read_json(HERE / "phrase_media_r2_audit.json")
    preserved = read_json(HERE / "source_preservation_final.json")
    for name in ("summary", "report", "parent_answers", "effective_parents",
                 "phrase_media_selection", "phrase_questions"):
        check_binding(handoff[name])
    for source in summary["sources"] + summary["independent_audits"]:
        check_binding(source)
    assert all(read_json(Path(row["path"]))["status"] == "passed"
               for row in summary["independent_audits"])
    ids = [f"Q{i:02}" for i in range(1, 9)]
    assert summary["status"] == handoff["current_status"] == (
        "parent_review_and_phrase_proposals_complete_phrase_boundary_review_pending")
    assert [r["review_id"] for r in effective["records"]] == ids
    assert summary["fixed_parents"] == len(effective["records"]) == 8
    assert summary["speakers"] == len({r["speaker_id"] for r in effective["records"]}) == 8
    assert summary["actual_parent_answers"] == len(answers["records"]) == 9
    assert summary["confirmed_parent_transcripts"] == effective["confirmed_parent_count"] == 8
    assert summary["qualitatively_confirmed_parent_audio"] == sum(
        r["parent_audio_qualitatively_confirmed"] for r in effective["records"])
    assert summary["original_parent_end_clipping_reports"] == parent_audit["preserved_original_end_clipping_reports"] == 1
    assert summary["separately_confirmed_parent_revisions"] == parent_audit["selected_native_revisions"] == 1
    assert summary["canonical_text_changes"] == parent_audit["parent_text_corrections"] == 0
    samples = sum(r["audio"]["samples"] for r in effective["records"])
    assert summary["selected_parent_audio_samples_16k"] == samples == 446549
    assert summary["selected_parent_audio_duration_seconds"] == samples / 16000
    assert summary["acoustic_annotation_forward_calls"] == alignment["neural_calls"] == 8
    assert summary["acoustic_posterior_frames"] == sum(r["posterior_shape"][0] for r in alignment["records"]) == 1390
    assert summary["acoustic_execution_seconds"] == alignment["elapsed_seconds"]
    assert summary["acoustic_greedy_parent_exact_matches"] == sum(
        r["acoustic_greedy_text"] == r["canonical_parent_text"] for r in alignment["records"]) == 0
    assert [r["review_id"] for r in summary["phrase_proposals"]] == ids
    for item, actual, question in zip(summary["phrase_proposals"], media["records"], questions["records"], strict=True):
        for key in ("review_id", "parent_id", "candidate_id", "text"):
            assert item[key] == actual[key] == question[key]
        assert item["duration_seconds"] == actual["phrase_audio"]["samples"] / 16000
        assert item["audio"] == {k: actual["phrase_audio"][k] for k in ("path", "sha256", "bytes")}
        for key in ("audio_boundary_status", "visual_boundary_status"):
            assert item[key] == "pending"
        assert all(item[k] is None for k in ("actual_split", "actual_training_frames", "ctc_feasible"))
        assert item["training_ready"] is False
    assert summary["unique_proposed_phrase_texts"] == len({r["text"] for r in media["records"]}) == 7
    assert summary["phrase_proposal_duration_seconds"] == phrase_audit["phrase_pcm_samples_total"] / 16000 == 10.04
    for key in ("phrase_audio_boundary_approvals", "phrase_visual_boundary_approvals", "visual_student_forward_calls",
                "optimizer_updates", "new_training_clips", "new_training_cache_files", "new_original_downloads", "service_changes"):
        assert summary[key] == 0
    for key, source in (("prior_frozen_artifact_entries_preserved", "frozen_artifact_entries_verified"),
                        ("additional_stage1_artifact_entries_preserved", "stage1_artifact_entries_verified"),
                        ("prior_scope_entries_preserved", "service_and_prior_scope_entries_verified")):
        assert summary[key] == preserved[source]
    test_specs = [("media_prepare/test_results.json", "tests_passed"),
                  ("media_prepare/phrase_media_tests.json", "tests_passed"),
                  ("alignment/test_results_after_parent_review.json", "test_count"),
                  ("audit/test_results.json", "observed_tests_passed"),
                  ("audit/viterbi_test_results.json", "observed_tests_passed")]
    assert summary["focused_tests_passed_unique"] == sum(read_json(stage / p)[k] for p, k in test_specs) == 43
    assert handoff["next_step"] == "5g_phrase_audio_then_qualitative_visual_review"
    assert handoff["review_ids"] == ids and handoff["first_question_batch"] == ids[:3]
    assert handoff["actual_phrase_answers"] == questions["actual_phrase_answers"] == 0
    assert handoff["actual_visual_answers"] == questions["actual_visual_answers"] == 0
    assert all(handoff[k] is True for k in handoff if k.startswith("do_not_"))
    assert summary["training_ready"] is handoff["training_ready"] is False
    assert summary["source_provisional"] is True and summary["actual_split"] is None
    assert summary["prior_5c_strict_gate"] is False and summary["prior_OOV_hold_and_pending_preserved"] is True
    report_path = Path(handoff["report"]["path"])
    report = report_path.read_text(encoding="utf8")
    for row in summary["phrase_proposals"]:
        assert f'| {row["review_id"]} | {row["text"]} | {row["duration_seconds"]:.2f}초 | 미확인 | 미확인 |' in report
    for literal in ("실제 답변은 9개", "446,549", "27.9093125", "10.04초", "1,390", "43개",
                    "1,910개", "109.5ms·118.8125ms", "training_ready=false", "다음 5g"):
        assert literal in report
    assert handoff["review_url"] in report
    agents_path = ROOT / "AGENTS.md"
    agents = agents_path.read_text(encoding="utf8")
    newest = agents.split("아래는 이전 시점이다.", 1)[0]
    for literal in ("녹화형 5f", "실제9답변", "전체8부모전사", "구절음성·시각승인0", "다음5g", "training_readyfalse"):
        assert literal in newest
    assert report_path.relative_to(ROOT).as_posix() in newest
    browser_path = stage / "results/browser_review_check.json"
    browser = read_json(browser_path)
    check_binding(browser["page"])
    assert browser["url"] == handoff["review_url"]
    assert browser["visible_parent_review_complete_count"] == browser["visible_phrase_review_pending_count"] == 8
    assert browser["human_audio_approval_inferred"] is browser["human_visual_approval_inferred"] is False
    return {
        "schema": "offline-stage5f-independent-final-report-audit-v1", "status": "passed",
        "summary": bind(summary_path), "handoff": bind(handoff_path), "report": bind(report_path),
        "agents_snapshot": bind(agents_path), "browser_observation": bind(browser_path),
        "parent_actual_answers": 9, "effective_confirmed_parents": 8,
        "parent_audio_samples_16k": samples, "acoustic_calls": 8, "posterior_frames": 1390,
        "phrase_proposals": 8, "unique_phrase_texts": 7, "phrase_seconds": 10.04,
        "phrase_audio_approvals": 0, "phrase_visual_approvals": 0,
        "focused_unique_tests_recorded": 43, "next_step": handoff["next_step"],
        "prior_source_hashes_repeated_this_check": False,
        "raw_media_or_model_reads_this_check": 0,
        "scope": "Final reporting and links match previously audited evidence; browser observation is root-supplied, not an independently repeated browser session.",
        "training_ready": False,
    }


if __name__ == "__main__":
    result = inspect()
    with (HERE / "final_report_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "actual_answers": 9, "effective_parents": 8, "phrase_approvals": 0}))
