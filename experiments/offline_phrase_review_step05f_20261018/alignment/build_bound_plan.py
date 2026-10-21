"""Bind actual eight-parent review evidence to a separate, still unexecuted plan."""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from protocol import BASE, STAGE, ROOT, OLD, binding, fraction, read, ratio, verify_binding, write
from run_alignment import validate_plan


def global_audio_origin(row: dict[str, Any]) -> Fraction:
    """Use the explicit recording origin; never interpret Q02's context index globally."""
    audio = row["audio"]
    first, end = row["original_recording_first_sample_48k"], row["original_recording_end_sample_48k"]
    if any(type(value) is not int for value in (first, end)) or not 0 <= first < end:
        raise ValueError("Invalid exact recording sample interval")
    if audio["source_sample_rate"] != 48000 or row["offset_applied_to_audio"]:
        raise ValueError("The origin requires unshifted original 48kHz annotation audio")
    if audio["clock"] == "saved_native_context_relative_samples":
        if (first, end) != (audio["original_recording_first_sample_48k"], audio["original_recording_end_sample_48k"]):
            raise ValueError("Context repair lacks the same explicit recording origin")
    elif audio["clock"] == "original_label_wav":
        if (first, end) != (audio["source_first_sample"], audio["source_end_sample_exclusive"]):
            raise ValueError("Native sample origin disagrees with the exact original slice")
    else:
        raise ValueError("Unknown annotation audio clock")
    origin = Fraction(first, 48000)
    if origin != fraction(row["audio_origin_seconds_rational"]):
        raise ValueError("Explicit rational origin does not match original samples")
    if abs((end - first) - 3 * audio["samples"]) > 2:
        raise ValueError("Resampled parent length disagrees with its exact native interval")
    return origin


def checked_answer(row: dict[str, Any], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Check the root-recorded actual answer against this exact text and WAV hash."""
    answer = answers[row["actual_answer_evidence_id"]]
    if answer["review_id"] != row["review_id"] or answer["parent_id"] != row["parent_id"]:
        raise ValueError("An actual answer belongs to another parent")
    if answer["question_item_id"] != row["actual_answer_question_item_id"]:
        raise ValueError("Question binding changed")
    if answer["source"] != "actual_user_chat_reply" or not answer["answer"]:
        raise ValueError("Actual chat evidence is required; no guessed or blank response")
    if row["parent_text"] not in answer["question"]:
        raise ValueError("The question did not present this exact canonical parent")
    for field in ("parent_transcript_confirmed", "parent_audio_qualitatively_confirmed"):
        if row[field] is not True or answer[field] is not True:
            raise ValueError("Selected parent evidence is not affirmative")
    for field in ("phrase_audio_boundary_approved", "phrase_visual_boundary_approved", "training_ready"):
        if row[field] is not False or answer[field] is not False:
            raise ValueError("Parent review cannot promote phrase or training approvals")
    if answer["question_audio"]["sha256"] != row["audio"]["sha256"]:
        raise ValueError("The actual answer was for different audio bytes")
    return answer


def main() -> None:
    """Prepare policy/plan and validate sources without loading the acoustic model."""
    effective_path = STAGE / "effective_parent_inputs.json"
    answers_path = STAGE / "user_evidence/parent_answers.json"
    initial_policy_path = STAGE / "execution_policy.json"
    queue_path = ROOT / "experiments/offline_phrase_design_step05e_20261018/design/review_queue.json"
    effective, evidence = read(effective_path), read(answers_path)
    rows = effective["records"]
    answers = {r["evidence_id"]: r for r in evidence["records"]}
    if effective["confirmed_parent_count"] != 8 or evidence["actual_answer_count"] != 9 or len(rows) != 8:
        raise ValueError("Expected the exact eight confirmed parents and nine preserved answers")
    for spec in effective["sources"] + evidence["sources"]:
        verify_binding(spec)
    records = []
    for row in rows:
        answer = checked_answer(row, answers)
        origin = global_audio_origin(row)
        media = read(verify_binding(row["media_record"]))
        if media["parent_id"] != row["parent_id"] or media["offset_native_samples"] != row["video_offset_native_samples"]:
            raise ValueError("Media identity or measured offset changed")
        if origin != Fraction(media["parent_native_start_sample"], 48000):
            raise ValueError("Original parent start changed unexpectedly")
        if row["effective_revision_record"]:
            revision = read(verify_binding(row["effective_revision_record"]))
            if row["review_id"] != "Q02" or revision["sentence_audio"]["sha256"] != row["audio"]["sha256"]:
                raise ValueError("Only the separately confirmed Q02 repair is selected")
            if (row["original_recording_first_sample_48k"], row["original_recording_end_sample_48k"]) != (13559808, 13706880):
                raise ValueError("Q02 repaired recording interval changed")
        elif row["audio"]["sha256"] != media["sentence_audio"]["sha256"]:
            raise ValueError("Unrevised parent WAV differs from the prepared media")
        audio = {**binding(verify_binding(row["audio"])), "sample_rate": 16000,
                 "sample_count": row["audio"]["samples"], "source_origin_seconds_exact": ratio(origin)}
        records.append({"review_id": row["review_id"], "parent_id": row["parent_id"],
                        "candidate_id": row["candidate_id"], "speaker_id": row["speaker_id"],
                        "canonical_parent_text": row["parent_text"], "canonical_phrase_text": row["phrase_text"],
                        "word_start": row["word_start"], "word_end": row["word_end"],
                        "human_parent_transcript_confirmed": True, "human_parent_audio_boundary_confirmed": True,
                        "parent_audio_confirmation_precision": "qualitative listening only; not phonetic or articulatory certification",
                        "parent_review_evidence": binding(answers_path),
                        "actual_answer_evidence_id": answer["evidence_id"], "actual_answer_question_item_id": answer["question_item_id"],
                        "parent_audio": audio, "video_offset_seconds_exact": ratio(Fraction(row["video_offset_native_samples"], 48000)),
                        "media_record_binding": row["media_record"], "effective_revision_record": row["effective_revision_record"],
                        "human_phrase_audio_boundary_approved": False, "human_phrase_visual_boundary_approved": False,
                        "frame_count": None, "ctc_feasible": None, "training_ready": False, "actual_split": None})
    policy_path = BASE / "execution_policy_after_parent_review.json"
    policy = {
        "schema": "offline-stage5f-after-parent-review-acoustic-policy-v1", "client_date": "2026-10-18",
        "status": "source_bound_plan_prepared_waiting_root_execution_dispatch",
        "initial_policy_preserved": binding(initial_policy_path), "effective_parents": binding(effective_path),
        "actual_parent_answers": binding(answers_path), "fixed_queue": binding(queue_path),
        "purpose": "Annotation assistance for the same fixed Q01-Q08 after actual full-parent review",
        "acoustic_alignment_neural_calls_allowed": 8, "max_calls_per_parent": 1,
        "review_ids": [r["review_id"] for r in records], "parent_ids": [r["parent_id"] for r in records],
        "source_audio_samples_16k": sum(r["parent_audio"]["sample_count"] for r in records),
        "Q02_origin": "13559808/48000 = 282.496 seconds in original recording; source_first_sample 16800 is context-relative",
        "model_input": "annotation_audio_only", "student_audio_input": False,
        "boundary_semantics": "CTC nonblank occupancy proposals; parent review does not approve phrase or visual boundaries",
        "out_of_vocabulary_policy": "Hold unchanged target before inference; no deletion, wildcard or automatic substitution",
        "no_automatic_boundary_extension": True, "no_context_audio_model_input": True,
        "training_updates": 0, "visual_model_calls": 0, "source_downloads": 0, "service_changes": 0,
        "actual_split": None, "training_ready": False, "source_provisional": True,
        "phrase_audio_approvals": 0, "phrase_visual_approvals": 0, "actual_new_training_clips": 0,
    }
    write(policy_path, policy)
    plan = {"schema": "offline-stage5f-authorized-acoustic-plan-v1", "client_date": "2026-10-18",
            "status": "prepared_not_executed", "neural_execution_authorized": True,
            "root_execution_dispatch_still_required": True, "execution_policy": binding(policy_path),
            "fixed_queue": binding(queue_path), "effective_parents": binding(effective_path),
            "model_setup": binding(OLD / "aligner_model_setup.json"), "model_input": "annotation_audio_only",
            "execution_code": [binding(p) for p in (BASE / "run_alignment.py", BASE / "protocol.py", BASE / "build_bound_plan.py", OLD / "ctc_alignment.py")],
            "records": records, "training_updates": 0, "actual_split": None, "training_ready": False,
            "device": "cuda:0", "dtype": "float32", "padding": False, "attention": "eager", "seed": 42,
            "sample_rule": "16kHz PCM16 input, exact original 48k origin; 320*t+40 sample CTC bin edges; offset once",
            "video_frames_deferred": True, "phrase_human_approvals_added": 0}
    validate_plan(plan)
    plan_path = BASE / "execution_plan_after_parent_review.json"
    write(plan_path, plan)
    write(BASE / "plan_preflight_after_parent_review.json", {
        "status": "passed_prepared_not_executed", "plan": binding(plan_path), "policy": binding(policy_path),
        "parent_count": 8, "actual_answer_count_preserved": 9,
        "source_samples_16k": policy["source_audio_samples_16k"], "neural_calls": 0,
        "Q02_source_origin_seconds_exact": records[1]["parent_audio"]["source_origin_seconds_exact"],
        "phrase_boundary_approvals": 0, "training_updates": 0, "training_ready": False})
    print("Separate eight-parent acoustic plan prepared and validated; neural calls remain zero.")


if __name__ == "__main__":
    main()
