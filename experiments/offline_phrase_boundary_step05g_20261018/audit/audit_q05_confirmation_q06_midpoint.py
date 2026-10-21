"""Bind Q05 confirmation/Q06 mixing reply and check Q06's midpoint trial."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from source_preflight import HERE, bind, check, read_json
from audit_q01_revision import pcm
from audit_q05_q06_revisions import rational, difference


def inspect() -> dict[str, Any]:
    """Verify new replies and two PCM slices without inferring boundary approval."""
    stage = HERE.parent
    evidence_paths = [stage / "user_evidence/006_Q05_revised_audio_confirmed.json",
                      stage / "user_evidence/007_Q06_plus80ms_mixing_reported.json"]
    evidences = [read_json(p) for p in evidence_paths]
    for index, evidence in enumerate(evidences):
        for key in ("question_manifest", "question_audio", "revision_record"):
            check(evidence[key])
        batch = read_json(Path(evidence["question_manifest"]["path"]))["records"]
        question = batch[index]
        assert evidence["question"] == question["question"]
        assert evidence["question_audio"] == question["question_audio"]
        assert evidence["revision_record"] == question["revision"]
        assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_24356cfdf7134061b0212ea4e81d6b88", index]
        assert evidence["answer"] == ["온전히 들림·잘림·혼입 없음", "뒤의 다른 말이 섞임"][index]
        assert evidence["phrase_audio_boundary_approved"] == (index == 0)
        assert evidence["phrase_visual_boundary_approved"] is evidence["original_candidate_approved"] is evidence["training_ready"] is False
    path = stage / "revisions/Q06_end_plus40ms/record.json"
    revision = read_json(path)
    for key in ("trigger", "original_candidate", "effective_parent_inputs", "previous_rejected_revision", "code"):
        check(revision[key])
    assert revision["trigger"] == bind(evidence_paths[1])
    previous = read_json(Path(revision["previous_rejected_revision"]["path"]))
    assert revision["previous_rejected_revision"] == evidences[1]["revision_record"]
    assert revision["parent_local_start_sample_16k"] == 40
    assert revision["parent_local_end_sample_16k"] == 19560
    assert revision["old_parent_local_end_sample_16k"] == 18920
    assert 19560 == (18920 + previous["parent_local_end_sample_16k"]) // 2
    assert revision["extension_samples_16k"] == 640 and revision["extension_seconds"] == 0.04
    payloads = {}
    for role, bounds in (("phrase", (40, 19560)), ("context", (2440, 28360))):
        item = revision["media"][role]
        check(item)
        check(item["source"])
        assert item["source"] == previous["media"][role]["source"]
        source, source_n = pcm(Path(item["source"]["path"]))
        actual, count = pcm(Path(item["path"]))
        first, stop = bounds
        assert (item["source_first_sample_16k"], item["source_end_sample_16k"]) == bounds
        assert actual == source[first * 2:stop * 2] and stop <= source_n
        assert item["samples"] == count == stop - first
        assert item["duration_seconds"] == count / 16000
        payloads[role] = actual
    original = read_json(Path(revision["original_candidate"]["path"]))
    old_pcm, _ = pcm(Path(original["phrase_audio"]["path"]))
    previous_pcm, _ = pcm(Path(previous["media"]["phrase"]["path"]))
    assert payloads["phrase"].startswith(old_pcm) and previous_pcm.startswith(payloads["phrase"])
    assert difference(payloads["phrase"], payloads["context"][6400:6400 + len(payloads["phrase"])]) == (0, 0)
    origin = revision["parent_origin_sample_48k"]
    assert origin == previous["parent_origin_sample_48k"] == 7655340
    start, end = Fraction(origin + 3 * 40, 48000), Fraction(origin + 3 * 19560, 48000)
    assert rational(revision["audio_start_seconds_exact"]) == start
    assert rational(revision["audio_end_seconds_exact"]) == end
    offset = rational(previous["video_start_seconds_exact"]) - rational(previous["audio_start_seconds_exact"])
    assert rational(revision["video_start_seconds_exact"]) == start + offset
    assert rational(revision["video_end_seconds_exact"]) == end + offset
    assert revision["video_offset_applied_count"] == 1
    assert revision["diagnostics"]["actual_intruding_word_identified_by_user"] is False
    assert revision["diagnostics"]["candidate_to_next_ctc_occupancy_samples"] == 20520 - 19560 == 960
    assert revision["phrase_audio_boundary_approved"] is revision["phrase_visual_boundary_approved"] is revision["training_ready"] is False
    question_path = stage / "question_batches/004_Q06_plus40ms.json"
    question = read_json(question_path)["records"][0]
    assert question["revision"] == bind(path)
    assert question["question_audio"] == bind(Path(revision["media"]["phrase"]["path"]))
    assert "0.04초" in question["question"] and "다른 말은 섞이지" in question["question"]
    assert question["actual_answer"] is None and question["phrase_audio_boundary_approved"] is False
    return {"schema": "offline-stage5g-independent-Q05-confirmation-Q06-midpoint-audit-v1", "status": "passed",
            "actual_answers": [bind(p) for p in evidence_paths], "new_trial": bind(path), "question": bind(question_path),
            "Q05_revised_audio_confirmed": True, "Q06_80ms_rejected_for_other_speech": True,
            "intruding_word_not_inferred": True, "Q06_40ms_PCM_samples": 19520, "Q06_40ms_duration": 1.22,
            "exact_pcm_slices": 2, "prior_PCM_prefix_relations_exact": True,
            "Q06_40ms_approved": False, "new_visual_approvals": 0, "TAR_reads": 0, "neural_calls": 0,
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q05_confirmation_Q06_midpoint_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "Q05_confirmed": True, "Q06_new_trial_pending": True}))
