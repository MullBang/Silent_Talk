"""Audit reported Q05/Q06 clipping and exact parent-PCM end revisions."""
from __future__ import annotations

from array import array
from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json
from audit_q01_revision import pcm


def rational(value: dict[str, Any]) -> Fraction:
    """Read a recorded exact clock."""
    return Fraction(value["numerator"], value["denominator"])


def difference(left: bytes, right: bytes) -> tuple[int, int]:
    """Count signed int16 differences independently without resampling."""
    assert len(left) == len(right)
    a, b = array("h", left), array("h", right)
    if sys.byteorder != "little":
        a.byteswap()
        b.byteswap()
    errors = [abs(x - y) for x, y in zip(a, b, strict=True)]
    return sum(x != 0 for x in errors), max(errors, default=0)


def inspect() -> dict[str, Any]:
    """Check two proposed repairs; preserve their human-review pending state."""
    stage = HERE.parent
    raw_message = 'Q05-맨 뒷부분 "어"가 잘려서 안들림, Q06-맨 뒷부분 "고"부분이 잘려서 안들림'
    parents = {r["review_id"]: r for r in read_json(PREVIOUS / "effective_parent_inputs.json")["records"]}
    output = []
    specs = [("Q05", "004_Q05_original_end_clipping.json", "Q05_end_confirmed_parent", 1),
             ("Q06", "005_Q06_original_end_clipping.json", "Q06_end_plus80ms", 2)]
    for qid, evidence_name, directory, question_index in specs:
        evidence_path = stage / "user_evidence" / evidence_name
        revision_path = stage / "revisions" / directory / "record.json"
        evidence, revision = read_json(evidence_path), read_json(revision_path)
        for row in (evidence["question_batch"], evidence["question_audio"], revision["trigger"],
                    revision["original_candidate"], revision["effective_parent_inputs"], revision["code"]):
            check(row)
        assert revision["trigger"] == bind(evidence_path)
        batch = read_json(Path(evidence["question_batch"]["path"]))
        question = next(r for r in batch["records"] if r["review_id"] == qid)
        old = read_json(Path(revision["original_candidate"]["path"]))
        parent = parents[qid]
        check(parent["media_record"])
        original = read_json(Path(parent["media_record"]["path"]))
        assert evidence["raw_user_message"] == raw_message
        assert evidence["mapped_answer_excerpt"] == raw_message.split(", ")[question_index - 1]
        assert evidence["reply_type"] == "free_text_not_structured_tool_reply"
        assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", "call_0db61fd52a5b4c349e274d8baa2a5d82", question_index]
        for key in ("review_id", "parent_id", "candidate_id"):
            assert evidence[key] == revision[key] == old[key] == question[key] == parent[key]
        assert evidence["question"] == question["question"]
        assert evidence["question_audio"] == question["question_audio"] == bind(Path(old["phrase_audio"]["path"]))
        assert revision["text"] == old["text"]
        assert parent["parent_audio_qualitatively_confirmed"] is True
        parent_origin = parent["original_recording_first_sample_48k"]
        context_origin = original["context_audio"]["source_first_sample"]
        assert revision["parent_origin_sample_48k"] == parent_origin
        first = revision["parent_local_start_sample_16k"]
        stop = revision["parent_local_end_sample_16k"]
        old_stop = revision["old_parent_local_end_sample_16k"]
        assert Fraction(old["phrase_audio"]["source_original_wav_first_sample_48k"] - parent_origin, 3) == first
        assert Fraction(old["phrase_audio"]["source_original_wav_end_sample_48k"] - parent_origin, 3) == old_stop
        assert revision["extension_samples_16k"] == stop - old_stop
        assert Fraction(str(revision["extension_seconds"])) == Fraction(stop - old_stop, 16000)
        source_start = Fraction(parent_origin + 3 * first - context_origin, 3)
        source_stop = Fraction(parent_origin + 3 * stop - context_origin, 3)
        assert source_start.denominator == source_stop.denominator == 1
        sources = {"phrase": parent["audio"], "context": original["context_audio"]}
        windows = {"phrase": (first, stop), "context": (int(source_start) - 3200, int(source_stop) + 3200)}
        payloads: dict[str, bytes] = {}
        for role in ("phrase", "context"):
            media = revision["media"][role]
            check(media)
            check(media["source"])
            assert media["source"] == bind(Path(sources[role]["path"]))
            source_pcm, source_count = pcm(Path(media["source"]["path"]))
            content, count = pcm(Path(media["path"]))
            left, right = windows[role]
            assert 0 <= left < right <= source_count
            assert (media["source_first_sample_16k"], media["source_end_sample_16k"]) == (left, right)
            assert content == source_pcm[2 * left:2 * right]
            assert media["samples"] == count == right - left
            assert media["duration_seconds"] == count / 16000
            assert media["resampling"] is media["padding"] is media["fade"] is False
            payloads[role] = content
        old_pcm, old_count = pcm(Path(evidence["question_audio"]["path"]))
        assert payloads["phrase"][:len(old_pcm)] == old_pcm
        assert old_count == old_stop - first
        compare = payloads["context"][6400:6400 + len(payloads["phrase"])]
        unequal, maximum = difference(payloads["phrase"], compare)
        assert revision["parent_vs_context_pcm"]["unequal_samples"] == unequal
        assert revision["parent_vs_context_pcm"]["max_absolute_int16_difference"] == maximum
        assert revision["phrase_within_context_samples_16k"] == [3200, 3200 + stop - first]
        audio_start, audio_end = Fraction(parent_origin + 3 * first, 48000), Fraction(parent_origin + 3 * stop, 48000)
        offset = Fraction(original["offset_native_samples"], 48000)
        assert rational(revision["audio_start_seconds_exact"]) == audio_start
        assert rational(revision["audio_end_seconds_exact"]) == audio_end
        assert rational(revision["video_start_seconds_exact"]) == audio_start + offset
        assert rational(revision["video_end_seconds_exact"]) == audio_end + offset
        assert revision["video_offset_applied_count"] == 1
        diagnostics = revision["diagnostics"]
        if qid == "Q05":
            assert (first, stop, stop - old_stop, unequal, maximum) == (12200, 49783, 6543, 3, 1)
            assert stop == parent["audio"]["samples"]
            assert diagnostics["native_parent_end_sample_48k"] == parent["original_recording_end_sample_48k"] == 4768078
            assert diagnostics["nominal_revised_end_sample_48k"] == parent_origin + 3 * stop == 4768077
            assert diagnostics["nominal_end_minus_native_annotation_samples_48k"] == -1
            assert rational(diagnostics["nominal_end_minus_native_annotation_seconds_exact"]) == Fraction(-1, 48000)
        else:
            assert (first, stop, stop - old_stop, unequal, maximum) == (40, 20200, 1280, 0, 0)
            check(diagnostics["source_alignment"])
            aligned = read_json(Path(diagnostics["source_alignment"]["path"]))
            idx = aligned["canonical_parent_text"].index("싶")
            assert len(aligned["canonical_parent_text"]) == len(aligned["alignment"]["token_spans"])
            next_start = 320 * aligned["alignment"]["token_spans"][idx]["start_frame"] + 40
            assert diagnostics["next_character_ctc_start_sample_16k"] == next_start == 20520
            assert diagnostics["candidate_to_next_ctc_occupancy_samples"] == next_start - stop == 320
            assert diagnostics["this_is_not_a_verified_speech_gap"] is True
        for obj in (evidence, revision):
            assert obj["phrase_audio_boundary_approved"] is obj["phrase_visual_boundary_approved"] is obj["training_ready"] is False
        assert revision["new_neural_calls"] == revision["new_tar_payload_bytes"] == revision["training_updates"] == 0
        assert revision["new_video_created"] is False
        assert revision["actual_split"] is revision["frame_count"] is revision["ctc_feasible"] is None
        output.append({"review_id": qid, "evidence": bind(evidence_path), "revision": bind(revision_path),
                       "phrase_samples": stop - first, "extension_samples": stop - old_stop,
                       "old_audio_exact_prefix": True, "parent_context_unequal_samples": unequal,
                       "parent_context_max_abs_difference": maximum, "offset_applied_once": True,
                       "new_audio_approval": False, "new_visual_approval": False})
    return {"schema": "offline-stage5g-independent-Q05-Q06-revision-audit-v1", "status": "passed",
            "records": output, "actual_free_text_message_count": 1, "mapped_question_answers": 2,
            "exact_pcm_slices": 4, "TAR_reads": 0, "neural_calls": 0,
            "auditor_calibration": "The first attempt compared a JSON float directly with Fraction and failed before producing an audit. Corrected to parse the recorded decimal string as Fraction; producer files and PCM unchanged.",
            "limit": "Stored CTC occupancy is not an independently heard speech gap; both new endings remain unconfirmed.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q05_Q06_revision_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "pcm_slices": 4, "new_approvals": 0}))
