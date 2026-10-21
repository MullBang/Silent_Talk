"""Check final original-audio answers and Q08's separately confirmed revision."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from source_preflight import HERE, PREVIOUS, bind, check, read_json
from audit_q01_revision import pcm
from audit_q05_q06_revisions import rational, difference


def inspect() -> dict[str, Any]:
    """Verify five actual answer bindings and two new PCM slices only."""
    stage = HERE.parent
    specifications = [
        ("009_Q02_original_audio_confirmed.json", "call_823ffdccee6048e39500b84f68a93f82", 0, True),
        ("010_Q03_original_audio_confirmed.json", "call_823ffdccee6048e39500b84f68a93f82", 1, True),
        ("011_Q07_original_audio_confirmed.json", "call_cd00f947c01442fab7aa8b8f495667ef", 0, True),
        ("012_Q08_original_end_clipping.json", "call_cd00f947c01442fab7aa8b8f495667ef", 1, False),
        ("013_Q08_plus80ms_audio_confirmed.json", "call_919a5f3e1c6a478482f87ffe8e91a2b1", 0, True),
    ]
    evidence_paths = []
    for filename, call, index, approved in specifications:
        path = stage / "user_evidence" / filename
        evidence = read_json(path)
        check(evidence["question_manifest"])
        check(evidence["question_audio"])
        batch = read_json(Path(evidence["question_manifest"]["path"]))
        question = batch["records"][index]
        assert evidence["review_id"] == question["review_id"]
        assert evidence["question"] == question["question"]
        assert evidence["question_audio"] == question["question_audio"]
        assert json.loads(evidence["question_item_id"]) == ["request_user_input_async", call, index]
        assert evidence["answer"] == ("온전히 들림·잘림·혼입 없음" if approved else '"타"까지만 들리고 "고"부분은 안들림')
        assert evidence["phrase_audio_boundary_approved"] is approved
        assert evidence["phrase_visual_boundary_approved"] is evidence["training_ready"] is False
        if "revision_record" in evidence:
            assert evidence["revision_record"] == question["revision"]
            assert evidence["original_candidate_approved"] is False
        else:
            assert evidence["previous_candidate_record"] == question["previous_candidate_record"]
        assert question["actual_answer"] is None
        evidence_paths.append(path)
    revision_path = stage / "revisions/Q08_end_plus80ms/record.json"
    revision = read_json(revision_path)
    for name in ("trigger", "original_candidate", "effective_parent_inputs", "code"):
        check(revision[name])
    assert revision["trigger"] == bind(evidence_paths[3])
    final_answer = read_json(evidence_paths[4])
    assert final_answer["revision_record"] == bind(revision_path)
    parent = next(r for r in read_json(Path(revision["effective_parent_inputs"]["path"]))["records"] if r["review_id"] == "Q08")
    original = read_json(Path(parent["media_record"]["path"]))
    sources = {"phrase": parent["audio"], "context": original["context_audio"]}
    payloads = {}
    for role, first, stop in (("phrase", 1000, 24040), ("context", 3400, 32840)):
        spec = revision["media"][role]
        check(spec)
        assert spec["source"] == bind(Path(sources[role]["path"]))
        source, count = pcm(Path(spec["source"]["path"]))
        result, n = pcm(Path(spec["path"]))
        assert 0 <= first < stop <= count and result == source[first * 2:stop * 2]
        assert spec["source_first_sample_16k"] == first and spec["source_end_sample_16k"] == stop
        assert spec["samples"] == n == stop - first
        assert spec["duration_seconds"] == n / 16000
        assert spec["resampling"] is spec["padding"] is spec["fade"] is False
        payloads[role] = result
    old = read_json(Path(revision["original_candidate"]["path"]))
    old_pcm, _ = pcm(Path(old["phrase_audio"]["path"]))
    assert payloads["phrase"].startswith(old_pcm)
    assert difference(payloads["phrase"], payloads["context"][6400:6400 + len(payloads["phrase"])]) == (0, 0)
    origin = parent["original_recording_first_sample_48k"]
    assert origin == revision["parent_origin_sample_48k"] == 5218360
    start, end = Fraction(origin + 3000, 48000), Fraction(origin + 3 * 24040, 48000)
    assert rational(revision["audio_start_seconds_exact"]) == start
    assert rational(revision["audio_end_seconds_exact"]) == end
    offset = Fraction(original["offset_native_samples"], 48000)
    assert rational(revision["video_start_seconds_exact"]) == start + offset
    assert rational(revision["video_end_seconds_exact"]) == end + offset
    assert revision["video_offset_applied_count"] == 1
    assert revision["extension_samples_16k"] == 24040 - revision["old_parent_local_end_sample_16k"] == 1280
    assert revision["extension_seconds"] == .08
    diagnostics = revision["diagnostics"]
    check(diagnostics["source_alignment"])
    aligned = read_json(Path(diagnostics["source_alignment"]["path"]))
    next_index = aligned["canonical_parent_text"].index("강")
    next_sample = aligned["alignment"]["token_spans"][next_index]["start_frame"] * 320 + 40
    assert diagnostics["next_character_ctc_start_sample_16k"] == next_sample == 26600
    assert diagnostics["candidate_to_next_ctc_occupancy_samples"] == next_sample - 24040 == 2560
    assert diagnostics["this_is_not_a_verified_speech_gap"] is True
    assert revision["phrase_audio_boundary_approved"] is revision["phrase_visual_boundary_approved"] is False
    assert final_answer["phrase_audio_boundary_approved"] is True
    return {"schema": "offline-stage5g-independent-remaining-answers-Q08-audit-v1", "status": "passed",
            "actual_answers": [bind(p) for p in evidence_paths], "Q08_revision": bind(revision_path),
            "Q08_phrase_samples": 23040, "Q08_duration_seconds": 1.44, "Q08_context_samples": 29440,
            "newly_confirmed_audio_ids": ["Q02", "Q03", "Q07", "Q08"],
            "Q08_original_clipping_preserved": True, "Q08_revision_snapshot_pending_preserved": True,
            "exact_new_PCM_slices": 2, "new_visual_approvals": 0, "TAR_reads": 0, "neural_calls": 0,
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "remaining_answers_Q08_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "new_answers": 5, "Q08_samples": 23040}))
