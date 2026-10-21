"""Check stage 5f policy and selected inputs against the frozen stage 5e queue."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from audit_preflight import HERE, ROOT, S5E, bind, check_binding, read_json


def inspect() -> dict[str, Any]:
    base = HERE.parent
    policy_path = base / "execution_policy.json"
    policy = read_json(policy_path)
    selected_path = base / "selected_inputs.json"
    selected = read_json(selected_path)
    queue = read_json(S5E / "design/review_queue.json")["records"]
    eligible = {r["id"]: r for r in read_json(S5E / "data/eligible_sentences.json")["records"]}
    for source in policy["bindings"] + selected["sources"]:
        check_binding(source)
    records = selected["records"]
    assert len(records) == 8
    assert [r["review"] for r in records] == queue[:8]
    assert [r["parent"]["id"] for r in records] == policy["record_ids"]
    assert [r["review"]["candidate_id"] for r in records] == policy["candidate_ids"]
    assert policy["review_ids"] == [f"Q{i:02}" for i in range(1, 9)]
    assert policy["acoustic_forward_in_this_initial_policy"] is False
    assert policy["parent_approval_implies_phrase_approval"] is False
    assert policy["unanswered_is_pending"] is True
    assert policy["actual_split"] is None
    assert policy["training_ready"] is False and selected["training_ready"] is False
    assert selected["new_human_approvals"] == 0
    assert policy["video"]["training_cache_creation"] is False
    assert policy["audio"]["parent_context_seconds_each_side"] == .35
    native = []
    for row in records:
        parent, review = row["parent"], row["review"]
        assert parent == eligible[parent["id"]]
        assert parent["id"] == review["parent_id"]
        assert parent["text"] == review["parent_text"]
        assert parent["source_text"] == review["parent_source_text"]
        assert parent["video_sha256"] == review["recording_video_sha256"]
        assert parent["label_sha256"] == review["recording_label_sha256"]
        assert abs(parent["start_time"] - parent["original_start_time"] - parent["offset_seconds"]) < 1e-10
        assert abs(parent["end_time"] - parent["original_end_time"] - parent["offset_seconds"]) < 1e-10
        first = round(Fraction(str(parent["original_start_time"])) * 48000)
        end = round(Fraction(str(parent["original_end_time"])) * 48000)
        offset_samples = round(parent["offset_seconds"] * 48000)
        assert abs(parent["offset_seconds"] - offset_samples / 48000) < 1e-10
        native.append({"review_id": review["review_id"], "parent_id": parent["id"],
                       "first_sample_48k": first, "end_sample_48k_exclusive": end,
                       "sample_count_48k": end - first, "offset_samples_48k": offset_samples,
                       "whole_parent_is_requested_phrase": parent["text"] == review["text"]})
    return {"schema": "offline-stage5f-independent-policy-audit-v1", "status": "passed",
            "policy": bind(policy_path), "selected_inputs": bind(selected_path),
            "exact_queue_rows": 8, "exact_parent_inventory_rows": 8,
            "independently_expected_native_geometry": native,
            "source_media_payload_verified_here": False,
            "review_origin_note": "Parent WAV is independently resampled from its exact native crop. Context WAV and D-grid muxed video may have distinct origins and need distinct UI offsets.",
            "parent_confirmation_does_not_confirm_phrase": True,
            "new_approvals": 0, "network_calls": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "policy_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "fixed_rows": result["exact_queue_rows"]}))
