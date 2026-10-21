"""Independently check each media source row against frozen old/new metadata."""
from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from audit_preflight import HERE, ROOT, S5E, bind, check_binding, read_json


def inspect() -> dict[str, Any]:
    base = HERE.parent
    plan_path = base / "media_prepare/source_plan.json"
    plan = read_json(plan_path)
    for source in plan["bindings"]:
        check_binding(source)
    selected = read_json(base / "selected_inputs.json")["records"]
    historical = read_json(ROOT / "experiments/local_source_step01_20261007/d_source_plan.json")
    sources = {r["stem"]: r for r in historical["sources"]}
    parents = {r["id"]: r for r in historical["records"]}
    mappings = {r["source_stem"]: r for r in read_json(ROOT / "experiments/training_alignment_step06/results/audited_time_mapping_candidates.json")["sources"]}
    assert len(plan["records"]) == 8
    assert plan["planned_members"] == 24
    byte_counts = []
    for actual, expected in zip(plan["records"], selected, strict=True):
        q, parent = expected["review"], expected["parent"]
        assert actual["queue_record"] == q
        assert actual["source_record"] == parents[parent["id"]]
        assert actual["source"] == sources[parent["source_stem"]]
        assert actual["global_mapping_evidence"] == mappings[parent["source_stem"]]
        assert actual["parent_id"] == parent["id"]
        assert actual["candidate_id"] == q["candidate_id"]
        assert actual["review_id"] == q["review_id"]
        assert actual["source"]["members"]["video"]["sha256"] == q["recording_video_sha256"]
        assert actual["source"]["members"]["label"]["sha256"] == q["recording_label_sha256"]
        a = round(Fraction(str(parent["original_start_time"])) * 48000)
        b = round(Fraction(str(parent["original_end_time"])) * 48000)
        assert actual["parent_native_start_sample"] == a
        assert actual["parent_native_end_sample"] == b
        assert actual["context_native_start_sample"] == a - 16800
        assert actual["context_native_end_sample"] == b + 16800
        off = actual["offset_native_samples"]
        assert abs(off / 48000 - parent["offset_seconds"]) < 1e-12
        assert actual["actual_split"] is None and not actual["training_ready"]
        assert set(actual["source"]["members"]) == {"audio", "video", "label"}
        byte_counts.append(sum(r["size"] for r in actual["source"]["members"].values()))
    assert sum(byte_counts) == plan["planned_payload_bytes"] == 2944839568
    assert max(byte_counts) == plan["maximum_temporary_member_bytes"] == 437217504
    assert [r["review_id"] for r in plan["records"]] == plan["allowed_review_ids"]
    assert [r["parent_id"] for r in plan["records"]] == plan["allowed_parent_ids"]
    assert plan["model_calls"] == plan["training_updates"] == 0
    assert not plan["training_ready"] and not plan["approval_added"]
    return {"schema": "offline-stage5f-independent-media-source-plan-audit-v1", "status": "passed",
            "source_plan": bind(plan_path), "fixed_records": 8, "members": 24,
            "expected_producer_payload_bytes": sum(byte_counts),
            "expected_producer_maximum_temporary_member_bytes": max(byte_counts),
            "queue_rows_equal": True, "historical_members_equal": True,
            "native_endpoint_and_offset_formulas_verified": True,
            "tar_payload_reads": 0, "new_approvals": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "source_plan_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "records": result["fixed_records"], "members": result["members"]}))
