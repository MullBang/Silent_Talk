"""Synthetic CPU-only checks for paired final-state aggregation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("offline_stage5_aggregate_tests", Path(__file__).with_name("aggregate.py"))
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def synthetic_records() -> list[dict]:
    ev = aggregate.evaluator
    plan = json.loads(ev.PLAN_PATH.read_text(encoding="utf-8"))
    records = []
    for seed in (42, 43):
        for arm in ("A_sentence", "B_curriculum"):
            for step in (420, 840, 1260):
                identity = {"seed": seed, "arm": arm, "completed_update": step}
                rows = []
                for item in plan["items"]:
                    for condition in plan["conditions"]:
                        # A's synthetic final primary has exactly one deletion per clip.
                        prediction = item["text"]
                        if arm == "A_sentence" and step == 1260 and item["split"] == "dev" and item["kind"] == "full_sentence" and condition == "normal":
                            prediction = prediction[1:]
                        rows.append({**{k: item[k] for k in ("clip_id", "parent_id", "split", "kind", "T")},
                            "identity": identity, "condition": condition, "reference_text": item["text"],
                            "raw_text": prediction, "status": "success",
                            "metrics": ev.contract.score_prediction(item["text"], prediction)})
                records.append({"identity": identity, "rows": rows, "metrics": ev.summarize_rows(rows),
                    "count": 90, "failures": 0, "evaluation_plan_sha256": ev.PLAN_SHA256,
                    "evaluation_contract_sha256": ev.CONTRACT_SHA256})
    return records


def test_aggregate_uses_final_primary_and_retains_all_diagnostics() -> None:
    result = aggregate.aggregate_records(synthetic_records())
    assert result["evaluations"] == 1080 and result["checkpoints"] == 12 and result["failures"] == 0
    assert result["primary_checkpoint"] == 1260 and result["best_checkpoint_selection"] is False
    assert result["mean_primary"]["A_CER"] == 3 / 74
    assert result["mean_primary"]["B_CER"] == 0
    assert result["mean_primary"]["B_minus_A_CER_pp"] == -300 / 74
    assert len(result["checkpoint_diagnostics"]) == 12
    assert all(len(r["cohorts"]) == 12 for r in result["checkpoint_diagnostics"])


@pytest.mark.parametrize("fault", ["missing_checkpoint", "duplicate_checkpoint", "omitted_row", "changed_denominator", "changed_metric"])
def test_incomplete_or_modified_evidence_is_rejected(fault: str) -> None:
    records = synthetic_records()
    if fault == "missing_checkpoint":
        records.pop()
    elif fault == "duplicate_checkpoint":
        records[-1] = copy.deepcopy(records[0])
    elif fault == "omitted_row":
        records[0]["rows"].pop()
    elif fault == "changed_denominator":
        records[0]["rows"][0]["reference_text"] += "추가"
    else:
        records[0]["metrics"]["cohorts"]["dev/full_sentence/normal"]["CER"] = .123
    with pytest.raises(ValueError):
        aggregate.aggregate_records(records)


def test_output_diagnostics_include_replacement_and_repetition_fractions() -> None:
    ev = aggregate.evaluator
    prediction = "아아아아아\ufffd"
    rows = [{"split": "dev", "kind": "full_sentence", "condition": "normal", "raw_text": prediction,
             "status": "success", "metrics": ev.contract.score_prediction("참조", prediction)}]
    result = aggregate.output_diagnostics(rows)["dev/full_sentence/normal"]
    assert result["replacement_character_fraction"] == 1 / 6
    assert result["repetition_ge5_clip_fraction"] == 1
    assert result["top10_nonspace_characters"][0] == {"character": "아", "count": 5, "fraction": 5 / 6}
