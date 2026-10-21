"""Aggregate all fixed phrase-only diagnostics, retaining every seed/checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stage5c_aggregate_evaluator", Path(__file__).with_name("evaluator.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def aggregate_summaries(paths: list[Path]) -> dict:
    ev.require(len(paths) == 8, "Exactly eight fixed checkpoint evaluations are required")
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    keys = [(r["identity"]["seed"], r["identity"]["completed_update"]) for r in records]
    ev.require(len(keys) == len(set(keys)) and set(keys) == {(s, u) for s in [42, 43] for u in ev.CHECKPOINTS},
               "Duplicate or omitted fixed checkpoint")
    records.sort(key=lambda r: (r["identity"]["seed"], r["identity"]["completed_update"]))
    source_plan = json.loads(ev.helper.PLAN_PATH.read_text(encoding="utf-8"))
    fixed_items = [r for r in source_plan["items"] if r["split"] == "train" and r["kind"] == "phrase"]
    texts = {r["clip_id"]: r["text"] for r in fixed_items}
    diagnostics = []
    for result in records:
        identity = result["identity"]
        count = 36 if identity["completed_update"] == 1680 else 12
        ev.require(result["count"] == count and len(result["rows"]) == count, "Checkpoint row count differs")
        ev.require(result["metrics"] == ev.summarize_rows(result["rows"]), "Checkpoint metrics differ from rows")
        ev.require(result["failures"] == 0 and result["status"] == "completed" and result["state_isolation_passed"], "Incomplete/failed checkpoint")
        expected = ev.CONDITIONS[str(identity["completed_update"])]
        ev.require(sorted({r["condition"] for r in result["rows"]}) == sorted(expected), "Unexpected condition cohort")
        ev.require({(r["clip_id"], r["condition"]) for r in result["rows"]} ==
                   {(r["clip_id"], c) for r in fixed_items for c in expected}, "Unexpected clip cohort")
        ev.require(all(r["identity"] == identity and r["reference_text"] == texts[r["clip_id"]] for r in result["rows"]),
                   "Row identity/reference differs from fixed source")
        for condition in expected:
            score = result["metrics"]["cohorts"][f"train/phrase/{condition}"]
            ev.require(score["count"] == 12 and score["failure_count"] == 0, "Invalid per-condition cohort")
            diagnostics.append({"seed": identity["seed"], "completed_update": identity["completed_update"],
                                "condition": condition, "metrics": score,
                                "predictions": [{"clip_id": r["clip_id"], "reference_text": r["reference_text"],
                                                 "prediction_text": r["raw_text"], "metrics": r["metrics"]}
                                                for r in result["rows"] if r["condition"] == condition]})
    gates = []
    for seed in [42, 43]:
        scores = [r for r in diagnostics if r["seed"] == seed and r["completed_update"] in [1260, 1680] and r["condition"] == "normal"]
        ev.require(len(scores) == 2, "Fit gate checkpoint missing")
        gates.append({"seed": seed, "passed": all(r["metrics"]["exact_count"] == 12 and r["metrics"]["CER"] == 0 and
                                                   r["metrics"]["failure_count"] == 0 for r in scores),
                      "checkpoints": [{"completed_update": r["completed_update"], "metrics": r["metrics"]} for r in scores]})
    return {"schema": "offline-stage5c-evaluation-aggregate-v1", "status": "completed", "sources": [ev.ref(Path(p)) for p in paths],
            "checkpoint_count": len(records), "evaluation_count": sum(r["count"] for r in records),
            "forward_calls": sum(r["forward_calls"] for r in records),
            "frames": sum(r["attempted_forward_frames"] for r in records),
            "failures": sum(r["failures"] for r in records), "diagnostics": diagnostics,
            "fit_gate": {"per_seed": gates, "passed": all(g["passed"] for g in gates),
                         "rule": "Each seed must have12/12 exact and NFC CER0 at BOTH1260 and1680.",
                         "technical_arbitrary_train_reproduction_gate_only": True},
            "dev_evaluated": False, "generalization_measured": False, "automatic_sentence_expansion": False}
