"""Fixed two-seed exploratory dev-phrase probe; no checkpoint selection."""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location("stage5d_aggregate_evaluator", Path(__file__).with_name("evaluator.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def aggregate_summaries(paths: list[Path]) -> dict:
    ev.require(len(paths) == 2, "Exactly two frozen final checkpoints are required")
    results = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    ev.require(sorted((r["identity"]["seed"], r["identity"]["completed_update"]) for r in results) == [(42, 1680), (43, 1680)],
               "Missing, duplicated or changed fixed checkpoint")
    old = json.loads(ev.helper.PLAN_PATH.read_text(encoding="utf-8"))
    items = [r for r in old["items"] if r["split"] == "dev" and r["kind"] == "phrase"]
    plan = {"items": items, "checkpoints": ev.CHECKPOINTS, "conditions_by_update": ev.CONDITIONS,
            "resource_caps": {"cuda_reserved_bytes": ev.CUDA_RESERVED_BUDGET_BYTES, "seconds_per_model": ev.SECONDS_PER_MODEL}}
    texts = {r["clip_id"]: r["text"] for r in items}
    diagnostics = []
    for result in sorted(results, key=lambda r: r["identity"]["seed"]):
        identity = result["identity"]
        conditions = ev.validate_plan(plan, identity, verify_sources=False)
        ev.require(result["count"] == 9 and len(result["rows"]) == 9, "Wrong per-seed row count")
        ev.require(result["metrics"] == ev.summarize_rows(result["rows"]), "Metrics differ from saved rows")
        ev.require(result["status"] == "completed" and result["failures"] == 0 and result["state_isolation_passed"], "Incomplete probe")
        ev.require({(r["clip_id"], r["condition"]) for r in result["rows"]} == ev.expected_pairs(plan, conditions), "Unexpected clip cohort")
        ev.require(all(r["identity"] == identity and r["reference_text"] == texts[r["clip_id"]] for r in result["rows"]), "Wrong row identity/reference")
        for condition in conditions:
            score = result["metrics"]["cohorts"][f"dev/phrase/{condition}"]
            ev.require(score["count"] == 3 and score["failure_count"] == 0, "Invalid per-condition cohort")
            diagnostics.append({"seed": identity["seed"], "completed_update": 1680, "condition": condition, "metrics": score,
                "predictions": [{"clip_id": r["clip_id"], "reference_text": r["reference_text"], "prediction_text": r["raw_text"], "metrics": r["metrics"]}
                                for r in result["rows"] if r["condition"] == condition]})
    primary = [r for r in diagnostics if r["condition"] == "normal"]
    return {"schema": "offline-stage5d-fixed-probe-aggregate-v1", "status": "completed", "sources": [ev.ref(Path(p)) for p in paths],
            "checkpoint_count": 2, "evaluation_count": sum(r["count"] for r in results),
            "forward_calls": sum(r["forward_calls"] for r in results), "frames": sum(r["attempted_forward_frames"] for r in results),
            "failures": 0, "unique_dev_phrase_count": 3, "diagnostics": diagnostics,
            "primary": {"per_seed": primary, "mean_CER": sum(r["metrics"]["CER"] for r in primary) / 2,
                        "mean_WER": sum(r["metrics"]["WER"] for r in primary) / 2,
                        "exact_observations": sum(r["metrics"]["exact_count"] for r in primary), "observations": 6,
                        "independent_utterances": 3},
            "optimizer_updates": 0, "exploratory_reused_development_probe": True, "generalization_established": False,
            "new_speaker_or_webcam_evaluation": False, "automatic_sentence_expansion": False,
            "prior5c_strict_fit_gate_remains_failed": True}
