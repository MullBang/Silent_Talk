"""Aggregate the fixed 12 checkpoint evaluations without selecting best weights."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

_spec = importlib.util.spec_from_file_location("offline_stage5_aggregation_evaluator", Path(__file__).with_name("evaluator.py"))
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)


def output_diagnostics(rows: Sequence[dict]) -> dict:
    """Describe blank/repetitive/invalid output without removing any characters."""
    result = {}
    for key in sorted({(r["split"], r["kind"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["kind"], r["condition"]) == key]
        predictions = [evaluator.contract.metric_text(r["raw_text"]) for r in group]
        characters = sum(len(p) for p in predictions)
        replacements = sum(p.count("\ufffd") for p in predictions)
        nonspace = Counter(c for p in predictions for c in p if not c.isspace())
        nonspace_count = sum(nonspace.values())
        repetitive = sum(r["metrics"]["repetition_ge5"] for r in group)
        result["/".join(key)] = {"count": len(group), "empty_prediction_count": sum(not p for p in predictions),
            "failed_count": sum(r["status"] == "failed" for r in group),
            "repetition_ge5_count": repetitive, "repetition_ge5_clip_fraction": repetitive / len(group),
            "replacement_characters": replacements, "prediction_characters_including_internal_spaces": characters,
            "replacement_character_fraction": replacements / characters if characters else 0.0,
            "replacement_fraction_empty_denominator": characters == 0,
            "predicted_nonspace_characters": nonspace_count,
            "top10_nonspace_characters": [{"character": c, "count": n, "fraction": n / nonspace_count}
                for c, n in sorted(nonspace.items(), key=lambda pair: (-pair[1], pair[0]))[:10]]}
    return result


def aggregate_records(records: Sequence[dict]) -> dict:
    """Recheck complete cohorts and compute paired final-state differences."""
    expected = {(seed, arm, step) for seed in (42, 43) for arm in ("A_sentence", "B_curriculum") for step in (420, 840, 1260)}
    identities = [(r["identity"]["seed"], r["identity"]["arm"], r["identity"]["completed_update"]) for r in records]
    evaluator.require(len(records) == 12 and len(set(identities)) == 12 and set(identities) == expected,
                      "Need all 12 distinct frozen checkpoints; do not select best checkpoints")
    plan = json.loads(evaluator.PLAN_PATH.read_text(encoding="utf-8"))
    expected_rows = {(item["clip_id"], c): item for item in plan["items"] for c in plan["conditions"]}
    by_key = {}
    for record, identity in zip(records, identities):
        evaluator.validate_plan(plan, record["identity"])
        evaluator.require(record["evaluation_plan_sha256"] == evaluator.PLAN_SHA256
                          and record["evaluation_contract_sha256"] == evaluator.CONTRACT_SHA256, "Metric/source contract changed")
        rows = record["rows"]
        evaluator.require(len(rows) == 90 and {(r["clip_id"], r["condition"]) for r in rows} == set(expected_rows),
                          "Fixed evaluation rows omitted or duplicated")
        for row in rows:
            item = expected_rows[(row["clip_id"], row["condition"])]
            evaluator.require(row["identity"] == record["identity"] and row["reference_text"] == item["text"]
                              and all(row[k] == item[k] for k in ("parent_id", "split", "kind", "T")), "Row identity/reference changed")
            evaluator.require(row["status"] in ("success", "failed"), "Unknown inference outcome")
            rescored = evaluator.contract.score_prediction(item["text"], row["raw_text"], failed=row["status"] == "failed")
            evaluator.require(rescored == row["metrics"], "Stored row metrics disagree with fixed definitions")
        rescored_corpus = evaluator.summarize_rows(rows)
        evaluator.require(rescored_corpus == record["metrics"], "Stored corpus metrics disagree with row totals")
        evaluator.require(record["count"] == 90 and record["failures"] == rescored_corpus["all_rows"]["failure_count"],
                          "Stored evaluation/failure counts disagree")
        by_key[identity] = record
    final = []
    paired = []
    for seed in (42, 43):
        for arm in ("A_sentence", "B_curriculum"):
            record = by_key[(seed, arm, 1260)]
            cohorts = record["metrics"]["cohorts"]
            primary = cohorts["dev/full_sentence/normal"]
            evaluator.require(primary["count"] == 3 and primary["reference_characters"] == 74
                              and primary["reference_words"] == 23, "Primary dev denominator changed")
            final.append({"seed": seed, "arm": arm, "completed_update": 1260,
                          "primary_dev_sentence_normal": primary, "cohorts": cohorts,
                          "output_diagnostics": output_diagnostics(record["rows"]),
                          "primary_static_minus_normal_CER_pp": 100 * (cohorts["dev/full_sentence/static"]["CER"] - primary["CER"]),
                          "primary_shuffled_minus_normal_CER_pp": 100 * (cohorts["dev/full_sentence/shuffled"]["CER"] - primary["CER"])})
        a = by_key[(seed, "A_sentence", 1260)]["metrics"]["cohorts"]["dev/full_sentence/normal"]
        b = by_key[(seed, "B_curriculum", 1260)]["metrics"]["cohorts"]["dev/full_sentence/normal"]
        paired.append({"seed": seed, "A_CER": a["CER"], "B_CER": b["CER"],
                       "B_minus_A_CER_pp": 100 * (b["CER"] - a["CER"]),
                       "A_WER": a["WER"], "B_WER": b["WER"], "B_minus_A_WER_pp": 100 * (b["WER"] - a["WER"]),
                       "A_exact": a["exact_count"], "B_exact": b["exact_count"],
                       "A_failures": a["failure_count"], "B_failures": b["failure_count"]})
    total_failures = sum(r["failures"] for r in records)
    return {"schema": "offline-stage5-fixed-comparison-aggregate-v1",
        "status": "completed" if total_failures == 0 else "completed_with_explicit_failures",
        "evaluations": 1080, "checkpoints": 12, "failures": total_failures,
        "final_models": final, "paired_primary": paired,
        "mean_primary": {"A_CER": sum(p["A_CER"] for p in paired) / 2,
                         "B_CER": sum(p["B_CER"] for p in paired) / 2,
                         "B_minus_A_CER_pp": sum(p["B_minus_A_CER_pp"] for p in paired) / 2,
                         "B_lower_CER_both_seeds": all(p["B_CER"] < p["A_CER"] for p in paired)},
        "checkpoint_diagnostics": [{"seed": seed, "arm": arm, "completed_update": step,
            "cohorts": by_key[(seed, arm, step)]["metrics"]["cohorts"],
            "output_diagnostics": output_diagnostics(by_key[(seed, arm, step)]["rows"])} for seed, arm, step in sorted(expected)],
        "primary_checkpoint": 1260, "best_checkpoint_selection": False,
        "limits": ["Only 12 training parents and 3 internal development parents; no unseen-speaker or webcam test.",
                   "A and B have equal updates and parent visits, but B uses 20.04% fewer true training frames.",
                   "Content duration and curriculum phase both differ; this does not isolate a pure order effect.",
                   "Phrase/sentence pairs share pixels and are not independent examples.",
                   "Failure rows remain full deletion errors; the pooled 1080-row total is not primary accuracy.",
                   "Two seeds are reported individually and do not establish population-level significance."]}


def aggregate_evaluations(summary_paths: Sequence[Path]) -> dict:
    """Read exact saved checkpoint summaries and return their bound aggregation."""
    paths = [Path(p).resolve() for p in summary_paths]
    result = aggregate_records([json.loads(p.read_text(encoding="utf-8")) for p in paths])
    result["sources"] = [{"path": str(p), "sha256": evaluator.file_sha256(p), "bytes": p.stat().st_size} for p in paths]
    return result
