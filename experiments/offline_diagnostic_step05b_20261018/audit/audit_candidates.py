"""Independent source/candidate membership, CPU CTC, ranking and group audit."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata

import numpy as np
import torch

from independent_ctc import collapse, minimum_frames, torch_nll
from verify_preservation import binding, check, read, sha

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S3 = ROOT / "experiments/offline_dataset_step03_20261013"
S5 = ROOT / "experiments/offline_comparison_step05_20261013"
TIE = 1e-9


def encode(text):
    values = []
    for char in unicodedata.normalize("NFD", text):
        code = ord(char)
        if 0x1100 <= code <= 0x1112:
            values.append(code - 0x1100 + 1)
        elif 0x1161 <= code <= 0x1175:
            values.append(code - 0x1161 + 20)
        elif 0x11A8 <= code <= 0x11C2:
            values.append(code - 0x11A8 + 41)
        elif char == " ":
            values.append(68)
        else:
            raise AssertionError((text, char, code))
    return values


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) < 1e-8 + 1e-10 * abs(b), (a, b)


def ranking(candidates, field):
    feasible = [c for c in candidates if c[field] is not None]
    reference = next(c for c in candidates if c["is_reference"])
    value = reference[field]
    assert value is not None
    best = min(c[field] for c in feasible)
    return {"reference_rank": 1 + sum(c[field] < value - TIE for c in feasible),
            "reference_tie_count": sum(abs(c[field] - value) <= TIE for c in feasible),
            "top_candidate_ids": [c["candidate_id"] for c in feasible if abs(c[field] - best) <= TIE],
            "reference_minus_best": value - best,
            "finite_candidates": len(feasible), "total_candidates": len(candidates)}


def inspect():
    torch.set_num_threads(1)
    result_path = BASE / "results/ctc_candidates.json"
    result = read(result_path)
    plan_path = BASE / "diagnostic_plan.json"
    assert sha(plan_path) == "ed0bc7ee9f89cc132538699eb82359d5980ee4fc90256e0908e85cf0f5aac8dc"
    plan = read(plan_path)
    check(result["plan"])
    assert result["plan"] == binding(plan_path)
    for key in ["code", "tests", "source_manifest"]:
        check(plan[key])
    assert plan["tie_tolerance"] == TIE
    assert read(HERE / "ctc_calibration.json")["status"] == "passed"
    handoff = read(S3 / "results/cache_handoff.json")
    records = {r["clip_id"]: r for r in handoff["records"]}
    assert len(plan["records"]) == len(records) == 30
    plan_records = {r["clip_id"]: r for r in plan["records"]}
    assert set(records) == set(plan_records)
    expected_pool = {kind: defaultdict(list) for kind in ["phrase", "full_sentence"]}
    for clip_id, record in records.items():
        declared = plan_records[clip_id]
        for key in ["clip_id", "parent_id", "kind", "split", "T", "text"]:
            assert record[key] == declared[key]
        assert declared["labels"] == encode(record["text"])
        assert declared["speaker_id"] == re.search(r"_C(\d+)_", clip_id).group(0)[1:-1]
        if record["split"] == "train":
            expected_pool[record["kind"]][tuple(declared["labels"])].append(declared)
    pool_lookup = {}
    for kind, pool in plan["candidate_pools"].items():
        assert len(pool) == len(expected_pool[kind]) == 11
        assert len({c["candidate_id"] for c in pool}) == 11
        assert {tuple(c["labels"]) for c in pool} == set(expected_pool[kind])
        for candidate in pool:
            sources = expected_pool[kind][tuple(candidate["labels"])]
            assert candidate["labels"] == encode(candidate["text"])
            assert set(candidate["source_clip_ids"]) == {s["clip_id"] for s in sources}
            assert set(candidate["speakers"]) == {s["speaker_id"] for s in sources}
            assert {s["text"] for s in sources} == {candidate["text"]}
        pool_lookup[kind] = {tuple(c["labels"]): c for c in pool}
    evaluation_rows = {}
    for source in plan["evaluations"]:
        check(source)
        ev = read(source["path"])
        for row in ev["rows"]:
            if row["condition"] != "normal":
                continue
            identity = row["identity"]
            key = (identity["seed"], identity["arm"], identity["completed_update"], row["clip_id"])
            assert key not in evaluation_rows
            evaluation_rows[key] = (row, source)
    expected_keys = {(seed, arm, step, clip_id) for seed in [42, 43] for arm in ["A_sentence", "B_curriculum"]
                     for step in [420, 840, 1260] for clip_id in records}
    assert set(evaluation_rows) == expected_keys
    assert len(result["rows"]) == 360
    seen, checks, groups = set(), [], defaultdict(list)
    pairs = greedy_pairs = impossible = frames = 0
    max_difference = 0.0
    for row in result["rows"]:
        key = (row["seed"], row["arm"], row["update"], row["clip_id"])
        assert key in expected_keys and key not in seen
        seen.add(key)
        ev, ev_binding = evaluation_rows[key]
        record = plan_records[row["clip_id"]]
        assert row["source_evaluation"] == ev_binding
        for field in ["parent_id", "kind", "split", "T"]:
            assert row[field] == ev[field] == record[field]
        assert row["speaker_id"] == record["speaker_id"]
        assert row["reference_text"] == ev["reference_text"] == record["text"]
        assert row["greedy_text"] == ev["raw_text"] and row["saved_metrics"] == ev["metrics"]
        assert row["greedy_exact"] == ev["metrics"]["exact"]
        assert ev["reference_label_ids"] == record["labels"]
        assert row["reference_label_length"] == len(record["labels"])
        assert row["reference_ctc_slack"] == row["T"] - minimum_frames(record["labels"]) >= 0
        assert row["posterior"] == ev["posterior"]
        check(row["posterior"])
        with np.load(row["posterior"]["path"], allow_pickle=False) as saved:
            matrix = saved["log_probs"].astype(np.float64)
            assert matrix.shape == (row["T"], 69) and np.isfinite(matrix).all()
            argmax = matrix.argmax(axis=1).tolist()
            greedy_ids = list(collapse(argmax))
            assert argmax == ev["raw_argmax_ids"] == saved["raw_argmax_ids"].tolist()
            assert greedy_ids == ev["collapsed_ids"] == saved["collapsed_ids"].tolist() == row["greedy"]["labels"]
        sum_error = float(np.abs(np.exp(matrix).sum(axis=1) - 1.0).max())
        close(row["log_probability_sum_max_error"], sum_error)
        assert sum_error < 1e-5
        frames += row["T"]
        expected_labels = set(pool_lookup[row["kind"]]) | {tuple(record["labels"])}
        assert len(row["candidates"]) == len(expected_labels)
        assert {tuple(c["labels"]) for c in row["candidates"]} == expected_labels
        assert sum(c["is_reference"] for c in row["candidates"]) == 1
        audited_candidates, candidate_checks = [], []
        for candidate in row["candidates"]:
            labels = tuple(candidate["labels"])
            assert candidate["labels"] == encode(candidate["text"])
            assert candidate["is_reference"] == (labels == tuple(record["labels"]))
            in_pool = labels in pool_lookup[row["kind"]]
            assert candidate["in_training_pool"] == in_pool
            if in_pool:
                expected = pool_lookup[row["kind"]][labels]
                for field in ["candidate_id", "labels", "text", "source_clip_ids", "speakers"]:
                    assert candidate[field] == expected[field]
            else:
                assert candidate["candidate_id"] == "reference"
                assert candidate["source_clip_ids"] == [row["clip_id"]]
                assert candidate["speakers"] == [row["speaker_id"]]
            audited, detail = audit_score(matrix, candidate)
            audited_candidates.append({**candidate, **audited})
            candidate_checks.append({"candidate_id": candidate["candidate_id"], **detail})
            pairs += 1
            impossible += not audited["feasible"]
            max_difference = max(max_difference, detail["absolute_difference"] or 0)
        greedy_score, greedy_detail = audit_score(matrix, row["greedy"])
        assert greedy_score["feasible"]
        greedy_pairs += 1
        max_difference = max(max_difference, greedy_detail["absolute_difference"])
        reference = next(c for c in audited_candidates if c["is_reference"])
        close(row["reference_nll"], reference["nll"])
        delta = reference["nll"] - greedy_score["nll"]
        close(row["reference_minus_greedy_nll"], delta)
        ranks = {}
        for field, output_key in [("nll", "total_score_ranking"), ("nll_per_token", "per_token_ranking")]:
            value = ranking(audited_candidates, field)
            for rank_field, rank_value in value.items():
                if rank_field == "reference_minus_best":
                    close(row[output_key][rank_field], rank_value)
                else:
                    assert row[output_key][rank_field] == rank_value
            ranks[output_key] = value
        group_key = (row["seed"], row["arm"], row["update"], row["split"], row["kind"])
        groups[group_key].append({"greedy_exact": row["greedy_exact"], "delta": delta,
                                  "candidate_count": len(audited_candidates),
                                  "impossible": sum(not c["feasible"] for c in audited_candidates), **ranks})
        checks.append({"seed": key[0], "arm": key[1], "update": key[2], "clip_id": key[3],
                       "candidate_checks": candidate_checks, "greedy_check": greedy_detail})
    assert seen == expected_keys
    assert (pairs, greedy_pairs, impossible, frames) == (4032, 360, 528, 24276)
    assert result["posterior_rows"] == 360 and result["posterior_frames"] == frames
    assert result["candidate_pairs"] == pairs and result["greedy_pairs"] == greedy_pairs
    assert result["ctc_computations"] == pairs + greedy_pairs
    assert result["impossible_candidate_pairs"] == impossible
    group_results = []
    assert len(result["groups"]) == len(groups) == 48
    for declared in result["groups"]:
        key = tuple(declared[k] for k in ["seed", "arm", "update", "split", "kind"])
        values = groups.pop(key)
        expected = {
            "count": len(values), "greedy_exact": sum(x["greedy_exact"] for x in values),
            "reference_top_total": sum(x["total_score_ranking"]["reference_rank"] == 1 for x in values),
            "reference_top_per_token": sum(x["per_token_ranking"]["reference_rank"] == 1 for x in values),
            "reference_total_better_than_greedy": sum(x["delta"] < -TIE for x in values),
            "reference_total_tied_with_greedy": sum(abs(x["delta"]) <= TIE for x in values),
            "reference_total_worse_than_greedy": sum(x["delta"] > TIE for x in values),
            "median_reference_minus_greedy_nll": statistics.median(x["delta"] for x in values),
            "impossible_candidates": sum(x["impossible"] for x in values),
            "total_candidate_pairs": sum(x["candidate_count"] for x in values),
        }
        for field, value in expected.items():
            if field.startswith("median"):
                close(declared[field], value)
            else:
                assert declared[field] == value, (key, field, declared[field], value)
        group_results.append({**dict(zip(["seed", "arm", "update", "split", "kind"], key)), **expected})
    assert not groups
    return {"schema": "offline-stage5b-independent-candidate-audit-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "source": binding(result_path),
            "plan": binding(plan_path), "calibration": binding(HERE / "ctc_calibration.json"),
            "posterior_rows": len(checks), "posterior_frames": frames,
            "candidate_pairs": pairs, "greedy_pairs": greedy_pairs, "CTC_computations": pairs + greedy_pairs,
            "impossible_candidate_pairs": impossible, "maximum_absolute_NLL_difference": max_difference,
            "verified_groups": group_results, "checks": checks,
            "candidate_pool_membership": "independently matched same-kind train labels; Unicode NFD encoding checked",
            "source_coverage": "all 4 runs x 3 checkpoints x 30 normal clips exactly once",
            "backend": "torch.nn.functional.ctc_loss CPU float64; saved float32 logits promoted without renormalizing",
            "producer_imports": 0, "neural_forward_calls": 0, "optimizer_updates": 0,
            "checkpoint_tensor_loads": 0, "GPU_calls": 0, "source_media_reads": 0}


def audit_score(matrix, candidate):
    labels = candidate["labels"]
    actual = torch_nll(matrix, labels)
    feasible = matrix.shape[0] >= minimum_frames(labels)
    assert candidate["label_length"] == len(labels)
    assert candidate["minimum_frames"] == minimum_frames(labels)
    assert candidate["feasible"] == feasible == math.isfinite(actual)
    if feasible:
        assert candidate["status"] == "finite"
        close(candidate["nll"], actual)
        per_token = actual / len(labels) if labels else None
        if per_token is None:
            assert candidate["nll_per_token"] is None
        else:
            close(candidate["nll_per_token"], per_token)
        difference = abs(candidate["nll"] - actual)
    else:
        assert candidate["status"] == "impossible_ctc_length"
        assert candidate["nll"] is None and candidate["nll_per_token"] is None and actual == math.inf
        actual = per_token = difference = None
    return ({"feasible": feasible, "nll": actual, "nll_per_token": per_token},
            {"torch_float64_NLL": actual, "producer_float64_NLL": candidate["nll"],
             "feasible": feasible, "absolute_difference": difference})


if __name__ == "__main__":
    output = HERE / "candidate_audit.json"
    assert not output.exists()
    value = inspect()
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: value[k] for k in ["status", "posterior_rows", "posterior_frames", "CTC_computations", "impossible_candidate_pairs", "maximum_absolute_NLL_difference"]}))
