"""Independent metadata, schedule and preservation checks; no producer imports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S3 = ROOT / "experiments/offline_dataset_step03_20261013"
S4 = ROOT / "experiments/offline_training_step04_20261013"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path: Path) -> dict:
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def check_binding(row: dict) -> None:
    path = Path(row["path"])
    if row.get("exists") is False:
        assert not path.exists(), f"Unexpected creation: {path}"
        return
    assert path.is_file(), str(path)
    assert digest(path) == row["sha256"], str(path)
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"], str(path)


def inspect() -> dict:
    manifests = []
    for stage, expected in [(S3, 114), (S4, 101)]:
        path = stage / "results/completion_manifest.json"
        entries = read(path)["artifacts"]
        assert len(entries) == expected
        assert len({r["path"] for r in entries}) == expected
        for row in entries:
            check_binding(row)
        manifests.append({**binding(path), "verified_artifacts": expected})
    scope_path = S3 / "results/scope_snapshot.json"
    scope = read(scope_path)["artifacts"]
    assert len(scope) == 10
    for row in scope:
        check_binding(row)
    handoff_path = S3 / "results/cache_handoff.json"
    records = read(handoff_path)["records"]
    assert len(records) == 30
    lookup = {r["clip_id"]: r for r in records}
    assert len(lookup) == 30
    assert Counter(r["split"] for r in records) == {"train": 24, "dev": 6}
    assert all(r["provisional"] and not r["training_ready"] for r in records)
    assert sum(r["boundary_followup_flag"] for r in records) == 2
    parents = sorted({r["parent_id"] for r in records if r["split"] == "train"})
    assert len(parents) == 12
    assert not set(parents) & {r["parent_id"] for r in records if r["split"] == "dev"}
    all_frames = all_dense = all_updates = 0
    max_true = max_dense = 0
    max_pairs = []
    counts = []
    for seed in [42, 43]:
        path = S4 / f"protocol/schedule_seed{seed}.json"
        schedule = read(path)
        assert schedule["seed"] == seed and len(schedule["updates"]) == 1260
        rng = random.Random(seed)
        expected_pairs = []
        for _ in range(210):
            order = parents.copy()
            rng.shuffle(order)
            expected_pairs.extend(order[i:i + 2] for i in range(0, 12, 2))
        exposures = {arm: Counter() for arm in ["A_sentence", "B_curriculum"]}
        sums = {arm: Counter() for arm in exposures}
        for n, (row, pair) in enumerate(zip(schedule["updates"], expected_pairs), 1):
            assert row["update"] == n and row["round_index"] == (n - 1) // 6 + 1
            assert row["parent_ids"] == pair
            assert row["lr"] == (0.001 if n <= 840 else 0.0003 if n <= 1050 else 0.0001)
            for arm, entry in row["arms"].items():
                kind = "phrase" if arm == "B_curriculum" and n <= 420 else "full_sentence"
                expected_ids = [p + "__" + kind for p in pair]
                assert entry["kind"] == kind and entry["clip_ids"] == expected_ids
                chosen = [lookup[x] for x in expected_ids]
                assert all(r["split"] == "train" for r in chosen)
                lengths = [r["T"] for r in chosen]
                actual, dense = sum(lengths), 2 * max(lengths)
                assert entry["lengths"] == lengths
                assert entry["target_lengths"] == [r["label_length"] for r in chosen]
                assert entry["actual_frames"] == actual and entry["dense_processed_frames"] == dense
                assert entry["padding_frames"] == dense - actual
                all_updates += 1
                all_frames += actual
                all_dense += dense
                max_true, max_dense = max(max_true, actual), max(max_dense, dense)
                if actual == 316:
                    max_pairs.append({"seed": seed, "arm": arm, "update": n, "ids": expected_ids, "lengths": lengths})
                sums[arm].update({"updates": 1, "true_frames": actual, "dense_frames": dense})
                exposures[arm].update((p, kind) for p in pair)
        assert set(exposures["A_sentence"].values()) == {210}
        assert all(exposures["B_curriculum"][(p, "phrase")] == 70 and exposures["B_curriculum"][(p, "full_sentence")] == 140 for p in parents)
        counts.append({"seed": seed, "schedule": binding(path), "arms": {k: dict(v) for k, v in sums.items()}})
    assert (all_updates, all_frames, all_dense, max_true, max_dense) == (5040, 911540, 1085994, 316, 358)
    evaluation_path = S4 / "protocol/evaluation_plan.json"
    evaluation = read(evaluation_path)
    assert evaluation["checkpoints"] == [420, 840, 1260]
    assert evaluation["conditions"] == ["normal", "static", "shuffled"]
    assert len(evaluation["items"]) == 30
    assert {r["clip_id"] for r in evaluation["items"]} == set(lookup)
    for item in evaluation["items"]:
        cache = lookup[item["clip_id"]]
        for key in ["parent_id", "split", "kind", "T", "text"]:
            assert item[key] == cache[key]
        assert item["static_frame_index"] == item["T"] // 2
        assert sorted(item["shuffled_frame_indices"]) == list(range(item["T"]))
        expected_permutation = list(range(item["T"]))
        random.Random(item["shuffle_seed"]).shuffle(expected_permutation)
        assert item["shuffled_frame_indices"] == expected_permutation
        assert item["primary_at_final_normal_only"] == (item["split"] == "dev" and item["kind"] == "full_sentence")
    eval_frames = sum(r["T"] for r in evaluation["items"]) * 3 * 3 * 4
    assert eval_frames == 72828 and 30 * 3 * 3 * 4 == 1080
    future = read(S4 / "results/resource_budget.json")["future_main_comparison"]
    assert future["max_true_frames_per_batch"] == max_true and future["max_dense_frames_per_batch"] == max_dense
    return {
        "schema": "offline-stage5-independent-source-schedule-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "producer_imports": 0, "GPU_calls": 0, "optimizer_updates": 0,
        "frozen_manifests": manifests, "scope": {**binding(scope_path), "verified_artifacts": 10},
        "handoff": binding(handoff_path), "cache_records": 30, "train_clips": 24, "dev_clips": 6,
        "provisional_flags_preserved": True, "boundary_followup_clips": 2,
        "schedule_checks": counts, "main_updates": all_updates, "true_frames": all_frames,
        "dense_frames": all_dense, "padding_frames": all_dense - all_frames,
        "max_true_batch_frames": max_true, "max_dense_batch_frames": max_dense,
        "max_true_pairs_count": len(max_pairs), "first_max_true_pair": max_pairs[0],
        "evaluation": binding(evaluation_path), "evaluation_rows": 1080, "evaluation_frames": eval_frames,
        "checkpoint_count": 12, "main_training_executed_by_audit": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="source_preflight.json")
    args = parser.parse_args()
    output = (HERE / args.output).resolve()
    assert output.parent == HERE and not output.exists()
    result = inspect()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output), "old_frozen_artifacts": 215, "scope": 10, "schedule_updates": result["main_updates"]}))
