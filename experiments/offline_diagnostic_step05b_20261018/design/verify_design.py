"""Read-only arithmetic and source verification of a proposed future design."""
import hashlib
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parent
root = base.parents[2]
plan_path = base / "proposed_phrase_only_design.json"
plan = json.loads(plan_path.read_text(encoding="utf-8"))
assert plan["status"] == "design_only_not_executable_or_authorized_training"
records = plan["train_records"]
lookup = {r["parent_id"]: r for r in records}
assert len(lookup) == 12 and len({r["text"] for r in records}) == 11
assert all(not r["training_ready"] and r["provisional"] for r in records)
assert all(r["T"] >= r["ctc_min_frames"] for r in records)
assert (min(r["frames_above_ctc_minimum"] for r in records), max(r["frames_above_ctc_minimum"] for r in records)) == (6, 33)
verified = []
for item in plan["sources"] + plan["schedule_bindings"]:
    path = Path(item["path"])
    assert path.stat().st_size == item["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    verified.append(item)
accounting = []
for item in plan["schedule_bindings"]:
    schedule = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
    old = json.loads(Path(schedule["source_schedule"]["path"]).read_text(encoding="utf-8"))["updates"]
    assert len(schedule["updates"]) == 1680
    visits = Counter()
    true_frames = dense_frames = tokens = 0
    for i, row in enumerate(schedule["updates"], 1):
        expected_old = old[i-1] if i <= 1260 else old[i-1261]
        assert row["update"] == i and row["parent_ids"] == expected_old["parent_ids"]
        selected = [lookup[p] for p in row["parent_ids"]]
        lengths = [r["T"] for r in selected]
        assert row["clip_ids"] == [r["clip_id"] for r in selected]
        assert row["lengths"] == lengths and row["kind"] == "phrase"
        assert row["lr"] == (.001 if i <= 840 else .0003 if i <= 1050 else .0001)
        assert row["true_frames"] == sum(lengths) and row["dense_frames"] == max(lengths) * 2
        if i <= 420:
            assert row["clip_ids"] == old[i-1]["arms"]["B_curriculum"]["clip_ids"]
            assert row["lr"] == old[i-1]["lr"]
        visits.update(row["parent_ids"])
        true_frames += sum(lengths)
        dense_frames += max(lengths) * 2
        tokens += sum(r["label_length"] for r in selected)
        if i % 420 == 0:
            assert set(visits.values()) == {i // 6}
    assert true_frames == 134680 and tokens == 76160
    assert dense_frames == schedule["accounting"]["dense_frames"]
    accounting.append({"seed": schedule["seed"], "updates": 1680, "true_frames": true_frames, "dense_frames": dense_frames, "target_tokens": tokens})
assert sum(r["dense_frames"] for r in accounting) == 315946
assert plan["evaluation"]["total_planned_forwards"] == 12 * 4 * 2 + 12 * 2 * 2 == 144
assert plan["evaluation"]["dev_clip_forward_calls"] == plan["evaluation"]["full_sentence_forward_calls"] == 0
assert plan["selection_policy"]["checkpoint_updates"] == [420, 840, 1260, 1680]
assert plan["planned_accounting"]["main_updates"] == 3360
assert plan["planned_accounting"]["max_total_optimizer_updates_before_any_replan"] == 3360 + 8
assert plan["planned_accounting"]["max_total_forward_calls_including_preflight"] == 3360 + 144 + 8
assert all(plan["actual_work_here"][k] == 0 for k in ("new_neural_calls", "new_optimizer_updates", "checkpoint_tensor_loads", "NPZ_pixel_loads", "raw_source_reads", "downloads", "service_changes"))
output = {"status": "passed_metadata_and_proposed_schedule_checks_no_training",
          "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
          "source_and_schedule_bindings_verified": len(verified), "verified_bindings": verified,
          "planned_accounting_recalculated": accounting,
          "new_neural_calls": 0, "new_optimizer_updates": 0,
          "meaning": "Arithmetic/schema/source checks only; not a future runtime or model validation."}
with (base / "design_verification.json").open("x", encoding="utf-8") as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps({k: v for k, v in output.items() if k != "verified_bindings"}))
