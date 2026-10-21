"""Bounded full-state continuation of frozen 5i B26, with unchanged inputs.

The inherited 1,820 updates are never repeated or counted as newly executed.
Only separately authorized epochs 141--280 run; source approval is unchanged.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD_BASE = ROOT / "experiments/offline_diversity_motion_step05i_20261019"
FROZEN = OLD_BASE / "runner.py"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require(ok, message):
    """Stop on an invalid scope or state instead of silently repairing it."""
    if not ok:
        raise ValueError(message)


def sha(path):
    """Read one explicit artifact with bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    """Bind exact bytes of a local artifact."""
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def bound(binding):
    """Reject any source byte or length change before reading its contents."""
    path = Path(binding["path"]).resolve()
    require(sha(path) == binding["sha256"], f"Bound bytes changed: {path}")
    if "bytes" in binding:
        require(path.stat().st_size == binding["bytes"], "Bound byte count changed")
    return path


def read(path):
    """Parse explicit UTF8 JSON without guessed defaults."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    """Exclusively save evidence inside this new training branch."""
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), "Output outside 5j training")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


require(sha(FROZEN) == "e0b59644c84c899fd7246263829342ef7c55ae0e279d211eed0d2017ba76fced", "Frozen 5i runner changed")
spec = importlib.util.spec_from_file_location("stage5j_frozen_5i", FROZEN)
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)
old, prior, torch, np, contract = source.old, source.prior, source.torch, source.np, source.contract


def verify_plan(path):
    """Validate the continuation scope and frozen arithmetic/data contracts."""
    plan = read(path)
    require(plan["schema"] == "offline-stage5j-budget-extension-plan-v1", "Wrong 5j plan")
    require(plan["seeds"] == [42, 43] and plan["start_updates"] == 1820 and
            plan["additional_steps"] == 1820 and plan["final_updates"] == 3640,
            "Continuation budget changed")
    require(plan["evaluations"] == [2730, 3640] and plan["batch_size"] == 2, "Evaluation/batch budget changed")
    require(plan["limits"] == {"wall_seconds_per_run": 1200, "cuda_reserved_bytes": 6*1024**3}, "Resource limit changed")
    bindings = {bound(item) for item in plan["bindings"]}
    require({Path(__file__).resolve(), FROZEN.resolve(), source.PRIOR.resolve(), source.CONTRACT.resolve(),
             ROOT/"backend/models/tokenizer.py"} <= bindings, "Missing executable bindings")
    bound(plan["policy"])
    require(bound(plan["old_plan"]) == OLD_BASE/"plan.json", "Wrong source plan")
    old_plan = source.verify_plan(bound(plan["old_plan"]))
    require(plan["runtime"] == old_plan["runtime"], "Runtime contract differs from source")
    require(bound(plan["old_data_plan"]) == bound(old_plan["data_plan"]), "Changed continuation data")
    require(set(plan["source_checkpoints"]) == {"42", "43"} and set(plan["schedules"]) == {"42", "43"}, "Missing seeds")
    for seed in plan["seeds"]:
        require(bound(plan["source_checkpoints"][str(seed)]) == OLD_BASE/f"results/B26_seed{seed}/update_1820.pt", "Wrong resume source")
    return plan


def expected_continuation(data, seed):
    """Reproduce local Random(seed) sorted26 permutations, without model RNG."""
    ids = sorted(data.arms["B26"])
    rng = random.Random(seed)
    all_tasks = []
    for epoch in range(1, 281):
        order = list(ids)
        rng.shuffle(order)
        for start in range(0, 26, 2):
            pair = order[start:start+2]
            update = len(all_tasks)+1
            lengths = [data.rows[s]["T"] for s in pair]
            all_tasks.append({"update": update, "epoch": epoch, "clip_ids": pair,
                              "lengths": lengths, "lr": .001 if update <= 910 else .0003 if update <= 1365 else .0001,
                              "true_frames": sum(lengths), "dense_frames": 2*max(lengths)})
    return all_tasks


def schedule(plan, data, seed):
    """Require the exact saved prefix and continuation epochs 141--280."""
    generated = expected_continuation(data, seed)
    old_plan = read(bound(plan["old_plan"]))
    previous = source.schedule(old_plan, data, "B26", seed)
    require(generated[:1820] == previous, "Original 140 shuffled epochs were not reproduced")
    tasks = read(bound(plan["schedules"][str(seed)]))["updates"]
    require(tasks == generated[1820:], "Continuation tasks differ from exact subsequent epochs")
    require(len(tasks) == 1820 and all(t["lr"] == .0001 for t in tasks), "Wrong continuation budget/LR")
    return tasks


def load_checkpoint(binding):
    """Restricted CPU tensor load from exactly bound old/new experiment paths."""
    path = bound(binding)
    require(path.is_relative_to(BASE) or path.is_relative_to(OLD_BASE/"results"), "Checkpoint outside authorized experiments")
    return old.runner._unpack_safe(torch.load(path, map_location="cpu", weights_only=True))


def validate_state(state, hashes, updates):
    """Check nonempty Adam state and every counter, LR, state/RNG digest."""
    require(set(state) == {"model", "optimizer", "rng"} and old.hashes(state) == hashes, "Incomplete/corrupt model Adam RNG state")
    optimizer = state["optimizer"]
    expected_ids = [i for group in optimizer["param_groups"] for i in group["params"]]
    require(set(optimizer["state"]) == set(expected_ids) and expected_ids, "Missing Adam tensors")
    require(all(float(item["step"]) == updates for item in optimizer["state"].values()), "Adam update counters differ")
    require(all(g["lr"] == .0001 and tuple(g["betas"]) == (.9, .999) and g["eps"] == 1e-8 and
                g["weight_decay"] == 0 for g in optimizer["param_groups"]), "Inherited Adam parameters differ")


def load_origin(plan, data, seed):
    """Prove the source checkpoint is the exact completed B26 trajectory."""
    binding = plan["source_checkpoints"][str(seed)]
    payload = load_checkpoint(binding)
    old_plan = read(bound(plan["old_plan"]))
    tasks = source.schedule(old_plan, data, "B26", seed)
    expected_identity = {"plan_sha256": plan["old_plan"]["sha256"], "arm": "B26", "seed": seed,
                         "runtime": plan["runtime"], "schedule_sha256": old_plan["schedules"]["B26"][str(seed)]["sha256"]}
    require(payload["schema"] == "offline-stage5i-complete-checkpoint-v1" and payload["identity"] == expected_identity,
            "Wrong source trajectory identity")
    require(payload["completed_updates"] == len(payload["history"]) == 1820 and payload["next_task"] is None,
            "Source checkpoint is not its final fixed endpoint")
    for row, task in zip(payload["history"], tasks):
        require(row["update"] == task["update"] and row["clip_ids"] == task["clip_ids"] and
                row["lengths"] == task["lengths"] and row["learning_rate"] == task["lr"] and
                row["optimizer_updated"] is True and row["training_condition"] == "normal", "Source history differs")
    validate_state(payload["state"], payload["state_hashes"], 1820)
    result = read(OLD_BASE/f"results/B26_seed{seed}/result.json")
    require(result["status"] == "completed" and result["identity"] == expected_identity and result["physical_updates"] == 1820 and
            result["checkpoints"][-1] == binding, "Source completion binding differs")
    return payload


def checkpoint(path, payload):
    """Save full state once, then verify its restricted-load round trip."""
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), "Unsafe/existing output checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.partial")
    with temporary.open("xb") as stream:
        torch.save(old.runner._pack_safe(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    binding = ref(path)
    restored = load_checkpoint(binding)
    require(old.hashes(restored["state"]) == payload["state_hashes"], "New checkpoint round trip differs")
    return binding


def verify_origin_predictions(model, optimizer, data, seed):
    """Reproduce four saved normal posteriors after restoring each 5i endpoint."""
    # Three internal dev phrases and the longest training input exercise both
    # evaluation scope and the inherited variable-length forward computation.
    ids = sorted(data.cohorts["dev_phrase3"])
    ids.append(max(data.arms["B26"], key=lambda s: (data.rows[s]["T"], s)))
    previous = read(OLD_BASE/f"results/B26_seed{seed}/eval_1820/summary.json")
    saved_rows = {r["clip_id"]: r for r in previous["rows"] if r["condition"] == "normal"}
    before = old.hashes(old.state(model, optimizer))
    rng = old.core.capture_rng()
    modes = {n: m.training for n, m in model.named_modules()}
    gradients = old.runner.nested_hash({n: p.grad for n, p in model.named_parameters()})
    comparisons = []
    try:
        model.eval()
        with torch.no_grad():
            for sid in ids:
                item = data.items[sid]
                pixels = np.ascontiguousarray(item["input"])
                values = model(torch.from_numpy(pixels[None]).to("cuda:0"),
                               torch.tensor([item["length"]], dtype=torch.long)).detach().cpu().numpy()[:, 0, :]
                row = saved_rows[sid]
                require(source.array_sha(pixels) == row["input_sha256"], "Source inference input differs")
                with np.load(bound(row["posterior"]), allow_pickle=False) as packed:
                    original = packed["log_probs"]
                require(np.array_equal(values, original) and source.array_sha(values) == row["posterior_array_sha256"],
                        "Restored source posterior does not reproduce the original")
                comparisons.append({"clip_id": sid, "T": item["length"], "source_posterior": row["posterior"],
                                    "reproduced_array_sha256": source.array_sha(values), "bitwise_exact": True})
    finally:
        for n, m in model.named_modules():
            m.training = modes[n]
        old.core.restore_rng(rng)
    require(old.hashes(old.state(model, optimizer)) == before and
            old.runner.nested_hash({n: p.grad for n, p in model.named_parameters()}) == gradients,
            "Source inference verification changed caller state")
    return {"seed": seed, "comparisons": comparisons, "forward_calls": 4, "state_isolation": True}


def evaluate(model, optimizer, data, seed, update, out, deadline):
    """Evaluate exactly 32 caches and frozen 5i controls, restoring all state."""
    out.mkdir(parents=True, exist_ok=False)
    (out/"posteriors").mkdir()
    before = old.hashes(old.state(model, optimizer))
    rng = old.core.capture_rng()
    modes = {name: m.training for name, m in model.named_modules()}
    gradients = old.runner.nested_hash({n: p.grad for n, p in model.named_parameters()})
    rows, began = [], time.perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for sid in sorted(data.rows):
                item, row_source = data.items[sid], data.rows[sid]
                cohort = next(c for c, ids in data.cohorts.items() if sid in ids)
                for condition, order in source.conditions(item["length"], sid).items():
                    require(time.perf_counter() < deadline, "Evaluation wall budget exceeded")
                    pixels = np.ascontiguousarray(item["input"][:, order])
                    row = {"clip_id": sid, "cohort": cohort, "condition": condition, "arm": "B26_extended", "seed": seed,
                           "update": update, "seen_in_optimizer": sid in data.arms["B26"], "reference": row_source["text"],
                           "T": item["length"], "frame_indices": order.tolist(), "input_sha256": source.array_sha(pixels),
                           "input_audit": item["audit"], "reference_label_ids": item["labels"].tolist(), "audio_input": False}
                    try:
                        values = model(torch.from_numpy(pixels[None]).to("cuda:0"),
                                       torch.tensor([item["length"]], dtype=torch.long)).detach().cpu().numpy()[:, 0, :]
                        require(values.shape == (item["length"], 69) and values.dtype == np.float32 and np.isfinite(values).all(), "Invalid posterior")
                        require(np.allclose(np.exp(values.astype(np.float64)).sum(1), 1, atol=1e-5, rtol=1e-5), "Unnormalized posterior")
                        require(torch.cuda.max_memory_reserved() <= 6*1024**3, "Evaluation CUDA budget exceeded")
                        decoded = contract.decode_prediction(values.argmax(1))
                        row.update({**decoded, "status": "success", "error": None,
                                    "metrics": contract.score_prediction(row_source["text"], decoded["raw_text"])})
                    except Exception as exc:
                        row.update({"status": "failed", "error": repr(exc), "posterior": None, "raw_text": "",
                                    "metrics": contract.score_prediction(row_source["text"], "", failed=True)})
                    if row["status"] == "success":
                        path = out/"posteriors"/f"{sid}__{condition}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream, log_probs=values, frame_indices=order, reference_labels=item["labels"],
                                                raw_argmax_ids=np.asarray(decoded["raw_argmax_ids"], np.int64))
                        row["posterior"] = ref(path)
                        row["posterior_array_sha256"] = source.array_sha(values)
                    rows.append(row)
                    with (out/"rows.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    finally:
        for name, m in model.named_modules():
            m.training = modes[name]
        old.core.restore_rng(rng)
        isolation = {"state_before": before, "state_after": old.hashes(old.state(model, optimizer)),
                     "gradients_before": gradients, "gradients_after": old.runner.nested_hash({n: p.grad for n, p in model.named_parameters()}),
                     "modes_before": modes, "modes_after": {n: m.training for n, m in model.named_modules()}}
        save(out/"isolation.json", isolation)
    require(isolation["state_before"] == isolation["state_after"] and isolation["gradients_before"] == isolation["gradients_after"] and
            isolation["modes_before"] == isolation["modes_after"], "Evaluation changed caller state")
    require(len(rows) == 128, "Missing evaluation rows")
    scores = {f"{cohort}/{condition}": contract.corpus_scores([r["metrics"] for r in rows if r["cohort"] == cohort and r["condition"] == condition])
              for cohort, condition in sorted({(r["cohort"], r["condition"]) for r in rows})}
    result = {"rows": rows, "cohorts": scores, "count": len(rows), "failures": sum(r["status"] == "failed" for r in rows),
              "seconds": time.perf_counter()-began, "frames": sum(r["T"] for r in rows), "state_isolation_passed": True,
              "optimizer_updates": 0, "audio_inputs": 0, "teacher_calls": 0,
              "dev_interpretation": "Two correlated views of three reused internal parents; not six independent examples/final test"}
    binding = save(out/"summary.json", result)
    return {"summary": binding, "cohorts": scores, "count": len(rows), "failures": result["failures"], "frames": result["frames"]}


def preflight(plan_path):
    """Check inherited state, actual next-update replay and maximum-size input (4 updates)."""
    plan = verify_plan(plan_path)
    data = source.Dataset(plan["old_data_plan"])
    runtime = old.setup()
    require(runtime == plan["runtime"], "Runtime differs")
    out = BASE/"results/preflight"
    out.mkdir(parents=True, exist_ok=False)
    tasks = {seed: schedule(plan, data, seed) for seed in plan["seeds"]}
    origins = {seed: load_origin(plan, data, seed) for seed in plan["seeds"]}
    maximum = max((t for ts in tasks.values() for t in ts), key=lambda t: (t["dense_frames"], t["true_frames"], tuple(t["clip_ids"])))
    model, optimizer = old.objects(42)
    reports = []
    source_checks = []
    try:
        old.restore(origins[42]["state"], model, optimizer)
        require(old.hashes(old.state(model, optimizer)) == origins[42]["state_hashes"], "Inherited state was not exactly restored")
        source_checks.append(verify_origin_predictions(model, optimizer, data, 42))
        reports.append(prior.update(model, optimizer, data.batch(tasks[42][0]["clip_ids"], "B26"), .0001))
        saved = old.state(model, optimizer)
        validate_state(saved, old.hashes(saved), 1821)
        binding = checkpoint(out/"after1821.pt", {"state": saved, "state_hashes": old.hashes(saved), "completed_updates": 1821, "mode": "isolated_preflight"})
        reference = prior.update(model, optimizer, data.batch(tasks[42][1]["clip_ids"], "B26"), .0001)
        reports.append(reference)
        expected = old.hashes(old.state(model, optimizer))
        del model, optimizer
        gc.collect()
        model, optimizer = old.objects(43)
        old.restore(origins[43]["state"], model, optimizer)
        require(old.hashes(old.state(model, optimizer)) == origins[43]["state_hashes"], "Second source restore differs")
        source_checks.append(verify_origin_predictions(model, optimizer, data, 43))
        reports.append(prior.update(model, optimizer, data.batch(maximum["clip_ids"], "B26"), .0001))
        restored = load_checkpoint(binding)
        old.restore(restored["state"], model, optimizer)
        require(old.hashes(old.state(model, optimizer)) == restored["state_hashes"], "Saved continuation restore differs")
        replay = prior.update(model, optimizer, data.batch(tasks[42][1]["clip_ids"], "B26"), .0001)
        reports.append(replay)
        require(old.hashes(old.state(model, optimizer)) == expected, "Persisted next-update replay differs")
        for key in ("loss", "gradient_norm_before_clip", "gradient_norm_after_clip", "nonzero_gradient_tensors"):
            require(reference[key] == replay[key], f"Replay {key} differs")
        validate_state(old.state(model, optimizer), expected, 1822)
        require(all(r["cuda_peak_reserved_bytes"] <= plan["limits"]["cuda_reserved_bytes"] for r in reports), "Preflight memory exceeded")
        return save(out/"result.json", {"status": "passed", "plan": ref(plan_path), "runtime": runtime,
                    "physical_updates": 4, "reports": reports, "checkpoint": binding,
                    "maxdense_pair": maximum["clip_ids"], "maximum_dense_frames": maximum["dense_frames"],
                    "maximum_true_frames": maximum["true_frames"], "source_state_hashes": {str(s): origins[s]["state_hashes"] for s in origins},
                    "source_restores_exact": True, "next_update_exact": True, "reference_final_hashes": expected,
                    "source_posterior_verification": source_checks, "source_posterior_forward_calls": 8,
                    "restored_final_hashes": old.hashes(old.state(model, optimizer)), "audio_inputs": 0, "source_approval_changed": False})
    except Exception:
        save(out/"failure.json", {"error": traceback.format_exc(), "physical_updates_returned": len(reports), "reports": reports,
                    "source_posterior_verification": source_checks})
        raise
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def train(plan_path, seed):
    """Continue one exact source state for 1,820 updates; never autoextend/restart."""
    plan = verify_plan(plan_path)
    require(seed in plan["seeds"], "Seed outside scope")
    pre = read(BASE/"results/preflight/result.json")
    require(pre["status"] == "passed" and pre["plan"] == ref(plan_path), "Missing matching preflight")
    runtime = old.setup()
    require(runtime == plan["runtime"], "Runtime differs")
    data = source.Dataset(plan["old_data_plan"])
    tasks = schedule(plan, data, seed)
    origin = load_origin(plan, data, seed)
    out = BASE/"results"/f"B26_extended_seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model, optimizer = old.objects(seed)
    history, checkpoints, evaluations = [], [], []
    started = time.perf_counter()
    try:
        old.restore(origin["state"], model, optimizer)
        initial = old.hashes(old.state(model, optimizer))
        require(initial == origin["state_hashes"], "Would train from changed initialization/Adam/RNG")
        identity = {"plan_sha256": sha(plan_path), "arm": "B26_extended", "seed": seed, "runtime": runtime,
                    "schedule_sha256": plan["schedules"][str(seed)]["sha256"], "source_checkpoint": plan["source_checkpoints"][str(seed)]}
        save(out/"run_contract.json", {"identity": identity, "initial_state_hashes": initial, "origin_completed_updates": 1820,
                "origin_initial_state_hashes": origin["initial_state_hashes"], "model_adam_rng_restored_exactly": True,
                "optimizer_ids": data.arms["B26"], "inputs": [data.items[s]["audit"] for s in sorted(data.items)],
                "training_condition": "normal", "new_updates_limit": 1820})
        for task in tasks:
            require(time.perf_counter()-started < plan["limits"]["wall_seconds_per_run"], "Run wall limit exceeded")
            batch = data.batch(task["clip_ids"], "B26")
            measured = prior.update(model, optimizer, batch, task["lr"])
            row = {"update": task["update"], "extension_update": len(history)+1, "epoch": task["epoch"],
                   "training_condition": "normal", "input_audits": batch["input_audits"],
                   "padded_batch_sha256": source.array_sha(batch["inputs"]), **measured}
            history.append(row)
            with (out/"progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            require(measured["cuda_peak_reserved_bytes"] <= plan["limits"]["cuda_reserved_bytes"], "CUDA budget exceeded")
            if task["update"] in plan["evaluations"]:
                state = old.state(model, optimizer)
                validate_state(state, old.hashes(state), task["update"])
                payload = {"schema": "offline-stage5j-full-continuation-checkpoint-v1", "identity": identity,
                           "initial_state_hashes": initial, "origin_completed_updates": 1820,
                           "completed_updates": task["update"], "completed_extension_updates": len(history), "extension_history": list(history),
                           "state": state, "state_hashes": old.hashes(state), "next_task": tasks[len(history)] if len(history) < 1820 else None}
                checkpoints.append(checkpoint(out/f"update_{task['update']:04d}.pt", payload))
                evaluated = evaluate(model, optimizer, data, seed, task["update"], out/f"eval_{task['update']:04d}", started+plan["limits"]["wall_seconds_per_run"])
                evaluations.append(evaluated)
                require(evaluated["failures"] == 0, "Evaluation failed; preserve results and stop")
            if len(history) % 130 == 0:
                print(json.dumps({"seed": seed, "total_update": task["update"], "new_updates": len(history),
                                  "loss": row["loss"], "seconds": time.perf_counter()-started}), flush=True)
        return save(out/"result.json", {"status": "completed", "identity": identity, "physical_updates": len(history),
                    "origin_updates": 1820, "total_updates": 3640, "initial_state_hashes": initial, "checkpoints": checkpoints,
                    "evaluations": evaluations, "seconds": time.perf_counter()-started,
                    "true_frames": sum(r["true_frames"] for r in history), "dense_frames": sum(r["dense_frames"] for r in history),
                    "audio_inputs": 0, "teacher_calls": 0, "service_changes": 0, "source_approval_changed": False,
                    "source_training_ready": False, "source_provisional": True})
    except Exception:
        save(out/"failure.json", {"error": traceback.format_exc(), "physical_updates_returned": len(history),
                    "last_checkpoint": checkpoints[-1] if checkpoints else None})
        raise
    finally:
        del model, optimizer, data
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("train")
    run.add_argument("--seed", type=int, choices=[42, 43], required=True)
    args = parser.parse_args()
    result = preflight(args.plan) if args.command == "preflight" else train(args.plan, args.seed)
    print(json.dumps(result, ensure_ascii=False), flush=True)
