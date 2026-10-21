"""Prepare or run the explicitly bounded stage4 loader/GPU/recovery pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
from typing import Any

import engine

BASE, ROOT = engine.BASE, engine.ROOT
PLAN = BASE / "smoke_plan.json"
POLICY = BASE / "controlled_pilot_policy.json"
torch, np = engine.torch, engine.np


def binding(path: Path) -> dict[str, Any]:
    """Reference a fixed local file by absolute path and SHA."""
    return {"path": str(path.resolve()), "sha256": engine.sha(path)}


def prepare() -> None:
    """Freeze data/trajectory/code/tolerances before observing any neural updates."""
    stage3 = ROOT / "experiments/offline_dataset_step03_20261013"
    handoff_path = stage3 / "results/cache_handoff.json"
    handoff = engine.read(handoff_path)
    chosen = {r["rank"]:r["parent_id"] for r in handoff["records"] if r["split"] == "train"}
    engine.require(all(rank in chosen for rank in (1, 5, 9)), "Expected representative train ranks missing")
    pairs = [(1,9),(5,9),(1,5),(9,5),(9,1),(5,1)]
    schedules = {}
    for arm in ("A_sentence", "B_curriculum"):
        tasks = []
        for index, pair in enumerate(pairs):
            kind = "phrase" if arm == "B_curriculum" and index < 2 else "full_sentence"
            tasks.append({"update": index+1, "kind": kind, "parent_ids": [chosen[r] for r in pair],
                "clip_ids": [chosen[r]+"__"+kind for r in pair], "lr": [.001,.001,.0003,.0003,.0001,.0001][index]})
        schedules[arm] = tasks
    paths = [BASE/name for name in ("engine.py", "run_smoke.py", "test_engine_contract.py", "data/loader.py", "data/test_loader.py")]
    paths += [engine.RUNNER_PATH, engine.POOL_PATH, ROOT/"experiments/feature_transfer_step07/train_compare.py",
        ROOT/"backend/models/baseline.py", ROOT/"backend/models/__init__.py", ROOT/"backend/models/tokenizer.py",
        handoff_path, stage3/"results/completion_manifest.json"]
    plan = {"schema": "offline-stage4-six-update-smoke-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User asked 다음 진행 after stage3; scope is the announced stage4 preparation and bounded real backprop/restart checks.",
        "updates": 6, "seeds": [42,43], "arms": ["A_sentence","B_curriculum"], "schedules": schedules,
        "selected_parent_ranks": [1,5,9], "selection_reason": "Preselected train-only minimum phrase T23, maximum full-sentence T179 and longer phrase T67; no scores used.",
        "segments": {"continuous": 6, "split": 2, "resumed": 4}, "planned_optimizer_updates": 48,
        "planned_main_comparison_updates": 0, "batch_size": 2, "batch_contract": "Dense last-frame padding, true lengths into every model call and CPU CTC.",
        "optimizer": {"name":"Adam","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"clip_norm":1.},
        "runtime": engine.setup(), "bindings": [binding(path) for path in paths],
        "model": "Fresh compact3D-CNN(32,64,96)/projection256/2-layer bidirectionalLSTM256/69CTC with frozen equivalent deterministic pooling; no adapter, teacher or historical weights.",
        "checkpoint": "model, Adam state and step, Python/NumPy/CPU/allCUDA RNG, global update, next data/phase/LR, plan/policy/runtime/code/data identity",
        "tolerances_fixed_before_execution": {"restoration": "all component hashes exact", "RNG_replay": "exact",
            "model": {"atol":1e-6,"rtol":1e-5}, "optimizer_exp_avg":{"atol":1e-7,"rtol":1e-4},
            "optimizer_exp_avg_sq":{"atol":1e-10,"rtol":1e-4}, "loss":{"atol":1e-5,"rtol":1e-5},
            "gradient_norm":{"atol":1e-4,"rtol":1e-4}, "padding_invariance":{"atol":1e-4,"rtol":1e-4}},
        "time_limit_seconds_per_segment": 180, "dev_forward_calls_allowed":0, "audio_inputs_allowed":0,
        "teacher_calls_allowed":0, "deployments_allowed":0,
        "stopping_rule":"Preserve any failure; no extra optimization or relaxed posthoc tolerances. Full comparison remains stage5."}
    engine.save(PLAN, plan)
    policy = {"schema": "lipnet-stage4-controlled-pilot-data-policy-v1", "mode": "controlled_pilot",
        "authorization":{"source":"current_user_request","request_text":"다음 진행","scope":"stage4_loader_backprop_resume_smoke_only"},
        "training_plan": binding(PLAN), "handoff": binding(handoff_path),
        "stage3_frozen_manifest": binding(stage3/"results/completion_manifest.json"),
        "allowed_optimizer_clip_ids": sorted(r["clip_id"] for r in handoff["records"] if r["split"] == "train"),
        "allowed_validation_clip_ids": sorted(r["clip_id"] for r in handoff["records"] if r["split"] == "dev"),
        "allow_provisional_cache_for_controlled_pilot":True,"preserve_provisional_flags":True,
        "full_training_authorized":False,"service_change_authorized":False}
    engine.save(POLICY, policy)
    print(json.dumps({"plan":binding(PLAN),"policy":binding(POLICY),"planned_optimizer_updates":48}))


def compare_state(left: dict[str, Any], right: dict[str, Any], tolerances: dict[str, Any]) -> dict[str, Any]:
    """Compare full model/Adam/RNG state with the predeclared component tolerances."""
    checks = []
    engine.require(engine.runner.nested_hash(left["rng"]) == engine.runner.nested_hash(right["rng"]), "Replay RNG differs")
    def walk(a: Any, b: Any, path: str) -> None:
        """Walk matching nested tensors and scalars without ignoring optimizer fields."""
        engine.require(type(a) is type(b), f"State type differs: {path}")
        if torch.is_tensor(a):
            engine.require(a.shape == b.shape and a.dtype == b.dtype, f"State layout differs: {path}")
            if not a.is_floating_point() or path.endswith("step"):
                engine.require(torch.equal(a,b), f"Exact state differs: {path}")
            else:
                key = "optimizer_exp_avg_sq" if path.endswith("exp_avg_sq") else "optimizer_exp_avg" if path.endswith("exp_avg") else "model"
                bound = tolerances[key]
                engine.require(torch.allclose(a,b,atol=bound["atol"],rtol=bound["rtol"]), f"Replay exceeds frozen tolerance: {path}")
                checks.append({"path":path,"exact":torch.equal(a,b),"maximum_abs_difference":float((a-b).abs().max()),"tolerance":key})
        elif isinstance(a, dict):
            engine.require(a.keys() == b.keys(), f"State keys differ: {path}")
            for key in a:
                walk(a[key],b[key],path+"/"+str(key))
        elif isinstance(a,(list,tuple)):
            engine.require(len(a) == len(b), f"State length differs: {path}")
            for index,(x,y) in enumerate(zip(a,b)):
                walk(x,y,path+"/"+str(index))
        else:
            engine.require(a == b, f"Exact scalar differs: {path}")
    walk(left["model"],right["model"],"model")
    walk(left["optimizer"],right["optimizer"],"optimizer")
    return {"passed":True,"all_component_hashes_exact":engine.hashes(left) == engine.hashes(right),
            "RNG_exact":True,"tensor_checks":checks}


def padding_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    """Check true-length CNN/LSTM output invariance against different padded companions."""
    data = engine.dataset(PLAN, POLICY)
    ids = [plan["schedules"]["B_curriculum"][0]["clip_ids"][0],
           plan["schedules"]["A_sentence"][1]["clip_ids"][0]]
    batch = engine.make_batch(data, ids)
    model, optimizer = engine.objects(42)
    model.eval()
    before = engine.hashes(engine.state(model,optimizer))
    tol = plan["tolerances_fixed_before_execution"]["padding_invariance"]
    with torch.no_grad():
        x = torch.from_numpy(batch["inputs"]).to("cuda:0")
        lengths = torch.from_numpy(batch["lengths"])
        outputs = model(x,lengths=lengths)
        checks = []
        single_losses = []
        target_start = 0
        for index in range(2):
            t = int(lengths[index])
            single = model(x[index:index+1,:,:t], lengths=lengths[index:index+1])
            difference = float((outputs[:t,index:index+1]-single).abs().max())
            engine.require(torch.allclose(outputs[:t,index:index+1],single,**tol), "Padding/companion changes true sequence output")
            n = int(batch["label_lengths"][index])
            targets = torch.from_numpy(batch["labels"][target_start:target_start+n])
            target_start += n
            single_losses.append(engine.F.ctc_loss(outputs[:t,index:index+1].cpu(),targets,torch.tensor([t]),torch.tensor([n]),blank=0,reduction="mean",zero_infinity=False))
            checks.append({"clip_id":ids[index],"T":t,"maximum_abs_logprob_difference":difference})
        batched_loss = engine.ctc(outputs,batch)
        mean_single = torch.stack(single_losses).mean()
        engine.require(torch.allclose(batched_loss,mean_single,atol=1e-6,rtol=1e-6), "Batch CTC is not mean per-target-length loss")
    engine.require(before == engine.hashes(engine.state(model,optimizer)), "Evaluation preflight mutated state or RNG")
    result = {"status":"passed","clip_ids":ids,"checks":checks,"tolerances":tol,
        "input_shape":list(batch["inputs"].shape),"true_frames":int(lengths.sum()),
        "padding_frames":int(2*lengths.max()-lengths.sum()),"forward_calls":3,"optimizer_updates":0,
        "batch_ctc_loss":float(batched_loss),"mean_of_individual_CTC":float(mean_single),
        "model_parameter_count":sum(p.numel() for p in model.parameters()),"state_unchanged":True,
        "dev_neural_inputs":0,"recognition_performance_measured":False}
    engine.save(BASE/"results/padding_preflight.json",result)
    del model,optimizer,data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def execute() -> None:
    """Run each continuous/restarted smoke exactly once, then compare full states."""
    plan = engine.read(PLAN)
    engine.verify_files(plan)
    engine.require(engine.setup() == plan["runtime"], "GPU runtime changed")
    gate = engine.read(BASE/"results/pre_execution_validation.json")
    engine.require(gate["status"] == "passed" and gate["plan"] == binding(PLAN) and gate["policy"] == binding(POLICY), "Passed pre-execution contract gate required")
    engine.require(not (BASE/"smoke").exists(), "Smoke already attempted; preserve it and use a new version")
    padding = padding_preflight(plan)
    runs, initial_by_seed = [], {}
    for seed in plan["seeds"]:
        for arm in plan["arms"]:
            out = BASE/"smoke"/f"seed{seed}"/arm
            continuous = engine.run_segment(PLAN,POLICY,seed,arm,out/"continuous",6)
            split = engine.run_segment(PLAN,POLICY,seed,arm,out/"split",2)
            resumed = engine.run_segment(PLAN,POLICY,seed,arm,out/"resumed",6,split["final_checkpoint"])
            left,right = engine.load_checkpoint(continuous["final_checkpoint"]),engine.load_checkpoint(resumed["final_checkpoint"])
            compared = compare_state(left["state"],right["state"],plan["tolerances_fixed_before_execution"])
            for l,r in zip(left["history"][2:],right["history"][2:]):
                for key, bound in (("loss","loss"),("gradient_norm_before_clip","gradient_norm")):
                    tol=plan["tolerances_fixed_before_execution"][bound]
                    engine.require(abs(l[key]-r[key]) <= tol["atol"]+tol["rtol"]*abs(r[key]), "Replay metric differs")
            if seed in initial_by_seed:
                engine.require(initial_by_seed[seed] == continuous["initial_state_hashes"], "A/B fresh initialization differs")
            initial_by_seed[seed] = continuous["initial_state_hashes"]
            comparison_path = out/"replay_comparison.json"
            engine.save(comparison_path,compared)
            runs.append({"seed":seed,"arm":arm,"continuous":continuous,"split":split,"resumed":resumed,"comparison":binding(comparison_path),
                         "final_replay_hashes_exact":compared["all_component_hashes_exact"]})
    actual = sum(len(run[segment]["executed_updates"]) for run in runs for segment in ("continuous","split","resumed"))
    engine.require(actual == 48, "Executed smoke budget differs")
    metrics = [step for run in runs for segment in ("continuous","split","resumed") for step in run[segment]["executed_updates"]]
    result={"schema":"offline-stage4-gpu-smoke-result-v1","status":"passed","plan":binding(PLAN),"policy":binding(POLICY),
        "runs":runs,"padding_preflight":binding(BASE/"results/padding_preflight.json"),
        "actual_smoke_optimizer_updates":actual,"actual_dense_training_forward_calls":actual,
        "additional_padding_eval_forward_calls":padding["forward_calls"],"main_comparison_updates":0,
        "paired_initial_state_hashes":initial_by_seed,"max_cuda_allocated_bytes":max(r["cuda_peak_allocated_bytes"] for r in metrics),
        "max_cuda_reserved_bytes":max(r["cuda_peak_reserved_bytes"] for r in metrics),
        "optimizer_seconds_total":sum(r["seconds"] for r in metrics),
        "actual_true_training_frames_including_replays":sum(r["true_frames"] for r in metrics),
        "actual_dense_training_frames_including_padding_and_replays":sum(r["dense_frames"] for r in metrics),
        "all_final_replays_bitwise_exact":all(r["final_replay_hashes_exact"] for r in runs),
        "dev_neural_inputs":0,"audio_inputs":0,"teacher_calls":0,"service_changes":0,
        "recognition_performance_claim":False,"checkpoint_deployment_allowed":False}
    engine.save(BASE/"results/smoke_summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("runs","paired_initial_state_hashes")},ensure_ascii=False))


def main() -> None:
    """Separate preparation from authorized, gated actual GPU execution."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    engine.require(args.prepare != args.execute, "Choose exactly one stage4 action")
    prepare() if args.prepare else execute()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
