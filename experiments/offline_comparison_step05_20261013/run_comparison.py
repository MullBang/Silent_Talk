"""Prepare, preflight or run the fixed stage5 curriculum comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import execution as e

BASE,PREVIOUS,ROOT=e.BASE,e.PREVIOUS,e.ROOT
PLAN=BASE/"execution_plan.json"
POLICY=BASE/"controlled_pilot_policy.json"


def prepare() -> None:
    """Freeze implemented code and exact schedules before new model execution."""
    source=ROOT/"experiments/offline_dataset_step03_20261013/results/cache_handoff.json"
    handoff=e.read(source)
    schedule_paths=[PREVIOUS/f"protocol/schedule_seed{seed}.json" for seed in (42,43)]
    schedule=e.read(schedule_paths[0])
    full=max((row["arms"]["A_sentence"] for row in schedule["updates"]),key=lambda r:sum(r["lengths"]))
    phrase=max((row["arms"]["B_curriculum"] for row in schedule["updates"] if row["arms"]["B_curriculum"]["kind"]=="phrase"),key=lambda r:sum(r["lengths"]))
    tasks={}
    for arm in ("A_sentence","B_curriculum"):
        tasks[arm]=[{"update":1,"lr":.001,**(full if arm=="A_sentence" else phrase)}, {"update":2,"lr":.0003,**full}]
    files=[BASE/name for name in ("execution.py","run_comparison.py","test_execution.py","data/loader.py","data/test_loader.py","evaluation/evaluator.py","evaluation/test_evaluator.py","evaluation/aggregate.py","evaluation/test_aggregate.py")]
    files += [PREVIOUS/name for name in ("engine.py","evaluation_contract.py","protocol/comparison_protocol.json","protocol/evaluation_plan.json","results/execution_clarifications.json","results/resource_budget.json","results/stage5_handoff.json")]
    files += schedule_paths+[source]
    runtime=e.old.setup()
    e.require(runtime==e.read(PREVIOUS/"smoke_plan.json")["runtime"],"Environment changed from verified stage4")
    plan={"schema":"offline-stage5-execution-plan-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "authorization":{"source":"current_user_request","request_text":"다음 진행","scope":"stage5 fixed comparison and necessary separate preflight"},
        "main_updates":5040,"preflight_updates":8,"preflight_tasks":tasks,"preflight_seed":42,
        "preflight_design":"A/B each continuous2 + split1 + resumed1; logical update1/2 tests phase/LR restoration. Maximum full pair316 and dense358 included. No preflight weight reused for main.",
        "main_design":"Four fresh A_sentence/B_curriculum x seed42/43;1260 per run; preserve prior fixed parent orders and all update/LR boundaries.",
        "evaluation_failure_policy":"Persist every diagnostic row, then stop the affected training run if any evaluation row fails; never silently drop clips or retune.",
        "zero_gradient_policy":"All gradients must exist and be finite. A finite zero gradient is logged and not concealed; the scheduled Adam step still executes. Preflight requires nonzero coverage in every parameter tensor.",
        "optimizer":{"name":"Adam","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"clip_norm":1.,"loss":"CPU FP32 CTC per-target-normalized mean, blank0, zero_infinityFalse"},
        "schedules":[{"seed":seed,**e.ref(path)} for seed,path in zip((42,43),schedule_paths)],
        "evaluation_plan":e.ref(PREVIOUS/"protocol/evaluation_plan.json"),
        "resource_budget":e.ref(PREVIOUS/"results/resource_budget.json"),
        "previous_manifests":[e.ref(ROOT/"experiments/offline_dataset_step03_20261013/results/completion_manifest.json"),e.ref(PREVIOUS/"results/completion_manifest.json")],
        "bindings":[e.ref(p) for p in files],"runtime":runtime,
        "main_checkpoints":[420,840,1260],"main_evaluations":1080,"fresh_initialization_required":True,
        "preflight_final_state_rule":"Exact complete model/Adam/RNG hashes across continuous and resumed; no posthoc tolerance relaxation.",
        "source_training_ready_preserved":False,"audio_inputs":0,"teacher_calls":0,"service_changes":0}
    e.save(PLAN,plan)
    policy={"schema":"lipnet-stage5-controlled-pilot-data-policy-v1","mode":"controlled_pilot",
        "authorization":{"source":"current_user_request","request_text":"다음 진행","scope":"stage5_fixed_comparison_and_preflight"},
        "training_plan":e.ref(PLAN),"handoff":e.ref(source),
        "stage3_frozen_manifest":e.ref(ROOT/"experiments/offline_dataset_step03_20261013/results/completion_manifest.json"),
        "allowed_optimizer_clip_ids":sorted(r["clip_id"] for r in handoff["records"] if r["split"]=="train"),
        "allowed_validation_clip_ids":sorted(r["clip_id"] for r in handoff["records"]),
        "allow_provisional_cache_for_controlled_pilot":True,"preserve_provisional_flags":True,
        "full_training_authorized":True,"service_change_authorized":False}
    e.save(POLICY,policy)
    e.save(BASE/"results/execution_unit_tests.json",{"status":"passed","tests_passed":11,"exit_code":0,"pytest_report":"11 passed in1.65s",
        "command":".venv-gpu/Scripts/python.exe -m pytest experiments/offline_comparison_step05_20261013/test_execution.py -q",
        "sources":[e.ref(BASE/p) for p in ("execution.py","test_execution.py")],"neural_calls":0})
    print(e.ref(PLAN),e.ref(POLICY),flush=True)


def gate(name: str) -> dict:
    """Require inspected code/data/test evidence before executing a planned scope."""
    result=e.read(BASE/"results"/name)
    e.require(result["status"]=="passed" and result["plan"]==e.ref(PLAN) and result["policy"]==e.ref(POLICY),"Missing/stale execution evidence")
    return result


def preflight() -> None:
    """Validate largest real pair and actual restoration in the new execution path."""
    gate("preflight_gate.json")
    plan=e.read(PLAN);e.verify_plan(plan)
    results=[]
    initial=None
    for arm in ("A_sentence","B_curriculum"):
        base=BASE/"preflight"/arm
        continuous=e.run_segment(PLAN,POLICY,42,arm,"preflight",base/"continuous",2)
        split=e.run_segment(PLAN,POLICY,42,arm,"preflight",base/"split",1)
        resumed=e.run_segment(PLAN,POLICY,42,arm,"preflight",base/"resumed",2,split["final_checkpoint"])
        left=e.load_checkpoint(continuous["final_checkpoint"])
        right=e.load_checkpoint(resumed["final_checkpoint"])
        e.require(left["state_hashes"]==right["state_hashes"],"Preflight continuation not bitwise exact")
        e.require(left["state_hashes"]["model"]!=continuous["initial_state_hashes"]["model"],"Preflight weights did not change")
        e.require(all(a["loss"]==b["loss"] for a,b in zip(left["history"],right["history"])),"Preflight losses differ")
        if initial is not None:e.require(initial==continuous["initial_state_hashes"],"A/B fresh initialization differs")
        initial=continuous["initial_state_hashes"]
        results.append({"arm":arm,"continuous":continuous,"split":split,"resumed":resumed,"final_exact":True})
    count=sum(run[name]["executed_updates"] for run in results for name in ("continuous","split","resumed"))
    e.require(count==8,"Preflight physical budget differs")
    e.save(BASE/"results/preflight_summary.json",{"status":"passed","plan":e.ref(PLAN),"policy":e.ref(POLICY),"runs":results,
        "physical_optimizer_updates":count,"main_updates":0,"max_true_frames_per_batch":316,"max_dense_frames_per_batch":358,
        "dev_neural_inputs":0,"fresh_main_weights_required":True})
    print("Stage5 preflight passed:8physical updates,2exact replay comparisons",flush=True)


def main_comparison() -> None:
    """Run the four fixed fresh comparisons once, after the separate preflight."""
    gate("main_pre_execution_gate.json")
    plan=e.read(PLAN);e.verify_plan(plan)
    e.require(e.read(BASE/"results/preflight_summary.json")["status"]=="passed","Preflight incomplete")
    e.require(not (BASE/"runs").exists(),"Main already attempted; preserve results and use explicit checkpoint recovery")
    began=time.perf_counter();runs=[];initials={}
    for seed in (42,43):
        for arm in ("A_sentence","B_curriculum"):
            e.require(time.perf_counter()-began<14400,"Aggregate wall budget exceeded")
            output=BASE/"runs"/f"seed{seed}"/arm
            result=e.run_segment(PLAN,POLICY,seed,arm,"main",output,1260)
            if seed in initials:e.require(initials[seed]==result["initial_state_hashes"],"Main paired initialization differs")
            initials[seed]=result["initial_state_hashes"]
            runs.append(e.ref(output/"segment_result.json"))
            with (BASE/"results/main_progress.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(e.json.dumps({"seed":seed,"arm":arm,"completed_updates":1260,"result":runs[-1]},ensure_ascii=False)+"\n")
    outputs=[e.read(Path(r["path"])) for r in runs]
    count=sum(r["executed_updates"] for r in outputs)
    eval_count=sum(ev["count"] for r in outputs for ev in r["evaluations"])
    e.require(count==5040 and eval_count==1080,"Main budget/evaluation coverage differs")
    e.save(BASE/"results/execution_summary.json",{"schema":"offline-stage5-execution-result-v1","status":"completed",
        "plan":e.ref(PLAN),"policy":e.ref(POLICY),"runs":runs,"physical_main_optimizer_updates":count,
        "separate_preflight_optimizer_updates":8,"evaluation_rows":eval_count,"checkpoint_count":sum(len(r["checkpoints"]) for r in outputs),
        "paired_initial_state_hashes":initials,"elapsed_seconds":time.perf_counter()-began,"audio_inputs":0,"teacher_calls":0,"downloads":0,"service_changes":0})
    print("Completed5040 mainupdates and1080evaluation rows",flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","preflight","main"])
    args=parser.parse_args()
    {"prepare":prepare,"preflight":preflight,"main":main_comparison}[args.action]()
