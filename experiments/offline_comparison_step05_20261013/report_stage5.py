"""Assemble audited stage5 accounting and final-state comparison evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def read(path: Path):
    """Read an existing UTF8 result without importing production modules."""
    return json.loads(path.read_text(encoding="utf-8"))


def ref(path: Path) -> dict:
    """Hash one locally stored artifact."""
    with path.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    return {"path":str(path.resolve()),"sha256":digest,"bytes":path.stat().st_size}


def save(path: Path,value: dict) -> None:
    """Create a new result only; prior evidence is never overwritten."""
    assert path.resolve().is_relative_to(BASE)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def main() -> None:
    """Derive physical counts and decision rule only after independent audits pass."""
    audit_names=["source_preflight.json","plan_preflight.json","preflight_checkpoint_audit.json",
        "main_checkpoint_audit.json","evaluation_audit.json","aggregate_audit.json","final_source_preservation.json"]
    for name in audit_names:
        assert read(BASE/"audit"/name)["status"]=="passed",name
    execution=read(BASE/"results/execution_summary.json")
    metrics=read(BASE/"results/aggregate_metrics.json")
    assert execution["status"]==metrics["status"]=="completed"
    logrows=[];runs=[];evaluations=[]
    for item in execution["runs"]:
        path=Path(item["path"])
        assert ref(path)["sha256"]==item["sha256"]
        run=read(path);runs.append(run)
        logpath=Path(run["progress"]["path"])
        assert ref(logpath)["sha256"]==run["progress"]["sha256"]
        rows=[json.loads(line) for line in logpath.read_text(encoding="utf-8").splitlines()]
        assert len(rows)==1260
        logrows.extend(rows)
        evaluations.extend(run["evaluations"])
    assert len(logrows)==5040 and sum(r["count"] for r in evaluations)==1080
    per_run=[]
    for run,final in zip(runs,metrics["final_models"]):
        assert run["identity"]["seed"]==final["seed"] and run["identity"]["arm"]==final["arm"]
        rows=[json.loads(line) for line in Path(run["progress"]["path"]).read_text(encoding="utf-8").splitlines()]
        per_run.append({"seed":final["seed"],"arm":final["arm"],"main_updates":len(rows),
            "true_frames":sum(r["true_frames"] for r in rows),"dense_frames":sum(r["dense_frames"] for r in rows),
            "padding_frames":sum(r["padding_frames"] for r in rows),"target_tokens":sum(r["target_tokens"] for r in rows),
            "optimizer_seconds":sum(r["seconds"] for r in rows),"segment_elapsed_seconds":run["elapsed_seconds"],
            "evaluation_seconds":sum(r["seconds"] for r in run["evaluations"]),
            "final_dev":final["primary_dev_sentence_normal"],"final_train":final["cohorts"]["train/full_sentence/normal"],
            "dev_static_minus_normal_CER_pp":final["primary_static_minus_normal_CER_pp"],
            "dev_shuffled_minus_normal_CER_pp":final["primary_shuffled_minus_normal_CER_pp"]})
    criterion=metrics["mean_primary"]["B_lower_CER_both_seeds"] and metrics["mean_primary"]["B_minus_A_CER_pp"]<=-5.
    accounting={"main_optimizer_updates":5040,"separate_preflight_updates":8,"total_new_physical_optimizer_updates":5048,
        "main_forward_calls":5040,"evaluation_forward_calls":sum(r["forward_calls"] for r in evaluations),
        "preflight_forward_calls":8,"total_new_neural_forward_calls":5048+sum(r["forward_calls"] for r in evaluations),
        "evaluation_frames":sum(r["frames"] for r in evaluations),
        "actual_main_frames":sum(r["true_frames"] for r in logrows),"dense_main_frames":sum(r["dense_frames"] for r in logrows),
        "main_padding_frames":sum(r["padding_frames"] for r in logrows),"main_target_tokens":sum(r["target_tokens"] for r in logrows),
        "main_optimizer_seconds":sum(r["seconds"] for r in logrows),"main_wall_seconds":execution["elapsed_seconds"],
        "evaluation_seconds":sum(r["seconds"] for r in evaluations),
        "max_training_cuda_allocated_bytes":max(r["cuda_peak_allocated_bytes"] for r in logrows),
        "max_training_cuda_reserved_bytes":max(r["cuda_peak_reserved_bytes"] for r in logrows),
        "max_evaluation_cuda_allocated_bytes":max(r["peak_allocated_cuda_bytes"] for r in evaluations),
        "max_evaluation_cuda_reserved_bytes":max(r["peak_reserved_cuda_bytes"] for r in evaluations),
        "main_checkpoints":12,"preflight_checkpoints":8,"posterior_files":1080,
        "checkpoint_bytes":sum(p.stat().st_size for p in BASE.rglob("*.pt")),
        "posterior_bytes":sum(p.stat().st_size for p in (BASE/"runs").rglob("*.npz")),
        "failed_evaluation_rows":metrics["failures"],"updates_with_any_zero_gradient_tensor":sum(r["nonzero_gradient_tensors"]<r["parameter_tensors"] for r in logrows)}
    assert accounting["actual_main_frames"]==911540 and accounting["dense_main_frames"]==1085994
    assert accounting["main_target_tokens"]==535780 and accounting["evaluation_frames"]==72828
    save(BASE/"results/resource_accounting.json",{"status":"completed","per_run":per_run,"totals":accounting,
        "scope":"Measured execution counts and times; whole interactive task/audit time is not included in main_wall_seconds.",
        "execution":ref(BASE/"results/execution_summary.json")})
    summary={"schema":"offline-stage5-completion-summary-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"completed_fixed_curriculum_comparison","data":{"train_parents":12,"dev_parents":3,"clips":30,"source_provisional":True,"training_ready":False},
        "model":"Fresh compact visual69CTC, no teacher/audio/pretrained checkpoint", "per_run":per_run,
        "paired_primary":metrics["paired_primary"],"mean_primary":metrics["mean_primary"],
        "predeclared_research_signal_met":criterion,
        "criterion":"Final1260 dev/full_sentence/normal CER lower for B in both seeds and average reduction>=5percentagepoints",
        "accounting":accounting,"tests_passed":50,
        "new_original_downloads":0,"source_TAR_reads":0,"audio_inputs":0,"teacher_calls":0,"service_changes":0,
        "preflight_replay_all_exact":True,"preserved_stage3_artifacts":114,"preserved_stage4_artifacts":101,
        "audits":[ref(BASE/"audit"/name) for name in audit_names],
        "results":[ref(BASE/"results"/p) for p in ("execution_summary.json","aggregate_metrics.json","resource_accounting.json")],
        "limitations":["Dev3 uses known historical speakers and is neither an unseen-speaker nor untouched final/webcam test.","Equal updates/parent visits are not equal frames/full-sentence exposures; no pure order-causality claim.","Phrase and sentence views share source pixels;30clips are only15parent utterances.","Checkpoint1260 is the fixed primary even if intermediate420/840 is better.","Training-ready/precise visual-boundary approvals and existing boundary flags are not promoted.","The curriculum comparison does not establish deployment readiness."]}
    save(BASE/"results/stage5_summary.json",summary)
    print(json.dumps({"status":summary["status"],"paired":summary["paired_primary"],"mean":summary["mean_primary"],"research_signal_met":criterion,"accounting":accounting},ensure_ascii=False))


if __name__=="__main__":
    main()
