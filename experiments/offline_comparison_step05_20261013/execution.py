"""Fixed stage5 dense visual CTC comparison with complete-state checkpointing."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = ROOT/"experiments/offline_training_step04_20261013"


def require(value: bool, message: str) -> None:
    """Fail explicitly before continuing an invalid computation."""
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash an explicit file using bounded memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def ref(path: Path) -> dict:
    """Bind bytes and an absolute path."""
    return {"path":str(path.resolve()),"sha256":sha(path)}


def read(path: Path) -> Any:
    """Read UTF8 JSON, propagating errors rather than falling back."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    """Create a new stage5 evidence file; never replace an existing result."""
    require(path.resolve().is_relative_to(BASE),"Output outside stage5")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def module(path: Path, name: str) -> Any:
    """Load a local module without invoking its CLI."""
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,"Missing module")
    obj=importlib.util.module_from_spec(spec)
    sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj


require(sha(PREVIOUS/"engine.py")=="fcd64be0d8731a75c10a3f7dfc41d58c814bda51b08ec2032c88ac50e80a2b2f","Frozen primitive source changed")
old=module(PREVIOUS/"engine.py","stage5_frozen_stage4_primitives")
data_module=module(BASE/"data/loader.py","stage5_data_loader")
torch,np=old.torch,old.np


def verify_plan(plan: dict) -> None:
    """Bind stage5 scope, code and the immutable stage4 comparison schedules."""
    require(plan["schema"]=="offline-stage5-execution-plan-v1","Wrong plan schema")
    require(plan["main_updates"]==5040 and plan["preflight_updates"]==8,"Budget changed")
    for item in plan["bindings"]:
        require(sha(Path(item["path"]))==item["sha256"],f"Bound input changed: {item['path']}")
    for item in plan["previous_manifests"]:
        require(sha(Path(item["path"]))==item["sha256"],"Snapshot changed")
        for artifact in read(Path(item["path"]))["artifacts"]:
            require(sha(Path(artifact["path"]))==artifact["sha256"],"Frozen artifact changed")


def load_datasets(plan_path: Path, policy_path: Path) -> dict:
    """Create explicitly bound train and dev loaders under the new stage5 policy."""
    policy=read(policy_path)
    require(policy["training_plan"]==ref(plan_path),"Policy plan differs")
    source=policy["handoff"]
    return {split:data_module.ControlledPilotDataset(source["path"],source["sha256"],policy_path,sha(policy_path),split=split) for split in ("train","dev")}


def scheduled_tasks(plan: dict, seed: int, arm: str, mode: str) -> list[dict]:
    """Use stored parent order; no sampling from a model's RNG."""
    require(arm in ("A_sentence","B_curriculum"),"Unknown arm")
    if mode=="preflight":
        require(seed==42,"Preflight is fixed to seed42")
        return plan["preflight_tasks"][arm]
    require(mode=="main" and seed in (42,43),"Unknown run")
    source=next(r for r in plan["schedules"] if r["seed"]==seed)
    schedule=read(Path(source["path"]))
    return [{"update":r["update"],"lr":r["lr"],**r["arms"][arm]} for r in schedule["updates"]]


def update(model: Any, optimizer: Any, batch: dict, lr: float) -> dict:
    """Apply one batch2 FP32 CTC update; finite zero gradients remain observable."""
    require(batch["purpose"]=="optimizer" and batch["splits"]==["train","train"],"Nontrain optimizer batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"]=lr
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter()
    x=torch.from_numpy(batch["inputs"]).to("cuda:0")
    lengths=torch.from_numpy(batch["lengths"])
    output=model(x,lengths=lengths)
    loss=old.ctc(output,batch)
    require(bool(torch.isfinite(loss)),"Nonfinite CTC loss")
    loss.backward()
    params=list(model.parameters())
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),"Missing/nonfinite gradient")
    nonzero=sum(bool(torch.count_nonzero(p.grad)) for p in params)
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    norm_after=math.sqrt(sum(float(p.grad.double().square().sum().cpu()) for p in params))
    require(norm_after<=1.0001,"Clipped gradient norm exceeded")
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in params),"Nonfinite parameter")
    torch.cuda.synchronize()
    return {"clip_ids":batch["clip_ids"],"lengths":batch["lengths"].tolist(),
        "true_frames":int(lengths.sum()),"dense_frames":int(2*lengths.max()),
        "padding_frames":int(2*lengths.max()-lengths.sum()),"target_tokens":int(batch["label_lengths"].sum()),
        "loss":float(loss.detach()),"gradient_norm_before_clip":float(norm),"gradient_norm_after_clip":norm_after,
        "nonzero_gradient_tensors":nonzero,"parameter_tensors":len(params),"optimizer_updated":True,"learning_rate":lr,
        "seconds":time.perf_counter()-began,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved()}


def load_checkpoint(binding: dict) -> dict:
    """Check checkpoint bytes before a restricted CPU tensor load."""
    path=Path(binding["path"]).resolve()
    require(path.is_relative_to(BASE) and sha(path)==binding["sha256"],"Wrong checkpoint path/hash")
    return old.runner._unpack_safe(torch.load(path,map_location="cpu",weights_only=True))


def validate_checkpoint(payload: dict, identity: dict, tasks: list[dict]) -> None:
    """Reject mismatched run/data/RNG/state, cursor, phase, LR and Adam step."""
    require(payload["schema"]=="offline-stage5-complete-checkpoint-v1" and payload["identity"]==identity,"Checkpoint identity differs")
    done=payload["completed_updates"]
    require(type(done) is int and 0<done<=len(tasks) and len(payload["history"])==done,"Invalid completed history")
    require(payload["state_hashes"]==old.hashes(payload["state"]),"Checkpoint state changed")
    require(payload["next_task"]==(tasks[done] if done<len(tasks) else None),"Next phase/LR/cursor differs")
    for index,row in enumerate(payload["history"]):
        task=tasks[index]
        require(row["update"]==index+1 and row["clip_ids"]==task["clip_ids"] and row["kind"]==task["kind"] and row["learning_rate"]==task["lr"] and row["optimizer_updated"] is True,"History order/input/LR differs")
    opt=payload["state"]["optimizer"]
    require(opt["state"] and all(float(v["step"])==done for v in opt["state"].values()),"Adam step differs")
    require(all(g["lr"]==tasks[done-1]["lr"] for g in opt["param_groups"]),"Stored optimizer LR differs")


def write_checkpoint(path: Path, payload: dict, identity: dict, tasks: list[dict]) -> dict:
    """Persist complete state atomically, then verify its restricted load."""
    require(path.resolve().is_relative_to(BASE) and not path.exists(),"Existing/outside checkpoint")
    part=path.with_suffix(".pt.partial")
    with part.open("xb") as stream:
        torch.save(old.runner._pack_safe(payload),stream)
        stream.flush()
        os.fsync(stream.fileno())
    part.rename(path)
    binding={**ref(path),"bytes":path.stat().st_size}
    validate_checkpoint(load_checkpoint(binding),identity,tasks)
    return binding


def run_segment(plan_path: Path, policy_path: Path, seed: int, arm: str, mode: str,
                output: Path, stop: int, resume: dict | None=None) -> dict:
    """Run a fixed segment and save all physical updates and evaluation references."""
    began=time.perf_counter()
    plan=read(plan_path)
    verify_plan(plan)
    runtime=old.setup()
    require(runtime==plan["runtime"],"GPU/runtime differs")
    tasks=scheduled_tasks(plan,seed,arm,mode)
    require(0<stop<=len(tasks) and (mode!="main" or stop in (420,840,1260)),"Invalid segment stop")
    require(output.resolve().is_relative_to(BASE) and not output.exists(),"Output must be a new segment directory")
    budget=read(Path(plan["resource_budget"]["path"]))["conservative_future_limits"]
    require(shutil.disk_usage(BASE).free>=budget["minimum_free_local_disk_bytes"],"Insufficient local space")
    require(torch.cuda.mem_get_info()[0]>=budget["minimum_free_cuda_bytes_at_start"],"Insufficient free CUDA memory")
    output.mkdir(parents=True)
    identity={"plan_sha256":sha(plan_path),"policy_sha256":sha(policy_path),"runtime":runtime,"mode":mode,"seed":seed,"arm":arm,
        "tasks_sha256":hashlib.sha256(json.dumps(tasks,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()}
    datasets=load_datasets(plan_path,policy_path)
    ids=sorted({sid for task in tasks for sid in task["clip_ids"]})
    items={sid:datasets["train"].load_clip(sid,purpose="optimizer") for sid in ids}
    model,optimizer=old.objects(seed)
    initial=old.hashes(old.state(model,optimizer))
    history,checkpoints,evaluations=[],[],[]
    restored=None
    prior_elapsed=0.
    if resume is not None:
        previous=load_checkpoint(resume)
        validate_checkpoint(previous,identity,tasks)
        require(previous["completed_updates"]<stop and previous["initial_state_hashes"]==initial,"Resume origin/cursor differs")
        old.restore(previous["state"],model,optimizer)
        require(old.hashes(old.state(model,optimizer))==previous["state_hashes"],"Restored state is not exact")
        history=list(previous["history"])
        prior_elapsed=previous["elapsed_seconds"]
        restored={"exact":True,"next_update":len(history)+1,"state_hashes":previous["state_hashes"]}
    save(output/"segment_contract.json",{"identity":identity,"initial_state_hashes":initial,"start_update":len(history),"stop_update":stop,"resume":resume,"restore":restored,
        "loaded_optimizer_clips":[items[s]["audit"] for s in ids],"normalized_cpu_bytes":sum(items[s]["input"].nbytes for s in ids)})
    executed=[]
    try:
        for index in range(len(history),stop):
            require(prior_elapsed+time.perf_counter()-began<budget["wall_seconds_per_run_including_evaluation"],"Run wall budget exceeded")
            task=tasks[index]
            batch=data_module.collate([items[s] for s in task["clip_ids"]],purpose="optimizer")
            require(batch["lengths"].tolist()==task["lengths"],"Scheduled lengths differ")
            measured=update(model,optimizer,batch,task["lr"])
            row={"update":index+1,"kind":task["kind"],**measured}
            history.append(row)
            executed.append(row)
            with (output/"progress.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            require(measured["cuda_peak_reserved_bytes"]<=budget["stop_if_cuda_reserved_exceeds_bytes"],"CUDA reservation budget exceeded")
            if mode=="preflight":
                require(measured["nonzero_gradient_tensors"]==measured["parameter_tensors"],"Preflight requires genuine gradient coverage")
            should_save=mode=="preflight" or index+1 in (420,840,1260)
            if should_save:
                current=old.state(model,optimizer)
                payload={"schema":"offline-stage5-complete-checkpoint-v1","identity":identity,"completed_updates":index+1,
                    "state":current,"state_hashes":old.hashes(current),"initial_state_hashes":initial,"history":list(history),
                    "next_task":tasks[index+1] if index+1<len(tasks) else None,"elapsed_seconds":prior_elapsed+time.perf_counter()-began}
                checkpoint=write_checkpoint(output/f"update_{index+1:04d}.pt",payload,identity,tasks)
                checkpoints.append(checkpoint)
                if mode=="main":
                    evaluator=module(BASE/"evaluation/evaluator.py",f"stage5_evaluator_{seed}_{arm}_{index}")
                    before=old.hashes(old.state(model,optimizer))
                    result=evaluator.evaluate_checkpoint(model,datasets,read(Path(plan["evaluation_plan"]["path"])),
                        output/"evaluations"/f"update_{index+1:04d}",
                        {**identity,"completed_update":index+1,"checkpoint":checkpoint})
                    require(before==old.hashes(old.state(model,optimizer)),"Evaluation changed training state/RNG")
                    evaluations.append(result)
                    save(output/f"evaluation_return_{index+1:04d}.json",result)
                    require(result["failures"]==0,"Evaluation failed rows preserved; stop before further training")
                    require(torch.cuda.max_memory_reserved()<=budget["stop_if_cuda_reserved_exceeds_bytes"],"Evaluation CUDA reservation budget exceeded")
            if mode=="preflight" or (index+1)%60==0:
                print(json.dumps({"seed":seed,"arm":arm,"mode":mode,"update":index+1,"loss":row["loss"],"seconds":round(time.perf_counter()-began,2)}),flush=True)
        result={"status":"completed","identity":identity,"initial_state_hashes":initial,"completed_updates":stop,"executed_updates":len(executed),
            "history_count":len(history),"restore":restored,"checkpoints":checkpoints,"evaluations":evaluations,
            "elapsed_seconds":time.perf_counter()-began,"cumulative_run_seconds":prior_elapsed+time.perf_counter()-began,
            "progress":ref(output/"progress.jsonl"),"final_checkpoint":checkpoints[-1],
            "audio_inputs":0,"teacher_calls":0,"service_changes":0}
        save(output/"segment_result.json",result)
        return result
    except Exception as exc:
        save(output/"failure.json",{"error":repr(exc),"completed_updates":len(history),"physical_updates_this_segment":len(executed),"last_complete_checkpoint":checkpoints[-1] if checkpoints else resume})
        raise
    finally:
        del model,optimizer,datasets,items
        gc.collect()
        torch.cuda.empty_cache()
