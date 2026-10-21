"""Independent complete-state, update, schedule and evaluation-transaction audit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from audit_common import state_hashes, unpack
from audit_sources import binding, check_binding, digest, read, S3, S4, S5B
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def check_state(state: dict, done: int, lr: float) -> dict:
    assert set(state) == {"model", "optimizer", "rng"}
    model, optimizer, rng = state["model"], state["optimizer"], state["rng"]
    assert sum(x.numel() for x in model.values()) == 3287781
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in model.values())
    groups = optimizer["param_groups"]
    assert len(groups) == 1
    group = groups[0]
    assert group["lr"] == lr and tuple(group["betas"]) == (.9, .999)
    assert group["eps"] == 1e-8 and group["weight_decay"] == 0 and group["amsgrad"] is False
    assert len(group["params"]) == len(set(group["params"])) == len(model) == len(optimizer["state"])
    assert set(group["params"]) == set(optimizer["state"])
    for ident, parameter in zip(group["params"], model.values()):
        saved = optimizer["state"][ident]
        assert set(saved) == {"step", "exp_avg", "exp_avg_sq"}
        assert saved["step"].numel() == 1 and float(saved["step"]) == done
        assert saved["exp_avg"].shape == saved["exp_avg_sq"].shape == parameter.shape
        assert saved["exp_avg"].dtype == saved["exp_avg_sq"].dtype == parameter.dtype
        assert torch.isfinite(saved["exp_avg"]).all() and torch.isfinite(saved["exp_avg_sq"]).all()
        assert (saved["exp_avg_sq"] >= 0).all()
    assert set(rng) == {"python", "numpy", "torch", "cuda"}
    assert len(rng["python"]) == 3 and rng["python"][0] == 3
    assert len(rng["numpy"]) == 5 and rng["numpy"][0] == "MT19937" and rng["numpy"][1].dtype == np.uint32
    assert rng["torch"].dtype == torch.uint8 and rng["torch"].ndim == 1
    assert len(rng["cuda"]) == 1 and rng["cuda"][0].dtype == torch.uint8
    return {"parameters": 3287781, "parameter_tensors": len(model), "Adam_states": len(optimizer["state"]), "Adam_steps": done}


def check_history(history: list[dict], tasks: list[dict], lookup: dict) -> None:
    for number, row in enumerate(history, 1):
        task = tasks[number - 1]
        assert row["update"] == number and row["kind"] == task["kind"]
        assert row["clip_ids"] == task["clip_ids"] and row["learning_rate"] == task["lr"]
        expected = [lookup[clip] for clip in row["clip_ids"]]
        assert all(r["split"] == "train" for r in expected)
        lengths = [r["T"] for r in expected]
        assert row["lengths"] == lengths and row["true_frames"] == sum(lengths)
        assert row["dense_frames"] == 2 * max(lengths) and row["padding_frames"] == 2 * max(lengths) - sum(lengths)
        assert row["target_tokens"] == sum(r["label_length"] for r in expected)
        assert row["optimizer_updated"] is True
        assert 0 <= row["nonzero_gradient_tensors"] <= row["parameter_tensors"]
        assert 0 <= row["gradient_norm_after_clip"] <= 1.0001
        assert all(math.isfinite(row[key]) for key in ["loss", "gradient_norm_before_clip", "gradient_norm_after_clip", "seconds"])
        assert row["seconds"] >= 0 and row["cuda_peak_reserved_bytes"] <= 6442450944
        assert row["cuda_peak_allocated_bytes"] <= row["cuda_peak_reserved_bytes"]


def inspect(mode):
    torch.set_num_threads(4)
    planpath=BASE/"execution_plan.json";policypath=BASE/"controlled_pilot_policy.json"
    plan=read(planpath);policy=read(policypath)
    for item in plan["bindings"]+plan["previous_manifests"]:
        check_binding(item)
    assert policy["training_plan"]=={"path":str(planpath),"sha256":digest(planpath)}
    lookup={r["clip_id"]:r for r in read(S3/"results/cache_handoff.json")["records"]}
    oldinitial=read(S4/"results/smoke_summary.json")["paired_initial_state_hashes"]
    results=[(p,read(p)) for p in sorted(BASE.glob("**/segment_result.json")) if read(p)["identity"]["mode"]==mode]
    assert len(results)==(6 if mode=="preflight" else 2),len(results)
    checkpoints=[];allprogress=[];segments=[];identities=set();maxmemory=0
    for path,value in results:
        identity=value["identity"];seed=identity["seed"]
        assert seed in [42,43] and identity["arm"]=="P_phrase"
        assert identity["plan_sha256"]==digest(planpath) and identity["policy_sha256"]==digest(policypath)
        assert identity["runtime"]==plan["runtime"]
        tasks=plan["preflight_tasks"] if mode=="preflight" else read(S5B/f"design/proposed_phrase_schedule_seed{seed}.json")["updates"]
        expectedtasksha=hashlib.sha256(json.dumps(tasks,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        assert identity["tasks_sha256"]==expectedtasksha
        initial=value["initial_state_hashes"];assert initial==oldinitial[str(seed)]
        contract=read(path.parent/"segment_contract.json")
        assert contract["identity"]==identity and contract["initial_state_hashes"]==initial
        start,stop=contract["start_update"],contract["stop_update"]
        progress=[json.loads(line) for line in (path.parent/"progress.jsonl").read_text(encoding="utf8").splitlines()]
        assert len(progress)==value["executed_updates"]==stop-start
        assert [r["update"] for r in progress]==list(range(start+1,stop+1))
        if start:
            check_binding(contract["resume"])
            previous=unpack(torch.load(contract["resume"]["path"],map_location="cpu",weights_only=True))
            assert previous["completed_updates"]==start and previous["state_hashes"]==state_hashes(previous["state"])
        else:
            assert contract["resume"] is None
        for reference in value["checkpoints"]:
            check_binding(reference)
            payload=unpack(torch.load(reference["path"],map_location="cpu",weights_only=True))
            assert payload["schema"]=="offline-stage5c-complete-checkpoint-v1" and payload["identity"]==identity
            done=payload["completed_updates"]
            assert payload["initial_state_hashes"]==initial and len(payload["history"])==done
            assert payload["history"][start:done]==progress[:done-start]
            check_history(payload["history"],tasks,lookup)
            assert payload["next_task"]==(tasks[done] if done<len(tasks) else None)
            assert payload["state_hashes"]==state_hashes(payload["state"])
            assert payload["state_hashes"]["model"]!=initial["model"]
            checked=check_state(payload["state"],done,tasks[done-1]["lr"])
            assert payload["evaluation_due"]==(mode=="main")
            if mode=="main":
                evaldir=Path(payload["evaluation_dir"])
                assert evaldir==path.parent/"evaluations"/f"update_{done:04d}"
                receiptpath=evaldir.parent/(evaldir.name+"_completed.json")
                receipt=read(receiptpath)
                assert receipt["status"]=="completed" and receipt["checkpoint"]==reference
                assert receipt["training_state_before"]==receipt["training_state_after"]==payload["state_hashes"]
                evaluation=receipt["evaluation"];check_binding(evaluation["summary"])
                assert evaluation["identity"]["completed_update"]==done
                assert all(evaluation["identity"][key]==val for key,val in identity.items())
                count=36 if done==1680 else 12
                frames=1443 if done==1680 else 481
                assert evaluation["count"]==evaluation["forward_calls"]==evaluation["new_forward_calls"]==count
                assert evaluation["frames"]==frames and evaluation["failures"]==0
                assert evaluation["state_isolation_passed"] is True and evaluation["reused"] is False
            else:
                assert payload["evaluation_dir"] is None
            checkpoints.append({"checkpoint":reference,"mode":mode,"seed":seed,"update":done,"state_hashes":payload["state_hashes"],**checked})
        assert value["status"]=="completed" and value["completed_updates"]==stop
        assert value["dev_forward_calls"]==value["full_sentence_forward_calls"]==value["service_changes"]==0
        if mode=="main":
            assert start==0 and stop==1680 and len(value["checkpoints"])==len(value["evaluations"])==4
            assert seed not in identities;identities.add(seed)
            replay=read(HERE/f"replay420_seed{seed}.json")
            assert replay["status"]=="passed" and replay["numerical_state_bitwise_exact"] is True
            check_binding(replay["checkpoint"])
        else:
            assert value["evaluations"]==[]
            assert all(r["nonzero_gradient_tensors"]==r["parameter_tensors"] for r in progress)
        allprogress.extend(progress)
        maxmemory=max(maxmemory,max(r["cuda_peak_reserved_bytes"] for r in progress))
        segments.append({"result":binding(path),"contract":binding(path.parent/"segment_contract.json"),"progress":binding(path.parent/"progress.jsonl"),"executed_updates":len(progress),"start":start,"stop":stop})
    actual=len(allprogress);true=sum(r["true_frames"] for r in allprogress);dense=sum(r["dense_frames"] for r in allprogress);tokens=sum(r["target_tokens"] for r in allprogress)
    assert actual==(8 if mode=="preflight" else 3360)
    assert len(checkpoints)==8
    assert max(r["true_frames"] for r in allprogress)==114 and max(r["dense_frames"] for r in allprogress)==134
    if mode=="main":
        assert identities=={42,43} and (true,dense,tokens)==(269360,315946,152320)
    else:
        for seed in [42,43]:
            terminal=[r["state_hashes"] for r in checkpoints if r["seed"]==seed and r["update"]==2]
            assert len(terminal)==2 and terminal[0]==terminal[1]
    return {"schema":"offline-stage5c-independent-complete-state-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","mode":mode,"producer_imports":0,"network_forward_calls":0,"optimizer_updates_by_audit":0,"execution_plan":binding(planpath),"policy":binding(policypath),"segment_count":len(results),"physical_training_updates":actual,"checkpoints":len(checkpoints),"true_frames":true,"dense_frames":dense,"target_tokens":tokens,"max_true_batch_frames":114,"max_dense_batch_frames":134,"max_cuda_reserved_bytes":maxmemory,"fresh_initial_state_exact":True,"resume_bitwise_exact":True if mode=="preflight" else None,"prior420_bitwise_exact":True if mode=="main" else None,"evaluation_receipts_verified":8 if mode=="main" else 0,"zero_gradient_tensor_update_count":sum(r["nonzero_gradient_tensors"]<r["parameter_tensors"] for r in allprogress),"segments":segments,"checkpoint_details":checkpoints}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--mode",required=True,choices=["preflight","main"]);args=parser.parse_args()
    output=HERE/f"{args.mode}_checkpoint_audit.json";assert not output.exists()
    result=inspect(args.mode);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:result[k] for k in ["status","mode","physical_training_updates","checkpoints","true_frames","dense_frames"]}))


