"""Independent CPU checkpoint and inference-state receipt verification."""
from datetime import datetime,timezone
import json
from pathlib import Path
import torch
from audit_common import unpack,state_hashes,fingerprint
from audit_sources import read,binding,check_binding,S5C
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    torch.set_num_threads(4)
    path=BASE/"results/execution_summary.json";summary=read(path)
    assert summary["status"]=="completed"
    assert summary["neural_forward_calls"]==18 and summary["evaluation_frames"]==576 and summary["failed_rows"]==0
    for key in ["optimizer_updates","optimizer_objects","train_forward_calls","full_sentence_forward_calls","source_media_reads","audio_inputs","teacher_calls","downloads","service_changes"]:assert summary[key]==0,key
    assert summary["checkpoint_tensor_loads"]==2
    plan=read(BASE/"execution_plan.json");policy=read(BASE/"controlled_pilot_policy.json")
    assert summary["plan"]==binding(BASE/"execution_plan.json") and summary["policy"]==binding(BASE/"controlled_pilot_policy.json")
    for b in plan["bindings"]+plan["previous_manifests"]+plan["checkpoints"]:check_binding(b)
    sources=read(S5C/"results/next_step_handoff.json")["checkpoints"]
    checks=[]
    assert len(summary["runs"])==2
    for seed,receipt,source in zip([42,43],summary["runs"],sources):
        receiptpath=BASE/f"results/seed{seed}_receipt.json";assert read(receiptpath)==receipt
        assert receipt["seed"]==seed and receipt["checkpoint"]==source
        check_binding(source)
        payload=unpack(torch.load(source["path"],map_location="cpu",weights_only=True))
        assert payload["schema"]=="offline-stage5c-complete-checkpoint-v1" and payload["completed_updates"]==1680
        assert payload["identity"]["seed"]==seed and payload["identity"]["mode"]=="main" and payload["identity"]["arm"]=="P_phrase"
        assert payload["identity"]["runtime"]==plan["runtime"]
        hashes=state_hashes(payload["state"])
        assert hashes==payload["state_hashes"]==receipt["complete_saved_state_hashes"]
        assert receipt["loaded_model_hash"]==receipt["after_model_hash"]==hashes["model"]
        assert receipt["rng_before"]==receipt["rng_after"]==hashes["rng"]
        assert sum(t.numel() for t in payload["state"]["model"].values())==3287781
        assert len(payload["state"]["model"])==31 and all(torch.isfinite(t).all() for t in payload["state"]["model"].values())
        assert all(float(r["step"])==1680 for r in payload["state"]["optimizer"]["state"].values())
        assert receipt["optimizer_objects_created"]==receipt["optimizer_updates"]==0
        evaluation=receipt["evaluation"];check_binding(evaluation["summary"])
        assert evaluation["count"]==evaluation["forward_calls"]==evaluation["new_forward_calls"]==9 and evaluation["frames"]==288
        assert evaluation["failures"]==0 and evaluation["reused"] is False and evaluation["state_isolation_passed"] is True
        identity=evaluation["identity"]
        assert identity["seed"]==seed and identity["mode"]=="evaluation_only" and identity["arm"]=="P_phrase" and identity["completed_update"]==1680
        assert identity["checkpoint"]==source and identity["runtime"]==plan["runtime"]
        checks.append({"seed":seed,"receipt":binding(receiptpath),"checkpoint":source,"saved_complete_state_hashes":hashes,"loaded_and_after_model_exact":True,"saved_RNG_before_after_exact":True,"Adam_only_historical_saved_tensors_no_optimizer_object":True})
    return {"schema":"offline-stage5d-independent-execution-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","execution_summary":binding(path),"plan":binding(BASE/"execution_plan.json"),"model_checks":checks,"frozen_checkpoints":2,"neural_forwards_observed":18,"evaluation_frames":576,"optimizer_objects":0,"optimizer_updates":0,"producer_imports":0,"network_forward_calls_by_audit":0,"CPU_checkpoint_tensor_loads_by_audit":2}

if __name__=="__main__":
    result=inspect();output=HERE/"execution_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"models":result["frozen_checkpoints"],"neural_forwards":result["neural_forwards_observed"],"optimizer_updates":0}))
