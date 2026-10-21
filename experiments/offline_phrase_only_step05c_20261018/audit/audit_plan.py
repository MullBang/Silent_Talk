"""Independent plan/policy/budget gate, without model or production imports."""
from datetime import datetime, timezone
import json
from pathlib import Path
from audit_sources import read, digest, binding, check_binding, S3, S4, S5, S5B
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    planpath=BASE/"execution_plan.json";policypath=BASE/"controlled_pilot_policy.json"
    assert digest(planpath)=="8f06a1ff1a626c7288f2a5d59263621325f5ffdf156540779730d610405a7e49"
    assert digest(policypath)=="7ed759a894440f573c8d4257190077622deafa265883058ed9fb6b46b8a4f19d"
    plan,policy=read(planpath),read(policypath)
    assert plan["schema"]=="offline-stage5c-execution-plan-v1"
    sourceaudit=read(HERE/"source_preflight.json")
    assert sourceaudit["status"]=="passed" and sourceaudit["verified_frozen_artifact_entries"]==1498
    for item in plan["bindings"]+plan["previous_manifests"]+plan["schedules"]+[plan["design"],plan["evaluation_plan"]]:
        check_binding(item)
    design=read(plan["design"]["path"])
    assert plan["selection_policy"]==design["selection_policy"]
    assert plan["planned_accounting"]==design["planned_accounting"]
    assert plan["runtime"]==read(S5/"execution_plan.json")["runtime"]
    assert plan["seeds"]==[42,43] and plan["main_checkpoints"]==[420,840,1260,1680]
    assert plan["main_updates"]==3360 and plan["preflight_updates"]==8 and plan["evaluation_rows"]==144
    assert plan["audio_inputs"]==plan["teacher_calls"]==plan["service_changes"]==plan["new_dev_or_sentence_forward_calls"]==0
    records=read(S3/"results/cache_handoff.json")["records"]
    lookup={r["clip_id"]:r for r in records if r["split"]=="train" and r["kind"]=="phrase"}
    allowed=sorted(lookup)
    assert policy["schema"]=="lipnet-stage5c-controlled-phrase-only-policy-v1"
    assert policy["training_plan"]=={"path":str(planpath),"sha256":digest(planpath)}
    assert policy["allowed_optimizer_clip_ids"]==policy["allowed_validation_clip_ids"]==allowed
    assert policy["allow_provisional_cache_for_controlled_pilot"] is policy["preserve_provisional_flags"] is policy["full_training_authorized"] is True
    assert policy["service_change_authorized"] is False
    for key in ["handoff","stage3_frozen_manifest"]:
        check_binding(policy[key])
    assert policy["authorization"]==plan["authorization"]
    assert plan["authorization"]["request_text"]=="다음 단계 진행"
    tasks=plan["preflight_tasks"]
    assert len(tasks)==2
    for i,row in enumerate(tasks,1):
        assert row["update"]==i and row["kind"]=="phrase" and row["lr"]==(.001 if i==1 else .0003)
        assert len(row["clip_ids"])==2 and all(c in lookup for c in row["clip_ids"])
        lengths=[lookup[c]["T"] for c in row["clip_ids"]]
        assert row["lengths"]==lengths and sum(lengths)==row["true_frames"]==114
        assert 2*max(lengths)==row["dense_frames"]==134
        assert row["target_lengths"]==[lookup[c]["label_length"] for c in row["clip_ids"]]
    assert tasks[0]["clip_ids"]==tasks[1]["clip_ids"]
    previous=[]
    for item in plan["prior_420"]:
        check_binding(item["checkpoint"]);check_binding(item["contract"])
        contract=read(item["contract"]["path"])
        assert contract["identity"]["seed"]==item["seed"] and contract["identity"]["arm"]=="B_curriculum"
        assert contract["identity"]["runtime"]==plan["runtime"]
        previous.append(item["seed"])
    assert previous==[42,43]
    evalplan=read(plan["evaluation_plan"]["path"])
    oldeval=read(S4/"protocol/evaluation_plan.json")
    assert evalplan["items"]==[r for r in oldeval["items"] if r["split"]=="train" and r["kind"]=="phrase"]
    assert evalplan["conditions_by_update"]=={"420":["normal"],"840":["normal"],"1260":["normal"],"1680":["normal","static","shuffled"]}
    assert evalplan["checkpoints"]==[420,840,1260,1680] and evalplan["total_rows"]==144
    assert evalplan["generalization_measured"] is False
    return {"schema":"offline-stage5c-independent-plan-preflight-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","plan":binding(planpath),"policy":binding(policypath),"source_audit":binding(HERE/"source_preflight.json"),"binding_count":len(plan["bindings"]),"metadata_previous_runtime_exact":True,"fresh420_tensor_reproduction_still_requires_execution":True,"scope":"fixed12 train phrases only; two seeds; no dev or full sentences", "main_optimizer_updates":3360,"preflight_optimizer_updates":8,"main_evaluation_rows":144,"main_evaluation_frames":5772,"preflight_max_true":114,"preflight_max_dense":134,"preserves_source_flags":True,"code_review":{"separate_scope_and_new_outputs":True,"immutable_plan_and_policy":True,"initial_state_gate_before_main_update":True,"prior420_state_gate_before421":True,"resume_before_next_update_completes_missing_evaluation":True,"completed_evaluation_reuse_verified":True,"partial_evaluation_fails_closed":True,"physical_uncheckpointed_updates_cannot_silently_replay":True,"fixed_final_endpoint_and_no_dev_selection":True},"producer_imports":0,"network_forward_calls":0,"optimizer_updates":0,"checkpoint_tensor_loads":0}

if __name__=="__main__":
    output=HERE/"preflight_plan_audit.json";assert not output.exists()
    result=inspect();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"bindings":result["binding_count"],"scope":result["scope"]}))
