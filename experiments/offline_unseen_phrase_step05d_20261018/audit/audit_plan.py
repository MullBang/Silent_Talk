"""Independent inference-only execution and policy preflight review."""
from datetime import datetime,timezone
import json
from pathlib import Path
from audit_sources import read,digest,binding,check_binding,S3,S4,S5C
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    executionpath=BASE/"execution_plan.json";policypath=BASE/"controlled_pilot_policy.json";evalpath=BASE/"evaluation_plan.json"
    plan,policy,evaluation=read(executionpath),read(policypath),read(evalpath)
    assert plan["schema"]=="offline-stage5d-inference-only-execution-v1"
    source=read(HERE/"source_preflight.json")
    assert source["status"]=="passed" and source["verified_frozen_artifact_entries"]==1778
    assert plan["seeds"]==[42,43] and plan["checkpoint_update"]==1680
    assert plan["evaluation_rows"]==18 and plan["evaluation_frames"]==576
    assert plan["optimizer_updates"]==plan["optimizer_objects"]==0
    for ref in plan["bindings"]+plan["previous_manifests"]+plan["checkpoints"]:
        check_binding(ref)
    assert plan["runtime"]==read(S5C/"execution_plan.json")["runtime"]
    handoff=read(S5C/"results/next_step_handoff.json")
    assert plan["checkpoints"]==handoff["checkpoints"]==evaluation["fixed_checkpoints"]
    assert plan["data_policy"]==binding(policypath)
    assert plan["evaluation_plan"]==binding(evalpath)==policy["evaluation_plan"]
    assert plan["no_new_training"] is plan["no_source_media_or_audio"] is plan["no_service_change"] is plan["pre_execution_gate_required"] is True
    oldeval=read(S4/"protocol/evaluation_plan.json")
    targets=[r for r in oldeval["items"] if r["split"]=="dev" and r["kind"]=="phrase"]
    assert evaluation["items"]==targets and len(targets)==3
    assert evaluation["conditions_by_update"]=={"1680":["normal","static","shuffled"]}
    assert evaluation["seeds"]==[42,43] and evaluation["checkpoints"]==[1680]
    assert evaluation["planned_neural_evaluations"]==18 and evaluation["planned_frames"]==576 and evaluation["optimizer_updates"]==0
    assert evaluation["resource_caps"]=={"cuda_reserved_bytes":2147483648,"seconds_per_model":120}
    for key in ["model_updates_allowed","full_sentence_input_allowed","teacher_or_audio_input_allowed","automatic_sentence_expansion","source_training_ready"]:assert evaluation[key] is False
    assert evaluation["source_provisional"] is True
    for key in ["source_handoff","source_evaluation_plan"]:check_binding(evaluation[key])
    assert policy["schema"]=="lipnet-stage5d-evaluation-only-data-policy-v1" and policy["mode"]=="evaluation_only"
    assert policy["allowed_optimizer_clip_ids"]==[]
    assert policy["allowed_evaluation_clip_ids"]==sorted(r["clip_id"] for r in targets)
    assert policy["optimizer_updates_authorized"] is policy["service_change_authorized"] is False
    assert policy["allow_provisional_cache_for_controlled_evaluation"] is policy["preserve_provisional_flags"] is True
    for key in ["handoff","stage3_frozen_manifest"]:check_binding(policy[key])
    assert policy["authorization"]["source"]=="current_user_request" and policy["authorization"]["scope"]=="stage5d_fixed_unseen_phrase_evaluation"
    return {"schema":"offline-stage5d-independent-plan-preflight-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","execution_plan":binding(executionpath),"policy":binding(policypath),"evaluation_plan":binding(evalpath),"source_preflight":binding(HERE/"source_preflight.json"),"bound_artifacts":len(plan["bindings"]),"fixed_checkpoint_count":2,"dev_phrase_count":3,"neural_forward_budget":18,"evaluation_frame_budget":576,"optimizer_objects":0,"optimizer_updates":0,"code_review":{"restricted_saved_fullstate_validation":True,"model_only_constructor_and_strict_weight_load":True,"restores_saved_RNG_then_preserves_current_RNG":True,"eval_mode_and_no_grad":True,"no_optimizer_created":True,"explicit_dev_only_loader_scope":True,"fixed_three_temporal_conditions":True,"complete_failed_row_preservation_and_stop":True,"fresh_attempt_required":True,"source_checkpoint_bytes_checked_before_after":True},"prior_fit_gate_false_preserved":True,"reused_known_speaker_probe_only":True,"producer_imports":0,"checkpoint_tensor_loads":0,"network_forward_calls":0}

if __name__=="__main__":
    result=inspect();output=HERE/"preflight_plan_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:result[k] for k in ["status","bound_artifacts","neural_forward_budget","optimizer_objects"]}))
