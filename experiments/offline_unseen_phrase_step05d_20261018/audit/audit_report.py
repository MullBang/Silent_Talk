"""Check resource and user-facing summary against independently audited results."""
from datetime import datetime,timezone
import json
from pathlib import Path
from audit_sources import read,binding,check_binding
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    for name in ["execution_audit","evaluation_audit","aggregate_audit","final_source_preservation"]:assert read(HERE/f"{name}.json")["status"]=="passed"
    rp=BASE/"results/resource_accounting.json";sp=BASE/"results/stage5d_summary.json"
    resource,summary=read(rp),read(sp)
    execution=read(BASE/"results/execution_summary.json");aggregate=read(BASE/"results/aggregate_metrics.json")
    evals=[read(BASE/f"runs/seed{s}/evaluation_summary.json") for s in [42,43]]
    expected={"neural_forward_calls":18,"evaluation_frames":576,"unique_phrase_clips":3,"unique_phrase_frames":96,"optimizer_updates":0,"optimizer_objects_created":0,"new_checkpoints":0,"main_checkpoint_tensor_loads":2,"evaluation_failures":0,"posterior_files":18,"train_forward_calls":0,"full_sentence_forward_calls":0,"source_media_reads":0,"audio_inputs":0,"teacher_calls":0,"downloads":0,"service_changes":0}
    for k,v in expected.items():assert resource[k]==v,(k,resource[k],v)
    assert resource["posterior_bytes"]==sum(r["posterior"]["bytes"] for e in evals for r in e["rows"])
    assert resource["execution_seconds"]==execution["seconds"]
    assert resource["evaluation_seconds_sum"]==sum(e["seconds"] for e in evals)
    assert resource["maximum_allocated_cuda_bytes"]==max(e["peak_allocated_cuda_bytes"] for e in evals)
    assert resource["maximum_reserved_cuda_bytes"]==max(e["peak_reserved_cuda_bytes"] for e in evals)
    assert summary["primary"]==aggregate["primary"] and summary["all_conditions"]==aggregate["diagnostics"]
    assert summary["neural_forward_calls"]==18 and summary["optimizer_updates"]==summary["new_checkpoints"]==summary["service_changes"]==0
    assert summary["unique_dev_phrase_clips"]==summary["known_speakers"]==3
    for k in ["new_speaker_or_webcam_evaluation","generalization_established","never_used_in_project_history","source_training_ready","prior5c_strict_fit_gate_passed","automatic_sentence_expansion"]:assert summary[k] is False,k
    for k in ["exploratory_reused_development_probe","new_in_this_model_training","source_provisional"]:assert summary[k] is True,k
    for k in ["execution","aggregate","resources"]:check_binding(summary[k])
    assert not list(BASE.rglob("*.pt")),"No new model checkpoint allowed"
    return {"schema":"offline-stage5d-independent-report-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","resource_accounting":binding(rp),"summary":binding(sp),"primary":aggregate["primary"],"checks":{"physical_neural_calls18":True,"optimizer_and_new_checkpoints_zero":True,"resource_counts_times_memory_bytes_match":True,"exploratory_reused_known_speaker_limits_preserved":True,"prior_fit_gate_remains_failed":True},"producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0}

if __name__=="__main__":
    result=inspect();output=HERE/"report_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"primary_mean_CER":result["primary"]["mean_CER"],"exact":result["primary"]["exact_observations"]}))
