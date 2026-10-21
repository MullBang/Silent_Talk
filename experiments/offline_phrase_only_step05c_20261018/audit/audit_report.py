"""Cross-check reported accounting and conclusions with audited evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
from audit_sources import read,binding,check_binding
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    main=read(HERE/"main_checkpoint_audit.json")
    pre=read(HERE/"preflight_checkpoint_audit.json")
    evaluation=read(HERE/"evaluation_audit.json")
    agg=read(HERE/"aggregate_audit.json")
    source=read(HERE/"final_source_preservation.json")
    assert all(v["status"]=="passed" for v in [main,pre,evaluation,agg,source])
    resourcepath=BASE/"results/resource_accounting.json";summarypath=BASE/"results/stage5c_summary.json"
    resources,summary=read(resourcepath),read(summarypath)
    expected={"main_optimizer_updates":main["physical_training_updates"],"preflight_optimizer_updates":pre["physical_training_updates"],"total_physical_optimizer_updates":3368,"main_neural_forward_calls":3360,"evaluation_neural_forward_calls":144,"preflight_neural_forward_calls":8,"total_neural_forward_calls":3512,"main_actual_frames":main["true_frames"],"main_dense_frames":main["dense_frames"],"main_padding_frames":main["dense_frames"]-main["true_frames"],"main_target_tokens":main["target_tokens"],"preflight_actual_frames":pre["true_frames"],"preflight_dense_frames":pre["dense_frames"],"evaluation_frames":evaluation["posterior_frames"],"main_checkpoints":main["checkpoints"],"preflight_checkpoints":pre["checkpoints"],"posterior_files":evaluation["posteriors"],"updates_with_zero_gradient_tensor":main["zero_gradient_tensor_update_count"],"evaluation_failures":evaluation["failed_rows"]}
    for key,value in expected.items():
        assert resources[key]==value,(key,resources[key],value)
    rows=[];segments=[]
    for item in main["segments"]:
        check_binding(item["progress"]);check_binding(item["result"])
        rows.extend(json.loads(line) for line in Path(item["progress"]["path"]).read_text(encoding="utf8").splitlines())
        segments.append(read(item["result"]["path"]))
    assert resources["main_optimizer_seconds"]==sum(r["seconds"] for r in rows)
    assert resources["maximum_training_cuda_allocated_bytes"]==max(r["cuda_peak_allocated_bytes"] for r in rows)
    assert resources["maximum_training_cuda_reserved_bytes"]==max(r["cuda_peak_reserved_bytes"] for r in rows)
    evals=[read(r["summary"]["path"]) for r in evaluation["details"]]
    assert resources["evaluation_seconds_sum"]==sum(r["seconds"] for r in evals)
    assert resources["maximum_evaluation_cuda_allocated_bytes"]==max(r["peak_allocated_cuda_bytes"] for r in evals)
    assert resources["maximum_evaluation_cuda_reserved_bytes"]==max(r["peak_reserved_cuda_bytes"] for r in evals)
    assert resources["per_run_segment_seconds_sum"]==sum(r["elapsed_seconds"] for r in segments)
    assert resources["main_execution_wall_seconds"]==read(BASE/"results/execution_summary.json")["elapsed_seconds"]
    assert resources["checkpoint_bytes"]==sum(Path(r["checkpoint"]["path"]).stat().st_size for audit in [main,pre] for r in audit["checkpoint_details"])
    assert resources["posterior_bytes"]==sum(r["posterior"]["bytes"] for e in evals for r in e["rows"])
    aggregate=read(BASE/"results/aggregate_metrics.json")
    assert summary["fit_gate"]==agg["fit_gate"]==aggregate["fit_gate"]
    assert summary["normal_trajectory"]==[r for r in aggregate["diagnostics"] if r["condition"]=="normal"]
    assert summary["final_temporal_controls"]==[r for r in aggregate["diagnostics"] if r["completed_update"]==1680]
    assert summary["training_parents"]==12 and summary["training_unique_phrase_texts"]==11 and summary["historical_speakers"]==8
    assert summary["data_train_only"] is summary["source_provisional"] is True
    assert summary["new_sentence_generalization_measured"] is summary["source_training_ready"] is summary["precise_visual_boundary_approval"] is summary["automatic_sentence_expansion"] is False
    assert summary["train_phrase_boundary_flag_count"]==1 and summary["original_all30_boundary_flag_count"]==2
    for key in ["resource_accounting","execution","aggregate"]:
        check_binding(summary[key])
    return {"schema":"offline-stage5c-independent-report-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","resources":binding(resourcepath),"summary":binding(summarypath),"audits":[binding(HERE/name) for name in ["main_checkpoint_audit.json","preflight_checkpoint_audit.json","evaluation_audit.json","aggregate_audit.json","final_source_preservation.json"]],"physical_updates":3368,"neural_forward_calls":3512,"fixed_gate_passed":False,"source_flags_preserved":True,"new_network_forward_calls":0,"optimizer_updates":0}

if __name__=="__main__":
    result=inspect();output=HERE/"report_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"updates":result["physical_updates"],"forwards":result["neural_forward_calls"]}))
