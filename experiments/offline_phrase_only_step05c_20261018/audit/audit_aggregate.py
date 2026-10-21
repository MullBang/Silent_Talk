"""Verify fixed aggregate and last-two-checkpoint gate from independent scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
from audit_sources import read, binding, check_binding

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    auditpath=HERE/"evaluation_audit.json"
    independent=read(auditpath)
    assert independent["status"]=="passed" and independent["evaluation_rows"]==144
    aggregatepath=BASE/"results/aggregate_metrics.json"
    aggregate=read(aggregatepath)
    assert aggregate["status"]=="completed"
    assert aggregate["checkpoint_count"]==8 and aggregate["evaluation_count"]==aggregate["forward_calls"]==144
    assert aggregate["frames"]==5772 and aggregate["failures"]==0
    assert aggregate["dev_evaluated"] is aggregate["generalization_measured"] is aggregate["automatic_sentence_expansion"] is False
    independent_lookup={}
    for detail in independent["details"]:
        check_binding(detail["summary"])
        seed,step=detail["identity"]["seed"],detail["identity"]["completed_update"]
        for cohort,metric in detail["corpus_metrics"].items():
            assert cohort.startswith("train/phrase/")
            condition=cohort.rsplit("/",1)[-1]
            independent_lookup[seed,step,condition]=metric
    assert len(independent_lookup)==12
    assert len(aggregate["diagnostics"])==12
    for row in aggregate["diagnostics"]:
        key=row["seed"],row["completed_update"],row["condition"]
        assert row["metrics"]==independent_lookup[key]
        assert len(row["predictions"])==12
    byseed=[]
    for seed in [42,43]:
        points=[(u,independent_lookup[seed,u,"normal"]) for u in [1260,1680]]
        passed=all(m["count"]==12 and m["exact_count"]==12 and m["CER"]==0 and m["failure_count"]==0 for _,m in points)
        byseed.append({"seed":seed,"passed":passed,"checkpoints":[{"completed_update":u,"metrics":m} for u,m in points]})
    assert aggregate["fit_gate"]["per_seed"]==byseed
    assert aggregate["fit_gate"]["passed"]==all(s["passed"] for s in byseed)
    assert aggregate["fit_gate"]["technical_arbitrary_train_reproduction_gate_only"] is True
    return {"schema":"offline-stage5c-independent-aggregate-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0,"independent_evaluation_audit":binding(auditpath),"aggregate":binding(aggregatepath),"independent_cohorts":12,"evaluations":144,"posterior_frames":5772,"fit_gate":aggregate["fit_gate"],"final_temporal_controls":[r for r in aggregate["diagnostics"] if r["completed_update"]==1680]}

if __name__=="__main__":
    result=inspect();output=HERE/"aggregate_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"cohorts":result["independent_cohorts"],"fit_gate_passed":result["fit_gate"]["passed"]}))
