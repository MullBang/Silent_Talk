"""Independent aggregate audit for the fixed exploratory dev3 probe."""
from datetime import datetime,timezone
import json
from pathlib import Path
from audit_sources import read,binding,check_binding
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def inspect():
    prior=read(HERE/"evaluation_audit.json");assert prior["status"]=="passed"
    path=BASE/"results/aggregate_metrics.json";aggregate=read(path)
    assert aggregate["status"]=="completed" and aggregate["checkpoint_count"]==2
    assert aggregate["evaluation_count"]==aggregate["forward_calls"]==18 and aggregate["frames"]==576 and aggregate["failures"]==0
    assert aggregate["unique_dev_phrase_count"]==3 and aggregate["optimizer_updates"]==0
    assert aggregate["exploratory_reused_development_probe"] is aggregate["prior5c_strict_fit_gate_remains_failed"] is True
    assert aggregate["generalization_established"] is aggregate["new_speaker_or_webcam_evaluation"] is aggregate["automatic_sentence_expansion"] is False
    lookup={}
    for detail in prior["details"]:
        check_binding(detail["summary"])
        for cohort,metrics in detail["corpus_metrics"].items():
            lookup[detail["identity"]["seed"],cohort.rsplit("/",1)[-1]]=metrics
    assert len(lookup)==len(aggregate["diagnostics"])==6
    for row in aggregate["diagnostics"]:
        assert row["completed_update"]==1680 and row["metrics"]==lookup[row["seed"],row["condition"]]
        assert len(row["predictions"])==3
    primary=aggregate["primary"]
    assert primary["per_seed"]==[r for r in aggregate["diagnostics"] if r["condition"]=="normal"]
    assert primary["mean_CER"]==sum(lookup[s,"normal"]["CER"] for s in [42,43])/2
    assert primary["mean_WER"]==sum(lookup[s,"normal"]["WER"] for s in [42,43])/2
    assert primary["exact_observations"]==sum(lookup[s,"normal"]["exact_count"] for s in [42,43])
    assert primary["observations"]==6 and primary["independent_utterances"]==3
    return {"schema":"offline-stage5d-independent-aggregate-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","aggregate":binding(path),"independent_evaluation_audit":binding(HERE/"evaluation_audit.json"),"cohorts":6,"evaluation_rows":18,"frames":576,"primary":primary,"producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0,"reused_known_speaker_probe_only":True,"strict_prior_fit_gate_still_false":True}

if __name__=="__main__":
    result=inspect();output=HERE/"aggregate_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"cohorts":6,"mean_CER":result["primary"]["mean_CER"],"exact":result["primary"]["exact_observations"]}))
