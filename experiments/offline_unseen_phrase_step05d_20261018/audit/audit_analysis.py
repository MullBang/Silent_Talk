"""Independent saved-output CTC, token, blank and nearest-text diagnostics."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import numpy as np
from audit_common import ctc_nll,collapsed_ids,edit_distance,metric_text,corpus
from audit_sources import read,binding,check_binding,S3,S5C
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def close(a,b):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)

def nearest(text,candidates):
    distances=[edit_distance(metric_text(r["text"]),metric_text(text)) for r in candidates]
    best=min(distances)
    return {"candidate_clip_count":len(candidates),"candidate_unique_texts":len({r["text"] for r in candidates}),"minimum_character_edits":best,"all_tied_nearest_records":[{**r,"character_edit_distance_to_prediction":d} for r,d in zip(candidates,distances) if d==best]}

def inspect():
    path=BASE/"analysis/result_interpretation.json";analysis=read(path)
    for r in analysis["sources"]+[analysis["analysis_code"]]:check_binding(r)
    records=read(S3/"results/cache_handoff.json")["records"]
    allsource={r["clip_id"]:r for r in records}
    train_source={r["clip_id"]:r for r in records if r["split"]=="train" and r["kind"]=="phrase"}
    ordered_ids=[r["clip_id"] for r in read(S5C/"evaluation_plan.json")["items"]]
    assert set(ordered_ids)==set(train_source) and len(ordered_ids)==12
    train=[{k:train_source[c][k] for k in ["clip_id","parent_id","speaker_id","text","T"]} for c in ordered_ids]
    assert train==analysis["train_metadata"] and len(train)==12
    saved={}
    for seed in [42,43]:
        for row in read(BASE/f"runs/seed{seed}/evaluation_summary.json")["rows"]:
            saved[seed,row["clip_id"],row["condition"]]=row
    assert len(saved)==len(analysis["rows"])==18
    maxdiff=0.;seen=set()
    for row in analysis["rows"]:
        key=row["seed"],row["clip_id"],row["condition"];assert key not in seen;seen.add(key)
        original=saved[key];source=allsource[row["clip_id"]]
        assert row["speaker_id"]==source["speaker_id"] and row["completed_update"]==1680
        assert row["reference_text"]==original["reference_text"]==source["text"] and row["prediction_text"]==original["raw_text"] and row["metrics"]==original["metrics"]
        assert row["posterior"]==original["posterior"];check_binding(row["posterior"])
        with np.load(row["posterior"]["path"],allow_pickle=False) as data:lp=data["log_probs"].astype(np.float64)
        raw=lp.argmax(axis=1).tolist();pred=collapsed_ids(raw);target=original["reference_label_ids"]
        assert row["T"]==len(lp) and row["reference_labels"]==target and row["collapsed_labels"]==pred
        assert row["reference_tokens"]==len(target) and row["predicted_tokens"]==len(pred)
        assert row["token_edits"]==edit_distance(target,pred)
        assert row["argmax_blank_frames"]==raw.count(0)
        assert row["ctc_minimum_frames"]==len(target)+sum(a==b for a,b in zip(target,target[1:]))
        targetnll=ctc_nll(lp,target);greedynll=ctc_nll(lp,pred) if pred else -float(lp[:,0].sum())
        values={"predicted_to_reference_tokens":len(pred)/len(target),"argmax_blank_ratio":raw.count(0)/len(lp),"posterior_blank_sum":float(np.exp(lp[:,0]).sum()),"frames_per_target_token":len(lp)/len(target),"target_ctc_nll":targetnll,"target_ctc_nll_per_token":targetnll/len(target),"greedy_ctc_nll":greedynll,"target_minus_greedy_nll":targetnll-greedynll}
        for k,v in values.items():close(row[k],v)
        maxdiff=max(maxdiff,abs(row["target_ctc_nll"]-targetnll),abs(row["greedy_ctc_nll"]-greedynll))
        assert row["target_probability_below_greedy"]==(targetnll-greedynll>1e-9)
        same=[r for r in train if r["speaker_id"]==source["speaker_id"]]
        assert len(same)==1 and row["same_speaker_training_phrases"]==same
        allnear=nearest(row["prediction_text"],train);samenear=nearest(row["prediction_text"],same)
        assert row["nearest_train_all"]==allnear and row["nearest_train_same_speaker"]==samenear
        assert row["nearest_all_contains_same_speaker"]==any(r["speaker_id"]==source["speaker_id"] for r in allnear["all_tied_nearest_records"])
        assert row["exactly_matches_train_text"]==any(metric_text(row["prediction_text"])==metric_text(r["text"]) for r in train)
    assert seen==set(saved)
    for cohort in analysis["cohorts"]:
        rows=[r for r in analysis["rows"] if r["seed"]==cohort["seed"] and r["condition"]==cohort["condition"]]
        assert len(rows)==cohort["count"]==3 and cohort["frames"]==sum(r["T"] for r in rows)==96
        assert cohort["reference_tokens"]==sum(r["reference_tokens"] for r in rows)==64
        assert cohort["metrics"]==corpus([r["metrics"] for r in rows])
        for key in ["predicted_tokens","token_edits"]:assert cohort[key]==sum(r[key] for r in rows)
        floats={"predicted_to_reference_tokens":sum(r["predicted_tokens"] for r in rows)/64,"token_error_rate":sum(r["token_edits"] for r in rows)/64,"argmax_blank_ratio":sum(r["argmax_blank_frames"] for r in rows)/96,"mean_blank_probability":sum(r["posterior_blank_sum"] for r in rows)/96,"target_nll_per_token_micro":sum(r["target_ctc_nll"] for r in rows)/64}
        for k,v in floats.items():close(cohort[k],v)
        for key,rowkey in [("target_probability_below_greedy_count","target_probability_below_greedy"),("exact_train_text_output_count","exactly_matches_train_text"),("nearest_train_contains_same_speaker_count","nearest_all_contains_same_speaker")]:assert cohort[key]==sum(r[rowkey] for r in rows)
        assert cohort["replacement_character_count"]==sum(r["metrics"]["replacement_characters"] for r in rows)
        assert cohort["repetition_ge5_count"]==sum(r["metrics"]["repetition_ge5"] for r in rows)
    for comp in analysis["condition_output_comparisons"]:
        rows={r["condition"]:r for r in analysis["rows"] if r["seed"]==comp["seed"] and r["clip_id"]==comp["clip_id"]}
        assert comp["predictions"]=={c:r["prediction_text"] for c,r in rows.items()}
        assert comp["character_edits"]=={c:r["metrics"]["character_edits"] for c,r in rows.items()}
        for c in ["static","shuffled"]:
            assert comp[c+"_identical_text_to_normal"]==(rows[c]["prediction_text"]==rows["normal"]["prediction_text"])
            assert comp[c+"_identical_tokens_to_normal"]==(rows[c]["collapsed_labels"]==rows["normal"]["collapsed_labels"])
    normal=[r for r in analysis["rows"] if r["condition"]=="normal"]
    assert analysis["normal_predictions"]==normal
    summary=analysis["normal_summary"]
    assert summary["observations"]==6 and summary["independent_dev_clips"]==3 and summary["correct"]==0
    assert summary["target_probability_below_greedy_count"]==sum(r["target_probability_below_greedy"] for r in normal)==6
    assert summary["exact_train_text_output_count"]==sum(r["exactly_matches_train_text"] for r in normal)==0
    assert summary["nearest_train_contains_same_speaker_count"]==sum(r["nearest_all_contains_same_speaker"] for r in normal)==2
    assert summary["replacement_character_count"]==sum(r["metrics"]["replacement_characters"] for r in normal)
    assert summary["mean_seed_CER"]==read(BASE/"results/aggregate_metrics.json")["primary"]["mean_CER"]
    return {"schema":"offline-stage5d-independent-analysis-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","analysis":binding(path),"saved_posteriors":18,"frames":576,"independent_CTC_calculations":36,"maximum_CTC_absolute_difference":maxdiff,"cohorts":6,"nearest_text_candidate_comparisons":18*13,"all_ties_preserved":True,"normal_findings":summary,"nearest_same_speaker_is_privileged_metadata_diagnostic_not_speaker_classifier":True,"producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0,"checkpoint_tensor_loads":0}

if __name__=="__main__":
    result=inspect();output=HERE/"analysis_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"CTC":36,"maxdiff":result["maximum_CTC_absolute_difference"],"normal":result["normal_findings"]}))
