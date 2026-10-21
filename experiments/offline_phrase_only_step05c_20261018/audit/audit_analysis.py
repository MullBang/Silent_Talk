"""Independently verify saved posterior/blank/token/CTC and speaker diagnostics."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import itertools
import json
import math
from pathlib import Path
import statistics
import numpy as np
from audit_common import ctc_nll,collapsed_ids,edit_distance,score,corpus
from audit_sources import read,binding,check_binding,S3
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def close(a,b):
    assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-11),(a,b)

def inspect():
    path=BASE/"analysis/result_interpretation.json";analysis=read(path)
    for r in analysis["sources"]+[analysis["source_code"]]:check_binding(r)
    source={r["clip_id"]:r for r in read(S3/"results/cache_handoff.json")["records"] if r["split"]=="train" and r["kind"]=="phrase"}
    saved={}
    for p in sorted((BASE/"runs").glob("**/evaluation_summary.json")):
        evaluation=read(p)
        for row in evaluation["rows"]:
            key=row["identity"]["seed"],row["identity"]["completed_update"],row["clip_id"],row["condition"]
            assert key not in saved;saved[key]=row
    assert len(saved)==len(analysis["rows"])==144
    maxdiff=0.;recomputed=[];seen=set()
    for row in analysis["rows"]:
        key=row["seed"],row["completed_update"],row["clip_id"],row["condition"]
        assert key not in seen;seen.add(key)
        original=saved[key];record=source[row["clip_id"]]
        assert row["reference_text"]==record["text"]==original["reference_text"]
        assert row["parent_id"]==record["parent_id"] and row["speaker_id"]==record["speaker_id"]
        assert row["prediction_text"]==original["raw_text"] and row["metrics"]==original["metrics"]
        assert row["posterior"]==original["posterior"];check_binding(row["posterior"])
        with np.load(row["posterior"]["path"],allow_pickle=False) as d:lp=d["log_probs"].astype(np.float64)
        raw=lp.argmax(axis=1).tolist();collapsed=collapsed_ids(raw);target=original["reference_label_ids"]
        assert row["collapsed_labels"]==collapsed and row["reference_labels"]==target
        assert row["T"]==len(lp)==record["T"] and row["reference_tokens"]==len(target) and row["predicted_tokens"]==len(collapsed)
        assert row["token_edits"]==edit_distance(target,collapsed)
        assert row["ctc_minimum_frames"]==len(target)+sum(a==b for a,b in zip(target,target[1:]))
        assert row["argmax_blank_frames"]==raw.count(0)
        assert row["maximum_collapsed_same_token_run"]==max((len(list(g)) for _,g in itertools.groupby(collapsed)),default=0)
        blank=float(np.exp(lp[:,0]).sum())
        targetnll=ctc_nll(lp,target)
        greedynll=ctc_nll(lp,collapsed) if collapsed else -float(lp[:,0].sum())
        checks={"predicted_to_reference_tokens":len(collapsed)/len(target),"frames_per_target_token":len(lp)/len(target),"argmax_blank_ratio":raw.count(0)/len(lp),"posterior_blank_sum":blank,"mean_blank_probability":blank/len(lp),"reference_ctc_nll":targetnll,"reference_ctc_nll_per_token":targetnll/len(target),"greedy_ctc_nll":greedynll,"target_minus_greedy_nll":targetnll-greedynll}
        for name,value in checks.items():close(row[name],value)
        maxdiff=max(maxdiff,abs(targetnll-row["reference_ctc_nll"]),abs(greedynll-row["greedy_ctc_nll"]))
        recomputed.append({**row,**checks})
    assert seen==set(saved)
    groups=defaultdict(list)
    for row in recomputed:groups[row["seed"],row["completed_update"],row["condition"]].append(row)
    assert len(groups)==len(analysis["cohorts"])==12
    for cohort in analysis["cohorts"]:
        rows=groups[cohort["seed"],cohort["completed_update"],cohort["condition"]]
        assert len(rows)==cohort["count"]==12 and sum(r["T"] for r in rows)==cohort["frames"]==481
        assert sum(r["reference_tokens"] for r in rows)==cohort["reference_tokens"]==272
        assert cohort["metrics"]==corpus([r["metrics"] for r in rows])
        for key in ["predicted_tokens","token_edits","argmax_blank_frames"]:assert cohort[key]==sum(r[key] for r in rows)
        assert cohort["replacement_characters"]==sum(r["metrics"]["replacement_characters"] for r in rows)
        assert cohort["repetition_ge5_count"]==sum(r["metrics"]["repetition_ge5"] for r in rows)
        checks={"token_error_rate":sum(r["token_edits"] for r in rows)/272,"predicted_to_reference_tokens":sum(r["predicted_tokens"] for r in rows)/272,"argmax_blank_ratio":sum(r["argmax_blank_frames"] for r in rows)/481,"mean_blank_probability":sum(r["posterior_blank_sum"] for r in rows)/481,"target_nll_per_token_micro":sum(r["reference_ctc_nll"] for r in rows)/272,"target_nll_per_token_clip_mean":statistics.fmean(r["reference_ctc_nll_per_token"] for r in rows)}
        for key,value in checks.items():close(cohort[key],value)
        assert cohort["target_probability_below_greedy_count"]==sum(r["target_minus_greedy_nll"]>1e-9 for r in rows)
        assert cohort["target_probability_above_greedy_count"]==sum(r["target_minus_greedy_nll"]<-1e-9 for r in rows)
        assert cohort["target_probability_tied_greedy_count"]==sum(abs(r["target_minus_greedy_nll"])<=1e-9 for r in rows)
    window_count=0
    for curve in analysis["loss_curves"]:
        logs=[json.loads(line) for line in (BASE/f"runs/seed{curve['seed']}/progress.jsonl").read_text(encoding="utf8").splitlines()]
        assert len(logs)==curve["count"]==curve["finite_losses"]==1680
        assert len(curve["windows"])==28
        for i,window in enumerate(curve["windows"]):
            rows=logs[i*60:(i+1)*60];loss=[r["loss"] for r in rows]
            assert window["first_update"]==i*60+1 and window["last_update"]==(i+1)*60
            close(window["loss_mean"],statistics.fmean(loss))
            assert window["loss_minimum"]==min(loss) and window["loss_maximum"]==max(loss)
            assert window["learning_rates"]==sorted({r["learning_rate"] for r in rows})
            assert window["gradient_clipped_updates"]==sum(r["gradient_norm_before_clip"]>1 for r in rows)
            window_count+=1
        close(curve["last60_loss_mean"],statistics.fmean(r["loss"] for r in logs[-60:]))
        close(curve["previous60_loss_mean"],statistics.fmean(r["loss"] for r in logs[-120:-60]))
        assert curve["updates_with_zero_gradient_tensor"]==sum(r["nonzero_gradient_tensors"]<r["parameter_tensors"] for r in logs)
    byspeaker=defaultdict(list);byt=defaultdict(list)
    for clip in source.values():byspeaker[clip["speaker_id"]].append(clip);byt[clip["T"]].append(clip)
    assert analysis["dataset_counts"]=={"clips":12,"unique_texts":len({r["text"] for r in source.values()}),"speakers":len(byspeaker),"distinct_T":len(byt),"speakers_with_single_phrase":sum(len(r)==1 for r in byspeaker.values()),"clips_with_unique_T":sum(len(r)==1 for r in byt.values())}
    final={(r["seed"],r["clip_id"],r["condition"]):r for r in recomputed if r["completed_update"]==1680}
    singles={k for k,v in byspeaker.items() if len(v)==1};assert len(singles)==4
    single_counts=[]
    for speaker in analysis["speaker_distribution"]:
        clips=byspeaker[speaker["speaker_id"]]
        assert speaker["phrase_count"]==len(clips) and speaker["unique_texts"]==len({r["text"] for r in clips})
        assert {r["clip_id"] for r in speaker["clips"]}=={r["clip_id"] for r in clips}
        for item in speaker["final_by_seed"]:
            expected={c:sum(final[item["seed"],r["clip_id"],c]["metrics"]["exact"] for r in clips) for c in ["normal","static","shuffled"]}
            assert item["exact_counts"]==expected
    for seed in [42,43]:
        counts={c:sum(final[seed,r["clip_id"],c]["metrics"]["exact"] for r in source.values() if r["speaker_id"] in singles) for c in ["normal","static","shuffled"]}
        assert counts=={"normal":4,"static":4,"shuffled":4}
        single_counts.append({"seed":seed,"exact_counts_among_single_phrase_speakers":counts})
    for item in analysis["final_retention_by_clip"]:
        clip=source[item["clip_id"]];chosen={c:final[item["seed"],item["clip_id"],c] for c in ["normal","static","shuffled"]}
        assert item["speaker_id"]==clip["speaker_id"] and item["phrases_for_speaker"]==len(byspeaker[clip["speaker_id"]])
        assert item["exact_by_condition"]=={c:r["metrics"]["exact"] for c,r in chosen.items()}
        assert item["prediction_by_condition"]=={c:r["prediction_text"] for c,r in chosen.items()}
        assert item["all_three_exact"]==all(r["metrics"]["exact"] for r in chosen.values())
        assert item["static_equals_normal_output"]==(chosen["static"]["prediction_text"]==chosen["normal"]["prediction_text"])
        assert item["shuffled_equals_normal_output"]==(chosen["shuffled"]["prediction_text"]==chosen["normal"]["prediction_text"])
    failures=[r for r in analysis["rows"] if r["condition"]=="normal" and r["completed_update"] in [1260,1680] and not r["metrics"]["exact"]]
    assert failures==analysis["gate_failure_rows"] and len(failures)==1
    failed=failures[0]
    assert failed["seed"]==43 and failed["completed_update"]==1260 and failed["clip_id"]=="lip_J_1_F_02_C032_A_002_030__phrase"
    assert failed["reference_text"]=="너무 많이 먹고 와" and failed["prediction_text"]=="너무 마이 먹고 와"
    assert failed["target_minus_greedy_nll"]<0
    return {"schema":"offline-stage5c-independent-saved-analysis-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","analysis":binding(path),"rows":144,"posterior_frames":5772,"independent_CTC_calculations":288,"maximum_CTC_absolute_difference":maxdiff,"cohorts":12,"loss_windows":window_count,"speaker_groups":8,"single_phrase_speakers":sorted(singles),"single_phrase_control_counts":single_counts,"unique_gate_error":{"seed":43,"update":1260,"clip_id":failed["clip_id"],"reference":failed["reference_text"],"prediction":failed["prediction_text"],"target_minus_greedy_nll":failed["target_minus_greedy_nll"],"target_sequence_probability_above_greedy":True,"interpretation":"Greedy decoding is not maximum sequence probability for this particular saved output; this does not establish that a specific beam decoder will recover it or that generalization improves."},"producer_imports":0,"new_network_forward_calls":0,"optimizer_updates":0,"checkpoint_tensor_loads":0}

if __name__=="__main__":
    result=inspect();output=HERE/"analysis_audit.json";assert not output.exists()
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"status":result["status"],"CTC_checks":result["independent_CTC_calculations"],"max_diff":result["maximum_CTC_absolute_difference"],"loss_windows":result["loss_windows"]}))
