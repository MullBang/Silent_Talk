"""Independent audit of the separately frozen 48kHz interpolation follow-up.

Reads saved workspace PCM only. Does not import either producer, load a model,
read source media, or change original 8kHz decisions. Full search numerators and
both sliding moments use FFT convolution; winning/runner-up Pearson is direct.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt

# Only the independently written local provenance helpers are shared. Neither
# audit_sources.py, align_step04_audio.py nor refine_alignment.py is imported.
from independent_audit import BASE, check_binding, close, pcm, read, require, sha

INPUT_RATE, RATE = 8000, 48000
SIGNAL_ONLY_REASON = "At least one strict anchor gate or constant-offset range failed"


def independent_score(reference: np.ndarray, search: np.ndarray) -> dict:
    require(reference.ndim == search.ndim == 1 and len(search)>len(reference)>100, "Invalid PCM shape")
    require(np.isfinite(reference).all() and np.isfinite(search).all(), "Nonfinite PCM")
    sos = butter(4, [100,3400], btype="bandpass", fs=INPUT_RATE, output="sos")
    a = resample_poly(sosfiltfilt(sos,reference), 6,1,window=("kaiser",5.0),padtype="constant")
    b = resample_poly(sosfiltfilt(sos,search), 6,1,window=("kaiser",5.0),padtype="constant")
    a -= a.mean()
    n = len(a)
    ones = np.ones(n,dtype=np.float64)
    sums = fftconvolve(b,ones,mode="valid")
    squares = fftconvolve(b*b,ones,mode="valid")
    window_energy = np.maximum(squares-sums*sums/n,0.0)
    a_energy = float(np.dot(a,a))
    denominator = np.sqrt(a_energy*window_energy)
    numerator = fftconvolve(b,a[::-1],mode="valid")
    curve = np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-12)
    require(np.isfinite(curve).all(), "Nonfinite independent curve")
    absolute = np.abs(curve)
    best = int(np.argmax(absolute))
    far = absolute.copy()
    exclusion = RATE//4
    far[max(0,best-exclusion):min(len(far),best+exclusion+1)] = -1.0
    second = int(np.argmax(far))

    def direct(index: int) -> float:
        window = b[index:index+n]
        centered = window-window.mean()
        denominator = float(np.linalg.norm(a)*np.linalg.norm(centered))
        return 0.0 if denominator <= 1e-12 else float(np.dot(a,centered)/denominator)

    best_direct, second_direct = direct(best),direct(second)
    close(best_direct,float(curve[best]),"Refined independent FFT/direct winner",1e-8)
    close(second_direct,float(curve[second]),"Refined independent FFT/direct runner-up",1e-8)
    return {"evaluation_grid_rate":RATE,"global_candidate_count":len(curve),
            "best_start_sample_in_search":best,"best_start_seconds_in_search":best/RATE,
            "signed_correlation":best_direct,"absolute_correlation":abs(best_direct),
            "runner_up_exclusion_seconds":.25,"runner_up_start_sample_in_search":second,
            "runner_up_start_seconds_in_search":second/RATE,"runner_up_absolute_correlation":abs(second_direct),
            "peak_margin":abs(best_direct)-abs(second_direct),
            "peak_ratio":abs(best_direct)/max(abs(second_direct),1e-12),
            "reference_rms_filtered_interpolated":math.sqrt(a_energy/n),
            "matched_search_rms_filtered_interpolated_centered":float(np.std(b[best:best+n])),
            "search_boundary_clearance_seconds":min(best,len(b)-n-best)/RATE,
            "direct_signed_correlations":{"best":best_direct,"far_runner_up":second_direct}}


def score_gates(score: dict,method: dict) -> dict:
    return {"correlation":score["absolute_correlation"]>=method["minimum_correlation"],
            "unique_far_peak":score["peak_ratio"]>=method["minimum_far_peak_ratio"],
            "reference_energy":score["reference_rms_filtered_interpolated"]>=method["minimum_filtered_rms"],
            "matched_energy":score["matched_search_rms_filtered_interpolated_centered"]>=method["minimum_filtered_rms"],
            "boundary_clearance":score["search_boundary_clearance_seconds"]>=method["minimum_search_boundary_clearance_seconds"]}


def main() -> None:
    started = time.perf_counter()
    plan_path = BASE/"refinement_plan.json"
    progress_path = BASE/"results/refinement_progress.json"
    plan,progress = read(plan_path),read(progress_path)
    plan_sha,progress_sha = sha(plan_path),sha(progress_path)
    for name in ["original_plan","inventory","trigger_probe","trigger_probe_code","code"]:
        check_binding(plan[name])
    require(progress["status"]=="completed" and progress["completed_sources"]==59 and not progress["pending_source_stems"],"Need all59 refined outcomes")
    require(progress["refinement_plan_sha256"]==plan_sha,"Refinement progress plan mismatch")
    require(plan["anchors_seconds"]==[30.0,150.0,270.0],"Refined anchor policy changed")
    method = plan["method"]
    for key,expected in {"input_rate":8000,"interpolation_factor":6,"evaluation_grid_rate":48000,
                         "reference_seconds":6.0,"search_seconds":36.0,"search_radius_seconds":15.0,
                         "minimum_correlation":.99,"minimum_far_peak_ratio":1.5,"minimum_filtered_rms":1e-4,
                         "minimum_search_boundary_clearance_seconds":.5,"maximum_offset_range_exclusive_seconds":.02}.items():
        close(method[key],expected,"Refinement method "+key,1e-12)
    coarse_audit_path = BASE/"results/independent_audit.json"
    coarse_audit = read(coarse_audit_path)
    require(coarse_audit["status"]=="passed" and coarse_audit["scope"]=="all59_source_outcomes","Coarse full independent audit must pass first")
    require(coarse_audit["plan_sha256"]==plan["original_plan"]["sha256"],"Original plan differs from coarse audit")
    original_progress_path = BASE/"results/progress.json"
    require(sha(original_progress_path)==coarse_audit["progress_snapshot_sha256"],"Original progress changed after audit")
    coarse_audits = {s["source_stem"]:s for s in coarse_audit["source_audits"]}
    inventory = read(plan["inventory"]["path"])
    sources = {s["source_stem"]:s for s in inventory["sources"]}
    require(len(sources)==59 and set(sources)==set(plan["source_order"])==set(coarse_audits),"Refined cohort mismatch")
    a_ids,b_ids = (set(inventory["branch_sample_ids"][b]) for b in ["A_keep256","B_expand512"])
    require(len(a_ids)==256 and len(b_ids)==512 and a_ids<b_ids,"Branch cohort mismatch")
    rows,seen,clip_outcomes,pcm_bindings = [],[],{},{}
    for binding in progress["source_results"]:
        check_binding(binding)
        row = read(binding["path"])
        stem = row["source_stem"]
        seen.append(stem)
        source = sources[stem]
        require(row["refinement_plan_sha256"]==plan_sha and row["speaker_id"]==source["speaker_id"] and row["phase"]==source["phase"],"Refined source provenance mismatch")
        require(row["branch_sample_ids"]==source["branch_sample_ids"],"Refined branch IDs mismatch")
        check_binding(row["original_result"])
        require(row["original_result"]["sha256"]==coarse_audits[stem]["source_result_sha256"],"Original source result differs from independently audited one")
        original = read(row["original_result"]["path"])
        require(row["original_status"]==original["status"] and row["original_reason"]==original.get("reason") and row["original_summary"]==original.get("summary"),"Original decisions not preserved")
        require(row["status"] in {"supported","quarantined"},"Unfinished refined source")
        anchors = []
        for index,anchor in enumerate(row["anchors"]):
            require(index<3 and index<len(original["anchors"]),"Unplanned refined anchor")
            original_anchor = original["anchors"][index]
            expected_anchor = plan["anchors_seconds"][index]
            for key in ["reference","mp4_search"]:
                expected_spec = {k:original_anchor[key][k] for k in ["path","sha256","start_seconds","frames"]}
                require(anchor[key]==expected_spec,"Refined PCM differs from original audited PCM")
                pcm_bindings[expected_spec["path"]] = expected_spec["sha256"]
            reference = pcm(original_anchor["reference"],6.0,source["audio"]["path"],expected_anchor)
            search = pcm(original_anchor["mp4_search"],36.0,source["video"]["path"],expected_anchor-15.0)
            actual = independent_score(reference,search)
            saved = anchor["score"]
            for key,value in actual.items():
                if key=="direct_signed_correlations":
                    for part,corr in value.items():
                        close(corr,saved[key][part],"Refined direct "+part,1e-8)
                else:
                    close(value,saved[key],"Refined "+key)
            gates = score_gates(actual,method)
            require(gates==saved["strict_gates"] and all(gates.values())==saved["strict_anchor_pass"],"Refined anchor gate mismatch")
            matched = expected_anchor-15.0+actual["best_start_seconds_in_search"]
            offset = matched-expected_anchor
            close(saved["matched_mp4_start_seconds"],matched,"Refined matched timestamp",1e-10)
            close(saved["offset_mp4_minus_original_seconds"],offset,"Refined offset",1e-10)
            anchors.append({"anchor_seconds":expected_anchor,**actual,"strict_gates":gates,
                            "strict_anchor_pass":all(gates.values()),"offset_seconds":offset})
        if "summary" in row:
            require(len(anchors)==3,"Refined summary needs3 complete anchors")
            offsets = [a["offset_seconds"] for a in anchors]
            median = round(statistics.median(offsets)*RATE)/RATE
            spread = max(offsets)-min(offsets)
            integrity = (original.get("summary") is not None and original["status"] in {"supported","quarantined"}
                         and original.get("reason") in (None,SIGNAL_ONLY_REASON))
            strict = all(a["strict_anchor_pass"] for a in anchors)
            supported = integrity and strict and spread<.02
            summary = row["summary"]
            for key,value in {"median_offset_seconds":median,"offset_range_seconds":spread,
                              "minimum_correlation":min(a["absolute_correlation"] for a in anchors),
                              "minimum_far_peak_ratio":min(a["peak_ratio"] for a in anchors)}.items():
                close(summary[key],value,"Refined summary "+key)
            require(summary["all_anchors_strict"]==strict and summary["original_integrity_checks_completed"]==integrity
                    and summary["constant_mapping_supported"]==supported,"Refined source gates mismatch")
            require(summary["abs_offset_at_least_100ms"]==(abs(median)>=.1),"Refined source significant flag mismatch")
            require((row["status"]=="supported")==supported,"Refined source disposition mismatch")
            require(summary["video_duration_seconds"]==original.get("summary",{}).get("video_duration_seconds"),"Refined video bound changed")
        if row["status"]=="supported":
            require(len(anchors)==3 and "summary" in row,"Refined supported source missing evidence")
            impacts = {r["id"]:r for r in row["training_frame_selection_impact"]}
            require(set(impacts)==set(source["branch_sample_ids"]["B_expand512"]),"Refined impact coverage mismatch")
            for record in source["training_records"]:
                impact = impacts[record["id"]]
                start,end = record["start_time"]+median,record["end_time"]+median
                frame_count = record["frames"]
                before = [int(round((record["start_time"]+k/25)*record["video_fps"])) for k in range(frame_count)]
                after = [int(round((start+k/25)*record["video_fps"])) for k in range(frame_count)]
                changed = sum(a!=b for a,b in zip(before,after))
                bounded = start>=0 and end<=summary["video_duration_seconds"]
                close(impact["video_start_time"],start,"Refined clip start")
                close(impact["video_end_time"],end,"Refined clip end")
                require(impact["target_frames"]==frame_count and impact["changed_selected_frame_indices"]==changed
                        and impact["fully_within_video_duration"]==bounded,"Refined frame-impact mismatch")
                clip_outcomes[record["id"]] = {"status":"supported","source_stem":stem,"offset_seconds":median,
                    "significant_offset":abs(median)>=.1,"target_frames":frame_count,
                    "changed_selected_frame_indices":changed,"within_video_duration":bounded}
        else:
            require(bool(row.get("reason")),"Refined quarantine missing reason")
            require("training_frame_selection_impact" not in row,"Quarantined refined source has impact candidate")
            for record in source["training_records"]:
                clip_outcomes[record["id"]] = {"status":"quarantined","source_stem":stem}
        rows.append({"source_stem":stem,"speaker_id":source["speaker_id"],"status":row["status"],
                     "original_status":original["status"],"refined_result_sha256":binding["sha256"],
                     "anchors":anchors,"summary":row.get("summary"),"reason":row.get("reason")})
        print(json.dumps({"independently_audited_refined_sources":len(rows),"anchors":sum(len(r["anchors"]) for r in rows)}),flush=True)
    require(seen==plan["source_order"] and len(seen)==59,"Full refined coverage/order mismatch")
    counts = Counter(r["status"] for r in rows)
    require(counts["supported"]==progress["supported_sources"] and counts["quarantined"]==progress["quarantined_sources"],"Refined aggregate mismatch")
    require(set(clip_outcomes)==b_ids,"Refined512 clip coverage mismatch")
    require(sha(plan_path)==plan_sha and sha(progress_path)==progress_sha,"Refined evidence changed during audit")
    require(sha(original_progress_path)==coarse_audit["progress_snapshot_sha256"],"Original progress changed during refined audit")
    branches = {}
    for branch,ids in [("A_keep256",a_ids),("B_expand512",b_ids)]:
        selected = [clip_outcomes[i] for i in ids]
        supported_clips = [v for v in selected if v["status"]=="supported"]
        branches[branch] = {"total_selected_clips":len(ids),"reviewed_clips":len(selected),
            "supported_mapping_clips":len(supported_clips),"quarantined_clips":len(selected)-len(supported_clips),
            "clips_on_at_least_100ms_offset_sources":sum(v["significant_offset"] for v in supported_clips),
            "clips_with_changed_frame_indices":sum(v["changed_selected_frame_indices"]>0 for v in supported_clips),
            "changed_selected_frame_indices":sum(v["changed_selected_frame_indices"] for v in supported_clips),
            "supported_clip_target_frames":sum(v["target_frames"] for v in supported_clips),
            "mapped_clip_out_of_bounds_ids":sorted(i for i in ids if clip_outcomes[i]["status"]=="supported" and not clip_outcomes[i]["within_video_duration"])}
    result = {"schema":"training-step06-independent-refinement-audit-v1","status":"passed",
        "created_utc":datetime.now(timezone.utc).isoformat(),"auditor_sha256":sha(Path(__file__)),
        "independent_coarse_helpers_sha256":sha(BASE/"independent_audit.py"),
        "refinement_plan_sha256":plan_sha,"refinement_progress_sha256":progress_sha,
        "original_independent_audit_sha256":sha(coarse_audit_path),
        "method":"Full48k-grid search on independently filtered/interpolated saved8k PCM; FFT convolution numerator AND both sliding moments; direct centered Pearson at winner and far runner-up for every available anchor; no producer imports",
        "verified":{"source_outcomes":len(rows),"supported_sources":counts["supported"],"quarantined_sources":counts["quarantined"],
                    "independent_full_search_anchors":sum(len(r["anchors"]) for r in rows),
                    "unique_PCM_hash_format_length_command_checks":len(pcm_bindings),
                    "original_decisions_preserved":True,"unique_training_clips":len(clip_outcomes)},
        "branch_impact":branches,"source_audits":rows,
        "scope_guards":{"cloud_source_read":False,"models_loaded":False,"test_payload_read":False,
                        "new_PCM_extraction":False,"original_results_modified":False,"thresholds_lowered":False},
        "limits":["Interpolating8kHz samples provides a finer evaluation grid, not new48kHz observations or an accuracy guarantee.",
                  "This independently verifies the separate posthoc follow-up; it does not reclassify or overwrite original8kHz results.",
                  "Timing-only analysis does not certify transcript truth, lip/audio synchrony, full media integrity or recognition improvement.",
                  "Three anchors cannot exclude unexamined local edits; frame-impact counts are arithmetic, not rebuilt-cache checks."],
        "seconds":time.perf_counter()-started}
    output = BASE/"results/independent_refined_audit.json"
    with output.open("x",encoding="utf8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"path":str(output),"sha256":sha(output),"verified":result["verified"],"branch_impact":branches},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
