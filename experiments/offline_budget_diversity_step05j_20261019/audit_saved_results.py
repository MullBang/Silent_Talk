"""Independent NumPy audit of saved 5j results, with no model/tensor deserialization."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import traceback
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / "experiments/offline_diversity_motion_step05i_20261019"
HELPER = OLD / "audit_saved_results.py"
spec = importlib.util.spec_from_file_location("frozen5i_numeric_audit_helpers", HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
read, check, bound, digest = h.read, h.check, h.bound, h.digest
CONDITIONS = ("normal", "static_mid", "shuffled", "reversed")

def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def load_sources(data):
    rows = {r["clip_id"]: r for r in data["records"]}
    check(len(rows) == len(data["records"]), "Duplicate source IDs")
    old_data = read(OLD / "data_plan.json")
    original = {r["clip_id"]: r for r in old_data["records"]}
    check(set(original) <= set(rows), "Missing preserved old32")
    for sid, original_row in original.items():
        for field in ("parent_id", "recording_group", "speaker_id", "split", "kind", "T", "text", "npz_path", "npz_sha256", "metadata_path", "metadata_sha256"):
            check(rows[sid][field] == original_row[field], "Old source field changed:"+field)
    check(not set(old_data["held_parent_ids"]) & {r["parent_id"] for r in rows.values()}, "Held Q03..Q08 entered")
    inputs, labels, metadata, prepared = {}, {}, {}, {}
    for sid, row in rows.items():
        meta = read(bound({"path": row["metadata_path"], "sha256": row["metadata_sha256"]}))
        metadata[sid] = meta
        path = bound({"path": row["npz_path"], "sha256": row["npz_sha256"], "bytes": row["npz_bytes"]})
        check(meta["training_ready"] is False and meta["provisional"] is True and row["training_ready"] is False and row["provisional"] is True, "Source approval promotion")
        with np.load(path, allow_pickle=False) as packed:
            check(set(packed.files) == {"schema", "frames_rgb", "length", "label", "metadata_json", "training_ready", "provisional"}, "Unexpected nonvisual cache content")
            check(json.loads(packed["metadata_json"].item()) == meta, "NPZ/sidecar mismatch")
            rgb = packed["frames_rgb"]
            check(rgb.dtype == np.uint8 and rgb.shape == (row["T"],96,96,3), "Source shape")
            check(int(packed["length"]) == row["T"] and h.array_digest(rgb) == meta["rgb_uint8_sha256"], "Source RGB/length")
            check(packed["training_ready"].item() is False and packed["provisional"].item() is True, "Packed flags")
            labels[sid] = packed["label"].copy()
            check(labels[sid].tolist() == meta["label"] and len(labels[sid]) == row["label_length"], "Label binding")
            check(len(labels[sid]) + int(np.sum(labels[sid][1:] == labels[sid][:-1])) <= row["T"], "CTC feasibility")
            # Insert blanks so the independent CTC decoder retains adjacent repeated label IDs.
            _, label_text = h.decode([v for tok in labels[sid].tolist() for v in (int(tok),0)])
            check(label_text == row["text"], "Independent target token/text decode")
            pixels = rgb.astype(np.float32) / np.float32(255)
            pixels -= np.array([.485,.456,.406],np.float32)
            pixels /= np.array([.229,.224,.225],np.float32)
            inputs[sid] = np.ascontiguousarray(np.moveaxis(pixels,-1,0))
            check(np.isfinite(inputs[sid]).all(), "Nonfinite normalized pixels")
        for condition in CONDITIONS:
            order = h.order_for(row["T"],sid,condition)
            prepared[sid,condition] = {"clip_id": sid,"condition": condition,"frame_indices": order.tolist(),"input_sha256": h.array_digest(inputs[sid][:,order])}
    dev = set(old_data["cohorts"]["dev_phrase3"] + old_data["cohorts"]["dev_sentence3"])
    train = set(rows)-dev
    for field in ("parent_id","recording_group","text"):
        check(not {rows[s][field] for s in train} & {rows[s][field] for s in dev}, "Train/dev leakage:"+field)
    for member in ("video","label"):
        check(not {metadata[s]["source_members"][member]["sha256"] for s in train} & {metadata[s]["source_members"][member]["sha256"] for s in dev}, "Train/dev source hash leakage")
    for a in train:
        for b in dev:
            ta,tb = h.text(rows[a]["text"]).replace(" ",""), h.text(rows[b]["text"]).replace(" ","")
            check(ta not in tb and tb not in ta, "Train/dev contained full text leakage")
    return rows,inputs,labels,metadata,prepared

def expected_budget_tasks(data, seed):
    rng = random.Random(seed)
    ids = sorted(data["arms"]["B26"])
    rows = {r["clip_id"]:r for r in data["records"]}
    tasks=[]
    for epoch in range(1,281):
        order=list(ids); rng.shuffle(order)
        for offset in range(0,26,2):
            pair=order[offset:offset+2]; update=len(tasks)+1
            lengths=[rows[s]["T"] for s in pair]
            tasks.append({"update":update,"epoch":epoch,"clip_ids":pair,"lengths":lengths,
                          "lr":.001 if update<=910 else .0003 if update<=1365 else .0001,
                          "true_frames":sum(lengths),"dense_frames":2*max(lengths)})
    old_plan=read(OLD/"plan.json")
    check(tasks[:1820] == read(bound(old_plan["schedules"]["B26"][str(seed)]))["updates"],"Independent original schedule prefix")
    return tasks[1820:]

def audit_evaluation(link, update, arm, seed, rows,inputs,labels,prepared,cohorts,optimizer_ids):
    summary_path=bound(link["summary"]); directory=summary_path.parent
    summary=read(summary_path)
    saved=[json.loads(line) for line in (directory/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    check(saved == summary["rows"],"Eval JSONL/summary mismatch")
    isolation=read(directory/"isolation.json")
    for field in ("state","gradients","modes"):
        check(isolation[field+"_before"] == isolation[field+"_after"],"Evaluation isolation:"+field)
    expected={(sid,c) for sid in rows for c in CONDITIONS}
    actual={(r["clip_id"],r["condition"]) for r in saved}
    check(len(saved)==len(actual) and actual==expected,"Missing/duplicate evaluations")
    buckets=defaultdict(list)
    for row in saved:
        sid,c=row["clip_id"],row["condition"]; source=rows[sid]; length=source["T"]
        check(row["status"]=="success" and row["audio_input"] is False,"Failed/nonvisual evaluation")
        check((row["arm"],row["seed"],row["update"])==(arm,seed,update),"Evaluation identity")
        check(row["reference"]==source["text"] and row["reference_label_ids"]==labels[sid].tolist(),"Evaluation target")
        check(row["seen_in_optimizer"]==(sid in optimizer_ids),"Evaluation train annotation")
        cohort=next(k for k,v in cohorts.items() if sid in v)
        check(row["cohort"]==cohort,"Evaluation cohort")
        check(row["frame_indices"]==prepared[sid,c]["frame_indices"] and row["input_sha256"]==prepared[sid,c]["input_sha256"],"Evaluation transformed input")
        check(row["input_audit"]["normalized_input_sha256"]==h.array_digest(inputs[sid]),"Normalized input")
        order=np.array(prepared[sid,c]["frame_indices"],np.int64)
        with np.load(bound(row["posterior"]),allow_pickle=False) as packed:
            values=packed["log_probs"]
            check(values.dtype==np.float32 and values.shape==(length,69) and np.isfinite(values).all(),"Posterior shape/dtype/finite")
            check(np.allclose(np.exp(values.astype(np.float64)).sum(1),1,atol=1e-5,rtol=1e-5),"Posterior normalization")
            check(h.array_digest(values)==row["posterior_array_sha256"],"Posterior content hash")
            check(np.array_equal(packed["frame_indices"],order) and np.array_equal(packed["reference_labels"],labels[sid]),"Posterior metadata")
            argmax=values.argmax(1).tolist()
            check(argmax==packed["raw_argmax_ids"].tolist()==row["raw_argmax_ids"],"Posterior argmax")
        collapsed,pred=h.decode(argmax)
        check(collapsed==row["collapsed_ids"] and pred==row["raw_text"] and h.text(pred)==row["metric_text"],"Independent numeric Hangul decoder")
        score=h.score(source["text"],pred)
        check(score==row["metrics"],"Independent Levenshtein metrics")
        buckets[cohort+"/"+c].append(score)
    calculated={k:h.aggregate(v) for k,v in buckets.items()}
    check(calculated==summary["cohorts"]==link["cohorts"],"Aggregate metrics")
    check(summary["count"]==link["count"]==len(saved) and summary["failures"]==link["failures"]==0,"Eval count/failures")
    frames=sum(r["T"] for r in saved)
    check(summary["frames"]==link["frames"]==frames,"Eval frame count")
    return {"rows":len(saved),"frames":frames,"cohorts":calculated}

def preservation():
    manifests={"stage5i":OLD/"results/completion_manifest.json",
               "review_r8":ROOT/"experiments/offline_phrase_boundary_step05g_20261018/results/review_snapshot_manifest_r8.json"}
    prior=read(ROOT/"experiments/offline_phrase_review_step05f_20261018/audit/source_preservation_final.json")
    manifests["service_scope"]=bound(prior["service_and_prior_scope_binding"])
    counts={}
    for name,path in manifests.items():
        entries=read(path)["artifacts"]
        for entry in entries:
            if entry.get("exists",True): bound(entry)
            else: check(not Path(entry["path"]).exists(),"Absent preserved scope appeared")
        counts[name]=len(entries)
    helper_entry=next(e for e in read(manifests["stage5i"])["artifacts"] if Path(e["path"])==HELPER)
    bound(helper_entry)
    return counts

def audit_budget():
    directory=BASE/"training"; plan_path=directory/"plan.json"; plan=read(plan_path)
    check(plan["start_updates"]==1820 and plan["additional_steps"]==1820 and plan["final_updates"]==3640 and plan["evaluations"]==[2730,3640],"Budget scope")
    data=read(bound(plan["old_data_plan"])); bound(plan["policy"]); bound(plan["old_plan"])
    for binding in plan["bindings"]: bound(binding)
    rows,inputs,labels,meta,prepared=load_sources(data)
    check(len(rows)==32,"Budget cache count")
    batches={}; reports={}; update_count=posterior_count=posterior_frames=checkpoint_count=0
    for seed in (42,43):
        arm="B26_extended"; out=directory/"results"/f"{arm}_seed{seed}"
        result,contract=read(out/"result.json"),read(out/"run_contract.json")
        check(result["status"]=="completed" and result["physical_updates"]==1820 and result["origin_updates"]==1820 and result["total_updates"]==3640,"Budget completion")
        check(result["identity"]==contract["identity"] and result["identity"]["plan_sha256"]==digest(plan_path),"Budget identity")
        old_state=read(OLD/f"results/B26_seed{seed}/eval_1820/isolation.json")["state_before"]
        check(result["initial_state_hashes"]==contract["initial_state_hashes"]==old_state,"Initial checkpoint model/Adam/RNG hashes")
        check(contract["model_adam_rng_restored_exactly"] and contract["origin_completed_updates"]==1820,"Full continuation contract")
        check(result["identity"]["source_checkpoint"]==plan["source_checkpoints"][str(seed)],"Source checkpoint binding")
        bound(plan["source_checkpoints"][str(seed)])
        tasks=read(bound(plan["schedules"][str(seed)]))["updates"]
        check(tasks==expected_budget_tasks(data,seed),"Independent subsequent Random epochs")
        logs=[json.loads(line) for line in (out/"progress.jsonl").read_text(encoding="utf-8").splitlines()]
        check(len(logs)==len(tasks)==1820,"Budget schedule/log count")
        for i,(task,log) in enumerate(zip(tasks,logs),1):
            check(log["update"]==task["update"]==1820+i and log["extension_update"]==i and log["epoch"]==task["epoch"],"Budget update order")
            audit_update(task,log,rows,inputs,prepared,set(data["arms"]["B26"]),batches,plan["limits"])
        exposure=Counter(s for task in tasks for s in task["clip_ids"])
        check(dict(exposure)==plan["planned_accounting"][str(seed)]["new_exposures"] and set(exposure.values())=={140},"Budget exposure")
        for field in ("true_frames","dense_frames"):
            check(sum(t[field] for t in tasks)==result[field]==plan["planned_accounting"][str(seed)]["new_"+field],"Budget frame accounting")
        check(result["seconds"]<=plan["limits"]["wall_seconds_per_run"],"Budget wall limit")
        for field in ("audio_inputs","teacher_calls","service_changes"): check(result[field]==0,"Unexpected side effect")
        check(result["source_approval_changed"] is False and result["source_training_ready"] is False,"Approval transition")
        check(len(result["checkpoints"])==len(result["evaluations"])==2,"Budget endpoint artifacts")
        for binding in result["checkpoints"]: bound(binding); checkpoint_count+=1
        reports[str(seed)]={}
        for update,link in zip(plan["evaluations"],result["evaluations"]):
            r=audit_evaluation(link,update,arm,seed,rows,inputs,labels,prepared,data["cohorts"],set(data["arms"]["B26"]))
            posterior_count+=r["rows"]; posterior_frames+=r["frames"]; reports[str(seed)][str(update)]=r["cohorts"]
        update_count+=len(logs)
    preflight=read(directory/"results/preflight/result.json")
    check(preflight["status"]=="passed" and preflight["physical_updates"]==4 and preflight["source_restores_exact"] and preflight["next_update_exact"],"Budget preflight")
    bound(preflight["checkpoint"])
    check(preflight["reference_final_hashes"]==preflight["restored_final_hashes"],"Budget replay hashes")
    for seed in (42,43):
        check(preflight["source_state_hashes"][str(seed)]==read(OLD/f"results/B26_seed{seed}/eval_1820/isolation.json")["state_before"],"Preflight origin hashes")
    check(update_count==3640 and posterior_count==512 and checkpoint_count==4,"Budget totals")
    return {"branch":"budget","status":"passed","plan_sha256":digest(plan_path),"cache_inputs_checked":len(rows),"new_updates_checked":update_count,"preflight_updates_recorded":4,"posteriors_checked":posterior_count,"posterior_frames":posterior_frames,"checkpoints_hash_checked":checkpoint_count,"padded_batches_checked":update_count,"distinct_padded_batch_hashes":len(batches),"cohorts":reports}

def audit_update(task,log,rows,inputs,prepared,optimizer_ids,batches,limits):
    ids=task["clip_ids"]; lengths=[rows[s]["T"] for s in ids]
    check(log["clip_ids"]==ids and log["lengths"]==task["lengths"]==lengths,"Logged batch IDs/lengths")
    check(set(ids)<=optimizer_ids and len(ids)==2,"Outside optimizer allowlist")
    check(log["learning_rate"]==task["lr"]==.0001 and log["training_condition"]=="normal","Continuation LR/condition")
    check(log["optimizer_updated"] and np.isfinite(log["loss"]),"Update/finite loss")
    check(log["input_audits"]==[prepared[s,"normal"] for s in ids],"Actual unpadded batch hash")
    pair=tuple(ids)
    if pair not in batches:
        maximum=max(lengths); padded=np.empty((2,3,maximum,96,96),np.float32)
        for i,sid in enumerate(ids):
            padded[i]=inputs[sid][:,np.minimum(np.arange(maximum),rows[sid]["T"]-1)]
        batches[pair]=h.array_digest(padded)
    check(log["padded_batch_sha256"]==batches[pair],"Actual last-frame padded batch hash")
    check(log["true_frames"]==task["true_frames"]==sum(lengths) and log["dense_frames"]==task["dense_frames"]==2*max(lengths),"Batch frame accounting")
    check(np.isfinite(log["gradient_norm_before_clip"]) and np.isfinite(log["gradient_norm_after_clip"]) and log["nonzero_gradient_tensors"]>0,"Finite nonzero gradients")
    check(log["cuda_peak_reserved_bytes"]<=limits["cuda_reserved_bytes"],"CUDA memory limit")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--branch",choices=["budget","addition","all"],required=True); args=parser.parse_args()
    target=BASE/"audit"/f"{args.branch}_saved_results_check.json"
    check(not target.exists(),"Audit output already exists")
    reports=[]
    if args.branch in ("budget","all"): reports.append(audit_budget())
    if args.branch in ("addition","all"): reports.append(audit_addition())
    report={"schema":"offline-stage5j-independent-saved-results-audit-v1","status":"passed","created_utc":datetime.now(timezone.utc).isoformat(),"auditor_sha256":digest(__file__),"frozen_numeric_helper_sha256":digest(HELPER),"branches":reports,"preservation_counts":preservation(),"network_forward_calls":0,"checkpoint_tensor_loads":0,"original_media_reads":0,"limitations":"Recomputes saved inputs/metrics and checks state hashes without independently executing neural layers or deserializing checkpoint tensors. Dev phrase and sentence share3parents and are not6independentexamples."}
    save(target,report)
    print(json.dumps({"status":"passed","path":str(target),"branches":[{k:v for k,v in r.items() if k!="cohorts"} for r in reports],"preservation":report["preservation_counts"]},ensure_ascii=False))

if __name__=="__main__":
    try: main()
    except Exception:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        save(BASE/"audit"/f"failure_{stamp}.json",{"status":"audit_failed","auditor_sha256":digest(__file__),"error":traceback.format_exc(),"training_mutated":False})
        raise
