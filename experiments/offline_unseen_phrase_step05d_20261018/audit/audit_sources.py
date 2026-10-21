"""Independent preservation and reused-dev phrase scope audit; metadata only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import unicodedata

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=HERE.parents[2]
S3=ROOT/"experiments/offline_dataset_step03_20261013"
S4=ROOT/"experiments/offline_training_step04_20261013"
S5=ROOT/"experiments/offline_comparison_step05_20261013"
S5B=ROOT/"experiments/offline_diagnostic_step05b_20261018"
S5C=ROOT/"experiments/offline_phrase_only_step05c_20261018"

def read(path):return json.loads(Path(path).read_text(encoding="utf8"))

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def binding(path):
    path=Path(path);return {"path":str(path),"sha256":digest(path),"bytes":path.stat().st_size}

def check_binding(row):
    path=Path(row["path"])
    if row.get("exists") is False:
        assert not path.exists(),str(path);return
    assert path.is_file() and digest(path)==row["sha256"],str(path)
    if "bytes" in row:assert path.stat().st_size==row["bytes"],str(path)

def textkey(value):return "".join(unicodedata.normalize("NFC",value).split())

def inspect():
    manifests=[]
    for folder,count in [(S3,114),(S4,101),(S5,1238),(S5B,45),(S5C,280)]:
        path=folder/"results/completion_manifest.json";manifest=read(path)
        assert len(manifest["artifacts"])==len({r["path"] for r in manifest["artifacts"]})==count
        for r in manifest["artifacts"]:check_binding(r)
        manifests.append({**binding(path),"verified_artifacts":count})
    scopepath=S3/"results/scope_snapshot.json";scope=read(scopepath)["artifacts"]
    assert len(scope)==10
    for r in scope:check_binding(r)
    cachepath=S3/"results/cache_handoff.json";handoff=read(cachepath);records=handoff["records"]
    assert len(records)==30 and all(r["provisional"] and not r["training_ready"] for r in records)
    assert sum(r["boundary_followup_flag"] for r in records)==2
    assert not handoff["training_ready"] and not handoff["new_original_pixels_human_approved"]
    train=[r for r in records if r["split"]=="train"]
    targets=[r for r in records if r["split"]=="dev" and r["kind"]=="phrase"]
    assert len(train)==24 and len(targets)==3
    expected_ids=["lip_J_1_F_02_C004_A_011_014__phrase","lip_J_1_F_03_C297_A_006_006__phrase","lip_J_1_F_03_C312_A_007_047__phrase"]
    assert [r["clip_id"] for r in targets]==expected_ids
    assert [r["T"] for r in targets]==[31,23,42] and sum(r["T"] for r in targets)==96
    assert [r["label_length"] for r in targets]==[24,16,24]
    assert not any(r["boundary_followup_flag"] or r["exact_visual_boundary_certified"] for r in targets)
    for key in ["parent_id","recording_group"]:
        assert not {r[key] for r in train}&{r[key] for r in targets}
    assert {r["speaker_id"] for r in targets}=={"C004","C297","C312"}
    assert {r["speaker_id"] for r in targets} <= {r["speaker_id"] for r in train}
    assert not {textkey(r["text"]) for r in train}&{textkey(r["text"]) for r in targets}
    splitpath=Path(handoff["split_plan"]["path"]);check_binding(handoff["split_plan"])
    split=read(splitpath);parents={r["parent_id"]:r for r in split["records"]}
    for key in ["source_stem","angle_independent_recording_key","video_sha256","label_sha256"]:
        trainkeys={parents[r["parent_id"]][key] for r in train};devkeys={parents[r["parent_id"]][key] for r in targets}
        assert not trainkeys&devkeys,key
    evalpath=S4/"protocol/evaluation_plan.json";evalplan=read(evalpath)
    targetitems=[r for r in evalplan["items"] if r["split"]=="dev" and r["kind"]=="phrase"]
    assert [r["clip_id"] for r in targetitems]==expected_ids
    targetbindings=[]
    for r,item in zip(targets,targetitems):
        assert all(r[k]==item[k] for k in ["clip_id","parent_id","kind","split","T","text"])
        assert item["static_frame_index"]==r["T"]//2
        permutation=list(range(r["T"]));random.Random(item["shuffle_seed"]).shuffle(permutation)
        assert item["shuffled_frame_indices"]==permutation
        check_binding({"path":r["npz_path"],"sha256":r["npz_sha256"],"bytes":r["npz_bytes"]})
        check_binding({"path":r["metadata_path"],"sha256":r["metadata_sha256"]})
        targetbindings.append({"clip_id":r["clip_id"],"parent_id":r["parent_id"],"speaker":r["speaker_id"],"text":r["text"],"T":r["T"],"label_length":r["label_length"],"npz":binding(Path(r["npz_path"])),"metadata":binding(Path(r["metadata_path"]))})
    s5chandoff=read(S5C/"results/next_step_handoff.json")
    assert s5chandoff["fixed_fit_gate_passed"] is False and s5chandoff["strict_gate_remains_failed"] is True
    assert s5chandoff["proposed_neural_evaluations"]==18 and s5chandoff["proposed_evaluation_frames"]==576
    assert s5chandoff["proposed_optimizer_updates"]==0
    assert [r["clip_id"] for r in s5chandoff["candidate_targets"]]==expected_ids
    checkpoints=[]
    for seed,checkpoint in zip([42,43],s5chandoff["checkpoints"]):
        check_binding(checkpoint)
        result=read(S5C/f"runs/seed{seed}/segment_result.json")
        assert result["completed_updates"]==1680 and result["final_checkpoint"]==checkpoint
        logs=[json.loads(line) for line in (S5C/f"runs/seed{seed}/progress.jsonl").read_text(encoding="utf8").splitlines()]
        assert len(logs)==1680
        ids={c for row in logs for c in row["clip_ids"]}
        assert ids=={r["clip_id"] for r in train if r["kind"]=="phrase"}
        assert not ids&set(expected_ids)
        for evaluation in result["evaluations"]:
            check_binding(evaluation["summary"])
            saved=read(evaluation["summary"]["path"])
            assert all(r["kind"]=="phrase" and r["split"]=="train" and r["clip_id"] not in expected_ids for r in saved["rows"])
        checkpoints.append({"seed":seed,**checkpoint,"trained_updates":1680,"never_optimized_or_evaluated_targets_in_stage5c":True})
    # Record their historical evaluation use explicitly: these are not untouched samples.
    old=read(S5/"runs/seed42/B_curriculum/evaluations/update_1260/evaluation_summary.json")
    previously_seen=[r["clip_id"] for r in old["rows"] if r["clip_id"] in expected_ids and r["condition"]=="normal"]
    assert previously_seen==expected_ids
    return {"schema":"offline-stage5d-independent-source-preflight-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","frozen_manifests":manifests,"verified_frozen_artifact_entries":1778,"scope":{"binding":binding(scopepath),"verified_artifacts":10},"handoff":binding(cachepath),"split_plan":binding(splitpath),"source_evaluation_plan":binding(evalpath),"stage5c_handoff":binding(S5C/"results/next_step_handoff.json"),"targets":targetbindings,"checkpoints":checkpoints,"evaluation_clips":3,"unique_phrases":3,"source_frames":96,"conditions":3,"models":2,"planned_evaluation_rows":18,"planned_evaluation_frames":576,"planned_optimizer_updates":0,"source_flags_preserved":True,"strict_stage5c_fit_gate_passed":False,"train_parent_recording_angle_key_video_label_sha_and_text_disjoint":True,"current_stage5c_models_never_used_targets":True,"same_speakers_as_training":True,"historically_evaluated_in_stage5":True,"untouched_independent_test":False,"new_speaker_evaluation":False,"producer_imports":0,"checkpoint_tensor_loads":0,"NPZ_pixel_arrays_loaded":0,"network_forward_calls":0,"optimizer_updates":0,"raw_media_reads":0}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="source_preflight.json");args=parser.parse_args()
    output=(HERE/args.output).resolve();assert output.parent==HERE and not output.exists()
    result=inspect();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:result[k] for k in ["status","verified_frozen_artifact_entries","evaluation_clips","planned_evaluation_rows","planned_evaluation_frames","planned_optimizer_updates"]}))
