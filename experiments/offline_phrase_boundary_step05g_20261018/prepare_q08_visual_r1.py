"""Render the confirmed Q08 +80ms audio window as two muted review previews.

Reuse bound read-only helpers from Q05/Q06 and5f without modifying them. Exact
original-WAV sample times select existing25fps display bins; no original video,
model inference, training pixels or automatic visual approval is involved.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OLD=ROOT/"experiments/offline_phrase_review_step05f_20261018"
OUTPUT=BASE/"visual_Q08_r1"
SHARED=BASE/"prepare_q05_q06_visual_r1.py"
SHARED_PLAN=BASE/"visual_Q05_r1/plan.json"
HELPER=OLD/"media_prepare/prepare_phrase_media_r2.py"
MANIFEST=OLD/"results/completion_manifest.json"
REVISION=BASE/"revisions/Q08_end_plus80ms/record.json"
EVIDENCE=BASE/"user_evidence/013_Q08_plus80ms_audio_confirmed.json"
MEDIA=OLD/"media/lip_J_1_F_03_C312_A_006_014/record.json"


def load(path: Path,name: str) -> Any:
    """Import only code that the caller has already bound and checked."""
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError("Cannot load helper")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run() -> None:
    """Bind new actual evidence and verify selected historical inputs only."""
    shared_plan=json.loads(SHARED_PLAN.read_text(encoding="utf-8"))
    shared_binding=next(b for b in shared_plan["source_bindings"] if Path(b["path"]).resolve()==SHARED.resolve())
    with SHARED.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=shared_binding["sha256"]: raise ValueError("Bound Q05/Q06 helper changed")
    common=load(SHARED,"bound_stage5g_visual_utilities")
    bind,verify,read,require=common.bind,common.verify,common.read,common.require
    require(not OUTPUT.exists(),"Existing Q08 visual result must be preserved")
    frozen={str(Path(b["path"]).resolve()):b for b in read(MANIFEST)["artifacts"]}
    selected={}
    def prior(path: Path) -> dict[str,Any]:
        """Check selected5f file against its frozen manifest entry."""
        key=str(path.resolve());require(key in frozen,"Historical input is not frozen")
        verify(frozen[key]);selected[key]=frozen[key];return frozen[key]
    prior(HELPER);prior(OLD/"media_prepare/prepare_media.py");prior(MEDIA)
    sys.path.insert(0,str(HELPER.parent))
    helper=load(HELPER,"bound_stage5f_review_encoder_Q08")
    revision,evidence,media=read(REVISION),read(EVIDENCE),read(MEDIA)
    require(revision["review_id"]==evidence["review_id"]=="Q08" and revision["parent_id"]==media["parent_id"],"Unexpected candidate")
    require(revision["revision_id"]==evidence["revision_id"]=="Q08_end_plus80ms_r1","Unexpected audio revision")
    require(evidence["revision_record"]==bind(REVISION) and evidence["phrase_audio_boundary_approved"] is True and evidence["phrase_visual_boundary_approved"] is False,"Expected actual audio-only confirmation")
    require(revision["training_ready"] is False and revision["actual_split"] is None,"Unexpected training state")
    require(evidence["question_audio"]=={k:revision["media"]["phrase"][k] for k in ("path","sha256","bytes")},"Confirmed question audio differs")
    inputs=[bind(p) for p in (REVISION,EVIDENCE,SHARED,SHARED_PLAN,MANIFEST,Path(__file__))]
    for role in ("phrase","context"):
        inputs.append({k:revision["media"][role][k] for k in ("path","sha256","bytes")})
        prior(Path(revision["media"][role]["source"]["path"]))
    preview=media["video"]["roi_preview"];prior(Path(preview["path"]))
    inputs.extend(selected.values())
    ffmpeg=bind(helper.media.FFMPEG)
    require(ffmpeg["sha256"]=="2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3","FFmpeg binding differs")
    inputs.append(ffmpeg)
    for b in inputs: verify(b)
    origin48=revision["parent_origin_sample_48k"]
    require(origin48==media["parent_native_start_sample"],"Native parent origin differs")
    start=Fraction(origin48+3*revision["parent_local_start_sample_16k"],48000)
    end=Fraction(origin48+3*revision["parent_local_end_sample_16k"],48000)
    require(start==helper.fraction(revision["audio_start_seconds_exact"]) and end==helper.fraction(revision["audio_end_seconds_exact"]),"Local/global audio clock differs")
    offset=Fraction(media["offset_native_samples"],48000)
    require(helper.fraction(revision["video_start_seconds_exact"])==start+offset and helper.fraction(revision["video_end_seconds_exact"])==end+offset and revision["video_offset_applied_count"]==1,"Video offset formula differs")
    display_origin=helper.fraction(media["video"]["display_video_start"])
    windows={}
    for role,s,e in (("phrase",start,end),("context",start-Fraction(1,5),end+Fraction(1,5))):
        audio=revision["media"][role]
        require(e-s==Fraction(audio["samples"],16000),"Audio range duration differs")
        if role=="context":
            original48=media["context_audio"]["source_first_sample"]
            require(s==Fraction(original48+3*audio["source_first_sample_16k"],48000) and e==Fraction(original48+3*audio["source_end_sample_16k"],48000),"Saved context is not the exact+-0.20s range")
        first,stop=helper.frame_window(s+offset,e+offset,display_origin,preview["frames"])
        windows[role]={"audio_binding":{k:audio[k] for k in ("path","sha256","bytes")},"semantic_original_wav_start":helper.ratio(s),"semantic_original_wav_end":helper.ratio(e),
            "semantic_video_start":helper.ratio(s+offset),"semantic_video_end":helper.ratio(e+offset),"source_first_display_frame":first,"source_stop_display_frame":stop,
            "preview_display_video_start":helper.ratio(display_origin+Fraction(first,25)),"preview_display_video_end":helper.ratio(display_origin+Fraction(stop,25)),
            "display_before_semantic_seconds":helper.ratio(s+offset-display_origin-Fraction(first,25)),"display_after_semantic_seconds":helper.ratio(display_origin+Fraction(stop,25)-e-offset),
            "original_grid_trace_slice":media["video"]["trace"][first:stop],"offset_applied_count":1}
    OUTPUT.mkdir()
    def save(name: str,value: Any) -> None:
        """Create a new artifact in only this owned output folder."""
        path=OUTPUT/name;require(path.parent==OUTPUT and not path.exists(),"Unsafe/existing result")
        with path.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    plan={"schema":"offline-stage5g-revised-visual-plan-v1","review_id":"Q08","revision_id":revision["revision_id"],"text":revision["text"],"source_bindings":inputs,"windows":windows,
        "selected_historical5f_files_verified":len(selected),"audit_scope":"Only used5f frozen artifact bindings and newQ08 source bindings; no full historic artifact audit.",
        "source_preview_binding":{k:preview[k] for k in ("path","sha256","bytes")},"parent_origin_sample_48k":origin48,
        "parent_local_start_sample_16k":revision["parent_local_start_sample_16k"],"parent_local_end_sample_16k":revision["parent_local_end_sample_16k"],
        "preserved_revision_diagnostics":revision["diagnostics"],"parent_vs_context_pcm":revision["parent_vs_context_pcm"],
        "selection":"Existing recording-origin25fps display bins intersecting exact revised audio windows; original PTS trace is copied provenance, not newly read original pixels.",
        "encoding":"Bound5f_r2 sequential preview decode verifies ordinal/PTS; raw25fps reencoding records selected BGR hashes and removes audio.",
        "actual_audio_confirmation":bind(EVIDENCE),"phrase_visual_boundary_approved":False,"training_ready":False,"actual_split":None,
        "new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_human_approval_added":False}
    save("plan.json",plan)
    videos={}
    for role,w in windows.items():
        videos[role]={**helper.muted_preview(preview,w["source_first_display_frame"],w["source_stop_display_frame"],OUTPUT/(role+"_lips_muted.mp4")),"clock_window":w}
    for b in inputs: verify(b)
    result={"schema":"offline-stage5g-revised-visual-result-v1","review_id":"Q08","revision_id":revision["revision_id"],"parent_id":revision["parent_id"],"text":revision["text"],
        "plan":bind(OUTPUT/"plan.json"),"audio_confirmation":bind(EVIDENCE),"videos":videos,"phrase_audio_boundary_approved":True,
        "audio_approval_scope":"Existing actual answer only for this exact+80ms revision; this producer adds no human approval.","phrase_visual_boundary_approved":False,
        "actual_visual_answer_pending":True,"visual_question_sent_by_producer":False,"training_ready":False,"actual_split":None,"original_training_frame_count":None,"ctc_feasible":None,
        "source_provisional":True,"new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_downloads":0,"new_human_approval_added":False,
        "selected_historical5f_files_verified":len(selected),"all_used_inputs_identical_after_execution":True,"historical_manifests_fully_reaudited":False,
        "precision_limits":["Muted recompressed review videos are not visual training inputs.","Selected display bins include recorded excess time before/after exact confirmed audio boundaries.","Copied original-frame PTS provenance is not a new extraction or visual approval.","Audio confirmation does not certify exact articulation boundaries."]}
    save("record.json",result)
    print(json.dumps({"record":str(OUTPUT/"record.json"),"phrase_frames":videos["phrase"]["frames"],"context_frames":videos["context"]["frames"]}),flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
