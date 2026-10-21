"""Prepare exact PCM and muted review video for eight fixed CTC proposals.

The 16kHz question audio is an exact slice of the existing context PCM. The
acoustic alignment used separately resampled parent audio; that distinction and
any observed PCM difference are retained. Preview display bins are not approved
articulation bounds or visual training caches.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import wave

import cv2
import numpy as np

import prepare_media as media

BASE=Path(__file__).resolve().parent
STAGE=BASE.parent
OUTPUT=STAGE/"phrase_media_r2"
ALIGNMENT=STAGE/"alignment/execution/alignment_results.json"
PARENTS=STAGE/"effective_parent_inputs.json"
PLAN=BASE/"phrase_media_plan_r2.json"


def fraction(spec: dict[str,Any]) -> Fraction:
    """Read a declared exact fraction while rejecting nonintegral fields."""
    media.require(type(spec["numerator"]) is int and type(spec["denominator"]) is int and spec["denominator"]>0,"Invalid rational")
    return Fraction(spec["numerator"],spec["denominator"])


def ratio(value: Fraction) -> dict[str,int]:
    """Save exact rational time without losing the source origin."""
    return {"numerator":value.numerator,"denominator":value.denominator}


def integer_sample(time_value: Fraction, origin: Fraction) -> int:
    """Require exact 16kHz sample coincidence, never silently round a proposal."""
    value=(time_value-origin)*16000
    media.require(value.denominator==1 and value>=0,"Proposal does not coincide with saved context sample")
    return int(value)


def frame_window(start: Fraction, end: Fraction, origin: Fraction, total: int) -> tuple[int,int]:
    """Select preexisting preview display bins intersecting the semantic window."""
    media.require(origin<=start<end<=origin+Fraction(total,25),"Proposal outside preview")
    s,e=(start-origin)*25,(end-origin)*25
    first=s.numerator//s.denominator
    stop=-(-e.numerator//e.denominator)
    media.require(0<=first<stop<=total,"Invalid display selection")
    return first,stop


def pcm(path: Path) -> np.ndarray:
    """Load the small mono16k review PCM, preserving signed integer samples."""
    with wave.open(str(path),"rb") as stream:
        media.require((stream.getframerate(),stream.getnchannels(),stream.getsampwidth())==(16000,1,2),"Unsupported review PCM")
        result=np.frombuffer(stream.readframes(stream.getnframes()),dtype="<i2").copy()
        media.require(len(result)==stream.getnframes(),"Truncated review PCM")
        return result


def save_slice(source: dict[str,Any], first: int, stop: int, path: Path) -> dict[str,Any]:
    """Write exact source-context sample bytes, with no resampling or padding."""
    media.require(media.bind(Path(source["path"]))=={k:source[k] for k in ("path","sha256","bytes")},"Saved context changed")
    values=pcm(Path(source["path"]))
    media.require(type(first) is int and type(stop) is int and 0<=first<stop<=len(values),"Invalid PCM slice")
    media.require(not path.exists(),"Refuse existing phrase PCM")
    samples=values[first:stop].tobytes()
    with wave.open(str(path),"wb") as stream:
        stream.setframerate(16000);stream.setnchannels(1);stream.setsampwidth(2);stream.writeframes(samples)
    media.require(pcm(path).tobytes()==samples,"Saved question differs from exact context slice")
    return {**media.bind(path),"sample_rate":16000,"channels":1,"sample_width_bytes":2,"samples":stop-first,
            "duration_seconds":(stop-first)/16000,"source_context_binding":{k:source[k] for k in ("path","sha256","bytes")},
            "source_context_first_sample_16k":first,"source_context_end_sample_16k":stop,"pcm_sha256":hashlib.sha256(samples).hexdigest(),
            "source_original_wav_first_sample_48k":source["source_first_sample"]+3*first,
            "source_original_wav_end_sample_48k":source["source_first_sample"]+3*stop,
            "new_resampling":False,"padding":False,"fade":False,"review_only":True}


def muted_preview(source: dict[str,Any], first: int, stop: int, output: Path) -> dict[str,Any]:
    """Re-encode selected decoded preview bins at explicit25fps without audio.

    Raw fixed-cadence input avoids inherited terminal-frame duration/edit-list
    metadata seen in the separately preserved first trimming attempt.
    """
    media.require(media.bind(Path(source["path"]))=={k:source[k] for k in ("path","sha256","bytes")},"Source preview changed")
    media.require(0<=first<stop<=source["frames"] and not output.exists(),"Invalid preview selection")
    capture=cv2.VideoCapture(source["path"])
    frames=[]
    try:
        media.require(capture.isOpened(),"Cannot decode source preview")
        for index in range(stop):
            ok,frame=capture.read()
            media.require(ok,"Source preview ended before selected display bin")
            media.require(capture.get(cv2.CAP_PROP_POS_FRAMES)-1==index and abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-index/25)<=1e-6,"Source display PTS/ordinal differs")
            if index>=first: frames.append(frame.copy())
    finally:
        capture.release()
    media.require(len(frames)==stop-first,"Selected raw preview frames incomplete")
    height,width=frames[0].shape[:2]
    command=[str(media.FFMPEG),"-nostdin","-v","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}","-r","25","-i","pipe:0","-map","0:v:0",
             "-an","-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart","-n",str(output)]
    result=subprocess.run(command,input=b"".join(frame.tobytes() for frame in frames),capture_output=True,timeout=120)
    media.require(result.returncode==0,"Preview trimming failed: "+result.stderr.decode(errors="replace"))
    capture=cv2.VideoCapture(str(output))
    try:
        media.require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==stop-first and abs(capture.get(cv2.CAP_PROP_FPS)-25)<1e-9,"Trimmed frame count/FPS differs")
        decoded=0
        while capture.read()[0]: decoded+=1
        media.require(decoded==stop-first,"Trimmed preview does not fully decode")
    finally:
        capture.release()
    # Map the first audio track explicitly: success would mean unwanted audio.
    check=subprocess.run([str(media.FFMPEG),"-nostdin","-v","error","-i",str(output),"-map","0:a:0","-f","null","-"],capture_output=True,timeout=30)
    media.require(check.returncode!=0 and b"matches no streams" in check.stderr,"Unexpected audio or failed audio absence check")
    return {**media.bind(output),"source_preview_binding":{k:source[k] for k in ("path","sha256","bytes")},"source_first_display_frame":first,"source_stop_display_frame":stop,
            "source_decoded_selected_bgr_sha256":[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],"encoding_clock":"explicit raw25fps; no input presentation metadata inherited",
            "frames":stop-first,"fps":25,"duration_seconds":(stop-first)/25,"audio_tracks":0,"decoded_all_frames":True,"command":command,
            "review_only":True,"recompressed_preview_not_training_input":True,"training_frame_count":None}


def prepare() -> None:
    """Bind the completed acoustic proposals and exact source media before writes."""
    alignment,parents=media.read(ALIGNMENT),media.read(PARENTS)
    media.require(alignment["neural_calls"]==8 and len(alignment["records"])==8,"Expected eight completed acoustic calls")
    media.require(alignment["training_ready"] is False and alignment["human_approvals_added"]==0,"Unexpected automatic approval")
    byid={r["review_id"]:r for r in parents["records"]}
    selected=[];bindings=[media.bind(ALIGNMENT),media.bind(PARENTS),media.bind(Path(__file__)),media.bind(BASE/"prepare_media.py"),media.bind(media.FFMPEG)]
    for index,a in enumerate(alignment["records"],1):
        ident=f"Q{index:02d}";p=byid[ident]
        media.require(a["review_id"]==ident and a["parent_id"]==p["parent_id"] and a["candidate_id"]==p["candidate_id"],"Fixed candidate changed")
        m=media.read(Path(p["media_record"]["path"]))
        media.require(media.bind(Path(p["media_record"]["path"]))==p["media_record"],"Media evidence changed")
        b={k:p["audio"][k] for k in ("path","sha256","bytes")}
        media.require(media.bind(Path(b["path"]))==b==a["audio"],"Acoustic input differs from reviewed parent")
        phrase=a["selected_phrase"]
        media.require(phrase["text"]==p["phrase_text"] and a["canonical_parent_text"]==p["parent_text"],"Text/indices changed")
        media.require(phrase["human_phrase_audio_boundary_approved"] is False and phrase["human_phrase_visual_boundary_approved"] is False and phrase["frame_count"] is None,"Unapproved proposal expected")
        start,end=fraction(phrase["wav_start_seconds_exact"]),fraction(phrase["wav_end_seconds_exact"])
        context_origin=Fraction(m["context_audio"]["source_first_sample"],48000)
        left,right=integer_sample(start,context_origin),integer_sample(end,context_origin)
        media.require(0<=left-3200<right+3200<=m["context_audio"]["samples"],"Exact +-0.20 context outside saved PCM; hold")
        off=Fraction(m["offset_native_samples"],48000)
        media.require(fraction(phrase["video_start_seconds_exact"])==start+off and fraction(phrase["video_end_seconds_exact"])==end+off and phrase["video_offset_applied_count"]==1,"Offset applied incorrectly")
        display_origin=fraction(m["video"]["display_video_start"])
        spans={}
        for role,s,e in (("phrase",start,end),("phrase_context",start-Fraction(1,5),end+Fraction(1,5))):
            first,stop=frame_window(s+off,e+off,display_origin,m["video"]["roi_preview"]["frames"])
            spans[role]={"first":first,"stop":stop,"semantic_video_start":ratio(s+off),"semantic_video_end":ratio(e+off),
                         "preview_display_video_start":ratio(display_origin+Fraction(first,25)),"preview_display_video_end":ratio(display_origin+Fraction(stop,25)),
                         "preview_before_semantic_seconds":ratio(s+off-display_origin-Fraction(first,25)),"preview_after_semantic_seconds":ratio(display_origin+Fraction(stop,25)-e-off)}
        selected.append({"review_id":ident,"parent_id":p["parent_id"],"candidate_id":p["candidate_id"],"text":phrase["text"],"parent_text":p["parent_text"],
                         "aligned_proposal":phrase,"parent":p,"original_media":m,"context_start_sample_16k":left,"context_end_sample_16k":right,"video_windows":spans})
        bindings.extend([p["media_record"],b,{k:m["context_audio"][k] for k in ("path","sha256","bytes")},{k:m["video"]["roi_preview"][k] for k in ("path","sha256","bytes")}])
    media.save(PLAN,{"schema":"offline-stage5f-phrase-review-media-plan-v1","bindings":bindings,"records":selected,"allowed_review_ids":[f"Q{i:02d}" for i in range(1,9)],
                     "audio_rule":"Exact integer slices of saved context16k; separate parent-resample input retained and compared without claiming byte equivalence.",
                     "video_rule":"Intersect existing D-grid compressed-preview bins; muted and recompressed; display overlap explicitly retained, never promoted to source-frame cache.",
                     "new_tar_payload_bytes":0,"new_neural_calls":0,"training_updates":0,"human_approvals_added":0,"training_ready":False})
    print(json.dumps({"plan":media.bind(PLAN),"records":8}),flush=True)


def execute() -> None:
    """Save only the bound eight review candidates, keeping all source bytes intact."""
    plan=media.read(PLAN)
    for binding in plan["bindings"]:
        media.require(media.bind(Path(binding["path"]))==binding,"Bound phrase-media input changed")
    media.require(not OUTPUT.exists(),"Refuse existing phrase-media results")
    OUTPUT.mkdir()
    results=[]
    for r in plan["records"]:
        folder=OUTPUT/r["review_id"];folder.mkdir()
        m,p,proposal=r["original_media"],r["parent"],r["aligned_proposal"]
        left,right=r["context_start_sample_16k"],r["context_end_sample_16k"]
        audio=save_slice(m["context_audio"],left,right,folder/"phrase.wav")
        context=save_slice(m["context_audio"],left-3200,right+3200,folder/"phrase_context.wav")
        exact=pcm(Path(audio["path"]))
        parent=pcm(Path(p["audio"]["path"]))[proposal["local_audio_start_sample_16k"]:proposal["local_audio_end_sample_16k"]]
        media.require(len(parent)==len(exact),"Parent and context proposal lengths differ")
        diff=exact.astype(np.int32)-parent.astype(np.int32)
        compare={"aligned_parent_input":{k:p["audio"][k] for k in ("path","sha256","bytes")},"parent_local_first_sample_16k":proposal["local_audio_start_sample_16k"],
                 "parent_local_end_sample_16k":proposal["local_audio_end_sample_16k"],"samples":len(exact),"unequal_samples":int(np.count_nonzero(diff)),
                 "maximum_absolute_int16_difference":int(np.max(np.abs(diff))),"context_and_parent_resampled_independently":True,
                 "same_exact_source_time_origin":True,"no_new_neural_inference":True,"interpretation":"Small edge-filter differences can arise from independently resampled native excerpts; this question is bound specifically to its exact context PCM slice."}
        clips={}
        for role,w in r["video_windows"].items():
            clips[role]={**muted_preview(m["video"]["roi_preview"],w["first"],w["stop"],folder/(role+"_lips_muted.mp4")),"clock_window":w}
        value={"schema":"offline-stage5f-phrase-review-candidate-v1","review_id":r["review_id"],"parent_id":r["parent_id"],"candidate_id":r["candidate_id"],"text":r["text"],"parent_text":r["parent_text"],
               "source_plan":media.bind(PLAN),"aligned_proposal":proposal,"phrase_audio":audio,"phrase_context_audio":context,"parent_vs_context_pcm_comparison":compare,"muted_video":clips,
               "phrase_within_phrase_context_samples_16k":[3200,3200+audio["samples"]],"phrase_audio_boundary_approved":False,"phrase_visual_boundary_approved":False,
               "parent_review_does_not_approve_phrase":True,"new_human_approval_added":False,"word_boundaries_are_ctc_proposals":True,"training_ready":False,"actual_split":None,
               "actual_training_frame_count":None,"ctc_feasible":None,"source_provisional":True}
        media.save(folder/"record.json",value);results.append(value)
        print(json.dumps({"review_id":r["review_id"],"audio":audio["path"],"duration_seconds":audio["duration_seconds"],"pcm_unequal_samples":compare["unequal_samples"]}),flush=True)
    for binding in plan["bindings"]:
        media.require(media.bind(Path(binding["path"]))==binding,"Phrase source changed during processing")
    media.save(OUTPUT/"manifest.json",{"schema":"offline-stage5f-phrase-review-manifest-v1","records":results,"records_count":len(results),"source_plan":media.bind(PLAN),
               "status":"eight_automatic_phrase_candidates_ready_human_review_pending","new_tar_payload_bytes":0,"new_neural_calls":0,"new_human_approvals":0,"training_updates":0,"training_ready":False})


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    media.require(args.prepare != args.execute,"Choose prepare or execute")
    if args.prepare: prepare()
    else: execute()
