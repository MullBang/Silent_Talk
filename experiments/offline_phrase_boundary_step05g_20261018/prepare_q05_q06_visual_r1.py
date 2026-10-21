"""Render revised Q05/Q06 muted review windows from existing preview pixels.

Only selected frozen input bindings are reverified, not all historical files.
Audio confirmation stays separate from pending visual review. Native source
sample quantization, including Q05's one-sample earlier nominal end, is retained.
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
FROZEN_F=OLD/"results/completion_manifest.json"
FROZEN_G=BASE/"results/review_snapshot_manifest_r3.json"
HELPER=OLD/"media_prepare/prepare_phrase_media_r2.py"
MEDIA_HELPER=OLD/"media_prepare/prepare_media.py"
SELECTED=[("Q05","Q05_end_confirmed_parent","006_Q05_revised_audio_confirmed.json","lip_J_1_F_03_C282_A_011_012"),
          ("Q06","Q06_end_plus40ms","008_Q06_plus40ms_audio_confirmed.json","lip_J_1_F_03_C293_A_008_022")]


def require(condition: bool,message: str) -> None:
    """Reject changed inputs or scope expansion explicitly."""
    if not condition: raise ValueError(message)


def sha(path: Path) -> str:
    """Compute a file digest without loading tensors or original media."""
    with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()


def bind(path: Path) -> dict[str,Any]:
    """Bind full local path, bytes and SHA."""
    return {"path":str(path.resolve()),"sha256":sha(path),"bytes":path.stat().st_size}


def read(path: Path) -> Any:
    """Read declared JSON evidence as data."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify(spec: dict[str,Any]) -> None:
    """Verify exactly the fields of an input binding."""
    require(bind(Path(spec["path"]))=={k:spec[k] for k in ("path","sha256","bytes")},"Input changed: "+spec["path"])


def save(folder: Path,name: str,value: Any) -> None:
    """Create a result only under one of the two assigned directories."""
    require(folder.resolve() in {(BASE/"visual_Q05_r1").resolve(),(BASE/"visual_Q06_r1").resolve()},"Unsafe output folder")
    path=folder/name
    require(path.parent==folder and not path.exists(),"Unsafe/existing result")
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def run() -> None:
    """Render exactly two revised phrases and their +-0.20s contexts."""
    manifests={"5f":read(FROZEN_F),"5g_r3":read(FROZEN_G)}
    require(len(manifests["5f"]["artifacts"])==247 and len(manifests["5g_r3"]["artifacts"])==75,"Unexpected historical snapshot")
    catalog={str(Path(b["path"]).resolve()):b for m in manifests.values() for b in m["artifacts"]}
    checked={}
    def frozen(path: Path) -> dict[str,Any]:
        """Require manifest membership then verify only this selected artifact."""
        key=str(path.resolve());require(key in catalog,"Input absent from frozen manifests: "+key)
        verify(catalog[key]);checked[key]=catalog[key]
        return catalog[key]
    frozen(HELPER);frozen(MEDIA_HELPER)
    sys.path.insert(0,str(HELPER.parent))
    spec=importlib.util.spec_from_file_location("stage5g_selected_review_encoder",HELPER)
    require(spec is not None and spec.loader is not None,"Cannot load selected frozen helper")
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    ffmpeg=bind(helper.media.FFMPEG)
    require(ffmpeg["sha256"]=="2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3","Bound FFmpeg executable differs")
    common=[bind(FROZEN_F),bind(FROZEN_G),bind(Path(__file__)),ffmpeg]
    prepared=[]
    for ident,revision_dir,evidence_name,parent_id in SELECTED:
        folder=BASE/("visual_"+ident+"_r1")
        require(not folder.exists(),"Existing visualization will not be overwritten")
        revision_path=BASE/"revisions"/revision_dir/"record.json"
        evidence_path=BASE/"user_evidence"/evidence_name
        media_path=OLD/"media"/parent_id/"record.json"
        own_inputs=[frozen(p) for p in (revision_path,evidence_path,media_path,HELPER,MEDIA_HELPER)]
        revision,evidence,media=read(revision_path),read(evidence_path),read(media_path)
        require(revision["review_id"]==evidence["review_id"]==ident and revision["parent_id"]==media["parent_id"]==parent_id,"Candidate identity differs")
        require(evidence["revision_record"]==bind(revision_path) and evidence["revision_id"]==revision["revision_id"],"Actual answer belongs to another revision")
        require(evidence["phrase_audio_boundary_approved"] is True and evidence["phrase_visual_boundary_approved"] is False,"Expected audio-only actual approval")
        require(revision["training_ready"] is False and revision["actual_split"] is None,"Unexpected training status")
        phrase_audio=revision["media"]["phrase"]
        require(evidence["question_audio"]=={k:phrase_audio[k] for k in ("path","sha256","bytes")},"Actual answer question audio differs")
        preview=media["video"]["roi_preview"]
        own_inputs.append(frozen(Path(preview["path"])))
        for role in ("phrase","context"):
            own_inputs.append(frozen(Path(revision["media"][role]["path"])))
            own_inputs.append(frozen(Path(revision["media"][role]["source"]["path"])))
        origin48=revision["parent_origin_sample_48k"]
        require(origin48==media["parent_native_start_sample"],"Parent native origin differs")
        start=Fraction(origin48+3*revision["parent_local_start_sample_16k"],48000)
        end=Fraction(origin48+3*revision["parent_local_end_sample_16k"],48000)
        require(start==helper.fraction(revision["audio_start_seconds_exact"]) and end==helper.fraction(revision["audio_end_seconds_exact"]),"Exact native/local audio clocks differ")
        require(end-start==Fraction(phrase_audio["samples"],16000),"Confirmed audio length differs")
        offset=Fraction(media["offset_native_samples"],48000)
        require(helper.fraction(revision["video_start_seconds_exact"])==start+offset and helper.fraction(revision["video_end_seconds_exact"])==end+offset and revision["video_offset_applied_count"]==1,"Offset formula differs")
        if ident=="Q05":
            diag=revision["diagnostics"]
            require(diag["nominal_revised_end_sample_48k"]==int(end*48000) and diag["nominal_end_minus_native_annotation_samples_48k"]==-1,"Q05 quantization evidence differs")
            require(int(end*48000)==media["parent_native_end_sample"]-1,"Q05 nominal earlier end was not preserved")
        display_origin=helper.fraction(media["video"]["display_video_start"])
        windows={}
        for role,s,e in (("phrase",start,end),("context",start-Fraction(1,5),end+Fraction(1,5))):
            audio=revision["media"][role]
            require(e-s==Fraction(audio["samples"],16000),"Context is not exactly +-0.20s")
            if role=="context":
                native_context=media["context_audio"]["source_first_sample"]
                require(s==Fraction(native_context+3*audio["source_first_sample_16k"],48000) and e==Fraction(native_context+3*audio["source_end_sample_16k"],48000),"Saved context clock differs")
            first,stop=helper.frame_window(s+offset,e+offset,display_origin,preview["frames"])
            windows[role]={"audio_binding":{k:audio[k] for k in ("path","sha256","bytes")},"semantic_original_wav_start":helper.ratio(s),"semantic_original_wav_end":helper.ratio(e),
                "semantic_video_start":helper.ratio(s+offset),"semantic_video_end":helper.ratio(e+offset),"source_first_display_frame":first,"source_stop_display_frame":stop,
                "preview_display_video_start":helper.ratio(display_origin+Fraction(first,25)),"preview_display_video_end":helper.ratio(display_origin+Fraction(stop,25)),
                "display_before_semantic_seconds":helper.ratio(s+offset-display_origin-Fraction(first,25)),"display_after_semantic_seconds":helper.ratio(display_origin+Fraction(stop,25)-e-offset),
                "original_grid_trace_slice":media["video"]["trace"][first:stop],"offset_applied_count":1}
        plan={"schema":"offline-stage5g-revised-visual-plan-v1","review_id":ident,"revision_id":revision["revision_id"],"text":revision["text"],"source_bindings":own_inputs+common,
            "snapshot_manifest_artifact_counts":{"5f":247,"5g_r3":75},"audit_scope":"Only used frozen artifact bindings verified; manifests read for membership. No full historical artifact re-audit.",
            "windows":windows,"source_preview_binding":{k:preview[k] for k in ("path","sha256","bytes")},"parent_origin_sample_48k":origin48,
            "parent_local_start_sample_16k":revision["parent_local_start_sample_16k"],"parent_local_end_sample_16k":revision["parent_local_end_sample_16k"],
            "preserved_revision_diagnostics":revision["diagnostics"],"parent_vs_context_pcm":revision["parent_vs_context_pcm"],
            "selection":"Existing recording-origin25fps display bins intersecting exact revised audio windows; copy existing PTS traces for provenance, do not reread original pixels.",
            "encoding":"Verified5f_r2 sequential preview decode with ordinal/PTS validation, selected BGR hashes, explicit raw25fps reencoding and no audio.",
            "actual_audio_confirmation":frozen(evidence_path),"phrase_visual_boundary_approved":False,"training_ready":False,"actual_split":None,
            "new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_human_approval_added":False}
        prepared.append((folder,revision,preview,windows,plan,evidence_path))
    for folder,revision,preview,windows,plan,evidence_path in prepared:
        folder.mkdir();save(folder,"plan.json",plan)
        videos={}
        for role,w in windows.items():
            videos[role]={**helper.muted_preview(preview,w["source_first_display_frame"],w["source_stop_display_frame"],folder/(role+"_lips_muted.mp4")),"clock_window":w}
        for b in plan["source_bindings"]: verify(b)
        result={"schema":"offline-stage5g-revised-visual-result-v1","review_id":revision["review_id"],"revision_id":revision["revision_id"],"parent_id":revision["parent_id"],"text":revision["text"],
            "plan":bind(folder/"plan.json"),"audio_confirmation":bind(evidence_path),"videos":videos,"phrase_audio_boundary_approved":True,
            "audio_approval_scope":"Existing actual answer only for this revised audio; no new approval is generated by this producer.","phrase_visual_boundary_approved":False,
            "actual_visual_answer_pending":True,"visual_question_sent_by_producer":False,"training_ready":False,"actual_split":None,"original_training_frame_count":None,"ctc_feasible":None,
            "source_provisional":True,"new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_downloads":0,"new_human_approval_added":False,
            "selected_frozen_inputs_verified":len({b["path"] for b in plan["source_bindings"] if b["path"] in catalog}),"all_used_inputs_identical_after_execution":True,
            "historical_manifests_fully_reaudited":False,"quantization_preserved":revision["diagnostics"],
            "precision_limits":["These are muted, recompressed review pixels, never student training inputs.","Display-bin overlap can include motion before or after confirmed audio; exact excesses are recorded separately.","Original source PTS traces are copied provenance, not newly extracted original pixels or human visual approval.","Confirmed audio does not establish exact articulatory boundaries."]}
        save(folder,"record.json",result)
        print(json.dumps({"review_id":revision["review_id"],"record":str(folder/"record.json"),"phrase_frames":videos["phrase"]["frames"],"context_frames":videos["context"]["frames"]}),flush=True)
    for b in checked.values(): verify(b)
    for b in common: verify(b)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
