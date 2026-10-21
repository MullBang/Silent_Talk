"""Create muted Q01 revised-window previews from the frozen5f review video.

This is visualization of existing recompressed review pixels, never original
training pixels. The separately confirmed audio range does not approve any
visual boundary. Only the Q01 parent-end revision and its +-0.20s context are
rendered; source files and previous snapshots remain immutable.
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
OUTPUT=BASE/"visual_Q01_r1"
REVISION=BASE/"revisions/Q01_end_parent/record.json"
EVIDENCE=BASE/"user_evidence/002_Q01_revised_audio_confirmed.json"
SNAPSHOT=BASE/"results/review_snapshot_manifest_r1.json"
OLD_MANIFEST=OLD/"results/completion_manifest.json"
MEDIA=OLD/"media/lip_J_1_F_02_C002_A_012_051/record.json"
HELPER=OLD/"media_prepare/prepare_phrase_media_r2.py"
HELPER_MEDIA=OLD/"media_prepare/prepare_media.py"


def require(condition: bool,message: str) -> None:
    """Reject changed evidence and scope expansion explicitly."""
    if not condition: raise ValueError(message)


def sha(path: Path) -> str:
    """Read a bound local artifact's digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def bind(path: Path) -> dict[str,Any]:
    """Capture full local source identity."""
    return {"path":str(path.resolve()),"sha256":sha(path),"bytes":path.stat().st_size}


def read(path: Path) -> Any:
    """Read UTF-8 JSON metadata only."""
    return json.loads(path.read_text(encoding="utf-8"))


def verify(binding: dict[str,Any]) -> None:
    """Compare immutable input bytes without loading model tensors."""
    p=Path(binding["path"])
    require(sha(p)==binding["sha256"] and p.stat().st_size==binding["bytes"],f"Bound source differs: {p}")


def save(path: Path,value: Any) -> None:
    """Write only a new artifact in this dedicated visualization directory."""
    require(path.resolve().is_relative_to(OUTPUT.resolve()) and not path.exists(),"Unsafe/existing visualization output")
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def run() -> None:
    """Verify both snapshots, select exact display bins, encode and check2videos."""
    require(not OUTPUT.exists(),"Existing Q01 visual evidence must be preserved")
    frozen5f=read(OLD_MANIFEST)["artifacts"]
    frozen5g=read(SNAPSHOT)["artifacts"]
    require(len(frozen5f)==247 and len(frozen5g)==18,"Unexpected frozen snapshot scope")
    for spec in frozen5f+frozen5g: verify(spec)
    source_bindings=[bind(p) for p in (REVISION,EVIDENCE,SNAPSHOT,OLD_MANIFEST,MEDIA,HELPER,HELPER_MEDIA,Path(__file__))]
    revision,evidence,media=read(REVISION),read(EVIDENCE),read(MEDIA)
    require(revision["revision_id"]==evidence["revision_id"]=="Q01_end_parent_r1" and revision["review_id"]=="Q01","Unexpected revision")
    require(evidence["revision_record"]==bind(REVISION) and evidence["phrase_audio_boundary_approved"] is True and evidence["phrase_visual_boundary_approved"] is False,"Expected actual revised-audio confirmation only")
    require(evidence["question_audio"]=={k:revision["media"]["phrase"][k] for k in ("path","sha256","bytes")},"Question audio identity differs")
    require(revision["parent_id"]==media["parent_id"] and revision["text"]=="너무 속상해","Q01 parent/text changed")
    require(revision["training_ready"] is False and revision["actual_split"] is None,"Review status changed")
    source_preview=media["video"]["roi_preview"]
    source_bindings.append({k:source_preview[k] for k in ("path","sha256","bytes")})
    for role in ("phrase","context"):
        source_bindings.append({k:revision["media"][role][k] for k in ("path","sha256","bytes")})
    for spec in source_bindings: verify(spec)
    sys.path.insert(0,str(HELPER.parent))
    module_spec=importlib.util.spec_from_file_location("frozen_stage5f_visual_r2",HELPER)
    require(module_spec is not None and module_spec.loader is not None,"Cannot import frozen review encoder")
    helper=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helper)
    offset=Fraction(media["offset_native_samples"],48000)
    origin=helper.fraction(media["video"]["display_video_start"])
    total=source_preview["frames"]
    windows={}
    for role in ("phrase","context"):
        audio=revision["media"][role]
        start=Fraction(audio["source_original_wav_first_sample_48k"],48000)
        end=Fraction(audio["source_original_wav_end_sample_48k"],48000)
        require(end-start==Fraction(audio["samples"],16000),"Audio source duration differs")
        first,stop=helper.frame_window(start+offset,end+offset,origin,total)
        windows[role]={"audio_binding":{k:audio[k] for k in ("path","sha256","bytes")},"semantic_original_wav_start":helper.ratio(start),"semantic_original_wav_end":helper.ratio(end),
                       "semantic_video_start":helper.ratio(start+offset),"semantic_video_end":helper.ratio(end+offset),"source_first_display_frame":first,"source_stop_display_frame":stop,
                       "preview_display_video_start":helper.ratio(origin+Fraction(first,25)),"preview_display_video_end":helper.ratio(origin+Fraction(stop,25)),
                       "display_before_semantic_seconds":helper.ratio(start+offset-origin-Fraction(first,25)),"display_after_semantic_seconds":helper.ratio(origin+Fraction(stop,25)-end-offset),
                       "original_grid_trace_slice":media["video"]["trace"][first:stop],"offset_applied_count":1}
    require(helper.fraction(windows["phrase"]["semantic_original_wav_start"])-helper.fraction(windows["context"]["semantic_original_wav_start"])==Fraction(1,5),"Context start differs from minus0.20s")
    require(helper.fraction(windows["context"]["semantic_original_wav_end"])-helper.fraction(windows["phrase"]["semantic_original_wav_end"])==Fraction(1,5),"Context end differs from plus0.20s")
    source_bindings.append(bind(helper.media.FFMPEG))
    OUTPUT.mkdir()
    plan={"schema":"offline-stage5g-Q01-revised-visual-plan-v1","review_id":"Q01","revision_id":"Q01_end_parent_r1","text":revision["text"],"source_bindings":source_bindings,"windows":windows,
          "source_preview_binding":{k:source_preview[k] for k in ("path","sha256","bytes")},"freeze_checks":{"stage5f_entries":247,"stage5g_r1_entries":18},
          "selection":"Intersect existing recording-origin25fps display bins. Exact copied5f native-PTS traces are provenance only; no original video pixels are read.",
          "encoding":"Sequentially decode existing source preview through selected bins; verify each ordinal/PTS; reencode raw25fps using the bound5f_r2 encoder; strip audio.",
          "phrase_audio_confirmed_by_actual_evidence":True,"phrase_visual_boundary_approved":False,"training_ready":False,"actual_split":None,
          "new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_human_approval_added":False}
    save(OUTPUT/"plan.json",plan)
    results={}
    for role,window in windows.items():
        result=helper.muted_preview(source_preview,window["source_first_display_frame"],window["source_stop_display_frame"],OUTPUT/(role+"_lips_muted.mp4"))
        results[role]={**result,"clock_window":window}
    for spec in source_bindings: verify(spec)
    for spec in frozen5f+frozen5g: verify(spec)
    result={"schema":"offline-stage5g-Q01-revised-visual-result-v1","review_id":"Q01","revision_id":"Q01_end_parent_r1","text":revision["text"],"parent_id":revision["parent_id"],
            "plan":bind(OUTPUT/"plan.json"),"audio_confirmation":bind(EVIDENCE),"revision_record":bind(REVISION),"videos":results,
            "phrase_audio_boundary_approved":True,"audio_approval_scope":"Existing actual answer for the revised exact audio only; this production adds no approval.","phrase_visual_boundary_approved":False,
            "actual_visual_answer_pending":True,"visual_question_sent_by_producer":False,"training_ready":False,"actual_split":None,
            "original_training_frame_count":None,"ctc_feasible":None,"source_provisional":True,"new_tar_payload_bytes":0,"new_neural_calls":0,"new_training_pixels":0,"new_human_approval_added":False,
            "source_preservation":{"stage5f_entries":247,"stage5g_r1_entries":18,"all_inputs_identical_after_execution":True},
            "precision_limits":["Both clips are reencoded review pixels; they must not be used as student training input.","Display-bin overlap can show movement before or after the exact confirmed audio range, with exact excesses recorded.","Video bins retain prior PTS-nearest source provenance but a human has not yet assessed this revised visual boundary.","This is an audio-confirmed phrase, not a newly certified phonetic or articulatory interval."]}
    save(OUTPUT/"record.json",result)
    print(json.dumps({"record":str(OUTPUT/"record.json"),"phrase_frames":results["phrase"]["frames"],"context_frames":results["context"]["frames"],"all_inputs_preserved":True}),flush=True)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
