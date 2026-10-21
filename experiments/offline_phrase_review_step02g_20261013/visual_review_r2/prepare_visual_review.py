"""Prepare seven review-only boundary contacts and a read-only muted review UI."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE = BASE.parent / "selected_phrase_manifest_r1.json"
MEDIA = ROOT / "experiments/offline_alignment_step02_20261012/results/media_manifest.json"


def sha(path: Path) -> str:
    """Hash local preview and metadata sources."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict[str, Any]:
    """Read metadata without modifying its approval state."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def decode(spec: dict[str, Any]) -> list[Any]:
    """Fully decode a hash-verified existing 25fps ROI preview."""
    cap = None
    try:
        if sha(Path(spec["path"])) != spec["sha256"]:
            raise ValueError("Source preview hash changed")
        cap = cv2.VideoCapture(spec["path"])
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-25)>1e-8:
            raise ValueError("Source preview FPS/open failure")
        result=[]
        while True:
            ok, frame=cap.read()
            if not ok:
                break
            if frame.shape!=(96,96,3):
                raise ValueError("Source ROI dimensions changed")
            result.append(frame)
        if len(result)!=spec["frames"]:
            raise ValueError("Source ROI frame count changed")
        return result
    except Exception as exc:
        raise RuntimeError("Read-only preview decode failed") from exc
    finally:
        if cap is not None:
            cap.release()


def contact(row: dict[str, Any], frames: list[Any]) -> dict[str, Any]:
    """Tile unchanged ROI pixels in chronological boundary rows."""
    try:
        font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",13)
        image=Image.new("RGB",(1320,510),"white")
        draw=ImageDraw.Draw(image)
        draw.text((10,4),f'{row["rank"]}. {row["parent_id"]} | {row["text"]}',font=font,fill="black")
        draw.text((10,26),"시작·끝 ±0.20초 연속 표본 / 기존 압축 검수영상 / 부모 밖 표본 별도 표시 / 학습 입력·승인 아님",font=font,fill="black")
        for ridx,window in enumerate(row["boundary_windows"]):
            top=64+ridx*218
            draw.text((10,top),f'{window["name"]} {window["boundary_seconds"]:.6f}s',font=font,fill="black")
            for col,s in enumerate(window["samples"]):
                x,y=12+col*120,top+27
                image.paste(Image.fromarray(cv2.cvtColor(frames[s["review_frame_index"]],cv2.COLOR_BGR2RGB)),(x,y))
                draw.text((x,y+97),f'#{s["review_frame_index"]} {s["review_time_seconds"]:.3f}s',font=font,fill="black")
                draw.text((x,y+115),f'Δ{s["seconds_from_boundary"]:+.3f}s',font=font,fill="black")
                draw.text((x,y+133),f'native {s["source_trace"]["native_frame_index"]}',font=font,fill="black")
                if not s["within_confirmed_parent_time"]:
                    draw.text((x,y+151),"부모 문장 밖",font=font,fill="#9e4e00")
        path=BASE/(row["parent_id"]+"_boundaries.png")
        if path.exists():
            raise ValueError("Diagnostic contact already exists")
        image.save(path)
        return {"path":str(path),"sha256":sha(path),"pixel_resize":False}
    except Exception as exc:
        raise RuntimeError("Contact creation failed") from exc


def main() -> None:
    """Bind exact candidate samples to existing preview media and safe UI routes."""
    try:
        source,media=read(SOURCE),read(MEDIA)
        rows={r["id"]:r for r in media["records"]}
        if len(source["records"])!=7:
            raise ValueError("Expected seven candidates")
        hashes={str(p):sha(p) for p in (SOURCE,MEDIA,Path(__file__).resolve())}
        output=[]
        routes={}
        decoded_total=0
        for candidate in source["records"]:
            if candidate["phrase_audio_approval"] is not True or candidate["visual_approval"] is not None:
                raise ValueError("Expected audio-confirmed, visual-pending candidate")
            evidence_path=Path(candidate["audio_evidence_path"])
            if sha(evidence_path)!=candidate["audio_evidence_sha256"]:
                raise ValueError("Audio evidence hash differs")
            hashes[str(evidence_path)]=candidate["audio_evidence_sha256"]
            old=rows[candidate["parent_id"]]
            video=old["video"]
            frames=decode(video["roi_preview"])
            decoded_total+=len(frames)
            if len(frames)!=len(video["trace"]):
                raise ValueError("Preview trace length differs")
            start,end=candidate["start_sample_16k"]/16000,candidate["end_sample_16k"]/16000
            pstart,pend=candidate["parent_start_sample_16k"]/16000,candidate["parent_end_sample_16k"]/16000
            if not pstart<=start<end<=pend:
                raise ValueError("Candidate is outside confirmed parent")
            clipleft,clipright=max(pstart,start-.20),min(pend,end+.20)
            if abs(clipleft-candidate["context_audio"]["start_sample_16k"]/16000)>1e-8 or abs(clipright-candidate["context_audio"]["end_sample_16k"]/16000)>1e-8:
                raise ValueError("UI context bounds differ from root exact-audio context")
            row={"rank":candidate["rank"],"parent_id":candidate["parent_id"],"segment_id":candidate["segment_id"],
                "text":candidate["text"],"parent_text":candidate["parent_text"],
                "start_sample_16k":candidate["start_sample_16k"],"end_sample_16k":candidate["end_sample_16k"],
                "parent_start_sample_16k":candidate["parent_start_sample_16k"],"parent_end_sample_16k":candidate["parent_end_sample_16k"],
                "start_seconds":start,"end_seconds":end,"context_start_seconds":clipleft,"context_end_seconds":clipright,
                "context_parent_clamped":True,"context_after_seconds":clipright-end,
                "automatic_flags":candidate["flags"],"boundary_windows":[],"human_visual_approval":None,
                "phrase_audio_approval_at_ui_build":candidate["phrase_audio_approval"],
                "audio_evidence_path":candidate["audio_evidence_path"],"audio_evidence_sha256":candidate["audio_evidence_sha256"],
                "actual_audio_question_id":candidate["actual_audio_reply"]["questionItemId"],"training_ready":False}
            for key,sourcekey in (("roi_url","roi_preview"),("full_url","full_preview")):
                spec=video[sourcekey]
                if sha(Path(spec["path"]))!=spec["sha256"]:
                    raise ValueError("Declared media hash differs")
                hashes[spec["path"]]=spec["sha256"]
                url=f'/media/{row["rank"]}/{"roi" if key=="roi_url" else "full"}.mp4'
                routes[url]={"path":spec["path"],"sha256":spec["sha256"]}
                row[key]=url
            for name,boundary in (("start",start),("end",end)):
                first,last=math.ceil((boundary-.20)*25-1e-9),math.floor((boundary+.20)*25+1e-9)
                if not 0<=first<=last<len(frames):
                    raise ValueError("Full diagnostic context unavailable")
                samples=[]
                for index in range(first,last+1):
                    time_s=index/25
                    trace=video["trace"][index]
                    if abs(trace["target_video_time"]-(video["video_clock_start"]+time_s))>1e-8:
                        raise ValueError("Native trace timing differs")
                    samples.append({"review_frame_index":index,"review_time_seconds":time_s,
                        "seconds_from_boundary":time_s-boundary,"source_trace":trace,
                        "within_confirmed_parent_time":pstart<=time_s<pend,
                        "decoded_roi_bgr_sha256":hashlib.sha256(frames[index].tobytes()).hexdigest()})
                row["boundary_windows"].append({"name":name,"boundary_seconds":boundary,
                    "requested_seconds":[boundary-.20,boundary+.20],"samples":samples})
            row["contact_sheet"]=contact(row,frames)
            output.append(row)
        result={"schema":"offline-seven-phrase-visual-review-r2-v1","source_hashes":hashes,"records":output,
            "routes":routes,"decoded_roi_frames":decoded_total,"chronological_windows":14,"roi_panels":140,
            "status":"current_seven_audio_confirmed_human_visual_review_pending",
            "human_visual_approvals_created":0,"training_ready":False,"new_tar_reads":0,"new_neural_calls":0,
            "limitation":"Existing 25fps recompressed review video. Full +/-0.2s diagnostic stills may include parent-external frames, explicitly marked; interactive context playback clamps to confirmed parent.",
            "build_time_evidence_note":"This selected_phrase_manifest_r1 snapshot binds seven real audio confirmations but creates no human visual approval; later changes require a new version."}
        path=BASE/"visual_review_manifest.json"
        if path.exists():
            raise ValueError("Manifest already exists")
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        html=(BASE/"review_template.html").read_text(encoding="utf-8")
        data={"manifest_sha256":sha(path),"records":[{k:v for k,v in r.items() if k not in ("boundary_windows","contact_sheet")} for r in output]}
        html=html.replace("__REVIEW_DATA__",json.dumps(data,ensure_ascii=False).replace("<","\\u003c"))
        index=BASE/"index.html"
        if index.exists():
            raise ValueError("Page already exists")
        index.write_text(html,encoding="utf-8")
        print(json.dumps({"records":len(output),"decoded_roi_frames":decoded_total,"roi_panels":140,"html_sha256":sha(index)}))
    except Exception as exc:
        raise RuntimeError("Seven-phrase visual preparation failed") from exc


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
