"""Prepare only Q01-Q08 review evidence from audited TAR members.

No labels are approved, no acoustic or visual model is loaded, and all saved
previews are review-only. Native PCM and recording-anchored presentation times
retain separate exact origins; their existing measured offset is applied once.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator
import wave

import cv2
import numpy as np
import scipy

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE.parents[1]
MEDIA = STAGE / "media"
PLAN = BASE / "source_plan.json"
QUEUE = ROOT / "experiments/offline_phrase_design_step05e_20261018/design/review_queue.json"
HANDOFF = ROOT / "experiments/offline_phrase_design_step05e_20261018/results/next_step_handoff.json"
ELIGIBLE = ROOT / "experiments/offline_phrase_design_step05e_20261018/data/eligible_sentences.json"
SOURCE = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"
MAPPING = ROOT / "experiments/training_alignment_step06/results/audited_time_mapping_candidates.json"
TAR = ROOT / "experiments/local_source_step01_20261007/tar_source.py"
SCORE = ROOT / "experiments/training_alignment_step06/refine_alignment.py"
OLD = ROOT / "experiments/offline_alignment_step02_20261012/prepare_media.py"
CLOCK = ROOT / "experiments/offline_dataset_step03_20261013/extraction/exact_clock.py"
FFMPEG = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
RESERVE = 10 * 1024**3


def require(condition: bool, message: str) -> None:
    """Reject unsupported input or changed evidence explicitly."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash a local artifact in bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path: Path) -> dict[str, Any]:
    """Bind the identity, digest and length of evidence."""
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def read(path: Path) -> Any:
    """Read JSON evidence without changing it."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: Any) -> None:
    """Create only new artifacts under this assigned experiment."""
    require(path.resolve().is_relative_to(STAGE.resolve()) and not path.exists(), "Unsafe/existing output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def helper(path: Path, name: str) -> Any:
    """Import a prebound helper without invoking its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(value: float | Fraction, rate: int = 48000) -> int:
    """Quantize decimal metadata to nearest native sample, ties-to-even."""
    return round((value if isinstance(value, Fraction) else Fraction(str(value))) * rate)


def prepare() -> None:
    """Bind exactly the fixed eight parents without reading TAR payloads."""
    handoff, queue, eligible, sources = read(HANDOFF), read(QUEUE), read(ELIGIBLE), read(SOURCE)
    require(bind(QUEUE) == handoff["queue"], "Frozen queue changed")
    expected = handoff["first_batch"]
    selected = queue["records"][:8]
    require([r["review_id"] for r in selected] == expected["review_ids"] == [f"Q{i:02}" for i in range(1,9)], "Wrong first batch")
    require([r["parent_id"] for r in selected] == expected["parent_ids"], "Parent identities changed")
    old = {r["id"]: r for r in sources["records"]}
    src = {r["stem"]: r for r in sources["sources"]}
    canonical = {r["id"]: r for r in eligible["records"]}
    mapped = {r["source_stem"]: r for r in read(MAPPING)["sources"]}
    tar = helper(TAR, "stage5f_tar_plan")
    rows, provenance = [], []
    for q in selected:
        ident, stem = q["parent_id"], q["source_stem"]
        r, s, c, m = old[ident], src[stem], canonical[ident], mapped[stem]
        require(r["source_text"] == q["parent_source_text"] and c["text"] == q["parent_text"], "Parent text differs")
        require(s["split"] == "train" and ident in s["record_ids"], "Source is not historical train candidate")
        require(s["members"]["video"]["sha256"] == q["recording_video_sha256"] and s["members"]["label"]["sha256"] == q["recording_label_sha256"], "Source member identity differs")
        require(q["actual_split"] is None and q["training_ready"] is False and q["parent_transcript_confirmed"] is False, "Approval/role changed")
        require(r["offset_seconds"] == m["offset_seconds"] and m["original_label_sha256"] == q["recording_label_sha256"], "Offset evidence differs")
        refined = read(Path(m["refined_result"]["path"]))
        require(sha(Path(m["refined_result"]["path"])) == m["refined_result"]["sha256"] and refined["status"] == "supported" and refined["summary"]["constant_mapping_supported"], "Global mapping unsupported")
        require(refined["summary"]["median_offset_seconds"] == r["offset_seconds"], "Global/local offset differs")
        off = sample(r["offset_seconds"])
        require(abs(off/48000-r["offset_seconds"]) < 1e-12, "Offset is not exact native sample")
        require(abs(r["start_time"]-r["original_start_time"]-r["offset_seconds"]) < 1e-10 and abs(r["end_time"]-r["original_end_time"]-r["offset_seconds"]) < 1e-10, "Offset formula invalid")
        for role, member in s["members"].items():
            require(Path(member["archive_path"]).drive.upper() == "D:", "Only D TARs allowed")
            tar._check_archive(tar._validate_member(stem,role,member))
        left, right = sample(r["original_start_time"]), sample(r["original_end_time"])
        require(0 < left < right, "Invalid native parent interval")
        rows.append({"review_id":q["review_id"], "parent_id":ident, "candidate_id":q["candidate_id"], "queue_record":q,
                     "source_record":r, "source":s, "global_mapping_evidence":m,
                     "parent_native_start_sample":left,"parent_native_end_sample":right,
                     "context_native_start_sample":left-16800,"context_native_end_sample":right+16800,
                     "offset_native_samples":off,"training_ready":False,"actual_split":None})
        provenance.append(bind(Path(m["refined_result"]["path"])))
    paths = [QUEUE,HANDOFF,ELIGIBLE,SOURCE,MAPPING,TAR,SCORE,OLD,CLOCK,FFMPEG,STAGE/"execution_policy.json",Path(__file__)]
    plan = {"schema":"offline-stage5f-media-source-plan-v1","client_date":"2026-10-18", "status":"source_plan_prepared_no_tar_payload_read",
            "bindings":[bind(p) for p in paths]+provenance,"records":rows,
            "allowed_review_ids":expected["review_ids"],"allowed_parent_ids":expected["parent_ids"],
            "recordings":8,"planned_members":24,"planned_payload_bytes":sum(m["size"] for r in rows for m in r["source"]["members"].values()),
            "maximum_temporary_member_bytes":max(sum(m["size"] for m in r["source"]["members"].values()) for r in rows),
            "minimum_free_reserve_bytes":RESERVE,
            "clock_policy":"native48k parent bounds half-even independently; parent and context +-16800samples resampled separately to16k; resampling lengths recorded and parent/context edge filter differences allowed; D grid display bins overlap context, original PTS nearest/ties-earlier, exact offset applied once; video mux audio begins at display_start-offset",
            "model_calls":0,"training_updates":0,"training_ready":False,"all_media_review_only":True,
            "full_archive_extraction":False,"audio_student_input":False,"approval_added":False}
    save(PLAN,plan)
    print(json.dumps({"source_plan":bind(PLAN),"payload_bytes":plan["planned_payload_bytes"],"maximum_temp_bytes":plan["maximum_temporary_member_bytes"]}),flush=True)


def event(value: dict[str, Any]) -> None:
    """Append actual byte accounting even if later validation fails."""
    with (BASE/"io_events.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


@contextmanager
def materialize(row: dict[str, Any], tar: Any, counts: dict[str,int]) -> Iterator[dict[str,Path]]:
    """Copy bounded approved members and always clean only generated temp files."""
    temp = (MEDIA/"_tmp").resolve()
    temp.mkdir(parents=True,exist_ok=True)
    require(temp.is_relative_to(MEDIA.resolve()) and not list(temp.iterdir()),"Unsafe/nonempty temp directory")
    source=row["source"]
    members={role:tar._validate_member(source["stem"],role,m) for role,m in source["members"].items()}
    require(set(members)=={"video","audio","label"},"Unexpected member roles")
    snaps={role:tar._check_archive(m) for role,m in members.items()}
    payload=sum(m["size"] for m in members.values())
    require(shutil.disk_usage(temp).free>=RESERVE+payload,"Insufficient temporary reserve")
    child=Path(tempfile.mkdtemp(prefix=source["stem"]+"-",dir=temp)).resolve()
    require(child.is_relative_to(temp) and child!=temp,"Unsafe temporary child")
    paths={}
    try:
        for role,m in members.items():
            target=child/m["output_name"]
            require(target.parent==child and not target.exists(),"Unsafe generated file")
            digest,n=hashlib.sha256(),0
            counts["member_attempts"]+=1
            event({"event":"member_start","parent_id":row["parent_id"],"role":role,"size":m["size"]})
            try:
                with m["archive_path"].open("rb") as src,target.open("xb") as dst:
                    tar._check_archive(m,snaps[role],src)
                    src.seek(m["offset"])
                    while n<m["size"]:
                        chunk=src.read(min(4*1024**2,m["size"]-n))
                        require(bool(chunk),"Truncated source member")
                        n+=len(chunk); counts["payload_bytes_read"]+=len(chunk)
                        dst.write(chunk); digest.update(chunk)
                    tar._check_archive(m,snaps[role],src)
                require(n==m["size"] and digest.hexdigest()==m["sha256"],"Member SHA/size differs")
                counts["members_sha_verified"]+=1
                paths[role]=target
            finally:
                counts["maximum_simultaneous_temp_bytes"]=max(counts["maximum_simultaneous_temp_bytes"],sum(p.stat().st_size for p in child.iterdir() if p.is_file()))
                event({"event":"member_finished","parent_id":row["parent_id"],"role":role,"bytes":n,"payload_total":counts["payload_bytes_read"]})
        yield paths
        for role,m in members.items():
            tar._check_archive(m,snaps[role])
    finally:
        require(child.is_relative_to(temp) and child!=temp,"Unsafe cleanup target")
        shutil.rmtree(child)
        event({"event":"temporary_cleanup","parent_id":row["parent_id"],"remaining_files":len(list(temp.iterdir()))})


def native_excerpt(source: Path, output: Path, left: int, right: int) -> dict[str,Any]:
    """Read exact native PCM samples, resample annotation audio to16kHz."""
    require(type(left) is int and type(right) is int and 0<=left<right and not output.exists(),"Invalid native slice")
    with wave.open(str(source),"rb") as stream:
        require(stream.getframerate()==48000 and stream.getsampwidth()==2 and stream.getcomptype()=="NONE","Unsupported native WAV")
        require(right<=stream.getnframes(),"Native PCM window out of range")
        channels=stream.getnchannels(); stream.setpos(left); pcm=stream.readframes(right-left)
    require(len(pcm)==(right-left)*channels*2,"Incomplete native PCM")
    native=io.BytesIO()
    with wave.open(native,"wb") as stream:
        stream.setframerate(48000); stream.setnchannels(channels); stream.setsampwidth(2); stream.writeframes(pcm)
    native_path=output.with_name(output.stem+"_native48k.wav")
    require(not native_path.exists(),"Native crop already exists")
    with native_path.open("xb") as stream:
        stream.write(native.getvalue())
    command=[str(FFMPEG),"-nostdin","-v","error","-f","wav","-i","pipe:0","-ac","1","-ar","16000","-c:a","pcm_s16le","-n",str(output)]
    process=subprocess.run(command,input=native.getvalue(),capture_output=True,timeout=120)
    require(process.returncode==0,"PCM conversion failed: "+process.stderr.decode(errors="replace"))
    with wave.open(str(output),"rb") as stream:
        require((stream.getframerate(),stream.getnchannels(),stream.getsampwidth())==(16000,1,2),"Wrong review PCM")
        n=stream.getnframes()
    require(abs(n-round(Fraction(right-left,3)))<=1,"Unexpected resampling length")
    return {**bind(output),"sample_rate":16000,"channels":1,"sample_width_bytes":2,"samples":n,"duration_seconds":n/16000,
            "source_sample_rate":48000,"source_first_sample":left,"source_end_sample_exclusive":right,"source_samples_read":right-left,
            "native_pcm_sha256":hashlib.sha256(pcm).hexdigest(),"native_crop":bind(native_path),"native_channels":channels,"source_sample_origin_verified":True,
            "source_start_seconds":left/48000,"source_end_seconds":right/48000,"clock":"original_label_wav","review_only":True,"command":command}


def slice_context(spec: dict[str,Any], output: Path, first: int, stop: int) -> dict[str,Any]:
    """Save a parent as exact integer PCM slice of the displayed review context."""
    require(sha(Path(spec["path"]))==spec["sha256"] and 0<=first<stop<=spec["samples"] and not output.exists(),"Invalid context slice")
    with wave.open(spec["path"],"rb") as stream:
        stream.setpos(first); pcm=stream.readframes(stop-first)
    require(len(pcm)==(stop-first)*2,"Incomplete context slice")
    with wave.open(str(output),"wb") as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000);stream.writeframes(pcm)
    return {**bind(output),"sample_rate":16000,"channels":1,"sample_width_bytes":2,"samples":stop-first,"duration_seconds":(stop-first)/16000,
            "context_start_sample_16k":first,"context_end_sample_16k":stop,"context_binding":{k:spec[k] for k in ("path","sha256","bytes")},
            "source_sample_origin_verified":True,"source_sample_rate":48000,"source_first_sample":spec["source_first_sample"]+3*first,
            "source_end_sample_exclusive":spec["source_first_sample"]+3*stop,"review_only":True,"clock":"context_16k_exact_slice"}


def videos(paths: dict[str,Path], label: dict[str,Any], row: dict[str,Any], folder: Path, old: Any, clock: Any) -> dict[str,Any]:
    """Decode actual bracketing PTS and save D-grid review-only full/lip previews."""
    timeline=clock.read_clock(paths["video"])
    off=Fraction(row["offset_native_samples"],48000)
    left=Fraction(row["context_native_start_sample"],48000)+off
    right=Fraction(row["context_native_end_sample"],48000)+off
    grid=clock.grid_window(left,right,timeline["origin"],timeline["coverage_end"])
    chosen=[clock.nearest(timeline["pts"],t) for t in grid["times"]]
    required=sorted({index for _,brackets in chosen for index in brackets})
    capture=cv2.VideoCapture(str(paths["video"]))
    require(capture.isOpened(),"Cannot open source MP4")
    try:
        fps=capture.get(cv2.CAP_PROP_FPS); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        require(count==timeline["sample_count"] and abs(fps-row["source"]["video_fps"])<1e-10,"Source clock count/FPS differs")
        boxes=np.asarray(label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"],dtype=np.float64)
        require(boxes.shape==(count,4) and np.isfinite(boxes).all(),"Invalid box sequence")
        boxes=boxes[:,[1,0,3,2]]
        decoded={};capture.set(cv2.CAP_PROP_POS_FRAMES,required[0]); current=required[0]
        for index in required:
            while current<=index:
                success,frame=capture.read();require(success,"Source ended early");current+=1
            actual=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            require(capture.get(cv2.CAP_PROP_POS_FRAMES)-1==index and abs(actual-float(timeline["pts"][index]))<=1e-6,"Decoded PTS/ordinal mismatch")
            decoded[index]=(frame.copy(),actual)
        rois,full,trace=[],[],[]
        for k,(target,(index,brackets)) in enumerate(zip(grid["times"],chosen,strict=True)):
            frame,actual=decoded[index]
            box=boxes[index].tolist();x1,y1,x2,y2=old.crop_rectangle(box,width,height)
            roi=cv2.resize(frame[y1:y2,x1:x2],(96,96),interpolation=cv2.INTER_LINEAR)
            rois.append(roi);full.append(cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA))
            trace.append({"display_grid_index":grid["first"]+k,"target_video_time":clock.rational(target),"native_frame_index":index,
                          "actual_pts_seconds":actual,"exact_pts":clock.rational(timeline["pts"][index]),"bracket_indices":list(brackets),
                          "bracket_decoded_pts":[decoded[i][1] for i in brackets],"annotation_index":index,"annotation_box_xyxy":box,"crop_rectangle_xyxy":[x1,y1,x2,y2],
                          "source_bgr_sha256":hashlib.sha256(frame.tobytes()).hexdigest(),"roi_bgr_sha256":hashlib.sha256(roi.tobytes()).hexdigest()})
        audio_left=(grid["display_start"]-off)*48000;audio_right=(grid["display_end"]-off)*48000
        require(audio_left.denominator==audio_right.denominator==1,"Unsupported nonnative display audio clock")
        display_audio=native_excerpt(paths["audio"],folder/"display_audio.wav",int(audio_left),int(audio_right))
        require(display_audio["samples"]==len(rois)*640,"Display audio/video lengths differ")
        lips=old.encode_preview(rois,Path(display_audio["path"]),folder/"context_lips.mp4")
        whole=old.encode_preview(full,Path(display_audio["path"]),folder/"context_video.mp4")
        contact=np.concatenate([cv2.resize(rois[i],(192,192)) for i in (0,len(rois)//2,len(rois)-1)],axis=1)
        require(cv2.imwrite(str(folder/"roi_contact.png"),contact),"Cannot save contact sheet")
        return {"sampling":"D_recording_anchored_25fps_display_overlap_v1","recording_origin":clock.rational(timeline["origin"]),"native_period":clock.rational(timeline["period"]),
                "native_fps":fps,"native_frame_count":count,"source_size":[width,height],"source_clock_cfr_verified":True,
                "semantic_context_video_start":clock.rational(left),"semantic_context_video_end":clock.rational(right),
                "display_video_start":clock.rational(grid["display_start"]),"display_video_end":clock.rational(grid["display_end"]),
                "display_before_context":clock.rational(grid["before_semantic"]),"display_after_context":clock.rational(grid["after_semantic"]),
                "offset_native_samples_applied_once":row["offset_native_samples"],"roi_coordinate_order":"yxyx -> xyxy","roi_margin":.2,"roi_size":[96,96],"no_padding":True,
                "trace":trace,"display_audio":display_audio,"roi_preview":lips,"full_preview":whole,"roi_contact":bind(folder/"roi_contact.png"),
                "review_only":True,"exact_articulation_boundary_approved":False}
    finally:
        capture.release()


def one(row: dict[str,Any], tar: Any, old: Any, scorer: Any, clock: Any, counts: dict[str,int]) -> dict[str,Any]:
    """Prepare one full parent without inventing or approving phrase boundaries."""
    folder=MEDIA/row["parent_id"]
    require(not folder.exists(),"Refuse existing parent output")
    folder.mkdir(parents=True)
    started=time.perf_counter()
    with materialize(row,tar,counts) as paths:
        raw=read(paths["label"]);require(isinstance(raw,list) and len(raw)==1,"Unexpected label schema")
        label=raw[0];r=row["source_record"];q=row["queue_record"]
        sentence=next(s for s in label["Sentence_info"] if int(s["ID"])==int(row["parent_id"].rsplit("_",1)[1]))
        require(sentence["sentence_text"]==r["source_text"] and abs(sentence["start_time"]-r["original_start_time"])<1e-9 and abs(sentence["end_time"]-r["original_end_time"])<1e-9,"Native parent label differs")
        require(Path(label["Video_info"]["video_Name"]).stem==r["source_stem"] and label["speaker_info"]["speaker_ID"]==r["speaker_id"],"Native label identity differs")
        mapping=old.mapping_check(paths,r,folder,scorer)
        context=native_excerpt(paths["audio"],folder/"context.wav",row["context_native_start_sample"],row["context_native_end_sample"])
        parent=native_excerpt(paths["audio"],folder/"sentence.wav",row["parent_native_start_sample"],row["parent_native_end_sample"])
        first=5600;stop=first+parent["samples"]
        left,right=row["context_native_start_sample"]/48000,row["context_native_end_sample"]/48000
        neighbors=[{"sentence_id":s["ID"],"source_text":s["sentence_text"],"start_time":s["start_time"],"end_time":s["end_time"],"intersection_seconds":min(right,s["end_time"])-max(left,s["start_time"])}
                   for s in label["Sentence_info"] if int(s["ID"])!=int(sentence["ID"]) and min(right,s["end_time"])>max(left,s["start_time"])]
        video=videos(paths,label,row,folder,old,clock)
        result={"schema":"offline-stage5f-parent-review-media-v1","review_id":row["review_id"],"id":row["parent_id"],"parent_id":row["parent_id"],"candidate_id":row["candidate_id"],
                "speaker_id":r["speaker_id"],"source_stem":r["source_stem"],"parent_text":q["parent_text"],"source_text":r["source_text"],"proposed_phrase":q["text"],"word_start":q["word_start"],"word_end":q["word_end"],
                "source_members":row["source"]["members"],"global_mapping_evidence":row["global_mapping_evidence"],"mapping_check":mapping,
                "original_start_time":r["original_start_time"],"original_end_time":r["original_end_time"],"offset_seconds":r["offset_seconds"],"offset_native_samples":row["offset_native_samples"],
                "parent_native_start_sample":row["parent_native_start_sample"],"parent_native_end_sample":row["parent_native_end_sample"],
                "parent_native_start_rounding_seconds":row["parent_native_start_sample"]/48000-r["original_start_time"],"parent_native_end_rounding_seconds":row["parent_native_end_sample"]/48000-r["original_end_time"],
                "parent_end_16k_duration_rounding_native_samples":3*parent["samples"]-(row["parent_native_end_sample"]-row["parent_native_start_sample"]),
                "parent_context_resampled_independently":True,"parent_is_not_pcm_slice_of_context":True,
                "context_audio":context,"sentence_audio":parent,"sentence_within_context_samples_16k":[first,stop],"context_radius_native_samples":16800,
                "context_neighbor_intersections":neighbors,"context_contains_neighbor_label_interval":bool(neighbors),"video":video,
                "all_member_sha256_verified":True,"archive_stats_checked_before_and_after":True,"parent_transcript_confirmed":False,"human_boundary_approved":False,"actual_split":None,
                "training_ready":False,"audio_used_as_student_input":False,"phrase_boundaries":None,"status":"parent_media_ready_human_transcript_and_boundary_pending"}
    result["temporary_originals_removed"]=True
    result["elapsed_seconds"]=time.perf_counter()-started
    save(folder/"record.json",result)
    return result


def execute(policy_path: Path, policy_sha: str, limit: int) -> None:
    """Run only under the root's bound policy and record every materialized byte."""
    require(sha(policy_path)==policy_sha,"Execution policy changed")
    policy=read(policy_path)
    require(policy.get("stage")=="5f", "Wrong execution stage")
    plan=read(PLAN)
    require(policy["record_ids"]==plan["allowed_parent_ids"] and policy["review_ids"]==plan["allowed_review_ids"],"Policy scope differs")
    require(policy["candidate_ids"]==[r["candidate_id"] for r in plan["records"]],"Policy candidate scope differs")
    require(policy["training_ready"] is False and policy["acoustic_forward_in_this_initial_policy"] is False,"Unexpected model/approval policy")
    for b in policy["bindings"]:
        require(bind(Path(b["path"]))==b,"Execution policy source changed")
    require(1<=limit<=8,"Only fixed first eight parents allowed")
    for b in plan["bindings"]:
        require(bind(Path(b["path"]))==b,"Media producer/source evidence changed")
    counts={"member_attempts":0,"members_sha_verified":0,"payload_bytes_read":0,"maximum_simultaneous_temp_bytes":0}
    tar,old,scorer,clock=helper(TAR,"stage5f_tar"),helper(OLD,"stage5f_oldmedia"),helper(SCORE,"stage5f_score"),helper(CLOCK,"stage5f_clock")
    records=[]
    try:
        for row in plan["records"][:limit]:
            record_path=MEDIA/row["parent_id"]/"record.json"
            if record_path.exists():
                records.append(read(record_path))
                continue
            print(json.dumps({"event":"source_start","review_id":row["review_id"],"parent_id":row["parent_id"]}),flush=True)
            result=one(row,tar,old,scorer,clock,counts);records.append(result)
            print(json.dumps({"event":"source_ready","review_id":row["review_id"],"record":str(record_path),"correlation":result["mapping_check"]["score"]["absolute_correlation"],"elapsed_seconds":result["elapsed_seconds"]}),flush=True)
        for b in plan["bindings"]:
            require(bind(Path(b["path"]))==b,"Evidence changed during execution")
        require(not list((MEDIA/"_tmp").iterdir()),"Temporary originals remain")
        report={"schema":"offline-stage5f-media-manifest-v1","status":"parent_media_ready_human_review_pending" if len(records)==8 else "partial_parent_media_ready",
                "source_plan":bind(PLAN),"execution_policy":bind(policy_path),"records":records,"completed":len(records),"io_this_invocation":counts,
                "recordings":len(records),"new_human_approvals":0,"word_alignment_calls":0,"neural_calls":0,"training_updates":0,"training_ready":False,"originals_remaining":0,
                "environment":{"executable":sys.executable,"python":sys.version,"opencv":cv2.__version__,"numpy":np.__version__,"scipy":scipy.__version__}}
        save(BASE/f"media_manifest_{len(records):02d}.json",report)
    except Exception as exc:
        save(BASE/f"failure_{time.time_ns()}.json",{"type":type(exc).__name__,"message":str(exc),"io_this_invocation":counts,"completed_parent_ids":[r["id"] for r in records]})
        raise


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--policy",type=Path)
    parser.add_argument("--policy-sha")
    parser.add_argument("--limit",type=int,default=8)
    args=parser.parse_args()
    require(args.prepare != args.execute,"Choose preparation or execution")
    if args.prepare: prepare()
    else:
        require(args.policy is not None and args.policy_sha is not None,"Execution policy required")
        execute(args.policy,args.policy_sha,args.limit)
