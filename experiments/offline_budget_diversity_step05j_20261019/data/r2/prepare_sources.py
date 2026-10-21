"""Six fixed other-recording whole sentences: original D-grid and annotation PCM.

One audited original member exists on disk at a time. New examples remain
provisional and are not human-approved; the separate ASR screen decides which
technically valid examples may enter the explicitly bounded experiment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import wave
import scipy
from scipy.signal import resample_poly

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
STAGE = BASE.parents[1]
ROOT = STAGE.parents[1]
OLD = ROOT / "experiments/offline_dataset_step03_20261013/extraction"
F = ROOT / "experiments/offline_phrase_review_step05f_20261018/media_prepare/prepare_media.py"
INVENTORY = STAGE / "inventory/new_recording_candidates.json"
PLAN = BASE / "extraction_plan.json"
sys.path.insert(0, str(OLD))
import prepare_and_extract as extract
from exact_clock import require, rational, fraction, read_clock

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def bind(path):
    p=Path(path).resolve()
    return {"path":str(p), "sha256":sha(p), "bytes":p.stat().st_size}

def save(path, value):
    p=Path(path).resolve()
    require(p.is_relative_to(BASE.resolve()) and not p.exists(), "Unsafe or existing output")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")

def prepare():
    inv=read(INVENTORY)
    rows=inv["records"]
    require(len(rows)==6 and [r["candidate_id"] for r in rows]==[f"N{i:02}" for i in range(1,7)], "Unexpected source scope")
    require(len({r["source_stem"] for r in rows})==6, "Recordings not distinct")
    tar=extract.load_module(extract.TAR_HELPER,"stage5j_tar_plan")
    codec=extract.load_module(extract.TOKENIZER,"stage5j_codec_plan")
    sources=[Path(__file__),BASE.parent/"prepare_sources.py",BASE.parent/"extraction_plan.json",INVENTORY,F,OLD/"prepare_and_extract.py",OLD/"exact_clock.py",extract.TAR_HELPER,extract.ROI_HELPER,extract.TOKENIZER]
    for b in inv["source_bindings"]:
        require(bind(b["path"])==b,"Inventory source changed")
        sources.append(Path(b["path"]))
    entries=[]
    for r in rows:
        require(not r["human_transcript_approved"] and not r["human_boundary_approved"] and not r["training_ready"],"Approval scope differs")
        require(r["source_stem"] not in inv["held_recordings_also_excluded_here"],"Held recording selected")
        require(bind(r["source_alignment"]["path"])==r["source_alignment"],"Alignment source changed")
        sources.append(Path(r["source_alignment"]["path"]))
        require(r["source_alignment_summary"]["constant_mapping_supported"],"Unsupported offset")
        labels=codec.encode_text(r["text"])
        require(labels==r["token_ids"] and codec.decode_labels(labels)==r["text"],"Target mismatch")
        offset=round(Fraction(str(r["offset_seconds"]))*48000)
        require(abs(offset/48000-r["offset_seconds"])<1e-12,"Non-native-sample offset")
        left,right=[round(Fraction(str(r[k]))*48000) for k in ("original_start_time","original_end_time")]
        require(left>9600 and right>left and 1<=float(Fraction(right-left,48000))<=3,"Unsupported source duration")
        for role,m in r["source_members"].items():
            require(Path(m["archive_path"]).drive.upper()=="D:","Only existing local D archives")
            tar._check_archive(tar._validate_member(r["source_stem"],role,m))
        entries.append({**r,"native_start_sample":left,"native_end_sample":right,"context_start_sample":left-9600,"context_end_sample":right+9600,"offset_native_samples":offset})
    payload=sum(m["size"] for r in entries for m in r["source_members"].values())
    peak=max(m["size"] for r in entries for m in r["source_members"].values())
    require(payload<=3*1024**3 and shutil.disk_usage(BASE).free>=peak+extract.RESERVE,"Source budget/reserve exceeded")
    prior=read(OLD/"extraction_plan.json")
    sources.append(OLD/"extraction_plan.json")
    policy={"schema":"offline-stage5j-new-source-technical-screen-policy-v1","actual_user_instruction":"다음 단계 진행해줘. 애매한 자료는 계속 보류하고, 다른 녹화의 짧은 표현을 추가하면서 확대군의 학습량이 부족한지도 확인해보자.","scope":"Only fixed six official whole short sentences, no internal cuts. Technical extraction and annotation-only audio screening precede bounded experimental inclusion. No inherited human approval.","held_review_ids":[f"Q{i:02}" for i in range(3,9)],"held_parent_ids":inv["held_parent_ids_unchanged"],"global_training_ready":False,"provisional":True,"human_approvals_added":0,"student_audio_inputs":0,"neural_calls_in_this_script":0,"new_downloads":0,"new_human_questions":0,"maximum_recordings":6,"maximum_original_payload_bytes":3*1024**3,"one_original_temporary_file_at_a_time":True,"screening_status_before_asr":"technical_only_pending_automatic_text_screen","automatic_acceptance_is_not_human_review":True}
    save(BASE/"source_policy.json",policy)
    sources.append(BASE/"source_policy.json")
    plan={"schema":"offline-stage5j-whole-sentence-source-plan-v1","records":entries,"source_hashes":{str(p.resolve()):sha(p) for p in sources},"normalization":prior["normalization"],"tokenizer":prior["tokenizer"],"planned_members":18,"planned_payload_bytes":payload,"maximum_original_temp_bytes":peak,"audio_context_native_samples_each_side":9600,"audio_is_annotation_only":True,"global_training_ready":False,"provisional":True,"new_human_approvals":0,"revision":"r2 CPU scipy PCM conversion after ffmpeg process permission failure; failed r1 original reads preserved separately","resampler":{"implementation":"scipy.signal.resample_poly","version":scipy.__version__,"up":1,"down":3,"window":["kaiser",5.0],"padtype":"constant","float":"float64","quantization":"numpy.rint half-even followed by int16 clipping"}}
    save(PLAN,plan)
    return {"plan":bind(PLAN),"payload_bytes":payload,"maximum_original_temp_bytes":peak}

def event(value):
    with (BASE/"io_events.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")

@contextmanager
def member_file(row,role,tar,counts):
    temp=(BASE/"_tmp").resolve()
    temp.mkdir(parents=True,exist_ok=True)
    require(temp.is_relative_to(BASE.resolve()) and not list(temp.iterdir()),"Unsafe/nonempty original temp")
    m=tar._validate_member(row["source_stem"],role,row["source_members"][role])
    target=(temp/m["output_name"]).resolve()
    require(target.parent==temp and not target.exists() and not target.is_symlink(),"Unsafe temporary original")
    snap=tar._check_archive(m)
    counts["member_attempts"]+=1
    n=0;digest=hashlib.sha256()
    try:
        with m["archive_path"].open("rb") as src,target.open("xb") as dst:
            tar._check_archive(m,snap,src);src.seek(m["offset"])
            while n<m["size"]:
                block=src.read(min(4*1024**2,m["size"]-n))
                require(bool(block),"Truncated TAR payload")
                dst.write(block);digest.update(block);n+=len(block);counts["payload_bytes_read"]+=len(block)
            tar._check_archive(m,snap,src)
        require(n==m["size"] and digest.hexdigest()==m["sha256"],"Original SHA/size mismatch")
        counts["members_sha_verified"]+=1
        counts["maximum_original_temp_bytes"]=max(counts["maximum_original_temp_bytes"],n)
        yield target
        tar._check_archive(m,snap)
    finally:
        event({"candidate_id":row["candidate_id"],"role":role,"actual_bytes":n,"payload_total":counts["payload_bytes_read"]})
        require(target.resolve()==target and target.parent==temp and not target.is_symlink(),"Unsafe cleanup")
        if target.exists(): target.unlink()
        require(not list(temp.iterdir()),"Temporary original remains")

class MemoryLabel:
    def __init__(self,txt): self.txt=txt
    def read_text(self,encoding=None): return self.txt

def native_excerpt(source, output, left, right):
    require(0<=left<right and not output.exists(), "Invalid PCM slice")
    with wave.open(str(source), "rb") as stream:
        require(stream.getframerate()==48000 and stream.getsampwidth()==2 and stream.getcomptype()=="NONE", "Unsupported native PCM")
        require(right<=stream.getnframes(), "PCM window out of range")
        channels=stream.getnchannels(); stream.setpos(left); raw=stream.readframes(right-left)
    require(len(raw)==(right-left)*channels*2, "Incomplete PCM")
    native=output.with_name(output.stem+"_native48k.wav")
    require(not native.exists(), "Native crop exists")
    with wave.open(str(native),"wb") as stream:
        stream.setframerate(48000); stream.setnchannels(channels); stream.setsampwidth(2); stream.writeframes(raw)
    x=np.frombuffer(raw,dtype="<i2").reshape(-1,channels).astype(np.float64).mean(axis=1)
    y=resample_poly(x,1,3,window=("kaiser",5.0),padtype="constant")
    pcm=np.clip(np.rint(y),-32768,32767).astype("<i2")
    require(len(pcm)==(right-left+2)//3, "Unexpected resample size")
    with wave.open(str(output),"wb") as stream:
        stream.setframerate(16000);stream.setnchannels(1);stream.setsampwidth(2);stream.writeframes(pcm.tobytes())
    return {**bind(output),"sample_rate":16000,"channels":1,"sample_width_bytes":2,"samples":len(pcm),"duration_seconds":len(pcm)/16000,"source_sample_rate":48000,"source_first_sample":left,"source_end_sample_exclusive":right,"source_samples_read":right-left,"native_pcm_sha256":hashlib.sha256(raw).hexdigest(),"native_crop":bind(native),"native_channels":channels,"source_sample_origin_verified":True,"clock":"original_label_wav","review_only":True,"resampler":"scipy.resample_poly float64 1:3 kaiser5 constant; rint clip int16","source_start_seconds":left/48000,"source_end_seconds":right/48000}


def execute():
    plan=read(PLAN)
    for p,h in plan["source_hashes"].items(): require(sha(p)==h,"Source changed: "+p)
    require(not (BASE/"cache").exists() and not (BASE/"audio").exists(),"Outputs exist")
    extract.BASE,extract.PLAN=BASE,PLAN
    tar=extract.load_module(extract.TAR_HELPER,"stage5j_tar_exec")
    cropper=extract.load_module(extract.ROI_HELPER,"stage5j_roi_exec")
    media=extract.load_module(F,"stage5j_pcm_exec")
    counts={"member_attempts":0,"payload_bytes_read":0,"members_sha_verified":0,"maximum_original_temp_bytes":0}
    records=[];audios=[];sheets=[];started=time.perf_counter()
    try:
        for r in plan["records"]:
            print("START",r["candidate_id"],r["id"],flush=True)
            with member_file(r,"label",tar,counts) as p: label_text=p.read_text(encoding="utf-8-sig")
            parsed=json.loads(label_text)
            require(isinstance(parsed,list) and len(parsed)==1,"Unexpected label structure")
            label=parsed[0]
            selected=next(s for s in label["Sentence_info"] if int(s["ID"])==int(r["id"].rsplit("_",1)[1]))
            require(selected["sentence_text"]==r["source_text"] and abs(selected["start_time"]-r["original_start_time"])<1e-6 and abs(selected["end_time"]-r["original_end_time"])<1e-6,"Raw sentence differs")
            audio_dir=BASE/"audio"/r["candidate_id"];audio_dir.mkdir(parents=True)
            with member_file(r,"audio",tar,counts) as p:
                whole=native_excerpt(p,audio_dir/"whole.wav",r["native_start_sample"],r["native_end_sample"])
                context=native_excerpt(p,audio_dir/"context.wav",r["context_start_sample"],r["context_end_sample"])
            audios.append({"candidate_id":r["candidate_id"],"id":r["id"],"text":r["text"],"whole":whole,"context":context,"sample_rate":16000,"annotation_only":True,"human_approved":False})
            with member_file(r,"video",tar,counts) as p:
                clock=read_clock(p)
                cap=cv2.VideoCapture(str(p))
                try:
                    require(cap.isOpened(),"Cannot open video")
                    shape=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                    require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==clock["sample_count"],"Count differs")
                finally: cap.release()
                old={"source_text":r["source_text"],"original_start_time":r["original_start_time"],"original_end_time":r["original_end_time"],"video":{"native_frame_count":clock["sample_count"],"native_fps":float(1/clock["period"]),"source_size":shape}}
                clip={"clip_id":r["id"]+"__full_sentence","kind":"full_sentence","text":r["text"],"label":r["token_ids"],"label_length":len(r["token_ids"]),"ctc_min_frames":r["ctc_min_frames"],"semantic_video_start":rational(Fraction(r["native_start_sample"]+r["offset_native_samples"],48000)),"semantic_video_end":rational(Fraction(r["native_end_sample"]+r["offset_native_samples"],48000)),"split":"train","candidate_id":r["candidate_id"],"start_sample_16k":None,"end_sample_16k":None,"provisional":True,"training_ready":False,"new_original_pixels_human_approved":False,"human_transcript_approved":False,"human_visual_boundary_approved":False,"automatic_asr_screening":"pending","new_internal_cut":False}
                evidence={"status":"automatic_source_label_candidate_not_human_review","inventory":bind(INVENTORY),"candidate_id":r["candidate_id"],"human_approved":False}
                entry={"rank":r["candidate_rank"],"parent_id":r["id"],"speaker_id":r["speaker_id"],"recording_group":r["source_stem"],"source_members":r["source_members"],"context_origin":rational(Fraction(r["context_start_sample"],48000)),"frozen_offset":rational(Fraction(r["offset_native_samples"],48000)),"clips":[clip],"historical_record":old,"review_references":[],"review_provenance":evidence,"parent_audio_review_provenance":evidence}
                result=extract.decode_record({"video":p,"label":MemoryLabel(label_text)},entry,plan,cropper)
            row=result["clips"][0];meta=read(row["metadata_path"])
            flag=bool(meta["neighbor_label_intersections"])
            row.update({"candidate_id":r["candidate_id"],"parent_id":r["id"],"rank":r["candidate_rank"],"speaker_id":r["speaker_id"],"recording_group":r["source_stem"],"text":r["text"],"label_length":len(r["token_ids"]),"ctc_min_frames":r["ctc_min_frames"],"technical_QC_passed":not flag,"display_neighbor_intersections":meta["neighbor_label_intersections"],"boundary_followup_flag":flag,"automatic_asr_screening":"pending","bounded_experiment_eligible":False,"human_approvals_added":0,"precise_articulatory_boundary_approved":False,"source_status":"technical_candidate_pending_automatic_screen"})
            records.append(row)
            save(BASE/"per_record"/(r["candidate_id"]+".json"),{"record":row,"source_decode":result,"audio":audios[-1]})
            with np.load(row["npz_path"],allow_pickle=False) as z:
                images=z["frames_rgb"];chosen=[0,len(images)//2,len(images)-1]
                panel=np.full((148,3*192,3),255,np.uint8)
                for k,index in enumerate(chosen): panel[32:128,k*192:k*192+96]=cv2.cvtColor(images[index],cv2.COLOR_RGB2BGR)
                cv2.putText(panel,r["candidate_id"]+" "+r["speaker_id"]+" first / mid / last",(6,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
                sheets.append(panel)
            print("READY",r["candidate_id"],row["T"],"neighbor_flag",flag,flush=True)
        require(counts["payload_bytes_read"]==plan["planned_payload_bytes"] and counts["members_sha_verified"]==18,"I/O budget mismatch")
        for p,h in plan["source_hashes"].items(): require(sha(p)==h,"Source changed during extraction")
        require(cv2.imwrite(str(BASE/"spatial_contact.png"),np.concatenate(sheets,axis=0)),"Contact image save failed")
        save(BASE/"audio_manifest.json",{"schema":"offline-stage5j-whole-context-annotation-audio-v1","records":audios,"audio_only_for_annotation":True,"student_audio_inputs":0,"new_human_approvals":0})
        result={"schema":"offline-stage5j-new-original-cache-handoff-v1","status":"technical_extraction_complete_automatic_ASR_and_spatial_review_pending","plan":bind(PLAN),"records":records,"audio_manifest":bind(BASE/"audio_manifest.json"),"spatial_contact":bind(BASE/"spatial_contact.png"),"io":counts,"total_frames":sum(r["T"] for r in records),"cache_bytes":sum(r["npz_bytes"] for r in records),"temporary_originals_remaining":0,"elapsed_seconds":time.perf_counter()-started,"new_human_approvals":0,"global_training_ready":False,"student_audio_inputs":0,"neural_calls":0,"new_downloads":0}
        save(BASE/"cache_handoff.json",result)
        return {"status":result["status"],"io":counts,"total_frames":result["total_frames"],"audio_manifest":result["audio_manifest"]}
    except Exception:
        save(BASE/("failure_"+str(time.time_ns())+".json"),{"traceback":traceback.format_exc(),"io":counts,"completed_records":records,"audio_records":audios})
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true");args=p.parse_args()
    print(json.dumps(execute() if args.execute else prepare(),ensure_ascii=False,indent=2),flush=True)
