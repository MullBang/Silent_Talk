"""Local independent-array and PCM checks, without new original reads or models."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,wave
import numpy as np
from scipy.signal import resample_poly

BASE=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x): return Fraction(x["numerator"],x["denominator"])
def require(x,msg):
    if not x: raise ValueError(msg)

handoff=read(BASE/"cache_handoff.json");audio=read(BASE/"audio_manifest.json")
out=[]
for row in handoff["records"]:
    require(sha(row["npz_path"])==row["npz_sha256"],"NPZ SHA")
    meta=read(row["metadata_path"])
    require(sha(row["metadata_path"])==row["metadata_sha256"],"Metadata SHA")
    with np.load(row["npz_path"],allow_pickle=False) as z:
        rgb=z["frames_rgb"];labels=z["label"].tolist()
        require(rgb.shape==(row["T"],96,96,3) and rgb.dtype==np.uint8,"Pixel shape")
        require(json.loads(str(z["metadata_json"].item()))==meta,"Embedded metadata")
        require(not z["training_ready"].item() and z["provisional"].item(),"Approval bits")
    require(labels==meta["label"] and len(rgb)>=len(labels)+sum(a==b for a,b in zip(labels,labels[1:])),"CTC")
    require(hashlib.sha256(rgb.tobytes()).hexdigest()==meta["rgb_uint8_sha256"],"Array hash")
    traces=meta["trace"];require(len(traces)==len(rgb),"Trace count")
    indices=[];times=[]
    for image,t in zip(rgb,traces):
        require(hashlib.sha256(image.tobytes()).hexdigest()==t["roi_rgb_sha256"],"Per-frame pixel hash")
        require(hashlib.sha256(image[:,:,::-1].copy().tobytes()).hexdigest()==t["roi_bgr_sha256"],"RGB/BGR")
        target=f(t["grid_time"]);bracket=t["selection_bracket"]
        chosen=min(bracket,key=lambda b:(abs(f(b["actual_pts"])-target),f(b["actual_pts"])))
        require(chosen["native_frame_index"]==t["native_frame_index"],"Nearest actual PTS/tie")
        require(abs(float(f(t["actual_pts"]))-t["decoded_pts_seconds"])<1e-6,"Decoded PTS")
        require(f(t["semantic_overlap"])>0,"No semantic overlap")
        indices.append(t["native_frame_index"]);times.append(target)
    require(len(set(indices))==len(indices) and indices==sorted(indices),"Native order")
    require(all(b-a==Fraction(1,25) for a,b in zip(times,times[1:])),"25fps grid")
    require(not meta["neighbor_label_intersections"],"Neighbor labels")
    require(not meta["human_transcript_approved"] and not meta["human_visual_boundary_approved"],"No approval transfer")
    require(not meta["audio_included"] and meta["raw_padding_frames"]==0,"No audio/padding")
    out.append({"candidate_id":row["candidate_id"],"T":len(rgb),"all_pixel_traces_verified":len(rgb),"CTC_feasible":True,"original_order_unique":True,"display_neighbor_overlap":False})
audio_checks=[]
for row in audio["records"]:
    for kind in ("whole","context"):
        a=row[kind];require(sha(a["path"])==a["sha256"] and sha(a["native_crop"]["path"])==a["native_crop"]["sha256"],"Audio SHA")
        with wave.open(a["native_crop"]["path"],"rb") as w:
            require(w.getframerate()==48000 and w.getsampwidth()==2 and w.getnframes()==a["source_end_sample_exclusive"]-a["source_first_sample"],"Native WAV metadata")
            channels=w.getnchannels();raw=w.readframes(w.getnframes())
        require(hashlib.sha256(raw).hexdigest()==a["native_pcm_sha256"],"Native PCM hash")
        x=np.frombuffer(raw,dtype="<i2").reshape(-1,channels).astype(np.float64).mean(axis=1)
        expect=np.clip(np.rint(resample_poly(x,1,3,window=("kaiser",5.0),padtype="constant")),-32768,32767).astype("<i2")
        with wave.open(a["path"],"rb") as w:
            require((w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,len(expect)),"16k WAV format")
            actual=w.readframes(w.getnframes())
        require(actual==expect.tobytes(),"PCM resample roundtrip")
        audio_checks.append({"candidate_id":row["candidate_id"],"kind":kind,"source_samples":len(x),"samples16k":len(expect),"sample_origin_metadata_and_pcm_roundtrip":True})
result={"schema":"offline-stage5j-original-cache-validation-v1","status":"passed","cache_records":out,"annotation_audio_checks":audio_checks,"pixel_frames":sum(r["T"] for r in out),"audio_files":len(audio_checks),"spatial_contact_review":{"path":str(BASE/"spatial_contact.png"),"sha256":sha(BASE/"spatial_contact.png"),"inspected_samples":18,"observer":"assistant_visual_inspection_of_rendered_contact","result":"All6 first/middle/last lip crops show mouth region without obvious spatial clipping, occlusion, or severe blur. Exact articulatory start/end is not certified.","human_approval_added":False},"technical_QC_passed_candidate_ids":[r["candidate_id"] for r in out],"new_original_reads":0,"NN_forward":0,"training":0,"approval_added":0,"training_ready":False,"provisional":True}
p=BASE/"cache_validation.json";require(not p.exists(),"Output exists")
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"status":result["status"],"frames":result["pixel_frames"],"audio_files":len(audio_checks)}))
