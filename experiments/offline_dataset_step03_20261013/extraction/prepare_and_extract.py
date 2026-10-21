"""Prepare or execute provisional visual-only caches for fifteen reviewed parents.

Default operation is metadata-only preparation. Original TAR payload I/O occurs
only with --execute after the saved plan and source-code hashes are checked.
No models, audio payloads, training or automatic QC approval are involved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator

import cv2
import numpy as np

from exact_clock import fraction, grid_window, nearest, rational, read_clock, require

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
EXPERIMENTS = ROOT / "experiments"
HANDOFF = EXPERIMENTS / "offline_phrase_review_step02g_20261013/results/verified_phrase_handoff.json"
DECISIONS = EXPERIMENTS / "offline_phrase_review_step02g_20261013/results/next_preparation_decisions_r2.json"
MEDIA = EXPERIMENTS / "offline_alignment_step02_20261012/results/media_manifest.json"
PARENT_OLD = EXPERIMENTS / "offline_boundary_finalize_step02d_20261013/review_manifest.json"
PARENT_NEW = EXPERIMENTS / "offline_boundary_remaining_step02f_20261013/remaining_parent_manifest.json"
PARENT_APPROVED = EXPERIMENTS / "offline_boundary_remaining_step02f_20261013/results/parent_audio_review_handoff.json"
SPLIT = BASE.parent / "split/split_plan.json"
TOKENIZER = ROOT / "backend/models/tokenizer.py"
TAR_HELPER = EXPERIMENTS / "local_source_step01_20261007/tar_source.py"
ROI_HELPER = EXPERIMENTS / "offline_boundary_remaining_step02f_20261013/native_check/check_original_roi_r4.py"
PLAN = BASE / "extraction_plan.json"
RESERVE = 10*1024**3
SCHEMA = "lipnet-recording-grid-D-rgb-provisional-v1"


def sha(path: Path) -> str:
    """Hash bounded-memory local evidence or a produced cache."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream,"sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> Any:
    """Read UTF-8 evidence without changing it."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError,ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def save(path: Path, value: Any) -> None:
    """Create an artifact only in the assigned extraction directory."""
    require(path.resolve().is_relative_to(BASE.resolve()) and not path.exists(), f"Unsafe/existing output {path}")
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {path}") from exc


def load_module(path: Path, name: str) -> Any:
    """Load explicitly hashed pure helper code, without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None, "Cannot load helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_time(record: dict[str, Any], sample: int) -> Fraction:
    """Map an exact context sample through native WAV time and offset once."""
    require(type(sample) is int and sample >= 0, "Invalid context sample")
    context = record["context_audio"]
    require(context["source_sample_origin_verified"] is True and context["source_sample_rate"] == 48000 and context["sample_rate"] == 16000, "Unsupported audio source clock")
    offset_samples = round(record["offset_seconds"]*48000)
    require(abs(offset_samples/48000-record["offset_seconds"]) < 1e-12, "Frozen offset is not an exact 48kHz sample count")
    return Fraction(context["source_first_sample"]+3*sample+offset_samples,48000)


def validate_clip_array(rgb: np.ndarray, labels: list[int]) -> int:
    """Reject invalid pixels and impossible CTC targets without adding frames."""
    require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[1:] == (96,96,3) and len(rgb)>0, "Invalid RGB cache shape/dtype")
    require(not np.any(np.all(rgb == 0,axis=(1,2,3))), "Zero/filler frame is forbidden")
    require(bool(labels) and all(type(x) is int and 0 < x < 69 for x in labels), "Invalid 69-class CTC target")
    minimum = len(labels)+sum(a == b for a,b in zip(labels,labels[1:]))
    require(len(rgb) >= minimum, "Actual extracted clip is CTC-infeasible")
    return minimum


def prepare(split_path: Path) -> dict[str, Any]:
    """Prepare thirty semantic windows and exact source provenance without TAR I/O."""
    handoff,media,decision,split = read(HANDOFF),read(MEDIA),read(DECISIONS),read(split_path)
    require(decision["sampling"]["selected_policy"] == "D_recording_anchored_25fps_display_overlap_v1", "Wrong sampling policy")
    require(len(handoff["records"]) == 15, "Expected fifteen reviewed phrases")
    require(split["schema"] == "lipnet-offline-dataset-stage3-split-plan-v1", "Unsupported split schema")
    split_rows = {r["parent_id"]:r for r in split["records"]}
    require(len(split_rows) == 15, "Split must bind exactly fifteen parents")
    codec,helper = load_module(TOKENIZER,"stage3_codec"),load_module(TAR_HELPER,"stage3_tar_contract")
    require(codec.NUM_CLASSES == 69, "Wrong token vocabulary")
    require(sha(TAR_HELPER) == "05f72840576046683d6b5a3c1121a94b7fe2099fd0d85596c30da511801effb2", "Frozen TAR helper changed")
    historical = {r["id"]:r for r in media["records"]}
    parents = {}
    for old in read(PARENT_OLD)["records"]:
        parents[old["id"]] = (old["sentence_start_sample_16k"],old["sentence_end_sample_16k"],old["text"])
    for old in read(PARENT_NEW)["records"]:
        parents[old["id"]] = (old["start_sample_16k"],old["end_sample_16k"],old["text"])
    approved_parents = {r["parent_id"]:r for r in read(PARENT_APPROVED)["records"]}
    require(len(approved_parents) == 16, "Expected sixteen parent audio reviews")
    source_paths = [HANDOFF,DECISIONS,MEDIA,PARENT_OLD,PARENT_NEW,PARENT_APPROVED,split_path,TOKENIZER,TAR_HELPER,ROI_HELPER,Path(__file__).resolve(),BASE/"exact_clock.py"]
    hashes = {str(p.resolve()):sha(p) for p in source_paths}
    entries = []
    for reviewed in handoff["records"]:
        ident = reviewed["parent_id"]
        old = historical[ident]
        require(reviewed["audio_boundary_confirmed"] is True and reviewed["visual_no_obvious_cut_reported"] is True, "Missing reviewed phrase evidence")
        require(reviewed["recording_group"] == old["source_stem"] and reviewed["speaker_id"] == old["speaker_id"], "Source identity differs")
        assignment = split_rows[ident]
        require(assignment["split"] in ("train","dev") and assignment["parent_text"] == reviewed["parent_text"], "Split identity/text differs")
        for key in ("segment_id","text","start_sample_16k","end_sample_16k"):
            require(assignment["phrase"][key] == reviewed[key], f"Split phrase differs: {key}")
        pleft,pright,ptext = parents[ident]
        parent_review = approved_parents[ident]
        require(parent_review["parent_transcript_confirmed"] is True and parent_review["parent_audio_boundary_confirmed"] is True, "Parent audio is not confirmed")
        require((pleft,pright,ptext) == (parent_review["audio_start_sample_16k"],parent_review["audio_end_sample_16k"],parent_review["canonical_parent_text"]), "Approved parent bounds/text differ")
        parent_evidence = Path(parent_review["evidence_path"])
        hashes[str(parent_evidence)] = sha(parent_evidence)
        require(ptext == reviewed["parent_text"] and pleft <= reviewed["start_sample_16k"] < reviewed["end_sample_16k"] <= pright, "Reviewed phrase/parent bounds differ")
        members = {role:old["source_members"][role] for role in ("video","label")}
        for role,member in members.items():
            require(Path(member["archive_path"]).drive.upper() == "D:", "Only audited D: TAR sources are allowed")
            helper._validate_member(old["source_stem"],role,member)
        for key in ("audio_scope_source","audio_evidence_path","visual_evidence_path","selection_source"):
            if key in reviewed:
                evidence = Path(reviewed[key])
                hashes[str(evidence)] = sha(evidence)
        clips = []
        for kind,left,right,text in (("phrase",reviewed["start_sample_16k"],reviewed["end_sample_16k"],reviewed["text"]),("full_sentence",pleft,pright,ptext)):
            labels = codec.encode_text(text)
            require(codec.decode_labels(labels) == text, "Canonical text would change during tokenization")
            s,e = original_time(old,left),original_time(old,right)
            require(s < e, "Empty source interval")
            clips.append({"clip_id":ident+"__"+kind,"kind":kind,"text":text,"label":labels,"label_length":len(labels),"ctc_min_frames":codec.ctc_min_input_len(labels),
                "start_sample_16k":left,"end_sample_16k":right,"semantic_video_start":rational(s),"semantic_video_end":rational(e),
                "split":assignment["split"],"training_ready":False,"provisional":True,"new_original_pixels_human_approved":False})
        refs = []
        for c in clips:
            for edge in ("start","end"):
                index = round(Fraction(c[edge+"_sample_16k"]*25,16000))
                require(0 <= index < len(old["video"]["trace"]), "Reference index outside preview")
                refs.append({"role":c["kind"]+"_"+edge,"preview_frame_index":index,"trace":old["video"]["trace"][index]})
        entries.append({"rank":reviewed["rank"],"parent_id":ident,"speaker_id":reviewed["speaker_id"],"recording_group":old["source_stem"],
            "source_members":members,"context_origin":rational(Fraction(old["context_audio"]["source_first_sample"],48000)),
            "frozen_offset":rational(Fraction(round(old["offset_seconds"]*48000),48000)),"clips":clips,"review_provenance":reviewed,"parent_audio_review_provenance":parent_review,
            "historical_record":old,"review_references":refs,"recording_origin_status":"measure_first_decoded_PTS_at_execution"})
    require(len({r["recording_group"] for r in entries}) == 15, "Expected one selected parent per recording")
    byte_count = sum(m["size"] for r in entries for m in r["source_members"].values())
    peak = max(sum(m["size"] for m in r["source_members"].values()) for r in entries)
    free = shutil.disk_usage(BASE).free
    require(free >= peak+RESERVE, "Insufficient reserve for one recording")
    plan = {"schema":"lipnet-stage3-original-extraction-plan-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"prepared_no_original_payload_read",
        "source_hashes":hashes,"records":entries,"planned_clips":30,"planned_recordings":15,"planned_payload_bytes":byte_count,
        "planned_member_count":30,"archive_paths":sorted({m["archive_path"] for r in entries for m in r["source_members"].values()}),
        "maximum_member_temp_bytes":peak,"minimum_free_reserve_bytes":RESERVE,"free_bytes_at_preflight":free,
        "cache_schema":SCHEMA,"normalization":{"stored":"uint8 RGB without normalization","future_input":"float32 RGB /255; channel (x-mean)/std; TCHW at loader","mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]},
        "tokenizer":{"path":str(TOKENIZER),"sha256":sha(TOKENIZER),"version":codec.TOKENIZER_VERSION,"normalization_version":codec.NORMALIZATION_VERSION,"vocab_sha256":codec.VOCAB_SHA256,"num_classes":69},
        "original_execution_authorized_by_this_preparation":False,"origin_assumed_zero":False,"audio_members":0,"neural_calls":0,"training_ready":False,
        "limits":"Exact CFR presentation table must agree with decoded bracketing PTS. Unsupported edits or VFR stop. New grid pixels remain provisional pending root QC."}
    save(PLAN,plan)
    return plan


def event(path: Path, value: dict[str, Any]) -> None:
    """Append bounded I/O progress for post-failure accounting."""
    require(path.resolve().is_relative_to(BASE.resolve()), "Unsafe log path")
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps({"utc":datetime.now(timezone.utc).isoformat(),**value},ensure_ascii=False,allow_nan=False)+"\n")


@contextmanager
def materialize(entry: dict[str, Any], helper: Any, io: dict[str, int], log: Path) -> Iterator[dict[str, Path]]:
    """Bound-copy only two audited members, log actual bytes, delete owned temp paths."""
    temp_root = (BASE/"_tmp").resolve()
    require(temp_root.is_relative_to(BASE.resolve()) and temp_root != BASE.resolve(), "Unsafe temporary root")
    temp_root.mkdir(exist_ok=True)
    require(not list(temp_root.iterdir()), "Temporary root has leftovers; inspect before proceeding")
    members = {role:helper._validate_member(entry["recording_group"],role,m) for role,m in entry["source_members"].items()}
    require(set(members) == {"video","label"}, "Audio or undeclared member requested")
    snapshots = {role:helper._check_archive(m) for role,m in members.items()}
    require(shutil.disk_usage(BASE).free >= RESERVE+sum(m["size"] for m in members.values()), "Insufficient temporary reserve")
    child_dir = Path(tempfile.mkdtemp(prefix=entry["recording_group"]+"-",dir=temp_root)).resolve()
    require(child_dir.is_relative_to(temp_root) and child_dir != temp_root, "Unsafe generated temporary child")
    paths = {}
    try:
        for role,m in members.items():
            target = (child_dir/m["output_name"]).resolve()
            require(target.parent == child_dir and not target.exists(), "Unsafe materialized file")
            digest,read_bytes = hashlib.sha256(),0
            io["member_read_attempts"] += 1
            event(log,{"kind":"member_start","role":role,"member":entry["source_members"][role],"payload_read_total":io["payload_bytes_read"]})
            try:
                with m["archive_path"].open("rb") as source,target.open("xb") as destination:
                    helper._check_archive(m,snapshots[role],source)
                    source.seek(m["offset"])
                    while read_bytes < m["size"]:
                        block = source.read(min(4*1024**2,m["size"]-read_bytes))
                        require(bool(block), "Truncated TAR payload")
                        read_bytes += len(block)
                        io["payload_bytes_read"] += len(block)
                        destination.write(block)
                        digest.update(block)
                    helper._check_archive(m,snapshots[role],source)
                require(digest.hexdigest() == m["sha256"] and target.stat().st_size == m["size"], "Original member SHA/size mismatch")
                io["members_sha_verified"] += 1
                paths[role] = target
            finally:
                event(log,{"kind":"member_read_finished","role":role,"parent_id":entry["parent_id"],"actual_payload_bytes":read_bytes,"payload_read_total":io["payload_bytes_read"]})
        yield paths
        for role,m in members.items():
            helper._check_archive(m,snapshots[role])
    finally:
        resolved = child_dir.resolve()
        require(resolved == child_dir and resolved.is_relative_to(temp_root) and resolved != temp_root and not child_dir.is_symlink(), "Unsafe cleanup path")
        # Only the verified freshly created child is removed; no computed parent deletion.
        shutil.rmtree(child_dir)
        event(log,{"kind":"temporary_cleanup","path":str(child_dir),"remaining":child_dir.exists()})


def decode_needed(path: Path, clock: dict[str, Any], needed: set[int]) -> tuple[dict[int, np.ndarray],dict[int,float],int]:
    """Decode short ranges containing actual neighbors; validate every decoded PTS/index."""
    cap = cv2.VideoCapture(str(path))
    try:
        require(cap.isOpened(), "Cannot open native video")
        require(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == clock["sample_count"], "Decoder count differs from presentation table")
        indices = sorted(needed)
        require(indices and indices[0] >= 0 and indices[-1] < clock["sample_count"], "Requested frame outside source")
        groups = []
        start = last = indices[0]
        for index in indices[1:]:
            if index-last > 4:
                groups.append((start,last))
                start = index
            last = index
        groups.append((start,last))
        frames,times,decoded = {},{},0
        for first,last in groups:
            require(cap.set(cv2.CAP_PROP_POS_FRAMES,first), "Decoder seek failed")
            for index in range(first,last+1):
                ok,frame = cap.read()
                require(ok and frame is not None, "Native decode ended early")
                actual_index = round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                actual_pts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                require(actual_index == index and np.isfinite(actual_pts), "Native decoded index/PTS invalid")
                require(abs(actual_pts-float(clock["pts"][index])) <= 1e-6, "Decoded PTS differs from exact MP4 presentation table")
                decoded += 1
                if index in needed:
                    frames[index],times[index] = frame,actual_pts
        require(set(frames) == needed, "Missing decoded target/bracket frames")
        return frames,times,decoded
    finally:
        cap.release()


def decode_record(paths: dict[str,Path], entry: dict[str,Any], plan: dict[str,Any], cropper: Any) -> dict[str,Any]:
    """Decode exact D targets and both neighbors, verify old raw pixels, save provisional NPZ."""
    old = entry["historical_record"]
    raw = read(paths["label"])
    require(isinstance(raw,list) and len(raw) == 1 and isinstance(raw[0],dict), "Expected single-record-list label")
    label = raw[0]
    require(Path(label["Video_info"]["video_Name"]).stem == entry["recording_group"] and label["speaker_info"]["speaker_ID"] == entry["speaker_id"], "Label identity mismatch")
    sentence = next(x for x in label["Sentence_info"] if int(x["ID"]) == int(entry["parent_id"].rsplit("_",1)[1]))
    require(sentence["sentence_text"] == old["source_text"] and abs(sentence["start_time"]-old["original_start_time"]) < 1e-6 and abs(sentence["end_time"]-old["original_end_time"]) < 1e-6, "Frozen original sentence changed")
    clock = read_clock(paths["video"])
    require(clock["sample_count"] == old["video"]["native_frame_count"], "Historical native frame count differs")
    require(abs(float(1/clock["period"])-old["video"]["native_fps"]) < 1e-8, "Historical FPS differs from exact presentation cadence")
    boxes_yxyx = np.asarray(label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"],dtype=np.float64)
    require(boxes_yxyx.shape == (clock["sample_count"],4) and np.isfinite(boxes_yxyx).all(), "Invalid per-frame yxyx bbox array")
    boxes_xyxy = boxes_yxyx[:,[1,0,3,2]]
    grids = []
    needed = {0,1,clock["sample_count"]-2,clock["sample_count"]-1}
    roi_needed = {r["trace"]["native_frame_index"] for r in entry["review_references"]}
    needed.update(roi_needed)
    for clip in entry["clips"]:
        grid = grid_window(fraction(clip["semantic_video_start"]),fraction(clip["semantic_video_end"]),clock["origin"],clock["coverage_end"])
        choices = [nearest(clock["pts"],t) for t in grid["times"]]
        require(len({i for i,_ in choices}) == len(choices), "Repeated source frames are not permitted by this CFR cache")
        for _,brackets in choices:
            needed.update(brackets)
        roi_needed.update(i for i,_ in choices)
        grids.append((grid,choices))
    frames,actual_pts,decoded = decode_needed(paths["video"],clock,needed)
    require(abs(actual_pts[0]-float(clock["origin"])) <= 1e-6, "First actual PTS is not the recording origin")
    h,w = frames[0].shape[:2]
    require([w,h] == old["video"]["source_size"], "Native geometry changed")
    traces,images = {},{}
    for index in sorted(roi_needed):
        frame = frames[index]
        require(frame.shape == (h,w,3) and frame.dtype == np.uint8, "Decoded geometry or pixel type changed")
        roi,rectangle = cropper.roi_crop(frame,boxes_xyxy[index])
        rgb = roi[:,:,::-1].copy()
        images[index] = rgb
        traces[index] = {"native_frame_index":index,"annotation_index":index,"actual_pts":rational(clock["pts"][index]),"decoded_pts_seconds":actual_pts[index],
            "source_bgr_sha256":hashlib.sha256(frame.tobytes()).hexdigest(),"roi_bgr_sha256":hashlib.sha256(roi.tobytes()).hexdigest(),"roi_rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),
            "annotation_box_xyxy":boxes_xyxy[index].tolist(),"crop_rectangle_xyxy":rectangle}
    comparisons = []
    for reference in entry["review_references"]:
        before = reference["trace"]
        after = traces[before["native_frame_index"]]
        for key in ("source_bgr_sha256","roi_bgr_sha256","annotation_index","annotation_box_xyxy","crop_rectangle_xyxy"):
            require(before[key] == after[key], f"Historical raw/ROI trace differs: {key}")
        require(abs(before["actual_pts_seconds"]-after["decoded_pts_seconds"]) < 1e-8, "Historical decoded PTS differs")
        comparisons.append({**reference,"raw_roi_pts_bbox_match":True})
    cache_records = []
    for clip,(grid,choices) in zip(entry["clips"],grids):
        rgb = np.stack([images[i] for i,_ in choices])
        minimum = validate_clip_array(rgb,clip["label"])
        require(minimum == clip["ctc_min_frames"], "CTC minimum differs")
        target_trace = []
        for position,(t,(index,bracket),overlap) in enumerate(zip(grid["times"],choices,grid["overlaps"])):
            # Selection uses exact source PTS already matched to these decoded neighbors.
            require(all(j in actual_pts for j in bracket), "Undecoded PTS bracket")
            target_trace.append({**traces[index],"grid_index":grid["first"]+position,"grid_time":rational(t),"semantic_overlap":rational(overlap),
                "selection_bracket":[{"native_frame_index":j,"actual_pts":rational(clock["pts"][j]),"decoded_pts_seconds":actual_pts[j]} for j in bracket]})
        offset = fraction(entry["frozen_offset"])
        neighbors = []
        for other in label["Sentence_info"]:
            if int(other["ID"]) == int(sentence["ID"]):
                continue
            a,b = Fraction(str(other["start_time"]))+offset,Fraction(str(other["end_time"]))+offset
            overlap = min(grid["display_end"],b)-max(grid["display_start"],a)
            if overlap > 0:
                neighbors.append({"sentence_id":other["ID"],"source_text":other["sentence_text"],"overlap":rational(overlap),"time_basis":"original label decimal times plus frozen offset"})
        metadata = {"schema":SCHEMA,**clip,"parent_id":entry["parent_id"],"speaker_id":entry["speaker_id"],"recording_group":entry["recording_group"],
            "source_members":entry["source_members"],"review_provenance":entry["review_provenance"],"parent_audio_review_provenance":entry["parent_audio_review_provenance"],"plan_sha256":sha(PLAN),
            "clock_mapping":{"context_origin":entry["context_origin"],"offset_applied_once":entry["frozen_offset"],"recording_origin":rational(clock["origin"]),"decoded_first_pts_seconds":actual_pts[0],"native_period":rational(clock["period"]),"source_coverage_end":rational(clock["coverage_end"]),"exact_CFR_table_and_decoded_brackets_agree":True},
            "grid":{"policy":"D_recording_anchored_25fps_display_overlap_v1","first":grid["first"],"stop":grid["stop"],"display_start":rational(grid["display_start"]),"display_end":rational(grid["display_end"]),"before_semantic":rational(grid["before_semantic"]),"after_semantic":rational(grid["after_semantic"]),"outside_semantic_bins_require_QC":True},
            "T":len(rgb),"shape":list(rgb.shape),"dtype":"uint8","color":"RGB","normalization":plan["normalization"],"tokenizer":plan["tokenizer"],
            "ctc_feasible":True,"trace":target_trace,"neighbor_label_intersections":neighbors,"rgb_uint8_sha256":hashlib.sha256(rgb.tobytes()).hexdigest(),
            "raw_padding_frames":0,"audio_included":False,"provisional":True,"training_ready":False,"root_QC_status":"pending"}
        target = BASE/"cache"/(clip["clip_id"]+".npz")
        target.parent.mkdir(exist_ok=True)
        require(not target.exists(), "Cache already exists")
        with target.open("xb") as stream:
            np.savez_compressed(stream,frames_rgb=rgb,label=np.asarray(clip["label"],dtype=np.int64),length=np.asarray(len(rgb),dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata,ensure_ascii=False,allow_nan=False)),schema=np.asarray(SCHEMA),training_ready=np.asarray(False),provisional=np.asarray(True))
        with np.load(target,allow_pickle=False) as packed:
            require(np.array_equal(packed["frames_rgb"],rgb) and np.array_equal(packed["label"],clip["label"]), "Lossless cache roundtrip failed")
        sidecar = BASE/"cache"/(clip["clip_id"]+".json")
        save(sidecar,metadata)
        cache_records.append({"clip_id":clip["clip_id"],"kind":clip["kind"],"split":clip["split"],"T":len(rgb),"npz_path":str(target),"npz_sha256":sha(target),"npz_bytes":target.stat().st_size,"metadata_path":str(sidecar),"metadata_sha256":sha(sidecar),"training_ready":False,"provisional":True})
    return {"parent_id":entry["parent_id"],"rank":entry["rank"],"recording_group":entry["recording_group"],"origin":rational(clock["origin"]),"native_period":rational(clock["period"]),"native_count":clock["sample_count"],
        "decoded_frames_in_short_ranges":decoded,"retained_native_frames":len(needed),"historical_raw_comparisons":comparisons,"clips":cache_records,"source_members_sha_verified":True,"training_ready":False}


def execute() -> dict[str,Any]:
    """Execute a fixed plan once; preserve partial outputs and exact failure/I/O evidence."""
    plan = read(PLAN)
    for path,expected in plan["source_hashes"].items():
        require(sha(Path(path)) == expected, f"Preflight-bound source changed: {path}")
    results_dir = BASE/"results"
    require(not results_dir.exists() and not (BASE/"cache").exists(), "Execution outputs exist; use a reviewed separate version")
    results_dir.mkdir()
    log = results_dir/"io_events.jsonl"
    io = {"member_read_attempts":0,"payload_bytes_read":0,"members_sha_verified":0}
    records = []
    helper,cropper = load_module(TAR_HELPER,"stage3_tar_runtime"),load_module(ROI_HELPER,"stage3_roi_runtime")
    started = time.perf_counter()
    try:
        for entry in plan["records"]:
            print(f"START rank {entry['rank']} {entry['parent_id']}",flush=True)
            with materialize(entry,helper,io,log) as paths:
                result = decode_record(paths,entry,plan,cropper)
            require(not list((BASE/"_tmp").iterdir()), "Temporary originals remain")
            save(results_dir/(entry["parent_id"]+".json"),result)
            records.append(result)
            print(f"DONE rank {entry['rank']}: {sum(c['T'] for c in result['clips'])} provisional RGB frames",flush=True)
        require(len(records) == 15 and io["payload_bytes_read"] == plan["planned_payload_bytes"] and io["members_sha_verified"] == 30, "Final source count differs from plan")
        output = {"schema":"lipnet-stage3-original-extraction-result-v1","status":"provisional_caches_created_root_QC_pending","plan_sha256":sha(PLAN),"records":records,"io":io,
            "cache_count":sum(len(r["clips"]) for r in records),"total_rgb_frames":sum(c["T"] for r in records for c in r["clips"]),"temporary_originals_remaining":0,
            "elapsed_seconds":time.perf_counter()-started,"audio_members_read":0,"neural_calls":0,"training_updates":0,"training_ready":False,"provisional":True}
        save(results_dir/"extraction_summary.json",output)
        return output
    except Exception as exc:
        remaining = [str(p) for p in (BASE/"_tmp").rglob("*")] if (BASE/"_tmp").exists() else []
        save(results_dir/"failure.json",{"status":"failed_preserved_partial_outputs","error":str(exc),"traceback":traceback.format_exc(),"io":io,"completed_parents":len(records),"remaining_temporary_paths":remaining,"training_ready":False})
        raise


def main() -> None:
    """Keep actual original extraction an explicit separate invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--split-plan",type=Path,default=SPLIT)
    args = parser.parse_args()
    try:
        result = execute() if args.execute else prepare(args.split_plan.resolve())
        print(json.dumps({k:result[k] for k in ("status","training_ready")},ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError("Stage3 provisional visual preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
