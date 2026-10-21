"""Create bounded review-only media for the frozen sixteen-sentence queue.

No model, training cache, service weight, or original label is modified. Audio
is annotation evidence only. Existing source members are verified one recording
at a time, and temporary original copies are removed by the audited reader.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import wave

import cv2
import numpy as np
import scipy

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
QUEUE = ROOT / "experiments/offline_curriculum_step01_20261012/results/alignment_review_queue.json"
QUEUE_SHA = "6940023f345a141af3929f818e15310e1595d0eccc9dbce99bb5cb9d0364111f"
SOURCE = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"
SOURCE_SHA = "d3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9"
TAR = ROOT / "experiments/local_source_step01_20261007/tar_source.py"
TAR_SHA = "05f72840576046683d6b5a3c1121a94b7fe2099fd0d85596c30da511801effb2"
SCORE = ROOT / "experiments/training_alignment_step06/refine_alignment.py"
SCORE_SHA = "70a35c12507eb0ed851045133cfb137bc22d7bae00adcd7662ceac572790bd5a"
FFMPEG = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
MEDIA = BASE / "media"
MANIFEST = BASE / "results/media_manifest.json"


def require(condition: bool, message: str) -> None:
    """Reject contract violations explicitly."""
    try:
        if not condition:
            raise ValueError(message)
    except ValueError:
        raise


def sha(path: Path) -> str:
    """Compute a bounded SHA-256 digest."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def bind(path: Path) -> dict[str, Any]:
    """Fingerprint local evidence."""
    try:
        return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f"Cannot bind {path}") from exc


def save(path: Path, data: Any) -> None:
    """Atomically save this stage's resumable metadata only."""
    try:
        require(path.resolve().is_relative_to(BASE), "Output must remain inside this experiment")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot save {path}") from exc


def load_helper(name: str, path: Path, expected: str) -> Any:
    """Load an immutable pure source or waveform helper."""
    try:
        require(sha(path) == expected, f"Frozen helper changed: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise RuntimeError(f"Cannot load {path}") from exc


def wav_info(path: Path) -> dict[str, Any]:
    """Inspect PCM without invoking an acoustic model."""
    try:
        with wave.open(str(path), "rb") as stream:
            return {"sample_rate": stream.getframerate(), "channels": stream.getnchannels(),
                    "sample_width_bytes": stream.getsampwidth(), "samples": stream.getnframes(),
                    "duration_seconds": stream.getnframes() / stream.getframerate()}
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect WAV {path}") from exc


def audio_excerpt(source: Path, output: Path, start: float, duration: float,
                  rate: int, clock: str) -> dict[str, Any]:
    """Accurately seek and save review PCM, preserving the declared clock."""
    try:
        require(start >= 0 and duration > 0 and not output.exists(), "Invalid or existing audio output")
        command = [str(FFMPEG), "-nostdin", "-v", "error", "-ss", f"{start:.9f}", "-i", str(source),
                   "-t", f"{duration:.9f}", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate),
                   "-c:a", "pcm_s16le", "-n", str(output)]
        origin = {"requested_source_start_seconds": start, "source_sample_origin_verified": False}
        input_bytes = None
        if clock == "original_label_wav":
            with wave.open(str(source), "rb") as stream:
                source_rate = stream.getframerate()
                first, size = round(start * source_rate), round(duration * source_rate)
                require(first >= 0 and first + size <= stream.getnframes(), "Exact PCM crop outside WAV")
                stream.setpos(first)
                value, channels, width = stream.readframes(size), stream.getnchannels(), stream.getsampwidth()
                require(len(value) == size * channels * width, "Incomplete native WAV sample read")
            native = io.BytesIO()
            with wave.open(native, "wb") as stream:
                stream.setnchannels(channels)
                stream.setsampwidth(width)
                stream.setframerate(source_rate)
                stream.writeframes(value)
            input_bytes = native.getvalue()
            command = [str(FFMPEG), "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
                       "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", "-n", str(output)]
            origin.update({"source_sample_origin_verified": True, "source_sample_rate": source_rate,
                           "source_first_sample": first, "source_samples_read": size,
                           "actual_source_start_seconds": first / source_rate,
                           "actual_source_end_seconds": (first + size) / source_rate,
                           "source_start_rounding_seconds": first / source_rate - start})
        process = subprocess.run(command, input=input_bytes, capture_output=True, timeout=120)
        require(process.returncode == 0, f"Audio extraction failed: {process.stderr.decode(errors='replace')}")
        info = wav_info(output)
        require((info["sample_rate"], info["channels"], info["sample_width_bytes"]) == (rate, 1, 2), "Invalid PCM")
        require(abs(info["samples"] - round(duration * rate)) <= (1 if input_bytes is not None else 0), "Incomplete PCM excerpt")
        return {**bind(output), **info, "clock": clock, "source_start_seconds": start,
                "source_end_seconds": start + duration, "requested_duration_seconds": duration,
                "command": command, "review_only": True, **origin}
    except Exception as exc:
        raise RuntimeError(f"Audio preparation failed: {output}") from exc


def pcm(spec: dict[str, Any]) -> np.ndarray:
    """Load the small saved 8 kHz signal used by the mapping diagnostic."""
    try:
        require(sha(Path(spec["path"])) == spec["sha256"] and spec["sample_rate"] == 8000, "PCM binding changed")
        with wave.open(spec["path"], "rb") as stream:
            return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float64) / 32768
    except Exception as exc:
        raise RuntimeError("Cannot read mapping PCM") from exc


def mapping_check(paths: dict[str, Path], row: dict[str, Any], folder: Path, scorer: Any) -> dict[str, Any]:
    """Check a six-second local WAV/MP4 match against the frozen offset."""
    try:
        start = (row["original_start_time"] + row["original_end_time"]) / 2 - 3
        search_start = start + row["offset_seconds"] - 1
        reference = audio_excerpt(paths["audio"], folder / "mapping_reference_8k.wav", start, 6., 8000, "original_label_wav")
        search = audio_excerpt(paths["video"], folder / "mapping_search_8k.wav", search_start, 8., 8000, "mp4")
        score = scorer.global_score(pcm(reference), pcm(search))
        measured = search_start + score["best_start_seconds_in_search"] - start
        residual = measured - row["offset_seconds"]
        require(score["strict_anchor_pass"] and abs(residual) <= .001, "Local waveform mapping failed; do not infer boundaries")
        return {"reference": reference, "search": search, "score": score,
                "measured_offset_seconds": measured, "frozen_offset_seconds": row["offset_seconds"],
                "residual_seconds": residual, "verified": True,
                "limit": "Audio-clock agreement does not verify transcript, word boundary, or visual speech synchrony."}
    except Exception as exc:
        raise RuntimeError(f"Waveform mapping failed for {row['id']}") from exc


def crop_rectangle(box: list[float], width: int, height: int) -> list[int]:
    """Match the production float32-corner, 20 percent margin contract."""
    try:
        x1, y1, x2, y2 = box
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "ROI outside source")
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        left, top = float(points[:, 0].min()), float(points[:, 1].min())
        right, bottom = float(points[:, 0].max()), float(points[:, 1].max())
        bw, bh = right - left, bottom - top
        left_i = max(0, min(int(np.floor(left - bw * .2)), width - 1))
        top_i = max(0, min(int(np.floor(top - bh * .2)), height - 1))
        return [left_i, top_i, max(left_i + 1, min(int(np.ceil(right + bw * .2)), width)),
                max(top_i + 1, min(int(np.ceil(bottom + bh * .2)), height))]
    except Exception as exc:
        raise RuntimeError("ROI crop failed") from exc


def encode_preview(frames: list[np.ndarray], audio: Path, output: Path) -> dict[str, Any]:
    """Encode explicitly selected BGR frames plus annotation-only audio."""
    try:
        require(bool(frames) and not output.exists(), "Empty or existing preview")
        height, width = frames[0].shape[:2]
        command = [str(FFMPEG), "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-i", str(audio),
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                   "-movflags", "+faststart", "-n", str(output)]
        process = subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames),
                                 capture_output=True, timeout=180)
        require(process.returncode == 0, "Preview encoding failed: " + process.stderr.decode(errors="replace"))
        capture = cv2.VideoCapture(str(output))
        try:
            require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames), "Preview frame count differs")
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            require(decoded == len(frames), "Preview does not fully decode")
        finally:
            capture.release()
        return {**bind(output), "fps": 25, "frames": len(frames), "width": width, "height": height,
                "video_duration_seconds": len(frames) / 25, "command": command,
                "review_only": True, "recompressed_not_a_training_input": True,
                "annotation_audio_attached": True, "visual_audio_alignment_precision": "nearest native frame; <= half a native frame"}
    except Exception as exc:
        raise RuntimeError(f"Cannot encode {output}") from exc


def video_previews(video: Path, label: dict[str, Any], row: dict[str, Any], source: dict[str, Any],
                   folder: Path, audio: dict[str, Any]) -> dict[str, Any]:
    """Decode timestamp-selected full frames and annotated lip ROIs once."""
    capture = None
    try:
        capture = cv2.VideoCapture(str(video))
        require(capture.isOpened(), "Source MP4 cannot open")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        require(abs(fps - source["video_fps"]) < 1e-10, "Native FPS differs")
        boxes = np.asarray(label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.float64)
        require(boxes.shape == (count, 4) and np.isfinite(boxes).all(), "ROI label count/shape invalid")
        boxes = boxes[:, [1, 0, 3, 2]]
        start = row["start_time"] - .35
        end = row["end_time"] + .35
        require(start >= 0 and end <= count / fps, "Context outside video")
        times = [start + k / 25 for k in range(round((end - start) * 25))]
        indices = [round(t * fps) for t in times]
        require(indices[-1] < count and all(a < b for a, b in zip(indices, indices[1:])), "Context indices invalid")
        capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        current, frame = indices[0], None
        rois, contexts, trace = [], [], []
        for target_time, selected in zip(times, indices, strict=True):
            while current <= selected:
                success, frame = capture.read()
                require(success, "Source video ended early")
                current += 1
            actual_index = float(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            require(actual_index == selected and abs(pts - selected / fps) <= .001
                    and abs(pts - target_time) <= .5 / fps + 1e-6, "Frame PTS/ordinal mismatch")
            box = boxes[selected].tolist()
            left, top, right, bottom = crop_rectangle(box, width, height)
            roi = cv2.resize(frame[top:bottom, left:right], (96, 96), interpolation=cv2.INTER_LINEAR)
            rois.append(roi)
            contexts.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
            trace.append({"native_frame_index": selected, "annotation_index": selected,
                          "target_video_time": target_time, "actual_pts_seconds": pts,
                          "annotation_box_xyxy": box, "crop_rectangle_xyxy": [left, top, right, bottom],
                          "source_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                          "roi_bgr_sha256": hashlib.sha256(roi.tobytes()).hexdigest()})
        roi_preview = encode_preview(rois, Path(audio["path"]), folder / "context_lips.mp4")
        full_preview = encode_preview(contexts, Path(audio["path"]), folder / "context_video.mp4")
        contact = np.concatenate([cv2.resize(rois[i], (192, 192)) for i in (0, len(rois) // 2, len(rois) - 1)], axis=1)
        require(cv2.imwrite(str(folder / "roi_contact.png"), contact), "Contact image failed")
        return {"native_fps": fps, "native_frame_count": count, "source_size": [width, height],
                "video_clock_start": start, "video_clock_end": end, "wav_clock_start": audio["source_start_seconds"],
                "offset_seconds": row["offset_seconds"], "target_fps": 25,
                "sampling": "round((start + k/25) * actual_fps), k < round(duration*25)",
                "roi_coordinate_order": "yxyx converted to xyxy", "roi_margin": .2,
                "roi_size": [96, 96], "no_frame_padding": True, "trace": trace,
                "roi_preview": roi_preview, "full_preview": full_preview, "roi_contact": bind(folder / "roi_contact.png")}
    except Exception as exc:
        raise RuntimeError(f"Video preparation failed for {row['id']}") from exc
    finally:
        if capture is not None:
            capture.release()


def prepare_one(row: dict[str, Any], source: dict[str, Any], tar: Any, scorer: Any) -> dict[str, Any]:
    """Materialize exactly one verified recording and persist only excerpts."""
    try:
        folder = MEDIA / row["id"]
        require(not folder.exists(), "Refuse to overwrite a prior media record")
        folder.mkdir(parents=True)
        started = time.perf_counter()
        with tar.materialize_source(source, MEDIA / "_tmp") as paths:
            raw = json.loads(paths["label"].read_text(encoding="utf-8-sig"))
            require(isinstance(raw, list) and len(raw) == 1, "Label schema changed")
            label = raw[0]
            sentence = next(x for x in label["Sentence_info"] if int(x["ID"]) == int(row["id"].rsplit("_", 1)[1]))
            require(sentence["sentence_text"] == row["source_text"], "Sentence text differs")
            require(abs(sentence["start_time"] - row["original_start_time"]) < 1e-6
                    and abs(sentence["end_time"] - row["original_end_time"]) < 1e-6, "Sentence boundary differs")
            require(Path(label["Video_info"]["video_Name"]).stem == row["source_stem"]
                    and label["speaker_info"]["speaker_ID"] == row["speaker_id"], "Label identity differs")
            duration = row["original_end_time"] - row["original_start_time"]
            require(row["original_start_time"] - .35 >= 0 and row["original_end_time"] + .35 <= wav_info(paths["audio"])["duration_seconds"], "WAV interval out of bounds")
            mapping = mapping_check(paths, row, folder, scorer)
            audio = audio_excerpt(paths["audio"], folder / "sentence.wav", row["original_start_time"], duration, 16000, "original_label_wav")
            context = audio_excerpt(paths["audio"], folder / "context.wav", row["original_start_time"] - .35, duration + .7, 16000, "original_label_wav")
            neighbors = [{"sentence_id": x["ID"], "source_text": x["sentence_text"],
                          "start_time": x["start_time"], "end_time": x["end_time"],
                          "overlap_seconds": min(x["end_time"], row["original_end_time"] + .35) - max(x["start_time"], row["original_start_time"] - .35)}
                         for x in label["Sentence_info"] if x["ID"] != sentence["ID"]
                         and min(x["end_time"], row["original_end_time"] + .35) > max(x["start_time"], row["original_start_time"] - .35)]
            videos = video_previews(paths["video"], label, row, source, folder, context)
            result = {"id": row["id"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                      "text": row["text"], "source_text": row["source_text"], "eojeols": row["eojeols"],
                      "source_members": source["members"], "original_start_time": row["original_start_time"],
                      "original_end_time": row["original_end_time"], "corrected_video_start": row["start_time"],
                      "corrected_video_end": row["end_time"], "offset_seconds": row["offset_seconds"],
                      "context_radius_seconds": .35, "sentence_within_context_seconds": [.35, .35 + duration],
                      "context_neighbor_intersections": neighbors, "context_contains_neighbor_label_interval": bool(neighbors),
                      "review_clock_contract": "context audio and video use matching WAV/video origins under frozen offset; exact WAV start rounds to native sample and video to nearest native frame",
                      "mapping_check": mapping, "sentence_audio": audio, "context_audio": context, "video": videos,
                      "archive_stats_checked_before_and_after": True, "all_member_sha256_verified": True,
                      "human_transcript_approved": False, "human_boundary_approved": False, "training_ready": False,
                      "word_boundaries": None, "audio_used_as_model_input": False,
                      "status": "media_ready_pending_alignment_and_human_review"}
        result["temporary_originals_removed"] = True
        result["elapsed_seconds"] = time.perf_counter() - started
        save(folder / "record.json", result)
        return result
    except Exception as exc:
        raise RuntimeError(f"Preparation failed for {row['id']}") from exc


def run(limit: int) -> None:
    """Validate the frozen queue, prepare a pilot, then resume without overwrite."""
    try:
        require(sha(QUEUE) == QUEUE_SHA and sha(SOURCE) == SOURCE_SHA, "Source metadata changed")
        queue = json.loads(QUEUE.read_text(encoding="utf-8"))
        plan = json.loads(SOURCE.read_text(encoding="utf-8"))
        sources = {r["stem"]: r for r in plan["sources"]}
        require(len(queue) == len({r["id"] for r in queue}) == 16, "Expected fixed sixteen records")
        require(1 <= limit <= 16, "Invalid run limit")
        for row in queue:
            source = sources[row["source_stem"]]
            require(source["members"] == row["d_source_members"] and source["speaker_id"] == row["speaker_id"], "Queue source binding differs")
            require(source["split"] == "train" and row["id"] in source["record_ids"], "Not an allowed training-source queue record")
            require(source["members"]["video"]["sha256"] == row["video_sha256"]
                    and source["members"]["label"]["sha256"] == row["label_sha256"], "Queue source SHA differs")
            require(row["training_ready_for_new_word_curriculum"] is False and row["stage1_human_approval_added"] is False, "Approval state changed")
            require(abs(row["start_time"] - row["original_start_time"] - row["offset_seconds"]) < 1e-10
                    and abs(row["end_time"] - row["original_end_time"] - row["offset_seconds"]) < 1e-10, "Clock formula invalid")
        inputs = {"queue": bind(QUEUE), "source_plan": bind(SOURCE), "producer": bind(Path(__file__)),
                  "tar_helper": bind(TAR), "waveform_helper": bind(SCORE), "ffmpeg": bind(FFMPEG)}
        tar, scorer = load_helper("offline2_tar", TAR, TAR_SHA), load_helper("offline2_scorer", SCORE, SCORE_SHA)
        result = {"schema": "lipnet-offline-word-alignment-media-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "inputs": inputs, "planned_ids": [r["id"] for r in queue], "records": [],
                  "environment": {"python": sys.version, "executable": sys.executable, "opencv": cv2.__version__,
                                  "numpy": np.__version__, "scipy": scipy.__version__},
                  "status": "preparing", "training_ready": False, "model_calls": 0, "training_updates": 0,
                  "limits": ["Review MP4s are recompressed and cannot be used as model inputs.",
                             "A local waveform match supports the existing time mapping, not human approval.",
                             "Neighboring utterances may be present in the 0.35-second review context."]}
        if MANIFEST.exists():
            result = json.loads(MANIFEST.read_text(encoding="utf-8"))
            require(result["inputs"] == inputs, "Cannot resume changed producer or evidence")
        done = {r["id"] for r in result["records"]}
        for row in queue[:limit]:
            if row["id"] in done:
                record_path = MEDIA / row["id"] / "record.json"
                prior = next(r for r in result["records"] if r["id"] == row["id"])
                require(json.loads(record_path.read_text(encoding="utf-8")) == prior, "Completed record changed")
                continue
            print(json.dumps({"event": "source_start", "id": row["id"]}), flush=True)
            value = prepare_one(row, sources[row["source_stem"]], tar, scorer)
            result["records"].append(value)
            result["completed_records"] = len(result["records"])
            result["status"] = "media_ready_pending_alignment_and_human_review" if len(result["records"]) == 16 else "pilot_or_partial_media_ready"
            save(MANIFEST, result)
            print(json.dumps({"event": "source_ready", "id": row["id"], "elapsed_seconds": value["elapsed_seconds"],
                              "mapping_correlation": value["mapping_check"]["score"]["absolute_correlation"]}), flush=True)
        require(inputs == {key: bind(Path(value["path"])) for key, value in inputs.items()}, "Input changed during run")
        require(not any((MEDIA / "_tmp").iterdir()), "Temporary source files remain")
        print(json.dumps({"manifest": str(MANIFEST), "sha256": sha(MANIFEST), "completed": len(result["records"])}), flush=True)
    except Exception as exc:
        raise RuntimeError("Review media preparation stopped without approving any training data") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    run(args.limit)
