"""Independently audit emitted parent media against bounded original evidence.

Reads eight label members plus native PCM windows directly from their TARs.
No original video is copied; original pixel values are consequently not
independently re-decoded here. Review video decoding, source annotation/PTS
geometry, exact PCM identity, resampling, and clock correspondence are checked.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any
import wave

import cv2
import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

from audit_preflight import HERE, ROOT, bind, check_binding, read_json
from tar_pcm import MemberReader, pcm_samples, validate_archive, wav_layout

BASE = HERE.parent
FFMPEG = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def ratio(row: dict[str, Any]) -> Fraction:
    """Read exact time, independently checking its optional display float."""
    value = Fraction(row["numerator"], row["denominator"])
    if "seconds" in row:
        assert abs(float(value) - row["seconds"]) < 1e-12
    return value


def pcm_file(path: Path) -> tuple[tuple[int, int, int, int], bytes]:
    """Read a small emitted PCM and reject truncated payloads."""
    with wave.open(str(path), "rb") as stream:
        rate, channels, width, count = stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()
        assert stream.getcomptype() == "NONE"
        value = stream.readframes(count)
    assert len(value) == count * channels * width
    return (rate, channels, width, count), value


def resample_source(pcm: bytes, channels: int) -> bytes:
    """Resample independently supplied archive PCM to pipe, never to a file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(pcm)
    command = [str(FFMPEG), "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    run = subprocess.run(command, input=buffer.getvalue(), capture_output=True, timeout=60, check=False)
    if run.returncode:
        raise RuntimeError(run.stderr.decode("utf8", errors="replace"))
    return run.stdout


def local_mapping(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute the complete lag curve without importing the producer scorer."""
    mapping = row["mapping_check"]
    values = []
    for name in ["reference", "search"]:
        spec = mapping[name]
        check_binding(spec)
        info, pcm = pcm_file(Path(spec["path"]))
        assert info[:3] == (8000, 1, 2)
        raw = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
        filt = butter(4, [100, 3400], btype="bandpass", fs=8000, output="sos")
        values.append(resample_poly(sosfiltfilt(filt, raw), 6, 1, window=("kaiser", 5.), padtype="constant"))
    reference, search = values
    reference -= reference.mean()
    n = reference.size
    sums = np.concatenate(([0.], np.cumsum(search)))
    squares = np.concatenate(([0.], np.cumsum(search ** 2)))
    energy = squares[n:] - squares[:-n] - (sums[n:] - sums[:-n]) ** 2 / n
    denominator = np.sqrt(np.maximum(energy, 0.) * np.dot(reference, reference))
    scores = np.divide(correlate(search, reference, mode="valid", method="fft"), denominator,
                       out=np.zeros_like(denominator), where=denominator > 1e-12)
    absolute = np.abs(scores)
    best = int(absolute.argmax())
    outside = absolute.copy()
    outside[max(0, best - 12000):min(outside.size, best + 12001)] = -1
    runner = int(outside.argmax())
    saved = mapping["score"]
    assert best == saved["best_start_sample_in_search"]
    assert runner == saved["runner_up_start_sample_in_search"]
    assert abs(absolute[best] - saved["absolute_correlation"]) < 1e-9
    assert abs(absolute[runner] - saved["runner_up_absolute_correlation"]) < 1e-9
    clearance = min(best, search.size - n - best) / 48000
    peak_ratio = absolute[best] / max(absolute[runner], 1e-12)
    rms_reference = math.sqrt(float(np.dot(reference, reference)) / n)
    rms_search = math.sqrt(float(max(energy[best], 0.)) / n)
    assert absolute[best] >= .99 and peak_ratio >= 1.5 and clearance >= .5
    assert rms_reference >= 1e-4 and rms_search >= 1e-4
    measured = mapping["search"]["source_start_seconds"] + best / 48000 - mapping["reference"]["source_start_seconds"]
    assert abs(measured - row["offset_seconds"]) <= .001
    assert abs(measured - mapping["measured_offset_seconds"]) < 1e-10
    return {"best_lag_samples_48k": best, "absolute_correlation": float(absolute[best]),
            "far_peak_ratio": float(peak_ratio), "measured_offset_seconds": measured,
            "saved_score_maximum_difference": abs(float(absolute[best]) - saved["absolute_correlation"]),
            "strict_gates_passed": True}


def intersections(label: dict[str, Any], parent_id: int, start: float, end: float) -> list[dict[str, Any]]:
    """Calculate adjacent-label overlap without interpreting it as heard speech."""
    rows = []
    for row in label["Sentence_info"]:
        overlap = min(end, row["end_time"]) - max(start, row["start_time"])
        if int(row["ID"]) != parent_id and overlap > 0:
            rows.append({"sentence_id": row["ID"], "source_text": row["sentence_text"],
                         "start_time": row["start_time"], "end_time": row["end_time"],
                         "intersection_seconds": overlap})
    return rows


def video_geometry(video: dict[str, Any], label: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Check every source-annotation/PTS selection and decode both review files."""
    origin, period = ratio(video["recording_origin"]), ratio(video["native_period"])
    left = Fraction(plan["context_native_start_sample"] + plan["offset_native_samples"], 48000)
    right = Fraction(plan["context_native_end_sample"] + plan["offset_native_samples"], 48000)
    assert ratio(video["semantic_context_video_start"]) == left
    assert ratio(video["semantic_context_video_end"]) == right
    first, end = math.floor((left - origin) * 25), math.ceil((right - origin) * 25)
    assert ratio(video["display_video_start"]) == origin + Fraction(first, 25)
    assert ratio(video["display_video_end"]) == origin + Fraction(end, 25)
    assert video["offset_native_samples_applied_once"] == plan["offset_native_samples"]
    assert len(video["trace"]) == end - first
    assert abs(1 / float(period) - video["native_fps"]) < 1e-9
    boxes = label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
    assert len(boxes) == video["native_frame_count"]
    width, height = video["source_size"]
    for k, trace in enumerate(video["trace"]):
        target = origin + Fraction(first + k, 25)
        assert ratio(trace["target_video_time"]) == target
        assert trace["display_grid_index"] == first + k
        exact_index = (target - origin) / period
        earlier, later = math.floor(exact_index), math.ceil(exact_index)
        brackets = ([i for i in (earlier - 1, earlier, earlier + 1)
                     if 0 <= i < video["native_frame_count"]]
                    if earlier == later else [earlier, later])
        assert trace["bracket_indices"] == brackets
        nearest = min(brackets, key=lambda i: (abs(origin + i * period - target), i))
        assert trace["native_frame_index"] == trace["annotation_index"] == nearest
        expected_pts = origin + nearest * period
        assert ratio(trace["exact_pts"]) == expected_pts
        assert abs(trace["actual_pts_seconds"] - float(expected_pts)) < 1e-6
        for index, pts in zip(brackets, trace["bracket_decoded_pts"], strict=True):
            assert abs(pts - float(origin + index * period)) < 1e-6
        y1, x1, y2, x2 = boxes[nearest]
        assert [x1, y1, x2, y2] == trace["annotation_box_xyxy"]
        x1, y1, x2, y2 = (float(x) for x in np.asarray([x1, y1, x2, y2], dtype=np.float32))
        bw, bh = x2 - x1, y2 - y1
        xl = max(0, min(math.floor(x1 - .2 * bw), width - 1))
        yl = max(0, min(math.floor(y1 - .2 * bh), height - 1))
        xr = max(xl + 1, min(math.ceil(x2 + .2 * bw), width))
        yr = max(yl + 1, min(math.ceil(y2 + .2 * bh), height))
        assert [xl, yl, xr, yr] == trace["crop_rectangle_xyxy"]
    decoded_counts = []
    for name, expected_size in [("roi_preview", (96, 96)), ("full_preview", (640, 360))]:
        spec = video[name]
        check_binding(spec)
        capture = cv2.VideoCapture(spec["path"])
        try:
            assert capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS) - 25) < 1e-9
            count = 0
            while True:
                good, frame = capture.read()
                if not good:
                    break
                assert frame.shape == (expected_size[1], expected_size[0], 3)
                count += 1
            assert count == len(video["trace"]) == spec["frames"]
            decoded_counts.append(count)
        finally:
            capture.release()
    check_binding(video["roi_contact"])
    assert video["review_only"] and not video["exact_articulation_boundary_approved"]
    return {"trace_count": len(video["trace"]), "native_annotation_geometry_verified": True,
            "nearest_actual_pts_and_earlier_ties_verified": True, "review_video_decoded_counts": decoded_counts,
            "original_pixel_values_independently_redecoded": False}


def inspect() -> dict[str, Any]:
    manifest_path = BASE / "media_prepare/media_manifest_08.json"
    manifest = read_json(manifest_path)
    check_binding(manifest["source_plan"])
    check_binding(manifest["execution_policy"])
    plan = read_json(Path(manifest["source_plan"]["path"]))
    assert len(manifest["records"]) == manifest["completed"] == 8
    output = []
    audit_bytes = 0
    for record, source in zip(manifest["records"], plan["records"], strict=True):
        record_read_start = audit_bytes
        assert record["id"] == source["parent_id"] and record["review_id"] == source["review_id"]
        saved_path = BASE / "media" / record["id"] / "record.json"
        assert read_json(saved_path) == record
        assert record["source_members"] == source["source"]["members"]
        q = source["queue_record"]
        assert record["parent_text"] == q["parent_text"] and record["proposed_phrase"] == q["text"]
        assert record["candidate_id"] == q["candidate_id"]
        assert record["parent_transcript_confirmed"] is False and record["human_boundary_approved"] is False
        assert record["actual_split"] is None and record["training_ready"] is False
        assert record["phrase_boundaries"] is None and record["audio_used_as_student_input"] is False
        label_spec = record["source_members"]["label"]
        label_path = validate_archive(label_spec)
        with label_path.open("rb") as stream:
            bounded = MemberReader(stream, label_spec["offset"], label_spec["size"])
            raw = bounded.read(0, label_spec["size"])
            audit_bytes += bounded.bytes_read
        assert hashlib.sha256(raw).hexdigest() == label_spec["sha256"]
        label = json.loads(raw.decode("utf-8-sig"))[0]
        sid = int(record["id"].rsplit("_", 1)[1])
        sentence = next(r for r in label["Sentence_info"] if int(r["ID"]) == sid)
        assert sentence["sentence_text"] == record["source_text"]
        assert sentence["start_time"] == record["original_start_time"]
        assert sentence["end_time"] == record["original_end_time"]
        assert label["speaker_info"]["speaker_ID"] == record["speaker_id"]
        assert Path(label["Video_info"]["video_Name"]).stem == record["source_stem"]
        off = Fraction(source["offset_native_samples"], 48000)
        windows = [
            ("sentence_audio", record["sentence_audio"], source["parent_native_start_sample"], source["parent_native_end_sample"]),
            ("context_audio", record["context_audio"], source["context_native_start_sample"], source["context_native_end_sample"]),
            ("display_audio", record["video"]["display_audio"],
             int((ratio(record["video"]["display_video_start"]) - off) * 48000),
             int((ratio(record["video"]["display_video_end"]) - off) * 48000)),
        ]
        audio_spec = record["source_members"]["audio"]
        audio_path = validate_archive(audio_spec)
        audio_checks = []
        with audio_path.open("rb") as stream:
            bounded = MemberReader(stream, audio_spec["offset"], audio_spec["size"])
            layout = wav_layout(bounded)
            assert layout["sample_rate"] == 48000 and layout["sample_width_bytes"] == 2
            for name, spec, first, end in windows:
                assert spec["source_first_sample"] == first and spec["source_end_sample_exclusive"] == end
                assert spec["source_samples_read"] == end - first
                assert spec["source_start_seconds"] == first / 48000 and spec["source_end_seconds"] == end / 48000
                original = pcm_samples(bounded, layout, first, end)
                assert hashlib.sha256(original).hexdigest() == spec["native_pcm_sha256"]
                check_binding(spec["native_crop"])
                native_info, native_pcm = pcm_file(Path(spec["native_crop"]["path"]))
                assert native_info == (48000, layout["channels"], 2, end - first)
                assert native_pcm == original
                check_binding(spec)
                review_info, review_pcm = pcm_file(Path(spec["path"]))
                assert review_info == (16000, 1, 2, spec["samples"])
                assert review_pcm == resample_source(original, layout["channels"])
                assert abs(spec["samples"] - round(Fraction(end - first, 3))) <= 1
                audio_checks.append({"role": name, "source_first_sample_48k": first,
                                     "source_end_sample_48k": end, "native_samples": end - first,
                                     "output_samples_16k": review_info[3], "native_pcm_exact": True,
                                     "resampled_pcm_exact": True})
            audit_bytes += bounded.bytes_read
        validate_archive(audio_spec)
        validate_archive(label_spec)
        context_neighbors = intersections(label, sid, windows[1][2] / 48000, windows[1][3] / 48000)
        display_neighbors = intersections(label, sid, windows[2][2] / 48000, windows[2][3] / 48000)
        assert context_neighbors == record["context_neighbor_intersections"]
        assert bool(context_neighbors) == record["context_contains_neighbor_label_interval"]
        geometry = video_geometry(record["video"], label, source)
        mapping = local_mapping(record)
        output.append({"review_id": record["review_id"], "parent_id": record["id"],
                       "media_record": bind(saved_path), "label_member_sha256_verified": True,
                       "audio_checks": audio_checks, "local_mapping": mapping,
                       "video": geometry, "context_neighbor_intersections": context_neighbors,
                       "display_neighbor_intersections": display_neighbors,
                       "additional_audit_tar_bytes_read": audit_bytes - record_read_start,
                       "new_approvals": 0})
    assert not any((BASE / "media/_tmp").iterdir())
    return {"schema": "offline-stage5f-independent-media-audit-v1", "status": "passed",
            "media_manifest": bind(manifest_path), "records": output,
            "parent_count": 8, "native_and_resampled_audio_checks": 24,
            "independent_lag_curve_checks": 8,
            "decoded_review_video_count": 16,
            "trace_entries_verified": sum(r["video"]["trace_count"] for r in output),
            "context_neighbor_parent_count": sum(bool(r["context_neighbor_intersections"]) for r in output),
            "display_neighbor_parent_count": sum(bool(r["display_neighbor_intersections"]) for r in output),
            "additional_audit_tar_bytes_read": audit_bytes,
            "prior_failed_attempt_tar_bytes_reconstructed": sum(r["additional_audit_tar_bytes_read"] for r in output[:6]),
            "all_attempts_additional_audit_tar_bytes_read": audit_bytes + sum(r["additional_audit_tar_bytes_read"] for r in output[:6]),
            "failed_attempt_note": "Initial auditor stopped at Q06 after label/PCM reads because it assumed a singleton exact-PTS bracket. Existing frozen rule includes both adjacent frames; producer unchanged. Failed-attempt byte count reconstructed from the identical six source/range reads before that assertion.",
            "additional_audit_tar_bytes_include": "Eight full JSON label payloads plus RIFF headers and three exact native PCM windows per WAV.",
            "source_video_payload_reads": 0, "source_original_materializations": 0,
            "media_files_written": 0, "new_human_approvals": 0, "training_ready": False,
            "limitations": ["Original frame pixel values were not independently re-decoded; only declared source annotation/PTS geometry and review video decoding were verified.",
                            "Waveform correspondence and file identity do not certify transcript correctness, phonetic boundaries, or lip/audio synchrony."]}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "media_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "parent_count", "native_and_resampled_audio_checks", "trace_entries_verified", "display_neighbor_parent_count", "additional_audit_tar_bytes_read"]}))
