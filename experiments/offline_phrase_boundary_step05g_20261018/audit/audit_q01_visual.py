"""Check Q01 revised preview windows and decoded review pixels independently."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import cv2

from source_preflight import HERE, PREVIOUS, ROOT, bind, check, read_json


def fraction(value: dict[str, Any]) -> Fraction:
    """Decode the recorded rational clock without float rounding."""
    return Fraction(value["numerator"], value["denominator"])


def decode_hashes(path: Path) -> list[str]:
    """Fully decode a 96-square 25fps review clip and verify each timestamp."""
    capture = cv2.VideoCapture(str(path))
    values: list[str] = []
    try:
        if not capture.isOpened():
            raise AssertionError(f"Cannot open review video: {path}")
        assert abs(capture.get(cv2.CAP_PROP_FPS) - 25) < 1e-9
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            good, frame = capture.read()
            if not good:
                break
            assert frame.shape == (96, 96, 3)
            assert abs(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - len(values) / 25) < 1e-6
            values.append(hashlib.sha256(frame.tobytes()).hexdigest())
        assert len(values) == count
    finally:
        capture.release()
    return values


def inspect() -> dict[str, Any]:
    """Verify only new previews and their selected source-preview frame hashes."""
    stage = HERE.parent
    result_path = stage / "visual_Q01_r1/record.json"
    result = read_json(result_path)
    plan = read_json(Path(result["plan"]["path"]))
    revision = read_json(Path(result["revision_record"]["path"]))
    original = read_json(PREVIOUS / "media/lip_J_1_F_02_C002_A_012_051/record.json")
    for row in (result["plan"], result["audio_confirmation"], result["revision_record"], *plan["source_bindings"]):
        check(row)
    assert result["review_id"] == plan["review_id"] == revision["review_id"] == "Q01"
    assert result["revision_id"] == revision["revision_id"] == "Q01_end_parent_r1"
    assert result["text"] == revision["text"] == "너무 속상해"
    source_binding = plan["source_preview_binding"]
    assert source_binding == bind(Path(original["video"]["roi_preview"]["path"]))
    source_hashes = decode_hashes(Path(source_binding["path"]))
    origin = fraction(original["video"]["display_video_start"])
    offset = Fraction(original["offset_native_samples"], 48000)
    ffmpeg = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
    records = []
    for role in ("phrase", "context"):
        spec = result["videos"][role]
        check(spec)
        window = plan["windows"][role]
        assert spec["clock_window"] == window
        audio = revision["media"][role]
        assert window["audio_binding"] == bind(Path(audio["path"]))
        start = Fraction(audio["source_original_wav_first_sample_48k"], 48000)
        end = Fraction(audio["source_original_wav_end_sample_48k"], 48000)
        assert fraction(window["semantic_original_wav_start"]) == start
        assert fraction(window["semantic_original_wav_end"]) == end
        assert fraction(window["semantic_video_start"]) == start + offset
        assert fraction(window["semantic_video_end"]) == end + offset
        assert window["offset_applied_count"] == 1
        first, stop = math.floor((start + offset - origin) * 25), math.ceil((end + offset - origin) * 25)
        assert spec["source_first_display_frame"] == window["source_first_display_frame"] == first
        assert spec["source_stop_display_frame"] == window["source_stop_display_frame"] == stop
        assert window["original_grid_trace_slice"] == original["video"]["trace"][first:stop]
        assert spec["source_decoded_selected_bgr_sha256"] == source_hashes[first:stop]
        display_start, display_end = origin + Fraction(first, 25), origin + Fraction(stop, 25)
        assert fraction(window["preview_display_video_start"]) == display_start
        assert fraction(window["preview_display_video_end"]) == display_end
        assert fraction(window["display_before_semantic_seconds"]) == start + offset - display_start == Fraction(143, 6000)
        assert fraction(window["display_after_semantic_seconds"]) == display_end - end - offset == Fraction(4, 375)
        frames = decode_hashes(Path(spec["path"]))
        assert len(frames) == spec["frames"] == stop - first
        assert spec["fps"] == 25 and spec["duration_seconds"] == len(frames) / 25
        probe = subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-i", spec["path"], "-map", "0:a:0", "-f", "null", "-"], capture_output=True, timeout=30, check=False)
        assert probe.returncode != 0 and b"matches no streams" in probe.stderr
        assert spec["audio_tracks"] == 0 and spec["training_frame_count"] is None
        assert spec["review_only"] is spec["recompressed_preview_not_training_input"] is True
        records.append({"role": role, "source_frame_start": first, "source_frame_stop": stop,
                        "decoded_frames": len(frames), "source_BGR_hashes_match": True,
                        "native_trace_copy_matches": True, "no_audio_track": True})
    assert result["phrase_audio_boundary_approved"] is True
    assert result["phrase_visual_boundary_approved"] is result["visual_question_sent_by_producer"] is result["training_ready"] is False
    assert result["actual_visual_answer_pending"] is True
    assert result["actual_split"] is result["original_training_frame_count"] is result["ctc_feasible"] is None
    assert result["new_tar_payload_bytes"] == result["new_neural_calls"] == result["new_training_pixels"] == 0
    return {"schema": "offline-stage5g-independent-Q01-visual-audit-v1", "status": "passed",
            "visual_record": bind(result_path), "records": records,
            "decoded_revised_video_frames": sum(r["decoded_frames"] for r in records),
            "revised_video_count": 2, "source_preview_frames_decoded_once": len(source_hashes),
            "display_before_seconds": float(Fraction(143, 6000)), "display_after_seconds": float(Fraction(4, 375)),
            "new_visual_approval": False, "TAR_reads": 0, "neural_calls": 0,
            "limit": "Verification concerns recompressed review previews and copied provenance; no original-TAR pixel redecoding or human articulation boundary approval.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q01_visual_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "frames": result["decoded_revised_video_frames"]}))
