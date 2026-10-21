"""Independently verify the two new Q08 review previews."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from source_preflight import HERE, ROOT, PREVIOUS, bind, check, read_json
from audit_q01_visual import fraction, decode_hashes


def inspect() -> dict[str, Any]:
    """Verify selected review pixels and timing without original-TAR decoding."""
    stage = HERE.parent
    path = stage / "visual_Q08_r1/record.json"
    result = read_json(path)
    plan = read_json(Path(result["plan"]["path"]))
    revision = read_json(stage / "revisions/Q08_end_plus80ms/record.json")
    original = read_json(PREVIOUS / "media" / result["parent_id"] / "record.json")
    for row in [result["plan"], result["audio_confirmation"], *plan["source_bindings"]]:
        check(row)
    confirmation = read_json(Path(result["audio_confirmation"]["path"]))
    assert confirmation["phrase_audio_boundary_approved"] is True
    assert confirmation["revision_record"] == bind(stage / "revisions/Q08_end_plus80ms/record.json")
    assert result["revision_id"] == revision["revision_id"] == "Q08_end_plus80ms_r1"
    assert plan["source_preview_binding"] == bind(Path(original["video"]["roi_preview"]["path"]))
    source_hashes = decode_hashes(Path(plan["source_preview_binding"]["path"]))
    origin = fraction(original["video"]["display_video_start"])
    offset = Fraction(original["offset_native_samples"], 48000)
    phrase_start = fraction(revision["audio_start_seconds_exact"])
    phrase_end = fraction(revision["audio_end_seconds_exact"])
    ffmpeg = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
    reports = []
    for role, bounds in (("phrase", (10, 47)), ("context", (5, 52))):
        spec, window = result["videos"][role], plan["windows"][role]
        check(spec)
        assert spec["clock_window"] == window
        assert window["audio_binding"] == bind(Path(revision["media"][role]["path"]))
        delta = Fraction(1, 5) if role == "context" else 0
        start, end = phrase_start - delta, phrase_end + delta
        assert fraction(window["semantic_original_wav_start"]) == start
        assert fraction(window["semantic_original_wav_end"]) == end
        assert fraction(window["semantic_video_start"]) == start + offset
        assert fraction(window["semantic_video_end"]) == end + offset
        assert window["offset_applied_count"] == 1
        first, stop = math.floor((start + offset - origin) * 25), math.ceil((end + offset - origin) * 25)
        assert (first, stop) == bounds == (spec["source_first_display_frame"], spec["source_stop_display_frame"])
        assert window["original_grid_trace_slice"] == original["video"]["trace"][first:stop]
        assert spec["source_decoded_selected_bgr_sha256"] == source_hashes[first:stop]
        display_start, display_end = origin + Fraction(first, 25), origin + Fraction(stop, 25)
        assert fraction(window["preview_display_video_start"]) == display_start
        assert fraction(window["preview_display_video_end"]) == display_end
        before, after = start + offset - display_start, display_end - end - offset
        assert fraction(window["display_before_semantic_seconds"]) == before
        assert fraction(window["display_after_semantic_seconds"]) == after
        hashes = decode_hashes(Path(spec["path"]))
        assert len(hashes) == spec["frames"] == stop - first
        assert spec["fps"] == 25 and spec["duration_seconds"] == len(hashes) / 25
        probe = subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-i", spec["path"], "-map", "0:a:0", "-f", "null", "-"], capture_output=True, timeout=30, check=False)
        assert probe.returncode != 0 and b"matches no streams" in probe.stderr
        assert spec["audio_tracks"] == 0 and spec["training_frame_count"] is None
        reports.append({"role": role, "decoded_frames": len(hashes), "source_BGR_hashes_exact": True,
                        "native_trace_copy_exact": True, "display_before_seconds": float(before),
                        "display_after_seconds": float(after), "no_audio": True})
    assert result["phrase_audio_boundary_approved"] is True
    assert result["phrase_visual_boundary_approved"] is result["training_ready"] is False
    assert result["actual_split"] is result["original_training_frame_count"] is result["ctc_feasible"] is None
    assert result["new_tar_payload_bytes"] == result["new_neural_calls"] == result["new_training_pixels"] == 0
    return {"schema": "offline-stage5g-independent-Q08-visual-audit-v1", "status": "passed",
            "visual_record": bind(path), "videos": reports, "decoded_new_frames": 84,
            "source_review_preview_decoded_once": len(source_hashes), "new_visual_approval": False,
            "TAR_reads": 0, "neural_calls": 0, "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q08_visual_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "videos": 2, "frames": 84}))
