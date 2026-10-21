"""Independently check four revised Q05/Q06 review previews only."""
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
    """Verify new preview windows, source-review hashes and full output decode."""
    stage = HERE.parent
    ffmpeg = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
    checked: set[str] = set()
    results = []
    expected = {"Q05": [(28, 88), (23, 93)], "Q06": [(9, 40), (4, 45)]}
    revisions = {"Q05": "Q05_end_confirmed_parent", "Q06": "Q06_end_plus40ms"}
    for qid in ("Q05", "Q06"):
        path = stage / f"visual_{qid}_r1/record.json"
        result = read_json(path)
        plan = read_json(Path(result["plan"]["path"]))
        revision = read_json(stage / "revisions" / revisions[qid] / "record.json")
        confirmation = read_json(Path(result["audio_confirmation"]["path"]))
        for binding in [result["plan"], result["audio_confirmation"], *plan["source_bindings"]]:
            if binding["path"] not in checked:
                check(binding)
                checked.add(binding["path"])
        assert confirmation["phrase_audio_boundary_approved"] is True
        assert result["review_id"] == plan["review_id"] == confirmation["review_id"] == qid
        assert result["revision_id"] == revision["revision_id"] == confirmation["revision_id"]
        assert result["text"] == revision["text"]
        original = read_json(PREVIOUS / "media" / result["parent_id"] / "record.json")
        assert plan["source_preview_binding"] == bind(Path(original["video"]["roi_preview"]["path"]))
        source_hashes = decode_hashes(Path(plan["source_preview_binding"]["path"]))
        origin = fraction(original["video"]["display_video_start"])
        offset = Fraction(original["offset_native_samples"], 48000)
        phrase_start = fraction(revision["audio_start_seconds_exact"])
        phrase_end = fraction(revision["audio_end_seconds_exact"])
        assert plan["preserved_revision_diagnostics"] == revision["diagnostics"]
        assert plan["parent_vs_context_pcm"] == revision["parent_vs_context_pcm"]
        videos = []
        for index, role in enumerate(("phrase", "context")):
            spec, window = result["videos"][role], plan["windows"][role]
            check(spec)
            assert spec["clock_window"] == window
            assert window["audio_binding"] == bind(Path(revision["media"][role]["path"]))
            start = phrase_start - (Fraction(1, 5) if role == "context" else 0)
            end = phrase_end + (Fraction(1, 5) if role == "context" else 0)
            assert fraction(window["semantic_original_wav_start"]) == start
            assert fraction(window["semantic_original_wav_end"]) == end
            assert fraction(window["semantic_video_start"]) == start + offset
            assert fraction(window["semantic_video_end"]) == end + offset
            assert window["offset_applied_count"] == 1
            first, stop = math.floor((start + offset - origin) * 25), math.ceil((end + offset - origin) * 25)
            assert (first, stop) == expected[qid][index]
            assert (spec["source_first_display_frame"], spec["source_stop_display_frame"]) == (first, stop)
            assert window["original_grid_trace_slice"] == original["video"]["trace"][first:stop]
            assert spec["source_decoded_selected_bgr_sha256"] == source_hashes[first:stop]
            display_start, display_end = origin + Fraction(first, 25), origin + Fraction(stop, 25)
            assert fraction(window["preview_display_video_start"]) == display_start
            assert fraction(window["preview_display_video_end"]) == display_end
            before, after = start + offset - display_start, display_end - end - offset
            assert fraction(window["display_before_semantic_seconds"]) == before
            assert fraction(window["display_after_semantic_seconds"]) == after
            assert 0 <= before < Fraction(1, 25) and 0 <= after < Fraction(1, 25)
            hashes = decode_hashes(Path(spec["path"]))
            assert len(hashes) == spec["frames"] == stop - first
            assert spec["fps"] == 25 and spec["duration_seconds"] == len(hashes) / 25
            probe = subprocess.run([str(ffmpeg), "-nostdin", "-v", "error", "-i", spec["path"], "-map", "0:a:0", "-f", "null", "-"], capture_output=True, timeout=30, check=False)
            assert probe.returncode != 0 and b"matches no streams" in probe.stderr
            assert spec["audio_tracks"] == 0 and spec["training_frame_count"] is None
            videos.append({"role": role, "source_start_frame": first, "source_stop_frame": stop,
                           "frames_decoded": len(hashes), "source_BGR_hashes_exact": True,
                           "copied_PTS_trace_exact": True, "no_audio": True,
                           "display_before_seconds": float(before), "display_after_seconds": float(after)})
        assert result["phrase_audio_boundary_approved"] is True
        assert result["phrase_visual_boundary_approved"] is result["training_ready"] is result["new_human_approval_added"] is False
        assert result["actual_split"] is result["original_training_frame_count"] is result["ctc_feasible"] is None
        assert result["new_tar_payload_bytes"] == result["new_neural_calls"] == result["new_training_pixels"] == 0
        results.append({"review_id": qid, "visual_record": bind(path), "videos": videos,
                        "source_review_preview_decoded_once": len(source_hashes), "new_visual_approval": False})
    return {"schema": "offline-stage5g-independent-Q05-Q06-visual-audit-v1", "status": "passed", "records": results,
            "new_preview_videos": 4, "decoded_output_frames": 202, "TAR_reads": 0, "neural_calls": 0,
            "older_frozen_set_reverification": False,
            "limit": "Technical verification concerns recompressed review pixels and copied original-time provenance; it is not original-TAR redecoding or a human visual-boundary approval.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q05_Q06_visual_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "videos": 4, "frames": 202}))
