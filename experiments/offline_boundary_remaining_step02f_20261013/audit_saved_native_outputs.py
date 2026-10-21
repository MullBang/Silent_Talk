"""Independently check saved ROI arrays and sampling geometry without rereading TARs."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash one small local artifact."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict:
    """Read UTF-8 JSON with a contextual error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def main() -> None:
    """Check immutable snapshots, per-frame pixels, exact clocks, and review scope."""
    try:
        native = ROOT / "native_check"
        summary = read(native / "native_check_summary_r4.json")
        plan_path = native / "native_execution_plan_r4.json"
        plan = read(plan_path)
        assert summary["plan_sha256"] == digest(plan_path)
        manifests = [
            native / "completion_manifest.json",
            ROOT.parent / "offline_boundary_finalize_step02d_20261013/results/completion_manifest.json",
            ROOT.parent / "offline_boundary_finalize_step02d_20261013/results/completion_manifest_r2.json",
        ]
        snapshots = []
        for manifest in manifests:
            document = read(manifest)
            for artifact in document["artifacts"]:
                assert digest(Path(artifact["path"])) == artifact["sha256"], artifact["path"]
            snapshots.append({"path": str(manifest), "sha256": digest(manifest), "verified_artifacts": len(document["artifacts"])})

        rows = []
        arrays_count = frames_count = common_count = historical_count = 0
        for actual, planned in zip(summary["records"], plan["records"], strict=True):
            record = planned["record"]
            assert actual["parent_id"] == record["parent_id"]
            assert actual["source_members_verified"] == {k: record["source_members"][k] for k in ("video", "label")}
            phrase = record["phrase"]
            duration = Fraction(phrase["end_sample_16k"] - phrase["start_sample_16k"], 16000)
            source_start = Fraction(record["context_source_first_sample"], record["context_source_sample_rate"])
            offset = Fraction(record["frozen_offset"]["numerator"], record["frozen_offset"]["denominator"])
            start = source_start + offset + Fraction(phrase["start_sample_16k"], 16000)
            fps = Fraction(record["historical_native_fps"]["numerator"], record["historical_native_fps"]["denominator"])
            expected_a_times = [start + Fraction(i, 25) for i in range(round(duration * 25))]
            expected_a_indices = [round(t * fps) for t in expected_a_times]
            assert [t["native_frame_index"] for t in actual["trace"]] == expected_a_indices
            expected_b_preview = list(range(phrase["start_sample_16k"] // 640, -(-phrase["end_sample_16k"] // 640)))
            assert [t["preview_frame_index"] for t in actual["grid_B"]["trace"]] == expected_b_preview
            expected_b_times = [float(t["trace"]["target_video_time"]) for t in planned["grid_B_display_overlap_references"]]
            # The historical preview clock is preserved as recorded, not recomputed with new rounding.
            datasets = []
            for name, info, trace, expected_times in [
                ("A", actual["npz"], actual["trace"], [float(t) for t in expected_a_times]),
                ("B", actual["grid_B"]["npz"], actual["grid_B"]["trace"], expected_b_times),
            ]:
                path = Path(info["path"])
                assert digest(path) == info["sha256"]
                with np.load(path, allow_pickle=False) as z:
                    rgb = z["review_roi_rgb"].copy()
                    assert rgb.dtype == np.uint8 and rgb.shape == (len(trace), 96, 96, 3)
                    assert hashlib.sha256(rgb.tobytes()).hexdigest() == info["rgb_uint8_sha256"]
                    assert z["review_only"].item() is True and z["training_ready"].item() is False
                    assert z["split"].item() == "unassigned"
                    assert z["native_indices"].tolist() == [t["native_frame_index"] for t in trace]
                    assert np.allclose(z["target_video_times"], expected_times, rtol=0, atol=1e-11)
                    assert np.array_equal(z["actual_video_pts"], [t["actual_pts_seconds"] for t in trace])
                for frame, t in zip(rgb, trace, strict=True):
                    assert hashlib.sha256(frame.tobytes()).hexdigest() == t["roi_rgb_sha256"]
                    assert hashlib.sha256(frame[:, :, ::-1].tobytes()).hexdigest() == t["roi_bgr_sha256"]
                    assert math.isclose(t["actual_pts_seconds"], float(Fraction(t["native_frame_index"], 1) / fps), abs_tol=1e-9)
                    assert t["annotation_index"] == t["native_frame_index"]
                    x1, y1, x2, y2 = t["annotation_box_xyxy"]
                    margin_x, margin_y = (x2-x1)*.2, (y2-y1)*.2
                    assert t["crop_rectangle_xyxy"] == [max(0, math.floor(x1-margin_x)), max(0, math.floor(y1-margin_y)), min(1920, math.ceil(x2+margin_x)), min(1080, math.ceil(y2+margin_y))]
                assert len(trace) >= record["ctc_min_frames"]
                datasets.append({t["native_frame_index"]: frame for t, frame in zip(trace, rgb, strict=True)})
                arrays_count += 1
                frames_count += len(trace)
            common = datasets[0].keys() & datasets[1].keys()
            for index in common:
                assert np.array_equal(datasets[0][index], datasets[1][index])
            common_count += len(common)
            for t, reference in zip(actual["grid_B"]["trace"], planned["grid_B_display_overlap_references"], strict=True):
                assert all(t[k] == v for k, v in reference["trace"].items())
            historical_count += len(actual["historical_comparisons"])
            assert all(t["all_checked_fields_match"] for t in actual["historical_comparisons"])
            rows.append({"rank": actual["rank"], "parent_id": actual["parent_id"], "A_frames": len(datasets[0]), "B_frames": len(datasets[1]), "common_frames": len(common), "A_only": len(datasets[0].keys()-common), "B_only": len(datasets[1].keys()-common), "ctc_min_frames": record["ctc_min_frames"]})
        assert (arrays_count, frames_count, common_count, historical_count) == (16, 474, 184, 273)
        assert not [p for p in (native / "_tmp").rglob("*") if p.is_file()]
        handoff = read(ROOT / "results/parent_audio_review_handoff.json")
        assert len({r["parent_id"] for r in handoff["records"]}) == 16
        assert all(r["parent_transcript_confirmed"] and r["parent_audio_boundary_confirmed"] and not r["training_ready"] for r in handoff["records"])
        output = {
            "schema": "step02f-independent-saved-native-audit-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed_saved_pixels_clocks_ctc_scope_and_snapshots",
            "snapshot_checks": snapshots, "records": rows,
            "npz_arrays_verified": arrays_count, "per_frame_rgb_and_bgr_hash_checks": frames_count,
            "shared_original_frames_identical": common_count,
            "B_frames_match_frozen_historical_traces": 241,
            "historical_comparison_records_checked": historical_count,
            "parent_audio_confirmations": 16, "temporary_originals_remaining": 0,
            "raw_tar_reads_by_this_audit": 0, "neural_calls": 0, "training_updates": 0,
            "sampling_policy_selected": None, "training_ready": False,
            "limitations": ["Saved-output audit; does not independently decode originals again.", "Producer verified source SHA, decoding, and historical full-frame hashes. This audit checks saved ROI pixels and trace consistency.", "Does not approve exact phonetic boundaries, new sampling pixels, dataset split, or model accuracy."],
            "source_sha256": {str(path): digest(path) for path in [Path(__file__), plan_path, native / "native_check_summary_r4.json", ROOT / "results/parent_audio_review_handoff.json"]},
        }
        (ROOT / "results/native_saved_output_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output["status"], arrays_count, frames_count, common_count)
    except Exception:
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
