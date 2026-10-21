"""Prepare separate current visual contacts for three sentence-tail proposals."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from prepare_visual_review import MEDIA, OUT, decode, read_json, sha, tile


def main() -> None:
    """Keep earlier candidates and bind the latest eight visual targets separately."""
    try:
        proposal_path = OUT.parent / "results/sentence_tail_proposals.json"
        proposal = read_json(proposal_path)
        media = read_json(MEDIA)
        base_path = OUT / "visual_contact_manifest.json"
        base = read_json(base_path)
        if sha(MEDIA) != base["sources"][str(MEDIA)]:
            raise ValueError("Media provenance changed")
        sources = {str(proposal_path): sha(proposal_path), str(MEDIA): sha(MEDIA),
            str(base_path): sha(base_path), str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
            str(OUT / "prepare_visual_review.py"): sha(OUT / "prepare_visual_review.py")}
        results = []
        for p in proposal["records"]:
            row = next(r for r in media["records"] if r["id"] == p["id"])
            video = row["video"]
            full, roi = decode(video["full_preview"]), decode(video["roi_preview"])
            if len(full) != len(roi) or len(roi) != len(video["trace"]):
                raise ValueError("Decoded frames/trace mismatch")
            for key in ("roi_preview", "full_preview"):
                spec = video[key]
                sources[spec["path"]] = spec["sha256"]
            start, end = p["start_sample_16k"] / 16000, p["end_sample_16k"] / 16000
            requests = [(f"start{d:+.2f}" if d else "start", start + d) for d in (-.08, -.04, 0, .04, .08)]
            requests += [("mid", (start + end) / 2)]
            requests += [(f"end{d:+.2f}" if d else "end", end + d) for d in (-.08, -.04, 0, .04, .08)]
            samples = []
            for label, time_s in requests:
                index = round(time_s * 25)
                if not 0 <= index < len(roi):
                    raise ValueError("Time outside review context")
                trace = video["trace"][index]
                target = video["video_clock_start"] + index / 25
                if abs(target - trace["target_video_time"]) > 1e-8 or round(target * video["native_fps"]) != trace["native_frame_index"]:
                    raise ValueError("Native trace mapping mismatch")
                requested_native = row["context_audio"]["actual_source_start_seconds"] + time_s + row["offset_seconds"]
                samples.append({"label": label, "requested_context_time_seconds": time_s,
                    "review_frame_index": index, "review_frame_time_seconds": index / 25,
                    "review_time_quantization_seconds": index / 25 - time_s,
                    "requested_native_video_time_seconds": requested_native,
                    "actual_native_minus_requested_seconds": trace["actual_pts_seconds"] - requested_native,
                    "source_trace": trace,
                    "decoded_roi_bgr_sha256": hashlib.sha256(roi[index].tobytes()).hexdigest(),
                    "decoded_full_bgr_sha256": hashlib.sha256(full[index].tobytes()).hexdigest()})
            display = {"id": p["id"] + "_sentence_tail", "text": p["text"]}
            result = {"id": p["id"], "segment_id": p["segment_id"], "text": p["text"],
                "recommendation_rank": p["rank"], "start_sample_16k": p["start_sample_16k"],
                "end_sample_16k": p["end_sample_16k"], "samples": samples,
                "contact_sheets": [tile(display, roi, samples, False), tile(display, full, samples, True)],
                "decoded_each_preview_frames": len(roi), "human_visual_approval": False,
                "training_ready": False, "model_calls": 0, "raw_tar_reads": 0}
            results.append(result)
        output = {"schema": "offline-visual-sentence-tail-supplement-v1", "records": results,
            "source_hashes": sources, "status": "prepared_for_actual_visual_observation",
            "human_visual_approvals_created": 0, "training_ready": False,
            "limitation": "Technical review preview only; human audio decisions remain in the parent stage2c evidence ledger."}
        path = OUT / "sentence_tail_visual_manifest.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        final_rows = sorted([r for r in base["records"] if r["recommendation_rank"] not in (4, 7, 8)] + results,
            key=lambda r: r["recommendation_rank"])
        handoff = {"schema": "offline-visual-latest-eight-v1", "records": final_rows,
            "source_hashes": {str(base_path): sha(base_path), str(path): sha(path)},
            "status": "current_visual_targets_audio_decisions_separate", "human_visual_approvals_created": 0,
            "training_ready": False, "superseded_diagnostics_preserved": ["C004 original", "C004 +80ms", "C293 original", "C297 original"]}
        (OUT / "latest_eight_visual_manifest.json").write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"records": len(results), "roi_samples": sum(len(r["samples"]) for r in results),
            "decoded_frames_total": sum(r["decoded_each_preview_frames"] * 2 for r in results)}))
    except Exception as exc:
        raise RuntimeError("Sentence-tail visual supplement failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
