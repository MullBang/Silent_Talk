"""Prepare a separate visual supplement for the user's C004 cut-tail report."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from prepare_visual_review import MEDIA, OUT, decode, read_json, sha, tile


def main() -> None:
    """Read the +80ms proposal and preserve the prior eight contact sheets."""
    try:
        proposal_path = OUT.parent / "results/repair04_proposal.json"
        proposal = read_json(proposal_path)
        media = read_json(MEDIA)
        row = next(r for r in media["records"] if r["id"] == proposal["id"])
        base = read_json(OUT / "visual_contact_manifest.json")
        if sha(MEDIA) != base["sources"][str(MEDIA)]:
            raise ValueError("Media provenance changed")
        video = row["video"]
        full, roi = decode(video["full_preview"]), decode(video["roi_preview"])
        start = proposal["proposed_audio"]["start_sample_16k"] / 16000
        end = proposal["proposed_audio"]["end_sample_16k"] / 16000
        old_end = proposal["base_audio"]["end_sample_16k"] / 16000
        if abs(end - old_end - .08) > 1e-10:
            raise ValueError("Expected +80ms proposal")
        requests = [("start", start), ("mid", (start + end) / 2), ("old_end", old_end)]
        requests += [(f"end{d:+.2f}" if d else "end", end + d) for d in (-.08, -.04, 0, .04, .08)]
        samples = []
        for label, time_s in requests:
            index = round(time_s * 25)
            if not 0 <= index < len(roi):
                raise ValueError("Time outside review context")
            trace = video["trace"][index]
            requested_native = row["context_audio"]["actual_source_start_seconds"] + time_s + row["offset_seconds"]
            samples.append({"label": label, "requested_context_time_seconds": time_s,
                "review_frame_index": index, "review_frame_time_seconds": index / 25,
                "review_time_quantization_seconds": index / 25 - time_s,
                "requested_native_video_time_seconds": requested_native,
                "actual_native_minus_requested_seconds": trace["actual_pts_seconds"] - requested_native,
                "source_trace": trace,
                "decoded_roi_bgr_sha256": hashlib.sha256(roi[index].tobytes()).hexdigest(),
                "decoded_full_bgr_sha256": hashlib.sha256(full[index].tobytes()).hexdigest()})
        display = {"id": proposal["id"] + "_end_plus80ms", "text": proposal["text"]}
        result = {"schema": "offline-visual-boundary-supplement-v1", "id": proposal["id"],
            "segment_id": proposal["base_segment_id"] + ":end_plus80ms",
            "start_sample_16k": proposal["proposed_audio"]["start_sample_16k"],
            "end_sample_16k": proposal["proposed_audio"]["end_sample_16k"],
            "samples": samples, "source_hashes": {str(proposal_path): sha(proposal_path),
                str(MEDIA): sha(MEDIA), str(OUT / "visual_contact_manifest.json"): sha(OUT / "visual_contact_manifest.json"),
                str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
                str(OUT / "prepare_visual_review.py"): sha(OUT / "prepare_visual_review.py"),
                video["roi_preview"]["path"]: video["roi_preview"]["sha256"],
                video["full_preview"]["path"]: video["full_preview"]["sha256"]},
            "contact_sheets": [tile(display, roi, samples, False), tile(display, full, samples, True)],
            "decoded_each_preview_frames": len(roi), "human_visual_approval": False,
            "training_ready": False, "model_calls": 0, "raw_tar_reads": 0,
            "limitation": "Diagnostic existing review-frame sampling only; no statement about final human audio decision."}
        (OUT / "repair04_visual_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"samples": len(samples), "decoded_frames_total": len(roi) + len(full)}))
    except Exception as exc:
        raise RuntimeError("Repair visual supplement failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
