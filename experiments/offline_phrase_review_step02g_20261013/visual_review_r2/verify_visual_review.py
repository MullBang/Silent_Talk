"""Verify review contact pixels, frame traces and clamped UI bounds without a server."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Read and hash an existing file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict[str, Any]:
    """Read a JSON evidence document."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def require(condition: bool, detail: str) -> None:
    """Reject an invalid evidence relationship without relying on assert flags."""
    if not condition:
        raise ValueError(detail)


def main() -> None:
    """Independently check every contact panel against a newly decoded preview."""
    try:
        manifest_path = BASE / "visual_review_manifest.json"
        manifest = read(manifest_path)
        hashes = manifest["source_hashes"]
        for path, expected in hashes.items():
            require(digest(Path(path)) == expected, f"Hash differs: {path}")
        media_path = next(Path(p) for p in hashes if p.endswith("media_manifest.json"))
        source_path = next(Path(p) for p in hashes if p.endswith("selected_phrase_manifest_r1.json"))
        media = {r["id"]: r for r in read(media_path)["records"]}
        proposed = {r["rank"]: r for r in read(source_path)["records"]}
        require([r["rank"] for r in manifest["records"]] == [9,10,11,12,13,15,16], "Candidate order")
        results = []
        decoded_count = panels = external = 0
        for row in manifest["records"]:
            candidate = proposed[row["rank"]]
            source = media[row["parent_id"]]["video"]
            cap = cv2.VideoCapture(source["roi_preview"]["path"])
            frames = []
            try:
                require(cap.isOpened(), "Cannot open preview")
                require(abs(cap.get(cv2.CAP_PROP_FPS)-25) < 1e-8, "Preview FPS")
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    require(frame.shape == (96,96,3), "Preview shape")
                    frames.append(frame)
            finally:
                cap.release()
            require(len(frames) == source["roi_preview"]["frames"] == len(source["trace"]), "Preview count")
            decoded_count += len(frames)
            for key in ("parent_id","segment_id","text","start_sample_16k","end_sample_16k","parent_start_sample_16k","parent_end_sample_16k"):
                require(row[key] == candidate[key], f"Candidate field differs: {key}")
            left, right = row["start_sample_16k"], row["end_sample_16k"]
            pleft, pright = row["parent_start_sample_16k"], row["parent_end_sample_16k"]
            require(pleft <= left < right <= pright, "Phrase outside parent")
            require(abs(row["context_start_seconds"] - max(pleft,left-3200)/16000) < 1e-9, "Context start")
            require(abs(row["context_end_seconds"] - min(pright,right+3200)/16000) < 1e-9, "Context end")
            require(row["training_ready"] is False and row["human_visual_approval"] is None, "Unapproved visual state")
            sheet_path = Path(row["contact_sheet"]["path"])
            require(digest(sheet_path) == row["contact_sheet"]["sha256"], "Contact hash")
            with Image.open(sheet_path) as loaded:
                sheet = np.asarray(loaded.convert("RGB"))
            row_external = 0
            for wi, window in enumerate(row["boundary_windows"]):
                boundary = (left if wi == 0 else right)/16000
                expected_indices = list(range(math.ceil((boundary-.2)*25-1e-9),math.floor((boundary+.2)*25+1e-9)+1))
                require([s["review_frame_index"] for s in window["samples"]] == expected_indices, "Chronological coverage")
                for col, sample in enumerate(window["samples"]):
                    idx = sample["review_frame_index"]
                    require(sample["source_trace"] == source["trace"][idx], "Historical native trace differs")
                    require(abs(sample["review_time_seconds"] - idx/25) < 1e-10, "Review time")
                    require(abs(sample["seconds_from_boundary"] - (idx/25-boundary)) < 1e-10, "Boundary delta")
                    require(hashlib.sha256(frames[idx].tobytes()).hexdigest() == sample["decoded_roi_bgr_sha256"], "Decoded pixel hash")
                    x, y = 12+col*120, 91+wi*218
                    rgb = cv2.cvtColor(frames[idx],cv2.COLOR_BGR2RGB)
                    require(np.array_equal(sheet[y:y+96,x:x+96],rgb), "Pasted panel pixels differ")
                    inside = pleft <= idx*640 < pright
                    require(sample["within_confirmed_parent_time"] == inside, "Parent containment flag")
                    row_external += not inside
                    panels += 1
            external += row_external
            results.append({"rank":row["rank"],"decoded_roi_frames":len(frames),"verified_pixels":20,"parent_external_panels":row_external,"parent_clamped_context_after_ms":(min(pright,right+3200)-right)/16})
        require(decoded_count == manifest["decoded_roi_frames"] == 750, "Total decoded frames")
        require(panels == manifest["roi_panels"] == 140, "Panel count")
        require(external == 29, "Parent-external count")
        report = {"schema":"seven-phrase-visual-verification-r2-v1","status":"passed","manifest_sha256":digest(manifest_path),"verifier_sha256":digest(Path(__file__)),"checked_source_hashes":len(hashes),"checked_contact_hashes":7,"checked_decoded_preview_frames":decoded_count,"checked_contact_pixels":panels,"historical_trace_equalities":panels,"parent_external_panels":external,"records":results,"server_started":False,"browser_playback_verified":False,"new_tar_reads":0,"new_neural_calls":0,"training_ready":False,"limitation":"Original native pixels were not reread. Historical original hashes are copied and equality-checked against the existing trace; actual panel pixels were checked against current decoding of the compressed preview."}
        output = BASE / "visual_verification.json"
        require(not output.exists(), "Preserve previous verification")
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":"passed","source_hashes":len(hashes),"decoded":decoded_count,"panels":panels,"parent_external":external}))
    except Exception as exc:
        raise RuntimeError("Read-only visual verification failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
