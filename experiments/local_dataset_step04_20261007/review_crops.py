"""Create a deterministic visual sample covering every selected source recording.

This reads only prepared student pixels and never opens original video/audio.
The contact sheets support a sampled crop review, not transcript validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PARENT = ROOT / "experiments/feature_transfer_step07/data_plan.json"
PARENT_SHA = "c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30"


def binding(path: Path) -> dict:
    """Hash a local prepared artifact for repeatable review provenance."""
    try:
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Only local project artifacts are allowed")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "sha256": digest}
    except OSError as exc:
        raise RuntimeError(f"Cannot bind review artifact {path}") from exc


def create(expected_manifest_sha: str) -> dict:
    """Sample first/middle/last frames from the earliest selected clip per source."""
    try:
        manifest_path = BASE / "inputs/manifest.json"
        manifest_binding = binding(manifest_path)
        if manifest_binding["sha256"] != expected_manifest_sha or binding(PARENT)["sha256"] != PARENT_SHA:
            raise ValueError("Prepared manifest or parent plan changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parent = json.loads(PARENT.read_text(encoding="utf-8"))
        if manifest["status"] != "passed" or manifest["ids"] != [row["id"] for row in parent["records"]]:
            raise ValueError("Expected completed unchanged full512 input manifest")
        entries = {entry["id"]: entry for entry in manifest["records"]}
        groups = {}
        for row in parent["records"]:
            groups.setdefault(row["source_stem"], []).append(row)
        selected = [min(rows, key=lambda row: (row["original_start_time"], row["id"]))
                    for _, rows in sorted(groups.items())]
        if len(selected) != 59:
            raise ValueError("Expected all59 source recordings")
        output = BASE / "review"
        output.mkdir(parents=True, exist_ok=True)
        records, sheets = [], []
        for page in range((len(selected) + 15) // 16):
            rows = selected[page * 16:(page + 1) * 16]
            sheet = np.full((len(rows) * 136 + 34, 510, 3), 245, np.uint8)
            cv2.putText(sheet, "Student first / middle / last (all source recordings)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (20, 20, 20), 1)
            for row_index, row in enumerate(rows):
                entry = entries[row["id"]]
                path = Path(entry["npz_path"])
                bound = binding(path)
                if bound["sha256"] != entry["npz_sha256"]:
                    raise ValueError("Prepared sample input changed")
                with np.load(path, allow_pickle=False) as data:
                    pixels = data["student_rgb"]
                    if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96, 3):
                        raise ValueError("Unexpected sampled student pixels")
                    positions = [0, len(pixels) // 2, len(pixels) - 1]
                    indices = data["student_source_indices"][positions].tolist()
                    y = 34 + row_index * 136
                    cv2.putText(sheet, f"{row['split']} {row['id']}", (8, y + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, .40, (20, 20, 20), 1)
                    for column, position in enumerate(positions):
                        x = 8 + column * 165
                        sheet[y + 25:y + 121, x:x + 96] = cv2.cvtColor(pixels[position], cv2.COLOR_RGB2BGR)
                records.append({"id": row["id"], "split": row["split"], "source_stem": row["source_stem"],
                                "input": bound, "student_positions": positions, "source_frame_indices": indices,
                                "sheet_index": page, "sheet_row": row_index})
            path = output / f"source_crops_{page + 1:02d}.png"
            ok, encoded = cv2.imencode(".png", sheet)
            if not ok:
                raise ValueError("Cannot encode contact sheet")
            with path.open("xb") as stream:
                stream.write(encoded.tobytes())
            sheets.append(binding(path))
        result = {"schema": "step04-all-source-crop-sample-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "generated_pending_visual_review", "manifest": manifest_binding,
                  "producer": binding(Path(__file__)), "parent_plan": binding(PARENT),
                  "selection": "earliest selected sentence per source, ordered by source_stem; first/middle/last student frame",
                  "sources": 59, "sampled_images": 177, "records": records, "sheets": sheets,
                  "original_media_read": False, "audio_read": False,
                  "limits": ["Only59 of512 sentences and177 sample frames are displayed.",
                             "No transcript truth or full temporal lip/audio synchronization conclusion."]}
        with (output / "sample_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as exc:
        raise RuntimeError("Full-source crop sampling did not finish") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    result = create(args.manifest_sha256)
    print(json.dumps({"sources": result["sources"], "images": result["sampled_images"], "sheets": result["sheets"]}), flush=True)
