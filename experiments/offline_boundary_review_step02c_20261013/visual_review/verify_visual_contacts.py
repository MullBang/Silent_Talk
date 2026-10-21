"""Verify diagnostic pixels, hashes, metadata IDs and unchanged input bindings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    """Compare every displayed pixel panel to its recorded decoded-frame hash."""
    try:
        root = Path(__file__).resolve().parent
        manifests = [json.loads((root / name).read_text(encoding="utf-8"))
            for name in ("visual_contact_manifest.json", "repair04_visual_manifest.json")]
        hash_checks = 0
        pixel_panels = 0
        for m in manifests:
            sources = m.get("sources", m.get("source_hashes"))
            for path, expected in sources.items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Source changed: {path}")
                hash_checks += 1
            records = m.get("records", [m])
            for r in records:
                for sheet in r["contact_sheets"]:
                    path = Path(sheet["path"])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != sheet["sha256"]:
                        raise ValueError("Sheet hash changed")
                    hash_checks += 1
                    is_full = path.stem.endswith("_full")
                    selected = [s for s in r["samples"] if s["label"] in ("start", "mid", "end")] if is_full else r["samples"]
                    columns, cell_w, cell_h = (3, 640, 418) if is_full else (5, 206, 164)
                    width, height = (640, 360) if is_full else (96, 96)
                    image = Image.open(path).convert("RGB")
                    for i, sample in enumerate(selected):
                        x = (i % columns) * cell_w + (0 if is_full else 55)
                        y = 70 + (i // columns) * cell_h
                        bgr = np.asarray(image.crop((x, y, x + width, y + height)))[:, :, ::-1].copy()
                        expected = sample["decoded_full_bgr_sha256" if is_full else "decoded_roi_bgr_sha256"]
                        if hashlib.sha256(bgr.tobytes()).hexdigest() != expected:
                            raise ValueError("Contact pixels differ from decoded preview")
                        pixel_panels += 1
        observations = json.loads((root / "visual_observations.json").read_text(encoding="utf-8"))
        if [(r["id"], r["segment_id"]) for r in observations["records"]] != [(r["id"], r["segment_id"]) for r in manifests[0]["records"]]:
            raise ValueError("Observation IDs differ")
        result = {"status": "passed", "source_and_sheet_hash_checks": hash_checks,
            "unchanged_decoded_pixel_panels": pixel_panels,
            "human_visual_approvals_created": 0,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observation_sha256": hashlib.sha256((root / "visual_observations.json").read_bytes()).hexdigest()}
        (root / "visual_artifact_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result))
    except Exception as exc:
        raise RuntimeError("Diagnostic contact verification failed") from exc


if __name__ == "__main__":
    main()
