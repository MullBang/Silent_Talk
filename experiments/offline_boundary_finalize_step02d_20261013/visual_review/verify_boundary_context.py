"""Verify boundary-only diagnostic source bindings, sequence coverage and pixels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    """Check actual artifact geometry without changing the frozen source records."""
    try:
        root = Path(__file__).resolve().parent
        manifest_path = root / "boundary_context_manifest.json"
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes = 0
        for path, expected in m["sources"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Source SHA differs")
            source_hashes += 1
        panels = 0
        for r in m["records"]:
            p = Path(r["contact_sheet"]["path"])
            if hashlib.sha256(p.read_bytes()).hexdigest() != r["contact_sheet"]["sha256"]:
                raise ValueError("Contact sheet changed")
            image = Image.open(p).convert("RGB")
            for row_index, window in enumerate(r["windows"]):
                low, high = window["requested_window_seconds"]
                expected = list(range(math.ceil(low * 25 - 1e-9), math.floor(high * 25 + 1e-9) + 1))
                if [s["review_frame_index"] for s in window["samples"]] != expected:
                    raise ValueError("Consecutive window coverage differs")
                for col, sample in enumerate(window["samples"]):
                    x, y = col * 120 + 12, 64 + row_index * 214 + 28
                    bgr = np.asarray(image.crop((x, y, x + 96, y + 96)))[:, :, ::-1].copy()
                    if hashlib.sha256(bgr.tobytes()).hexdigest() != sample["decoded_roi_bgr_sha256"]:
                        raise ValueError("Panel pixels differ")
                    panels += 1
        observations_path = root / "boundary_context_observations.json"
        obs = json.loads(observations_path.read_text(encoding="utf-8"))
        if [r["id"] for r in obs["records"]] != [r["id"] for r in m["records"]]:
            raise ValueError("Observation IDs differ")
        artifact = {"status": "passed", "source_hashes_verified": source_hashes,
            "sheets_verified": len(m["records"]), "chronological_windows": m["boundary_windows"],
            "unchanged_decoded_pixel_panels": panels,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "observations_sha256": hashlib.sha256(observations_path.read_bytes()).hexdigest(),
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "human_visual_approvals_created": 0, "training_ready": False}
        (root / "boundary_context_verification.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(artifact))
    except Exception as exc:
        raise RuntimeError("Boundary diagnostic verification failed") from exc


if __name__ == "__main__":
    main()
