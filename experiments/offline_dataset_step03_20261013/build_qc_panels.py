"""Render hash-checked cached RGB pixels for scoped human-readable technical QC."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    """Hash a single evidence file with bounded memory."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def main() -> None:
    """Display first three, middle and last three unmodified RGB frames per clip."""
    try:
        source = BASE / "extraction/results/extraction_summary.json"
        summary = json.loads(source.read_text(encoding="utf-8"))
        if summary["cache_count"] != 30 or summary["temporary_originals_remaining"] != 0:
            raise ValueError("Incomplete extraction")
        output = BASE / "qc"
        if output.exists():
            raise FileExistsError(output)
        output.mkdir()
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
        small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12)
        rows, panel_files = [], []
        records = sorted(summary["records"], key=lambda r: r["rank"])
        for page, first in enumerate(range(0, len(records), 5), 1):
            # 10 clip rows; seven 96px panels, each preserved without scaling.
            sheet = Image.new("RGB", (860, 50 + 10 * 158), "#f5f7f9")
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 10), f"Stage 3 / D grid / RGB cache QC {page} — first3, middle, last3", font=font, fill="#101b2e")
            row_index = 0
            for parent in records[first:first+5]:
                for clip in parent["clips"]:
                    npz_path = Path(clip["npz_path"])
                    sidecar = Path(clip["metadata_path"])
                    if digest(npz_path) != clip["npz_sha256"] or digest(sidecar) != clip["metadata_sha256"]:
                        raise ValueError("Cache source changed")
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    with np.load(npz_path, allow_pickle=False) as loaded:
                        frames = loaded["frames_rgb"]
                        embedded = json.loads(loaded["metadata_json"].item())
                    if embedded != metadata or frames.dtype != np.uint8 or frames.shape != (clip["T"], 96, 96, 3):
                        raise ValueError("Cache layout/metadata mismatch")
                    indices = [0, 1, 2, len(frames)//2, len(frames)-3, len(frames)-2, len(frames)-1]
                    if len(set(indices)) != 7:
                        raise ValueError("Unexpectedly short clip for QC contract")
                    y = 50 + row_index * 158
                    title = f"{parent['rank']:02d} {metadata['speaker_id']} / {clip['kind']} / {clip['split']} / T={clip['T']}"
                    draw.text((12, y), title, font=small, fill="#17243a")
                    draw.text((12, y+18), metadata["text"], font=small, fill="#17243a")
                    placements = []
                    for column, index in enumerate(indices):
                        x = 12 + column * 120
                        xy = (x, y+39)
                        sheet.paste(Image.fromarray(frames[index]), xy)
                        native = metadata["trace"][index]["native_frame_index"]
                        draw.text((x, y+138), f"t{index} / src{native}", font=small, fill="#28384e")
                        placements.append({"frame_index": index, "native_frame_index": native,
                                           "xy": list(xy), "rgb_sha256": hashlib.sha256(frames[index].tobytes()).hexdigest()})
                    rows.append({"parent_id": parent["parent_id"], "rank": parent["rank"], "clip_id": clip["clip_id"],
                                 "kind": clip["kind"], "npz_path": str(npz_path), "npz_sha256": digest(npz_path),
                                 "sheet": str(output / f"cache_qc_{page:02d}.png"), "panels": placements})
                    row_index += 1
            image_path = output / f"cache_qc_{page:02d}.png"
            sheet.save(image_path)
            # Compare saved image pixels against source panel hashes, excluding labels.
            saved = Image.open(image_path).convert("RGB")
            for row in [r for r in rows if r["sheet"] == str(image_path)]:
                for panel in row["panels"]:
                    x, y = panel["xy"]
                    raw = np.asarray(saved.crop((x, y, x+96, y+96))).tobytes()
                    if hashlib.sha256(raw).hexdigest() != panel["rgb_sha256"]:
                        raise ValueError("Rendered panel pixels changed")
            panel_files.append({"path": str(image_path), "sha256": digest(image_path)})
        result = {"schema": "offline-stage3-cached-pixel-contact-QC-v1", "source": str(source),
                  "source_sha256": digest(source), "clip_count": len(rows), "panel_count": sum(len(r["panels"]) for r in rows),
                  "images": panel_files, "rows": rows, "saved_panels_match_raw_RGB": True,
                  "visual_inspection_status": "pending", "scope": "Technical spatial/color/boundary-support samples; not exact phonetic boundary certification or exhaustive frame viewing."}
        (output / "panel_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"clips": len(rows), "panels": result["panel_count"], "images": panel_files}))
    except Exception as exc:
        raise RuntimeError("Cache QC panel generation failed") from exc


if __name__ == "__main__":
    main()
