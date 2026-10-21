"""고정 명세의 모든 ROI에서 처음/중간/끝을 검수용 모음 이미지로 저장한다."""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.cache import load_clip
from models.dataset_manifest import load_manifest, sha256_file
from models.tokenizer import ctc_min_input_len
from build_speaker_dataset import write_json
from expand_speaker_dataset import assert_parent_preserved


def review_font(size: int):
    for name in ("C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def inspect(manifest: Path) -> dict:
    document, validation = load_manifest(manifest)
    out = manifest.parent / "visual_qa"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("검수 폴더가 이미 있습니다. 기존 검수 기록을 덮지 않습니다.")
    out.mkdir(exist_ok=True)
    parent_path = document["policy"].get("frozen_parent_manifest")
    parent_ids = set()
    if parent_path:
        if sha256_file(parent_path) != document["policy"]["frozen_parent_sha256"]:
            raise ValueError("부모 명세가 변경되었습니다.")
        parent, _ = load_manifest(parent_path)
        assert_parent_preserved(document, parent, check_hash=True)
        parent_ids = {r["id"] for r in parent["records"]}
    # 새 train만 영상 검수. 보존 표본은 부모에서 검수했고 해시가 동일함을 별도 검사한다.
    rows = [r for r in document["records"] if r["id"] not in parent_ids]
    if not rows:
        rows = document["records"]
    font = review_font(14)
    heading = review_font(20)
    metrics, sheets = [], []
    for offset in range(0, len(rows), 40):
        batch = rows[offset:offset + 40]
        sheet = Image.new("RGB", (1500, 1094), "#f7f8fa")
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 8), f"Lip ROI review | {offset + 1}-{offset + len(batch)} / {len(rows)} new clips | first / middle / last", font=heading, fill="#13263f")
        for i, row in enumerate(batch):
            clip = load_clip(manifest.parent / row["cache_path"])
            frames = clip["frames"]
            x, y = (i % 5) * 300 + 5, (i // 5) * 132 + 38
            video_no = row["recording_id"].rsplit("_", 1)[-1]
            label = f"{row['speaker_id']}/v{video_no}/s{row['sentence_id']:03d} T={len(frames)}"
            draw.text((x, y), label, font=font, fill="#13263f")
            for j, frame in enumerate((frames[0], frames[len(frames)//2], frames[-1])):
                sheet.paste(Image.fromarray(frame), (x + 98*j, y + 21))
            pixels = frames.astype(np.float32)
            gray = pixels.mean(axis=3)
            metric = {"id": row["id"], "frames": len(frames),
                      "minimum_frame_spatial_std": float(gray.std(axis=(1, 2)).min()),
                      "mean_absolute_frame_change": float(np.abs(np.diff(pixels, axis=0)).mean()),
                      "near_black_fraction": float((frames <= 3).all(axis=3).mean()),
                      "near_white_fraction": float((frames >= 252).all(axis=3).mean()),
                      "ctc_slack_frames": int(len(frames) - ctc_min_input_len(clip["label"]))}
            metric["review_flags"] = []
            if metric["minimum_frame_spatial_std"] < 3:
                metric["review_flags"].append("low_spatial_variance")
            if metric["mean_absolute_frame_change"] < 0.1:
                metric["review_flags"].append("almost_static")
            if max(metric["near_black_fraction"], metric["near_white_fraction"]) > 0.5:
                metric["review_flags"].append("mostly_black_or_white")
            metrics.append(metric)
        path = out / f"roi_samples_{offset // 40 + 1:02d}.png"
        sheet.save(path)
        sheets.append({"path": path.name, "sha256": sha256_file(path), "clip_ids": [r["id"] for r in batch]})
        print(f"[검수 이미지] {offset + len(batch)}/{len(rows)}", flush=True)
    report = {"manifest_sha256": validation["manifest_sha256"], "new_clips_reviewable": len(rows),
              "thumbnails_per_clip": 3, "thumbnail_count": len(rows) * 3,
              "sheet_count": len(sheets), "sheets": sheets,
              "automatic_flag_count": sum(bool(r["review_flags"]) for r in metrics),
              "automatic_checks_are_not_recognition_accuracy": True,
              "manual_review_status": "pending", "metrics": metrics,
              "new_clips_per_speaker": dict(Counter(r["speaker_id"] for r in rows))}
    write_json(out / "qa_inventory.json", report)
    return report


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data_cache/datasets/aihub_expanded_v3/manifest.json")
    inspect(Path(parser.parse_args().manifest).resolve())
