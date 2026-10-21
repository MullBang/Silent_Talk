"""Verify completed local-only input cohort and make a paired visual QA sheet."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_inputs import BASE, SOURCE_PLAN, ALIGNMENT, anchor, save_new_json
from models.cache import load_clip


def main() -> None:
    manifest_path = BASE / "inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    offsets = {r["source_name"]: r["label_to_video_offset_seconds"] for r in alignment["sources"]}
    references = {r["id"]: r for r in source["records"]}
    assert manifest["status"] == "completed" and len(manifest["records"]) == 40
    assert manifest["reproduction_gate"] == {"passed": True, "records": 20, "pixel_exact": 20, "label_exact": 20, "text_exact": 20}
    data = {(r["id"], r["arm"]): r for r in manifest["records"]}
    expected_keys = {(sample, arm) for sample in references for arm in ("old_timing", "aligned_timing")}
    assert set(data) == expected_keys and len(data) == len(manifest["records"])
    checked_frames = 0
    max_pts_residual = 0.
    frame_totals = {"old_timing": 0, "aligned_timing": 0}
    pair_middles = []
    for sample, ref in references.items():
        images = []
        for arm in ("old_timing", "aligned_timing"):
            row = data[(sample, arm)]
            anchor(Path(row["npz_path"]), row["npz_sha256"])
            anchor(Path(row["selected_trace_path"]), row["selected_trace_sha256"])
            clip = load_clip(row["npz_path"])
            pixels = clip["frames"]
            assert clip["text"] == row["reference"] == ref["text"]
            assert clip["source_text"] == ref["source_text"]
            assert pixels.shape == (row["frame_count"], 96, 96, 3) and pixels.dtype == np.uint8
            assert len(pixels) == ref["frames"]
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"]
            assert hashlib.sha256(clip["label"].astype("<i8").tobytes()).hexdigest() == row["label_tokens_sha256"]
            offset = 0. if arm == "old_timing" else offsets[Path(row["source_video"]["path"]).name]
            assert row["label_to_video_offset_seconds"] == offset
            assert row["video_start_time"] == ref["start_time"] + offset
            assert row["video_end_time"] == ref["end_time"] + offset
            trace = json.loads(Path(row["selected_trace_path"]).read_text(encoding="utf-8"))
            assert len(trace["selected"]) == len(pixels)
            for k, selection in enumerate(trace["selected"]):
                target = ref["start_time"] + offset + k / 25
                index = round(target * row["native_fps"])
                assert selection["source_frame_index"] == selection["annotation_box_index"] == index
                assert selection["target_video_seconds"] == target
                residual = abs(selection["actual_pts_seconds"] - index / row["native_fps"])
                assert residual <= 1e-3
                assert abs(selection["actual_pts_seconds"] - target) <= .5 / row["native_fps"] + 1e-6
                max_pts_residual = max(max_pts_residual, residual)
                checked_frames += 1
            frame_totals[arm] += len(pixels)
            images.append(Image.fromarray(pixels[len(pixels) // 2], mode="RGB").resize((144, 144)))
        pair_middles.append((sample, images))
    # Each paired cell uses RGB arrays directly. Frames can legitimately differ
    # because the clock intervention selects different video moments.
    sheet = Image.new("RGB", (960, 7 * 186), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (sample, images) in enumerate(pair_middles):
        x, y = (n % 3) * 320, (n // 3) * 186
        draw.text((x + 5, y + 3), sample.replace("lip_J_1_F_", ""), fill="black")
        draw.text((x + 5, y + 18), "old timing", fill="black")
        draw.text((x + 157, y + 18), "aligned timing", fill="black")
        sheet.paste(images[0], (x + 5, y + 35))
        sheet.paste(images[1], (x + 157, y + 35))
    contact = BASE / "inputs/contact_sheet.png"
    if contact.exists():
        raise FileExistsError(contact)
    sheet.save(contact)
    qa = {"schema": "legacy-step05-input-qa-v1", "status": "pass", "manifest": anchor(manifest_path),
          "records": 40, "ids": 20, "reference_characters_per_arm": sum(len(r["text"]) for r in references.values()),
          "reference_words_per_arm": sum(len(r["text"].split()) for r in references.values()),
          "selected_frames_verified": checked_frames, "frames_per_arm": frame_totals,
          "maximum_native_pts_residual_seconds": max_pts_residual, "old_timing_pixel_reproduction_exact": 20,
          "same_IDs_GT_tokens_and_lengths_across_arms": True, "local_only": True,
          "test_payloads_read": False, "training": False, "contact_sheet": anchor(contact),
          "total_local_new_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in data.values() if r["arm"] == "aligned_timing")}
    save_new_json(BASE / "inputs/input_qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
