"""Verify saved original RGB diagnostics and compare the two sampling grids."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    """Hash one small diagnostic artifact."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict[str, Any]:
    """Load saved execution evidence."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def check_npz(spec: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Check all RGB samples and timing arrays without invoking any neural model."""
    try:
        path = Path(spec["path"])
        if sha(path) != spec["sha256"]:
            raise ValueError("NPZ SHA changed")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        rgb = arrays["review_roi_rgb"]
        if rgb.dtype != np.uint8 or rgb.shape != (len(trace),96,96,3):
            raise ValueError("RGB dtype/shape differs")
        if not bool(arrays["review_only"].item()) or bool(arrays["training_ready"].item()) or arrays["split"].item() != "unassigned":
            raise ValueError("Diagnostic scope flags differ")
        if hashlib.sha256(rgb.tobytes()).hexdigest() != spec["rgb_uint8_sha256"]:
            raise ValueError("Full RGB hash differs")
        for k, sample in enumerate(trace):
            if hashlib.sha256(rgb[k].tobytes()).hexdigest() != sample["roi_rgb_sha256"]:
                raise ValueError("Per-frame RGB hash differs")
            if arrays["native_indices"][k] != sample["native_frame_index"] or arrays["target_video_times"][k] != sample["target_video_time"] or arrays["actual_video_pts"][k] != sample["actual_pts_seconds"]:
                raise ValueError("NPZ time/index differs")
        return arrays
    except Exception as exc:
        raise RuntimeError("Saved native diagnostic verification failed") from exc


def main() -> None:
    """Record sampling differences and preserve all earlier failed attempts."""
    try:
        summary_path = BASE / "native_check_summary_r4.json"
        summary = read(summary_path)
        plan_path = BASE / "native_execution_plan_r4.json"
        plan = read(plan_path)
        if sha(plan_path) != summary["plan_sha256"]:
            raise ValueError("Executed plan changed")
        for path, expected in plan["source_hashes"].items():
            if sha(Path(path)) != expected:
                raise ValueError("Frozen code/source changed")
        failures = [read(BASE / f"failed_attempt_r{r}.json") for r in (2,3)]
        results = []
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 13)
        for r in summary["records"]:
            a = check_npz(r["npz"], r["trace"])
            b = check_npz(r["grid_B"]["npz"], r["grid_B"]["trace"])
            ai = {int(idx): k for k,idx in enumerate(a["native_indices"])}
            bi = {int(idx): k for k,idx in enumerate(b["native_indices"])}
            common, aonly, bonly = sorted(ai.keys() & bi.keys()), sorted(ai.keys() - bi.keys()), sorted(bi.keys() - ai.keys())
            if any(not np.array_equal(a["review_roi_rgb"][ai[i]],b["review_roi_rgb"][bi[i]]) for i in common):
                raise ValueError("Common native frame pixels differ between grids")
            panels = [("A first", a, 0), ("B first", b, 0),
                ("A middle", a, len(ai)//2), ("B middle", b, len(bi)//2),
                ("A last", a, len(ai)-1), ("B last", b, len(bi)-1)]
            image = Image.new("RGB", (1000, 255), "white")
            draw = ImageDraw.Draw(image)
            draw.text((8,5),r["parent_id"] + " | " + r["phrase"]["text"],font=font,fill="black")
            draw.text((8,28),"A: 구절 시작 25fps / B: 기존 검수 표시격자 겹침 / 원본 ROI 진단 · 학습승인 아님",font=font,fill="black")
            panel_meta = []
            for col,(role,arr,index) in enumerate(panels):
                x,y = 10 + col*164,65
                image.paste(Image.fromarray(arr["review_roi_rgb"][index]), (x,y))
                draw.text((x,y+100),role,font=font,fill="black")
                draw.text((x,y+118),f'native #{arr["native_indices"][index]}',font=font,fill="black")
                draw.text((x,y+136),f'PTS {arr["actual_video_pts"][index]:.4f}',font=font,fill="black")
                panel_meta.append({"role":role,"native_index":int(arr["native_indices"][index]),
                    "rgb_sha256":hashlib.sha256(arr["review_roi_rgb"][index].tobytes()).hexdigest()})
            out=BASE/"ab_contacts"/(r["parent_id"]+".png")
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                raise ValueError("AB contact already exists")
            image.save(out)
            loaded=Image.open(out).convert("RGB")
            for col,panel in enumerate(panel_meta):
                x,y=10+col*164,65
                if hashlib.sha256(np.asarray(loaded.crop((x,y,x+96,y+96))).tobytes()).hexdigest()!=panel["rgb_sha256"]:
                    raise ValueError("Contact pixels differ")
            results.append({"rank":r["rank"],"parent_id":r["parent_id"],"text":r["phrase"]["text"],
                "A_frames":len(ai),"B_frames":len(bi),"common_native_frames":len(common),
                "A_only_native_indices":aonly,"B_only_native_indices":bonly,
                "A_last_native_pts_to_selected_end_ms":r["last_selected_native_pts_to_target_end_seconds"]*1000,
                "B_last_native_pts_to_selected_end_ms":r["grid_B"]["last_native_pts_to_selected_end_seconds"]*1000,
                "A_first_native_pts_minus_selected_start_ms":(r["trace"][0]["actual_pts_seconds"]-r["trace"][0]["target_video_time"])*1000,
                "B_first_native_pts_minus_selected_start_ms":r["grid_B"]["first_native_pts_minus_selected_start_seconds"]*1000,
                "B_display_start_minus_selected_start_ms":r["grid_B"]["display_span_start_minus_selected_start_seconds"]*1000,
                "B_display_end_minus_selected_end_ms":r["grid_B"]["display_span_end_minus_selected_end_seconds"]*1000,
                "AB_common_frame_pixels_match":True,
                "historical_reference_comparisons":len(r["historical_comparisons"]),
                "unique_historical_raw_indices_checked":len({x["trace"]["native_frame_index"] for x in r["historical_comparisons"]}),
                "contact":{"path":str(out),"sha256":sha(out),"panels":panel_meta}})
        output={"schema":"native-grid-diagnostic-comparison-v1","status":"passed_saved_array_and_grid_comparison",
            "source_hashes":{str(summary_path):sha(summary_path),str(plan_path):sha(plan_path),str(Path(__file__).resolve()):sha(Path(__file__).resolve()),
                **{str(BASE/f"failed_attempt_r{r}.json"):sha(BASE/f"failed_attempt_r{r}.json") for r in (2,3)}},
            "records":results,"A_total_frames":sum(r["A_frames"] for r in results),"B_total_frames":sum(r["B_frames"] for r in results),
            "common_native_frames":sum(r["common_native_frames"] for r in results),
            "A_only_native_frames":sum(len(r["A_only_native_indices"]) for r in results),
            "B_only_native_frames":sum(len(r["B_only_native_indices"]) for r in results),
            "successful_execution_member_bytes":summary["source_member_bytes_read"],
            "failed_attempt_member_bytes":sum(f["source_member_bytes_read"] for f in failures),
            "all_attempts_member_bytes":summary["source_member_bytes_read"]+sum(f["source_member_bytes_read"] for f in failures),
            "all_attempts_member_reads":16+sum(f["source_video_members"]+f["source_label_members"] for f in failures),
            "unique_member_count":16,"unique_archive_count":len(summary["unique_archive_paths"]),
            "temporary_originals_remaining":len(list((BASE/"_tmp").iterdir())),
            "npz_arrays_verified":16,"per_frame_pixels_verified":sum(r["A_frames"]+r["B_frames"] for r in results),
            "new_AB_contact_pixel_panels_verified":48,"sampling_policy_selected":None,
            "review_only":True,"training_ready":False,"split":"unassigned","training_cache_count":0,
            "audio_members_read":0,"neural_calls":0,"training_updates":0,
            "limitations":["A and B use different grids; B is not simply A with appended endpoints.",
                "Native frame endpoint residual is sampling geometry, not a measured phonetic truncation error.",
                "B selects display bins overlapping the target and may include a source-frame timestamp outside its exact audio bounds.",
                "Prior qualitative review does not approve all newly sampled A/B pixels or their training policy."]}
        path=BASE/"grid_comparison.json"
        if path.exists():
            raise ValueError("Comparison already exists")
        path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in output.items() if not isinstance(v,(dict,list))}))
    except Exception as exc:
        raise RuntimeError("Post-extraction grid diagnostic failed") from exc


if __name__=="__main__":
    main()
