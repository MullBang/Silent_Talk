"""Read-only, end-to-end software decoding of the project's frontal training sources.

Archives stay intact. At most --workers selected members are copied to a private
workspace temporary directory and removed after decoding. This is a media audit,
not a transcript/alignment or model-performance evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = Path(r"D:\009.립리딩(입모양) 음성인식 데이터")
FFMPEG = ROOT / "experiments/olkavs_official/audio_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snap(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def create_plan() -> dict:
    priority = set()
    inventory = json.loads((ROOT / "experiments/training_alignment_step06/training_inventory.json").read_text("utf-8"))
    priority.update(s["source_name"] for s in inventory["sources"])
    intake = json.loads((ROOT / "results/step18-intake/run01/selection.json").read_text("utf-8"))
    priority.update(Path(s["source_video"]["path"]).name for s in intake["records"])
    records = []
    seen = set()
    for archive in sorted(DATA.rglob("TS*.tar")):
        before = snap(archive)
        with tarfile.open(archive, "r:") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                ext = Path(name).suffix.lower()
                if ext != ".wav" and not (ext == ".mp4" and Path(name).stem.split("_")[-2] == "A"):
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate selected basename: {name}")
                seen.add(name)
                if member.offset_data + member.size > before["size"]:
                    raise ValueError(f"Selected member exceeds archive: {name}")
                records.append({"name": name, "package": archive.stem, "archive": str(archive),
                                "archive_stat": before, "member": member.name, "offset": member.offset_data,
                                "size": member.size, "priority_existing_or_pilot": name in priority})
        if snap(archive) != before:
            raise ValueError(f"Archive changed during indexing: {archive}")
    records.sort(key=lambda r: (not r["priority_existing_or_pilot"], Path(r["name"]).suffix != ".mp4", r["package"], r["name"]))
    return {"schema": "lipnet-full-source-decode-plan-v1", "scope": "All frontal A MP4 and all WAV in five D: Training TARs; other video angles receive structural checks separately.",
            "priority_names": len(priority), "records": records, "script_sha256": sha_file(Path(__file__)),
            "ffmpeg_sha256": sha_file(FFMPEG), "read_only_sources": True, "training": False}


def decode_one(record: dict, out: Path) -> dict:
    started = time.perf_counter()
    result = {**record, "status": "not_checked"}
    try:
        archive = Path(record["archive"])
        if snap(archive) != record["archive_stat"]:
            raise ValueError("archive_changed_since_plan")
        if shutil.disk_usage(out).free < 8 * 1024**3:
            raise ValueError("workspace_free_space_below_8GiB")
        with tempfile.TemporaryDirectory(prefix="decode_", dir=out / "tmp") as temp_dir:
            temp = Path(temp_dir) / record["name"]
            digest = hashlib.sha256()
            remaining = record["size"]
            with archive.open("rb") as source, temp.open("xb") as target:
                source.seek(record["offset"])
                while remaining:
                    chunk = source.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError("member_truncated_during_copy")
                    target.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            result["member_sha256"] = digest.hexdigest()
            if snap(archive) != record["archive_stat"]:
                raise ValueError("archive_changed_during_copy")
            metadata = {}
            if temp.suffix == ".mp4":
                import cv2
                cap = cv2.VideoCapture(str(temp))
                try:
                    if not cap.isOpened():
                        raise ValueError("opencv_cannot_open")
                    metadata = {"frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
                    if not math.isfinite(metadata["fps"]) or metadata["fps"] <= 0 or metadata["frame_count"] <= 0:
                        raise ValueError("invalid_video_metadata")
                finally:
                    cap.release()
                result["video_metadata"] = metadata
            maps = ["-map", "0:v:0", "-map", "0:a:0?", "-fps_mode", "passthrough"] if temp.suffix == ".mp4" else ["-map", "0:a:0"]
            command = [str(FFMPEG), "-hide_banner", "-v", "warning", "-xerror", "-err_detect", "explode", "-nostdin",
                       "-threads", "2", "-i", str(temp), *maps, "-max_error_rate", "0", "-f", "null", "-progress", "pipe:1", "-nostats", "NUL"]
            decoded = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
            fields = {}
            for line in decoded.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    fields[key] = value
            result.update({"returncode": decoded.returncode, "progress": fields,
                           "decoder_messages": decoded.stderr[-16000:], "decoder_messages_total_chars": len(decoded.stderr)})
            (out / "logs" / (record["name"] + ".log")).write_text(decoded.stderr, encoding="utf-8")
            if decoded.returncode != 0 or fields.get("progress") != "end":
                result["status"] = "decode_fail"
            elif metadata and int(fields.get("frame", "0")) != metadata["frame_count"]:
                result["status"] = "decoded_frame_count_mismatch"
            elif int(fields.get("out_time_us", "0")) <= 0:
                result["status"] = "empty_decode"
            else:
                result["status"] = "full_decode_pass"
    except Exception as exc:
        result.update({"status": "audit_error", "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("workers must be 1..6")
    out = BASE / "decode_results"
    for child in ("records", "logs", "tmp"):
        (out / child).mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text("utf-8"))
        if plan["script_sha256"] != sha_file(Path(__file__)) or plan["ffmpeg_sha256"] != sha_file(FFMPEG):
            raise ValueError("audit implementation changed; use a separately named audit run")
    else:
        plan = create_plan()
        save(plan_path, plan)
    print(json.dumps({"selected": len(plan["records"]), "by_extension": dict(Counter(Path(r["name"]).suffix for r in plan["records"])), "priority": plan["priority_names"]}), flush=True)
    if args.plan_only:
        return
    started = time.perf_counter()
    results, pending = [], []
    for record in plan["records"]:
        path = out / "records" / (record["name"] + ".json")
        if path.exists():
            previous = json.loads(path.read_text("utf-8"))
            if any(previous.get(k) != v for k, v in record.items()) or snap(Path(record["archive"])) != record["archive_stat"]:
                raise ValueError("saved record/source differs from plan")
            results.append(previous)
        else:
            pending.append(record)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(decode_one, record, out) for record in pending]
        for future in as_completed(futures):
            result = future.result()
            save(out / "records" / (result["name"] + ".json"), result)
            results.append(result)
            if len(results) % 10 == 0 or result["status"] != "full_decode_pass":
                print(json.dumps({"done": len(results), "total": len(plan["records"]), "counts": dict(Counter(r["status"] for r in results)), "elapsed_s": round(time.perf_counter()-started)}), flush=True)
    summary = {"schema": "lipnet-full-source-decode-summary-v1", "plan_sha256": sha_file(plan_path), "scope": plan["scope"],
               "count": len(results), "counts": dict(Counter(r["status"] for r in results)),
               "by_package": {p: dict(Counter(r["status"] for r in results if r["package"] == p)) for p in sorted({r["package"] for r in results})},
               "by_extension": {e: dict(Counter(r["status"] for r in results if Path(r["name"]).suffix == e)) for e in (".mp4", ".wav")},
               "warning_files": [r["name"] for r in results if r.get("decoder_messages")],
               "failures": [r for r in results if r["status"] != "full_decode_pass"],
               "elapsed_this_invocation_s": time.perf_counter()-started,
               "limitations": ["No vendor checksum available: hashes fingerprint local member bytes only.", "Successful decoding does not prove transcript correctness, A/V synchronization, ROI semantic correctness or model accuracy.", "Nonfrontal MP4 payloads are not fully decoded in this run."]}
    save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
