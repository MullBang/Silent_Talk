"""Read-only SHA-256 equality audit of loose media and their TAR members.

This proves local extraction-copy equality only. It does not compare against an
AIHub publisher checksum, validate codecs, or prove annotation correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone


CHUNK = 8 * 1024 * 1024
ROOT = Path(r"D:\009.립리딩(입모양) 음성인식 데이터")
OUT = Path(__file__).resolve().parent / "copy_results"


def utc():
    return datetime.now(timezone.utc).isoformat()


def snapshot(path):
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "dev": st.st_dev, "ino": st.st_ino}


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def hash_exact(handle, size):
    digest = hashlib.sha256()
    left = size
    while left:
        block = handle.read(min(CHUNK, left))
        if not block:
            raise EOFError(f"Unexpected EOF with {left} bytes left")
        digest.update(block)
        left -= len(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    if root == out or root in out.parents:
        raise ValueError("Audit outputs must not be inside the original data directory")
    out.mkdir(parents=True, exist_ok=True)
    records_dir = out / "records"
    records_dir.mkdir(exist_ok=True)
    started = utc()
    started_clock = time.monotonic()
    loose = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".wav"})
    archives = sorted(root.rglob("*.tar"))
    by_name = defaultdict(list)
    archive_status = []
    targets = {p.name for p in loose}
    for archive in archives:
        before = snapshot(archive)
        count = 0
        error = None
        try:
            with tarfile.open(archive, "r:") as tf:
                for member in tf:
                    count += 1
                    name = PurePosixPath(member.name).name
                    if name not in targets or not member.isfile():
                        continue
                    if member.offset_data < 0 or member.offset_data + member.size > before["size"]:
                        raise ValueError(f"Member extends past archive EOF: {member.name}")
                    by_name[name].append({"archive": str(archive), "member": member.name,
                        "offset": member.offset_data, "size": member.size,
                        "archive_snapshot": before})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        after = snapshot(archive)
        archive_status.append({"archive": str(archive), "before": before, "after": after,
                               "member_count": count, "error": error,
                               "stable": before == after})
        print(json.dumps({"event": "indexed", "archive": archive.name,
                          "members": count, "error": error}, ensure_ascii=False), flush=True)
    archive_map = {r["archive"]: r for r in archive_status}
    jobs = []
    for path in loose:
        candidates = by_name[path.name]
        jobs.append((path, candidates))
    jobs.sort(key=lambda job: (job[1][0]["archive"], job[1][0]["offset"])
              if len(job[1]) == 1 else ("", str(job[0])))
    if args.max_files:
        jobs = jobs[:args.max_files]
    save(out / "index.json", {"created_utc": utc(), "root": str(root),
         "loose_count": len(loose), "loose_bytes": sum(p.stat().st_size for p in loose),
         "archives": archive_status,
         "candidate_counts": dict(Counter(len(candidates) for _, candidates in jobs))})
    results = []
    total_bytes_read = 0
    reused = 0
    for number, (path, candidates) in enumerate(jobs, 1):
        key = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
        record_path = records_dir / f"{key}.json"
        before = snapshot(path)
        result = {"loose_file": str(path), "name": path.name, "loose_before": before,
                  "candidate_count": len(candidates), "candidates": candidates,
                  "started_utc": utc()}
        cached = None
        if record_path.exists() and len(candidates) == 1:
            try:
                prev = json.loads(record_path.read_text(encoding="utf-8"))
                if (prev.get("status") in {"match", "hash_mismatch"}
                    and prev.get("loose_before") == before == prev.get("loose_after")
                    and prev.get("candidates") == candidates
                    and prev.get("archive_before") == snapshot(Path(candidates[0]["archive"]))
                    == prev.get("archive_after")):
                    cached = prev
            except (OSError, ValueError):
                pass
        if cached is not None:
            result = cached
            reused += 1
        else:
            try:
                if not candidates:
                    result["status"] = "no_matching_member"
                elif len(candidates) != 1:
                    result["status"] = "ambiguous_matching_members"
                else:
                    candidate = candidates[0]
                    arc = Path(candidate["archive"])
                    result["archive_before"] = snapshot(arc)
                    if not archive_map[str(arc)]["stable"] or archive_map[str(arc)]["error"]:
                        result["status"] = "archive_index_invalid"
                    elif result["archive_before"] != candidate["archive_snapshot"]:
                        result["status"] = "archive_changed_after_index"
                    elif before["size"] != candidate["size"]:
                        result["status"] = "size_mismatch"
                    else:
                        with arc.open("rb", buffering=0) as stream:
                            stream.seek(candidate["offset"])
                            result["archive_member_sha256"] = hash_exact(stream, candidate["size"])
                        total_bytes_read += candidate["size"]
                        with path.open("rb", buffering=0) as stream:
                            result["loose_sha256"] = hash_exact(stream, before["size"])
                            if stream.read(1):
                                raise ValueError("Loose media grew during hashing")
                        total_bytes_read += before["size"]
                        result["status"] = ("match" if result["archive_member_sha256"] ==
                                            result["loose_sha256"] else "hash_mismatch")
                    result["archive_after"] = snapshot(arc)
                result["loose_after"] = snapshot(path)
                if result["loose_after"] != before:
                    result["status"] = "loose_changed_during_check"
                if "archive_before" in result and result["archive_before"] != result["archive_after"]:
                    result["status"] = "archive_changed_during_check"
            except Exception as exc:
                result["status"] = "read_error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["finished_utc"] = utc()
            save(record_path, result)
        results.append(result)
        progress = {"event": "progress", "completed": number, "planned": len(jobs),
                    "status_counts": dict(Counter(r["status"] for r in results)),
                    "bytes_read_this_run": total_bytes_read, "resumed_records": reused,
                    "elapsed_seconds": round(time.monotonic() - started_clock, 2),
                    "last_file": path.name, "updated_utc": utc()}
        save(out / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    changed_archives = []
    for row in archive_status:
        row["at_audit_end"] = snapshot(Path(row["archive"]))
        if row["at_audit_end"] != row["before"]:
            changed_archives.append(row["archive"])
    for result in results:
        if any(c["archive"] in changed_archives for c in result["candidates"]):
            result["status_before_archive_change"] = result["status"]
            result["status"] = "archive_changed_during_audit"
    save(out / "details.json", results)
    summary = {"started_utc": started, "finished_utc": utc(),
               "root": str(root), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "Every loose MP4/WAV under root, compared against uniquely named regular TAR member",
               "proof_limit": "Local extraction-copy equality only; no publisher checksum or codec/label validation",
               "loose_inventory_count": len(loose), "checked_count": len(results),
               "full_inventory_covered": len(results) == len(loose),
               "status_counts": dict(Counter(r["status"] for r in results)),
               "checked_loose_bytes": sum(r["loose_before"]["size"] for r in results),
               "bytes_read_this_run": total_bytes_read, "resumed_records": reused,
               "elapsed_seconds": round(time.monotonic() - started_clock, 2),
               "archives": archive_status, "changed_archives": changed_archives,
               "all_match": len(results) == len(loose) and all(r["status"] == "match" for r in results)}
    save(out / "summary.json", summary)
    print(json.dumps({"event": "complete", "all_match": summary["all_match"],
                      "status_counts": summary["status_counts"]}, ensure_ascii=False), flush=True)
    return 0 if summary["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
