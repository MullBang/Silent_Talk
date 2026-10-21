"""Read only top-level MP4 box headers to screen stage18 new frontal sources.

A structural pass is NOT successful decoding, label alignment or ROI validation.
No media payload, audio or Validation source is inspected by this command.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from models.dataset_manifest import sha256_file


def probe(path: Path) -> dict:
    """Bounds-check ISO BMFF box lengths without reading box payloads."""
    size = path.stat().st_size
    offset, bytes_read, boxes = 0, 0, []
    result = {"size_bytes": size, "boxes": boxes, "payload_decoded": False}
    with path.open("rb") as stream:
        while offset < size:
            if len(boxes) >= 10000:
                return {**result, "status": "too_many_top_level_boxes", "header_bytes_read": bytes_read}
            stream.seek(offset)
            header = stream.read(8)
            bytes_read += len(header)
            if len(header) != 8:
                return {**result, "status": "truncated_box_header", "offset": offset, "header_bytes_read": bytes_read}
            length, kind = struct.unpack(">I4s", header)
            header_length = 8
            if length == 1:
                extended = stream.read(8)
                bytes_read += len(extended)
                if len(extended) != 8:
                    return {**result, "status": "truncated_extended_header", "offset": offset, "header_bytes_read": bytes_read}
                length = struct.unpack(">Q", extended)[0]
                header_length = 16
            elif length == 0:
                length = size - offset
            box = {"type": kind.decode("ascii", errors="replace"), "offset": offset,
                   "declared_size": length, "declared_end": offset + length}
            boxes.append(box)
            if length < header_length:
                return {**result, "status": "invalid_box_size", "header_bytes_read": bytes_read}
            if offset + length > size:
                return {**result, "status": "box_extends_past_eof", "minimum_expected_bytes": offset + length,
                        "minimum_missing_bytes": offset + length - size, "header_bytes_read": bytes_read}
            offset += length
    types = [b["type"] for b in boxes]
    if not types or types[0] != "ftyp":
        status = "missing_initial_ftyp"
    elif "moov" not in types:
        status = "missing_moov"
    elif "mdat" not in types:
        status = "missing_mdat"
    else:
        status = "structural_screen_pass"
    return {**result, "status": status, "header_bytes_read": bytes_read}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/step18-intake/run01")
    args = parser.parse_args()
    out = args.run.resolve()
    if not out.is_relative_to(ROOT / "results/step18-intake"):
        raise ValueError("Stage18 output required")
    index = json.loads((out / "source_index.json").read_text(encoding="utf-8"))
    results, started = [], time.perf_counter()
    for number, source in enumerate(index["sources"], 1):
        video = source["video"]
        path = Path(video["path"])
        before = path.stat()
        if (before.st_size, before.st_mtime_ns) != (video["size_bytes"], video["mtime_ns"]):
            result = {"status": "metadata_changed_since_inventory", "size_bytes": before.st_size}
        else:
            try:
                result = probe(path)
            except OSError as exc:
                result = {"status": "read_error", "error": str(exc)}
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                result["status_before_change"] = result["status"]
                result["status"] = "metadata_changed_during_probe"
        results.append({"stem": source["stem"], "source_path": str(path), "package": video["package"],
                        "speaker_id": video["speaker_id"], "recording_id": video["recording_id"], **result})
        print(f"[container {number}/{len(index['sources'])}] {source['stem']}: {result['status']}", flush=True)
    grouped = {}
    for field in ("package", "speaker_id"):
        values = defaultdict(Counter)
        for result in results:
            values[result[field]][result["status"]] += 1
        grouped[field] = {k: dict(v) for k, v in sorted(values.items())}
    report = {"schema": "lipnet-intake-container-screen-v1", "source_index_sha256": sha256_file(out / "source_index.json"),
        "inventory_sha256": sha256_file(out / "inventory.json"), "probe_script_sha256": sha256_file(Path(__file__)),
        "scope": "new Training speakers, frontal A only; top-level headers, not decoding",
        "count": len(results), "counts": dict(Counter(r["status"] for r in results)), "by": grouped,
        "header_bytes_read": sum(r.get("header_bytes_read", 0) for r in results),
        "elapsed_seconds": time.perf_counter()-started, "records": results}
    with (out / "container_probe.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in ("count", "counts", "by", "header_bytes_read")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
