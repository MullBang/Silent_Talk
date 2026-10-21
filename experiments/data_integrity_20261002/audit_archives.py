"""Read-only TAR and media-container structure audit; no payload decoding.

Physical TAR headers (including GNU/PAX auxiliary headers) are checksum and
bounds checked. tarfile separately resolves logical member names. Media probes
only read bounded top-level headers, never data outside an indexed member.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import struct
import tarfile
import time

HERE = Path(__file__).resolve().parent
ZERO = bytes(512)


def snapshot(path):
    s = path.stat()
    return {"size": s.st_size, "mtime_ns": s.st_mtime_ns}


def unsafe_name(name):
    p = PurePosixPath(name.replace("\\", "/"))
    w = PureWindowsPath(name)
    return p.is_absolute() or bool(w.drive) or ".." in p.parts or "\x00" in name


def bounded_read(stream, base, size, offset, count):
    if offset < 0 or count < 0 or offset + count > size:
        return b""
    stream.seek(base + offset)
    return stream.read(count)


def mp4_probe(stream, base, size):
    boxes, pos, read = [], 0, 0
    def result(status, **extra):
        return {"status": status, "boxes": boxes, "header_bytes_read": read,
                "payload_decoded": False, **extra}
    while pos < size:
        if len(boxes) >= 10000:
            return result("too_many_top_level_boxes")
        raw = bounded_read(stream, base, size, pos, 8)
        read += len(raw)
        if len(raw) != 8:
            return result("truncated_box_header", offset=pos)
        length, kind = struct.unpack(">I4s", raw)
        header = 8
        if length == 1:
            raw = bounded_read(stream, base, size, pos + 8, 8)
            read += len(raw)
            if len(raw) != 8:
                return result("truncated_extended_header", offset=pos)
            length, header = struct.unpack(">Q", raw)[0], 16
        elif length == 0:
            length = size - pos
        boxes.append({"type": kind.decode("ascii", "replace"), "offset": pos,
                      "size": length, "end": pos + length})
        if length < header:
            return result("invalid_box_size")
        if pos + length > size:
            return result("box_extends_past_eof", missing_bytes=pos + length - size)
        pos += length
    kinds = [b["type"] for b in boxes]
    if not kinds or kinds[0] != "ftyp":
        return result("missing_initial_ftyp")
    for needed in ("moov", "mdat"):
        if needed not in kinds:
            return result("missing_" + needed)
    return result("structural_screen_pass")


def wav_probe(stream, base, size):
    chunks, fmt, pos, read = [], None, 12, 0
    def result(status, **extra):
        return {"status": status, "chunks": chunks, "format": fmt,
                "header_bytes_read": read, "payload_decoded": False, **extra}
    raw = bounded_read(stream, base, size, 0, 12)
    read += len(raw)
    if len(raw) != 12:
        return result("truncated_riff_header")
    if raw[:4] != b"RIFF" or raw[8:] != b"WAVE":
        return result("unsupported_or_invalid_wave_header", signature=raw.hex())
    riff_end = struct.unpack("<I", raw[4:8])[0] + 8
    if riff_end > size:
        return result("riff_extends_past_eof", missing_bytes=riff_end - size)
    if riff_end < 12:
        return result("invalid_riff_size")
    while pos < riff_end:
        if len(chunks) >= 10000:
            return result("too_many_wave_chunks")
        raw = bounded_read(stream, base, riff_end, pos, 8)
        read += len(raw)
        if len(raw) != 8:
            return result("truncated_wave_chunk_header", offset=pos)
        kind, length = struct.unpack("<4sI", raw)
        end = pos + 8 + length
        chunks.append({"type": kind.decode("ascii", "replace"), "offset": pos,
                       "size": length, "end": end})
        if end > riff_end:
            return result("wave_chunk_extends_past_riff")
        if kind == b"fmt ":
            if length < 16:
                return result("short_fmt_chunk")
            raw = bounded_read(stream, base, size, pos + 8, 16)
            read += len(raw)
            if len(raw) != 16:
                return result("truncated_fmt_chunk")
            fields = struct.unpack("<HHIIHH", raw)
            fmt = dict(zip(("format_tag", "channels", "sample_rate", "byte_rate",
                            "block_align", "bits_per_sample"), fields))
            if any(fmt[k] <= 0 for k in ("channels", "sample_rate", "byte_rate", "block_align")):
                return result("invalid_fmt_values")
            if fmt["format_tag"] == 1:
                if fmt["bits_per_sample"] <= 0 or fmt["bits_per_sample"] % 8:
                    return result("invalid_pcm_bit_depth")
                if fmt["block_align"] != fmt["channels"] * fmt["bits_per_sample"] // 8:
                    return result("pcm_block_align_mismatch")
                if fmt["byte_rate"] != fmt["sample_rate"] * fmt["block_align"]:
                    return result("pcm_byte_rate_mismatch")
        pos = end + length % 2
        if pos > riff_end:
            return result("wave_padding_extends_past_riff")
    if fmt is None:
        return result("missing_fmt_chunk")
    data = [c for c in chunks if c["type"] == "data"]
    if not data:
        return result("missing_data_chunk")
    if any(c["size"] == 0 for c in data):
        return result("empty_data_chunk")
    if fmt["format_tag"] == 1 and any(c["size"] % fmt["block_align"] for c in data):
        return result("partial_pcm_frame")
    if riff_end != size:
        return result("trailing_bytes_after_riff", trailing_bytes=size-riff_end)
    return result("structural_screen_pass")


def physical_scan(stream, size):
    pos, headers, payload_bytes = 0, 0, 0
    issues, special = [], Counter()
    terminated = False
    trailer_bytes = 0
    while pos < size:
        stream.seek(pos)
        block = stream.read(512)
        if len(block) != 512:
            issues.append({"code": "truncated_tar_header", "offset": pos})
            break
        if block == ZERO:
            terminated = True
            stream.seek(pos)
            while True:
                tail = stream.read(1024 * 1024)
                if not tail:
                    break
                if tail.strip(b"\x00"):
                    issues.append({"code": "nonzero_after_end_marker", "offset_at_or_after": pos})
                    break
                trailer_bytes += len(tail)
            if trailer_bytes < 1024:
                issues.append({"code": "missing_second_zero_end_block", "trailer_bytes": trailer_bytes})
            break
        try:
            item = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        except tarfile.HeaderError as exc:
            issues.append({"code": "invalid_tar_header_or_checksum", "offset": pos, "error": str(exc)})
            break
        headers += 1
        if item.type in (tarfile.GNUTYPE_SPARSE,):
            issues.append({"code": "unsupported_sparse_tar_member", "offset": pos, "name": item.name})
            break
        if item.size < 0:
            issues.append({"code": "negative_tar_size", "offset": pos})
            break
        payload_end = pos + 512 + item.size
        next_pos = pos + 512 + ((item.size + 511) // 512) * 512
        if payload_end > size or next_pos > size:
            issues.append({"code": "tar_member_extends_past_eof", "offset": pos,
                           "name": item.name, "declared_size": item.size,
                           "minimum_missing_bytes": max(payload_end, next_pos) - size})
            break
        if item.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK,
                         tarfile.XHDTYPE, tarfile.XGLTYPE):
            special[item.type.decode("ascii")] += 1
        payload_bytes += item.size
        pos = next_pos
    if not terminated:
        issues.append({"code": "missing_tar_end_marker"})
    if size % 512:
        issues.append({"code": "tar_size_not_multiple_of_block", "remainder": size % 512})
    return {"status": "structure_pass" if not issues else "structure_fail",
            "physical_headers_checked": headers, "special_headers": dict(special),
            "declared_payload_bytes": payload_bytes, "trailer_bytes": trailer_bytes,
            "issues": issues, "payloads_read_or_hashed": False}


def audit_one(path):
    before = snapshot(path)
    start = time.perf_counter()
    members, probes, issues, seen = [], [], [], set()
    with path.open("rb") as stream:
        physical = physical_scan(stream, before["size"])
        stream.seek(0)
        try:
            with tarfile.open(fileobj=stream, mode="r:", encoding="utf-8", errors="surrogateescape") as archive:
                for member in archive:
                    row = {"archive_path": str(path), "archive_name": path.name,
                           "member_name": member.name, "offset_header": member.offset,
                           "offset_data": member.offset_data, "size": member.size,
                           "type": member.type.decode("ascii", "replace"),
                           "is_file": member.isfile(), "is_dir": member.isdir()}
                    members.append(row)
                    if member.name in seen:
                        issues.append({"code": "duplicate_member_name", "member_name": member.name})
                    seen.add(member.name)
                    if unsafe_name(member.name):
                        issues.append({"code": "unsafe_member_name", "member_name": member.name})
                    if member.issym() or member.islnk():
                        issues.append({"code": "link_member", "member_name": member.name,
                                       "linkname": member.linkname})
                    if not member.isfile():
                        continue
                    if member.size == 0:
                        issues.append({"code": "empty_regular_file", "member_name": member.name})
                    if member.offset_data < 0 or member.size < 0 or member.offset_data + member.size > before["size"]:
                        issues.append({"code": "logical_member_extends_past_eof", "member_name": member.name})
                        continue
                    suffix = PurePosixPath(member.name).suffix.lower()
                    if suffix in (".mp4", ".wav"):
                        probe = mp4_probe if suffix == ".mp4" else wav_probe
                        # tarfile seeks on its next iteration, so a separate bounded
                        # probe on the same uncompressed stream is safe.
                        media = probe(stream, member.offset_data, member.size)
                        probes.append({**row, "media_type": suffix[1:], **media})
        except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
            issues.append({"code": "logical_archive_read_error", "error": str(exc)})
    after = snapshot(path)
    if after != before:
        issues.append({"code": "archive_changed_during_read", "before": before, "after": after})
    summary = {"archive_path": str(path), "archive_name": path.name, "before": before,
               "after": after, "physical": physical,
               "status": "structure_pass" if not issues and physical["status"] == "structure_pass" else "structure_fail",
               "member_count": len(members), "file_count": sum(x["is_file"] for x in members),
               "extensions": dict(Counter(PurePosixPath(x["member_name"]).suffix.lower() for x in members if x["is_file"])),
               "media_status_counts": {kind: dict(Counter(x["status"] for x in probes if x["media_type"] == kind)) for kind in ("mp4", "wav")},
               "issues": issues, "elapsed_seconds": time.perf_counter() - start}
    return summary, members, probes


def save(path, value):
    # Exclusive creation avoids silently replacing prior audit evidence.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def self_test():
    def box(kind, payload=b""):
        return struct.pack(">I4s", 8 + len(payload), kind) + payload
    good = box(b"ftyp", b"isom") + box(b"mdat", b"abc") + box(b"moov")
    assert mp4_probe(io.BytesIO(b"prefix" + good + b"suffix"), 6, len(good))["status"] == "structural_screen_pass"
    bad = box(b"ftyp") + struct.pack(">I4s", 100, b"mdat")
    assert mp4_probe(io.BytesIO(bad), 0, len(bad))["status"] == "box_extends_past_eof"
    assert mp4_probe(io.BytesIO(good), 0, len(good) - 3)["status"] == "truncated_box_header"
    wav = b"RIFF" + struct.pack("<I", 38) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16) + b"data" + struct.pack("<I", 2) + b"\0\0"
    assert wav_probe(io.BytesIO(wav), 0, len(wav))["status"] == "structural_screen_pass"
    assert wav_probe(io.BytesIO(wav[:-1]), 0, len(wav)-1)["status"] == "riff_extends_past_eof"
    for format in (tarfile.USTAR_FORMAT, tarfile.GNU_FORMAT, tarfile.PAX_FORMAT):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w", format=format) as archive:
            info = tarfile.TarInfo("folder/" + ("a" * 120 if format != tarfile.USTAR_FORMAT else "b") + ".mp4")
            info.size = len(good)
            archive.addfile(info, io.BytesIO(good))
        raw = data.getvalue()
        assert physical_scan(io.BytesIO(raw), len(raw))["status"] == "structure_pass"
        corrupt = bytearray(raw)
        corrupt[0] ^= 1
        assert physical_scan(io.BytesIO(corrupt), len(corrupt))["status"] == "structure_fail"
        assert physical_scan(io.BytesIO(raw[:520]), 520)["status"] == "structure_fail"
    assert unsafe_name("../../evil") and unsafe_name("C:/evil") and unsafe_name("\\\\server\\x")
    assert not unsafe_name("TS2/소음환경1/test.mp4")
    print("synthetic_tests_pass", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "archive_results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.root is None:
        if not args.self_test:
            parser.error("--root or --self-test is required")
        return
    output = args.output.resolve()
    if not output.is_relative_to(HERE / "archive_results"):
        parser.error("output must stay within owned archive_results directory")
    output.mkdir(parents=True, exist_ok=True)
    archives = sorted(args.root.rglob("*.tar"))
    summaries, members, probes = [], [], []
    started = time.perf_counter()
    for number, path in enumerate(archives, 1):
        summary, index, media = audit_one(path)
        save(output / (path.stem + "_structure.json"), {"summary": summary, "media": media})
        summaries.append(summary)
        members.extend(index)
        probes.extend(media)
        print(json.dumps({"progress": [number, len(archives)], "archive": path.name,
                          "status": summary["status"], "files": summary["file_count"],
                          "media": summary["media_status_counts"],
                          "seconds": round(summary["elapsed_seconds"], 2)}), flush=True)
    save(output / "member_index.json", {"schema": "archive-member-index-v1", "members": members})
    final = {"schema": "archive-structure-audit-v1", "date": "2026-10-02",
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope": "All TAR physical header checksums and bounds; all indexed MP4 top-level box headers and WAV RIFF chunk headers. No full payload hashes, media decoding, or official checksum comparison.",
             "archive_count": len(summaries), "archive_bytes": sum(x["before"]["size"] for x in summaries),
             "member_count": len(members), "file_count": sum(x["is_file"] for x in members),
             "archive_status_counts": dict(Counter(x["status"] for x in summaries)),
             "media_status_counts": {kind: dict(Counter(x["status"] for x in probes if x["media_type"] == kind)) for kind in ("mp4", "wav")},
             "archives": summaries, "media_failures": [x for x in probes if x["status"] != "structural_screen_pass"],
             "elapsed_seconds": time.perf_counter() - started}
    save(output / "summary.json", final)
    print(json.dumps({k: final[k] for k in ("archive_count", "archive_bytes", "file_count", "archive_status_counts", "media_status_counts", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
