"""Exact display-bin selection and a deliberately restricted nonfragmented MP4 clock.

Only one video track with constant presentation cadence is supported. Metadata
provides exact rational PTS; the caller must match them to decoded frame PTS.
Unknown edit lists, fragmented streams, duplicate PTS and VFR fail explicitly.
"""
from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from pathlib import Path
import struct
from typing import Any, Iterator


def require(condition: bool, message: str) -> None:
    """Reject unsupported inputs instead of inventing a clock."""
    if not condition:
        raise ValueError(message)


def rational(value: Fraction) -> dict[str, int | float]:
    """Serialize an exact value with an explicitly redundant display value."""
    return {"numerator": value.numerator, "denominator": value.denominator, "seconds": float(value)}


def fraction(value: dict[str, Any]) -> Fraction:
    """Read strict integral fraction fields, checking optional display seconds."""
    require(type(value["numerator"]) is int and type(value["denominator"]) is int, "Fraction fields must be integers")
    result = Fraction(value["numerator"], value["denominator"])
    require("seconds" not in value or abs(float(result)-value["seconds"]) < 1e-9, "Fraction/display mismatch")
    return result


def grid_window(start: Fraction, end: Fraction, origin: Fraction, coverage_end: Fraction) -> dict[str, Any]:
    """Return all recording-anchored 25fps display bins intersecting [start,end)."""
    require(origin <= start < end <= coverage_end, "Semantic interval outside supported recording")
    left, right = (start-origin)*25, (end-origin)*25
    first = left.numerator // left.denominator
    stop = -(-right.numerator // right.denominator)
    require(0 <= first < stop, "Empty or negative display grid")
    times = [origin + Fraction(k,25) for k in range(first,stop)]
    require(origin+Fraction(stop,25) <= coverage_end, "Display support exceeds recording coverage; hold partial recording edge")
    overlaps = [min(end,t+Fraction(1,25))-max(start,t) for t in times]
    require(all(o > 0 for o in overlaps), "Nonintersecting display bin")
    return {"first":first,"stop":stop,"times":times,"overlaps":overlaps,
            "display_start":times[0],"display_end":origin+Fraction(stop,25),
            "before_semantic":start-times[0],"after_semantic":origin+Fraction(stop,25)-end}


def constant_period(pts: list[Fraction]) -> Fraction:
    """Reject nonmonotone, repeated or variable presentation timestamps."""
    require(len(pts) >= 2, "At least two presentation timestamps required")
    period = pts[1]-pts[0]
    require(period > 0 and all(b-a == period for a,b in zip(pts,pts[1:])), "Unsupported VFR, repeated or nonmonotone PTS")
    return period


def nearest(pts: list[Fraction], target: Fraction) -> tuple[int, tuple[int, ...]]:
    """Choose nearest presentation time with exact ties going to the earlier PTS.

    Both bracket frames must exist, except an exact match to the first/last PTS.
    The returned bracket must be decoded and PTS-validated by the caller.
    """
    require(bool(pts) and pts[0] <= target <= pts[-1], "Missing decoded bracketing coverage")
    right = bisect_left(pts,target)
    if pts[right] == target:
        adjacent = tuple(i for i in (right-1,right,right+1) if 0 <= i < len(pts))
        return right, adjacent
    left = right-1
    return (left if target-pts[left] <= pts[right]-target else right), (left,right)


def boxes(data: bytes) -> Iterator[tuple[bytes, bytes]]:
    """Iterate bounded ISO BMFF child boxes, refusing truncation and invalid sizes."""
    offset = 0
    while offset < len(data):
        require(len(data)-offset >= 8, "Truncated box header")
        size,kind = struct.unpack_from(">I4s",data,offset)
        head = 8
        if size == 1:
            require(len(data)-offset >= 16, "Truncated extended box")
            size = struct.unpack_from(">Q",data,offset+8)[0]
            head = 16
        elif size == 0:
            size = len(data)-offset
        require(head <= size <= len(data)-offset, "Invalid child box length")
        yield kind,data[offset+head:offset+size]
        offset += size


def child(data: bytes, name: bytes, optional: bool = False) -> bytes | None:
    """Require one named child, allowing an absent optional table only."""
    matches = [body for kind,body in boxes(data) if kind == name]
    require(len(matches) == 1 or (optional and not matches), f"Missing or duplicate {name!r}")
    return matches[0] if matches else None


def u32(data: bytes, offset: int) -> int:
    """Read a big-endian integer with explicit length validation."""
    require(offset >= 0 and offset+4 <= len(data), "Truncated clock integer")
    return struct.unpack_from(">I",data,offset)[0]


def timescale(header: bytes) -> int:
    """Read mdhd/mvhd v0 or v1 timescale."""
    require(len(header) >= 4 and header[0] in (0,1), "Unsupported clock header")
    scale = u32(header,20 if header[0] else 12)
    require(scale > 0, "Invalid clock timescale")
    return scale


def expand_table(data: bytes, signed: bool = False) -> list[int]:
    """Expand bounded stts/ctts run-length tables to one value per sample."""
    require(len(data) >= 8 and data[0] in (0,1), "Unsupported timestamp table")
    count = u32(data,4)
    require(len(data) == 8+8*count, "Timestamp table size mismatch")
    result = []
    for i in range(count):
        repeats = u32(data,8+i*8)
        require(repeats > 0 and len(result)+repeats <= 2_000_000, "Invalid or oversized timestamp table")
        value = struct.unpack_from(">i" if signed else ">I",data,12+i*8)[0]
        result.extend([value]*repeats)
    return result


def parse_moov(moov: bytes) -> dict[str, Any]:
    """Parse exact sorted presentation times for one untrimmed CFR video track."""
    require(child(moov,b"mvex",optional=True) is None, "Fragmented MP4 is unsupported")
    movie_scale = timescale(child(moov,b"mvhd"))
    candidates = []
    for kind,trak in boxes(moov):
        if kind != b"trak":
            continue
        mdia = child(trak,b"mdia")
        handler = child(mdia,b"hdlr")
        require(len(handler) >= 12, "Truncated handler")
        if handler[8:12] == b"vide":
            candidates.append((trak,mdia))
    require(len(candidates) == 1, "Exactly one video track required")
    trak,mdia = candidates[0]
    scale = timescale(child(mdia,b"mdhd"))
    stbl = child(child(mdia,b"minf"),b"stbl")
    durations = expand_table(child(stbl,b"stts"))
    require(all(x > 0 for x in durations), "Nonpositive decode duration")
    stsz = child(stbl,b"stsz")
    require(len(stsz) >= 12 and u32(stsz,8) == len(durations), "Sample count differs from clock")
    ctts = child(stbl,b"ctts",optional=True)
    offsets = [0]*len(durations) if ctts is None else expand_table(ctts,signed=ctts[0] == 1)
    require(len(offsets) == len(durations), "Composition table count differs")
    dts = 0
    values = []
    for duration,offset in zip(durations,offsets):
        values.append(dts+offset)
        dts += duration
    values.sort()
    edit = child(trak,b"edts",optional=True)
    media_start = 0
    edit_duration = None
    if edit is not None:
        elst = child(edit,b"elst")
        require(len(elst) >= 8 and elst[0] in (0,1) and u32(elst,4) == 1, "Only one nonempty edit is supported")
        fmt = ">Qqhh" if elst[0] else ">Iihh"
        require(len(elst) == 8+struct.calcsize(fmt), "Edit list size mismatch")
        duration,media_start,rate_int,rate_frac = struct.unpack_from(fmt,elst,8)
        require(duration > 0 and media_start >= 0 and (rate_int,rate_frac) == (1,0), "Empty, negative or rate-changing edit unsupported")
        require(values[0] == media_start, "Edit trims or omits source presentation frames")
        edit_duration = Fraction(duration,movie_scale)
    pts = [Fraction(value-media_start,scale) for value in values]
    period = constant_period(pts)
    require(pts[0] >= 0, "Negative recording origin unsupported")
    if edit_duration is not None:
        require(abs(edit_duration-(pts[-1]+period)) <= Fraction(1,movie_scale), "Edit does not cover full CFR timeline")
    return {"pts":pts,"period":period,"origin":pts[0],"coverage_end":pts[-1]+period,
            "timescale":scale,"movie_timescale":movie_scale,"media_edit_start":media_start,
            "sample_count":len(pts),"cfr_verified_by_exact_presentation_table":True}


def read_clock(path: Path) -> dict[str, Any]:
    """Read only box headers and moov metadata, never scan/decode the full video."""
    try:
        length = path.stat().st_size
        found = []
        with path.open("rb") as stream:
            cursor = 0
            while cursor < length:
                stream.seek(cursor)
                header = stream.read(8)
                require(len(header) == 8, "Truncated top-level box")
                size,kind = struct.unpack(">I4s",header)
                head = 8
                if size == 1:
                    extended = stream.read(8)
                    require(len(extended) == 8, "Truncated extended box")
                    size,head = struct.unpack(">Q",extended)[0],16
                elif size == 0:
                    size = length-cursor
                require(head <= size <= length-cursor, "MP4 box exceeds source file")
                require(kind != b"moof", "Fragmented MP4 unsupported")
                if kind == b"moov":
                    require(size <= 128*1024**2, "Unexpectedly large moov")
                    found.append(stream.read(size-head))
                cursor += size
        require(len(found) == 1, "Exactly one moov required")
        return parse_moov(found[0])
    except (OSError,ValueError,struct.error) as exc:
        raise RuntimeError(f"Unsupported/invalid original MP4 clock: {path}") from exc
