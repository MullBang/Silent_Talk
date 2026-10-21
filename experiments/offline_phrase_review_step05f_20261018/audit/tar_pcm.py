"""Independent bounded TAR/RIFF reader for exact native-PCM audit comparisons.

No archive extraction or output media creation is performed. Member offsets
are supplied by the previously frozen source inventory; every read remains
within that declared byte range.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, BinaryIO


class AuditReadError(ValueError):
    """A source range, WAV format, or source identity failed validation."""


@dataclass
class MemberReader:
    stream: BinaryIO
    offset: int
    size: int
    bytes_read: int = 0

    def read(self, position: int, length: int) -> bytes:
        """Read a complete relative span without permitting member escape."""
        if (type(position) is not int or type(length) is not int
                or position < 0 or length < 0 or position + length > self.size):
            raise AuditReadError("Read exceeds the declared member range")
        self.stream.seek(self.offset + position)
        data = self.stream.read(length)
        self.bytes_read += len(data)
        if len(data) != length:
            raise AuditReadError("Archive member is truncated")
        return data


def validate_archive(spec: dict[str, Any]) -> Path:
    """Validate a bound plain-TAR identity without interpreting member names."""
    path = Path(spec["archive_path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (spec["archive_size"], spec["archive_mtime_ns"]):
        raise AuditReadError("Archive size or modification time changed")
    if not path.is_absolute() or path.suffix.lower() != ".tar":
        raise AuditReadError("Expected an absolute plain-TAR path")
    offset, size = spec["offset"], spec["size"]
    if (type(offset) is not int or type(size) is not int or offset < 512
            or offset % 512 or size <= 0 or offset + size > stat.st_size):
        raise AuditReadError("Invalid frozen member byte range")
    return path


def wav_layout(member: MemberReader) -> dict[str, int]:
    """Parse RIFF chunks, including odd padding, and locate PCM samples."""
    header = member.read(0, 12)
    if header[:4] != b"RIFF" or header[8:] != b"WAVE":
        raise AuditReadError("Expected RIFF/WAVE; RF64 is not accepted implicitly")
    riff_end = 8 + struct.unpack("<I", header[4:8])[0]
    if riff_end > member.size or riff_end < 12:
        raise AuditReadError("RIFF length exceeds member size")
    position = 12
    fmt: dict[str, int] | None = None
    data: tuple[int, int] | None = None
    while position + 8 <= riff_end:
        chunk = member.read(position, 8)
        name, length = chunk[:4], struct.unpack("<I", chunk[4:])[0]
        first = position + 8
        if first + length > riff_end:
            raise AuditReadError("RIFF chunk exceeds its declared container")
        if name == b"fmt ":
            if fmt is not None or length < 16 or length > 65536:
                raise AuditReadError("Ambiguous or invalid WAV fmt chunk")
            value = member.read(first, length)
            codec, channels, rate, byte_rate, block_align, bits = struct.unpack("<HHIIHH", value[:16])
            if codec == 0xFFFE and len(value) >= 40:
                if value[24:40] != bytes.fromhex("0100000000001000800000aa00389b71"):
                    raise AuditReadError("Unsupported extensible WAV subtype")
                codec = 1
            if codec != 1 or channels < 1 or bits != 16 or rate <= 0:
                raise AuditReadError("Only explicit 16-bit integer PCM is accepted")
            if block_align != channels * 2 or byte_rate != rate * block_align:
                raise AuditReadError("Inconsistent WAV PCM geometry")
            fmt = {"channels": channels, "sample_rate": rate,
                   "sample_width_bytes": 2, "frame_bytes": block_align}
        elif name == b"data":
            if data is not None:
                raise AuditReadError("Multiple WAV data chunks are not accepted")
            data = first, length
        position = first + length + (length % 2)
    if fmt is None or data is None:
        raise AuditReadError("Missing WAV fmt or data chunk")
    if data[1] % fmt["frame_bytes"]:
        raise AuditReadError("WAV data contains an incomplete sample frame")
    return {**fmt, "data_offset": data[0], "data_bytes": data[1],
            "sample_count": data[1] // fmt["frame_bytes"], "riff_end": riff_end}


def pcm_samples(member: MemberReader, layout: dict[str, int], first: int, end: int) -> bytes:
    """Read a half-open integer sample window without rounding or resampling."""
    if type(first) is not int or type(end) is not int or not 0 <= first < end <= layout["sample_count"]:
        raise AuditReadError("PCM window lies outside source WAV")
    width = layout["frame_bytes"]
    return member.read(layout["data_offset"] + first * width, (end - first) * width)
