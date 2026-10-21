"""Focused negative cases for independent RIFF/sample-range validation."""
from __future__ import annotations

import io
import struct
import pytest

from tar_pcm import AuditReadError, MemberReader, pcm_samples, wav_layout


def wav(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"WAVE" + b"".join(name + struct.pack("<I", len(value)) + value + b"\0" * (len(value) % 2) for name, value in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


FMT = struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16)


def reader(data: bytes) -> MemberReader:
    return MemberReader(io.BytesIO(b"x" * 512 + data), 512, len(data))


def test_odd_metadata_padding_and_pcm_origin() -> None:
    data = wav([(b"JUNK", b"abc"), (b"fmt ", FMT), (b"data", struct.pack("<hhhh", 7, -3, 11, 5))])
    source = reader(data)
    layout = wav_layout(source)
    assert layout["sample_count"] == 4 and layout["sample_rate"] == 48000
    assert pcm_samples(source, layout, 1, 3) == struct.pack("<hh", -3, 11)


def test_declared_riff_exceeds_member() -> None:
    data = bytearray(wav([(b"fmt ", FMT), (b"data", b"\0\0")]))
    data[4:8] = struct.pack("<I", len(data) + 100)
    with pytest.raises(AuditReadError, match="RIFF length"):
        wav_layout(reader(bytes(data)))


def test_duplicate_data_is_rejected() -> None:
    with pytest.raises(AuditReadError, match="Multiple"):
        wav_layout(reader(wav([(b"fmt ", FMT), (b"data", b"\0\0"), (b"data", b"\0\0")])))


def test_incomplete_pcm_frame_rejected() -> None:
    with pytest.raises(AuditReadError, match="incomplete"):
        wav_layout(reader(wav([(b"fmt ", FMT), (b"data", b"\0")])))


def test_pcm_outside_window_rejected() -> None:
    source = reader(wav([(b"fmt ", FMT), (b"data", b"\0\0")]))
    layout = wav_layout(source)
    with pytest.raises(AuditReadError, match="outside"):
        pcm_samples(source, layout, 0, 2)


def test_member_range_cannot_escape() -> None:
    source = reader(b"abc")
    with pytest.raises(AuditReadError, match="member range"):
        source.read(-1, 1)
    with pytest.raises(AuditReadError, match="member range"):
        source.read(2, 2)


def test_nonpcm_codec_rejected() -> None:
    float_fmt = struct.pack("<HHIIHH", 3, 1, 48000, 96000, 2, 16)
    with pytest.raises(AuditReadError, match="integer PCM"):
        wav_layout(reader(wav([(b"fmt ", float_fmt), (b"data", b"\0\0")])))
