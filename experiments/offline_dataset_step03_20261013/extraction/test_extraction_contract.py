"""Meaningful synthetic clock, boundary, source-copy and CTC failure checks.

These tests never open AIHub TARs, call a model or create training inputs.
"""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
from pathlib import Path
import struct
from typing import Any

import numpy as np
import pytest

import exact_clock as clock
import prepare_and_extract as extraction


def atom(kind: bytes, payload: bytes) -> bytes:
    """Construct a tiny well-bounded ISO box for parser tests."""
    return struct.pack(">I4s",len(payload)+8,kind)+payload


def fake_moov(durations: list[int], offsets: list[int] | None = None, edit: tuple[int,int,int,int] | None = None) -> bytes:
    """Build only the relevant metadata, including optional reordered PTS/edit."""
    header = b"\0"*12+struct.pack(">II",1000,sum(durations))+b"\0"*4
    stts = b"\0"*4+struct.pack(">I",len(durations))+b"".join(struct.pack(">II",1,d) for d in durations)
    tables = atom(b"stts",stts)+atom(b"stsz",b"\0"*4+struct.pack(">II",1,len(durations)))
    if offsets is not None:
        ctts = b"\0"*4+struct.pack(">I",len(offsets))+b"".join(struct.pack(">II",1,o) for o in offsets)
        tables += atom(b"ctts",ctts)
    mdia = atom(b"mdhd",header)+atom(b"hdlr",b"\0"*8+b"vide"+b"\0"*12)+atom(b"minf",atom(b"stbl",tables))
    trak = atom(b"mdia",mdia)
    if edit is not None:
        trak += atom(b"edts",atom(b"elst",b"\0"*4+struct.pack(">I",1)+struct.pack(">Iihh",*edit)))
    return atom(b"mvhd",header)+atom(b"trak",trak)


@pytest.mark.parametrize("start,end,first,stop",[(F(1,25),F(2,25),1,2),(F(41,1000),F(79,1000),1,2),(F(39,1000),F(81,1000),0,3),(F(0),F(1,25),0,1)])
def test_exact_floor_ceil_overlap(start: F,end: F,first: int,stop: int) -> None:
    """Aligned edges exclude the next bin; fractional edges include both sides."""
    result = clock.grid_window(start,end,F(0),F(10))
    assert (result["first"],result["stop"]) == (first,stop)
    assert sum(result["overlaps"]) == end-start


def test_nonzero_recording_origin_is_preserved() -> None:
    """Grid phase follows measured recording origin, never an implicit zero."""
    origin = F(7,1000)
    result = clock.grid_window(origin+F(1,25),origin+F(2,25),origin,F(10))
    assert result["times"] == [F(47,1000)]
    assert result["before_semantic"] == result["after_semantic"] == 0


@pytest.mark.parametrize("start,end,origin,coverage",[(F(-1,100),F(1),F(0),F(10)),(F(1),F(1),F(0),F(10)),(F(2),F(1),F(0),F(10)),(F(1),F(11),F(0),F(10)),(F(0),F(39,1000),F(0),F(39,1000))])
def test_invalid_or_partial_recording_edges_are_held(start: F,end: F,origin: F,coverage: F) -> None:
    """Reject semantic and display coverage errors instead of padding/trimming."""
    with pytest.raises(ValueError):
        clock.grid_window(start,end,origin,coverage)


def test_nearest_tie_uses_earlier_actual_pts() -> None:
    """Exact ties and tiny perturbations must not depend on float rounding."""
    pts = [F(7,1000),F(107,3000),F(193,3000)]
    midpoint = (pts[0]+pts[1])/2
    assert clock.nearest(pts,midpoint) == (0,(0,1))
    assert clock.nearest(pts,midpoint+F(1,10**12))[0] == 1
    assert clock.nearest(pts,pts[0]) == (0,(0,1))
    assert clock.nearest(pts,pts[-1]) == (2,(1,2))


@pytest.mark.parametrize("target",[F(-1,100),F(11,100)])
def test_missing_bracket_is_rejected(target: F) -> None:
    """A nominal nearest-frame guess cannot hide missing real brackets."""
    with pytest.raises(ValueError):
        clock.nearest([F(0),F(1,10)],target)


@pytest.mark.parametrize("pts",[[F(0),F(0)],[F(1),F(0)],[F(0),F(1,25),F(9,100)]])
def test_non_cfr_and_repeated_pts_rejected(pts: list[F]) -> None:
    """Global table cadence cannot silently collapse VFR or duplicate frames."""
    with pytest.raises(ValueError):
        clock.constant_period(pts)


def test_composition_order_and_edit_offset() -> None:
    """B-frame composition order plus one full edit yields exact presentation order."""
    parsed = clock.parse_moov(fake_moov([40]*4,[80,160,40,40],(160,80,1,0)))
    assert parsed["pts"] == [F(0),F(1,25),F(2,25),F(3,25)]
    assert parsed["media_edit_start"] == 80
    assert parsed["coverage_end"] == F(4,25)


def test_metadata_nonzero_first_pts() -> None:
    """A track without an edit may start later than zero and retains its origin."""
    parsed = clock.parse_moov(fake_moov([40]*4,[20]*4))
    assert parsed["origin"] == F(1,50)


@pytest.mark.parametrize("edit",[(120,40,1,0),(160,-1,1,0),(160,0,2,0),(40,0,1,0)])
def test_trim_empty_rate_or_incomplete_edit_rejected(edit: tuple[int,int,int,int]) -> None:
    """Unsupported edit semantics cannot be mistaken for ordinary CFR origin."""
    with pytest.raises(ValueError):
        clock.parse_moov(fake_moov([40]*4,edit=edit))


def test_vfr_metadata_rejected() -> None:
    """Varying sample duration changes presentation cadence and must stop."""
    with pytest.raises(ValueError):
        clock.parse_moov(fake_moov([40,41,40,40]))


def test_truncated_metadata_and_fragmented_file_rejected(tmp_path: Path) -> None:
    """Box bounds and fragmented source rejection work without original media."""
    with pytest.raises(ValueError):
        list(clock.boxes(struct.pack(">I4s",100,b"bad!")+b"short"))
    path = tmp_path/"fake.mp4"
    path.write_bytes(atom(b"moov",fake_moov([40]*4))+atom(b"moof",b""))
    with pytest.raises(RuntimeError):
        clock.read_clock(path)


def test_native48k_mapping_applies_offset_once() -> None:
    """Source sample rounding stays exact through context16k and offset48k."""
    record = {"context_audio":{"source_sample_origin_verified":True,"source_sample_rate":48000,"sample_rate":16000,"source_first_sample":13230688},"offset_seconds":1024/48000}
    assert extraction.original_time(record,44680) == F(13230688+3*44680+1024,48000)
    assert extraction.original_time(record,44681)-extraction.original_time(record,44680) == F(1,16000)
    record["offset_seconds"] = 0.00001
    with pytest.raises(ValueError):
        extraction.original_time(record,44680)


def test_ctc_repeats_zero_and_shape_failures() -> None:
    """Adjacent identical labels require an extra time step; zero fill is never used."""
    good = np.ones((3,96,96,3),dtype=np.uint8)
    assert extraction.validate_clip_array(good,[1,1]) == 3
    for pixels,labels in ((good[:2],[1,1]),(np.zeros_like(good),[1]),(good.astype(np.float32),[1]),(good,[0]),(good,[69])):
        with pytest.raises(ValueError):
            extraction.validate_clip_array(pixels,labels)


def test_copy_failure_records_actual_bytes_and_cleans_owned_child(tmp_path: Path,monkeypatch: Any) -> None:
    """A synthetic member hash failure preserves accounting and only deletes its temp child."""
    monkeypatch.setattr(extraction,"BASE",tmp_path)
    monkeypatch.setattr(extraction,"RESERVE",0)
    helper = extraction.load_module(extraction.TAR_HELPER,"test_tar_contract")
    path = tmp_path/"synthetic.tar"
    payload = b"synthetic no real AIHub payload"
    path.write_bytes(b"\0"*512+payload+b"\0"*(512-len(payload)))
    stat = path.stat()
    member = {"archive_path":str(path.resolve()),"archive_size":stat.st_size,"archive_mtime_ns":stat.st_mtime_ns,"member_name":"prefix/test_video.mp4","offset":512,"size":len(payload),"sha256":"0"*64}
    label = {**member,"member_name":"prefix/test_video.json","sha256":hashlib.sha256(payload).hexdigest()}
    entry = {"recording_group":"test_video","parent_id":"synthetic","source_members":{"video":member,"label":label}}
    io = {"member_read_attempts":0,"payload_bytes_read":0,"members_sha_verified":0}
    sentinel = tmp_path/"keep.txt"
    sentinel.write_text("preserve",encoding="utf-8")
    with pytest.raises(ValueError):
        with extraction.materialize(entry,helper,io,tmp_path/"io.jsonl"):
            pytest.fail("Bad member must never be yielded")
    assert io == {"member_read_attempts":1,"payload_bytes_read":len(payload),"members_sha_verified":0}
    assert list((tmp_path/"_tmp").iterdir()) == []
    assert sentinel.read_text(encoding="utf-8") == "preserve"
