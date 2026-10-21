"""Bounds-only container screening: synthetic local data, no cloud or decoding."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import probe_intake_containers as screening


def box(kind, payload=b"", *, size=None):
    return struct.pack(">I4s", len(payload) + 8 if size is None else size, kind) + payload


def extended_box(kind, payload=b"", *, size=None):
    return struct.pack(">I4sQ", 1, kind, len(payload) + 16 if size is None else size) + payload


FTYP = box(b"ftyp", b"isom\x00\x00\x02\x00isom")
MOOV = box(b"moov")
MDAT = box(b"mdat", b"synthetic media bytes")


def screen_bytes(tmp_path, contents):
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(contents)
    return screening.probe(path)


@pytest.mark.parametrize("parts", [(FTYP, MOOV, MDAT), (FTYP, MDAT, MOOV), (FTYP, box(b"free"), MDAT, MOOV)])
def test_both_moov_positions_pass_only_shallow_structure(tmp_path, parts):
    result = screen_bytes(tmp_path, b"".join(parts))
    assert result["status"] == "structural_screen_pass"
    assert result["payload_decoded"] is False
    assert result["header_bytes_read"] == 8 * len(parts)
    # The deliberately empty moov is not a playable movie; pass is bounds-only.
    assert next(b for b in result["boxes"] if b["type"] == "moov")["declared_size"] == 8


@pytest.mark.parametrize("tail", [b"x", b"1234567"])
def test_partial_basic_header_is_not_silently_ignored(tmp_path, tail):
    result = screen_bytes(tmp_path, FTYP + MOOV + MDAT + tail)
    assert result["status"] == "truncated_box_header"
    assert result["offset"] == len(FTYP + MOOV + MDAT)
    assert result["header_bytes_read"] == 24 + len(tail)


@pytest.mark.parametrize("tail", [b"", b"1234567"])
def test_incomplete_extended_header_is_reported(tmp_path, tail):
    result = screen_bytes(tmp_path, FTYP + struct.pack(">I4s", 1, b"mdat") + tail)
    assert result["status"] == "truncated_extended_header"
    assert result["offset"] == len(FTYP)
    assert result["header_bytes_read"] == 16 + len(tail)


@pytest.mark.parametrize("length", [2, 7])
def test_nonzero_basic_size_cannot_be_smaller_than_header(tmp_path, length):
    result = screen_bytes(tmp_path, FTYP + box(b"mdat", size=length))
    assert result["status"] == "invalid_box_size"


@pytest.mark.parametrize("length", [0, 1, 8, 15])
def test_extended_size_cannot_be_smaller_than_extended_header(tmp_path, length):
    result = screen_bytes(tmp_path, FTYP + extended_box(b"mdat", size=length))
    assert result["status"] == "invalid_box_size"


def test_complete_extended_header_is_counted_and_skips_payload(tmp_path):
    data = FTYP + extended_box(b"mdat", b"payload") + MOOV
    result = screen_bytes(tmp_path, data)
    assert result["status"] == "structural_screen_pass"
    assert result["header_bytes_read"] == 32
    assert result["boxes"][1]["declared_size"] == 23


def test_declared_payload_past_eof_reports_exact_minimum_missing_bytes(tmp_path):
    data = FTYP + box(b"mdat", b"abc", size=100)
    result = screen_bytes(tmp_path, data)
    assert result["status"] == "box_extends_past_eof"
    assert result["minimum_expected_bytes"] == len(FTYP) + 100
    assert result["minimum_missing_bytes"] == 89
    assert result["header_bytes_read"] == 16


def test_64bit_length_is_not_truncated_to_32bits(tmp_path):
    declared = 2**40 + 17
    result = screen_bytes(tmp_path, FTYP + extended_box(b"mdat", size=declared))
    assert result["status"] == "box_extends_past_eof"
    assert result["minimum_expected_bytes"] == len(FTYP) + declared
    assert result["minimum_missing_bytes"] == declared - 16


def test_zero_size_mdat_consumes_remaining_file_once(tmp_path):
    result = screen_bytes(tmp_path, FTYP + MOOV + box(b"mdat", b"remaining payload", size=0))
    assert result["status"] == "structural_screen_pass"
    assert result["boxes"][-1]["declared_end"] == result["size_bytes"]
    assert result["header_bytes_read"] == 24


def test_moov_bytes_inside_mdat_are_not_a_top_level_movie_box(tmp_path):
    result = screen_bytes(tmp_path, FTYP + box(b"mdat", b"prefix" + MOOV + b"suffix"))
    assert result["status"] == "missing_moov"
    assert [b["type"] for b in result["boxes"]] == ["ftyp", "mdat"]
    assert result["header_bytes_read"] == 16


@pytest.mark.parametrize("data,status", [(b"", "missing_initial_ftyp"), (MOOV + MDAT, "missing_initial_ftyp"),
                                        (FTYP + MDAT, "missing_moov"), (FTYP + MOOV, "missing_mdat")])
def test_missing_required_top_level_boxes_are_reported(tmp_path, data, status):
    assert screen_bytes(tmp_path, data)["status"] == status


def test_large_payload_is_skipped_by_seek_and_never_read_or_allocated():
    mdat_offset = len(FTYP)
    mdat_size = 2**34 + 16
    moov_offset = mdat_offset + mdat_size
    size = moov_offset + len(MOOV)
    chunks = {0: FTYP[:8], mdat_offset: struct.pack(">I4s", 1, b"mdat"),
              mdat_offset + 8: struct.pack(">Q", mdat_size), moov_offset: MOOV}
    class Stream:
        def __init__(self):
            self.pos = 0
            self.reads = []
            self.seeks = []
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def seek(self, offset):
            assert offset in (0, mdat_offset, moov_offset)
            self.seeks.append(offset)
            self.pos = offset
        def read(self, count):
            assert count == 8 and self.pos in chunks, "Attempt to read media payload"
            self.reads.append((self.pos, count))
            value = chunks[self.pos]
            self.pos += len(value)
            return value
    stream = Stream()
    class VirtualPath:
        def stat(self):
            return SimpleNamespace(st_size=size)
        def open(self, mode):
            assert mode == "rb"
            return stream
    result = screening.probe(VirtualPath())
    assert result["status"] == "structural_screen_pass"
    assert result["header_bytes_read"] == 32
    assert len(stream.reads) == 4 and len(stream.seeks) == 3


def test_box_count_limit_bounds_header_work(tmp_path):
    result = screen_bytes(tmp_path, box(b"free") * 10001)
    assert result["status"] == "too_many_top_level_boxes"
    assert len(result["boxes"]) == 10000
    assert result["header_bytes_read"] == 80000


@pytest.fixture
def main_case(tmp_path, monkeypatch):
    work, source = tmp_path / "work", tmp_path / "source"
    out = work / "results/step18-intake/run01"
    out.mkdir(parents=True)
    video = source / "Training" / "TS2" / "new.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(FTYP + MDAT + MOOV)
    for relative in ("Training/TS2/audio.wav", "Training/TS1/old.mp4", "Validation/VS2/held.mp4"):
        forbidden = source / relative
        forbidden.parent.mkdir(parents=True, exist_ok=True)
        forbidden.write_bytes(b"excluded payload")
    st = video.stat()
    record = {"stem": "new", "video": {"path": str(video), "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns, "package": "TS2", "speaker_id": "C011", "recording_id": "recording1"}}
    (out / "source_index.json").write_text(json.dumps({"sources": [record]}), encoding="utf-8")
    (out / "inventory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(screening, "ROOT", work)
    monkeypatch.setattr(sys, "argv", ["probe", "--run", str(out)])
    return SimpleNamespace(work=work, source=source, out=out, video=video)


def test_main_reads_only_preselected_index_videos_and_never_writes_source(main_case, monkeypatch):
    """Scope comes from the already-validated frozen index, not from a fresh scan."""
    case = main_case
    real_open = Path.open
    opened = []
    def guard(path, mode="r", *args, **kwargs):
        if path.is_relative_to(case.source):
            assert path == case.video and mode == "rb"
            opened.append(path)
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    screening.main()
    result = json.loads((case.out / "container_probe.json").read_text(encoding="utf-8"))
    assert opened == [case.video]
    assert result["count"] == 1 and result["header_bytes_read"] == 24
    assert result["counts"] == {"structural_screen_pass": 1}
    assert result["records"][0]["payload_decoded"] is False
    assert "not decoding" in result["scope"]


def test_metadata_change_before_probe_stops_before_payload_access(main_case, monkeypatch):
    case = main_case
    case.video.write_bytes(case.video.read_bytes() + b" ")
    monkeypatch.setattr(screening, "probe", lambda _: pytest.fail("Changed source opened"))
    screening.main()
    result = json.loads((case.out / "container_probe.json").read_text(encoding="utf-8"))
    assert result["counts"] == {"metadata_changed_since_inventory": 1}
    assert result["header_bytes_read"] == 0


def test_metadata_change_during_probe_overrides_initial_pass(main_case, monkeypatch):
    real_probe = screening.probe
    def mutate_after_read(path):
        result = real_probe(path)
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(screening, "probe", mutate_after_read)
    screening.main()
    result = json.loads((main_case.out / "container_probe.json").read_text(encoding="utf-8"))
    record = result["records"][0]
    assert record["status"] == "metadata_changed_during_probe"
    assert record["status_before_change"] == "structural_screen_pass"


def test_read_failure_is_reported_without_false_structural_pass(main_case, monkeypatch):
    def unreadable(_):
        raise OSError("synthetic read failure")
    monkeypatch.setattr(screening, "probe", unreadable)
    screening.main()
    result = json.loads((main_case.out / "container_probe.json").read_text(encoding="utf-8"))
    assert result["counts"] == {"read_error": 1}
    assert result["header_bytes_read"] == 0


def test_existing_probe_result_is_never_overwritten(main_case):
    screening.main()
    path = main_case.out / "container_probe.json"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        screening.main()
    assert path.read_bytes() == before


def test_cli_refuses_output_outside_stage18_before_file_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(screening, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["probe", "--run", str(tmp_path / "other")])
    with pytest.raises(ValueError, match="Stage18 output required"):
        screening.main()
