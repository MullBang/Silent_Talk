"""Tests for bounded, temporary materialization from audited plain TAR ranges."""
from __future__ import annotations

import copy
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tarfile
from typing import Any

import pytest

from experiments.local_source_step01_20261007 import tar_source


def make_source(root: Path, *, long_names: bool = False) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build real regular or PAX TAR members and their audited reference fields."""
    try:
        stem = "lip_J_1_F_02_C002_A_004"
        payloads = {"video": b"video-payload" * 101, "label": b'{"Sentence_info": []}',
                    "audio": b"RIFF-audio-payload" * 31}
        archive = root / "source.tar"
        prefix = "TS1/" + ("long-name-" * 30 if long_names else "source")
        with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as handle:
            for role, data in payloads.items():
                entry = tarfile.TarInfo(f"{prefix}/{stem}{tar_source.ROLE_SUFFIXES[role]}")
                entry.size = len(data)
                handle.addfile(entry, io.BytesIO(data))
        stat = archive.stat()
        members = {}
        with tarfile.open(archive, "r:") as handle:
            for entry in handle:
                role = next(role for role, suffix in tar_source.ROLE_SUFFIXES.items()
                            if entry.name.endswith(suffix))
                members[role] = {"archive_path": str(archive), "archive_size": stat.st_size,
                                 "archive_mtime_ns": stat.st_mtime_ns,
                                 "member_name": entry.name, "offset": entry.offset_data,
                                 "size": entry.size,
                                 "sha256": hashlib.sha256(payloads[role]).hexdigest()}
        return {"stem": stem, "members": members}, payloads
    except (OSError, tarfile.TarError):
        raise


@pytest.mark.parametrize("long_names", [False, True])
def test_payload_identity_original_preservation_and_cleanup(tmp_path: Path, long_names: bool) -> None:
    """Only planned members appear locally and archive bytes remain unchanged."""
    source, payloads = make_source(tmp_path, long_names=long_names)
    archive = Path(source["members"]["video"]["archive_path"])
    original = archive.read_bytes()
    target = tmp_path / "local"
    with tar_source.materialize_source(source, target, minimum_free_bytes=0) as paths:
        assert set(paths) == set(payloads)
        assert all(path.read_bytes() == payloads[role] for role, path in paths.items())
        assert len({path.parent for path in paths.values()}) == 1
    assert list(target.iterdir()) == []
    assert all(not path.exists() for path in paths.values())
    assert archive.read_bytes() == original


def test_label_only_and_caller_failure_cleanup(tmp_path: Path) -> None:
    """A caller's processing failure cannot leave a copied recording behind."""
    source, _ = make_source(tmp_path)
    source["members"] = {"label": source["members"]["label"]}
    target = tmp_path / "local"
    with pytest.raises(RuntimeError, match="caller failure"):
        with tar_source.materialize_source(source, target, 0) as paths:
            assert set(paths) == {"label"}
            raise RuntimeError("caller failure")
    assert list(target.iterdir()) == []


def test_hash_mismatch_cleans_partial_recording(tmp_path: Path) -> None:
    """A later member failing its digest removes earlier verified copies too."""
    source, _ = make_source(tmp_path)
    source["members"]["audio"]["sha256"] = "0" * 64
    target = tmp_path / "local"
    with pytest.raises(tar_source.TarSourceError, match="SHA-256 mismatch"):
        with tar_source.materialize_source(source, target, 0):
            pytest.fail("corrupt payload must not be yielded")
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("member_name", ["../lip_J_1_F_02_C002_A_004.mp4",
                                        "/absolute/lip_J_1_F_02_C002_A_004.mp4",
                                        "C:/drive/lip_J_1_F_02_C002_A_004.mp4",
                                        "folder\\lip_J_1_F_02_C002_A_004.mp4",
                                        "folder/./lip_J_1_F_02_C002_A_004.mp4",
                                        "folder//lip_J_1_F_02_C002_A_004.mp4"])
def test_unsafe_member_names_rejected_before_copy(tmp_path: Path, member_name: str) -> None:
    """Archive names cannot traverse or choose the destination filesystem path."""
    source, _ = make_source(tmp_path)
    source["members"]["video"]["member_name"] = member_name
    with pytest.raises(tar_source.TarSourceError, match="Unsafe member path"):
        with tar_source.materialize_source(source, tmp_path / "local", 0):
            pytest.fail("unsafe path must not be yielded")


@pytest.mark.parametrize("field,value", [("offset", 513), ("offset", -512),
                                       ("size", 10**9), ("size", True)])
def test_invalid_ranges_rejected(tmp_path: Path, field: str, value: int) -> None:
    """Only aligned and bounded byte ranges from the audit are accepted."""
    source, _ = make_source(tmp_path)
    source["members"]["video"][field] = value
    with pytest.raises(tar_source.TarSourceError):
        with tar_source.materialize_source(source, tmp_path / "local", 0):
            pytest.fail("invalid range must not be yielded")


def test_truncated_archive_rejected_before_copy(tmp_path: Path) -> None:
    """A file shortened after its audit cannot be treated as the verified TAR."""
    source, _ = make_source(tmp_path)
    archive = Path(source["members"]["video"]["archive_path"])
    with archive.open("r+b") as handle:
        handle.truncate(100)
    with pytest.raises(tar_source.TarSourceError, match="Archive stat differs"):
        with tar_source.materialize_source(source, tmp_path / "local", 0):
            pytest.fail("truncation must not be yielded")


def test_source_change_during_caller_processing_detected_and_cleaned(tmp_path: Path) -> None:
    """A changed source is reported even if the copies were already verified."""
    source, _ = make_source(tmp_path)
    archive = Path(source["members"]["video"]["archive_path"])
    target = tmp_path / "local"
    with pytest.raises(tar_source.TarSourceError, match="Archive stat differs"):
        with tar_source.materialize_source(source, target, 0):
            current = archive.stat()
            os.utime(archive, ns=(current.st_atime_ns, current.st_mtime_ns + 10_000_000))
    assert list(target.iterdir()) == []


def test_free_space_includes_all_simultaneous_members_and_reserve(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """The reserve is additional to all copied video, audio and label bytes."""
    source, _ = make_source(tmp_path)
    reserve = 1000
    needed = reserve + sum(item["size"] for item in source["members"].values())
    monkeypatch.setattr(tar_source.shutil, "disk_usage", lambda _: SimpleNamespace(free=needed - 1))
    target = tmp_path / "local"
    with pytest.raises(tar_source.TarSourceError, match="Insufficient temporary space"):
        with tar_source.materialize_source(source, target, reserve):
            pytest.fail("insufficient free space must not start copying")
    assert list(target.iterdir()) == []


def test_copy_never_reads_bytes_outside_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small blocks and adjacent member payloads do not affect copied content."""
    source, payloads = make_source(tmp_path)
    monkeypatch.setattr(tar_source, "COPY_BLOCK_BYTES", 7)
    source = copy.deepcopy(source)
    source["members"] = {"video": source["members"]["video"]}
    with tar_source.materialize_source(source, tmp_path / "local", 0) as paths:
        assert paths["video"].read_bytes() == payloads["video"]


def test_wrong_recording_name_rejected(tmp_path: Path) -> None:
    """A valid SHA on a different recording cannot satisfy a source identity."""
    source, _ = make_source(tmp_path)
    source["members"]["label"]["member_name"] = "TL1/different_recording.json"
    with pytest.raises(tar_source.TarSourceError, match="Member identity mismatch"):
        with tar_source.materialize_source(source, tmp_path / "local", 0):
            pytest.fail("wrong recording identity must not be yielded")
