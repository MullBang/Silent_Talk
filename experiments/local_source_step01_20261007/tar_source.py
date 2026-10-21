"""Materialize explicitly audited members of plain TAR files, one recording at a time.

The caller must bind the member specifications to the integrity audit and its
frozen source plan. This reader deliberately does not scan archives or interpret
TAR paths: audited offsets resolve GNU/PAX names, and the payload SHA-256 is
verified while copying a bounded range into generated local filenames. Original
archives are opened read-only. Returned paths exist only inside the context.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping


ROLE_SUFFIXES = {"video": ".mp4", "label": ".json", "audio": ".wav"}
COPY_BLOCK_BYTES = 4 * 1024 * 1024
DEFAULT_MINIMUM_FREE_BYTES = 10 * 1024**3


class TarSourceError(RuntimeError):
    """A source contract or bounded copy failed validation."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    """Accept strict nonnegative integers, including a configurable minimum."""
    try:
        if type(value) is not int or value < minimum:
            raise TarSourceError(f"{name} must be an integer >= {minimum}")
        return value
    except TarSourceError:
        raise


def _validate_member(stem: str, role: str, member: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an audited reference without opening its archive."""
    try:
        if role not in ROLE_SUFFIXES:
            raise TarSourceError(f"Unsupported source role: {role}")
        member_name = member["member_name"]
        if not isinstance(member_name, str) or not member_name:
            raise TarSourceError("member_name must be a nonempty string")
        parts = member_name.split("/")
        logical = PurePosixPath(member_name)
        if (logical.is_absolute() or PureWindowsPath(member_name).drive
                or "\\" in member_name or "\x00" in member_name
                or any(part in ("", ".", "..") for part in parts)
                or ":" in member_name):
            raise TarSourceError(f"Unsafe member path: {member_name!r}")
        output_name = stem + ROLE_SUFFIXES[role]
        if logical.name != output_name:
            raise TarSourceError(f"Member identity mismatch for {role}: {member_name}")
        archive_path = Path(member["archive_path"])
        if not archive_path.is_absolute() or archive_path.suffix.lower() != ".tar":
            raise TarSourceError("archive_path must name an absolute plain .tar file")
        archive_size = _integer(member["archive_size"], "archive_size", 1024)
        mtime_ns = _integer(member["archive_mtime_ns"], "archive_mtime_ns")
        offset = _integer(member["offset"], "offset", 512)
        size = _integer(member["size"], "size", 1)
        if archive_size % 512 or offset % 512:
            raise TarSourceError("Plain TAR archive size and member offset must be 512-byte aligned")
        if offset + size > archive_size:
            raise TarSourceError("Member range exceeds the audited archive size")
        digest = member["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise TarSourceError("sha256 must contain exactly 64 hexadecimal characters")
        return {"archive_path": archive_path, "archive_size": archive_size,
                "archive_mtime_ns": mtime_ns, "offset": offset, "size": size,
                "sha256": digest.lower(), "member_name": member_name,
                "output_name": output_name}
    except TarSourceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TarSourceError(f"Invalid {role} member specification: {exc}") from exc


def _snapshot(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    """Include filesystem identity as well as the audit's size and mtime."""
    try:
        return (stat_result.st_size, stat_result.st_mtime_ns,
                stat_result.st_dev, stat_result.st_ino)
    except AttributeError as exc:
        raise TarSourceError(f"Incomplete archive stat: {exc}") from exc


def _check_archive(member: Mapping[str, Any],
                   expected: tuple[int, int, int, int] | None = None,
                   stream: BinaryIO | None = None) -> tuple[int, int, int, int]:
    """Reject a changed archive path or a different file behind an open handle."""
    try:
        path = member["archive_path"]
        current = _snapshot(path.stat())
        if current[:2] != (member["archive_size"], member["archive_mtime_ns"]):
            raise TarSourceError(f"Archive stat differs from audited source: {path}")
        if expected is not None and current != expected:
            raise TarSourceError(f"Archive identity changed during materialization: {path}")
        if stream is not None and _snapshot(os.fstat(stream.fileno())) != current:
            raise TarSourceError(f"Open archive and archive path disagree: {path}")
        return current
    except TarSourceError:
        raise
    except OSError as exc:
        raise TarSourceError(f"Cannot validate archive {member['archive_path']}: {exc}") from exc


def _copy_member(member: Mapping[str, Any], destination: Path,
                 expected_stat: tuple[int, int, int, int]) -> None:
    """Copy only the approved byte range and verify the copied file's digest."""
    try:
        with member["archive_path"].open("rb") as source:
            _check_archive(member, expected_stat, source)
            source.seek(member["offset"])
            digest = hashlib.sha256()
            remaining = member["size"]
            with destination.open("xb") as target:
                while remaining:
                    block = source.read(min(COPY_BLOCK_BYTES, remaining))
                    if not block:
                        raise TarSourceError(f"Truncated archive member: {member['member_name']}")
                    target.write(block)
                    digest.update(block)
                    remaining -= len(block)
            _check_archive(member, expected_stat, source)
        if destination.stat().st_size != member["size"]:
            raise TarSourceError(f"Copied size differs for {member['member_name']}")
        if digest.hexdigest() != member["sha256"]:
            raise TarSourceError(f"SHA-256 mismatch for {member['member_name']}")
    except TarSourceError:
        raise
    except OSError as exc:
        raise TarSourceError(f"Cannot materialize {member['member_name']}: {exc}") from exc


@contextmanager
def materialize_source(source: Mapping[str, Any], temp_root: Path,
                       minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
                       ) -> Iterator[dict[str, Path]]:
    """Yield verified local video/label/audio paths, then delete only owned files.

    ``source`` contains ``stem`` and a nonempty ``members`` mapping. Each member
    contains archive_path, archive_size, archive_mtime_ns, member_name, offset,
    size and sha256. Supported roles are video, label and audio; subsets allow
    label-only inspection. Reserve space is checked in addition to the sum of
    simultaneous payload sizes. The parent temp directory is retained, while
    the unique child created here is always cleaned, including on caller errors.
    """
    try:
        stem = source["stem"]
        if not isinstance(stem, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
            raise TarSourceError("stem must be a safe recording basename")
        members = source["members"]
        if not isinstance(members, Mapping) or not members:
            raise TarSourceError("members must be a nonempty mapping")
        minimum_free_bytes = _integer(minimum_free_bytes, "minimum_free_bytes")
        validated = {role: _validate_member(stem, role, member)
                     for role, member in members.items()}
        snapshots = {role: _check_archive(member) for role, member in validated.items()}
        temp_root = Path(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        required = minimum_free_bytes + sum(member["size"] for member in validated.values())
        free = shutil.disk_usage(temp_root).free
        if free < required:
            raise TarSourceError(f"Insufficient temporary space: need {required} bytes, have {free}")
        with tempfile.TemporaryDirectory(prefix=f"{stem}-", dir=temp_root) as directory:
            paths: dict[str, Path] = {}
            for role, member in validated.items():
                if shutil.disk_usage(temp_root).free < minimum_free_bytes + member["size"]:
                    raise TarSourceError("Insufficient temporary space before copying the next member")
                destination = Path(directory) / member["output_name"]
                _copy_member(member, destination, snapshots[role])
                paths[role] = destination
            try:
                yield paths
            finally:
                for role, member in validated.items():
                    _check_archive(member, snapshots[role])
    except TarSourceError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise TarSourceError(f"Source materialization failed: {exc}") from exc
