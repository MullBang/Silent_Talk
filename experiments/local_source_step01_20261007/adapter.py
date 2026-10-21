"""Read audited local TAR members while preserving the frozen logical records.

The original step07 producer is imported unchanged. Temporary paths exist only
inside one source context; persisted metadata identifies the D: TAR transport.
The original G: paths are historical record identifiers, never opened here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

from build_plan import load_plan
from tar_source import materialize_source

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def sha256(path: Path) -> str:
    """Hash a local producer or manifest in bounded blocks."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise OSError(f"Cannot fingerprint {path}") from exc


def load_producer() -> ModuleType:
    """Import the frozen step07 producer without modifying its globals."""
    try:
        path = ROOT / "experiments/feature_transfer_step07/prepare_inputs.py"
        spec = importlib.util.spec_from_file_location("step01_frozen_step07_producer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise RuntimeError("Cannot import frozen input producer") from exc


def code_bindings() -> list[dict]:
    """Bind the transport implementation separately from the original producer."""
    try:
        return [{"path": str(BASE / name), "sha256": sha256(BASE / name)}
                for name in ("adapter.py", "build_plan.py", "tar_source.py")]
    except OSError as exc:
        raise RuntimeError("Cannot bind D transport implementation") from exc


class LocalSourceAdapter:
    """Source-at-a-time transport; the immutable source plan is the authority."""

    def __init__(self, plan_path: Path, expected_sha256: str,
                 temp_root: Path | None = None) -> None:
        try:
            self.plan_path = plan_path.resolve()
            self.plan_sha256 = expected_sha256
            self.plan = load_plan(self.plan_path, expected_sha256)
            self.parent = self.plan["parent_plan"]
            self.producer = load_producer()
            self.logical_plan = self.producer.validate_plan(
                Path(self.parent["path"]), self.parent["sha256"])
            if self.plan["records"] != self.logical_plan["records"]:
                raise ValueError("Transport plan changed frozen logical records")
            self.rows = {row["id"]: row for row in self.plan["records"]}
            self.sources = {source["stem"]: source for source in self.plan["sources"]}
            self.temp_root = (temp_root or BASE / "tmp").resolve()
            if not self.temp_root.is_relative_to(BASE):
                raise ValueError("Temporary source copies must remain in this experiment")
            self.code = code_bindings()
        except Exception as exc:
            raise RuntimeError("D source adapter initialization failed") from exc

    def prepare_source(self, rows: list[dict], *, reproduction: bool,
                       include_audio: bool = False) -> list[tuple[dict, dict]]:
        """Reuse preprocessing with local paths, then restore logical provenance.

        Audio is optionally hashed for source verification; the input producer
        receives only video and labels. It never receives audio samples.
        """
        try:
            if not rows or len({row["id"] for row in rows}) != len(rows):
                raise ValueError("A nonempty unique source batch is required")
            if any(self.rows.get(row["id"]) != row for row in rows):
                raise ValueError("Only exact frozen records may be prepared")
            stems = {row["source_stem"] for row in rows}
            if len(stems) != 1:
                raise ValueError("Materialize exactly one source recording at a time")
            source = copy.deepcopy(self.sources[next(iter(stems))])
            if not include_audio:
                source["members"].pop("audio")
            # Do not expose caller-owned rows to a legacy producer.
            logical = copy.deepcopy(rows)
            with materialize_source(source, self.temp_root) as paths:
                state = paths["video"].stat()
                execution_rows = []
                for row in logical:
                    execution = copy.deepcopy(row)
                    execution.update(video_path=str(paths["video"]),
                                     label_path=str(paths["label"]),
                                     video_size_bytes=state.st_size,
                                     video_mtime_ns=state.st_mtime_ns)
                    execution_rows.append(execution)
                values = self.producer.prepare_source(
                    execution_rows, self.parent["sha256"], reproduction=reproduction)
                if [row["id"] for row, _ in values] != [row["id"] for row in logical]:
                    raise ValueError("Producer changed the source batch order or identity")
                prepared = []
                for frozen, (_, value) in zip(logical, values, strict=True):
                    metadata = value if reproduction else value["meta"]
                    local_verification = metadata["source_verification"]
                    if not reproduction:
                        metadata["record"] = frozen
                    metadata["source_verification"] = {
                        "transport": "audited-local-tar-v1",
                        "source_plan_path": str(self.plan_path),
                        "source_plan_sha256": self.plan_sha256,
                        "parent_plan_sha256": self.parent["sha256"],
                        "source_stem": source["stem"],
                        "members": copy.deepcopy(source["members"]),
                        "temporary_payload_verification": local_verification,
                        "temporary_paths_persisted": False,
                        "archive_stats_checked_before_and_after": True,
                        "member_hashes_checked_on_materialization": True,
                        "audio_used_as_model_input": False,
                        "transport_code": self.code,
                    }
                    prepared.append((frozen, value))
            # Context exit rechecks source archives and cleans temporary copies.
            if code_bindings() != self.code:
                raise ValueError("Transport implementation changed during preparation")
            return prepared
        except Exception as exc:
            raise RuntimeError("D source preparation failed; no result accepted") from exc
