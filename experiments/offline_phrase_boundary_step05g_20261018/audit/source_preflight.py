"""Verify only the immediate frozen stage 5f source set for a Q01 revision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "experiments/offline_phrase_review_step05f_20261018"
MANIFEST_SHA = "153d12419b93790be72d1fdb91230d40510d00bd54e3f7dbcbf791f82a042281"


def digest(path: Path) -> str:
    """Hash one file with bounded memory."""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_json(path: Path) -> Any:
    """Read only a repository JSON."""
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {path}")
    return json.loads(path.read_text(encoding="utf8"))


def bind(path: Path) -> dict[str, Any]:
    """Record a local immutable byte identity."""
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def check(binding: dict[str, Any]) -> None:
    """Fail on a changed or missing binding."""
    path = Path(binding["path"])
    if not path.is_file() or digest(path) != binding["sha256"]:
        raise AssertionError(f"Binding mismatch: {path}")
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise AssertionError(f"Size mismatch: {path}")


def inspect() -> dict[str, Any]:
    """Verify the immediate 247 frozen entries without historical re-expansion."""
    path = PREVIOUS / "results/completion_manifest.json"
    if digest(path) != MANIFEST_SHA:
        raise AssertionError("Unexpected stage 5f completion manifest")
    rows = read_json(path)["artifacts"]
    assert len(rows) == len({r["path"] for r in rows}) == 247
    for row in rows:
        check(row)
    return {
        "schema": "offline-stage5g-independent-source-preflight-v1",
        "status": "passed", "stage5f_manifest": bind(path),
        "frozen_entries_verified": len(rows),
        "frozen_bytes_verified": sum(r["bytes"] for r in rows),
        "historical_1910_entries_reexpanded": False,
        "TAR_payload_reads": 0, "neural_calls": 0, "training_updates": 0,
        "scope": "Stage 5f's 247 frozen artifacts only; current Q01 answer and new candidate are separate stage 5g evidence.",
    }


if __name__ == "__main__":
    result = inspect()
    with (HERE / "source_preflight.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "frozen_entries": result["frozen_entries_verified"]}))
