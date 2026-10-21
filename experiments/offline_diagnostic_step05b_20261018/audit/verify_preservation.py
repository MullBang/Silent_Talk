"""Read-only verification of the immutable source snapshots for Stage 5b."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFESTS = (
    ("offline_dataset_step03_20261013", 114, "20b783427fbb7c43a61be73227e3a22b20650a2b3b80624ee115051a6afeea1f"),
    ("offline_training_step04_20261013", 101, "c6f7cd6569d9885d030376dbd95b8a24d39f1109ac68d46733f05c294e11dd1f"),
    ("offline_comparison_step05_20261013", 1238, "30d82e9ee65631ed1ec4e11961d7d45610ec13ac21c6f3708c0539a947fb9807"),
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binding(path):
    p = Path(path)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def check(row):
    p = Path(row["path"])
    assert p.resolve().is_relative_to(ROOT.resolve()), str(p)
    if row.get("exists") is False:
        assert not p.exists(), "Unexpected creation: " + str(p)
        return
    assert p.is_file(), "Missing source: " + str(p)
    assert sha(p) == row["sha256"], "SHA mismatch: " + str(p)
    assert p.stat().st_size == row["bytes"], "Size mismatch: " + str(p)


def verify():
    manifests, all_paths = [], set()
    for stage, count, expected_sha in MANIFESTS:
        p = ROOT / "experiments" / stage / "results/completion_manifest.json"
        assert sha(p) == expected_sha, "Manifest mismatch: " + str(p)
        value = read(p)
        rows = value["artifacts"]
        assert len(rows) == len({r["path"] for r in rows}) == count
        for row in rows:
            check(row)
            all_paths.add(row["path"])
        manifests.append({**binding(p), "verified_artifacts": count,
                          "verified_artifact_bytes": sum(r["bytes"] for r in rows)})
    scope_path = ROOT / "experiments/offline_dataset_step03_20261013/results/scope_snapshot.json"
    assert sha(scope_path) == "a64470d1eb50a45187d9b51f6871e5620dd110b5a0582a9ac1d4ef73d356f144"
    scope = read(scope_path)["artifacts"]
    assert len(scope) == 10
    for row in scope:
        check(row)
    handoff_path = ROOT / "experiments/offline_dataset_step03_20261013/results/cache_handoff.json"
    records = read(handoff_path)["records"]
    assert len(records) == 30
    assert all(r["provisional"] is True and r["training_ready"] is False for r in records)
    assert sum(r["boundary_followup_flag"] for r in records) == 2
    assert {r["split"] for r in records} == {"train", "dev"}
    assert sum(r["split"] == "train" for r in records) == 24
    assert sum(r["split"] == "dev" for r in records) == 6
    assert not any("C293_A_012_019" in r["parent_id"] for r in records)
    return {
        "schema": "offline-stage5b-independent-preservation-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "manifests": manifests,
        "manifest_artifact_entries": sum(r["verified_artifacts"] for r in manifests),
        "distinct_manifest_artifact_paths": len(all_paths),
        "scope": {**binding(scope_path), "verified_entries": 10,
                  "preserved_absent_paths": [r["path"] for r in scope if r.get("exists") is False]},
        "handoff": binding(handoff_path), "cache_records": 30,
        "provisional_preserved": True, "training_ready_preserved_false": True,
        "boundary_followup_flags_preserved": 2, "OOV_parent_excluded": True,
        "checkpoint_processing": "bytes read for SHA-256 only; no tensor deserialization",
        "neural_forward_calls": 0, "optimizer_updates": 0, "raw_TAR_reads": 0,
        "new_downloads": 0, "source_mutations": 0, "producer_imports": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="initial_preservation.json")
    args = parser.parse_args()
    output = (HERE / args.output).resolve()
    assert output.parent == HERE.resolve() and not output.exists()
    result = verify()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "manifest_artifact_entries": result["manifest_artifact_entries"],
                      "scope_entries": result["scope"]["verified_entries"], "output": str(output)}))
