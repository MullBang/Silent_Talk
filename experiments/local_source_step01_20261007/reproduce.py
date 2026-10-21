"""Verify the four frozen original-time controls through D: TAR transport.

This step does not generate a pilot, teacher features, or training checkpoints.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from adapter import BASE, LocalSourceAdapter, code_bindings, sha256
from build_plan import load_plan


def reproduce(plan_path: Path, expected_sha256: str) -> dict:
    """Accept a new gate only after all payload and golden-cache checks pass."""
    try:
        destination = BASE / "results" / "reproduction_gate.json"
        if destination.exists():
            raise FileExistsError(f"Keep the accepted evidence unchanged: {destination}")
        started = time.perf_counter()
        adapter = LocalSourceAdapter(plan_path, expected_sha256)
        initial_code = code_bindings()
        runner = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}
        selected = adapter.logical_plan["preparation"]["reproduction_ids"]
        free_before = shutil.disk_usage(BASE).free
        controls = []
        for record_id in selected:
            row = adapter.rows[record_id]
            values = adapter.prepare_source([row], reproduction=True, include_audio=True)
            controls.append(values[0][1])
            print(f"[D reproduced] {record_id}: {controls[-1]['frames']} frames pixel-exact", flush=True)
        if load_plan(plan_path, expected_sha256) != adapter.plan:
            raise ValueError("Source connection plan changed")
        adapter.producer.validate_plan(Path(adapter.parent["path"]), adapter.parent["sha256"])
        if code_bindings() != initial_code or sha256(Path(__file__)) != runner["sha256"]:
            raise ValueError("Verification implementation changed")
        if list(adapter.temp_root.iterdir()):
            raise ValueError("Unexpected temporary files remain; inspect before accepting")
        result = {
            "schema": "local-tar-step01-reproduction-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_plan_path": str(plan_path.resolve()), "source_plan_sha256": expected_sha256,
            "parent_plan": adapter.parent, "transport_code": initial_code, "runner": runner,
            "ids": selected, "records": controls,
            "frames": sum(control["frames"] for control in controls),
            "logical_record_count": len(adapter.rows),
            "logical_split_counts": dict(Counter(row["split"] for row in adapter.rows.values())),
            "mapped_source_count": len(adapter.sources),
            "materialized_source_count": len(selected),
            "materialized_member_count": sum(len(c["source_verification"]["members"]) for c in controls),
            "materialized_bytes": sum(m["size"] for c in controls for m in c["source_verification"]["members"].values()),
            "maximum_temporary_payload_bytes": max(sum(m["size"] for m in c["source_verification"]["members"].values()) for c in controls),
            "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(BASE).free,
            "temporary_payloads_remaining": 0,
            "original_times_used_for_golden_reproduction": True,
            "corrected_pilot_inputs_created": False, "teacher_features_created": False,
            "audio_read_for_hash_verification": True, "audio_used_as_model_input": False,
            "G_drive_accessed": False, "test_payloads_read": False,
            "model_inference_or_training_run": False, "original_files_modified": False,
            "elapsed_seconds": time.perf_counter() - started,
            "limits": [
                "Mapping 59 recordings reuses the Oct2 audit plus current archive size/mtime; all 59 payloads were not rehashed today.",
                "Four recordings received fresh full member hashes and four sentences received exact original-time pixel reproduction.",
                "Reproduction does not validate all corrected-time inputs, transcript truth, webcam accuracy or teacher transfer efficacy.",
            ],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as exc:
        raise RuntimeError("Step1 reproduction did not publish a new accepted gate") from exc


def main() -> None:
    """Run only the Step1 reproduction gate with an explicit plan fingerprint."""
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, default=BASE / "d_source_plan.json")
        parser.add_argument("--source-plan-sha256", required=True)
        args = parser.parse_args()
        result = reproduce(args.plan, args.source_plan_sha256)
        print(json.dumps({key: result[key] for key in ("status", "frames", "materialized_member_count", "elapsed_seconds")}), flush=True)
    except Exception:
        raise


if __name__ == "__main__":
    main()
