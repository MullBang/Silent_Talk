"""Prepare only the eight frozen training pilots through audited D: transport.

The prior logical plan and original preprocessing implementation remain intact.
The new execution plan pins the D source plan and this runner before extraction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP01 = ROOT / "experiments/local_source_step01_20261007"
sys.path.insert(0, str(STEP01))
from adapter import LocalSourceAdapter, code_bindings, sha256
from build_plan import load_plan

SOURCE_SHA = "d3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9"
REPRODUCTION_SHA = "173ea79a90c22f59ec5294d423f10c1462192415ba7e9fa5ccce35261343d739"
OUTPUT = BASE / "inputs"


def binding(path: Path) -> dict:
    """Record a local artifact's exact bytes."""
    try:
        return {"path": str(path.resolve()), "sha256": sha256(path)}
    except OSError as exc:
        raise RuntimeError(f"Cannot bind pilot artifact: {path}") from exc


def read_json(path: Path) -> dict:
    """Read experiment JSON with a contextual error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read pilot artifact: {path}") from exc


def verified_gate(adapter: LocalSourceAdapter) -> dict:
    """Require the accepted D reproduction evidence before generating inputs."""
    try:
        path = STEP01 / "results/reproduction_gate.json"
        gate_binding = binding(path)
        if gate_binding["sha256"] != REPRODUCTION_SHA:
            raise ValueError("The accepted D reproduction gate changed")
        gate = read_json(path)
        if (gate["status"] != "passed" or gate["parent_plan"] != adapter.parent
                or gate["source_plan_sha256"] != adapter.plan_sha256
                or gate["ids"] != adapter.logical_plan["preparation"]["reproduction_ids"]
                or gate["transport_code"] != adapter.code):
            raise ValueError("Reproduction gate belongs to another source/producer")
        return gate_binding
    except Exception as exc:
        raise RuntimeError("Pilot requires the accepted Step1 reproduction gate") from exc


def validate_saved_transport(adapter: LocalSourceAdapter, row: dict, meta: dict,
                             runner: dict) -> None:
    """Refuse a resumed input made by different code or from a different source."""
    try:
        source = adapter.sources[row["source_stem"]]
        check = meta["source_verification"]
        members = {role: source["members"][role] for role in ("video", "label")}
        if (meta["local_pilot_preparation"] != runner
                or check["transport"] != "audited-local-tar-v1"
                or check["source_plan_path"] != str(adapter.plan_path)
                or check["source_plan_sha256"] != adapter.plan_sha256
                or check["parent_plan_sha256"] != adapter.parent["sha256"]
                or check["source_stem"] != row["source_stem"]
                or check["members"] != members or check["transport_code"] != adapter.code
                or check["audio_used_as_model_input"] is not False
                or check["temporary_paths_persisted"] is not False
                or check["archive_stats_checked_before_and_after"] is not True
                or check["member_hashes_checked_on_materialization"] is not True):
            raise ValueError("Saved pilot provenance differs")
        local = check["temporary_payload_verification"]
        if (local["video_size_bytes"] != members["video"]["size"]
                or local["label_sha256_before_and_after"] != row["label_sha256"]
                or local["source_video_unchanged"] is not True
                or type(local["video_mtime_ns"]) is not int):
            raise ValueError("Saved materialization verification differs")
    except Exception as exc:
        raise RuntimeError(f"Reject saved pilot input {row['id']}") from exc


def prepare() -> dict:
    """Generate/resume the eight input payloads without model inference."""
    try:
        started = time.perf_counter()
        adapter = LocalSourceAdapter(STEP01 / "d_source_plan.json", SOURCE_SHA)
        producer = adapter.producer
        runner = binding(Path(__file__))
        gate = verified_gate(adapter)
        selected_ids = adapter.logical_plan["preparation"]["pilot_ids"]
        rows = [adapter.rows[key] for key in selected_ids]
        if (len(rows) != 8 or len({r["speaker_id"] for r in rows}) != 8
                or any(r["split"] != "train" for r in rows)):
            raise ValueError("Exactly one fixed training pilot per eight speakers required")
        execution = {
            "schema": "local-tar-step02-input-plan-v1",
            "parent_plan": adapter.parent, "source_plan": binding(adapter.plan_path),
            "preparation_runner": runner, "transport_code": adapter.code,
            "reproduction_gate": gate, "ids": selected_ids,
            "output_directory": str(OUTPUT),
            "environment": adapter.logical_plan["preparation"]["environment"],
            "scope": "fixed8 train inputs only; no audio, dev/test inputs, models or training",
            "source_strategy": "materialize one source video and label, SHA verify, prepare and clean",
        }
        plan_path = BASE / "input_plan.json"
        if plan_path.exists():
            if read_json(plan_path) != execution:
                raise ValueError("Existing preparation plan differs; retain accepted artifacts")
        else:
            producer.json_new(plan_path, execution)
        plan_binding = binding(plan_path)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        completed, timings = [], []
        for row in rows:
            existing = producer.existing_record(OUTPUT, row, adapter.parent["sha256"])
            if existing is not None:
                meta = producer.verify_payload(Path(existing["npz_path"]), row, adapter.parent["sha256"])
                validate_saved_transport(adapter, row, meta, runner)
                completed.append(existing)
                print(f"[resumed input] {row['id']}", flush=True)
                continue
            free_before = shutil.disk_usage(OUTPUT).free
            source_started = time.perf_counter()
            values = adapter.prepare_source([row], reproduction=False, include_audio=False)
            logical, value = values[0]
            value["meta"]["local_pilot_preparation"] = runner
            validate_saved_transport(adapter, logical, value["meta"], runner)
            entry = producer.write_record(OUTPUT, logical, value)
            completed.append(entry)
            timings.append({"id": row["id"], "materialization_preparation_save_seconds": time.perf_counter() - source_started,
                            "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(OUTPUT).free})
            print(f"[pilot input] {len(completed)}/8 {row['id']} student={entry['student_frames']} teacher={entry['teacher_frames']}", flush=True)
        if load_plan(adapter.plan_path, SOURCE_SHA) != adapter.plan:
            raise ValueError("D source connection changed during pilot")
        producer.validate_plan(Path(adapter.parent["path"]), adapter.parent["sha256"])
        if code_bindings() != adapter.code or binding(Path(__file__)) != runner:
            raise ValueError("Pilot implementation changed during execution")
        if list(adapter.temp_root.iterdir()):
            raise ValueError("Temporary source payload remains")
        sheet = producer.contact_sheet(OUTPUT, rows)
        result = {
            "schema": "feature-transfer-step07-input-manifest-v1", "status": "passed", "phase": "pilot",
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": adapter.parent["sha256"],
            "source_plan": binding(adapter.plan_path), "preparation_plan": plan_binding,
            "preparation_runner": runner, "transport_code": adapter.code, "reproduction_gate": gate,
            "record_count": len(completed), "ids": selected_ids, "records": completed,
            "student_frames": sum(r["student_frames"] for r in completed),
            "teacher_frames": sum(r["teacher_frames"] for r in completed),
            "source_recordings": len({r["source_stem"] for r in completed}),
            "elapsed_seconds": time.perf_counter() - started, "per_source_execution": timings,
            "contact_sheet": binding(sheet),
            "reproduction_gate_sha256": gate["sha256"], "audio_read": False, "test_payloads_read": False,
            "G_drive_accessed": False, "original_files_modified": False,
            "teacher_model_run": False, "student_model_run": False,
            "temporary_source_payloads_remaining": 0,
            "input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in completed),
        }
        destination = OUTPUT / "pilot_manifest.json"
        if destination.exists():
            prior = read_json(destination)
            for key in ("records", "ids", "source_plan", "preparation_plan", "preparation_runner", "transport_code", "reproduction_gate", "contact_sheet"):
                if prior[key] != result[key]:
                    raise ValueError(f"Accepted pilot manifest changed: {key}")
            return prior
        producer.json_new(destination, result)
        return result
    except Exception as exc:
        raise RuntimeError("Step2 input pilot failed; do not accept an incomplete manifest") from exc


def main() -> None:
    """Run this explicitly scoped pilot command."""
    try:
        argparse.ArgumentParser(description=__doc__).parse_args()
        result = prepare()
        print(json.dumps({key: result[key] for key in ("status", "record_count", "student_frames", "teacher_frames", "elapsed_seconds")}), flush=True)
    except Exception:
        raise


if __name__ == "__main__":
    main()
