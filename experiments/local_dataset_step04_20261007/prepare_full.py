"""Prepare the frozen 512 corrected inputs through the audited D: TAR adapter.

Eight accepted pilots remain references to their original bytes. New training
inputs retain the original teacher/student schema; development inputs persist
only student arrays. The unchanged producer briefly prepares paired pixels for
development correspondence checking, but no teacher model or features run here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STEP01 = ROOT / "experiments/local_source_step01_20261007"
STEP02 = ROOT / "experiments/local_pilot_step02_20261007"
STEP03 = ROOT / "experiments/local_training_step03_20261007"
sys.path.insert(0, str(STEP01))
from adapter import LocalSourceAdapter, code_bindings, sha256
from build_plan import load_plan

SOURCE_SHA = "d3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9"
STEP02_SHA = "c26a4b9b4a90ad553cacddecfd68fe22aad6835af507b9ce921ccfb6d5ec856e"
STEP03_SHA = "aa62fc9f6ad16f6d2b05843096baf2cd3b3f4332108f2883417f39fa76b224fd"
PILOT_SHA = "d807b5a67e109019b1cba82181dbba8c6204e736ecc78db09b5c73fbcf381fac"
OUTPUT = BASE / "inputs"
DEV_ARRAYS = {"student_rgb", "student_source_indices", "label_tokens"}
DEV_META = {"schema", "id", "record", "plan_sha256", "source_frame_count",
            "student_rgb_sha256", "student_trace", "source_verification",
            "preparation_seconds", "local_full_preparation"}


def binding(path: Path) -> dict:
    """Fingerprint a local experiment file."""
    try:
        path = path.resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Only workspace evidence can be bound")
        return {"path": str(path), "sha256": sha256(path)}
    except Exception as exc:
        raise RuntimeError(f"Cannot bind artifact {path}") from exc


def read_json(path: Path, expected: str | None = None) -> dict:
    """Read local JSON after optional exact-byte verification."""
    try:
        if expected is not None and binding(path)["sha256"] != expected:
            raise ValueError("Bound artifact hash changed")
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot accept JSON {path}") from exc


def check_binding(item: dict) -> None:
    """Verify a previously bound evidence or code file."""
    try:
        if set(item) != {"path", "sha256"} or binding(Path(item["path"])) != item:
            raise ValueError("Evidence binding changed")
    except Exception as exc:
        raise RuntimeError(f"Invalid binding {item.get('path')}") from exc


def accepted_evidence() -> tuple[list[dict], dict]:
    """Require completed, unchanged Step2/3 evidence and the fixed pilot."""
    try:
        two_path = STEP02 / "results/step02_summary.json"
        three_path = STEP03 / "results/step03_summary.json"
        two, three = read_json(two_path, STEP02_SHA), read_json(three_path, STEP03_SHA)
        if two["status"] != "completed" or three["status"] != "completed":
            raise ValueError("Previous steps are incomplete")
        evidence = [binding(two_path), binding(three_path), *two["artifacts"].values(),
                    three["preflight_plan"], three["replay_audit"],
                    three["independent_review"], three["future_comparison_recipe"]]
        for item in evidence:
            check_binding(item)
        pilot = read_json(STEP02 / "inputs/pilot_manifest.json", PILOT_SHA)
        if pilot["status"] != "passed" or pilot["phase"] != "pilot":
            raise ValueError("Pilot was not accepted")
        check_binding(pilot["preparation_runner"])
        check_binding(pilot["reproduction_gate"])
        return evidence, pilot
    except Exception as exc:
        raise RuntimeError("Full extraction requires accepted Step2/3 evidence") from exc


def validate_transport(adapter: LocalSourceAdapter, row: dict, meta: dict,
                       runner: dict, runner_key: str) -> None:
    """Check the saved producer and D transport against the frozen source map."""
    try:
        source = adapter.sources[row["source_stem"]]
        check = meta["source_verification"]
        expected = {"transport": "audited-local-tar-v1", "source_plan_path": str(adapter.plan_path),
                    "source_plan_sha256": adapter.plan_sha256,
                    "parent_plan_sha256": adapter.parent["sha256"], "source_stem": row["source_stem"],
                    "members": {role: source["members"][role] for role in ("video", "label")},
                    "transport_code": adapter.code, "audio_used_as_model_input": False,
                    "temporary_paths_persisted": False, "archive_stats_checked_before_and_after": True,
                    "member_hashes_checked_on_materialization": True}
        if meta[runner_key] != runner or set(check) != {*expected, "temporary_payload_verification"}:
            raise ValueError("Saved preparation provenance differs")
        for key, value in expected.items():
            if check[key] != value:
                raise ValueError(f"Saved D provenance differs: {key}")
        local = check["temporary_payload_verification"]
        if (set(local) != {"video_size_bytes", "video_mtime_ns", "label_sha256_before_and_after", "source_video_unchanged"}
                or local["video_size_bytes"] != source["members"]["video"]["size"]
                or local["label_sha256_before_and_after"] != row["label_sha256"]
                or local["source_video_unchanged"] is not True
                or type(local["video_mtime_ns"]) is not int or local["video_mtime_ns"] <= 0):
            raise ValueError("Saved temporary payload verification differs")
    except Exception as exc:
        raise RuntimeError(f"Reject transport for {row['id']}") from exc


def dev_value(row: dict, value: dict) -> dict:
    """Drop paired teacher arrays/traces after unchanged extraction verifies them."""
    try:
        if row["split"] != "dev":
            raise ValueError("Student-only storage is reserved for development")
        meta = {key: value["meta"][key] for key in DEV_META}
        meta["schema"] = "local-tar-step04-dev-input-v1"
        return {"arrays": {key: value["arrays"][key] for key in DEV_ARRAYS}, "meta": meta}
    except Exception as exc:
        raise RuntimeError(f"Cannot isolate development input {row['id']}") from exc


def verify_dev(path: Path, row: dict, plan_sha: str, producer: object) -> dict:
    """Validate student-only NPZ bytes, identity, labels and source-frame indices."""
    try:
        if row["split"] != "dev":
            raise ValueError("Development payload cannot stand in for training")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != DEV_ARRAYS | {"metadata_json"}:
                raise ValueError("Development payload has teacher or unknown arrays")
            meta = json.loads(str(payload["metadata_json"].item()))
            if (set(meta) != DEV_META or meta["schema"] != "local-tar-step04-dev-input-v1"
                    or meta["id"] != row["id"] or meta["record"] != row or meta["plan_sha256"] != plan_sha):
                raise ValueError("Development payload identity differs")
            contract = producer.interval_contract(row["start_time"], row["end_time"], row["video_fps"], meta["source_frame_count"])
            pixels, indices, tokens = (payload[key] for key in ("student_rgb", "student_source_indices", "label_tokens"))
            if pixels.dtype != np.uint8 or pixels.shape != (row["frames"], 96, 96, 3):
                raise ValueError("Development pixels have wrong shape/dtype")
            if hashlib.sha256(pixels.tobytes()).hexdigest() != meta["student_rgb_sha256"]:
                raise ValueError("Development pixels changed")
            if indices.dtype != np.int64 or not np.array_equal(indices, contract["student_indices"]):
                raise ValueError("Development source frame indices changed")
            if tokens.dtype != np.int64 or tokens.tolist() != row["token_ids"]:
                raise ValueError("Development labels changed")
            trace = meta["student_trace"]["selected"]
            if len(trace) != row["frames"] or [x["source_frame_index"] for x in trace] != indices.tolist():
                raise ValueError("Development source trace changed")
        return meta
    except Exception as exc:
        raise RuntimeError(f"Invalid development NPZ {path}") from exc


def dev_entry(path: Path, meta: dict) -> dict:
    """Build the student-only sidecar using the common manifest fields."""
    try:
        row = meta["record"]
        return {"id": meta["id"], "split": row["split"], "source_stem": row["source_stem"],
                "plan_sha256": meta["plan_sha256"], "npz_path": str(path.resolve()), "npz_sha256": sha256(path),
                "student_frames": row["frames"], "teacher_frames": 0,
                "student_rgb_sha256": meta["student_rgb_sha256"], "source_frame_count": meta["source_frame_count"],
                "video_fps": row["video_fps"], "source_verification": meta["source_verification"],
                "preparation_seconds": meta["preparation_seconds"]}
    except Exception as exc:
        raise RuntimeError("Cannot describe development artifact") from exc


def saved_record(output: Path, row: dict, adapter: LocalSourceAdapter,
                 runner: dict, expected: dict | None = None) -> dict | None:
    """Resume only fully validated bytes; pilot references must remain identical."""
    try:
        path = output / "records" / f"{row['id']}.npz"
        sidecar = output / "metadata" / f"{row['id']}.json"
        if not path.exists():
            if sidecar.exists() or expected is not None:
                raise ValueError("Required payload is missing")
            return None
        producer = adapter.producer
        if row["split"] == "dev":
            meta = verify_dev(path, row, adapter.parent["sha256"], producer)
            entry = dev_entry(path, meta)
        else:
            meta = producer.verify_payload(path, row, adapter.parent["sha256"])
            entry = producer.metadata_record(path, meta)
        runner_key = "local_pilot_preparation" if expected is not None else "local_full_preparation"
        validate_transport(adapter, row, meta, runner, runner_key)
        if expected is not None and entry != expected:
            raise ValueError("Accepted pilot dictionary or NPZ changed")
        if sidecar.exists():
            if read_json(sidecar) != entry:
                raise ValueError("Saved sidecar/hash differs")
        elif expected is not None:
            raise ValueError("Accepted pilot sidecar missing")
        else:
            producer.json_new(sidecar, entry)
        return entry
    except Exception as exc:
        raise RuntimeError(f"Cannot resume {row['id']}") from exc


def write_dev(output: Path, row: dict, value: dict, producer: object) -> dict:
    """Atomically publish a checked development payload without overwriting."""
    try:
        path = output / "records" / f"{row['id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(f".pending-{os.getpid()}.npz")
        with pending.open("xb") as stream:
            np.savez_compressed(stream, **value["arrays"], metadata_json=json.dumps(value["meta"], ensure_ascii=False, allow_nan=False))
        verify_dev(pending, row, value["meta"]["plan_sha256"], producer)
        if path.exists():
            raise FileExistsError(path)
        pending.rename(path)
        entry = dev_entry(path, value["meta"])
        producer.json_new(output / "metadata" / f"{row['id']}.json", entry)
        return entry
    except Exception as exc:
        raise RuntimeError(f"Cannot publish development input {row['id']}") from exc


def prepare() -> dict:
    """Prepare/resume 504 new inputs source-by-source and reference eight pilots."""
    try:
        started = time.perf_counter()
        adapter = LocalSourceAdapter(STEP01 / "d_source_plan.json", SOURCE_SHA)
        producer, rows = adapter.producer, adapter.logical_plan["records"]
        evidence, pilot = accepted_evidence()
        runner = binding(Path(__file__))
        if (pilot["plan_sha256"] != adapter.parent["sha256"]
                or pilot["ids"] != adapter.logical_plan["preparation"]["pilot_ids"]
                or [r["id"] for r in pilot["records"]] != pilot["ids"]
                or Counter(r["split"] for r in rows) != {"train": 441, "dev": 71}):
            raise ValueError("Frozen cohort/pilot changed")
        execution = {"schema": "local-tar-step04-input-plan-v1", "parent_plan": adapter.parent,
                     "source_plan": binding(adapter.plan_path), "preparation_runner": runner,
                     "transport_code": adapter.code, "accepted_evidence": evidence,
                     "pilot_manifest": binding(STEP02 / "inputs/pilot_manifest.json"),
                     "reproduction_gate": pilot["reproduction_gate"],
                     "ids": [r["id"] for r in rows], "reused_pilot_ids": pilot["ids"],
                     "output_directory": str(OUTPUT), "environment": adapter.logical_plan["preparation"]["environment"],
                     "minimum_free_bytes": 10 * 1024**3,
                     "dev_storage": "student_rgb,student_source_indices,label_tokens,metadata_json only; teacher_frames=0",
                     "dev_intermediate": "unchanged paired-pixel producer verifies correspondence; teacher arrays/traces discarded before saving; no teacher model/features",
                     "scope": "512 student inputs, paired teacher pixels only for train441; no audio/test, models, features, training or service changes"}
        plan_path = BASE / "input_plan.json"
        if plan_path.exists():
            if read_json(plan_path) != execution:
                raise ValueError("Full preparation plan changed; preserve prior artifacts")
        else:
            producer.json_new(plan_path, execution)
        plan_binding = binding(plan_path)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        completed, groups = {}, defaultdict(list)
        pilots = {entry["id"]: entry for entry in pilot["records"]}
        for row in rows:
            if row["id"] in pilots:
                existing = saved_record(STEP02 / "inputs", row, adapter, pilot["preparation_runner"], pilots[row["id"]])
            else:
                existing = saved_record(OUTPUT, row, adapter, runner)
            if existing is None:
                groups[row["source_stem"]].append(row)
            else:
                completed[row["id"]] = existing
        resumed_count = len(completed) - len(pilots)
        timings = []
        for number, (stem, group) in enumerate(groups.items(), 1):
            free_before = shutil.disk_usage(OUTPUT).free
            if free_before < execution["minimum_free_bytes"]:
                raise ValueError("Less than 10 GiB free before another source")
            source_started = time.perf_counter()
            values = adapter.prepare_source(group, reproduction=False, include_audio=False)
            for row, value in values:
                value["meta"]["local_full_preparation"] = runner
                validate_transport(adapter, row, value["meta"], runner, "local_full_preparation")
                if row["split"] == "dev":
                    entry = write_dev(OUTPUT, row, dev_value(row, value), producer)
                else:
                    entry = producer.write_record(OUTPUT, row, value)
                completed[row["id"]] = entry
            timing = {"source_stem": stem, "new_records": len(group), "split": group[0]["split"],
                      "elapsed_seconds": time.perf_counter() - source_started,
                      "free_bytes_before": free_before, "free_bytes_after": shutil.disk_usage(OUTPUT).free}
            timings.append(timing)
            producer.json_progress(OUTPUT / "progress.json", {"phase": "all", "plan_sha256": adapter.parent["sha256"],
                "preparation_plan": plan_binding, "completed": len(completed), "target": 512,
                "reused_pilot_count": 8, "resumed_new_count": resumed_count, "sources_processed_this_run": number,
                "last_source": stem, "per_source_execution": timings, "elapsed_seconds": time.perf_counter() - started})
            print(f"[full inputs] source {number}/{len(groups)} {stem} saved={len(completed)}/512 split={group[0]['split']}", flush=True)
            del values
        if load_plan(adapter.plan_path, SOURCE_SHA) != adapter.plan:
            raise ValueError("D source connection changed during extraction")
        producer.validate_plan(Path(adapter.parent["path"]), adapter.parent["sha256"])
        if code_bindings() != adapter.code or binding(Path(__file__)) != runner:
            raise ValueError("Preparation implementation changed during execution")
        for item in evidence:
            check_binding(item)
        if list(adapter.temp_root.iterdir()):
            raise ValueError("Temporary source payload remains")
        records = [completed[row["id"]] for row in rows]
        result = {"schema": "feature-transfer-step07-input-manifest-v1", "status": "passed", "phase": "all",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": adapter.parent["sha256"],
                  "source_plan": binding(adapter.plan_path), "preparation_plan": plan_binding,
                  "preparation_runner": runner, "transport_code": adapter.code,
                  "reproduction_gate": pilot["reproduction_gate"], "pilot_manifest": execution["pilot_manifest"],
                  "record_count": len(records), "ids": execution["ids"], "records": records,
                  "student_frames": sum(r["student_frames"] for r in records),
                  "teacher_frames": sum(r["teacher_frames"] for r in records), "source_recordings": 59,
                  "split_counts": {"train": 441, "dev": 71}, "reused_pilot_count": 8,
                  "new_train_input_count": 433, "student_only_dev_count": 71,
                  "elapsed_seconds": time.perf_counter() - started, "per_source_execution": timings,
                  "reproduction_gate_sha256": pilot["reproduction_gate"]["sha256"],
                  "audio_read": False, "test_payloads_read": False, "G_drive_accessed": False,
                  "original_files_modified": False, "teacher_model_run": False, "student_model_run": False,
                  "temporary_source_payloads_remaining": 0, "dev_intermediate": execution["dev_intermediate"],
                  "input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in records),
                  "new_input_npz_bytes": sum(Path(r["npz_path"]).stat().st_size for r in records if r["id"] not in pilots)}
        destination = OUTPUT / "manifest.json"
        if destination.exists():
            prior = read_json(destination)
            for key in ("records", "ids", "source_plan", "preparation_plan", "preparation_runner", "transport_code", "pilot_manifest"):
                if prior[key] != result[key]:
                    raise ValueError(f"Accepted full manifest changed: {key}")
            return prior
        producer.json_new(destination, result)
        return result
    except Exception as exc:
        raise RuntimeError("Step4 input preparation failed; incomplete inputs must not be accepted") from exc


def main() -> None:
    """Run full input preparation, without any model or training invocation."""
    try:
        argparse.ArgumentParser(description=__doc__).parse_args()
        result = prepare()
        print(json.dumps({key: result[key] for key in ("status", "record_count", "student_frames", "teacher_frames", "elapsed_seconds", "input_npz_bytes")}), flush=True)
    except Exception:
        raise


if __name__ == "__main__":
    main()
