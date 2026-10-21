"""Reevaluate both fixed legacy endpoints after a strict historical replay gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OFFICIAL = ROOT / "experiments/olkavs_official"
HISTORY = ROOT / "results/step17-expansion/run01"
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts"), str(OFFICIAL)]
import diagnose_temporal_use as legacy
from models.cache import load_clip
from models.registry import load_model_checkpoint
from evaluation_metrics import counts, aggregate

BRANCHES = ["A_keep256", "B_expand512"]
ARMS = ["old_timing", "aligned_timing"]
METHODS = ["greedy", "beam10"]


def require(ok: bool, message: str) -> None:
    """Stop on provenance or replay failure without relaxing the protocol."""
    if not ok:
        raise ValueError(message)


def sha(path: Path | str) -> str:
    """Hash a file without loading the entire content into RAM."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    """Read an existing UTF-8 evidence object."""
    return json.loads(path.read_text(encoding="utf8"))


def write(path: Path, value: dict, *, exclusive: bool = True) -> None:
    """Write new evidence or atomically replace this run's progress checkpoint."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf8") as stream:
            stream.write(text)
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text, encoding="utf8")
        temp.replace(path)


def evidence(path: Path) -> dict:
    """Bind an existing file by absolute location and digest."""
    return {"path": str(path.resolve()), "sha256": sha(path)}


def environment() -> dict:
    """Describe the same numerical runtime fields pinned in stage17."""
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "numpy": np.__version__, "python": sys.version}


def freeze() -> None:
    """Freeze all20 targets, both endpoints, source code and prepared inputs."""
    require(not (BASE / "eval_plan.json").exists(), "Evaluation plan already exists")
    previous = read(HISTORY / "plan.json")
    completed = read(HISTORY / "completed.json")
    require(completed["status"] == "completed" and completed["plan_sha256"] == sha(HISTORY / "plan.json"), "Historical plan changed")
    require(environment() == completed["environment"], "Historical runtime differs")
    for name, digest in previous["source_hashes"].items():
        require(sha(ROOT / name) == digest, f"Historical implementation changed: {name}")
    quality = read(HISTORY / "quality_checks.json")
    require(quality["status"] == "passed", "Historical quality audit not passed")
    source = read(OFFICIAL / "step03_source_plan.json")
    sidecar = read(OFFICIAL / "results/step04_audio_offset_candidates.json")
    require(sidecar["status"] == "validated", "Alignment not validated")
    manifest = read(BASE / "inputs/manifest.json")
    require(manifest["status"] == "completed", "Incomplete input preparation")
    require(manifest["reproduction_gate"]["passed"], "Original-input replay gate failed")
    source_ids = [r["id"] for r in source["records"]]
    require(len(set(source_ids)) == 20 and source_ids == previous["cohorts"]["val20"], "Different development cohort/order")
    old_sources = {r["id"]: r for r in previous["records"] if r["id"] in source_ids}
    for row in source["records"]:
        old = old_sources[row["id"]]
        require(row["split"] == old["split"] == "val" and
                all(row[k] == old[k] for k in ["text", "source_text", "frames", "speaker_id", "start_time", "end_time"]),
                "Historical target, duration or split changed")
    denominator = [counts(r["text"], "") for r in source["records"]]
    require(sum(r["reference_characters"] for r in denominator) == 667 and
            sum(r["reference_words"] for r in denominator) == 189, "Unexpected fixed metric denominator")
    require(len(manifest["records"]) == 40 and {(r["arm"], r["id"]) for r in manifest["records"]} ==
            {(a, s) for a in ARMS for s in source_ids}, "Incomplete two-arm input cohort")
    models = {}
    for branch in BRANCHES:
        folder = HISTORY / branch
        summary_path = folder / "step_6400/summary.json"
        summary = read(summary_path)
        rows_path = folder / "step_6400/rows.jsonl"
        posterior_path = folder / "step_6400/posteriors.npz"
        checkpoint_path = folder / "step_6400.pt"
        require(summary["branch"] == branch and summary["step"] == 6400, "Wrong endpoint")
        for path in [summary_path, rows_path, posterior_path]:
            require(sha(path) == quality["artifacts"][path.relative_to(ROOT).as_posix()], "Historical audited artifact changed")
        require(sha(checkpoint_path) == summary["checkpoint_sha256"], "Checkpoint changed")
        require(sha(rows_path) == summary["rows_sha256"] and sha(posterior_path) == summary["posteriors_sha256"], "Historical outputs changed")
        models[branch] = {"checkpoint": evidence(checkpoint_path), "summary": evidence(summary_path),
                          "rows": evidence(rows_path), "posteriors": evidence(posterior_path)}
    files = {"source_plan": OFFICIAL / "step03_source_plan.json", "input_manifest": BASE / "inputs/manifest.json",
             "alignment_sidecar": OFFICIAL / "results/step04_audio_offset_candidates.json",
             "official_comparison": OFFICIAL / "results/step04.json", "legacy_plan": HISTORY / "plan.json",
             "legacy_completed": HISTORY / "completed.json", "legacy_quality": HISTORY / "quality_checks.json"}
    for field in ["source_plan", "alignment_sidecar", "preparation_plan"]:
        require(sha(manifest[field]["path"]) == manifest[field]["sha256"], f"Input evidence changed: {field}")
    require(manifest["source_plan"]["sha256"] == sha(files["source_plan"]) and
            manifest["alignment_sidecar"]["sha256"] == sha(files["alignment_sidecar"]), "Input evidence refers to other data")
    plan = {"schema": "legacy-step05-evaluation-plan-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_new_model_inference": True, "source_ids": source_ids, "branches": BRANCHES,
            "arms": ARMS, "methods": METHODS, "models": models,
            "files": {name: evidence(path) for name, path in files.items()},
            "legacy_source_hashes": previous["source_hashes"],
            "runner": evidence(Path(__file__)),
            "metric_sources": {name: evidence(OFFICIAL / name) for name in ["evaluation_metrics.py", "diagnostic_utils.py"]},
            "environment": environment(),
            "settings": {"batch_size": 1, "precision": "FP32", "autocast": False, "lengths": None,
                         "torch_threads": 2, "seed": 42, "cudnn_deterministic": True,
                         "cudnn_benchmark": False, "tf32": False, "beam_width": 10,
                         "normalization": "legacy.forward RGB /255 ImageNet mean/std; no new normalization",
                         "checkpoint_selection": "both preexisting fixed step6400 endpoints, no best-checkpoint selection"},
            "gate": "All40 original-input log posteriors and all80 decoded strings must reproduce historical values exactly before any aligned-input inference",
            "metrics": {"primary": "pooled NFC composed-character edit distance including internal spaces /667",
                        "secondary": "WER /189, exact20, per-speaker; historical mean and pooled NFD jamo CER separately",
                        "failures": "retain every ID with empty prediction; source/provenance/replay failure aborts before comparison"},
            "scope": {"training": False, "test50": False, "audio_model_input": False, "new_download": False,
                      "production_changes": False, "new_geometry": False, "development_only": True},
            "interpretation": ["Within each checkpoint only the source-specific video clock mapping changes.",
                               "Official model comparison is between full pipelines, not isolated neural architectures.",
                               "Neither correcting evaluation nor poor results establish whether the training sources are aligned."]}
    write(BASE / "eval_plan.json", plan)
    print(json.dumps({"plan": str(BASE / "eval_plan.json"), "ids": 20, "models": BRANCHES}, ensure_ascii=False), flush=True)


def aggregate_all(records: list[dict]) -> dict:
    """Report every fixed arm/method with both current and historical metrics."""
    result = {}
    for branch in BRANCHES:
        for arm in ARMS:
            selected = [r for r in records if r["branch"] == branch and r["arm"] == arm]
            if not selected:
                continue
            for method in METHODS:
                def summarize(rows: list[dict]) -> dict:
                    entries = [dict(id=r["id"], **r["outputs"][method]) for r in rows]
                    value = aggregate(entries)
                    value["mean_jamo_cer_percent"] = 100 * statistics.mean(e["legacy_metrics"]["cer"] for e in entries)
                    value["jamo_edits"] = sum(e["legacy_metrics"]["jamo_edit_distance"] for e in entries)
                    value["reference_jamos"] = sum(e["legacy_metrics"]["reference_jamo_count"] for e in entries)
                    value["pooled_jamo_cer_percent"] = 100 * value["jamo_edits"] / value["reference_jamos"]
                    value["replacement_characters"] = sum(e["prediction"].count("\ufffd") for e in entries)
                    return value
                item = summarize(selected)
                item["per_speaker"] = {s: summarize([r for r in selected if r["speaker_id"] == s])
                                       for s in sorted({r["speaker_id"] for r in selected})}
                result[f"{branch}/{arm}/{method}"] = item
    return result


@torch.inference_mode()
def run() -> None:
    """Replay the old inputs, enforce the gate, then evaluate corrected inputs."""
    plan = read(BASE / "eval_plan.json")
    output = BASE / "results"
    require(not output.exists(), "Preserve previous run; result directory exists")
    require(environment() == plan["environment"], "Runtime changed")
    for item in [*plan["files"].values(), plan["runner"], *plan["metric_sources"].values()]:
        require(sha(item["path"]) == item["sha256"], "Frozen file changed")
    for name, digest in plan["legacy_source_hashes"].items():
        require(sha(ROOT / name) == digest, "Frozen legacy implementation changed")
    for spec in plan["models"].values():
        for item in spec.values():
            require(sha(item["path"]) == item["sha256"], "Frozen model or historical evidence changed")
    manifest = read(Path(plan["files"]["input_manifest"]["path"]))
    sources = {r["id"]: r for r in read(Path(plan["files"]["source_plan"]["path"]))["records"]}
    inputs = {(r["arm"], r["id"]): r for r in manifest["records"]}
    require(len(inputs) == 40, "Duplicate input")
    cache = {}
    for key, row in inputs.items():
        require(key in {(a, s) for a in ARMS for s in plan["source_ids"]}, "Unplanned input")
        source = sources[row["id"]]
        path = Path(row["npz_path"]).resolve()
        require(path == Path(source["legacy_cache_absolute_path"]).resolve() if row["arm"] == "old_timing"
                else path.is_relative_to((BASE / "inputs").resolve()), "Unapproved cache location")
        require(sha(path) == row["npz_sha256"], "Cache hash changed")
        item = load_clip(path)
        require(item["text"] == row["reference"] == source["text"] and item["source_text"] == source["source_text"], "Target changed")
        require(len(item["frames"]) == row["frame_count"] == source["frames"], "Frame count changed")
        require(hashlib.sha256(item["frames"].tobytes()).hexdigest() == row["pixel_sha256"], "Pixels changed")
        cache[key] = item
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir()
    result = {"schema": "legacy-step05-result-v1", "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha(BASE / "eval_plan.json"), "plan": plan, "records": [],
              "historical_replay": {"passed": False, "records": []}, "scope": plan["scope"],
              "timing_scope": "forward includes uint8 normalization, H2D transfer, model and D2H; decoding CPU separately; excludes original video/crop/capture"}
    try:
        for arm in ARMS:
            if arm == "aligned_timing":
                require(result["historical_replay"]["passed"], "Historical replay gate not passed")
            for branch in BRANCHES:
                spec = plan["models"][branch]
                model = load_model_checkpoint(spec["checkpoint"]["path"]).to("cuda").eval()
                require(model.architecture_version == "lipnet-compact-v1" and not model.training, "Wrong model or mode")
                provenance = model.training_data
                require(provenance["branch"] == branch and provenance["cumulative_attempt"] == 19200
                        and provenance["development_only"] and not provenance["deployment"], "Unexpected checkpoint provenance")
                historical = [json.loads(line) for line in Path(spec["rows"]["path"]).read_text(encoding="utf8").splitlines()]
                historical = {r["id"]: r for r in historical if r["cohort"] == "val20" and r["condition"] == "normal"}
                require(set(historical) == set(plan["source_ids"]), "Historical target set differs")
                with np.load(spec["posteriors"]["path"], allow_pickle=False) as saved:
                    for index, sid in enumerate(plan["source_ids"], 1):
                        source, item = sources[sid], cache[arm, sid]
                        row = {"branch": branch, "arm": arm, "id": sid, "speaker_id": source["speaker_id"],
                               "reference": source["text"], "frames": len(item["frames"]), "outputs": {}}
                        try:
                            torch.cuda.synchronize(); started = time.perf_counter()
                            lp, _ = legacy.forward(model, item["frames"], capture=False)
                            torch.cuda.synchronize(); row["forward_seconds"] = time.perf_counter() - started
                            path = output / f"{branch}_{arm}_{sid}.npz"
                            np.savez_compressed(path, log_probs=lp)
                            row["posterior"] = evidence(path)
                            row["blank_fraction"] = float(np.mean(lp.argmax(-1) == 0))
                            for method in METHODS:
                                started = time.perf_counter()
                                prediction = legacy.ctc_greedy_decode(lp) if method == "greedy" else legacy.ctc_beam_search_decode(lp, beam_width=10)
                                row["outputs"][method] = {"status": "ok", "prediction": prediction,
                                    "decoder_seconds": time.perf_counter() - started, "metrics": counts(source["text"], prediction),
                                    "legacy_metrics": legacy.prediction_row(source, prediction)}
                            labels = torch.tensor(item["label"], dtype=torch.long)
                            loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:, None, :]), labels,
                                torch.tensor([len(lp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
                            require(bool(torch.isfinite(loss)), "Nonfinite CTC loss")
                            row["ctc_loss_per_target_token"] = float(loss)
                            if arm == "old_timing":
                                old = historical[sid]
                                old_lp = saved[old["posterior_key"]]
                                gate = {"branch": branch, "id": sid, "posterior_exact": np.array_equal(lp, old_lp),
                                        "max_abs_posterior_difference": float(np.max(np.abs(lp-old_lp))),
                                        "greedy_exact": row["outputs"]["greedy"]["prediction"] == old["greedy"]["prediction"],
                                        "beam10_exact": row["outputs"]["beam10"]["prediction"] == old["beam"]["prediction"]}
                                result["historical_replay"]["records"].append(gate)
                                require(all(gate[k] for k in ["posterior_exact", "greedy_exact", "beam10_exact"]), "Historical output replay differs")
                        except Exception as exc:
                            if arm == "old_timing":
                                raise
                            row["error"] = repr(exc)
                            row["outputs"] = {m: {"status": "error", "prediction": "", "metrics": counts(source["text"], ""),
                                                "legacy_metrics": legacy.prediction_row(source, "")} for m in METHODS}
                        result["records"].append(row)
                        write(output / "result.json", result, exclusive=False)
                        print(json.dumps({"branch": branch, "arm": arm, "completed": index, "id": sid,
                                          "greedy": row["outputs"]["greedy"]["prediction"]}, ensure_ascii=False), flush=True)
                del model
                torch.cuda.empty_cache()
            if arm == "old_timing":
                require(len(result["historical_replay"]["records"]) == 40, "Incomplete historical replay")
                result["historical_replay"]["passed"] = True
                write(output / "result.json", result, exclusive=False)
        require(len(result["records"]) == 80, "Incomplete fixed evaluation")
        result["aggregate"] = aggregate_all(result["records"])
        result["status"] = "completed"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write(output / "result.json", result, exclusive=False)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        write(output / "result.json", result, exclusive=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
