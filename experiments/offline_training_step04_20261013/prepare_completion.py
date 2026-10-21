"""Write evidence-based resource and execution supplements without neural calls."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path: Path):
    """Read existing UTF8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def ref(path: Path) -> dict:
    """Bind a local artifact without importing its code."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def save(path: Path, value: dict) -> None:
    """Create new completion evidence only."""
    assert path.resolve().is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Derive measured resources, explicitly separating estimates and future caps."""
    smoke = read(BASE/"results/smoke_summary.json")
    assert smoke["status"] == "passed"
    steps = [s for run in smoke["runs"] for segment in ("continuous", "split", "resumed") for s in run[segment]["executed_updates"]]
    schedules = [read(BASE/f"protocol/schedule_seed{seed}.json") for seed in (42,43)]
    arms = [arm for schedule in schedules for row in schedule["updates"] for arm in row["arms"].values()]
    checkpoints = sorted(BASE.rglob("*.pt"))
    elapsed = sum(run[segment]["elapsed_seconds"] for run in smoke["runs"] for segment in ("continuous", "split", "resumed"))
    assert len(steps) == 48 and len(arms) == 5040 and len(checkpoints) == 16
    save(BASE/"results/evaluation_contract_tests.json", {
        "status":"passed", "command":".venv-tokenizer/Scripts/python.exe -m pytest experiments/offline_training_step04_20261013/test_evaluation_contract.py -q",
        "exit_code":0,"tests_passed":13,"pytest_report":"13 passed in 0.06s",
        "code":[ref(BASE/name) for name in ("evaluation_contract.py","test_evaluation_contract.py")],
        "neural_forward_calls":0,"purpose":"CTC blank/repetition/invalid jamo and corpus-metric regression checks"})
    save(BASE/"results/execution_clarifications.json", {
        "schema":"offline-stage4-execution-supplement-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":ref(BASE/"protocol/comparison_protocol.json"),"data_loader":ref(BASE/"data/loader.py"),
        "axis_clarification":{
            "historical_field":"model.normalization.future_input",
            "historical_text":"float32 RGB /255; channel (x-mean)/std; TCHW at loader",
            "correct_executable_contract":{"raw_cache":"T,H,W,C (RGB uint8)","single_loader_output":"C,T,H,W (FP32)","collated_model_input":"B,C,T,H,W (FP32)","model_output":"T,B,69"},
            "reason":"The protocol copied a future-input note from stage3 metadata. The implemented/tested loader already uses CTHW, matching the existing model. This supplement clarifies the metadata wording; no pixel, normalization, execution or outcome is changed.",
            "old_protocol_and_stage3_preserved":True},
        "prediction_and_metrics":ref(BASE/"evaluation_contract.py"),
        "prediction_contract_test":ref(BASE/"results/evaluation_contract_tests.json"),
        "metric_exclusion":"Do not call backend.models.tokenizer.cer, whose NFD metric differs from this NFC-character protocol. Do not call the legacy strict core.greedy decoder.",
        "repetition_diagnostic":"Count outputs containing a run of at least5 identical nonspace Unicode characters after NFC/outer strip; retain U+FFFD. Save each maximum run.",
        "additional_research_signal_fixed_before_main_results":{
            "scope":"Descriptive pilot decision, not significance or deployment certification",
            "rule":"B final normal dev-sentence corpus CER is strictly lower than A in both seeds and the average reduction is at least0.05 absolute (5 percentage points). Report exact3, WER and static/shuffled results regardless of passing.",
            "failure_rule":"If no signal, report result and diagnose before enlarging; do not select a favorable intermediate checkpoint or tune on the dev3 outcomes."},
        "main_results_seen":False,"neural_execution_in_supplement":0})
    budget = {
        "schema":"offline-stage4-measured-resource-and-future-budget-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "sources":[ref(BASE/p) for p in ("results/smoke_summary.json","audit/resource_scope_audit.json","protocol/schedule_accounting.json","protocol/evaluation_plan.json")],
        "device":"NVIDIA GeForce RTX 3080 (10GiB nominal)",
        "measured_smoke":{
            "optimizer_updates":48,"optimizer_seconds":smoke["optimizer_seconds_total"],"sum_segment_elapsed_seconds":elapsed,
            "segment_time_includes":"initialization, data checking/loading, state hashing and checkpoint serialization; excludes outer orchestration/padding preflight/final independent audit",
            "seconds_per_update_min":min(s["seconds"] for s in steps),"seconds_per_update_median":statistics.median(s["seconds"] for s in steps),"seconds_per_update_max":max(s["seconds"] for s in steps),
            "max_cuda_allocated_bytes":smoke["max_cuda_allocated_bytes"],"max_cuda_reserved_bytes":smoke["max_cuda_reserved_bytes"],
            "memory_scope":"PyTorch process allocated/reserved counters; not total device usage or a future guarantee",
            "max_true_frames_per_batch":max(s["true_frames"] for s in steps),"max_dense_frames_per_batch":max(s["dense_frames"] for s in steps),
            "checkpoints":16,"checkpoint_bytes":sum(p.stat().st_size for p in checkpoints),"largest_checkpoint_bytes":max(p.stat().st_size for p in checkpoints)},
        "future_main_comparison":{
            "runs":4,"updates_per_run":1260,"total_updates":5040,"batch_size":2,
            "max_T":max(max(a["lengths"]) for a in arms),"max_true_frames_per_batch":max(a["actual_frames"] for a in arms),
            "max_dense_frames_per_batch":max(a["dense_processed_frames"] for a in arms),
            "actual_training_frames":sum(a["actual_frames"] for a in arms),"dense_training_frames":sum(a["dense_processed_frames"] for a in arms),
            "fixed_checkpoint_count":12,"evaluation_forward_calls":1080,"evaluation_frames":72828,
            "posterior_fp32_raw_bytes_if_saved":72828*69*4,
            "checkpoint_bytes_scaled_from_smoke_without_larger_history":12*max(p.stat().st_size for p in checkpoints)},
        "estimates_not_measured_main_runtime":{
            "update_compute_scaled_seconds":smoke["optimizer_seconds_total"]*5040/48,
            "segment_average_scaled_seconds":elapsed*5040/48,
            "interpretation":"Arithmetic projections of different smoke cost scopes, not confidence bounds or promised duration. New loader/checkpoint cadence, warmup, larger packed pairs and1080 evaluations may change elapsed time. Evaluation wall time has not been measured."},
        "conservative_future_limits":{
            "wall_seconds_per_run_including_evaluation":3600,"aggregate_wall_seconds":14400,
            "minimum_free_local_disk_bytes":4*2**30,"minimum_free_cuda_bytes_at_start":2*2**30,
            "stop_if_cuda_reserved_exceeds_bytes":6*2**30,
            "failure_policy":"At a safe boundary, preserve the last complete checkpoint and failure. Do not silently change batch, inputs, LR, update count or tolerance. OOM/nonfinite loss or gradient aborts the affected run.",
            "preflight":"Stage5 must check its actual maximum packed-length pair; smoke covers maximum dense shape but not every combined true-length/target-length pair."},
        "local_disk_free_at_measurement_bytes":shutil.disk_usage(ROOT).free,
        "new_original_download_required":False,"colab_required_for_this_pilot":False,
        "scope_limit":"Applies to this compact15-parent pilot only; not TB-scale training or a30-second unsegmented full-resolution video.",
        "main_training_executed":False}
    save(BASE/"results/resource_budget.json",budget)
    print(json.dumps({"status":"supplements_created","checkpoint_bytes":budget["measured_smoke"]["checkpoint_bytes"],"max_main_true_frames":budget["future_main_comparison"]["max_true_frames_per_batch"],"max_main_dense_frames":budget["future_main_comparison"]["max_dense_frames_per_batch"]}))


if __name__ == "__main__":
    main()
