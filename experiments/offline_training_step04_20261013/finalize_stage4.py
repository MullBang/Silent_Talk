"""Finalize stage4 handoff after all measured checks; never runs a model."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_completion import BASE, ROOT, read, ref, save


def main() -> None:
    """Bind preserved inputs, measured results and the separately planned stage5."""
    evidence_names = [
        "results/engine_test_results.json", "data/loader_test_results.json", "results/evaluation_contract_tests.json",
        "audit/stage3_source_preflight.json", "audit/protocol_audit.json", "audit/loader_behavior_audit.json",
        "audit/pre_execution_audit.json", "audit/smoke_checkpoint_audit.json", "audit/resource_scope_audit.json",
        "audit/completion_supplement_audit.json", "results/padding_preflight.json", "results/smoke_summary.json"]
    for name in evidence_names:
        item = read(BASE/name)
        assert item.get("status") == "passed" or item.get("passed") is True, name
    smoke = read(BASE/"results/smoke_summary.json")
    smoke_plan = read(BASE/"smoke_plan.json")
    for binding in smoke_plan["bindings"]:
        assert ref(Path(binding["path"]))["sha256"] == binding["sha256"], binding["path"]
    stage3 = ROOT/"experiments/offline_dataset_step03_20261013"
    for item in read(stage3/"results/completion_manifest.json")["artifacts"]:
        assert ref(Path(item["path"]))["sha256"] == item["sha256"], item["path"]
    for item in read(stage3/"results/scope_snapshot.json")["artifacts"]:
        path = Path(item["path"])
        assert path.exists() is item["exists"]
        if item["exists"]:
            assert ref(path)["sha256"] == item["sha256"]
    timestamp = datetime.now(timezone.utc).isoformat()
    handoff = {
        "schema":"offline-stage5-prepared-comparison-handoff-v1", "created_utc":timestamp,
        "status":"prepared_for_separate_stage5_runner_and_main_comparison",
        "data":ref(stage3/"results/cache_handoff.json"),
        "protocol":ref(BASE/"protocol/comparison_protocol.json"),
        "protocol_frozen_manifest":ref(BASE/"protocol/completion_manifest.json"),
        "execution_supplement":ref(BASE/"results/execution_clarifications.json"),
        "resource_budget":ref(BASE/"results/resource_budget.json"),
        "verified_code":[ref(BASE/p) for p in ("data/loader.py","engine.py","evaluation_contract.py")],
        "smoke_plan":ref(BASE/"smoke_plan.json"),"smoke_result":ref(BASE/"results/smoke_summary.json"),
        "runtime":smoke_plan["runtime"],
        "data_counts":{"train_parents":12,"dev_parents":3,"train_clips":24,"dev_clips":6,"independent_clips":False},
        "planned_main":{"seeds":[42,43],"arms":["A_sentence","B_curriculum"],"updates_per_run":1260,"updates_total":5040,"evaluation_rows":1080,"checkpoint_count":12},
        "next_required_work":[
            "Implement a separate frozen stage5 runner/loader policy around these same fixed schedules and inputs; the stage4 policy and6-update CLI cannot authorize/execute full training.",
            "Verify actual maximum packed-length batch316 and dense358, resource envelope, fresh paired initialization and complete checkpoint/restart in the new runner.",
            "Bind prediction helper, all30 evaluation IDs, raw argmax/collapsed IDs/posteriors and explicit failed rows; use final1260 dev full-sentence3 normal CER as primary.",
            "Run the four main comparisons only as stage5; count any preparatory updates separately and preserve failures.",
            "Preserve all stage3/stage4 frozen artifacts. Never initialize from smoke or historical trained weights, or silently replace the service checkpoint."],
        "source_flags":{"training_ready":False,"provisional":True,"boundary_followup_clips":2,"precise_visual_boundary_certified":False},
        "new_original_download_required":False,"new_human_answers_required_for_this_prepared_pilot":False,
        "main_training_executed":False,"service_deployment_authorized":False}
    save(BASE/"results/stage5_handoff.json",handoff)
    summary = {
        "schema":"offline-stage4-completion-summary-v1","created_utc":timestamp,
        "status":"completed_model_execution_and_comparison_preparation",
        "goal":"Under30-second recorded webcam video to Korean text using visual lip input only",
        "model":"Fresh compact3D-CNN/projection256/2-layerBiLSTM256/69CTC", "model_parameters":3287781,
        "actual_smoke_optimizer_updates":48,"actual_training_forward_calls":48,"padding_eval_forward_calls":3,
        "checkpoint_count":16,"exact_resume_comparisons":4,"all_final_replays_bitwise_exact":smoke["all_final_replays_bitwise_exact"],
        "actual_true_training_frames_including_replays":9932,"actual_dense_training_frames_including_replays":13112,
        "unit_and_live_cache_tests_passed":38,"additional_independent_negative_cases":21,
        "all30_label_prediction_roundtrips_checked":True,
        "preserved_stage3_frozen_artifacts":114,"preserved_service_scope_items":10,
        "max_cuda_allocated_bytes":smoke["max_cuda_allocated_bytes"],"max_cuda_reserved_bytes":smoke["max_cuda_reserved_bytes"],
        "main_comparison_updates":0,"recognition_performance_measured":False,"source_TAR_reads":0,
        "audio_inputs":0,"teacher_calls":0,"downloads":0,"service_changes":0,
        "source_training_ready":False,"source_provisional":True,
        "clarification":"Preserved historical TCHW future-input wording; executable single input CTHW and batch BCTHW are explicitly bound in the supplement.",
        "evidence":[ref(BASE/name) for name in evidence_names],
        "next_stage":ref(BASE/"results/stage5_handoff.json"),
        "limitations":["The48updates validate execution/recovery, not recognition success.","The future dev3 cohort is a tiny internal pilot of previously used speakers.","A/B parent visits/updates match but frames/tokens/full-sentence exposures differ.","Exact original visual phonetic boundaries are not certified; existing boundary flags remain.","The future316true-frame pair needs its own runner/resource preflight."]}
    save(BASE/"results/stage4_summary.json",summary)
    report = ROOT/"docs/녹화형_단계학습_2026-10-12/04_학습실행_검증과_비교규약.md"
    paths = sorted(p for p in BASE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    paths.append(report)
    assert not any(p.name == "failure.json" for p in paths)
    manifest = {"schema":"offline-stage4-completion-snapshot-v1","created_utc":timestamp,
        "status":summary["status"],"artifacts":[ref(p) for p in paths],
        "training_ready":False,"smoke_updates":48,"main_updates":0,
        "policy":"Keep these code/plans/checkpoints/results as frozen evidence; any modification requires a separate version. AGENTS.md is a mutable navigation index and is not included."}
    save(BASE/"results/completion_manifest.json",manifest)
    print(json.dumps({"status":summary["status"],"frozen_artifacts":len(paths),"summary":ref(BASE/"results/stage4_summary.json"),"next_stage":ref(BASE/"results/stage5_handoff.json")},ensure_ascii=False))


if __name__ == "__main__":
    main()
