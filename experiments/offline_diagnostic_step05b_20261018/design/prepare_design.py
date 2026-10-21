"""Prepare a future, non-executable phrase-only experiment design from metadata.

This script does not import torch, load checkpoints/NPZ arrays, or run a model.
It creates files only inside this new design directory and never overwrites.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
STAGE3 = ROOT / "experiments/offline_dataset_step03_20261013"
STAGE4 = ROOT / "experiments/offline_training_step04_20261013"
STAGE5 = ROOT / "experiments/offline_comparison_step05_20261013"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path: Path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def save(name: str, value):
    path = BASE / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return binding(path)


def main():
    handoff_path = STAGE3 / "results/cache_handoff.json"
    handoff = read(handoff_path)
    records = [r for r in handoff["records"] if r["split"] == "train" and r["kind"] == "phrase"]
    assert len(records) == 12
    by_parent = {r["parent_id"]: r for r in records}
    assert len(by_parent) == 12
    assert sum(r["T"] for r in records) == 481
    assert sum(r["label_length"] for r in records) == 272
    assert all(r["T"] >= r["ctc_min_frames"] and not r["training_ready"] and r["provisional"] for r in records)
    schedule_bindings = []
    budgets = []
    for seed in (42, 43):
        source_path = STAGE4 / f"protocol/schedule_seed{seed}.json"
        original = read(source_path)["updates"]
        assert len(original) == 1260
        parent_rows = original + original[:420]
        schedule = []
        visits = Counter()
        for index, old in enumerate(parent_rows, 1):
            selected = [by_parent[p] for p in old["parent_ids"]]
            visits.update(old["parent_ids"])
            lengths = [r["T"] for r in selected]
            lr = .001 if index <= 840 else .0003 if index <= 1050 else .0001
            row = {"update": index, "kind": "phrase", "parent_ids": old["parent_ids"],
                   "clip_ids": [r["clip_id"] for r in selected], "lengths": lengths,
                   "target_lengths": [r["label_length"] for r in selected], "lr": lr,
                   "true_frames": sum(lengths), "dense_frames": 2 * max(lengths),
                   "source_schedule_update": old["update"],
                   "source_schedule_replayed_for_order_only": index > 1260}
            if index <= 420:
                assert row["clip_ids"] == old["arms"]["B_curriculum"]["clip_ids"]
                assert row["lengths"] == old["arms"]["B_curriculum"]["lengths"]
                assert row["lr"] == old["lr"]
            schedule.append(row)
        assert set(visits.values()) == {280}
        budget = {"seed": seed, "logical_updates": len(schedule), "parent_visits_each": 280,
                  "true_frames": sum(r["true_frames"] for r in schedule),
                  "dense_frames": sum(r["dense_frames"] for r in schedule),
                  "target_tokens": sum(sum(r["target_lengths"]) for r in schedule),
                  "max_true_batch_frames": max(r["true_frames"] for r in schedule),
                  "max_dense_batch_frames": max(r["dense_frames"] for r in schedule)}
        assert budget["true_frames"] == 134680 and budget["target_tokens"] == 76160
        schedule_bindings.append(save(f"proposed_phrase_schedule_seed{seed}.json", {
            "schema": "offline-stage5b-proposed-phrase-only-schedule-v1", "status": "design_only_not_execution_authorization",
            "seed": seed, "source_schedule": binding(source_path),
            "rule": "Preserve first1260 parent batches; repeat first420 parent batches for updates1261-1680. All inputs are the fixed train phrases.",
            "first420_matches_prior_B_clip_order_and_lr": True, "accounting": budget, "updates": schedule}))
        budgets.append(budget)
    compact_records = [{k: r[k] for k in ("clip_id", "parent_id", "rank", "speaker_id", "T", "label_length", "ctc_min_frames", "text", "provisional", "training_ready", "boundary_followup_flag")} | {
        "frames_above_ctc_minimum": r["T"] - r["ctc_min_frames"],
        "frames_over_ctc_minimum": r["T"] / r["ctc_min_frames"]} for r in records]
    source_paths = [handoff_path, STAGE4 / "protocol/comparison_protocol.json", STAGE4 / "protocol/evaluation_plan.json",
                    STAGE4 / "engine.py", ROOT / "backend/models/baseline.py",
                    STAGE5 / "results/stage5_summary.json", STAGE5 / "results/resource_accounting.json",
                    STAGE5 / "results/next_step_handoff.json", STAGE5 / "analysis/result_interpretation.json"]
    plan = {
        "schema": "offline-stage5b-future-phrase-learnability-design-v1",
        "created_date_local": "2026-10-18", "status": "design_only_not_executable_or_authorized_training",
        "question_ko": "동일한 compact/CTC/입력/옵티마이저에서 문장으로 바꾸지 않고 구절 학습을 계속하면 고정12구절을 안정적으로 재현하는가?",
        "sources": [binding(p) for p in source_paths],
        "prior_observation": {"B420_train_phrase_exact_by_seed": {"42": "1/12", "43": "0/12"},
                              "B420_train_phrase_CER_percent_by_seed": {"42": 100 * 78 / 132, "43": 100 * 80 / 132},
                              "prior_final_dev_exact_all_models": "0/3"},
        "train_records": compact_records, "train_clip_count": 12,
        "train_unique_phrase_texts": len({r["text"] for r in records}),
        "train_speakers": sorted({r["speaker_id"] for r in records}),
        "unchanged": {"model": "fresh compact3D-CNN32/64/96 + projection256/dropout0.1 + 2-layer BiLSTM256 per direction + 69CTC",
                      "parameter_count": 3287781, "seeds": [42, 43], "pretrained_or_teacher": False,
                      "input": "same25fps D-grid96RGB fixed caches, float32 RGB/255 then ImageNet normalization; true-length-aware BCTHW",
                      "optimizer": {"name": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0., "gradient_clip_norm": 1.},
                      "loss": "same CPU FP32 CTC per-target-length mean; blank0; zero_infinity=False",
                      "batch_size": 2, "precision": "FP32, AMP/TF32 off, same deterministic spatial pooling", "augmentation": "none"},
        "changes_from_stage5_B": ["Keep phrase inputs after420 instead of switching to full sentences.",
                                  "Run to1680 total updates, continuing final existing learning rate0.0001 after1260.",
                                  "Use only train-phrase evaluations; no dev evaluations or optimization."],
        "schedule_bindings": schedule_bindings,
        "lr_schedule": [{"from": 1, "through": 840, "lr": .001}, {"from": 841, "through": 1050, "lr": .0003}, {"from": 1051, "through": 1680, "lr": .0001}],
        "selection_policy": {"fixed_final_update": 1680, "checkpoint_updates": [420, 840, 1260, 1680],
                             "early_best_checkpoint_or_seed_selection": False, "dev_dependent_decisions": False,
                             "per_seed_gate": "Both1260 and1680 normal train-phrase evaluations must have12/12 exact and corpus NFC CER0 with no failed rows.",
                             "overall_gate": "Both seed42 and seed43 must pass their last-two-checkpoint rule.",
                             "gate_meaning_ko": "현재12구절 재현의 엄격한 기술적 임의 기준이다. 발음 이해, 일반화, 학습자료전체승인, 서비스성공 기준이 아니다.",
                             "if_not_met": "Stop at1680, report all seeds/curves and analyze residual failures. Do not auto-add updates, alter LR, drop difficult clips, or advance to sentence expansion.",
                             "if_met": "Record evidence and reconsider the separately proposed same-full-sentence-exposure experiment. No automatic next training is authorized."},
        "evaluation": {"split": "train_only", "normal": "12 phrases at420/840/1260/1680 for both seeds=96 forwards",
                       "temporal_control": "At fixed1680: same12 phrases, center-frame repeat and previously fixed per-clip shuffle, both seeds=48 forwards",
                       "total_planned_forwards": 144, "not_independent_utterance_count": True,
                       "decoder": "same argmax-collapse then blank removal; compose_jamos(strict=False), preserveU+FFFD",
                       "metrics": ["corpus NFC CER including inner spaces", "word error rate", "exact12 count", "per-clip CTC/target length", "argmax/collapsed token errors", "replacement/repetition statistics"],
                       "temporal_control_is_gate": False,
                       "temporal_limit": "If static/shuffled also fit, normal fit alone does not establish motion-based recognition. Review before further comparisons.",
                       "dev_clip_forward_calls": 0, "full_sentence_forward_calls": 0},
        "planned_accounting": {"per_seed": budgets, "main_updates": 3360,
                               "main_true_frames": sum(b["true_frames"] for b in budgets),
                               "main_dense_frames": sum(b["dense_frames"] for b in budgets),
                               "main_target_tokens": sum(b["target_tokens"] for b in budgets),
                               "max_separate_preflight_updates": 8, "max_total_optimizer_updates_before_any_replan": 3368,
                               "main_forward_calls": 3360, "evaluation_forward_calls": 144,
                               "max_total_forward_calls_including_preflight": 3512,
                               "main_checkpoints": 8, "max_preflight_checkpoints": 8,
                               "max_wall_seconds_per_seed_proposed": 900, "max_wall_seconds_all_main_proposed": 1800,
                               "max_cuda_reserved_bytes_proposed": 6 * 1024**3,
                               "minimum_free_disk_bytes_proposed": 4 * 1024**3,
                               "time_limits_are_caps_not_measured_ETA": True},
        "pre_execution_requirements": [
            "Create new separate execution folder, immutable code/runtime/plan/policy bindings; this design has no model launcher.",
            "Preserve all stage3/4/5 frozen artifacts and source flags, scope boundaries and held-out recordings.",
            "Validate exact actual phrase data order, labels, max lengths and new output directory; no dev in optimizer or evaluator.",
            "Run at most8 explicitly accounted maximum-input/save-resume smoke updates; require exact complete-state recovery under declared runtime.",
            "With unchanged numerical runtime, require fresh initial-state and update420 model/Adam/RNG reproduction of prior B before continuing. If runtime differs or replay fails, stop and separately characterize rather than claim pure-budget replication.",
            "Save model/Adam/RNG/cursor/LR/evaluation completion transactionally; complete fixed missing evaluations before continuation on resume.",
            "Count every physical replay/preflight update separately; no silent extension beyond fixed budget. Source mismatch, nonfinite loss/gradient, data-boundary or resource violations stop execution."],
        "conditional_followup_comparison": {
            "status": "conditional_outline_only_requires_separate_fixed_execution_design",
            "trigger": "Both seeds satisfy phrase-only fit gate and diagnostic review supports proceeding; not automatic.",
            "question": "With the same full-sentence optimization opportunities, does a well-fitted phrase initialization help this internal pilot?",
            "A": "fresh compact -> S=1260 full-sentence updates",
            "B": "fresh compact -> P=1680 phrase updates -> S=1260 full-sentence updates",
            "paired_controls": ["Same12 train parents and full-sentence batches for all S updates", "Same seed-paired model initialization", "Same full-sentence-stage LR indexed by sentence update", "Same preprocessing/loss/decoder and final-S endpoint"],
            "optimizer_transition_proposal": "Reset Adam at the start of S in both arms; B transfers phrase-trained model weights only. This removes an unequal Adam-step/momentum history at the sentence stage but is a two-stage initialization comparison, not continuous-optimizer curriculum.",
            "alternative_not_simultaneously_run": "If continuous Adam state is desired instead, predeclare it and explicitly include optimizer-history effects; do not select the option by dev3 results.",
            "logical_updates_if_all_four_runs_are_new": 2 * (1260 + 1680 + 1260),
            "phrase_pilot_updates_not_included_in_this_conditional_number": 3360,
            "cost_limit": "B includes extra phrase updates/frames/tokens. Equal sentence exposure is not equal total compute or a pure order effect. An equal-total-update A would answer a different question and must not replace this control post hoc.",
            "dev_limit": "The current dev3 has already influenced analysis/design; any later scores are exploratory reused-development results, not independent evidence. An untouched reviewed recording/transcript-disjoint development/evaluation set is required before asserting generalization.",
            "new_data_policy": "Immediate phrase diagnostic needs no downloads. For independent generalization, inventory and review eligible existing data first; the additional254 metadata are not automatically approved training/evaluation clips.",
            "not_stage6_expansion_authorization": True},
        "actual_work_here": {"metadata_reads_only": True, "new_neural_calls": 0, "new_optimizer_updates": 0,
                             "checkpoint_tensor_loads": 0, "NPZ_pixel_loads": 0, "raw_source_reads": 0, "downloads": 0, "service_changes": 0},
    }
    out = save("proposed_phrase_only_design.json", plan)
    print(json.dumps({"status": "design_prepared_no_execution", "design": out,
                      "accounting": plan["planned_accounting"], "train_unique_phrase_texts": plan["train_unique_phrase_texts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
