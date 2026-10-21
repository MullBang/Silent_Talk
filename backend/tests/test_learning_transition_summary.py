"""Small saved-posterior/telemetry fixtures; no model forward, training or GPU use."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
import summarize_learning_transition as summary
from models.tokenizer import encode_text, tokenizer_metadata
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import (GROUP_PREFIXES, make_training_schedule, gradient_metrics,
                                    snapshot_parameters, parameter_update_metrics)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in rows), encoding="utf-8")


def source(sample_id):
    return {"id": sample_id, "text": "가", "frames": 4, "speaker_id": "speaker", "split": "train"}


def log_probs(path=(1, 0, 20, 0)):
    logits = np.full((len(path), 69), -9., np.float32)
    for index, token in enumerate(path):
        logits[index, token] = 0.
    return logits-np.log(np.exp(logits).sum(axis=1, keepdims=True))


def make_row(record, cohort, condition="normal", step=1, beam=False, lp=None):
    lp = log_probs() if lp is None else lp
    labels = np.asarray(encode_text(record["text"]), np.int64)
    ctc = analyze_ctc(lp, labels, beam_text=record["text"] if beam else "")
    if not beam:
        ctc = {k: v for k, v in ctc.items() if not k.startswith("beam")}
    _, transform = transform_frames(np.zeros((4, 96, 96, 3), np.uint8),
        "shuffle" if condition == "shuffle_42" else condition, sample_id=record["id"], seed=42)
    loss = torch.nn.functional.ctc_loss(torch.from_numpy(lp[:, None, :]), torch.from_numpy(labels),
        torch.tensor([len(lp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
    return {"id": record["id"], "step": step, "split": "train", "condition": condition,
        "cohort": cohort, "speaker_id": record["speaker_id"], "reference": record["text"], "frames": 4,
        "greedy": summary.prediction_row(record, ctc["greedy_text"]),
        "beam": summary.prediction_row(record, record["text"]) if beam else None,
        "ctc": ctc, "ctc_loss_per_target_token": float(loss), "posterior_key": record["id"]+"_"+condition,
        "transform": transform, "comparison_to_normal": None}, lp


def make_snapshot(folder, plan, records, step, checkpoint_sha="a"*64):
    folder.mkdir(parents=True)
    conditions = ["normal", "static_middle", "shuffle_42"] if step in summary.CONTROL_STEPS else ["normal"]
    beam = step in summary.BEAM_STEPS
    rows, arrays = [], {}
    for sample_id in plan["cohorts"]["expanded64"]:
        normal = None
        for condition in conditions:
            row, lp = make_row(records[sample_id], "short16" if sample_id in plan["cohorts"]["short16"] else "additional48",
                               condition, step, beam)
            if condition == "normal":
                normal = row, lp
            else:
                row["comparison_to_normal"] = compare_posteriors(normal[1], lp,
                    normal_text=normal[0]["greedy"]["prediction"], control_text=row["greedy"]["prediction"])
            rows.append(row)
            arrays[row["posterior_key"]] = lp
    write_rows(folder / "rows.jsonl", rows)
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    stored = {"step": step, "beam_performed": beam, "conditions": conditions, "rows": len(rows),
              "checkpoint_sha256": checkpoint_sha, "rows_sha256": summary.sha256_file(folder / "rows.jsonl"),
              "posteriors_sha256": summary.sha256_file(folder / "posteriors.npz"), "seconds": 0., "cohorts": {}}
    for cohort in ("short16", "additional48"):
        selected = [r for r in rows if r["cohort"] == cohort and r["condition"] == "normal"]
        stored["cohorts"][cohort] = {"count": len(selected), "greedy_exact": sum(r["greedy"]["exact"] for r in selected),
                                    "greedy_mean_cer": float(np.mean([r["greedy"]["cer"] for r in selected]))}
    write_json(folder / "summary.json", stored)
    return rows, stored


@pytest.fixture
def snapshot(tmp_path):
    records = {sample_id: source(sample_id) for sample_id in ("old", "new")}
    plan = {"cohorts": {"short16": ["old"], "additional48": ["new"], "expanded64": ["old", "new"]}}
    folder = tmp_path / "step_0001"
    rows, stored = make_snapshot(folder, plan, records, 1)
    return folder, plan, records, rows, stored


def replace_snapshot_rows(fixture, rows):
    folder, _, _, _, stored = fixture
    write_rows(folder / "rows.jsonl", rows)
    stored["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    write_json(folder / "summary.json", stored)


def test_saved_posteriors_are_recomputed_without_beam_decoding(snapshot, monkeypatch):
    import temporal_diagnostics
    monkeypatch.setattr(temporal_diagnostics, "ctc_beam_search_decode", lambda *a, **k: pytest.fail("Unexpected beam decode"))
    folder, plan, records, _, _ = snapshot
    result = summary.validate_snapshot(folder, plan, records, 1, "a"*64)
    assert result["rows"] == 2
    assert result["cohorts"]["short16"]["normal"]["greedy"]["exact_count"] == 1
    assert "beam" not in result["cohorts"]["short16"]["normal"]


@pytest.mark.parametrize("mutation", ["blank", "token_path", "greedy_text", "loss", "missing_ctc", "forbidden_beam"])
def test_rehashed_but_semantically_altered_row_is_rejected(snapshot, mutation):
    folder, plan, records, rows, _ = snapshot
    if mutation == "blank":
        rows[0]["ctc"]["blank_posterior_mean"] += .1
    elif mutation == "token_path":
        rows[0]["ctc"]["greedy_collapsed_token_ids"] = [1, 21]
    elif mutation == "greedy_text":
        rows[0]["greedy"]["prediction"] = "나"
    elif mutation == "loss":
        rows[0]["ctc_loss_per_target_token"] += .5
    elif mutation == "missing_ctc":
        rows[0]["ctc"].pop("argmax_blank_fraction")
    else:
        rows[0]["ctc"]["beam_text"] = ""
    replace_snapshot_rows(snapshot, rows)
    with pytest.raises(ValueError, match="mismatch"):
        summary.validate_snapshot(folder, plan, records, 1, "a"*64)


@pytest.mark.parametrize("mutation", ["validation", "duplicate_id", "wrong_cohort", "beam_present"])
def test_snapshot_rejects_scope_or_decoder_schedule_mutations(snapshot, mutation):
    folder, plan, records, rows, _ = snapshot
    if mutation == "validation":
        rows[0]["split"] = "val"
    elif mutation == "duplicate_id":
        rows[1]["id"] = rows[0]["id"]
    elif mutation == "wrong_cohort":
        rows[0]["cohort"] = "additional48"
    else:
        rows[0]["beam"] = copy.deepcopy(rows[0]["greedy"])
    replace_snapshot_rows(snapshot, rows)
    with pytest.raises(ValueError, match="mismatch"):
        summary.validate_snapshot(folder, plan, records, 1, "a"*64)


def test_hash_failure_occurs_even_for_valid_json(snapshot):
    folder, plan, records, _, _ = snapshot
    with (folder / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summary.validate_snapshot(folder, plan, records, 1, "a"*64)


def test_nonfinite_raw_posteriors_cannot_be_hidden_by_new_hash(snapshot):
    folder, plan, records, rows, stored = snapshot
    arrays = {row["posterior_key"]: log_probs() for row in rows}
    arrays[rows[0]["posterior_key"]][0, 0] = np.nan
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    stored["posteriors_sha256"] = summary.sha256_file(folder / "posteriors.npz")
    write_json(folder / "summary.json", stored)
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.validate_snapshot(folder, plan, records, 1, "a"*64)


def test_control_is_checked_against_normal_greedy_not_beam(tmp_path):
    records = {s: source(s) for s in ("old", "new")}
    plan = {"cohorts": {"short16": ["old"], "additional48": ["new"], "expanded64": ["old", "new"]}}
    folder = tmp_path / "step_0000"
    rows, stored = make_snapshot(folder, plan, records, 0)
    result = summary.validate_snapshot(folder, plan, records, 0, "a"*64)
    assert result["cohorts"]["short16"]["static_middle"]["comparison_to_normal"]["greedy_text_agreement_count"] == 1
    rows[1]["comparison_to_normal"]["mean_total_variation"] = .5
    replace_snapshot_rows((folder, plan, records, rows, stored), rows)
    with pytest.raises(ValueError, match="normal/control posterior"):
        summary.validate_snapshot(folder, plan, records, 0, "a"*64)


def test_token_pooling_uses_raw_collapsed_tokens_and_does_not_pool_cohorts():
    first, _ = make_row(source("a"), "short16", lp=log_probs((1, 0, 1, 0)))
    second, _ = make_row(source("b"), "short16", lp=log_probs((20, 21, 22, 23)))
    result = summary.summarize_group([first, second])
    assert result["token_frequency"]["collapsed_token_count"] == 6
    assert result["token_frequency"]["collapsed_unique_token_count"] == 5
    assert result["token_frequency"]["pooled_top4_fraction"] == pytest.approx(5/6)
    assert result["token_frequency"]["collapsed_counts_by_token_id"] == {"1": 2, "20": 1, "21": 1, "22": 1, "23": 1}
    second["cohort"] = "additional48"
    with pytest.raises(ValueError, match="must not be pooled"):
        summary.summarize_group([first, second])


def test_sentence_mean_and_corpus_cer_have_distinct_denominators():
    records = [{"id": "a", "text": "가", "frames": 4}, {"id": "b", "text": "가가가", "frames": 8}]
    rows = [{"greedy": summary.prediction_row(row, "가" if row["id"] == "a" else "")} for row in records]
    result = summary.aggregate_predictions(rows, "greedy")
    assert result["mean_sentence_jamo_cer"] == .5
    assert result["corpus_jamo_cer"] == .75
    assert result["exact_count"] == 1


def telemetry_fixture(attempts=4):
    records = {s: source(s) for s in ("a", "b")}
    settings = {"grad_scaler_init_scale": 1024., "learning_rate": .0003, "grad_clip": 1.}
    schedule = make_training_schedule(["a", "b"], attempt_updates=attempts, batch_size=2)
    groups = {name: [(prefix+"weight", torch.nn.Parameter(torch.ones(2)))] for name, prefix in GROUP_PREFIXES.items()}
    rows, scale, updates, skips = [], 1024., 0, 0
    for entry in schedule["batches"]:
        attempt = entry["attempt"]
        for pairs in groups.values():
            pairs[0][1].grad = torch.full((2,), .01 if attempt == 1 else (0. if attempt == 4 else 1.))
        skipped = attempt == 3
        if skipped:
            groups["cnn"][0][1].grad[0] = float("inf")
        preclip = gradient_metrics(groups)
        coefficient = None if skipped else min(1., 1./(preclip["total"]["l2_norm"]+1e-6))
        if not skipped:
            for pairs in groups.values():
                pairs[0][1].grad.mul_(coefficient)
        postclip = gradient_metrics(groups)
        before = snapshot_parameters(groups)
        if not skipped and attempt != 4:
            with torch.no_grad():
                for pairs in groups.values():
                    pairs[0][1].add_(-.0001)
        delta = parameter_update_metrics(groups, before)
        updates += int(not skipped)
        skips += int(skipped)
        new_scale = scale/2 if skipped else scale
        rows.append({**copy.deepcopy(entry), "loss": .1, "preclip": preclip, "postclip": postclip,
                     "clip_coefficient": coefficient, "clip_applied": not skipped,
                     "scale_before": scale, "scale_after": new_scale, "skipped": skipped,
                     "parameter_update": delta, "learning_rate": .0003, "seconds": .01,
                     "optimizer_steps": updates, "skipped_optimizer_steps": skips,
                     "frames": [4, 4], "target_lengths": [2, 2]})
        scale = new_scale
    return rows, schedule, records, settings


def test_attempted_and_successful_exposures_are_distinct_and_zero_update_is_not_skip():
    rows, schedule, records, settings = telemetry_fixture()
    result = summary.validate_telemetry(rows, schedule, records, settings)
    assert result["attempt_updates"] == 4 and result["optimizer_steps"] == 3 and result["skipped_optimizer_steps"] == 1
    assert result["attempted_exposures"] == {"a": 4, "b": 4}
    assert result["successful_update_exposures"] == {"a": 3, "b": 3}
    window = summary.fixed_window_summary(rows)["0001-0100"]
    assert window["clip_invoked_count"] == 3 and window["effective_clipping_count"] == 1
    assert window["zero_parameter_update_count"] == 2 and window["skipped_count"] == 1
    norm = window["groups"]["total"]["preclip"]["l2_norm"]
    assert norm["count"] == 4 and norm["null_count"] == 1 and norm["finite_count"] == 3
    assert norm["minimum"] == 0. and norm["median"] > 0.


@pytest.mark.parametrize("mutation", ["ids", "epoch", "optimizer", "exposures", "scale", "clip", "norm", "missing", "continuity"])
def test_telemetry_rejects_schedule_and_accounting_tampering(mutation):
    rows, schedule, records, settings = telemetry_fixture()
    if mutation == "ids":
        rows[0]["ids"].reverse()
    elif mutation == "epoch":
        rows[0]["epoch"] = 2
    elif mutation == "optimizer":
        rows[2]["optimizer_steps"] = 3
    elif mutation == "exposures":
        schedule["exposures"]["a"] = 5
    elif mutation == "scale":
        rows[3]["scale_before"] = 1024.
    elif mutation == "clip":
        rows[1]["clip_coefficient"] = 1.
    elif mutation == "norm":
        rows[0]["preclip"]["total"]["l2_norm"] += 1.
    elif mutation == "missing":
        rows[0]["preclip"]["groups"]["cnn"]["missing_grad_tensors"] = 1
    else:
        for group in [*rows[1]["parameter_update"]["groups"].values(), rows[1]["parameter_update"]["total"]]:
            group["before_l2"] *= 2
            group["relative_l2"] /= 2
    with pytest.raises(ValueError):
        summary.validate_telemetry(rows, schedule, records, settings)


def test_fixed_windows_are_disjoint_attempt_ranges_not_epochs():
    template = telemetry_fixture(1)[0][0]
    rows = [{**copy.deepcopy(template), "attempt": attempt, "epoch": 999} for attempt in (1, 100, 101, 400, 401, 800, 801, 1600)]
    windows = summary.fixed_window_summary(rows)
    assert list(windows) == ["0001-0100", "0101-0400", "0401-0800", "0801-1600"]
    assert [v["attempts_observed"] for v in windows.values()] == [2, 2, 2, 2]
    assert sum(v["attempts_observed"] for v in windows.values()) == len(rows)


def test_first_observed_drop_does_not_invent_unobserved_exact_onset():
    snapshots = {f"{step:04d}": {"step": step, "cohorts": {"short16": {"normal": {"greedy": {"exact_count": exact}}}}}
                 for step, exact in ((0, 16), (1, 16), (5, 16), (10, 15), (25, 16))}
    result = summary.first_observed_drop(snapshots)
    assert result["step"] == 10 and result["previous_observed_step"] == 5
    assert "exact onset" in result["note"]
    for snapshot in snapshots.values():
        snapshot["cohorts"]["short16"]["normal"]["greedy"]["exact_count"] = 16
    assert summary.first_observed_drop(snapshots)["step"] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_or_infinity_is_rejected_but_explicit_null_norm_is_counted(bad, tmp_path):
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.describe([1., bad])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nested": [bad]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.read_json(path)
    result = summary.describe([None, None])
    assert result["finite_count"] == 0 and result["mean"] is None


def test_duplicate_json_keys_and_path_escape_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"completed","status":"running"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        summary.read_json(path)
    with pytest.raises(ValueError, match="escapes"):
        summary.inside(tmp_path, "../outside.json")


@pytest.mark.parametrize("status,sha", [("running", "a"*64), ("completed", "a"*64)])
def test_incomplete_or_rehashed_plan_rejected_before_output(tmp_path, monkeypatch, status, sha):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "plan.json", {})
    write_json(run / "completed.json", {"status": status, "plan_sha256": sha})
    output = tmp_path / "docs" / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summarize", "--run", str(run), "--out", str(output)])
    with pytest.raises(ValueError):
        summary.main()
    assert not output.exists() and not output.parent.exists()


def test_existing_report_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    output.write_bytes(b"keep original")
    monkeypatch.setattr(sys, "argv", ["summarize", "--run", str(tmp_path), "--out", str(output)])
    with pytest.raises(FileExistsError, match="overwrite"):
        summary.main()
    assert output.read_bytes() == b"keep original"


@pytest.mark.parametrize("mutation", [None, "wrong_purpose", "wrong_architecture", "nonfinite", "optimizer_state", "different_branch"])
def test_checkpoint_diagnostic_provenance_and_inference_only_contract(tmp_path, mutation):
    plan = {"manifest_sha256": "m"*64, "initial_checkpoint_sha256": "a"*64,
            "settings": {"learning_rate": .0003}, "schedules": {"A_keep16": {"sample_ids": ["old"]}}}
    checkpoint = {"architecture_version": "lipnet-compact-v1", "num_classes": 69, "lstm_hidden": 256,
        **tokenizer_metadata(), "vocab": summary.GRAPHEME_VOCAB, "state_dict": {"conv.weight": torch.ones(2)},
        "training_data": {"mode": "smoke_only", "purpose": "train_subset_transition_diagnostic",
            "manifest_sha256": plan["manifest_sha256"], "sample_ids": ["old"],
            "generalization_evaluation": False, "deployment": False},
        "training_run": {"transition_diagnostic": True, "branch": "A_keep16", "attempt_updates": 1,
            "optimizer_steps": 1, "skipped_optimizer_steps": 0, "plan_sha256": "p"*64,
            "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"], "inference_only": True,
            "selection": "fixed_attempt_schedule", "settings": plan["settings"]}}
    if mutation == "wrong_purpose":
        checkpoint["training_data"]["purpose"] = "train_subset_overfit"
    elif mutation == "wrong_architecture":
        checkpoint["architecture_version"] = "lipnet-residual-v1"
    elif mutation == "nonfinite":
        checkpoint["state_dict"]["conv.weight"][0] = float("nan")
    elif mutation == "optimizer_state":
        checkpoint["optimizer_state_dict"] = {}
    elif mutation == "different_branch":
        checkpoint["training_run"]["branch"] = "B_expand64"
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    def check():
        summary.validate_checkpoint(path, plan, branch="A_keep16", step=1, updates=1, skips=0, plan_sha="p"*64)
    if mutation is None:
        check()
    else:
        with pytest.raises(ValueError):
            check()


def test_complete_artifact_graph_with_shared_common_snapshot(tmp_path, monkeypatch):
    """Exercise all file/hash joins with a one-attempt protocol and synthetic outputs.

    Only external source/model provenance are stubbed; raw posteriors, summary
    hashes, checkpoint hashes, telemetry, schedule and exposure joins remain real.
    """
    monkeypatch.setattr(summary, "STEPS", [0, 1])
    monkeypatch.setattr(summary, "CONTROL_STEPS", [0, 1])
    monkeypatch.setattr(summary, "BEAM_STEPS", [0, 1])
    ids = [f"s{i:02d}" for i in range(64)]
    records = {s: source(s) for s in ids}
    settings = {"grad_scaler_init_scale": 1024., "learning_rate": .0003, "grad_clip": 1.}
    plan = {"cohorts": {"short16": ids[:16], "additional48": ids[16:], "expanded64": ids},
            "settings": settings, "scope": {"train_only": True}, "manifest_sha256": "m"*64,
            "initial_checkpoint_sha256": "a"*64, "source_hashes": {}, "schedules": {}}
    for branch, cohort in summary.BRANCHES.items():
        plan["schedules"][branch] = make_training_schedule(plan["cohorts"][cohort], attempt_updates=1)
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "plan.json", plan)
    make_snapshot(run / "common/step_0000", plan, records, 0)
    monkeypatch.setattr(summary, "validate_plan", lambda p: records)
    monkeypatch.setattr(summary, "validate_checkpoint", lambda *a, **k: None)
    completed = {"status": "completed", "plan_sha256": summary.sha256_file(run / "plan.json"),
                 "common_summary_sha256": summary.sha256_file(run / "common/step_0000/summary.json"),
                 "train_cache_ids_opened": ids, "validation_cache_access": False, "test_cache_access": False,
                 "deployment": False, "branches": {}, "branch_completed_sha256": {}}
    for branch in summary.BRANCHES:
        folder = run / branch
        folder.mkdir()
        checkpoint = folder / "step_0001.pt"
        checkpoint.write_bytes(branch.encode())
        checkpoint_sha = summary.sha256_file(checkpoint)
        make_snapshot(folder / "step_0001", plan, records, 1, checkpoint_sha)
        telemetry = telemetry_fixture(1)[0]
        telemetry[0].update(copy.deepcopy(plan["schedules"][branch]["batches"][0]))
        write_rows(folder / "telemetry.jsonl", telemetry)
        validation = summary.validate_telemetry(telemetry, plan["schedules"][branch], records, settings)
        spec = {"step": 1, "folder": str((folder / "step_0001").relative_to(run)),
                "checkpoint": str(checkpoint.relative_to(run)), "checkpoint_sha256": checkpoint_sha,
                "summary_sha256": summary.sha256_file(folder / "step_0001/summary.json")}
        report = {"branch": branch, "status": "completed", **validation, "snapshots": [spec],
                  "step0_folder": "common/step_0000", "telemetry_sha256": summary.sha256_file(folder / "telemetry.jsonl"),
                  "last_checkpoint_sha256": checkpoint_sha, "peak_allocated_mib": 0., "peak_reserved_mib": 0., "seconds": 1.}
        write_json(folder / "completed.json", report)
        completed["branch_completed_sha256"][branch] = summary.sha256_file(folder / "completed.json")
        completed["branches"][branch] = {k: report[k] for k in ("attempt_updates", "optimizer_steps", "skipped_optimizer_steps", "seconds")}
    write_json(run / "completed.json", completed)
    result = summary.build_summary(run)
    assert result["status"] == "completed_and_verified"
    assert result["common_step0"]["cohorts"]["short16"]["normal"]["count"] == 16
    assert result["common_step0"]["cohorts"]["additional48"]["normal"]["count"] == 48
    assert all("0000" not in branch["snapshots"] for branch in result["branches"].values())
    assert result["provenance"]["model_forward_calls"] == result["provenance"]["beam_decodes"] == 0
    assert result["final_comparison"]["short16"]["normal"]["B_minus_A"]["greedy"]["exact_count"] == 0
    json.dumps(result, allow_nan=False)
    # A changed branch file must fail even while the top-level completion stays valid.
    with (run / "A_keep16/telemetry.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summary.build_summary(run)
