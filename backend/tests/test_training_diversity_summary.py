"""Stage16 raw output audits fail closed even after an attacker refreshes hashes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts"), str(Path(__file__).parent)]
import summarize_training_diversity as summary
from models.tokenizer import encode_text


def fixture(step=0):
    records = {name: {"id": name, "text": text, "frames": 5 + index, "split": "val" if name == "v" else "train",
        "speaker_id": "unseen" if name == "v" else "seen", "recording_id": "r_" + name}
        for index, (name, text) in enumerate(zip("oahv", ["가", "나", "다", "라"]))}
    plan = {"cohorts": {cohort: [name] for cohort, name in zip(summary.COHORTS, "oahv")}}
    membership = {ids[0]: cohort for cohort, ids in plan["cohorts"].items()}
    rows, arrays, normal = [], {}, {}
    for name, condition in summary.snapshot_pairs(plan, step):
        source = records[name]
        labels = np.asarray(encode_text(source["text"]), dtype=np.int64)
        logits = np.full((source["frames"], 69), -9., dtype=np.float32)
        logits[:, 0] = 0.
        if condition != "static_middle":
            logits[0, labels[0]], logits[2, labels[1]] = 2., 3.
        logp = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
        has_beam = step in summary.ENDPOINTS and condition == "normal"
        beam = summary.ctc_beam_search_decode(logp, beam_width=10) if has_beam else ""
        diagnostic = summary.analyze_ctc(logp, labels, beam_text=beam)
        if not has_beam:
            diagnostic = {k: v for k, v in diagnostic.items() if not k.startswith("beam")}
        _, transform = summary.transform_frames(np.zeros((source["frames"], 1, 1, 3), np.uint8),
            "shuffle" if condition == "shuffle_42" else condition, sample_id=name, seed=42)
        transform["input_shape"] = [source["frames"], 96, 96, 3]
        loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
            torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
        row = {"id": name, "condition": condition, "branch": "A_keep64", "step": step,
            "cumulative_attempt": 9600 + step, "split": source["split"], "cohort": membership[name],
            "speaker_id": source["speaker_id"], "recording_id": source["recording_id"],
            "reference": source["text"], "frames": source["frames"], "length_bin": summary.length_bin(source["frames"]),
            "posterior_key": name + "__" + condition, "greedy": summary.prediction_row(source, diagnostic["greedy_text"]),
            "beam": summary.prediction_row(source, beam) if has_beam else None,
            "ctc": diagnostic, "ctc_loss_per_target_token": float(loss), "transform": transform, "comparison_to_normal": None}
        if condition == "normal":
            normal[name] = (logp, diagnostic["greedy_text"])
        else:
            row["comparison_to_normal"] = summary.compare_posteriors(normal[name][0], logp,
                normal_text=normal[name][1], control_text=diagnostic["greedy_text"])
        rows.append(row)
        arrays[row["posterior_key"]] = logp
    report = {"step": step, "cumulative_attempt": 9600 + step, "branch": "A_keep64", "rows": len(rows),
        "checkpoint_sha256": "c" * 64, "seconds": .1, "conditions": summary.CONDITIONS if step in summary.ENDPOINTS else ["normal"],
        "control_cohorts": summary.HELDOUT if step in summary.ENDPOINTS else [], "beam_performed": step in summary.ENDPOINTS}
    return plan, records, rows, arrays, report


def save(folder, rows, arrays, report):
    (folder / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    report["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    report["posteriors_sha256"] = summary.sha256_file(folder / "posteriors.npz")
    (folder / "summary.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@pytest.mark.parametrize("step", [0, 800, 1600, 3200])
def test_exact_schedule_redecodes_endpoint_normal_beams_only(tmp_path, monkeypatch, step):
    plan, records, rows, arrays, report = fixture(step)
    save(tmp_path, rows, arrays, report)
    original, calls = summary.ctc_beam_search_decode, []
    def counted(value, beam_width):
        calls.append(value.copy())
        return original(value, beam_width=beam_width)
    monkeypatch.setattr(summary, "ctc_beam_search_decode", counted)
    result, raw = summary.validate_snapshot(tmp_path, plan, records, "A_keep64", step, "c" * 64)
    assert raw == rows and len(calls) == (4 if step in summary.ENDPOINTS else 0)
    assert result["rows"] == (8 if step in summary.ENDPOINTS else 4)
    assert set(result["cohorts"]["original64"]) == {"normal"}
    assert result["cohorts"]["additional192"]["normal"]["count"] == 1
    assert set(result["normal_by_speaker"]["val20"]) == {"unseen"}


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "reorder", "branch", "step", "cumulative_attempt", "split",
    "cohort", "recording_id", "speaker_id", "reference", "length_bin", "beam_forged", "greedy_forged",
    "control_beam", "missing_beam", "posterior_dtype", "posterior_shape", "posterior_nan", "posterior_logits",
    "extra_posterior", "ctc", "loss", "comparison", "transform", "pixels", "control_cohorts", "beam_flag", "seconds"])
def test_tampering_rejected_even_when_file_hashes_are_refreshed(tmp_path, mutation):
    plan, records, rows, arrays, report = fixture()
    if mutation == "drop":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation in ("branch", "split", "cohort", "recording_id", "speaker_id", "reference", "length_bin"):
        rows[0][mutation] = "wrong"
    elif mutation in ("step", "cumulative_attempt"):
        rows[0][mutation] += 1
    elif mutation == "beam_forged":
        rows[0]["beam"] = summary.prediction_row(records["o"], "허위")
        rows[0]["ctc"] = summary.analyze_ctc(arrays["o__normal"], np.asarray(encode_text("가")), beam_text="허위")
    elif mutation == "greedy_forged":
        rows[0]["greedy"] = summary.prediction_row(records["o"], "허위")
    elif mutation == "control_beam":
        rows[3]["beam"] = rows[2]["beam"]
    elif mutation == "missing_beam":
        rows[0]["beam"] = None
    elif mutation == "posterior_dtype":
        arrays["o__normal"] = arrays["o__normal"].astype(np.float64)
    elif mutation == "posterior_shape":
        arrays["o__normal"] = arrays["o__normal"][:-1]
    elif mutation == "posterior_nan":
        arrays["o__normal"][0, 0] = np.nan
    elif mutation == "posterior_logits":
        arrays["o__normal"] += 2
    elif mutation == "extra_posterior":
        arrays["extra"] = arrays["o__normal"]
    elif mutation == "ctc":
        rows[0]["ctc"]["blank_posterior_mean"] += .1
    elif mutation == "loss":
        rows[0]["ctc_loss_per_target_token"] += 1
    elif mutation == "comparison":
        rows[3]["comparison_to_normal"]["mean_total_variation"] += .1
    elif mutation == "transform":
        rows[4]["transform"]["source_frame_indices"].reverse()
    elif mutation == "pixels":
        rows[0]["transform"]["pixel_changed_frame_count"] = 1
    elif mutation == "control_cohorts":
        report["control_cohorts"] = summary.COHORTS
    elif mutation == "beam_flag":
        report["beam_performed"] = False
    elif mutation == "seconds":
        report["seconds"] = -1
    save(tmp_path, rows, arrays, report)
    with pytest.raises(ValueError):
        summary.validate_snapshot(tmp_path, plan, records, "A_keep64", 0, "c" * 64)


def test_hash_checked_before_beam_execution(tmp_path, monkeypatch):
    plan, records, rows, arrays, report = fixture()
    save(tmp_path, rows, arrays, report)
    with (tmp_path / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Hash failure must precede decoding")
    monkeypatch.setattr(summary, "ctc_beam_search_decode", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summary.validate_snapshot(tmp_path, plan, records, "A_keep64", 0, "c" * 64)


def test_paired_differences_use_ids_and_after_minus_before():
    _, _, rows, _, _ = fixture()
    after = copy.deepcopy(rows)[::-1]
    for row in after:
        if row["condition"] == "normal":
            row["greedy"]["cer"] += .25
            row["beam"]["cer"] -= .25
    result = summary.paired_normal(rows, after)
    assert all(result[c]["greedy"]["worsened_count"] == 1 for c in summary.COHORTS)
    assert all(result[c]["beam"]["improved_count"] == 1 for c in summary.COHORTS)
    assert result["train60"]["greedy"]["cer_delta"]["mean"] == .25


def test_paired_differences_reject_missing_sample():
    _, _, rows, _, _ = fixture()
    after = copy.deepcopy(rows)
    after[0]["id"] = "different"
    with pytest.raises(ValueError, match="Unmatched"):
        summary.paired_normal(rows, after)


def test_midpoint_beam_is_forbidden_even_if_score_and_diagnostics_are_valid(tmp_path):
    plan, records, rows, arrays, report = fixture(800)
    beam = summary.ctc_beam_search_decode(arrays["o__normal"], beam_width=10)
    rows[0]["beam"] = summary.prediction_row(records["o"], beam)
    rows[0]["ctc"] = summary.analyze_ctc(arrays["o__normal"], np.asarray(encode_text("가")), beam_text=beam)
    save(tmp_path, rows, arrays, report)
    with pytest.raises(ValueError, match="Beam schedule"):
        summary.validate_snapshot(tmp_path, plan, records, "A_keep64", 800, "c" * 64)


def telemetry_fixture():
    import test_learning_rate_summary as previous
    rows, _, records, settings, source = previous.continuation_fixture()
    schedule = summary.make_training_schedule(["a", "b"], attempt_updates=4, batch_size=2, seed=42)
    source["attempt"] = 9600
    source["counters"] = {"optimizer_steps": 9599, "skipped_optimizer_steps": 1,
        "attempted_exposures": {"a": 300}, "successful_update_exposures": {"a": 299}}
    records["b"]["frames"] = 7
    settings["learning_rate"] = .0001
    for row, batch in zip(rows, schedule["batches"]):
        row.update(copy.deepcopy(batch))
        row["optimizer_steps"] -= 7
        row["skipped_optimizer_steps"] -= 1
        row["learning_rate"] = .0001
        row["cumulative_attempt"] = 9600 + row["attempt"]
        row["cumulative_optimizer_steps"] = 9599 + row["optimizer_steps"]
        row["cumulative_skipped_optimizer_steps"] = 1 + row["skipped_optimizer_steps"]
        row["frames"] = [records[s]["frames"] for s in batch["ids"]]
        row["raw_frame_presentations"] = sum(row["frames"])
        row["padded_frame_presentations"] = max(row["frames"]) * 2
    return rows, schedule, records, settings, source


def test_local_counters_preserve_source_adam_history_and_measure_real_frames():
    args = telemetry_fixture()
    before = copy.deepcopy(args)
    result = summary.validate_extension_telemetry(*args)
    assert result["extension"]["optimizer_steps"] == 3
    assert result["extension"]["skipped_optimizer_steps"] == 1
    assert result["cumulative"] == {"optimizer_steps": 9602, "skipped_optimizer_steps": 2,
        "attempted_exposures": {"a": 304, "b": 4}, "successful_update_exposures": {"a": 302, "b": 3}}
    assert result["final_scaler_state"]["scale"] == 1024.
    assert result["final_scaler_state"]["_growth_tracker"] == 1
    assert result["compute"]["attempted_unpadded_frames"] == 44
    assert result["compute"]["attempted_padded_frames"] == 56
    assert result["compute"]["successful_update_unpadded_frames"] == 33
    assert summary.compare_nested(before, args)["exact"]


@pytest.mark.parametrize("mutation", ["attempt", "global_updates", "global_skips", "local_updates", "new_exposures",
    "raw_frames", "padded_frames", "source_weights", "source_tracker", "skip", "lr", "batch"])
def test_telemetry_rejects_history_reset_or_false_compute_accounting(mutation):
    rows, schedule, records, settings, source = telemetry_fixture()
    if mutation == "attempt":
        rows[0]["cumulative_attempt"] = 1
    elif mutation == "global_updates":
        rows[0]["cumulative_optimizer_steps"] = 1
    elif mutation == "global_skips":
        rows[0]["cumulative_skipped_optimizer_steps"] = 0
    elif mutation == "local_updates":
        rows[0]["optimizer_steps"] += 9599
    elif mutation == "new_exposures":
        schedule["exposures"]["b"] += 1
    elif mutation == "raw_frames":
        rows[0]["raw_frame_presentations"] += 1
    elif mutation == "padded_frames":
        rows[0]["padded_frame_presentations"] = rows[0]["raw_frame_presentations"]
    elif mutation == "source_weights":
        source["model_state_dict"]["fc.weight"][0] += .01
    elif mutation == "source_tracker":
        source["scaler_state_dict"]["_growth_tracker"] = 0
    elif mutation == "skip":
        rows[2]["skipped"] = False
    elif mutation == "lr":
        rows[0]["learning_rate"] = .0003
    elif mutation == "batch":
        rows[0]["ids"].reverse()
    with pytest.raises(ValueError):
        summary.validate_extension_telemetry(rows, schedule, records, settings, source)


def state_fixture(tmp_path, step=0):
    import test_learning_budget_summary as previous
    previous.resume_fixture(tmp_path)
    source = torch.load(tmp_path / "training_state_6400.pt", map_location="cpu", weights_only=True)
    source["attempt"] = 9600
    source["counters"] = {"optimizer_steps": 9599, "skipped_optimizer_steps": 1,
        "attempted_exposures": {"old": 300}, "successful_update_exposures": {"old": 299}}
    source["optimizer_state_dict"]["param_groups"][0]["lr"] = .0001
    source["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(9599.)
    source["scaler_state_dict"] = {"scale": 8192., "growth_factor": 2., "backoff_factor": .5,
        "growth_interval": 2000, "_growth_tracker": 1551}
    schedule = summary.make_training_schedule(["old", "new"], attempt_updates=3200, batch_size=2, seed=42)
    plan = {"branches": {"B_expand256": {"train_ids": ["old", "new"], "schedule": schedule}},
        "cohorts": {"original64": ["old"]}, "prior_state_sha256": "s" * 64,
        "manifest_sha256": "m" * 64, "source_hashes": {"test_source": "a" * 64}, "settings": {"learning_rate": .0001}}
    telemetry = [{"ids": ["old", "new"], "skipped": False, "scale_before": 8192., "scale_after": 8192.}
        for _ in range(step)]
    candidate = copy.deepcopy(source)
    candidate["attempt"] = step
    candidate["counters"] = summary.local_counters(telemetry, ["old", "new"])
    candidate["schedule"] = summary.schedule_metadata(schedule)
    candidate["provenance"] = {**summary.expected_provenance(plan, "B_expand256", step), "plan_sha256": "p" * 64}
    candidate["source_hashes"] = plan["source_hashes"]
    candidate["settings"] = plan["settings"]
    candidate["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(float(9599 + step))
    candidate["scaler_state_dict"] = summary.scaler_after(telemetry, source["scaler_state_dict"])
    return candidate, source, copy.deepcopy(candidate["model_state_dict"]), plan, "B_expand256", step, telemetry, "p" * 64


@pytest.mark.parametrize("step", [0, 4])
def test_saved_state_has_local_attempt_but_cumulative_adam_step(tmp_path, step):
    args = state_fixture(tmp_path, step)
    before = copy.deepcopy(args)
    result = summary.validate_resume_state(*args)
    assert result["local_attempt"] == step
    assert result["cumulative_attempt"] == 9600 + step
    assert result["local_optimizer_steps"] == step
    assert result["cumulative_optimizer_steps"] == 9599 + step
    assert result["initial_source_runtime_exact"] == (step == 0)
    assert summary.compare_nested(before, args)["exact"]


@pytest.mark.parametrize("mutation", ["adam_reset", "attempt_global", "counters_global", "wrong_schedule", "wrong_scope",
    "initial_rng", "initial_moment", "initial_scaler", "initial_mode", "state_weights", "nonfinite_moment", "negative_variance"])
def test_saved_state_rejects_reset_optimizer_and_local_global_confusion(tmp_path, mutation):
    args = list(state_fixture(tmp_path))
    candidate = args[0]
    if mutation == "adam_reset":
        candidate["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(0.)
    elif mutation == "attempt_global":
        candidate["attempt"] = 9600
    elif mutation == "counters_global":
        candidate["counters"]["optimizer_steps"] = 9599
    elif mutation == "wrong_schedule":
        candidate["schedule"]["attempt_updates"] = 12800
    elif mutation == "wrong_scope":
        candidate["provenance"]["deployment"] = True
    elif mutation == "initial_rng":
        candidate["rng_state"]["torch_cpu"][0] = 1
    elif mutation == "initial_moment":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg"][0] += 1e-7
    elif mutation == "initial_scaler":
        candidate["scaler_state_dict"]["_growth_tracker"] = 0
    elif mutation == "initial_mode":
        candidate["module_training"][""] = False
    elif mutation == "state_weights":
        candidate["model_state_dict"]["weight"][0] += 1e-7
    elif mutation == "nonfinite_moment":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg"][0] = float("nan")
    elif mutation == "negative_variance":
        candidate["optimizer_state_dict"]["state"][0]["exp_avg_sq"][0] = -1.
    with pytest.raises(ValueError):
        summary.validate_resume_state(*args)


def test_saved_weights_must_match_telemetry_endpoint_norms():
    state = {prefix + "weight": torch.ones(2) * (index + 1)
        for index, prefix in enumerate(summary.GROUP_PREFIXES.values())}
    norms = summary.parameter_norms(state)
    row = {"parameter_update": {"total": {"after_l2": norms["total"]},
        "groups": {group: {"after_l2": norms[group]} for group in summary.GROUP_PREFIXES}}}
    summary.validate_checkpoint_telemetry(state, row)
    state["fc.weight"][0] += .01
    with pytest.raises(ValueError, match="weight norm"):
        summary.validate_checkpoint_telemetry(state, row)
