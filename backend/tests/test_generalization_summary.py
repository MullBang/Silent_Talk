"""Saved posterior audit rejects leakage, corrupt metrics and dishonest scope."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
import summarize_generalization as summary
from learning_budget_state import compare_nested, compare_state_dicts
from models.decoder import ctc_beam_search_decode
from models.tokenizer import encode_text


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fixture_rows():
    records = {"a": {"id": "a", "text": "가", "frames": 5, "split": "train", "speaker_id": "seen", "recording_id": "new_a"},
               "b": {"id": "b", "text": "나", "frames": 6, "split": "val", "speaker_id": "unseen", "recording_id": "new_b"}}
    state = {"weight": torch.ones(2)}
    plan = {"selection": {"cohorts": {"train60": ["a"], "val20": ["b"]}},
            "models": {"source6400": {"checkpoint_sha256": "c"*64, "state_sha256": summary.state_digest(state)}}}
    rows, arrays = [], {}
    for source in records.values():
        normal, normal_text = None, None
        labels = np.asarray(encode_text(source["text"]), dtype=np.int64)
        for index, condition in enumerate(summary.CONDITIONS):
            logits = np.full((source["frames"], 69), -9., dtype=np.float32)
            logits[:, 0] = 0
            if condition != "static_middle":
                logits[0, labels[0]] = 2
                logits[2, labels[1]] = 3
            logits[-1, labels[-1]] = float(index)/2
            logp = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
            beam = ctc_beam_search_decode(logp, beam_width=10) if condition == "normal" else None
            diagnostic = summary.analyze_ctc(logp, labels, beam_text=beam or "")
            if beam is None:
                diagnostic = {key: value for key, value in diagnostic.items() if not key.startswith("beam")}
            _, transform = summary.transform_frames(np.zeros((source["frames"], 1, 1, 3), np.uint8),
                "shuffle" if condition == "shuffle_42" else condition, sample_id=source["id"], seed=42)
            transform["input_shape"] = [source["frames"], 96, 96, 3]
            loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
            posterior_key = source["id"] + "__" + condition
            row = {"id": source["id"], "condition": condition, "model": "source6400", "split": source["split"],
                "cohort": "train60" if source["split"] == "train" else "val20", "speaker_id": source["speaker_id"],
                "recording_id": source["recording_id"], "reference": source["text"], "frames": source["frames"],
                "length_bin": summary.length_bin(source["frames"]), "posterior_key": posterior_key,
                "greedy": summary.prediction_row(source, diagnostic["greedy_text"]),
                "beam": summary.prediction_row(source, beam) if beam is not None else None,
                "ctc": diagnostic, "ctc_loss_per_target_token": float(loss), "transform": transform,
                "comparison_to_normal": None}
            if condition == "normal":
                normal, normal_text = logp, diagnostic["greedy_text"]
            else:
                row["comparison_to_normal"] = summary.compare_posteriors(normal, logp,
                    normal_text=normal_text, control_text=diagnostic["greedy_text"])
            arrays[posterior_key] = logp
            rows.append(row)
    report = {"status": "completed", "model": "source6400", "plan_sha256": "p"*64,
        "checkpoint_sha256": "c"*64, "clips": 2, "forwards": 6, "beam_rows": 2, "weights_unchanged": True,
        "rng_unchanged": True, "training_performed": False, "test_cache_access": False, "seconds": .1,
        "evaluation_mode_unchanged": True, "state_before_sha256": summary.state_digest(state),
        "state_after_sha256": summary.state_digest(state), "state_audit": compare_state_dicts(state, state),
        "rng_audit": compare_nested({"seed": 42}, {"seed": 42})}
    return plan, records, rows, arrays, report


def save_fixture(folder, rows, arrays, report):
    (folder / "rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n" for row in rows), encoding="utf-8")
    np.savez_compressed(folder / "posteriors.npz", **arrays)
    report["rows_sha256"] = summary.sha256_file(folder / "rows.jsonl")
    report["posteriors_sha256"] = summary.sha256_file(folder / "posteriors.npz")
    save_json(folder / "report.json", report)


def test_valid_saved_grid_redecodes_beam_and_keeps_cohorts_separate(tmp_path, monkeypatch):
    plan, records, rows, arrays, report = fixture_rows()
    save_fixture(tmp_path, rows, arrays, report)
    original = summary.ctc_beam_search_decode
    calls = []
    def counted(values, beam_width):
        calls.append(values.copy())
        return original(values, beam_width=beam_width)
    monkeypatch.setattr(summary, "ctc_beam_search_decode", counted)
    result, reread = summary.validate_model(tmp_path, plan, records, "source6400", "p"*64)
    assert len(calls) == 2 and result["forwards"] == 6 and reread == rows
    assert result["cohorts"]["train60"]["normal"]["count"] == 1
    assert "beam" in result["cohorts"]["val20"]["normal"]
    assert "beam" not in result["cohorts"]["val20"]["shuffle_42"]
    assert set(result["normal_by_speaker"]["val20"]) == {"unseen"}


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "reorder", "wrong_split", "wrong_cohort", "wrong_speaker",
    "wrong_recording", "wrong_bin", "wrong_reference", "wrong_model", "beam_control", "beam_missing", "beam_forged",
    "greedy_forged", "ctc_forged", "loss_forged", "comparison_forged", "transform_indices", "transform_pixels",
    "key_alias", "extra_array", "posterior_dtype", "posterior_shape", "posterior_nan", "posterior_logits",
    "trained", "test_access", "weights_changed", "rng_changed", "state_digest", "rng_difference", "hidden_exclusion",
    "missing_state_tensor", "mode_changed", "beam_count", "negative_duration"])
def test_tampered_self_consistent_hashes_cannot_hide_substantive_errors(tmp_path, mutation):
    plan, records, rows, arrays, report = fixture_rows()
    if mutation == "drop":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation.startswith("wrong_"):
        field = {"split": "split", "cohort": "cohort", "speaker": "speaker_id", "recording": "recording_id",
                 "bin": "length_bin", "reference": "reference", "model": "model"}[mutation[6:]]
        rows[0][field] = "test" if field == "split" else "wrong"
    elif mutation == "beam_control":
        rows[1]["beam"] = rows[0]["beam"]
    elif mutation == "beam_missing":
        rows[0]["beam"] = None
    elif mutation == "beam_forged":
        rows[0]["beam"] = summary.prediction_row(records["a"], "허위")
        rows[0]["ctc"] = summary.analyze_ctc(arrays["a__normal"], np.asarray(encode_text("가")), beam_text="허위")
    elif mutation == "greedy_forged":
        rows[0]["greedy"] = summary.prediction_row(records["a"], "허위")
    elif mutation == "ctc_forged":
        rows[0]["ctc"]["blank_posterior_mean"] += .1
    elif mutation == "loss_forged":
        rows[0]["ctc_loss_per_target_token"] += 1
    elif mutation == "comparison_forged":
        rows[1]["comparison_to_normal"]["mean_total_variation"] += .1
    elif mutation == "transform_indices":
        rows[2]["transform"]["source_frame_indices"].reverse()
    elif mutation == "transform_pixels":
        rows[0]["transform"]["pixel_changed_frame_count"] = 1
    elif mutation == "key_alias":
        arrays["alias"] = arrays.pop(rows[0]["posterior_key"])
        rows[0]["posterior_key"] = "alias"
    elif mutation == "extra_array":
        arrays["extra"] = arrays["a__normal"]
    elif mutation == "posterior_dtype":
        arrays["a__normal"] = arrays["a__normal"].astype(np.float64)
    elif mutation == "posterior_shape":
        arrays["a__normal"] = arrays["a__normal"][:-1]
    elif mutation == "posterior_nan":
        arrays["a__normal"][0, 0] = np.nan
    elif mutation == "posterior_logits":
        arrays["a__normal"] += 2
    elif mutation in ("trained", "test_access", "weights_changed", "rng_changed", "mode_changed"):
        field = {"trained": "training_performed", "test_access": "test_cache_access", "weights_changed": "weights_unchanged",
                 "rng_changed": "rng_unchanged", "mode_changed": "evaluation_mode_unchanged"}[mutation]
        report[field] = not report[field]
    elif mutation == "state_digest":
        report["state_after_sha256"] = "d"*64
    elif mutation == "rng_difference":
        report["rng_audit"]["difference_count"] = 1
    elif mutation == "hidden_exclusion":
        report["state_audit"]["excluded_keys"] = ["weight"]
    elif mutation == "missing_state_tensor":
        report["state_audit"]["tensors"] = {}
    elif mutation == "beam_count":
        report["beam_rows"] += 1
    elif mutation == "negative_duration":
        report["seconds"] = -1
    else:
        raise AssertionError(mutation)
    save_fixture(tmp_path, rows, arrays, report)
    with pytest.raises(ValueError):
        summary.validate_model(tmp_path, plan, records, "source6400", "p"*64)


def test_hash_change_detected_before_redecoding(tmp_path, monkeypatch):
    plan, records, rows, arrays, report = fixture_rows()
    save_fixture(tmp_path, rows, arrays, report)
    with (tmp_path / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must fail hash before beam decoding")
    monkeypatch.setattr(summary, "ctc_beam_search_decode", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summary.validate_model(tmp_path, plan, records, "source6400", "p"*64)


def test_paired_differences_are_matched_by_id_and_sign_is_explicit():
    _, _, rows, _, _ = fixture_rows()
    models = {key: copy.deepcopy(rows) for key in summary.MODEL_PATHS}
    for row in models["B_reduce9600"]:
        for decoder in ("greedy", "beam"):
            if row[decoder] is not None:
                row[decoder]["cer"] += .5
    models["B_reduce9600"].reverse()
    result = summary.paired_normal(models)["B_reduce9600_minus_A_keep9600"]
    assert result["train60"]["greedy"]["worsened_count"] == 1
    assert result["val20"]["greedy"]["cer_delta"]["mean"] == .5
    assert result["train60"]["greedy"]["improved_count"] == 0


def test_paired_differences_reject_unmatched_ids():
    _, _, rows, _, _ = fixture_rows()
    models = {key: copy.deepcopy(rows) for key in summary.MODEL_PATHS}
    models["B_reduce9600"][0]["id"] = "leaked"
    with pytest.raises(ValueError, match="Unmatched"):
        summary.paired_normal(models)


def test_state_digest_is_order_independent_but_shape_name_dtype_and_values_sensitive():
    a = {"a": torch.ones(2), "b": torch.zeros(2)}
    expected = summary.state_digest(a)
    assert expected == summary.state_digest(dict(reversed(list(a.items()))))
    for changed in ({"a": torch.ones(1, 2), "b": a["b"]}, {"a": a["a"].double(), "b": a["b"]},
                    {"renamed": a["a"], "b": a["b"]}, {"a": a["a"]+1, "b": a["b"]}):
        assert summary.state_digest(changed) != expected


def test_json_duplicate_key_and_nonfinite_values_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"status": "completed", "status": "different"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        summary.read_json(path)
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Nonfinite"):
        summary.read_json(path)
