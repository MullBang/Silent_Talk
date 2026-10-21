"""Recompute fixed evaluation inputs, Unicode predictions and corpus metrics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_common import collapsed_ids, compose_ids, corpus, ctc_nll, evaluation_digest, score, self_check, unpack
from audit_sources import binding, check_binding, read, S5C

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
S4 = ROOT / "experiments/offline_training_step04_20261013"
S3 = ROOT / "experiments/offline_dataset_step03_20261013"


def array_digest(value: np.ndarray) -> str:
    """Published evaluator byte digest, implemented from its array encoding."""
    header = b"numpy:" + value.dtype.str.encode() + repr(value.shape).encode() + b":"
    return hashlib.sha256(header + np.ascontiguousarray(value).tobytes()).hexdigest()


def inspect() -> dict:
    self_tests = self_check()
    torch.set_num_threads(4)
    plan = read(BASE / "evaluation_plan.json")
    old_plan = read(S4 / "protocol/evaluation_plan.json")
    assert plan["items"] == [r for r in old_plan["items"] if r["split"] == "dev" and r["kind"] == "phrase"]
    assert len(plan["items"]) == 3
    assert plan["conditions_by_update"] == {"1680":["normal","static","shuffled"]}
    sources = {r["clip_id"]: r for r in read(S3 / "results/cache_handoff.json")["records"]}
    prepared = {}
    for item in plan["items"]:
        row = sources[item["clip_id"]]
        check_binding({"path": row["npz_path"], "sha256": row["npz_sha256"], "bytes": row["npz_bytes"]})
        with np.load(row["npz_path"], allow_pickle=False) as data:
            pixels = data["frames_rgb"].astype(np.float32) / np.float32(255.)
            pixels = (pixels - np.asarray([.485, .456, .406], dtype=np.float32)) / np.asarray([.229, .224, .225], dtype=np.float32)
            pixels = np.ascontiguousarray(pixels.transpose(3, 0, 1, 2))
            labels = data["label"].tolist()
        assert compose_ids(labels) == item["text"]
        inputs = {}
        for condition in ["normal", "static", "shuffled"]:
            indices = (list(range(item["T"])) if condition == "normal" else
                       [item["T"] // 2] * item["T"] if condition == "static" else item["shuffled_frame_indices"])
            inputs[condition] = {"indices": indices, "digest": array_digest(np.ascontiguousarray(pixels[:, indices]))}
        prepared[item["clip_id"]] = {"item": item, "labels": labels, "inputs": inputs}
    paths = sorted(BASE.glob("runs/**/evaluation_summary.json"))
    assert len(paths) == 2, (len(paths), paths)
    expected_identities = {(seed, update) for seed in [42, 43] for update in [1680]}
    seen = set()
    all_rows = posterior_count = all_frames = failures = 0
    maximum_ctc_difference = maximum_probability_sum_error = 0.
    groups = []
    primary = []
    for path in paths:
        value = read(path)
        identity = value["identity"]
        key = (identity["seed"], identity["completed_update"])
        conditions = plan["conditions_by_update"][str(key[1])]
        expected_count = 3 * len(conditions)
        assert key in expected_identities and key not in seen
        seen.add(key)
        assert len(value["rows"]) == value["count"] == expected_count
        jsonl = [json.loads(line) for line in (path.parent / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        assert jsonl == value["rows"]
        check_binding(value["state_isolation"])
        isolation = read(Path(value["state_isolation"]["path"]))
        assert isolation["before"] == isolation["after"] and isolation["fatal"] is None
        assert isolation["rows_saved"] == expected_count
        check_binding(identity["checkpoint"])
        payload = unpack(torch.load(identity["checkpoint"]["path"], map_location="cpu", weights_only=True))
        assert payload["completed_updates"] == identity["completed_update"]
        assert payload["identity"]["seed"] == identity["seed"]
        assert payload["identity"]["runtime"] == identity["runtime"]
        assert evaluation_digest(payload["state"]["model"]) == isolation["before"]["model"]
        source_handoff = read(S5C / "results/next_step_handoff.json")
        assert identity["checkpoint"] == source_handoff["checkpoints"][[42,43].index(identity["seed"])]
        assert isolation["before"]["rng"] == isolation["after"]["rng"]
        rng = payload["state"]["rng"]
        expected_rng = {"python": rng["python"], "numpy": rng["numpy"], "torch_cpu": rng["torch"], "torch_cuda": rng["cuda"]}
        assert evaluation_digest(expected_rng) == isolation["before"]["rng"]
        assert all(not flag for flag in isolation["before"]["module_training_flags"].values())
        del payload
        for flag in ["model_unchanged", "gradients_unchanged", "rng_restored_exact", "module_training_flags_restored"]:
            assert isolation[flag] is True
        assert value["state_isolation_passed"] is True and value["optimizer_updates"] == 0
        assert value["audio_input"] is False and value["teacher_model_used"] is False
        expected_rows = [(item["clip_id"], cond) for item in plan["items"] for cond in conditions]
        assert [(row["clip_id"], row["condition"]) for row in value["rows"]] == expected_rows
        recomputed = []
        nll_records = []
        for row in value["rows"]:
            prepared_row = prepared[row["clip_id"]]
            item = prepared_row["item"]
            assert row["identity"] == identity
            for field in ["parent_id", "split", "kind", "T"]:
                assert row[field] == item[field]
            assert row["reference_text"] == item["text"]
            is_failed = row["status"] == "failed"
            if is_failed:
                assert row["error"] and row["posterior"] is None
                assert row["raw_argmax_ids"] == row["collapsed_ids"] == []
                assert row["raw_text"] == row["metric_text"] == ""
                failures += 1
            else:
                assert row["status"] == "success" and row["error"] is None
                assert row["forward_attempted"] is True
                expected_input = prepared_row["inputs"][row["condition"]]
                assert row["frame_indices"] == expected_input["indices"]
                assert row["input_array_sha256"] == expected_input["digest"]
                assert row["reference_label_ids"] == prepared_row["labels"]
                check_binding(row["posterior"])
                posterior = Path(row["posterior"]["path"])
                assert posterior.parent == path.parent / "posteriors"
                with np.load(posterior, allow_pickle=False) as data:
                    assert set(data.files) == {"log_probs", "raw_argmax_ids", "collapsed_ids", "frame_indices", "length", "metadata_json"}
                    values = data["log_probs"]
                    assert values.dtype == np.float32 and values.shape == (row["T"], 69) and np.isfinite(values).all()
                    assert int(data["length"]) == row["T"] and data["frame_indices"].tolist() == expected_input["indices"]
                    metadata = json.loads(str(data["metadata_json"]))
                    assert metadata["identity"] == identity and metadata["clip_id"] == row["clip_id"] and metadata["condition"] == row["condition"]
                    assert metadata["audio_input"] is False and metadata["layout"] == "T,69"
                    assert array_digest(values) == row["log_probs_array_sha256"]
                    sums = np.exp(values.astype(np.float64)).sum(axis=1)
                    maximum_probability_sum_error = max(maximum_probability_sum_error, float(np.abs(sums - 1.).max()))
                    assert np.allclose(sums, 1., atol=1e-5, rtol=1e-5)
                    raw = values.argmax(axis=1).tolist()
                    collapsed = collapsed_ids(raw)
                    predicted = compose_ids(collapsed)
                    assert row["raw_argmax_ids"] == data["raw_argmax_ids"].tolist() == raw
                    assert row["collapsed_ids"] == data["collapsed_ids"].tolist() == collapsed
                    assert row["raw_text"] == predicted
                    assert row["metric_text"] == predicted.strip()
                    numpy_loss = ctc_nll(values, prepared_row["labels"])
                    reference_loss = float(torch.nn.functional.ctc_loss(
                        torch.from_numpy(values.astype(np.float64)).unsqueeze(1), torch.tensor(prepared_row["labels"], dtype=torch.int64),
                        torch.tensor([row["T"]]), torch.tensor([len(prepared_row["labels"])]), reduction="sum", zero_infinity=False))
                    difference = abs(numpy_loss - reference_loss)
                    maximum_ctc_difference = max(maximum_ctc_difference, difference)
                    assert difference <= 1e-8 + 1e-10 * abs(reference_loss)
                    nll_records.append({"clip_id": row["clip_id"], "condition": row["condition"], "numpy_float64_CTC_NLL": numpy_loss,
                                        "torch_float64_CTC_NLL": reference_loss, "absolute_difference": difference})
                posterior_count += 1
                all_frames += row["T"]
            scoring = score(item["text"], row["raw_text"], is_failed)
            assert scoring == row["metrics"]
            recomputed.append((row, scoring))
            all_rows += 1
        cohorts = {}
        for cohort in sorted({(row["split"], row["kind"], row["condition"]) for row, _ in recomputed}):
            cohorts["/".join(cohort)] = corpus([scoring for row, scoring in recomputed if (row["split"], row["kind"], row["condition"]) == cohort])
        aggregate = corpus([scoring for _, scoring in recomputed])
        assert value["metrics"]["cohorts"] == cohorts and value["metrics"]["all_rows"] == aggregate
        assert value["failures"] == aggregate["failure_count"]
        assert value["posterior_count"] == expected_count - aggregate["failure_count"]
        assert value["successful_posterior_frames"] == sum(r["T"] for r in value["rows"] if r["status"] == "success")
        assert value["forward_calls"] == sum(r["forward_attempted"] for r in value["rows"])
        assert value["attempted_forward_frames"] == sum(r["T"] for r in value["rows"] if r["forward_attempted"])
        assert value["dev_evaluated"] is True and value["generalization_measured"] is False
        assert value["metrics"]["primary_cohort_key"] == "dev/phrase/normal"
        assert value["planned_frames"] == 96 * len(conditions)
        check_binding(value["rows_jsonl"])
        primary.append({"seed": key[0], "update": key[1], **cohorts["dev/phrase/normal"]})
        groups.append({"summary": binding(path), "identity": identity, "corpus_metrics": cohorts, "CTC_checks": nll_records})
    assert seen == expected_identities and all_rows == 18 and all_frames == 576
    return {"schema": "offline-stage5d-independent-evaluation-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed", "algorithm_self_checks": self_tests, "producer_imports": 0,
            "network_forward_calls": 0, "optimizer_updates": 0, "checkpoint_evaluations": 2,
            "evaluation_rows": all_rows, "posteriors": posterior_count, "posterior_frames": all_frames, "failed_rows": failures,
            "max_probability_sum_error": maximum_probability_sum_error, "max_Float64_CTC_absolute_difference": maximum_ctc_difference,
            "primary_final_results": primary, "details": groups}


if __name__ == "__main__":
    result = inspect()
    output = HERE / "evaluation_audit.json"
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "evaluation_rows", "posteriors", "posterior_frames", "failed_rows", "max_Float64_CTC_absolute_difference"]}))
