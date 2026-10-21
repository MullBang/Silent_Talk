"""Check final paired comparisons and output statistics against independent rows."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from audit_sources import binding, check_binding, read

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def diagnostics(rows: list[dict]) -> dict:
    result = {}
    for key in sorted({(r["split"], r["kind"], r["condition"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["kind"], r["condition"]) == key]
        predictions = [r["metric_text"] for r in group]
        characters = sum(map(len, predictions))
        replacements = sum(text.count("\ufffd") for text in predictions)
        counts = Counter(char for text in predictions for char in text if not char.isspace())
        nonspace = sum(counts.values())
        repeats = sum(row["metrics"]["repetition_ge5"] for row in group)
        result["/".join(key)] = {
            "count":len(group), "empty_prediction_count":sum(not x for x in predictions),
            "failed_count":sum(row["status"] == "failed" for row in group),
            "repetition_ge5_count":repeats, "repetition_ge5_clip_fraction":repeats / len(group),
            "replacement_characters":replacements, "prediction_characters_including_internal_spaces":characters,
            "replacement_character_fraction":replacements / characters if characters else 0.,
            "replacement_fraction_empty_denominator":characters == 0,
            "predicted_nonspace_characters":nonspace,
            "top10_nonspace_characters":[{"character":c, "count":n, "fraction":n / nonspace} for c,n in sorted(counts.items(), key=lambda item:(-item[1], item[0]))[:10]],
        }
    return result


def inspect() -> dict:
    aggregate_path = BASE / "results/aggregate_metrics.json"
    aggregate = read(aggregate_path)
    audit_path = HERE / "evaluation_audit.json"
    audit = read(audit_path)
    assert audit["status"] == "passed"
    assert aggregate["evaluations"] == audit["evaluation_rows"] == 1080
    assert aggregate["checkpoints"] == 12 and aggregate["failures"] == audit["failed_rows"]
    assert aggregate["primary_checkpoint"] == 1260 and aggregate["best_checkpoint_selection"] is False
    independent = {}
    output_rows = {}
    for group in audit["details"]:
        check_binding(group["summary"])
        row = group["identity"]
        key = (row["seed"],row["arm"],row["completed_update"])
        independent[key] = group["corpus_metrics"]
        output_rows[key] = read(Path(group["summary"]["path"]))["rows"]
    for row in aggregate["sources"]:
        check_binding(row)
    assert {r["path"] for r in aggregate["sources"]} == {g["summary"]["path"] for g in audit["details"]}
    assert len(aggregate["final_models"]) == 4 and len(aggregate["checkpoint_diagnostics"]) == 12
    for model in aggregate["final_models"]:
        key = (model["seed"], model["arm"], model["completed_update"])
        assert key[2] == 1260
        cohorts = independent[key]
        primary = cohorts["dev/full_sentence/normal"]
        assert model["cohorts"] == cohorts and model["primary_dev_sentence_normal"] == primary
        assert primary["count"] == 3 and primary["reference_characters"] == 74 and primary["reference_words"] == 23
        assert model["primary_static_minus_normal_CER_pp"] == 100 * (cohorts["dev/full_sentence/static"]["CER"] - primary["CER"])
        assert model["primary_shuffled_minus_normal_CER_pp"] == 100 * (cohorts["dev/full_sentence/shuffled"]["CER"] - primary["CER"])
        assert model["output_diagnostics"] == diagnostics(output_rows[key])
    for row in aggregate["checkpoint_diagnostics"]:
        key = (row["seed"], row["arm"], row["completed_update"])
        assert row["cohorts"] == independent[key]
        assert row["output_diagnostics"] == diagnostics(output_rows[key])
    assert [r["seed"] for r in aggregate["paired_primary"]] == [42,43]
    paired = []
    for seed in [42,43]:
        a = independent[(seed,"A_sentence",1260)]["dev/full_sentence/normal"]
        b = independent[(seed,"B_curriculum",1260)]["dev/full_sentence/normal"]
        paired.append({"seed":seed, "A_CER":a["CER"], "B_CER":b["CER"], "B_minus_A_CER_pp":100*(b["CER"]-a["CER"]),
                       "A_WER":a["WER"], "B_WER":b["WER"], "B_minus_A_WER_pp":100*(b["WER"]-a["WER"]),
                       "A_exact":a["exact_count"], "B_exact":b["exact_count"], "A_failures":a["failure_count"], "B_failures":b["failure_count"]})
    assert aggregate["paired_primary"] == paired
    mean = {"A_CER":sum(p["A_CER"] for p in paired)/2, "B_CER":sum(p["B_CER"] for p in paired)/2,
            "B_minus_A_CER_pp":sum(p["B_minus_A_CER_pp"] for p in paired)/2,
            "B_lower_CER_both_seeds":all(p["B_CER"] < p["A_CER"] for p in paired)}
    assert aggregate["mean_primary"] == mean
    return {"schema":"offline-stage5-independent-aggregate-audit-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
            "status":"passed", "aggregate":binding(aggregate_path), "independent_evaluation_audit":binding(audit_path),
            "final_models":4, "checkpoint_diagnostics":12, "output_statistic_cohorts":144,
            "paired_primary":paired, "mean_primary":mean, "producer_imports":0, "new_forward_calls":0,
            "claim_scope":"Internal dev3 final1260, no selection of best checkpoint and no generalization certification"}


if __name__ == "__main__":
    result = inspect()
    output = HERE / "aggregate_audit.json"
    assert not output.exists()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status":result["status"],"mean_primary":result["mean_primary"]}))
