"""Read-only output diagnostics from the four completed evaluation JSON files.

No model/runtime imports or new inference. Examples are the first five frozen
development rows, never ranked by accuracy. Text follows NFC plus outer strip;
internal spaces remain significant, matching the primary evaluation metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

CONDITIONS = ("normal", "static", "shuffled")
PAIRS = ((42, "A_ctc"), (42, "B_feature"), (43, "A_ctc"), (43, "B_feature"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(text: str) -> str:
    require(isinstance(text, str), "Text must be a string")
    return unicodedata.normalize("NFC", text).strip()


def binding(path: Path) -> dict:
    path = path.resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def validate_evaluation(value: dict) -> None:
    """Keep the complete same ordered 71-sentence cohort under every condition."""
    require(value.get("schema") == "lipnet-step05-final-development-evaluation-v1"
            and value.get("status") == "completed", "Wrong or incomplete evaluation")
    require((value.get("seed"), value.get("arm")) in PAIRS
            and value.get("completed_updates") == 3200, "Unexpected seed/arm/checkpoint budget")
    ids = value["ids"]
    require(value.get("record_count") == 71 and len(ids) == len(set(ids)) == 71,
            "Expected all 71 unique development IDs")
    require(set(value["conditions"]) == set(CONDITIONS), "Missing or extra temporal condition")
    normal = value["conditions"]["normal"]["records"]
    require(len(normal) == 71, "Normal condition must retain every development row")
    for condition in CONDITIONS:
        records = value["conditions"][condition]["records"]
        require([row["id"] for row in records] == ids, "Condition changed fixed development order")
        for reference, row in zip(normal, records):
            require(row["condition"] == condition and row["split"] == "dev", "Condition/split mismatch")
            require(canonical(reference["reference"]) == canonical(row["reference"]), "Reference changed between conditions")
            require(row["speaker_id"] == reference["speaker_id"] and row["source_stem"] == reference["source_stem"],
                    "Recording identity changed between conditions")
            require(row["status"] in ("ok", "failed"), "Unrecognized inference status")
            require(row["status"] == "ok" or row["prediction"] == "", "Failed inference must retain empty prediction")
            require(len(canonical(row["reference"])) > 0, "Empty reference")
            canonical(row["prediction"])


def condition_diagnostics(records: list[dict]) -> dict:
    predictions = [canonical(row["prediction"]) for row in records]
    references = [canonical(row["reference"]) for row in records]
    frequency = Counter(predictions)
    predicted_chars = sum(map(len, predictions))
    reference_chars = sum(map(len, references))
    count = len(records)
    return {"sentences": count, "failed_inputs": sum(row["status"] == "failed" for row in records),
            "empty_prediction_count": frequency.get("", 0), "unique_prediction_count": len(frequency),
            "unique_nonempty_prediction_count": sum(bool(text) for text in frequency),
            "most_frequent_predictions": [{"prediction": text, "count": occurrences}
                for text, occurrences in sorted(frequency.items(), key=lambda item: (-item[1], item[0]))[:5]],
            "total_reference_characters": reference_chars, "total_prediction_characters": predicted_chars,
            "mean_reference_characters": reference_chars / count,
            "mean_prediction_characters": predicted_chars / count,
            "prediction_to_reference_character_ratio": predicted_chars / reference_chars}


def analyze_evaluation(value: dict) -> dict:
    validate_evaluation(value)
    rows = {condition: value["conditions"][condition]["records"] for condition in CONDITIONS}
    normalized = {condition: [canonical(row["prediction"]) for row in records] for condition, records in rows.items()}
    equality = {}
    for first, second in (("normal", "static"), ("normal", "shuffled"), ("static", "shuffled")):
        equal_ids = [sample_id for sample_id, a, b in zip(value["ids"], normalized[first], normalized[second]) if a == b]
        equality[f"{first}_equals_{second}"] = {"sentences": len(equal_ids), "ids": equal_ids}
    all_equal = [sample_id for sample_id, a, b, c in zip(value["ids"], normalized["normal"], normalized["static"], normalized["shuffled"])
                 if a == b == c]
    equality["all_three_equal"] = {"sentences": len(all_equal), "ids": all_equal}
    examples = [{"id": row["id"], "speaker_id": row["speaker_id"], "reference": canonical(row["reference"]),
                 "predictions": {condition: normalized[condition][index] for condition in CONDITIONS},
                 "statuses": {condition: rows[condition][index]["status"] for condition in CONDITIONS}}
                for index, row in enumerate(rows["normal"][:5])]
    return {"seed": value["seed"], "arm": value["arm"], "completed_updates": value["completed_updates"],
            "checkpoint": value["checkpoint"], "plan": value["plan"], "ids": value["ids"],
            "conditions": {condition: condition_diagnostics(records) for condition, records in rows.items()},
            "prediction_equality": equality, "examples_first_five_fixed_development_rows": examples}


def build_report(evaluations: list[dict]) -> dict:
    require(len(evaluations) == 4, "Exactly four fixed seed/arm evaluations are required")
    for value in evaluations:
        validate_evaluation(value)
    lookup = {(value["seed"], value["arm"]): value for value in evaluations}
    require(set(lookup) == set(PAIRS), "Missing or duplicated seed/arm evaluation")
    first = lookup[PAIRS[0]]
    for value in evaluations:
        require(value["ids"] == first["ids"], "Development ordering differs across models")
        require([canonical(row["reference"]) for row in value["conditions"]["normal"]["records"]]
                == [canonical(row["reference"]) for row in first["conditions"]["normal"]["records"]],
                "References differ across models")
        require(value["full_data_plan"] == first["full_data_plan"], "Full data handoff differs across models")
    return {"schema": "lipnet-step05-saved-prediction-diagnostics-v1", "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "full_data_plan": first["full_data_plan"],
            "text_policy": "NFC plus outer strip; internal whitespace preserved",
            "cohort_policy": "All 71 fixed development sentences retained per model and condition; failures included",
            "example_policy": "First five sentences in unchanged development order, not selected by prediction or error",
            "frequency_policy": "Top five distinct outputs by descending frequency; Unicode lexical order resolves ties",
            "character_ratio_policy": "Total predicted characters divided by total reference characters, including internal spaces",
            "inference_executed": False, "training_executed": False,
            "models": [analyze_evaluation(lookup[pair]) for pair in PAIRS],
            "limits": ["Output collapse and temporal invariance are diagnostics, not standalone proof of their causes",
                       "Same-speaker development records are not a final unseen-speaker or webcam test"]}


def save_new(path: Path, value: dict) -> None:
    require(not path.exists(), f"Refusing to overwrite diagnostics: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix="." + path.name, suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    references = [binding(path) for path in args.evaluations]
    values = [json.loads(path.read_text(encoding="utf-8")) for path in args.evaluations]
    report = build_report(values)
    require(references == [binding(path) for path in args.evaluations], "Evaluation JSON changed during diagnostics")
    report["evaluation_files"] = references
    report["analysis_code"] = binding(Path(__file__))
    report["analysis_tests"] = binding(Path(__file__).with_name("test_analyze_predictions.py"))
    save_new(args.output, report)
    print(json.dumps({"output": str(args.output.resolve()), "models": len(report["models"]),
                      "new_inference": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
