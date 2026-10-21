"""Re-score saved stage17 B normal val20 text only; no model or media imports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

from evaluation_metrics import counts as runner_counts, aggregate as runner_aggregate


BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(left, right):
    # Independent full-matrix implementation, not the runner's rolling-row helper.
    matrix = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) + 1):
        matrix[i][0] = i
    for j in range(len(right) + 1):
        matrix[0][j] = j
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            matrix[i][j] = min(matrix[i-1][j] + 1, matrix[i][j-1] + 1,
                               matrix[i-1][j-1] + (left[i-1] != right[j-1]))
    return matrix[-1][-1]


def independent_counts(reference: str, prediction: str) -> dict:
    ref = unicodedata.normalize("NFC", reference).strip()
    pred = unicodedata.normalize("NFC", prediction).strip()
    ref_norm, pred_norm = " ".join(ref.split()), " ".join(pred.split())
    return {
        "character_edits": distance(ref, pred), "reference_characters": len(ref),
        "word_edits": distance(ref.split(), pred.split()), "reference_words": len(ref.split()),
        "exact": ref == pred, "normalized_character_edits": distance(ref_norm, pred_norm),
        "normalized_reference_characters": len(ref_norm), "normalized_exact": ref_norm == pred_norm,
    }


def main():
    output = BASE / "results" / "step03_legacy_val20.json"
    if output.exists():
        raise FileExistsError("Preserve previous audit output")
    source_path = BASE / "step03_source_plan.json"
    plan_path = BASE / "step03_eval_plan.json"
    source = json.loads(source_path.read_text(encoding="utf8"))
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    if sha256(source_path) != plan["source_plan_sha256"]:
        raise ValueError("Frozen source plan differs")
    original_manifest = Path(source["manifest"]["path"])
    if sha256(original_manifest) != source["manifest"]["sha256"]:
        raise ValueError("Original manifest differs")
    expected = {r["id"]: r for r in source["records"]}
    original_val = {r["id"]: r for r in json.loads(original_manifest.read_text(encoding="utf8"))["records"]
                    if r["split"] == "val"}
    if set(expected) != set(original_val) or list(expected) != plan["source_ids"] or len(expected) != 20:
        raise ValueError("Expected all and only frozen val20, in frozen order")
    for sid in expected:
        if expected[sid]["split"] != "val" or expected[sid]["text"] != original_val[sid]["text"]:
            raise ValueError("Source reference mismatch")
    folder = PROJECT / "results" / "step17-expansion" / "run01"
    quality_path = folder / "quality_checks.json"
    quality = json.loads(quality_path.read_text(encoding="utf8"))
    run_path = folder / "B_expand512" / "step_6400"
    summary_path, rows_path = run_path / "summary.json", run_path / "rows.jsonl"
    for path in (summary_path, rows_path):
        if sha256(path) != quality["artifacts"][path.relative_to(PROJECT).as_posix()]:
            raise ValueError(f"Historical quality hash differs: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    if sha256(rows_path) != summary["rows_sha256"]:
        raise ValueError("Historical rows no longer match summary")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf8").splitlines()]
    if len(rows) != summary["rows"] or summary["branch"] != "B_expand512" or summary["step"] != 6400:
        raise ValueError("Wrong frozen historical checkpoint output")
    rows = [r for r in rows if r["condition"] == "normal" and r["split"] == "val"]
    if len(rows) != 20 or {r["id"] for r in rows} != set(expected):
        raise ValueError("Historical normal/val rows differ from val20")
    lookup = {r["id"]: r for r in rows}
    records = []
    for sid in plan["source_ids"]:
        row, ref = lookup[sid], expected[sid]["text"]
        if row["reference"] != ref or row["cohort"] != "val20":
            raise ValueError("Historical reference/cohort mismatch")
        item = {"id": sid, "speaker_id": row["speaker_id"], "reference": ref,
                "legacy_frames": row["frames"], "outputs": {}}
        for method in ("greedy", "beam"):
            original = row[method]
            if original["id"] != sid or original["reference"] != ref:
                raise ValueError("Historical method reference mismatch")
            metrics = independent_counts(ref, original["prediction"])
            if metrics != runner_counts(ref, original["prediction"]):
                raise ValueError("Independent and runner metrics disagree")
            item["outputs"][method] = {
                "status": "ok", "prediction": original["prediction"], "metrics": metrics,
                "replacement_character_count": original["prediction"].count("\ufffd"),
                "legacy_reported_jamo_cer": original["cer"],
            }
        records.append(item)
    aggregate = {}
    for method in ("greedy", "beam"):
        entries = [dict(id=r["id"], **r["outputs"][method]) for r in records]
        aggregate[method] = runner_aggregate(entries)
        aggregate[method]["per_speaker"] = {
            speaker: runner_aggregate([dict(id=r["id"], **r["outputs"][method])
                                       for r in records if r["speaker_id"] == speaker])
            for speaker in sorted({r["speaker_id"] for r in records})}
        for key in ("character_edits", "reference_characters", "word_edits", "reference_words",
                    "normalized_character_edits", "normalized_reference_characters"):
            assert aggregate[method][key] == sum(x["metrics"][key] for x in entries)
    result = {
        "schema": "olkavs-step03-historical-legacy-text-audit-v1", "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Same existing development20, fixed stage17 B512 step6400 saved normal predictions, re-scored with stage03 text definitions",
        "selection": "all and only condition=normal, split=val, cohort=val20; greedy and existing beam10; no output-based checkpoint selection",
        "sources": {"evaluation_plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
                    "source_plan": {"path": str(source_path), "sha256": sha256(source_path)},
                    "original_manifest": {"path": str(original_manifest), "sha256": sha256(original_manifest)},
                    "historical_quality_checks": {"path": str(quality_path), "sha256": sha256(quality_path)},
                    "historical_summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
                    "historical_rows": {"path": str(rows_path), "sha256": sha256(rows_path)},
                    "audit_script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
                    "metrics_code": {"path": str(BASE / "evaluation_metrics.py"), "sha256": sha256(BASE / "evaluation_metrics.py")}},
        "historical_checkpoint_sha256_from_verified_summary": summary["checkpoint_sha256"],
        "new_model_inference": False, "video_or_cache_payload_read": False,
        "checkpoint_or_posterior_payload_read": False, "test50_used": False, "training_run": False,
        "metrics": plan["metrics"], "independent_full_matrix_metric_check": True,
        "records": records, "aggregate": aggregate,
        "limitations": [
            "Previously repeatedly used development20; not untouched test or webcam evaluation",
            "Official checkpoint pretraining overlap unknown",
            "Legacy path includes its own 25fps RGB margin20 ImageNet normalized 69-token inputs; this is whole-system comparison, not architecture-only comparison",
            "Historical jamo CER is a different metric from the current composed-character CER and is kept only as provenance",
            "Saved invalid/replacement characters are preserved and counted, never repaired",
            "No timing re-measured; historical full-run time cannot be compared to current single-method latency",
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"output": str(output), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
