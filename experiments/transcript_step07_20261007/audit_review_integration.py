"""Independent review-package join/metrics audit; production code is not imported."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "new_data_step06_20261007"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


AUDIT = BASE / "results/independent_stage07_audit.json"
if digest(AUDIT) != "0bfc60e91b8c7dce3820a4ea8f22e2aeeb344d8f93f511d929d466e2b97e7852":
    raise ValueError("Prior independent audit changed")
prior = json.loads(AUDIT.read_text(encoding="utf8"))
if digest(prior["code"]["path"]) != prior["code"]["sha256"]:
    raise ValueError("Frozen independent code changed")
spec = importlib.util.spec_from_file_location("frozen_stage07_independent", prior["code"]["path"])
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
require, same, read, bind, verify = (getattr(independent, key) for key in ("require", "same", "read", "bind", "verify"))


def compare(reference, prediction):
    value = independent.old.comparison(reference, prediction)
    metric = value["metrics"]
    return {"normalized_reference": value["normalized_reference"], "normalized_prediction": value["normalized_asr"],
            "exact": metric["exact"], "without_spaces_exact": metric["without_spaces_exact"],
            "edits": metric["character_edits"], "characters": metric["reference_characters"],
            "without_spaces_edits": metric["without_spaces_character_edits"],
            "without_spaces_characters": metric["without_spaces_reference_characters"]}


def aggregate(rows):
    totals = {key: sum(row[key] for row in rows) for key in ("exact", "without_spaces_exact", "edits", "characters", "without_spaces_edits", "without_spaces_characters")}
    totals["cer_percent"] = 100 * totals["edits"] / totals["characters"]
    totals["without_spaces_cer_percent"] = 100 * totals["without_spaces_edits"] / totals["without_spaces_characters"]
    return totals


def run():
    output = BASE / "results/review_integration_audit.json"
    require(not output.exists(), "Preserve existing integration audit")
    package_ref = bind(BASE / "results/review_package.json")
    package = read(package_ref["path"])
    require(package["schema"] == "lipnet-step07-review-package-v1", "Review package schema")
    refs = [package_ref, bind(AUDIT)] + list(package["inputs"].values()) + [package[key] for key in ("builder", "template", "data", "html")]
    for ref in refs:
        verify(ref)
    data = read(package["data"]["path"])
    require(data["schema"] == "lipnet-step07-review-data-v1", "Review data schema")
    same(data["bindings"], package["inputs"], "Package/data inputs")
    expected_paths = [OLD / "results/review_package.json"] + [BASE / name for name in
        ("results/label_targets.json", "results/context_asr_results.json", "user_alternative_choice.json", "label_plan.json", "context_plan.json")]
    expected_inputs = {str(path.relative_to(BASE.parent)).replace("\\", "/"): bind(path) for path in expected_paths}
    same(package["inputs"], expected_inputs, "Exact review input inventory")
    require(expected_inputs["new_data_step06_20261007/results/review_package.json"]["sha256"] ==
            "22287f7f8c208446b0172a0600fd96ee37df451aafac15d09f6ce0e79e2d3d29", "Frozen Step6 review package")
    for name in ("results/label_targets.json", "results/context_asr_results.json", "user_alternative_choice.json", "label_plan.json", "context_plan.json"):
        same(expected_inputs["transcript_step07_20261007/" + name], prior["inputs"][name], "Prior independent audit binding " + name)
    old_package = read(OLD / "results/review_package.json")
    for key in ("data", "media_manifest", "source_plan", "asr_results", "scoped_user_review"):
        verify(old_package[key])
    previous = read(old_package["data"]["path"])
    media = read(old_package["media_manifest"]["path"])
    source = read(old_package["source_plan"]["path"])
    asr = read(old_package["asr_results"]["path"])
    targets = read(BASE / "results/label_targets.json")
    context = read(BASE / "results/context_asr_results.json")
    ids = [row["id"] for row in source["records"]]
    require(len(ids) == len(set(ids)) == len(data["records"]) == 120, "120 record denominator")
    for rows in (data["records"], previous["records"], targets["records"], context["records"], media["clips"], asr["records"]):
        same([row["id"] for row in rows], ids, "Fixed original order")
    same(data["media_sha256"], old_package["media_manifest"]["sha256"], "Unchanged media hash")
    groups, calculations, partial = Counter(), {"baseline": [], "full_context": []}, []
    reused_media = set()
    inherited_mutations = {"audio", "context_audio", "mp4_audio", "preview_video", "face_context", "roi_samples", "user_note"}
    for row, old, target, ctx, clip, original, baseline in zip(data["records"], previous["records"], targets["records"], context["records"], media["clips"], source["records"], asr["records"]):
        for key in set(old) - inherited_mutations:
            same(row[key], old[key], "Unchanged prior review field " + key)
        same(row["reference"], original["source_text"], "Untouched source text")
        same(row["split"], original["split"], "Unchanged role")
        same(row["asr_text"], baseline["asr_text"], "Baseline transcription")
        same(row["canonical_target"], target["canonical_target"], "Canonical reference")
        same(row["canonical_target_sha256"], independent.raw_sha(target["canonical_target"]), "Canonical raw SHA")
        same(row["target_status"], target["target_status"], "Target status")
        same(row["context_text"], ctx["full_context_text"], "Full context text")
        same(row["selected_text_diagnostic_only"], ctx["selected_text"], "Diagnostic-only selected text")
        same(row["words"], ctx["words"], "Unchanged word evidence")
        for name, prediction in (("baseline", baseline["asr_text"]), ("full_context", ctx["full_context_text"])):
            expected = compare(target["canonical_target"], prediction)
            same(row["canonical_comparisons"][name], expected, "Independent canonical comparison " + name)
            calculations[name].append(expected)
        matches = [value["without_spaces_exact"] for value in row["canonical_comparisons"].values()]
        group = "both_supported" if all(matches) else "one_supported" if any(matches) else "both_differ"
        same(row["comparison_group"], group, "Display group")
        groups[group] += 1
        require(row["comparison_scope"] == "same_model_diagnostic_not_independent_verification", "Comparison scope")
        require(all(row[key] is False for key in ("automatic_approval", "full_transcript_human_approved", "human_boundary_approved", "training_ready", "auto_supported")), "Unwarranted review approval")
        if target["selection_evidence"]:
            same(row["alternative_choice_evidence"], target["selection_evidence"]["binding"], "Partial user approval binding")
            require("대안 선택만 확인" in row["user_note"] and "전체 문장·경계의 승인 기록은 아닙니다" in row["user_note"], "Partial choice note scope")
            partial.append(row["id"])
        else:
            require("alternative_choice_evidence" not in row, "Spurious alternative evidence")
            same(row["user_note"], old["user_note"], "Prior scoped user note preserved")
        for key in ("audio", "context_audio", "mp4_audio", "preview_video", "face_context"):
            if row.get(key):
                linked = (BASE / "review" / row[key]).resolve()
                original_link = (OLD / "review" / old[key]).resolve()
                media_ref = clip[key]["image"] if key == "face_context" else clip[key]
                require(linked == original_link == Path(media_ref["path"]).resolve(), "Changed media target " + key)
                verify(media_ref); reused_media.add(str(linked))
        require(len(row["roi_samples"]) == len(clip["roi_samples"]) == 3, "ROI count")
        for item, previous_roi, roi in zip(row["roi_samples"], old["roi_samples"], clip["roi_samples"]):
            same({k: v for k, v in item.items() if k != "url"}, {k: v for k, v in previous_roi.items() if k != "url"}, "ROI metadata")
            require((BASE / "review" / item["url"]).resolve() == Path(roi["image"]["path"]).resolve(), "ROI target")
            verify(roi["image"]); reused_media.add(roi["image"]["path"])
    same(dict(Counter(row["split"] for row in data["records"])), {"train": 96, "val": 24}, "Role totals")
    same(dict(groups), {"both_supported": 85, "both_differ": 29, "one_supported": 6}, "Expected display groups")
    same(data["group_counts"], dict(groups), "Data group summary")
    same(package["group_counts"], dict(groups), "Package group summary")
    metrics = {key: aggregate(rows) for key, rows in calculations.items()}
    same(data["canonical_comparison_metrics"], metrics, "Data metrics")
    same(package["canonical_comparison_metrics"], metrics, "Package metrics")
    diagnostic = {"first_word_starts_at_zero": sum(bool(row["words"]) and row["words"][0]["start"] == 0 for row in context["records"]),
                  "first_word_excluded_by_midpoint": sum(bool(row["words"]) and not row["words"][0]["selected_by_midpoint"] for row in context["records"]),
                  "selected_text_usable_as_labels": False, "word_time_cues_prove_clipping": False}
    require(diagnostic["first_word_starts_at_zero"] == 120 and diagnostic["first_word_excluded_by_midpoint"] == 86, "Timing diagnosis counts")
    same(data["word_time_diagnostic"], diagnostic, "Data timing diagnosis")
    same(package["word_time_diagnostic"], diagnostic, "Package timing diagnosis")
    same(partial, ["lip_J_1_F_05_C057_A_005_005"], "Individual alternative scope")
    require(data["training_ready"] is False and data["automatic_approvals"] == data["full_transcript_human_approvals"] == 0
            and package["training_ready"] is False and package["manual_approvals_added"] == 0, "Package approval scope")
    html = Path(package["html"]["path"]).read_text(encoding="utf8")
    embedded = re.search(r'<script id="review-data" type="application/json">(.*?)</script>', html, re.S)
    require(embedded is not None, "Missing embedded review data")
    same(json.loads(embedded.group(1)), data, "HTML payload equals audited JSON")
    template = Path(package["template"]["path"]).read_text(encoding="utf8")
    require(template.count("__REVIEW_DATA__") == 1, "Template data marker")
    require(html == template.replace("__REVIEW_DATA__", embedded.group(1)), "HTML/template correspondence")
    same(data["review_order_rule"], "post_diagnostic_display_only: both_differ, one_supported, both_supported; original order within each", "Display-only priority scope")
    for ref in refs:
        verify(ref)
    result = {"schema": "lipnet-step07-independent-review-integration-audit-v1", "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "code": bind(__file__),
        "upstream_independent_audit": bind(AUDIT), "review_package": package_ref,
        "review_data": package["data"], "review_html": package["html"], "builder": package["builder"],
        "template": package["template"], "records_checked": 120, "split_counts": {"train": 96, "val": 24},
        "group_counts": dict(groups), "canonical_comparison_metrics": metrics, "word_time_diagnostic": diagnostic,
        "raw_source_preserved_count": 120, "scoped_alternative_choice_ids": partial,
        "reused_media_files_checked": len(reused_media), "html_embedded_data_identical": True,
        "full_transcript_human_approvals_added": 0, "automatic_approvals": 0, "training_ready": False,
        "models_executed": False, "production_imports": False,
        "limits": ["Display groups compare whitespace-removed outputs from the same recognizer and are not independent transcript verification",
                   "Canonical-reference metrics are separate from preserved raw-reference Stage6/Stage7 metrics",
                   "The one user choice concerns only the ambiguous word and does not approve the full utterance or its boundaries",
                   "Word timestamps and midpoint exclusion cannot establish real audio clipping",
                   "HTML data bindings were checked statically; root handles browser appearance and interaction checks"]}
    with output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": result["status"], "groups": dict(groups), "metrics": metrics, "output": bind(output)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
