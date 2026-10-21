"""Independent Step9 selection arithmetic; no producer or tokenizer imports."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA_SHA = "251d28ee44ea934ec2db7925821b0821dd4487f9e2f64c219f86ee87d5529585"
OLD_SOURCE_SHA = "d3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9"

def same(actual, expected, note):
    if actual != expected:
        raise ValueError(f"{note}: {actual!r} != {expected!r}")

def near(actual, expected, tolerance, note):
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{note}: {actual!r} != {expected!r}")

def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Bound artifact outside project")
    return path

def sha(path):
    h = hashlib.sha256()
    with local(path).open("rb") as stream:
        while block := stream.read(2**20):
            h.update(block)
    return h.hexdigest()

def binding(path):
    return {"path": str(local(path)), "sha256": sha(path)}

def read(path, expected=None):
    if expected is not None:
        same(sha(path), expected, "Bound artifact SHA")
    return json.loads(local(path).read_text(encoding="utf-8"))

def bound(ref):
    return read(ref["path"], ref["sha256"])

def tokens(text):
    """Independent modern-Hangul arithmetic with the fixed 69-class IDs."""
    chars = []
    for c in unicodedata.normalize("NFC", text):
        if 0xAC00 <= ord(c) <= 0xD7A3:
            chars.append(c)
        elif c.isspace() or unicodedata.category(c).startswith("P"):
            chars.append(" ")
        else:
            raise ValueError("Unsupported label character")
    normalized = " ".join("".join(chars).split())
    values = []
    for c in normalized:
        if c == " ":
            values.append(68)
        else:
            n = ord(c) - 0xAC00
            values.extend([n // 588 + 1, n % 588 // 28 + 20])
            if n % 28:
                values.append(n % 28 + 40)
    return normalized, values

def ctc_min(values):
    return len(values) + sum(a == b for a, b in zip(values, values[1:]))

def audit(plan_path=BASE / "data_plan.json"):
    plan = read(plan_path, DATA_SHA)
    same(plan["schema"], "lipnet-step09-provisional-data-plan-v1", "Plan schema")
    source = bound(plan["inputs"]["source_plan"])
    media = bound(plan["inputs"]["media_manifest"])
    targets = bound(plan["inputs"]["label_targets"])
    review = bound(plan["inputs"]["review_data"])
    bound(plan["inputs"]["step08_summary"])
    same(sha(plan["generator"]["path"]), plan["generator"]["sha256"], "Planner SHA")
    old = bound(plan["original512_plan"])
    old_source_path = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"
    old_source = read(old_source_path, OLD_SOURCE_SHA)
    source_by_id = {r["id"]: r for r in source["records"]}
    review_by_id = {r["id"]: r for r in review["records"]}
    target_by_id = {r["id"]: r for r in targets["records"]}
    media_by_id = {r["id"]: r for r in media["clips"]}
    expected_ids = [r["id"] for r in source["records"]]
    rows = plan["records"]
    same([r["id"] for r in rows], expected_ids, "All120 original order")
    same(plan["ids"], expected_ids, "Plan ID list")
    same(len(set(expected_ids)), 120, "Unique120")
    same(set(review_by_id), set(expected_ids), "Review cohort")
    same(set(target_by_id), set(expected_ids), "Target cohort")
    same(set(media_by_id), set(expected_ids), "Media cohort")
    same(plan["sources"], source["sources"], "Unchanged source plan")
    candidates, held, development = [], [], []
    for r in rows:
        s, v, t, m = (d[r["id"]] for d in (source_by_id, review_by_id, target_by_id, media_by_id))
        for k in ("id", "split", "speaker_id", "sentence_id", "source_text", "label_sha256", "label_path"):
            same(r[k], s[k], "Unchanged " + k)
        same(r["source_stem"], r["id"].rsplit("_", 1)[0], "Source ID")
        same(r["text"], t["canonical_target"], "Versioned target")
        normalized, values = tokens(r["text"])
        same(normalized, r["text"], "Normalized candidate")
        same(values, r["token_ids"], "Independent Hangul token IDs")
        same(values, t["canonical_tokens"], "Step7 unchanged tokens")
        same(ctc_min(values), r["ctc_min_frames"], "CTC minimum")
        if r["frames"] < ctc_min(values):
            raise ValueError("Impossible CTC length")
        same(r["original_start_time"], s["start_time"], "Original start")
        same(r["original_end_time"], s["end_time"], "Original end")
        near(r["start_time"], s["start_time"] + r["offset_seconds"], 1e-12, "Corrected start")
        near(r["end_time"], s["end_time"] + r["offset_seconds"], 1e-12, "Corrected end")
        same(r["start_time"], m["timing"]["corrected_start_time"], "Step6 video start")
        same(r["end_time"], m["timing"]["corrected_end_time"], "Step6 video end")
        same(r["frames"], round((r["end_time"] - r["start_time"]) * 25), "Student25fps count")
        indices = [round((r["start_time"] + k / 25) * r["video_fps"]) for k in range(r["frames"])]
        same(indices, r["student_native_frame_indices"], "Native indices")
        same(indices, m["timing"]["native_frame_indices"], "Unchanged Step6 indices")
        same(r["comparison_group"], v["comparison_group"], "Saved diagnostic group")
        candidate = r["split"] == "train" and v["comparison_group"] == "both_supported"
        same(r["teacher_candidate"], candidate, "Candidate role from independent predicate")
        role = "provisional_train_candidate" if candidate else "train_hold" if r["split"] == "train" else "development_all_unfiltered"
        same(r["preparation_role"], role, "Preparation role")
        for flag in ("full_transcript_human_approved", "human_boundary_approved", "training_ready"):
            same(r[flag], False, "No implicit approval")
        (candidates if candidate else held if r["split"] == "train" else development).append(r)
    same([len(candidates), len(held), len(development)], [69, 27, 24], "Roles69/27/24")
    for key, group in (("teacher_candidate_ids", candidates), ("held_train_ids", held), ("development_ids", development)):
        same(plan[key], [r["id"] for r in group], "Declared role IDs")
    for key in ("training_ready", "student_training_executed", "service_weights_changed"):
        same(plan[key], False, "Plan scope")
    old_overlap = {}
    for key in ("id", "source_stem", "speaker_id"):
        old_overlap[key] = sorted({r[key] for r in rows} & {r[key] for r in old["records"]})
        same(old_overlap[key], [], "New/old " + key)
    byte_overlap = {}
    for role in ("video", "audio", "label"):
        byte_overlap[role] = sorted({s["members"][role]["sha256"] for s in source["sources"]} & {s["members"][role]["sha256"] for s in old_source["sources"]})
        same(byte_overlap[role], [], "Source byte SHA overlap")
    text_pairs = [{"train_id": a["id"], "development_id": b["id"], "without_spaces_text": a["text"].replace(" ", "")} for a in candidates for b in development if a["text"].replace(" ", "") == b["text"].replace(" ", "")]
    same([(p["train_id"], p["development_id"]) for p in text_pairs], [("lip_J_1_F_02_C075_A_003_014", "lip_J_1_F_05_C059_A_002_036")], "Known one new train/dev text pair")
    same({r["text"].replace(" ", "") for r in rows} & {r["text"].replace(" ", "") for r in old["records"]}, set(), "New/old text overlap")
    same({r["speaker_id"] for r in candidates} & {r["speaker_id"] for r in development}, set(), "New speaker separation")
    def counts(group):
        return {"records": len(group), "student_frames": sum(r["frames"] for r in group), "speakers": dict(sorted(Counter(r["speaker_id"] for r in group).items())), "source_recordings": len({r["source_stem"] for r in group}), "packages": dict(Counter(r["package"] for r in group))}
    return {"schema": "lipnet-step09-independent-selection-audit-v1", "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "code": binding(Path(__file__)), "data_plan": binding(plan_path), "old_source_plan": binding(old_source_path), "inputs": plan["inputs"], "ids": expected_ids, "candidate_train": counts(candidates), "held_train": counts(held), "development_all": counts(development), "all120": counts(rows), "old512_identity_overlap": old_overlap, "old512_source_byte_sha_overlap": byte_overlap, "old512_text_overlap": 0, "candidate_vs_dev_text_overlap": text_pairs, "original_dev71_shares_train_speakers": True, "new_development_is_untouched_test": False, "new_human_approvals": 0, "training_ready": False, "limitations": ["Same-ASR agreement is provisional selection, not independent transcription truth.", "New24 keeps ASR-disagreeing samples and one text shared with new train.", "Old71 is recording-held-out with the same eight training speakers and prior project exposure.", "This selection audit reads bound local artifacts, not original D media."]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "selection_audit.json")
    args = parser.parse_args()
    result = audit()
    with local(args.output).open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2); out.write("\n")
    print(json.dumps({"status": result["status"], "candidate_train": result["candidate_train"]["records"], "held_train": result["held_train"]["records"], "development": result["development_all"]["records"]}))
