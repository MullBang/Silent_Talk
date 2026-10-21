"""Map audited filenames to frozen dataset roles using local metadata only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evidence(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    paths = {
        "v4_manifest": ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json",
        "protected_reference": ROOT / "results/step18-intake/run01/protected_reference.json",
        "stage17_inventory": ROOT / "experiments/training_alignment_step06/training_inventory.json",
        "stage18_selection": ROOT / "results/step18-intake/run01/selection.json",
        "stage7_plan": ROOT / "experiments/feature_transfer_step07/data_plan.json",
        "archive_member_index": OUT / "member_index.json",
    }
    d = {key: read(path) for key, path in paths.items()}
    hashes = {key: evidence(path) for key, path in paths.items()}
    if hashes["v4_manifest"]["sha256"] != d["protected_reference"]["parents"]["manifest"]["sha256"]:
        raise ValueError("Protected v4 manifest changed")
    original_roles = {}
    original_counts = Counter()
    for row in d["v4_manifest"]["records"]:
        speaker, split = row["speaker_id"], row["split"]
        if original_roles.setdefault(speaker, split) != split:
            raise ValueError("Original speaker appears across splits")
        original_counts[split] += 1
    for split, contract in d["protected_reference"]["existing_manifest_split_contract"].items():
        if sorted(s for s, role in original_roles.items() if role == split) != contract["speaker_ids"]:
            raise ValueError("Protected speaker contract mismatch")
    pilot_roles = d["stage18_selection"]["speaker_mapping"]
    if set(pilot_roles) & set(original_roles):
        raise ValueError("New pilot speakers overlap original roles")
    stage17 = {r["source_stem"]: r for r in d["stage17_inventory"]["sources"]}
    selected18 = defaultdict(list)
    for row in d["stage18_selection"]["records"]:
        if pilot_roles[row["speaker_id"]] != row["split"]:
            raise ValueError("Pilot clip role differs from frozen speaker role")
        selected18[row["id"].rsplit("_", 1)[0]].append(row)
    stage7 = defaultdict(list)
    for row in d["stage7_plan"]["records"]:
        if row["source_stem"] not in stage17:
            raise ValueError("Stage7 source outside audited stage17 cohort")
        stage7[row["source_stem"]].append(row)
    protected_recordings = defaultdict(list)
    for cohort, spec in d["protected_reference"]["reference_cohorts"].items():
        for recid in spec["recording_ids"]:
            protected_recordings[recid].append(cohort)
    rows = []
    for member in d["archive_member_index"]["members"]:
        stem = PurePosixPath(member["member_name"]).stem
        if not member["is_file"] or not member["member_name"].lower().endswith(".mp4") or stem.split("_")[-2] != "A":
            continue
        speaker = re.search(r"_(C\d+)_", stem).group(1)
        recid = stem.replace("_A_", "_")
        original = original_roles.get(speaker)
        pilot = pilot_roles.get(speaker)
        if original:
            speaker_role = "existing_" + original
            if original == "train":
                if stem in stage7:
                    splits = {r["split"] for r in stage7[stem]}
                    if len(splits) != 1:
                        raise ValueError("Stage7 recording spans roles")
                    role = "stage7_selected_" + next(iter(splits))
                elif "train60" in protected_recordings[recid]:
                    role = "protected_familiar_speaker_development_recording"
                else:
                    role = "existing_train_speaker_unselected_recording"
            else:
                role = "protected_original_" + original + "_speaker_recording"
        elif pilot:
            speaker_role = "new_pilot_train" if pilot == "train" else "new_pilot_development"
            role = speaker_role + ("_selected_recording" if stem in selected18 else "_unselected_recording")
        else:
            raise ValueError("Unassigned speaker: " + speaker)
        rows.append({
            "source_name": PurePosixPath(member["member_name"]).name,
            "source_stem": stem, "speaker_id": speaker, "recording_id": recid,
            "archive_path": member["archive_path"], "archive_member": member["member_name"],
            "speaker_role": speaker_role, "readiness_role": role,
            "original_v4_split": original, "stage18_pilot_split": pilot,
            "protected_reference_cohorts": protected_recordings[recid],
            "stage17_recording_selected": stem in stage17,
            "stage17_branch_clip_ids": stage17.get(stem, {}).get("branch_sample_ids", {}),
            "stage7_planned_selected_clip_ids": [r["id"] for r in stage7.get(stem, [])],
            "stage7_planned_split": stage7[stem][0]["split"] if stage7.get(stem) else None,
            "stage18_pilot_selected_clip_ids": [r["id"] for r in selected18.get(stem, [])],
            "all_sentences_in_recording_approved_for_training": False,
            "integrity_audit_changes_split_or_authorizes_training": False,
        })
    seen = {r["source_stem"] for r in rows}
    if len(seen) != len(rows) or len(rows) != 300:
        raise ValueError("Expected exactly 300 unique frontal source recordings")
    if set(stage17) - seen or set(selected18) - seen:
        raise ValueError("A previously selected source is absent from current archive index")
    report = {
        "schema": "integrity-readiness-role-map-v1", "date": "2026-10-02",
        "source_payloads_read": False, "model_inference_or_training": False,
        "evidence": hashes,
        "original_v4_contract": {split: {"speaker_ids": sorted(s for s, r in original_roles.items() if r == split),
                                        "clips": original_counts[split]} for split in ("train", "val", "test")},
        "summary": {
            "frontal_recordings": len(rows), "speakers": len({r["speaker_id"] for r in rows}),
            "recordings_by_speaker_role": dict(Counter(r["speaker_role"] for r in rows)),
            "recordings_by_readiness_role": dict(Counter(r["readiness_role"] for r in rows)),
            "stage17_selected_recordings": len(stage17),
            "stage18_selected_recordings": len(selected18),
            "stage18_selected_clip_counts": dict(Counter(r["split"] for r in d["stage18_selection"]["records"])),
            "original_test_speakers_present": sorted({r["speaker_id"] for r in rows if r["original_v4_split"] == "test"}),
        },
        "guards": [
            "The 300 recordings are integrity-audit coverage, not a 300-recording training set.",
            "Original validation speakers C025/C294 and every recording/angle of those speakers remain development-only.",
            "New pilot development speakers C040/C059/C769 and every recording/angle remain development-only.",
            "Original frozen final-test speakers retain their role; none of their source videos is in these five Training TARs.",
            "The 37 train60 protected recordings remain familiar-speaker development. Stage17 selected 59 different recordings; the stage7 plan further splits those into 51 train and 8 dev recordings.",
            "Only frozen selected clip IDs are named here. Passing source-file integrity does not authorize all sentences in that recording; transcript exclusions, timing, ROI and cache checks still apply.",
            "Structural/label/decode integrity checks on held-out development files are file-quality inspection, not model-performance evaluation or permission to train on them.",
            "No existing split, selection, training manifest, source path or source metadata is modified by this mapping.",
        ],
        "records": sorted(rows, key=lambda r: r["source_name"]),
    }
    with (OUT / "readiness_role_map.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
