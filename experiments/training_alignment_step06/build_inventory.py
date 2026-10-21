"""Join frozen stage17 training IDs to the local stage18 file inventory.

This module reads local JSON metadata only: no G: stat, payload, cache, weight,
prediction, or held-out/test payload is opened. Cloud availability and media
integrity are deliberately left for the subsequent bounded waveform audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "training_inventory.json"
PLAN = ROOT / "results/step17-expansion/run01/plan.json"
INVENTORY = ROOT / "results/step18-intake/run01/inventory.json"
MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
BRANCHES = ("A_keep256", "B_expand512")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recording_stem(row: dict) -> str:
    """Use the archived member, rather than a tar container's stem, if present."""
    stem = row["id"].rsplit("_", 1)[0]
    source = row["source_video"]
    source_name = source.get("archive_member", source["path"])
    require(PureWindowsPath(source_name).stem == stem, f"Source identity mismatch: {row['id']}")
    require(PureWindowsPath(row["label_path"]).stem == stem, f"Label identity mismatch: {row['id']}")
    require(row["angle"] == "A" and "_A_" in stem, "Frontal inputs only")
    return stem


def single_match(entries: list[dict], *, stem: str, kind: str) -> dict:
    if len(entries) != 1:
        return {"status": "missing" if not entries else "ambiguous", "candidate_count": len(entries),
                "candidates": entries}
    item = entries[0]
    require(PureWindowsPath(item["path"]).stem == stem, f"Inventory {kind} filename mismatch: {stem}")
    require(type(item["size_bytes"]) is int and item["size_bytes"] > 0, f"Invalid {kind} size: {stem}")
    return {"status": "unique_metadata_match", "candidate_count": 1, **item}


def build() -> dict:
    plan, inventory, manifest = read(PLAN), read(INVENTORY), read(MANIFEST)
    require(plan["manifest_sha256"] == sha(MANIFEST), "Old manifest hash changed")
    require(inventory["payloads_read"] is False, "Expected metadata-only source inventory")
    branch_ids = {key: plan["branches"][key]["train_ids"] for key in BRANCHES}
    for key, expected in zip(BRANCHES, (256, 512)):
        require(len(branch_ids[key]) == len(set(branch_ids[key])) == expected, f"Unexpected {key} IDs")
        schedule = plan["branches"][key]["schedule"]
        require(set(schedule["exposures"]) == set(branch_ids[key]), f"Schedule IDs changed: {key}")
        require(all(value > 0 for value in schedule["exposures"].values()), f"Unscheduled selected ID: {key}")
    selected = set().union(*(set(ids) for ids in branch_ids.values()))
    require(set(branch_ids[BRANCHES[0]]) <= set(branch_ids[BRANCHES[1]]), "A is not nested in B")
    records = {row["id"]: row for row in plan["records"] if row["id"] in selected}
    original = {row["id"]: row for row in manifest["records"] if row["id"] in selected}
    require(set(records) == selected == set(original), "Selected metadata missing")
    require(records == original, "Stage17 training metadata differs from the frozen old manifest")
    heldout = set(plan["cohorts"]["train60"] + plan["cohorts"]["val20"])
    require(not selected & heldout, "Held-out ID selected")
    excluded_recordings = set(plan["selection"]["excluded_recording_ids"])
    grouped = defaultdict(list)
    for sample_id in sorted(selected):
        row = records[sample_id]
        require(row["split"] == "train" and row["source_partition"] == "Training", "Training only")
        require(row["recording_id"] not in excluded_recordings, "Held-out recording selected")
        grouped[recording_stem(row)].append(row)
    require(len(grouped) == 59, "Unexpected training recording count")
    part = inventory["partitions"]["Training"]
    audio_by_stem = defaultdict(list)
    for item in part["audio"]:
        audio_by_stem[PureWindowsPath(item["path"]).stem].append(item)
    sources = []
    for stem, rows in sorted(grouped.items()):
        speaker_ids = {r["speaker_id"] for r in rows}
        recording_ids = {r["recording_id"] for r in rows}
        label_hashes = {r["label_sha256"] for r in rows}
        video_sizes = {r["source_video"]["size_bytes"] for r in rows}
        require(len(speaker_ids) == len(recording_ids) == len(label_hashes) == len(video_sizes) == 1,
                f"Inconsistent within-recording provenance: {stem}")
        matches = {}
        for kind, table in (("video", "videos"), ("label", "labels")):
            candidates = [part[table][stem]] if stem in part[table] else []
            matches[kind] = single_match(candidates, stem=stem, kind=kind)
        matches["audio"] = single_match(audio_by_stem[stem], stem=stem, kind="audio")
        issues = [f"{kind}_{match['status']}" for kind, match in matches.items()
                  if match["status"] != "unique_metadata_match"]
        for kind in ("video", "label"):
            match = matches[kind]
            if match["status"] == "unique_metadata_match":
                require(match["speaker_id"] in speaker_ids and match["recording_id"] in recording_ids
                        and match["angle"] == "A", f"Inventory identity mismatch: {stem}/{kind}")
        if matches["video"]["status"] == "unique_metadata_match":
            if matches["video"]["size_bytes"] != next(iter(video_sizes)):
                issues.append("video_size_differs_from_training_metadata")
        if all(matches[kind]["status"] == "unique_metadata_match" for kind in ("video", "label")):
            require(matches["video"]["package"][2:] == matches["label"]["package"][2:],
                    f"Video/label package mismatch: {stem}")
        sample_ids = {r["id"] for r in rows}
        per_branch = {key: [i for i in branch_ids[key] if i in sample_ids] for key in BRANCHES}
        sources.append({"source_stem": stem, "source_name": stem + ".mp4",
            "speaker_id": next(iter(speaker_ids)), "recording_id": next(iter(recording_ids)),
            "angle": "A", **matches, "expected_label_sha256": next(iter(label_hashes)),
            "expected_training_video_size_bytes": next(iter(video_sizes)),
            "status": "metadata_ready" if not issues else "metadata_issue", "issues": issues,
            "branch_sample_ids": per_branch, "branch_counts": {key: len(ids) for key, ids in per_branch.items()},
            "training_records": rows,
            "selected_label_start_min_seconds": min(r["start_time"] for r in rows),
            "selected_label_end_max_seconds": max(r["end_time"] for r in rows)})
    speakers = sorted({source["speaker_id"] for source in sources})
    require(len(speakers) == 8, "Unexpected training speaker count")
    screen_stems = [min(s["source_stem"] for s in sources if s["speaker_id"] == speaker) for speaker in speakers]
    for source in sources:
        source["phase"] = "screen8" if source["source_stem"] in screen_stems else "remaining51"
    branch_recordings = {key: sorted(s["source_stem"] for s in sources if s["branch_counts"][key]) for key in BRANCHES}
    a, b = (set(branch_ids[key]) for key in BRANCHES)
    ar, br = (set(branch_recordings[key]) for key in BRANCHES)
    distributions = {}
    for speaker in speakers:
        subset = [s for s in sources if s["speaker_id"] == speaker]
        distributions[speaker] = {"recordings": len(subset),
            "branch_selected_utterances": {key: sum(s["branch_counts"][key] for s in subset) for key in BRANCHES},
            "screen8_source_stem": next(s for s in screen_stems if s in {x["source_stem"] for x in subset})}
    totals = {kind: sum(s[kind].get("size_bytes", 0) for s in sources) for kind in ("video", "audio", "label")}
    return {"schema": "lipnet-training-alignment-inventory-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"local_metadata_only": True, "cloud_paths_statted": False, "source_payloads_read": False,
                  "cache_payloads_read": False, "model_loaded": False, "predictions_read": False,
                  "heldout_or_test_payloads_read": False, "training_performed": False},
        "evidence_files": [{"path": str(p), "sha256": sha(p)} for p in
            (PLAN, INVENTORY, MANIFEST, ROOT / "scripts/diagnose_training_expansion.py",
             ROOT / "scripts/expansion_selection.py", ROOT / "scripts/intake_new_sources.py", Path(__file__).resolve())],
        "status": "metadata_ready" if all(s["status"] == "metadata_ready" for s in sources) else "metadata_issues",
        "selection_policy": "All stage17 A_keep256/B_expand512 selected training IDs. One screen recording per speaker: lexical minimum source_stem, independent of audio, video, predictions and outcome. Remaining 51 recordings sorted by source_stem.",
        "source_stem_resolution": "ID without final sentence suffix must equal source_video.archive_member stem if present, otherwise source_video.path stem, and label_path stem.",
        "limits": ["Unique MP4/JSON matches rely on the frozen local inventory; its producer rejects duplicate stems. WAV matches are rechecked here.",
                   "Inventory sizes/timestamps are historical metadata, not a current cloud availability or integrity check.",
                   "Selected IDs and schedule exposures establish planned training coverage; no weight or live training state is loaded."],
        "summary": {"sources": len(sources), "unique_training_utterances": len(selected), "speakers": speakers,
                    "metadata_status_counts": dict(Counter(s["status"] for s in sources)),
                    "source_file_bytes_if_fully_materialized": totals,
                    "branch_utterances": {key: len(ids) for key, ids in branch_ids.items()},
                    "branch_recordings": {key: len(ids) for key, ids in branch_recordings.items()},
                    "utterance_intersection": len(a & b), "utterance_A_only": len(a - b), "utterance_B_only": len(b - a),
                    "recording_intersection": len(ar & br), "recording_A_only": len(ar - br), "recording_B_only": len(br - ar),
                    "speaker_distribution": distributions},
        "branch_sample_ids": branch_ids, "branch_source_stems": branch_recordings,
        "screen8_source_stems": screen_stems,
        "remaining51_source_stems": [s["source_stem"] for s in sources if s["phase"] == "remaining51"],
        "sources": sources}


if __name__ == "__main__":
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(OUT), "sha256": sha(OUT), "status": result["status"], "summary": result["summary"],
                      "screen8_source_stems": result["screen8_source_stems"]}, ensure_ascii=False, indent=2))
