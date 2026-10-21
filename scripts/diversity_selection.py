"""Stage16 metadata-only expansion with the stage15 held-out recordings fixed.

This module reads no files or pixels and never considers predictions. Its
production cohort labels document the fixed 64/192/256 and 60/20 experiment;
small synthetic fixtures may use other counts to exercise the same policy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from generalization_selection import LENGTH_BINS, length_bin, normalize_transcript


SELECTION_VERSION = "heldout-recordings-fixed-length-speaker-expansion-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rank(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"step16-{namespace}-v1:{seed}:{value}".encode("utf-8")).hexdigest(), value


def _distribution(rows: list[dict]) -> dict:
    return {
        "clips": len(rows),
        "speaker": dict(sorted(Counter(row["speaker_id"] for row in rows).items())),
        "length_bin": {bucket: sum(length_bin(row["frames"]) == bucket for row in rows)
                       for bucket in LENGTH_BINS},
        "speaker_length_bin": dict(sorted(Counter(
            f"{row['speaker_id']}/{length_bin(row['frames'])}" for row in rows).items())),
        "recordings": len({row["recording_id"] for row in rows}),
    }


def _ids(values: object, label: str) -> list[str]:
    require(isinstance(values, list) and bool(values), f"{label} must be a nonempty list")
    require(all(isinstance(value, str) and value for value in values),
            f"{label} must contain nonempty strings")
    require(len(values) == len(set(values)), f"duplicate {label}")
    return list(values)


def select_diversity(records: list[dict], original_ids: list[str],
                     heldout_cohorts: dict[str, list[str]], total: int = 256,
                     seed: int = 42) -> dict:
    """Keep original clips and add deterministic length/speaker-stratified clips.

    All train clips in a train60 recording are forbidden, as are normalized
    exact transcripts from either evaluation cohort. Validation is retained in
    full and must have different speakers and recordings. Additions receive
    equal round-robin length quotas, skipping exhausted bins; each bin cycles
    through SHA256-ranked speakers and consumes SHA256-ranked IDs. Original
    and held-out order are preserved, while manifest input order has no effect.
    """
    require(isinstance(total, int) and not isinstance(total, bool) and total > 0,
            "total must be a positive integer")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            "seed must be a nonnegative integer")
    require(isinstance(records, list) and bool(records), "records must be a nonempty list")
    original_ids = _ids(original_ids, "original IDs")
    require(total > len(original_ids), "total must exceed original count")
    require(isinstance(heldout_cohorts, dict) and set(heldout_cohorts) == {"train60", "val20"},
            "heldout cohorts must contain exactly train60 and val20")
    heldout = {name: _ids(heldout_cohorts[name], f"{name} IDs") for name in ("train60", "val20")}
    require(not set(heldout["train60"]) & set(heldout["val20"]), "heldout IDs overlap")
    require(not set(original_ids) & set(heldout["train60"] + heldout["val20"]),
            "original IDs overlap heldout IDs")

    by_id = {}
    for row in records:
        require(isinstance(row, dict), "record must be an object")
        sample_id = row.get("id")
        require(isinstance(sample_id, str) and bool(sample_id), "record ID must be a nonempty string")
        require(sample_id not in by_id, f"duplicate record ID: {sample_id}")
        require(row.get("split") in ("train", "val", "test"), f"unknown split: {sample_id}")
        by_id[sample_id] = row
        # Test metadata is only counted and checked for duplicate IDs/split.
        if row["split"] == "test":
            continue
        length_bin(row.get("frames"))
        for key in ("speaker_id", "recording_id"):
            require(isinstance(row.get(key), str) and bool(row[key]), f"missing {key}: {sample_id}")
        require(bool(normalize_transcript(row.get("text"))), f"empty transcript: {sample_id}")

    for name, ids, split in (("original", original_ids, "train"),
                             ("train60", heldout["train60"], "train"),
                             ("val20", heldout["val20"], "val")):
        require(all(sample_id in by_id for sample_id in ids), f"missing {name} ID in manifest")
        require(all(by_id[sample_id]["split"] == split for sample_id in ids),
                f"{name} IDs must belong to {split}")
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: row["id"])
    validation = sorted((row for row in records if row["split"] == "val"), key=lambda row: row["id"])
    require(set(heldout["val20"]) == {row["id"] for row in validation},
            "val20 must retain all validation IDs")
    original = [by_id[sample_id] for sample_id in original_ids]
    familiar = [by_id[sample_id] for sample_id in heldout["train60"]]
    validation_ordered = [by_id[sample_id] for sample_id in heldout["val20"]]
    train_speakers = {row["speaker_id"] for row in train}
    require({row["speaker_id"] for row in original} == train_speakers,
            "original IDs must cover every train speaker")
    require(not train_speakers & {row["speaker_id"] for row in validation},
            "validation speakers overlap train speakers")
    require(not {row["recording_id"] for row in train} & {row["recording_id"] for row in validation},
            "validation recordings overlap train recordings")

    heldout_recordings = {row["recording_id"] for row in familiar}
    heldout_texts = {normalize_transcript(row["text"]) for row in familiar + validation}
    excluded_by_recording = [row["id"] for row in train if row["recording_id"] in heldout_recordings]
    remaining = [row for row in train if row["recording_id"] not in heldout_recordings]
    excluded_by_transcript = [row["id"] for row in remaining
                              if normalize_transcript(row["text"]) in heldout_texts]
    eligible = [row for row in remaining if normalize_transcript(row["text"]) not in heldout_texts]
    eligible_ids = [row["id"] for row in eligible]
    original_set = set(original_ids)
    require(original_set <= set(eligible_ids),
            "original IDs excluded by heldout recording or normalized transcript")
    require(len(eligible) >= total, f"insufficient eligible train clips: {len(eligible)} < {total}")
    candidates = [row for row in eligible if row["id"] not in original_set]
    additions_count = total - len(original_ids)
    available = Counter(length_bin(row["frames"]) for row in candidates)
    quota_additions = {bucket: 0 for bucket in LENGTH_BINS}
    while sum(quota_additions.values()) < additions_count:
        for bucket in LENGTH_BINS:
            if quota_additions[bucket] < available[bucket] and sum(quota_additions.values()) < additions_count:
                quota_additions[bucket] += 1

    speaker_order = sorted(train_speakers, key=lambda value: _rank(seed, "speaker", value))
    selection_order = {}
    for bucket in LENGTH_BINS:
        groups = defaultdict(list)
        for row in candidates:
            if length_bin(row["frames"]) == bucket:
                groups[row["speaker_id"]].append(row["id"])
        for speaker in groups:
            groups[speaker].sort(key=lambda value: _rank(seed, "id", value))
        positions = Counter()
        selected = []
        while len(selected) < quota_additions[bucket]:
            for speaker in speaker_order:
                if positions[speaker] < len(groups[speaker]) and len(selected) < quota_additions[bucket]:
                    selected.append(groups[speaker][positions[speaker]])
                    positions[speaker] += 1
        selection_order[bucket] = selected
    additions = sorted(sample_id for ids in selection_order.values() for sample_id in ids)
    expanded = original_ids + additions
    require(len(additions) == len(set(additions)) == additions_count, "invalid additions count")
    require(len(expanded) == len(set(expanded)) == total, "invalid expanded count")
    expanded_rows = [by_id[sample_id] for sample_id in expanded]
    require({row["speaker_id"] for row in expanded_rows} == train_speakers,
            "expanded training does not preserve every train speaker")
    disjointness = {
        "expanded_ids_vs_heldout": not set(expanded) & set(heldout["train60"] + heldout["val20"]),
        "expanded_recordings_vs_train60": not {row["recording_id"] for row in expanded_rows} & heldout_recordings,
        "expanded_recordings_vs_val20": not {row["recording_id"] for row in expanded_rows}
                                          & {row["recording_id"] for row in validation},
        "expanded_transcripts_vs_heldout": not {normalize_transcript(row["text"]) for row in expanded_rows}
                                          & heldout_texts,
        "expanded_speakers_vs_val20": not train_speakers & {row["speaker_id"] for row in validation},
        "all_original_ids_retained": expanded[:len(original_ids)] == original_ids,
        "only_train_selected": all(row["split"] == "train" for row in expanded_rows),
        "test_selected": False,
    }
    require(all(value for key, value in disjointness.items() if key != "test_selected"),
            "failed final disjointness check")
    return {
        "selection_version": SELECTION_VERSION, "seed": seed, "total": total,
        "training64": original_ids, "additional192": additions, "expanded256": expanded,
        "heldout": heldout, "eligible_pool_ids": eligible_ids,
        "excluded_recording_ids": sorted(heldout_recordings),
        "excluded_clip_ids": excluded_by_recording,
        "excluded_transcript_ids": excluded_by_transcript,
        "performance_used": False, "cache_accessed": False,
        "algorithm": "Exclude all train60 recording IDs and NFC/whitespace-normalized exact train60/val20 transcripts; retain original IDs in original order; length quotas for additions cycle ascending bins and skip exhausted bins; within each bin cycle SHA256-ranked speakers and take SHA256-ranked clip IDs; sort additions lexically; preserve both heldout cohorts unchanged.",
        "source_counts": {
            "train": len(train), "val": len(validation),
            "test": sum(row["split"] == "test" for row in records),
            "excluded_recording": len(excluded_by_recording),
            "excluded_transcript_after_recording": len(excluded_by_transcript),
            "eligible": len(eligible), "original": len(original_ids),
            "addition_candidates": len(candidates), "additions": len(additions),
        },
        "quota_additions": quota_additions, "speaker_order": speaker_order,
        "selection_order": selection_order,
        "distributions": {
            "training64": _distribution(original),
            "additional192": _distribution([by_id[sample_id] for sample_id in additions]),
            "expanded256": _distribution(expanded_rows), "eligible617": _distribution(eligible),
            "train60": _distribution(familiar), "val20": _distribution(validation_ordered),
        },
        "disjointness": disjointness,
    }
