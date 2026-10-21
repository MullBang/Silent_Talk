"""Stage15 metadata-only held-out selection; no cache, model or filesystem I/O.

The fixed models learned training64. Eligible familiar-speaker clips exclude
those exact IDs, every recording represented there, and normalized exact
transcript matches. Validation is retained in full as a development cohort.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import unicodedata


LENGTH_BINS = ("001-050", "051-100", "101-150", "151-200")
SELECTION_VERSION = "heldout-recording-length-speaker-round-robin-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def length_bin(frames: int) -> str:
    require(isinstance(frames, int) and not isinstance(frames, bool)
            and 1 <= frames <= 200, "frames must be an integer in 1..200")
    return LENGTH_BINS[(frames - 1) // 50]


def normalize_transcript(text: str) -> str:
    require(isinstance(text, str), "text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _distribution(rows: list[dict]) -> dict:
    return {"clips": len(rows),
            "speaker": dict(sorted(Counter(row["speaker_id"] for row in rows).items())),
            "length_bin": {bucket: sum(length_bin(row["frames"]) == bucket for row in rows)
                           for bucket in LENGTH_BINS},
            "speaker_length_bin": dict(sorted(Counter(
                f"{row['speaker_id']}/{length_bin(row['frames'])}" for row in rows).items())),
            "recordings": len({row["recording_id"] for row in rows})}


def _rank(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"step15-{namespace}-v1:{seed}:{value}".encode("utf-8")).hexdigest(), value


def select_generalization(records: list[dict], training_ids: list[str],
                          count: int = 60, seed: int = 42) -> dict:
    """Return reproducible train60/val20 cohort metadata without opening pixels.

    Length quotas cycle through ascending 50-frame bins, skipping exhausted
    bins. In each bin, cycle through speakers ranked by SHA256 of
    ``step15-speaker-v1:{seed}:{speaker}``; candidates within speakers are ranked
    by SHA256 of ``step15-id-v1:{seed}:{id}``, with lexical ties. Output cohort
    IDs sort lexically; selection_order records the allocation order. Small
    synthetic fixtures may override count; the runner enforces 60/20/64 and
    the production speaker counts before freezing the actual experiment.
    """
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "count must be a positive integer")
    require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
            "seed must be a nonnegative integer")
    require(isinstance(records, list) and records, "records must be a nonempty list")
    require(isinstance(training_ids, list) and training_ids, "training_ids must be a nonempty list")
    require(all(isinstance(x, str) and x for x in training_ids), "training IDs must be nonempty strings")
    require(len(training_ids) == len(set(training_ids)), "duplicate training IDs")
    by_id = {}
    for row in records:
        require(isinstance(row, dict), "record must be an object")
        sample_id = row.get("id")
        require(isinstance(sample_id, str) and sample_id, "record ID must be a nonempty string")
        require(sample_id not in by_id, f"duplicate record ID: {sample_id}")
        require(row.get("split") in ("train", "val", "test"), f"unknown split: {sample_id}")
        by_id[sample_id] = row
        # Test metadata is never used beyond identity/split inventory.
        if row["split"] == "test":
            continue
        length_bin(row.get("frames"))
        for key in ("speaker_id", "recording_id"):
            require(isinstance(row.get(key), str) and row[key], f"missing {key}: {sample_id}")
        require(bool(normalize_transcript(row.get("text"))), f"empty transcript: {sample_id}")
    require(all(x in by_id for x in training_ids), "missing training ID in manifest")
    require(all(by_id[x]["split"] == "train" for x in training_ids), "training IDs must belong to train")
    training_ids = sorted(training_ids)
    training = [by_id[x] for x in training_ids]
    trained = set(training_ids)
    train = sorted((row for row in records if row["split"] == "train"), key=lambda row: row["id"])
    validation = sorted((row for row in records if row["split"] == "val"), key=lambda row: row["id"])
    require(bool(validation), "validation cohort must not be empty")
    training_speakers = {row["speaker_id"] for row in training}
    training_recordings = {row["recording_id"] for row in training}
    training_texts = {normalize_transcript(row["text"]) for row in training}
    require({row["speaker_id"] for row in train} <= training_speakers,
            "train cohort contains a speaker absent from training IDs")
    require(not training_speakers & {row["speaker_id"] for row in validation},
            "validation speakers overlap training speakers")
    require(not training_recordings & {row["recording_id"] for row in validation},
            "validation recordings overlap training recordings")

    excluded_id = [row["id"] for row in train if row["id"] in trained]
    remaining = [row for row in train if row["id"] not in trained]
    excluded_recording = [row["id"] for row in remaining if row["recording_id"] in training_recordings]
    remaining = [row for row in remaining if row["recording_id"] not in training_recordings]
    excluded_transcript = [row["id"] for row in remaining if normalize_transcript(row["text"]) in training_texts]
    eligible = [row for row in remaining if normalize_transcript(row["text"]) not in training_texts]
    require(len(eligible) >= count, f"insufficient eligible train clips: {len(eligible)} < {count}")
    available = Counter(length_bin(row["frames"]) for row in eligible)
    quotas = {bucket: 0 for bucket in LENGTH_BINS}
    while sum(quotas.values()) < count:
        for bucket in LENGTH_BINS:
            if quotas[bucket] < available[bucket] and sum(quotas.values()) < count:
                quotas[bucket] += 1
    speaker_order = sorted(training_speakers, key=lambda speaker: _rank(seed, "speaker", speaker))
    selection_order = {}
    for bucket in LENGTH_BINS:
        groups = defaultdict(list)
        for row in eligible:
            if length_bin(row["frames"]) == bucket:
                groups[row["speaker_id"]].append(row["id"])
        for speaker in groups:
            groups[speaker].sort(key=lambda sample_id: _rank(seed, "id", sample_id))
        positions = Counter()
        selected = []
        while len(selected) < quotas[bucket]:
            for speaker in speaker_order:
                if positions[speaker] < len(groups[speaker]) and len(selected) < quotas[bucket]:
                    selected.append(groups[speaker][positions[speaker]])
                    positions[speaker] += 1
        selection_order[bucket] = selected
    selected_ids = sorted(sample_id for ids in selection_order.values() for sample_id in ids)
    selected_rows = [by_id[x] for x in selected_ids]
    require(len(selected_ids) == len(set(selected_ids)) == count, "invalid selected count")
    if count >= len(training_speakers):
        require({row["speaker_id"] for row in selected_rows} == training_speakers,
                "selection does not cover every training speaker")
    val_ids = [row["id"] for row in validation]
    val_text_overlap = sorted(row["id"] for row in validation
                              if normalize_transcript(row["text"]) in training_texts)
    disjointness = {
        "train60_ids_vs_training": not set(selected_ids) & trained,
        "train60_recordings_vs_training": not {r["recording_id"] for r in selected_rows} & training_recordings,
        "train60_transcripts_vs_training": not {normalize_transcript(r["text"]) for r in selected_rows} & training_texts,
        "val20_ids_vs_training": not set(val_ids) & trained,
        "val20_speakers_vs_training": not {r["speaker_id"] for r in validation} & training_speakers,
        "val20_recordings_vs_training": not {r["recording_id"] for r in validation} & training_recordings,
        "val20_transcripts_vs_training": not val_text_overlap,
        "evaluation_cohort_ids": not set(selected_ids) & set(val_ids),
        "test_selected": False,
    }
    return {"selection_version": SELECTION_VERSION, "seed": seed, "requested_count": count,
            "cohorts": {"train60": selected_ids, "val20": val_ids}, "training_ids": training_ids,
            "performance_used": False, "cache_accessed": False,
            "algorithm": "Exclude training IDs, then their recording IDs, then NFC/whitespace-normalized exact training transcripts; length quotas cycle ascending bins and skip exhausted bins; within each bin cycle SHA256-ranked speakers and take SHA256-ranked clip IDs; retain all val IDs.",
            "source_counts": {"train": len(train), "val": len(validation),
                              "test": sum(r["split"] == "test" for r in records), "eligible": len(eligible)},
            "exclusions": {"training_id": {"count": len(excluded_id), "ids": excluded_id},
                           "training_recording_after_id": {"count": len(excluded_recording), "ids": excluded_recording},
                           "training_transcript_after_recording": {"count": len(excluded_transcript), "ids": excluded_transcript}},
            "quotas": quotas, "speaker_order": speaker_order, "selection_order": selection_order,
            "distributions": {"training": _distribution(training), "eligible": _distribution(eligible),
                              "train60": _distribution(selected_rows), "val20": _distribution(validation)},
            "disjointness": disjointness, "val_training_transcript_overlap_ids": val_text_overlap}
