"""Deterministic stage18 metadata-only pilot selection for new source packages.

This module performs no file, video, cache, model or network access. It adds a
pilot train/val role to copied source records and never rewrites their metadata.
The pilot is an intake-readiness sample, not a replacement training manifest.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib

from models.dataset_manifest import parse_identity
from models.tokenizer import ctc_min_input_len, encode_text, normalize_text, tokenizer_metadata

PACKAGES = ("TS2", "TS20", "TS200")
POLICY_VERSION = "new-speaker-package-intake-pilot-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, name: str, minimum: int = 0) -> None:
    require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")


def _rank(seed: int, kind: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"18:{seed}:{kind}:{value}".encode("utf-8")).hexdigest(), value


def _identity(row: dict, label: str) -> dict:
    sample_id = row.get("id")
    require(isinstance(sample_id, str) and sample_id and not any(ch in sample_id for ch in "/\\.:"),
            f"{label}: canonical filename ID required")
    identity = parse_identity(sample_id)
    for key in ("speaker_id", "recording_id", "utterance_id", "angle"):
        require(identity[key] is not None and row.get(key) == identity[key], f"{label}: filename {key} mismatch")
    return identity


def _text(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label}: transcript must be a string")
    normalized = normalize_text(value)
    require(bool(normalized), f"{label}: empty normalized transcript")
    return normalized


def _distribution(rows: list[dict]) -> dict:
    return {"clips": len(rows), "speakers": len({r["speaker_id"] for r in rows}),
        "recordings": len({r["recording_id"] for r in rows}),
        "unique_texts": len({normalize_text(r["text"]) for r in rows})}


def select_intake(candidates: list[dict], old_records: list[dict], protected_recording_ids: list[str],
                  seed: int = 42, videos_per_speaker: int = 2, clips_per_video: int = 4) -> dict:
    """Choose fixed videos/clips for every new speaker without silent fallback.

    First validate every source record, then exclude existing speakers and
    protected recordings. Each required package assigns its SHA256-first new
    speaker to pilot val and all remaining new speakers to pilot train. Each
    speaker's SHA256-first recordings are frozen BEFORE transcript filtering.
    Val clips are chosen first; both roles reject all old normalized texts and
    train also rejects selected val texts. Within each split, selected texts
    are unique. If any fixed speaker/video quota cannot be met, raise rather
    than changing the split, picking another video, or relaxing exclusions.

    Existing records supply identity/text protection only; their source files
    and their train/val/test roles are never opened, changed or reassigned.
    """
    _integer(seed, "seed")
    _integer(videos_per_speaker, "videos_per_speaker", 1)
    _integer(clips_per_video, "clips_per_video", 1)
    require(isinstance(candidates, list) and bool(candidates), "candidates must be a nonempty list")
    require(isinstance(old_records, list) and bool(old_records), "old_records must be a nonempty list")
    require(isinstance(protected_recording_ids, list)
            and all(isinstance(value, str) and value for value in protected_recording_ids),
            "protected_recording_ids must contain strings")
    require(len(protected_recording_ids) == len(set(protected_recording_ids)), "duplicate protected recording IDs")
    protected = set(protected_recording_ids)
    old_speakers, old_texts, old_ids = set(), set(), set()
    for row in old_records:
        require(isinstance(row, dict), "old record must be an object")
        _identity(row, "old record")
        require(row["id"] not in old_ids, "duplicate old record ID")
        old_ids.add(row["id"])
        old_speakers.add(row["speaker_id"])
        old_texts.add(_text(row.get("text"), "old record"))

    by_id, normalized, speaker_package, utterances = {}, {}, {}, set()
    for row in candidates:
        require(isinstance(row, dict), "candidate must be an object")
        identity = _identity(row, "candidate")
        sample_id = row["id"]
        require(sample_id not in by_id, f"duplicate candidate ID: {sample_id}")
        require(identity["utterance_id"] not in utterances, f"duplicate candidate utterance: {sample_id}")
        require("split" not in row, f"candidate already has a split: {sample_id}")
        require(row.get("source_partition") == "Training", f"candidate must be official Training: {sample_id}")
        require(identity["angle"] == "A", f"candidate must have frontal angle A: {sample_id}")
        package = row.get("package")
        require(package in PACKAGES, f"unsupported source package: {package}")
        speaker = identity["speaker_id"]
        require(speaker not in speaker_package or speaker_package[speaker] == package,
                f"speaker appears in multiple packages: {speaker}")
        speaker_package[speaker] = package
        text = _text(row.get("text"), f"candidate {sample_id}")
        if "source_text" in row:
            require(_text(row["source_text"], sample_id) == text, f"source_text/text mismatch: {sample_id}")
        frames, minimum = row.get("expected_frames"), row.get("ctc_min_frames")
        _integer(frames, "expected_frames", 1)
        require(frames <= 200, f"expected_frames exceeds200: {sample_id}")
        _integer(minimum, "ctc_min_frames", 1)
        labels = encode_text(text)
        require(minimum == ctc_min_input_len(labels), f"ctc_min_frames/tokenizer mismatch: {sample_id}")
        require(frames >= minimum, f"CTC target cannot fit expected_frames: {sample_id}")
        if "label" in row:
            require(row["label"] == labels, f"candidate label/tokenizer mismatch: {sample_id}")
        by_id[sample_id], normalized[sample_id] = row, text
        utterances.add(identity["utterance_id"])

    excluded = {name: set() for name in ("old_speaker", "protected_recording", "old_transcript",
        "selected_val_transcript", "duplicate_selected_val_transcript", "duplicate_selected_train_transcript")}
    remaining = []
    for sample_id in sorted(by_id):
        row = by_id[sample_id]
        if row["speaker_id"] in old_speakers:
            excluded["old_speaker"].add(sample_id)
        elif row["recording_id"] in protected:
            excluded["protected_recording"].add(sample_id)
        else:
            remaining.append(row)
    package_speakers = {package: sorted({r["speaker_id"] for r in remaining if r["package"] == package},
        key=lambda value: _rank(seed, "speaker", value)) for package in PACKAGES}
    for package, speakers in package_speakers.items():
        require(len(speakers) >= 2, f"{package}: at least2 new eligible speakers required")
    val_speakers = {package: speakers[0] for package, speakers in package_speakers.items()}
    speaker_mapping = {speaker: "val" if speaker == val_speakers[package] else "train"
        for package in PACKAGES for speaker in package_speakers[package]}
    recordings = defaultdict(lambda: defaultdict(list))
    for row in remaining:
        recordings[row["speaker_id"]][row["recording_id"]].append(row)
        if normalized[row["id"]] in old_texts:
            excluded["old_transcript"].add(row["id"])
    chosen_recordings = {}
    for speaker in speaker_mapping:
        ranked = sorted(recordings[speaker], key=lambda value: _rank(seed, "recording", value))
        require(len(ranked) >= videos_per_speaker,
                f"speaker {speaker}: insufficient recordings ({len(ranked)} < {videos_per_speaker})")
        chosen_recordings[speaker] = ranked[:videos_per_speaker]

    selected = {"train": [], "val": []}
    seen_texts = {"train": set(), "val": set()}
    for split in ("val", "train"):
        if split == "train":
            excluded["selected_val_transcript"] = {r["id"] for r in remaining
                if speaker_mapping[r["speaker_id"]] == "train" and r["id"] not in excluded["old_transcript"]
                and normalized[r["id"]] in seen_texts["val"]}
        for package in PACKAGES:
            for speaker in package_speakers[package]:
                if speaker_mapping[speaker] != split:
                    continue
                for recording in chosen_recordings[speaker]:
                    chosen = []
                    ordered = sorted(recordings[speaker][recording], key=lambda row: _rank(seed, "clip", row["id"]))
                    for row in ordered:
                        sample_id, text = row["id"], normalized[row["id"]]
                        if sample_id in excluded["old_transcript"] or sample_id in excluded["selected_val_transcript"]:
                            continue
                        if text in seen_texts[split]:
                            excluded[f"duplicate_selected_{split}_transcript"].add(sample_id)
                            continue
                        if len(chosen) < clips_per_video:
                            chosen.append(row)
                            seen_texts[split].add(text)
                    require(len(chosen) == clips_per_video,
                        f"{split}/{speaker}/{recording}: insufficient unique eligible clips ({len(chosen)} < {clips_per_video}); no fallback")
                    selected[split].extend(chosen)

    records = [{**copy.deepcopy(row), "split": split} for split in ("train", "val")
        for row in sorted(selected[split], key=lambda row: row["id"])]
    selected_ids = {r["id"] for r in records}
    excluded_ids = set().union(*excluded.values())
    require(len(selected_ids) == len(records) and not selected_ids & excluded_ids, "Selected/excluded identity overlap")
    require(not seen_texts["train"] & seen_texts["val"] and not (seen_texts["train"] | seen_texts["val"]) & old_texts,
            "Final transcript leakage")
    require(not {r["speaker_id"] for r in records} & old_speakers
            and not {r["recording_id"] for r in records} & protected, "Final protected identity leakage")
    unselected = sorted(set(by_id) - selected_ids - excluded_ids)
    counts = {"candidates": len(candidates), "selected": len(records), "excluded": len(excluded_ids),
        "unselected": len(unselected),
        **{split: _distribution([r for r in records if r["split"] == split]) for split in ("train", "val")},
        "test": _distribution([]),
        "by_package": {package: {"train": sum(r["package"] == package and r["split"] == "train" for r in records),
            "val": sum(r["package"] == package and r["split"] == "val" for r in records),
            "speakers": len(package_speakers[package])} for package in PACKAGES}}
    require(counts["selected"] == len(speaker_mapping) * videos_per_speaker * clips_per_video, "Pilot quota mismatch")
    return {"records": records, "speaker_mapping": dict(sorted(speaker_mapping.items())),
        "policy": {"version": POLICY_VERSION, "purpose": "new_data_intake_readiness_only", "seed": seed,
            "videos_per_speaker": videos_per_speaker, "clips_per_video": clips_per_video,
            "packages": list(PACKAGES), "official_partition": "Training", "angle": "A", "max_frames": 200,
            "split_roles": {"train": "pilot_train", "val": "pilot_development", "test": "not_selected"},
            "dev_speakers_by_package": val_speakers, "selected_recordings": dict(sorted(chosen_recordings.items())),
            "excluded_old_speakers": sorted(old_speakers), "protected_recording_ids": sorted(protected),
            "speaker_packages": {speaker: speaker_package[speaker] for speaker in sorted(speaker_mapping)},
            "ranking": "SHA256 UTF-8 18:{seed}:{speaker|recording|clip}:{identity}, lexical identity tie break",
            "selection_order": "required packages TS2,TS20,TS200; ranked speakers; ranked recordings frozen before text filtering; val clips before train; ranked clips",
            "text_policy": "Tokenizer-normalized exact text: exclude every old train/val/test text from both pilot roles; exclude selected pilot val text from train; unique selected text within each split; insufficient fixed quotas fail without fallback",
            "tokenizer": tokenizer_metadata(), "metadata_only": True, "performance_used": False,
            "source_metadata_rewritten": False, "training_performed": False, "test_access": False,
            "generic_manifest_compatible": False},
        "excluded_summary": {**{name: {"count": len(ids), "ids": sorted(ids)} for name, ids in excluded.items()},
            "not_selected_by_fixed_quota": {"count": len(unselected), "ids": unselected}},
        "counts": counts}
