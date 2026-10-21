"""Stage17 metadata-only expansion, preserving both previous training cohorts.

No files, cached pixels, predictions or models are accessed. The frozen
stage16 selection policy is reused with all existing256 as the retained set;
its SHA256 namespace remains ``step16`` so the policy itself does not change.
Names denote the production 64/192/256/512 cohorts, while small synthetic
fixtures may supply smaller cohorts and a different total.
"""
from __future__ import annotations

from collections import Counter

from diversity_selection import SELECTION_VERSION as BASE_SELECTION_VERSION
from diversity_selection import select_diversity
from generalization_selection import LENGTH_BINS, length_bin


SELECTION_VERSION = "preserve-stage16-cohorts-expand-training-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ids(values: object, label: str) -> list[str]:
    require(isinstance(values, list) and bool(values), f"{label} must be a nonempty list")
    require(all(isinstance(value, str) and value for value in values),
            f"{label} must contain nonempty strings")
    require(len(values) == len(set(values)), f"duplicate {label}")
    return list(values)


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


def select_expansion(records: list[dict], original64_ids: list[str],
                     previous192_ids: list[str], heldout_cohorts: dict[str, list[str]],
                     total: int = 512, seed: int = 42) -> dict:
    """Retain original64 + previous192 in order and select a new disjoint set.

    Exclusion, quotas and SHA256 speaker/ID round-robin use the unchanged
    stage16 helper. Every train60 recording is excluded in full, normalized
    exact train60/val20 transcripts are forbidden, and all validation IDs are
    held fixed. A conflict with any retained ID fails instead of dropping it.
    """
    original = _ids(original64_ids, "original64 IDs")
    previous = _ids(previous192_ids, "previous192 IDs")
    require(not set(original) & set(previous), "original64 and previous192 IDs overlap")
    existing = original + previous
    base = select_diversity(records, existing, heldout_cohorts, total=total, seed=seed)
    require(base["selection_version"] == BASE_SELECTION_VERSION, "Base selection version mismatch")
    require(base["training64"] == existing, "Base policy changed retained cohort order")
    additions = base["additional192"]
    expanded = base["expanded256"]
    by_id = {row["id"]: row for row in records}
    original_distribution = _distribution([by_id[sample_id] for sample_id in original])
    previous_distribution = _distribution([by_id[sample_id] for sample_id in previous])
    require(set(original_distribution["speaker"]) == set(base["distributions"]["training64"]["speaker"]),
            "original64 must preserve the existing training speakers")

    disjointness = {key: value for key, value in base["disjointness"].items()
                    if key != "all_original_ids_retained"}
    disjointness.update({
        "original64_vs_additional192": not set(original) & set(previous),
        "additional256_vs_existing256": not set(additions) & set(existing),
        "all_original64_ids_retained_in_order": expanded[:len(original)] == original,
        "all_additional192_ids_retained_in_order": expanded[len(original):len(existing)] == previous,
        "all_existing256_ids_retained_in_order": expanded[:len(existing)] == existing,
        "heldout_cohorts_unchanged": base["heldout"] == heldout_cohorts,
    })
    require(all(value for key, value in disjointness.items() if key != "test_selected"),
            "Failed expansion lineage or disjointness check")
    counts = {key: value for key, value in base["source_counts"].items() if key != "original"}
    counts.update({"original64": len(original), "additional192": len(previous),
                   "existing256": len(existing), "expanded512": len(expanded)})
    return {
        "selection_version": SELECTION_VERSION, "base_selection_version": BASE_SELECTION_VERSION,
        "seed": seed, "total": total,
        "original64": original, "additional192": previous, "existing256": existing,
        "additional256": additions, "expanded512": expanded, "heldout": base["heldout"],
        "eligible_pool_ids": base["eligible_pool_ids"],
        "excluded_recording_ids": base["excluded_recording_ids"],
        "excluded_clip_ids": base["excluded_clip_ids"],
        "excluded_transcript_ids": base["excluded_transcript_ids"],
        "performance_used": False, "cache_accessed": False,
        "algorithm": "Retain original64 then previous192 in their existing order, with no shared IDs. Reuse frozen stage16 metadata policy and step16 SHA256 ranking namespace: exclude whole train60 recordings and normalized exact heldout transcripts; distribute new additions across ascending length bins, skipping exhausted bins; within each bin cycle SHA256-ranked speakers and consume SHA256-ranked IDs. Sort new additions lexically and append to existing256. Preserve both heldout cohorts unchanged.",
        "source_counts": counts, "quota_additions": base["quota_additions"],
        "speaker_order": base["speaker_order"], "selection_order": base["selection_order"],
        "distributions": {
            "original64": original_distribution, "additional192": previous_distribution,
            "existing256": base["distributions"]["training64"],
            "additional256": base["distributions"]["additional192"],
            "expanded512": base["distributions"]["expanded256"],
            "eligible617": base["distributions"]["eligible617"],
            "train60": base["distributions"]["train60"], "val20": base["distributions"]["val20"],
        },
        "disjointness": disjointness,
    }
