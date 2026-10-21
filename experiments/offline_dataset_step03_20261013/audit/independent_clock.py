"""Independent audit arithmetic; never imports the production preparation code."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from fractions import Fraction
import unicodedata


def rational(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal_seconds": float(value)}


def parse_rational(value: dict) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def token_ids(text: str) -> list[int]:
    vocabulary = (["<blank>"] + [chr(x) for x in range(0x1100, 0x1113)]
                  + [chr(x) for x in range(0x1161, 0x1176)]
                  + [chr(x) for x in range(0x11A8, 0x11C3)] + [" "])
    # Inputs here must already be the frozen human-linked canonical text.
    return [vocabulary.index(x) for x in unicodedata.normalize("NFD", text)]


def ctc_minimum(labels: list[int]) -> int:
    if not labels or any(not isinstance(x, int) or not 1 <= x <= 68 for x in labels):
        raise ValueError("Nonempty canonical labels with no blank are required")
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))


def mapped_interval(context_first_48k: int, start_16k: int, end_16k: int,
                    offset_48k: int) -> tuple[Fraction, Fraction]:
    if not (0 <= start_16k < end_16k) or context_first_48k < 0:
        raise ValueError("Invalid source sample interval")
    # Integer 16k sample origin maps to 48k source samples exactly; offset once.
    return (Fraction(context_first_48k + start_16k * 3 + offset_48k, 48000),
            Fraction(context_first_48k + end_16k * 3 + offset_48k, 48000))


def display_grid(start: Fraction, end: Fraction, origin: Fraction | None,
                 origin_measured: bool) -> dict:
    if origin is None or origin_measured is not True:
        raise ValueError("An actually measured first valid PTS is required")
    if start < origin or end <= start:
        raise ValueError("Invalid source coverage")
    a, b = (start - origin) * 25, (end - origin) * 25
    first = a.numerator // a.denominator
    stop = -(-b.numerator // b.denominator)
    if not 0 <= first < stop:
        raise ValueError("Empty or negative display grid")
    grid = [origin + Fraction(k, 25) for k in range(first, stop)]
    display_start, display_end = origin + Fraction(first, 25), origin + Fraction(stop, 25)
    return {"first": first, "stop": stop, "count": stop - first, "grid": grid,
            "display_start": display_start, "display_end": display_end,
            "left_extension": start - display_start, "right_extension": display_end - end}


def nearest_actual_pts(targets: list[Fraction], pts: list[Fraction]) -> list[int]:
    if not pts or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Strictly increasing actual PTS values are required")
    indices = []
    for target in targets:
        if target < pts[0] or target > pts[-1]:
            raise ValueError("Actual PTS coverage does not bracket a grid target")
        upper = bisect_left(pts, target)
        if upper == 0 or pts[upper] == target:
            indices.append(upper)
        else:
            lower = upper - 1
            indices.append(lower if target - pts[lower] <= pts[upper] - target else upper)
    return indices


def assert_partition(rows: list[dict], expected: list[dict]) -> dict:
    """Use an explicit schema adapter before calling; no production schema guesses.

    Rows contain parent_id, kind (phrase/full_sentence), split, recording_group,
    video_sha256, label_sha256, text, canonical_parent_text, start/end_sample_16k.
    This validator deliberately does not choose a split.
    """
    keyed = {(r["parent_id"], r["kind"]): r for r in rows}
    if len(keyed) != len(rows) or set(keyed) != {(r["parent_id"], r["kind"]) for r in expected}:
        raise ValueError("Missing, repeated or extra parent/clip roles")
    groups = defaultdict(set)
    parent_splits = defaultdict(set)
    counts = defaultdict(int)
    for ref in expected:
        r = keyed[(ref["parent_id"], ref["kind"])]
        for key in ("text", "canonical_parent_text", "recording_group", "video_sha256",
                    "label_sha256", "start_sample_16k", "end_sample_16k"):
            if r[key] != ref[key]:
                raise ValueError(f"Reference mismatch: {ref['parent_id']} {ref['kind']} {key}")
        if r["split"] not in ("train", "dev"):
            raise ValueError("Only explicit train/dev assignments accepted")
        parent_splits[r["parent_id"]].add(r["split"])
        counts[r["split"]] += 1
        for kind, value in (("recording", r["recording_group"]), ("video_hash", r["video_sha256"]),
                            ("label_hash", r["label_sha256"]),
                            ("parent_text", "".join(r["canonical_parent_text"].split())),
                            ("target_text", "".join(r["text"].split()))):
            groups[(kind, value)].add(r["split"])
    crossing = [key for key, values in groups.items() if len(values) > 1]
    if crossing or any(len(x) != 1 for x in parent_splits.values()):
        raise ValueError(f"Cross-split recording/copy/text leakage: {crossing}")
    if dict(counts) != {"train": 24, "dev": 6}:
        raise ValueError(f"Expected 12 train and 3 dev parents, each with two roles: {dict(counts)}")
    return {"clip_counts": dict(counts), "parent_count": len(parent_splits),
            "crossing_groups": [], "role_count_per_parent": 2}
