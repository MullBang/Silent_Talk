"""Stage17 cohort identity, fixed holdouts and metadata-only expansion guards."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys
import unicodedata

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from expansion_selection import select_expansion


def row(sample_id, speaker="a", frames=75, split="train", recording=None, text=None):
    return {"id": sample_id, "speaker_id": speaker, "frames": frames, "split": split,
            "recording_id": recording or f"record-{sample_id}",
            "text": text if text is not None else f"문장 {sample_id}"}


def fixture():
    original = [row("original-b", "b"), row("original-a")]
    previous = [row("previous-b", "b"), row("previous-a")]
    candidates = [row(f"candidate-{speaker}-{bucket}-{i}", speaker, bucket * 50 + 25)
                  for speaker in ("a", "b") for bucket in range(4) for i in range(5)]
    evaluation = [row("held-b", "b"), row("held-a"),
                  row("val-2", "v", split="val"), row("val-1", "v", split="val")]
    records = original + previous + candidates + evaluation + [{"id": "test", "split": "test"}]
    return (records, [r["id"] for r in original], [r["id"] for r in previous],
            {"train60": ["held-b", "held-a"], "val20": ["val-2", "val-1"]})


def test_manifest_order_invariance_preserves_both_historical_cohorts_without_mutation():
    records, original, previous, heldout = fixture()
    before = copy.deepcopy((records, original, previous, heldout))
    result = select_expansion(records, original, previous, heldout, total=20)
    assert (records, original, previous, heldout) == before
    random.Random(617).shuffle(records)
    assert select_expansion(records, original, previous, heldout, total=20) == result
    assert result["original64"] == original
    assert result["additional192"] == previous
    assert result["existing256"] == original + previous
    assert result["expanded512"] == original + previous + result["additional256"]
    assert result["additional256"] == sorted(result["additional256"])
    assert result["heldout"] == heldout
    assert result["source_counts"]["original64"] == 2
    assert result["source_counts"]["additional192"] == 2
    assert result["source_counts"]["existing256"] == 4
    assert result["source_counts"]["expanded512"] == 20
    assert result["distributions"]["additional192"]["clips"] == 2
    assert result["distributions"]["additional256"]["clips"] == 16
    assert all(value for key, value in result["disjointness"].items() if key != "test_selected")
    assert result["disjointness"]["test_selected"] is False


def test_heldout_recording_siblings_and_normalized_transcripts_are_never_added():
    records, original, previous, heldout = fixture()
    next(r for r in records if r["id"] == "held-a")["text"] = "한국 어"
    records += [row("recording-sibling", recording="record-held-a"),
                row("train-text-copy", text=" \t" + unicodedata.normalize("NFD", "한국") + "\n어 "),
                row("val-text-copy", text="  문장\tval-1 ")]
    result = select_expansion(records, original, previous, heldout, total=20)
    assert result["excluded_clip_ids"] == ["held-a", "held-b", "recording-sibling"]
    assert result["excluded_transcript_ids"] == ["train-text-copy", "val-text-copy"]
    assert not set(result["excluded_clip_ids"] + result["excluded_transcript_ids"]) & set(result["expanded512"])
    assert result["source_counts"]["eligible"] == 44


def test_empty_short_bin_and_exhausted_long_bin_redistribute_new_quotas():
    records, original, previous, heldout = fixture()
    records = [r for r in records if not r["id"].startswith("candidate") or 50 < r["frames"] <= 150]
    records += [row("only-long", frames=180)]
    result = select_expansion(records, original, previous, heldout, total=20)
    assert result["quota_additions"] == {"001-050": 0, "051-100": 8, "101-150": 7, "151-200": 1}
    assert "only-long" in result["additional256"]
    assert len(result["expanded512"]) == len(set(result["expanded512"])) == 20
    cells = result["distributions"]["additional256"]["speaker_length_bin"]
    for bucket in ("051-100", "101-150"):
        assert abs(cells[f"a/{bucket}"] - cells[f"b/{bucket}"]) <= 1


def test_performance_and_nonexistent_cache_fields_cannot_influence_selection():
    records, original, previous, heldout = fixture()
    expected = select_expansion(records, original, previous, heldout, total=20)
    for item in records:
        item.update(cer=-999, prediction="chosen", cache_path="never/open/absent.npz", cache_sha256="invalid")
    records[-1].update(text=None, frames=False, speaker_id=None, recording_id=None)
    assert select_expansion(records, original, previous, heldout, total=20) == expected
    assert expected["performance_used"] is False
    assert expected["cache_accessed"] is False


def test_changing_seed_changes_only_new_selection_not_existing_cohorts_or_pool():
    records, original, previous, heldout = fixture()
    first = select_expansion(records, original, previous, heldout, total=20, seed=42)
    second = select_expansion(records, original, previous, heldout, total=20, seed=43)
    assert first["additional256"] != second["additional256"]
    for key in ("original64", "additional192", "existing256", "heldout", "eligible_pool_ids", "quota_additions"):
        assert first[key] == second[key]


@pytest.mark.parametrize("change,message", [
    (lambda r, o, p, h: o.append(o[0]), "duplicate original64"),
    (lambda r, o, p, h: p.append(p[0]), "duplicate previous192"),
    (lambda r, o, p, h: p.append(o[0]), "original64 and previous192 IDs overlap"),
    (lambda r, o, p, h: p.append("missing"), "missing original ID"),
    (lambda r, o, p, h: p.append("test"), "original IDs must belong to train"),
    (lambda r, o, p, h: p.append("held-b"), "original IDs overlap heldout"),
    (lambda r, o, p, h: r[2].update(recording_id="record-held-a"), "original IDs excluded"),
    (lambda r, o, p, h: r[2].update(text="문장 val-1"), "original IDs excluded"),
    (lambda r, o, p, h: h["val20"].pop(), "retain all validation"),
    (lambda r, o, p, h: h["train60"].append("test"), "train60 IDs must belong to train"),
    (lambda r, o, p, h: r.append(copy.deepcopy(r[0])), "duplicate record ID"),
    (lambda r, o, p, h: r[-2].update(speaker_id="a"), "validation speakers overlap"),
    (lambda r, o, p, h: o.remove("original-b"), "original64 must preserve"),
])
def test_invalid_historical_cohorts_or_holdouts_fail_closed(change, message):
    records, original, previous, heldout = fixture()
    change(records, original, previous, heldout)
    with pytest.raises(ValueError, match=message):
        select_expansion(records, original, previous, heldout, total=20)


@pytest.mark.parametrize("kwargs,message", [
    ({"total": True}, "positive integer"),
    ({"total": 4}, "exceed original"),
    ({"total": 45}, "insufficient eligible"),
    ({"total": 20, "seed": -1}, "nonnegative integer"),
])
def test_invalid_total_or_seed_cannot_change_frozen_cohorts(kwargs, message):
    records, original, previous, heldout = fixture()
    with pytest.raises(ValueError, match=message):
        select_expansion(records, original, previous, heldout, **kwargs)


@pytest.mark.skipif(
    not (ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").exists()
    or not (ROOT / "results/step16-diversity/run01/plan.json").exists(),
    reason="로컬 AI Hub 데이터 명세·이전 단계 결과가 없음 (저장소에 포함하지 않는 자료)",
)
def test_actual_manifest_retains_exact_stage16_cohorts_and_heldout_pool():
    manifest = json.loads((ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/step16-diversity/run01/plan.json").read_text(encoding="utf-8"))
    cohorts = prior["cohorts"]
    heldout = {name: cohorts[name] for name in ("train60", "val20")}
    result = select_expansion(manifest["records"], cohorts["original64"], cohorts["additional192"], heldout)
    assert result["original64"] == cohorts["original64"]
    assert result["additional192"] == cohorts["additional192"]
    assert result["existing256"] == prior["selection"]["expanded256"]
    assert result["heldout"] == prior["selection"]["heldout"]
    assert result["eligible_pool_ids"] == prior["selection"]["eligible_pool_ids"]
    assert result["excluded_clip_ids"] == prior["selection"]["excluded_clip_ids"]
    assert result["excluded_recording_ids"] == prior["selection"]["excluded_recording_ids"]
    assert result["source_counts"] == {
        "train": 1000, "val": 20, "test": 50, "excluded_recording": 383,
        "excluded_transcript_after_recording": 0, "eligible": 617,
        "addition_candidates": 361, "additions": 256,
        "original64": 64, "additional192": 192, "existing256": 256, "expanded512": 512,
    }
    assert result["quota_additions"] == {"001-050": 0, "051-100": 111, "101-150": 111, "151-200": 34}
    assert result["distributions"]["expanded512"]["length_bin"] == {
        "001-050": 28, "051-100": 187, "101-150": 187, "151-200": 110,
    }
    assert result["distributions"]["expanded512"]["recordings"] == 59
    assert len(result["distributions"]["expanded512"]["speaker"]) == 8
    assert len(set(result["expanded512"])) == 512
    assert all(value for key, value in result["disjointness"].items() if key != "test_selected")
