"""Stage16 train expansion preserves frozen evaluation recordings and lineage."""
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
from diversity_selection import select_diversity


def row(sample_id, speaker="a", frames=75, split="train", recording=None, text=None):
    return {"id": sample_id, "speaker_id": speaker, "frames": frames, "split": split,
            "recording_id": recording or f"record-{sample_id}",
            "text": text if text is not None else f"문장 {sample_id}"}


def fixture():
    original = [row("original-b", "b"), row("original-a")]
    candidates = [row(f"candidate-{speaker}-{bucket}-{i}", speaker, bucket * 50 + 25)
                  for speaker in ("a", "b") for bucket in range(4) for i in range(5)]
    evaluation = [row("held-b", "b"), row("held-a"),
                  row("val-2", "v", split="val"), row("val-1", "v", split="val")]
    records = original + candidates + evaluation + [{"id": "test", "split": "test"}]
    return records, [r["id"] for r in original], {"train60": ["held-b", "held-a"], "val20": ["val-2", "val-1"]}


def test_input_order_invariant_without_changing_original_or_heldout_order():
    records, original, heldout = fixture()
    before = copy.deepcopy((records, original, heldout))
    result = select_diversity(records, original, heldout, total=14)
    assert (records, original, heldout) == before
    random.Random(211).shuffle(records)
    assert select_diversity(records, original, heldout, total=14) == result
    assert result["training64"] == original
    assert result["expanded256"][:2] == original
    assert result["heldout"] == heldout
    assert result["additional192"] == sorted(result["additional192"])
    assert result["eligible_pool_ids"] == sorted(result["eligible_pool_ids"])
    assert result["disjointness"]["test_selected"] is False
    assert all(value for key, value in result["disjointness"].items() if key != "test_selected")


def test_entire_heldout_recordings_and_normalized_transcripts_are_excluded():
    records, original, heldout = fixture()
    by_id = {r["id"]: r for r in records}
    by_id["held-a"]["text"] = "한국 어"
    by_id["val-1"]["text"] = "한글 음성"
    records += [row("recording-sibling", recording="record-held-a"),
                row("train-text-copy", text=" \t" + unicodedata.normalize("NFD", "한국") + "\n어  "),
                row("val-text-copy", text=" 한글\t음성 ")]
    result = select_diversity(records, original, heldout, total=14)
    assert result["excluded_recording_ids"] == ["record-held-a", "record-held-b"]
    assert result["excluded_clip_ids"] == ["held-a", "held-b", "recording-sibling"]
    assert result["excluded_transcript_ids"] == ["train-text-copy", "val-text-copy"]
    assert result["source_counts"]["eligible"] == 42
    assert result["source_counts"]["addition_candidates"] == 40
    assert not set(result["excluded_clip_ids"] + result["excluded_transcript_ids"]) & set(result["expanded256"])


def test_length_scarcity_redistributes_and_speaker_cells_balance():
    records, original, heldout = fixture()
    records = [r for r in records if not r["id"].startswith("candidate") or r["frames"] > 50]
    records.append(row("only-short", frames=25))
    result = select_diversity(records, original, heldout, total=14)
    assert result["quota_additions"] == {"001-050": 1, "051-100": 4, "101-150": 4, "151-200": 3}
    assert "only-short" in result["additional192"]
    cells = result["distributions"]["additional192"]["speaker_length_bin"]
    for bucket in ("051-100", "101-150", "151-200"):
        assert abs(cells[f"a/{bucket}"] - cells[f"b/{bucket}"]) <= 1


def test_exhausted_length_and_speaker_groups_cannot_stall_or_duplicate():
    records, original, heldout = fixture()
    records = [r for r in records if not r["id"].startswith("candidate")
               or r["speaker_id"] == "a" and r["frames"] == 125]
    result = select_diversity(records, original, heldout, total=7)
    assert result["quota_additions"] == {"001-050": 0, "051-100": 0, "101-150": 5, "151-200": 0}
    assert len(set(result["expanded256"])) == 7
    assert set(result["distributions"]["expanded256"]["speaker"]) == {"a", "b"}


def test_predictions_cache_paths_and_test_payload_do_not_affect_selection():
    records, original, heldout = fixture()
    expected = select_diversity(records, original, heldout, total=14)
    for item in records:
        item.update(prediction="perfect", cer=-999, cache_path="absent/never/open.npz", cache_sha256="not-a-hash")
    records[-1].update(frames=False, text=None, speaker_id=None, recording_id=None)
    assert select_diversity(records, original, heldout, total=14) == expected
    assert expected["cache_accessed"] is False
    assert expected["performance_used"] is False


def test_seed_controls_additions_but_not_frozen_cohorts_or_quotas():
    records, original, heldout = fixture()
    first = select_diversity(records, original, heldout, total=14, seed=42)
    second = select_diversity(records, original, heldout, total=14, seed=19)
    assert first["additional192"] != second["additional192"]
    for key in ("training64", "heldout", "quota_additions", "eligible_pool_ids"):
        assert first[key] == second[key]


@pytest.mark.parametrize("change,message", [
    (lambda r, o, h: o.append(o[0]), "duplicate original"),
    (lambda r, o, h: o.append("absent"), "missing original"),
    (lambda r, o, h: o.append("test"), "original IDs must belong to train"),
    (lambda r, o, h: h["train60"].append(h["train60"][0]), "duplicate train60"),
    (lambda r, o, h: h["train60"].append("absent"), "missing train60"),
    (lambda r, o, h: h["train60"].append("test"), "train60 IDs must belong to train"),
    (lambda r, o, h: h["train60"].append(o[0]), "original IDs overlap heldout"),
    (lambda r, o, h: h["val20"].pop(), "retain all validation"),
    (lambda r, o, h: r.append(copy.deepcopy(r[0])), "duplicate record"),
    (lambda r, o, h: r[0].update(recording_id="record-held-a"), "original IDs excluded"),
    (lambda r, o, h: r[0].update(text="문장 val-1"), "original IDs excluded"),
    (lambda r, o, h: r[2].update(speaker_id="extra"), "cover every train speaker"),
    (lambda r, o, h: r[-2].update(speaker_id="a"), "validation speakers overlap"),
    (lambda r, o, h: r[-2].update(recording_id="record-candidate-a-0-0"), "validation recordings overlap"),
    (lambda r, o, h: r[2].update(frames=True), "frames"),
    (lambda r, o, h: r[2].update(text="\t "), "empty transcript"),
    (lambda r, o, h: r[2].update(split="other"), "unknown split"),
])
def test_metadata_and_lineage_violations_fail_closed(change, message):
    records, original, heldout = fixture()
    change(records, original, heldout)
    with pytest.raises(ValueError, match=message):
        select_diversity(records, original, heldout, total=14)


@pytest.mark.parametrize("kwargs,message", [
    ({"total": True}, "positive integer"),
    ({"total": 2}, "exceed original"),
    ({"total": 43}, "insufficient eligible"),
    ({"total": 14, "seed": -1}, "nonnegative integer"),
])
def test_invalid_budgets_and_seed_fail_closed(kwargs, message):
    records, original, heldout = fixture()
    with pytest.raises(ValueError, match=message):
        select_diversity(records, original, heldout, **kwargs)


@pytest.mark.skipif(
    not (ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").exists()
    or not (ROOT / "results/step15-generalization/run01/plan.json").exists(),
    reason="로컬 AI Hub 데이터 명세·이전 단계 결과가 없음 (저장소에 포함하지 않는 자료)",
)
def test_actual_manifest_expansion_is_disjoint_from_fixed_stage15_evaluation():
    manifest = json.loads((ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/step15-generalization/run01/plan.json").read_text(encoding="utf-8"))
    original = prior["models"]["B_reduce9600"]["training_data"]["sample_ids"]
    result = select_diversity(manifest["records"], original, prior["selection"]["cohorts"])
    assert result["source_counts"] == {
        "train": 1000, "val": 20, "test": 50, "excluded_recording": 383,
        "excluded_transcript_after_recording": 0, "eligible": 617,
        "original": 64, "addition_candidates": 553, "additions": 192,
    }
    assert result["training64"] == original
    assert result["heldout"] == prior["selection"]["cohorts"]
    assert result["quota_additions"] == {"001-050": 4, "051-100": 63, "101-150": 63, "151-200": 62}
    assert result["distributions"]["expanded256"]["length_bin"] == {
        "001-050": 28, "051-100": 76, "101-150": 76, "151-200": 76,
    }
    assert result["distributions"]["expanded256"]["recordings"] == 59
    assert result["distributions"]["eligible617"]["recordings"] == 59
    assert len(result["excluded_recording_ids"]) == 37
    assert len(result["expanded256"]) == len(set(result["expanded256"])) == 256
    assert len(result["distributions"]["expanded256"]["speaker"]) == 8
    assert all(value for key, value in result["disjointness"].items() if key != "test_selected")
