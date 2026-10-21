"""Stage15 metadata cohort exclusions, balancing, invariance and guard checks."""
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
from generalization_selection import length_bin, normalize_transcript, select_generalization


def row(sample_id, speaker="a", frames=75, split="train", recording=None, text=None):
    return {"id": sample_id, "speaker_id": speaker, "frames": frames, "split": split,
            "recording_id": recording or f"record-{sample_id}", "text": text or f"문장 {sample_id}"}


def fixture():
    training = [row("trained-a"), row("trained-b", speaker="b")]
    candidates = [row(f"candidate-{speaker}-{bucket}-{i}", speaker, bucket*50+25)
                  for speaker in ("a", "b") for bucket in range(4) for i in range(5)]
    return training + candidates + [row("validation", "v", split="val"), {"id": "test", "split": "test"}], [r["id"] for r in training]


def test_deterministic_under_record_and_training_id_reordering():
    records, training = fixture()
    original = copy.deepcopy(records)
    expected = select_generalization(records, training, count=12)
    random.Random(209).shuffle(records)
    assert select_generalization(records, list(reversed(training)), count=12) == expected
    assert sorted(records, key=lambda r: r["id"]) == sorted(original, key=lambda r: r["id"])
    assert len(expected["cohorts"]["train60"]) == 12
    assert expected["cohorts"]["val20"] == ["validation"]
    assert expected["disjointness"]["test_selected"] is False


def test_excludes_training_ids_recordings_and_normalized_exact_transcripts():
    records, training = fixture()
    records[0]["text"] = "한국 어"
    records += [row("same-record", recording="record-trained-a"),
                row("same-text", text=" \t" + unicodedata.normalize("NFD", "한국") + "\n어  ")]
    result = select_generalization(records, training, count=12)
    excluded = result["exclusions"]
    assert excluded["training_id"]["ids"] == sorted(training)
    assert excluded["training_recording_after_id"]["ids"] == ["same-record"]
    assert excluded["training_transcript_after_recording"]["ids"] == ["same-text"]
    assert not set(training + ["same-record", "same-text"]) & set(result["cohorts"]["train60"])
    assert result["source_counts"]["eligible"] == 40


def test_length_scarcity_redistributes_quotas_and_speaker_cells_balance():
    records, training = fixture()
    records = [r for r in records if not r["id"].startswith("candidate") or r["frames"] > 50]
    records += [row("only-short", frames=25)]
    result = select_generalization(records, training, count=12)
    assert result["quotas"] == {"001-050": 1, "051-100": 4, "101-150": 4, "151-200": 3}
    assert "only-short" in result["cohorts"]["train60"]
    distribution = result["distributions"]["train60"]["speaker_length_bin"]
    for bucket in ("051-100", "101-150", "151-200"):
        assert abs(distribution[f"a/{bucket}"] - distribution[f"b/{bucket}"]) <= 1


def test_performance_and_cache_fields_have_no_influence_and_are_not_opened():
    records, training = fixture()
    expected = select_generalization(records, training, count=12)
    for item in records:
        item.update(prediction="perfect", cer=-999, cache_path="does/not/exist", cache_sha256="invalid")
    assert select_generalization(records, training, count=12) == expected
    assert expected["cache_accessed"] is False


@pytest.mark.parametrize("change,message", [
    (lambda r, t: t.append(t[0]), "duplicate training"),
    (lambda r, t: t.append("missing"), "missing training"),
    (lambda r, t: t.append("validation"), "belong to train"),
    (lambda r, t: r.append(copy.deepcopy(r[0])), "duplicate record"),
    (lambda r, t: r[-2].update(speaker_id="a"), "validation speakers overlap"),
    (lambda r, t: r[-2].update(recording_id="record-trained-a"), "validation recordings overlap"),
    (lambda r, t: r[2].update(speaker_id="untrained"), "absent from training"),
    (lambda r, t: r[2].update(frames=0), "frames"),
    (lambda r, t: r[2].update(frames=201), "frames"),
    (lambda r, t: r[2].update(frames=True), "frames"),
    (lambda r, t: r[2].update(text=" \t"), "empty transcript"),
    (lambda r, t: r[2].update(split="other"), "unknown split"),
])
def test_invalid_metadata_fails_closed(change, message):
    records, training = fixture()
    change(records, training)
    with pytest.raises(ValueError, match=message):
        select_generalization(records, training, count=12)


def test_insufficient_eligible_count_fails():
    records, training = fixture()
    with pytest.raises(ValueError, match="insufficient"):
        select_generalization(records, training, count=41)


def test_validation_is_retained_even_when_reference_repeats_but_overlap_is_disclosed():
    records, training = fixture()
    records[-2]["text"] = records[0]["text"]
    result = select_generalization(records, training, count=12)
    assert result["cohorts"]["val20"] == ["validation"]
    assert result["val_training_transcript_overlap_ids"] == ["validation"]
    assert result["disjointness"]["val20_transcripts_vs_training"] is False


def test_transcript_normalization_retains_punctuation_and_case():
    assert normalize_transcript(" \t한\n국   ") == "한 국"
    assert normalize_transcript(unicodedata.normalize("NFD", "한국")) == "한국"
    assert normalize_transcript("A!") != normalize_transcript("a")


@pytest.mark.parametrize("value,expected", [(1, "001-050"), (50, "001-050"), (51, "051-100"),
                                             (100, "051-100"), (101, "101-150"), (150, "101-150"),
                                             (151, "151-200"), (200, "151-200")])
def test_length_boundaries(value, expected):
    assert length_bin(value) == expected


@pytest.mark.skipif(
    not (ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").exists()
    or not (ROOT / "results/step14-learning-rate/run01/plan.json").exists(),
    reason="로컬 AI Hub 데이터 명세·이전 단계 결과가 없음 (저장소에 포함하지 않는 자료)",
)
def test_actual_manifest_metadata_has_sixty_plus_twenty_without_training_overlap():
    manifest = json.loads((ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/step14-learning-rate/run01/plan.json").read_text(encoding="utf-8"))
    result = select_generalization(manifest["records"], prior["cohorts"]["expanded64"])
    assert result["source_counts"] == {"train": 1000, "val": 20, "test": 50, "eligible": 497}
    assert result["quotas"] == {"001-050": 5, "051-100": 19, "101-150": 18, "151-200": 18}
    assert len(result["cohorts"]["train60"]) == 60
    assert len(result["cohorts"]["val20"]) == 20
    assert len(result["distributions"]["train60"]["speaker"]) == 8
    assert len(result["distributions"]["val20"]["speaker"]) == 2
    assert all(value for key, value in result["disjointness"].items() if key != "test_selected")
