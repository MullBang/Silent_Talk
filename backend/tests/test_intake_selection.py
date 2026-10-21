"""Stage18 pilot sampling guards; these tests never open videos or train a model."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import sys
import unicodedata

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from intake_selection import PACKAGES, select_intake
from models.dataset_manifest import parse_identity
from models.tokenizer import ctc_min_input_len, encode_text, normalize_text


def rank(kind, identity, seed=42):
    # Deliberately independent of the implementation's private ranking helper.
    return hashlib.sha256(f"18:{seed}:{kind}:{identity}".encode("utf-8")).hexdigest(), identity


def row(speaker=100, recording=1, sentence=1, package="TS2", text="새로운 문장", angle="A"):
    sample_id = f"lip_J_1_F_02_C{speaker:03d}_{angle}_{recording:03d}_{sentence:03d}"
    labels = encode_text(text)
    return {"id": sample_id, **parse_identity(sample_id), "package": package,
            "source_partition": "Training", "text": text, "source_text": text,
            "expected_frames": 80, "ctc_min_frames": ctc_min_input_len(labels),
            "opaque_source_metadata": {"filename": "do-not-open.mp4", "nested": [1, 2]}}


def set_text(record, text):
    record["text"] = record["source_text"] = text
    record["ctc_min_frames"] = ctc_min_input_len(encode_text(text))


@pytest.fixture
def source():
    candidates = []
    for p, package in enumerate(PACKAGES):
        for s in range(3):
            for video in range(1, 4):
                for clip in range(1, 7):
                    text = chr(0xAC00 + len(candidates)) + " 문장"
                    candidates.append(row(100 + p * 10 + s, video, clip, package, text))
    old = [row(900 + i, text=text) for i, text in enumerate(("기존 학습", "기존 개발", "기존 시험"))]
    for item, split in zip(old, ("train", "val", "test")):
        item["split"] = split
    return candidates, old, []


def call(source, **kwargs):
    return select_intake(*source, **kwargs)


def test_fixed_hash_split_and_quotas_are_independently_recomputed(source):
    result = call(source)
    assert result["counts"]["selected"] == 72
    assert result["counts"]["train"] == {"clips": 48, "speakers": 6, "recordings": 12, "unique_texts": 48}
    assert result["counts"]["val"] == {"clips": 24, "speakers": 3, "recordings": 6, "unique_texts": 24}
    assert result["counts"]["test"] == {"clips": 0, "speakers": 0, "recordings": 0, "unique_texts": 0}
    expected_ids = set()
    for package in PACKAGES:
        speakers = sorted({r["speaker_id"] for r in source[0] if r["package"] == package}, key=lambda x: rank("speaker", x))
        assert result["policy"]["dev_speakers_by_package"][package] == speakers[0]
        for speaker in speakers:
            assert result["speaker_mapping"][speaker] == ("val" if speaker == speakers[0] else "train")
            videos = sorted({r["recording_id"] for r in source[0] if r["speaker_id"] == speaker}, key=lambda x: rank("recording", x))[:2]
            assert result["policy"]["selected_recordings"][speaker] == videos
            for video in videos:
                ids = sorted((r["id"] for r in source[0] if r["recording_id"] == video), key=lambda x: rank("clip", x))[:4]
                expected_ids.update(ids)
    assert {r["id"] for r in result["records"]} == expected_ids
    train = [r for r in result["records"] if r["split"] == "train"]
    val = [r for r in result["records"] if r["split"] == "val"]
    for field in ("id", "speaker_id", "recording_id", "utterance_id", "text"):
        assert not {r[field] for r in train} & {r[field] for r in val}
    assert result["policy"]["training_performed"] is False
    assert result["policy"]["metadata_only"] is True
    assert result["policy"]["performance_used"] is False
    assert result["policy"]["test_access"] is False
    assert result["policy"]["generic_manifest_compatible"] is False


def test_input_order_and_rng_cannot_change_selection(source):
    expected = call(source)
    shuffled = copy.deepcopy(source)
    rng = random.Random(127)
    rng.shuffle(shuffled[0])
    rng.shuffle(shuffled[1])
    state = random.getstate()
    assert call(shuffled) == expected
    assert random.getstate() == state
    alternate = call(source, seed=43)
    assert alternate == call(shuffled, seed=43)
    assert {r["id"] for r in alternate["records"]} != {r["id"] for r in expected["records"]}


def test_output_preserves_every_source_field_and_deep_copies_nested_metadata(source):
    before = copy.deepcopy(source)
    result = call(source)
    assert source == before
    original = {r["id"]: r for r in source[0]}
    for item in result["records"]:
        assert set(item) == set(original[item["id"]]) | {"split"}
        assert {k: v for k, v in item.items() if k != "split"} == original[item["id"]]
    result["records"][0]["opaque_source_metadata"]["nested"].append(999)
    assert source == before


def test_nondefault_quotas_remain_fixed_per_speaker(source):
    result = call(source, videos_per_speaker=1, clips_per_video=2)
    assert result["counts"]["selected"] == 9 * 1 * 2
    assert result["counts"]["train"]["clips"] == 12
    assert result["counts"]["val"]["clips"] == 6


def test_all_old_train_val_test_speakers_are_excluded_and_auditable(source):
    extras = [row(900 + i, recording=2, text="기존 화자의 새 문장") for i in range(3)]
    source[0].extend(extras)
    result = call(source)
    assert result["excluded_summary"]["old_speaker"] == {"count": 3, "ids": sorted(r["id"] for r in extras)}
    assert result["policy"]["excluded_old_speakers"] == ["C900", "C901", "C902"]
    assert not {r["speaker_id"] for r in result["records"]} & {r["speaker_id"] for r in source[1]}
    assert result["counts"]["excluded"] == 3


def test_protected_recording_is_excluded_with_all_sibling_utterances(source):
    protected = source[0][0]["recording_id"]
    source[2].append(protected)
    result = call(source)
    ids = sorted(r["id"] for r in source[0] if r["recording_id"] == protected)
    assert result["excluded_summary"]["protected_recording"] == {"count": 6, "ids": ids}
    assert protected not in {r["recording_id"] for r in result["records"]}
    assert result["counts"]["selected"] == 72


def test_protected_recording_order_does_not_change_result(source):
    # These unrelated recording IDs have no matching candidate and still must be preserved.
    source[2].extend([r["recording_id"] for r in source[1]])
    expected = call(source)
    source[2].reverse()
    assert call(source) == expected


def test_old_transcript_exclusion_uses_tokenizer_normalization_for_both_roles(source):
    baseline = call(source)
    for split, old in zip(("train", "val"), source[1]):
        chosen = next(r for r in baseline["records"] if r["split"] == split)
        target = next(r for r in source[0] if r["id"] == chosen["id"])
        set_text(target, " \t" + unicodedata.normalize("NFD", old["text"].replace(" ", ",\n")) + "! ")
    result = call(source)
    rejected = result["excluded_summary"]["old_transcript"]
    assert rejected["count"] == 2
    assert not {normalize_text(r["text"]) for r in result["records"]} & {normalize_text(r["text"]) for r in source[1]}
    assert result["counts"]["selected"] == 72


def test_train_cannot_reuse_selected_new_dev_transcript(source):
    baseline = call(source)
    dev = next(r for r in baseline["records"] if r["split"] == "val")
    train = next(r for r in baseline["records"] if r["split"] == "train")
    changed = next(r for r in source[0] if r["id"] == train["id"])
    set_text(changed, "  " + dev["text"].replace(" ", "! ") + " ")
    result = call(source)
    assert result["excluded_summary"]["selected_val_transcript"] == {"count": 1, "ids": [train["id"]]}
    assert dev["id"] in {r["id"] for r in result["records"]}
    assert train["id"] not in {r["id"] for r in result["records"]}
    assert result["speaker_mapping"] == baseline["speaker_mapping"]
    assert result["policy"]["selected_recordings"] == baseline["policy"]["selected_recordings"]


@pytest.mark.parametrize("split", ["train", "val"])
def test_same_split_duplicate_texts_are_skipped_without_relaxing_quota(source, split):
    baseline = call(source)
    selected = next(r for r in baseline["records"] if r["split"] == split)
    same_video = sorted((r for r in source[0] if r["recording_id"] == selected["recording_id"]), key=lambda r: rank("clip", r["id"]))
    set_text(same_video[1], same_video[0]["text"])
    result = call(source)
    assert same_video[1]["id"] in result["excluded_summary"][f"duplicate_selected_{split}_transcript"]["ids"]
    rows = [r for r in result["records"] if r["split"] == split]
    assert len(rows) == len({normalize_text(r["text"]) for r in rows})
    assert result["counts"]["selected"] == 72


def test_exclusion_and_unselected_accounting_forms_exact_partition(source):
    protected = source[0][0]["recording_id"]
    source[2].append(protected)
    extra = row(900, recording=2)
    source[0].append(extra)
    result = call(source)
    groups = [{r["id"] for r in result["records"]}]
    for details in result["excluded_summary"].values():
        assert details["count"] == len(details["ids"])
        assert details["ids"] == sorted(set(details["ids"]))
        groups.append(set(details["ids"]))
    for i, group in enumerate(groups):
        assert not group & set().union(*groups[:i]) if i else True
    assert set().union(*groups) == {r["id"] for r in source[0]}
    counts = result["counts"]
    assert counts["candidates"] == counts["selected"] + counts["excluded"] + counts["unselected"]


def test_too_few_recordings_fails(source):
    speaker = source[0][0]["speaker_id"]
    source[0][:] = [r for r in source[0] if r["speaker_id"] != speaker or r["id"].endswith(tuple(f"_001_{n:03d}" for n in range(1, 7)))]
    with pytest.raises(ValueError, match="insufficient recordings"):
        call(source)


def test_chosen_short_video_fails_instead_of_substituting_third_video(source):
    baseline = call(source)
    video = next(iter(baseline["policy"]["selected_recordings"].values()))[0]
    retained_ids = {r["id"] for r in source[0] if r["recording_id"] == video}
    retained_ids = set(sorted(retained_ids)[:3])
    source[0][:] = [r for r in source[0] if r["recording_id"] != video or r["id"] in retained_ids]
    with pytest.raises(ValueError, match="insufficient unique eligible clips.*no fallback"):
        call(source)


def test_all_old_texts_in_fixed_video_fail_without_video_or_role_fallback(source):
    baseline = call(source)
    video = next(iter(baseline["policy"]["selected_recordings"].values()))[0]
    for item in source[0]:
        if item["recording_id"] == video:
            set_text(item, source[1][2]["text"])
    with pytest.raises(ValueError, match="insufficient unique eligible clips.*no fallback"):
        call(source)


def test_cross_split_transcript_shortage_fails_without_reassigning_speaker(source):
    baseline = call(source)
    train = next(r for r in baseline["records"] if r["split"] == "train")
    dev = next(r for r in baseline["records"] if r["split"] == "val")
    for item in source[0]:
        if item["recording_id"] == train["recording_id"]:
            set_text(item, dev["text"])
    with pytest.raises(ValueError, match="train/.+insufficient unique eligible clips.*no fallback"):
        call(source)


@pytest.mark.parametrize("name,value", [("seed", -1), ("seed", True), ("seed", 42.0),
    ("videos_per_speaker", 0), ("videos_per_speaker", False), ("videos_per_speaker", 1.0),
    ("clips_per_video", -1), ("clips_per_video", True), ("clips_per_video", 4.0)])
def test_bad_sampling_arguments_fail(source, name, value):
    with pytest.raises(ValueError, match=name):
        call(source, **{name: value})


@pytest.mark.parametrize("which,value,match", [(0, [], "candidates"), (0, {}, "candidates"),
    (1, [], "old_records"), (1, {}, "old_records"), (2, "record", "contain strings"),
    (2, [None], "contain strings"), (2, [""], "contain strings"), (2, ["duplicate", "duplicate"], "duplicate protected")])
def test_bad_collection_arguments_fail(source, which, value, match):
    args = list(source)
    args[which] = value
    with pytest.raises(ValueError, match=match):
        select_intake(*args)


def test_duplicate_candidate_and_old_ids_fail(source):
    duplicate = copy.deepcopy(source[0][0])
    source[0].append(duplicate)
    with pytest.raises(ValueError, match="duplicate candidate ID"):
        call(source)
    source[0].pop()
    source[1].append(copy.deepcopy(source[1][0]))
    with pytest.raises(ValueError, match="duplicate old record ID"):
        call(source)


def test_alternate_filename_padding_cannot_duplicate_an_utterance(source):
    alias = copy.deepcopy(source[0][0])
    alias["id"] = alias["id"].replace("_A_001_001", "_A_1_1")
    alias.update(parse_identity(alias["id"]))
    source[0].append(alias)
    with pytest.raises(ValueError, match="duplicate candidate utterance"):
        call(source)


@pytest.mark.parametrize("field,forgery", [("speaker_id", "C999"), ("recording_id", "fake-recording"),
    ("utterance_id", "fake:001"), ("angle", "B")])
def test_filename_identity_cannot_be_forged(source, field, forgery):
    source[0][0][field] = forgery
    with pytest.raises(ValueError, match=f"filename {field} mismatch"):
        call(source)


@pytest.mark.parametrize("forgery", ["bad-id", "../lip_J_1_F_02_C100_A_001_001", "C:\\unsafe", "lip_J_1_F_02_C100_A_001_001.mp4", "lip_J_1_F_02_C100_A_001"])
def test_malformed_path_extension_or_recording_only_id_fails(source, forgery):
    source[0][0]["id"] = forgery
    with pytest.raises(ValueError):
        call(source)


def test_excluded_old_speaker_cannot_hide_invalid_identity_metadata(source):
    item = row(900, recording=2)
    item["utterance_id"] = "forged"
    source[0].append(item)
    with pytest.raises(ValueError, match="filename utterance_id mismatch"):
        call(source)


@pytest.mark.parametrize("field,value,match", [("source_partition", "Validation", "official Training"),
    ("source_partition", "training", "official Training"), ("package", "TS201", "unsupported source package"),
    ("split", "train", "already has a split"), ("expected_frames", 0, "expected_frames"),
    ("expected_frames", 201, "expected_frames"), ("expected_frames", True, "expected_frames"),
    ("expected_frames", 80.0, "expected_frames"), ("ctc_min_frames", 0, "ctc_min_frames"),
    ("ctc_min_frames", True, "ctc_min_frames"), ("ctc_min_frames", 3.0, "ctc_min_frames"),
    ("ctc_min_frames", 1, "tokenizer mismatch"), ("text", " ", "empty normalized"),
    ("text", None, "transcript must be a string"), ("text", "한글 123", "지원하지 않는"),
    ("source_text", "다른 정답", "source_text/text mismatch"), ("label", [1, 2, 3], "label/tokenizer mismatch")])
def test_candidate_contract_rejects_bad_metadata(source, field, value, match):
    source[0][0][field] = value
    with pytest.raises(ValueError, match=match):
        call(source)


def test_valid_side_camera_identity_is_still_rejected(source):
    item = source[0][0]
    item["id"] = item["id"].replace("_A_", "_B_")
    item.update(parse_identity(item["id"]))
    with pytest.raises(ValueError, match="frontal angle A"):
        call(source)


def test_one_speaker_cannot_have_multiple_package_roles(source):
    source[0][0]["package"] = "TS20"
    with pytest.raises(ValueError, match="multiple packages"):
        call(source)


def test_ctc_minimum_must_fit_temporal_frames(source):
    source[0][0]["expected_frames"] = source[0][0]["ctc_min_frames"] - 1
    with pytest.raises(ValueError, match="CTC target cannot fit"):
        call(source)


def test_boundary_two_hundred_frames_and_exact_ctc_minimum_are_valid(source):
    source[0][0]["expected_frames"] = 200
    source[0][1]["expected_frames"] = source[0][1]["ctc_min_frames"]
    source[0][2]["label"] = encode_text(source[0][2]["text"])
    assert call(source)["counts"]["selected"] == 72


@pytest.mark.parametrize("remove_package", [True, False])
def test_every_required_package_needs_two_new_speakers(source, remove_package):
    if remove_package:
        source[0][:] = [r for r in source[0] if r["package"] != "TS200"]
    else:
        source[0][:] = [r for r in source[0] if r["package"] != "TS200" or r["speaker_id"] == "C120"]
    with pytest.raises(ValueError, match="TS200: at least2 new eligible speakers"):
        call(source)


def test_existing_speaker_exclusion_can_make_package_ineligible(source):
    source[1].extend([copy.deepcopy(next(r for r in source[0] if r["speaker_id"] == speaker)) for speaker in ("C120", "C121")])
    with pytest.raises(ValueError, match="TS200: at least2 new eligible speakers"):
        call(source)


def test_model_performance_fields_cannot_change_selected_ids(source):
    expected = {r["id"] for r in call(source)["records"]}
    for i, record in enumerate(source[0]):
        record.update(cer=i * -100, predictions="irrelevant", source_video_path="nonexistent.mp4")
    assert {r["id"] for r in call(source)["records"]} == expected
