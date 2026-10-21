"""Hand-specified prediction/metric regressions; no neural execution."""
import pytest

from evaluation_contract import corpus_scores, decode_prediction, score_prediction


@pytest.mark.parametrize("frames,collapsed,text", [
    ([1,1,20,20], [1,20], "가"),
    ([1,0,1,20], [1,1,20], "�가"),
    ([1,20,0,1,20], [1,20,1,20], "가가"),
    ([20,41,1], [20,41,1], "���"),
    ([0,0], [], ""),
    ([68,1,20,68,68,0,68,1,20,68], [68,1,20,68,68,1,20,68], " 가  가 "),
])
def test_explicit_ctc_and_invalid_jamo(frames, collapsed, text):
    result = decode_prediction(frames)
    assert result["raw_argmax_ids"] == frames
    assert result["collapsed_ids"] == collapsed
    assert result["raw_text"] == text
    assert result["metric_text"] == text.strip()


@pytest.mark.parametrize("ids", [[-1], [69], [True], [1.0]])
def test_out_of_vocabulary_or_noninteger_ids_rejected(ids):
    with pytest.raises(ValueError):
        decode_prediction(ids)


def test_corpus_weighting_and_failed_empty_prediction():
    rows = [score_prediction("가", "", failed=True), score_prediction("가가가", "가가가")]
    result = corpus_scores(rows)
    assert result["CER"] == .25 and result["WER"] == .5
    assert result["exact_count"] == 1 and result["failure_count"] == 1


def test_nfc_internal_spaces_errors_and_repetition_kept():
    assert score_prediction("가", "가")["exact"]
    assert score_prediction("가 가", "가  가")["character_edits"] == 1
    assert score_prediction("가", "�")["character_edits"] == 1
    assert score_prediction("가", "아아아아아")["repetition_ge5"]
    assert not score_prediction("가", "아 아 아 아 아")["repetition_ge5"]


def test_failure_cannot_hide_nonempty_prediction():
    with pytest.raises(ValueError):
        score_prediction("가", "나", failed=True)
