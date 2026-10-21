"""baseline 모델 단위 테스트.

설계문서 1.5절 — 입력/출력 형상, log-softmax, 체크포인트 저장/로드를 검증한다.
"""

from __future__ import annotations

import os
import sys

import torch
import pytest

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from models import NUM_CLASSES  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402


def test_compact_parameter_budget():
    model = LipNetBaseline()
    assert sum(p.numel() for p in model.parameters()) < 4_000_000
    assert model.lstm.input_size == 256


@pytest.mark.parametrize("length", [1, 17, 200])
def test_variable_time_axis_is_preserved(length):
    with torch.no_grad():
        output = LipNetBaseline().eval()(torch.randn(1, 3, length, 96, 96))
    assert output.shape == (length, 1, 69)
    assert torch.isfinite(output).all()


def test_padding_cannot_change_real_frame_predictions():
    """CNN 경계/정규화/역방향 LSTM도 다른 표본의 길이에 의존하지 않아야 한다."""
    torch.manual_seed(7)
    model = LipNetBaseline().eval()
    short = torch.randn(1, 3, 5, 96, 96)
    long = torch.randn(1, 3, 11, 96, 96)
    # 잘못된 값이 들어온 패딩도 lengths를 통해 마지막 유효 특징으로 덮는다.
    padded = torch.cat([short, torch.randn(1, 3, 6, 96, 96) * 100], dim=2)
    with torch.no_grad():
        alone = model(short)
        together = model(torch.cat([padded, long]), torch.tensor([5, 11]))
    torch.testing.assert_close(alone[:, 0], together[:5, 0], atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("lengths", [[0], [8], [2, 3], [2.5]])
def test_invalid_lengths_rejected(lengths):
    with pytest.raises(ValueError, match="lengths"):
        LipNetBaseline()(torch.randn(1, 3, 7, 96, 96), torch.tensor(lengths))


def test_old_architecture_with_current_tokens_is_rejected(tmp_path):
    from models import GRAPHEME_VOCAB
    from models.baseline import ModelCompatibilityError
    from models.tokenizer import tokenizer_metadata
    path = tmp_path / "old_structure.pt"
    torch.save({**tokenizer_metadata(), "vocab": GRAPHEME_VOCAB, "state_dict": {}}, path)
    with pytest.raises(ModelCompatibilityError, match="경량 모델"):
        LipNetBaseline.load_from_checkpoint(str(path))


def test_num_classes_is_69() -> None:
    """초성 19 + 중성 21 + 종성 27 + 공백 1 + blank 1 = 69."""
    assert NUM_CLASSES == 69


def test_forward_output_shape() -> None:
    """forward 출력이 (T, B, num_classes) = (75, B, 69)인지 검증한다."""
    model = LipNetBaseline().eval()
    x = torch.randn(2, 3, 75, 96, 96)
    with torch.no_grad():
        out = model(x)
    assert tuple(out.shape) == (75, 2, NUM_CLASSES)


def test_output_is_log_softmax() -> None:
    """출력이 log-softmax(클래스 합 확률 ≈ 1)인지 검증한다."""
    model = LipNetBaseline().eval()
    x = torch.randn(1, 3, 75, 96, 96)
    with torch.no_grad():
        out = model(x)
    probs = out.exp().sum(dim=-1)  # (T, B)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_save_and_load_checkpoint(tmp_path) -> None:
    """체크포인트 저장 후 로드 시 동일 출력을 내는지 검증한다."""
    model = LipNetBaseline().eval()
    path = str(tmp_path / "ckpt.pt")
    provenance = {"mode": "speaker_disjoint", "manifest_sha256": "test-manifest"}
    model.save_checkpoint(path, training_data=provenance)
    assert os.path.exists(path)

    loaded = LipNetBaseline.load_from_checkpoint(path)
    assert loaded.training_data == provenance
    x = torch.randn(1, 3, 75, 96, 96)
    with torch.no_grad():
        a = model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_legacy_checkpoint_rejected_before_model_creation(tmp_path):
    from models.tokenizer import TokenizerMismatchError
    path = tmp_path / "legacy.pt"
    torch.save({"num_classes": 41, "state_dict": {}}, path)
    with pytest.raises(TokenizerMismatchError):
        LipNetBaseline.load_from_checkpoint(str(path))


def test_wrong_token_order_checkpoint_rejected(tmp_path):
    from models.tokenizer import TokenizerMismatchError, tokenizer_metadata
    path = tmp_path / "wrong.pt"
    torch.save({**tokenizer_metadata(), "vocab": ["wrong"], "state_dict": {}}, path)
    with pytest.raises(TokenizerMismatchError):
        LipNetBaseline.load_from_checkpoint(str(path))
