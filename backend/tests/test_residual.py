"""residual 후보의 시간·패딩·수치 정밀도·체크포인트 계약을 CPU로 검증한다."""
from __future__ import annotations

import os
import sys

import pytest
import torch

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from models import GRAPHEME_VOCAB, NUM_CLASSES  # noqa: E402
from models.baseline import LipNetBaseline, ModelCompatibilityError  # noqa: E402
from models.residual import ARCHITECTURE_VERSION, LipNetResidual  # noqa: E402
from models.tokenizer import TokenizerMismatchError, tokenizer_metadata  # noqa: E402


def test_parameter_budget_and_sequence_contract() -> None:
    model = LipNetResidual()
    assert model.architecture_version == ARCHITECTURE_VERSION == "lipnet-residual-v1"
    assert sum(parameter.numel() for parameter in model.parameters()) < 8_000_000
    assert model.lstm.input_size == 256
    assert model.lstm.hidden_size == 256
    assert model.lstm.num_layers == 2
    assert model.lstm.bidirectional
    assert isinstance(model.projection[2], torch.nn.GELU)


@pytest.mark.parametrize("length", [1, 9, 200])
def test_variable_time_axis_and_normalized_finite_output(length: int) -> None:
    with torch.inference_mode():
        output = LipNetResidual().eval()(torch.randn(1, 3, length, 96, 96))
    assert output.shape == (length, 1, NUM_CLASSES)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.exp().sum(-1), torch.ones(length, 1), atol=1e-5, rtol=1e-5)


def test_padding_values_and_other_samples_cannot_change_valid_predictions() -> None:
    """NaN/Inf 패딩도 차단하며 CNN 두 층과 역방향 LSTM 경계까지 검사한다."""
    torch.manual_seed(19)
    model = LipNetResidual().eval()
    short = torch.randn(1, 3, 3, 96, 96)
    padding = torch.full((1, 3, 4, 96, 96), float("nan"))
    padding[:, :, 2:] = float("inf")
    padded = torch.cat([short, padding], dim=2)
    long = torch.randn(1, 3, 7, 96, 96) * 3
    with torch.inference_mode():
        alone = model(short)
        together = model(torch.cat([padded, long]), torch.tensor([3, 7]))
    assert torch.isfinite(together).all()
    torch.testing.assert_close(alone[:, 0], together[:3, 0], atol=2e-5, rtol=1e-5)


def test_ctc_backward_is_finite_and_padding_has_no_gradient() -> None:
    """CTC 학습 경로와 각 residual 분기의 유한 기울기를 확인한다."""
    torch.manual_seed(23)
    model = LipNetResidual().train()
    inputs = torch.randn(2, 3, 7, 96, 96, requires_grad=True)
    lengths = torch.tensor([4, 7])
    outputs = model(inputs, lengths)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        outputs, torch.tensor([1, 2, 3, 4]), lengths, torch.tensor([2, 2]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad[0, :, 4:]) == 0
    for block in model.conv:
        assert torch.count_nonzero(block.conv1.weight.grad) > 0
        assert torch.count_nonzero(block.conv2.weight.grad) > 0
        assert torch.count_nonzero(block.shortcut[0].weight.grad) > 0


def test_cpu_autocast_retains_fp32_finite_log_probabilities() -> None:
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = LipNetResidual().eval()(torch.randn(1, 3, 4, 96, 96), torch.tensor([4]))
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.exp().sum(-1), torch.ones(4, 1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("lengths", [[0], [8], [2, 3], [2.5], [True]])
def test_invalid_lengths_are_rejected(lengths: list) -> None:
    with pytest.raises(ValueError, match="lengths"):
        LipNetResidual()(torch.randn(1, 3, 7, 96, 96), torch.tensor(lengths))


@pytest.mark.parametrize("shape", [(1, 1, 3, 96, 96), (1, 3, 3, 48, 48), (1, 3, 0, 96, 96)])
def test_invalid_input_shape_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="모델 입력"):
        LipNetResidual()(torch.empty(shape))


def test_checkpoint_round_trip_preserves_predictions_and_provenance(tmp_path) -> None:
    model = LipNetResidual().eval()
    path = str(tmp_path / "residual.pt")
    provenance = {"mode": "speaker_disjoint", "manifest_sha256": "example"}
    run = {"epoch": 3, "optimizer_steps": 18, "initialization": "random"}
    model.save_checkpoint(path, training_data=provenance, training_run=run)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    assert stored["architecture_version"] == ARCHITECTURE_VERSION
    assert not {"optimizer", "optimizer_state_dict", "scaler", "rng_state"}.intersection(stored)
    loaded = LipNetResidual.load_from_checkpoint(path)
    assert loaded.training is False
    assert loaded.training_data == provenance
    assert loaded.training_run == run
    inputs = torch.randn(1, 3, 5, 96, 96)
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), loaded(inputs), atol=0, rtol=0)


def test_compact_and_residual_checkpoints_cannot_be_cross_loaded(tmp_path) -> None:
    compact_path = str(tmp_path / "compact.pt")
    residual_path = str(tmp_path / "residual.pt")
    LipNetBaseline().save_checkpoint(compact_path)
    LipNetResidual().save_checkpoint(residual_path)
    with pytest.raises(ModelCompatibilityError, match="lipnet-residual-v1"):
        LipNetResidual.load_from_checkpoint(compact_path)
    with pytest.raises(ModelCompatibilityError, match="lipnet-compact-v1"):
        LipNetBaseline.load_from_checkpoint(residual_path)


@pytest.mark.parametrize("architecture", [None, "unrecognized-v9"])
def test_missing_or_unknown_architecture_is_rejected_before_state_loading(tmp_path, architecture) -> None:
    path = str(tmp_path / "unknown.pt")
    torch.save({**tokenizer_metadata(), "vocab": GRAPHEME_VOCAB,
                "architecture_version": architecture, "state_dict": {}}, path)
    with pytest.raises(ModelCompatibilityError, match="lipnet-residual-v1"):
        LipNetResidual.load_from_checkpoint(path)


def test_wrong_token_order_is_rejected(tmp_path) -> None:
    path = str(tmp_path / "wrong_tokens.pt")
    torch.save({**tokenizer_metadata(), "vocab": ["wrong"],
                "architecture_version": ARCHITECTURE_VERSION, "state_dict": {}}, path)
    with pytest.raises(TokenizerMismatchError):
        LipNetResidual.load_from_checkpoint(path)


def test_wrong_class_count_is_rejected() -> None:
    with pytest.raises(TokenizerMismatchError):
        LipNetResidual(num_classes=41)
