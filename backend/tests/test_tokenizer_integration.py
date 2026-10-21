"""학습/서비스 경로가 구형 자료를 조용히 사용하지 않는지 확인한다."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
import train
from api import infer
from models.baseline import LipNetBaseline, ModelCompatibilityError
from models.tokenizer import TokenizerMismatchError, encode_text, tokenizer_metadata
from services import ws_handler


def test_training_rejects_legacy_before_filtering_or_model_creation(tmp_path, monkeypatch):
    np.savez_compressed(tmp_path / "old.npz", frames=np.zeros((1, 96, 96, 3), dtype=np.uint8),
                        label=np.array([1]), text="가")
    monkeypatch.setattr(train, "LipNetBaseline", lambda: pytest.fail("구형 캐시를 먼저 거부해야 함"))
    with pytest.raises(TokenizerMismatchError):
        train.train(SimpleNamespace(device="cpu", smoke_data=str(tmp_path), max_frames=1))


def test_training_collate_preserves_new_labels_and_true_lengths(tmp_path):
    path = tmp_path / "v2.npz"
    np.savez_compressed(path, frames=np.zeros((30, 96, 96, 3), dtype=np.uint8),
                        label=np.asarray(encode_text("값이 없다")), text="값이 없다",
                        source_text="값이 없다.", **tokenizer_metadata())
    item = train.ClipDataset([str(path)])[0]
    tensor, labels, in_lengths, target_lengths = train.collate([item])
    assert tuple(tensor.shape) == (1, 3, 30, 96, 96)
    assert labels.tolist() == encode_text("값이 없다")
    assert in_lengths.tolist() == [30]
    assert target_lengths.tolist() == [len(encode_text("값이 없다"))]


@pytest.mark.parametrize("error_type", [TokenizerMismatchError, ModelCompatibilityError])
def test_inference_does_not_hide_or_cache_tokenizer_mismatch(tmp_path, monkeypatch, error_type):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(infer, "MODEL_WEIGHTS_PATH", str(path))
    monkeypatch.setattr(infer, "_model_loaded", False)
    monkeypatch.setattr(infer, "_model", None)
    def mismatch(*args, **kwargs):
        raise error_type("구형 사전 또는 구조")
    monkeypatch.setattr(LipNetBaseline, "load_from_checkpoint", mismatch)
    for _ in range(2):
        with pytest.raises(error_type):
            infer._get_model()
    assert not infer._model_loaded


def test_webcam_without_v2_weights_requests_training(tmp_path, monkeypatch):
    monkeypatch.setattr(ws_handler, "_model", None)
    monkeypatch.setattr(ws_handler, "MODEL_WEIGHTS_PATH", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="새 토큰 사전"):
        ws_handler._get_model()
