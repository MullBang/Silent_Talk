"""baseline 모델 단위 테스트.

설계문서 1.5절 — 입력/출력 형상, log-softmax, 체크포인트 저장/로드를 검증한다.
"""

from __future__ import annotations

import os
import sys

import torch

# backend 패키지 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from models import NUM_CLASSES  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402


def test_num_classes_is_41() -> None:
    """자음 19 + 모음 21 + blank 1 = 41."""
    assert NUM_CLASSES == 41


def test_forward_output_shape() -> None:
    """forward 출력이 (T, B, num_classes) = (75, B, 41)인지 검증한다."""
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
    model.save_checkpoint(path)
    assert os.path.exists(path)

    loaded = LipNetBaseline.load_from_checkpoint(path)
    x = torch.randn(1, 3, 75, 96, 96)
    with torch.no_grad():
        a = model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-5)
