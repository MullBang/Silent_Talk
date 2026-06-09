"""베이스라인 모델: 3D-CNN + LSTM + CTC.

입력 형상 MODEL_INPUT_SHAPE (B, C, T, H, W) = (B, 3, 75, 96, 96).
전처리 출력 (B, T, H, W, C)는 permute((0, 4, 1, 2, 3))로 변환 후 전달한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LipReadingBaseline(nn.Module):
    """3D-CNN 시공간 특징 추출 + 양방향 LSTM + CTC 분류 헤드."""

    def __init__(self, num_classes: int) -> None:
        """모델 레이어를 구성한다.

        Args:
            num_classes: CTC 출력 클래스 수 (자모 + blank).
        """
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: MODEL_INPUT_SHAPE (B, 3, 75, 96, 96) 입력 텐서.

        Returns:
            (T, B, num_classes) CTC log-softmax 로그 확률.
        """
        pass


def load_baseline(weights_path: str, num_classes: int) -> LipReadingBaseline:
    """가중치를 로드한 베이스라인 모델을 반환한다.

    Args:
        weights_path: .pt/.pth 가중치 파일 경로.
        num_classes: CTC 출력 클래스 수.

    Returns:
        eval 모드로 설정된 모델 인스턴스.
    """
    pass
