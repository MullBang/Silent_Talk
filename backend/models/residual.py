"""학습 비교용 3D residual CNN + BiLSTM + CTC 후보.

RGB 96×96, 시간축 T, 토큰/정규화 계약은 compact 모델과 같다. 각 CNN
stage의 합성곱 두 개와 shortcut이 특징/기울기 전달을 돕도록 설계했다.
공간 크기는 96→24→12→6→4이고 projection 활성화는 GELU이다.
사전학습 가중치를 사용하지 않으며 성능 향상은 별도 실험으로 판단한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import GRAPHEME_VOCAB, NUM_CLASSES
from .baseline import ModelCompatibilityError
from .tokenizer import TokenizerMismatchError, tokenizer_metadata, validate_tokenizer_metadata

ARCHITECTURE_VERSION = "lipnet-residual-v1"


def _replicate_valid_end(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """패딩된 시점의 값을 마지막 유효 프레임/특징으로 교체한다."""
    if lengths is None:
        return x
    indices = torch.arange(x.shape[2], device=x.device)[None, :]
    indices = torch.minimum(indices, lengths.to(x.device)[:, None] - 1)
    return x.gather(2, indices[:, None, :, None, None].expand_as(x))


class FramewiseGroupNorm(nn.Module):
    """배치·시간·패딩을 섞지 않고 각 프레임의 공간/채널만 정규화한다."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W)에 프레임별 GroupNorm을 적용한다."""
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frames = self.norm(frames)
        return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResidualSpatialTemporalBlock(nn.Module):
    """시간 해상도를 유지하는 두 합성곱과 projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, first: bool = False) -> None:
        super().__init__()
        spatial_kernel = 5 if first else 3
        spatial_stride = 2 if first else 1
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, (3, spatial_kernel, spatial_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2), bias=False,
        )
        self.norm1 = FramewiseGroupNorm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=(0, 1, 1), bias=False)
        self.norm2 = FramewiseGroupNorm(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1,
                      stride=(1, spatial_stride, spatial_stride), bias=False),
            FramewiseGroupNorm(out_channels),
        )
        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """각 시간 합성곱의 경계에서 마지막 실제 특징을 복제한다."""
        x = _replicate_valid_end(x, lengths)
        residual = self.shortcut(x)
        features = self.conv1(F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate"))
        features = F.gelu(self.norm1(features))
        # 첫 합성곱이 만든 패딩 구간도 두 번째 합성곱에 섞이지 않게 한다.
        features = _replicate_valid_end(features, lengths)
        features = self.conv2(F.pad(features, (0, 0, 0, 0, 1, 1), mode="replicate"))
        features = self.norm2(features)
        return self.pool(F.gelu(features + residual))


class LipNetResidual(nn.Module):
    """새 구조 버전으로 분리한 residual CNN과 기존 크기의 BiLSTM."""

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 256) -> None:
        super().__init__()
        if num_classes != NUM_CLASSES:
            raise TokenizerMismatchError(f"현재 토큰 사전은 {NUM_CLASSES}클래스입니다.")
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.conv = nn.ModuleList([
            ResidualSpatialTemporalBlock(3, 32, first=True),
            ResidualSpatialTemporalBlock(32, 64),
            ResidualSpatialTemporalBlock(64, 96),
        ])
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 4, 4))
        self.projection = nn.Sequential(
            nn.Linear(96 * 4 * 4, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(256, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """(B,3,T,96,96) → (T,B,69) FP32 로그 확률, lengths는 실제 길이."""
        if (x.ndim != 5 or x.shape[0] < 1 or x.shape[1] != 3
                or x.shape[2] < 1 or x.shape[3:] != (96, 96)):
            raise ValueError("모델 입력은 (B,3,T,96,96), B>=1, T>=1이어야 합니다.")
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device="cpu")
            if (lengths.shape != (x.shape[0],) or lengths.dtype not in (torch.int32, torch.int64)
                    or (lengths < 1).any() or (lengths > x.shape[2]).any()):
                raise ValueError("lengths는 배치별 실제 프레임 수(1<=T<=입력 길이)여야 합니다.")
        features = x
        for block in self.conv:
            features = block(features, lengths)
        features = self.spatial_pool(features)
        b, c, t, h, w = features.shape
        features = self.projection(features.permute(0, 2, 1, 3, 4).reshape(b, t, c * h * w))
        if lengths is None:
            sequence, _ = self.lstm(features)
        else:
            packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
            sequence, _ = self.lstm(packed)
            sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=t)
        return F.log_softmax(self.fc(sequence).float(), dim=-1).permute(1, 0, 2).contiguous()

    def save_checkpoint(self, path: str, training_data: dict | None = None,
                        training_run: dict | None = None) -> None:
        """구조/토큰/출처 포함 추론용 가중치. optimizer/RNG 등 재개 상태는 없다."""
        try:
            torch.save({
                "architecture_version": ARCHITECTURE_VERSION,
                "state_dict": self.state_dict(), "num_classes": self.num_classes,
                "lstm_hidden": self.lstm_hidden, **tokenizer_metadata(),
                "vocab": GRAPHEME_VOCAB, "training_data": training_data,
                "training_run": training_run,
            }, path)
        except OSError as exc:
            raise OSError(f"체크포인트 저장 실패: {path} ({exc})") from exc

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "LipNetResidual":
        """동일 토큰·구조의 가중치만 CPU에 읽고 평가 모드로 반환한다."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except OSError as exc:
            raise OSError(f"체크포인트 로드 실패: {path} ({exc})") from exc
        validate_tokenizer_metadata(checkpoint, path)
        if checkpoint.get("vocab") != GRAPHEME_VOCAB:
            raise TokenizerMismatchError(f"{path}: 체크포인트 토큰 순서가 일치하지 않습니다.")
        if checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ModelCompatibilityError(
                f"{path}: {ARCHITECTURE_VERSION} 구조의 가중치가 필요합니다. "
                "compact 모델 등 다른 구조의 가중치는 residual 모델과 호환되지 않습니다."
            )
        model = cls(num_classes=checkpoint["num_classes"], lstm_hidden=checkpoint["lstm_hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.training_data = checkpoint.get("training_data")
        model.training_run = checkpoint.get("training_run")
        return model.eval()
