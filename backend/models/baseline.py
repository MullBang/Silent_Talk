"""경량 3D-CNN + BiLSTM + CTC. RGB 96×96, 가변 T, 출력 T를 유지한다.

공간 크기 96→48→24→12→6→4, 프레임 특징 1536→256, BiLSTM 2층.
시간축 풀링은 하지 않는다. 학습 배치의 복제 패딩은 CNN 경계와 LSTM에서 제외한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import GRAPHEME_VOCAB, NUM_CLASSES
from .tokenizer import TokenizerMismatchError, tokenizer_metadata, validate_tokenizer_metadata

ARCHITECTURE_VERSION = "lipnet-compact-v1"


class ModelCompatibilityError(ValueError):
    """토큰 사전은 같아도 모델 구조가 다른 가중치를 사용할 수 없다."""


class SpatialTemporalBlock(nn.Module):
    """시간 해상도를 유지하는 합성곱과 프레임별 GroupNorm/공간 풀링."""

    def __init__(self, in_channels: int, out_channels: int, first: bool = False) -> None:
        super().__init__()
        spatial_kernel = 5 if first else 3
        self.conv = nn.Conv3d(
            in_channels, out_channels, (3, spatial_kernel, spatial_kernel),
            stride=(1, 2 if first else 1, 2 if first else 1),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2), bias=False,
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.pool = nn.MaxPool3d((1, 2, 2))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """매 층의 실제 문장 끝을 복제해 다른 배치 길이에 영향을 받지 않게 한다."""
        if lengths is not None:
            indices = torch.arange(x.shape[2], device=x.device)[None, :]
            indices = torch.minimum(indices, lengths.to(x.device)[:, None] - 1)
            x = x.gather(2, indices[:, None, :, None, None].expand_as(x))
        x = self.conv(F.pad(x, (0, 0, 0, 0, 1, 1), mode="replicate"))
        b, c, t, h, w = x.shape
        # 프레임별 정규화: 다른 시점/패딩/표본을 통계에 섞지 않는다.
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.relu(self.norm(x))
        x = x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return self.pool(x)


class LipNetBaseline(nn.Module):
    """약 300만 변수의 경량 시공간 특징 추출기와 양방향 LSTM."""

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 256) -> None:
        super().__init__()
        if num_classes != NUM_CLASSES:
            raise TokenizerMismatchError(f"현재 토큰 사전은 {NUM_CLASSES}클래스입니다.")
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden
        self.conv = nn.ModuleList([
            SpatialTemporalBlock(3, 32, first=True),
            SpatialTemporalBlock(32, 64),
            SpatialTemporalBlock(64, 96),
        ])
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 4, 4))
        self.projection = nn.Sequential(
            nn.Linear(96 * 4 * 4, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
        )
        self.lstm = nn.LSTM(256, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """(B,3,T,96,96) → (T,B,69) FP32 로그 확률. lengths는 패딩 전 길이."""
        if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] < 1 or x.shape[3:] != (96, 96):
            raise ValueError("모델 입력은 (B,3,T,96,96), T>=1이어야 합니다.")
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device="cpu")
            if (lengths.shape != (x.shape[0],) or lengths.dtype not in (torch.int32, torch.int64)
                    or (lengths < 1).any() or (lengths > x.shape[2]).any()):
                raise ValueError("lengths는 배치별 실제 프레임 수(1<=T<=입력 길이)여야 합니다.")
        feat = x
        for block in self.conv:
            feat = block(feat, lengths)
        feat = self.spatial_pool(feat)
        b, c, t, h, w = feat.shape
        feat = self.projection(feat.permute(0, 2, 1, 3, 4).reshape(b, t, c * h * w))
        if lengths is None:
            seq, _ = self.lstm(feat)
        else:
            packed = pack_padded_sequence(feat, lengths, batch_first=True, enforce_sorted=False)
            seq, _ = self.lstm(packed)
            seq, _ = pad_packed_sequence(seq, batch_first=True, total_length=t)
        logits = self.fc(seq)
        # AMP에서도 log-softmax와 CTC는 FP32로 수치 정밀도를 유지한다.
        return F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2).contiguous()

    def save_checkpoint(self, path: str, training_data: dict | None = None,
                        training_run: dict | None = None) -> None:
        """구조/토큰/출처 포함 추론용 가중치 저장. 학습 재개 상태는 포함하지 않는다."""
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
    def load_from_checkpoint(cls, path: str) -> "LipNetBaseline":
        """동일 구조의 가중치만 CPU에 로드한다. 필요하면 이후 .to(device)를 적용한다."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except OSError as exc:
            raise OSError(f"체크포인트 로드 실패: {path} ({exc})") from exc
        validate_tokenizer_metadata(ckpt, path)
        if ckpt.get("vocab") != GRAPHEME_VOCAB:
            raise TokenizerMismatchError(f"{path}: 체크포인트 토큰 순서가 일치하지 않습니다.")
        if ckpt.get("architecture_version") != ARCHITECTURE_VERSION:
            raise ModelCompatibilityError(
                f"{path}: {ARCHITECTURE_VERSION} 구조의 새 학습 가중치가 필요합니다. "
                "기존 대형 모델의 가중치는 경량 모델과 호환되지 않습니다."
            )
        model = cls(num_classes=ckpt["num_classes"], lstm_hidden=ckpt["lstm_hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.training_data = ckpt.get("training_data")
        model.training_run = ckpt.get("training_run")
        return model.eval()


def load_baseline(weights_path: str, num_classes: int = NUM_CLASSES) -> LipNetBaseline:
    """가중치를 로드하는 호환용 래퍼."""
    if num_classes != NUM_CLASSES:
        raise TokenizerMismatchError(f"현재 토큰 사전은 {NUM_CLASSES}클래스입니다.")
    return LipNetBaseline.load_from_checkpoint(weights_path)
