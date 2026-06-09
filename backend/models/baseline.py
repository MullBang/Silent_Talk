"""베이스라인 모델: 3D-CNN + BiLSTM + CTC (LipNet 구조 참고).

설계문서 1.5절. 입력 MODEL_INPUT_SHAPE (B, 3, 75, 96, 96).

차원 흐름 (MaxPool3d는 시간축 보존, 공간축 1/2):
  (B,3,75,96,96)
  → conv1(3→32) + pool(1,2,2)  → (B,32,75,48,48)
  → conv2(32→64) + pool(1,2,2) → (B,64,75,24,24)
  → conv3(64→96)               → (B,96,75,24,24)
  → temporal flatten           → (B,75, 96*24*24)
  → BiLSTM(hidden=256, 2층)     → (B,75,512)
  → Linear(512→41) + LogSoftmax→ (B,75,41)
  → permute                    → (75,B,41)  (CTC Loss 입력 형태)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import NUM_CLASSES


class LipNetBaseline(nn.Module):
    """3D-CNN 시공간 특징 추출 + 양방향 LSTM + CTC 분류 헤드."""

    def __init__(self, num_classes: int = NUM_CLASSES, lstm_hidden: int = 256) -> None:
        """모델 레이어를 구성한다.

        Args:
            num_classes: CTC 출력 클래스 수 (자모 + blank, 기본 41).
            lstm_hidden: BiLSTM hidden 크기 (양방향이므로 출력은 2배).
        """
        super().__init__()
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden

        # 1. 3D-CNN 블록 3개 (공간축만 풀링, 시간축 75 유지)
        self.conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
        )

        # 2~3. temporal flatten 후 BiLSTM. 입력 차원 = 96 * H' * W' = 96*24*24.
        self._feat_channels = 96
        self._feat_hw = 24  # 96 → /2 → /2 = 24
        lstm_input = self._feat_channels * self._feat_hw * self._feat_hw

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # 4~5. 분류 헤드
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: MODEL_INPUT_SHAPE (B, 3, 75, 96, 96) 입력 텐서.

        Returns:
            (T, B, num_classes) CTC log-softmax 로그 확률.
        """
        # 3D-CNN
        feat = self.conv(x)  # (B, 96, T, H', W')
        b, c, t, h, w = feat.shape

        # temporal flatten: (B, C, T, H, W) → (B, T, C*H*W)
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        feat = feat.view(b, t, c * h * w)  # (B, T, C*H*W)

        # BiLSTM
        seq, _ = self.lstm(feat)  # (B, T, 2*hidden)

        # 분류 + log-softmax
        logits = self.fc(seq)  # (B, T, num_classes)
        log_probs = self.log_softmax(logits)  # (B, T, num_classes)

        # CTC 입력 형태 (T, B, num_classes)
        return log_probs.permute(1, 0, 2).contiguous()

    def save_checkpoint(self, path: str) -> None:
        """모델 가중치를 체크포인트로 저장한다.

        Args:
            path: 저장 경로 (.pt/.pth).
        """
        try:
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "num_classes": self.num_classes,
                    "lstm_hidden": self.lstm_hidden,
                },
                path,
            )
        except OSError as exc:
            raise OSError(f"체크포인트 저장 실패: {path} ({exc})") from exc

    @classmethod
    def load_from_checkpoint(cls, path: str) -> "LipNetBaseline":
        """체크포인트에서 모델을 로드한다 (eval 모드).

        Args:
            path: 가중치 파일 경로.

        Returns:
            가중치가 로드된 eval 모드 모델.
        """
        try:
            ckpt = torch.load(path, map_location="cpu")
        except OSError as exc:
            raise OSError(f"체크포인트 로드 실패: {path} ({exc})") from exc

        model = cls(
            num_classes=ckpt.get("num_classes", NUM_CLASSES),
            lstm_hidden=ckpt.get("lstm_hidden", 256),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model


def load_baseline(weights_path: str, num_classes: int = NUM_CLASSES) -> LipNetBaseline:
    """가중치를 로드한 베이스라인 모델을 반환한다 (호환용 래퍼).

    Args:
        weights_path: .pt/.pth 가중치 파일 경로.
        num_classes: CTC 출력 클래스 수.

    Returns:
        eval 모드 모델 인스턴스.
    """
    return LipNetBaseline.load_from_checkpoint(weights_path)
