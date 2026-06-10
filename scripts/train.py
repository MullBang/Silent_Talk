"""LipNet 베이스라인 CTC 학습 스크립트.

prepare_data.py가 생성한 문장별 .npz(frames, label)를 로드해 3D-CNN+BiLSTM+CTC
모델을 학습하고 체크포인트를 저장한다. GPU가 있으면 자동 사용한다.

mediapipe를 import하지 않는다(정규화는 인라인) → torch와의 네이티브 충돌 회피.

사용:
  python scripts/train.py --data data_cache/clips --epochs 50 \
      --batch-size 2 --lr 1e-4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# backend 루트 경로 추가
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models import IDX_TO_GRAPHEME  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402
from models.decoder import decode  # noqa: E402

_MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
_STD = np.array(NORMALIZE_STD, dtype=np.float32)


def _normalize(frames_uint8: np.ndarray) -> np.ndarray:
    """(T,96,96,3) uint8 → ImageNet 정규화 float32 (roi_extractor와 동일 기준)."""
    return (frames_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD


class ClipDataset(Dataset):
    """문장별 .npz(frames, label) 데이터셋."""

    def __init__(self, data_dir: str) -> None:
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"학습 데이터(.npz)가 없습니다: {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        d = np.load(self.files[idx], allow_pickle=True)
        return {"frames": d["frames"], "label": d["label"], "text": str(d["text"])}


def collate(batch: list[dict]):
    """가변 길이 T를 last-frame 패딩으로 배치화한다 (zero-padding 금지)."""
    t_max = max(b["frames"].shape[0] for b in batch)
    tensors, in_lens, targets, tgt_lens = [], [], [], []
    for b in batch:
        f = _normalize(b["frames"])  # (T,96,96,3)
        t = f.shape[0]
        if t < t_max:
            pad = np.repeat(f[-1:], t_max - t, axis=0)
            f = np.concatenate([f, pad], axis=0)
        tensors.append(torch.from_numpy(f).permute(3, 0, 1, 2))  # (3,Tmax,96,96)
        in_lens.append(t)
        targets.append(torch.from_numpy(b["label"]))
        tgt_lens.append(len(b["label"]))
    x = torch.stack(tensors, dim=0)  # (B,3,Tmax,96,96)
    return (
        x,
        torch.cat(targets),
        torch.tensor(in_lens, dtype=torch.long),
        torch.tensor(tgt_lens, dtype=torch.long),
    )


def train(data_dir: str, epochs: int, batch_size: int, lr: float,
          out_path: str, device: str) -> None:
    """학습 루프."""
    ds = ClipDataset(data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    print(f"[데이터] {len(ds)}개 클립 · device={device} · batch={batch_size}")

    model = LipNetBaseline().to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, targets, in_lens, tgt_lens in loader:
            x = x.to(device)
            log_probs = model(x)  # (Tmax, B, C)
            loss = ctc(log_probs, targets, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_checkpoint(out_path)
    print(f"[저장] 체크포인트 → {out_path}")

    # 과적합/학습 확인용 디코딩 샘플
    model.eval()
    with torch.no_grad():
        sample = ds[0]
        x, _t, _il, _tl = collate([sample])
        log_probs = model(x.to(device)).cpu()
        text, conf, _raw = decode(log_probs)
    print(f"[샘플] 정답='{sample['text']}'")
    print(f"       예측='{text}'  (conf={conf:.3f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LipNet CTC 학습")
    p.add_argument("--data", default="data_cache/clips", help="npz 데이터 디렉토리")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default=MODEL_WEIGHTS_PATH, help="체크포인트 경로")
    p.add_argument("--device", default=None, help="cuda | cpu (기본: 자동)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train(args.data, args.epochs, args.batch_size, args.lr, args.out, device)


if __name__ == "__main__":
    main()
