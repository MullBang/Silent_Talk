"""LipNet 베이스라인 CTC 학습 스크립트.

prepare_data/prepare_batch가 생성한 문장별 .npz(frames, label)를 로드해
3D-CNN+BiLSTM+CTC 모델을 학습한다. GPU가 있으면 자동 사용한다.
train/val 분리, val CER 모니터링, best 체크포인트 저장을 지원한다.

mediapipe를 import하지 않는다(정규화 인라인) → torch와의 네이티브 충돌 회피.

사용:
  python scripts/train.py --data data_cache/clips --epochs 150 \
      --batch-size 2 --lr 1e-4 --val-split 0.15 --eval-every 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from jamo import h2j
from torch.utils.data import DataLoader, Dataset

# backend 루트 경로 추가
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402
from models.decoder import decode  # noqa: E402

_MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
_STD = np.array(NORMALIZE_STD, dtype=np.float32)


def _normalize(frames_uint8: np.ndarray) -> np.ndarray:
    """(T,96,96,3) uint8 → ImageNet 정규화 float32."""
    return (frames_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD


def _levenshtein(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def _cer(reference: str, hypothesis: str) -> float:
    """자모 단위 CER."""
    ref = list("".join(h2j(reference)))
    hyp = list("".join(h2j(hypothesis)))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


class ClipDataset(Dataset):
    """문장별 .npz(frames, label, text) 데이터셋."""

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        d = np.load(self.files[idx], allow_pickle=True)
        return {"frames": d["frames"], "label": d["label"], "text": str(d["text"])}


def collate(batch: list[dict]):
    """가변 길이 T를 last-frame 패딩으로 배치화 (zero-padding 금지)."""
    t_max = max(b["frames"].shape[0] for b in batch)
    tensors, in_lens, targets, tgt_lens = [], [], [], []
    for b in batch:
        f = _normalize(b["frames"])
        t = f.shape[0]
        if t < t_max:
            f = np.concatenate([f, np.repeat(f[-1:], t_max - t, axis=0)], axis=0)
        tensors.append(torch.from_numpy(f).permute(3, 0, 1, 2))
        in_lens.append(t)
        targets.append(torch.from_numpy(b["label"]))
        tgt_lens.append(len(b["label"]))
    return (
        torch.stack(tensors, dim=0),
        torch.cat(targets),
        torch.tensor(in_lens, dtype=torch.long),
        torch.tensor(tgt_lens, dtype=torch.long),
    )


@torch.no_grad()
def evaluate_val(model, files: list[str], device: str) -> tuple[float, list[tuple[str, str]]]:
    """검증셋 평균 CER과 (정답, 예측) 샘플을 반환한다."""
    model.eval()
    ds = ClipDataset(files)
    cers, samples = [], []
    for i in range(len(ds)):
        item = ds[i]
        x, _t, _il, _tl = collate([item])
        log_probs = model(x.to(device)).cpu()
        pred, _conf, _raw = decode(log_probs)
        cers.append(_cer(item["text"], pred))
        if i < 3:
            samples.append((item["text"], pred))
    model.train()
    return (sum(cers) / len(cers) if cers else 1.0), samples


def train(args) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not files:
        raise FileNotFoundError(f"학습 데이터(.npz)가 없습니다: {args.data}")

    # train/val 분리 (마지막 비율을 val로)
    n_val = int(len(files) * args.val_split)
    val_files = files[len(files) - n_val:] if n_val > 0 else []
    train_files = files[: len(files) - n_val] if n_val > 0 else files
    print(f"[데이터] 총 {len(files)} (train {len(train_files)} / val {len(val_files)}) "
          f"· device={device} · batch={args.batch_size}")

    loader = DataLoader(
        ClipDataset(train_files), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    model = LipNetBaseline().to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_cer = float("inf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, targets, in_lens, tgt_lens in loader:
            log_probs = model(x.to(device))
            loss = ctc(log_probs, targets, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            if val_files:
                val_cer, samples = evaluate_val(model, val_files, device)
                marker = ""
                if val_cer < best_cer:
                    best_cer = val_cer
                    model.save_checkpoint(args.out)
                    marker = "  ★best 저장"
                print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  val_CER={val_cer:.3f}{marker}")
            else:
                model.save_checkpoint(args.out)
                print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}")

    # val이 없으면 마지막 모델이 저장됨. 샘플 디코딩 출력.
    eval_files = val_files or train_files[:3]
    _cer_final, samples = evaluate_val(model, eval_files[:3], device)
    print(f"[저장] best 체크포인트 → {args.out}  (best val_CER={best_cer if val_files else 'N/A'})")
    for ref, pred in samples:
        print(f"  정답='{ref}'")
        print(f"  예측='{pred}'")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LipNet CTC 학습")
    p.add_argument("--data", default="data_cache/clips")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default=MODEL_WEIGHTS_PATH)
    p.add_argument("--device", default=None, help="cuda | cpu (기본: 자동)")
    p.add_argument("--val-split", type=float, default=0.15, help="검증셋 비율")
    p.add_argument("--eval-every", type=int, default=10, help="검증 주기(epoch)")
    return p.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
