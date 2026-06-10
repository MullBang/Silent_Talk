"""학습된 모델 성능 테스트 — 캐시된 문장 클립(.npz)에 대해 예측/정답/CER 출력.

기본은 학습에서 제외된 검증셋(마지막 --val-split 비율)으로 테스트하여 일반화
성능을 본다. --all로 전체 클립을 테스트할 수도 있다.

torch만 사용(mediapipe 미포함). 새 영상을 테스트하려면 먼저 prepare_data.py로
.npz를 만든 뒤 그 디렉토리를 --data로 지정한다.

사용:
  python scripts/test_model.py                          # 검증셋 테스트
  python scripts/test_model.py --data data_cache/clips --all
  python scripts/test_model.py --data data_cache/test_clips --all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
from jamo import h2j

# Windows 콘솔(cp949)에서도 한글이 깨지지 않도록 UTF-8 출력 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402
from models.decoder import decode  # noqa: E402

_MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
_STD = np.array(NORMALIZE_STD, dtype=np.float32)


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


def _cer(ref: str, hyp: str) -> float:
    r = list("".join(h2j(ref)))
    h = list("".join(h2j(hyp)))
    if not r:
        return 0.0 if not h else 1.0
    return _levenshtein(r, h) / len(r)


def _to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    x = (frames_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,96,96)


def main() -> None:
    p = argparse.ArgumentParser(description="학습 모델 성능 테스트")
    p.add_argument("--data", default="data_cache/clips", help="npz 디렉토리")
    p.add_argument("--weights", default=MODEL_WEIGHTS_PATH, help="체크포인트 경로")
    p.add_argument("--val-split", type=float, default=0.15, help="검증셋 비율(뒤에서)")
    p.add_argument("--all", action="store_true", help="전체 클립 테스트")
    p.add_argument("--n", type=int, default=None, help="테스트 클립 수 제한")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if not os.path.exists(args.weights):
        print(f"[오류] 가중치가 없습니다: {args.weights}")
        print("       먼저 scripts/train.py로 학습하세요.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not files:
        print(f"[오류] 클립(.npz)이 없습니다: {args.data}")
        sys.exit(1)

    if not args.all and args.val_split > 0:
        n_val = max(1, int(len(files) * args.val_split))
        files = files[len(files) - n_val:]
    if args.n:
        files = files[: args.n]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = LipNetBaseline.load_from_checkpoint(args.weights).to(device)
    model.eval()
    print(f"[모델] {args.weights}  · device={device}")
    print(f"[테스트] {len(files)}개 클립 {'(전체)' if args.all else '(검증셋)'}\n")

    cers = []
    with torch.no_grad():
        for f in files:
            d = np.load(f, allow_pickle=True)
            ref = str(d["text"])
            log_probs = model(_to_tensor(d["frames"]).to(device)).cpu()
            pred, conf, _raw = decode(log_probs)
            c = _cer(ref, pred)
            cers.append(c)
            print(f"  정답: {ref}")
            print(f"  예측: {pred or '(빈 출력)'}   [CER={c:.3f} conf={conf:.2f}]\n")

    avg = sum(cers) / len(cers)
    print(f"=== 평균 CER: {avg:.4f}  ({len(cers)}개) ===")
    print(f"    (CER 0=완벽, 1=전부 오류 · 참고: 문자정확도 약 {max(0.0, 1 - avg) * 100:.1f}%)")


if __name__ == "__main__":
    main()
