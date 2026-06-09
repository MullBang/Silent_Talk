"""성능 평가 스크립트 (설계문서 F-MVP-05).

등록된 test_set_id 기준으로 모델을 추론하고 CER/WER/지연/검출률을 산출하여
evaluation_log.csv에 영구 기록하고 CER/WER 바 그래프를 저장한다.

test_set_path 직접 전달 금지 → config.TEST_SETS에 등록된 test_set_id로만 조회.

사용:
  python scripts/evaluate.py --test_set_id aihub_subset \
      --model_version baseline_v1 --eval_unit word

테스트셋 디렉토리 구조:
  <test_set_path>/
    manifest.json   # [{"video": "a.mp4", "text": "정답 텍스트"}, ...]
    a.mp4 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 디스플레이 없는 환경
import matplotlib.pyplot as plt  # noqa: E402

# backend 루트를 import 경로에 추가
_BACKEND_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from config import MODEL_WEIGHTS_PATH, TEST_SETS  # noqa: E402

from jamo import h2j  # noqa: E402

# 평가 결과 출력 디렉토리 (테스트에서 monkeypatch 가능)
RESULTS_DIR = os.path.join(os.getcwd(), "results")


# ---------------------------------------------------------------------------
# 편집 거리 / 지표
# ---------------------------------------------------------------------------

def _levenshtein(a, b) -> int:
    """두 시퀀스 간 Levenshtein 편집 거리."""
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


def cer(reference: str, hypothesis: str) -> float:
    """문자 오류율 (CER) — python-jamo 자모 단위 편집 거리.

    Args:
        reference: 정답 텍스트.
        hypothesis: 예측 텍스트.

    Returns:
        CER (0.0~). 정답이 비고 예측이 있으면 1.0.
    """
    ref = list("".join(h2j(reference)))
    hyp = list("".join(h2j(hypothesis)))
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """단어(어절) 오류율 (WER) — 공백 분리 후 편집 거리.

    Args:
        reference: 정답 텍스트.
        hypothesis: 예측 텍스트.

    Returns:
        WER (0.0~). 정답이 비고 예측이 있으면 1.0.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# 테스트셋 / 추론
# ---------------------------------------------------------------------------

def load_test_set(test_set_id: str) -> list[dict]:
    """등록된 test_set_id로 테스트셋(영상+정답) 목록을 로드한다.

    Args:
        test_set_id: config.TEST_SETS에 등록된 식별자.

    Returns:
        [{video: 절대경로, text: 정답}, ...].

    Raises:
        ValueError: 미등록 test_set_id.
        FileNotFoundError: manifest.json 부재.
    """
    path = TEST_SETS.get(test_set_id)
    if path is None:
        raise ValueError(f"등록되지 않은 test_set_id입니다: {test_set_id}")
    manifest = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"manifest.json이 없습니다: {manifest}")
    with open(manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [
        {"video": os.path.join(path, e["video"]), "text": e["text"]}
        for e in entries
    ]


def _load_model():
    """가중치가 있으면 로드, 없으면 미학습 모델로 폴백한다."""
    from models.baseline import LipNetBaseline

    if os.path.exists(MODEL_WEIGHTS_PATH):
        return LipNetBaseline.load_from_checkpoint(MODEL_WEIGHTS_PATH)
    return LipNetBaseline().eval()


def predict_video(video_path: str, model) -> tuple[str, float, bool]:
    """단일 영상을 추론한다.

    Args:
        video_path: 영상 경로.
        model: LipNetBaseline 인스턴스.

    Returns:
        (예측 텍스트, 추론 지연 ms, 얼굴 검출 성공 여부).
    """
    import torch

    from models.decoder import decode
    from preprocessing.pipeline import run_preprocessing_pipeline

    t0 = time.perf_counter()
    segments = run_preprocessing_pipeline(video_path)
    detected = any(not s["skipped"] for s in segments)
    texts: list[str] = []
    for seg in segments:
        if seg["skipped"]:
            continue
        with torch.no_grad():
            log_probs = model(seg["tensor"])
        text, _conf, _raw = decode(log_probs)
        if text:
            texts.append(text)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return " ".join(texts), latency_ms, detected


# ---------------------------------------------------------------------------
# 저장 (CSV / 그래프)
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "eval_id", "model_version", "cer", "wer", "avg_latency_ms",
    "detection_rate", "test_set_id", "eval_unit", "test_date",
]


def _next_eval_id(csv_path: str) -> int:
    """기존 CSV 행 수 기준 eval_id 발급."""
    if not os.path.exists(csv_path):
        return 1
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return max(1, len(rows))  # 헤더 1행 포함 → 데이터 n행이면 다음 id는 n+1


def _append_csv(row: dict, csv_path: str) -> None:
    """evaluation_log.csv에 한 행을 추가한다 (없으면 헤더 생성)."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _plot_metrics(metrics: dict, eval_id: int, test_set_id: str) -> str:
    """CER/WER 바 그래프를 생성/저장하고 경로를 반환한다."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["CER", "WER"], [metrics["cer"], metrics["wer"]], color=["#3b82f6", "#ef4444"])
    ax.set_ylim(0, max(1.0, metrics["cer"], metrics["wer"]))
    ax.set_ylabel("error rate")
    ax.set_title(f"eval #{eval_id} · {test_set_id}")
    for i, v in enumerate([metrics["cer"], metrics["wer"]]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"eval_{eval_id}_{test_set_id}.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def evaluate(test_set_id: str, model_version: str, eval_unit: str = "word") -> dict:
    """테스트셋 평가를 수행하고 CSV/그래프를 저장한다.

    Args:
        test_set_id: 등록된 테스트셋 식별자.
        model_version: 모델 버전 라벨.
        eval_unit: 'word' | 'sentence'.

    Returns:
        지표 + 산출물 경로를 담은 딕셔너리.
    """
    entries = load_test_set(test_set_id)
    model = _load_model()

    cers, wers, lats, detected = [], [], [], 0
    for e in entries:
        hyp, latency, det = predict_video(e["video"], model)
        cers.append(cer(e["text"], hyp))
        wers.append(wer(e["text"], hyp))
        lats.append(latency)
        if det:
            detected += 1

    metrics = {
        "cer": round(_mean(cers), 4),
        "wer": round(_mean(wers), 4),
        "avg_latency_ms": round(_mean(lats), 2),
        "detection_rate": round(detected / len(entries), 4) if entries else 0.0,
    }

    csv_path = os.path.join(RESULTS_DIR, "evaluation_log.csv")
    eval_id = _next_eval_id(csv_path)
    row = {
        "eval_id": eval_id,
        "model_version": model_version,
        **metrics,
        "test_set_id": test_set_id,
        "eval_unit": eval_unit,
        "test_date": datetime.now().isoformat(timespec="seconds"),
    }
    _append_csv(row, csv_path)
    graph = _plot_metrics(metrics, eval_id, test_set_id)

    return {
        **metrics,
        "eval_id": eval_id,
        "num_samples": len(entries),
        "csv": csv_path,
        "graph": graph,
    }


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱."""
    p = argparse.ArgumentParser(description="립리딩 모델 성능 평가 (CER/WER)")
    p.add_argument("--test_set_id", required=True, help="config.TEST_SETS 등록 식별자")
    p.add_argument("--model_version", required=True, help="모델 버전 라벨")
    p.add_argument(
        "--eval_unit", choices=["word", "sentence"], default="word", help="평가 단위"
    )
    return p.parse_args()


def main() -> None:
    """엔트리포인트: 인자 파싱 → 평가 실행 → 결과 출력."""
    args = parse_args()
    try:
        result = evaluate(args.test_set_id, args.model_version, args.eval_unit)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[평가 실패] {exc}")
        sys.exit(1)
    print("=== 평가 결과 ===")
    print(f"  eval_id        : {result['eval_id']}")
    print(f"  샘플 수        : {result['num_samples']}")
    print(f"  CER            : {result['cer']}")
    print(f"  WER            : {result['wer']}")
    print(f"  avg_latency_ms : {result['avg_latency_ms']}")
    print(f"  detection_rate : {result['detection_rate']}")
    print(f"  CSV            : {result['csv']}")
    print(f"  graph          : {result['graph']}")


if __name__ == "__main__":
    main()
