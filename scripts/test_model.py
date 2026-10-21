"""학습된 모델 성능 테스트 — 캐시된 문장 클립(.npz)에 대해 예측/정답/CER 출력.

고정 명세의 val이 기본이며 최종 평가는 --split test로 별도 실행한다.
체크포인트의 학습 명세 해시가 일치해야 하며 test로 모델을 선정하지 않는다.

torch만 사용(mediapipe 미포함). 새 영상을 테스트하려면 먼저 prepare_data.py로
.npz를 만든 뒤 --smoke-data로 디버깅한다(일반화 평가 아님).

사용:
  python scripts/test_model.py                          # 검증셋 테스트
  python scripts/test_model.py --split test
  python scripts/test_model.py --smoke-data data_cache/test_clips
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

# Windows 콘솔(cp949)에서도 한글이 깨지지 않도록 UTF-8 출력 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import MODEL_WEIGHTS_PATH, NORMALIZE_MEAN, NORMALIZE_STD  # noqa: E402
from models.registry import load_model_checkpoint  # noqa: E402
from models.decoder import decode, ctc_greedy_decode  # noqa: E402
from models.cache import load_clip  # noqa: E402
from models.tokenizer import cer as _cer, wer as _wer, normalize_text  # noqa: E402
from models.dataset_manifest import DEFAULT_MANIFEST, load_manifest, split_files, sha256_file  # noqa: E402

_MEAN = np.array(NORMALIZE_MEAN, dtype=np.float32)
_STD = np.array(NORMALIZE_STD, dtype=np.float32)


def _to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    x = (frames_uint8.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,96,96)


def evaluation_files(args) -> tuple[list[str], dict | None]:
    """평가 시에도 같은 명세를 검증하며 명시한 집합만 선택한다."""
    if args.smoke_data:
        files = sorted(glob.glob(os.path.join(args.smoke_data, "*.npz")))
        if not files:
            raise FileNotFoundError(f"스모크 캐시가 없습니다: {args.smoke_data}")
        return files, None
    document, summary = load_manifest(args.manifest)
    return split_files(document, args.manifest, args.split), summary


def validate_evaluation_provenance(provenance: dict | None, summary: dict) -> None:
    """다른 분할 또는 스모크 학습 가중치로 고정 평가 점수를 내지 않는다."""
    if (not provenance or provenance.get("mode") != "speaker_disjoint"
            or provenance.get("manifest_sha256") != summary["manifest_sha256"]):
        raise ValueError("체크포인트의 학습 분할과 평가 명세가 다릅니다. 이 명세로 학습한 가중치가 필요합니다.")


def aggregate_predictions(rows: list[dict]) -> dict:
    """문장 평균과 총 편집거리/총 길이 오류율을 구분한다. CER을 정확도로 바꾸지 않는다."""
    if not rows:
        raise ValueError("평가 문장이 없습니다.")
    n = len(rows)
    chars = sum(r["reference_jamo_count"] for r in rows)
    words = sum(r["reference_word_count"] for r in rows)
    return {"clips": n, "mean_sentence_cer": sum(r["cer"] for r in rows) / n,
            "corpus_cer": sum(r["jamo_edit_distance"] for r in rows) / chars,
            "mean_sentence_wer": sum(r["wer"] for r in rows) / n,
            "corpus_wer": sum(r["word_edit_distance"] for r in rows) / words,
            "exact_count": sum(r["exact"] for r in rows),
            "empty_count": sum(not r["prediction"] for r in rows),
            "invalid_jamo_sentence_count": sum("\ufffd" in r["prediction"] for r in rows),
            "reference_jamo_count": chars, "reference_word_count": words}


@torch.no_grad()
def evaluate_files(model, files: list[str], device: str, *, decoder: str = "beam",
                   control: str | None = None) -> dict:
    """캐시의 영상을 추론하고 정답은 출력 후에만 비교한다."""
    if decoder not in ("beam", "greedy") or control not in (None, "reversed", "constant_gray"):
        raise ValueError("지원하지 않는 decoder/control입니다.")
    model.eval()
    started = time.perf_counter()
    rows = []
    for path in files:
        item = load_clip(path)
        frames = item["frames"]
        if control == "reversed":
            frames = frames[::-1].copy()
        elif control == "constant_gray":
            frames = np.full_like(frames, 128)
        log_probs = model(_to_tensor(frames).to(device)).cpu()
        if decoder == "beam":
            pred, confidence, raw = decode(log_probs)
        else:
            pred, confidence, raw = ctc_greedy_decode(log_probs), None, None
        ref = normalize_text(item["text"])
        cer, wer = _cer(ref, pred), _wer(ref, pred)
        char_count = len(unicodedata.normalize("NFD", ref))
        word_count = len(ref.split())
        rows.append({"id": Path(path).stem, "reference": ref, "prediction": pred,
                     "frames": len(frames), "cer": cer, "wer": wer, "exact": ref == pred,
                     "reference_jamo_count": char_count, "reference_word_count": word_count,
                     "jamo_edit_distance": round(cer * char_count),
                     "word_edit_distance": round(wer * word_count),
                     "confidence": confidence, "raw_score": raw})
    return {"decoder": decoder, "beam_width": 10 if decoder == "beam" else None,
            "control": control, "metrics": aggregate_predictions(rows), "predictions": rows,
            "elapsed_seconds_including_loading_and_decoding": time.perf_counter() - started}


def main() -> None:
    """개발 검증 또는 명시적으로 선택한 최종 평가를 실행한다."""
    p = argparse.ArgumentParser(description="학습 모델 성능 테스트")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--manifest", default=DEFAULT_MANIFEST)
    source.add_argument("--smoke-data", help="단순 동작 확인용 캐시 폴더; 일반화 평가 아님")
    p.add_argument("--weights", default=MODEL_WEIGHTS_PATH, help="체크포인트 경로")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--n", type=int, default=None, help="테스트 클립 수 제한")
    p.add_argument("--device", default=None)
    p.add_argument("--report", help="전체 문장 예측/오류율/가중치 출처 JSON")
    p.add_argument("--decoder", choices=["beam", "greedy"], default="beam")
    p.add_argument("--control", choices=["reversed", "constant_gray"], help="진단 전용 대조 입력")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        print(f"[오류] 가중치가 없습니다: {args.weights}")
        print("       먼저 scripts/train.py로 학습하세요.")
        sys.exit(1)

    files, summary = evaluation_files(args)
    if args.n is not None:
        if args.n < 1:
            p.error("--n은 1 이상이어야 합니다.")
        files = files[: args.n]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(args.weights).to(device)
    if summary is not None:
        validate_evaluation_provenance(getattr(model, "training_data", None), summary)
    model.eval()
    print(f"[모델] {args.weights}  · device={device}")
    label = 'smoke_only (일반화 평가 아님)' if args.smoke_data else args.split
    subset = ' / 일부 문장만 평가' if args.n else ''
    print(f"[테스트] {len(files)}개 클립 / {label}{subset}\n")

    report = evaluate_files(model, files, device, decoder=args.decoder, control=args.control)
    report.update({"split": label, "subset": args.n is not None,
                   "checkpoint": str(Path(args.weights).resolve()),
                   "checkpoint_sha256": sha256_file(args.weights),
                   "manifest_sha256": summary["manifest_sha256"] if summary else None,
                   "training_data": model.training_data, "training_run": model.training_run})
    for row in report["predictions"]:
        print(f"  정답: {row['reference']}")
        print(f"  예측: {row['prediction'] or '(빈 출력)'}   [CER={row['cer']:.3f}]\n")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print("CER/WER는 오류율이며 100%를 초과할 수 있습니다. train 결과는 일반화 성능이 아닙니다.")


if __name__ == "__main__":
    main()
