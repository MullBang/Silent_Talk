"""학습 데이터 준비 (AI Hub 립리딩 → 문장 단위 ROI/라벨 캐시).

영상 + 라벨 JSON(Sentence_info)을 받아 각 문장 구간을 잘라 25fps 리샘플 →
MediaPipe 입술 ROI(96×96) 추출 → 텍스트를 자모 인덱스로 인코딩 →
문장별 .npz(frames, label, text)로 저장한다.

torch를 import하지 않는다(전처리/라벨만) → mediapipe와의 네이티브 충돌 회피.
학습은 train.py에서 캐시 npz를 로드해 수행한다.

사용:
  python scripts/prepare_data.py \
      --video "D:\\...\\lip_..._001.mp4" \
      --label "data_cache\\labels\\lip_..._001.json" \
      --out data_cache/clips --max-sentences 3 --scale 640
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np
from jamo import h2j, j2hcj

# backend 루트 경로 추가
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from models import GRAPHEME_TO_IDX  # noqa: E402 (순수 파이썬, torch 미포함)
from preprocessing.resampler import resample_to_fps  # noqa: E402
from preprocessing.roi_extractor import extract_roi_from_segment  # noqa: E402


def encode_text(text: str) -> list[int]:
    """한국어 텍스트를 자모 인덱스 시퀀스로 인코딩한다.

    한글 음절 → 자모 분해(h2j) → 호환 자모(j2hcj) → GRAPHEME_TO_IDX 매핑.
    한글이 아닌 문자(공백/구두점/숫자)는 제외(자모 41클래스 기준).

    Args:
        text: 정답 문장.

    Returns:
        자모 인덱스 리스트.
    """
    indices: list[int] = []
    for ch in h2j(text):
        compat = j2hcj(ch)
        if compat in GRAPHEME_TO_IDX:
            indices.append(GRAPHEME_TO_IDX[compat])
    return indices


def _cut_sentence_clip(video_path: str, start_sec: float, end_sec: float,
                       dst: str, scale_w: int) -> bool:
    """영상에서 [start,end] 구간을 잘라 scale_w로 축소해 임시 mp4로 저장한다."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w == 0:
        cap.release()
        return False
    ow = scale_w
    oh = int(round(h * scale_w / w))
    start_f = int(start_sec * fps)
    end_f = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    count = 0
    for _ in range(max(1, end_f - start_f)):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, (ow, oh)))
        count += 1
    writer.release()
    cap.release()
    return count > 0


def prepare_video(video_path: str, label_path: str, out_dir: str,
                  max_sentences: int | None = None, scale_w: int = 640) -> int:
    """단일 영상+라벨을 문장 단위 npz로 변환한다.

    Args:
        video_path: 원본 영상 경로.
        label_path: 라벨 JSON 경로(Sentence_info 포함).
        out_dir: npz 저장 디렉토리.
        max_sentences: 처리할 최대 문장 수(None=전체).
        scale_w: 전처리 가속을 위한 다운스케일 폭(px).

    Returns:
        저장된 문장 클립 수.
    """
    os.makedirs(out_dir, exist_ok=True)
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)
    # AI Hub 라벨은 [{...}] 리스트로 감싸진 경우가 있다.
    if isinstance(label, list):
        label = label[0]
    sentences = label["Sentence_info"]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    base = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    tmp_dir = tempfile.mkdtemp(prefix="prep_")
    try:
        for s in sentences:
            text = s["sentence_text"].strip()
            label_idx = encode_text(text)
            if not label_idx:
                continue
            tmp_clip = os.path.join(tmp_dir, f"{base}_{s['ID']}.mp4")
            if not _cut_sentence_clip(video_path, s["start_time"], s["end_time"],
                                      tmp_clip, scale_w):
                continue
            try:
                frames, _fps = resample_to_fps(tmp_clip)  # 25fps 리샘플
                roi = extract_roi_from_segment(np.stack(frames, axis=0))
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] id={s['ID']} 전처리 실패: {exc}")
                continue
            if roi is None:
                print(f"  [skip] id={s['ID']} 얼굴 미검출")
                continue
            # CTC 제약: 입력 길이(T) >= 라벨 길이
            if roi.shape[0] < len(label_idx):
                print(f"  [skip] id={s['ID']} T({roi.shape[0]}) < label({len(label_idx)})")
                continue
            out = os.path.join(out_dir, f"{base}_{s['ID']:03d}.npz")
            np.savez_compressed(
                out,
                frames=roi.astype(np.uint8),
                label=np.array(label_idx, dtype=np.int64),
                text=text,
            )
            saved += 1
            print(f"  [ok] id={s['ID']} T={roi.shape[0]} label={len(label_idx)} -> {os.path.basename(out)}")
    finally:
        for fn in os.listdir(tmp_dir):
            try:
                os.remove(os.path.join(tmp_dir, fn))
            except OSError:
                pass
        os.rmdir(tmp_dir)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="립리딩 학습 데이터 준비")
    p.add_argument("--video", required=True, help="원본 영상 경로")
    p.add_argument("--label", required=True, help="라벨 JSON 경로")
    p.add_argument("--out", default="data_cache/clips", help="npz 출력 디렉토리")
    p.add_argument("--max-sentences", type=int, default=None, help="최대 문장 수")
    p.add_argument("--scale", type=int, default=640, help="다운스케일 폭(px)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[준비] {os.path.basename(args.video)}")
    n = prepare_video(args.video, args.label, args.out, args.max_sentences, args.scale)
    print(f"[완료] {n}개 문장 클립 저장 → {args.out}")


if __name__ == "__main__":
    main()
