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

from config import FPS  # noqa: E402 (학습 리샘플 목표 fps=25)
from models import GRAPHEME_TO_IDX  # noqa: E402 (순수 파이썬, torch 미포함)
from preprocessing.resampler import resample_to_fps  # noqa: E402
from preprocessing.roi_extractor import (  # noqa: E402
    crop_roi_from_box,
    extract_roi_from_segment,
)


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


def ctc_min_input_len(label_idx: list[int]) -> int:
    """CTC가 해당 라벨을 정렬 가능하게 하는 최소 입력 길이(T)를 계산한다.

    CTC는 인접한 동일 라벨 사이에 blank가 반드시 있어야 하므로, 필요한 최소
    타임스텝은 len(label) + (인접 중복 라벨 쌍의 수)이다. 단순 T>=L 검사보다
    엄밀하여, 긴 문장에서 조용히 그래디언트가 0이 되는 표본을 걸러낸다.

    Args:
        label_idx: 자모 인덱스 시퀀스.

    Returns:
        정렬 가능 최소 입력 길이.
    """
    if not label_idx:
        return 0
    repeats = sum(1 for i in range(1, len(label_idx)) if label_idx[i] == label_idx[i - 1])
    return len(label_idx) + repeats


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


def _extract_lip_boxes(label: dict) -> tuple[list | None, float]:
    """라벨에서 프레임별 입술 박스와 원본 FPS를 추출한다.

    AI Hub 라벨의 Bounding_box_info.Lip_bounding_box.xtl_ytl_xbr_ybr는 원본 fps
    기준 프레임별 입술 박스 [xtl,ytl,xbr,ybr] 리스트다.

    Args:
        label: 라벨 dict.

    Returns:
        (박스 리스트 또는 None, 원본 fps). 박스가 없으면 (None, fps).
    """
    bbi = label.get("Bounding_box_info") or {}
    lip = (bbi.get("Lip_bounding_box") or {}).get("xtl_ytl_xbr_ybr")
    fps = float((label.get("Video_info") or {}).get("FPS", 30) or 30)
    if not lip:
        return None, fps
    return lip, fps


def _detect_rotation(frame_w: int, frame_h: int, lip_boxes: list):
    """라벨 좌표계와 저장 영상 방향의 불일치(90° 회전)를 감지한다.

    AI Hub 영상은 세로(예: 1080×1920)로 촬영·라벨링되었으나 파일은 가로(1920×
    1080)로 저장되어(회전 플래그 미적용), cv2가 읽은 프레임과 라벨 좌표계가
    90° 어긋난다. 라벨 y좌표가 실제 프레임 높이를 초과하면 세로 좌표계이므로
    프레임을 세로로 회전해 정합한다(데이터셋 관측상 반시계 방향으로 정합됨).

    Args:
        frame_w: cv2가 읽은 프레임 너비.
        frame_h: cv2가 읽은 프레임 높이.
        lip_boxes: 프레임별 [xtl,ytl,xbr,ybr] 리스트.

    Returns:
        cv2.ROTATE_* 상수 또는 회전 불필요 시 None.
    """
    arr = np.asarray(lip_boxes)
    x_max = float(arr[:, 2].max())
    y_max = float(arr[:, 3].max())
    # 좌표 y가 프레임 높이를 넘고 x는 프레임 높이 이내 → 세로 좌표계(회전 필요)
    if y_max > frame_h and x_max <= max(frame_w, frame_h):
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _roi_seq_from_boxes(cap, start_sec: float, end_sec: float,
                        lip_boxes: list, orig_fps: float,
                        rotate=None) -> np.ndarray | None:
    """라벨 입술 박스로 [start,end] 구간의 25fps ROI 시퀀스를 만든다.

    타임스탬프 기반 리샘플링(FPS=25)으로 각 출력 프레임에 대응하는 원본 프레임
    인덱스를 계산하고, 그 인덱스의 입술 박스로 원본 해상도 프레임에서 ROI를
    크롭한다(MediaPipe 미사용). MediaPipe 경로와 동일하게 RGB로 저장한다.

    Args:
        cap: 열려 있는 cv2.VideoCapture(원본 해상도).
        start_sec: 문장 시작(초).
        end_sec: 문장 끝(초).
        lip_boxes: 원본 fps 기준 프레임별 [xtl,ytl,xbr,ybr] 리스트.
        orig_fps: 원본 영상 fps.
        rotate: 라벨 좌표계 정합을 위한 cv2.ROTATE_* (None=회전 안 함).

    Returns:
        (T,96,96,3) uint8 ROI. 프레임을 하나도 못 읽으면 None.
    """
    n_boxes = len(lip_boxes)
    duration = max(0.0, end_sec - start_sec)
    n_out = int(round(duration * FPS))
    if n_out <= 0:
        return None
    # 각 출력 프레임의 원본 프레임 인덱스(타임스탬프 기반, 단조 증가)
    targets = [min(n_boxes - 1, int(round((start_sec + k / FPS) * orig_fps)))
               for k in range(n_out)]

    lo = targets[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)  # 구간 시작으로 시크(fps 변경 아님)
    rois: list[np.ndarray] = []
    cur = lo
    frame = None
    for need in targets:
        while cur <= need:
            ok, f = cap.read()
            if not ok:
                f = None
                break
            frame = f
            cur += 1
        if frame is None:
            break
        oriented = cv2.rotate(frame, rotate) if rotate is not None else frame
        rgb = cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB)
        rois.append(crop_roi_from_box(rgb, lip_boxes[need]))
    if not rois:
        return None
    return np.stack(rois, axis=0).astype(np.uint8)


def prepare_video(video_path: str, label_path: str, out_dir: str,
                  max_sentences: int | None = None, scale_w: int = 640,
                  max_frames: int = 0, roi_mode: str = "mediapipe") -> int:
    """단일 영상+라벨을 문장 단위 npz로 변환한다.

    Args:
        video_path: 원본 영상 경로.
        label_path: 라벨 JSON 경로(Sentence_info 포함).
        out_dir: npz 저장 디렉토리.
        max_sentences: 처리할 최대 문장 수(None=전체 문장, 긴 문장 포함).
        scale_w: mediapipe 경로의 다운스케일 폭(px). bbox 경로에서는 미사용.
        max_frames: 클립 최대 프레임 수 상한(0=무제한). 초과 클립은 학습 시
            메모리 스파이크를 유발하므로 스킵한다(25fps 리샘플 후 기준).
        roi_mode: ROI 추출 방식.
            "mediapipe"(기본) = 다운스케일 후 MediaPipe Face Mesh로 추출.
            "bbox"  = 라벨 Lip_bounding_box로 원본 해상도에서 추출(MediaPipe
                      미사용, 훨씬 빠름). 좌표가 없으면 mediapipe로 폴백.
            "auto"  = Lip_bounding_box가 있으면 bbox, 없으면 mediapipe.

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

    # ROI 방식 결정: bbox/auto면 라벨 입술 박스 확보
    lip_boxes, orig_fps = (None, 30.0)
    use_bbox = False
    if roi_mode in ("bbox", "auto"):
        lip_boxes, orig_fps = _extract_lip_boxes(label)
        use_bbox = lip_boxes is not None
        if roi_mode == "bbox" and not use_bbox:
            print("  [경고] Lip_bounding_box 없음 → mediapipe 폴백")
    cap = cv2.VideoCapture(video_path) if use_bbox else None

    # 라벨 좌표계 정합용 회전 감지(가로 저장 vs 세로 라벨)
    rotate = None
    if use_bbox:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate = _detect_rotation(fw, fh, lip_boxes)
        if rotate is not None:
            print(f"  [회전] 프레임 {fw}x{fh} vs 세로 라벨좌표 → CCW 90° 정합")

    base = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    tmp_dir = tempfile.mkdtemp(prefix="prep_")
    try:
        for s in sentences:
            text = s["sentence_text"].strip()
            label_idx = encode_text(text)
            if not label_idx:
                continue
            try:
                if use_bbox:
                    roi_arr = _roi_seq_from_boxes(
                        cap, s["start_time"], s["end_time"], lip_boxes, orig_fps,
                        rotate=rotate)
                else:
                    tmp_clip = os.path.join(tmp_dir, f"{base}_{s['ID']}.mp4")
                    if not _cut_sentence_clip(video_path, s["start_time"],
                                              s["end_time"], tmp_clip, scale_w):
                        continue
                    frames, _fps = resample_to_fps(tmp_clip)  # 25fps 리샘플
                    roi_arr = extract_roi_from_segment(np.stack(frames, axis=0))
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] id={s['ID']} 전처리 실패: {exc}")
                continue
            if roi_arr is None:
                print(f"  [skip] id={s['ID']} {'프레임 없음' if use_bbox else '얼굴 미검출'}")
                continue
            # 메모리 보호: 과도하게 긴 클립은 스킵(0=무제한)
            if max_frames and roi_arr.shape[0] > max_frames:
                print(f"  [skip] id={s['ID']} T({roi_arr.shape[0]}) > max_frames({max_frames})")
                continue
            # CTC 제약: 입력 길이(T) >= 라벨 정렬 최소 길이(인접 중복 자모 고려)
            min_t = ctc_min_input_len(label_idx)
            if roi_arr.shape[0] < min_t:
                print(f"  [skip] id={s['ID']} T({roi_arr.shape[0]}) < CTC_min({min_t}) "
                      f"[label={len(label_idx)}]")
                continue
            out = os.path.join(out_dir, f"{base}_{s['ID']:03d}.npz")
            np.savez_compressed(
                out,
                frames=roi_arr.astype(np.uint8),
                label=np.array(label_idx, dtype=np.int64),
                text=text,
            )
            saved += 1
            print(f"  [ok] id={s['ID']} T={roi_arr.shape[0]} label={len(label_idx)} -> {os.path.basename(out)}")
    finally:
        if cap is not None:
            cap.release()
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
    p.add_argument("--max-sentences", type=int, default=None,
                   help="최대 문장 수(기본 None=전체 문장, 긴 문장 포함)")
    p.add_argument("--scale", type=int, default=640, help="다운스케일 폭(px)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="클립 최대 프레임 상한(0=무제한). 초과 클립은 스킵")
    p.add_argument("--roi", choices=["mediapipe", "bbox", "auto"], default="mediapipe",
                   help="ROI 추출: mediapipe(기본) | bbox(라벨 입술좌표, 빠름) | auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[준비] {os.path.basename(args.video)}")
    n = prepare_video(args.video, args.label, args.out, args.max_sentences,
                      args.scale, args.max_frames, args.roi)
    print(f"[완료] {n}개 문장 클립 저장 → {args.out}")


if __name__ == "__main__":
    main()
