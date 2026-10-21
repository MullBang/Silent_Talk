"""학습 데이터 준비 (AI Hub 립리딩 → 문장 단위 ROI/라벨 캐시).

영상 + 라벨 JSON(Sentence_info)을 받아 각 문장 구간을 잘라 25fps 리샘플 →
MediaPipe 입술 ROI(96×96) 추출 → 텍스트를 자모 인덱스로 인코딩 →
문장별 .npz(frames, label, text)로 저장한다.

토큰화는 ML 라이브러리와 독립적이다. ROI 모듈은 torch를 간접 import한다.
학습은 train.py에서 버전이 검증된 캐시 npz를 로드해 수행한다.

사용:
  python scripts/prepare_data.py \
      --video "D:\\...\\lip_..._001.mp4" \
      --label "data_cache\\labels\\lip_..._001.json" \
      --out data_cache/jamo_v2/clips --max-sentences 3 --scale 640
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import cv2
import numpy as np

# backend 루트 경로 추가
_BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from config import FPS  # noqa: E402 (학습 리샘플 목표 fps=25)
from models.tokenizer import (  # noqa: E402
    encode_text, normalize_text, ctc_min_input_len, tokenizer_metadata,
)
from preprocessing.resampler import resample_to_fps  # noqa: E402
from preprocessing.roi_extractor import (  # noqa: E402
    crop_roi_from_box,
    extract_roi_from_segment,
)

TEMPORAL_METHOD = "video-cfr-timestamps-v1"


def _video_timing_metadata(cap, lip_boxes: list) -> tuple[float, int]:
    """영상의 실제 CFR 메타데이터와 프레임별 bbox 개수를 검증한다."""
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("영상의 실제 FPS가 유효하지 않습니다.")
    if not np.isfinite(raw_count) or raw_count < 1 or raw_count != int(raw_count):
        raise ValueError("영상의 실제 프레임 수가 유효하지 않습니다.")
    frame_count = int(raw_count)
    if len(lip_boxes) != frame_count:
        raise ValueError("영상 프레임 수와 프레임별 bbox 수가 일치하지 않습니다.")
    return fps, frame_count


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


def _extract_lip_boxes(label: dict, coordinate_order: str = "yxyx") -> tuple[list | None, float]:
    """라벨에서 프레임별 입술 박스와 원본 FPS를 추출한다.

    이번 로컬 AI Hub 자료의 xtl_ytl_xbr_ybr 필드는 이름과 달리 실제 값이
    [y1,x1,y2,x2]였다. 영상과 대조한 좌표 순서를 명시하여 [x1,y1,x2,y2]로 바꾼다.
    영상의 90도 회전으로 좌표 순서를 대신 교정하지 않는다.

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
    boxes = np.asarray(lip, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("입술 좌표 형식이 올바르지 않습니다.")
    if coordinate_order == "yxyx":
        boxes = boxes[:, [1, 0, 3, 2]]
    elif coordinate_order != "xyxy":
        raise ValueError("좌표 순서는 yxyx 또는 xyxy여야 합니다.")
    return boxes.tolist(), fps


def _roi_seq_from_boxes(cap, start_sec: float, end_sec: float,
                        lip_boxes: list, orig_fps: float) -> np.ndarray | None:
    """라벨 입술 박스로 [start,end] 구간의 25fps ROI 시퀀스를 만든다.

    영상 자체의 실제 FPS를 사용해 목표 시각의 가장 가까운 프레임을 선택하고,
    읽은 프레임의 실제 PTS와 인덱스를 대조한다. 라벨의 반올림 FPS로 시크하지
    않는다. 프레임별 bbox 수가 실제 영상 프레임 수와 다르거나 선택 프레임이
    CFR 시각을 따르지 않으면 거부한다. MediaPipe 경로와 동일하게 RGB로 저장한다.

    Args:
        cap: 열려 있는 cv2.VideoCapture(원본 해상도).
        start_sec: 문장 시작(초).
        end_sec: 문장 끝(초).
        lip_boxes: 원본 fps 기준 프레임별 [xtl,ytl,xbr,ybr] 리스트.
        orig_fps: 라벨에 기록된 FPS(호환 인자, 시크 계산에는 사용하지 않음).

    Returns:
        (T,96,96,3) uint8 ROI. 프레임을 하나도 못 읽으면 None.
    """
    video_fps, frame_count = _video_timing_metadata(cap, lip_boxes)
    if (not np.isfinite(start_sec) or not np.isfinite(end_sec)
            or start_sec < 0 or end_sec <= start_sec
            or end_sec > frame_count / video_fps + 1e-6):
        raise ValueError("문장 시간이 영상의 실제 재생 범위를 벗어납니다.")
    duration = end_sec - start_sec
    n_out = int(round(duration * FPS))
    if n_out <= 0:
        return None
    # 각 출력 프레임의 원본 프레임 인덱스(타임스탬프 기반, 단조 증가)
    target_times = [start_sec + k / FPS for k in range(n_out)]
    targets = [int(round(target_time * video_fps))
               for target_time in target_times]
    if targets[0] < 0 or targets[-1] >= frame_count:
        raise ValueError("문장 전체에 대응하는 bbox가 없습니다.")

    lo = targets[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)  # 구간 시작으로 시크(fps 변경 아님)
    rois: list[np.ndarray] = []
    cur = lo
    frame = None
    for target_time, need in zip(target_times, targets):
        while cur <= need:
            ok, f = cap.read()
            if not ok:
                return None
            frame = f
            cur += 1
        if frame is None:
            break
        actual_index_after_read = float(cap.get(cv2.CAP_PROP_POS_FRAMES))
        actual_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if (not np.isfinite(actual_index_after_read)
                or abs(actual_index_after_read - (need + 1)) > 1e-6):
            raise ValueError("영상 시크 결과의 프레임 인덱스가 일치하지 않습니다.")
        if (not np.isfinite(actual_time)
                or abs(actual_time - need / video_fps) > 1e-3
                or abs(actual_time - target_time) > 0.5 / video_fps + 1e-6):
            raise ValueError("영상 PTS가 CFR/가장 가까운 프레임 시각과 일치하지 않습니다.")
        x1, y1, x2, y2 = lip_boxes[need]
        height, width = frame.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"영상 범위 밖 bbox: {lip_boxes[need]} / {width}x{height}")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rois.append(crop_roi_from_box(rgb, lip_boxes[need]))
    if not rois:
        return None
    return np.stack(rois, axis=0).astype(np.uint8)


def prepare_video(video_path: str, label_path: str, out_dir: str,
                  max_sentences: int | None = None, scale_w: int = 640,
                  max_frames: int = 0, roi_mode: str = "mediapipe",
                  sentence_ids: list[int] | None = None,
                  provenance: dict[int, dict] | None = None,
                  bbox_coordinate_order: str = "yxyx") -> int:
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
        sentence_ids: 사전 검증된 명세에 지정한 문장만 처리한다.
        provenance: 문장 ID별 화자·발화·라벨 출처 메타데이터.
        bbox_coordinate_order: 실제 라벨 좌표 순서(yxyx 또는 xyxy).

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
    if sentence_ids is not None:
        requested = set(sentence_ids)
        available = {s["ID"] for s in sentences}
        if not requested <= available:
            raise ValueError(f"존재하지 않는 문장 ID: {sorted(requested - available)}")
        sentences = [s for s in sentences if s["ID"] in requested]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    # 먼저 모든 선택 정답을 검사하여, 잘못된 표기를 만난 뒤 부분 캐시를 남기지 않는다.
    normalized_sentences = []
    for sentence in sentences:
        try:
            text = normalize_text(sentence["sentence_text"])
        except ValueError as exc:
            raise ValueError(f"{label_path}, 문장 ID={sentence['ID']}: {exc}") from exc
        normalized_sentences.append((sentence, text, encode_text(text)))

    # ROI 방식 결정: bbox/auto면 라벨 입술 박스 확보
    lip_boxes, orig_fps = (None, 30.0)
    use_bbox = False
    if roi_mode in ("bbox", "auto"):
        lip_boxes, orig_fps = _extract_lip_boxes(label, bbox_coordinate_order)
        use_bbox = lip_boxes is not None
        if roi_mode == "bbox" and not use_bbox:
            print("  [경고] Lip_bounding_box 없음 → mediapipe 폴백")
    cap = cv2.VideoCapture(video_path) if use_bbox else None

    base = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    tmp_dir = tempfile.mkdtemp(prefix="prep_")
    try:
        temporal_metadata = {}
        if use_bbox:
            video_fps, _ = _video_timing_metadata(cap, lip_boxes)
            if not np.isfinite(orig_fps) or orig_fps <= 0:
                raise ValueError("라벨 FPS가 유효하지 않습니다.")
            temporal_metadata = {"temporal_method": TEMPORAL_METHOD,
                                 "video_fps": video_fps, "label_fps": orig_fps}
        for s, text, label_idx in normalized_sentences:
            source_text = s["sentence_text"]
            if not label_idx:
                continue
            try:
                if use_bbox:
                    roi_arr = _roi_seq_from_boxes(
                        cap, s["start_time"], s["end_time"], lip_boxes, orig_fps)
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
            if os.path.exists(out):
                raise FileExistsError(f"기존 캐시를 덮어쓰지 않습니다: {out}")
            extra = (provenance or {}).get(s["ID"], {})
            allowed = {"speaker_id", "utterance_id", "recording_id", "angle",
                       "source_partition", "roi_method", "label_sha256"}
            if not extra.keys() <= allowed:
                raise ValueError("지원하지 않는 출처 필드입니다.")
            np.savez_compressed(
                out,
                frames=roi_arr.astype(np.uint8),
                label=np.array(label_idx, dtype=np.int64),
                text=text,
                source_text=source_text,
                **extra,
                **temporal_metadata,
                **tokenizer_metadata(),
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
    p.add_argument("--out", default="data_cache/jamo_v2/clips", help="npz 출력 디렉토리")
    p.add_argument("--max-sentences", type=int, default=None,
                   help="최대 문장 수(기본 None=전체 문장, 긴 문장 포함)")
    p.add_argument("--scale", type=int, default=640, help="다운스케일 폭(px)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="클립 최대 프레임 상한(0=무제한). 초과 클립은 스킵")
    p.add_argument("--roi", choices=["mediapipe", "bbox", "auto"], default="mediapipe",
                   help="ROI 추출: mediapipe(기본) | bbox(라벨 입술좌표, 빠름) | auto")
    p.add_argument("--bbox-coordinate-order", choices=["yxyx", "xyxy"], default="yxyx",
                   help="이번 AI Hub 로컬 라벨은 yxyx. 다른 자료는 영상과 좌표를 확인 후 지정")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[준비] {os.path.basename(args.video)}")
    n = prepare_video(args.video, args.label, args.out, args.max_sentences,
                      args.scale, args.max_frames, args.roi, bbox_coordinate_order=args.bbox_coordinate_order)
    print(f"[완료] {n}개 문장 클립 저장 → {args.out}")


if __name__ == "__main__":
    main()
