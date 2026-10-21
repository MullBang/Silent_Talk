"""OLKAVS 공식 영상 모델로 영상 파일 하나를 한국어 텍스트로 바꾼다 (2단계 단독 실행기).

두 가상환경을 서브프로세스로 묶는다. 서버에 넣기 전의 명령줄 도구이며 학습·서비스 코드를
바꾸지 않는다. 검증된 실험 코드(experiments/olkavs_official)를 그대로 import 한다.

  roi   : .venv-olkavs-roi (mediapipe 0.10.21)  영상 → 96x96 BGR 입술 프레임 NPZ
  infer : .venv-olkavs     (torch cu128)        NPZ → CTC greedy / hybrid 텍스트 + 시간
  run   : 위 둘을 순서대로 실행하는 드라이버 (현재 인터프리터는 무엇이든 된다)

사용 예:
  python scripts/olkavs_transcribe.py run --video clip.mp4 --out out/clip
  python scripts/olkavs_transcribe.py run --video src.mp4 --start 1.707 --end 3.115 --out out/x
  python scripts/olkavs_transcribe.py infer --frames out/x/roi.npz --device cpu --out out/x_cpu

입술 자르기 규약은 stable_context_square(정사각형·1.5배 문맥·EMA 0.4)이며, 이 규약으로
val20 문자 CER 33.58%·5/20을 얻었다(experiments/olkavs_official/results/step04.json).
로컬 AI Hub 자료는 OLKAVS 학습 분포일 가능성이 높으므로 그 수치는 회귀 검사용이다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OLKAVS = ROOT / "experiments" / "olkavs_official"
UPSTREAM = OLKAVS / "upstream"
CHECKPOINT = OLKAVS / "weights" / "multiview_lip.pt"
CHECKPOINT_SHA256 = "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"
EXPECTED_COMMIT = "af2feda3e3ac91ff3fef2454fcee662366ce9f4b"
ROI_PYTHON = ROOT / ".venv-olkavs-roi" / "Scripts" / "python.exe"
INFER_PYTHON = ROOT / ".venv-olkavs" / "Scripts" / "python.exe"
NATIVE_FPS_NOMINAL = 30.0
GEOMETRY_ARM = "stable_context_square"


def _ensure_path(path: Path) -> None:
    """sys.path에 경로를 한 번만 넣는다 (상주 워커가 요청마다 경로를 쌓지 않게)."""
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha256_file(path: Path) -> str:
    """파일 SHA-256 (1MiB 블록)."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise OSError(f"해시를 계산할 수 없습니다: {path}") from exc


def save_json(path: Path, value: Any) -> None:
    """UTF-8 JSON 저장 (디렉터리 자동 생성)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"JSON을 저장할 수 없습니다: {path}") from exc


# ----------------------------------------------------------------------------------
# roi: 영상 → 입술 프레임 (mediapipe 환경)
# ----------------------------------------------------------------------------------
def select_frame_range(fps: float, frame_count: int, start: float | None, end: float | None) -> tuple[int, int]:
    """시작·끝 초를 공식 선택 규칙(ceil(t*fps))으로 정수 프레임 구간으로 바꾼다."""
    import math
    try:
        first = 0 if start is None else math.ceil(start * fps - 1e-9)
        stop = frame_count if end is None else math.ceil(end * fps - 1e-9)
        first, stop = max(0, first), min(frame_count, stop)
        if stop <= first:
            raise ValueError(f"빈 구간입니다: frames [{first}, {stop})")
        return first, stop
    except (TypeError, ValueError) as exc:
        raise ValueError(f"프레임 구간 계산 실패: {exc}") from exc


EBML_SIGNATURE = b"\x1a\x45\xdf\xa3"
GRID_FPS = 30.0


def is_webm(path: Path) -> bool:
    """EBML(WebM/Matroska) 시그니처 검사. 브라우저 MediaRecorder 기본 형식."""
    try:
        with path.open("rb") as stream:
            return stream.read(4) == EBML_SIGNATURE
    except OSError:
        return False


def iter_native_frames(cap: Any, first: int, stop: int) -> Any:
    """정수 프레임 [first, stop) 을 순서대로 낸다 (AI Hub 등 고정 fps 파일용)."""
    if first > 0 and not cap.set(1, first):  # cv2.CAP_PROP_POS_FRAMES == 1
        raise ValueError("시작 프레임으로 이동하지 못했습니다")
    for index in range(first, stop):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"프레임 {index} 디코딩 실패 (파일이 조기 종료됨)")
        yield index, float(cap.get(0)), frame  # cv2.CAP_PROP_POS_MSEC == 0


class GridSelector:
    """실제 PTS 프레임 흐름에서 1/grid_fps 격자점마다 가장 가까운 프레임을 고른다 (한 프레임 앞보기).

    - 동률이면 앞 프레임. 같은 PTS가 반복되거나 한 칸 이내로 살짝 뒤로 가는 프레임은 건너뛰고 세기만 한다.
    - 한 칸보다 크게 뒤로 가면 손상으로 보고 ValueError.
    - 프레임이 드문드문하면 같은 프레임이 여러 격자점에 쓰일 수 있다(시간축 유지).
    """

    def __init__(self, grid_fps: float, target_ms: float = 0.0, end_ms: float | None = None,
                 stats: dict[str, Any] | None = None):
        self.step = 1000.0 / grid_fps
        self.target = target_ms
        self.end = end_ms
        self.prev: tuple[int, float, Any] | None = None
        self.stats = stats if stats is not None else {}

    def feed(self, item: tuple[int, float, Any]) -> list[tuple[int, float, Any]]:
        """원본 프레임 하나를 넣고, 이번에 확정된 격자 프레임들을 돌려준다."""
        index, pts, _frame = item
        if self.prev is not None and pts <= self.prev[1]:
            if self.prev[1] - pts > self.step:
                raise ValueError(f"PTS가 단조 증가하지 않습니다 (frame {index}: {pts} <= {self.prev[1]})")
            self.stats["pts_non_monotonic_skipped"] = self.stats.get("pts_non_monotonic_skipped", 0) + 1
            return []
        out = []
        while pts >= self.target and (self.end is None or self.target < self.end):
            chosen = item
            if self.prev is not None and abs(self.prev[1] - self.target) <= abs(pts - self.target):
                chosen = self.prev
            out.append(chosen)
            self.target += self.step
        self.prev = item
        return out

    def flush(self) -> list[tuple[int, float, Any]]:
        """파일 끝: 마지막 프레임이 다음 격자점과 반 칸 이내면 그 격자점에 쓴다(반올림 오차로 잃지 않게)."""
        if (self.prev is not None and (self.end is None or self.target < self.end)
                and abs(self.prev[1] - self.target) <= self.step / 2):
            self.target += self.step
            return [self.prev]
        return []


def iter_grid_frames(cap: Any, grid_fps: float, start: float | None, end: float | None,
                     stats: dict[str, Any] | None = None, input_gaps: list[float] | None = None) -> Any:
    """실제 PTS를 기준으로 1/grid_fps 격자마다 가장 가까운 원본 프레임을 낸다 (가변 프레임률용).

    한 프레임만 앞서 보는 스트리밍 방식이라 전체를 메모리에 올리지 않는다(GridSelector 규칙).
    WebM(MediaRecorder)은 fps/프레임 수 메타데이터가 없으므로 이 경로를 쓴다.
    input_gaps(list)를 주면 격자 선택 전 원본 프레임 사이의 양수 PTS 간격(ms)을 모두 덧붙인다
    (프레임당 실수 하나). 메타데이터가 없는 영상의 실제 원본 fps 추정(estimate_fps_from_gaps)용이다.
    """
    end_ms = None if end is None else end * 1000.0
    selector = GridSelector(grid_fps, 0.0 if start is None else start * 1000.0, end_ms, stats)
    index = -1
    previous_pts: float | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        pts = float(cap.get(0))  # cv2.CAP_PROP_POS_MSEC
        if input_gaps is not None:
            if previous_pts is not None and pts > previous_pts:
                input_gaps.append(pts - previous_pts)
            previous_pts = pts if previous_pts is None else max(previous_pts, pts)
        yield from selector.feed((index, pts, frame))
        if end_ms is not None and pts >= end_ms:
            return
    yield from selector.flush()


class VariableFrameRateError(ValueError):
    """고정 프레임률로 보고된 영상의 실제 PTS가 index/fps와 어긋남."""


NATIVE_FPS_CANDIDATES = (30000 / 1001, 30.0)
NATIVE_FPS_TOLERANCE = 0.06
LOW_FPS_THRESHOLD = 27.0


def metadata_usable(fps: float, frame_count: int) -> bool:
    """컨테이너 fps(1~240)와 프레임 수(양수)를 믿을 수 있는지. MediaRecorder WebM은 보통 fps 1000·음수 프레임 수."""
    try:
        return bool(1.0 <= float(fps) <= 240.0 and int(frame_count) > 0)
    except (TypeError, ValueError, OverflowError):
        return False


def estimate_fps_from_gaps(gaps: list[float], min_gaps: int = 2) -> float | None:
    """원본 프레임 사이 양수 PTS 간격(ms)으로 실제 원본 fps를 추정한다. 간격이 모자라면 None.

    중앙값 간격의 ±50% 안에 드는 간격만 평균한다. 중앙값은 프레임 누락·멈춤에 강하고, 평균은 WebM의
    1ms 단위 시각 반올림(15fps → 66/67ms 번갈아)을 없앤다.
    """
    import math
    import statistics
    try:
        clean = [float(g) for g in gaps if g is not None and math.isfinite(float(g)) and float(g) > 0]
        if len(clean) < max(1, min_gaps):
            return None
        median_gap = float(statistics.median(clean))
        inliers = [g for g in clean if 0.5 * median_gap <= g <= 1.5 * median_gap]
        mean_gap = sum(inliers) / len(inliers) if inliers else median_gap
        return 1000.0 / mean_gap if mean_gap > 0 else None
    except (TypeError, ValueError, statistics.StatisticsError):
        return None


def validate_duration_hint(value: Any) -> float | None:
    """진행률 분모로 쓸 영상 길이(초) 힌트를 검사한다. None 또는 유한한 양수만 허용, 나머지는 ValueError."""
    import math
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration_seconds는 양수 실수 또는 null이어야 합니다")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("duration_seconds는 유한한 양수여야 합니다")
    return number


def choose_selection(fps: float, frame_count: int, webm: bool, resample: bool = False) -> dict[str, Any]:
    """프레임 선택 방식을 정한다 (순수 함수).

    모델은 약 30fps 원본 프레임으로 학습·검증됐다. 약 30fps(29.97/30) 고정 프레임률 파일만 원본 정수
    프레임을 그대로 쓰고(검증된 step04 경로), 그 밖에는 실제 PTS 기준 30fps 격자로 다시 뽑는다
    (WebM, 25/60/120fps, 메타데이터 이상, 명시 요청). 27fps 미만은 정보가 부족해 품질 경고를 붙인다.
    여기서는 메타데이터만 본다(fps_source='metadata'). 메타데이터를 쓸 수 없으면 extract_segments가
    읽은 원본 PTS 간격으로 source_fps/low_fps를 다시 정한다(fps_source='decoded_pts').
    """
    metadata_ok = metadata_usable(fps, frame_count)
    near_30 = metadata_ok and min(abs(fps - c) for c in NATIVE_FPS_CANDIDATES) <= NATIVE_FPS_TOLERANCE
    if resample:
        reason = "resample requested"
    elif webm:
        reason = "webm container"
    elif not metadata_ok:
        reason = "invalid fps/frame-count metadata"
    elif not near_30:
        reason = f"source fps {fps:.3f} is not ~30"
    else:
        reason = None
    mode = "native_integer" if reason is None else "pts_grid"
    return {"mode": mode, "rate": fps if mode == "native_integer" else GRID_FPS, "reason": reason,
            "source_fps": fps, "low_fps": bool(metadata_ok and fps < LOW_FPS_THRESHOLD),
            "fps_source": "metadata"}


def iter_all_native_frames(cap: Any, frame_count: int, fps: float | None = None) -> Any:
    """처음부터 순서대로 모든 프레임을 낸다. 메타데이터보다 일찍 끝나도 오류 없이 멈춘다.

    fps를 주면 각 프레임의 실제 PTS가 index*1000/fps에서 반 프레임 넘게 벗어나는지 검사하고,
    벗어나면 VariableFrameRateError를 낸다(가변 프레임률을 고정으로 착각하지 않게).
    """
    index = -1
    half_frame_ms = 500.0 / fps if fps else None
    while frame_count <= 0 or index + 1 < frame_count:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        pts = float(cap.get(0))  # cv2.CAP_PROP_POS_MSEC
        if half_frame_ms is not None and abs(pts - index * 1000.0 / fps) > half_frame_ms:
            raise VariableFrameRateError(f"frame {index}: PTS {pts:.3f}ms != {index * 1000.0 / fps:.3f}ms")
        yield index, pts, frame


def iter_adaptive_frames(cap: Any, frame_count: int, fps: float, selection: dict[str, Any],
                         stats: dict[str, Any]) -> Any:
    """약 30fps 고정 프레임률로 보이는 파일: 원본 정수 프레임을 쓰다가, PTS가 index/fps에서 반 프레임 넘게
    어긋나는 순간부터 같은 스트림을 30fps 격자 선택으로 이어서 읽는다(앞부분 결과를 버리지 않는다).

    전환하면 selection을 mode='native_then_grid', switch_frame, reason으로 갱신한다.
    """
    half_frame_ms = 500.0 / fps
    index, last, grid = -1, None, None
    while True:
        if grid is None and 0 < frame_count <= index + 1:
            break
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        pts = float(cap.get(0))  # cv2.CAP_PROP_POS_MSEC
        if grid is None:
            if abs(pts - index * 1000.0 / fps) <= half_frame_ms:
                last = (index, pts, frame)
                yield last
                continue
            selection.update(mode="native_then_grid", switch_frame=index,
                             reason=f"variable frame rate detected at frame {index} "
                                    f"(PTS {pts:.3f}ms != {index * 1000.0 / fps:.3f}ms)")
            grid = GridSelector(GRID_FPS, (last[1] + 1000.0 / GRID_FPS) if last else 0.0, None, stats)
            grid.prev = last
        yield from grid.feed((index, pts, frame))
    if grid is not None:
        yield from grid.flush()


def rearm_tracker(tracker: Any) -> None:
    """검출 실패로 멈춘 AutomaticLipRoi를 다시 얼굴을 찾는 상태로 되돌린다 (검출기 인스턴스는 유지).

    검증 규약의 '실패 시 그 발화 전체 거부'는 짧은 발화 단위 규칙이다. 서비스는 긴 영상을 다루므로
    실패한 범위만 실패 구간으로 보고하고, 이후 프레임에서 다시 검출을 시도한다(미래 프레임은 쓰지 않음).
    """
    tracker.failed = False
    tracker.failure_reason = None
    tracker.last_box_xyxy = None
    tracker.consecutive_miss = 0


def motion_score(previous: Any, current: Any) -> float:
    """연속한 두 입술 crop의 움직임 정도 (24x24 회색조 평균 절대차). 작을수록 입이 멈춰 있다."""
    import numpy as np
    a = previous[::4, ::4].astype(np.float32).mean(axis=2)
    b = current[::4, ::4].astype(np.float32).mean(axis=2)
    return float(np.abs(a - b).mean())


def best_cut(motion: list[float], low: int, high: int, smooth: int = 5) -> int:
    """[low, high] 범위에서 주변 smooth개 평균 움직임이 가장 작은 위치(다음 창의 첫 프레임 번호)를 고른다."""
    if not motion:
        return high
    n = len(motion)
    low, high = max(1, min(low, high)), max(1, min(high, n))
    half = smooth // 2
    best_index, best_key = high, (float("inf"), float("inf"))
    for i in range(low, high + 1):
        # 끝에서 창이 잘리면 가장자리 값을 반복한다(잘린 창이 먼 곳의 작은 값 쪽으로 치우치지 않게).
        window = [motion[min(max(j, 0), n - 1)] for j in range(i - half, i + half + 1)]
        key = (sum(window) / len(window), motion[min(i, n - 1)])  # 동률이면 그 지점 자체가 조용한 쪽
        if key < best_key:
            best_index, best_key = i, key
    return best_index


# ----------------------------------------------------------------------------------
# 입 움직임 기반 발화 구간 검출 (VAD, 6단계)
# ----------------------------------------------------------------------------------
# 입 벌림 비율 = d_raw / W. CLAUDE.md·frontend/src/webcamUtils.js와 같은 정의(픽셀 좌표).
VAD_TOP_IDX = (13, 14, 312, 311, 310, 415)     # 윗입술 내측 → 평균점
VAD_BOT_IDX = (17, 18, 84, 181, 180, 314)      # 아랫입술 → 평균점
VAD_WIDTH_IDX = (234, 454)                     # 좌우 광대(얼굴 폭 W)
VAD_REQUIRED_MEDIAPIPE = "0.10.21"
# 보정값(2026-10-20, 보정 담당). 근거: experiments/stage6_vad_calibration_20261020/calibration.json
# (scripts/calibrate_vad.py). AI Hub 6명·6녹화(C297_A_008, C002_A_004, C017_A_001, C032_A_001, C282_A_003,
# C312_A_002; 각 약 5분, 1080p 29.97fps, 문장 라벨 269개)에서 실제 추적기로 비율을 뽑아 144개 후보를 비교했다.
# 이전 임시값(문턱 0.008·창 0.5초·메움 0.4초, C297 한 명 기준)은 입 움직임이 작은 화자(C002·C282)에서
# 문장 안 재현율 0.73·0.83, 녹화 모의에서 문장 앞을 최대 1.7초·끝을 최대 1.2초 잘라 냈다.
# 선정 조건: 라벨 문장·발화와 1초 넘게 겹치는 6초 창·문장±2.5초 녹화 모의가 하나도 '발화 없음'이 되지 않고,
# 문장 안 프레임 재현율 ≥ 0.97(라벨 0.975, 소리 나는 프레임 0.989)인 후보 중 먼 무발화 활성 비율이 가장
# 낮은 것(0.154). 녹화 모의 269개에서 여유 0.2/0.3초로도 문장이 잘리지 않았고(최소 여유 0.09/0.13초),
# 웹캠·다른 사람에 대비해 여유를 0.4/0.5초로 넉넉히 뒀다(말을 지우는 쪽보다 남기는 쪽을 택함).
# 대가: 말하지 않을 때도 입을 움직이는 화자(C002·C017)는 무발화 프레임의 약 40%가 활성으로 잡혀, 문장 사이
# 3초 무발화 모의 37개 중 26개만 '발화 없음'이 된다. 정지 영상(입이 전혀 움직이지 않음)은 발화 없음이다.
# 고해상도 AI Hub 자료 기준이므로 팀 웹캠 파일럿 녹화로 scripts/calibrate_vad.py를 다시 돌려 확인한다.
# 모든 값은 결과 "vad" 프로필에 기록되며 함수는 호출 시점의 모듈 상수를 읽는다.
VAD_MEDIAN_FRAMES = 3            # 비율 계열 중앙값 필터 폭(프레임, 홀수)
VAD_WINDOW_SECONDS = 0.4         # 움직임 범위를 재는 중심 창 길이
VAD_MIN_WINDOW_KNOWN = 0.5       # 창 안에서 비율을 아는 프레임 비율 하한(미만이면 그 프레임은 비활성)
VAD_ACTIVITY_THRESHOLD = 0.005   # 창 안 비율 (최대-최소) ≥ 이 값이면 말하는 프레임
VAD_FILL_GAP_SECONDS = 0.8       # 활성 사이의 이보다 짧은 비활성은 메운다(문장 안 짧은 멈춤)
VAD_MIN_BURST_SECONDS = 0.3      # 이보다 짧은 활성 덩어리는 버린다(깜빡임·튐)
VAD_PRE_PAD_SECONDS = 0.4        # 발화 구간 앞 여유(녹화 모드 자르기; 창 범위가 이미 약 0.2초 넓힌다)
VAD_POST_PAD_SECONDS = 0.5       # 발화 구간 뒤 여유
VAD_MIN_SPEECH_SECONDS = 0.4     # 활성 시간이 이보다 짧으면 발화 없음
VAD_MIN_COVERAGE = 0.5           # 비율을 아는 프레임 비율이 이보다 낮으면 VAD를 끈다(판정 불가)
NO_SPEECH_REASON = "no_mouth_activity"


def vad_profile() -> dict[str, Any]:
    """실제로 쓰는 VAD 상수 묶음 (결과·로그 기록용)."""
    return {"profile_id": "mouth_ratio_range_v1",
            "ratio": "d_raw/W pixel coords; d_raw=|mean(top)-mean(bot)|, W=|p234-p454|",
            "top_idx": list(VAD_TOP_IDX), "bot_idx": list(VAD_BOT_IDX), "width_idx": list(VAD_WIDTH_IDX),
            "median_frames": VAD_MEDIAN_FRAMES, "window_seconds": VAD_WINDOW_SECONDS,
            "min_window_known": VAD_MIN_WINDOW_KNOWN, "activity_threshold": VAD_ACTIVITY_THRESHOLD,
            "fill_gap_seconds": VAD_FILL_GAP_SECONDS, "min_burst_seconds": VAD_MIN_BURST_SECONDS,
            "pre_pad_seconds": VAD_PRE_PAD_SECONDS, "post_pad_seconds": VAD_POST_PAD_SECONDS,
            "min_speech_seconds": VAD_MIN_SPEECH_SECONDS, "min_coverage": VAD_MIN_COVERAGE,
            "ratio_source": "detected frames only (previous-bbox frames unknown)"}


def _finite_or_none(value: Any) -> float | None:
    """유한한 실수면 float, 아니면 None."""
    import math
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def mouth_ratio(points_xy: Any) -> float | None:
    """랜드마크 픽셀 좌표 배열(N,2; 행 번호 = 랜드마크 번호)에서 입 벌림 비율 d_raw/W.

    점이 모자라거나 비유한 값이거나 얼굴 폭이 0이면 None.
    """
    import numpy as np
    try:
        points = np.asarray(points_xy, dtype=np.float64)
        need = max(VAD_TOP_IDX + VAD_BOT_IDX + VAD_WIDTH_IDX)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] <= need:
            return None
        used = points[list(VAD_TOP_IDX + VAD_BOT_IDX + VAD_WIDTH_IDX)]
        if not np.isfinite(used).all():
            return None
        top = points[list(VAD_TOP_IDX)].mean(axis=0)
        bot = points[list(VAD_BOT_IDX)].mean(axis=0)
        width = float(np.linalg.norm(points[VAD_WIDTH_IDX[0]] - points[VAD_WIDTH_IDX[1]]))
        if width <= 1e-9:
            return None
        return _finite_or_none(np.linalg.norm(top - bot) / width)
    except (TypeError, ValueError, IndexError):
        return None


def face_mouth_ratio(face: Any, width: int, height: int) -> float | None:
    """MediaPipe 얼굴 하나(정규화 좌표 landmark 목록)를 원본 픽셀 좌표로 바꿔 입 벌림 비율을 구한다."""
    try:
        landmarks = face.landmark
        need = max(VAD_TOP_IDX + VAD_BOT_IDX + VAD_WIDTH_IDX)
        points = [(0.0, 0.0)] * (need + 1)
        for i in set(VAD_TOP_IDX + VAD_BOT_IDX + VAD_WIDTH_IDX):
            points[i] = (float(landmarks[i].x) * width, float(landmarks[i].y) * height)
        return mouth_ratio(points)
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def select_lip_face(faces: Any, width: int, height: int) -> Any:
    """AutomaticLipRoi와 같은 얼굴(입술 상자 면적 최대)을 고른다. 동결 함수 largest_lip_points를 그대로 써서
    고른 입술 점과 정확히 같은 점을 가진 얼굴을 돌려준다. 없으면 None."""
    import numpy as np
    _ensure_path(OLKAVS)
    from automatic_roi import LIP_INDICES, largest_lip_points
    try:
        best = largest_lip_points(faces, width, height)
        if best is None:
            return None
        scale = np.array([width, height], dtype=np.float32)
        for face in faces or ():
            try:
                points = np.array([(face.landmark[i].x, face.landmark[i].y) for i in LIP_INDICES],
                                  dtype=np.float32) * scale
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if points.shape == best.shape and np.array_equal(points, best):
                return face
        return None
    except (TypeError, ValueError):
        return None


class RecordingDetector:
    """검출기(process/close)를 감싸 마지막 결과를 보관한다. 동결된 AutomaticLipRoi를 고치지 않고 랜드마크를 본다."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.last_result: Any = None

    def process(self, rgb: Any) -> Any:
        """원래 검출기를 호출하고 결과를 기억한다 (예외는 그대로 올린다)."""
        self.last_result = None
        self.last_result = self.inner.process(rgb)
        return self.last_result

    def close(self) -> None:
        """원래 검출기를 해제한다."""
        self.last_result = None
        self.inner.close()


_VAD_TRACKER_CLASS: Any = None


def vad_tracker_class() -> Any:
    """AutomaticLipRoi 하위 클래스: 처리 후 tracker.mouth_ratio(같은 얼굴, 검출 프레임만)를 노출한다.

    검출기를 주입하면 AutomaticLipRoi가 mediapipe 버전 검사를 건너뛰므로 여기서 직접 검사한다.
    FaceMesh 설정은 동결 프로필(automatic_roi.profile())의 값을 그대로 쓴다.
    """
    global _VAD_TRACKER_CLASS
    if _VAD_TRACKER_CLASS is not None:
        return _VAD_TRACKER_CLASS
    _ensure_path(OLKAVS)
    from automatic_roi import AutomaticLipRoi, profile as detector_profile

    class MouthRatioLipRoi(AutomaticLipRoi):
        """입 벌림 비율을 함께 내는 입술 추적기 (입술 상자·crop·실패 규칙은 AutomaticLipRoi 그대로)."""

        def __init__(self, detector: Any = None):
            started = time.perf_counter()
            if detector is None:
                import mediapipe as mp
                if mp.__version__ != VAD_REQUIRED_MEDIAPIPE:
                    raise RuntimeError(f"입술 추출 규약은 mediapipe {VAD_REQUIRED_MEDIAPIPE}이 필요합니다: "
                                       f"{mp.__version__}")
                spec = detector_profile()
                detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=spec["static_image_mode"], max_num_faces=spec["max_num_faces"],
                    refine_landmarks=spec["refine_landmarks"],
                    min_detection_confidence=spec["min_detection_confidence"],
                    min_tracking_confidence=spec["min_tracking_confidence"])
            self.mouth_ratio: float | None = None
            super().__init__(detector=RecordingDetector(detector))
            self.initialization_seconds = time.perf_counter() - started

        def process(self, frame_bgr: Any) -> Any:
            """AutomaticLipRoi.process 후, 이번 프레임이 '검출'이면 같은 얼굴의 입 벌림 비율을 기록한다."""
            self.mouth_ratio = None
            roi = super().process(frame_bgr)
            try:
                if self.frame_records and self.frame_records[-1].get("status") == "detected":
                    height, width = frame_bgr.shape[:2]
                    faces = getattr(self.detector.last_result, "multi_face_landmarks", None)
                    face = select_lip_face(faces, width, height)
                    self.mouth_ratio = face_mouth_ratio(face, width, height) if face is not None else None
            except (AttributeError, TypeError, ValueError):
                self.mouth_ratio = None
            return roi

    _VAD_TRACKER_CLASS = MouthRatioLipRoi
    return MouthRatioLipRoi


def _median_filter(values: list[float | None], width: int) -> list[float | None]:
    """알려진 값만으로 중심 중앙값 필터 (모르는 프레임은 그대로 None)."""
    import statistics
    half = max(0, width // 2)
    if half == 0:
        return list(values)
    out: list[float | None] = []
    n = len(values)
    for i, value in enumerate(values):
        if value is None:
            out.append(None)
            continue
        known = [v for v in values[max(0, i - half):min(n, i + half + 1)] if v is not None]
        out.append(float(statistics.median(known)))
    return out


def rolling_ratio_range(ratios: list[float | None], half: int, min_known: float | None = None
                        ) -> list[float | None]:
    """중심 창 [i-half, i+half]의 (최대-최소). 창 안 아는 값이 (2*half+1)*min_known 미만이면 None.

    min_known 생략 시 호출 시점의 VAD_MIN_WINDOW_KNOWN.
    """
    min_known = VAD_MIN_WINDOW_KNOWN if min_known is None else min_known
    n = len(ratios)
    need = max(1, int(round((2 * half + 1) * min_known)))
    out: list[float | None] = []
    for i in range(n):
        known = [v for v in ratios[max(0, i - half):min(n, i + half + 1)] if v is not None]
        out.append(max(known) - min(known) if len(known) >= need else None)
    return out


def fill_and_filter(active: list[bool], fill_frames: int, min_burst_frames: int) -> list[bool]:
    """활성 사이의 짧은 비활성(< fill_frames)을 메우고, 그 뒤 짧은 활성 덩어리(< min_burst_frames)를 지운다."""
    out = list(active)
    n = len(out)
    runs: list[tuple[bool, int, int]] = []  # (값, 시작, 끝 exclusive)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        runs.append((out[i], i, j))
        i = j
    for k, (value, start, stop) in enumerate(runs):
        if not value and 0 < k < len(runs) - 1 and stop - start < fill_frames:
            for p in range(start, stop):
                out[p] = True
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if j - i < min_burst_frames:
            for p in range(i, j):
                out[p] = False
        i = j
    return out


def speech_activity(ratios: list[float | None], rate: float) -> list[bool]:
    """프레임별 말하는 중 여부. 중앙값 필터 → 0.5초 중심 창 범위 ≥ 문턱 → 짧은 틈 메움·짧은 덩어리 제거."""
    if rate <= 0:
        raise ValueError("rate는 양수여야 합니다")
    clean = [_finite_or_none(v) for v in ratios]
    smoothed = _median_filter(clean, VAD_MEDIAN_FRAMES)
    half = max(1, int(round(VAD_WINDOW_SECONDS * rate / 2)))
    spread = rolling_ratio_range(smoothed, half)
    raw = [s is not None and s >= VAD_ACTIVITY_THRESHOLD for s in spread]
    return fill_and_filter(raw, max(0, int(round(VAD_FILL_GAP_SECONDS * rate))),
                           max(1, int(round(VAD_MIN_BURST_SECONDS * rate))))


def speech_span(active: list[bool], rate: float, pre_seconds: float | None = None,
                post_seconds: float | None = None) -> tuple[int, int, int, int] | None:
    """첫 활성~마지막 활성 프레임과 앞뒤 여유를 더한 구간. (first, last, padded_first, padded_last) 포함 범위.

    활성이 없으면 None. 여유를 생략하면 호출 시점의 VAD_PRE/POST_PAD_SECONDS를 쓴다(보정 시 상수만 바꾸면 됨).
    """
    pre_seconds = VAD_PRE_PAD_SECONDS if pre_seconds is None else pre_seconds
    post_seconds = VAD_POST_PAD_SECONDS if post_seconds is None else post_seconds
    hits = [i for i, value in enumerate(active) if value]
    if not hits:
        return None
    first, last = hits[0], hits[-1]
    pre, post = int(round(pre_seconds * rate)), int(round(post_seconds * rate))
    return first, last, max(0, first - pre), min(len(active) - 1, last + post)


def active_seconds(active: list[bool], positions: list[int], rate: float) -> float:
    """주어진 프레임 위치들 중 활성 프레임 수 / rate."""
    return sum(1 for p in positions if 0 <= p < len(active) and active[p]) / rate


def _remove_file(path: Any) -> None:
    """임시 NPZ 삭제 (없으면 무시)."""
    try:
        if path:
            os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[roi] NPZ 삭제 실패: {exc}", file=sys.stderr, flush=True)


def _no_speech_entry(start_ms: int, end_ms: int, frames: int, seconds: float, ratio_coverage: float
                     ) -> dict[str, Any]:
    """녹화 모드의 no_speech 구간 하나 (NPZ 없음, 인식하지 않음)."""
    return {"index": 0, "start_ms": int(start_ms), "end_ms": int(end_ms), "frames": int(frames),
            "status": "no_speech", "reason": NO_SPEECH_REASON, "npz": None,
            "speech": {"active_seconds": seconds, "ratio_coverage": ratio_coverage}}


def _load_window_frames(path: Any, expected: int) -> Any:
    """창 NPZ의 frames 배열을 읽는다. 길이가 창 위치 수와 다르면 ValueError."""
    import numpy as np
    try:
        with np.load(path) as data:
            frames = np.asarray(data["frames"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"창 NPZ를 읽지 못했습니다: {exc}") from exc
    if frames.shape[0] != expected:
        raise ValueError(f"창 NPZ 길이({frames.shape[0]})가 창 프레임 수({expected})와 다릅니다")
    return frames


def apply_speech_activity(segments: list[dict[str, Any]], window_meta: dict[int, dict[str, Any]],
                          ratios: list[float | None], pts_series: list[float], rate: float, frame_ms: float,
                          utterance: bool, min_frames: int, max_frames: int | None = None) -> dict[str, Any]:
    """추출이 끝난 구간 목록에 입 움직임 판정을 적용한다(segments를 제자리에서 바꾼다). 반환: 최상위 speech.

    - 비율을 아는 프레임이 ok 창 프레임의 VAD_MIN_COVERAGE 미만이면 VAD를 끈다(5단계와 같은 결과).
    - 업로드(utterance=False): 자르지 않는다. 활성 시간이 VAD_MIN_SPEECH_SECONDS 미만인 ok 창은 no_speech
      (NPZ 삭제, 인식 안 함). 창 자체의 비율 범위가 모자라면 판정하지 않고 ok로 둔다.
    - 녹화(utterance=True): 녹화 한 건을 한 발화로 판정하지만 모델 창(최대 segment_seconds, 기본 12초)은
      그대로다. 12초보다 긴 녹화(30초 실험 모드 등)는 여전히 여러 창으로 나뉠 수 있다.
      * 발화가 없을 때: 실패 구간(얼굴·입술 미관측)이 없으면 읽은 영상 전체를 덮는 no_speech 하나.
        실패 구간이 있으면 실패 구간은 그대로 두고, 그 사이의 관측된 ok 창들(이어진 창은 하나로 합침)만
        no_speech로 바꾼다(시간 순서 유지, 번호 다시 매김, NPZ 삭제). 보지 못한 구간을 '발화 없음'이라 하지 않는다.
      * 발화가 있을 때: 남길 창은 여유를 더하기 전의 활성 프레임으로 정한다. 활성 프레임이
        VAD_MIN_BURST_SECONDS 이상인 창만 남기고, 여유(앞 VAD_PRE_PAD·뒤 VAD_POST_PAD)는 남긴 창을
        그 창의 활성 범위 밖으로 넓히는 데만 쓴다. 활성이 그보다 짧은 창은, 그 활성이 경계를 넘어 이어진
        남긴 이웃 창이 있고 합친 길이가 max_frames + min_frames 이하면 이웃에 붙이고(프레임 이어 붙임),
        아니면 버린다(NPZ 삭제). 남긴 창이 하나도 없으면(활성이 창 경계로 쪼개짐) 활성이 가장 많은 창을 남긴다.
        여유를 포함한 전체 발화 구간과 겹치지 않는 실패 구간은 버린다. 너무 짧게 잘린 창은 창 안에서
        min_frames까지 넓힌다. max_frames가 None이면 이웃에 붙이지 않는다.
    """
    import numpy as np
    speech: dict[str, Any] = {"enabled": False, "detected": None, "start_ms": None, "end_ms": None,
                              "active_seconds": 0.0, "trimmed_head_ms": 0, "trimmed_tail_ms": 0}
    ok_segments = [s for s in segments if s["status"] == "ok"]

    def coverage(positions: list[int]) -> float:
        return (sum(1 for p in positions if ratios[p] is not None) / len(positions)) if positions else 0.0

    ok_positions = [p for s in ok_segments for p in window_meta[s["index"]]["positions"]]
    total_coverage = coverage(ok_positions)
    enabled = bool(ok_positions) and total_coverage >= VAD_MIN_COVERAGE
    speech["enabled"] = enabled
    if not enabled:
        for seg in ok_segments:
            seg["speech"] = {"active_seconds": None,
                             "ratio_coverage": coverage(window_meta[seg["index"]]["positions"])}
        return speech

    active = speech_activity(ratios, rate)
    in_ok = [False] * len(active)
    for p in ok_positions:
        in_ok[p] = True
    active = [a and b for a, b in zip(active, in_ok)]  # 창으로 쓰이지 않은 프레임의 움직임은 세지 않는다
    hits = [i for i, value in enumerate(active) if value]
    speech["active_seconds"] = len(hits) / rate
    if hits:
        speech["start_ms"] = int(round(pts_series[hits[0]]))
        speech["end_ms"] = int(round(pts_series[hits[-1]] + frame_ms))

    if not utterance:
        verdicts: list[float | None] = []
        for seg in ok_segments:
            positions = window_meta[seg["index"]]["positions"]
            cov = coverage(positions)
            seconds = active_seconds(active, positions, rate) if cov >= VAD_MIN_COVERAGE else None
            seg["speech"] = {"active_seconds": seconds, "ratio_coverage": cov}
            verdicts.append(seconds)
            if seconds is not None and seconds < VAD_MIN_SPEECH_SECONDS:
                _remove_file(seg.get("npz"))
                seg.update(status="no_speech", reason=NO_SPEECH_REASON, npz=None)
        if any(v is not None and v >= VAD_MIN_SPEECH_SECONDS for v in verdicts):
            speech["detected"] = True
        else:
            speech["detected"] = None if any(v is None for v in verdicts) else False
        return speech

    ordered = sorted(segments, key=lambda s: (s["start_ms"], s["end_ms"]))  # 기록 순서가 곧 시간 순서(안정 정렬)

    # ---- 녹화 모드: 발화 없음 ----
    if speech["active_seconds"] < VAD_MIN_SPEECH_SECONDS:
        speech["detected"] = False
        for seg in ok_segments:
            _remove_file(seg.get("npz"))
        if not any(s["status"] == "failed" for s in segments):  # 전부 관측됨 → 영상 전체가 발화 없음
            start_pts, end_pts = pts_series[0], pts_series[-1]
            segments[:] = [_no_speech_entry(int(round(start_pts)), int(round(end_pts + frame_ms)), len(pts_series),
                                            speech["active_seconds"], total_coverage)]
            return speech
        out: list[dict[str, Any]] = []
        group: list[dict[str, Any]] = []

        def flush_group() -> None:
            if not group:
                return
            positions = [p for s in group for p in window_meta[s["index"]]["positions"]]
            out.append(_no_speech_entry(group[0]["start_ms"], group[-1]["end_ms"], positions[-1] - positions[0] + 1,
                                        active_seconds(active, positions, rate), coverage(positions)))
            group.clear()

        for seg in ordered:
            if seg["status"] == "ok":
                group.append(seg)  # 실패 구간 없이 이어진 ok 창은 no_speech 하나로 합친다
                continue
            flush_group()
            out.append(seg)  # 얼굴·입술을 보지 못한 구간은 그대로 보고한다
        flush_group()
        for i, seg in enumerate(out):
            seg["index"] = i
        segments[:] = out
        return speech

    # ---- 녹화 모드: 발화 있음 ----
    speech["detected"] = True
    span = speech_span(active, rate)
    if span is None:  # hits가 있으므로 생기지 않는다
        raise ValueError("발화 구간 계산 실패")
    _first, _last, lo, hi = span
    span_start_ms, span_end_ms = pts_series[lo], pts_series[hi] + frame_ms
    old_first_start = min(s["start_ms"] for s in ok_segments)
    old_last_end = max(s["end_ms"] for s in ok_segments)
    burst_frames = max(1, int(round(VAD_MIN_BURST_SECONDS * rate)))
    pre_frames = int(round(VAD_PRE_PAD_SECONDS * rate))
    post_frames = int(round(VAD_POST_PAD_SECONDS * rate))
    merge_limit = None if max_frames is None else int(max_frames) + int(min_frames)

    oks = [s for s in ordered if s["status"] == "ok"]
    infos = [{"seg": s, "positions": list(window_meta[s["index"]]["positions"]),
              "detected": list(window_meta[s["index"]]["detected"])} for s in oks]
    for info in infos:
        info["active"] = sum(1 for p in info["positions"] if active[p])
    keep = [info["active"] >= burst_frames for info in infos]
    if not any(keep):  # 활성이 창 경계로 쪼개져 어느 창도 덩어리 기준을 넘지 못함 → 가장 많은 창
        keep[max(range(len(infos)), key=lambda k: infos[k]["active"])] = True
    groups: dict[int, list[int]] = {k: [k] for k, flag in enumerate(keep) if flag}  # 남긴 창 → 구성 창(시간순)

    def trim_range(members: list[int]) -> tuple[int, int] | None:
        """구성 창들을 이어 붙인 위치 목록에서 (첫 활성 - 앞 여유, 마지막 활성 + 뒤 여유) 포함 범위."""
        combined = [p for m in members for p in infos[m]["positions"]]
        marks = [k for k, p in enumerate(combined) if active[p]]
        if not marks:
            return None
        return max(0, marks[0] - pre_frames), min(len(combined) - 1, marks[-1] + post_frames)

    def touches(left: int, right: int) -> bool:
        """두 ok 창이 프레임이 끊기지 않고 이어지며 활성이 그 경계를 넘어 이어지는가."""
        a_pos, b_pos = infos[left]["positions"], infos[right]["positions"]
        return bool(a_pos and b_pos and a_pos[-1] + 1 == b_pos[0] and active[a_pos[-1]] and active[b_pos[0]])

    merged: set[int] = set()
    for k, info in enumerate(infos):
        if keep[k] or info["active"] == 0 or merge_limit is None:
            continue
        candidates = []
        if k > 0 and keep[k - 1] and touches(k - 1, k):
            candidates.append((k - 1, groups[k - 1] + [k]))   # 앞 창의 꼬리
        if k + 1 < len(infos) and keep[k + 1] and touches(k, k + 1):
            candidates.append((k + 1, [k] + groups[k + 1]))   # 뒤 창의 머리
        for owner, members in candidates:
            rng = trim_range(members)
            if rng is not None and rng[1] - rng[0] + 1 <= merge_limit:
                groups[owner] = members
                merged.add(k)
                break

    kept: list[dict[str, Any]] = []
    ok_number = {id(info["seg"]): k for k, info in enumerate(infos)}
    for seg in ordered:
        if seg["status"] == "failed":
            if seg["end_ms"] <= span_start_ms or seg["start_ms"] >= span_end_ms:
                continue  # 발화 구간 밖의 얼굴 미검출 → 버림
            kept.append(seg)
            continue
        k = ok_number[id(seg)]
        if k not in groups:
            if k not in merged:
                _remove_file(seg.get("npz"))  # 활성이 덩어리 기준에 못 미치는 창 → 인식하지 않음
            continue
        members = groups[k]
        positions = [p for m in members for p in infos[m]["positions"]]
        detected = [d for m in members for d in infos[m]["detected"]]
        rng = trim_range(members)
        if rng is None:  # 남긴 창에는 활성이 있으므로 생기지 않는다
            raise ValueError("발화 창 범위 계산 실패")
        a, b = rng
        need = min(min_frames, len(positions))
        while b - a + 1 < need:  # 너무 짧게 잘리면 창 안에서 양쪽으로 넓힌다(인식기 최소 길이 확보)
            if a > 0:
                a -= 1
            if b - a + 1 < need and b < len(positions) - 1:
                b += 1
        if len(members) > 1 or (a, b) != (0, len(positions) - 1):
            try:
                parts = [_load_window_frames(infos[m]["seg"]["npz"], len(infos[m]["positions"])) for m in members]
                frames = np.concatenate(parts, axis=0)[a:b + 1]
                np.savez_compressed(seg["npz"], frames=frames.astype(np.uint8))
            except (OSError, ValueError) as exc:
                raise ValueError(f"발화 구간 자르기 실패: {exc}") from exc
            for m in members:
                if infos[m]["seg"] is not seg:
                    _remove_file(infos[m]["seg"].get("npz"))
                    infos[m]["seg"]["npz"] = None
            seg["start_ms"] = int(round(pts_series[positions[a]]))
            seg["end_ms"] = int(round(pts_series[positions[b]] + frame_ms))
            seg["frames"] = b - a + 1
            sub = detected[a:b + 1]
            seg["detection"] = {"detected": sum(sub), "previous_bbox": len(sub) - sum(sub)}
            positions = positions[a:b + 1]
            if len(members) > 1:
                seg["merged_windows"] = len(members)
        seg["speech"] = {"active_seconds": active_seconds(active, positions, rate),
                         "ratio_coverage": coverage(positions)}
        kept.append(seg)
    for i, seg in enumerate(kept):
        seg["index"] = i
    segments[:] = kept
    new_ok = [s for s in kept if s["status"] == "ok"]
    if new_ok:
        speech["trimmed_head_ms"] = max(0, min(s["start_ms"] for s in new_ok) - old_first_start)
        speech["trimmed_tail_ms"] = max(0, old_last_end - max(s["end_ms"] for s in new_ok))
    return speech


class WindowBuilder:
    """연속 검출 구간(run)의 crop 흐름을 모델 입력 창으로 나눈다.

    - 창 길이는 최대 max_frames. 넘으면 [max_frames - search_frames, max_frames]에서 입 움직임이 가장
      작은 지점에서 자른다. 자르는 위치는 항상 min_frames 이상이라 짧은 조각 창이 생기지 않는다.
    - run이 끝날 때 마지막 창이 min_frames보다 짧으면 직전 창에 붙인다(최대 max_frames + min_frames).
    - crop(96x96x3)만 들고 있으므로 메모리는 최대 약 2창 분량이다.
    """

    def __init__(self, max_frames: int, search_frames: int, min_frames: int, emit: Any):
        if max_frames < 2:
            raise ValueError("max_frames는 2 이상이어야 합니다")
        self.max_frames = max_frames
        self.min_frames = max(1, min(min_frames, max_frames // 2))
        self.search_frames = max(0, min(search_frames, max_frames - self.min_frames))
        self.emit = emit
        self.buffer: list[dict[str, Any]] = []
        self.pending: list[dict[str, Any]] | None = None

    def push(self, item: dict[str, Any]) -> None:
        """crop 하나를 넣는다. item: {crop, pts, motion, status}."""
        self.buffer.append(item)
        if self.pending is not None and len(self.buffer) >= self.min_frames:
            self._flush_pending()
        if len(self.buffer) > self.max_frames:
            motion = [entry["motion"] for entry in self.buffer]
            cut = best_cut(motion, self.max_frames - self.search_frames, self.max_frames)
            window, self.buffer = self.buffer[:cut], self.buffer[cut:]
            self._flush_pending()
            self.pending = window
            if len(self.buffer) >= self.min_frames:
                self._flush_pending()

    def end_run(self) -> None:
        """run이 끝났다: 남은 crop을 창으로 내보낸다(짧은 꼬리는 직전 창에 합침)."""
        if self.buffer:
            if self.pending is not None and len(self.buffer) < self.min_frames:
                self.pending = self.pending + self.buffer
            else:
                self._flush_pending()
                self.pending = self.buffer
        self.buffer = []
        self._flush_pending()

    def _flush_pending(self) -> None:
        if self.pending:
            self.emit(self.pending)
        self.pending = None


def extract_segments(video: Path, out_dir: Path, segment_seconds: float = 12.0, min_window_seconds: float = 1.0,
                     progress: Any = None, resample: bool = False, cut_search_seconds: float = 6.0,
                     min_gap_seconds: float = 0.5, min_run_seconds: float = 0.3,
                     tracker_factory: Any = None, geometry_factory: Any = None,
                     utterance: bool = False, duration_hint_seconds: float | None = None) -> dict[str, Any]:
    """영상 전체를 스트리밍으로 읽어 입술 96x96 BGR NPZ 창들과 얼굴 미검출 구간을 만든다 (서비스 워커용).

    - 검출기(AutomaticLipRoi) 하나로 영상 전체를 인과적으로 추적한다. 직전 상자로 채운 미검출
      프레임(1~11장)은 다음 검출이 올 때까지 보류한다. 다음 검출이 오면 run에 넣고, 12장 연속
      미검출로 추적을 잃으면 실패 구간에 넣는다(얼굴 없는 crop이 창에 섞이지 않게).
    - 검출 프레임이 min_run_seconds 미만인 run(깜빡이는 검출, 연속 기하 오류)은 앞뒤 실패 구간과
      합쳐 하나의 실패 구간이 된다. min_gap_seconds 미만의 고립된 실패 구간은 보고하지 않는다.
    - 연속 run마다 새 기하 상태(stable_context_square EMA)를 쓰고, WindowBuilder로 최대
      segment_seconds 창으로 나눈다. 창마다 인식기가 발화 단위 정규화를 따로 한다.
    - 약 30fps가 아니거나 가변 프레임률이면 실제 PTS 기준 30fps 격자로 읽는다(도중 전환 포함).
    - 영상 중간에 해상도가 바뀌면 run 경계로 보고 검출기를 새로 만든다.
    - 입 움직임 판정(6단계, apply_speech_activity): 검출 프레임마다 tracker.mouth_ratio(없거나 None이면
      모름)를 모아 둔다(프레임당 실수 하나). utterance=False(업로드)는 자르지 않고 말이 없는 창만
      no_speech로, utterance=True(녹화)는 녹화 한 건을 한 발화로 판정해 앞뒤 무발화 부분을 잘라 낸다.
      녹화 모드도 모델 창(최대 segment_seconds, 기본 12초)은 그대로라 12초보다 긴 녹화(30초 실험 모드)는
      여러 창으로 나뉠 수 있다. 말이 없으면 관측한 구간만 no_speech이고 얼굴·입술 실패 구간은 남긴다.
      비율을 거의 모르면 VAD를 끄며 결과는 5단계와 같다.
    - 컨테이너 메타데이터(fps 1~240, 프레임 수 양수)를 쓸 수 없으면(MediaRecorder WebM 등) 읽은 원본
      프레임의 PTS 간격 중앙값으로 selection의 source_fps/low_fps를 정한다(fps_source='decoded_pts').
    - 진행률 분모: 메타데이터 길이, 없으면 duration_hint_seconds(서버가 잰 길이; 유한한 양수 또는 None,
      아니면 ValueError), 둘 다 없으면 fraction=None.
    tracker_factory/geometry_factory는 테스트용 주입 지점이다.
    반환: {video, selection, profiles, segments:[{index,start_ms,end_ms,frames,status,reason,npz,detection,
    speech}], speech, vad, stats, timing}. status는 ok|failed|no_speech.
    """
    import math
    import cv2
    import numpy as np
    _ensure_path(OLKAVS)
    if tracker_factory is None or geometry_factory is None:
        from roi_geometry_step04 import RoiGeometry
        tracker_factory = tracker_factory or vad_tracker_class()
        geometry_factory = geometry_factory or (lambda: RoiGeometry(GEOMETRY_ARM))
    from automatic_roi import PROFILE_ID as DETECTOR_PROFILE_ID
    from roi_geometry_step04 import PROFILE_ID as GEOMETRY_PROFILE_ID

    started = time.perf_counter()
    duration_hint = validate_duration_hint(duration_hint_seconds)
    for name, value in (("segment_seconds", segment_seconds), ("min_window_seconds", min_window_seconds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}는 양수여야 합니다: {value}")
    video, out_dir = Path(video).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    tracker = None
    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selection = choose_selection(fps, frame_count, is_webm(video), resample)
        metadata_ok = metadata_usable(fps, frame_count)
        input_gaps: list[float] = []         # 격자 경로: 원본 프레임 PTS 간격(ms), 메타데이터 없을 때 fps 추정용
        stats: dict[str, Any] = {"frames": 0, "gap_frames_dropped": 0, "short_runs_merged": 0,
                                 "held_frames_to_gap": 0, "detector_resets": 0}
        rate = selection["rate"]
        if selection["mode"] == "native_integer":
            frames_iter = iter_adaptive_frames(cap, frame_count, fps, selection, stats)
        else:
            frames_iter = iter_grid_frames(cap, GRID_FPS, None, None, stats, input_gaps=input_gaps)
        frame_ms = 1000.0 / rate
        max_frames = max(2, round(segment_seconds * rate))
        search_frames = max(0, round(cut_search_seconds * rate))
        min_frames = max(1, round(min_window_seconds * rate))
        min_gap_frames = max(1, round(min_gap_seconds * rate))
        min_run_frames = max(1, round(min_run_seconds * rate))
        duration_ms = frame_count / fps * 1000.0 if metadata_ok else None
        duration_source = "metadata" if duration_ms is not None else None
        if duration_ms is None and duration_hint is not None:
            duration_ms, duration_source = duration_hint * 1000.0, "hint"  # MediaRecorder WebM: 서버가 잰 길이

        segments: list[dict[str, Any]] = []
        ratios: list[float | None] = []      # 읽은 프레임마다 입 벌림 비율(모르면 None) — 프레임당 실수 하나
        pts_series: list[float] = []         # 읽은 프레임마다 PTS(ms)
        window_meta: dict[int, dict[str, Any]] = {}  # ok 창 index → 프레임 위치·검출 여부
        last_progress = {"pts": 0.0}

        def report(pts_ms: float) -> None:
            """시간 기준 진행률 (원본·격자 모드 공통, 메타데이터 길이나 길이 힌트가 있을 때)."""
            last_progress["pts"] = max(last_progress["pts"], pts_ms)
            if progress is not None:
                fraction = min(1.0, last_progress["pts"] / duration_ms) if duration_ms else None
                progress({"stage": "roi", "segments_done": len(segments), "frames_done": stats["frames"],
                          "fraction": fraction})

        def record(status: str, items: list[dict[str, Any]] | None, start_pts: float, end_pts: float,
                   frames: int, reason: str | None) -> None:
            """구간 하나를 기록한다 (ok면 NPZ 저장)."""
            entry: dict[str, Any] = {"index": len(segments), "start_ms": int(round(start_pts)),
                                     "end_ms": int(round(end_pts + frame_ms)), "frames": frames,
                                     "status": status, "reason": reason, "npz": None}
            if status == "ok" and items:
                npz = out_dir / f"seg_{entry['index']:03d}.npz"
                np.savez_compressed(npz, frames=np.stack([it["crop"] for it in items]).astype(np.uint8))
                entry["npz"] = str(npz)
                entry["detection"] = {"detected": sum(it["status"] == "detected" for it in items),
                                      "previous_bbox": sum(it["status"] != "detected" for it in items)}
                window_meta[entry["index"]] = {"positions": [it["pos"] for it in items],
                                               "detected": [it["status"] == "detected" for it in items]}
            segments.append(entry)
            report(end_pts)

        builder = WindowBuilder(max_frames, search_frames, min_frames,
                                emit=lambda items: record("ok", items, items[0]["pts"], items[-1]["pts"], len(items), None))
        run: dict[str, Any] | None = None   # {"geometry", "prev_crop", "detected", "held": [...], "staged": [...]}
        gap: dict[str, Any] | None = None   # {"start", "end", "frames", "reasons": {...}}

        def add_gap(pts: float, reason: str) -> None:
            nonlocal gap
            if gap is None:
                gap = {"start": pts, "end": pts, "frames": 0, "reasons": {}}
            gap["end"], gap["frames"] = pts, gap["frames"] + 1
            gap["reasons"][reason] = gap["reasons"].get(reason, 0) + 1

        def close_gap() -> None:
            nonlocal gap
            if gap is not None:
                if gap["frames"] >= min_gap_frames:
                    reason = max(gap["reasons"].items(), key=lambda kv: kv[1])[0]
                    record("failed", None, gap["start"], gap["end"], gap["frames"], reason)
                else:
                    stats["gap_frames_dropped"] += gap["frames"]
            gap = None

        def push(item: dict[str, Any]) -> None:
            item["motion"] = motion_score(run["prev_crop"], item["crop"]) if run["prev_crop"] is not None else 0.0
            run["prev_crop"] = item["crop"]
            builder.push(item)

        def close_run(reason: str, tracking_lost: bool = False) -> None:
            """run을 닫는다. 검출이 적은 run은 실패 구간에 합치고, 추적을 잃은 경우 보류 프레임은 실패 구간으로."""
            nonlocal run
            if run is None:
                return
            held, run_items = run["held"], None
            if run["detected"] < min_run_frames:
                # 확정되지 않은 run: 창으로 보낸 crop이 없다(staged에만 있음) → 전부 실패 구간으로
                stats["short_runs_merged"] += 1
                run_items = run["staged"] + held
                held = []
            elif not tracking_lost:
                for item in held:  # 영상 끝·기하 오류 등: 검증 규약대로 직전 상자 프레임을 발화에 포함
                    push(item)
                held = []
            builder.end_run()
            run = None
            for item in (run_items or []) + held:  # 실패 구간은 시간 순서대로 (창 뒤에 온다)
                add_gap(item["pts"], reason)
            stats["held_frames_to_gap"] += len(held)

        def new_tracker() -> Any:
            stats["detector_resets"] += 1
            return tracker_factory()

        tracker = tracker_factory()
        frame_shape = None
        first_pts = last_pts = None
        for _frame_index, pts_ms, frame in frames_iter:
            stats["frames"] += 1
            pos = stats["frames"] - 1
            ratios.append(None)
            pts_series.append(pts_ms)
            first_pts = pts_ms if first_pts is None else first_pts
            last_pts = pts_ms
            if stats["frames"] % 30 == 0:
                report(pts_ms)
            shape = getattr(frame, "shape", (0, 0))[:2]
            if frame_shape is not None and shape != frame_shape:
                close_run("frame_size_changed")
                tracker.close()
                tracker = new_tracker()
            frame_shape = shape
            try:
                tracker.process(frame)  # 직사각형 crop은 쓰지 않고 상자만 쓴다
            except (ValueError, RuntimeError):
                close_run("detector_error", tracking_lost=True)
                add_gap(pts_ms, "detector_error")
                tracker.close()
                tracker = new_tracker()
                continue
            ratios[pos] = _finite_or_none(getattr(tracker, "mouth_ratio", None))
            if tracker.failed:
                reason = str(tracker.failure_reason)
                close_run(reason, tracking_lost=True)
                add_gap(pts_ms, "no_face" if reason == "initial_detection_miss" else reason)
                rearm_tracker(tracker)
                continue
            geometry = run["geometry"] if run is not None else geometry_factory()
            try:
                crop, _telemetry = geometry.process(frame, tracker.last_box_xyxy)
            except ValueError:  # 얼굴은 보이지만 문맥 정사각형이 화면보다 큼 등 → 그 프레임은 실패 구간
                close_run("geometry_error")
                add_gap(pts_ms, "geometry_error")
                continue
            if run is None:
                run = {"geometry": geometry, "prev_crop": None, "detected": 0, "held": [], "staged": []}
            item = {"crop": crop, "pts": pts_ms, "status": tracker.frame_records[-1]["status"], "pos": pos}
            if item["status"] != "detected":
                run["held"].append(item)  # 다음 검출이 올 때까지 보류 (최대 11장)
                continue
            run["staged"].extend(run["held"])
            run["held"] = []
            run["staged"].append(item)
            run["detected"] += 1
            if run["detected"] >= min_run_frames:
                if run["detected"] == min_run_frames:
                    close_gap()  # run이 확정되면 그 앞의 실패 구간을 먼저 내보낸다 (창보다 먼저 → 시간 순서 유지)
                for staged in run["staged"]:
                    push(staged)
                run["staged"] = []
        close_run("end_of_video")
        close_gap()
        if not metadata_ok:  # 메타데이터를 믿을 수 없으면 실제 원본 PTS 간격으로 원본 fps·저fps 경고를 정한다
            estimated = estimate_fps_from_gaps(input_gaps)
            if estimated is not None:
                selection.update(source_fps=estimated, low_fps=bool(estimated < LOW_FPS_THRESHOLD),
                                 fps_source="decoded_pts")
        if stats["frames"] == 0:
            raise ValueError("영상에서 프레임을 읽지 못했습니다")
        if not segments:  # 프레임은 읽었지만 짧은 실패만 있었다 → 전체를 실패 구간으로 보고
            record("failed", None, first_pts, last_pts, stats["frames"], "no_usable_lip_frames")
        speech = apply_speech_activity(segments, window_meta, ratios, pts_series, rate, frame_ms,
                                       bool(utterance), builder.min_frames, max_frames=builder.max_frames)
        speech["utterance"] = bool(utterance)
        stats["ratio_known_frames"] = sum(1 for value in ratios if value is not None)
        report(last_pts)
        return {
            "video": {"fps_metadata": fps, "frame_count_metadata": frame_count, "frames_read": stats["frames"],
                      "duration_hint_seconds": duration_hint, "progress_duration_source": duration_source},
            "selection": {**selection, "segment_frames": max_frames, "cut_search_frames": builder.search_frames,
                          "min_window_frames": builder.min_frames},
            "profiles": {"detector": DETECTOR_PROFILE_ID, "geometry": GEOMETRY_PROFILE_ID, "arm": GEOMETRY_ARM},
            "segments": segments,
            "speech": speech,
            "vad": vad_profile(),
            "stats": stats,
            "timing_seconds": {"total": time.perf_counter() - started},
        }
    finally:
        if tracker is not None:
            tracker.close()  # 예외로 끝나도 FaceMesh 그래프·스레드를 반드시 해제한다 (멱등)
        cap.release()


def run_roi(args: argparse.Namespace) -> int:
    """영상에서 입술 96x96 BGR 프레임을 추출해 NPZ와 텔레메트리 JSON을 저장한다."""
    try:
        import cv2
        import numpy as np
        _ensure_path(OLKAVS)
        from automatic_roi import AutomaticLipRoi, profile as detector_profile
        from roi_geometry_step04 import RoiGeometry, profile as geometry_profile
    except ImportError as exc:
        print(f"[roi] import 실패 — .venv-olkavs-roi 인터프리터로 실행하십시오: {exc}", file=sys.stderr)
        return 2

    video = Path(args.video).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        webm = is_webm(video)
        metadata_ok = 1.0 <= fps <= 120.0 and frame_count > 0
        use_grid = args.resample or webm or not metadata_ok
        if use_grid:
            selection = {"mode": "pts_grid", "grid_fps": GRID_FPS, "start_seconds": args.start, "end_seconds": args.end,
                         "reason": "resample flag" if args.resample else ("webm container" if webm else "invalid fps/frame-count metadata"),
                         "rule": "nearest decoded PTS to each 1/30 s grid point; tie -> earlier frame; streaming one-frame lookahead"}
            frames_iter = iter_grid_frames(cap, GRID_FPS, args.start, args.end)
        else:
            first, stop = select_frame_range(fps, frame_count, args.start, args.end)
            selection = {"mode": "native_integer", "first": first, "stop_exclusive": stop, "start_seconds": args.start,
                         "end_seconds": args.end, "rule": "ceil(t*fps) native integer frames (official nominal selection)"}
            frames_iter = iter_native_frames(cap, first, stop)

        rois: list[np.ndarray] = []
        frame_meta: list[dict[str, Any]] = []
        decode_s = detect_s = geom_s = 0.0
        tracker = AutomaticLipRoi()
        geometry = RoiGeometry(GEOMETRY_ARM)
        frame = None
        try:
            t_read = time.perf_counter()
            for index, pts_ms, frame in frames_iter:
                decode_s += time.perf_counter() - t_read
                t0 = time.perf_counter()
                tracker.process(frame)  # 직사각형 크롭은 쓰지 않고 상자만 사용
                detect_s += time.perf_counter() - t0
                if tracker.failed:
                    raise ValueError(f"입술 검출 실패: {tracker.failure_reason} (frame {index})")
                t0 = time.perf_counter()
                crop, telemetry = geometry.process(frame, tracker.last_box_xyxy)
                geom_s += time.perf_counter() - t0
                rois.append(crop)
                frame_meta.append({"source_frame_index": index, "pts_ms": pts_ms,
                                   "detector_status": tracker.frame_records[-1]["status"],
                                   "lip_bbox_xyxy": list(map(int, tracker.last_box_xyxy)),
                                   "final_square_xyxy": telemetry["final_square_xyxy"],
                                   "integer_side": telemetry["integer_side"],
                                   "boundary_shifted": telemetry["boundary_shifted"]})
                t_read = time.perf_counter()
        finally:
            tracker.close()
            cap.release()

        if not rois or frame is None:
            raise ValueError("선택 구간에 프레임이 없습니다 (영상이 비었거나 start/end가 범위 밖)")
        frames = np.stack(rois).astype(np.uint8)  # (T, 96, 96, 3) BGR
        npz_path = out / "roi.npz"
        np.savez_compressed(npz_path, frames=frames, source_frame_index=np.array([m["source_frame_index"] for m in frame_meta]))
        summary = {
            "schema": "olkavs-transcribe-roi-v1",
            "video": {"path": str(video), "sha256": sha256_file(video), "fps_metadata": fps, "frame_count_metadata": frame_count,
                      "webm": webm, "metadata_ok": metadata_ok, "width": int(frame.shape[1]), "height": int(frame.shape[0])},
            "selected_frames": {**selection, "count": int(frames.shape[0]),
                                "first_pts_ms": frame_meta[0]["pts_ms"], "last_pts_ms": frame_meta[-1]["pts_ms"]},
            "detector_profile": detector_profile(), "geometry_profile": geometry_profile(), "geometry_arm": GEOMETRY_ARM,
            "detector_stats": tracker.stats(),
            "frames": frame_meta,
            "output": {"npz": str(npz_path), "npz_sha256": sha256_file(npz_path), "shape": list(frames.shape),
                       "color": "BGR", "dtype": "uint8", "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest()},
            "timing_seconds": {"decode": decode_s, "detect": detect_s, "geometry": geom_s,
                               "detector_init": tracker.initialization_seconds,
                               "total": time.perf_counter() - started},
            "environment": {"python": sys.version.split()[0], "cv2": cv2.__version__, "numpy": np.__version__},
        }
        save_json(out / "roi.json", summary)
        print(json.dumps({"stage": "roi", "frames": int(frames.shape[0]),
                          "detect_seconds": round(detect_s, 3), "total_seconds": round(summary["timing_seconds"]["total"], 3)},
                         ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - 명령줄 도구의 최종 보고
        save_json(out / "roi.json", {"schema": "olkavs-transcribe-roi-v1", "status": "failed",
                                     "error": {"type": type(exc).__name__, "message": str(exc)}})
        print(f"[roi] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# ----------------------------------------------------------------------------------
# infer: 입술 프레임 → 텍스트 (torch 환경)
# ----------------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def load_olkavs(device: str) -> dict[str, Any]:
    """고정 commit·가중치 SHA를 검사한 뒤 공식 영상 모델과 사전을 메모리에 올린다."""
    key = device
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import inspect
    import numpy as np
    import torch
    import yaml
    _ensure_path(OLKAVS)
    from verify_model import checked_state
    from corrected_search import CorrectedSearchSequence  # noqa: F401 - upstream 경로 등록 겸 검증
    try:
        commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
        if commit != EXPECTED_COMMIT:
            raise ValueError(f"upstream commit 불일치: {commit}")
        if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
            raise ValueError("가중치 SHA-256 불일치")
    except (subprocess.CalledProcessError, OSError) as exc:
        raise RuntimeError(f"upstream/가중치 검증 실패: {exc}") from exc
    sys.path.insert(0, str(UPSTREAM))
    cwd = os.getcwd()
    os.chdir(UPSTREAM)  # vocabulary csv 상대 경로
    try:
        from avsr.utils.model_builder import build_model
        from avsr.utils.korean_funcs import grp2char
        from vocabulary.utils import KsponSpeechVocabulary
        config = yaml.safe_load((UPSTREAM / "configs" / "v_inference.yaml").read_text(encoding="utf-8"))
        vocab = KsponSpeechVocabulary(unit="grapheme")
        if (len(vocab), vocab.unk_id, vocab.eos_id, vocab.sos_id, vocab.pad_id) != (72, 3, 2, 1, 0):
            raise ValueError("공식 사전 구성이 예상과 다릅니다")
        torch.manual_seed(2207)
        np.random.seed(2207)
        torch.backends.cudnn.benchmark = False
        kwargs = {k: config[k] for k in inspect.signature(build_model).parameters if k in config}
        model = build_model(vocab_size=72, pad_id=0, pass_visual_frontend=False, verbose=False, **kwargs)
        contract = checked_state(model, torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        model.to(torch.device(device)).eval()
    finally:
        os.chdir(cwd)
    bundle = {"model": model, "vocab": vocab, "grp2char": grp2char, "contract": contract, "device": device,
              "search_cls": CorrectedSearchSequence}
    _MODEL_CACHE[key] = bundle
    return bundle


def normalize_frames(frames: Any) -> tuple[Any, dict[str, float]]:
    """공식 규약: 발화 전체 float32 평균/표준편차(correction=1)로 정규화 → (1,3,T,96,96)."""
    import numpy as np
    import torch
    if frames.ndim != 4 or frames.shape[1:] != (96, 96, 3) or frames.dtype != np.uint8:
        raise ValueError(f"(T,96,96,3) uint8 BGR 프레임이 필요합니다: {frames.shape} {frames.dtype}")
    x = torch.from_numpy(frames).float()
    raw_mean = x.mean().item()
    x -= x.mean()
    raw_std = x.std(correction=1)
    if not torch.isfinite(raw_std) or raw_std <= 0:
        raise ValueError("영상 분산이 0이거나 비정상입니다")
    x /= raw_std
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous(), {"raw_mean": raw_mean, "raw_std": raw_std.item()}


def transcribe_frames(frames: Any, device: str, methods: tuple[str, ...]) -> dict[str, Any]:
    """정규화 → encoder+medium → CTC greedy / hybrid 디코딩. 각 구간 시간을 기록한다."""
    import torch
    _ensure_path(OLKAVS)
    from diagnostic_utils import collapse_ctc
    bundle = load_olkavs(device)
    model, vocab, grp2char = bundle["model"], bundle["vocab"], bundle["grp2char"]
    use_cuda = device.startswith("cuda")

    def clock() -> float:
        if use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    result: dict[str, Any] = {"device": device, "frames": int(frames.shape[0]), "outputs": {}, "timing_seconds": {}}
    t0 = time.perf_counter()
    video, norm = normalize_frames(frames)
    video = video.to(device)
    lengths = torch.tensor([video.shape[2]], device=device)
    result["normalization"] = norm
    result["timing_seconds"]["normalize_and_transfer"] = clock() - t0
    with torch.inference_mode():
        t0 = clock()
        features = model.medium(model.encoder(video, lengths, None, None))
        result["timing_seconds"]["encoder"] = clock() - t0
        if not torch.isfinite(features).all():
            raise ValueError("encoder 출력에 비정상 값이 있습니다")
        result["feature_shape"] = list(features.shape)
        for method in methods:
            try:
                t0 = clock()
                if method == "ctc_greedy":
                    logits = model.decoder.ctcdecoder(inputs=features)
                    path = logits.argmax(-1)[0].cpu().tolist()
                    tokens = collapse_ctc(path, blank_id=3)
                    extra = {"blank_fraction": path.count(3) / len(path)}
                elif method == "hybrid":
                    search = bundle["search_cls"](model=model, vocab_size=72, pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                                                  method="hybrid", max_len=200, ctc_rate=0.6, mp_num=1)
                    ids, lens = search.decode_features(features, beam_size=1, D_end=-14, M_end=16)
                    tokens = ids[0, :int(lens[0])].cpu().tolist()
                    extra = {}
                else:
                    raise ValueError(f"알 수 없는 디코딩 방식: {method}")
                elapsed = clock() - t0
                graphemes = vocab.label_to_string(torch.tensor(tokens, dtype=torch.long))
                text = grp2char(graphemes).strip()
                result["outputs"][method] = {"status": "ok", "text": text, "graphemes": graphemes,
                                             "token_ids": tokens, "seconds": elapsed, **extra}
            except (ValueError, RuntimeError, OSError) as exc:
                result["outputs"][method] = {"status": "failed", "text": "",
                                             "error": {"type": type(exc).__name__, "message": str(exc)}}
    if use_cuda:
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    return result


SPECIAL_TOKEN_IDS = (0, 1, 2, 3)  # pad, sos, eos, unk(=CTC blank)
HYBRID_MAX_LEN = 200  # 공식 v_inference.yaml / evaluate_step04.py 와 같은 값
MAX_GRAPHEME_TOKENS = 450  # upstream grp2char 반복 상한(1000) 안쪽 (반복 ≤ 2×토큰)


def _observed_search_cls(base: Any) -> Any:
    """탐색 깊이를 기록하는 CorrectedSearchSequence 하위 클래스 (점수·선택은 그대로)."""
    class ObservedSearch(base):
        observed_max_search_length = 0

        def _beam_search(self, features, current_hypothesis, length, *args, **kwargs):
            self.observed_max_search_length = max(self.observed_max_search_length, length)
            return super()._beam_search(features, current_hypothesis, length, *args, **kwargs)
    return ObservedSearch


def ctc_forced_score(log_probs: Any, tokens: list[int], blank: int = 3) -> dict[str, Any]:
    """문장 토큰열의 CTC 강제 정렬 로그우도(FP64 CPU 로그확률 T×V)와 토큰당 평균."""
    import torch
    from torch.nn import functional as F
    try:
        need = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        if not tokens or need > len(log_probs):
            return {"status": "infeasible", "total": None, "per_token": None, "tokens": len(tokens)}
        loss = F.ctc_loss(log_probs.unsqueeze(1), torch.tensor(tokens, dtype=torch.long),
                          torch.tensor([len(log_probs)]), torch.tensor([len(tokens)]),
                          blank=blank, reduction="none", zero_infinity=False)
        total = -float(loss[0])
        return {"status": "ok", "total": total, "per_token": total / len(tokens), "tokens": len(tokens)}
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"CTC 점수 계산 실패: {exc}") from exc


def recognize(frames: Any, device: str, decoder: str = "hybrid") -> dict[str, Any]:
    """서비스용 단일 구간 인식: 텍스트 + confidence(0~1) + raw_score(CTC 로그우도).

    confidence = exp(프레임별 최대 CTC 로그확률의 평균) — compact 서비스와 같은 정의.
    raw_score  = 출력 토큰열의 CTC 강제 정렬 로그우도(원본 log-prob 합). 정렬 불가면 None.
    """
    import torch
    _ensure_path(OLKAVS)
    from diagnostic_utils import collapse_ctc
    if decoder not in ("hybrid", "ctc"):
        raise ValueError(f"알 수 없는 디코더: {decoder}")
    bundle = load_olkavs(device)
    model, vocab, grp2char = bundle["model"], bundle["vocab"], bundle["grp2char"]
    use_cuda = device.startswith("cuda")

    def clock() -> float:
        if use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    timing: dict[str, float] = {}
    t0 = clock()
    video, _norm = normalize_frames(frames)
    video = video.to(device)
    lengths = torch.tensor([video.shape[2]], device=device)
    with torch.inference_mode():
        features = model.medium(model.encoder(video, lengths, None, None))
        if not torch.isfinite(features).all():
            raise ValueError("encoder 출력에 비정상 값이 있습니다")
        logits = model.decoder.ctcdecoder(inputs=features)
        timing["encoder"] = clock() - t0
        log_probs = torch.log_softmax(logits[0].double(), dim=-1).cpu()
        t0 = clock()
        search_cap_reached = False
        if decoder == "ctc":
            tokens = collapse_ctc(log_probs.argmax(-1).tolist(), blank_id=3)
        else:
            search = _observed_search_cls(bundle["search_cls"])(
                model=model, vocab_size=72, pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                method="hybrid", max_len=HYBRID_MAX_LEN, ctc_rate=0.6, mp_num=1)
            ids, lens = search.decode_features(features, beam_size=1, D_end=-14, M_end=16)
            tokens = ids[0, :int(lens[0])].cpu().tolist()
            # evaluate_step04.py와 같은 기준: 탐색이 상한에 닿으면 끝부분이 잘렸을 수 있다.
            search_cap_reached = search.observed_max_search_length >= HYBRID_MAX_LEN - 1
        timing["decode"] = clock() - t0
    if 2 in tokens:  # EOS 이후는 버린다
        tokens = tokens[:tokens.index(2)]
    content = [t for t in tokens if t not in SPECIAL_TOKEN_IDS]
    if len(content) > MAX_GRAPHEME_TOKENS:
        # upstream grp2char는 반복 1000회를 넘으면 pdb.set_trace()로 멈춘다(토큰당 최대 2회 반복).
        raise ValueError(f"출력 자모가 너무 깁니다({len(content)}개). 구간 길이를 줄이세요.")
    graphemes = vocab.label_to_string(torch.tensor(content, dtype=torch.long)) if content else ""
    text = grp2char(graphemes).strip() if graphemes else ""
    confidence = float(torch.exp(log_probs.max(dim=-1).values.mean()))
    score = ctc_forced_score(log_probs, content, blank=3) if content else {"status": "empty", "total": None}
    return {"text": text, "confidence": min(1.0, max(0.0, confidence)), "raw_score": score["total"],
            "raw_score_status": score["status"], "decoder": decoder, "frames": int(frames.shape[0]),
            "search_cap_reached": search_cap_reached, "token_ids": content, "timing_seconds": timing}


def read_frames(path: Path) -> Any:
    """roi.npz 또는 96x96 mp4(step04 입력 형식)에서 (T,96,96,3) BGR uint8을 읽는다."""
    import numpy as np
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return np.asarray(data["frames"], dtype=np.uint8)
    import cv2
    cap, frames = cv2.VideoCapture(str(path)), []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"프레임을 읽지 못했습니다: {path}")
    return np.stack(frames).astype(np.uint8)


def run_infer(args: argparse.Namespace) -> int:
    """NPZ/mp4 입술 프레임을 읽어 텍스트와 시간을 JSON으로 저장한다."""
    try:
        import torch
    except ImportError as exc:
        print(f"[infer] torch 없음 — .venv-olkavs 인터프리터로 실행하십시오: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[infer] CUDA를 쓸 수 없어 cpu로 실행합니다", file=sys.stderr)
        device = "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, args.threads))
    methods = tuple(args.methods.split(","))
    try:
        t0 = time.perf_counter()
        bundle = load_olkavs(device)
        load_seconds = time.perf_counter() - t0
        warm = None
        if args.warmup:
            import numpy as np
            dummy = (np.random.RandomState(0).rand(24, 96, 96, 3) * 255).astype(np.uint8)
            t0 = time.perf_counter()
            transcribe_frames(dummy, device, ("ctc_greedy",))
            warm = time.perf_counter() - t0
        frames = read_frames(Path(args.frames).resolve())
        result = transcribe_frames(frames, device, methods)
        result.update({"schema": "olkavs-transcribe-infer-v1", "frames_path": str(Path(args.frames).resolve()),
                       "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                       "checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA256, "upstream_commit": EXPECTED_COMMIT,
                                      "strict_load": bundle["contract"]["strict"]},
                       "model_load_seconds": load_seconds, "warmup_seconds": warm,
                       "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
                       "cpu_threads": torch.get_num_threads() if device == "cpu" else None})
        if args.reference is not None:
            _ensure_path(OLKAVS)
            from evaluation_metrics import counts
            result["reference"] = args.reference
            for method, output in result["outputs"].items():
                output["metrics"] = counts(args.reference, output.get("text", ""))
        save_json(out / "infer.json", result)
        print(json.dumps({"stage": "infer", "device": device,
                          **{m: {"text": o.get("text", ""), "seconds": round(o.get("seconds", 0.0), 3)} for m, o in result["outputs"].items()}},
                         ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        save_json(out / "infer.json", {"schema": "olkavs-transcribe-infer-v1", "status": "failed",
                                       "error": {"type": type(exc).__name__, "message": str(exc)}})
        print(f"[infer] 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


# ----------------------------------------------------------------------------------
# run: 두 단계를 순서대로 실행
# ----------------------------------------------------------------------------------
def run_driver(args: argparse.Namespace) -> int:
    """roi → infer 를 각 가상환경 인터프리터로 실행하고 결과를 합친다."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    for python in (ROI_PYTHON, INFER_PYTHON):
        if not python.exists():
            print(f"인터프리터가 없습니다: {python}", file=sys.stderr)
            return 2
    roi_cmd = [str(ROI_PYTHON), script, "roi", "--video", str(args.video), "--out", str(out)]
    if args.start is not None:
        roi_cmd += ["--start", str(args.start)]
    if args.end is not None:
        roi_cmd += ["--end", str(args.end)]
    if args.resample:
        roi_cmd.append("--resample")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.perf_counter()
    roi_rc = subprocess.run(roi_cmd, env=env).returncode
    roi_wall = time.perf_counter() - t0
    if roi_rc != 0:
        return roi_rc
    infer_cmd = [str(INFER_PYTHON), script, "infer", "--frames", str(out / "roi.npz"), "--out", str(out),
                 "--device", args.device, "--methods", args.methods, "--threads", str(args.threads)]
    if args.warmup:
        infer_cmd.append("--warmup")
    if args.reference is not None:
        infer_cmd += ["--reference", args.reference]
    t0 = time.perf_counter()
    infer_rc = subprocess.run(infer_cmd, env=env).returncode
    infer_wall = time.perf_counter() - t0
    try:
        roi = json.loads((out / "roi.json").read_text(encoding="utf-8"))
        infer = json.loads((out / "infer.json").read_text(encoding="utf-8"))
        combined = {"schema": "olkavs-transcribe-v1", "video": roi.get("video"), "selected_frames": roi.get("selected_frames"),
                    "text": {m: o.get("text", "") for m, o in infer.get("outputs", {}).items()},
                    "outputs": infer.get("outputs"), "reference": infer.get("reference"),
                    "wall_seconds": {"roi_process": roi_wall, "infer_process": infer_wall,
                                     "roi_inner": roi.get("timing_seconds"), "infer_inner": infer.get("timing_seconds"),
                                     "model_load": infer.get("model_load_seconds")},
                    "device": infer.get("device"), "detector_stats": roi.get("detector_stats")}
        save_json(out / "result.json", combined)
        print(json.dumps({"text": combined["text"], "wall_seconds": {"roi": round(roi_wall, 2), "infer": round(infer_wall, 2)}},
                         ensure_ascii=False))
    except (OSError, ValueError) as exc:
        print(f"결과 병합 실패: {exc}", file=sys.stderr)
        return 1
    return infer_rc


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_roi = sub.add_parser("roi", help="영상 → 입술 프레임 NPZ (.venv-olkavs-roi)")
    p_roi.add_argument("--video", required=True)
    p_roi.add_argument("--out", required=True)
    p_roi.add_argument("--start", type=float, default=None, help="시작 초 (생략 시 처음부터)")
    p_roi.add_argument("--end", type=float, default=None, help="끝 초 (생략 시 끝까지)")
    p_roi.add_argument("--resample", action="store_true", help="PTS 기준 30fps 격자 리샘플 강제 (WebM은 자동)")
    p_roi.set_defaults(func=run_roi)

    p_inf = sub.add_parser("infer", help="입술 프레임 → 텍스트 (.venv-olkavs)")
    p_inf.add_argument("--frames", required=True, help="roi.npz 또는 96x96 mp4")
    p_inf.add_argument("--out", required=True)
    p_inf.add_argument("--device", default="cuda:0")
    p_inf.add_argument("--methods", default="ctc_greedy,hybrid")
    p_inf.add_argument("--threads", type=int, default=4, help="CPU 스레드 수")
    p_inf.add_argument("--warmup", action="store_true", help="측정 전에 더미 추론 1회")
    p_inf.add_argument("--reference", default=None, help="정답 문장 (있으면 CER 계산)")
    p_inf.set_defaults(func=run_infer)

    p_run = sub.add_parser("run", help="roi → infer 순서 실행")
    for name in ("--video", "--out"):
        p_run.add_argument(name, required=True)
    p_run.add_argument("--start", type=float, default=None)
    p_run.add_argument("--end", type=float, default=None)
    p_run.add_argument("--resample", action="store_true")
    p_run.add_argument("--device", default="cuda:0")
    p_run.add_argument("--methods", default="ctc_greedy,hybrid")
    p_run.add_argument("--threads", type=int, default=4)
    p_run.add_argument("--warmup", action="store_true")
    p_run.add_argument("--reference", default=None)
    p_run.set_defaults(func=run_driver)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
