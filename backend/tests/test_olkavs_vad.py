"""6단계 입 움직임 기반 발화 구간 검출(VAD) 검사 (mediapipe·torch·GPU 불필요).

scripts/olkavs_transcribe.py 의 입 벌림 비율·활성 판정·구간 계산 순수 함수와, 가짜 검출기(mouth_ratio
대본)로 extract_segments 의 업로드(utterance=False)/녹화(utterance=True) 동작을 확인한다.
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import olkavs_transcribe as ot  # noqa: E402

RATE = 30.0


# ---------------------------------------------------------------------------
# 비율 계열 대본
# ---------------------------------------------------------------------------
def still(n: int, level: float = 0.13) -> list[float]:
    """입을 다물고 가만히 있는 계열 (랜드마크 잡음 ±0.0003)."""
    return [level + (0.0003 if i % 2 else -0.0003) for i in range(n)]


def talk(n: int, level: float = 0.13, amplitude: float = 0.015) -> list[float]:
    """말하는 계열: 4Hz로 입이 열렸다 닫힌다(비율 변화 폭 2*amplitude)."""
    return [level + amplitude * (1 + math.sin(2 * math.pi * 4 * i / RATE)) for i in range(n)]


# ---------------------------------------------------------------------------
# 입 벌림 비율
# ---------------------------------------------------------------------------
def _points(top_y: float, bot_y: float, cheek_x: tuple[float, float], n: int = 468) -> np.ndarray:
    points = np.zeros((n, 2), dtype=np.float64)
    for k, i in enumerate(ot.VAD_TOP_IDX):
        points[i] = (90 + 4 * k, top_y)
    for k, i in enumerate(ot.VAD_BOT_IDX):
        points[i] = (90 + 4 * k, bot_y)
    points[ot.VAD_WIDTH_IDX[0]] = (cheek_x[0], 115)
    points[ot.VAD_WIDTH_IDX[1]] = (cheek_x[1], 115)
    return points


def test_mouth_ratio_is_mean_point_distance_over_cheek_width() -> None:
    """d_raw = 윗/아랫 평균점 거리, W = 234–454 거리. 같은 x 배치면 d_raw는 y 차이."""
    assert ot.mouth_ratio(_points(100, 130, (0, 200))) == pytest.approx(30 / 200)
    assert ot.mouth_ratio(_points(100, 112, (50, 150))) == pytest.approx(12 / 100)


def test_mouth_ratio_rejects_invalid_points() -> None:
    """점 부족·비유한 값·얼굴 폭 0이면 None."""
    assert ot.mouth_ratio(np.zeros((400, 2))) is None
    assert ot.mouth_ratio(_points(100, 130, (100, 100))) is None
    bad = _points(100, 130, (0, 200))
    bad[13] = (np.nan, 1.0)
    assert ot.mouth_ratio(bad) is None
    assert ot.mouth_ratio("not points") is None


class _Landmark:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


def _make_face(lip_half: float, top_y: float, bot_y: float, cheeks: tuple[float, float] = (0.4, 0.6),
               n: int = 468) -> SimpleNamespace:
    """정규화 좌표 얼굴 하나. 입술 외곽 20점은 원 위(상자 면적 ∝ lip_half²), VAD 점은 지정 높이."""
    marks = [_Landmark(0.5, 0.5) for _ in range(n)]
    lip = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
    for k, i in enumerate(lip):
        angle = 2 * math.pi * k / len(lip)
        marks[i] = _Landmark(0.5 + lip_half * math.cos(angle), 0.55 + lip_half * 0.5 * math.sin(angle))
    for k, i in enumerate(ot.VAD_TOP_IDX):
        marks[i] = _Landmark(0.48 + 0.005 * k, top_y)
    for k, i in enumerate(ot.VAD_BOT_IDX):
        marks[i] = _Landmark(0.48 + 0.005 * k, bot_y)
    marks[ot.VAD_WIDTH_IDX[0]] = _Landmark(cheeks[0], 0.5)
    marks[ot.VAD_WIDTH_IDX[1]] = _Landmark(cheeks[1], 0.5)
    return SimpleNamespace(landmark=marks)


def test_face_mouth_ratio_uses_pixel_coordinates_not_normalized() -> None:
    """정규화 좌표를 원본 가로·세로 픽셀로 바꾼 뒤 잰다(16:9에서 정규화 그대로와 다르다)."""
    face = _make_face(0.05, 0.50, 0.55)
    got = ot.face_mouth_ratio(face, 1920, 1080)
    assert got == pytest.approx((0.05 * 1080) / (0.2 * 1920))  # 54 / 384
    assert got != pytest.approx(0.05 / 0.2)
    assert ot.face_mouth_ratio(face, 1000, 1000) == pytest.approx(0.25)
    assert ot.face_mouth_ratio(SimpleNamespace(landmark=[]), 640, 480) is None
    assert ot.face_mouth_ratio(None, 640, 480) is None


def _require_cv2() -> Any:
    """automatic_roi·extract_segments는 cv2가 필요하다(없으면 건너뜀)."""
    return pytest.importorskip("cv2")


def test_select_lip_face_matches_frozen_largest_lip_choice() -> None:
    """AutomaticLipRoi와 같은 얼굴(입술 상자 면적 최대)을 고른다."""
    _require_cv2()
    small, big = _make_face(0.03, 0.50, 0.52), _make_face(0.08, 0.50, 0.58)
    assert ot.select_lip_face([small, big], 640, 480) is big
    assert ot.select_lip_face([big, small], 640, 480) is big
    assert ot.select_lip_face([], 640, 480) is None
    assert ot.select_lip_face(None, 640, 480) is None


class _ScriptedDetector:
    """FaceMesh 대체: 프레임마다 정해 둔 얼굴 목록(없으면 미검출)을 돌려준다."""

    def __init__(self, frames: list[list[Any] | None]):
        self.frames = frames
        self.i = -1
        self.closed = False

    def process(self, rgb: Any) -> Any:
        self.i += 1
        return SimpleNamespace(multi_face_landmarks=self.frames[self.i])

    def close(self) -> None:
        self.closed = True


def test_vad_tracker_exposes_ratio_of_selected_face_on_detected_frames_only() -> None:
    """실제 AutomaticLipRoi 하위 클래스: 검출 프레임은 고른 얼굴의 비율, 직전 상자 프레임은 None."""
    _require_cv2()
    small, big = _make_face(0.03, 0.50, 0.52), _make_face(0.08, 0.50, 0.58)
    detector = _ScriptedDetector([[small, big], None, [big]])
    tracker = ot.vad_tracker_class()(detector=detector)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.process(frame)
    assert tracker.frame_records[-1]["status"] == "detected"
    assert tracker.mouth_ratio == pytest.approx(ot.face_mouth_ratio(big, 640, 480))
    assert tracker.mouth_ratio != pytest.approx(ot.face_mouth_ratio(small, 640, 480))
    tracker.process(frame)
    assert tracker.frame_records[-1]["status"] == "previous_bbox_current_frame"
    assert tracker.mouth_ratio is None
    tracker.process(frame)
    assert tracker.mouth_ratio == pytest.approx(ot.face_mouth_ratio(big, 640, 480))
    tracker.close()
    assert detector.closed


# ---------------------------------------------------------------------------
# 활성 판정·구간
# ---------------------------------------------------------------------------
def test_speech_activity_still_and_constant_open_mouth_are_inactive() -> None:
    """가만히 다문 입, 계속 벌린 입(변화 없음), 전부 모름은 모두 비활성."""
    assert not any(ot.speech_activity(still(120), RATE))
    assert not any(ot.speech_activity(still(120, level=0.19), RATE))
    assert not any(ot.speech_activity([None] * 120, RATE))
    assert ot.speech_activity([], RATE) == []


def test_speech_activity_finds_talking_in_middle() -> None:
    """2초 정지 → 2초 말함 → 2초 정지: 말하는 부분은 활성, 먼 정지 부분은 비활성."""
    series = still(60) + talk(60) + still(60)
    active = ot.speech_activity(series, RATE)
    assert len(active) == 180
    assert all(active[70:110])
    assert not any(active[:40]) and not any(active[140:])
    first = active.index(True)
    last = len(active) - 1 - active[::-1].index(True)
    assert 60 - 12 <= first <= 60 and 119 <= last <= 120 + 12  # 창 범위만큼만 넓어진다


def test_speech_activity_tolerates_unknown_frames_and_single_spikes() -> None:
    """한 프레임 걸러 모르는 값이어도 말하는 부분을 찾고, 한 프레임 튐은 중앙값으로 지운다."""
    series: list[float | None] = still(60) + talk(60) + still(60)
    sparse = [v if i % 2 == 0 else None for i, v in enumerate(series)]
    assert all(ot.speech_activity(sparse, RATE)[75:105])
    spiky: list[float | None] = still(90)
    spiky[45] = 0.25
    assert not any(ot.speech_activity(spiky, RATE))


def test_fill_and_filter_fills_short_gaps_and_drops_short_bursts() -> None:
    """활성 사이 짧은 틈은 메우고, 짧은 덩어리는 지운다. 양 끝의 비활성은 메우지 않는다."""
    T, F = True, False
    assert ot.fill_and_filter([F, T, T, F, T, T, F], fill_frames=2, min_burst_frames=1) == [F, T, T, T, T, T, F]
    assert ot.fill_and_filter([T, F, F, F, T], fill_frames=2, min_burst_frames=1) == [T, F, F, F, T]
    assert ot.fill_and_filter([F, T, F, F, T, T, T, F], fill_frames=1, min_burst_frames=2) == [F, F, F, F, T, T, T, F]
    assert ot.fill_and_filter([], 3, 3) == []


def test_speech_span_padding_and_clamping() -> None:
    """첫·마지막 활성에 앞뒤 여유를 더하고 범위로 자른다. 활성이 없으면 None."""
    active = [False] * 30 + [True] * 10 + [False] * 30
    assert ot.speech_span(active, 30.0, 0.2, 0.3) == (30, 39, 24, 48)
    assert ot.speech_span(active, 30.0, 5.0, 5.0) == (30, 39, 0, 69)
    assert ot.speech_span([False] * 5, 30.0) is None
    assert ot.active_seconds(active, list(range(35, 50)), 30.0) == pytest.approx(5 / 30)


def test_vad_profile_lists_constants() -> None:
    """결과에 기록할 VAD 프로필에 실제 상수가 들어 있다."""
    profile = ot.vad_profile()
    assert profile["activity_threshold"] == ot.VAD_ACTIVITY_THRESHOLD
    assert profile["min_speech_seconds"] == ot.VAD_MIN_SPEECH_SECONDS
    assert profile["top_idx"] == [13, 14, 312, 311, 310, 415]
    assert profile["bot_idx"] == [17, 18, 84, 181, 180, 314]
    assert profile["width_idx"] == [234, 454]


# ---------------------------------------------------------------------------
# extract_segments (가짜 검출기·기하 주입)
# ---------------------------------------------------------------------------
class _Script:
    """프레임 순서별 대본. codes: D=검출, H=직전 상자, F=추적 실패, N=얼굴 없음, G=기하 오류.
    ratios: 프레임별 입 벌림 비율(None=모름)."""

    def __init__(self, codes: list[str], ratios: list[float | None] | None):
        self.codes = codes
        self.ratios = ratios
        self.i = -1
        self.trackers: list[Any] = []

    def current(self) -> str:
        return self.codes[self.i]


class _RatioTracker:
    """mouth_ratio를 노출하는 가짜 추적기 (검출 프레임만 비율을 낸다)."""

    expose_ratio = True

    def __init__(self, script: _Script):
        self.script = script
        self.failed = False
        self.failure_reason = None
        self.last_box_xyxy = None
        self.consecutive_miss = 0
        self.frame_records: list[dict[str, Any]] = []
        self.closed = False
        if self.expose_ratio:
            self.mouth_ratio = None
        script.trackers.append(self)

    def process(self, frame: Any) -> None:
        self.script.i += 1
        code = self.script.current()
        if self.expose_ratio:
            self.mouth_ratio = None
        if code in ("F", "N"):
            self.failed = True
            self.failure_reason = "tracking_lost" if code == "F" else "initial_detection_miss"
            return None
        self.last_box_xyxy = (0, 0, 10, 10)
        status = "previous_bbox" if code == "H" else "detected"
        self.frame_records.append({"status": status})
        if self.expose_ratio and status == "detected" and self.script.ratios is not None:
            self.mouth_ratio = self.script.ratios[self.script.i]
        return None

    def close(self) -> None:
        self.closed = True


class _NoRatioTracker(_RatioTracker):
    """5단계와 같은 가짜 추적기: mouth_ratio 속성이 없다."""

    expose_ratio = False


class _FakeGeometry:
    def __init__(self, script: _Script):
        self.script = script

    def process(self, frame: Any, box: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if self.script.current() == "G":
            raise ValueError("context square larger than frame")
        return np.full((96, 96, 3), self.script.i % 251, dtype=np.uint8), {}


def _extract(tmp_path: Any, monkeypatch: Any, codes: str, ratios: list[float | None] | None,
             tracker_cls: Any = _RatioTracker, sub: str = "out", **kw: Any) -> tuple[dict[str, Any], Any]:
    """대본대로 extract_segments를 돌린다(30fps 원본 정수 프레임 경로)."""
    cv2 = _require_cv2()
    n = len(codes)
    video = tmp_path / "v.avi"
    if not video.exists():
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32))
        for _ in range(4):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

    def frames(*_a: Any, **_k: Any) -> Any:
        for i in range(n):
            yield i, i * 1000.0 / 30.0, np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(ot, "iter_adaptive_frames", frames)
    monkeypatch.setattr(ot, "iter_grid_frames", frames)
    monkeypatch.setattr(ot, "choose_selection",
                        lambda *_a, **_k: {"mode": "native_integer", "rate": 30.0, "reason": None,
                                           "source_fps": 30.0, "low_fps": False})
    script = _Script(list(codes), ratios)
    out = tmp_path / sub
    result = ot.extract_segments(video, out, tracker_factory=lambda: tracker_cls(script),
                                 geometry_factory=lambda: _FakeGeometry(script), **kw)
    return result, out


def _summary(result: dict[str, Any]) -> list[tuple[str, int, Any]]:
    return [(s["status"], s["frames"], s.get("reason")) for s in result["segments"]]


def _npz_frames(seg: dict[str, Any]) -> np.ndarray:
    with np.load(seg["npz"]) as data:
        return np.asarray(data["frames"])


def _expected_span(series: list[float | None]) -> tuple[int, int, int, int]:
    span = ot.speech_span(ot.speech_activity(series, RATE), RATE)
    assert span is not None
    return span


def test_upload_mode_marks_silent_window_no_speech_and_deletes_npz(tmp_path, monkeypatch) -> None:
    """업로드: 자르지 않는다. 말이 없는 창은 no_speech(NPZ 삭제), 말한 창은 ok 그대로."""
    codes = "D" * 60 + "G" * 20 + "D" * 90
    ratios: list[float | None] = still(60) + [None] * 20 + still(15) + talk(60) + still(15)
    result, out = _extract(tmp_path, monkeypatch, codes, ratios)
    assert _summary(result) == [("no_speech", 60, ot.NO_SPEECH_REASON), ("failed", 20, "geometry_error"),
                                ("ok", 90, None)]
    silent, _failed, spoken = result["segments"]
    assert silent["npz"] is None and silent["speech"]["active_seconds"] == pytest.approx(0.0)
    assert silent["speech"]["ratio_coverage"] == pytest.approx(1.0)
    assert _npz_frames(spoken).shape[0] == 90  # 업로드 경로는 자르지 않는다
    assert spoken["speech"]["active_seconds"] >= ot.VAD_MIN_SPEECH_SECONDS
    assert [p.name for p in out.glob("*.npz")] == [os.path.basename(spoken["npz"])]
    speech = result["speech"]
    assert speech["enabled"] is True and speech["detected"] is True
    assert speech["trimmed_head_ms"] == 0 and speech["trimmed_tail_ms"] == 0
    assert result["vad"]["activity_threshold"] == ot.VAD_ACTIVITY_THRESHOLD


def test_upload_mode_window_with_unknown_ratios_is_not_judged(tmp_path, monkeypatch) -> None:
    """VAD가 켜져도 비율을 거의 모르는 창은 판정하지 않고 ok(active_seconds None)로 둔다."""
    codes = "D" * 60 + "G" * 20 + "D" * 60
    ratios: list[float | None] = talk(60) + [None] * 80
    result, _out = _extract(tmp_path, monkeypatch, codes, ratios)
    assert _summary(result) == [("ok", 60, None), ("failed", 20, "geometry_error"), ("ok", 60, None)]
    assert result["segments"][2]["speech"] == {"active_seconds": None, "ratio_coverage": 0.0}
    assert result["speech"]["enabled"] is True and result["speech"]["detected"] is True


def test_utterance_mode_trims_head_and_tail(tmp_path, monkeypatch) -> None:
    """녹화: 앞뒤 정지 부분을 잘라 NPZ를 다시 저장하고 시작·끝 ms를 고친다."""
    series: list[float | None] = still(45) + talk(60) + still(45)
    result, _out = _extract(tmp_path, monkeypatch, "D" * 150, series, utterance=True)
    _first, _last, lo, hi = _expected_span(series)
    assert 0 < lo < 45 and 105 < hi < 149
    [seg] = result["segments"]
    assert seg["status"] == "ok" and seg["frames"] == hi - lo + 1
    frames = _npz_frames(seg)
    assert frames.shape == (hi - lo + 1, 96, 96, 3)
    assert int(frames[0, 0, 0, 0]) == lo and int(frames[-1, 0, 0, 0]) == hi  # 가짜 crop 값 = 프레임 번호
    assert seg["start_ms"] == round(lo * 1000 / 30) and seg["end_ms"] == round((hi + 1) * 1000 / 30)
    assert seg["detection"] == {"detected": hi - lo + 1, "previous_bbox": 0}
    speech = result["speech"]
    assert speech["detected"] is True and speech["utterance"] is True
    assert speech["trimmed_head_ms"] == seg["start_ms"]
    assert speech["trimmed_tail_ms"] == round(150 * 1000 / 30) - seg["end_ms"]
    assert speech["start_ms"] <= round(45 * 1000 / 30) and speech["end_ms"] >= round(105 * 1000 / 30)


def test_utterance_mode_drops_windows_and_failed_spans_outside_speech(tmp_path, monkeypatch) -> None:
    """녹화: 발화 구간 밖의 실패 구간·창은 버리고(NPZ 삭제), 남은 창만 번호를 다시 매긴다."""
    codes = "N" * 30 + "D" * 60 + "G" * 20 + "D" * 120 + "G" * 20 + "D" * 60
    ratios: list[float | None] = ([None] * 30 + still(60) + [None] * 20 + still(30) + talk(60) + still(30)
                                  + [None] * 20 + still(60))
    result, out = _extract(tmp_path, monkeypatch, codes, ratios, utterance=True)
    _first, _last, lo, hi = _expected_span(ratios)
    assert 110 < lo < 140 and 200 < hi < 230
    assert _summary(result) == [("ok", hi - lo + 1, None)]
    [seg] = result["segments"]
    assert seg["index"] == 0
    assert int(_npz_frames(seg)[0, 0, 0, 0]) == lo
    assert [p.name for p in out.glob("*.npz")] == [os.path.basename(seg["npz"])]


def test_utterance_mode_keeps_failed_span_inside_speech(tmp_path, monkeypatch) -> None:
    """녹화: 발화 사이에 낀 실패 구간은 남기고 시간 순서를 지킨다."""
    codes = "D" * 75 + "G" * 20 + "D" * 75
    ratios: list[float | None] = still(15) + talk(60) + [None] * 20 + talk(60) + still(15)
    result, _out = _extract(tmp_path, monkeypatch, codes, ratios, utterance=True)
    assert [s["status"] for s in result["segments"]] == ["ok", "failed", "ok"]
    assert [s["index"] for s in result["segments"]] == [0, 1, 2]
    starts = [s["start_ms"] for s in result["segments"]]
    assert starts == sorted(starts)


@pytest.mark.parametrize("codes", ["D" * 120, "D" * 50 + "G" * 5 + "D" * 65])
def test_utterance_mode_silent_recording_is_single_no_speech(tmp_path, monkeypatch, codes) -> None:
    """녹화: 실패 구간 없이 얼굴이 보였고 말이 없으면 읽은 영상 전체를 덮는 no_speech 하나, NPZ 0개.

    보고하지 않는 짧은 실패(0.5초 미만, 창 두 개로 나뉨)만 있으면 여전히 전체가 관측된 녹화다.
    """
    ratios: list[float | None] = [None if c == "G" else v for c, v in zip(codes, still(120))]
    result, out = _extract(tmp_path, monkeypatch, codes, ratios, utterance=True)
    assert len(result["segments"]) == 1
    [seg] = result["segments"]
    assert (seg["status"], seg["reason"], seg["npz"], seg["frames"]) == ("no_speech", ot.NO_SPEECH_REASON, None, 120)
    assert seg["index"] == 0
    assert seg["start_ms"] == 0 and seg["end_ms"] == round(120 * 1000 / 30)
    assert seg["speech"]["active_seconds"] == pytest.approx(0.0)
    assert result["speech"]["enabled"] is True and result["speech"]["detected"] is False
    assert not list(out.glob("*.npz"))


def _rows(result: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [(s["index"], s["status"], s["start_ms"], s["end_ms"], s["frames"], s.get("reason"))
            for s in result["segments"]]


@pytest.mark.parametrize("lost,reason", [("N", "no_face"), ("G", "geometry_error"), ("F", "tracking_lost")])
def test_utterance_mode_silent_face_loss_dominates_keeps_failed_span(tmp_path, monkeypatch, lost, reason) -> None:
    """녹화: 1.5초 가만히 보이다가 6.5초 얼굴·입술을 잃으면, 본 1.5초만 no_speech이고 실패 구간은 남는다.

    (이전에는 0–8초 전체가 '[발화 없음]'이 되어 보지 못한 구간을 말하지 않은 것으로 알렸다.)
    """
    codes = "D" * 45 + lost * 195
    ratios: list[float | None] = still(45) + [None] * 195
    up, _o = _extract(tmp_path, monkeypatch, codes, ratios, sub="up", utterance=False)
    rec, out = _extract(tmp_path, monkeypatch, codes, ratios, sub="rec", utterance=True)
    expected = [(0, "no_speech", 0, 1500, 45, ot.NO_SPEECH_REASON), (1, "failed", 1500, 8000, 195, reason)]
    assert _rows(rec) == expected
    assert _rows(up) == expected  # 업로드 경로와 같은 보고
    assert rec["segments"][0]["npz"] is None and rec["segments"][0]["speech"]["active_seconds"] == pytest.approx(0.0)
    assert rec["speech"]["enabled"] is True and rec["speech"]["detected"] is False
    assert not list(out.glob("*.npz"))


def test_utterance_mode_silent_with_failed_spans_keeps_time_order(tmp_path, monkeypatch) -> None:
    """녹화·발화 없음: 실패 구간 사이의 관측 구간은 각각 no_speech, 이어진 ok 창들은 하나로 합친다."""
    codes = "N" * 30 + "D" * 400 + "G" * 20 + "D" * 50
    ratios: list[float | None] = [None] * 30 + still(400) + [None] * 20 + still(50)
    upload, _o = _extract(tmp_path, monkeypatch, codes, ratios, sub="up")
    assert [s["status"] for s in upload["segments"]] == ["failed", "no_speech", "no_speech", "failed", "no_speech"]
    rec, out = _extract(tmp_path, monkeypatch, codes, ratios, sub="rec", utterance=True)
    ms = lambda f: round(f * 1000 / 30)  # noqa: E731
    assert _rows(rec) == [(0, "failed", 0, ms(30), 30, "no_face"),
                          (1, "no_speech", ms(30), ms(430), 400, ot.NO_SPEECH_REASON),  # 12초 창 두 개 → 하나
                          (2, "failed", ms(430), ms(450), 20, "geometry_error"),
                          (3, "no_speech", ms(450), ms(500), 50, ot.NO_SPEECH_REASON)]
    assert all(s["npz"] is None for s in rec["segments"])
    assert not list(out.glob("*.npz"))


def test_utterance_mode_face_never_found_keeps_failed_span(tmp_path, monkeypatch) -> None:
    """녹화: 얼굴을 끝내 못 찾으면 no_speech가 아니라 실패 구간을 그대로 둔다."""
    result, _out = _extract(tmp_path, monkeypatch, "N" * 60, [None] * 60, utterance=True)
    assert _summary(result) == [("failed", 60, "no_face")]
    assert result["speech"]["enabled"] is False and result["speech"]["detected"] is None


def test_utterance_mode_short_speech_keeps_minimum_window(tmp_path, monkeypatch) -> None:
    """녹화: 잘라 낸 창이 최소 창 길이보다 짧으면 창 안에서 넓혀 최소 길이를 지킨다."""
    series: list[float | None] = still(90) + talk(12) + still(90)
    result, _out = _extract(tmp_path, monkeypatch, "D" * 192, series, utterance=True, min_window_seconds=3.0)
    _first, _last, lo, hi = _expected_span(series)
    assert hi - lo + 1 < 90
    [seg] = result["segments"]
    assert seg["status"] == "ok" and seg["frames"] == 90
    first = int(_npz_frames(seg)[0, 0, 0, 0])
    assert first <= lo and first + 89 >= hi


def _ok_windows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in result["segments"] if s["status"] == "ok"]


def test_utterance_mode_drops_window_left_with_only_padding(tmp_path, monkeypatch) -> None:
    """녹화(12초 초과): 11.6초 말하고 6초 정지, 창이 12초에서 나뉘면 뒤 창(활성 0)은 인식하지 않는다.

    뒤 여유(0.5초)가 창 경계를 넘어도 그 여유만으로 창을 남기지 않는다(NPZ 삭제).
    """
    series: list[float | None] = talk(348) + still(180)
    result, out = _extract(tmp_path, monkeypatch, "D" * len(series), series, utterance=True, cut_search_seconds=0.0)
    [seg] = result["segments"]
    assert seg["status"] == "ok" and seg["start_ms"] == 0 and seg["end_ms"] <= 12000
    assert seg["speech"]["active_seconds"] >= 11.0
    assert [p.name for p in out.glob("*.npz")] == [os.path.basename(seg["npz"])]
    upload, _o = _extract(tmp_path, monkeypatch, "D" * len(series), series, sub="up", cut_search_seconds=0.0)
    assert [s["status"] for s in upload["segments"]] == ["ok", "no_speech"]  # 같은 창 나눔


def test_utterance_mode_merges_short_speech_head_into_next_window(tmp_path, monkeypatch) -> None:
    """녹화(12초 초과): 말이 12초 경계 직전에 시작해 앞 창에 활성이 몇 프레임만 있으면 뒤 창에 이어 붙인다."""
    series: list[float | None] = still(365) + talk(175)
    result, out = _extract(tmp_path, monkeypatch, "D" * len(series), series, utterance=True, cut_search_seconds=0.0)
    active = [a for a in ot.speech_activity(series, RATE)]
    first = active.index(True)
    assert 350 < first < 360  # 앞 창(0..359)에 덩어리 기준 미만의 활성만 있다
    [seg] = result["segments"]
    lo = first - round(ot.VAD_PRE_PAD_SECONDS * RATE)
    assert seg["status"] == "ok" and seg["frames"] == len(series) - lo and seg.get("merged_windows") == 2
    frames = _npz_frames(seg)
    assert frames.shape[0] == seg["frames"]
    assert int(frames[0, 0, 0, 0]) == lo % 251 and int(frames[-1, 0, 0, 0]) == (len(series) - 1) % 251
    assert seg["start_ms"] == round(lo * 1000 / 30)
    assert seg["speech"]["active_seconds"] == pytest.approx(sum(active) / RATE)
    assert [p.name for p in out.glob("*.npz")] == [os.path.basename(seg["npz"])]


def test_utterance_mode_drops_short_head_when_merge_would_exceed_limit(tmp_path, monkeypatch) -> None:
    """이어 붙이면 최대 창+최소 창(360+30)을 넘으면 짧은 활성 창은 버린다."""
    series: list[float | None] = still(365) + talk(384)   # 창 0..359 / 360..748(꼬리 29장 합침, 389장)
    result, out = _extract(tmp_path, monkeypatch, "D" * len(series), series, utterance=True, cut_search_seconds=0.0)
    [seg] = _ok_windows(result)
    assert len(result["segments"]) == 1
    assert seg["start_ms"] == round(360 * 1000 / 30) and seg["frames"] == 389 and "merged_windows" not in seg
    assert int(_npz_frames(seg)[0, 0, 0, 0]) == 360 % 251
    assert [p.name for p in out.glob("*.npz")] == [os.path.basename(seg["npz"])]


def _windows(tmp_path: Any, spans: list[tuple[int, int]]) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """apply_speech_activity 직접 검사용 ok 창들(가짜 crop 값 = 프레임 번호 % 251)과 window_meta."""
    segments, meta = [], {}
    for k, (a, b) in enumerate(spans):
        npz = tmp_path / f"seg_{k:03d}.npz"
        np.savez_compressed(npz, frames=np.stack([np.full((4, 4, 3), p % 251, np.uint8) for p in range(a, b)]))
        segments.append({"index": k, "start_ms": round(a * 1000 / 30), "end_ms": round(b * 1000 / 30),
                         "frames": b - a, "status": "ok", "reason": None, "npz": str(npz),
                         "detection": {"detected": b - a, "previous_bbox": 0}})
        meta[k] = {"positions": list(range(a, b)), "detected": [True] * (b - a)}
    return segments, meta


def test_utterance_mode_split_speech_keeps_most_active_window_and_merges(tmp_path, monkeypatch) -> None:
    """활성이 창 경계로 쪼개져 어느 창도 0.3초에 못 미치면, 활성이 많은 창을 남기고 다른 쪽을 이어 붙인다."""
    n = 120
    active = [False] * n
    for p in range(52, 66):          # 14프레임(0.47초) 발화가 60에서 나뉨: 앞 8 / 뒤 6
        active[p] = True
    monkeypatch.setattr(ot, "speech_activity", lambda _r, _rate: list(active))
    segments, meta = _windows(tmp_path, [(0, 60), (60, 120)])
    pts = [i * 1000 / 30 for i in range(n)]
    speech = ot.apply_speech_activity(segments, meta, [0.13] * n, pts, RATE, 1000 / 30, True, 30, max_frames=360)
    assert speech["detected"] is True
    [seg] = segments
    pre, post = round(ot.VAD_PRE_PAD_SECONDS * RATE), round(ot.VAD_POST_PAD_SECONDS * RATE)
    lo, hi = 52 - pre, 65 + post
    assert seg["index"] == 0 and seg["frames"] == hi - lo + 1 and seg["merged_windows"] == 2
    frames = _npz_frames(seg)
    assert [int(f[0, 0, 0]) for f in frames] == list(range(lo, hi + 1))
    assert seg["speech"]["active_seconds"] == pytest.approx(14 / 30)
    assert not os.path.exists(tmp_path / "seg_001.npz")


def test_utterance_mode_short_activity_not_touching_neighbour_is_dropped(tmp_path, monkeypatch) -> None:
    """짧은 활성이 남긴 창과 경계에서 이어지지 않으면 붙이지 않고 버린다. max_frames 없으면 붙이지 않는다."""
    n = 180
    active = [False] * n
    for p in list(range(10, 60)) + list(range(140, 145)):
        active[p] = True
    monkeypatch.setattr(ot, "speech_activity", lambda _r, _rate: list(active))
    pts = [i * 1000 / 30 for i in range(n)]
    for limit in (360, None):
        segments, meta = _windows(tmp_path, [(0, 90), (90, 180)])
        ot.apply_speech_activity(segments, meta, [0.13] * n, pts, RATE, 1000 / 30, True, 30, max_frames=limit)
        assert [(s["status"], s["start_ms"]) for s in segments] == [("ok", round(0 * 1000 / 30))]
        assert not os.path.exists(tmp_path / "seg_001.npz")


@pytest.mark.parametrize("utterance", [False, True])
def test_vad_disabled_without_ratios_matches_stage5(tmp_path, monkeypatch, utterance) -> None:
    """mouth_ratio가 없는(또는 대부분 모르는) 추적기는 VAD를 끄고 5단계와 같은 구간을 낸다."""
    codes = "N" * 20 + "D" * 100 + "G" * 20 + "D" * 40
    base, _o = _extract(tmp_path, monkeypatch, codes, None, tracker_cls=_NoRatioTracker, sub="base")
    got, _o = _extract(tmp_path, monkeypatch, codes, None, tracker_cls=_NoRatioTracker, sub=f"u{utterance}",
                       utterance=utterance)
    sparse: list[float | None] = [0.13 if i % 3 == 0 else None for i in range(len(codes))]  # 약 33%만 앎
    low, _o = _extract(tmp_path, monkeypatch, codes, sparse, sub=f"s{utterance}", utterance=utterance)
    for result in (got, low):
        assert result["speech"]["enabled"] is False and result["speech"]["detected"] is None
        assert [(s["status"], s["start_ms"], s["end_ms"], s["frames"], s.get("reason"), s.get("detection"))
                for s in result["segments"]] == \
               [(s["status"], s["start_ms"], s["end_ms"], s["frames"], s.get("reason"), s.get("detection"))
                for s in base["segments"]]
        for seg in result["segments"]:
            if seg["status"] == "ok":
                assert seg["speech"]["active_seconds"] is None
                assert _npz_frames(seg).shape[0] == seg["frames"]
    assert _summary(base) == [("failed", 20, "no_face"), ("ok", 100, None), ("failed", 20, "geometry_error"),
                              ("ok", 40, None)]


# ---------------------------------------------------------------------------
# 워커 전달
# ---------------------------------------------------------------------------
def test_worker_forwards_utterance_flag() -> None:
    """olkavs_worker._handle_roi는 요청의 utterance(기본 False)를 extract_segments에 넘긴다."""
    from services import olkavs_worker

    calls: list[dict[str, Any]] = []
    core = SimpleNamespace(extract_segments=lambda video, out_dir, **kw: calls.append(kw) or {"segments": []})
    channel = SimpleNamespace(send=lambda message: None)
    base = {"id": 1, "cmd": "roi", "video": "C:/x/v.webm", "out_dir": "C:/x/out"}
    olkavs_worker._handle_roi(core, dict(base), channel)
    olkavs_worker._handle_roi(core, dict(base, utterance=True), channel)
    olkavs_worker._handle_roi(core, dict(base, utterance=False), channel)
    assert [c["utterance"] for c in calls] == [False, True, False]
    assert calls[1]["segment_seconds"] == 12.0 and callable(calls[1]["progress"])
    with pytest.raises(ValueError):
        olkavs_worker._handle_roi(core, dict(base, utterance="yes"), channel)


def test_worker_forwards_duration_seconds_and_rejects_invalid() -> None:
    """duration_seconds(선택): 없거나 null이면 None, 유한한 양수면 그대로 넘기고, 나머지는 요청 거부."""
    from services import olkavs_worker

    calls: list[dict[str, Any]] = []
    core = SimpleNamespace(extract_segments=lambda video, out_dir, **kw: calls.append(kw) or {"segments": []})
    channel = SimpleNamespace(send=lambda message: None)
    base = {"id": 1, "cmd": "roi", "video": "C:/x/v.webm", "out_dir": "C:/x/out", "utterance": True}
    olkavs_worker._handle_roi(core, dict(base), channel)
    olkavs_worker._handle_roi(core, dict(base, duration_seconds=None), channel)
    olkavs_worker._handle_roi(core, dict(base, duration_seconds=8.04), channel)
    olkavs_worker._handle_roi(core, dict(base, duration_seconds=8), channel)
    assert [c["duration_hint_seconds"] for c in calls] == [None, None, 8.04, 8.0]
    for bad in (0, -1.0, float("nan"), float("inf"), "8", True, [8]):
        with pytest.raises(ValueError):
            olkavs_worker._handle_roi(core, dict(base, duration_seconds=bad), channel)
    assert len(calls) == 4


# ---------------------------------------------------------------------------
# 메타데이터 없는 영상(MediaRecorder WebM): 원본 fps 추정·진행률 분모
# ---------------------------------------------------------------------------
class _PtsCap:
    """cv2.VideoCapture 대체: read()와 get(POS_MSEC)만 흉내 낸다."""

    def __init__(self, pts_list: list[float]):
        self.pts_list = list(pts_list)
        self.i = -1

    def read(self) -> tuple[bool, Any]:
        if self.i + 1 >= len(self.pts_list):
            return False, None
        self.i += 1
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def get(self, prop: int) -> float:
        assert prop == 0
        return float(self.pts_list[self.i])


def test_iter_grid_frames_collects_input_pts_gaps_before_grid_selection() -> None:
    """input_gaps에는 격자 선택 전 원본 프레임의 양수 PTS 간격만 들어간다(반복·역행 PTS 제외)."""
    gaps: list[float] = []
    pts = [0.0, 66.7, 133.3, 133.3, 200.0, 266.7]
    out = list(ot.iter_grid_frames(_PtsCap(pts), 30.0, None, None, {}, input_gaps=gaps))
    assert len(out) >= 8  # 15fps 원본을 30fps 격자로 → 같은 프레임 반복
    assert gaps == pytest.approx([66.7, 66.6, 66.7, 66.7])
    assert ot.estimate_fps_from_gaps(gaps) == pytest.approx(15.0, abs=0.05)


def test_estimate_fps_from_gaps_and_metadata_usable() -> None:
    """중앙값 근처 간격 평균 → fps. 간격이 모자라거나 비정상이면 None. 메타데이터는 fps 1~240·프레임 수 양수만."""
    assert ot.estimate_fps_from_gaps([33.3, 33.4, 100.0, 33.3]) == pytest.approx(1000 / 33.3333, rel=1e-4)
    assert ot.estimate_fps_from_gaps([66.0, 67.0] * 50 + [500.0]) == pytest.approx(1000 / 66.5)  # 1ms 반올림·멈춤 1회
    assert ot.estimate_fps_from_gaps([33.3]) is None
    assert ot.estimate_fps_from_gaps([0.0, -5.0, float("nan")]) is None
    assert ot.metadata_usable(30.0, 240) and ot.metadata_usable(15.0, 120)
    assert not ot.metadata_usable(1000.0, -9223372036854775808)
    assert not ot.metadata_usable(15.0, -138000000000000000) and not ot.metadata_usable(30.0, 0)
    assert ot.choose_selection(1000.0, -9, True)["fps_source"] == "metadata"


def test_validate_duration_hint() -> None:
    """길이 힌트는 None 또는 유한한 양수만."""
    assert ot.validate_duration_hint(None) is None
    assert ot.validate_duration_hint(8) == 8.0 and ot.validate_duration_hint(7.5) == 7.5
    for bad in (0, -1, float("nan"), float("inf"), "8", True):
        with pytest.raises(ValueError):
            ot.validate_duration_hint(bad)


def _extract_no_metadata(tmp_path: Any, monkeypatch: Any, source_gap_ms: float, n: int,
                         **kw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """메타데이터를 쓸 수 없는 영상(격자 경로)을 흉내 낸다: 원본 간격 source_gap_ms, 30fps 격자 n장."""
    cv2 = _require_cv2()
    video = tmp_path / "v.avi"
    if not video.exists():
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32))
        for _ in range(4):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()

    def frames(*_a: Any, input_gaps: list[float] | None = None, **_k: Any) -> Any:
        if input_gaps is not None:
            input_gaps.extend([source_gap_ms] * max(0, int(n * (1000 / 30) / source_gap_ms) - 1))
        for i in range(n):
            yield i, i * 1000.0 / 30.0, np.zeros((32, 32, 3), dtype=np.uint8)

    monkeypatch.setattr(ot, "metadata_usable", lambda *_a, **_k: False)
    monkeypatch.setattr(ot, "iter_grid_frames", frames)
    script = _Script(list("D" * n), still(n))
    progress: list[dict[str, Any]] = []
    result = ot.extract_segments(video, tmp_path / f"out{source_gap_ms}", progress=progress.append,
                                 tracker_factory=lambda: _RatioTracker(script),
                                 geometry_factory=lambda: _FakeGeometry(script), **kw)
    return result, progress


def test_low_fps_from_decoded_pts_when_metadata_unusable(tmp_path, monkeypatch) -> None:
    """메타데이터가 없으면 원본 PTS 간격으로 source_fps·low_fps를 정한다(15fps → 경고, 30fps → 경고 없음)."""
    low, _p = _extract_no_metadata(tmp_path, monkeypatch, 1000 / 15, 240)
    assert low["selection"]["mode"] == "pts_grid" and low["selection"]["fps_source"] == "decoded_pts"
    assert low["selection"]["source_fps"] == pytest.approx(15.0) and low["selection"]["low_fps"] is True
    normal, _p = _extract_no_metadata(tmp_path, monkeypatch, 1000 / 30, 240)
    assert normal["selection"]["source_fps"] == pytest.approx(30.0) and normal["selection"]["low_fps"] is False
    assert normal["selection"]["fps_source"] == "decoded_pts"


def test_progress_uses_duration_hint_when_metadata_unusable(tmp_path, monkeypatch) -> None:
    """메타데이터 길이가 없으면 진행률 분모로 duration_hint_seconds를 쓴다. 없으면 fraction=None 그대로."""
    hinted, progress = _extract_no_metadata(tmp_path, monkeypatch, 1000 / 30, 240, duration_hint_seconds=8.0)
    fractions = [p["fraction"] for p in progress]
    assert fractions and all(f is not None for f in fractions)
    assert fractions == sorted(fractions) and 0 < fractions[0] < 0.5 and fractions[-1] == pytest.approx(1.0, abs=0.01)
    assert hinted["video"]["progress_duration_source"] == "hint"
    assert hinted["video"]["duration_hint_seconds"] == 8.0
    _r, plain = _extract_no_metadata(tmp_path, monkeypatch, 1000 / 30, 240)
    assert plain and all(p["fraction"] is None for p in plain)
    with pytest.raises(ValueError):
        _extract_no_metadata(tmp_path, monkeypatch, 1000 / 30, 60, duration_hint_seconds=float("nan"))
