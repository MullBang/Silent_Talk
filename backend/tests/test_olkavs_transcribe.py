"""scripts/olkavs_transcribe.py 의 환경 비의존 순수 함수 검사 (mediapipe·GPU·가중치 불필요)."""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import olkavs_transcribe as ot  # noqa: E402

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# 프레임 구간·선택 방식
# ---------------------------------------------------------------------------
def test_select_frame_range_uses_ceil_native_selection() -> None:
    """공식 규칙 ceil(t*fps)로 정수 프레임 구간을 만들고 영상 범위로 자른다."""
    fps = 29.97002997002997
    assert ot.select_frame_range(fps, 9047, 1.7073333333333327, 3.115333333333334) == (52, 94)
    assert ot.select_frame_range(fps, 100, None, None) == (0, 100)
    assert ot.select_frame_range(fps, 100, -5.0, 1000.0) == (0, 100)


def test_select_frame_range_rejects_empty_interval() -> None:
    """끝이 시작보다 앞이면 빈 구간으로 거부한다."""
    with pytest.raises(ValueError):
        ot.select_frame_range(30.0, 100, 2.0, 1.0)


@pytest.mark.parametrize("fps,frames,webm,resample,mode,low", [
    (30000 / 1001, 9047, False, False, "native_integer", False),   # AI Hub 29.97 → 검증된 경로
    (30.0, 300, False, False, "native_integer", False),
    (29.94, 300, False, False, "native_integer", False),
    (25.0, 250, False, False, "pts_grid", True),                    # 27fps 미만 → 품질 경고
    (59.94, 600, False, False, "pts_grid", False),
    (60.0, 600, False, False, "pts_grid", False),
    (120.0, 1200, False, False, "pts_grid", False),
    (240.0, 2400, False, False, "pts_grid", False),
    (15.0, 150, False, False, "pts_grid", True),
    (30.0, 300, True, False, "pts_grid", False),                    # WebM은 항상 격자
    (1000.0, -9, True, False, "pts_grid", False),                   # MediaRecorder WebM 메타데이터
    (0.0, 0, False, False, "pts_grid", False),
    (30.0, 0, False, False, "pts_grid", False),
    (30.0, 300, False, True, "pts_grid", False),
])
def test_choose_selection(fps, frames, webm, resample, mode, low) -> None:
    """약 30fps 고정 프레임률 파일만 원본 정수 프레임, 나머지는 실제 PTS 30fps 격자."""
    sel = ot.choose_selection(fps, frames, webm, resample)
    assert sel["mode"] == mode
    assert sel["rate"] == (fps if mode == "native_integer" else 30.0)
    assert (sel["reason"] is None) == (mode == "native_integer")
    assert sel["low_fps"] is low
    assert sel["fps_source"] == "metadata"  # 메타데이터가 없으면 extract_segments가 원본 PTS로 다시 정한다


class FakeCap:
    """cv2.VideoCapture 대체: read()와 get(POS_MSEC)만 흉내 낸다."""

    def __init__(self, pts_list):
        self.pts_list = list(pts_list)
        self.i = -1

    def read(self):
        if self.i + 1 >= len(self.pts_list):
            return False, None
        self.i += 1
        return True, np.full((2, 2, 3), self.i, dtype=np.uint8)

    def get(self, prop):
        assert prop == 0
        return float(self.pts_list[self.i])


def test_iter_grid_frames_nearest_tie_earlier_and_vfr() -> None:
    """1/30초 격자마다 가장 가까운 실제 프레임, 동률은 앞 프레임. 간격이 들쭉날쭉해도 시간축을 지킨다."""
    pts = [0, 15, 42, 90, 100, 133.3333333]
    out = [(i, p) for i, p, _f in ot.iter_grid_frames(FakeCap(pts), 30.0, None, None)]
    # 격자 0, 33.3, 66.7, 100, 133.3 → 0, 42(33.3에 가까움), 90(66.7은 42와 90 중 ... 42:24.7, 90:23.3), 100, 133.3
    assert [p for _i, p in out] == [0, 42, 90, 100, 133.3333333]
    tie = [(i, p) for i, p, _f in ot.iter_grid_frames(FakeCap([0, 20, 46.6666667]), 30.0, None, None)]
    assert tie[1][1] == 20  # 33.33 격자: 20(13.33)와 46.67(13.33) 동률 → 앞 프레임


def test_iter_grid_frames_rejects_large_pts_regression() -> None:
    """PTS가 한 칸(1/30초)보다 크게 거꾸로 가면 손상으로 보고 오류."""
    with pytest.raises(ValueError, match="단조"):
        list(ot.iter_grid_frames(FakeCap([0, 33, -10, 66]), 30.0, None, None))


def test_iter_all_native_frames_early_end_and_vfr_detection() -> None:
    """메타데이터보다 일찍 끝나도 멈추기만 하고, fps를 주면 가변 프레임률을 감지한다."""
    cfr = [i * 1000 / 30 for i in range(5)]
    assert [i for i, _p, _f in ot.iter_all_native_frames(FakeCap(cfr), 10, 30.0)] == [0, 1, 2, 3, 4]
    assert len(list(ot.iter_all_native_frames(FakeCap(cfr), 3, 30.0))) == 3
    vfr = [0, 33.3, 66.7, 140.0, 173.3]
    with pytest.raises(ot.VariableFrameRateError):
        list(ot.iter_all_native_frames(FakeCap(vfr), 5, 30.0))
    assert len(list(ot.iter_all_native_frames(FakeCap(vfr), 5, None))) == 5  # 검사 끔


# ---------------------------------------------------------------------------
# 창 나누기 (입 움직임 최소 지점)
# ---------------------------------------------------------------------------
def _items(motions):
    return [{"crop": None, "pts": i * 33.3, "motion": m, "status": "detected", "i": i} for i, m in enumerate(motions)]


def _run_builder(motions, max_frames, search, min_frames):
    windows = []
    builder = ot.WindowBuilder(max_frames, search, min_frames, emit=lambda w: windows.append([x["i"] for x in w]))
    for item in _items(motions):
        builder.push(item)
    builder.end_run()
    return windows


def test_window_builder_cuts_at_lowest_motion_and_keeps_order() -> None:
    """최대 길이를 넘으면 [max-search, max] 안의 움직임 최소 지점에서 자르고, 순서·누락이 없다."""
    motions = [5.0] * 25
    motions[8] = 0.1  # 8번 프레임 앞에서 자르는 것이 가장 조용하다
    windows = _run_builder(motions, max_frames=10, search=4, min_frames=3)
    assert windows[0] == list(range(8))
    assert [i for w in windows for i in w] == list(range(25))
    assert all(len(w) <= 10 + 3 for w in windows)


def test_window_builder_merges_short_tail_and_short_runs() -> None:
    """마지막 창이 min_frames보다 짧으면 직전 창에 붙인다. 최대 길이 이하 run은 창 하나."""
    assert _run_builder([1.0] * 7, 10, 2, 3) == [list(range(7))]
    windows = _run_builder([1.0] * 11, 10, 0, 3)  # 10 + 1 → 꼬리 1개는 앞 창에 합침
    assert windows == [list(range(11))]
    windows = _run_builder([1.0] * 14, 10, 0, 3)  # 10 + 4 → 꼬리 4개는 따로
    assert [len(w) for w in windows] == [10, 4]
    assert _run_builder([], 10, 2, 3) == []


def test_window_builder_long_run_lengths() -> None:
    """300프레임 최대·90 탐색·30 최소: 모든 창이 210~329프레임, 전부 이어진다."""
    rng = np.random.RandomState(0)
    windows = _run_builder(list(rng.rand(1000)), 300, 90, 30)
    assert [i for w in windows for i in w] == list(range(1000))
    assert all(210 <= len(w) <= 329 for w in windows[:-1]) and 1 <= len(windows[-1]) <= 329


def test_window_builder_default_service_window_lengths() -> None:
    """서비스 기본값(최대 12초·탐색 6초·최소 1초, 30fps): 중간 창은 180~360프레임, 꼬리 포함 390 이하."""
    rng = np.random.RandomState(7)
    windows = _run_builder(list(rng.rand(3000)), 360, 180, 30)
    assert [i for w in windows for i in w] == list(range(3000))
    assert all(180 <= len(w) <= 360 for w in windows[:-1]) and 1 <= len(windows[-1]) <= 390


def test_best_cut_bounds() -> None:
    """탐색 범위 밖으로 나가지 않는다."""
    assert ot.best_cut([], 3, 5) == 5
    assert 3 <= ot.best_cut([1.0] * 10, 3, 6) <= 6
    assert ot.best_cut([9, 9, 9, 9, 0, 9, 9, 9], 2, 7, smooth=1) == 4


def test_motion_score_zero_for_identical_crops() -> None:
    """같은 crop은 움직임 0, 다르면 양수."""
    a = np.zeros((96, 96, 3), dtype=np.uint8)
    b = np.full((96, 96, 3), 10, dtype=np.uint8)
    assert ot.motion_score(a, a) == 0.0 and ot.motion_score(a, b) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 격자 선택기·가변 프레임률 전환
# ---------------------------------------------------------------------------
def test_grid_selector_skips_small_pts_regression_and_counts() -> None:
    """같은 PTS 반복·한 칸 이내 역행은 건너뛰고 세기만, 한 칸보다 큰 역행은 오류."""
    stats = {}
    out = [p for _i, p, _f in ot.iter_grid_frames(FakeCap([0, 33.3, 33.3, 20.0, 66.7]), 30.0, None, None, stats)]
    assert out == [0, 33.3, 66.7] and stats["pts_non_monotonic_skipped"] == 2


def test_iter_adaptive_frames_switches_to_grid_mid_stream() -> None:
    """원본 정수 프레임으로 읽다가 PTS가 반 프레임 넘게 어긋나면 격자로 이어 읽는다 (앞부분 유지)."""
    pts = [i * 1000 / 30 for i in range(5)] + [200.0, 233.3333, 266.6667, 300.0]
    selection = {"mode": "native_integer", "reason": None}
    stats = {}
    out = list(ot.iter_adaptive_frames(FakeCap(pts), len(pts), 30.0, selection, stats))
    assert [i for i, _p, _f in out[:5]] == [0, 1, 2, 3, 4]
    assert selection["mode"] == "native_then_grid" and selection["switch_frame"] == 5
    assert "variable frame rate" in selection["reason"]
    indices = [i for i, _p, _f in out]
    assert indices == sorted(indices) and indices[-1] == len(pts) - 1
    assert len(out) == 10  # 0~300ms를 1/30초 격자로: 빠진 166.7ms 자리는 가까운 프레임으로 채움
    cfr = [i * 1000 / 30 for i in range(6)]
    plain = {"mode": "native_integer", "reason": None}
    assert len(list(ot.iter_adaptive_frames(FakeCap(cfr), 6, 30.0, plain, {}))) == 6
    assert plain["mode"] == "native_integer"


def test_window_builder_never_emits_short_windows_with_wide_search() -> None:
    """탐색 폭이 최대 길이만큼 넓어도(90/90/30) min보다 짧은 창이 없다."""
    rng = np.random.RandomState(3)
    builder = ot.WindowBuilder(90, 90, 30, emit=lambda w: None)
    assert builder.search_frames == 60 and builder.min_frames == 30
    windows = _run_builder(list(rng.rand(400)), 90, 90, 30)
    assert [i for w in windows for i in w] == list(range(400))
    assert all(30 <= len(w) <= 120 for w in windows)


# ---------------------------------------------------------------------------
# extract_segments 상태 전이 (가짜 검출기·기하 주입, mediapipe 불필요)
# ---------------------------------------------------------------------------
class _Script:
    """프레임 순서별 검출 결과 대본. D=검출, H=직전 상자(미검출 1~11), F=추적 실패, N=처음부터 얼굴 없음,
    E=검출기 예외, G=검출됐지만 기하(문맥 정사각형) 오류."""

    def __init__(self, codes):
        self.codes = codes
        self.i = -1
        self.trackers = []

    def current(self):
        return self.codes[self.i]


class _FakeTracker:
    def __init__(self, script):
        self.script = script
        self.failed = False
        self.failure_reason = None
        self.last_box_xyxy = None
        self.consecutive_miss = 0
        self.frame_records = []
        self.closed = False
        script.trackers.append(self)

    def process(self, frame):
        self.script.i += 1
        code = self.script.current()
        if code == "E":
            raise RuntimeError("detector exploded")
        if code in ("F", "N"):
            self.failed = True
            self.failure_reason = "tracking_lost" if code == "F" else "initial_detection_miss"
            return None
        self.last_box_xyxy = (0, 0, 10, 10)
        self.frame_records.append({"status": "previous_bbox" if code == "H" else "detected"})
        return None

    def close(self):
        self.closed = True


class _FakeGeometry:
    def __init__(self, script):
        self.script = script

    def process(self, frame, box):
        if self.script.current() == "G":
            raise ValueError("context square larger than frame")
        crop = np.full((96, 96, 3), self.script.i % 251, dtype=np.uint8)
        return crop, {}


def _extract(tmp_path, monkeypatch, codes, shapes=None, **kw):
    """대본대로 extract_segments를 돌린다. 실제 작은 영상으로 메타데이터를 만들고 프레임 흐름은 대체한다."""
    cv2 = pytest.importorskip("cv2")  # extract_segments 자체가 cv2를 쓴다 (.venv-gpu에는 없음)

    n = len(codes)
    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32))
    for _ in range(n):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    shapes = shapes or [(32, 32)] * n

    def frames(*_a, **_k):
        for i in range(n):
            yield i, i * 1000.0 / 30.0, np.zeros((*shapes[i], 3), dtype=np.uint8)

    monkeypatch.setattr(ot, "iter_adaptive_frames", frames)
    monkeypatch.setattr(ot, "iter_grid_frames", frames)
    script = _Script(codes)
    progress = []
    result = ot.extract_segments(video, tmp_path / "out", progress=progress.append,
                                 tracker_factory=lambda: _FakeTracker(script),
                                 geometry_factory=lambda: _FakeGeometry(script), **kw)
    return result, script, progress


def _summary(result):
    return [(s["status"], s["frames"], s.get("reason")) for s in result["segments"]]


def _assert_time_order(result):
    starts = [s["start_ms"] for s in result["segments"]]
    assert starts == sorted(starts)
    for a, b in zip(result["segments"], result["segments"][1:]):
        assert a["end_ms"] <= b["start_ms"] + 1


def test_extract_flicker_detection_becomes_one_failed_span(tmp_path, monkeypatch) -> None:
    """검출이 짧게 깜빡이는 구간(검출 9장 미만 run)은 창을 만들지 않고 앞뒤 실패와 합쳐 하나의 실패 구간."""
    codes = list("N" * 30 + "DDDF" * 15 + "N" * 30)
    result, _script, _p = _extract(tmp_path, monkeypatch, codes)
    assert _summary(result) == [("failed", len(codes), "no_face")]
    assert result["stats"]["short_runs_merged"] == 15
    assert not list((tmp_path / "out").glob("*.npz"))


def test_extract_held_frames_join_run_or_gap(tmp_path, monkeypatch) -> None:
    """직전 상자 프레임: 다음 검출이 오면 run에 들어가고, 추적을 잃으면 실패 구간으로 간다."""
    codes = list("D" * 40 + "H" * 5 + "D" * 40 + "H" * 5 + "F" + "N" * 20)
    result, _script, progress = _extract(tmp_path, monkeypatch, codes)
    ok, failed = result["segments"]
    assert (ok["status"], ok["frames"]) == ("ok", 85)
    assert ok["detection"] == {"detected": 80, "previous_bbox": 5}
    assert (failed["status"], failed["frames"]) == ("failed", 26)
    assert failed["start_ms"] == round(85 * 1000 / 30)  # 뒤쪽 보류 5장부터 실패 구간
    assert result["stats"]["held_frames_to_gap"] == 5
    frames = np.load(ok["npz"])["frames"]
    assert frames.shape == (85, 96, 96, 3)
    _assert_time_order(result)
    fractions = [p["fraction"] for p in progress if p["fraction"] is not None]
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0, abs=0.02)


def test_extract_geometry_error_and_detector_error_spans(tmp_path, monkeypatch) -> None:
    """기하 오류·검출기 예외 구간은 실패 구간으로 보고하고, 앞뒤 정상 run은 각각 창이 된다."""
    result, _s, _p = _extract(tmp_path, monkeypatch, list("D" * 30 + "G" * 20 + "D" * 30))
    assert _summary(result) == [("ok", 30, None), ("failed", 20, "geometry_error"), ("ok", 30, None)]
    _assert_time_order(result)
    result, script, _p = _extract(tmp_path, monkeypatch, list("D" * 30 + "E" * 20 + "D" * 30))
    assert _summary(result) == [("ok", 30, None), ("failed", 20, "detector_error"), ("ok", 30, None)]
    assert result["stats"]["detector_resets"] == 20
    assert all(t.closed for t in script.trackers)  # 예외로 버린 검출기와 마지막 검출기 모두 해제


def test_extract_short_gap_is_dropped_but_splits_run(tmp_path, monkeypatch) -> None:
    """0.5초 미만의 짧은 실패는 보고하지 않지만 run은 끊는다(창 두 개)."""
    result, _s, _p = _extract(tmp_path, monkeypatch, list("D" * 30 + "G" * 5 + "D" * 30))
    assert _summary(result) == [("ok", 30, None), ("ok", 30, None)]
    assert result["stats"]["gap_frames_dropped"] == 5


def test_extract_frame_size_change_starts_new_tracker(tmp_path, monkeypatch) -> None:
    """도중에 해상도가 바뀌면 run 경계로 보고 검출기를 새로 만든다."""
    codes = list("D" * 80)
    shapes = [(32, 32)] * 40 + [(24, 32)] * 40
    result, script, _p = _extract(tmp_path, monkeypatch, codes, shapes)
    assert _summary(result) == [("ok", 40, None), ("ok", 40, None)]
    assert result["stats"]["detector_resets"] == 1 and len(script.trackers) == 2
    assert all(t.closed for t in script.trackers)


def test_extract_gap_before_run_is_reported_first(tmp_path, monkeypatch) -> None:
    """앞쪽 얼굴 없음 구간이 뒤의 창보다 먼저 기록된다(시간 순서)."""
    result, _s, _p = _extract(tmp_path, monkeypatch, list("N" * 20 + "D" * 40))
    assert _summary(result) == [("failed", 20, "no_face"), ("ok", 40, None)]
    _assert_time_order(result)


def test_extract_only_short_failures_reports_whole_span(tmp_path, monkeypatch) -> None:
    """프레임은 읽었지만 쓸 만한 입술도 보고할 실패 구간도 없으면 전체를 하나의 실패 구간으로."""
    result, _s, _p = _extract(tmp_path, monkeypatch, list("N" * 10))
    assert _summary(result) == [("failed", 10, "no_usable_lip_frames")]
    assert result["segments"][0]["start_ms"] == 0


def test_extract_rejects_bad_window_seconds(tmp_path, monkeypatch) -> None:
    """창 길이 설정이 양수가 아니면 거부한다."""
    with pytest.raises(ValueError, match="양수"):
        _extract(tmp_path, monkeypatch, list("D" * 5), segment_seconds=float("nan"))


# ---------------------------------------------------------------------------
# 정규화·CTC 점수·인식 후처리
# ---------------------------------------------------------------------------
def test_normalize_frames_matches_published_contract() -> None:
    """발화 전체 평균 0·표준편차 1(correction=1), 출력 (1,3,T,96,96)."""
    rng = np.random.RandomState(0)
    frames = (rng.rand(7, 96, 96, 3) * 255).astype(np.uint8)
    tensor, stats = ot.normalize_frames(frames)
    assert tuple(tensor.shape) == (1, 3, 7, 96, 96)
    assert abs(tensor.mean().item()) < 1e-5
    assert abs(tensor.std(correction=1).item() - 1.0) < 1e-5
    assert stats["raw_mean"] == pytest.approx(frames.astype(np.float32).mean(), rel=1e-6)
    expected = (frames[3, :, :, 1].astype(np.float32) - stats["raw_mean"]) / stats["raw_std"]
    assert np.allclose(tensor[0, 1, 3].numpy(), expected, atol=1e-5)


def test_normalize_frames_rejects_wrong_shape_or_constant() -> None:
    """96x96x3 uint8가 아니거나 분산이 0이면 거부한다."""
    with pytest.raises(ValueError):
        ot.normalize_frames(np.zeros((3, 64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        ot.normalize_frames(np.full((3, 96, 96, 3), 7, dtype=np.uint8))


def _brute_ctc_log_prob(log_probs, tokens, blank):
    """모든 정렬 경로를 열거한 CTC 로그확률 (작은 T·V 전용)."""
    T, V = log_probs.shape
    total = -math.inf
    for path in itertools.product(range(V), repeat=T):
        collapsed, prev = [], None
        for p in path:
            if p != prev and p != blank:
                collapsed.append(p)
            prev = p
        if collapsed == tokens:
            total = np.logaddexp(total, sum(float(log_probs[t, p]) for t, p in enumerate(path)))
    return total


def test_ctc_forced_score_matches_brute_force() -> None:
    """강제 정렬 점수가 전수 열거와 같고, 정렬 불가·빈 토큰은 infeasible."""
    rng = np.random.RandomState(1)
    lp = torch.log_softmax(torch.from_numpy(rng.randn(5, 4)), dim=-1).double()
    for tokens in ([1], [1, 2], [2, 2], [1, 2, 1]):
        got = ot.ctc_forced_score(lp, tokens, blank=3)
        assert got["status"] == "ok"
        assert got["total"] == pytest.approx(_brute_ctc_log_prob(lp.numpy(), tokens, 3), abs=1e-9)
        assert got["total"] <= 0 and got["per_token"] == pytest.approx(got["total"] / len(tokens))
    exact = ot.ctc_forced_score(lp, [1, 1, 1], blank=3)  # 반복 사이 blank 포함 정확히 5프레임 필요
    assert exact["status"] == "ok"
    assert exact["total"] == pytest.approx(_brute_ctc_log_prob(lp.numpy(), [1, 1, 1], 3), abs=1e-9)
    assert ot.ctc_forced_score(lp, [1, 1, 1, 2], blank=3)["status"] == "infeasible"  # 필요 6 > 5
    assert ot.ctc_forced_score(lp, [], blank=3)["status"] == "infeasible"


class _FakeVocab:
    def label_to_string(self, labels):
        return "".join(chr(0x3131 + int(t)) for t in labels.tolist())


class _FakeModel:
    """encoder/medium/ctcdecoder만 흉내: 정해 둔 로짓을 돌려준다."""

    def __init__(self, logits):
        self.logits = logits
        self.decoder = type("D", (), {"ctcdecoder": lambda _s, inputs: self.logits})()

    def encoder(self, video, lengths, *_a):
        return torch.zeros(1, video.shape[2], 4)

    def medium(self, x):
        return x


def test_recognize_ctc_postprocessing(monkeypatch) -> None:
    """EOS 이후 절단, 특수 토큰 제거, confidence 0~1, raw_score = 출력의 CTC 로그우도."""
    T = 8
    path = [5, 5, 3, 6, 3, 2, 7, 3]  # 5 5 _ 6 _ EOS 7 _ → 내용 [5, 6], EOS 뒤 7은 버림
    logits = torch.full((1, T, 72), -8.0)
    for t, k in enumerate(path):
        logits[0, t, k] = 4.0
    bundle = {"model": _FakeModel(logits), "vocab": _FakeVocab(), "grp2char": lambda g: f" {g} ",
              "search_cls": None, "contract": {"strict": True}, "device": "cpu"}
    monkeypatch.setitem(ot._MODEL_CACHE, "cpu", bundle)
    frames = (np.random.RandomState(0).rand(T, 96, 96, 3) * 255).astype(np.uint8)
    out = ot.recognize(frames, "cpu", "ctc")
    assert out["token_ids"] == [5, 6]
    assert out["text"] == _FakeVocab().label_to_string(torch.tensor([5, 6]))
    assert 0.0 <= out["confidence"] <= 1.0 and out["search_cap_reached"] is False
    lp = torch.log_softmax(logits[0].double(), dim=-1)
    assert out["raw_score"] == pytest.approx(ot.ctc_forced_score(lp, [5, 6], 3)["total"])
    assert out["raw_score"] < 0


def test_recognize_empty_output(monkeypatch) -> None:
    """모두 blank면 빈 텍스트, raw_score None."""
    logits = torch.full((1, 6, 72), -8.0)
    logits[0, :, 3] = 4.0
    bundle = {"model": _FakeModel(logits), "vocab": _FakeVocab(), "grp2char": lambda g: g, "search_cls": None}
    monkeypatch.setitem(ot._MODEL_CACHE, "cpu", bundle)
    out = ot.recognize((np.random.RandomState(1).rand(6, 96, 96, 3) * 255).astype(np.uint8), "cpu", "ctc")
    assert out["text"] == "" and out["raw_score"] is None and out["token_ids"] == []


def test_read_frames_npz_roundtrip(tmp_path) -> None:
    """roi.npz 의 frames 배열을 그대로 돌려준다."""
    frames = (np.random.RandomState(1).rand(4, 96, 96, 3) * 255).astype(np.uint8)
    path = tmp_path / "roi.npz"
    np.savez_compressed(path, frames=frames, source_frame_index=np.arange(4))
    loaded = ot.read_frames(path)
    assert loaded.dtype == np.uint8 and np.array_equal(loaded, frames)
