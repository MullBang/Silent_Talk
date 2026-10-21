"""입 움직임 발화 검출(VAD) 상수 보정 도구 (6단계).

서비스 입술 추출 경로(scripts/olkavs_transcribe.py)의 실제 추적기(MouthRatioLipRoi)와 VAD 함수를 그대로 써서,
영상 + 발화 시간 라벨로 프레임별 입 벌림 비율을 뽑고 후보 상수별 오류율을 계산한다.
팀 파일럿 웹캠 녹화에도 그대로 다시 돌리도록 만들었다.

하위 명령
  dump      (.venv-olkavs-roi) 영상 하나 → 프레임별 PTS·입 벌림 비율 NPZ (+ 발화 구간 라벨 내장)
  aihub     (아무 Python) D: TAR의 감사된 AI Hub 녹화를 하나씩 임시로 꺼내 dump를 실행하고 원본 사본을 지운다
  audio     (아무 Python, numpy) aihub dump에 같은 녹화 WAV의 프레임 에너지를 덧붙인다(라벨 문장 안 실제 쉼 진단)
  evaluate  (아무 Python, numpy 필요) dump NPZ 여러 개 → 활동량 분포, 프레임·구간 단위 오류율, 후보 상수 비교 JSON

라벨 형식(--labels)
  - AI Hub 라벨 JSON: {"Sentence_info":[{"start_time","end_time",...}]} (또는 그런 객체의 목록)
  - 단순 목록: [[start_sec, end_sec], ...]  또는 {"speech": [[start, end], ...]}
  - 빈 목록 [] = 말하지 않는 영상(전체가 무발화). 라벨을 주지 않으면 그 영상은 비율 분포에만 쓴다.

프레임 라벨: 문장 안(경계 0.15초 제외) = 발화, 문장 사이 0.6초 이상 틈 안(경계 0.15초 제외) = 무발화.
녹화 앞·뒤의 무발화도 틈으로 본다(영상 끝 쪽 경계는 제외하지 않음).

사용 예
  .venv-olkavs-roi/Scripts/python.exe scripts/calibrate_vad.py dump --video v.webm --labels v.json --out d/v.npz
  py -3.11 scripts/calibrate_vad.py aihub --stem lip_J_1_F_02_C002_A_004 --work-dir tmp/calib --max-seconds 300
  py -3.11 scripts/calibrate_vad.py evaluate --dump d/*.npz --out experiments/<폴더>/calibration_run.json
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TAR_SOURCE_DIR = ROOT / "experiments" / "local_source_step01_20261007"
DEFAULT_PLAN = TAR_SOURCE_DIR / "d_source_plan.json"
ROI_PYTHON = ROOT / ".venv-olkavs-roi" / "Scripts" / "python.exe"

BOUNDARY_MARGIN = 0.15      # 문장 경계 근처 프레임은 발화/무발화 어느 쪽으로도 세지 않는다
MIN_PAUSE_GAP = 0.6         # 이 길이 이상의 문장 사이 틈만 무발화 프레임으로 쓴다
FAR_PAUSE_DISTANCE = 1.0    # 문장에서 이만큼 떨어진 무발화 = '먼 무발화'(말 직전 준비 동작 제외)
CLIP_CONTEXT = 2.5          # 녹화 모의: 문장 앞뒤로 붙이는 길이
WINDOW_TEST_SECONDS = 6.0   # 업로드 창 모의: 실제 창은 대부분 6~12초
WINDOW_TEST_STEP = 0.5
WINDOW_SPEECH_OVERLAP = 1.0  # 라벨 발화와 이보다 많이 겹치는 창은 no_speech가 되면 안 된다
PAUSE_CLIP_SECONDS = 3.0    # 무발화 모의 녹화 길이
PAUSE_CLIP_MARGIN = 0.5     # 무발화 모의 녹화는 문장에서 이만큼 떨어진 곳에서만 자른다

DEFAULT_GRID = {
    "VAD_ACTIVITY_THRESHOLD": [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012],
    "VAD_WINDOW_SECONDS": [0.4, 0.5, 0.7],
    "VAD_FILL_GAP_SECONDS": [0.4, 0.6, 0.8],
    "VAD_MIN_BURST_SECONDS": [0.2, 0.3],
}
QUICK_GRID = {
    "VAD_ACTIVITY_THRESHOLD": [0.005, 0.006, 0.008],
    "VAD_WINDOW_SECONDS": [0.5],
    "VAD_FILL_GAP_SECONDS": [0.4, 0.6],
    "VAD_MIN_BURST_SECONDS": [0.3],
}
MIN_SPEECH_CANDIDATES = (0.3, 0.4, 0.5)
PAD_CANDIDATES = ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6))
TARGET_RECALL = 0.97


def _vad_module() -> Any:
    """서비스와 같은 olkavs_transcribe 모듈을 읽는다 (VAD 함수·상수의 기준)."""
    try:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import olkavs_transcribe
        return olkavs_transcribe
    except ImportError as exc:
        raise RuntimeError(f"olkavs_transcribe를 불러올 수 없습니다: {exc}") from exc


# ----------------------------------------------------------------------------------
# 라벨
# ----------------------------------------------------------------------------------
def parse_labels(value: Any) -> list[tuple[float, float]]:
    """라벨 JSON 값 → 정렬된 (start, end) 초 목록. 형식이 맞지 않으면 ValueError."""
    try:
        if isinstance(value, dict) and "speech" in value:
            value = value["speech"]
        if isinstance(value, dict) and "Sentence_info" in value:
            value = value["Sentence_info"]
        if isinstance(value, list) and value and isinstance(value[0], dict) and "Sentence_info" in value[0]:
            value = [s for item in value for s in item["Sentence_info"]]
        if not isinstance(value, list):
            raise ValueError("라벨은 목록이거나 Sentence_info/speech 키를 가진 객체여야 합니다")
        out: list[tuple[float, float]] = []
        for item in value:
            if isinstance(item, dict):
                start, end = float(item["start_time"]), float(item["end_time"])
            else:
                start, end = float(item[0]), float(item[1])
            if not (math.isfinite(start) and math.isfinite(end)) or end <= start or start < 0:
                raise ValueError(f"잘못된 발화 구간: {item}")
            out.append((start, end))
        return sorted(out)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"라벨 형식 오류: {exc}") from exc


def load_labels(path: Path) -> list[tuple[float, float]]:
    """라벨 파일을 읽는다 (UTF-8, BOM 허용)."""
    try:
        return parse_labels(json.loads(path.read_text(encoding="utf-8-sig")))
    except OSError as exc:
        raise ValueError(f"라벨 파일을 읽을 수 없습니다: {path}") from exc


# ----------------------------------------------------------------------------------
# dump: 실제 추적기로 프레임별 입 벌림 비율
# ----------------------------------------------------------------------------------
def dump_ratios(video: Path, max_seconds: float | None = None) -> dict[str, Any]:
    """extract_segments와 같은 프레임 선택·추적기·재무장 규칙으로 프레임별 (PTS ms, 비율|None)을 모은다.

    기하(crop)·창 나누기는 비율에 영향이 없어 생략한다. 서비스는 ok 창 프레임만 활동으로 세지만 여기서는
    모든 프레임의 활동을 계산한다(검출 프레임만 비율을 가지므로 차이는 짧은 실패 run 정도다).
    """
    import cv2
    ot = _vad_module()
    cap = cv2.VideoCapture(str(video))
    tracker = None
    try:
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        selection = ot.choose_selection(fps, frame_count, ot.is_webm(video), False)
        stats: dict[str, Any] = {"frames": 0, "detector_resets": 0, "detector_errors": 0, "failed_frames": 0}
        if selection["mode"] == "native_integer":
            frames_iter = ot.iter_adaptive_frames(cap, frame_count, fps, selection, stats)
        else:
            frames_iter = ot.iter_grid_frames(cap, ot.GRID_FPS, None, None, stats)
        tracker_cls = ot.vad_tracker_class()
        tracker = tracker_cls()
        pts_list: list[float] = []
        ratios: list[float | None] = []
        statuses: list[str] = []
        frame_shape = None
        started = time.perf_counter()
        for _index, pts_ms, frame in frames_iter:
            if max_seconds is not None and pts_ms > max_seconds * 1000.0:
                break
            stats["frames"] += 1
            shape = getattr(frame, "shape", (0, 0))[:2]
            if frame_shape is not None and shape != frame_shape:
                tracker.close()
                tracker = tracker_cls()
                stats["detector_resets"] += 1
            frame_shape = shape
            pts_list.append(pts_ms)
            try:
                tracker.process(frame)
            except (ValueError, RuntimeError):
                stats["detector_errors"] += 1
                ratios.append(None)
                statuses.append("detector_error")
                tracker.close()
                tracker = tracker_cls()
                continue
            ratios.append(ot._finite_or_none(getattr(tracker, "mouth_ratio", None)))
            if tracker.failed:
                stats["failed_frames"] += 1
                statuses.append("failed")
                ot.rearm_tracker(tracker)
                continue
            statuses.append(str(tracker.frame_records[-1].get("status")))
        if not pts_list:
            raise ValueError("영상에서 프레임을 읽지 못했습니다")
        stats["seconds"] = round(time.perf_counter() - started, 2)
        return {"pts_ms": pts_list, "ratios": ratios, "statuses": statuses, "rate": float(selection["rate"]),
                "selection": selection, "stats": stats, "fps_metadata": fps, "frame_count_metadata": frame_count}
    finally:
        if tracker is not None:
            tracker.close()
        cap.release()


def save_dump(path: Path, dump: dict[str, Any], labels: list[tuple[float, float]] | None, name: str,
              meta: dict[str, Any] | None = None) -> None:
    """dump 결과를 NPZ(작은 수치 배열 + JSON 메타)로 저장한다."""
    import numpy as np
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {"name": name, "rate": dump["rate"], "selection": dump["selection"], "stats": dump["stats"],
                "fps_metadata": dump["fps_metadata"], "frame_count_metadata": dump["frame_count_metadata"],
                "has_labels": labels is not None, "labels": [list(x) for x in (labels or [])],
                "vad_profile_at_dump": _vad_module().vad_profile(), **(meta or {})}
        np.savez_compressed(path, pts_ms=np.asarray(dump["pts_ms"], dtype=np.float64),
                            ratio=np.asarray([np.nan if r is None else r for r in dump["ratios"]], dtype=np.float64),
                            detected=np.asarray([s == "detected" for s in dump["statuses"]], dtype=bool),
                            info=np.asarray(json.dumps(info, ensure_ascii=False)))
    except (OSError, ValueError, TypeError) as exc:
        raise OSError(f"dump 저장 실패: {path}: {exc}") from exc


def run_dump(args: argparse.Namespace) -> int:
    """dump 하위 명령."""
    try:
        video = Path(args.video).resolve()
        labels = load_labels(Path(args.labels)) if args.labels else None
        dump = dump_ratios(video, args.max_seconds)
        name = args.name or video.stem
        save_dump(Path(args.out), dump, labels, name, {"video_name": video.name, "max_seconds": args.max_seconds})
        known = sum(r is not None for r in dump["ratios"])
        print(json.dumps({"name": name, "frames": len(dump["ratios"]), "known": known, "rate": dump["rate"],
                          "mode": dump["selection"]["mode"], "seconds": dump["stats"]["seconds"]},
                         ensure_ascii=False), flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[dump] 실패: {exc}", file=sys.stderr, flush=True)
        return 1


# ----------------------------------------------------------------------------------
# aihub: D: TAR에서 녹화 하나씩
# ----------------------------------------------------------------------------------
def run_aihub(args: argparse.Namespace) -> int:
    """감사된 TAR 원본(d_source_plan.json)에서 영상·라벨을 하나씩 꺼내 dump하고, 원본 사본은 즉시 지운다."""
    try:
        if str(TAR_SOURCE_DIR) not in sys.path:
            sys.path.insert(0, str(TAR_SOURCE_DIR))
        from tar_source import materialize_source
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        sources = {s["stem"]: s for s in plan["sources"]}
        work = Path(args.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
        for stem in args.stem:
            if stem not in sources:
                raise ValueError(f"계획에 없는 녹화: {stem}")
            source = {"stem": stem, "members": {k: v for k, v in sources[stem]["members"].items()
                                                if k in ("video", "label")}}
            out = work / f"{stem}.npz"
            with materialize_source(source, work / "media") as paths:
                raw = json.loads(paths["label"].read_text(encoding="utf-8-sig"))
                infos = raw if isinstance(raw, list) else [raw]
                sentences = [{"start_time": s["start_time"], "end_time": s["end_time"],
                              "sentence_text": s.get("sentence_text")}
                             for item in infos for s in item["Sentence_info"]]
                label_path = work / f"{stem}_labels.json"
                label_path.write_text(json.dumps({"Sentence_info": sentences}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
                cmd = [str(args.roi_python), "-u", str(Path(__file__).resolve()), "dump", "--video",
                       str(paths["video"]), "--labels", str(label_path), "--out", str(out), "--name", stem]
                if args.max_seconds:
                    cmd += ["--max-seconds", str(args.max_seconds)]
                result = subprocess.run(cmd, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
                                        timeout=args.timeout)
                if result.returncode != 0:
                    raise RuntimeError(f"dump 실패: {stem} (exit {result.returncode})")
            print(f"[aihub] {stem}: {out} (원본 사본 삭제됨)", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print(f"[aihub] 실패: {exc}", file=sys.stderr, flush=True)
        return 1


def frame_audio_db(wav_path: Path, pts_ms: Any, rate: float) -> Any:
    """WAV를 읽어 영상 프레임마다(PTS 중심, 프레임 길이 창) RMS dBFS를 계산한다 (진단 전용, 모델 입력 아님)."""
    import wave
    import numpy as np
    try:
        with wave.open(str(wav_path), "rb") as w:
            channels, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if width != 2:
            raise ValueError(f"16비트 PCM만 지원합니다 (sampwidth={width})")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0
        half = 0.5 / rate
        out = np.full(len(pts_ms), np.nan)
        for i, p in enumerate(np.asarray(pts_ms, dtype=np.float64) / 1000.0):
            a, b = int(max(0.0, p - half) * sr), int((p + half) * sr)
            if b > a and b <= x.size:
                seg = x[a:b]
                out[i] = 10.0 * math.log10(float(np.mean(seg * seg)) + 1e-12)
        return out
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"WAV를 읽을 수 없습니다: {exc}") from exc


def run_audio(args: argparse.Namespace) -> int:
    """aihub dump에 같은 녹화의 WAV 프레임 에너지를 덧붙인다 (라벨 문장 안 실제 쉼 확인용 진단)."""
    import numpy as np
    try:
        if str(TAR_SOURCE_DIR) not in sys.path:
            sys.path.insert(0, str(TAR_SOURCE_DIR))
        from tar_source import materialize_source
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        sources = {s["stem"]: s for s in plan["sources"]}
        work = Path(args.work_dir).resolve()
        for stem in args.stem:
            dump_path = work / f"{stem}.npz"
            with np.load(dump_path, allow_pickle=False) as data:
                arrays = {k: np.asarray(data[k]) for k in data.files}
            info = json.loads(str(arrays["info"]))
            source = {"stem": stem, "members": {"audio": sources[stem]["members"]["audio"]}}
            with materialize_source(source, work / "media") as paths:
                arrays["audio_db"] = frame_audio_db(paths["audio"], arrays["pts_ms"], float(info["rate"]))
            info["audio_db"] = "per-frame RMS dBFS of the source WAV (diagnostic only; not a model input)"
            arrays["info"] = np.asarray(json.dumps(info, ensure_ascii=False))
            np.savez_compressed(dump_path, **arrays)
            print(f"[audio] {stem}: {int(np.isfinite(arrays['audio_db']).sum())} frames (원본 사본 삭제됨)", flush=True)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"[audio] 실패: {exc}", file=sys.stderr, flush=True)
        return 1


AUDIO_VOICED_PAUSE_PERCENTILE = 80.0  # 진단: 그 녹화 '먼 무발화' 프레임 에너지의 80번째 백분위보다 크면 소리 남


def voiced_mask(audio_db: Any, classes: dict[str, Any]) -> tuple[Any, float | None]:
    """프레임별 소리 남 여부와 쓴 문턱(dBFS). 5프레임 중앙값 에너지 > 그 녹화 무발화 에너지의 80번째 백분위.

    소음 환경 녹화가 섞여 있어 고정 dB 대신 녹화별 무발화 분포를 기준으로 한다(라벨 정보 사용).
    라벨 문장 안의 긴 실제 쉼(말을 멈추고 다시 시작 등)을 재현율 분모에서 빼기 위한 진단이며 모델 입력이 아니다.
    """
    import numpy as np
    db = np.asarray(audio_db, dtype=np.float64)
    ref = db[classes["far_pause"] & np.isfinite(db)]
    if ref.size < 30:
        ref = db[classes["pause"] & np.isfinite(db)]
    if ref.size < 30:
        return np.zeros(len(db), bool), None
    level = float(np.percentile(ref, AUDIO_VOICED_PAUSE_PERCENTILE))
    padded = np.pad(np.nan_to_num(db, nan=-120.0), 2, mode="edge")
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    return smooth > level, level


# ----------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------
@contextlib.contextmanager
def patched(module: Any, values: dict[str, Any]) -> Iterator[None]:
    """모듈 상수를 잠시 바꾼다 (VAD 함수는 호출 시점의 상수를 읽는다)."""
    old = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def load_dump(path: Path) -> dict[str, Any]:
    """dump NPZ를 읽는다."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(str(data["info"]))
            t = np.asarray(data["pts_ms"], dtype=np.float64) / 1000.0
            r = np.asarray(data["ratio"], dtype=np.float64)
            audio_db = np.asarray(data["audio_db"], dtype=np.float64) if "audio_db" in data.files else None
        ratios = [None if not math.isfinite(x) else float(x) for x in r]
        return {"name": info["name"], "t": t, "ratios": ratios, "rate": float(info["rate"]), "info": info,
                "labels": [tuple(x) for x in info["labels"]] if info.get("has_labels") else None,
                "audio_db": audio_db, "voiced": None}
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"dump를 읽을 수 없습니다: {path}: {exc}") from exc


def frame_classes(t: Any, labels: list[tuple[float, float]]) -> dict[str, Any]:
    """프레임 분류: speech / pause / far_pause 불 배열과 무발화 틈 목록."""
    import numpy as np
    m = BOUNDARY_MARGIN
    speech = np.zeros(len(t), bool)
    near = np.zeros(len(t), bool)
    for s, e in labels:
        speech |= (t >= s + m) & (t <= e - m)
        near |= (t >= s - FAR_PAUSE_DISTANCE) & (t <= e + FAR_PAUSE_DISTANCE)
    begin, end = float(t[0]), float(t[-1])
    gaps: list[tuple[float, float, bool, bool]] = []  # (start, end, 시작이 문장 끝인가, 끝이 문장 시작인가)
    edges = [(begin, False)] + [x for s, e in labels for x in ((s, True), (e, True))] + [(end, False)]
    for k in range(0, len(edges), 2):
        (a, a_sent), (b, b_sent) = edges[k], edges[k + 1]
        if b - a >= MIN_PAUSE_GAP:
            gaps.append((a, b, a_sent, b_sent))
    pause = np.zeros(len(t), bool)
    for a, b, a_sent, b_sent in gaps:
        lo, hi = a + (m if a_sent else 0.0), b - (m if b_sent else 0.0)
        pause |= (t >= lo) & (t <= hi)
    pause &= ~speech
    return {"speech": speech, "pause": pause, "far_pause": pause & ~near, "gaps": gaps}


def percentiles(values: Any) -> dict[str, float] | None:
    """분포 요약."""
    import numpy as np
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return None
    q = np.percentile(v, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {"n": int(v.size), **{f"p{p}": round(float(x), 5) for p, x in zip((1, 5, 10, 25, 50, 75, 90, 95, 99), q)}}


def activity_measure(ot: Any, ratios: list[float | None], rate: float) -> list[float | None]:
    """speech_activity 안의 활동량(중앙값 필터 → 중심 창 최대-최소)만 계산한다."""
    clean = [ot._finite_or_none(v) for v in ratios]
    smoothed = ot._median_filter(clean, ot.VAD_MEDIAN_FRAMES)
    half = max(1, int(round(ot.VAD_WINDOW_SECONDS * rate / 2)))
    return ot.rolling_ratio_range(smoothed, half)


def evaluate_recording(ot: Any, rec: dict[str, Any], min_speech_values: tuple[float, ...],
                       pads: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    """현재 패치된 상수로 녹화 하나를 평가한다."""
    import numpy as np
    t, ratios, rate, labels = rec["t"], rec["ratios"], rec["rate"], rec["labels"]
    cls = rec["classes"]
    active = np.asarray(ot.speech_activity(ratios, rate), dtype=bool)
    out: dict[str, Any] = {
        "speech_frames": int(cls["speech"].sum()), "pause_frames": int(cls["pause"].sum()),
        "far_pause_frames": int(cls["far_pause"].sum()),
        "speech_recall_hits": int(active[cls["speech"]].sum()),
        "speech_voiced_frames": int((cls["speech"] & rec["voiced"]).sum()) if rec["voiced"] is not None else 0,
        "speech_voiced_hits": int(active[cls["speech"] & rec["voiced"]].sum()) if rec["voiced"] is not None else 0,
        "pause_active": int(active[cls["pause"]].sum()), "far_pause_active": int(active[cls["far_pause"]].sum()),
    }
    csum = np.concatenate([[0], np.cumsum(active)])

    def active_sec(lo: float, hi: float) -> float:
        i, j = np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "right")
        return float(csum[j] - csum[i]) / rate

    sent_active = [active_sec(s, e) for s, e in labels]
    out["sentence_active_seconds_min"] = round(min(sent_active), 3) if sent_active else None
    # 업로드 창 모의: 6초 창이 라벨 발화와 1초 넘게 겹치면 no_speech가 되면 안 된다.
    windows = []
    start = float(t[0])
    while start + WINDOW_TEST_SECONDS <= float(t[-1]) + 1e-9:
        end = start + WINDOW_TEST_SECONDS
        overlap = sum(max(0.0, min(e, end) - max(s, start)) for s, e in labels)
        windows.append((overlap, active_sec(start, end)))
        start += WINDOW_TEST_STEP
    # 녹화 모의: 문장 ±2.5초 클립에서 다시 계산(가장자리 효과 포함).
    clips = []
    for s, e in labels:
        sel = np.where((t >= s - CLIP_CONTEXT) & (t <= e + CLIP_CONTEXT))[0]
        if sel.size == 0:
            continue
        a = ot.speech_activity([ratios[i] for i in sel], rate)
        hits = [k for k, v in enumerate(a) if v]
        clips.append({"sel": sel, "active_s": len(hits) / rate, "hits": hits, "s": s, "e": e})
    # 무발화 모의 녹화: 문장에서 0.5초 이상 떨어진 3초 클립.
    pause_clips = []
    for a, b, a_sent, b_sent in cls["gaps"]:
        lo, hi = a + (PAUSE_CLIP_MARGIN if a_sent else 0.0), b - (PAUSE_CLIP_MARGIN if b_sent else 0.0)
        cs = lo
        while cs + PAUSE_CLIP_SECONDS <= hi + 1e-9:
            sel = np.where((t >= cs) & (t < cs + PAUSE_CLIP_SECONDS))[0]
            if sel.size:
                act = ot.speech_activity([ratios[i] for i in sel], rate)
                pause_clips.append(sum(act) / rate)
            cs += PAUSE_CLIP_SECONDS / 2
    # 말하지 않는 녹화(라벨 []): 녹화 모드 전체 판정이 '발화 없음'이어야 한다. 비율을 아는 프레임이
    # VAD_MIN_COVERAGE 미만이면 서비스는 VAD를 끄므로(얼굴 미검출 등) 이 판정에서 뺀다.
    coverage = sum(r is not None for r in ratios) / max(1, len(ratios))
    whole_active = float(active.sum()) / rate if (not labels and coverage >= ot.VAD_MIN_COVERAGE) else None
    by_min: dict[str, Any] = {}
    for ms in min_speech_values:
        speech_windows = [w for w in windows if w[0] > WINDOW_SPEECH_OVERLAP]
        silent_windows = [w for w in windows if w[0] == 0.0]
        by_min[str(ms)] = {
            "sentences_below_min": sum(1 for x in sent_active if x < ms),
            "speech_windows": len(speech_windows),
            "speech_windows_no_speech": sum(1 for w in speech_windows if w[1] < ms),
            "silent_windows": len(silent_windows),
            "silent_windows_no_speech": sum(1 for w in silent_windows if w[1] < ms),
            "record_clips": len(clips), "record_clips_no_speech": sum(1 for c in clips if c["active_s"] < ms),
            "pause_clips": len(pause_clips), "pause_clips_no_speech": sum(1 for x in pause_clips if x < ms),
            "nonspeech_recordings": int(whole_active is not None),
            "nonspeech_recordings_no_speech": int(whole_active is not None and whole_active < ms),
        }
    by_pad: dict[str, Any] = {}
    for pre, post in pads:
        heads, tails, head_slack, tail_slack = [], [], [], []
        for c in clips:
            if not c["hits"]:
                continue
            flags = [False] * len(c["sel"])
            for k in c["hits"]:
                flags[k] = True
            span = ot.speech_span(flags, rate, pre, post)
            _f, _l, pf, pl = span
            span_start, span_end = t[c["sel"][pf]], t[c["sel"][pl]] + 1.0 / rate
            heads.append(max(0.0, span_start - c["s"]))
            tails.append(max(0.0, c["e"] - span_end))
            head_slack.append(c["s"] - span_start)
            tail_slack.append(span_end - c["e"])
        by_pad[f"{pre}/{post}"] = {
            "clips": len(heads), "head_cut_count": sum(1 for x in heads if x > 1e-6),
            "head_cut_max": round(max(heads), 3) if heads else None,
            "tail_cut_count": sum(1 for x in tails if x > 1e-6),
            "tail_cut_max": round(max(tails), 3) if tails else None,
            "head_slack_median": round(float(np.median(head_slack)), 3) if head_slack else None,
            "head_slack_min": round(float(min(head_slack)), 3) if head_slack else None,
            "tail_slack_median": round(float(np.median(tail_slack)), 3) if tail_slack else None,
            "tail_slack_min": round(float(min(tail_slack)), 3) if tail_slack else None,
        }
    out["by_min_speech"] = by_min
    out["by_pad"] = by_pad
    return out


def merge(per: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """녹화별 결과를 합친다."""
    tot: dict[str, Any] = {}
    keys = ("speech_frames", "pause_frames", "far_pause_frames", "speech_recall_hits", "pause_active",
            "far_pause_active", "speech_voiced_frames", "speech_voiced_hits")
    for k in keys:
        tot[k] = sum(v[k] for v in per.values())
    tot["speech_recall"] = round(tot["speech_recall_hits"] / max(1, tot["speech_frames"]), 4)
    tot["voiced_speech_recall"] = (round(tot["speech_voiced_hits"] / tot["speech_voiced_frames"], 4)
                                   if tot["speech_voiced_frames"] else None)
    tot["pause_active_rate"] = round(tot["pause_active"] / max(1, tot["pause_frames"]), 4)
    tot["far_pause_active_rate"] = round(tot["far_pause_active"] / max(1, tot["far_pause_frames"]), 4)
    tot["min_recording_recall"] = round(min(v["speech_recall_hits"] / max(1, v["speech_frames"])
                                            for v in per.values() if v["speech_frames"]), 4) \
        if any(v["speech_frames"] for v in per.values()) else None
    mins = [v["sentence_active_seconds_min"] for v in per.values() if v["sentence_active_seconds_min"] is not None]
    tot["sentence_active_seconds_min"] = min(mins) if mins else None
    first = next(iter(per.values()))
    tot["by_min_speech"] = {ms: {k: sum(v["by_min_speech"][ms][k] for v in per.values())
                                 for k in first["by_min_speech"][ms]} for ms in first["by_min_speech"]}
    tot["by_pad"] = {}
    for pad in first["by_pad"]:
        rows = [v["by_pad"][pad] for v in per.values() if v["by_pad"][pad]["clips"]]
        tot["by_pad"][pad] = {
            "clips": sum(r["clips"] for r in rows),
            "head_cut_count": sum(r["head_cut_count"] for r in rows),
            "head_cut_max": max((r["head_cut_max"] for r in rows), default=None),
            "tail_cut_count": sum(r["tail_cut_count"] for r in rows),
            "tail_cut_max": max((r["tail_cut_max"] for r in rows), default=None),
            "head_slack_min": min((r["head_slack_min"] for r in rows), default=None),
            "tail_slack_min": min((r["tail_slack_min"] for r in rows), default=None),
        }
    return tot


def feasible(total: dict[str, Any], min_speech: str) -> bool:
    """선정 조건: 라벨 발화 창·문장·녹화 클립을 no_speech로 만들지 않고 프레임 재현율 ≥ 목표.

    WAV 에너지가 있으면 '라벨 문장 안 + 소리 나는' 프레임의 재현율을 쓴다(라벨 문장 안의 긴 실제 쉼 제외).
    없으면 라벨 문장 프레임 재현율을 쓴다.
    """
    b = total["by_min_speech"][min_speech]
    recall = total["voiced_speech_recall"] if total.get("voiced_speech_recall") is not None else total["speech_recall"]
    return (b["sentences_below_min"] == 0 and b["speech_windows_no_speech"] == 0
            and b["record_clips_no_speech"] == 0 and recall >= TARGET_RECALL)


def run_evaluate(args: argparse.Namespace) -> int:
    """evaluate 하위 명령: 후보 격자를 돌려 JSON으로 저장한다."""
    try:
        ot = _vad_module()
        recs = [load_dump(Path(p)) for p in args.dump]
        labelled = [r for r in recs if r["labels"] is not None]
        if not labelled:
            raise ValueError("라벨이 있는 dump가 하나 이상 필요합니다")
        for r in labelled:
            # --max-seconds로 잘린 dump: 영상 범위 밖 라벨은 버리고 끝에 걸친 문장은 영상 끝에서 자른다.
            end = float(r["t"][-1])
            r["labels"] = [(s, min(e, end)) for s, e in r["labels"] if s < end - 2 * BOUNDARY_MARGIN]
            r["classes"] = frame_classes(r["t"], r["labels"])
            if r["audio_db"] is not None:
                r["voiced"], r["voiced_level_db"] = voiced_mask(r["audio_db"], r["classes"])
        grid = QUICK_GRID if args.grid == "quick" else DEFAULT_GRID
        report: dict[str, Any] = {
            "schema": "lipnet-vad-calibration-v1",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_profile": ot.vad_profile(),
            "frame_labels": {"boundary_margin_s": BOUNDARY_MARGIN, "min_pause_gap_s": MIN_PAUSE_GAP,
                             "far_pause_distance_s": FAR_PAUSE_DISTANCE},
            "simulation": {"record_clip_context_s": CLIP_CONTEXT, "upload_window_s": WINDOW_TEST_SECONDS,
                           "upload_window_step_s": WINDOW_TEST_STEP,
                           "upload_window_speech_overlap_s": WINDOW_SPEECH_OVERLAP,
                           "pause_clip_s": PAUSE_CLIP_SECONDS, "pause_clip_margin_s": PAUSE_CLIP_MARGIN},
            "recordings": [],
        }
        for r in recs:
            entry = {"name": r["name"], "frames": len(r["ratios"]), "rate": r["rate"],
                     "seconds": round(float(r["t"][-1] - r["t"][0]), 2),
                     "ratio_known_fraction": round(sum(x is not None for x in r["ratios"]) / len(r["ratios"]), 4),
                     "sentences": len(r["labels"]) if r["labels"] is not None else None,
                     "selection_mode": r["info"]["selection"]["mode"],
                     "audio_voiced_level_db": r.get("voiced_level_db"),
                     "labelled_speech_frames_voiced_fraction":
                         (round(float((r["classes"]["speech"] & r["voiced"]).sum() / max(1, r["classes"]["speech"].sum())), 4)
                          if r.get("voiced") is not None else None)}
            report["recordings"].append(entry)
        # 1) 활동량 분포 (창 길이별)
        dist: dict[str, Any] = {}
        for window in grid["VAD_WINDOW_SECONDS"]:
            with patched(ot, {"VAD_WINDOW_SECONDS": window}):
                pooled: dict[str, list[float | None]] = {"speech": [], "pause": [], "far_pause": []}
                per: dict[str, Any] = {}
                for r in labelled:
                    measure = activity_measure(ot, r["ratios"], r["rate"])
                    row = {}
                    for key in pooled:
                        vals = [measure[i] for i in range(len(measure)) if r["classes"][key][i]]
                        pooled[key].extend(vals)
                        row[key] = percentiles(vals)
                    per[r["name"]] = row
                dist[str(window)] = {"pooled": {k: percentiles(v) for k, v in pooled.items()}, "per_recording": per}
        report["activity_distribution"] = dist
        # 2) 후보 격자
        names = list(grid)
        candidates = []
        for combo in itertools.product(*(grid[k] for k in names)):
            values = dict(zip(names, combo))
            with patched(ot, values):
                per = {r["name"]: evaluate_recording(ot, r, MIN_SPEECH_CANDIDATES, PAD_CANDIDATES) for r in labelled}
            total = merge(per)
            candidates.append({"params": values, "total": total,
                               "per_recording": {n: {k: v[k] for k in ("speech_recall_hits", "speech_frames",
                                                                       "speech_voiced_hits", "speech_voiced_frames",
                                                                       "far_pause_active", "far_pause_frames",
                                                                       "sentence_active_seconds_min")}
                                                 for n, v in per.items()},
                               "feasible": {ms: feasible(total, ms) for ms in total["by_min_speech"]}})
            print(json.dumps({**values, "recall": total["speech_recall"], "voiced": total["voiced_speech_recall"],
                              "far_pause": total["far_pause_active_rate"],
                              "feasible": candidates[-1]["feasible"]}), flush=True)
        report["grid"] = grid
        report["candidates"] = candidates
        # 3) 무라벨 dump(예: 말하지 않는 영상)는 현재 상수로 활동 시간만 보고
        unlabelled = []
        for r in recs:
            if r["labels"] is None:
                act = ot.speech_activity(r["ratios"], r["rate"])
                unlabelled.append({"name": r["name"], "active_seconds": sum(act) / r["rate"]})
        report["unlabelled_current_profile"] = unlabelled
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[evaluate] {len(candidates)}개 후보 → {out}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[evaluate] 실패: {exc}", file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 콘솔 한글
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="(ROI 환경) 영상 → 프레임별 입 벌림 비율 NPZ")
    d.add_argument("--video", required=True)
    d.add_argument("--labels", default=None, help="발화 구간 라벨 JSON (없으면 분포에만 사용)")
    d.add_argument("--out", required=True)
    d.add_argument("--name", default=None)
    d.add_argument("--max-seconds", type=float, default=None)
    a = sub.add_parser("aihub", help="D: TAR 원본을 하나씩 꺼내 dump (원본 사본은 지움)")
    a.add_argument("--plan", default=str(DEFAULT_PLAN))
    a.add_argument("--stem", nargs="+", required=True)
    a.add_argument("--work-dir", required=True)
    a.add_argument("--max-seconds", type=float, default=None)
    a.add_argument("--roi-python", default=str(ROI_PYTHON))
    a.add_argument("--timeout", type=float, default=3600.0)
    w = sub.add_parser("audio", help="aihub dump에 원본 WAV 프레임 에너지 추가(진단)")
    w.add_argument("--plan", default=str(DEFAULT_PLAN))
    w.add_argument("--stem", nargs="+", required=True)
    w.add_argument("--work-dir", required=True)
    e = sub.add_parser("evaluate", help="dump들 → 후보 상수 비교 JSON")
    e.add_argument("--dump", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--grid", choices=("default", "quick"), default="default")
    args = parser.parse_args(argv)
    return {"dump": run_dump, "aihub": run_aihub, "audio": run_audio, "evaluate": run_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
