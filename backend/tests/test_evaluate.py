"""scripts/evaluate.py 단위/통합 테스트 (설계문서 F-MVP-05 / TC-10).

CER/WER 편집 거리와 evaluation_log.csv·그래프 생성을 검증한다.
무거운 모델/전처리는 monkeypatch로 대체한다.
"""

from __future__ import annotations

import csv
import json
import os
import sys

# scripts / backend 경로 추가
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev  # noqa: E402


def test_cer_identical() -> None:
    """동일 문자열의 CER은 0."""
    assert ev.cer("안녕", "안녕") == 0.0


def test_cer_partial_error() -> None:
    """일부 다른 문자열의 CER은 (0, 1) 사이."""
    v = ev.cer("안녕", "안뇽")
    assert 0.0 < v < 1.0


def test_cer_empty_ref() -> None:
    """정답이 비고 예측이 있으면 CER=1."""
    assert ev.cer("", "안녕") == 1.0


def test_wer_word_level() -> None:
    """어절 단위 WER."""
    assert ev.wer("안녕 하세요", "안녕 하세요") == 0.0
    assert ev.wer("안녕 하세요", "안녕") == 0.5


def test_evaluate_writes_csv_and_graph(tmp_path, monkeypatch) -> None:
    """TC-10: 평가 실행 시 evaluation_log.csv + 그래프 생성."""
    ts_dir = tmp_path / "ts"
    ts_dir.mkdir()
    manifest = [
        {"video": "a.mp4", "text": "안녕"},
        {"video": "b.mp4", "text": "하세요"},
    ]
    (ts_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    monkeypatch.setitem(ev.TEST_SETS, "demo", str(ts_dir))
    monkeypatch.setattr(ev, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(ev, "_load_model", lambda: None)
    # 모델/전처리 대체: 항상 '안녕' 예측, 12ms, 검출 성공
    monkeypatch.setattr(ev, "predict_video", lambda vp, model: ("안녕", 12.0, True))

    result = ev.evaluate("demo", "baseline_v1", "word")

    assert os.path.exists(result["csv"])
    assert os.path.exists(result["graph"])
    assert result["num_samples"] == 2
    assert 0.0 <= result["cer"] <= 1.0
    assert result["detection_rate"] == 1.0

    with open(result["csv"], encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "eval_id"  # 헤더
    assert len(rows) == 2  # 헤더 + 1 데이터 행


def test_evaluate_unregistered_test_set(monkeypatch, tmp_path) -> None:
    """미등록 test_set_id는 ValueError."""
    monkeypatch.setattr(ev, "RESULTS_DIR", str(tmp_path / "results"))
    try:
        ev.evaluate("no_such_set", "v1", "word")
        assert False, "ValueError가 발생해야 합니다"
    except ValueError:
        pass
