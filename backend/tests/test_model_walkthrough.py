"""모델 과정 보기 HTML 만들기 검사 (모델·mediapipe·GPU 불필요, 가짜 결과 파일로 render만 확인)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import model_walkthrough as mw  # noqa: E402


def _write_fixture(out: Path) -> None:
    """입력 영상 없이 입술 영상만 넣은 경우의 중간 결과를 만든다."""
    film = out / "film"
    film.mkdir(parents=True)
    (film / "00_0000.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (out / "roi_steps.json").write_text(json.dumps({"mode": "lip_video", "rate": 30.0, "frames_log": [], "segments": []}), encoding="utf-8")
    window = {"start_ms": 0, "end_ms": 100, "frames": 3, "input_shape": [1, 3, 3, 96, 96], "feature_shape": [1, 3, 256],
              "hybrid": {"text": "나", "confidence": 0.6, "raw_score": -1.0, "timing_seconds": {"encoder": 0.1, "decode": 0.2}},
              "ctc": {"text": "나", "confidence": 0.6, "timing_seconds": {}}, "ctc_jamo": "ㄴ ㅏ \ㄴ",
              "per_frame": [[{"id": 3, "token": "·", "p": 0.9}], [{"id": 9, "token": "ㄴ", "p": 0.8}], [{"id": 70, "token": "\ㄴ", "p": 0.7}]]}
    (out / "model.json").write_text(json.dumps({"device": "cpu", "windows": [window]}, ensure_ascii=False), encoding="utf-8")


def test_render_marks_low_confidence_final_consonants_and_error_rate(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    assert mw.main(["render", "--out", str(tmp_path), "--reference", "나라", "--title", "<시험>"]) == 0
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&lt;시험&gt;" in page and "<시험>" not in page
    assert "[인식 불가]" in page, "신뢰도 0.6은 기준 0.75보다 낮다"
    assert "<u>ㄴ</u>" in page, "받침은 밑줄"
    assert "50.0% (1/2" in page


def test_char_error_ignores_spaces() -> None:
    assert mw.char_error("나 같이", "나같이") == (0, 3)
    assert mw.char_error("가나", "가다라") == (2, 2)
