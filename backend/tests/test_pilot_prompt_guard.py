"""웹캠 촬영 클립 평가 전 검사: 지정 문장 목록 변경·촬영 묶음 섞임을 거부한다 (모델·GPU 불필요).

scripts/pilot_evaluate.py 의 load_clips + check_clip_prompts 를 가짜 클립 폴더(빈 영상 파일 + 메타 JSON)로 확인한다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import pilot_evaluate as pe  # noqa: E402

PROMPTS = json.loads(pe.PROMPTS_PATH.read_text(encoding="utf-8"))
FIXED = PROMPTS["fixed"]


def _clip(folder: Path, name: str, **meta: object) -> None:
    """빈 영상 파일과 CapturePage 형식의 메타 JSON을 만든다."""
    (folder / f"{name}.webm").write_bytes(b"")
    (folder / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _fixed_meta(sentence_id: str, session: str = "pilot", text: str | None = None) -> dict:
    """지정 문장 클립 메타 (text를 주지 않으면 지금 prompts.json의 문장)."""
    current = next(p["text"] for p in FIXED if p["id"] == sentence_id)
    return {"kind": "fixed", "condition": "voiced", "sentenceId": sentence_id, "rep": 1,
            "text": current if text is None else text, "session": session, "promptsVersion": PROMPTS["version"]}


def test_current_prompts_pass_and_metadata_is_kept(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_voiced_N01_1", **_fixed_meta("N01"))
    _clip(tmp_path, "P01_none_X01_1", kind="nonspeech", condition="none", sentenceId="X01", rep=1,
          text="(말하지 않기)", session="pilot")
    rows = pe.load_clips(tmp_path)
    pe.check_clip_prompts(rows, FIXED)
    assert {r["session"] for r in rows} == {"pilot"}
    assert {r["prompts_version"] for r in rows if r["kind"] == "fixed"} == {PROMPTS["version"]}


def test_clip_recorded_with_old_sentence_is_rejected(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_voiced_N01_1", **_fixed_meta("N01", text="오늘 점심은 뭐 먹을까"))
    with pytest.raises(ValueError, match="지정 문장이 지금 prompts.json과 다른 클립 1개"):
        pe.check_clip_prompts(pe.load_clips(tmp_path), FIXED)


def test_pilot_and_eval_sessions_cannot_be_scored_together(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_voiced_N01_1", **_fixed_meta("N01", session="pilot"))
    _clip(tmp_path, "P05_voiced_N01_1", **_fixed_meta("N01", session="eval"))
    with pytest.raises(ValueError, match="촬영 묶음이 섞여"):
        pe.check_clip_prompts(pe.load_clips(tmp_path), FIXED)


def test_free_and_nonspeech_clips_are_not_compared_to_fixed_list(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_voiced_F01_1", kind="free", condition="voiced", sentenceId="F01", rep=1,
          text="자유 문장 1: 아무 말이나 한 문장으로", spoken="오늘은 집에 일찍 갈게")
    pe.check_clip_prompts(pe.load_clips(tmp_path), FIXED)


def test_folder_report_counts_expected_clips_and_missing_transcripts(tmp_path: Path) -> None:
    meta = _fixed_meta("N01", session="eval")
    _clip(tmp_path, "P05_voiced_N01_1", participant="P05", repeats=3, **meta)
    _clip(tmp_path, "P05_voiced_F01_1", participant="P05", kind="free", condition="voiced", sentenceId="F01", rep=1,
          text="자유 문장 1", session="eval", repeats=3)
    report = pe.folder_report(pe.load_clips(tmp_path), PROMPTS)
    assert report["ok"] is True and report["sessions"] == ["eval"]
    assert report["participants"]["P05"]["expected"] == 140
    assert report["participants"]["P05"]["clips"] == 2
    assert any("2/140" in w for w in report["warnings"])
    assert any("spoken" in w for w in report["warnings"])


def test_check_command_fails_on_duplicate_clip_and_passes_clean_folder(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _clip(tmp_path, "P01_voiced_N01_1", participant="P01", repeats=2, **_fixed_meta("N01"))
    assert pe.main(["check", "--clips", str(tmp_path)]) == 0
    (tmp_path / "P01_voiced_N01_1.mp4").write_bytes(b"")
    assert pe.main(["check", "--clips", str(tmp_path)]) == 1
    assert "같은 클립이 여러 개" in capsys.readouterr().out


def test_mac_sidecar_files_are_ignored_and_nested_clip_folder_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _clip(tmp_path, "P02_voiced_N01_1", participant="P02", repeats=2, **_fixed_meta("N01"))
    (tmp_path / "._P02_voiced_N01_1.webm").write_bytes(b"\x00\x05\x16\x07")
    (tmp_path / "._P02_voiced_N01_1.json").write_bytes(b"\x00\x05\x16\x07")
    assert [r["file"] for r in pe.load_clips(tmp_path)] == ["P02_voiced_N01_1.webm"]
    nested = tmp_path / "silent_talk_P03"
    nested.mkdir()
    _clip(nested, "P03_voiced_N01_1", participant="P03", repeats=2, **_fixed_meta("N01"))
    assert pe.main(["check", "--clips", str(tmp_path)]) == 0
    assert "하위 폴더 안의 클립은 세지 않았습니다" in capsys.readouterr().out
