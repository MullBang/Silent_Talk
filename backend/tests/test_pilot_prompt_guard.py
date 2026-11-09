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


# ── 병실 명령 촬영 묶음(ward_tune·ward_eval): ward_prompts.json으로 점검 ──────────────────────────────

WARD = json.loads(pe.WARD_PROMPTS_PATH.read_text(encoding="utf-8"))
COMMANDS = json.loads(pe.WARD_COMMANDS_PATH.read_text(encoding="utf-8"))


def _ward_meta(kind: str, condition: str, sentence_id: str, rep: int = 1, session: str = "ward_tune",
               participant: str = "P01", text: str | None = None, posture: str = "upright") -> dict:
    """CapturePage가 병실 명령 묶음에서 저장하는 형식의 클립 메타 (text를 주지 않으면 지금 목록의 문장)."""
    if text is None:
        pool = COMMANDS["fixed"] + WARD["decoys"]["tune"] + WARD["decoys"]["eval"] + WARD["short"] + WARD["nonspeech"]
        text = next(p["text"] for p in pool if p["id"] == sentence_id)
    repeats = WARD["sessions"][session]["repeats"] if session in WARD["sessions"] else None
    return {"participant": participant, "kind": kind, "condition": condition, "sentenceId": sentence_id, "rep": rep,
            "text": text, "session": session, "repeats": repeats, "promptsVersion": WARD["version"],
            "commandsVersion": COMMANDS["version"], "posture": posture, "maxSeconds": 5}


def _ward_folder(folder: Path, session: str, participant: str = "P01") -> int:
    """한 사람의 병실 명령 묶음 전체(촬영 화면과 같은 구성)를 가짜 클립으로 만든다. 만든 클립 수를 돌려준다."""
    spec = WARD["sessions"][session]
    n = 0
    for cond, reps in spec["repeats"].items():
        for rep in range(1, reps + 1):
            for p in COMMANDS["fixed"]:
                _clip(folder, f"{participant}_{cond}_{p['id']}_{rep}",
                      **_ward_meta("fixed", cond, p["id"], rep, session, participant))
                n += 1
    extras = [("decoy", "silent", WARD["decoys"][spec["decoys"]]), ("short", "silent", WARD["short"]),
              ("nonspeech", "none", WARD["nonspeech"])]
    for kind, cond, items in extras:
        for p in items:
            _clip(folder, f"{participant}_{cond}_{p['id']}_1", **_ward_meta(kind, cond, p["id"], 1, session, participant))
            n += 1
    if spec["reclined"]:
        for cid in WARD["reclined"]:
            _clip(folder, f"{participant}_reclined_{cid}_1",
                  **_ward_meta("fixed", "reclined", cid, 1, session, participant, posture="reclined"))
            n += 1
    return n


def test_ward_prompts_commands_match_commands_json() -> None:
    assert [(p["id"], p["text"]) for p in WARD["fixed"]] == [(p["id"], p["text"]) for p in COMMANDS["fixed"]]
    assert pe.load_ward_prompts()["version"] == "ward-prompts-v1"
    assert {k: v["clips"] for k, v in WARD["sessions"].items()} == {"ward_tune": 93, "ward_eval": 103}
    assert pe.ward_expected(WARD, "ward_tune") == {"fixed": 70, "decoy": 10, "short": 3, "nonspeech": 10}
    assert pe.ward_expected(WARD, "ward_eval") == {"fixed": 80, "decoy": 10, "short": 3, "nonspeech": 10}


@pytest.mark.parametrize("session,expected", [("ward_tune", 93), ("ward_eval", 103)])
def test_full_ward_folder_passes_check_with_expected_counts(tmp_path: Path, capsys: pytest.CaptureFixture,
                                                           session: str, expected: int) -> None:
    assert _ward_folder(tmp_path, session) == expected
    assert pe.main(["check", "--clips", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["sessions"] == [session] and report["prompts_version"] == "ward-prompts-v1"
    person = report["participants"]["P01"]
    assert person["clips"] == person["expected"] == expected
    assert person["by_kind"] == person["expected_by_kind"]
    assert person["by_posture"]["reclined"] == (10 if session == "ward_tune" else 0)
    assert report["warnings"] == []


def test_ward_folder_reports_missing_clips(tmp_path: Path) -> None:
    _clip(tmp_path, "P03_silent_W01_1", **_ward_meta("fixed", "silent", "W01", participant="P03"))
    report = pe.folder_report(pe.load_clips(tmp_path), pe.load_ward_prompts())
    assert report["ok"] is True
    assert report["participants"]["P03"]["expected"] == 93
    assert any("1/93" in w for w in report["warnings"])


def test_ward_and_sentence_clips_cannot_be_checked_together(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _clip(tmp_path, "P01_silent_W01_1", **_ward_meta("fixed", "silent", "W01"))
    _clip(tmp_path, "P01_voiced_N01_1", **_fixed_meta("N01", session="pilot"))
    with pytest.raises(ValueError, match="촬영 묶음이 섞여"):
        pe.check_rows(pe.load_clips(tmp_path))
    assert pe.main(["check", "--clips", str(tmp_path)]) == 1
    assert "촬영 묶음이 섞여" in capsys.readouterr().out


def test_ward_folder_with_v1_clip_without_session_is_mixed(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_silent_W01_1", **_ward_meta("fixed", "silent", "W01"))
    _clip(tmp_path, "P01_voiced_A01_1", kind="fixed", condition="voiced", sentenceId="A01", rep=1,
          text=next(p["text"] for p in FIXED if p["id"] == "A01"))
    with pytest.raises(ValueError, match="섞여"):
        pe.prompts_for_rows(pe.load_clips(tmp_path))


def test_ward_tune_and_ward_eval_cannot_be_checked_together(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_silent_W01_1", **_ward_meta("fixed", "silent", "W01", session="ward_tune"))
    _clip(tmp_path, "P02_silent_W01_1", **_ward_meta("fixed", "silent", "W01", session="ward_eval", participant="P02"))
    with pytest.raises(ValueError, match="ward_eval, ward_tune"):
        pe.check_rows(pe.load_clips(tmp_path))


def test_ward_clip_with_changed_command_or_decoy_text_is_rejected(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_silent_W05_1", **_ward_meta("fixed", "silent", "W05", text="목이 말라 물 좀 주세요"))
    with pytest.raises(ValueError, match="ward_prompts.json과 다른 클립 1개"):
        pe.check_rows(pe.load_clips(tmp_path))
    _clip(tmp_path, "P01_silent_W05_1", **_ward_meta("fixed", "silent", "W05"))
    pe.check_rows(pe.load_clips(tmp_path))
    _clip(tmp_path, "P01_silent_D01_1", **_ward_meta("decoy", "silent", "D01", text="숨쉬기는 아직 힘들어요"))
    with pytest.raises(ValueError, match="다른 클립 1개"):
        pe.check_rows(pe.load_clips(tmp_path))


def test_eval_decoy_in_tune_folder_is_rejected(tmp_path: Path) -> None:
    """조정용 폴더에 평가용 함정(D11~)이 있으면 그 묶음 목록에 없는 문장으로 거부한다."""
    _clip(tmp_path, "P01_silent_D11_1", **_ward_meta("decoy", "silent", "D11", session="ward_tune"))
    with pytest.raises(ValueError, match="다른 클립 1개"):
        pe.check_rows(pe.load_clips(tmp_path))


def test_ward_prompts_drift_from_commands_json_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drifted = json.loads(json.dumps(WARD, ensure_ascii=False))
    drifted["fixed"][0]["text"] = "간호사 좀 불러 주세요"
    path = tmp_path / "ward_prompts.json"
    path.write_text(json.dumps(drifted, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(pe, "WARD_PROMPTS_PATH", path)
    with pytest.raises(ValueError, match="commands.json과 다릅니다"):
        pe.load_ward_prompts()


def test_ward_score_is_refused_until_supported(tmp_path: Path) -> None:
    """병실 명령 묶음은 아직 채점하지 않는다 (문장 목록으로 잘못 채점하지 않게 score가 거부)."""
    row = {"file": "P01_silent_W01_1.webm", "status": "ok", "session": "ward_tune", "kind": "fixed"}
    (tmp_path / "roi_index.json").write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    assert pe.main(["score", "--out", str(tmp_path), "--device", "cpu"]) == 2


def test_sentence_check_still_uses_prompts_json(tmp_path: Path) -> None:
    _clip(tmp_path, "P01_voiced_N01_1", **_fixed_meta("N01"))
    prompts, source = pe.prompts_for_rows(pe.load_clips(tmp_path))
    assert source == "prompts.json" and prompts["version"] == PROMPTS["version"]
    assert pe.check_rows(pe.load_clips(tmp_path))["version"] == PROMPTS["version"]


@pytest.mark.parametrize("name,meta", [
    # 기댄 자세 클립인데 자세가 upright로 저장됨
    ("P01_reclined_W02_1", dict(kind="fixed", condition="reclined", sentence_id="W02", posture="upright")),
    # 함정 문장을 소리 내어 찍음 (함정은 소리 없이만)
    ("P01_voiced_D01_1", dict(kind="decoy", condition="voiced", sentence_id="D01")),
    # 조정용에는 무음 3회차가 없다
    ("P01_silent_W01_3", dict(kind="fixed", condition="silent", sentence_id="W01", rep=3)),
    # 기댄 자세 목록(reclined)에 없는 명령
    ("P01_reclined_W01_1", dict(kind="fixed", condition="reclined", sentence_id="W01", posture="reclined")),
])
def test_ward_clip_outside_session_plan_is_a_problem(tmp_path: Path, name: str, meta: dict) -> None:
    _ward_folder(tmp_path, "ward_tune")
    _clip(tmp_path, name, **_ward_meta(meta.pop("kind"), meta.pop("condition"), meta.pop("sentence_id"), **meta))
    report = pe.folder_report(pe.load_clips(tmp_path), pe.load_ward_prompts())
    assert report["ok"] is False
    assert any("촬영 계획에 없는 클립 1개" in p and f"{name}.webm" in p for p in report["problems"])


def test_ward_folder_with_swapped_clip_warns_by_kind_and_posture(tmp_path: Path) -> None:
    """클립 수는 93개로 같아도 함정 하나가 빠지고 계획에 없는 기댄 자세 클립이 들어오면 알려 준다."""
    _ward_folder(tmp_path, "ward_tune")
    for suffix in (".webm", ".json"):
        (tmp_path / f"P01_silent_D01_1{suffix}").unlink()
    _clip(tmp_path, "P01_reclined_W01_1", **_ward_meta("fixed", "reclined", "W01", posture="reclined"))
    report = pe.folder_report(pe.load_clips(tmp_path), pe.load_ward_prompts())
    person = report["participants"]["P01"]
    assert person["clips"] == person["expected"] == 93
    assert person["by_kind"] == {"fixed": 71, "decoy": 9, "short": 3, "nonspeech": 10}
    assert person["missing"] == 1
    assert report["ok"] is False and any("P01_reclined_W01_1.webm" in p for p in report["problems"])
    assert any("종류별 클립 수가 촬영 계획과 다릅니다" in w for w in report["warnings"])
    assert any("기댄 자세 클립 11/10개" in w for w in report["warnings"])
    assert any("빠진 클립 1개" in w and "P01_silent_D01_1" in w for w in report["warnings"])


def test_ward_plan_keys_match_capture_order_and_counts() -> None:
    keys = pe.ward_plan_keys(WARD, "ward_tune")
    assert len(keys) == len(set(keys)) == 93
    assert keys[0] == ("fixed", "voiced", "W01", 1, "upright")
    assert keys[20] == ("fixed", "silent", "W01", 1, "upright")
    assert keys[60] == ("decoy", "silent", "D01", 1, "upright")
    assert keys[-1] == ("fixed", "reclined", WARD["reclined"][-1], 1, "reclined")
    assert len(pe.ward_plan_keys(WARD, "ward_eval")) == 103
    with pytest.raises(ValueError, match="ward_test"):
        pe.ward_plan_keys(WARD, "ward_test")


def test_run_refuses_ward_folder_before_lip_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture) -> None:
    """병실 명령 폴더에 run을 걸면 몇 분 걸리는 roi-all을 시작하기 전에 거부한다."""
    _clip(tmp_path, "P01_silent_W01_1", **_ward_meta("fixed", "silent", "W01"))
    started: list[object] = []
    monkeypatch.setattr(pe.subprocess, "run", lambda *a, **k: started.append(a))
    assert pe.main(["run", "--clips", str(tmp_path), "--out", str(tmp_path / "out"), "--device", "cpu"]) == 2
    assert started == [] and not (tmp_path / "out").exists()
    assert "채점은 아직 지원하지 않습니다" in capsys.readouterr().err


def _fake_run_roi(ns: object) -> int:
    """입술 추출 대신 실패 기록만 남긴다 (모델·영상 없이 roi-all 흐름 확인용)."""
    out = Path(ns.out)  # type: ignore[attr-defined]
    out.mkdir(parents=True, exist_ok=True)
    (out / "roi.json").write_text(json.dumps({"error": {"message": "가짜"}}), encoding="utf-8")
    return 1


def test_roi_all_refuses_sealed_ward_eval_unless_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture) -> None:
    clips, out = tmp_path / "clips", tmp_path / "out"
    clips.mkdir()
    _clip(clips, "P02_silent_W01_1", **_ward_meta("fixed", "silent", "W01", session="ward_eval", participant="P02"))
    called: list[str] = []
    monkeypatch.setattr(pe.ot, "run_roi", lambda ns: called.append(ns.video) or _fake_run_roi(ns))
    assert pe.main(["roi-all", "--clips", str(clips), "--out", str(out)]) == 2
    assert called == [] and not out.exists()
    assert "봉인된 평가용 폴더" in capsys.readouterr().err
    # 동결 파일을 만든 뒤에는 명시적으로 허용해서 추출한다.
    assert pe.main(["roi-all", "--clips", str(clips), "--out", str(out), "--allow-sealed"]) == 0
    assert len(called) == 1 and (out / "roi_index.json").exists()


def test_roi_all_runs_on_ward_tune_without_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clips, out = tmp_path / "clips", tmp_path / "out"
    clips.mkdir()
    _clip(clips, "P01_silent_W01_1", **_ward_meta("fixed", "silent", "W01"))
    monkeypatch.setattr(pe.ot, "run_roi", _fake_run_roi)
    assert pe.main(["roi-all", "--clips", str(clips), "--out", str(out)]) == 0
    index = json.loads((out / "roi_index.json").read_text(encoding="utf-8"))
    assert index["rows"][0]["roi_status"] == "failed"
