"""병실 '내 입 모양 등록' 서버 쪽 검사 (GPU·실제 모델 불필요, 합성 배열만 쓴다).

- config: 등록 거리 D·긴급 가능성 점수 하한·저장 폴더 환경변수 (기동 시 검사).
- 저장소: 등록 이름 규칙(경로 문자·장치 이름·대소문자 충돌), 저장·바꾸기·개수·완료·stale, 원자적 쓰기.
- 판정: missing·incomplete·stale·no_phrase·error·ok, within·agree·accepted 계산(D 없음 포함), 오프라인 거리와 같음.
- 거리: services/ward_templates.py와 scripts/ward_tools/enroll.py가 무작위 배열에서 같은 값, 함수 본문 글자 그대로 같음.
- 워커·인식: hidden_out이 있을 때만 은닉 특징 저장(값이 순전파 특징과 비트 단위로 같음), 없으면 결과 키가 이전과 같음.
- 서비스: 은닉 특징 임시 npz를 읽은 즉시 지움, 저장·판정 결과, 문장 묶음 요청은 바이트 단위로 이전과 같음.
- API: POST /api/record 등록 필드 400, 등록 목록·상세·삭제, /health 기준값, CORS DELETE 사전 요청.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_TESTS_DIR)
_SCRIPTS = os.path.join(os.path.dirname(_BACKEND_ROOT), "scripts")
for _path in (_TESTS_DIR, _BACKEND_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config  # noqa: E402
from schemas.models import ResultItem  # noqa: E402
from services import olkavs_worker  # noqa: E402
from services import ward_profiles as wp  # noqa: E402
from services import ward_templates  # noqa: E402
from test_olkavs_service import _StubWorker, _roi_result, svc  # noqa: E402
from test_record_api import client, env, jm, make_video, record_module  # noqa: E402,F401 - env는 fixture

from ward_tools import enroll as offline_enroll  # noqa: E402

COMMANDS = wp.current_commands(config.OLKAVS_WARD_PHRASE_FILE)
IDS = COMMANDS["ids"]
CKPT = "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"


def _hidden(seed: int, frames: int = 20) -> np.ndarray:
    """합성 은닉 특징 (frames, 256) float32."""
    return np.random.RandomState(seed).randn(frames, wp.HIDDEN_DIM).astype(np.float32)


@pytest.fixture()
def store(tmp_path, monkeypatch) -> wp.ProfileStore:
    """임시 등록 폴더 (config.WARD_PROFILE_DIR도 바꾼다)."""
    root = tmp_path / "profiles"
    monkeypatch.setattr(config, "WARD_PROFILE_DIR", str(root))
    return wp.ProfileStore(root)


def _complete(store: wp.ProfileStore, profile: str = "P01", checkpoint: str | None = CKPT,
              arrays: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """명령 20개를 모두 등록한다. 등록한 배열을 돌려준다."""
    arrays = arrays or {cid: _hidden(100 + k, 15 + k) for k, cid in enumerate(IDS)}
    for cid in IDS:
        store.store_template(profile, cid, arrays[cid], {"start_ms": 0, "end_ms": 1000, "text_top": cid,
                                                         "text_top_matches": True}, COMMANDS, checkpoint)
    return arrays


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _import_config(overrides: dict[str, str | None]) -> subprocess.CompletedProcess:
    """새 프로세스에서 config를 import해 등록 기준값을 출력한다 (기동 시 검사를 실제로 거친다)."""
    names = ("LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE", "LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE", "LIPNET_WARD_PROFILE_DIR")
    child = {k: v for k, v in os.environ.items() if k not in names}
    child.update({k: v for k, v in overrides.items() if v is not None}, PYTHONIOENCODING="utf-8")
    code = ("import config, json; print(json.dumps([config.OLKAVS_WARD_ENROLL_MAX_DISTANCE, "
            "config.OLKAVS_WARD_POSSIBLE_MIN_SCORE, config.WARD_PROFILE_DIR]))")
    return subprocess.run([sys.executable, "-c", code], cwd=_BACKEND_ROOT, env=child, capture_output=True,
                          text=True, encoding="utf-8", timeout=60)


def test_enroll_thresholds_defaults() -> None:
    """환경변수가 없으면 D=0.27(임시값), 점수 하한 None(꺼짐), 저장 폴더는 backend/data/ward_profiles."""
    out = _import_config({})
    assert out.returncode == 0, out.stderr
    distance, floor, folder = json.loads(out.stdout)
    assert (distance, floor) == (0.27, None)
    assert folder == os.path.join(config.BACKEND_ROOT, "data", "ward_profiles")


@pytest.mark.parametrize("raw,expected", [("0.3", 0.3), ("", None), ("none", None), ("NONE", None), (" None ", None),
                                          ("1e-3", 0.001)])
def test_enroll_max_distance_env_accepts(raw, expected) -> None:
    """D: 0보다 큰 유한한 값, 또는 빈 값·none(대소문자 무관) = 등록으로 고르지 않음."""
    out = _import_config({"LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE": raw})
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)[0] == expected


@pytest.mark.parametrize("raw", ["0", "-0.1", "nan", "inf", "-inf", "abc", "0.27x"])
def test_enroll_max_distance_env_rejects(raw) -> None:
    """D가 0 이하·NaN·무한대·숫자 아님이면 서버 기동(config import)을 막는다."""
    out = _import_config({"LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE": raw})
    assert out.returncode != 0 and "ValueError" in out.stderr


@pytest.mark.parametrize("raw,expected", [("-3.5", -3.5), ("0", 0.0), ("", None), ("none", None)])
def test_possible_min_score_env_accepts(raw, expected) -> None:
    """긴급 가능성 점수 하한: 0 이하의 유한한 값, 또는 빈 값·none = 꺼짐."""
    out = _import_config({"LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE": raw})
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)[1] == expected


@pytest.mark.parametrize("raw", ["0.1", "-inf", "nan", "abc"])
def test_possible_min_score_env_rejects(raw) -> None:
    """점수 하한이 0보다 크거나 NaN·무한대·숫자 아님이면 기동을 막는다."""
    out = _import_config({"LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE": raw})
    assert out.returncode != 0 and "ValueError" in out.stderr


def test_profile_dir_env_and_ward_rule(tmp_path, monkeypatch) -> None:
    """LIPNET_WARD_PROFILE_DIR로 저장 폴더를 바꾼다. ward_rule()은 지금 쓰는 ward 기준값 네 개다."""
    out = _import_config({"LIPNET_WARD_PROFILE_DIR": str(tmp_path / "x")})
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)[2] == str(tmp_path / "x")
    monkeypatch.setattr(config, "OLKAVS_WARD_MIN_MARGIN", 0.7)
    monkeypatch.setattr(config, "OLKAVS_WARD_POSSIBLE_MIN_SCORE", -3.6)
    monkeypatch.setattr(config, "OLKAVS_WARD_ENROLL_MAX_DISTANCE", None)
    assert config.ward_rule() == {"min_margin": 0.7, "min_score": config.OLKAVS_WARD_MIN_SCORE,
                                  "possible_min_score": -3.6, "enroll_max_distance": None}


# ---------------------------------------------------------------------------
# 등록 이름·명령 목록
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["P01", "a", "A-b_9", "x" * 32, "SMOKE", "con1", "COMX"])
def test_profile_id_accepts(value) -> None:
    """영문·숫자·_·- 1~32자."""
    assert wp.is_valid_profile_id(value) and wp.validate_profile_id(value) == value


@pytest.mark.parametrize("value", ["", "x" * 33, "../P01", "..", ".", "a/b", "a\\b", "C:", "C:\\x", "P01 ", " P01",
                                   "한글", "a.b", "a*", "%2e%2e", "P01\n", "CON", "con", "NUL", "aux", "COM1", "lpt9",
                                   None, 1, ["P01"]])
def test_profile_id_rejects_paths_and_reserved(value) -> None:
    """경로 문자·빈 값·길이 초과·Windows 장치 이름·문자열 아님은 거부한다(보낸 값을 문구에 넣지 않는다)."""
    assert not wp.is_valid_profile_id(value)
    with pytest.raises(wp.InvalidProfileId) as caught:
        wp.validate_profile_id(value)
    assert caught.value.user_message == wp.MSG_BAD_PROFILE


_ID_EDGE_CASES = ["P01", "a", "A-b_9", "x" * 32, "x" * 33, "", "P01\n", "\nP01", "P01\r", "CON", "con", "Con", "nul",
                  "PRN", "aux", "COM0", "com9", "LPT1", "lpt0", "COM10", "con1", "COMX", "../P01", "a/b", "a\\b", "C:",
                  "P01 ", "한글", "a.b", "%2e", "P\x0001", None, 1, ["P01"]]


@pytest.mark.parametrize("value", _ID_EDGE_CASES)
def test_profile_id_rule_same_in_batch_tool_and_server(value) -> None:
    """채점 도구(ward_service_batch → common.valid_profile_id)가 서버와 같은 이름만 받아, 틀린 이름은 서버를 부르기 전에 멈춘다."""
    from ward_tools import common

    assert common.valid_profile_id(value) == wp.is_valid_profile_id(value), repr(value)


def test_commands_sha_matches_scoring_tools() -> None:
    """commands_sha256은 commands.json 바이트 전체의 SHA-256이다(채점 도구 common.load_commands와 같다)."""
    from ward_tools import common

    with open(config.OLKAVS_WARD_PHRASE_FILE, "rb") as handle:
        raw = handle.read()
    assert COMMANDS["sha256"] == hashlib.sha256(raw).hexdigest()
    loaded = common.load_commands()
    assert COMMANDS["sha256"] == loaded["sha256"] and IDS == loaded["ids"] and COMMANDS["version"] == loaded["version"]
    assert IDS == [f"W{i:02d}" for i in range(1, 21)]
    assert (wp.FEATURE_NAME, wp.DISTANCE_NAME) == (common.FEATURE_NAME, common.DISTANCE_NAME)


# ---------------------------------------------------------------------------
# 저장소
# ---------------------------------------------------------------------------

def test_store_creates_profile_and_bit_identical_npz(store) -> None:
    """첫 저장이 등록을 만든다. npz는 받은 float32 배열과 비트 단위로 같고, profile.json은 계약 형식이다."""
    hidden = _hidden(1, 37)
    count = store.store_template("P01", "W05", hidden, {"start_ms": 120, "end_ms": 2600, "text_top": "W05",
                                                        "text_top_matches": True}, COMMANDS, CKPT)
    assert count == 1
    folder = store.root / "P01"
    with np.load(folder / "W05.npz") as data:
        saved = data["hidden"]
        assert saved.dtype == np.float32 and saved.tobytes() == hidden.tobytes()
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    assert set(profile) == {"schema", "profile", "created", "updated", "commands_version", "commands_sha256",
                            "checkpoint_sha256", "feature", "distance", "template_mode", "templates"}
    assert (profile["schema"], profile["profile"], profile["template_mode"]) == ("ward-profile-v1", "P01", "voiced")
    assert profile["commands_sha256"] == COMMANDS["sha256"] and profile["checkpoint_sha256"] == CKPT
    assert profile["feature"] == "olkavs_medium_hidden_256"
    assert profile["distance"] == "dtw_cosine_unit_steps_over_path_length"
    assert profile["templates"]["W05"] == {"created": profile["templates"]["W05"]["created"], "frames": 37,
                                           "start_ms": 120, "end_ms": 2600, "text_top": "W05",
                                           "text_top_matches": True}
    assert sorted(os.listdir(folder)) == ["W05.npz", "profile.json"]  # 임시 파일이 남지 않는다
    assert store.summary("P01", CKPT) == {"profile": "P01", "count": 1, "complete": False, "status": "incomplete",
                                          "updated": profile["updated"]}


def test_store_replaces_template_and_completes(store) -> None:
    """같은 명령을 다시 찍으면 바꾼다(개수 그대로). 20개를 모두 등록하면 complete·ok."""
    store.store_template("P01", "W01", _hidden(1, 10), {}, COMMANDS, CKPT)
    replacement = _hidden(2, 12)
    assert store.store_template("P01", "W01", replacement, {"text_top": "W03", "text_top_matches": False},
                                COMMANDS, CKPT) == 1
    detail = store.detail("P01", CKPT)
    assert detail["count"] == 1 and detail["templates"]["W01"]["text_top"] == "W03"
    assert detail["templates"]["W01"]["text_top_matches"] is False
    assert np.array_equal(wp.load_hidden_npz(store.root / "P01" / "W01.npz"), replacement)
    _complete(store)
    assert store.summary("P01", CKPT)["status"] == "ok"
    assert store.detail("P01", CKPT)["complete"] is True and set(store.detail("P01", CKPT)["templates"]) == set(IDS)


def test_status_missing_incomplete_stale_ok(store, tmp_path, monkeypatch) -> None:
    """missing → incomplete → ok, 명령 목록이나 가중치 SHA가 바뀌면 stale (가중치를 모르면 목록만 본다)."""
    assert store.summary("P09", CKPT)["status"] == "missing"
    assert store.detail("P09", CKPT) == {"profile": "P09", "count": 0, "complete": False, "status": "missing",
                                         "templates": {}}
    store.store_template("P01", "W01", _hidden(1), {}, COMMANDS, CKPT)
    assert store.summary("P01", CKPT)["status"] == "incomplete"
    _complete(store)
    assert store.summary("P01", CKPT)["status"] == "ok"
    assert store.summary("P01", None)["status"] == "ok"  # 워커 기동 전: 가중치는 비교하지 않는다
    assert store.summary("P01", "other-checkpoint")["status"] == "stale"
    changed = tmp_path / "commands.json"
    shutil.copyfile(config.OLKAVS_WARD_PHRASE_FILE, changed)
    changed.write_bytes(changed.read_bytes() + b"\n")  # 한 바이트만 달라도 다른 목록이다
    monkeypatch.setattr(config, "OLKAVS_WARD_PHRASE_FILE", str(changed))
    summary = store.summary("P01", CKPT)
    assert summary["status"] == "stale" and summary["count"] == 20 and summary["complete"] is True


def test_store_into_stale_profile_starts_fresh(store) -> None:
    """stale 등록에 저장하면 이전 템플릿을 모두 지우고 새 가중치·목록으로 다시 시작한다(섞어 쓰지 않는다)."""
    _complete(store, checkpoint="old-checkpoint")
    assert store.summary("P01", CKPT)["status"] == "stale"
    assert store.store_template("P01", "W07", _hidden(3), {}, COMMANDS, CKPT) == 1
    assert sorted(os.listdir(store.root / "P01")) == ["W07.npz", "profile.json"]
    profile = json.loads((store.root / "P01" / "profile.json").read_text(encoding="utf-8"))
    assert profile["checkpoint_sha256"] == CKPT and list(profile["templates"]) == ["W07"]
    assert store.summary("P01", CKPT)["status"] == "incomplete"


def test_missing_npz_is_not_counted(store) -> None:
    """profile.json에 있어도 특징 파일이 없으면 세지 않는다(판정에 쓸 수 없다)."""
    _complete(store)
    os.remove(store.root / "P01" / "W20.npz")
    summary = store.summary("P01", CKPT)
    assert (summary["count"], summary["complete"], summary["status"]) == (19, False, "incomplete")
    assert "W20" not in store.detail("P01", CKPT)["templates"]


def test_store_rejects_bad_inputs(store) -> None:
    """목록에 없는 명령, 모양이 틀린 특징, 비정상 값은 저장하지 않는다(폴더도 만들지 않는다)."""
    with pytest.raises(ValueError):
        store.store_template("P01", "W21", _hidden(1), {}, COMMANDS, CKPT)
    for bad in (np.zeros((5, 128), np.float32), np.zeros((0, 256), np.float32), np.zeros(256, np.float32),
                np.full((3, 256), np.nan, np.float32), np.zeros((3, 256), np.int32)):
        with pytest.raises(ValueError):
            store.store_template("P01", "W01", bad, {}, COMMANDS, CKPT)
    with pytest.raises(wp.InvalidProfileId):
        store.store_template("../evil", "W01", _hidden(1), {}, COMMANDS, CKPT)
    assert not store.root.exists() or os.listdir(store.root) == []


def test_case_only_conflict_never_touches_other_profile(store) -> None:
    """대소문자만 다른 이름은 충돌로 거부한다. 다른 등록을 덮어쓰거나 지우지 않는다(Windows 폴더 이름)."""
    store.store_template("Abc", "W01", _hidden(1), {}, COMMANDS, CKPT)
    for call in (lambda: store.store_template("abc", "W02", _hidden(2), {}, COMMANDS, CKPT),
                 lambda: store.delete("ABC"), lambda: store.summary("abc", CKPT)):
        with pytest.raises(wp.ProfileConflict):
            call()
    assert sorted(os.listdir(store.root)) == ["Abc"] and store.summary("Abc", CKPT)["count"] == 1


def test_list_and_delete(store) -> None:
    """목록은 이름순이고 규칙 밖 폴더·profile.json 없는 폴더는 건너뛴다. 삭제하면 폴더째 사라지고, 없으면 ProfileNotFound."""
    store.store_template("P03", "W01", _hidden(1), {}, COMMANDS, CKPT)
    _complete(store, "P01")
    (store.root / "not valid").mkdir()
    (store.root / "P05").mkdir()  # profile.json 없음
    listed = store.list_profiles(CKPT)
    assert [p["profile"] for p in listed] == ["P01", "P03"]
    assert [(p["count"], p["complete"], p["status"]) for p in listed] == [(20, True, "ok"), (1, False, "incomplete")]
    store.delete("P03")
    assert not (store.root / "P03").exists()
    with pytest.raises(wp.ProfileNotFound):
        store.delete("P03")


def test_corrupt_profile_json_is_reported(store) -> None:
    """깨진 profile.json은 고정 문구 오류로 알리고 목록에서는 건너뛴다. 삭제는 된다."""
    store.store_template("P01", "W01", _hidden(1), {}, COMMANDS, CKPT)
    (store.root / "P01" / "profile.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(wp.ProfileError) as caught:
        store.summary("P01", CKPT)
    assert caught.value.user_message == wp.MSG_UNREADABLE
    assert store.list_profiles(CKPT) == []
    store.delete("P01")


def test_profile_dir_writable(tmp_path) -> None:
    """있는 폴더·아직 없는 폴더(상위에 써 봄, 만들지 않음)는 쓸 수 있고, 상위가 파일이면 쓸 수 없다."""
    assert wp.profile_dir_writable(tmp_path) is True
    assert wp.profile_dir_writable(tmp_path / "new" / "deeper") is True
    assert not (tmp_path / "new").exists() and os.listdir(tmp_path) == []
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    assert wp.profile_dir_writable(blocker / "sub") is False


# ---------------------------------------------------------------------------
# 거리 (오프라인 도구와 같은 값)
# ---------------------------------------------------------------------------

def test_dtw_functions_are_verbatim_copies() -> None:
    """unit_rows·dtw_distance·template_distances 본문이 scripts/ward_tools/enroll.py와 글자 그대로 같다."""
    for name in ("unit_rows", "dtw_distance", "template_distances"):
        assert inspect.getsource(getattr(ward_templates, name)) == inspect.getsource(getattr(offline_enroll, name))


def test_dtw_distances_equal_offline_on_random_data() -> None:
    """무작위 길이·값의 특징열에서 서버와 오프라인 거리가 정확히 같다."""
    rng = np.random.RandomState(7)
    for _ in range(12):
        test = rng.randn(rng.randint(1, 60), 256)
        temps = {f"W{k:02d}": [rng.randn(rng.randint(1, 60), 256) for _ in range(rng.randint(1, 3))]
                 for k in range(1, 6)}
        temps["W06"] = []
        server = ward_templates.template_distances(ward_templates.unit_rows(test),
                                                   {c: [ward_templates.unit_rows(t) for t in ts]
                                                    for c, ts in temps.items()})
        offline = offline_enroll.template_distances(offline_enroll.unit_rows(test),
                                                    {c: [offline_enroll.unit_rows(t) for t in ts]
                                                     for c, ts in temps.items()})
        assert server == offline and server["W06"] is None
    a = ward_templates.unit_rows(rng.randn(9, 256))
    assert ward_templates.dtw_distance(a, a) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 판정 (match)
# ---------------------------------------------------------------------------

def _phrase(top: str) -> dict:
    return {"id": top, "text": "x", "score": -3.6, "margin": 0.1, "runner_up_id": "W01", "accepted": False,
            "set": "ward"}


def test_match_statuses_before_distances(store) -> None:
    """등록 상태가 먼저(missing·incomplete·stale), 그다음 녹화 조건(no_phrase), 특징이 없으면 error. 거리 항목은 null."""
    nulls = {"top_id": None, "distance": None, "runner_up_id": None, "runner_up_distance": None, "distances": None,
             "within": False, "agree": False, "accepted": False}
    out = wp.match_result(store, "P01", None, _phrase("W05"), _hidden(1), CKPT, 0.27)
    assert out == {"mode": "match", "profile": "P01", "status": "missing", "count": 0, "max_distance": 0.27, **nulls}
    store.store_template("P01", "W05", _hidden(1), {}, COMMANDS, CKPT)
    assert wp.match_result(store, "P01", None, _phrase("W05"), _hidden(1), CKPT, 0.27)["status"] == "incomplete"
    _complete(store)
    stale = wp.match_result(store, "P01", None, _phrase("W05"), _hidden(1), "new-ckpt", 0.27)
    assert (stale["status"], stale["count"]) == ("stale", 20)
    for reason, phrase in (("multi_window", _phrase("W05")), ("no_speech", None), (None, None)):
        out = wp.match_result(store, "P01", reason, phrase, _hidden(1), CKPT, 0.27)
        assert out == {"mode": "match", "profile": "P01", "status": "no_phrase", "count": 20, "max_distance": 0.27,
                       **nulls}
    err = wp.match_result(store, "P01", None, _phrase("W05"), None, CKPT, 0.27)
    assert err["status"] == "error" and err["distances"] is None and err["accepted"] is False
    bad = wp.match_result(store, "P01", None, _phrase("W05"), np.zeros((4, 3), np.float32), CKPT, 0.27)
    assert bad["status"] == "error"


def test_match_distances_and_decision_maths(store) -> None:
    """ok: 20개 거리 = 오프라인 template_distances, 1·2등, within = 거리 ≤ D(같으면 포함), agree = 글자 1등과 같음."""
    arrays = _complete(store)
    probe = arrays["W07"] + 0.05 * _hidden(9, arrays["W07"].shape[0])
    out = wp.match_result(store, "P01", None, _phrase("W07"), probe, CKPT, 0.27)
    expected = offline_enroll.template_distances(offline_enroll.unit_rows(probe),
                                                 {c: [offline_enroll.unit_rows(a)] for c, a in arrays.items()})
    assert out["status"] == "ok" and out["distances"] == expected and list(out["distances"]) == IDS
    order = sorted(IDS, key=lambda c: (expected[c], IDS.index(c)))
    assert (out["top_id"], out["runner_up_id"]) == (order[0], order[1]) == ("W07", order[1])
    assert out["distance"] == expected["W07"] and out["runner_up_distance"] == expected[order[1]]
    assert out["distance"] < 0.27 and (out["within"], out["agree"], out["accepted"]) == (True, True, True)
    exact = wp.match_result(store, "P01", None, _phrase("W07"), probe, CKPT, out["distance"])
    assert exact["within"] is True and exact["accepted"] is True  # 거리 = D도 통과
    tight = wp.match_result(store, "P01", None, _phrase("W07"), probe, CKPT, out["distance"] * 0.999)
    assert (tight["within"], tight["agree"], tight["accepted"]) == (False, True, False)
    disagree = wp.match_result(store, "P01", None, _phrase("W03"), probe, CKPT, 0.27)
    assert (disagree["within"], disagree["agree"], disagree["accepted"]) == (True, False, False)
    off = wp.match_result(store, "P01", None, _phrase("W07"), probe, CKPT, None)
    assert off["status"] == "ok" and off["distances"] == expected and off["max_distance"] is None
    assert (off["within"], off["agree"], off["accepted"]) == (False, True, False)  # D 없음: 거리만 남긴다


def test_match_ties_break_by_command_order(store) -> None:
    """거리가 같으면 명령 목록 순서가 앞선 쪽이 1등이다."""
    same = _hidden(5, 10)
    _complete(store, arrays={cid: same for cid in IDS})
    out = wp.match_result(store, "P01", None, _phrase("W01"), same, CKPT, 0.27)
    assert (out["top_id"], out["runner_up_id"]) == ("W01", "W02")


def test_store_result_reasons(store) -> None:
    """저장 모드: 사유가 있거나 지정 문장·특징이 없으면 저장하지 않고 사유를 남긴다. 글자 1등이 달라도 저장은 한다."""
    meta = {"start_ms": 0, "end_ms": 2000}
    for reason, phrase, hidden, expected in (("multi_window", _phrase("W05"), _hidden(1), "multi_window"),
                                             ("partial_detection", None, None, "partial_detection"),
                                             ("no_speech", None, None, "no_speech"),
                                             (None, None, _hidden(1), "no_phrase"),
                                             (None, _phrase("W05"), None, "no_features")):
        out = wp.store_result(store, "P01", "W05", reason, phrase, hidden, meta, CKPT)
        assert (out["stored"], out["reason"], out["count"], out["complete"]) == (False, expected, 0, False)
    out = wp.store_result(store, "P01", "W05", None, _phrase("W09"), _hidden(1), meta, CKPT)
    assert out == {"mode": "store", "profile": "P01", "command": "W05", "stored": True, "reason": None,
                   "text_top": "W09", "text_top_matches": False, "count": 1, "complete": False}
    store.store_template("Q1", "W01", _hidden(1), {}, COMMANDS, CKPT)
    conflict = wp.store_result(store, "q1", "W02", None, _phrase("W02"), _hidden(1), meta, CKPT)
    assert (conflict["stored"], conflict["reason"]) == (False, "error")


@pytest.mark.parametrize("segments,expected", [
    ([{"status": "ok", "npz": "a"}], None),
    ([{"status": "no_speech"}], "no_speech"),
    ([{"status": "no_speech"}, {"status": "no_speech"}], "no_speech"),
    ([{"status": "failed"}], "partial_detection"),
    ([{"status": "ok", "npz": "a"}, {"status": "failed"}], "partial_detection"),
    ([{"status": "ok"}], "partial_detection"),
    ([{"status": "ok", "npz": "a"}, {"status": "ok", "npz": "b"}], "multi_window"),
    ([{"status": "ok", "npz": "a"}, {"status": "no_speech"}], "multi_window"),
    ([], "partial_detection"),
])
def test_segments_reason(segments, expected) -> None:
    """입술 구간 → 등록에 쓸 수 없는 사유 (창 하나·ok만 None)."""
    assert wp.segments_reason(segments) == expected


# ---------------------------------------------------------------------------
# 워커·인식 (hidden_out)
# ---------------------------------------------------------------------------

class _HiddenModel:
    """encoder/medium/ctcdecoder 흉내: 무작위 은닉 특징(1, T, 256)과 정해 둔 로짓."""

    def __init__(self, logits, hidden):
        import torch

        self.logits = logits
        self.hidden = torch.from_numpy(hidden).unsqueeze(0)
        self.decoder = type("D", (), {"ctcdecoder": lambda _s, inputs: self.logits})()

    def encoder(self, video, lengths, *_a):
        return self.hidden.clone()

    def medium(self, x):
        return x * 1.0


def _recognize_bundle(monkeypatch, frames_count: int = 8):
    """가짜 모델을 cpu 캐시에 등록하고 (ot 모듈, 입력 프레임, 순전파 은닉 특징)을 돌려준다."""
    torch = pytest.importorskip("torch")
    import olkavs_transcribe as ot
    from test_olkavs_transcribe import _FakeVocab

    logits = torch.full((1, frames_count, 72), -8.0)
    for t, k in enumerate([5, 3, 6, 3, 2, 7, 3, 3][:frames_count]):
        logits[0, t, k] = 4.0
    hidden = np.random.RandomState(3).randn(frames_count, 256).astype(np.float64)
    bundle = {"model": _HiddenModel(logits, hidden), "vocab": _FakeVocab(), "grp2char": lambda g: g,
              "search_cls": None, "contract": {"strict": True}, "device": "cpu"}
    monkeypatch.setitem(ot._MODEL_CACHE, "cpu", bundle)
    frames = (np.random.RandomState(0).rand(frames_count, 96, 96, 3) * 255).astype(np.uint8)
    return ot, frames, bundle


_RECOGNIZE_KEYS = {"text", "confidence", "raw_score", "raw_score_status", "decoder", "frames", "search_cap_reached",
                   "token_ids", "timing_seconds"}


def test_recognize_without_hidden_out_keeps_result_keys(monkeypatch, tmp_path) -> None:
    """hidden_out이 없으면 결과 키가 이 기능 이전과 같고 파일을 쓰지 않는다."""
    ot, frames, _bundle = _recognize_bundle(monkeypatch)
    out = ot.recognize(frames, "cpu", "ctc")
    assert set(out) == _RECOGNIZE_KEYS
    phrases = [{"id": "A", "text": "가"}, {"id": "B", "text": "나"}]
    monkeypatch.setattr(ot, "phrase_tokens", lambda text, _b: {"가": [5, 6], "나": [7, 8]}[text])
    assert set(ot.recognize(frames, "cpu", "ctc", phrases)) == _RECOGNIZE_KEYS | {"phrase"}
    assert list(tmp_path.iterdir()) == []


def test_recognize_hidden_out_saves_forward_features(monkeypatch, tmp_path) -> None:
    """hidden_out: 같은 순전파의 features[0].float().cpu().numpy()를 저장하고 hidden_saved=True. 다른 결과는 그대로."""
    ot, frames, bundle = _recognize_bundle(monkeypatch)
    before = ot.recognize(frames, "cpu", "ctc")
    path = tmp_path / "hidden.npz"
    out = ot.recognize(frames, "cpu", "ctc", None, hidden_out=str(path))
    assert out.pop("hidden_saved") is True and set(out) == _RECOGNIZE_KEYS
    assert {k: v for k, v in out.items() if k != "timing_seconds"} == \
        {k: v for k, v in before.items() if k != "timing_seconds"}
    expected = (bundle["model"].hidden * 1.0)[0].float().cpu().numpy()
    with np.load(path) as data:
        assert data["hidden"].dtype == np.float32 and data["hidden"].tobytes() == expected.tobytes()
    failed = ot.recognize(frames, "cpu", "ctc", None, hidden_out=str(tmp_path / "missing_dir" / "h.npz"))
    assert failed["hidden_saved"] is False and failed["text"] == before["text"]  # 저장 실패가 인식을 막지 않는다


def test_worker_handle_infer_hidden_out_only_when_requested() -> None:
    """워커: hidden_out이 없으면 공용 코어 호출이 이전과 같고, 있으면 키워드로 넘긴다. 잘못된 값은 거부."""
    calls = []

    def recognize(frames, device, decoder, *args, **kwargs):
        calls.append((args, kwargs))
        return {"text": "가", "token_ids": [1]}

    core = types.SimpleNamespace(read_frames=lambda p: [0], recognize=recognize)
    olkavs_worker._handle_infer(core, {"npz": "a.npz"}, "cpu", "hybrid")
    olkavs_worker._handle_infer(core, {"npz": "a.npz", "phrases": [{"id": "A"}]}, "cpu", "hybrid")
    target = os.path.abspath("hidden_x.npz")
    out = olkavs_worker._handle_infer(core, {"npz": "a.npz", "hidden_out": target}, "cpu", "ctc")
    olkavs_worker._handle_infer(core, {"npz": "a.npz", "phrases": [{"id": "A"}], "hidden_out": target}, "cpu", "ctc")
    assert calls == [((), {}), (([{"id": "A"}],), {}), ((None,), {"hidden_out": target}),
                     (([{"id": "A"}],), {"hidden_out": target})]
    assert "token_ids" not in out
    for bad in ("relative.npz", os.path.abspath("x.txt"), 3, True, ""):
        with pytest.raises(ValueError):
            olkavs_worker._handle_infer(core, {"npz": "a.npz", "hidden_out": bad}, "cpu", "ctc")


# ---------------------------------------------------------------------------
# 서비스 (대체 워커)
# ---------------------------------------------------------------------------

def _one_window() -> list[dict]:
    return [{"index": 0, "start_ms": 300, "end_ms": 2800, "status": "ok", "npz": "a.npz"}]


def _infer_handler(hidden: np.ndarray | None, top: str = "W05", saved: bool = True, seen: list | None = None):
    """대체 인식 워커: hidden_out이 오면 그 경로에 은닉 특징을 저장하고 hidden_saved를 알린다."""
    def handler(payload, _progress):
        if seen is not None:
            seen.append(payload.get("hidden_out"))
        out = {"ok": True, "text": "목이 말라", "confidence": 0.9, "raw_score": -3.0, "decoder": payload.get("decoder"),
               "phrase": {"id": top, "text": "x", "score": -3.6, "margin": 0.2, "runner_up_id": "W03"}}
        if "hidden_out" in payload:
            if saved and hidden is not None:
                np.savez(payload["hidden_out"], hidden=hidden)
            out["hidden_saved"] = bool(saved and hidden is not None)
        return out
    return handler


def _ward_service(handler, segments=None):
    roi = _StubWorker(_roi_result(segments or _one_window()))
    infer = _StubWorker(handler)
    infer.info = {"checkpoint_sha256": CKPT, "decoder": "hybrid"}
    return svc.OlkavsService(roi, infer), roi, infer


def test_service_store_mode_saves_template_and_removes_temp(store, tmp_path) -> None:
    """저장 모드: 인식 요청에 hidden_out(작업 폴더 안 .npz 절대 경로)을 넣고, 읽은 즉시 지운 뒤 템플릿으로 저장한다."""
    hidden = _hidden(11, 33)
    seen: list = []
    service, _roi, infer = _ward_service(_infer_handler(hidden, top="W05", seen=seen))
    work = tmp_path / "work"
    results = service.transcribe_video("v.webm", str(work), utterance=True, phrase_set="ward",
                                       enroll_profile="P01", enroll_command="W05")
    target = infer.calls[0]["hidden_out"]
    assert os.path.isabs(target) and target.endswith(".npz") and os.path.dirname(target) == os.path.abspath(work)
    assert infer.calls[0]["decoder"] == "ctc" and len(infer.calls[0]["phrases"]) == 20
    assert not os.path.exists(target) and os.listdir(work) == []  # 임시 특징 파일이 남지 않는다
    assert results[0]["enroll"] == {"mode": "store", "profile": "P01", "command": "W05", "stored": True,
                                    "reason": None, "text_top": "W05", "text_top_matches": True, "count": 1,
                                    "complete": False}
    assert results[0]["phrase"]["rule"] == config.ward_rule() and results[0]["phrase"]["set"] == "ward"
    assert wp.load_hidden_npz(store.root / "P01" / "W05.npz").tobytes() == hidden.tobytes()
    meta = store.detail("P01", CKPT)["templates"]["W05"]
    assert meta["text_top"] == "W05" and meta["text_top_matches"] is True
    profile = json.loads((store.root / "P01" / "profile.json").read_text(encoding="utf-8"))
    assert (profile["templates"]["W05"]["start_ms"], profile["templates"]["W05"]["end_ms"]) == (300, 2800)
    assert profile["checkpoint_sha256"] == CKPT
    ResultItem(**results[0])  # 응답 스키마로 검증된다


def test_service_match_mode_uses_profile(store, tmp_path) -> None:
    """판정 모드: 등록 20개와 거리 판정 결과를 results[0].enroll에 붙인다. 글자 결과·accepted는 그대로다."""
    arrays = _complete(store)
    probe = arrays["W05"] + 0.01 * _hidden(4, arrays["W05"].shape[0])
    service, _roi, infer = _ward_service(_infer_handler(probe, top="W05"))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                    enroll_profile="P01")[0]
    enroll = item["enroll"]
    assert enroll["mode"] == "match" and enroll["status"] == "ok" and enroll["top_id"] == "W05"
    assert (enroll["within"], enroll["agree"], enroll["accepted"]) == (True, True, True)
    assert enroll["max_distance"] == config.OLKAVS_WARD_ENROLL_MAX_DISTANCE and enroll["count"] == 20
    assert item["phrase"]["accepted"] is False  # 서버의 글자 판정은 등록과 관계없다
    assert not os.path.exists(infer.calls[0]["hidden_out"])
    ResultItem(**item)


def test_service_enroll_reasons_and_no_hidden_out(store, tmp_path) -> None:
    """여러 창·무발화·검출 실패는 hidden_out을 보내지 않고 사유를 남긴다. 워커가 저장하지 못하면 no_features."""
    two = [{"index": i, "start_ms": i * 1000, "end_ms": (i + 1) * 1000, "status": "ok", "npz": f"{i}.npz"}
           for i in range(2)]
    cases = [(two, "multi_window"), ([{"index": 0, "start_ms": 0, "end_ms": 900, "status": "no_speech"}], "no_speech"),
             ([{"index": 0, "start_ms": 0, "end_ms": 900, "status": "failed"}], "partial_detection")]
    for segments, reason in cases:
        service, _roi, infer = _ward_service(_infer_handler(_hidden(1)), segments)
        results = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                           enroll_profile="P01", enroll_command="W01")
        assert all("hidden_out" not in c for c in infer.calls)
        assert results[0]["enroll"]["reason"] == reason and results[0]["enroll"]["stored"] is False
        assert all("enroll" not in r for r in results[1:])  # 첫 결과에만 붙는다
    service, _roi, infer = _ward_service(_infer_handler(_hidden(1), saved=False))
    out = service.transcribe_video("v.webm", str(tmp_path / "w2"), utterance=True, phrase_set="ward",
                                   enroll_profile="P01", enroll_command="W01")[0]["enroll"]
    assert (out["stored"], out["reason"]) == (False, "no_features")
    assert not (store.root / "P01").exists()


def test_service_removes_temp_npz_when_infer_fails(store, tmp_path) -> None:
    """인식 요청이 실패해도 은닉 특징 임시 파일을 지운다(Job은 원래대로 실패한다)."""
    def handler(payload, _progress):
        np.savez(payload["hidden_out"], hidden=_hidden(1))
        raise svc.WorkerError(svc.MSG_FAILED, "boom")

    service, _roi, infer = _ward_service(handler)
    with pytest.raises(svc.WorkerError):
        service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                 enroll_profile="P01", enroll_command="W01")
    assert not os.path.exists(infer.calls[0]["hidden_out"])
    assert service.gate.waiting() == 0


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04garbage-not-a-zip", b"\x93NUMPY not really"])
@pytest.mark.parametrize("command", ["W05", None])
def test_service_corrupt_hidden_npz_only_degrades_enroll(store, tmp_path, content, command) -> None:
    """워커가 hidden_saved=True라고 했는데 임시 npz가 비었거나 깨졌어도 녹화 결과는 나오고 등록만 no_features(저장)·
    error(판정)가 된다. 임시 파일은 손잡이가 닫혀 바로 지워진다."""
    def handler(payload, _progress):
        with open(payload["hidden_out"], "wb") as handle:
            handle.write(content)
        return {"ok": True, "text": "x", "confidence": 0.9, "raw_score": -3.0, "decoder": payload.get("decoder"),
                "phrase": {"id": "W05", "text": "x", "score": -3.0, "margin": 0.6, "runner_up_id": "W03"},
                "hidden_saved": True}

    if command is None:
        _complete(store)
    service, _roi, infer = _ward_service(handler)
    results = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                       enroll_profile="P01", enroll_command=command)
    assert results[0]["phrase"]["id"] == "W05" and results[0]["phrase"]["rule"] == config.ward_rule()
    enroll = results[0]["enroll"]
    if command is None:
        assert (enroll["mode"], enroll["status"], enroll["distance"], enroll["accepted"]) == ("match", "error", None, False)
    else:
        assert (enroll["stored"], enroll["reason"]) == (False, "no_features")
        assert not (store.root / "P01").exists()
    assert not os.path.exists(infer.calls[0]["hidden_out"])
    bad = tmp_path / "bad.npz"
    bad.write_bytes(content)
    with pytest.raises(ValueError):
        wp.load_hidden_npz(bad)
    os.remove(bad)  # 손잡이가 열려 있으면 Windows에서 WinError 32
    ResultItem(**results[0])


def test_service_without_enroll_is_unchanged(store, tmp_path) -> None:
    """등록 인자가 없으면 hidden_out·enroll이 없다. 문장 묶음은 rule도 없고 인식 요청이 바이트 단위로 같다."""
    for phrase_set in ("ward", "sentences"):
        service, _roi, infer = _ward_service(_infer_handler(_hidden(1)))
        item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set=phrase_set)[0]
        assert "hidden_out" not in infer.calls[0] and "enroll" not in item
        assert ("rule" in item["phrase"]) is (phrase_set == "ward")
    service, _roi, infer = _ward_service(_infer_handler(_hidden(1)))
    service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True)
    expected = {"cmd": "infer", "npz": "a.npz", "phrases": svc.load_phrases(config.OLKAVS_PHRASE_FILE)}
    assert json.dumps(infer.calls[0], ensure_ascii=True) == json.dumps(expected, ensure_ascii=True)


@pytest.mark.parametrize("kwargs", [{"phrase_set": "sentences", "enroll_profile": "P01"},
                                    {"enroll_profile": "P01"},
                                    {"phrase_set": "sentences", "enroll_command": "W01"},
                                    {"phrase_set": "ward", "enroll_command": "W01"},
                                    {"phrase_set": "ward", "enroll_profile": "../P01"},
                                    {"phrase_set": "ward", "enroll_profile": "CON"}])
def test_service_rejects_bad_enroll_args_before_workers(tmp_path, kwargs) -> None:
    """ward가 아닌 묶음의 등록 인자·이름 없는 명령·규칙 밖 이름은 워커를 쓰기 전에 ValueError."""
    service, roi, infer = _ward_service(_infer_handler(_hidden(1)))
    with pytest.raises(ValueError):
        service.transcribe_video("v.webm", str(tmp_path), utterance=True, **kwargs)
    assert roi.calls == [] and infer.calls == [] and service.gate.waiting() == 0


def test_service_aborted_job_does_not_store(store, tmp_path) -> None:
    """인식이 끝난 뒤 중단 신호(시간 초과)가 오면 등록을 저장하지 않는다."""
    state = {"calls": 0}

    def abort() -> bool:
        state["calls"] += 1
        return state["calls"] > 3  # 대기·입술 추출 전·창 확인 뒤, 등록 직전에 중단

    service, _roi, _infer = _ward_service(_infer_handler(_hidden(1)))
    with pytest.raises(svc.JobAborted):
        service.transcribe_video("v.webm", str(tmp_path / "w"), should_abort=abort, utterance=True,
                                 phrase_set="ward", enroll_profile="P01", enroll_command="W01")
    assert not (store.root / "P01").exists()


# ---------------------------------------------------------------------------
# API: POST /api/record 등록 필드
# ---------------------------------------------------------------------------

def _post(env, data: dict):
    """2초 WebM을 폼 필드와 함께 보낸다 (값이 목록이면 같은 필드를 여러 번 보낸다)."""
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        content = handle.read()
    return client.post("/api/record", files={"file": ("recording.webm", content, "video/webm")}, data=data)


@pytest.mark.parametrize("data,message", [
    ({"enroll_profile": "P01"}, "MSG_ENROLL_WARD_ONLY"),
    ({"phrase_set": "sentences", "enroll_profile": "P01"}, "MSG_ENROLL_WARD_ONLY"),
    ({"phrase_set": "sentences", "enroll_profile": "P01", "enroll_command": "W01"}, "MSG_ENROLL_WARD_ONLY"),
    ({"enroll_command": "W01"}, "MSG_ENROLL_WARD_ONLY"),
    ({"phrase_set": "ward", "enroll_command": "W01"}, "MSG_ENROLL_NEEDS_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": ""}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "../P01"}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "a/b"}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "x" * 33}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "NUL"}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": ["P01", "P01"]}, "MSG_BAD_ENROLL_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "W21"}, "MSG_BAD_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "w01"}, "MSG_BAD_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "../W01"}, "MSG_BAD_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": ""}, "MSG_BAD_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": ["W01", "W02"]}, "MSG_BAD_ENROLL_COMMAND"),
])
def test_record_rejects_bad_enroll_fields(env, data, message) -> None:
    """규칙 위반은 파일을 저장하기 전에 400 + 고정 한국어 문구. 서비스를 부르지 않고 파일을 남기지 않는다."""
    resp = _post(env, data)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == getattr(record_module, message)
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


@pytest.mark.parametrize("data,expected", [
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "W05"},
     {"enroll_profile": "P01", "enroll_command": "W05"}),
    ({"phrase_set": "ward", "enroll_profile": "P01"}, {"enroll_profile": "P01", "enroll_command": None}),
    ({"phrase_set": "ward"}, {}),
    ({"phrase_set": "sentences"}, {}),
    ({}, {}),
])
def test_record_enroll_fields_reach_service(env, data, expected) -> None:
    """등록 필드는 ward에서만 서비스로 간다. 등록이 없으면 서비스 인자는 이 기능 이전과 같다. 응답 형식은 그대로다."""
    resp = _post(env, data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"job_id", "session_id", "status", "duration_sec"} and body["status"] == "processing"
    try:
        call = env["service"].calls[-1]
        assert {k: call[k] for k in ("enroll_profile", "enroll_command") if k in call} == expected
        assert set(call) - {"enroll_profile", "enroll_command"} == {"video", "work_dir", "remaining_seconds",
                                                                    "video_seconds", "warnings", "utterance",
                                                                    "phrase_set"}
    finally:
        jm.clear_job(body["session_id"])


def _post_with_origin(env, data: dict, origin: str | None):
    """_post와 같되 Origin 헤더를 붙인다 (브라우저의 여러 부분 폼 POST는 사전 요청 없이 Origin만 보낸다)."""
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        content = handle.read()
    headers = {"Origin": origin} if origin is not None else {}
    return client.post("/api/record", files={"file": ("recording.webm", content, "video/webm")}, data=data,
                       headers=headers)


@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://localhost:5174"])
def test_record_enroll_from_other_origin_is_403(env, origin) -> None:
    """화이트리스트 밖 Origin의 등록 요청(저장·판정 모두)은 파일을 저장하기 전에 403이고 서비스를 부르지 않는다."""
    for data in ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "W02"},
                 {"phrase_set": "ward", "enroll_profile": "P01"}):
        resp = _post_with_origin(env, data, origin)
        assert resp.status_code == 403, resp.text
        assert resp.json()["detail"] == record_module.MSG_ENROLL_ORIGIN
        assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


@pytest.mark.parametrize("origin", [None, "http://localhost:5173", "http://127.0.0.1:5173"])
def test_record_enroll_from_app_origin_or_tools_is_accepted(env, origin) -> None:
    """화면 주소(화이트리스트)나 Origin을 보내지 않는 도구(채점 도구·점검 스크립트)의 등록 요청은 받는다."""
    resp = _post_with_origin(env, {"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "W05"}, origin)
    assert resp.status_code == 200, resp.text
    try:
        assert env["service"].calls[-1]["enroll_command"] == "W05"
    finally:
        jm.clear_job(resp.json()["session_id"])


@pytest.mark.parametrize("data", [{}, {"phrase_set": "sentences"}, {"phrase_set": "ward"}])
def test_record_without_enroll_ignores_origin_as_before(env, data) -> None:
    """등록 필드가 없는 요청은 Origin과 관계없이 이 기능 이전과 같이 받는다 (문장 모드 동작 그대로)."""
    resp = _post_with_origin(env, data, "http://evil.example")
    assert resp.status_code == 200, resp.text
    jm.clear_job(resp.json()["session_id"])


_SENTENCE_ITEM_KEYS = {"start_ms", "end_ms", "text", "confidence", "raw_score", "candidate", "phrase"}
_SENTENCE_PHRASE_KEYS = {"id", "text", "score", "margin", "runner_up_id", "accepted", "set"}


def test_result_json_keys_sentences_unchanged_ward_extended(env) -> None:
    """GET /api/result: 문장 묶음 결과의 키는 이 기능 이전과 같다(rule·enroll 없음). ward에는 rule·enroll이 붙는다."""
    phrase = {"id": "A01", "text": "x", "score": -2.0, "margin": 0.8, "runner_up_id": "N01", "accepted": True,
              "set": "sentences"}
    env["service"].results = [{"start_ms": 0, "end_ms": 2000, "text": "가", "confidence": 0.9, "raw_score": -1.0,
                               "phrase": phrase}, {"start_ms": 2000, "end_ms": 3000, "text": "[발화 없음]",
                                                   "confidence": None, "raw_score": None}]
    resp = _post(env, {})
    session = resp.json()["session_id"]
    try:
        raw = client.get(f"/api/result/{session}")
        body = raw.json()
        assert body["status"] == "done"
        assert [set(r) for r in body["results"]] == [_SENTENCE_ITEM_KEYS, _SENTENCE_ITEM_KEYS]
        assert set(body["results"][0]["phrase"]) == _SENTENCE_PHRASE_KEYS and body["results"][1]["phrase"] is None
        assert '"rule"' not in raw.text and '"enroll"' not in raw.text
    finally:
        jm.clear_job(session)
    rule = config.ward_rule()
    enroll = {"mode": "match", "profile": "P01", "status": "incomplete", "count": 3, "top_id": None, "distance": None,
              "runner_up_id": None, "runner_up_distance": None, "distances": None, "max_distance": 0.27,
              "within": False, "agree": False, "accepted": False}
    env["service"].results = [{"start_ms": 0, "end_ms": 2000, "text": "가", "confidence": 0.9, "raw_score": -1.0,
                               "phrase": {**phrase, "id": "W05", "set": "ward", "rule": rule}, "enroll": enroll}]
    session = _post(env, {"phrase_set": "ward", "enroll_profile": "P01"}).json()["session_id"]
    try:
        item = client.get(f"/api/result/{session}").json()["results"][0]
        assert set(item) == _SENTENCE_ITEM_KEYS | {"enroll"} and item["enroll"] == enroll
        assert set(item["phrase"]) == _SENTENCE_PHRASE_KEYS | {"rule"} and item["phrase"]["rule"] == rule
    finally:
        jm.clear_job(session)


def test_result_item_schema_rejects_bad_enroll() -> None:
    """enroll은 mode로 저장·판정 형식을 가린다. 알 수 없는 mode·사유·상태는 거부한다."""
    base = {"start_ms": 0, "end_ms": 1, "text": "가"}
    store_ok = {"mode": "store", "profile": "P01", "command": "W01", "stored": True, "reason": None, "text_top": "W01",
                "text_top_matches": True, "count": 1, "complete": False}
    assert ResultItem(**base, enroll=store_ok).enroll.mode == "store"
    for bad in ({**store_ok, "mode": "other"}, {**store_ok, "reason": "weird"}):
        with pytest.raises(ValueError):
            ResultItem(**base, enroll=bad)
    with pytest.raises(ValueError):
        ResultItem(**base, enroll={"mode": "match", "profile": "P01", "status": "great", "count": 0,
                                   "within": False, "agree": False, "accepted": False})
    assert "enroll" not in ResultItem(**base).model_dump() and "enroll" not in ResultItem(**base).model_dump_json()


# ---------------------------------------------------------------------------
# API: 등록 목록·상세·삭제, /health, CORS
# ---------------------------------------------------------------------------

def test_profiles_api_list_get_delete(store) -> None:
    """목록·상세(없는 등록은 missing)·삭제({deleted:true}, 다시 지우면 404)."""
    assert client.get("/api/ward/profiles").json() == {"profiles": []}
    assert client.get("/api/ward/profiles/P01").json() == {"profile": "P01", "count": 0, "complete": False,
                                                           "status": "missing", "templates": {}}
    store.store_template("P01", "W02", _hidden(1), {"text_top": "W03", "text_top_matches": False}, COMMANDS, CKPT)
    listed = client.get("/api/ward/profiles").json()["profiles"]
    assert len(listed) == 1 and set(listed[0]) == {"profile", "count", "complete", "status", "updated"}
    assert (listed[0]["profile"], listed[0]["count"], listed[0]["status"]) == ("P01", 1, "incomplete")
    detail = client.get("/api/ward/profiles/P01").json()
    assert set(detail) == {"profile", "count", "complete", "status", "templates"}
    assert list(detail["templates"]) == ["W02"]
    assert set(detail["templates"]["W02"]) == {"created", "text_top", "text_top_matches"}
    assert detail["templates"]["W02"]["text_top_matches"] is False
    assert client.delete("/api/ward/profiles/P01").json() == {"deleted": True}
    assert not (store.root / "P01").exists()
    resp = client.delete("/api/ward/profiles/P01")
    assert resp.status_code == 404 and resp.json()["detail"] == wp.MSG_NOT_FOUND


@pytest.mark.parametrize("name", ["..", "%2e%2e", "a%2Fb", "x" * 33, "CON", "a.b", "%20P01"])
def test_profiles_api_rejects_bad_names(store, name) -> None:
    """규칙 밖 이름은 400(또는 경로가 맞지 않아 404/405)이며 저장 폴더 밖을 건드리지 않는다."""
    (store.root.parent / "keep").mkdir(exist_ok=True)
    for method in (client.get, client.delete):
        resp = method(f"/api/ward/profiles/{name}")
        assert resp.status_code in (400, 404, 405), (name, resp.status_code)
        if resp.status_code == 400:
            assert resp.json()["detail"] == wp.MSG_BAD_PROFILE
    assert (store.root.parent / "keep").is_dir()


def test_profiles_api_case_conflict_409(store) -> None:
    """대소문자만 다른 등록이 있으면 상세·삭제는 409이고 그 등록은 그대로다."""
    store.store_template("Abc", "W01", _hidden(1), {}, COMMANDS, CKPT)
    assert client.get("/api/ward/profiles/abc").status_code == 409
    assert client.delete("/api/ward/profiles/ABC").status_code == 409
    assert (store.root / "Abc" / "W01.npz").is_file()


def test_health_reports_ward_thresholds_and_enroll(store, monkeypatch) -> None:
    """/health model.ward_phrases에 thresholds(지금 기준값)·enroll{enabled, profile_dir_writable}을 더한다(경로 없음)."""
    service = svc.OlkavsService(_StubWorker(lambda p, _p: {}), _StubWorker(lambda p, _p: {}))
    monkeypatch.setattr(svc, "_service", service)
    monkeypatch.setattr(config, "OLKAVS_WARD_POSSIBLE_MIN_SCORE", -3.6)
    body = client.get("/health").json()
    ward = body["model"]["ward_phrases"]
    assert ward["thresholds"] == {"min_margin": config.OLKAVS_WARD_MIN_MARGIN, "min_score": config.OLKAVS_WARD_MIN_SCORE,
                                  "possible_min_score": -3.6,
                                  "enroll_max_distance": config.OLKAVS_WARD_ENROLL_MAX_DISTANCE}
    assert ward["enroll"] == {"enabled": True, "profile_dir_writable": True}
    assert {"enabled", "count", "version"} <= set(ward) and ward["count"] == 20
    assert str(store.root) not in json.dumps(body, ensure_ascii=False)


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
def test_cors_allows_delete_preflight_for_whitelist(origin) -> None:
    """화이트리스트 오리진은 DELETE 사전 요청을 허용하고, 다른 오리진은 허용하지 않는다(와일드카드 없음)."""
    resp = client.options("/api/ward/profiles/P01", headers={"Origin": origin,
                                                              "Access-Control-Request-Method": "DELETE"})
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == origin
    assert "DELETE" in resp.headers["access-control-allow-methods"]
    bad = client.options("/api/ward/profiles/P01", headers={"Origin": "http://evil.example",
                                                             "Access-Control-Request-Method": "DELETE"})
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers
