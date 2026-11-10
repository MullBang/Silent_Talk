"""병실 '내 목록'(사용자가 만든 명령 목록) 서버 검사 (GPU·실제 모델 불필요).

- 검사기: 정규화, 오류(개수·글자·길이·토큰·화면 문장·종류·긴급·취소·중복·이름), 주의(길이·긴급 수·입 모양 같음·반대말·
  비슷함·끝 같음), 토큰 수가 인식 워커 phrase_tokens와 같음(공식 모델 코드가 있을 때), 입 모양 규칙이 기본 목록 설계값과 같음.
- 저장소: ID 규칙·대소문자 충돌·원자적 쓰기·만든 시각 유지·SHA-256·20개 상한(동시 만들기 포함)·깨진 파일·삭제.
- API: 목록·조회·저장·검사·삭제, 400/403/404/409/413, /health custom_lists.
- 녹화: command_list는 ward 전용, 업로드 저장 전 검사, 서비스에 목록 전달, phrase.list는 내 목록일 때만, 기본 JSON 그대로.
- 등록: 목록마다 이름공간(기본 등록은 그대로), 문장으로 찾는 템플릿(문장 하나를 고치면 그 템플릿만 빠짐, 순서를 바꿔도 그대로),
  상태(missing·incomplete·ok·stale), 판정은 지금 목록 명령만, 목록을 지우면 등록도 지움.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import threading
import types
import unicodedata

import numpy as np
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_TESTS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_ROOT)
_SCRIPTS = os.path.join(_PROJECT_ROOT, "scripts")
for _path in (_TESTS_DIR, _BACKEND_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config  # noqa: E402
from api import infer as infer_module  # noqa: E402
from schemas.models import PhraseMatch, ResultItem  # noqa: E402
from services import ward_lists as wl  # noqa: E402
from services import ward_profiles as wp  # noqa: E402
from test_olkavs_service import _StubWorker, _roi_result, svc  # noqa: E402
from test_record_api import client, env, jm, make_video, record_module  # noqa: E402,F401 - env는 fixture

CKPT = "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"

# 이 프로젝트에서 시험용으로 쓴 문장 (AI Hub 문장 아님). 오류·주의가 하나도 없는 목록이다.
CLEAN_ROWS = [
    ("가슴이 답답하고 숨이 차요", "숨이 차요", "emergency", True, False),
    ("머리가 깨질 것처럼 아파요", "머리가 많이 아파요", "emergency", True, False),
    ("보호자를 불러 주실래요", "보호자를 불러 주세요", "call", False, False),
    ("식사 시간이 언제인가요", "식사는 언제예요?", "answer", False, False),
    ("휠체어를 타고 나가고 싶어요", "산책하고 싶어요", "room", False, False),
    ("방금 부탁은 없던 걸로 해요", "방금 요청은 취소할게요", "call", False, True),
]

with open(config.OLKAVS_WARD_PHRASE_FILE, encoding="utf-8") as _handle:
    DEFAULT_LIST = json.load(_handle)


def _row(text: str, display: str, category: str = "care", emergency: bool = False, cancel: bool = False) -> dict:
    return {"text": text, "display": display, "category": category, "emergency": emergency, "cancel": cancel}


def _body(rows=None, title: str = "재활 병동 시험") -> dict:
    """검사기·PUT 본문 {title, commands}."""
    rows = CLEAN_ROWS if rows is None else rows
    return {"title": title, "commands": [_row(*r) if isinstance(r, tuple) else r for r in rows]}


def _codes(items: list[dict]) -> list[str]:
    return [item["code"] for item in items]


def _hidden(seed: int, frames: int = 20) -> np.ndarray:
    """합성 은닉 특징 (frames, 256) float32."""
    return np.random.RandomState(seed).randn(frames, wp.HIDDEN_DIM).astype(np.float32)


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    """임시 내 목록 폴더·등록 폴더 (config 값도 바꾼다)."""
    lists = tmp_path / "ward_lists"
    profiles = tmp_path / "ward_profiles"
    monkeypatch.setattr(config, "WARD_LIST_DIR", str(lists))
    monkeypatch.setattr(config, "WARD_PROFILE_DIR", str(profiles))
    return types.SimpleNamespace(lists=lists, profiles=profiles, store=wl.ListStore(lists), tmp=tmp_path)


def test_list_dir_config_default_and_env(tmp_path) -> None:
    """LIPNET_WARD_LIST_DIR가 없으면 backend/data/ward_lists(gitignore), 있으면 그 폴더 (새 프로세스에서 config import)."""
    import subprocess

    code = "import config, json; print(json.dumps(config.WARD_LIST_DIR))"
    base = {k: v for k, v in os.environ.items() if k != "LIPNET_WARD_LIST_DIR"}
    for extra, expected in (({}, os.path.join(config.BACKEND_ROOT, "data", "ward_lists")),
                            ({"LIPNET_WARD_LIST_DIR": str(tmp_path / "x")}, str(tmp_path / "x"))):
        out = subprocess.run([sys.executable, "-c", code], cwd=_BACKEND_ROOT, env={**base, **extra},
                             capture_output=True, text=True, encoding="utf-8", timeout=60)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout) == expected
    with open(os.path.join(_PROJECT_ROOT, ".gitignore"), encoding="utf-8") as handle:
        assert "backend/data/ward_lists/" in handle.read().splitlines()


# ---------------------------------------------------------------------------
# 토큰 수·입 모양 단위
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,tokens,syllables", [("가", 2, 1), ("각", 3, 1), ("닭", 3, 1), ("값", 3, 1),
                                                    ("가 각", 6, 2), ("물 좀 주세요", 14, 5)])
def test_token_and_syllable_counts(text, tokens, syllables) -> None:
    """토큰 = 받침 없는 음절 2 + 받침 있는 음절 3(겹받침도 3) + 띄어쓰기 1. 음절 수는 띄어쓰기를 빼고 센다."""
    assert wl.token_count(text) == tokens and wl.syllable_count(text) == syllables


def _upstream_vocab() -> dict[str, int]:
    """공식 모델의 72토큰 자모 사전(kor_grapheme.csv). 없으면 건너뛴다."""
    path = os.path.join(_PROJECT_ROOT, "experiments", "olkavs_official", "upstream", "vocabulary", "kor_grapheme.csv")
    if not os.path.isfile(path):
        pytest.skip("공식 모델 코드(experiments/olkavs_official/upstream)가 없어 토큰 대조를 건너뜀")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        vocab = {row["char"]: int(row["id"]) for row in csv.DictReader(handle)}
    assert len(vocab) == 72
    return vocab


def test_token_count_equals_worker_phrase_tokens() -> None:
    """서버 토큰 수가 인식 워커의 phrase_tokens(char2grp + 72토큰 사전) 길이와 같다(기본 목록·시험 목록·무작위 문장)."""
    vocab = _upstream_vocab()
    try:
        import olkavs_transcribe as ot

        bundle = {"vocab": types.SimpleNamespace(vocab_dict=vocab)}
        ot.phrase_tokens("가", bundle)
    except ImportError as exc:
        pytest.skip(f"phrase_tokens를 불러오지 못함: {exc}")
    rng = random.Random(5)
    texts = [c["text"] for c in DEFAULT_LIST["fixed"]] + [r[0] for r in CLEAN_ROWS] + ["닭 값 삶 앉아 읽어 없어"]
    for _ in range(200):
        words = ["".join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(rng.randint(1, 5)))
                 for _ in range(rng.randint(1, 4))]
        texts.append(" ".join(words))
    for text in texts:
        assert wl.token_count(text) == len(ot.phrase_tokens(text, bundle)), text


def test_default_list_tokens_match_design_note() -> None:
    """기본 목록 20문장의 토큰 수는 commands.json 설계 기록대로 23~34다."""
    counts = [wl.token_count(c["text"]) for c in DEFAULT_LIST["fixed"]]
    assert (min(counts), max(counts)) == (23, 34)


@pytest.mark.parametrize("text,units", [("같이", "KaJi"), ("좋아요", "Juau"), ("읽어", "iDKe"), ("꽃이", "KuJi"),
                                        ("옷을", "uDiD"), ("합니다", "KaBDiDa"), ("국물", "KuKBuD"),
                                        ("밤바다", "BaBaDa"), ("물 좀", "BuDJuB"), ("화장실", "KuaJaKDiD")])
def test_lip_units_rules(text, units) -> None:
    """연음·겹받침 나눔·구개음화(같이→가치)·받침 ㅎ 없음·첫소리 ㅇ 없음·같은 단위 합침(밤바다)·이중 모음(화)."""
    assert wl.lip_units(text) == units


def test_lip_rules_reproduce_default_list_design_values() -> None:
    """기본 목록의 입 모양 최소 거리 0.500·끝 공유 최대 4 (commands.json note에 적힌 설계값)와 같다."""
    units = [wl.lip_units(c["text"]) for c in DEFAULT_LIST["fixed"]]
    pairs = [(a, b) for i, a in enumerate(units) for b in units[i + 1:]]
    assert min(wl.lip_distance(a, b) for a, b in pairs) == 0.5
    assert max(wl.shared_ending(a, b) for a, b in pairs) == 4
    assert "입 모양 최소 거리 0.500" in DEFAULT_LIST["note"] and "끝말 공유 최대 4" in DEFAULT_LIST["note"]


@pytest.mark.parametrize("a,b,expected", [("불 켜 줘", "불 꺼 줘", True), ("불 좀 켜 줄래", "불 좀 꺼 줄래", True),
                                          ("등받이 올려 줘요", "등받이 내려 줘요", True),
                                          ("창문 열어 주세요", "창문 닫아 주세요", True),
                                          ("네 좋아요", "아니요 좋아요", True), ("불 켜 줘", "불 켜 줘", False),
                                          ("불 켜 줘", "물 좀 줘", False), ("창문 열어 주세요", "창문 닫고 가세요", False)])
def test_opposite_pairs(a, b, expected) -> None:
    """반대말 하나만 바꾸면 같은 문장이 되는가."""
    assert wl.is_opposite_pair(a, b) is expected and wl.is_opposite_pair(b, a) is expected


def test_categories_equal_default_list() -> None:
    """종류는 기본 목록(commands.json categories)의 다섯 개와 같다."""
    assert list(wl.CATEGORIES) == [c["id"] for c in DEFAULT_LIST["categories"]]


# ---------------------------------------------------------------------------
# 검사기
# ---------------------------------------------------------------------------

def test_clean_list_has_no_errors_or_warnings() -> None:
    """시험 목록: 오류·주의 없음, 정규화 결과에 U01부터 ID, 줄별 음절·토큰 수."""
    out = wl.check_list(_body())
    assert out["errors"] == [] and out["warnings"] == []
    commands = out["normalized"]["commands"]
    assert [c["id"] for c in commands] == [f"U{i:02d}" for i in range(1, 7)]
    assert set(commands[0]) == {"id", "text", "display", "category", "emergency", "cancel"}
    assert out["normalized"]["title"] == "재활 병동 시험"
    assert out["stats"][0] == {"syllables": 11, "tokens": 29}


def test_default_list_passes_validator_with_only_ending_warnings() -> None:
    """기본 목록을 내 목록으로 옮겨도 오류가 없다(취소 W20). 주의는 끝 입 모양 4단위가 같은 네 쌍뿐이다."""
    rows = [_row(c["text"], c["display"], c["category"], c["emergency"], c["action"] == "cancel_last")
            for c in DEFAULT_LIST["fixed"]]
    out = wl.check_list(_body(rows))
    assert out["errors"] == []
    assert _codes(out["warnings"]) == ["lip_ending"] * 4


def test_normalization_and_client_ids_ignored() -> None:
    """입 모양 문장: NFC·앞뒤 공백·공백 묶음 정리. 화면 문장: 앞뒤 공백·띄어쓰기 묶음. 보낸 id는 무시한다."""
    rows = [dict(_row(*r), id="W99") for r in CLEAN_ROWS]
    rows[2]["text"] = unicodedata.normalize("NFD", "  보호자를\t불러   주실래요 \n")
    rows[2]["display"] = "  보호자를   불러 주세요 "
    out = wl.check_list(_body(rows, title="  재활   병동 "))
    assert out["errors"] == []
    command = out["normalized"]["commands"][2]
    assert (command["id"], command["text"], command["display"]) == ("U03", "보호자를 불러 주실래요", "보호자를 불러 주세요")
    assert out["normalized"]["title"] == "재활 병동"


def _with(index: int, **changes) -> dict:
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[index].update(changes)
    return _body(rows)


@pytest.mark.parametrize("changes,code,field", [
    ({"text": ""}, "text_empty", "text"),
    ({"text": None}, "text_empty", "text"),
    ({"text": "물 2컵 주세요 please"}, "text_chars", "text"),
    ({"text": "ㅋㅋ 너무 아파요"}, "text_chars", "text"),
    ({"text": "물 좀 주세요!"}, "text_chars", "text"),
    ({"text": "물"}, "text_length", "text"),
    ({"text": "가" * 41}, "text_length", "text"),
    ({"text": "각" * 17}, "text_tokens", "text"),
    ({"text": "가" * 40}, "text_tokens", "text"),
    ({"display": ""}, "display_empty", "display"),
    ({"display": "   "}, "display_empty", "display"),
    ({"display": 3}, "display_empty", "display"),
    ({"display": "물\n주세요"}, "display_chars", "display"),
    ({"display": "물​주세요"}, "display_chars", "display"),
    ({"display": "가" * 31}, "display_length", "display"),
    ({"category": "urgent"}, "category_invalid", "category"),
    ({"category": None}, "category_invalid", "category"),
    ({"emergency": "true"}, "emergency_type", "emergency"),
    ({"emergency": 1}, "emergency_type", "emergency"),
    ({"emergency": True}, "emergency_mismatch", "category"),
    ({"cancel": "yes"}, "cancel_type", "cancel"),
])
def test_row_errors(changes, code, field) -> None:
    """줄 하나의 오류: 그 줄 번호(0부터)·필드·한국어 문구, level 'error'."""
    out = wl.check_list(_with(2, **changes))
    hits = [e for e in out["errors"] if e["code"] == code]
    assert hits, out["errors"]
    assert (hits[0]["row"], hits[0]["rows"], hits[0]["field"], hits[0]["level"]) == (2, [2], field, "error")
    assert any("가" <= ch <= "힣" for ch in hits[0]["message"])  # 한국어 문구


def test_text_error_lists_bad_characters_and_boundaries() -> None:
    """쓸 수 없는 글자를 알려 준다. 2자·토큰 50개(16음절 받침+1음절)는 받는다."""
    out = wl.check_list(_with(2, text="물 2컵 주세요 ok"))
    assert "2 o k" in out["errors"][0]["message"]
    for text in ("물을", "각" * 16 + "가"):
        assert wl.check_list(_with(2, text=text))["errors"] == [], text
    assert wl.token_count("각" * 16 + "가") == 50


def test_emergency_category_mismatch_both_ways() -> None:
    """긴급 표시 ⇔ 종류 '긴급'. 종류가 긴급인데 긴급이 아니면 emergency 칸에 오류."""
    out = wl.check_list(_with(0, emergency=False))
    assert [(e["code"], e["row"], e["field"]) for e in out["errors"]] == [("emergency_mismatch", 0, "emergency")]


def test_duplicates_after_normalization() -> None:
    """정규화 뒤 같은 입 모양 문장·같은 화면 문장은 뒤에 나온 줄의 오류다(rows에 두 줄)."""
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[4]["text"] = unicodedata.normalize("NFD", " 보호자를  불러 주실래요")
    rows[5]["display"] = "숨이 차요"
    out = wl.check_list(_body(rows))
    text_dup = [e for e in out["errors"] if e["code"] == "text_duplicate"]
    display_dup = [e for e in out["errors"] if e["code"] == "display_duplicate"]
    assert [(e["row"], e["rows"]) for e in text_dup] == [(4, [2, 4])]
    assert [(e["row"], e["rows"]) for e in display_dup] == [(5, [0, 5])]
    assert "3번" in text_dup[0]["message"] and "1번" in display_dup[0]["message"]
    assert not any(w["code"].startswith("lip_") and 4 in w["rows"] for w in out["warnings"])  # 중복 줄은 비교하지 않는다


def test_texts_differing_only_in_spacing_are_duplicates() -> None:
    """띄어쓰기만 다른 입 모양 문장은 오류다(띄어쓰기는 입 모양이 없어 1·2등을 가를 수 없고 등록 템플릿도 같은 입 모양이 된다)."""
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[4]["text"] = "보호자를불러 주실래요"
    out = wl.check_list(_body(rows))
    hits = [e for e in out["errors"] if e["code"] == "text_duplicate_spacing"]
    assert [(e["row"], e["rows"], e["field"], e["level"]) for e in hits] == [(4, [2, 4], "text", "error")]
    assert "3번 문장과 띄어쓰기만 다르고 같은 문장이에요" in hits[0]["message"]
    assert not any(w["code"] == "lip_same" for w in out["warnings"])  # 오류로 올렸으니 주의는 두 번 내지 않는다
    rows[4]["text"] = "보호자를 불러 주실래요"
    assert _codes(wl.check_list(_body(rows))["errors"]) == ["text_duplicate"]  # 완전히 같으면 예전 오류 그대로


def test_display_spaces_of_other_kinds_count_as_spaces() -> None:
    """화면 문장·목록 이름의 NBSP·전각 공백 같은 공백 문자는 띄어쓰기로 바꿔 비교한다(같아 보이는 카드 두 장을 막는다)."""
    assert wl.normalize_display("물\u00a0주세요") == "물 주세요"
    assert wl.normalize_display("\u3000물\u2003\u2003주세요 ") == "물 주세요"
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[4]["display"] = "숨이\u00a0차요"
    out = wl.check_list(_body(rows))
    assert [(e["code"], e["row"], e["rows"]) for e in out["errors"]] == [("display_duplicate", 4, [0, 4])]
    assert out["normalized"]["commands"][4]["display"] == "숨이 차요"
    assert wl.check_list(_body(title="재활\u00a0병동"))["normalized"]["title"] == "재활 병동"
    assert _codes(wl.check_list(_with(2, display="물\n주세요"))["errors"]) == ["display_chars"]  # 줄바꿈은 그대로 오류


def test_invisible_bad_characters_are_named_by_code() -> None:
    """입 모양 문장의 보이지 않는 글자(BOM 등)도 무엇을 지울지 알 수 있게 'U+FEFF'처럼 알린다."""
    out = wl.check_list(_with(2, text="\ufeff보호자를 불러 주실래요"))
    assert [(e["code"], e["row"]) for e in out["errors"]] == [("text_chars", 2)]
    assert "(쓸 수 없는 글자: U+FEFF)" in out["errors"][0]["message"]
    out = wl.check_list(_with(2, text="물\u200b 2컵"))
    assert "U+200B 2" in out["errors"][0]["message"]


def test_cancel_rules() -> None:
    """취소 문장은 하나까지(둘째부터 오류), 긴급일 수 없다. 취소가 없어도 된다."""
    two = _with(3, cancel=True)
    assert [(e["code"], e["row"], e["rows"]) for e in wl.check_list(two)["errors"]] == [("cancel_multiple", 5, [3, 5])]
    both = _with(0, cancel=True)
    both["commands"][5]["cancel"] = False
    assert [(e["code"], e["row"]) for e in wl.check_list(both)["errors"]] == [("cancel_emergency", 0)]
    assert wl.check_list(_with(5, cancel=False))["errors"] == []


def test_list_level_errors() -> None:
    """개수 4~20, 긴급이 아닌 문장 하나 이상, 목록 이름 1~30자·제어 문자 없음, 형식 오류."""
    assert "too_few" in _codes(wl.check_list(_body(CLEAN_ROWS[:3]))["errors"])
    many = [_row(f"{'가' * (i % 5 + 2)} {'나' * (i // 5 + 2)} 다라마", f"화면 {i}") for i in range(21)]
    assert "too_many" in _codes(wl.check_list(_body(many))["errors"])
    all_emergency = [(t, d, "emergency", True, False) for t, d, *_ in CLEAN_ROWS[:5]]
    out = wl.check_list(_body(all_emergency))
    assert "no_normal" in _codes(out["errors"]) and "emergency_many" in _codes(out["warnings"])
    for title, code in (("", "title_empty"), (None, "title_empty"), ("가" * 31, "title_length"),
                        ("재활\t병동", "title_chars")):
        errors = wl.check_list(_body(title=title))["errors"]
        assert _codes(errors) == [code] and errors[0]["field"] == "title" and errors[0]["row"] is None
    assert _codes(wl.check_list([])["errors"]) == ["body_invalid"]
    assert _codes(wl.check_list({"title": "가", "commands": "x"})["errors"]) == ["commands_missing", "too_few"]
    bad_row = _body()
    bad_row["commands"][1] = 7
    assert [(e["code"], e["row"]) for e in wl.check_list(bad_row)["errors"]] == [("row_invalid", 1)]


def test_warnings_length_and_emergency_counts() -> None:
    """음절 9~13·토큰 23~34 밖은 길이 주의. 긴급 0개·5개 이상은 목록 주의(문구는 기획서 그대로)."""
    out = wl.check_list(_with(3, text="베개를 하나 더 주세요"))  # 9음절이지만 토큰 22
    assert [(w["code"], w["row"], w["level"]) for w in out["warnings"]] == [("length", 3, "warning")]
    assert out["warnings"][0]["message"].startswith("짧거나 길어서 잘 읽히지 않을 수 있어요")
    long = wl.check_list(_with(3, text="가" * 14))
    assert "length" in _codes(long["warnings"])
    none = [(t, d, "care", False, c) for t, d, _cat, _e, c in CLEAN_ROWS]
    warning = [w for w in wl.check_list(_body(none))["warnings"] if w["code"] == "emergency_none"]
    assert len(warning) == 1 and "스위치 3번 긴급 호출만 남아요" in warning[0]["message"] and warning[0]["row"] is None
    many = [(t, d, "emergency", True, False) for t, d, *_ in CLEAN_ROWS[:5]] + [CLEAN_ROWS[5]]
    warning = [w for w in wl.check_list(_body(many))["warnings"] if w["code"] == "emergency_many"]
    assert len(warning) == 1 and "긴급이 많으면 거짓 긴급 알림이 늘어요" in warning[0]["message"]


@pytest.mark.parametrize("a,b,code,level", [
    ("물을 한 컵 마시고 싶어요", "불을 한 컵 마시고 싶어요", "lip_same", "strong"),
    ("창문을 조금 열어 주세요", "창문을 조금 닫아 주세요", "lip_opposite", "strong"),
    ("약을 먹을 시간인가요", "약을 먹을 시간이에요", "lip_close", "warning"),
    ("간호사 선생님 좀 불러 주세요", "우리 가족에게 전화해 볼래요", "lip_ending", "warning"),
])
def test_lip_pair_warnings(a, b, code, level) -> None:
    """두 문장의 입 모양 주의는 가장 강한 하나만, 두 줄 번호와 함께 (rows, 문구의 'N번·M번')."""
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[3]["text"], rows[4]["text"] = a, b
    out = wl.check_list(_body(rows))
    assert out["errors"] == []
    lip = [w for w in out["warnings"] if w["code"].startswith("lip_")]
    assert [(w["code"], w["level"], w["row"], w["rows"]) for w in lip] == [(code, level, 3, [3, 4])]
    assert lip[0]["message"].startswith("4번·5번 문장은")
    expected = {"lip_same": "입 모양이 거의 같아서 구분하기 어려워요", "lip_opposite": "구분하기 어려워요",
                "lip_close": "헷갈릴 수 있어요", "lip_ending": "끝부분이 같아요"}[code]
    assert expected in lip[0]["message"]


def test_examples_file_has_no_errors() -> None:
    """화면의 예시 목록(frontend/src/ward/custom_list_examples.json)은 검사기 오류가 없다(파일이 있을 때)."""
    path = os.path.join(_PROJECT_ROOT, "frontend", "src", "ward", "custom_list_examples.json")
    if not os.path.isfile(path):
        pytest.skip("예시 목록 파일이 아직 없음")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["schema"] == "ward-custom-list-examples-v1" and data["examples"]
    for example in data["examples"]:
        out = wl.check_list(example)
        assert out["errors"] == [], (example.get("title"), out["errors"])


def test_commands_sha256_is_canonical_json() -> None:
    """sha256 = commands 배열의 정규 JSON(키 정렬, ensure_ascii False, 구분자 ',' ':')의 SHA-256."""
    commands = wl.check_list(_body())["normalized"]["commands"]
    canonical = json.dumps(commands, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert wl.commands_sha256(commands) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    reordered = [dict(reversed(list(c.items()))) for c in commands]
    assert wl.commands_sha256(reordered) == wl.commands_sha256(commands)  # 키 순서와 관계없다


# ---------------------------------------------------------------------------
# 저장소
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["L1", "a", "my-list_2", "x" * 32])
def test_list_id_accepts(value) -> None:
    """영문·숫자·_·- 1~32자."""
    assert wl.is_valid_list_id(value) and wl.validate_list_id(value) == value


@pytest.mark.parametrize("value", ["", "x" * 33, "../L1", "a/b", "a\\b", "C:", "L1 ", "한글", "a.b", "CON", "nul",
                                   "COM1", "lpt9", None, 3])
def test_list_id_rejects(value) -> None:
    """경로 문자·빈 값·길이 초과·Windows 장치 이름·문자열 아님은 거부(보낸 값을 문구에 넣지 않는다)."""
    assert not wl.is_valid_list_id(value)
    with pytest.raises(wl.InvalidListId) as caught:
        wl.validate_list_id(value)
    assert caught.value.user_message == wl.MSG_BAD_ID


def test_save_writes_schema_and_preserves_created(roots) -> None:
    """저장: 계약 형식·U01부터 ID·SHA-256, 임시 파일 없음. 다시 저장하면 만든 시각은 그대로, 제목만 바꾸면 SHA 그대로."""
    doc, warnings = roots.store.save("L1", _body())
    assert warnings == []
    on_disk = json.loads((roots.lists / "L1.json").read_text(encoding="utf-8"))
    assert on_disk == doc and list(on_disk) == ["schema", "id", "title", "created", "updated", "commands", "sha256"]
    assert (on_disk["schema"], on_disk["id"]) == ("ward-custom-list-v1", "L1")
    assert [c["id"] for c in on_disk["commands"]] == [f"U{i:02d}" for i in range(1, 7)]
    assert on_disk["sha256"] == wl.commands_sha256(on_disk["commands"])
    assert os.listdir(roots.lists) == ["L1.json"]
    retitled, _ = roots.store.save("L1", _body(title="다른 이름"))
    assert (retitled["created"], retitled["sha256"], retitled["title"]) == (doc["created"], doc["sha256"], "다른 이름")
    changed, _ = roots.store.save("L1", _with(3, text="밥은 언제 나오나요"))
    assert changed["created"] == doc["created"] and changed["sha256"] != doc["sha256"]
    assert roots.store.get("L1") == changed


def test_save_with_errors_writes_nothing(roots) -> None:
    """검사 오류가 있으면 저장하지 않고 ListInvalid(errors, warnings)."""
    with pytest.raises(wl.ListInvalid) as caught:
        roots.store.save("L1", _with(2, text="hello"))
    assert _codes(caught.value.errors) == ["text_chars"] and isinstance(caught.value.warnings, list)
    assert not roots.lists.exists() or os.listdir(roots.lists) == []


def test_atomic_write_keeps_old_file_on_failure(roots, monkeypatch) -> None:
    """바꾸기(os.replace)가 실패하면 이전 파일이 그대로이고 임시 파일이 남지 않는다."""
    doc, _ = roots.store.save("L1", _body())
    before = (roots.lists / "L1.json").read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    with monkeypatch.context() as patch, pytest.raises(wl.ListError) as caught:
        patch.setattr(os, "replace", boom)
        roots.store.save("L1", _with(3, text="밥은 언제 나오나요"))
    assert caught.value.user_message == wl.MSG_STORAGE
    assert (roots.lists / "L1.json").read_bytes() == before and os.listdir(roots.lists) == ["L1.json"]
    assert roots.store.get("L1") == doc


def test_case_conflict_and_not_found(roots) -> None:
    """대소문자만 다른 목록은 충돌(그 목록은 그대로). 없는 목록은 ListNotFound."""
    roots.store.save("Ward", _body())
    for call in (lambda: roots.store.save("ward", _body()), lambda: roots.store.get("WARD"),
                 lambda: roots.store.delete("wArD")):
        with pytest.raises(wl.ListConflict):
            call()
    assert os.listdir(roots.lists) == ["Ward.json"]
    with pytest.raises(wl.ListNotFound):
        roots.store.get("Other")
    with pytest.raises(wl.ListNotFound):
        roots.store.delete("Other")


def test_list_limit_and_concurrent_creates(roots) -> None:
    """20개까지. 상한에서도 있는 목록은 고칠 수 있다. 동시에 만들어도 상한을 넘지 않는다."""
    for i in range(wl.MAX_LISTS - 1):
        roots.store.save(f"L{i:02d}", _body())
    results: list = []

    def create(name: str) -> None:
        try:
            roots.store.save(name, _body())
            results.append("ok")
        except wl.ListLimit:
            results.append("limit")

    threads = [threading.Thread(target=create, args=(f"N{i}",)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["limit"] * 5 + ["ok"] and roots.store.count() == wl.MAX_LISTS
    with pytest.raises(wl.ListLimit) as caught:
        roots.store.save("Extra", _body())
    assert caught.value.user_message == wl.MSG_LIMIT
    roots.store.save("L00", _body(title="고친 이름"))  # 있는 목록은 고칠 수 있다


def test_corrupt_or_tampered_files(roots) -> None:
    """깨진 JSON·바뀐 SHA·틀린 ID는 읽기 오류(고정 문구). 요약 목록에서는 건너뛰고, 지울 수는 있다."""
    roots.store.save("Good", _body())
    roots.store.save("Tampered", _body())
    (roots.lists / "Broken.json").write_text("{broken", encoding="utf-8")
    data = json.loads((roots.lists / "Tampered.json").read_text(encoding="utf-8"))
    data["commands"][0]["text"] = "몰래 바꾼 문장이에요"
    (roots.lists / "Tampered.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (roots.lists / "notes.txt").write_text("x", encoding="utf-8")
    for name in ("Broken", "Tampered"):
        with pytest.raises(wl.ListError) as caught:
            roots.store.get(name)
        assert caught.value.user_message == wl.MSG_UNREADABLE
    assert [s["id"] for s in roots.store.summaries()] == ["Good"]
    roots.store.delete("Broken")
    assert not (roots.lists / "Broken.json").exists()


def test_load_valid_rejects_list_failing_current_validator(roots) -> None:
    """파일 형식·SHA는 맞아도 지금 검사기로 오류가 있는 목록은 녹화에 쓰지 않는다(ListInvalid)."""
    _write_raw_list(roots.lists, "Old", [dict(_row(*r)) for r in CLEAN_ROWS[:3]])  # 3개뿐
    assert roots.store.get("Old")["id"] == "Old"
    with pytest.raises(wl.ListInvalid):
        roots.store.load_valid("Old")
    assert _codes(roots.store.detail("Old")["checks"]["errors"]) == ["too_few"]


def _write_raw_list(folder, list_id: str, rows: list[dict]) -> dict:
    """검사기를 거치지 않고 형식·SHA만 맞춘 목록 파일을 쓴다(예전 규칙으로 저장된 목록 흉내)."""
    commands = [{"id": f"U{i + 1:02d}", **row} for i, row in enumerate(rows)]
    doc = {"schema": wl.SCHEMA, "id": list_id, "title": "예전 목록", "created": "2026-11-11T00:00:00+00:00",
           "updated": "2026-11-11T00:00:00+00:00", "commands": commands, "sha256": wl.commands_sha256(commands)}
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{list_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False)
    return doc


# ---------------------------------------------------------------------------
# API: /api/ward/lists
# ---------------------------------------------------------------------------

def test_api_lists_crud(roots) -> None:
    """목록(빈 목록) → 저장(목록+checks+warnings) → 요약 → 조회(checks) → 삭제({deleted:true}) → 다시 지우면 404."""
    assert client.get("/api/ward/lists").json() == {"lists": []}
    resp = client.put("/api/ward/lists/L1", json=_body())
    assert resp.status_code == 200, resp.text
    saved = resp.json()
    assert set(saved) == {"schema", "id", "title", "created", "updated", "commands", "sha256", "checks", "warnings"}
    assert saved["checks"] == {"errors": [], "warnings": []} and saved["warnings"] == []
    assert saved["schema"] == "ward-custom-list-v1" and saved["commands"][5]["cancel"] is True
    listed = client.get("/api/ward/lists").json()["lists"]
    assert listed == [{"id": "L1", "title": "재활 병동 시험", "count": 6, "emergency_count": 2,
                       "updated": saved["updated"], "sha256": saved["sha256"]}]
    detail = client.get("/api/ward/lists/L1").json()
    assert set(detail) == {"schema", "id", "title", "created", "updated", "commands", "sha256", "checks"}
    assert detail["commands"] == saved["commands"] and detail["checks"] == {"errors": [], "warnings": []}
    assert client.delete("/api/ward/lists/L1").json() == {"deleted": True}
    resp = client.delete("/api/ward/lists/L1")
    assert resp.status_code == 404 and resp.json()["detail"] == wl.MSG_NOT_FOUND
    assert client.get("/api/ward/lists/L1").status_code == 404


def test_api_put_returns_warnings_and_400_with_errors(roots) -> None:
    """주의가 있으면 저장하고 warnings로 알린다. 오류가 있으면 400 {detail, errors, warnings}이고 저장하지 않는다."""
    resp = client.put("/api/ward/lists/L1", json=_with(3, text="베개를 하나 더 주세요"))
    assert resp.status_code == 200 and _codes(resp.json()["warnings"]) == ["length"]
    assert resp.json()["checks"]["warnings"] == resp.json()["warnings"]
    item = resp.json()["warnings"][0]
    assert set(item) == {"code", "level", "message", "row", "rows", "field"}
    bad = client.put("/api/ward/lists/L2", json=_with(1, text="help me", emergency="x"))
    assert bad.status_code == 400
    body = bad.json()
    assert set(body) == {"detail", "errors", "warnings"} and body["detail"] == wl.MSG_INVALID
    assert {(e["code"], e["row"]) for e in body["errors"]} == {("text_chars", 1), ("emergency_type", 1)}
    assert client.get("/api/ward/lists/L2").status_code == 404


@pytest.mark.parametrize("name", ["..", "%2e%2e", "a%2Fb", "x" * 33, "CON", "a.b", "%20L1"])
def test_api_rejects_bad_ids(roots, name) -> None:
    """규칙 밖 ID는 400(또는 경로가 맞지 않아 404/405)이고 폴더 밖을 건드리지 않는다."""
    (roots.tmp / "keep.json").write_text("{}", encoding="utf-8")
    for method in ("get", "put", "delete"):
        kwargs = {"json": _body()} if method == "put" else {}
        resp = getattr(client, method)(f"/api/ward/lists/{name}", **kwargs)
        assert resp.status_code in (400, 404, 405), (name, method, resp.status_code)
        if resp.status_code == 400:
            assert resp.json()["detail"] == wl.MSG_BAD_ID
    assert (roots.tmp / "keep.json").is_file() and not roots.lists.exists()


def test_api_conflict_409_and_limit_400(roots) -> None:
    """대소문자만 다른 목록 409. 20개가 차면 새 목록은 400(고정 문구)."""
    assert client.put("/api/ward/lists/Ward", json=_body()).status_code == 200
    for method in ("get", "put", "delete"):
        kwargs = {"json": _body()} if method == "put" else {}
        assert getattr(client, method)("/api/ward/lists/ward", **kwargs).status_code == 409
    for i in range(wl.MAX_LISTS - 1):
        roots.store.save(f"L{i:02d}", _body())
    resp = client.put("/api/ward/lists/Extra", json=_body())
    assert resp.status_code == 400 and resp.json()["detail"] == wl.MSG_LIMIT


def test_api_put_create_only_and_if_match(roots) -> None:
    """If-None-Match: * 는 새 목록만 만들고(이미 있으면 409, 덮어쓰지 않음), If-Match: <sha256>은 그 판일 때만 고친다."""
    new = {"If-None-Match": "*"}
    first = client.put("/api/ward/lists/L1", json=_body(), headers=new)
    assert first.status_code == 200, first.text
    before = (roots.lists / "L1.json").read_bytes()
    again = client.put("/api/ward/lists/L1", json=_body(title="덮어쓰기"), headers=new)
    assert again.status_code == 409 and again.json()["detail"] == wl.MSG_EXISTS
    assert (roots.lists / "L1.json").read_bytes() == before
    sha = first.json()["sha256"]
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[3]["text"] = "식사 시간이 언제쯤인가요"
    edited = client.put("/api/ward/lists/L1", json=_body(rows), headers={"If-Match": f'"{sha}"'})
    assert edited.status_code == 200, edited.text
    new_sha = edited.json()["sha256"]
    assert new_sha != sha
    # 옛 판(sha)으로 다시 고치면 다른 곳에서 바뀐 것이니 409, 파일은 그대로
    after = (roots.lists / "L1.json").read_bytes()
    stale = client.put("/api/ward/lists/L1", json=_body(title="옛 화면"), headers={"If-Match": sha})
    assert stale.status_code == 409 and stale.json()["detail"] == wl.MSG_CHANGED_ELSEWHERE
    assert (roots.lists / "L1.json").read_bytes() == after
    # 지워진(없는) 목록을 고치려 하면 409 (조용히 다시 만들지 않는다)
    missing = client.put("/api/ward/lists/L9", json=_body(), headers={"If-Match": new_sha})
    assert missing.status_code == 409 and not (roots.lists / "L9.json").exists()
    # 제목만 바꾸기: 같은 판이면 받는다(대문자 sha도 받음)
    title_only = client.put("/api/ward/lists/L1", json=_body(rows, title="새 이름"),
                            headers={"If-Match": new_sha.upper()})
    assert title_only.status_code == 200 and title_only.json()["title"] == "새 이름"
    # 헤더가 없으면 예전처럼 조건 없이 저장한다
    assert client.put("/api/ward/lists/L1", json=_body()).status_code == 200


@pytest.mark.parametrize("headers", [
    {"If-None-Match": "abc"}, {"If-Match": "x" * 64}, {"If-Match": "*"}, {"If-Match": ""},
    {"If-None-Match": "*", "If-Match": "0" * 64},
])
def test_api_put_rejects_bad_precondition_headers(roots, headers) -> None:
    """형식이 틀린 조건 헤더는 400(고정 문구)이고 아무것도 쓰지 않는다."""
    resp = client.put("/api/ward/lists/L1", json=_body(), headers=headers)
    assert resp.status_code == 400 and "저장 조건" in resp.json()["detail"]
    assert not roots.lists.exists()


def test_store_save_expect_rules(roots) -> None:
    """ListStore.save(expect=...)는 목록 잠금 안에서 조건을 본다(화면 탭 둘이 같은 ID로 새 목록을 저장해도 하나만 남는다)."""
    doc, _ = roots.store.save("L1", _body(), expect=wl.EXPECT_NEW)
    with pytest.raises(wl.ListPrecondition):
        roots.store.save("L1", _body(title="둘째 탭"), expect=wl.EXPECT_NEW)
    with pytest.raises(wl.ListPrecondition):
        roots.store.save("L1", _body(title="옛 판"), expect="0" * 64)
    assert roots.store.get("L1")["title"] == "재활 병동 시험"
    saved, _ = roots.store.save("L1", _body(title="같은 판"), expect=doc["sha256"])
    assert saved["title"] == "같은 판"
    results = []

    def make() -> None:
        try:
            roots.store.save("Race", _body(), expect=wl.EXPECT_NEW)
            results.append("ok")
        except wl.ListPrecondition:
            results.append("exists")

    threads = [threading.Thread(target=make) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["exists"] * 5 + ["ok"]


@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://localhost:5174"])
def test_api_state_changes_from_other_origin_are_403(roots, origin) -> None:
    """화이트리스트 밖 Origin의 PUT·DELETE는 403이고 아무것도 바꾸지 않는다. 검사(check)·조회는 받는다."""
    roots.store.save("L1", _body())
    before = (roots.lists / "L1.json").read_bytes()
    headers = {"Origin": origin}
    for resp in (client.put("/api/ward/lists/L1", json=_body(title="바꿈"), headers=headers),
                 client.put("/api/ward/lists/L2", json=_body(), headers=headers),
                 client.delete("/api/ward/lists/L1", headers=headers)):
        assert resp.status_code == 403 and "허용하지 않은" in resp.json()["detail"]
    assert (roots.lists / "L1.json").read_bytes() == before and os.listdir(roots.lists) == ["L1.json"]
    assert client.post("/api/ward/lists/check", json=_body(), headers=headers).status_code == 200


@pytest.mark.parametrize("origin", [None, "http://localhost:5173", "http://127.0.0.1:5173"])
def test_api_state_changes_from_app_origin_or_tools(roots, origin) -> None:
    """화면 주소(화이트리스트)나 Origin을 보내지 않는 도구의 PUT·DELETE는 받는다."""
    headers = {"Origin": origin} if origin else {}
    assert client.put("/api/ward/lists/L1", json=_body(), headers=headers).status_code == 200
    assert client.delete("/api/ward/lists/L1", headers=headers).status_code == 200


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173"])
def test_cors_preflight_allows_put_and_delete_for_whitelist(origin) -> None:
    """화이트리스트 오리진은 PUT·DELETE 사전 요청을 허용하고, 다른 오리진은 허용하지 않는다."""
    for method in ("PUT", "DELETE"):
        resp = client.options("/api/ward/lists/L1", headers={"Origin": origin, "Access-Control-Request-Method": method})
        assert resp.status_code == 200 and resp.headers["access-control-allow-origin"] == origin
        bad = client.options("/api/ward/lists/L1", headers={"Origin": "http://evil.example",
                                                            "Access-Control-Request-Method": method})
        assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers


def test_api_body_limits_and_bad_json(roots) -> None:
    """64KB를 넘는 본문은 413(Content-Length가 있든 없든), JSON이 아니면 400. 아무것도 저장하지 않는다."""
    big = json.dumps({"title": "x" * (wl.MAX_BODY_BYTES + 10), "commands": []}).encode("utf-8")
    for path, method in (("/api/ward/lists/L1", "put"), ("/api/ward/lists/check", "post")):
        resp = getattr(client, method)(path, content=big, headers={"Content-Type": "application/json"})
        assert resp.status_code == 413 and resp.json()["detail"].startswith("요청이 너무 커요")

        def chunks():
            for i in range(0, len(big), 4096):
                yield big[i:i + 4096]

        resp = getattr(client, method)(path, content=chunks(), headers={"Content-Type": "application/json"})
        assert resp.status_code == 413
        for raw in (b"{not json", b"\xff\xfe", b"[" * 5000 + b"]" * 5000):
            resp = getattr(client, method)(path, content=raw, headers={"Content-Type": "application/json"})
            assert resp.status_code == 400 and resp.json()["detail"] == "요청 내용을 읽지 못했어요. JSON 형식으로 보내 주세요."
    assert not roots.lists.exists()
    resp = client.put("/api/ward/lists/L1", json=[1, 2])
    assert resp.status_code == 400 and _codes(resp.json()["errors"]) == ["body_invalid"]


def test_api_check_does_not_save(roots) -> None:
    """검사는 저장하지 않고 {errors, warnings, normalized, stats}를 준다(오류가 있어도 200)."""
    resp = client.post("/api/ward/lists/check", json=_with(2, text="물 2컵"))
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"errors", "warnings", "normalized", "stats"}
    assert _codes(body["errors"]) == ["text_chars"] and body["errors"][0]["row"] == 2
    assert body["normalized"]["commands"][2]["id"] == "U03" and len(body["stats"]) == 6
    assert body["stats"][2] == {"syllables": None, "tokens": None}
    assert body["stats"][0] == {"syllables": 11, "tokens": 29}
    clean = client.post("/api/ward/lists/check", json=_body()).json()
    assert clean["errors"] == [] and clean["warnings"] == []
    assert not roots.lists.exists()


def test_api_check_echoes_lone_surrogates_safely(roots) -> None:
    """짝 없는 대리 문자(JSON의 \\ud800)가 든 본문도 500이 아니라 오류로 알리고, 되돌려 보내는 값은 U+FFFD로 바꾼다."""
    raw = ('{"title":"\\ud800","commands":[' + ",".join(
        ['{"text":"\\udc00","display":"\\ud800 %d","category":"care","emergency":false,"cancel":false}' % i
         for i in range(4)]) + "]}").encode("ascii")
    resp = client.post("/api/ward/lists/check", content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {"title_chars", "text_chars", "display_chars"} <= set(_codes(body["errors"]))
    replacement = "\N{REPLACEMENT CHARACTER}"
    assert body["normalized"]["title"] == replacement and body["normalized"]["commands"][0]["text"] == replacement
    put = client.put("/api/ward/lists/L1", content=raw, headers={"Content-Type": "application/json"})
    assert put.status_code == 400 and not roots.lists.exists()


def test_delete_keeps_list_when_enrollments_cannot_be_removed(roots, saved, monkeypatch) -> None:
    """등록(개인 특징)을 지우지 못하면 목록도 남겨 다시 지울 수 있게 한다(500, 고정 문구)."""
    _store_all(roots, saved)

    def boom(*_a, **_k):
        raise OSError("locked")

    calls = []

    def counting_boom(*a, **k):
        calls.append(1)
        boom(*a, **k)

    with monkeypatch.context() as patch:
        patch.setattr(wp.shutil, "rmtree", counting_boom)
        patch.setattr(wl, "DELETE_RETRY_SECONDS", 0.0)
        resp = client.delete("/api/ward/lists/L1")
    assert resp.status_code == 500 and resp.json()["detail"] == wl.MSG_DELETE_FAILED
    assert "지우지 못했어요" in wl.MSG_DELETE_FAILED and "저장" not in wl.MSG_DELETE_FAILED
    assert len(calls) == wl.DELETE_RETRIES  # 잠깐 열린 파일 때문일 수 있어 몇 번 다시 해 본다
    assert (roots.lists / "L1.json").is_file() and _namespace(roots).is_dir()
    assert client.delete("/api/ward/lists/L1").json() == {"deleted": True}
    assert not _namespace(roots).exists() and not (roots.lists / "L1.json").exists()


def test_delete_retries_when_enrollment_folder_is_briefly_locked(roots, saved, monkeypatch) -> None:
    """등록 폴더 지우기가 한 번 실패해도(다른 요청이 파일을 잠깐 염) 다시 해 보아 지운다."""
    _store_all(roots, saved)
    real = wp.shutil.rmtree
    calls = []

    def flaky(path, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("in use")
        return real(path, *a, **k)

    monkeypatch.setattr(wp.shutil, "rmtree", flaky)
    monkeypatch.setattr(wl, "DELETE_RETRY_SECONDS", 0.0)
    assert client.delete("/api/ward/lists/L1").json() == {"deleted": True}
    assert len(calls) == 2 and not _namespace(roots).exists() and not (roots.lists / "L1.json").exists()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_adds_custom_lists_only(roots, monkeypatch) -> None:
    """/health model.ward_phrases에 custom_lists {enabled, dir_writable, count}만 더한다(다른 키·값 그대로, 경로 없음)."""
    service = svc.OlkavsService(_StubWorker(lambda p, _p: {}), _StubWorker(lambda p, _p: {}))
    monkeypatch.setattr(svc, "_service", service)
    body = client.get("/health").json()
    ward = body["model"]["ward_phrases"]
    assert set(ward) == {"enabled", "count", "version", "thresholds", "enroll", "custom_lists"}
    assert (ward["enabled"], ward["count"], ward["version"]) == (True, 20, "ward-commands-v1")
    assert ward["thresholds"] == config.ward_rule()
    assert ward["enroll"] == {"enabled": True, "profile_dir_writable": True}
    assert ward["custom_lists"] == {"enabled": True, "dir_writable": True, "count": 0}
    roots.store.save("L1", _body())
    assert client.get("/health").json()["model"]["ward_phrases"]["custom_lists"]["count"] == 1
    assert str(roots.lists) not in json.dumps(body, ensure_ascii=False)
    assert set(body) == {"status", "model_backend", "model"}


# ---------------------------------------------------------------------------
# POST /api/record command_list
# ---------------------------------------------------------------------------

def _post_record(env, data: dict, headers: dict | None = None):
    """2초 WebM을 폼 필드와 함께 보낸다 (값이 목록이면 같은 필드를 여러 번 보낸다)."""
    video = make_video(str(env["tmp"] / "clip.webm"), 2.0)
    with open(video, "rb") as handle:
        content = handle.read()
    return client.post("/api/record", files={"file": ("recording.webm", content, "video/webm")}, data=data,
                       headers=headers or {})


@pytest.fixture()
def saved(roots):
    """저장된 내 목록 L1 (시험 목록 6문장)."""
    doc, _ = roots.store.save("L1", _body())
    return doc


@pytest.mark.parametrize("data,message", [
    ({"command_list": "L1"}, "MSG_LIST_WARD_ONLY"),
    ({"phrase_set": "sentences", "command_list": "L1"}, "MSG_LIST_WARD_ONLY"),
    ({"phrase_set": "ward", "command_list": ""}, "MSG_BAD_COMMAND_LIST"),
    ({"phrase_set": "ward", "command_list": ["L1", "L1"]}, "MSG_BAD_COMMAND_LIST"),
    ({"phrase_set": "ward", "command_list": "../L1"}, "MSG_BAD_COMMAND_LIST"),
    ({"phrase_set": "ward", "command_list": "CON"}, "MSG_BAD_COMMAND_LIST"),
    ({"phrase_set": "ward", "command_list": "x" * 33}, "MSG_BAD_COMMAND_LIST"),
    ({"phrase_set": "ward", "command_list": "Missing"}, "MSG_LIST_NOT_FOUND"),
    ({"phrase_set": "ward", "command_list": "l1"}, "MSG_LIST_NOT_FOUND"),
    ({"phrase_set": "ward", "command_list": "Old"}, "MSG_LIST_INVALID"),
    ({"phrase_set": "ward", "command_list": "Broken"}, "MSG_LIST_INVALID"),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01", "enroll_command": "W01"},
     "MSG_BAD_LIST_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01", "enroll_command": "U07"},
     "MSG_BAD_LIST_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01", "enroll_command": ""},
     "MSG_BAD_LIST_ENROLL_COMMAND"),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_command": "U01"}, "MSG_ENROLL_NEEDS_PROFILE"),
    ({"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "U01"}, "MSG_BAD_ENROLL_COMMAND"),
])
def test_record_rejects_bad_command_list_before_saving(env, roots, saved, data, message) -> None:
    """규칙 위반·없는 목록·오류가 있는 목록은 업로드를 저장하기 전에 400 + 고정 문구. 서비스를 부르지 않는다."""
    _write_raw_list(roots.lists, "Old", [dict(_row(*r)) for r in CLEAN_ROWS[:3]])
    (roots.lists / "Broken.json").write_text("{broken", encoding="utf-8")
    resp = _post_record(env, data)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == getattr(record_module, message)
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


@pytest.mark.parametrize("data,enroll", [
    ({"phrase_set": "ward", "command_list": "L1"}, {}),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01"},
     {"enroll_profile": "P01", "enroll_command": None}),
    ({"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01", "enroll_command": "U03"},
     {"enroll_profile": "P01", "enroll_command": "U03"}),
])
def test_record_passes_checked_list_to_service(env, saved, data, enroll) -> None:
    """검사한 저장 목록을 그대로 서비스에 넘긴다(command_list). 응답 형식은 그대로다."""
    resp = _post_record(env, data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    try:
        assert set(body) == {"job_id", "session_id", "status", "duration_sec"}
        call = env["service"].calls[-1]
        assert call["command_list"] == saved and call["phrase_set"] == "ward"
        assert {k: call[k] for k in ("enroll_profile", "enroll_command") if k in call} == enroll
    finally:
        jm.clear_job(body["session_id"])


@pytest.mark.parametrize("data", [{}, {"phrase_set": "sentences"}, {"phrase_set": "ward"},
                                  {"phrase_set": "ward", "enroll_profile": "P01", "enroll_command": "W05"}])
def test_record_without_command_list_calls_service_as_before(env, saved, data) -> None:
    """command_list가 없으면 서비스 인자에 그 키가 없다(이 기능 이전과 같다)."""
    resp = _post_record(env, data)
    assert resp.status_code == 200, resp.text
    try:
        call = env["service"].calls[-1]
        assert "command_list" not in call
        assert set(call) - {"enroll_profile", "enroll_command"} == {"video", "work_dir", "remaining_seconds",
                                                                    "video_seconds", "warnings", "utterance",
                                                                    "phrase_set"}
    finally:
        jm.clear_job(resp.json()["session_id"])


def test_record_list_origin_rules(env, saved) -> None:
    """내 목록만 쓰는 녹화는 Origin을 보지 않는다(상태를 바꾸지 않음). 등록 필드가 있으면 기존 규칙대로 403."""
    evil = {"Origin": "http://evil.example"}
    ok = _post_record(env, {"phrase_set": "ward", "command_list": "L1"}, evil)
    assert ok.status_code == 200
    jm.clear_job(ok.json()["session_id"])
    denied = _post_record(env, {"phrase_set": "ward", "command_list": "L1", "enroll_profile": "P01"}, evil)
    assert denied.status_code == 403 and denied.json()["detail"] == record_module.MSG_ENROLL_ORIGIN


def test_record_list_sha_must_match_current_list(env, roots, saved) -> None:
    """command_list_sha256(화면이 불러온 판)가 지금 목록과 같으면 받고, 다르면 업로드를 저장하기 전에 400(MSG_LIST_CHANGED)이다."""
    ok = _post_record(env, {"phrase_set": "ward", "command_list": "L1", "command_list_sha256": saved["sha256"],
                            "enroll_profile": "P01", "enroll_command": "U03"})
    assert ok.status_code == 200, ok.text
    try:
        assert env["service"].calls[-1]["command_list"] == saved
    finally:
        jm.clear_job(ok.json()["session_id"])
    calls = len(env["service"].calls)
    rows = [_row(*r) for r in CLEAN_ROWS]
    rows[2], rows[3] = rows[3], rows[2]  # 다른 탭에서 순서를 바꿈 → U03의 문장이 바뀐다
    roots.store.save("L1", _body(rows))
    stale = _post_record(env, {"phrase_set": "ward", "command_list": "L1", "command_list_sha256": saved["sha256"],
                               "enroll_profile": "P01", "enroll_command": "U03"})
    assert stale.status_code == 400 and stale.json()["detail"] == record_module.MSG_LIST_CHANGED
    assert os.listdir(env["upload_dir"]) == [] and len(env["service"].calls) == calls


@pytest.mark.parametrize("data,message", [
    ({"phrase_set": "ward", "command_list_sha256": "0" * 64}, "MSG_LIST_SHA_NEEDS_LIST"),
    ({"phrase_set": "ward", "command_list": "L1", "command_list_sha256": ""}, "MSG_BAD_LIST_SHA"),
    ({"phrase_set": "ward", "command_list": "L1", "command_list_sha256": "A" * 64}, "MSG_BAD_LIST_SHA"),
    ({"phrase_set": "ward", "command_list": "L1", "command_list_sha256": "0" * 63}, "MSG_BAD_LIST_SHA"),
    ({"phrase_set": "ward", "command_list": "L1", "command_list_sha256": ["0" * 64, "0" * 64]}, "MSG_BAD_LIST_SHA"),
    ({"phrase_set": "ward", "command_list": "L1", "command_list_sha256": "0" * 64}, "MSG_LIST_CHANGED"),
])
def test_record_rejects_bad_list_sha_before_saving(env, saved, data, message) -> None:
    """목록 지문 규칙 위반은 업로드를 저장하기 전에 400 + 고정 문구. 서비스를 부르지 않는다."""
    resp = _post_record(env, data)
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == getattr(record_module, message)
    assert os.listdir(env["upload_dir"]) == [] and env["service"].calls == []


def test_list_changed_message_matches_frontend() -> None:
    """화면(client.js LIST_CHANGED_MESSAGE)이 같은 문구로 '목록이 바뀜'을 알아보고 목록을 다시 불러온다."""
    path = os.path.join(_PROJECT_ROOT, "frontend", "src", "api", "client.js")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert f"export const LIST_CHANGED_MESSAGE = '{record_module.MSG_LIST_CHANGED}';" in source


def test_result_json_list_key_only_with_command_list(env, saved) -> None:
    """GET /api/result: 기본 ward 결과의 phrase에는 list 키가 없고(이전과 같은 JSON), 내 목록 결과에만 list {id, sha256}."""
    rule = config.ward_rule()
    phrase = {"id": "W05", "text": "x", "score": -2.0, "margin": 0.8, "runner_up_id": "W03", "accepted": True,
              "set": "ward", "rule": rule}
    env["service"].results = [{"start_ms": 0, "end_ms": 2000, "text": "가", "confidence": 0.9, "raw_score": -1.0,
                               "phrase": phrase}]
    session = _post_record(env, {"phrase_set": "ward"}).json()["session_id"]
    try:
        raw = client.get(f"/api/result/{session}")
        assert raw.json()["status"] == "done" and '"list"' not in raw.text
        assert set(raw.json()["results"][0]["phrase"]) == {"id", "text", "score", "margin", "runner_up_id", "accepted",
                                                           "set", "rule"}
    finally:
        jm.clear_job(session)
    ref = {"id": "L1", "sha256": saved["sha256"]}
    env["service"].results = [{"start_ms": 0, "end_ms": 2000, "text": "가", "confidence": 0.9, "raw_score": -1.0,
                               "phrase": {**phrase, "id": "U03", "runner_up_id": "U01", "list": ref}}]
    session = _post_record(env, {"phrase_set": "ward", "command_list": "L1"}).json()["session_id"]
    try:
        item = client.get(f"/api/result/{session}").json()["results"][0]
        assert item["phrase"]["list"] == ref and item["phrase"]["set"] == "ward" and item["phrase"]["rule"] == rule
    finally:
        jm.clear_job(session)


def test_phrase_schema_list_key_absent_by_default() -> None:
    """PhraseMatch: list가 없으면 직렬화에 키가 없다(문장·기본 ward 결과의 JSON이 그대로다)."""
    base = {"id": "W01", "text": "x", "score": -1.0, "margin": 1.0, "runner_up_id": "W02", "accepted": True,
            "set": "ward"}
    assert "list" not in PhraseMatch(**base).model_dump() and '"list"' not in PhraseMatch(**base).model_dump_json()
    with_list = PhraseMatch(**base, list={"id": "L1", "sha256": "a" * 64})
    assert with_list.model_dump()["list"] == {"id": "L1", "sha256": "a" * 64}
    item = ResultItem(start_ms=0, end_ms=1, text="가", phrase=base)
    assert '"list"' not in item.model_dump_json()


def test_start_background_job_rejects_list_outside_ward_record(env, saved) -> None:
    """Job 등록 단계에서도 내 목록은 ward 녹화(record) Job에만 받는다."""
    from fastapi import BackgroundTasks

    for kwargs in ({"mode": "upload", "phrase_set": "ward"}, {"mode": "record", "phrase_set": "sentences"}):
        with pytest.raises(ValueError):
            infer_module.start_background_job("5e0f6a4c-1b2d-4c3e-8f9a-0b1c2d3e4f5a", "x.webm", BackgroundTasks(),
                                              command_list=saved, **kwargs)


# ---------------------------------------------------------------------------
# 서비스 (대체 워커)
# ---------------------------------------------------------------------------

def _one_window() -> list[dict]:
    return [{"index": 0, "start_ms": 300, "end_ms": 2800, "status": "ok", "npz": "a.npz"}]


def _infer_handler(hidden: np.ndarray | None, top: str = "U03"):
    """대체 인식 워커: hidden_out이 오면 은닉 특징을 저장하고, 받은 후보 1·2등으로 phrase를 만든다."""
    def handler(payload, _progress):
        phrases = payload.get("phrases") or []
        runner = next((p["id"] for p in phrases if p["id"] != top), None)
        out = {"ok": True, "text": "가", "confidence": 0.9, "raw_score": -3.0, "decoder": payload.get("decoder")}
        if phrases:
            out["phrase"] = {"id": top, "text": "x", "score": -2.0, "margin": 0.9, "runner_up_id": runner}
        if "hidden_out" in payload:
            if hidden is not None:
                np.savez(payload["hidden_out"], hidden=hidden)
            out["hidden_saved"] = hidden is not None
        return out
    return handler


def _service(handler, segments=None):
    roi = _StubWorker(_roi_result(segments or _one_window()))
    infer = _StubWorker(handler)
    infer.info = {"checkpoint_sha256": CKPT, "decoder": "hybrid"}
    return svc.OlkavsService(roi, infer), roi, infer


def test_service_uses_list_phrases_and_marks_phrase(roots, saved, tmp_path) -> None:
    """내 목록: 후보는 그 목록 [{id, text}], 디코더 ctc, 기준값은 ward 그대로, phrase에 list {id, sha256}."""
    service, _roi, infer = _service(_infer_handler(None))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                    command_list=saved)[0]
    call = infer.calls[0]
    assert call["phrases"] == [{"id": c["id"], "text": c["text"]} for c in saved["commands"]]
    assert call["decoder"] == "ctc" and "hidden_out" not in call
    phrase = item["phrase"]
    assert phrase["list"] == {"id": "L1", "sha256": saved["sha256"]} and phrase["set"] == "ward"
    assert phrase["rule"] == config.ward_rule()
    margin_ok = phrase["margin"] >= config.OLKAVS_WARD_MIN_MARGIN and phrase["score"] >= config.OLKAVS_WARD_MIN_SCORE
    assert phrase["accepted"] is margin_ok
    ResultItem(**item)


def test_service_without_list_has_no_list_key(tmp_path) -> None:
    """command_list가 없으면 기본 목록 후보 그대로이고 phrase에 list 키가 없다."""
    service, _roi, infer = _service(_infer_handler(None, top="W05"))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward")[0]
    assert [p["id"] for p in infer.calls[0]["phrases"]] == [f"W{i:02d}" for i in range(1, 21)]
    assert "list" not in item["phrase"] and set(item["phrase"]) == {"id", "text", "score", "margin", "runner_up_id",
                                                                    "accepted", "set", "rule"}


def test_service_rejects_list_outside_ward_or_malformed(saved, tmp_path) -> None:
    """ward가 아닌 묶음의 내 목록, 형식이 틀린 목록은 워커를 쓰기 전에 ValueError."""
    bad_lists = [dict(saved, id="../x"), dict(saved, sha256=None), dict(saved, commands=saved["commands"][:1]),
                 {"id": "L1"}]
    for phrase_set, command_list in [("sentences", saved)] + [("ward", b) for b in bad_lists]:
        service, roi, infer = _service(_infer_handler(None))
        with pytest.raises(ValueError):
            service.transcribe_video("v.webm", str(tmp_path), utterance=True, phrase_set=phrase_set,
                                     command_list=command_list)
        assert roi.calls == [] and infer.calls == [] and service.gate.waiting() == 0


# ---------------------------------------------------------------------------
# 내 목록 등록 (이름공간·문장 키 템플릿)
# ---------------------------------------------------------------------------

def _namespace(roots, list_id: str = "L1"):
    return roots.profiles / wp.LIST_NAMESPACE_DIR / list_id


def _store_all(roots, doc: dict, profile: str = "P01", checkpoint: str | None = CKPT) -> dict[str, np.ndarray]:
    """목록의 모든 명령을 등록한다. 문장 → 배열을 돌려준다."""
    commands = wp.list_commands(doc)
    store = wp.ListProfileStore(doc["id"], commands)
    arrays = {}
    for k, cid in enumerate(commands["ids"]):
        arrays[commands["texts"][cid]] = _hidden(200 + k, 12 + k)
        store.store_template(profile, cid, arrays[commands["texts"][cid]],
                             {"start_ms": 0, "end_ms": 1000, "text_top": cid, "text_top_matches": True},
                             commands, checkpoint)
    return arrays


def test_service_store_mode_uses_list_namespace_only(roots, saved, tmp_path) -> None:
    """등록 저장: <등록 폴더>/lists.d/<목록>/<이름>/에 문장 키 npz·profile.json. 기본 등록 폴더에는 아무것도 생기지 않는다."""
    hidden = _hidden(7, 31)
    service, _roi, infer = _service(_infer_handler(hidden, top="U03"))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                    enroll_profile="P01", enroll_command="U03", command_list=saved)[0]
    assert item["enroll"] == {"mode": "store", "profile": "P01", "command": "U03", "stored": True, "reason": None,
                              "text_top": "U03", "text_top_matches": True, "count": 1, "complete": False}
    assert not os.path.exists(infer.calls[0]["hidden_out"])
    folder = _namespace(roots) / "P01"
    key = wp.text_key(saved["commands"][2]["text"])
    assert sorted(os.listdir(folder)) == sorted([f"{key[:32]}.npz", "profile.json"])
    assert wp.load_hidden_npz(folder / f"{key[:32]}.npz").tobytes() == hidden.tobytes()
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    assert (profile["schema"], profile["list_id"], profile["profile"]) == ("ward-list-profile-v1", "L1", "P01")
    assert profile["templates"][key]["text"] == saved["commands"][2]["text"]
    assert profile["templates"][key]["command"] == "U03" and profile["checkpoint_sha256"] == CKPT
    assert os.listdir(roots.profiles) == [wp.LIST_NAMESPACE_DIR]  # 기본 등록 이름 폴더가 생기지 않는다
    assert wp.get_store().list_profiles(CKPT) == [] and wp.get_store().summary("P01", CKPT)["status"] == "missing"
    ResultItem(**item)


def test_default_and_list_profiles_never_touch_each_other(roots, saved) -> None:
    """같은 이름(P01)의 기본 등록과 내 목록 등록이 따로 있다. 한쪽을 지워도 다른 쪽은 그대로다."""
    default_commands = wp.current_commands()
    wp.get_store().store_template("P01", "W01", _hidden(1), {}, default_commands, CKPT)
    _store_all(roots, saved)
    default_before = sorted(os.listdir(roots.profiles / "P01"))
    listed = wp.get_store().list_profiles(CKPT)
    assert [p["profile"] for p in listed] == ["P01"] and listed[0]["count"] == 1  # lists.d는 등록으로 보이지 않는다
    ns = wp.ListProfileStore("L1", wp.list_commands(saved))
    assert ns.summary("P01", CKPT)["status"] == "ok"
    ns.delete("P01")
    assert sorted(os.listdir(roots.profiles / "P01")) == default_before
    _store_all(roots, saved)
    wp.get_store().delete("P01")
    assert ns.summary("P01", CKPT)["count"] == len(saved["commands"])
    assert not wp.is_valid_profile_id(wp.LIST_NAMESPACE_DIR)  # 기본 등록 이름이 될 수 없다


def test_text_keyed_templates_survive_reorder_and_edit_only_drops_one(roots, saved) -> None:
    """순서를 바꿔도 모두 그대로(ok), 문장 하나를 고치면 그 명령만 빠진다(incomplete). 다시 찍으면 ok."""
    arrays = _store_all(roots, saved)
    total = len(saved["commands"])
    reversed_doc, _ = roots.store.save("L1", _body(list(reversed(CLEAN_ROWS))))
    ns = wp.ListProfileStore("L1", wp.list_commands(reversed_doc))
    summary = ns.summary("P01", CKPT)
    assert (summary["status"], summary["count"], summary["complete"]) == ("ok", total, True)
    status, count, loaded = ns.templates_for_match("P01", None, CKPT)
    assert status == "ok" and count == total
    for command in reversed_doc["commands"]:
        assert loaded[command["id"]].tobytes() == arrays[command["text"]].tobytes()
    detail = ns.detail("P01", CKPT)
    assert list(detail["templates"]) == [c["id"] for c in reversed_doc["commands"]]
    assert detail["templates"]["U01"]["text_top"] == "U01"  # 찍을 때 1등이던 문장의 지금 ID
    edited, _ = roots.store.save("L1", _with(3, text="밥은 언제 나오나요"))
    ns = wp.ListProfileStore("L1", wp.list_commands(edited))
    summary = ns.summary("P01", CKPT)
    assert (summary["status"], summary["count"], summary["complete"]) == ("incomplete", total - 1, False)
    assert "U04" not in ns.detail("P01", CKPT)["templates"]
    assert ns.templates_for_match("P01", None, CKPT)[0] == "incomplete"
    commands = wp.list_commands(edited)
    assert ns.store_template("P01", "U04", _hidden(9), {}, commands, CKPT) == total
    assert ns.summary("P01", CKPT)["status"] == "ok"
    folder = _namespace(roots) / "P01"
    assert len([n for n in os.listdir(folder) if n.endswith(".npz")]) == total + 1  # 옛 문장 템플릿도 지우지 않는다


def test_list_profile_status_rules(roots, saved) -> None:
    """missing → incomplete → ok, 가중치가 바뀔 때만 stale(기동 전에는 비교하지 않음). stale에 저장하면 새로 시작한다."""
    commands = wp.list_commands(saved)
    ns = wp.ListProfileStore("L1", commands)
    assert ns.summary("P01", CKPT)["status"] == "missing"
    assert ns.detail("P01", CKPT) == {"profile": "P01", "count": 0, "complete": False, "status": "missing",
                                      "templates": {}}
    ns.store_template("P01", "U01", _hidden(1), {}, commands, CKPT)
    assert ns.summary("P01", CKPT)["status"] == "incomplete"
    _store_all(roots, saved)
    assert ns.summary("P01", CKPT)["status"] == "ok" and ns.summary("P01", None)["status"] == "ok"
    stale = ns.summary("P01", "new-checkpoint")
    assert (stale["status"], stale["count"]) == ("stale", len(commands["ids"]))
    assert ns.store_template("P01", "U02", _hidden(2), {}, commands, "new-checkpoint") == 1
    folder = _namespace(roots) / "P01"
    assert len([n for n in os.listdir(folder) if n.endswith(".npz")]) == 1
    with pytest.raises(ValueError):
        ns.store_template("P01", "W01", _hidden(1), {}, commands, CKPT)  # 목록에 없는 명령


def test_list_match_compares_current_list_commands_only(roots, saved, tmp_path) -> None:
    """판정: 지금 목록 명령의 템플릿과만 비교한다(distances 키 = 지금 ID). 서비스 판정 모드도 같은 이름공간을 쓴다."""
    arrays = _store_all(roots, saved)
    extra_doc, _ = roots.store.save("L1", _with(3, text="밥은 언제 나오나요"))
    ns_old = wp.ListProfileStore("L1", wp.list_commands(extra_doc))
    ns_old.store_template("P01", "U04", _hidden(77), {}, wp.list_commands(extra_doc), CKPT)
    back, _ = roots.store.save("L1", _body())  # 원래 문장으로 되돌림: 옛 템플릿이 다시 쓰인다
    probe = arrays[back["commands"][2]["text"]] + 0.01 * _hidden(5, arrays[back["commands"][2]["text"]].shape[0])
    ns = wp.ListProfileStore("L1", wp.list_commands(back))
    out = wp.match_result(ns, "P01", None, {"id": "U03"}, probe, CKPT, 0.26)
    assert out["status"] == "ok" and list(out["distances"]) == [c["id"] for c in back["commands"]]
    assert (out["top_id"], out["within"], out["agree"], out["accepted"]) == ("U03", True, True, True)
    service, _roi, _infer = _service(_infer_handler(probe, top="U03"))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                    enroll_profile="P01", command_list=back)[0]
    enroll = item["enroll"]
    assert (enroll["mode"], enroll["status"], enroll["top_id"], enroll["accepted"]) == ("match", "ok", "U03", True)
    assert set(enroll["distances"]) == {c["id"] for c in back["commands"]}
    missing = service.transcribe_video("v.webm", str(tmp_path / "w2"), utterance=True, phrase_set="ward",
                                       enroll_profile="P09", command_list=back)[0]["enroll"]
    assert (missing["status"], missing["count"]) == ("missing", 0)


def test_store_skipped_when_list_deleted_meanwhile(roots, saved, tmp_path) -> None:
    """녹화 요청 뒤 목록이 지워졌으면 등록을 저장하지 않는다(reason error, 이름공간을 만들지 않음). 녹화 결과는 그대로."""
    roots.store.delete("L1")
    service, _roi, _infer = _service(_infer_handler(_hidden(3), top="U03"))
    item = service.transcribe_video("v.webm", str(tmp_path / "w"), utterance=True, phrase_set="ward",
                                    enroll_profile="P01", enroll_command="U03", command_list=saved)[0]
    assert (item["enroll"]["stored"], item["enroll"]["reason"]) == (False, "error")
    assert item["phrase"]["list"]["id"] == "L1" and not _namespace(roots).exists()


def test_delete_list_removes_its_enrollments_only(roots, saved) -> None:
    """목록을 지우면 그 목록의 등록 이름공간도 지운다. 기본 등록·다른 목록의 등록은 그대로다."""
    wp.get_store().store_template("P01", "W01", _hidden(1), {}, wp.current_commands(), CKPT)
    other, _ = roots.store.save("L2", _body(title="다른 목록"))
    _store_all(roots, saved)
    _store_all(roots, other)
    assert client.delete("/api/ward/lists/L1").json() == {"deleted": True}
    assert not _namespace(roots, "L1").exists() and _namespace(roots, "L2").is_dir()
    assert (roots.profiles / "P01" / "W01.npz").is_file()
    assert wl.get_store().count() == 1


def test_profiles_api_with_list_query(roots, saved) -> None:
    """GET/DELETE /api/ward/profiles[/{p}]?list=<ID>는 그 목록의 등록. 쿼리가 없으면 기본 등록 그대로."""
    _store_all(roots, saved)
    wp.get_store().store_template("P02", "W01", _hidden(1), {}, wp.current_commands(), CKPT)
    assert [p["profile"] for p in client.get("/api/ward/profiles").json()["profiles"]] == ["P02"]
    listed = client.get("/api/ward/profiles?list=L1").json()["profiles"]
    assert [(p["profile"], p["count"], p["complete"], p["status"]) for p in listed] == [("P01", 6, True, "ok")]
    detail = client.get("/api/ward/profiles/P01?list=L1").json()
    assert detail["status"] == "ok" and list(detail["templates"]) == [f"U{i:02d}" for i in range(1, 7)]
    assert client.get("/api/ward/profiles/P01").json()["status"] == "missing"  # 기본 등록에는 없다
    assert client.get("/api/ward/profiles/P02?list=L1").json()["status"] == "missing"
    for bad in ("", "../x", "CON", "x" * 33):
        resp = client.get(f"/api/ward/profiles?list={bad}")
        assert resp.status_code == 400 and resp.json()["detail"] == wl.MSG_BAD_ID
    assert client.get("/api/ward/profiles?list=Nope").status_code == 404
    assert client.get("/api/ward/profiles/P01?list=Nope").status_code == 404
    assert client.delete("/api/ward/profiles/P02?list=L1").status_code == 404
    assert client.delete("/api/ward/profiles/P01?list=L1").json() == {"deleted": True}
    assert not (_namespace(roots) / "P01").exists() and (roots.profiles / "P02").is_dir()
