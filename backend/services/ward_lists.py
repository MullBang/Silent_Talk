"""병실 무음 도우미 '내 목록'(사용자가 직접 만든 명령 목록) 검사기와 저장소 (서버 프로세스, 표준 라이브러리만).

기본 목록(frontend/src/ward/commands.json)은 그대로 둔다. 내 목록은 서버 폴더(config.WARD_LIST_DIR, 저장소 밖·gitignore)에
목록마다 JSON 파일 하나로 둔다. 이 모듈의 검사기(check_list)가 저장·녹화·화면 점검 모두의 기준이다. 화면은 한글만 썼는지·길이
같은 것을 미리 알려 줄 수 있지만, 결과는 늘 이 검사기를 따른다.

저장 형식 "ward-custom-list-v1" (<폴더>/<목록 ID>.json):
  {schema, id, title, created, updated, commands:[{id, text, display, category, emergency, cancel}], sha256}
  - 명령 ID는 저장할 때 서버가 순서대로 U01, U02, … 를 붙인다(기본 목록의 W.. 와 겹치지 않는다).
  - sha256 = commands 배열의 정규 JSON(키 정렬, ensure_ascii=False, separators=(',', ':'))의 SHA-256. 제목은 넣지 않는다.
  - 목록 ID는 등록 이름과 같은 규칙(^[A-Za-z0-9_-]{1,32}$, Windows 장치 이름 제외, services/ward_profiles.py)이다.
    Windows 파일 이름은 대소문자를 가리지 않으므로 대소문자만 다른 목록이 있으면 충돌(ListConflict, 409)이다.
  - 파일은 임시 파일에 쓴 뒤 os.replace로 바꾼다. 목록마다 잠금 하나, 새 목록 만들기(개수 상한 확인)는 폴더 잠금 하나를
    목록 잠금 안에서 잡는다(언제나 목록 → 폴더 순서라 서로 기다리다 멈추지 않는다).
  - 목록은 MAX_LISTS(20)개까지. 목록을 지우면 그 목록의 등록 이름공간(ward_profiles.ListProfileStore)도 함께 지운다.

검사 (check_list) — 오류는 저장을 막고, 주의는 보여 주기만 한다:
  오류: 문장 4~20개 / 입 모양 문장(text)은 NFC·앞뒤 공백 제거·공백 하나로 줄인 뒤 한글 음절(가-힣)과 띄어쓰기만, 2~40자,
        OLKAVS 토큰 50개 이하 / 화면 문장(display) 1~30자, 제어 문자 없음 / 종류는 기본 목록의 다섯 종류,
        긴급 표시 ⇔ 종류 '긴급' / 입 모양 문장(띄어쓰기를 빼고 비교)·화면 문장(공백 문자를 띄어쓰기로 바꿔 비교)은 서로
        달라야 함 / 취소 문장은 하나까지, 긴급일 수 없음 /
        긴급이 아닌 문장이 하나 이상 / 목록 이름 1~30자.
  주의: 음절 9~13·토큰 23~34 밖 / 긴급 4개 초과 또는 0개 / 두 문장의 입 모양이 같음(강한 주의), 반대말 한 단어만 다름
        (강한 주의), 입 모양 거리 < 0.3, 끝 입 모양 4단위 이상 같음.
  항목 형식: {code, level('error'|'strong'|'warning'), message, row(0부터, 목록 전체는 None), rows(관련 줄 전부), field}.

토큰 수 (token_count): 받침 없는 음절 2, 받침 있는 음절 3, 띄어쓰기 1. OLKAVS 72토큰 사전에서 char2grp가 내는 자모 수와 같다
  (초성·중성 각 1, 겹받침도 한 토큰, 띄어쓰기 한 토큰). 인식 워커의 phrase_tokens(scripts/olkavs_transcribe.py)와 같은지는
  backend/tests/test_ward_lists.py가 확인한다(공식 모델 코드가 있을 때).

입 모양 단위 (lip_units): 기본 목록을 만들 때 쓴 설계 규칙을 옮겼다. 자음은 입술 닫힘 B(ㅁㅂㅃㅍ)·혀끝 D(ㄴㄷㄸㄹㅅㅆㅌ)·
  J(ㅈㅉㅊ)·거의 안 보임 K(ㄱㄲㅋㅎ, 받침 ㅇ), 모음은 벌림 a(ㅏㅑ)·둥글게 u(ㅗㅛㅜㅠ)·옆으로 i(ㅣㅡㅢ)·중간 e(ㅓㅕㅔㅐㅖㅒ),
  이중 모음 ㅘ=ua, ㅝ·ㅙ·ㅞ·ㅚ=ue, ㅟ=ui. 첫소리 ㅇ은 소리가 없고, 받침은 대표음으로 바꾸며(ㅅ·ㅈ·ㅊ·ㅌ→D, ㅍ→B, ㄲ·ㅋ→K,
  ㅎ은 보이지 않음), 한 어절 안에서 모음 앞 받침은 다음 음절로 넘어가고(연음, 겹받침은 나뉨), ㄷ·ㅌ+이는 ㅈ·ㅊ(구개음화)이다.
  띄어쓰기는 입 모양이 없고, 같은 단위가 이어지면 하나로 본다. 기본 목록(commands.json)에서 최소 거리 0.500·끝 공유 최대 4다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from services import ward_profiles as wp  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA = "ward-custom-list-v1"
ID_PREFIX = "U"
MAX_LISTS = 20
MIN_COMMANDS = 4
MAX_COMMANDS = 20
TITLE_MAX = 30
TEXT_MIN_CHARS = 2
TEXT_MAX_CHARS = 40
TEXT_MAX_TOKENS = 50
DISPLAY_MAX = 30
RECOMMENDED_SYLLABLES = (9, 13)
RECOMMENDED_TOKENS = (23, 34)
RECOMMENDED_MAX_EMERGENCY = 4
LIP_CLOSE_DISTANCE = 0.3
LIP_ENDING_UNITS = 4
MAX_BODY_BYTES = 64 * 1024
"""PUT·check 요청 본문 상한(바이트). 넘으면 413."""

CATEGORY_EMERGENCY = "emergency"
CATEGORIES: tuple[str, ...] = ("emergency", "call", "care", "room", "answer")
"""기본 목록(commands.json categories)의 다섯 종류. 화면 이름표도 그 파일을 쓴다(테스트가 같은지 확인한다)."""

LEVEL_ERROR = "error"
LEVEL_STRONG = "strong"
LEVEL_WARNING = "warning"

# 사용자 문구 (경로·예외 원문을 넣지 않는다)
MSG_BAD_ID = "목록 ID는 영문·숫자·밑줄(_)·하이픈(-)으로 1~32자만 쓸 수 있어요."
MSG_CONFLICT = "대소문자만 다른 같은 ID의 목록이 이미 있어요. 다른 ID를 쓰거나 그 ID를 그대로 써 주세요."
MSG_NOT_FOUND = "내 목록을 찾을 수 없어요."
MSG_LIMIT = f"내 목록은 {MAX_LISTS}개까지 만들 수 있어요. 안 쓰는 목록을 지운 뒤 다시 해 주세요."
MSG_INVALID = "목록에 고칠 곳이 있어요. 빨간 글씨를 확인해 주세요."
MSG_UNREADABLE = "내 목록 파일을 읽지 못했어요. 목록을 지우고 다시 만들어 주세요."
MSG_STORAGE = "내 목록을 저장하지 못했어요. 저장 폴더를 확인해 주세요."
MSG_DELETE_FAILED = "내 목록을 지우지 못했어요. 잠시 뒤 다시 지워 주세요."
MSG_EXISTS = "같은 ID의 목록이 이미 있어요. 다른 ID를 쓰거나 그 목록을 열어서 고쳐 주세요."
MSG_CHANGED_ELSEWHERE = "다른 곳에서 이 목록이 바뀌었거나 지워졌어요. 목록을 다시 불러온 뒤 고쳐 주세요."

EXPECT_NEW = "new"
"""save()의 expect 값: 새 목록일 때만 저장한다(PUT If-None-Match: *). 이미 있으면 ListPrecondition(409)."""

# 연결이 끊기지 않은 비교(match) Job이 등록 파일을 잠깐 열고 있으면 Windows에서 지우기가 실패할 수 있어 몇 번 다시 해 본다.
DELETE_RETRIES = 3
DELETE_RETRY_SECONDS = 0.2

_ROW_FIELDS = ("text", "display", "category", "emergency", "cancel")


class ListError(ValueError):
    """내 목록 오류. user_message는 사용자에게 보내도 되는 고정 문구다(경로·예외 원문 없음)."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class InvalidListId(ListError):
    """규칙에 맞지 않는 목록 ID (400)."""


class ListConflict(ListError):
    """대소문자만 다른 목록이 이미 있음 (409)."""


class ListNotFound(ListError):
    """목록이 없음 (404)."""


class ListLimit(ListError):
    """목록 개수 상한 (400)."""


class ListPrecondition(ListError):
    """저장 조건이 맞지 않음 (409): 새로 만들려는데 이미 있거나(If-None-Match: *), 고친 목록이 그사이 바뀜(If-Match)."""


class ListInvalid(ListError):
    """검사 오류가 있는 목록 (400). errors·warnings는 check_list의 항목 목록이다."""

    def __init__(self, errors: list[dict[str, Any]], warnings: list[dict[str, Any]], detail: str = ""):
        super().__init__(MSG_INVALID, detail)
        self.errors = errors
        self.warnings = warnings


# ---------------------------------------------------------------------------
# 목록 ID
# ---------------------------------------------------------------------------

def is_valid_list_id(value: Any) -> bool:
    """목록 ID 규칙 ^[A-Za-z0-9_-]{1,32}$ (Windows 장치 이름 제외). 등록 이름과 같은 규칙이다."""
    return wp.is_valid_profile_id(value)


def validate_list_id(value: Any) -> str:
    """목록 ID를 확인해 그대로 돌려준다.

    Raises:
        InvalidListId: 규칙에 맞지 않을 때 (보낸 값을 문구에 넣지 않는다).
    """
    if not is_valid_list_id(value):
        shown = value[:40] if isinstance(value, str) else type(value).__name__
        raise InvalidListId(MSG_BAD_ID, f"list_id={shown!r}")
    return value


# ---------------------------------------------------------------------------
# 글자·토큰·입 모양
# ---------------------------------------------------------------------------

_HANGUL_FIRST = 0xAC00
_HANGUL_LAST = 0xD7A3
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = ("ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ",
         "ㅡ", "ㅢ", "ㅣ")
_JONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
         "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

_CONSONANT_UNIT = {**dict.fromkeys("ㅁㅂㅃㅍ", "B"), **dict.fromkeys("ㄴㄷㄸㄹㅅㅆㅌ", "D"),
                   **dict.fromkeys("ㅈㅉㅊ", "J"), **dict.fromkeys("ㄱㄲㅋㅎ", "K")}
_VOWEL_UNITS = {"ㅏ": "a", "ㅑ": "a", "ㅗ": "u", "ㅛ": "u", "ㅜ": "u", "ㅠ": "u", "ㅣ": "i", "ㅡ": "i", "ㅢ": "i",
                "ㅓ": "e", "ㅕ": "e", "ㅔ": "e", "ㅐ": "e", "ㅖ": "e", "ㅒ": "e",
                "ㅘ": "ua", "ㅝ": "ue", "ㅙ": "ue", "ㅞ": "ue", "ㅚ": "ue", "ㅟ": "ui"}
# 자음 앞·어절 끝의 받침 (대표음). 빈 값은 입 모양이 없다.
_CODA_UNIT = {"": "", "ㄱ": "K", "ㄲ": "K", "ㅋ": "K", "ㄳ": "K", "ㄺ": "K",
              "ㄴ": "D", "ㄵ": "D", "ㄶ": "D", "ㄷ": "D", "ㅅ": "D", "ㅆ": "D", "ㅈ": "D", "ㅊ": "D", "ㅌ": "D",
              "ㄹ": "D", "ㄼ": "D", "ㄽ": "D", "ㄾ": "D", "ㅀ": "D",
              "ㅁ": "B", "ㄻ": "B", "ㅂ": "B", "ㅍ": "B", "ㅄ": "B", "ㄿ": "B", "ㅇ": "K", "ㅎ": ""}
# 같은 어절 안 모음으로 시작하는 음절 앞의 받침: (남는 받침, 다음 음절 첫소리로 넘어가는 자음)
_CODA_LIAISON = {"": ("", ""), "ㄱ": ("", "ㄱ"), "ㄲ": ("", "ㄲ"), "ㅋ": ("", "ㅋ"), "ㄳ": ("ㄱ", "ㅅ"),
                 "ㄴ": ("", "ㄴ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("", "ㄴ"), "ㄷ": ("", "ㄷ"), "ㅅ": ("", "ㅅ"),
                 "ㅆ": ("", "ㅆ"), "ㅈ": ("", "ㅈ"), "ㅊ": ("", "ㅊ"), "ㅌ": ("", "ㅌ"), "ㄹ": ("", "ㄹ"),
                 "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"),
                 "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("", "ㄹ"), "ㅁ": ("", "ㅁ"), "ㅂ": ("", "ㅂ"), "ㅍ": ("", "ㅍ"),
                 "ㅄ": ("ㅂ", "ㅅ"), "ㅇ": ("ㅇ", ""), "ㅎ": ("", "")}
_PALATAL = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}

# 한 단어만 바꾸면 같은 문장이 되는 반대말 짝 (입 모양으로 가르기 어렵고, 잘못 고르면 정반대 요청이 된다)
OPPOSITE_PAIRS: tuple[tuple[str, str], ...] = (
    ("켜", "꺼"), ("켜", "끄"), ("올려", "내려"), ("올리", "내리"), ("열어", "닫아"), ("열", "닫"),
    ("높여", "낮춰"), ("높이", "낮추"), ("늘려", "줄여"), ("키워", "줄여"), ("크게", "작게"), ("밝게", "어둡게"),
    ("세게", "약하게"), ("올라", "내려"),
    ("빨리", "천천히"), ("더", "덜"), ("위로", "아래로"), ("앞으로", "뒤로"), ("왼쪽", "오른쪽"), ("넣어", "빼"),
    ("입혀", "벗겨"), ("좋아", "싫어"), ("네", "아니요"), ("있어", "없어"), ("들어와", "나가"), ("와", "가"),
)

_SPACE_RE = re.compile(r" {2,}")


def normalize_text(value: str) -> str:
    """입 모양 문장 정규화: NFC, 앞뒤 공백 제거, 공백(줄바꿈·탭 포함) 묶음을 띄어쓰기 하나로."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_display(value: str) -> str:
    """화면 문장·목록 이름 정규화: NFC, 공백 문자(Zs: 띄어쓰기·NBSP·전각 공백 등)를 띄어쓰기로 바꾸고 앞뒤 공백 제거,
    이어진 띄어쓰기를 하나로. 줄바꿈·탭 같은 제어 문자는 바꾸지 않고 검사에서 거른다(앞뒤에 있으면 지운다).

    그래서 '물 주세요'와 '물<NBSP>주세요'는 같은 화면 문장으로 보고 중복 검사에 걸린다.
    """
    nfc = unicodedata.normalize("NFC", value)
    spaced = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in nfc)
    return _SPACE_RE.sub(" ", spaced.strip())


def _is_syllable(ch: str) -> bool:
    """완성형 한글 음절(가-힣)인지."""
    return len(ch) == 1 and _HANGUL_FIRST <= ord(ch) <= _HANGUL_LAST


def _decompose(syllable: str) -> tuple[str, str, str]:
    """완성형 음절 → (초성, 중성, 종성) 호환 자모. 종성 없으면 ''.

    Raises:
        ValueError: 완성형 한글 음절이 아닐 때.
    """
    if not _is_syllable(syllable):
        raise ValueError(f"한글 음절이 아닙니다: {syllable!r}")
    code = ord(syllable) - _HANGUL_FIRST
    return _CHO[code // 588], _JUNG[(code % 588) // 28], _JONG[code % 28]


def syllable_count(text: str) -> int:
    """정규화한 문장의 한글 음절 수 (띄어쓰기 제외)."""
    return sum(1 for ch in text if _is_syllable(ch))


def token_count(text: str) -> int:
    """OLKAVS 72토큰 수 = 받침 없는 음절 2 + 받침 있는 음절 3 + 띄어쓰기 1 (char2grp 자모 분해 수와 같다).

    Raises:
        ValueError: 한글 음절·띄어쓰기 밖의 글자가 있을 때(그런 문장은 검사에서 먼저 거른다).
    """
    total = 0
    for ch in text:
        if ch == " ":
            total += 1
        else:
            total += 3 if _decompose(ch)[2] else 2
    return total


def _word_units(word: str) -> list[str]:
    """어절 하나 → 입 모양 단위 목록 (연음·대표음·구개음화 반영, 합치기 전)."""
    sylls = [list(_decompose(ch)) for ch in word]
    out: list[str] = []
    for idx, (onset, nucleus, coda) in enumerate(sylls):
        if onset != "ㅇ":
            out.append(_CONSONANT_UNIT[onset])
        out.extend(_VOWEL_UNITS[nucleus])
        nxt = sylls[idx + 1] if idx + 1 < len(sylls) else None
        if nxt is not None and nxt[0] == "ㅇ" and coda:
            stay, move = _CODA_LIAISON[coda]
            if stay:
                out.append(_CODA_UNIT[stay])
            if move:
                if move in _PALATAL and nxt[1] == "ㅣ":
                    move = _PALATAL[move]
                nxt[0] = move  # 다음 음절이 넘어온 자음으로 시작한다
        elif _CODA_UNIT[coda]:
            out.append(_CODA_UNIT[coda])
    return out


def lip_units(text: str) -> str:
    """정규화한 문장 → 입 모양 단위 문자열 (띄어쓰기 없음, 같은 단위가 이어지면 하나).

    Raises:
        ValueError: 한글 음절·띄어쓰기 밖의 글자가 있을 때.
    """
    try:
        seq: list[str] = []
        for word in text.split():
            seq.extend(_word_units(word))
        merged: list[str] = []
        for unit in seq:
            if not merged or merged[-1] != unit:
                merged.append(unit)
        return "".join(merged)
    except KeyError as exc:
        raise ValueError(f"입 모양으로 바꾸지 못한 자모: {exc}") from exc


def edit_distance(a: str, b: str) -> int:
    """레벤시테인 거리 (넣기·빼기·바꾸기 각 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def lip_distance(a: str, b: str) -> float:
    """입 모양 단위 문자열의 정규화 거리 = 편집 거리 / 긴 쪽 길이 (0이면 입 모양이 같다)."""
    longest = max(len(a), len(b))
    return edit_distance(a, b) / longest if longest else 0.0


def shared_ending(a: str, b: str) -> int:
    """두 입 모양 단위 문자열의 끝에서 같은 단위 수."""
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def is_opposite_pair(a: str, b: str) -> bool:
    """정규화한 두 문장이 반대말 하나(OPPOSITE_PAIRS)만 바꾸면 같아지는가 (예: '불 켜 줘' / '불 꺼 줘')."""
    if a == b:
        return False
    for left, right in OPPOSITE_PAIRS:
        for x, y, src, dst in ((left, right, a, b), (right, left, a, b)):
            start = src.find(x)
            while start != -1:
                if src[:start] + y + src[start + len(x):] == dst:
                    return True
                start = src.find(x, start + 1)
    return False


# ---------------------------------------------------------------------------
# 검사기
# ---------------------------------------------------------------------------

def _issue(code: str, level: str, message: str, row: int | None = None, rows: list[int] | None = None,
           field: str | None = None) -> dict[str, Any]:
    """검사 항목 하나 {code, level, message, row, rows, field}. rows가 없으면 [row] 또는 []."""
    if rows is None:
        rows = [row] if row is not None else []
    return {"code": code, "level": level, "message": message, "row": row, "rows": list(rows), "field": field}


def _bad_chars(value: str) -> bool:
    """제어·서식·대리·줄/문단 구분 문자가 있는가 (화면 문장·목록 이름)."""
    return any(unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Zl", "Zp") for ch in value)


def _shown_chars(text: str) -> str:
    """입 모양 문장에서 쓸 수 없는 글자 (5개까지, 나온 순서). 보이지 않는 글자(BOM·제어 문자 등)는 'U+FEFF'처럼 코드로 보인다."""
    seen: list[str] = []
    for ch in text:
        if ch != " " and not _is_syllable(ch) and ch not in seen:
            seen.append(ch)
    shown = [ch if ch.isprintable() and not 0xD800 <= ord(ch) <= 0xDFFF else f"U+{ord(ch):04X}" for ch in seen[:5]]
    return " ".join(shown)


def _echo(value: str) -> str:
    """응답에 되돌려 보낼 문자열: 짝 없는 대리 문자(UTF-8로 쓸 수 없음)를 U+FFFD로 바꾼다. 그런 글자는 이미 오류로 알린다."""
    return "".join("\N{REPLACEMENT CHARACTER}" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)


def command_id(index: int) -> str:
    """줄 번호(0부터) → 서버가 붙이는 명령 ID (U01, U02, …)."""
    return f"{ID_PREFIX}{index + 1:02d}"


def _check_title(raw: Any, errors: list[dict[str, Any]]) -> str:
    """목록 이름 정규화·검사. 오류는 errors에 더한다."""
    if not isinstance(raw, str) or not normalize_display(raw):
        errors.append(_issue("title_empty", LEVEL_ERROR, "목록 이름을 적어 주세요.", field="title"))
        return normalize_display(raw) if isinstance(raw, str) else ""
    title = normalize_display(raw)
    if _bad_chars(title):
        errors.append(_issue("title_chars", LEVEL_ERROR, "목록 이름에 쓸 수 없는 글자가 있어요.", field="title"))
    elif len(title) > TITLE_MAX:
        errors.append(_issue("title_length", LEVEL_ERROR, f"목록 이름은 {TITLE_MAX}자까지 쓸 수 있어요.",
                             field="title"))
    return title


def _check_row(index: int, raw: Any, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """한 줄의 정규화·검사. (정규화한 명령, 입 모양 문장이 올바른가). 오류는 errors에 더한다."""
    cid = command_id(index)
    if not isinstance(raw, dict):
        errors.append(_issue("row_invalid", LEVEL_ERROR, "이 줄의 형식이 올바르지 않아요.", row=index))
        return {"id": cid, "text": "", "display": "", "category": "", "emergency": False, "cancel": False}, False

    text_ok = False
    raw_text = raw.get("text")
    text = normalize_text(raw_text) if isinstance(raw_text, str) else ""
    if not text:
        errors.append(_issue("text_empty", LEVEL_ERROR, "입 모양 문장을 적어 주세요.", row=index, field="text"))
    elif any(ch != " " and not _is_syllable(ch) for ch in text):
        shown = _shown_chars(text)
        errors.append(_issue("text_chars", LEVEL_ERROR,
                             "입 모양 문장은 한글과 띄어쓰기만 쓸 수 있어요."
                             + (f" (쓸 수 없는 글자: {shown})" if shown else ""), row=index, field="text"))
    elif not TEXT_MIN_CHARS <= len(text) <= TEXT_MAX_CHARS:
        errors.append(_issue("text_length", LEVEL_ERROR,
                             f"입 모양 문장은 {TEXT_MIN_CHARS}~{TEXT_MAX_CHARS}자로 적어 주세요.", row=index,
                             field="text"))
    elif token_count(text) > TEXT_MAX_TOKENS:
        errors.append(_issue("text_tokens", LEVEL_ERROR, "입 모양 문장이 너무 길어요. 조금 줄여 주세요.", row=index,
                             field="text"))
    else:
        text_ok = True

    raw_display = raw.get("display")
    display = normalize_display(raw_display) if isinstance(raw_display, str) else ""
    if not display:
        errors.append(_issue("display_empty", LEVEL_ERROR, "화면 문장을 적어 주세요.", row=index, field="display"))
    elif _bad_chars(display):
        errors.append(_issue("display_chars", LEVEL_ERROR, "화면 문장에 쓸 수 없는 글자가 있어요.", row=index,
                             field="display"))
    elif len(display) > DISPLAY_MAX:
        errors.append(_issue("display_length", LEVEL_ERROR, f"화면 문장은 {DISPLAY_MAX}자까지 쓸 수 있어요.",
                             row=index, field="display"))

    category = raw.get("category")
    if not isinstance(category, str) or category not in CATEGORIES:
        errors.append(_issue("category_invalid", LEVEL_ERROR, "종류를 골라 주세요.", row=index, field="category"))
        category = category if isinstance(category, str) else ""

    emergency = raw.get("emergency", False)
    if not isinstance(emergency, bool):
        errors.append(_issue("emergency_type", LEVEL_ERROR, "긴급 여부가 올바르지 않아요.", row=index,
                             field="emergency"))
        emergency = False
    elif emergency and category in CATEGORIES and category != CATEGORY_EMERGENCY:
        errors.append(_issue("emergency_mismatch", LEVEL_ERROR, "긴급으로 표시한 문장은 종류도 '긴급'이어야 해요.",
                             row=index, field="category"))
    elif not emergency and category == CATEGORY_EMERGENCY:
        errors.append(_issue("emergency_mismatch", LEVEL_ERROR, "종류가 '긴급'이면 긴급으로 표시해야 해요.",
                             row=index, field="emergency"))

    cancel = raw.get("cancel", False)
    if not isinstance(cancel, bool):
        errors.append(_issue("cancel_type", LEVEL_ERROR, "취소 여부가 올바르지 않아요.", row=index, field="cancel"))
        cancel = False
    elif cancel and emergency:
        errors.append(_issue("cancel_emergency", LEVEL_ERROR, "취소 문장은 긴급일 수 없어요.", row=index,
                             field="cancel"))

    return {"id": cid, "text": text, "display": display, "category": category, "emergency": emergency,
            "cancel": cancel}, text_ok


def _pair_warning(i: int, j: int, a: str, b: str, units: dict[int, str]) -> dict[str, Any] | None:
    """두 줄의 입 모양 주의 하나 (가장 강한 것만). 없으면 None."""
    pair = f"{i + 1}번·{j + 1}번 문장은"
    if is_opposite_pair(a, b):
        return _issue("lip_opposite", LEVEL_STRONG,
                      f"{pair} 반대말 한 단어만 달라요. 입 모양이 거의 같아서 구분하기 어려워요.", row=i,
                      rows=[i, j], field="text")
    if units[i] == units[j]:
        return _issue("lip_same", LEVEL_STRONG, f"{pair} 입 모양이 거의 같아서 구분하기 어려워요.", row=i,
                      rows=[i, j], field="text")
    if lip_distance(units[i], units[j]) < LIP_CLOSE_DISTANCE:
        return _issue("lip_close", LEVEL_WARNING, f"{pair} 입 모양이 비슷해서 헷갈릴 수 있어요.", row=i,
                      rows=[i, j], field="text")
    if shared_ending(units[i], units[j]) >= LIP_ENDING_UNITS:
        return _issue("lip_ending", LEVEL_WARNING, f"{pair} 끝부분이 같아요. 앞부분을 더 다르게 하면 좋아요.",
                      row=i, rows=[i, j], field="text")
    return None


def check_list(body: Any) -> dict[str, Any]:
    """목록 하나를 검사한다(저장하지 않는다).

    Args:
        body: {title, commands:[{text, display, category, emergency, cancel}]} (JSON 객체). 줄의 다른 키(id 등)는 무시한다.
            emergency·cancel이 없으면 false로 본다.

    Returns:
        {errors, warnings, normalized:{title, commands:[{id, text, display, category, emergency, cancel}]},
         stats:[{syllables, tokens}]}. 오류가 없을 때의 normalized.commands가 저장될 내용이다.
         stats는 줄마다(입 모양 문장이 올바르지 않으면 null).
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if not isinstance(body, dict):
        errors.append(_issue("body_invalid", LEVEL_ERROR, "목록 형식이 올바르지 않아요."))
        return {"errors": errors, "warnings": warnings, "normalized": {"title": "", "commands": []}, "stats": []}
    title = _check_title(body.get("title"), errors)
    rows = body.get("commands")
    if not isinstance(rows, list):
        errors.append(_issue("commands_missing", LEVEL_ERROR, "문장 목록이 없어요.", field="commands"))
        rows = []
    if len(rows) < MIN_COMMANDS:
        errors.append(_issue("too_few", LEVEL_ERROR, f"문장은 {MIN_COMMANDS}개 이상이어야 해요 (지금 {len(rows)}개).",
                             field="commands"))
    elif len(rows) > MAX_COMMANDS:
        errors.append(_issue("too_many", LEVEL_ERROR,
                             f"문장은 {MAX_COMMANDS}개까지 넣을 수 있어요 (지금 {len(rows)}개).", field="commands"))
        rows = rows[:MAX_COMMANDS + 20]  # 지나치게 긴 목록은 앞부분만 줄별로 본다(검사 시간 상한)

    commands: list[dict[str, Any]] = []
    valid_text: dict[int, bool] = {}
    for index, raw in enumerate(rows):
        command, ok = _check_row(index, raw, errors)
        commands.append(command)
        valid_text[index] = ok

    # 서로 같은 문장 (뒤에 나온 줄에 오류를 둔다). 입 모양 문장은 띄어쓰기를 빼고 비교한다: 띄어쓰기는 입 모양이 없어
    # 띄어쓰기만 다른 두 문장은 인식 점수가 거의 같고(1·2등 차이 기준을 넘지 못함) 등록 템플릿도 같은 입 모양으로 두 번 남는다.
    first_text: dict[str, int] = {}
    first_display: dict[str, int] = {}
    for index, command in enumerate(commands):
        text, display = command["text"], command["display"]
        if text:
            key = text.replace(" ", "")
            if key in first_text:
                other = first_text[key]
                if commands[other]["text"] == text:
                    errors.append(_issue("text_duplicate", LEVEL_ERROR, f"{other + 1}번 문장과 입 모양 문장이 같아요.",
                                         row=index, rows=[other, index], field="text"))
                else:
                    errors.append(_issue("text_duplicate_spacing", LEVEL_ERROR,
                                         f"{other + 1}번 문장과 띄어쓰기만 다르고 같은 문장이에요.",
                                         row=index, rows=[other, index], field="text"))
                valid_text[index] = False
            else:
                first_text[key] = index
        if display:
            if display in first_display:
                other = first_display[display]
                errors.append(_issue("display_duplicate", LEVEL_ERROR, f"{other + 1}번 문장과 화면 문장이 같아요.",
                                     row=index, rows=[other, index], field="display"))
            else:
                first_display[display] = index

    cancels = [i for i, c in enumerate(commands) if c["cancel"]]
    for index in cancels[1:]:
        errors.append(_issue("cancel_multiple", LEVEL_ERROR, "취소 문장은 하나만 둘 수 있어요.", row=index,
                             rows=cancels, field="cancel"))
    emergency_count = sum(1 for c in commands if c["emergency"])
    if commands and emergency_count == len(commands):
        errors.append(_issue("no_normal", LEVEL_ERROR, "긴급이 아닌 문장이 하나 이상 있어야 해요.", field="commands"))

    # 주의
    stats: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        if not valid_text.get(index) and not (command["text"] and _all_hangul(command["text"])):
            stats.append({"syllables": None, "tokens": None})
            continue
        syllables, tokens = syllable_count(command["text"]), token_count(command["text"])
        stats.append({"syllables": syllables, "tokens": tokens})
        if valid_text.get(index) and not (RECOMMENDED_SYLLABLES[0] <= syllables <= RECOMMENDED_SYLLABLES[1]
                                          and RECOMMENDED_TOKENS[0] <= tokens <= RECOMMENDED_TOKENS[1]):
            warnings.append(_issue("length", LEVEL_WARNING,
                                   f"짧거나 길어서 잘 읽히지 않을 수 있어요 (지금 {syllables}음절·토큰 {tokens}개, "
                                   f"권장 {RECOMMENDED_SYLLABLES[0]}~{RECOMMENDED_SYLLABLES[1]}음절).",
                                   row=index, field="text"))
    if commands and len(commands) >= MIN_COMMANDS:
        if emergency_count > RECOMMENDED_MAX_EMERGENCY:
            warnings.append(_issue("emergency_many", LEVEL_WARNING,
                                   f"긴급이 많으면 거짓 긴급 알림이 늘어요 (지금 {emergency_count}개, "
                                   f"권장 {RECOMMENDED_MAX_EMERGENCY}개 이하).", field="emergency"))
        elif emergency_count == 0:
            warnings.append(_issue("emergency_none", LEVEL_WARNING,
                                   "긴급 문장이 없으면 입 읽기로는 빨간 카드가 나지 않고 스위치 3번 긴급 호출만 남아요.",
                                   field="emergency"))
    usable = [i for i in range(len(commands)) if valid_text.get(i)]
    units = {i: lip_units(commands[i]["text"]) for i in usable}
    for pos, i in enumerate(usable):
        for j in usable[pos + 1:]:
            warning = _pair_warning(i, j, commands[i]["text"], commands[j]["text"], units)
            if warning is not None:
                warnings.append(warning)
    echoed = [{k: _echo(v) if isinstance(v, str) else v for k, v in c.items()} for c in commands]
    return {"errors": errors, "warnings": warnings, "normalized": {"title": _echo(title), "commands": echoed},
            "stats": stats}


def _all_hangul(text: str) -> bool:
    """한글 음절과 띄어쓰기만 있는가 (길이 오류가 있어도 음절·토큰 수는 보여 준다)."""
    return all(ch == " " or _is_syllable(ch) for ch in text)


def commands_sha256(commands: list[dict[str, Any]]) -> str:
    """commands 배열의 정규 JSON(키 정렬, ensure_ascii=False, separators=(',', ':'), UTF-8)의 SHA-256."""
    canonical = json.dumps(commands, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _as_input(doc: dict[str, Any]) -> dict[str, Any]:
    """저장된 목록 → 검사기 입력 {title, commands:[{text, display, category, emergency, cancel}]}."""
    return {"title": doc.get("title"),
            "commands": [{k: c.get(k) for k in _ROW_FIELDS} if isinstance(c, dict) else c
                         for c in doc.get("commands") or []]}


# ---------------------------------------------------------------------------
# 저장소
# ---------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(key: str) -> threading.Lock:
    """이름 하나의 잠금 (처음 부를 때 만든다)."""
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    """저장된 목록 → 목록 화면용 요약 {id, title, count, emergency_count, updated, sha256}."""
    commands = doc["commands"]
    return {"id": doc["id"], "title": doc["title"], "count": len(commands),
            "emergency_count": sum(1 for c in commands if c.get("emergency") is True),
            "updated": doc.get("updated"), "sha256": doc["sha256"]}


class ListStore:
    """내 목록 폴더 하나(config.WARD_LIST_DIR)의 읽기·쓰기."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        """root가 None이면 config.WARD_LIST_DIR(만들 때의 값)을 쓴다."""
        self.root = Path(root if root is not None else config.WARD_LIST_DIR)

    # -- 잠금 ------------------------------------------------------------------
    def _root_key(self) -> str:
        """잠금 키에 쓰는 폴더 이름 (대소문자·구분자 차이를 없앤다)."""
        return os.path.normcase(str(self.root.resolve()))

    def lock(self, list_id: str) -> threading.Lock:
        """목록 하나의 잠금 (대소문자만 다른 ID도 같은 잠금). 등록 저장(olkavs_service)도 이 잠금 안에서 한다."""
        return _lock(f"list|{self._root_key()}|{list_id.lower()}")

    def _create_lock(self) -> threading.Lock:
        """새 목록 만들기(개수 상한 확인 + 쓰기) 잠금. 언제나 목록 잠금 안에서 잡는다."""
        return _lock(f"create|{self._root_key()}")

    # -- 찾기·읽기 -------------------------------------------------------------
    def _entries(self) -> list[str]:
        """폴더의 항목 이름 (폴더가 없으면 빈 목록)."""
        try:
            return sorted(os.listdir(self.root)) if self.root.is_dir() else []
        except OSError as exc:
            raise ListError(MSG_UNREADABLE, f"목록 폴더 읽기 실패: {exc}") from exc

    def _ids(self) -> list[str]:
        """규칙에 맞는 이름의 목록 파일 ID들 (이름순)."""
        out = []
        for name in self._entries():
            if name.endswith(".json") and is_valid_list_id(name[:-5]) and (self.root / name).is_file():
                out.append(name[:-5])
        return out

    def _find(self, list_id: str) -> Path | None:
        """정확히 같은 이름의 목록 파일. 없으면 None.

        Raises:
            ListConflict: 대소문자만 다른 목록 파일이 있을 때.
        """
        name = f"{list_id}.json"
        names = self._entries()
        if name in names:
            path = self.root / name
            return path if path.is_file() else None
        if any(entry.lower() == name.lower() for entry in names):
            raise ListConflict(MSG_CONFLICT, f"대소문자만 다른 목록: {list_id!r}")
        return None

    @staticmethod
    def _read(path: Path, list_id: str) -> dict[str, Any]:
        """목록 파일을 읽고 형식·ID·SHA-256을 확인한다.

        Raises:
            ListError: 깨졌거나 다른 목록의 파일이거나 SHA-256이 맞지 않을 때.
        """
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as exc:
            raise ListError(MSG_UNREADABLE, f"목록 파일 읽기 실패: {type(exc).__name__}") from exc
        try:
            commands = doc["commands"]
            ok = (isinstance(doc, dict) and doc.get("schema") == SCHEMA and doc.get("id") == list_id
                  and isinstance(doc.get("title"), str) and isinstance(commands, list)
                  and all(isinstance(c, dict) and set(c) == {"id", *_ROW_FIELDS} for c in commands)
                  and [c["id"] for c in commands] == [command_id(i) for i in range(len(commands))]
                  and doc.get("sha256") == commands_sha256(commands))
        except (KeyError, TypeError) as exc:
            raise ListError(MSG_UNREADABLE, f"목록 파일 형식 오류: {type(exc).__name__}") from exc
        if not ok:
            raise ListError(MSG_UNREADABLE, f"목록 파일 형식·SHA-256이 다릅니다: {list_id!r}")
        return doc

    def exists(self, list_id: str) -> bool:
        """목록 파일이 있는가 (대소문자만 다른 파일은 없음으로 본다)."""
        try:
            return self._find(validate_list_id(list_id)) is not None
        except ListError:
            return False

    def count(self) -> int:
        """저장된 목록 수 (규칙에 맞는 이름의 파일)."""
        return len(self._ids())

    def get(self, list_id: str) -> dict[str, Any]:
        """저장된 목록 하나.

        Raises:
            InvalidListId / ListConflict / ListNotFound / ListError(읽기 실패).
        """
        validate_list_id(list_id)
        with self.lock(list_id):
            path = self._find(list_id)
            if path is None:
                raise ListNotFound(MSG_NOT_FOUND, f"없는 목록: {list_id!r}")
            return self._read(path, list_id)

    def detail(self, list_id: str) -> dict[str, Any]:
        """저장된 목록 + 지금 검사기로 본 {checks:{errors, warnings}}."""
        doc = self.get(list_id)
        result = check_list(_as_input(doc))
        return {**doc, "checks": {"errors": result["errors"], "warnings": result["warnings"]}}

    def load_valid(self, list_id: str) -> dict[str, Any]:
        """녹화에 쓸 목록: 있고, 지금 검사기로 오류가 없어야 한다.

        Raises:
            InvalidListId / ListConflict / ListNotFound / ListInvalid / ListError(읽기 실패).
        """
        doc = self.get(list_id)
        result = check_list(_as_input(doc))
        if result["errors"]:
            raise ListInvalid(result["errors"], result["warnings"], f"검사 오류가 있는 목록: {list_id!r}")
        return doc

    def summaries(self) -> list[dict[str, Any]]:
        """저장된 목록 요약 (ID순). 깨진 파일·대소문자 충돌은 건너뛴다(서버 로그)."""
        out = []
        for list_id in self._ids():
            try:
                out.append(_summary(self.get(list_id)))
            except ListError as exc:
                logger.warning("내 목록 요약에서 건너뜀 (%s): %s", list_id, exc.detail or exc)
        return out

    # -- 쓰기 ------------------------------------------------------------------
    def _write(self, path: Path, doc: dict[str, Any]) -> None:
        """목록 파일을 원자적으로 쓴다 (폴더가 없으면 만든다).

        Raises:
            ListError: 쓰지 못했을 때.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            wp._atomic_write(path, json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8"))
        except OSError as exc:
            raise ListError(MSG_STORAGE, f"목록 저장 실패: {exc}") from exc

    def save(self, list_id: str, body: Any, expect: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """목록을 검사하고 저장한다(있으면 바꾼다, 만든 시각은 그대로). 명령 ID는 U01부터 다시 붙인다.

        Args:
            list_id: 목록 ID.
            body: {title, commands}.
            expect: 저장 조건. None이면 조건 없음(없으면 만들고 있으면 바꾼다, 이 기능의 처음 동작).
                EXPECT_NEW('new')면 그 ID의 목록이 아직 없을 때만 만든다(화면의 '새 목록' 저장).
                SHA-256 문자열이면 지금 저장된 목록의 sha256이 그 값일 때만 바꾼다(화면이 불러온 판을 고칠 때).
                조건은 목록 잠금 안에서 본다.

        Returns:
            (저장한 목록, 주의 목록).

        Raises:
            InvalidListId / ListInvalid(검사 오류) / ListConflict / ListPrecondition(조건 불일치) / ListLimit /
            ListError(쓰기 실패).
        """
        validate_list_id(list_id)
        result = check_list(body)
        if result["errors"]:
            raise ListInvalid(result["errors"], result["warnings"], f"검사 오류 {len(result['errors'])}개")
        commands = result["normalized"]["commands"]
        now = wp._now()
        with self.lock(list_id):
            path = self._find(list_id)
            self._check_expect(list_id, path, expect)
            doc = {"schema": SCHEMA, "id": list_id, "title": result["normalized"]["title"], "created": now,
                   "updated": now, "commands": commands, "sha256": commands_sha256(commands)}
            if path is not None:
                try:
                    doc["created"] = self._read(path, list_id).get("created") or now
                except ListError as exc:
                    logger.warning("내 목록 %s: 이전 파일을 읽지 못해 새로 씁니다 (%s)", list_id, exc.detail or exc)
                self._write(path, doc)
            else:
                with self._create_lock():
                    if self.count() >= MAX_LISTS:
                        raise ListLimit(MSG_LIMIT, f"목록 {MAX_LISTS}개 상한")
                    # 잠금을 기다리는 사이 대소문자만 다른 목록이 생겼을 수 있다
                    path = self._find(list_id) or self.root / f"{list_id}.json"
                    self._write(path, doc)
        logger.info("내 목록 저장: %s (%d문장, sha256 %s)", list_id, len(commands), doc["sha256"][:12])
        return doc, result["warnings"]

    def _check_expect(self, list_id: str, path: Path | None, expect: str | None) -> None:
        """save()의 저장 조건을 확인한다 (목록 잠금 안에서 부른다).

        Raises:
            ListPrecondition: 새 목록인데 이미 있거나, 고칠 목록이 없거나 그 sha256이 다를 때.
        """
        if expect is None:
            return
        if expect == EXPECT_NEW:
            if path is not None:
                raise ListPrecondition(MSG_EXISTS, f"이미 있는 목록을 새로 만들려 함: {list_id!r}")
            return
        if path is None:
            raise ListPrecondition(MSG_CHANGED_ELSEWHERE, f"고칠 목록이 없음: {list_id!r}")
        try:
            current = self._read(path, list_id).get("sha256")
        except ListError as exc:
            raise ListPrecondition(MSG_CHANGED_ELSEWHERE, f"고칠 목록을 읽지 못함: {exc.detail or exc}") from exc
        if current != expect:
            raise ListPrecondition(MSG_CHANGED_ELSEWHERE,
                                   f"목록이 그사이 바뀜: {list_id!r} {str(current)[:12]} != {str(expect)[:12]}")

    def delete(self, list_id: str, profile_root: str | os.PathLike[str] | None = None) -> None:
        """그 목록의 등록 이름공간을 먼저 지우고, 목록 파일을 지운다.

        등록(개인 입 모양 특징)을 먼저 지운다. 그것을 지우지 못하면 목록도 남겨 두어 다시 지울 수 있게 한다.

        Args:
            list_id: 목록 ID.
            profile_root: 등록 폴더. None이면 config.WARD_PROFILE_DIR(부를 때의 값).

        등록 폴더 지우기는 몇 번(DELETE_RETRIES) 다시 해 본다. 같은 목록의 비교 Job이 등록 파일을 잠깐 열고 있으면 Windows에서
        지우기가 실패할 수 있기 때문이다. 그래도 실패하면 지우기 전용 문구(MSG_DELETE_FAILED)로 알린다.

        Raises:
            InvalidListId / ListNotFound / ListConflict / ListError(지우지 못함, MSG_DELETE_FAILED).
        """
        validate_list_id(list_id)
        with self.lock(list_id):
            path = self._find(list_id)
            if path is None:
                raise ListNotFound(MSG_NOT_FOUND, f"없는 목록: {list_id!r}")
            try:
                wp.delete_list_namespace(list_id, profile_root, retries=DELETE_RETRIES,
                                         retry_seconds=DELETE_RETRY_SECONDS)
            except wp.ProfileError as exc:
                raise ListError(MSG_DELETE_FAILED, f"등록 이름공간 삭제 실패: {exc.detail or exc}") from exc
            try:
                os.remove(path)
            except OSError as exc:
                raise ListError(MSG_DELETE_FAILED, f"목록 삭제 실패: {exc}") from exc
        logger.info("내 목록 삭제: %s", list_id)

    @contextmanager
    def guard(self, list_id: str) -> Iterator[bool]:
        """목록 잠금을 잡고 그 목록 파일이 아직 있는지 알려 준다(등록 저장이 지워진 목록에 쓰지 않게)."""
        with self.lock(list_id):
            try:
                present = self._find(list_id) is not None
            except ListError:
                present = False
            yield present


def get_store() -> ListStore:
    """현재 config.WARD_LIST_DIR의 저장소 (부를 때마다 설정 값을 다시 읽는다)."""
    return ListStore(config.WARD_LIST_DIR)


def health_status() -> dict[str, Any]:
    """/health model.ward_phrases.custom_lists: {enabled, dir_writable, count}. 경로·예외 원문은 넣지 않는다."""
    try:
        store = get_store()
        return {"enabled": True, "dir_writable": wp.profile_dir_writable(store.root), "count": store.count()}
    except Exception as exc:  # noqa: BLE001 - 상태 조회가 /health를 깨지 않는다
        logger.warning("내 목록 상태 조회 실패: %s", exc)
        return {"enabled": True, "dir_writable": False, "count": 0}


def record_phrases(doc: dict[str, Any]) -> list[dict[str, str]]:
    """저장된(검사를 통과한) 목록 → 인식 워커 후보 [{id, text}]."""
    return [{"id": str(c["id"]), "text": str(c["text"])} for c in doc["commands"]]
