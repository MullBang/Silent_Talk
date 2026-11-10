"""웹캠 녹화 접수 라우트 (6단계).

POST /api/record — 브라우저 MediaRecorder 녹화(WebM, Safari는 MP4)를 받아 녹화 모드로 인식하는
백그라운드 Job을 등록한다. 녹화 모드는 앞뒤 무발화를 잘라 내고, 12초 이하 녹화는 창 하나로 인식한다
(30초 실험 모드처럼 12초를 넘으면 모델 창 길이 때문에 여러 창으로 나뉠 수 있다). 결과는 즉시 반환하지 않고 GET /api/result/{session_id} 폴링으로만 준다.

- 요청 본문은 limits.RequestBodyLimitMiddleware가 RECORD_MAX_SIZE_MB(+멀티파트 여유 1MB)를 넘으면 받는 즉시
  413으로 끊는다. 받아 둔 사본은 RECORD_MAX_SIZE_MB+1바이트까지만 옮긴다(넘으면 400). 내용 시그니처로만 형식을 판정한다
  (WebM: EBML 1A45DFA3 + DocType webm/matroska, MP4: ftyp). 클라이언트 파일명·확장자·MIME은 믿지 않는다.
- MediaRecorder WebM에는 길이 메타데이터가 없고 cv2가 fps=1000·프레임 수 음수를 돌려준다. 그래서 서버가
  실제 프레임 시각(CAP_PROP_POS_MSEC)을 읽어 길이를 잰다. RECORD_MIN_SECONDS 이상, RECORD_MAX_SECONDS
  미만만 접수한다. 측정은 asyncio.to_thread로 돌려 이벤트 루프를 막지 않는다.
- 거부 사유는 고정 한국어 문구로만 알린다(서버 경로·예외 원문 금지). 거부하면 저장한 파일을 바로 지운다.
- 공식 모델(olkavs) 백엔드에서만 지원한다. compact면 503.
- 선택 필드 phrase_set: 지정 문장 묶음 ID('sentences' 기본, 'ward' 병실 명령). 필드를 보내지 않을 때만 기본값을 쓰고,
  화이트리스트(config.PHRASE_SET_IDS) 밖의 값·빈 값·같은 필드 여러 개는 400이다. FastAPI는 빈 폼 값을 '보내지 않음'으로
  바꾸므로 원래 폼(request.form(), 이미 읽은 것을 다시 씀)에서 직접 확인한다. 목록 파일 경로는 클라이언트에게서 받지 않는다.
- 선택 필드 enroll_profile·enroll_command(내 입 모양 등록): phrase_set=ward에서만 받는다. enroll_command는 enroll_profile과
  함께, 병실 명령 목록의 ID만 받는다. 규칙 위반·빈 값·중복 필드는 파일을 저장하기 전에 400(고정 문구)이다.
  응답 형식은 그대로이고, 등록 결과는 폴링 결과 results[0].enroll로 준다(services/ward_profiles.py).
- 선택 필드 command_list(내 목록 ID, services/ward_lists.py): phrase_set=ward에서만 받는다. 값 하나, 목록 ID 규칙
  (^[A-Za-z0-9_-]{1,32}$, Windows 장치 이름 제외), 목록이 있고 지금 검사기로 오류가 없어야 한다. 아니면 파일을 저장하기 전에
  400(고정 문구)이다. 그 목록이 후보가 되고 결과 phrase에 list {id, sha256}가 붙는다. 등록 필드를 함께 보내면 그 목록의
  등록 이름공간을 쓰고, enroll_command는 그 목록의 ID(U01…)만 받는다. command_list가 없으면 이 기능 이전과 같다.
- 선택 필드 command_list_sha256(command_list와 함께만): 화면이 불러온 목록의 sha256. 서버의 지금 목록과 다르면 파일을
  저장하기 전에 400(MSG_LIST_CHANGED)이다. 등록 화면이 다른 탭에서 고친 목록의 새 문장으로 옛 입 모양을 저장하지 않게 한다.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import statistics
import sys
import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402
from api import infer as infer_api  # noqa: E402
from api import limits  # noqa: E402
from schemas.models import RecordStartResponse  # noqa: E402

router = APIRouter()
logger = logging.getLogger(__name__)

# 사용자에게 보내는 고정 문구 (경로·예외 원문을 넣지 않는다)
MSG_BACKEND_UNSUPPORTED = "녹화 인식은 공식 인식 모델(olkavs) 서버에서만 사용할 수 있습니다."
MSG_EMPTY = "녹화 파일이 비어 있거나 너무 작습니다. 다시 녹화해 주세요."
MSG_UNSUPPORTED = "지원하지 않는 녹화 형식입니다 (WebM 또는 MP4만 가능)."
MSG_TOO_LARGE = "녹화 파일이 너무 큽니다 ({mb}MB 이하만 접수합니다)."
MSG_UNREADABLE = "녹화 영상을 읽을 수 없습니다. 다시 녹화해 주세요."
MSG_TOO_SHORT = "녹화가 너무 짧습니다 ({sec:g}초 이상 녹화해 주세요)."
MSG_TOO_LONG = "녹화 길이는 {sec:g}초 미만이어야 합니다."
MSG_SAVE_FAILED = "녹화 파일을 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."
MSG_BAD_PHRASE_SET = "알 수 없는 문장 묶음입니다 (sentences 또는 ward만 사용할 수 있습니다)."
MSG_ENROLL_WARD_ONLY = "내 입 모양 등록은 병실 명령 묶음(phrase_set=ward)에서만 쓸 수 있습니다."
MSG_ENROLL_NEEDS_PROFILE = "등록할 명령(enroll_command)을 보낼 때는 등록 이름(enroll_profile)도 함께 보내야 합니다."
MSG_BAD_ENROLL_PROFILE = "등록 이름은 영문·숫자·밑줄(_)·하이픈(-)으로 1~32자만 쓸 수 있습니다."
MSG_BAD_ENROLL_COMMAND = "알 수 없는 병실 명령입니다 (병실 명령 목록의 ID만 쓸 수 있습니다)."
MSG_ENROLL_ORIGIN = "허용하지 않은 화면 주소에서 보낸 등록 요청입니다. 이 프로그램의 화면에서 다시 해 주세요."
MSG_LIST_WARD_ONLY = "내 목록은 병실 명령 묶음(phrase_set=ward)에서만 쓸 수 있어요."
MSG_BAD_COMMAND_LIST = "내 목록 ID는 영문·숫자·밑줄(_)·하이픈(-)으로 1~32자만 쓸 수 있어요."
MSG_LIST_NOT_FOUND = "내 목록을 찾을 수 없어요."
MSG_LIST_INVALID = "내 목록에 오류가 있어요. 내 목록 관리에서 고친 뒤 다시 해 주세요."
MSG_BAD_LIST_ENROLL_COMMAND = "내 목록에 없는 명령이에요 (그 목록의 명령 ID만 쓸 수 있어요)."
MSG_LIST_SHA_NEEDS_LIST = "목록 지문(command_list_sha256)은 내 목록(command_list)과 함께만 보낼 수 있어요."
MSG_BAD_LIST_SHA = "목록 지문(command_list_sha256)은 SHA-256 16진수 64자여야 해요."
MSG_LIST_CHANGED = "목록이 바뀌었어요. 다시 불러와 주세요."
"""화면이 불러온 목록 판(sha256)과 서버의 지금 목록이 다를 때. 화면(frontend/src/api/client.js LIST_CHANGED_MESSAGE)도 이 문구로 알아본다."""

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_MIN_FILE_BYTES = 32
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_EBML_DOCTYPE_ID = 0x4282
_HEADER_BYTES = 4096


class RecordingRejected(ValueError):
    """녹화를 접수하지 않는 사유. user_message는 사용자에게 보내도 되는 고정 문구다."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


# ---------------------------------------------------------------------------
# 형식 판정 (내용 시그니처)
# ---------------------------------------------------------------------------

def _read_vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int] | None:
    """EBML 가변 길이 정수를 읽는다.

    Args:
        data: 바이트열.
        pos: 시작 위치.
        keep_marker: True면 요소 ID처럼 길이 표시 비트를 남기고, False면 크기 값처럼 지운다.

    Returns:
        (값, 다음 위치) 또는 형식이 잘못되면 None.
    """
    if pos >= len(data):
        return None
    first = data[pos]
    length = 1
    mask = 0x80
    while length <= 8 and not (first & mask):
        length += 1
        mask >>= 1
    if length > 8 or pos + length > len(data):
        return None
    value = first if keep_marker else (first & (mask - 1))
    for byte in data[pos + 1:pos + length]:
        value = (value << 8) | byte
    return value, pos + length


def _ebml_doctype(head: bytes) -> str | None:
    """EBML 헤더의 DocType 문자열을 찾는다 (없거나 형식이 깨졌으면 None)."""
    try:
        if not head.startswith(_EBML_MAGIC):
            return None
        size = _read_vint(head, 4, keep_marker=False)
        if size is None:
            return None
        header_size, pos = size
        end = min(len(head), pos + header_size)
        while pos < end:
            element = _read_vint(head, pos, keep_marker=True)
            if element is None:
                return None
            element_id, pos = element
            length = _read_vint(head, pos, keep_marker=False)
            if length is None:
                return None
            element_size, pos = length
            if pos + element_size > len(head):
                return None
            if element_id == _EBML_DOCTYPE_ID:
                return head[pos:pos + element_size].rstrip(b"\x00").decode("ascii", "strict")
            pos += element_size
        return None
    except (UnicodeDecodeError, ValueError, IndexError):
        return None


def detect_recording_container(head: bytes) -> str | None:
    """녹화 파일 선두 바이트로 컨테이너를 판정한다.

    Args:
        head: 파일 앞부분(수 KB).

    Returns:
        'webm'(EBML + DocType webm/matroska), 'mp4'(ISO BMFF ftyp) 또는 지원하지 않으면 None.
    """
    if head.startswith(_EBML_MAGIC):
        doctype = _ebml_doctype(head)
        return "webm" if doctype in ("webm", "matroska") else None
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    return None


# ---------------------------------------------------------------------------
# 실제 길이 측정
# ---------------------------------------------------------------------------

def measure_recording_seconds(path: str, max_seconds: float = config.RECORD_MAX_SECONDS,
                              max_frames: int = config.RECORD_MAX_FRAMES,
                              time_budget: float = config.RECORD_PROBE_TIMEOUT_SEC) -> float:
    """프레임 시각(PTS)을 실제로 읽어 녹화 길이(초)를 잰다.

    길이 = 마지막 PTS − 첫 PTS + 대표 프레임 간격(PTS 차이의 중앙값). 메타데이터(fps·프레임 수)는
    MediaRecorder WebM에서 믿을 수 없으므로 쓰지 않는다(간격을 모를 때의 보조값으로만 사용).
    PTS가 상한을 1초 넘게 지나면 거기서 멈춰 그때까지의 길이(상한 이상)를 돌려준다.

    Args:
        path: 저장된 녹화 파일 경로.
        max_seconds: 길이 상한(초). 조기 종료 판정에만 쓴다.
        max_frames: 읽을 최대 프레임 수. 넘으면 비정상 파일로 거부한다.
        time_budget: 측정에 쓸 최대 시간(초). 넘으면 거부한다.

    Returns:
        녹화 길이(초).

    Raises:
        RecordingRejected: 열 수 없음·첫 프레임 디코딩 실패·PTS 비정상·프레임 수/시간 초과.
    """
    import cv2  # 서버 환경의 cv2 (지연 import)

    started = time.monotonic()
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RecordingRejected(MSG_UNREADABLE, "VideoCapture를 열 수 없음")
        ok, frame = cap.read()  # 첫 프레임은 실제로 디코딩해 본다
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            raise RecordingRejected(MSG_UNREADABLE, "첫 프레임 디코딩 실패")
        first = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not math.isfinite(first):
            raise RecordingRejected(MSG_UNREADABLE, "첫 프레임 PTS 비정상")
        last = first
        prev = first
        gaps: list[float] = []
        frames = 1
        limit_ms = max_seconds * 1000.0 + 1000.0
        while True:
            if last - first > limit_ms:
                break  # 상한을 충분히 넘었다: 더 읽지 않는다
            if frames >= max_frames:
                raise RecordingRejected(MSG_UNREADABLE, f"프레임 수 상한({max_frames}) 초과")
            if time.monotonic() - started > time_budget:
                raise RecordingRejected(MSG_UNREADABLE, f"길이 측정 시간 상한({time_budget:.0f}초) 초과")
            if not cap.grab():
                break
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            if not math.isfinite(pts):
                raise RecordingRejected(MSG_UNREADABLE, "PTS 비정상")
            frames += 1
            if pts > prev:
                gaps.append(pts - prev)
            prev = pts
            last = max(last, pts)
        if gaps:
            gap = statistics.median(gaps)
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            gap = 1000.0 / fps if 1.0 <= fps <= 240.0 else 0.0
        return max(0.0, (last - first + gap) / 1000.0)
    except RecordingRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - 디코더 오류는 모두 '읽을 수 없음'으로 본다
        raise RecordingRejected(MSG_UNREADABLE, f"길이 측정 오류: {type(exc).__name__}: {exc}") from exc
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001 - 해제 실패는 무시
                pass


# ---------------------------------------------------------------------------
# 업로드 수신
# ---------------------------------------------------------------------------

def _safe_remove(path: str | None) -> None:
    """파일을 조용히 지운다 (없으면 무시, 실패는 서버 로그)."""
    limits.safe_remove(path)


_REJECT_MESSAGES = {
    limits.REJECT_EMPTY: lambda: MSG_EMPTY,
    limits.REJECT_UNSUPPORTED: lambda: MSG_UNSUPPORTED,
    limits.REJECT_TOO_LARGE: lambda: MSG_TOO_LARGE.format(mb=config.RECORD_MAX_SIZE_MB),
    limits.REJECT_SAVE_FAILED: lambda: MSG_SAVE_FAILED,
}


def resolve_phrase_set(values: list[Any] | None) -> str:
    """phrase_set 폼 필드 값 전체를 화이트리스트로 확인한다. 필드가 아예 없을 때만 기본 묶음('sentences').

    대소문자·공백을 고쳐 주지 않는다(정확히 일치하는 ID만 받는다). 경로나 파일 이름은 받지 않는다.
    빈 문자열, 같은 필드를 여러 번 보낸 요청, 문자열이 아닌 파트(파일)는 기본값으로 바꾸지 않고 거부한다.

    Args:
        values: 원래 폼의 'phrase_set' 값 목록(form.getlist). 없거나 비었으면 필드를 보내지 않은 것이다.

    Returns:
        config.PHRASE_SET_IDS 중 하나.

    Raises:
        RecordingRejected: 화이트리스트 밖의 값 (고정 문구, 보낸 값을 되돌려 보여 주지 않는다).
    """
    if not values:
        return config.PHRASE_SET_DEFAULT
    if len(values) == 1 and isinstance(values[0], str) and values[0] in config.PHRASE_SET_IDS:
        return values[0]
    shown = [str(v)[:40] if isinstance(v, str) else type(v).__name__ for v in values[:3]]
    raise RecordingRejected(MSG_BAD_PHRASE_SET, f"phrase_set={shown!r} ({len(values)}개)")


async def _form_phrase_set_values(request: Request) -> list[Any]:
    """원래 멀티파트 폼의 phrase_set 값 목록. FastAPI가 이미 읽어 둔 폼을 다시 쓴다(본문을 다시 읽지 않는다)."""
    return await _form_values(request, "phrase_set")


async def _form_values(request: Request, name: str) -> list[Any]:
    """원래 멀티파트 폼의 name 필드 값 목록 (FastAPI가 이미 읽어 둔 폼을 다시 쓴다)."""
    form = await request.form()
    return list(form.getlist(name))


def _single_text(values: list[Any], message: str, name: str) -> str | None:
    """선택 폼 필드 값: 보내지 않았으면 None, 문자열 하나면 그 값. 빈 값·여러 개·파일 파트는 거부한다.

    Raises:
        RecordingRejected: 빈 값·중복·문자열이 아닌 파트 (고정 문구, 보낸 값을 되돌려 보여 주지 않는다).
    """
    if not values:
        return None
    if len(values) == 1 and isinstance(values[0], str) and values[0] != "":
        return values[0]
    shown = [str(v)[:40] if isinstance(v, str) else type(v).__name__ for v in values[:3]]
    raise RecordingRejected(message, f"{name}={shown!r} ({len(values)}개)")


def resolve_command_list(values: list[Any] | None, phrase_set: str) -> dict[str, Any] | None:
    """command_list 폼 필드(내 목록 ID)를 확인하고 그 목록을 읽는다 (선택, ward 묶음 전용).

    - 보내지 않았으면 None(기본 목록, 이 기능 이전과 같다).
    - phrase_set이 ward가 아니면 거부한다. 빈 값·같은 필드 여러 번·파일 파트·ID 규칙 위반도 거부한다.
    - 목록이 없거나(대소문자만 다른 목록 포함) 지금 검사기로 오류가 있거나 파일을 읽지 못하면 거부한다.

    Returns:
        저장된 목록(services/ward_lists.py 형식) 또는 None.

    Raises:
        RecordingRejected: 위 규칙 위반 (고정 한국어 문구).
    """
    from services import ward_lists

    values = list(values or [])
    if not values:
        return None
    if phrase_set != config.PHRASE_SET_WARD:
        raise RecordingRejected(MSG_LIST_WARD_ONLY, f"phrase_set={phrase_set!r}")
    list_id = _single_text(values, MSG_BAD_COMMAND_LIST, "command_list")
    if not ward_lists.is_valid_list_id(list_id):
        raise RecordingRejected(MSG_BAD_COMMAND_LIST, f"command_list={str(list_id)[:40]!r}")
    try:
        return ward_lists.get_store().load_valid(list_id)
    except (ward_lists.ListNotFound, ward_lists.ListConflict) as exc:
        raise RecordingRejected(MSG_LIST_NOT_FOUND, exc.detail) from None
    except ward_lists.ListError as exc:  # 검사 오류(ListInvalid)·읽기 실패
        raise RecordingRejected(MSG_LIST_INVALID, exc.detail) from None


def resolve_command_list_sha(values: list[Any] | None, custom_list: dict[str, Any] | None) -> str | None:
    """command_list_sha256 폼 필드(화면이 불러온 목록 판)를 확인한다 (선택, command_list와 함께만).

    - 보내지 않았으면 None(지금 목록을 그대로 쓴다, 이 필드 이전과 같다).
    - command_list 없이 보냈거나, 빈 값·여러 번·64자 소문자 16진수가 아니면 거부한다.
    - 지금 목록의 sha256과 다르면 MSG_LIST_CHANGED로 거부한다(목록이 그사이 바뀌었다).

    Returns:
        확인한 sha256 또는 None.

    Raises:
        RecordingRejected: 위 규칙 위반 (고정 한국어 문구).
    """
    values = list(values or [])
    if not values:
        return None
    if custom_list is None:
        raise RecordingRejected(MSG_LIST_SHA_NEEDS_LIST, "command_list 없이 command_list_sha256")
    sha = _single_text(values, MSG_BAD_LIST_SHA, "command_list_sha256")
    if sha is None or not _SHA256_RE.match(sha):
        raise RecordingRejected(MSG_BAD_LIST_SHA, f"command_list_sha256={str(sha)[:70]!r}")
    current = str(custom_list.get("sha256") or "")
    if sha != current:
        raise RecordingRejected(MSG_LIST_CHANGED, f"목록 {custom_list.get('id')!r} 판이 다름: {sha[:12]} != {current[:12]}")
    return sha


def resolve_enroll(profile_values: list[Any] | None, command_values: list[Any] | None,
                   phrase_set: str, command_ids: list[str] | None = None) -> tuple[str | None, str | None]:
    """enroll_profile·enroll_command 폼 필드를 확인한다 (둘 다 선택, ward 묶음 전용).

    - 둘 중 하나라도 보냈는데 phrase_set이 ward가 아니면 거부한다.
    - enroll_command는 enroll_profile과 함께만, 현재 병실 명령 목록의 ID만 받는다. 내 목록(command_ids)을 쓰면 그 목록의 ID만 받는다.
    - 등록 이름은 ^[A-Za-z0-9_-]{1,32}$ (Windows 장치 이름 제외). 경로 문자는 받지 않는다.
    - 빈 값·같은 필드 여러 번·파일 파트는 '보내지 않음'으로 바꾸지 않고 거부한다.

    Args:
        profile_values: 원래 폼의 enroll_profile 값 목록.
        command_values: 원래 폼의 enroll_command 값 목록.
        phrase_set: 확인을 마친 묶음 ID.
        command_ids: 내 목록을 쓸 때 그 목록의 명령 ID들. None이면 기본 병실 명령 목록.

    Returns:
        (enroll_profile 또는 None, enroll_command 또는 None).

    Raises:
        RecordingRejected: 위 규칙 위반 (고정 한국어 문구).
    """
    from services import ward_profiles

    profile_values = list(profile_values or [])
    command_values = list(command_values or [])
    if not profile_values and not command_values:
        return None, None
    if phrase_set != config.PHRASE_SET_WARD:
        raise RecordingRejected(MSG_ENROLL_WARD_ONLY, f"phrase_set={phrase_set!r}")
    if not profile_values:
        raise RecordingRejected(MSG_ENROLL_NEEDS_PROFILE, "enroll_profile 없음")
    profile = _single_text(profile_values, MSG_BAD_ENROLL_PROFILE, "enroll_profile")
    if not ward_profiles.is_valid_profile_id(profile):
        raise RecordingRejected(MSG_BAD_ENROLL_PROFILE, f"enroll_profile={str(profile)[:40]!r}")
    if command_ids is not None:
        command = _single_text(command_values, MSG_BAD_LIST_ENROLL_COMMAND, "enroll_command")
        if command is not None and command not in command_ids:
            raise RecordingRejected(MSG_BAD_LIST_ENROLL_COMMAND, f"enroll_command={command[:40]!r}")
        return profile, command
    command = _single_text(command_values, MSG_BAD_ENROLL_COMMAND, "enroll_command")
    if command is not None:
        commands = ward_profiles.current_commands()
        if commands is None or command not in commands["ids"]:
            raise RecordingRejected(MSG_BAD_ENROLL_COMMAND, f"enroll_command={command[:40]!r}")
    return profile, command


def record_body_limit() -> tuple[float, str]:
    """요청 본문 크기 제한 미들웨어 규칙: (RECORD_MAX_SIZE_MB, 413 고정 문구). 요청마다 설정을 다시 읽는다."""
    return float(config.RECORD_MAX_SIZE_MB), MSG_TOO_LARGE.format(mb=config.RECORD_MAX_SIZE_MB)


async def _save_capped(file: UploadFile, session_id: str) -> tuple[str, int]:
    """이미 받아 둔 업로드 사본을 RECORD_MAX_SIZE_MB+1바이트까지만 서버가 정한 이름으로 옮긴다.

    앞 4KB로 컨테이너를 판정하고 나머지는 1MB 단위로 스레드에서 복사한다(limits.save_upload_capped).
    서버가 '받는' 양의 제한은 이 함수가 아니라 limits.RequestBodyLimitMiddleware(413)가 맡는다.

    Returns:
        (저장 경로, 바이트 수).

    Raises:
        RecordingRejected: 비었음·형식 불일치·크기 초과·저장 실패. 이때 파일은 이미 지워져 있다.
    """
    max_bytes = int(config.RECORD_MAX_SIZE_MB) * 1024 * 1024
    upload_dir = infer_api.TMP_UPLOAD_DIR
    try:
        saved = await limits.save_upload_capped(
            file, max_bytes, detect_recording_container,
            lambda container: os.path.join(upload_dir, f"{session_id}_record.{container}"),
            min_bytes=_MIN_FILE_BYTES, head_bytes=_HEADER_BYTES)
    except limits.UploadRejected as exc:
        message = _REJECT_MESSAGES.get(exc.reason, lambda: MSG_UNREADABLE)()
        raise RecordingRejected(message, exc.detail) from None
    return saved.path, saved.size


class EnrollOriginRejected(Exception):
    """등록 필드가 있는 요청이 화이트리스트 밖 Origin에서 왔다 (403)."""


def origin_allowed(origin: str | None) -> bool:
    """Origin 헤더가 없거나(같은 출처의 일부 요청·채점 도구·점검 스크립트) config.CORS_ORIGINS 안이면 참.

    여러 부분 폼 POST는 CORS 사전 요청 없이 오고, CORS 미들웨어는 허용하지 않은 출처의 요청도 앱까지 보낸다(응답 헤더만 뺀다).
    등록은 서버에 템플릿을 남기므로 같은 브라우저에 열린 다른 사이트가 등록을 만들거나 덮어쓰지 못하게 여기서 막는다.
    """
    try:
        if origin is None:
            return True
        return origin in config.CORS_ORIGINS
    except (AttributeError, TypeError):
        return False


@router.post("/record", response_model=RecordStartResponse)
async def start_record(request: Request,
                       background_tasks: BackgroundTasks,
                       file: UploadFile = File(...),
                       phrase_set: str | None = Form(None),
                       enroll_profile: str | None = Form(None),
                       enroll_command: str | None = Form(None),
                       command_list: str | None = Form(None),
                       command_list_sha256: str | None = Form(None)) -> RecordStartResponse:
    """웹캠 녹화를 받아 발화 전체 인식 Job을 등록한다 (결과 즉시 반환 금지).

    Args:
        request: 원래 폼에서 phrase_set·등록 필드를 확인하는 데 쓴다(빈 값·중복 필드 구분).
        background_tasks: FastAPI 백그라운드 태스크 큐.
        file: 멀티파트 'file' 필드의 녹화 파일.
        phrase_set: 멀티파트 'phrase_set' 필드(선택). 'sentences'(기본) 또는 'ward'. API 문서용이며,
            FastAPI가 빈 값을 None으로 바꾸므로 판정은 원래 폼 값(resolve_phrase_set)으로 한다.
        enroll_profile: (선택, ward 전용) 내 입 모양 등록 이름. 보내면 결과 results[0]에 enroll이 붙는다.
        enroll_command: (선택, ward 전용, enroll_profile 필요) 이 녹화를 그 명령의 등록 템플릿으로 저장한다.
            없으면 등록 템플릿과 거리 판정만 한다. 두 필드도 원래 폼 값(resolve_enroll)으로 판정한다.
        command_list: (선택, ward 전용) 내 목록 ID. 그 목록을 후보로 쓰고 결과 phrase에 list가 붙는다. 원래 폼 값
            (resolve_command_list)으로 판정한다.
        command_list_sha256: (선택, command_list와 함께) 화면이 불러온 목록의 sha256. 지금 목록과 다르면 400이다.

    Returns:
        {job_id, session_id, status:'processing', duration_sec}.

    Raises:
        HTTPException: 접수 거부 400(고정 문구, 알 수 없는·빈·중복 phrase_set, ward가 아닌 묶음의 등록 필드·내 목록, 이름 없는
            등록 명령, 규칙에 맞지 않는 등록 이름·명령·목록 ID·목록 지문, 없거나 오류가 있거나 그사이 바뀐 내 목록 포함),
            허용하지 않은 Origin의 등록 요청 403, compact 백엔드 503, 같은 세션 처리 중 409.
    """
    if infer_api.MODEL_BACKEND != "olkavs":
        raise HTTPException(status_code=503, detail=MSG_BACKEND_UNSUPPORTED)
    session_id = str(uuid.uuid4())
    path: str | None = None
    try:
        # 파일을 저장하기 전에 거른다. phrase_set 인자 대신 원래 폼 값으로 판정한다(빈 값이 기본값으로 바뀌지 않게).
        phrase_set = resolve_phrase_set(await _form_phrase_set_values(request))
        list_values = await _form_values(request, "command_list")
        custom_list = (await asyncio.to_thread(resolve_command_list, list_values, phrase_set)) if list_values else None
        sha_values = await _form_values(request, "command_list_sha256")
        if sha_values:
            resolve_command_list_sha(sha_values, custom_list)
        command_ids = [c["id"] for c in custom_list["commands"]] if custom_list is not None else None
        enroll_profile, enroll_command = resolve_enroll(await _form_values(request, "enroll_profile"),
                                                        await _form_values(request, "enroll_command"), phrase_set,
                                                        command_ids)
        if enroll_profile is not None and not origin_allowed(request.headers.get("origin")):
            raise EnrollOriginRejected()
        path, size = await _save_capped(file, session_id)
        duration = await asyncio.to_thread(measure_recording_seconds, path, config.RECORD_MAX_SECONDS,
                                           config.RECORD_MAX_FRAMES, config.RECORD_PROBE_TIMEOUT_SEC)
        if duration < config.RECORD_MIN_SECONDS:
            raise RecordingRejected(MSG_TOO_SHORT.format(sec=config.RECORD_MIN_SECONDS), f"{duration:.3f}초")
        if not (duration < config.RECORD_MAX_SECONDS):
            raise RecordingRejected(MSG_TOO_LONG.format(sec=config.RECORD_MAX_SECONDS), f"{duration:.3f}초")
    except EnrollOriginRejected:
        _safe_remove(path)
        logger.warning("등록 요청 거부: 허용하지 않은 Origin %r", request.headers.get("origin"))
        raise HTTPException(status_code=403, detail=MSG_ENROLL_ORIGIN) from None
    except RecordingRejected as exc:
        _safe_remove(path)
        logger.info("녹화 접수 거부: %s (%s)", exc.user_message, exc.detail)
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    except Exception as exc:  # noqa: BLE001 - 예상 밖 오류도 경로·원문 없이 400
        _safe_remove(path)
        logger.exception("녹화 접수 중 예기치 않은 오류: %s", type(exc).__name__)
        raise HTTPException(status_code=400, detail=MSG_UNREADABLE) from None
    finally:
        try:
            await file.close()
        except Exception:  # noqa: BLE001 - 업로드 임시 파일 정리 실패는 무시
            pass
    logger.info("녹화 접수: %s %d바이트 %.3f초 묶음=%s%s%s", os.path.basename(path), size, duration, phrase_set,
                f" 내목록={custom_list['id']}" if custom_list is not None else "",
                f" 등록={enroll_profile}/{enroll_command or 'match'}" if enroll_profile else "")
    enroll_kwargs: dict[str, Any] = ({"enroll_profile": enroll_profile, "enroll_command": enroll_command}
                                     if enroll_profile is not None else {})
    if custom_list is not None:
        enroll_kwargs["command_list"] = custom_list  # 녹화 요청 때 검사한 목록 그대로 Job에 넘긴다
    try:
        start = infer_api.start_background_job(session_id, path, background_tasks,
                                               mode="record", video_seconds=duration, phrase_set=phrase_set,
                                               **enroll_kwargs)
    except Exception:
        _safe_remove(path)
        raise
    return RecordStartResponse(job_id=start.job_id, session_id=start.session_id,
                               status=start.status, duration_sec=round(float(duration), 3))
