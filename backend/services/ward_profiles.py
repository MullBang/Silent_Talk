"""병실 무음 도우미 '내 입 모양 등록' 저장소와 판정 (서버 프로세스, numpy만).

등록은 사람마다 병실 명령 20개를 소리 내어 한 번씩 찍어, 인식 모델의 CTC 직전 은닉 특징(T x 256)을 명령별 템플릿으로 저장한다.
이후 ward 녹화는 같은 특징을 템플릿 20개와 DTW 거리(services/ward_templates.py)로 비교한다. 오프라인 도구
scripts/ward_enroll.py·scripts/ward_tools/enroll.py와 같은 특징·거리·템플릿 방식(voiced)이다.

저장 구조 (config.WARD_PROFILE_DIR, 저장소 밖·gitignore):
  <폴더>/<등록 이름>/profile.json   {schema, profile, created, updated, commands_version, commands_sha256, checkpoint_sha256,
                                    feature, distance, template_mode, templates:{명령ID:{created, frames, start_ms, end_ms,
                                    text_top, text_top_matches}}}
  <폴더>/<등록 이름>/<명령ID>.npz    hidden: float32 (T, 256)

- 등록 이름은 영문·숫자·_·- 1~32자만 받는다(경로 문자 차단). Windows 장치 이름(CON·NUL·COM1 등)도 거부한다.
  Windows 폴더는 대소문자를 가리지 않으므로 대소문자만 다른 등록이 이미 있으면 충돌(ProfileConflict)로 거부한다.
- 파일은 임시 파일에 쓴 뒤 os.replace로 바꾼다. 같은 등록은 잠금(threading.Lock) 하나로 차례대로 고친다.
- 명령마다 템플릿은 하나다(다시 찍으면 바꾼다). 명령 목록(commands.json SHA-256)이나 인식 가중치 SHA-256이 등록 때와
  다르면 'stale'이다. stale 등록에 새 템플릿을 저장하면 이전 템플릿을 모두 지우고 새로 시작한다(섞어 쓰지 않는다).
- 상태: missing(없음) · stale(목록·가중치가 바뀜) · incomplete(20개 미만) · ok(20/20).
- 판정(match)은 상태 ok이고 녹화가 지정 문장 결과가 있는 창 하나일 때만 거리를 계산한다.
  within = 1등 거리 ≤ D(config.OLKAVS_WARD_ENROLL_MAX_DISTANCE, None이면 항상 false), agree = 등록 1등 == 글자 1등,
  accepted = within and agree. 긴급 여부 해석과 화면 판정표는 화면(wardResult.js)이 scripts/ward_tools/metrics.decide()와 같게 한다.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import config  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA = "ward-profile-v1"
# scripts/ward_tools/common.FEATURE_NAME·DISTANCE_NAME과 같은 이름 (오프라인 등록 평가와 같은 특징·거리)
FEATURE_NAME = "olkavs_medium_hidden_256"
DISTANCE_NAME = "dtw_cosine_unit_steps_over_path_length"
TEMPLATE_MODE = "voiced"
HIDDEN_DIM = 256

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_INCOMPLETE = "incomplete"
STATUS_STALE = "stale"
STATUS_NO_PHRASE = "no_phrase"
STATUS_ERROR = "error"

REASON_MULTI_WINDOW = "multi_window"
REASON_PARTIAL_DETECTION = "partial_detection"
REASON_NO_SPEECH = "no_speech"
REASON_NO_PHRASE = "no_phrase"
REASON_NO_FEATURES = "no_features"
REASON_ERROR = "error"

PROFILE_FILE = "profile.json"
_PROFILE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                               *(f"LPT{i}" for i in range(10))})

MSG_BAD_PROFILE = "등록 이름은 영문·숫자·밑줄(_)·하이픈(-)으로 1~32자만 쓸 수 있습니다."
MSG_CONFLICT = "대소문자만 다른 같은 이름의 등록이 이미 있습니다. 다른 이름을 쓰거나 그 이름을 그대로 쓰세요."
MSG_NOT_FOUND = "등록을 찾을 수 없습니다."
MSG_UNREADABLE = "등록 파일을 읽지 못했습니다. 등록을 지우고 다시 만들어 주세요."
MSG_STORAGE = "등록 파일을 저장하지 못했습니다. 저장 폴더를 확인해 주세요."


class ProfileError(ValueError):
    """등록 저장소 오류. user_message는 사용자에게 보내도 되는 고정 문구다(경로·예외 원문 없음)."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail


class InvalidProfileId(ProfileError):
    """규칙에 맞지 않는 등록 이름."""


class ProfileConflict(ProfileError):
    """대소문자만 다른 등록이 이미 있음 (Windows 폴더 이름 충돌)."""


class ProfileNotFound(ProfileError):
    """등록이 없음."""


# ---------------------------------------------------------------------------
# 이름·명령 목록
# ---------------------------------------------------------------------------

def is_valid_profile_id(value: Any) -> bool:
    """등록 이름 규칙 ^[A-Za-z0-9_-]{1,32}$ (Windows 장치 이름 제외)."""
    try:
        return (isinstance(value, str) and _PROFILE_ID_RE.fullmatch(value) is not None
                and value.upper() not in _WINDOWS_RESERVED)
    except TypeError:
        return False


def validate_profile_id(value: Any) -> str:
    """등록 이름을 확인해 그대로 돌려준다.

    Raises:
        InvalidProfileId: 규칙에 맞지 않을 때 (보낸 값을 문구에 넣지 않는다).
    """
    if not is_valid_profile_id(value):
        shown = value[:40] if isinstance(value, str) else type(value).__name__
        raise InvalidProfileId(MSG_BAD_PROFILE, f"profile={shown!r}")
    return value


def current_commands(path: str | None = None) -> dict[str, Any] | None:
    """현재 병실 명령 목록 {version, sha256, ids}. 읽지 못하면 None.

    sha256은 파일 바이트 전체의 SHA-256이다(scripts/ward_tools/common.load_commands()['sha256']와 같은 계산).

    Args:
        path: 목록 파일. None이면 config.OLKAVS_WARD_PHRASE_FILE(부를 때의 값).
    """
    path = config.OLKAVS_WARD_PHRASE_FILE if path is None else path
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
        data = json.loads(raw.decode("utf-8"))
        ids = [str(p["id"]) for p in data["fixed"]]
        version = data.get("version")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("병실 명령 목록을 읽지 못했습니다: %s", type(exc).__name__)
        return None
    if len(ids) < 2 or len(set(ids)) != len(ids):
        return None
    return {"version": version if isinstance(version, str) else None,
            "sha256": hashlib.sha256(raw).hexdigest(), "ids": ids}


def _now() -> str:
    """UTC ISO 시각 (초 단위)."""
    try:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    except (OverflowError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# 파일 쓰기·읽기
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: bytes) -> None:
    """같은 폴더의 임시 파일에 쓴 뒤 os.replace로 바꾼다. 실패하면 임시 파일을 지운다.

    Raises:
        OSError: 쓰기 실패.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                os.remove(tmp)
        except OSError:
            logger.warning("등록 임시 파일 삭제 실패: %s", tmp.name)


def check_hidden(hidden: Any) -> Any:
    """은닉 특징 배열 확인: 2차원 (T ≥ 1, 256), 유한한 값. float32로 돌려준다(이미 float32면 값이 그대로다).

    Raises:
        ValueError: 모양·값이 틀렸을 때.
    """
    import numpy as np

    try:
        arr = np.asarray(hidden)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != HIDDEN_DIM:
            raise ValueError(f"은닉 특징 모양이 (T, {HIDDEN_DIM})이 아닙니다: {arr.shape}")
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"은닉 특징이 실수가 아닙니다: {arr.dtype}")
        arr = arr.astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise ValueError("은닉 특징에 비정상 값이 있습니다")
        return arr
    except (TypeError, AttributeError) as exc:
        raise ValueError(f"은닉 특징 확인 실패: {exc}") from exc


def load_hidden_npz(path: str | os.PathLike[str]) -> Any:
    """npz의 hidden 배열을 메모리로 읽는다(파일을 닫은 뒤 돌려준다). allow_pickle은 쓰지 않는다.

    파일을 직접 열어 np.load에 넘기므로 빈 파일·깨진 zip이어도 파일 손잡이가 늘 닫힌다(Windows에서 바로 지울 수 있다).

    Raises:
        ValueError: 파일·배열이 틀렸을 때(빈 파일 EOFError, 깨진 zip BadZipFile 포함).
    """
    import zipfile

    import numpy as np

    try:
        with open(path, "rb") as handle, np.load(handle, allow_pickle=False) as data:
            hidden = np.array(data["hidden"], copy=True)
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile, AttributeError, TypeError) as exc:
        # AttributeError·TypeError: npz가 아닌 단일 배열(.npy) 내용이면 np.load가 with를 지원하지 않는 배열을 돌려준다.
        raise ValueError(f"특징 파일을 읽지 못했습니다: {type(exc).__name__}") from exc
    return check_hidden(hidden)


def _npz_bytes(hidden: Any) -> bytes:
    """np.savez(hidden=...) 결과 바이트."""
    import numpy as np

    try:
        buffer = io.BytesIO()
        np.savez(buffer, hidden=hidden)
        return buffer.getvalue()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"특징 직렬화 실패: {exc}") from exc


# ---------------------------------------------------------------------------
# 저장소
# ---------------------------------------------------------------------------

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(root: Path, profile_id: str) -> threading.Lock:
    """등록 하나의 잠금 (폴더·이름은 대소문자를 가리지 않고 같은 잠금을 쓴다)."""
    key = f"{os.path.normcase(str(root))}|{profile_id.lower()}"
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def profile_status(data: dict[str, Any] | None, count: int, commands: dict[str, Any] | None,
                   checkpoint: str | None) -> str:
    """등록 상태 (missing → stale → incomplete → ok 순서로 본다).

    Args:
        data: profile.json 내용 또는 None(등록 없음).
        count: 현재 명령 목록에 있고 특징 파일도 있는 템플릿 수.
        commands: current_commands() 결과. None이면 목록을 확인할 수 없어 stale로 본다.
        checkpoint: 지금 인식 가중치 SHA-256. None이면(워커 기동 전) 가중치는 비교하지 않는다.
    """
    try:
        if data is None:
            return STATUS_MISSING
        if (commands is None or data.get("commands_sha256") != commands["sha256"]
                or (checkpoint is not None and data.get("checkpoint_sha256") != checkpoint)
                or data.get("feature") != FEATURE_NAME or data.get("distance") != DISTANCE_NAME
                or data.get("template_mode") != TEMPLATE_MODE):
            return STATUS_STALE
        return STATUS_OK if count >= len(commands["ids"]) else STATUS_INCOMPLETE
    except (AttributeError, KeyError, TypeError) as exc:
        logger.warning("등록 상태 판정 실패: %s", exc)
        return STATUS_STALE


class ProfileStore:
    """등록 폴더 하나(config.WARD_PROFILE_DIR)의 읽기·쓰기."""

    def __init__(self, root: str | os.PathLike[str] | None = None):
        """root가 None이면 config.WARD_PROFILE_DIR(만들 때의 값)을 쓴다."""
        self.root = Path(root if root is not None else config.WARD_PROFILE_DIR)

    # -- 찾기·읽기 ---------------------------------------------------------------
    def _entries(self) -> list[str]:
        """저장 폴더의 항목 이름 (폴더가 없으면 빈 목록)."""
        try:
            return sorted(os.listdir(self.root)) if self.root.is_dir() else []
        except OSError as exc:
            raise ProfileError(MSG_UNREADABLE, f"등록 폴더 목록 실패: {exc}") from exc

    def _find(self, profile_id: str) -> Path | None:
        """정확히 같은 이름의 등록 폴더. 없으면 None.

        Raises:
            ProfileConflict: 대소문자만 다른 항목이 있을 때 (Windows에서는 같은 폴더라 덮어쓰지 않는다).
        """
        names = self._entries()
        if profile_id in names:
            path = self.root / profile_id
            return path if path.is_dir() else None
        if any(name.lower() == profile_id.lower() for name in names):
            raise ProfileConflict(MSG_CONFLICT, f"대소문자만 다른 등록: {profile_id!r}")
        return None

    @staticmethod
    def _read(path: Path, profile_id: str) -> dict[str, Any] | None:
        """profile.json을 읽는다. 없으면 None.

        Raises:
            ProfileError: 깨졌거나 다른 등록의 파일일 때.
        """
        file = path / PROFILE_FILE
        try:
            if not file.is_file():
                return None
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProfileError(MSG_UNREADABLE, f"profile.json 읽기 실패: {type(exc).__name__}") from exc
        if (not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("profile") != profile_id
                or not isinstance(data.get("templates"), dict)):
            raise ProfileError(MSG_UNREADABLE, f"profile.json 형식이 다릅니다: {profile_id!r}")
        return data

    @staticmethod
    def _count(path: Path, data: dict[str, Any] | None, commands: dict[str, Any] | None) -> int:
        """현재 명령 목록에 있고 특징 파일도 있는 템플릿 수."""
        if data is None or commands is None:
            return 0
        try:
            return sum(1 for cid in commands["ids"]
                       if cid in data["templates"] and (path / f"{cid}.npz").is_file())
        except (OSError, KeyError, TypeError):
            return 0

    def _snapshot(self, profile_id: str, commands: dict[str, Any] | None,
                  checkpoint: str | None) -> tuple[Path | None, dict[str, Any] | None, int, str]:
        """(폴더, profile.json, 템플릿 수, 상태). 잠금 없이 부르지 않는다."""
        path = self._find(profile_id)
        data = self._read(path, profile_id) if path is not None else None
        count = self._count(path, data, commands) if path is not None else 0
        return path, data, count, profile_status(data, count, commands, checkpoint)

    def summary(self, profile_id: str, checkpoint: str | None = None) -> dict[str, Any]:
        """{profile, count, complete, status, updated}. 없는 등록은 status 'missing'.

        Raises:
            InvalidProfileId / ProfileConflict / ProfileError.
        """
        validate_profile_id(profile_id)
        commands = current_commands()
        total = len(commands["ids"]) if commands else 0
        with _lock_for(self.root, profile_id):
            _path, data, count, status = self._snapshot(profile_id, commands, checkpoint)
        return {"profile": profile_id, "count": count, "complete": bool(total) and count >= total,
                "status": status, "updated": (data or {}).get("updated")}

    def detail(self, profile_id: str, checkpoint: str | None = None) -> dict[str, Any]:
        """{profile, count, complete, status, templates:{명령ID:{created, text_top, text_top_matches}}}.

        templates에는 현재 명령 목록에 있고 특징 파일이 있는 템플릿만 넣는다. 없는 등록은 status 'missing'·빈 templates.
        """
        validate_profile_id(profile_id)
        commands = current_commands()
        total = len(commands["ids"]) if commands else 0
        with _lock_for(self.root, profile_id):
            path, data, count, status = self._snapshot(profile_id, commands, checkpoint)
            templates: dict[str, Any] = {}
            if path is not None and data is not None and commands is not None:
                for cid in commands["ids"]:
                    entry = data["templates"].get(cid)
                    if isinstance(entry, dict) and (path / f"{cid}.npz").is_file():
                        templates[cid] = {"created": entry.get("created"), "text_top": entry.get("text_top"),
                                          "text_top_matches": bool(entry.get("text_top_matches"))}
        return {"profile": profile_id, "count": count, "complete": bool(total) and count >= total,
                "status": status, "templates": templates}

    def list_profiles(self, checkpoint: str | None = None) -> list[dict[str, Any]]:
        """저장된 등록 요약 목록 (이름순). 규칙에 맞지 않는 이름·profile.json이 없거나 깨진 폴더는 건너뛴다."""
        out = []
        for name in self._entries():
            if not is_valid_profile_id(name) or not (self.root / name / PROFILE_FILE).is_file():
                continue
            try:
                out.append(self.summary(name, checkpoint))
            except ProfileError as exc:
                logger.warning("등록 목록에서 건너뜀 (%s): %s", name, exc.detail or exc)
        return out

    def delete(self, profile_id: str) -> None:
        """등록 폴더를 통째로 지운다.

        Raises:
            ProfileNotFound: 없을 때. ProfileConflict: 대소문자만 다른 등록만 있을 때(그 등록은 지우지 않는다).
            ProfileError: 지우지 못했을 때.
        """
        validate_profile_id(profile_id)
        with _lock_for(self.root, profile_id):
            path = self._find(profile_id)
            if path is None:
                raise ProfileNotFound(MSG_NOT_FOUND, f"없는 등록: {profile_id!r}")
            try:
                shutil.rmtree(path)
            except OSError as exc:
                raise ProfileError(MSG_STORAGE, f"등록 삭제 실패: {exc}") from exc
            if path.exists():
                raise ProfileError(MSG_STORAGE, f"등록 폴더가 남았습니다: {profile_id!r}")
        logger.info("등록 삭제: %s", profile_id)

    # -- 저장 ----------------------------------------------------------------------
    def store_template(self, profile_id: str, command_id: str, hidden: Any, meta: dict[str, Any],
                       commands: dict[str, Any], checkpoint: str | None) -> int:
        """명령 하나의 템플릿을 저장한다(있으면 바꾼다). 등록이 없으면 만들고, stale이면 이전 템플릿을 지우고 새로 시작한다.

        Args:
            profile_id: 등록 이름.
            command_id: 현재 명령 목록의 ID.
            hidden: 은닉 특징 (T, 256).
            meta: {frames, start_ms, end_ms, text_top, text_top_matches}.
            commands: current_commands() 결과.
            checkpoint: 지금 인식 가중치 SHA-256.

        Returns:
            저장 뒤 템플릿 수.

        Raises:
            ProfileError(하위 클래스 포함), ValueError: 이름·명령·특징이 틀렸거나 쓰지 못했을 때.
        """
        validate_profile_id(profile_id)
        if command_id not in commands["ids"]:
            raise ValueError(f"현재 명령 목록에 없는 명령: {command_id!r}")
        hidden = check_hidden(hidden)
        with _lock_for(self.root, profile_id):
            path, data, _count, status = self._snapshot(profile_id, commands, checkpoint)
            now = _now()
            try:
                if path is None:
                    path = self.root / profile_id
                    path.mkdir(parents=True, exist_ok=True)
                if data is None or status == STATUS_STALE:
                    if data is not None:
                        logger.warning("등록 %s: 명령 목록·가중치가 등록 때와 달라 이전 템플릿을 지우고 새로 시작합니다",
                                       profile_id)
                        for name in os.listdir(path):
                            if name.lower().endswith(".npz"):
                                os.remove(path / name)
                    data = {"schema": SCHEMA, "profile": profile_id, "created": now, "updated": now,
                            "commands_version": commands.get("version"), "commands_sha256": commands["sha256"],
                            "checkpoint_sha256": checkpoint, "feature": FEATURE_NAME, "distance": DISTANCE_NAME,
                            "template_mode": TEMPLATE_MODE, "templates": {}}
                _atomic_write(path / f"{command_id}.npz", _npz_bytes(hidden))
                data["templates"][command_id] = {
                    "created": now, "frames": int(hidden.shape[0]),
                    "start_ms": meta.get("start_ms"), "end_ms": meta.get("end_ms"),
                    "text_top": meta.get("text_top"), "text_top_matches": bool(meta.get("text_top_matches"))}
                data["updated"] = now
                _atomic_write(path / PROFILE_FILE,
                              json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
            except OSError as exc:
                raise ProfileError(MSG_STORAGE, f"등록 저장 실패: {exc}") from exc
            count = self._count(path, data, commands)
        logger.info("등록 저장: %s %s (%d/%d)", profile_id, command_id, count, len(commands["ids"]))
        return count

    def templates_for_match(self, profile_id: str, commands: dict[str, Any] | None,
                            checkpoint: str | None) -> tuple[str, int, dict[str, Any]]:
        """(상태, 템플릿 수, 명령ID → 템플릿 배열). 배열은 상태 ok일 때만 읽는다(잠금 안에서 읽어 저장 중간을 보지 않는다).

        Raises:
            ProfileError / ValueError: 등록 파일을 읽지 못했을 때.
        """
        validate_profile_id(profile_id)
        with _lock_for(self.root, profile_id):
            path, _data, count, status = self._snapshot(profile_id, commands, checkpoint)
            if status != STATUS_OK or path is None or commands is None:
                return status, count, {}
            arrays = {cid: load_hidden_npz(path / f"{cid}.npz") for cid in commands["ids"]}
        return status, count, arrays


def get_store() -> ProfileStore:
    """현재 config.WARD_PROFILE_DIR의 저장소 (부를 때마다 설정 값을 다시 읽는다)."""
    return ProfileStore(config.WARD_PROFILE_DIR)


_WRITABLE_CACHE: dict[str, tuple[float, bool]] = {}
_WRITABLE_TTL_SEC = 30.0


def profile_dir_writable(root: str | os.PathLike[str] | None = None) -> bool:
    """저장 폴더에 실제로 파일을 쓸 수 있는지 (/health). 폴더가 아직 없으면 가장 가까운 상위 폴더에 써 본다.

    폴더를 새로 만들지 않는다. 시험 파일은 바로 지운다(TemporaryFile). /health를 자주 불러도 같은 폴더는 30초에
    한 번만 써 본다.
    """
    try:
        path = Path(root if root is not None else config.WARD_PROFILE_DIR).resolve()
        key = os.path.normcase(str(path))
        cached = _WRITABLE_CACHE.get(key)
        now = time.monotonic()
        if cached is not None and now - cached[0] < _WRITABLE_TTL_SEC:
            return cached[1]
        probe = path
        while not probe.exists():
            if probe.parent == probe:
                break
            probe = probe.parent
        writable = probe.is_dir()
        if writable:
            try:
                with tempfile.TemporaryFile(dir=probe, prefix=".ward_probe_"):
                    pass
            except OSError:
                writable = False
        _WRITABLE_CACHE[key] = (now, writable)
        return writable
    except (OSError, ValueError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# 녹화 결과 → 등록 결과 (results[0].enroll)
# ---------------------------------------------------------------------------

def segments_reason(segments: list[dict[str, Any]]) -> str | None:
    """입술 구간으로 본, 등록에 쓸 수 없는 사유. 지정 문장 결과가 붙는 창 하나면 None.

    모든 구간이 무발화 → no_speech, 검출 실패 구간이 있음 → partial_detection, 창이 여러 개 → multi_window.
    """
    try:
        if not segments:
            return REASON_PARTIAL_DETECTION
        statuses = [s.get("status") for s in segments]
        if all(st == "no_speech" for st in statuses):
            return REASON_NO_SPEECH
        if any(not ((s.get("status") == "ok" and s.get("npz")) or s.get("status") == "no_speech") for s in segments):
            return REASON_PARTIAL_DETECTION
        if len(segments) > 1:
            return REASON_MULTI_WINDOW
        return None
    except (AttributeError, TypeError):
        return REASON_ERROR


def store_result(store: ProfileStore, profile_id: str, command_id: str, reason: str | None,
                 phrase: dict[str, Any] | None, hidden: Any, meta: dict[str, Any],
                 checkpoint: str | None) -> dict[str, Any]:
    """저장 모드 결과 {mode:'store', profile, command, stored, reason, text_top, text_top_matches, count, complete}.

    창 하나·지정 문장 결과·은닉 특징이 모두 있을 때만 저장한다. 글자 1등이 찍은 명령과 달라도 저장한다(화면이 경고하고
    다시 찍기를 권한다). 저장소 오류는 reason 'error'로 남기고 녹화 결과는 그대로 돌려준다.
    """
    text_top = str(phrase["id"]) if isinstance(phrase, dict) and phrase.get("id") is not None else None
    out: dict[str, Any] = {"mode": "store", "profile": profile_id, "command": command_id, "stored": False,
                           "reason": None, "text_top": text_top, "text_top_matches": text_top == command_id,
                           "count": 0, "complete": False}
    commands = current_commands()
    total = len(commands["ids"]) if commands else 0
    try:
        if reason is None and not isinstance(phrase, dict):
            reason = REASON_NO_PHRASE
        if reason is None and hidden is None:
            reason = REASON_NO_FEATURES
        if reason is None and commands is None:
            reason = REASON_ERROR
        if reason is None:
            count = store.store_template(profile_id, command_id, hidden,
                                         {**meta, "text_top": text_top, "text_top_matches": text_top == command_id},
                                         commands, checkpoint)
            out.update(stored=True, count=count, complete=bool(total) and count >= total)
            return out
        out["reason"] = reason
    except (ProfileError, ValueError, KeyError, TypeError) as exc:
        logger.error("등록 저장 실패 (%s %s): %s", profile_id, command_id, getattr(exc, "detail", "") or exc)
        out.update(stored=False, reason=REASON_ERROR)
    try:
        current = store.summary(profile_id, checkpoint)
        out.update(count=current["count"], complete=current["complete"])
    except (ProfileError, ValueError) as exc:
        logger.warning("등록 수 확인 실패 (%s): %s", profile_id, getattr(exc, "detail", "") or exc)
    return out


def match_result(store: ProfileStore, profile_id: str, reason: str | None, phrase: dict[str, Any] | None,
                 hidden: Any, checkpoint: str | None, max_distance: float | None) -> dict[str, Any]:
    """판정 모드 결과 {mode:'match', profile, status, count, top_id, distance, runner_up_id, runner_up_distance,
    distances, max_distance, within, agree, accepted}.

    상태 순서: 등록 상태(missing·stale·incomplete) → 지정 문장 결과가 붙는 창 하나가 아님(no_phrase) → 거리 계산(ok).
    은닉 특징이 없거나 계산이 실패하면 error. ok가 아니면 거리 항목은 null이고 within·agree·accepted는 false다.
    """
    from services import ward_templates

    out: dict[str, Any] = {"mode": "match", "profile": profile_id, "status": STATUS_ERROR, "count": 0,
                           "top_id": None, "distance": None, "runner_up_id": None, "runner_up_distance": None,
                           "distances": None, "max_distance": max_distance, "within": False, "agree": False,
                           "accepted": False}
    try:
        commands = current_commands()
        status, count, arrays = store.templates_for_match(profile_id, commands, checkpoint)
        out["count"] = count
        if status != STATUS_OK:
            out["status"] = status
            return out
        if reason is not None or not isinstance(phrase, dict):
            out["status"] = STATUS_NO_PHRASE
            return out
        if hidden is None:
            logger.error("등록 판정: 은닉 특징이 없습니다 (%s)", profile_id)
            return out
        ids = list(commands["ids"])
        test = ward_templates.unit_rows(check_hidden(hidden))
        raw = ward_templates.template_distances(test, {cid: [ward_templates.unit_rows(arrays[cid])] for cid in ids})
        if any(raw.get(cid) is None for cid in ids):
            raise ValueError("거리를 계산하지 못한 명령이 있습니다")
        distances = {cid: float(raw[cid]) for cid in ids}
        order = sorted(ids, key=lambda cid: (distances[cid], ids.index(cid)))
        top, second = order[0], order[1]
        within = max_distance is not None and distances[top] <= float(max_distance)
        agree = top == str(phrase.get("id"))
        out.update(status=STATUS_OK, top_id=top, distance=distances[top], runner_up_id=second,
                   runner_up_distance=distances[second], distances=distances, within=bool(within),
                   agree=bool(agree), accepted=bool(within and agree))
        return out
    except (ProfileError, ValueError, KeyError, TypeError, IndexError) as exc:
        logger.error("등록 판정 실패 (%s): %s", profile_id, getattr(exc, "detail", "") or exc)
        out.update(status=STATUS_ERROR, top_id=None, distance=None, runner_up_id=None, runner_up_distance=None,
                   distances=None, within=False, agree=False, accepted=False)
        return out
