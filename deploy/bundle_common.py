"""시연 노트북 묶음(설치·실행·점검·묶음 만들기)이 함께 쓰는 경로·상수·작은 도구.

표준 라이브러리만 쓴다. 설치 전(시스템 파이썬)과 설치 후(.venv-server) 어느 쪽에서도 import 할 수 있어야 한다.
경로는 이 파일 위치(deploy/) 기준이므로 개발 PC 저장소와 노트북 묶음 폴더에서 똑같이 동작한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from git_shim import read_head_commit  # noqa: E402  (같은 폴더의 표준 라이브러리 전용 모듈)

ROOT: Path = Path(__file__).resolve().parents[1]
"""묶음 최상위 폴더(개발 PC에서는 저장소 최상위)."""

DEPLOY: Path = ROOT / "deploy"
RUNTIME: Path = DEPLOY / "runtime"
"""노트북에서 생기는 설치 기록·로그·프로세스 기록 폴더 (저장소에서는 .gitignore 대상)."""
SAMPLES: Path = DEPLOY / "samples"
"""점검용 예시 영상 폴더 (묶음 만들 때 --sample-clips로만 채운다. 저장소에는 넣지 않는다)."""
LOG_DIR: Path = RUNTIME / "logs"
GITSHIM_DIR: Path = RUNTIME / "gitshim"
INSTALL_STATE: Path = RUNTIME / "install_state.json"
PIDS_FILE: Path = RUNTIME / "pids.json"
MANIFEST_NAME: str = "bundle_manifest.json"

BACKEND: Path = ROOT / "backend"
FRONTEND_DIST: Path = ROOT / "frontend" / "dist"
CORE_SCRIPT: Path = ROOT / "scripts" / "olkavs_transcribe.py"
WORKER_SCRIPT: Path = BACKEND / "services" / "olkavs_worker.py"
OLKAVS: Path = ROOT / "experiments" / "olkavs_official"
UPSTREAM: Path = OLKAVS / "upstream"
WEIGHTS: Path = OLKAVS / "weights" / "multiview_lip.pt"
WARD_COMMANDS: Path = ROOT / "frontend" / "src" / "ward" / "commands.json"

LOCK_SERVER: Path = DEPLOY / "requirements-server-lock.txt"
LOCK_ROI: Path = OLKAVS / "requirements-lock-roi.txt"
LOCK_INFER: Path = OLKAVS / "requirements-lock.txt"

VENV_SERVER: str = ".venv-server"
VENV_ROI: str = ".venv-olkavs-roi"
VENV_INFER: str = ".venv-olkavs"

BACKEND_PORT: int = 8000
FRONTEND_PORT: int = 5173
HEALTH_URL: str = f"http://127.0.0.1:{BACKEND_PORT}/health"
API_URL: str = f"http://127.0.0.1:{BACKEND_PORT}/api"
FRONT_URL: str = f"http://127.0.0.1:{FRONTEND_PORT}/"
APP_ORIGIN: str = f"http://localhost:{FRONTEND_PORT}"
"""두 창은 반드시 이 주소로 연다. BroadcastChannel은 주소(scheme·host·port)가 같은 창끼리만 통한다."""
WARD_URL: str = f"{APP_ORIGIN}/#ward"
NURSE_URL: str = f"{APP_ORIGIN}/#nurse"

REQUIRED_PYTHON: tuple[int, int] = (3, 11)
PYTHON_DOWNLOAD_URL: str = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
VC_REDIST_URL: str = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
"""Microsoft Visual C++ 2015-2022 재배포 패키지(x64). torch·mediapipe가 MSVCP140.dll을 쓰는데 파이썬에는 들어 있지 않다."""
MIN_MSVCP_VERSION: tuple[int, int] = (14, 40)
"""이보다 오래된 MSVCP140.dll은 최신 컴파일러로 만든 torch에서 'DLL 초기화 실패(WinError 1114)'를 낼 수 있다(주의만)."""
PYTORCH_INDEX: str = "https://download.pytorch.org/whl/{flavor}"
CUDA_FLAVOR: str = "cu128"
CPU_FLAVOR: str = "cpu"
MIN_CUDA_VERSION: tuple[int, int] = (12, 8)
MIN_CUDA_COMPAT: tuple[int, int] = (12, 0)
"""CUDA 12.0~12.7 드라이버도 CUDA 12 하위 호환으로 cu128 torch가 도는 경우가 많다. 설치 후 실제 GPU 계산 시험이 실패하면
CPU판으로 자동으로 바꾼다."""
MIN_WINDOWS_DRIVER: tuple[int, int] = (570, 65)
"""CUDA 12.8을 지원하는 Windows NVIDIA 드라이버 최소 버전 (nvidia-smi가 CUDA 버전을 안 보여 줄 때 쓴다)."""
MIN_WINDOWS_DRIVER_COMPAT: tuple[int, int] = (528, 33)
"""CUDA 12 하위 호환이 되는 Windows 드라이버 최소 버전."""
MIN_COMPUTE_CAP: tuple[int, int] = (7, 0)
"""torch 2.10.0+cu128이 지원하는 가장 오래된 그래픽카드 세대(sm_70, Volta). 개발 PC에서 torch.cuda.get_arch_list()로
확인: sm_70·75·80·86·90·100·120. GTX 10xx(Pascal, 6.1)는 지원하지 않는다."""
SPACE_NEED_GB: dict[str, dict[str, float]] = {
    CUDA_FLAVOR: {"venvs": 8.0, "downloads": 8.0},
    CPU_FLAVOR: {"venvs": 5.0, "downloads": 5.0},
}
"""설치에 필요한 남은 공간(GB, 여유 포함). venvs는 묶음 폴더 드라이브(설치 후 GPU판 약 6.2GB, CPU판 약 2.4GB),
downloads는 내려받은 파일 드라이브(pip 임시 폴더 + pip 보관 폴더: GPU판은 torch 2.87GB 파일이 두 곳에 동시에 있어 약 6.3GB).
같은 드라이브면 더해서 GPU판 16GB, CPU판 10GB가 필요하다."""
MIN_FREE_GB: float = SPACE_NEED_GB[CPU_FLAVOR]["venvs"] + SPACE_NEED_GB[CPU_FLAVOR]["downloads"]
EXPECTED_WARD_PHRASES: int = 20

WARD_THRESHOLDS: Path = DEPLOY / "ward_thresholds.json"
"""묶음에 넣은 동결 병실 기준값(build_bundle.py --freeze). 있으면 서버를 이 값으로 켜고(service_env), 시연 점검이
/health model.ward_phrases.thresholds와 같은지 본다. 없으면 코드 기본값(임시값)으로 켠다. 저장소에는 넣지 않는다."""
WARD_THRESHOLDS_SCHEMA: str = "silenttalk-ward-thresholds-v1"
WARD_THRESHOLD_ENV: dict[str, str] = {
    "min_margin": "LIPNET_OLKAVS_WARD_MIN_MARGIN",
    "min_score": "LIPNET_OLKAVS_WARD_MIN_SCORE",
    "possible_min_score": "LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE",
    "enroll_max_distance": "LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE",
}
"""병실 기준값 4개 → 서버 환경 변수 (scripts/ward_tools/common.py SERVER_RULE_ENV와 같은 이름)."""
WARD_THRESHOLD_TOL: float = 1e-9
WARD_THRESHOLD_KO: dict[str, str] = {"min_margin": "1·2등 차이", "min_score": "1등 점수", "possible_min_score": "긴급 가능성 하한",
                                     "enroll_max_distance": "등록 거리"}

_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
"""로컬 서버(127.0.0.1)용 요청기. Windows에 프록시가 설정돼 있어도 거치지 않는다(파이썬은 127.0.0.1을 프록시로 보낼 수 있다).
인터넷 확인(pypi·pytorch)은 시스템 프록시를 그대로 쓴다."""

_DISK_FULL_MARKERS = ("errno 28", "no space left", "winerror 112", "winerror 39", "not enough space on the disk",
                      "디스크 공간이 부족")
_DLL_MARKERS = ("dll load failed", "winerror 126", "winerror 127", "winerror 1114", "msvcp140", "vcruntime140",
                "dll initialization routine failed", "dll 초기화")

_SHA_LINE = re.compile(r'^CHECKPOINT_SHA256\s*=\s*"([0-9a-f]{64})"', re.M)
_COMMIT_LINE = re.compile(r'^EXPECTED_COMMIT\s*=\s*"([0-9a-f]{40})"', re.M)
_CUDA_HEADER = re.compile(r"CUDA (?:UMD )?Version:\s*([0-9]+)\.([0-9]+)")


class BundleError(RuntimeError):
    """사용자에게 그대로 보여 줄 한국어 문구를 담은 실패. hint는 다음에 할 일이다."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.message = message
        self.hint = hint


def setup_console() -> None:
    """Windows 콘솔(cp949)에서 한글 출력이 깨지거나 멈추지 않게 표준출력을 UTF-8로 바꾼다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def disable_quick_edit() -> None:
    """이 콘솔 창의 '빠른 편집' 모드를 끈다.

    켜져 있으면 창을 한 번 클릭해 글자가 선택되는 순간 출력이 멈추고, 출력하던 서버도 같이 멈춘다
    (시연 중 아무도 모르게 요청이 걸리는 원인). 콘솔이 없으면 아무것도 하지 않는다.
    """
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        enable_quick_edit, enable_extended_flags = 0x0040, 0x0080
        kernel32.SetConsoleMode(handle, (mode.value & ~enable_quick_edit) | enable_extended_flags)
    except (OSError, AttributeError, ValueError):
        pass


def core_constants(root: Path = ROOT) -> dict[str, str]:
    """scripts/olkavs_transcribe.py에 적힌 가중치 SHA-256과 공식 코드 commit을 읽는다(같은 값을 두 곳에 두지 않는다).

    Returns:
        {"checkpoint_sha256": ..., "upstream_commit": ...}

    Raises:
        BundleError: 파일이 없거나 값을 찾지 못했을 때.
    """
    path = root / "scripts" / "olkavs_transcribe.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BundleError(f"인식 코드 파일을 읽지 못했습니다: {path}", "묶음이 온전히 복사됐는지 확인하세요.") from exc
    sha, commit = _SHA_LINE.search(text), _COMMIT_LINE.search(text)
    if not sha or not commit:
        raise BundleError("인식 코드에서 가중치 SHA 또는 commit 값을 찾지 못했습니다.", "묶음을 다시 만들어 주세요.")
    return {"checkpoint_sha256": sha.group(1), "upstream_commit": commit.group(1)}


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    """파일 SHA-256 (소문자 16진수)."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
    except OSError as exc:
        raise BundleError(f"파일을 읽지 못했습니다: {path} ({exc})") from exc
    return digest.hexdigest()


def check_model_files(root: Path = ROOT) -> dict[str, Any]:
    """가중치 SHA-256과 공식 코드 commit을 확인한다.

    Returns:
        {"weights_sha256", "weights_ok", "weights_bytes", "upstream_commit", "upstream_ok", "expected": {...}}
    """
    expected = core_constants(root)
    weights = root / "experiments" / "olkavs_official" / "weights" / "multiview_lip.pt"
    upstream = root / "experiments" / "olkavs_official" / "upstream"
    result: dict[str, Any] = {"expected": expected}
    if weights.is_file():
        result["weights_bytes"] = weights.stat().st_size
        result["weights_sha256"] = sha256_file(weights)
    else:
        result["weights_bytes"] = None
        result["weights_sha256"] = None
    result["weights_ok"] = result["weights_sha256"] == expected["checkpoint_sha256"]
    try:
        result["upstream_commit"] = read_head_commit(upstream)
    except ValueError:
        result["upstream_commit"] = None
    result["upstream_ok"] = result["upstream_commit"] == expected["upstream_commit"]
    return result


def venv_python(name: str, root: Path | None = None) -> Path:
    """묶음 가상환경의 파이썬 경로 (Windows Scripts\\python.exe, 그 밖에는 bin/python). root 기본값은 ROOT."""
    root = root if root is not None else ROOT
    if os.name == "nt":
        return root / name / "Scripts" / "python.exe"
    return root / name / "bin" / "python"


def local_urlopen(request: urllib.request.Request | str, timeout: float) -> Any:
    """로컬 서버 요청을 프록시 없이 연다(urllib.request.urlopen과 같은 반환값·예외)."""
    return _LOCAL_OPENER.open(request, timeout=timeout)


def http_json(url: str, timeout: float = 3.0, headers: dict[str, str] | None = None
              ) -> tuple[int | None, Any, dict[str, str]]:
    """GET 요청 후 JSON을 읽는다(프록시를 거치지 않는다).

    Returns:
        (HTTP 상태 코드 또는 연결 실패 시 None, JSON 값 또는 None, 응답 헤더(소문자 키)).
    """
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with local_urlopen(request, timeout=timeout) as response:
            body = response.read()
            head = {k.lower(): v for k, v in response.headers.items()}
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except OSError:
            body = b""
        head = {k.lower(): v for k, v in (exc.headers or {}).items()}
        status = exc.code
    except (urllib.error.URLError, OSError, ValueError):
        return None, None, {}
    try:
        return status, json.loads(body.decode("utf-8")), head
    except (UnicodeDecodeError, ValueError):
        return status, None, head


def http_bytes(url: str, timeout: float = 3.0) -> tuple[int | None, bytes, dict[str, str]]:
    """GET 요청 후 본문 바이트를 돌려준다(프록시를 거치지 않는다). 연결 실패면 (None, b'', {})."""
    try:
        with local_urlopen(url, timeout=timeout) as response:
            return response.status, response.read(), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, b"", {k.lower(): v for k, v in (exc.headers or {}).items()}
    except (urllib.error.URLError, OSError, ValueError):
        return None, b"", {}


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    """그 포트에서 누가 접속을 받고 있으면 True."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def backend_state() -> tuple[str, Any]:
    """8000번 포트 상태: ('ours', health) | ('busy', None) | ('free', None)."""
    status, data, _ = http_json(HEALTH_URL, timeout=3.0)
    if status == 200 and isinstance(data, dict) and "model_backend" in data:
        return "ours", data
    return ("busy", None) if port_open(BACKEND_PORT) else ("free", None)


def frontend_state() -> str:
    """5173번 포트 상태: 'ours'(빌드한 화면) | 'busy'(다른 프로그램) | 'free'."""
    status, body, _ = http_bytes(FRONT_URL, timeout=3.0)
    if status == 200 and b'id="root"' in body:
        return "ours"
    return "busy" if port_open(FRONTEND_PORT) else "free"


def workers_ready(health: Any) -> bool:
    """두 워커가 실제로 준비를 마쳤는지. state()는 기동 중에도 'ready'라서 기동 정보(device·profile)까지 본다."""
    try:
        model = health["model"]
        return (model.get("roi_worker") == "ready" and model.get("infer_worker") == "ready"
                and bool(model.get("device")) and bool(model.get("roi_profile")))
    except (KeyError, TypeError, AttributeError):
        return False


_ERROR_CODES = {"start_failed": "시작 실패", "crashed": "비정상 종료", "send_failed": "요청 전달 실패"}


def worker_error_text(raw: Any) -> str | None:
    """/health의 last_error 값('start_failed at 2026-11-10 01:02:03')을 한국어로. 오류가 없으면 None."""
    if not raw:
        return None
    code, _, when = str(raw).partition(" at ")
    label = _ERROR_CODES.get(code.strip(), f"오류({code.strip()})")
    return f"{label} ({when.strip()})" if when.strip() else label


def worker_states(health: Any) -> dict[str, str]:
    """두 워커 상태를 학생이 읽을 한국어로: {"roi": ..., "infer": ...}.

    서버의 원래 값은 stopped(아직 안 켬) | dead(켜는 중이거나 꺼짐) | ready다. 켜는 동안에도 dead라서 오류가 없으면
    '켜는 중'으로 보인다.
    """
    model = health.get("model") if isinstance(health, dict) else None
    if not isinstance(model, dict):
        return {"roi": "서버 켜는 중", "infer": "서버 켜는 중"}
    errors = model.get("last_error") if isinstance(model.get("last_error"), dict) else {}
    started = {"roi": bool(model.get("roi_profile")), "infer": bool(model.get("device"))}
    out: dict[str, str] = {}
    for key in ("roi", "infer"):
        state = model.get(f"{key}_worker")
        error = worker_error_text(errors.get(key))
        if error:
            out[key] = error
        elif state == "ready" and started[key]:
            out[key] = "준비됨"
        elif state == "stopped":
            out[key] = "대기"
        else:
            out[key] = "켜는 중"
    return out


def describe_workers(health: Any) -> str:
    """'입술 추출 준비됨, 인식 켜는 중' 같은 한 줄."""
    states = worker_states(health)
    return f"입술 추출 {states['roi']}, 인식 {states['infer']}"


def worker_errors(health: Any) -> dict[str, str]:
    """오류가 있는 워커만 {"roi"|"infer": 한국어 오류}."""
    model = health.get("model") if isinstance(health, dict) else None
    errors = model.get("last_error") if isinstance(model, dict) and isinstance(model.get("last_error"), dict) else {}
    return {key: text for key in ("roi", "infer") if (text := worker_error_text(errors.get(key)))}


def text_mentions_disk_full(text: str) -> bool:
    """오류 문구가 디스크 공간 부족(Errno 28, WinError 112 등)을 말하는지."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _DISK_FULL_MARKERS)


def text_mentions_dll_problem(text: str) -> bool:
    """오류 문구가 DLL을 불러오지 못한 것(대개 Visual C++ 재배포 패키지 없음·오래됨)인지."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _DLL_MARKERS)


def vc_redist_hint() -> str:
    """Visual C++ 재배포 패키지 안내 문구."""
    return (f"Microsoft Visual C++ 재배포 패키지(x64)를 설치한 뒤 다시 실행하세요: {VC_REDIST_URL} "
            "(받은 파일 실행 → 동의 → 설치. 이미 있으면 '복구'를 누릅니다. 끝나면 노트북을 다시 시작합니다.)")


def read_tail(path: Path, start: int = 0, limit: int = 65536) -> str:
    """파일의 start 바이트 이후 내용 중 마지막 limit 바이트(없으면 빈 문자열)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(start, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def pip_cache_dir() -> Path:
    """pip가 내려받은 파일을 보관하는 폴더(PIP_CACHE_DIR 또는 %LOCALAPPDATA%\\pip\\Cache)."""
    configured = os.environ.get("PIP_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "pip" / "Cache"


def _existing_anchor(path: Path) -> Path:
    """아직 없는 폴더면 있는 부모 폴더까지 올라간다(남은 공간 조회용)."""
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def space_requirements(flavor: str, root: Path, download_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """드라이브별 필요한 남은 공간(GB).

    Args:
        flavor: 새로 설치할 인식 환경의 torch 종류(설치할 것이 없으면 CPU_FLAVOR 기준으로 충분하다).
        root: 묶음 폴더(가상환경이 생기는 곳).
        download_dirs: pip 임시 폴더·보관 폴더.

    Returns:
        {"C:": {"need_gb": 16.0, "path": Path}, ...} — 같은 드라이브면 필요량을 더한다.
    """
    need = SPACE_NEED_GB.get(flavor, SPACE_NEED_GB[CPU_FLAVOR])
    out: dict[str, dict[str, Any]] = {}

    def add(path: Path, amount: float) -> None:
        anchor = _existing_anchor(path)
        drive = (anchor.drive or str(anchor)).upper()
        entry = out.setdefault(drive, {"need_gb": 0.0, "path": anchor})
        entry["need_gb"] += amount

    add(root, need["venvs"])
    drives = {(_existing_anchor(p).drive or str(p)).upper(): p for p in download_dirs}
    for path in drives.values():
        add(path, need["downloads"])
    return out


def _find_nvidia_smi() -> str | None:
    """nvidia-smi 실행 파일 위치 (PATH → 기본 설치 폴더)."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
                      Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI"
                      / "nvidia-smi.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


def _version_tuple(text: str) -> tuple[int, ...]:
    """'570.65' → (570, 65). 숫자가 아닌 조각은 0."""
    parts = []
    for piece in text.strip().split("."):
        digits = re.match(r"\d+", piece)
        parts.append(int(digits.group(0)) if digits else 0)
    return tuple(parts)


def nvidia_info() -> dict[str, Any]:
    """NVIDIA 그래픽카드·드라이버·지원 CUDA 버전. 없거나 nvidia-smi가 실패하면 present=False.

    Returns:
        {"present", "name", "driver", "cuda": "13.3" 또는 None, "memory_mb", "compute_cap": "8.6" 또는 None, "error"}
    """
    info: dict[str, Any] = {"present": False, "name": None, "driver": None, "cuda": None, "memory_mb": None,
                            "compute_cap": None, "error": None}
    exe = _find_nvidia_smi()
    if not exe:
        info["error"] = "nvidia-smi 없음"
        return info
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        query = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20, creationflags=flags)
        header = subprocess.run([exe], capture_output=True, text=True, timeout=20, creationflags=flags)
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = f"nvidia-smi 실행 실패: {exc}"
        return info
    if query.returncode != 0 or not query.stdout.strip():
        info["error"] = (query.stderr or query.stdout or "nvidia-smi 오류").strip()[:200]
        return info
    first = query.stdout.strip().splitlines()[0].split(",")
    info["present"] = True
    info["name"] = first[0].strip()
    info["driver"] = first[1].strip() if len(first) > 1 else None
    try:
        info["memory_mb"] = int(float(first[2])) if len(first) > 2 else None
    except ValueError:
        info["memory_mb"] = None
    match = _CUDA_HEADER.search(header.stdout or "")
    if match:
        info["cuda"] = f"{match.group(1)}.{match.group(2)}"
    try:  # 오래된 드라이버는 compute_cap 항목을 모른다 → 조용히 넘어간다
        cap = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20, creationflags=flags)
        value = (cap.stdout or "").strip().splitlines()[0].strip() if cap.returncode == 0 and cap.stdout.strip() else ""
        if re.fullmatch(r"\d+\.\d+", value):
            info["compute_cap"] = value
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return info


def choose_torch_flavor(info: dict[str, Any]) -> tuple[str, str]:
    """그래픽카드 정보로 torch 종류를 고른다.

    - 그래픽카드 세대(compute capability)가 7.0보다 낮으면 cu128 torch가 지원하지 않으므로 CPU판.
    - 드라이버가 CUDA 12.8 이상이면 GPU판. CUDA 12.0~12.7이면 하위 호환으로 GPU판을 시도한다(설치 후 실제 GPU 계산
      시험이 실패하면 install.py가 CPU판으로 바꾼다). 그보다 낮으면 CPU판.

    Returns:
        (CUDA_FLAVOR 또는 CPU_FLAVOR, 이유 한국어 문장)
    """
    if not info.get("present"):
        return CPU_FLAVOR, "NVIDIA 그래픽카드(또는 드라이버)를 찾지 못해 CPU판 torch를 씁니다."
    name = info.get("name") or "NVIDIA 그래픽카드"
    cap = info.get("compute_cap")
    if cap and _version_tuple(cap)[:2] < MIN_COMPUTE_CAP:
        return CPU_FLAVOR, (f"그래픽카드({name}, 세대 {cap})가 GPU판 torch(cu128)가 지원하는 세대 "
                            f"{MIN_COMPUTE_CAP[0]}.{MIN_COMPUTE_CAP[1]}보다 오래돼 CPU판 torch를 씁니다.")
    cuda = info.get("cuda")
    if cuda:
        version = _version_tuple(cuda)[:2]
        if version >= MIN_CUDA_VERSION:
            return CUDA_FLAVOR, f"드라이버가 CUDA {cuda}까지 지원해 GPU판(cu128) torch를 씁니다."
        if version >= MIN_CUDA_COMPAT:
            return CUDA_FLAVOR, (f"드라이버가 CUDA {cuda}까지 지원합니다(12.8보다 낮음). 하위 호환으로 GPU판(cu128)을 "
                                 "설치해 보고, 실제 GPU 계산이 안 되면 CPU판으로 바꿉니다. 드라이버를 최신으로 바꾸면 "
                                 "더 확실합니다.")
        return CPU_FLAVOR, (f"드라이버가 CUDA {cuda}까지만 지원합니다(12.0 이상 필요). CPU판 torch를 씁니다. "
                            "NVIDIA 드라이버를 최신으로 바꾸면 GPU판을 쓸 수 있습니다.")
    driver = info.get("driver") or ""
    version = _version_tuple(driver)[:2] if driver else (0, 0)
    if version >= MIN_WINDOWS_DRIVER:
        return CUDA_FLAVOR, f"드라이버 {driver}가 CUDA 12.8을 지원해 GPU판(cu128) torch를 씁니다."
    if version >= MIN_WINDOWS_DRIVER_COMPAT:
        return CUDA_FLAVOR, (f"드라이버 {driver}는 CUDA 12.8보다 낮은 버전입니다. 하위 호환으로 GPU판(cu128)을 설치해 "
                             "보고, 실제 GPU 계산이 안 되면 CPU판으로 바꿉니다.")
    return CPU_FLAVOR, (f"드라이버 {driver or '버전 모름'}로는 CUDA 12 지원을 확인하지 못했습니다. "
                        "CPU판 torch를 씁니다.")


def normalize_name(name: str) -> str:
    """패키지 이름 비교용 정규화 (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    """'이름==버전' 줄만 읽어 {정규화 이름: 버전}. 주석·빈 줄은 건너뛴다."""
    pins: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BundleError(f"패키지 목록 파일을 읽지 못했습니다: {path}") from exc
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise BundleError(f"패키지 목록에 버전이 고정되지 않은 줄이 있습니다: {path.name}: {raw}")
        name, version = line.split("==", 1)
        name = name.split("[", 1)[0].strip()
        pins[normalize_name(name)] = version.strip()
    return pins


def load_json(path: Path, default: Any = None) -> Any:
    """JSON 파일을 읽는다. 없거나 깨졌으면 default."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path: Path, data: Any) -> None:
    """JSON을 임시 파일에 쓴 뒤 바꿔 넣는다(쓰는 도중 끊겨도 반쪽 파일이 남지 않는다)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        raise BundleError(f"기록 파일을 쓰지 못했습니다: {path} ({exc})") from exc


def is_ascii_path(path: Path) -> bool:
    """경로가 영문·숫자·기호(ASCII)로만 돼 있으면 True. mediapipe는 한글 경로에서 모델 파일을 못 여는 일이 있다."""
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def free_gb(path: Path) -> float:
    """그 경로가 있는 드라이브의 남은 공간(GB, 10^9 아닌 2^30 기준)."""
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return float("nan")


def _threshold_value(key: str, value: Any) -> float | None:
    """병실 기준값 하나를 검사한다(서버 config.py와 같은 허용 범위). 하한·등록 거리는 None(쓰지 않음)도 된다.

    Raises:
        ValueError: 숫자가 아니거나 범위 밖일 때
    """
    if value is None:
        if key in ("possible_min_score", "enroll_max_distance"):
            return None
        raise ValueError(f"{key}는 비워 둘 수 없습니다")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}가 숫자가 아닙니다: {value!r}")
    try:
        number = float(value)
        ok = {"min_margin": lambda v: v >= 0.0, "min_score": lambda v: v <= 0.0,
              "possible_min_score": lambda v: v <= 0.0, "enroll_max_distance": lambda v: v > 0.0}[key]
    except (KeyError, OverflowError, TypeError) as exc:
        raise ValueError(f"{key}: 알 수 없는 기준값이거나 숫자가 아닙니다") from exc
    if not (number == number and abs(number) != float("inf") and ok(number)):
        raise ValueError(f"{key} 값이 허용 범위 밖입니다: {value!r}")
    return number


def threshold_env_value(value: float | None) -> str:
    """기준값 → 환경 변수 문자열 (None이면 'none', 서버는 'none'을 '쓰지 않음'으로 읽는다).

    Raises:
        ValueError: 숫자가 아닐 때
    """
    try:
        return "none" if value is None else repr(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"환경 변수 값으로 바꿀 수 없습니다: {value!r}") from exc


def ward_thresholds_doc(values: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """묶음에 넣을 기준값 파일 내용 {schema, values, env, source}. values는 기준값 4개다.

    Raises:
        BundleError: 기준값이 빠졌거나 범위 밖일 때
    """
    try:
        checked = {key: _threshold_value(key, values.get(key)) for key in WARD_THRESHOLD_ENV}
    except (AttributeError, ValueError) as exc:
        raise BundleError(f"병실 기준값이 올바르지 않습니다: {exc}") from exc
    return {"schema": WARD_THRESHOLDS_SCHEMA, "values": checked,
            "env": {WARD_THRESHOLD_ENV[k]: threshold_env_value(v) for k, v in checked.items()}, "source": source}


def load_ward_thresholds(path: Path | None = None) -> dict[str, Any] | None:
    """묶음의 동결 병실 기준값 파일을 읽어 검사한다. 파일이 없으면 None(코드 기본값으로 켠다).

    Returns:
        {schema, values, env, source} (env는 values에서 다시 만든 값)
    Raises:
        BundleError: 파일이 깨졌거나 값이 틀릴 때(옛 값이나 기본값으로 조용히 켜지지 않게 막는다)
    """
    target = path or WARD_THRESHOLDS
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleError(f"병실 기준값 파일을 읽지 못했습니다: {target.name} ({exc})",
                          "zip을 처음부터 다시 풀거나 묶음을 다시 만드세요.") from exc
    if not isinstance(data, dict) or data.get("schema") != WARD_THRESHOLDS_SCHEMA or not isinstance(data.get("values"), dict):
        raise BundleError(f"병실 기준값 파일 형식이 다릅니다: {target.name}", "묶음을 다시 만드세요.")
    doc = ward_thresholds_doc(data["values"], data.get("source") if isinstance(data.get("source"), dict) else {})
    if isinstance(data.get("env"), dict) and data["env"] != doc["env"]:
        raise BundleError(f"병실 기준값 파일의 env가 values와 다릅니다: {target.name}", "묶음을 다시 만드세요.")
    return doc


def describe_thresholds(values: Any) -> str:
    """'1·2등 차이 0.5, 1등 점수 -3.4, 긴급 가능성 하한 없음, 등록 거리 0.27'처럼."""
    try:
        if not isinstance(values, dict):
            return "알 수 없음"
        return ", ".join(f"{name} {'없음' if values.get(key) is None else values.get(key)}"
                         for key, name in WARD_THRESHOLD_KO.items())
    except (TypeError, ValueError):
        return "알 수 없음"


def thresholds_problems(server: Any, expected: dict[str, Any]) -> list[str]:
    """서버 /health model.ward_phrases.thresholds가 묶음 기준값과 다른 항목(한국어 문제 목록, 같으면 빈 목록)."""
    if not isinstance(server, dict):
        return ["서버가 병실 기준값(thresholds)을 알려 주지 않습니다(옛 서버이거나 병실 명령이 꺼짐)"]
    problems = []
    for key, name in WARD_THRESHOLD_KO.items():
        want, got = expected.get(key), server.get(key)
        if want is None and got is None:
            continue
        try:
            same = (want is not None and got is not None and not isinstance(got, bool) and isinstance(got, (int, float))
                    and abs(float(got) - float(want)) <= WARD_THRESHOLD_TOL)
        except (TypeError, ValueError):
            same = False
        if not same:
            problems.append(f"{name}: 서버 {'없음' if got is None else got} ≠ 묶음 {'없음' if want is None else want}")
    return problems


def service_env(device: str | None = None) -> dict[str, str]:
    """서버·워커에 넘길 환경변수.

    - LIPNET_* 값은 모두 지우고 저장소 기본값으로 실행한다. 노트북에 남은 옛 설정이 섞이지 않게 한다.
    - 묶음에 동결 병실 기준값 파일(deploy/ward_thresholds.json)이 있으면 그 4개(LIPNET_OLKAVS_WARD_*)만 다시 넣는다.
      없으면 병실 기준값도 코드 기본값(임시값)이다.
    - device가 'cpu'·'cuda:0'이면 인식 장치를 그 값으로 고정한다(기본 auto: CUDA가 되면 GPU).
    - git 대체 명령을 만들었으면 PATH 맨 앞에 둔다(인식 워커의 commit 확인용).

    Raises:
        BundleError: 기준값 파일이 깨졌거나 값이 틀릴 때
    """
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("LIPNET_") and k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["LIPNET_MODEL_BACKEND"] = "olkavs"
    thresholds = load_ward_thresholds()
    if thresholds is not None:
        env.update(thresholds["env"])
    if device and device != "auto":
        env["LIPNET_OLKAVS_DEVICE"] = device
    env["PYTHONIOENCODING"] = "utf-8"
    if (GITSHIM_DIR / "git.exe").is_file():
        env["PATH"] = str(GITSHIM_DIR) + os.pathsep + env.get("PATH", "")
    return env


def windows_name() -> str:
    """'Windows 11 (빌드 22631)'처럼. 파이썬 3.11의 platform.release()는 Windows 11도 '10'으로 돌려준다."""
    try:
        import platform  # noqa: PLC0415

        release, version = platform.release(), platform.version()
        build = int(version.rsplit(".", 1)[-1]) if version else 0
        if release == "10" and build >= 22000:
            release = "11"
        return f"Windows {release} (빌드 {build or version})"
    except (ValueError, OSError):
        return "Windows (버전 모름)"


def timestamp() -> str:
    """파일 이름용 현재 시각 (YYYYmmdd_HHMMSS)."""
    return time.strftime("%Y%m%d_%H%M%S")


def percentile(values: list[float], q: float) -> float | None:
    """가장 가까운 순위 방식 백분위수 (값이 없으면 None). q는 0~100."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 100))))
    return ordered[rank - 1]


def median(values: list[float]) -> float | None:
    """중앙값 (값이 없으면 None)."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
