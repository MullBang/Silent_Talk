"""시연 노트북 설치 (설치.bat이 Python 3.11로 실행한다. 인터넷은 처음 설치할 때 한 번만 필요하다).

하는 일:
  1. 파이썬 3.11(64비트)·Windows·Visual C++ 재배포 패키지(MSVCP140.dll) 확인
  2. 묶음 파일 확인: 폴더 경로(영문), 가중치 SHA-256, 공식 코드 commit, 병실 명령 20개, 빌드한 화면
  3. 시연 프로그램이 켜져 있지 않은지 확인 (켜져 있으면 파일이 잠겨 설치가 깨진다)
  4. 그래픽카드 확인 → GPU판(cu128) 또는 CPU판 torch 선택 (세대 7.0 미만·CUDA 12.0 미만 드라이버는 CPU판)
  5. 할 일이 있으면 디스크 여유(같은 드라이브 기준 GPU판 16GB, CPU판 10GB)·인터넷 확인.
     GPU판에 공간이 모자라고 CPU판에는 충분하면 CPU판으로 설치한다(--device auto일 때).
  6~8. 가상환경 3개를 고정 목록으로 만든다: .venv-server(서버), .venv-olkavs-roi(입술 추출), .venv-olkavs(인식)
       GPU판이면 실제 CUDA 계산을 해 보고, 안 되면 CPU판으로 바꿔 설치한다(--device auto일 때).
  9. git 확인: 없으면 commit 확인용 대체 명령(deploy/runtime/gitshim/git.exe)을 만든다
 10. 실제 기동 확인: 서버 코드 import, 입술 추출 워커·인식 워커를 실제로 띄워 준비 완료 신호를 받는다
 11. 요약 출력, deploy/runtime/install_state.json 기록

다시 실행해도 안전하다(이미 끝난 환경은 고정 목록 지문이 같으면 건너뛴다). 중간에 끊겼으면 다시 실행하면 이어서 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402

TOTAL_STEPS = 11
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WORKER_TIMEOUT_SEC = 420.0
"""워커 기동 확인 상한. 첫 실행은 라이브러리 첫 import·모델 로드가 겹쳐 느리다(서버 자체 상한은 180초)."""

CUDA_TEST = r"""
import json, sys
out = {"torch": None, "cuda_build": None, "available": False, "ok": False}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_build"] = torch.version.cuda
    out["available"] = bool(torch.cuda.is_available())
    if out["available"]:
        x = torch.randn(256, 256, device="cuda")
        conv = torch.nn.Conv3d(3, 8, 3).cuda()
        y = conv(torch.randn(1, 3, 8, 32, 32, device="cuda")).sum() + (x @ x).sum()
        y.item()
        torch.cuda.synchronize()
        out["name"] = torch.cuda.get_device_name(0)
        out["capability"] = list(torch.cuda.get_device_capability(0))
        out["ok"] = True
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:400]
print(json.dumps(out))
"""

SERVER_IMPORT_TEST = r"""
import json, sys
sys.path.insert(0, ".")
import uvicorn, fastapi, multipart, cv2, numpy, torch, apscheduler
import main
print(json.dumps({"python": sys.version.split()[0], "fastapi": fastapi.__version__, "uvicorn": uvicorn.__version__,
                  "torch": torch.__version__, "cv2": cv2.__version__, "numpy": numpy.__version__,
                  "routes": len(main.app.routes)}))
"""

VERSIONS_SNIPPET = r"""
import json, importlib.metadata as md
print(json.dumps({d.metadata["Name"]: d.version for d in md.distributions() if d.metadata["Name"]}))
"""

SHIM_MAKER = r"""
import sys
from pip._vendor.distlib.scripts import ScriptMaker
maker = ScriptMaker(sys.argv[1], sys.argv[2], add_launchers=True)
maker.executable = sys.argv[3]
maker.clobber = True
print(maker.make("git.py"))
"""


class Log:
    """화면과 설치 기록 파일에 같은 내용을 쓴다."""

    def __init__(self, path: Path):
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(path, "a", encoding="utf-8")  # noqa: SIM115 - 설치 동안 유지
        except OSError:
            self.handle = None

    def write(self, text: str) -> None:
        """한 줄을 쓴다."""
        print(text, flush=True)
        if self.handle:
            try:
                self.handle.write(text + "\n")
                self.handle.flush()
            except OSError:
                pass

    def step(self, number: int, title: str) -> None:
        """단계 제목."""
        self.write(f"\n[{number}/{TOTAL_STEPS}] {title}")

    def ok(self, text: str) -> None:
        """통과 줄."""
        self.write(f"  [통과] {text}")

    def info(self, text: str) -> None:
        """정보 줄."""
        self.write(f"  - {text}")

    def warn(self, text: str) -> None:
        """주의 줄."""
        self.write(f"  [주의] {text}")


@dataclass
class VenvSpec:
    """가상환경 하나의 설치 명세."""

    name: str
    label: str
    lock: Path
    torch_flavor: str | None  # None이면 torch 없음


def _run(cmd: list[str], log: Log, cwd: Path | None = None, env: dict[str, str] | None = None,
         timeout: float | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """명령을 실행한다. capture가 아니면 출력(pip 진행 표시 포함)을 그대로 화면에 보인다."""
    if not capture:
        log.write(f"    $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, timeout=timeout, text=True,
                              capture_output=capture, encoding="utf-8" if capture else None,
                              errors="replace" if capture else None, creationflags=NO_WINDOW if capture else 0)
    except FileNotFoundError as exc:
        raise bc.BundleError(f"실행 파일을 찾지 못했습니다: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise bc.BundleError(f"명령이 {timeout:.0f}초 안에 끝나지 않았습니다: {' '.join(cmd[:4])} ...") from exc


def _pip_env() -> dict[str, str]:
    """pip 실행 환경: 다른 파이썬 설정이 섞이지 않게 정리한다."""
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _python_works(python: Path) -> str | None:
    """가상환경 파이썬이 실행되면 버전 문자열(예: '3.11.9'), 아니면 None."""
    if not python.is_file():
        return None
    try:
        out = subprocess.run([str(python), "-c", "import sys; print(sys.version.split()[0])"], capture_output=True,
                             text=True, timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _marker_path(spec: VenvSpec) -> Path:
    """설치 완료 표시 파일."""
    return bc.ROOT / spec.name / "silenttalk_install.json"


def _fingerprint(spec: VenvSpec) -> dict[str, Any]:
    """고정 목록 내용 + torch 종류 + 파이썬 버전. 하나라도 바뀌면 그 환경을 다시 맞춘다."""
    try:
        lock_sha = hashlib.sha256(spec.lock.read_bytes()).hexdigest()
    except OSError as exc:
        raise bc.BundleError(f"고정 패키지 목록이 없습니다: {spec.lock}", "묶음을 다시 복사하세요.") from exc
    return {"lock_sha256": lock_sha, "torch": spec.torch_flavor,
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}"}


def venv_is_current(spec: VenvSpec) -> bool:
    """표시 파일의 지문이 지금 명세와 같고 파이썬이 실행되면 True."""
    marker = bc.load_json(_marker_path(spec), {}) or {}
    want = _fingerprint(spec)
    same = all(marker.get(k) == v for k, v in want.items()) and marker.get("complete") is True
    return same and _python_works(bc.venv_python(spec.name)) is not None


def _derived_requirements(spec: VenvSpec) -> tuple[Path, str | None]:
    """고정 목록에서 torch 줄을 뺀 PyPI용 목록(ASCII, 주석 없음)과 torch 버전(+종류 제외)을 만든다.

    pip는 목록 파일을 시스템 문자셋(cp949)으로 읽으므로 한글 주석이 있으면 깨진다. 그래서 핀 줄만 옮겨 쓴다.
    """
    pins = bc.parse_lock(spec.lock)
    torch_version = None
    lines = []
    for raw in spec.lock.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = bc.normalize_name(line.split("==", 1)[0].split("[", 1)[0])
        if name == "torch":
            torch_version = pins["torch"].split("+", 1)[0]
            continue
        lines.append(line)
    if spec.torch_flavor and not torch_version:
        raise bc.BundleError(f"{spec.lock.name}에 torch 줄이 없습니다.")
    out = bc.RUNTIME / "pip" / f"{spec.name.lstrip('.')}-pypi.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out, torch_version


def _installed_versions(python: Path) -> dict[str, str]:
    """가상환경에 설치된 {정규화 이름: 버전}."""
    out = subprocess.run([str(python), "-c", VERSIONS_SNIPPET], capture_output=True, text=True, timeout=120,
                         creationflags=NO_WINDOW)
    if out.returncode != 0:
        raise bc.BundleError("설치된 패키지 목록을 읽지 못했습니다.", out.stderr.strip()[-300:])
    return {bc.normalize_name(k): v for k, v in json.loads(out.stdout).items()}


def _compare_with_lock(spec: VenvSpec, installed: dict[str, str]) -> list[str]:
    """고정 목록과 다른 패키지 목록(문장). torch는 '+종류' 꼬리를 설치한 종류로 바꿔 비교한다."""
    problems = []
    for name, version in bc.parse_lock(spec.lock).items():
        want = version
        if name == "torch" and spec.torch_flavor:
            want = f"{version.split('+', 1)[0]}+{spec.torch_flavor}"
        have = installed.get(name)
        if have != want:
            problems.append(f"{name}: 필요 {want}, 설치 {have or '없음'}")
    return problems


def _log_size(path: Path) -> int:
    """pip 기록 파일의 지금 크기(--log는 이어 쓰므로 이번 실행분만 읽으려고 잰다)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _pip_failure(message: str, pip_log: Path, since: int, default_hint: str) -> bc.BundleError:
    """pip 실패를 원인별 안내로 바꾼다. 이번 실행분 기록에 디스크 부족이 보이면 공간 안내를 먼저 한다."""
    tail = bc.read_tail(pip_log, since)
    if bc.text_mentions_disk_full(tail):
        drives = sorted({(bc.ROOT.drive or "").upper(), (bc.pip_cache_dir().drive or "").upper(),
                         (Path(os.environ.get("TEMP", str(bc.ROOT))).drive or "").upper()} - {""})
        return bc.BundleError(message + " (디스크 공간 부족)",
                              f"{', '.join(drives)} 드라이브의 필요 없는 파일을 지워 공간을 만든 뒤 설치.bat을 다시 "
                              "실행하세요(GPU판은 약 16GB, CPU판은 약 10GB 필요). 명령 프롬프트에서 "
                              f"'py -3.11 -m pip cache purge'로 내려받은 파일 보관분도 지울 수 있습니다. 기록: {pip_log}")
    return bc.BundleError(message, f"{default_hint} 자세한 기록: {pip_log}")


def install_torch(spec: VenvSpec, torch_version: str, log: Log) -> None:
    """PyTorch 저장소에서 정해진 종류의 torch를 받는다(의존 패키지는 고정 목록에서 이미 설치)."""
    python = bc.venv_python(spec.name)
    flavor = spec.torch_flavor or bc.CPU_FLAVOR
    pip_log = bc.LOG_DIR / f"pip_{spec.name.lstrip('.')}_torch.log"
    since = _log_size(pip_log)
    result = _run([str(python), "-m", "pip", "install", "--no-input", "--no-deps", "--only-binary=:all:",
                   "--index-url", bc.PYTORCH_INDEX.format(flavor=flavor), "--log", str(pip_log),
                   f"torch=={torch_version}+{flavor}"], log, env=_pip_env())
    if result.returncode != 0:
        raise _pip_failure(f"torch({flavor}) 설치에 실패했습니다.", pip_log, since,
                           "인터넷 연결을 확인하고 설치.bat을 다시 실행하세요.")


def _pip_works(python: Path) -> bool:
    """가상환경에 pip가 제대로 들어 있는지(가상환경 만들기가 중간에 끊기면 파이썬만 있고 pip가 없다)."""
    try:
        out = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, text=True, timeout=120,
                             env=_pip_env(), creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def ensure_venv(spec: VenvSpec, log: Log, need_online: Callable[[], None]) -> dict[str, Any]:
    """가상환경 하나를 고정 목록대로 맞춘다. 이미 맞으면 건너뛴다.

    Returns:
        {"name", "status": "kept"|"installed", "seconds", "python"}
    """
    started = time.perf_counter()
    venv_dir = bc.ROOT / spec.name
    python = bc.venv_python(spec.name)
    if venv_is_current(spec):
        log.ok(f"{spec.label}({spec.name})은 이미 설치돼 있습니다.")
        return {"name": spec.name, "status": "kept", "seconds": 0.0, "python": _python_works(python)}
    need_online()
    marker = _marker_path(spec)
    if venv_dir.exists() and _python_works(python) is None:
        log.info(f"{spec.name} 폴더가 망가져 있어 지우고 새로 만듭니다.")
        _remove_venv(venv_dir)
    elif venv_dir.exists() and not _pip_works(python):
        log.info(f"{spec.name} 폴더에 pip가 없습니다(가상환경 만들기가 중간에 끊긴 것 같습니다). 지우고 새로 만듭니다.")
        _remove_venv(venv_dir)
    if not python.is_file():
        log.info(f"{spec.name} 가상환경을 만듭니다.")
        result = _run([sys.executable, "-m", "venv", str(venv_dir)], log, env=_pip_env())
        if result.returncode != 0 or not python.is_file() or not _pip_works(python):
            raise bc.BundleError(f"{spec.name} 가상환경을 만들지 못했습니다.",
                                 "디스크 공간을 확인하고, 파이썬 3.11을 python.org 설치 파일로 다시 설치해 보세요. "
                                 "백신이 막았으면 C:\\SilentTalk 폴더를 예외로 두세요.")
    try:
        marker.unlink()  # 설치 도중 끊기면 다음 실행에서 다시 맞추게 한다
    except OSError:
        pass
    requirements, torch_version = _derived_requirements(spec)
    pip_log = bc.LOG_DIR / f"pip_{spec.name.lstrip('.')}.log"
    log.info(f"{spec.label} 패키지를 내려받아 설치합니다 (기록: {pip_log}).")
    since = _log_size(pip_log)
    result = _run([str(python), "-m", "pip", "install", "--no-input", "--only-binary=:all:", "-r", str(requirements),
                   "--log", str(pip_log)], log, env=_pip_env())
    if result.returncode != 0:
        raise _pip_failure(f"{spec.label} 패키지 설치에 실패했습니다.", pip_log, since,
                           "인터넷 연결을 확인하고 다시 실행하세요. 'No matching distribution'이면 이 파이썬 버전에 "
                           "맞는 패키지가 없다는 뜻입니다(파이썬 3.11인지 확인).")
    if spec.torch_flavor:
        install_torch(spec, torch_version or "", log)
    check = _run([str(python), "-m", "pip", "check"], log, env=_pip_env(), capture=True, timeout=300)
    if check.returncode != 0:
        raise bc.BundleError(f"{spec.label}의 패키지 의존성이 맞지 않습니다.", (check.stdout or check.stderr)[-500:])
    problems = _compare_with_lock(spec, _installed_versions(python))
    if problems:
        raise bc.BundleError(f"{spec.label}의 패키지 버전이 고정 목록과 다릅니다: " + "; ".join(problems[:5]),
                             "설치.bat을 다시 실행하세요. 계속되면 이 폴더를 지우고 다시 설치하세요: " + str(venv_dir))
    bc.save_json(marker, {**_fingerprint(spec), "complete": True, "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "base_python": sys.executable})
    seconds = time.perf_counter() - started
    log.ok(f"{spec.label}({spec.name}) 설치 완료 ({seconds:.0f}초).")
    return {"name": spec.name, "status": "installed", "seconds": round(seconds, 1), "python": _python_works(python)}


def _remove_venv(path: Path) -> None:
    """이 묶음의 가상환경 폴더만 지운다(다른 경로는 거부)."""
    resolved = path.resolve()
    if resolved.parent != bc.ROOT.resolve() or resolved.name not in (bc.VENV_SERVER, bc.VENV_ROI, bc.VENV_INFER):
        raise bc.BundleError(f"지울 수 없는 경로입니다: {resolved}")
    shutil.rmtree(resolved, ignore_errors=False)


def mark_cuda_fallback(spec: VenvSpec, reason: str) -> None:
    """인식 환경을 CPU판으로 바꾼 이유를 표시 파일에 남긴다(다음 실행에서 cu128을 다시 받지 않게)."""
    marker = _marker_path(spec)
    data = bc.load_json(marker, {}) or {}
    data["cuda_fallback"] = reason
    bc.save_json(marker, data)


def check_python(log: Log) -> dict[str, Any]:
    """파이썬 3.11 64비트·Windows인지, Visual C++ 재배포 패키지가 있는지 확인한다.

    Returns:
        {"vc_runtime": {...}} (check_vc_runtime 결과)
    """
    version = sys.version.split()[0]
    if os.name != "nt":
        raise bc.BundleError("이 설치 스크립트는 Windows 노트북 전용입니다.")
    if sys.version_info[:2] != bc.REQUIRED_PYTHON:
        raise bc.BundleError(f"파이썬 3.11이 필요합니다. 지금 실행한 파이썬은 {version}입니다 ({sys.executable}).",
                             f"Python 3.11.9 (64-bit)을 설치하세요: {bc.PYTHON_DOWNLOAD_URL}")
    if sys.maxsize <= 2 ** 32:
        raise bc.BundleError("64비트 파이썬이 필요합니다(지금은 32비트).", f"64-bit 설치 파일: {bc.PYTHON_DOWNLOAD_URL}")
    log.ok(f"파이썬 {version} 64비트 ({sys.executable})")
    log.info(bc.windows_name())
    return {"vc_runtime": check_vc_runtime(log)}


def _file_version(path: str) -> tuple[int, int, int, int] | None:
    """Windows 파일 버전(예: (14, 44, 35211, 0)). 읽지 못하면 None."""
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        version = ctypes.WinDLL("version")
        version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
        version.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        version.GetFileVersionInfoW.restype = wintypes.BOOL
        version.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.POINTER(wintypes.UINT)]
        version.VerQueryValueW.restype = wintypes.BOOL
        size = version.GetFileVersionInfoSizeW(path, None)
        if not size:
            return None
        buffer = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(path, 0, size, buffer):
            return None
        pointer, length = ctypes.c_void_p(), wintypes.UINT()
        if not version.VerQueryValueW(buffer, "\\", ctypes.byref(pointer), ctypes.byref(length)) or not pointer.value:
            return None
        fixed = (wintypes.DWORD * 4).from_address(pointer.value + 8)  # dwSignature, dwStrucVersion 다음 MS·LS
        return fixed[0] >> 16, fixed[0] & 0xFFFF, fixed[1] >> 16, fixed[1] & 0xFFFF
    except (OSError, AttributeError, ValueError, TypeError):
        return None


def check_vc_runtime(log: Log) -> dict[str, Any]:
    """Visual C++ 2015-2022 재배포 패키지(MSVCP140.dll 등)를 불러올 수 있는지 확인한다.

    torch(c10.dll)와 mediapipe가 MSVCP140.dll을 쓰는데, 파이썬(python.org)에는 vcruntime140만 들어 있다.
    없으면 나중에 'DLL load failed'만 보이므로 여기서 먼저 멈추고 설치 주소를 알려 준다.
    오래된 판(14.40 미만)은 주의만 보인다(실제로 문제가 되면 10단계 기동 확인에서 같은 안내가 나온다).

    Raises:
        BundleError: DLL을 불러오지 못했을 때.
    """
    info: dict[str, Any] = {}
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    except (ImportError, OSError, AttributeError) as exc:
        log.warn(f"Visual C++ 재배포 패키지를 확인하지 못했습니다({exc}). 계속합니다.")
        return {"checked": False}
    missing = []
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        try:
            handle = ctypes.WinDLL(name)
        except OSError:
            missing.append(name)
            continue
        buffer = ctypes.create_unicode_buffer(1024)
        kernel32.GetModuleFileNameW(wintypes.HMODULE(handle._handle), buffer, 1024)
        info[name] = {"path": buffer.value or None}
    if missing:
        raise bc.BundleError("Microsoft Visual C++ 재배포 패키지가 없습니다(" + ", ".join(missing) + "). torch·입술 추출 "
                             "프로그램이 이 파일을 씁니다.", bc.vc_redist_hint())
    path = info["msvcp140.dll"]["path"]
    found = _file_version(path) if path else None
    info["msvcp140.dll"]["version"] = ".".join(map(str, found)) if found else None
    info["checked"] = True
    if found and found[:2] < bc.MIN_MSVCP_VERSION:
        info["old"] = True
        log.warn(f"Visual C++ 재배포 패키지가 오래됐습니다(MSVCP140 {info['msvcp140.dll']['version']}). torch를 불러올 때 "
                 f"오류가 나면 최신판을 설치하세요: {bc.VC_REDIST_URL}")
    else:
        log.ok(f"Visual C++ 재배포 패키지 있음 (MSVCP140 {info['msvcp140.dll']['version'] or '버전 모름'})")
    return info


def check_bundle(log: Log, allow_non_ascii: bool) -> dict[str, Any]:
    """묶음 파일이 온전한지 확인한다."""
    if not bc.is_ascii_path(bc.ROOT):
        message = (f"묶음 폴더 경로에 한글·특수문자가 있습니다: {bc.ROOT}. 입술 추출(mediapipe)이 이런 경로에서 "
                   "모델 파일을 열지 못할 수 있습니다.")
        if not allow_non_ascii:
            raise bc.BundleError(message, r"묶음 폴더를 C:\SilentTalk 처럼 영문 경로로 옮긴 뒤 다시 실행하세요.")
        log.warn(message)
    else:
        log.ok(f"묶음 위치: {bc.ROOT}")
    required = [bc.BACKEND / "main.py", bc.WORKER_SCRIPT, bc.CORE_SCRIPT, bc.FRONTEND_DIST / "index.html",
                bc.WARD_COMMANDS, bc.LOCK_SERVER, bc.LOCK_ROI, bc.LOCK_INFER, bc.UPSTREAM / "configs" / "v_inference.yaml"]
    missing = [str(p.relative_to(bc.ROOT)) for p in required if not p.is_file()]
    if missing:
        raise bc.BundleError("묶음에 필요한 파일이 없습니다: " + ", ".join(missing),
                             "zip을 처음부터 다시 풀어 주세요(일부만 복사하면 이렇게 됩니다).")
    log.info("가중치 파일 지문(SHA-256)을 확인합니다 (약 300MB)...")
    model = bc.check_model_files()
    if not model["weights_ok"]:
        raise bc.BundleError("모델 가중치 파일이 없거나 원본과 다릅니다(SHA-256 불일치).",
                             "개발 PC에서 받은 zip을 다시 복사해 풀어 주세요. 복사 중 손상됐을 수 있습니다.")
    log.ok(f"가중치 SHA-256 일치 ({model['weights_sha256'][:16]}..., {model['weights_bytes'] / 1e6:.0f}MB)")
    if not model["upstream_ok"]:
        raise bc.BundleError("공식 모델 코드(upstream)의 commit이 다릅니다.", "zip을 다시 풀어 주세요(.git 폴더 포함).")
    log.ok(f"공식 모델 코드 commit 일치 ({model['upstream_commit'][:12]})")
    try:
        ward = json.loads(bc.WARD_COMMANDS.read_text(encoding="utf-8"))
        count = len(ward["fixed"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise bc.BundleError("병실 명령 목록 파일을 읽지 못했습니다.") from exc
    if count != bc.EXPECTED_WARD_PHRASES:
        raise bc.BundleError(f"병실 명령이 {count}개입니다({bc.EXPECTED_WARD_PHRASES}개여야 합니다).")
    log.ok(f"병실 명령 {count}개, 빌드한 화면(frontend/dist) 있음")
    manifest = bc.load_json(bc.DEPLOY / bc.MANIFEST_NAME, {}) or {}
    if manifest:
        log.info(f"묶음 정보: {manifest.get('built_at')} 빌드, 코드 commit {str(manifest.get('source_commit'))[:12]}")
    return model


def check_not_running(log: Log) -> None:
    """시연 서버·화면이 켜져 있으면 멈춘다(가상환경 파일이 잠겨 설치가 깨진다)."""
    back, _ = bc.backend_state()
    front = bc.frontend_state()
    if back == "ours" or front == "ours":
        raise bc.BundleError("시연 프로그램(서버 또는 화면)이 켜져 있습니다.",
                             "시연_중지.bat을 실행하거나 'Silent Talk 서버'·'Silent Talk 화면' 창을 닫은 뒤 다시 실행하세요.")
    if back == "busy":
        log.warn(f"{bc.BACKEND_PORT}번 포트를 다른 프로그램이 쓰고 있습니다. 설치는 계속하지만 시연 전에 그 프로그램을 끄세요.")
    if front == "busy":
        log.warn(f"{bc.FRONTEND_PORT}번 포트를 다른 프로그램이 쓰고 있습니다. 설치는 계속하지만 시연 전에 그 프로그램을 끄세요.")
    if back == "free" and front == "free":
        log.ok(f"시연 프로그램이 꺼져 있습니다 (포트 {bc.BACKEND_PORT}·{bc.FRONTEND_PORT} 비어 있음).")


def choose_flavor(device: str, log: Log, infer_spec_marker: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """그래픽카드와 옵션으로 인식 환경의 torch 종류를 정한다."""
    gpu = bc.nvidia_info()
    if gpu["present"]:
        log.info(f"그래픽카드: {gpu['name']} ({gpu.get('memory_mb') or '?'}MB, 세대 {gpu.get('compute_cap') or '모름'}), "
                 f"드라이버 {gpu['driver']}, CUDA 지원 {gpu['cuda'] or '표시 없음'}")
    else:
        log.info(f"NVIDIA 그래픽카드를 찾지 못했습니다 ({gpu['error']}). 게이밍 노트북이면 그래픽 모드가 "
                 "'절전(Eco)·내장 그래픽 전용'인지 확인하세요(표준·Optimus로 바꾼 뒤 다시 설치하면 GPU판을 씁니다).")
    if device == "cpu":
        log.ok("옵션 --device cpu: CPU판 torch를 설치합니다.")
        return bc.CPU_FLAVOR, gpu
    flavor, reason = bc.choose_torch_flavor(gpu)
    if device == "cuda":
        if flavor != bc.CUDA_FLAVOR:
            raise bc.BundleError("옵션 --device cuda를 줬지만 GPU판을 쓸 수 없습니다: " + reason,
                                 "NVIDIA 드라이버를 최신으로 설치하거나 --device 옵션 없이 다시 실행하세요.")
        log.ok("옵션 --device cuda: GPU판(cu128) torch를 설치합니다.")
        return bc.CUDA_FLAVOR, gpu
    if flavor == bc.CUDA_FLAVOR and infer_spec_marker.get("cuda_fallback") and infer_spec_marker.get("torch") == "cpu":
        log.warn("이전 설치에서 GPU 계산 확인이 실패해 CPU판을 유지합니다 "
                 f"(이유: {infer_spec_marker['cuda_fallback']}). GPU로 다시 시도하려면: 설치.bat --device cuda")
        return bc.CPU_FLAVOR, gpu
    log.ok(reason)
    return flavor, gpu


def space_report(flavor: str) -> list[tuple[str, float, float]]:
    """그 torch 종류로 설치할 때 드라이브별 [(드라이브, 남은 GB, 필요 GB)].

    필요량은 묶음 드라이브(가상환경)와 pip 임시·보관 폴더 드라이브(내려받는 파일)를 따로 세고, 같은 드라이브면 더한다.
    """
    temp = Path(os.environ.get("TEMP") or os.environ.get("TMP") or str(bc.ROOT))
    needs = bc.space_requirements(flavor, bc.ROOT.resolve(), [temp, bc.pip_cache_dir()])
    return [(drive, bc.free_gb(entry["path"]), entry["need_gb"]) for drive, entry in needs.items()]


def check_space_and_internet(log: Log, need_install: bool, flavor: str = bc.CPU_FLAVOR) -> Callable[[], None]:
    """할 일이 있을 때만 디스크와 인터넷을 확인한다. 돌려준 함수를 부르면 인터넷 확인을 (한 번만) 한다.

    Args:
        flavor: 새로 설치할 인식 환경의 torch 종류(GPU판이면 내려받는 파일이 커서 더 많은 공간이 필요하다).
    """
    state = {"online": None}
    if not need_install:
        log.ok("새로 설치할 것이 없어 디스크·인터넷 확인을 건너뜁니다.")
    else:
        for drive, free, need in space_report(flavor):
            if not free >= need:
                kind = "GPU판" if flavor == bc.CUDA_FLAVOR else "CPU판"
                raise bc.BundleError(f"{drive} 드라이브 남은 공간이 {free:.1f}GB입니다 ({kind} 설치에 {need:.0f}GB 이상 "
                                     "필요: 가상환경과 내려받는 파일·임시 파일).",
                                     "필요 없는 파일을 지워 공간을 만든 뒤 다시 실행하세요. 명령 프롬프트에서 "
                                     "'py -3.11 -m pip cache purge'로 예전에 내려받은 파일 보관분을 지울 수 있습니다.")
            log.ok(f"{drive} 드라이브 남은 공간 {free:.0f}GB (필요 {need:.0f}GB)")

    def need_online() -> None:
        if state["online"] is True:
            return
        targets = ["https://pypi.org/simple/pip/", "https://files.pythonhosted.org/",
                   "https://download.pytorch.org/whl/cpu/torch/"]
        for url in targets:
            try:
                request = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(request, timeout=15):  # noqa: S310 - 고정된 패키지 저장소 주소
                    pass
            except urllib.error.HTTPError:
                pass  # 응답이 왔으면 연결은 된다
            except (urllib.error.URLError, OSError, ValueError) as exc:
                raise bc.BundleError(f"인터넷에 연결되지 않았습니다 ({url}: {getattr(exc, 'reason', exc)}).",
                                     "설치할 때 한 번은 인터넷이 필요합니다. Wi-Fi를 연결한 뒤 설치.bat을 다시 실행하세요. "
                                     "학교·회사 망에서 막히면 휴대폰 테더링으로 해 보세요.") from exc
        state["online"] = True
        log.ok("인터넷 연결 확인 (pypi.org, download.pytorch.org)")

    if need_install:
        need_online()
    return need_online


def cuda_works(spec: VenvSpec, log: Log) -> tuple[bool, dict[str, Any]]:
    """인식 환경에서 실제 CUDA 계산(행렬곱·3D 합성곱)이 되는지."""
    out = _run([str(bc.venv_python(spec.name)), "-c", CUDA_TEST], log, capture=True, timeout=300, env=_pip_env())
    try:
        data = json.loads((out.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        data = {"ok": False, "error": (out.stderr or "결과 없음")[-300:]}
    if out.stderr and "not compatible" in out.stderr:
        data["warning"] = out.stderr.strip()[-300:]
    if not data.get("torch") and bc.text_mentions_dll_problem(f"{data.get('error')} {out.stderr}"):
        # torch 자체를 못 불러온 것은 GPU 문제가 아니다. CPU판으로 바꿔도 같은 오류가 나므로 여기서 알린다.
        raise bc.BundleError(f"torch를 불러오지 못했습니다(DLL 오류): {str(data.get('error'))[:200]}",
                             bc.vc_redist_hint())
    return bool(data.get("ok")), data


def ensure_git(mode: str, log: Log) -> dict[str, Any]:
    """인식 워커의 commit 확인에 쓸 git을 준비한다. 진짜 git이 맞는 값을 주면 그대로 쓰고, 아니면 대체 명령을 만든다."""
    expected = bc.core_constants()["upstream_commit"]
    real = shutil.which("git")
    if real and mode != "always":
        try:
            out = subprocess.run([real, "-C", str(bc.UPSTREAM), "rev-parse", "HEAD"], capture_output=True, text=True,
                                 timeout=60, creationflags=NO_WINDOW)
            if out.returncode == 0 and out.stdout.strip() == expected:
                if bc.GITSHIM_DIR.exists():
                    shutil.rmtree(bc.GITSHIM_DIR, ignore_errors=True)
                log.ok(f"git이 설치돼 있어 그대로 씁니다 ({real}).")
                return {"mode": "git", "path": real}
            log.info(f"git이 있지만 commit 확인이 안 됩니다 ({(out.stderr or out.stdout).strip()[:150]}). 대체 명령을 만듭니다.")
        except (OSError, subprocess.SubprocessError) as exc:
            log.info(f"git 실행 실패({exc}). 대체 명령을 만듭니다.")
    elif mode != "always":
        log.info("git이 없습니다. 인식 워커의 commit 확인용 대체 명령을 만듭니다(Git 설치는 필요 없음).")
    if mode == "never":
        raise bc.BundleError("git으로 공식 코드 commit을 확인하지 못했습니다(--git-shim never).",
                             "Git for Windows를 설치하거나 --git-shim 옵션 없이 다시 실행하세요.")
    source = bc.RUNTIME / "gitshim_src"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bc.DEPLOY / "git_shim.py", source / "git.py")
    bc.GITSHIM_DIR.mkdir(parents=True, exist_ok=True)
    server_python = bc.venv_python(bc.VENV_SERVER)
    made = _run([str(server_python), "-c", SHIM_MAKER, str(source), str(bc.GITSHIM_DIR), str(server_python)], log,
                capture=True, timeout=120, env=_pip_env())
    shim = bc.GITSHIM_DIR / "git.exe"
    if made.returncode != 0 or not shim.is_file():
        raise bc.BundleError("git 대체 명령을 만들지 못했습니다.",
                             "Git for Windows(https://git-scm.com/download/win)를 설치한 뒤 다시 실행하세요. "
                             + (made.stderr or "")[-200:])
    # 인식 워커와 같은 조건으로 확인한다: 서버 환경변수(PATH 맨 앞에 대체 명령)를 받은 파이썬이 이름 'git'으로 부른다.
    # (Windows는 subprocess에 넘긴 env의 PATH가 아니라 부르는 프로세스 자신의 PATH로 실행 파일을 찾는다.)
    env = bc.service_env()
    resolved = shutil.which("git", path=env.get("PATH"))
    probe = ("import subprocess, sys; print(subprocess.check_output(['git', '-C', sys.argv[1], 'rev-parse', 'HEAD'], "
             "text=True).strip())")
    out = subprocess.run([str(server_python), "-c", probe, str(bc.UPSTREAM)], capture_output=True, text=True,
                         timeout=60, env=env, creationflags=NO_WINDOW)
    if out.returncode != 0 or out.stdout.strip() != expected or not resolved or Path(resolved) != shim:
        raise bc.BundleError("git 대체 명령이 맞는 commit을 내지 않습니다.",
                             f"Git for Windows를 설치한 뒤 다시 실행하세요. ({(out.stderr or '').strip()[-200:]})")
    log.ok(f"git 대체 명령 준비 완료 ({shim}) — 시연 실행 때 자동으로 씁니다.")
    return {"mode": "shim", "path": str(shim)}


def probe_worker(python: Path, mode: str, extra: list[str], timeout: float, log_path: Path) -> dict[str, Any]:
    """워커를 서버와 같은 방식으로 띄워 ready/fatal 신호를 기다린 뒤 끈다.

    Returns:
        {"ok", "seconds", "info" 또는 "error"}
    """
    env = bc.service_env()
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", TF_CPP_MIN_LOG_LEVEL="2", GLOG_minloglevel="2")
    cmd = [str(python), "-u", str(bc.WORKER_SCRIPT), "--mode", mode, *extra]
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    since = _log_size(log_path)
    messages: queue.Queue = queue.Queue()
    with open(log_path, "ab") as err:
        err.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} install probe {mode} =====\n".encode())
        err.flush()
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, cwd=str(bc.ROOT),
                                    env=env, creationflags=NO_WINDOW)
        except OSError as exc:
            return {"ok": False, "seconds": 0.0, "error": f"워커 실행 실패: {exc}"}

        def reader() -> None:
            try:
                for raw in iter(proc.stdout.readline, b""):  # type: ignore[union-attr]
                    try:
                        messages.put(json.loads(raw.decode("utf-8")))
                    except (UnicodeDecodeError, ValueError):
                        continue
            finally:
                messages.put(None)

        threading.Thread(target=reader, daemon=True).start()
        result: dict[str, Any] = {"ok": False, "error": f"{timeout:.0f}초 안에 준비되지 않았습니다"}
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                try:
                    message = messages.get(timeout=1.0)
                except queue.Empty:
                    continue
                if message is None:
                    result = {"ok": False, "error": f"워커가 바로 끝났습니다(종료 코드 {proc.poll()})"}
                    break
                if isinstance(message, dict) and message.get("type") == "ready":
                    result = {"ok": True, "info": message.get("info") or {}}
                    break
                if isinstance(message, dict) and message.get("type") == "fatal":
                    error = message.get("error") or {}
                    result = {"ok": False, "error": f"{error.get('type')}: {error.get('message')}"}
                    break
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
                proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()
    result["seconds"] = round(time.perf_counter() - started, 1)
    if not result["ok"]:
        recent = bc.read_tail(log_path, since, limit=16384)  # 이번 실행에서 워커가 남긴 오류만 본다
        result["dll_problem"] = bc.text_mentions_dll_problem(f"{result.get('error')} {recent}")
    return result


def verify_runtime(log: Log, infer_spec: VenvSpec) -> dict[str, Any]:
    """서버 코드 import와 두 워커의 실제 기동을 확인한다."""
    out: dict[str, Any] = {}
    server = _run([str(bc.venv_python(bc.VENV_SERVER)), "-c", SERVER_IMPORT_TEST], log, cwd=bc.BACKEND,
                  env=bc.service_env(), capture=True, timeout=300)
    try:
        out["server"] = json.loads(server.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        detail = (server.stderr or "")[-500:]
        if bc.text_mentions_dll_problem(server.stderr or ""):
            last = (detail.strip().splitlines() or [""])[-1][:200]
            raise bc.BundleError(f"서버 코드를 불러오지 못했습니다(DLL 오류: {last}).", bc.vc_redist_hint()) from exc
        raise bc.BundleError("서버 코드를 불러오지 못했습니다.", detail) from exc
    log.ok(f"서버: fastapi {out['server']['fastapi']}, uvicorn {out['server']['uvicorn']}, "
           f"torch {out['server']['torch']}, opencv {out['server']['cv2']}")
    log.info("입술 추출 워커를 띄워 봅니다 (처음에는 1분 정도 걸릴 수 있습니다)...")
    roi = probe_worker(bc.venv_python(bc.VENV_ROI), "roi", [], WORKER_TIMEOUT_SEC, bc.LOG_DIR / "install_roi_worker.log")
    out["roi_worker"] = roi
    if not roi["ok"]:
        hint = f"기록: {bc.LOG_DIR / 'install_roi_worker.log'}"
        if roi.get("dll_problem"):
            hint = bc.vc_redist_hint() + " " + hint
        raise bc.BundleError(f"입술 추출 워커가 뜨지 않습니다: {roi['error']}", hint)
    info = roi["info"]
    log.ok(f"입술 추출 워커 준비 ({roi['seconds']}초): mediapipe {info.get('mediapipe')}, opencv {info.get('cv2')}, "
           f"파이썬 {info.get('python')}")
    log.info("인식 워커를 띄워 봅니다 (모델 로드·예열, 처음에는 1~3분 걸릴 수 있습니다)...")
    infer = probe_worker(bc.venv_python(bc.VENV_INFER), "infer", ["--device", "auto", "--decoder", "hybrid", "--warmup"],
                         WORKER_TIMEOUT_SEC, bc.LOG_DIR / "install_infer_worker.log")
    out["infer_worker"] = infer
    if not infer["ok"]:
        hint = f"기록: {bc.LOG_DIR / 'install_infer_worker.log'}"
        if infer.get("dll_problem"):
            hint = bc.vc_redist_hint() + " " + hint
        elif infer_spec.torch_flavor == bc.CUDA_FLAVOR:
            hint += " — GPU에서만 실패하면 '설치.bat --device cpu'로 CPU판을 설치해 보세요."
        raise bc.BundleError(f"인식 워커가 뜨지 않습니다: {infer['error']}", hint)
    info = infer["info"]
    log.ok(f"인식 워커 준비 ({infer['seconds']}초): 장치 {info.get('device')} {info.get('cuda_device') or ''}, "
           f"torch {info.get('torch')}, 모델 로드 {float(info.get('model_load_seconds') or 0):.1f}초, "
           f"예열 {float(info.get('warmup_seconds') or 0):.2f}초")
    if infer_spec.torch_flavor == bc.CUDA_FLAVOR and not str(info.get("device", "")).startswith("cuda"):
        log.warn("GPU판 torch를 설치했지만 인식 워커가 CPU로 떴습니다. 병실 모드는 CPU로도 됩니다.")
    return out


def summarize(log: Log, state: dict[str, Any]) -> None:
    """마지막 요약."""
    infer = (state.get("runtime") or {}).get("infer_worker", {}).get("info", {})
    roi = (state.get("runtime") or {}).get("roi_worker", {}).get("info", {})
    log.write("\n" + "=" * 64)
    log.write(" 설치 완료")
    log.write("=" * 64)
    log.write(f"  파이썬          : {state['python']}")
    gpu = state.get("gpu") or {}
    log.write(f"  그래픽카드      : {gpu.get('name') or '없음(NVIDIA 아님)'}")
    skipped = "(기동 확인 생략)"
    log.write(f"  인식용 torch    : {infer.get('torch') or skipped} "
              f"({'GPU판' if state['torch_flavor'] == bc.CUDA_FLAVOR else 'CPU판'})")
    log.write(f"  인식 장치       : {infer.get('device') or skipped} {infer.get('cuda_device') or ''}")
    log.write(f"  입술 추출       : mediapipe {roi.get('mediapipe') or skipped}")
    vc = (state.get("vc_runtime") or {}).get("msvcp140.dll") or {}
    log.write(f"  VC++ 런타임     : MSVCP140 {vc.get('version') or '확인 못 함'}")
    log.write(f"  가중치 지문     : 일치 ({state['model']['weights_sha256'][:16]}...)")
    log.write(f"  git             : {'설치된 git' if state['git']['mode'] == 'git' else '대체 명령(git 설치 불필요)'}")
    log.write(f"  걸린 시간       : {state['seconds'] / 60:.1f}분")
    log.write(f"  기록            : {log.path}")
    log.write("\n  다음 단계: 시연_실행.bat 을 두 번 클릭하세요. 그다음 시연_점검.bat 으로 처리 시간을 확인합니다.")
    if state["torch_flavor"] == bc.CPU_FLAVOR:
        log.write("  (CPU로 인식합니다. 병실 모드는 CPU로도 되지만 시연_점검.bat으로 처리 시간을 꼭 확인하세요.)")


def run(args: argparse.Namespace) -> int:
    """설치 전체 흐름. 성공하면 0."""
    started = time.perf_counter()
    log = Log(bc.LOG_DIR / f"install_{bc.timestamp()}.log")
    log.write("Silent Talk 병실 무음 도우미 — 시연 노트북 설치")
    log.write(f"시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    state: dict[str, Any] = {"ok": False}
    try:
        log.step(1, "파이썬·Visual C++ 재배포 패키지 확인")
        state.update(check_python(log))
        state["python"] = sys.version.split()[0]

        log.step(2, "묶음 파일 확인")
        state["model"] = check_bundle(log, args.allow_non_ascii_path)

        log.step(3, "시연 프로그램이 꺼져 있는지 확인")
        check_not_running(log)

        log.step(4, "그래픽카드 확인")
        infer_marker = bc.load_json(bc.ROOT / bc.VENV_INFER / "silenttalk_install.json", {}) or {}
        flavor, gpu = choose_flavor(args.device, log, infer_marker)
        state["gpu"], state["torch_flavor"] = gpu, flavor
        specs = [VenvSpec(bc.VENV_SERVER, "서버 환경", bc.LOCK_SERVER, bc.CPU_FLAVOR),
                 VenvSpec(bc.VENV_ROI, "입술 추출 환경", bc.LOCK_ROI, None),
                 VenvSpec(bc.VENV_INFER, "인식 환경", bc.LOCK_INFER, flavor)]

        log.step(5, "디스크·인터넷 확인")
        current = {spec.name: venv_is_current(spec) for spec in specs}
        infer_new = specs[2].torch_flavor if not current[bc.VENV_INFER] else bc.CPU_FLAVOR
        if not all(current.values()) and infer_new == bc.CUDA_FLAVOR and args.device == "auto":
            short = [row for row in space_report(bc.CUDA_FLAVOR) if not row[1] >= row[2]]
            if short and all(free >= need for _, free, need in space_report(bc.CPU_FLAVOR)):
                drive, free, need = short[0]
                log.warn(f"{drive} 드라이브 남은 공간({free:.1f}GB)이 GPU판 설치에 필요한 {need:.0f}GB보다 적어 CPU판으로 "
                         "설치합니다. 병실 모드는 CPU로도 됩니다. 공간을 비운 뒤 설치.bat을 다시 실행하면 GPU판으로 바꿉니다.")
                specs[2].torch_flavor = flavor = bc.CPU_FLAVOR
                state["torch_flavor"], state["space_fallback"] = flavor, True
                current[bc.VENV_INFER] = venv_is_current(specs[2])
                infer_new = bc.CPU_FLAVOR
        need_install = not all(current.values())
        need_online = check_space_and_internet(log, need_install, infer_new)

        state["venvs"] = []
        for number, spec in zip((6, 7, 8), specs):
            log.step(number, f"{spec.label} ({spec.name})")
            state["venvs"].append(ensure_venv(spec, log, need_online))
        infer_spec = specs[2]
        if infer_spec.torch_flavor == bc.CUDA_FLAVOR:
            ok, detail = cuda_works(infer_spec, log)
            state["cuda_test"] = detail
            if ok:
                log.ok(f"GPU 계산 확인: {detail.get('name')} (compute {detail.get('capability')}), "
                       f"torch {detail.get('torch')} CUDA {detail.get('cuda_build')}")
            elif args.device == "cuda":
                raise bc.BundleError(f"GPU 계산이 되지 않습니다: {detail.get('error') or detail}",
                                     "옵션 없이 다시 실행하면 CPU판으로 설치합니다.")
            else:
                reason = str(detail.get("error") or detail.get("warning") or "torch.cuda.is_available()=False")[:200]
                log.warn(f"GPU 계산이 되지 않아 CPU판 torch로 바꿉니다 ({reason}).")
                need_online()
                infer_spec.torch_flavor = bc.CPU_FLAVOR
                _, torch_version = _derived_requirements(infer_spec)
                install_torch(infer_spec, torch_version or "", log)
                problems = _compare_with_lock(infer_spec, _installed_versions(bc.venv_python(infer_spec.name)))
                if problems:
                    raise bc.BundleError("CPU판 torch로 바꾸지 못했습니다: " + "; ".join(problems[:3]))
                bc.save_json(_marker_path(infer_spec), {**_fingerprint(infer_spec), "complete": True,
                                                        "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                        "base_python": sys.executable})
                mark_cuda_fallback(infer_spec, reason)
                state["torch_flavor"] = bc.CPU_FLAVOR
                log.ok("CPU판 torch로 바꿨습니다.")

        log.step(9, "git 확인 (공식 모델 코드 commit 확인용)")
        state["git"] = ensure_git(args.git_shim, log)

        log.step(10, "실제 기동 확인")
        if args.skip_worker_test:
            log.warn("옵션 --skip-worker-test: 워커 기동 확인을 건너뜁니다.")
            state["runtime"] = {}
        else:
            state["runtime"] = verify_runtime(log, infer_spec)

        log.step(11, "요약")
        state["ok"] = True
        state["seconds"] = round(time.perf_counter() - started, 1)
        state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["log"] = str(log.path)
        bc.save_json(bc.INSTALL_STATE, state)
        summarize(log, state)
        return 0
    except bc.BundleError as exc:
        log.write("\n" + "!" * 64)
        log.write(f" 설치 실패: {exc.message}")
        if exc.hint:
            log.write(f" 해결 방법: {exc.hint}")
        log.write(f" 기록 파일: {log.path}")
        log.write(" 고친 뒤 설치.bat을 다시 실행하면 끝난 부분은 건너뛰고 이어서 합니다.")
        log.write("!" * 64)
        state["error"] = exc.message
        try:
            bc.save_json(bc.RUNTIME / "install_failed.json", state)
        except bc.BundleError:
            pass
        return 1
    except KeyboardInterrupt:
        log.write("\n중단했습니다. 설치.bat을 다시 실행하면 이어서 설치합니다.")
        return 1


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    bc.setup_console()
    parser = argparse.ArgumentParser(description="Silent Talk 시연 노트북 설치")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="인식 환경 torch 종류: auto(그래픽카드 보고 결정, 기본) | cuda(GPU판 강제) | cpu(CPU판)")
    parser.add_argument("--git-shim", choices=("auto", "always", "never"), default="auto",
                        help="git 대체 명령: auto(git이 없거나 안 될 때만, 기본) | always(시험용) | never")
    parser.add_argument("--allow-non-ascii-path", action="store_true",
                        help="한글이 들어간 폴더 경로에서도 설치를 계속한다(권장하지 않음)")
    parser.add_argument("--skip-worker-test", action="store_true", help="마지막 워커 기동 확인을 건너뛴다")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - 예상 못 한 오류도 학생이 읽을 수 있게 알린다
        print(f"\n예상하지 못한 오류로 설치를 멈췄습니다: {type(exc).__name__}: {exc}")
        print(f"이 화면을 찍어 개발 담당에게 보내 주세요. 기록 폴더: {bc.LOG_DIR}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
