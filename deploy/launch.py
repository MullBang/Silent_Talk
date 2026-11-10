"""시연 실행·중지 (시연_실행.bat / 시연_중지.bat이 .venv-server 파이썬으로 실행한다).

실행:
  1. 설치 확인(가상환경 3개, 빌드한 화면)
  2. 8000번(서버)·5173번(화면) 포트 확인 — 이미 우리 프로그램이 켜져 있거나 켜지는 중이면 다시 켜지 않는다.
     다른 프로그램이 쓰고 있으면 멈추고 알린다. 두 번 눌러 실행기가 동시에 돌면 나중 것은 아무것도 켜지 않고 끝난다
     (deploy/runtime/launch.lock).
  3. 서버 창("Silent Talk 서버")과 화면 창("Silent Talk 화면")을 새 콘솔로 켠다(--background면 창 없이).
  4. /health에서 두 워커(입술 추출·인식)가 실제로 준비될 때까지 기다린다.
  5. Chrome(없으면 Edge)으로 환자 창 http://localhost:5173/#ward 와 간호사 창 http://localhost:5173/#nurse 를
     따로 연다. 두 창은 같은 주소·같은 브라우저 프로필이어야 BroadcastChannel로 통한다.
중지(--stop): 이 묶음이 켠 서버·화면 프로세스만 끈다(다른 프로그램은 건드리지 않는다).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402

NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
SERVER_TITLE = "Silent Talk 서버"
FRONT_TITLE = "Silent Talk 화면"
SERVER_SCRIPT = "run_server.py"
FRONT_SCRIPT = "serve_frontend.py"
OUR_SCRIPTS = (SERVER_SCRIPT, FRONT_SCRIPT)
APP_PATHS = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
LOCK_FILE = bc.RUNTIME / "launch.lock"
LOCK_STALE_SEC = 900.0
"""실행기 잠금이 이보다 오래됐으면(실행기가 비정상 종료) 무시한다. 모델 대기 상한(기본 360초)보다 길다."""
BROWSER_PARTICLES = {"Chrome": ("Chrome으로", "Chrome을"), "Edge": ("Edge로", "Edge를")}
"""브라우저 이름 뒤 조사(으로/로, 을/를)."""
WORKER_NAMES = {"roi": "입술 추출", "infer": "인식"}


def fail(message: str, hint: str = "") -> int:
    """실패 문구를 보이고 1을 돌려준다."""
    print(f"\n[실패] {message}")
    if hint:
        print(f"       해결 방법: {hint}")
    return 1


def _record_pid(kind: str, pid: int) -> None:
    """켠 프로세스 번호를 기록한다(시연_중지.bat이 이 묶음 프로세스만 끄도록)."""
    try:
        data = bc.load_json(bc.PIDS_FILE, {}) or {}
        data[kind] = {"pid": pid, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        bc.save_json(bc.PIDS_FILE, data)
    except bc.BundleError as exc:
        print(f"  [주의] 프로세스 기록 실패: {exc.message}")


def start_window(kind: str, title: str, script: Path, env: dict[str, str], background: bool) -> int:
    """서버 또는 화면 서버를 새 콘솔 창(또는 창 없이)으로 켠다.

    Returns:
        켠 프로세스 번호(창 모드는 cmd.exe, 창 없는 모드는 python).
    """
    python = bc.venv_python(bc.VENV_SERVER)
    if background:
        bc.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = open(bc.LOG_DIR / f"{kind}.log", "ab")  # noqa: SIM115 - 자식 프로세스가 계속 쓴다
        proc = subprocess.Popen([str(python), str(script)], cwd=str(bc.ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW | NEW_GROUP)
    else:
        # cmd /s /k "..." : 바깥 따옴표를 벗긴 나머지를 실행하고 창을 남긴다(오류가 나도 창에서 읽을 수 있다).
        command = f'cmd.exe /s /k "title {title} && "{python}" "{script}""'
        proc = subprocess.Popen(command, cwd=str(bc.ROOT), env=env, creationflags=NEW_CONSOLE)
    _record_pid(kind, proc.pid)
    return proc.pid


def find_browser(prefer: str = "auto") -> tuple[str, str] | None:
    """Chrome 우선, 없으면 Edge. (이름, 실행 파일 경로) 또는 None."""
    order = {"auto": ("chrome", "edge"), "chrome": ("chrome",), "edge": ("edge",)}[prefer]
    exe_names = {"chrome": "chrome.exe", "edge": "msedge.exe"}
    labels = {"chrome": "Chrome", "edge": "Edge"}
    local = os.environ.get("LOCALAPPDATA", "")
    program_files = [os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")]
    folders = {"chrome": [Path(p) / "Google" / "Chrome" / "Application" for p in [*program_files, local] if p],
               "edge": [Path(p) / "Microsoft" / "Edge" / "Application" for p in [*program_files, local] if p]}
    for key in order:
        exe = exe_names[key]
        try:
            import winreg  # noqa: PLC0415

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    value = winreg.QueryValue(hive, f"{APP_PATHS}\\{exe}")
                except OSError:
                    continue
                path = value.strip().strip('"')
                if path and Path(path).is_file():
                    return labels[key], path
        except ImportError:
            pass
        for folder in folders[key]:
            if (folder / exe).is_file():
                return labels[key], str(folder / exe)
    return None


def open_windows(prefer: str) -> bool:
    """환자 창과 간호사 창을 같은 브라우저(같은 프로필)의 새 창 두 개로 연다."""
    found = find_browser(prefer)
    if not found:
        print("  [주의] Chrome·Edge를 찾지 못했습니다. 브라우저에서 아래 두 주소를 직접 여세요.")
        return False
    label, path = found
    with_particle, object_particle = BROWSER_PARTICLES.get(label, (label + "(으)로", label + "을(를)"))
    try:
        subprocess.Popen([path, "--new-window", bc.WARD_URL])
        time.sleep(3.0)  # 첫 창이 브라우저를 다 띄운 뒤 두 번째 창을 연다(같은 프로필이어야 두 창이 통한다)
        subprocess.Popen([path, "--new-window", bc.NURSE_URL])
    except OSError as exc:
        print(f"  [주의] {object_particle} 열지 못했습니다({exc}). 아래 두 주소를 직접 여세요.")
        return False
    print(f"  [통과] {with_particle} 환자 창과 간호사 창을 열었습니다.")
    return True


def wait_ready(timeout: float) -> dict[str, Any] | None:
    """두 워커가 준비될 때까지 /health를 본다. 준비되면 health, 워커 오류가 15초 이어지거나 시간 초과면 None."""
    started = time.monotonic()
    last_text, last_print = "", -100.0
    error_since: float | None = None
    while time.monotonic() - started < timeout:
        status, health, _ = bc.http_json(bc.HEALTH_URL, timeout=3.0)
        if status == 200 and bc.workers_ready(health):
            return health
        elapsed = time.monotonic() - started
        text = f"서버 응답함, {bc.describe_workers(health)}" if status == 200 else "서버 켜는 중"
        if text != last_text or elapsed - last_print >= 20:
            print(f"  ... {text} ({elapsed:.0f}초)", flush=True)
            last_text, last_print = text, elapsed
        errors = bc.worker_errors(health) if status == 200 else {}
        if errors:
            error_since = error_since or time.monotonic()
            if time.monotonic() - error_since > 15:
                for key, message in errors.items():
                    print(f"  [실패] {WORKER_NAMES[key]} 워커: {message}")
                return None
        else:
            error_since = None
        time.sleep(2.0)
    return None


def _pid_alive(pid: int) -> bool:
    """그 번호의 프로세스가 아직 살아 있는지(Windows)."""
    if pid <= 0:
        return False
    try:
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        return False


def acquire_start_lock(path: Path = LOCK_FILE) -> bool:
    """실행기 잠금을 잡는다. 다른 실행기가 이미 켜는 중이면 False.

    잠금 파일에는 실행기 프로세스 번호를 쓴다. 그 프로세스가 끝났거나 파일이 LOCK_STALE_SEC보다 오래됐으면 지우고 다시 잡는다.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return True  # 기록 폴더를 못 만들면 잠금 없이 진행한다(포트·프로세스 확인은 그대로 한다)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime
                text = path.read_text(encoding="ascii").strip()
            except (OSError, UnicodeDecodeError):
                age, text = LOCK_STALE_SEC + 1, ""
            if age < LOCK_STALE_SEC:
                if not text and age < 5:
                    return False  # 다른 실행기가 방금 만들고 번호를 쓰는 중
                if text.isdigit() and int(text) != os.getpid() and _pid_alive(int(text)):
                    return False
            try:
                path.unlink()
            except OSError:
                return False
            continue
        except OSError:
            return True
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return True
    return False


def release_start_lock(path: Path = LOCK_FILE) -> None:
    """이 실행기가 잡은 잠금만 푼다."""
    try:
        if path.read_text(encoding="ascii").strip() == str(os.getpid()):
            path.unlink()
    except (OSError, UnicodeDecodeError):
        pass


def _our_processes() -> dict[int, str]:
    """이 묶음의 run_server.py·serve_frontend.py를 실행 중인 파이썬 프로세스 {번호: 명령줄}.

    포트를 열기 전(서버는 torch·FastAPI를 불러오는 몇 초 동안 8000번이 비어 있다)에도 '켜지는 중'을 알아보려고 쓴다.
    창 모드의 cmd.exe는 세지 않는다(서버가 오류로 끝나도 창은 남기 때문).
    """
    script = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return {}
    found: dict[int, str] = {}
    for line in (out.stdout or "").splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit() and int(pid.strip()) != os.getpid() and _is_ours(cmd):
            found[int(pid.strip())] = cmd
    return found


def _running(processes: dict[int, str], script: str) -> bool:
    """그 스크립트를 실행 중인 프로세스가 있는지."""
    return any(script in cmd.lower() for cmd in processes.values())


def run_start(args: argparse.Namespace) -> int:
    """시연 실행."""
    print("Silent Talk 병실 무음 도우미 — 시연 실행\n")
    for name in (bc.VENV_SERVER, bc.VENV_ROI, bc.VENV_INFER):
        if not bc.venv_python(name).is_file():
            return fail(f"{name} 가상환경이 없습니다(설치가 끝나지 않았습니다).", "설치.bat을 먼저 실행하세요.")
    if not (bc.FRONTEND_DIST / "index.html").is_file():
        return fail("빌드한 화면(frontend/dist)이 없습니다.", "zip을 처음부터 다시 풀어 주세요.")
    state = bc.load_json(bc.INSTALL_STATE, {}) or {}
    if not state.get("ok"):
        print("  [주의] 설치 완료 기록이 없습니다. 실패하면 설치.bat을 다시 실행하세요.")
    device = {"auto": None, "cpu": "cpu", "cuda": "cuda:0"}[args.device]
    try:
        env = bc.service_env(device)
        thresholds = bc.load_ward_thresholds()
    except bc.BundleError as exc:
        return fail(exc.message, exc.hint)
    if thresholds is None:
        print("  [주의] 이 묶음에는 동결한 병실 기준값이 없어 코드 기본값(임시값)으로 판정합니다.")
    env["PYTHONUNBUFFERED"] = "1"
    if not acquire_start_lock():
        print("  [주의] 다른 창에서 이미 시연을 켜는 중입니다. 먼저 연 '시연 실행' 창에 '준비 완료'가 뜰 때까지 "
              "기다려 주세요. (이 창은 아무것도 새로 켜지 않습니다)")
        return 0
    try:
        return _start_locked(args, device, env)
    finally:
        release_start_lock()


def _start_locked(args: argparse.Namespace, device: str | None, env: dict[str, str]) -> int:
    """실행기 잠금을 잡은 상태에서 서버·화면을 켜고 준비를 기다린 뒤 브라우저를 연다."""
    back, health = bc.backend_state()
    if back == "busy":
        return fail(f"{bc.BACKEND_PORT}번 포트를 다른 프로그램이 쓰고 있습니다.",
                    "시연_중지.bat을 실행해 보고, 그래도 안 되면 그 프로그램을 끄거나 노트북을 다시 시작하세요.")
    front = bc.frontend_state()
    if front == "busy":
        return fail(f"{bc.FRONTEND_PORT}번 포트를 다른 프로그램이 쓰고 있습니다(개발용 화면 서버 등).",
                    "그 프로그램을 끄거나 시연_중지.bat을 실행한 뒤 다시 실행하세요.")
    processes = _our_processes() if "free" in (back, front) else {}
    if back == "ours":
        print("  [통과] 서버가 이미 켜져 있습니다(다시 켜지 않습니다).")
        running_device = ((health or {}).get("model") or {}).get("device")
        if device and running_device and not str(running_device).startswith(device.split(":")[0]):
            print(f"  [주의] 켜져 있는 서버의 인식 장치는 {running_device}입니다. 바꾸려면 시연_중지.bat 후 다시 실행하세요.")
    elif _running(processes, SERVER_SCRIPT):
        print("  [통과] 서버가 이미 켜지는 중입니다(다시 켜지 않습니다).")
    else:
        pid = start_window("server", SERVER_TITLE, bc.DEPLOY / SERVER_SCRIPT, env, args.background)
        print(f"  서버와 모델을 켭니다... ({'창 없이' if args.background else SERVER_TITLE + ' 창'}, PID {pid})")
    if front == "ours":
        print("  [통과] 화면 서버가 이미 켜져 있습니다(다시 켜지 않습니다).")
    elif _running(processes, FRONT_SCRIPT):
        print("  [통과] 화면 서버가 이미 켜지는 중입니다(다시 켜지 않습니다).")
    else:
        pid = start_window("frontend", FRONT_TITLE, bc.DEPLOY / FRONT_SCRIPT, env, args.background)
        print(f"  화면을 켭니다... ({'창 없이' if args.background else FRONT_TITLE + ' 창'}, PID {pid})")

    print("\n모델을 불러오는 중입니다. 잠시 기다려 주세요 (GPU 약 30초, CPU는 1~3분)...")
    health = wait_ready(args.timeout)
    if health is None:
        return fail(f"{args.timeout:.0f}초 안에 모델이 준비되지 않았습니다.",
                    f"'{SERVER_TITLE}' 창의 오류와 backend\\logs\\olkavs\\roi_worker.log, infer_worker.log를 확인하세요. "
                    "설치.bat을 다시 실행하면 원인을 다시 확인합니다.")
    for _ in range(10):
        if bc.frontend_state() == "ours":
            break
        time.sleep(1.0)
    else:
        return fail("화면 서버가 응답하지 않습니다.", f"'{FRONT_TITLE}' 창의 오류를 확인하세요.")
    model = health.get("model") or {}
    ward = model.get("ward_phrases") or {}
    device_text = (f"GPU ({model.get('cuda_device') or model.get('device')})"
                   if str(model.get("device") or "").startswith("cuda") else "CPU")
    print(f"  [통과] 준비 완료: 인식 장치 {device_text}, 병실 명령 {ward.get('count')}개")
    if ward.get("count") != bc.EXPECTED_WARD_PHRASES:
        print(f"  [주의] 병실 명령이 {ward.get('count')}개입니다({bc.EXPECTED_WARD_PHRASES}개여야 합니다). 시연_점검.bat을 실행하세요.")
    try:
        thresholds = bc.load_ward_thresholds()
    except bc.BundleError as exc:
        thresholds = None
        print(f"  [주의] {exc.message}")
    if thresholds is not None:
        problems = bc.thresholds_problems(ward.get("thresholds"), thresholds["values"])
        if problems:
            print(f"  [주의] 서버 병실 기준값이 묶음의 동결 값과 다릅니다: {'; '.join(problems)}. "
                  "시연_중지.bat 후 시연_실행.bat으로 다시 켜세요.")
        else:
            print(f"  [통과] 병실 기준값: 동결 값 ({bc.describe_thresholds(thresholds['values'])})")

    if not args.no_browser:
        open_windows(args.browser)
    print(f"\n  환자 화면:   {bc.WARD_URL}")
    print(f"  간호사 화면: {bc.NURSE_URL}")
    print("   - 간호사 화면에서 [소리 켜기]를 한 번 눌러 주세요.")
    print("   - 두 화면 배치면 Windows 키 + P → '확장'을 고르고 간호사 창을 프로젝터 쪽으로 옮기세요.")
    print("   - 프로젝터(HDMI)를 연결했으면 작업 표시줄 스피커 아이콘에서 소리 출력 장치가 노트북 스피커인지 확인하세요.")
    print(f"   - 카메라 없이 연습: {bc.WARD_URL}?practice=1")
    print("   - 처리 시간 확인: 시연_점검.bat")
    if args.background:
        print(f"   - 끄기: 시연_중지.bat (기록: {bc.LOG_DIR})")
    else:
        print(f"   - 끄기: '{SERVER_TITLE}'·'{FRONT_TITLE}' 창을 닫거나 시연_중지.bat")
    return 0


def _command_lines(pids: list[int]) -> dict[int, str]:
    """프로세스 번호 → 명령줄 (PowerShell CIM 조회). 실패하면 빈 사전."""
    if not pids:
        return {}
    ids = ",".join(str(int(p)) for p in pids)
    script = (f"Get-CimInstance Win32_Process | Where-Object {{ @({ids}) -contains $_.ProcessId }} | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[int, str] = {}
    for line in (out.stdout or "").splitlines():
        pid, _, cmd = line.partition("\t")
        if pid.strip().isdigit():
            result[int(pid.strip())] = cmd
    return result


def _listening_pids(ports: tuple[int, ...]) -> dict[int, int]:
    """포트 → 그 포트를 듣는 프로세스 번호 (netstat)."""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, errors="replace",
                             timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return {}
    found: dict[int, int] = {}
    for line in (out.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            for port in ports:
                if parts[1].endswith(f":{port}") and parts[4].isdigit():
                    found[port] = int(parts[4])
    return found


def _is_ours(command_line: str) -> bool:
    """이 묶음의 run_server.py / serve_frontend.py를 실행하는 명령줄인지."""
    text = command_line.lower()
    return str(bc.DEPLOY).lower() in text and any(name in text for name in OUR_SCRIPTS)


def run_stop() -> int:
    """이 묶음이 켠 서버·화면만 끈다."""
    print("Silent Talk 시연 중지\n")
    recorded = bc.load_json(bc.PIDS_FILE, {}) or {}
    pids = {int(v["pid"]) for v in recorded.values() if isinstance(v, dict) and str(v.get("pid", "")).isdigit()}
    listeners = _listening_pids((bc.BACKEND_PORT, bc.FRONTEND_PORT))
    pids |= set(listeners.values())
    pids |= set(_our_processes())  # 기록 파일이 없거나 아직 포트를 열기 전인 서버도 끈다
    lines = _command_lines(sorted(pids))
    stopped = 0
    for pid in sorted(pids):
        cmd = lines.get(pid)
        if cmd is None:
            continue  # 이미 끝난 프로세스
        if not _is_ours(cmd):
            continue
        out = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True,
                             errors="replace", creationflags=NO_WINDOW)
        if out.returncode == 0:
            stopped += 1
            print(f"  [통과] 프로세스 번호 {pid}: 하위 프로세스까지 껐습니다.")
    for _ in range(20):
        if not bc.port_open(bc.BACKEND_PORT) and not bc.port_open(bc.FRONTEND_PORT):
            break
        time.sleep(0.5)
    try:
        bc.PIDS_FILE.unlink()
    except OSError:
        pass
    still = _listening_pids((bc.BACKEND_PORT, bc.FRONTEND_PORT))
    if still:
        others = _command_lines(sorted(set(still.values())))
        for port, pid in still.items():
            print(f"  [주의] {port}번 포트를 다른 프로그램(PID {pid})이 쓰고 있습니다: {others.get(pid, '')[:120]}")
        print("         이 묶음이 켠 것이 아니라 끄지 않았습니다. 필요하면 그 창을 직접 닫으세요.")
        return 1
    print(f"  [통과] {bc.BACKEND_PORT}번·{bc.FRONTEND_PORT}번 포트가 비었습니다. (끈 프로세스 {stopped}개)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    bc.setup_console()
    parser = argparse.ArgumentParser(description="Silent Talk 시연 실행/중지")
    parser.add_argument("--stop", action="store_true", help="이 묶음이 켠 서버·화면을 끈다")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="인식 장치: auto(기본, GPU가 되면 GPU) | cpu | cuda")
    parser.add_argument("--background", action="store_true", help="콘솔 창 없이 켠다(기록은 deploy/runtime/logs)")
    parser.add_argument("--no-browser", action="store_true", help="브라우저 창을 열지 않는다")
    parser.add_argument("--browser", choices=("auto", "chrome", "edge"), default="auto")
    parser.add_argument("--timeout", type=float, default=360.0, help="모델 준비 대기 상한(초)")
    args = parser.parse_args(argv)
    try:
        return run_stop() if args.stop else run_start(args)
    except KeyboardInterrupt:
        return 1
    except Exception as exc:  # noqa: BLE001 - 예상 못 한 오류도 읽을 수 있게 알린다
        return fail(f"예상하지 못한 오류: {type(exc).__name__}: {exc}", "이 화면을 찍어 개발 담당에게 보내 주세요.")


if __name__ == "__main__":
    sys.exit(main())
