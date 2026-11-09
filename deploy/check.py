"""시연 점검 (시연_점검.bat이 .venv-server 파이썬으로 실행한다. 먼저 시연_실행.bat으로 켜 둔다).

확인하는 것:
  1. 컴퓨터: CPU·메모리·전원 연결·NVIDIA 그래픽카드·Windows 한국어 음성(음성 읽기용)
  2. 서버 상태(/health): 두 워커 준비, 인식 장치, 병실 명령 20개, 가중치 SHA·공식 코드 commit, CORS
  3. 화면 서버: index.html과 화면 스크립트가 올바른 종류(MIME)로 오는지
  4. 카메라: 0번 카메라에서 영상 한 장을 읽을 수 있는지(--no-camera로 끔. 브라우저가 쓰는 중이면 실패할 수 있다)
  5. 처리 시간: 예시 영상(deploy/samples)을 병실(ward)·문장(sentences) 묶음으로 보내 결과가 오는지,
     (P02 최종 평가 대상 영상은 폴더·파일 이름에 P02가 있으면 보내지 않는다)
     보내기 시작 → 결과 도착 시간의 중앙값·90%·최댓값 (기본 병실 7회 + 문장 3회 = 10회)
  6. 화면 파일 오프라인 점검: frontend/dist에 인터넷 주소가 남아 있는지 정적 검사
  7. 직접 확인할 항목 목록
결과는 deploy/runtime/check_reports/check_<시각>.json 에도 저장한다(개발 담당에게 보내면 된다).
예시 영상은 조정 참가자(P01)의 문장 영상이라 병실 명령이 아니다. 결과가 '고르지 않음'이어도 정상이며,
이 점검은 결과가 돌아오는지와 시간만 본다(정확도 측정이 아니다).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402

POLL_SEC = 0.3
"""결과 확인 간격(병실 화면과 같은 약 300ms)."""
WARD_TARGET_SEC = 3.0
"""기획서 10.2절: 녹화 끝 → 화면 표시 3초 이내(참고 목표)."""

LOCAL_HOSTS = ("localhost", "127.0.0.1")
CDN_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com",
             "storage.googleapis.com", "ajax.googleapis.com", "esm.sh", "cdn.skypack.dev")
"""화면 파일을 내려주는 곳으로 흔히 쓰이는 주소. 여기 주소가 있으면 내려받는 코드로 본다."""
RESOURCE_EXT = (".js", ".mjs", ".css", ".wasm", ".task", ".tflite", ".json", ".woff", ".woff2", ".ttf", ".png",
                ".jpg", ".svg", ".mp3", ".wav", ".onnx", ".bin")
"""주소가 이 확장자로 끝나면 화면이 내려받는 파일로 본다. 나머지(설명 문서·오류 안내·저작권 주석)는 내려받지 않는다."""
URL_RE = re.compile(rb"https?://[A-Za-z0-9.\-]+(?::\d+)?[^\s\"'`<>)\\]*")


class Report:
    """점검 결과를 모으고 화면에 찍는다."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.data: dict[str, Any] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    def section(self, title: str) -> None:
        """구역 제목."""
        print(f"\n== {title}")

    def add(self, level: str, text: str) -> None:
        """level: 통과|주의|실패|정보."""
        self.items.append({"level": level, "text": text})
        mark = {"통과": "[통과]", "주의": "[주의]", "실패": "[실패]"}.get(level, "  -  ")
        print(f"  {mark} {text}", flush=True)

    def count(self, level: str) -> int:
        """그 수준의 항목 수."""
        return sum(1 for item in self.items if item["level"] == level)


def computer_info(report: Report) -> None:
    """CPU·메모리·전원·GPU."""
    report.section("1. 컴퓨터")
    info: dict[str, Any] = {"os": bc.windows_name(), "cores": os.cpu_count()}
    try:
        import winreg  # noqa: PLC0415

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["cpu"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        info["cpu"] = platform.processor()
    try:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            info["ram_gb"] = round(status.ullTotalPhys / 1024 ** 3, 1)
            info["ram_free_gb"] = round(status.ullAvailPhys / 1024 ** 3, 1)

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        power = PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):  # type: ignore[attr-defined]
            info["ac_power"] = {0: False, 1: True}.get(power.ACLineStatus)
            info["battery_percent"] = None if power.BatteryLifePercent == 255 else power.BatteryLifePercent
            info["battery_saver"] = bool(power.SystemStatusFlag & 1)
    except (AttributeError, OSError, ValueError):
        pass
    report.add("정보", f"CPU {info.get('cpu')} (논리 코어 {info.get('cores')}), 메모리 {info.get('ram_gb')}GB "
                       f"(남은 {info.get('ram_free_gb')}GB), {info['os']}")
    if info.get("ac_power") is False:
        report.add("주의", f"전원 어댑터가 연결돼 있지 않습니다(배터리 {info.get('battery_percent')}%). 시연 때는 꼭 연결하세요.")
    elif info.get("ac_power") is True:
        report.add("통과", "전원 어댑터 연결됨")
    if info.get("battery_saver"):
        report.add("주의", "배터리 절약 모드가 켜져 있습니다. 끄면 처리가 빨라질 수 있습니다.")
    gpu = bc.nvidia_info()
    info["gpu"] = gpu
    if gpu["present"]:
        report.add("정보", f"그래픽카드 {gpu['name']} ({gpu.get('memory_mb')}MB), 드라이버 {gpu['driver']}, "
                           f"CUDA 지원 {gpu['cuda'] or '표시 없음'}")
    else:
        report.add("정보", "NVIDIA 그래픽카드 없음 — CPU로 인식합니다(병실 모드는 CPU로도 됩니다).")
    state = bc.load_json(bc.INSTALL_STATE, {}) or {}
    if state:
        info["install"] = {"torch_flavor": state.get("torch_flavor"), "finished_at": state.get("finished_at")}
        report.add("정보", f"설치 기록: {state.get('finished_at')}, 인식용 torch "
                           f"{'GPU판' if state.get('torch_flavor') == bc.CUDA_FLAVOR else 'CPU판'}")
    else:
        report.add("주의", "설치 완료 기록(deploy/runtime/install_state.json)이 없습니다. 설치.bat을 먼저 실행하세요.")
    voices = korean_voices()
    info["korean_voices"] = voices
    if voices:
        report.add("통과", f"Windows 한국어 음성 있음 ({', '.join(sorted(set(voices))[:2])}) — 음성 읽기는 인터넷 없이 됩니다")
    else:
        report.add("주의", "Windows 한국어 음성을 찾지 못했습니다. 음성 읽기를 쓰려면 인터넷이 될 때 설정 > 시간 및 언어 > "
                           "음성 > 음성 추가 > 한국어를 설치하세요(음성 읽기는 선택 기능이며 꺼도 시연은 됩니다).")
    report.data["computer"] = info


VOICE_KEYS = (r"SOFTWARE\Microsoft\Speech\Voices\Tokens", r"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens")
"""Windows 음성 목록 위치(데스크톱 음성·OneCore 음성). Chrome·Edge는 이 음성을 '기기 음성(localService)'으로 보인다."""


def korean_voices() -> list[str]:
    """Windows에 설치된 한국어(언어 코드 412) 음성 이름 목록. 확인하지 못하면 빈 목록.

    병실 화면의 음성 읽기는 인터넷 없이 되는 기기 음성(ko-KR, localService)만 쓰므로, 이 음성이 없으면 소리 읽기가 안 된다.
    """
    names: list[str] = []
    try:
        import winreg  # noqa: PLC0415
    except ImportError:
        return names
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for base in VOICE_KEYS:
            try:
                with winreg.OpenKey(hive, base) as tokens:
                    count = winreg.QueryInfoKey(tokens)[0]
                    for index in range(count):
                        token = winreg.EnumKey(tokens, index)
                        try:
                            with winreg.OpenKey(tokens, token + r"\Attributes") as attributes:
                                language = str(winreg.QueryValueEx(attributes, "Language")[0])
                                try:
                                    name = str(winreg.QueryValueEx(attributes, "Name")[0])
                                except OSError:
                                    name = token
                        except OSError:
                            continue
                        if "412" in [part.strip().lower() for part in language.split(";")]:
                            names.append(name)
            except OSError:
                continue
    return names


def server_status(report: Report) -> dict[str, Any] | None:
    """/health 확인. 서버가 꺼져 있으면 None."""
    report.section("2. 서버 상태")
    status, health, _ = bc.http_json(bc.HEALTH_URL, timeout=5.0)
    report.data["health"] = health
    if status != 200 or not isinstance(health, dict):
        report.add("실패", "서버가 응답하지 않습니다. 먼저 시연_실행.bat을 실행하고 '준비 완료'가 뜬 뒤 다시 점검하세요.")
        return None
    model = health.get("model") or {}
    if health.get("model_backend") == "olkavs":
        report.add("통과", "서버 응답, 공식 인식 모델(olkavs) 사용")
    else:
        report.add("실패", f"인식 모델이 olkavs가 아닙니다: {health.get('model_backend')}")
    if bc.workers_ready(health):
        report.add("통과", f"입술 추출·인식 워커 준비됨 (입술 검출 {model.get('roi_profile')})")
    else:
        report.add("실패", f"워커가 준비되지 않았습니다({bc.describe_workers(health)}). 켜는 중이면 1~3분 뒤 다시 점검하고, "
                           "오류가 보이면 시연_중지.bat 후 시연_실행.bat을 다시 실행하세요.")
    device = str(model.get("device") or "")
    if device.startswith("cuda"):
        report.add("통과", f"인식 장치 GPU: {model.get('cuda_device')}")
    elif device:
        report.add("정보", "인식 장치 CPU (그래픽카드 없음 또는 CPU판 torch)")
    ward = model.get("ward_phrases") or {}
    if ward.get("enabled") and ward.get("count") == bc.EXPECTED_WARD_PHRASES:
        report.add("통과", f"병실 명령 {ward.get('count')}개 (목록 {ward.get('version')})")
    else:
        report.add("실패", f"병실 명령 목록 이상: {ward}")
    try:
        expected = bc.core_constants()
        if model.get("checkpoint_sha256") == expected["checkpoint_sha256"]:
            report.add("통과", "가중치 SHA-256 일치")
        else:
            report.add("실패", f"가중치 SHA-256 불일치: {model.get('checkpoint_sha256')}")
        if model.get("upstream_commit") == expected["upstream_commit"]:
            report.add("통과", "공식 모델 코드 commit 일치")
        else:
            report.add("실패", f"공식 모델 코드 commit 불일치: {model.get('upstream_commit')}")
    except bc.BundleError as exc:
        report.add("실패", exc.message)
    _, _, headers = bc.http_json(bc.HEALTH_URL, timeout=5.0, headers={"Origin": bc.APP_ORIGIN})
    if headers.get("access-control-allow-origin") == bc.APP_ORIGIN:
        report.add("통과", f"CORS: {bc.APP_ORIGIN} 허용")
    else:
        report.add("실패", f"CORS 응답에 {bc.APP_ORIGIN}가 없습니다: {headers.get('access-control-allow-origin')}")
    return health


def frontend_status(report: Report) -> None:
    """화면 서버가 index.html과 스크립트를 올바른 MIME으로 주는지."""
    report.section("3. 화면 서버")
    status, body, _ = bc.http_bytes(bc.FRONT_URL, timeout=5.0)
    if status != 200 or b'id="root"' not in body:
        report.add("실패", f"화면 서버가 응답하지 않습니다({bc.FRONT_URL}). 시연_실행.bat으로 켜 주세요.")
        return
    report.add("통과", f"화면 서버 응답 ({bc.FRONT_URL})")
    scripts = re.findall(rb'src="(/assets/[^"]+\.js)"', body)
    if not scripts:
        report.add("실패", "index.html에서 화면 스크립트를 찾지 못했습니다.")
        return
    url = bc.FRONT_URL.rstrip("/") + scripts[0].decode("ascii", "replace")
    status, data, headers = bc.http_bytes(url, timeout=10.0)
    kind = headers.get("content-type", "")
    if status == 200 and "javascript" in kind and data:
        report.add("통과", f"화면 스크립트 {len(data) / 1e3:.0f}KB, 종류 {kind}")
    else:
        report.add("실패", f"화면 스크립트를 받지 못했거나 종류가 틀립니다: {status} {kind}")


def camera_check(report: Report) -> None:
    """0번 카메라를 열어 영상 한 장을 읽는다(저장하지 않는다)."""
    report.section("4. 카메라")
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        report.add("주의", "opencv가 없어 카메라를 확인하지 못했습니다.")
        return
    cap = None
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ok, frame = (cap.read() if cap.isOpened() else (False, None))
        if ok and frame is not None:
            report.add("통과", f"0번 카메라에서 영상을 읽었습니다 ({frame.shape[1]}x{frame.shape[0]}). 화면에서는 브라우저가 "
                               "720p 이상으로 다시 엽니다.")
            report.data["camera"] = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
        else:
            report.add("주의", "카메라를 열지 못했습니다. 환자 창이 카메라를 쓰는 중이면 정상입니다. 아니면 카메라 연결·"
                               "Windows 설정 > 개인정보 > 카메라 권한을 확인하세요.")
    except Exception as exc:  # noqa: BLE001 - 카메라 드라이버 오류도 점검 결과로 남긴다
        report.add("주의", f"카메라 확인 중 오류: {type(exc).__name__}: {exc}")
    finally:
        if cap is not None:
            cap.release()


def _multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes, content_type: str
               ) -> tuple[bytes, str]:
    """multipart/form-data 본문과 Content-Type."""
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                  f"Content-Type: {content_type}\r\n\r\n").encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send_clip(path: Path, phrase_set: str, timeout: float) -> dict[str, Any]:
    """녹화 한 건을 화면과 같은 방식(POST /api/record → 300ms 간격 결과 확인)으로 처리한다.

    Returns:
        {"clip", "phrase_set", "ok", "accept_s", "total_s", "status", "phrase", "text", "error"}
    """
    row: dict[str, Any] = {"clip": path.name, "phrase_set": phrase_set, "ok": False}
    content = path.read_bytes()
    kind = "video/mp4" if path.suffix.lower() == ".mp4" else "video/webm"
    body, content_type = _multipart({"phrase_set": phrase_set}, "file", path.name, content, kind)
    request = urllib.request.Request(f"{bc.API_URL}/record", data=body, method="POST",
                                     headers={"Content-Type": content_type, "Origin": bc.APP_ORIGIN})
    started = time.perf_counter()
    try:
        with bc.local_urlopen(request, timeout=60) as response:  # 로컬 서버: 시스템 프록시를 거치지 않는다
            accepted = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        row["error"] = f"접수 거부 {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}"
        return row
    except (urllib.error.URLError, OSError, ValueError) as exc:
        row["error"] = f"접수 실패: {exc}"
        return row
    row["accept_s"] = round(time.perf_counter() - started, 3)
    row["duration_sec"] = accepted.get("duration_sec")
    session = accepted.get("session_id")
    deadline = started + timeout
    while time.perf_counter() < deadline:
        time.sleep(POLL_SEC)
        status, data, _ = bc.http_json(f"{bc.API_URL}/result/{session}", timeout=10.0)
        if status != 200 or not isinstance(data, dict):
            continue
        if data.get("status") in ("done", "error", "timeout"):
            row["total_s"] = round(time.perf_counter() - started, 3)
            row["status"] = data.get("status")
            results = data.get("results") or []
            first = results[0] if results else {}
            row["text"] = first.get("text")
            phrase = first.get("phrase")
            if isinstance(phrase, dict):
                row["phrase"] = {k: phrase.get(k) for k in ("id", "set", "accepted", "score", "margin", "runner_up_id")}
            row["windows"] = len(results)
            row["error"] = data.get("error_message")
            row["decoder"] = (data.get("model") or {}).get("decoder")
            row["ok"] = data.get("status") == "done"
            return row
    row["error"] = f"{timeout:.0f}초 안에 결과가 오지 않았습니다"
    return row


def select_sample_clips(folder: Path) -> tuple[list[Path], str | None]:
    """점검에 보낼 영상 목록. P02(최종 평가 대상)가 폴더 경로나 파일 이름에 있으면 하나도 보내지 않는다.

    Returns:
        (영상 목록, 거부 사유 또는 None)
    """
    if "p02" in str(folder).lower():
        return [], str(folder)
    try:
        clips = sorted(p for p in folder.glob("*") if p.is_file() and p.suffix.lower() in (".webm", ".mp4"))
    except OSError:
        return [], None
    sealed = [p.name for p in clips if "p02" in p.name.lower()]
    if sealed:
        return [], ", ".join(sealed[:3]) + (f" 외 {len(sealed) - 3}개" if len(sealed) > 3 else "")
    return clips, None


def latency_check(report: Report, args: argparse.Namespace, health_ok: bool) -> None:
    """예시 영상으로 처리 시간을 잰다."""
    report.section("5. 처리 시간 (예시 영상)")
    folder = Path(args.samples) if args.samples else bc.SAMPLES
    clips, refused = select_sample_clips(folder)
    if refused:
        report.add("실패", f"P02(최종 평가 대상) 영상은 점검에 쓰지 않습니다: {refused}. P01 예시 영상 폴더를 주세요.")
        report.data["latency_refused"] = refused
        return
    if not clips:
        report.add("주의", f"예시 영상이 없습니다({folder}). 처리 시간은 환자 화면에서 직접 10번 녹화해 재 주세요.")
        return
    if not health_ok:
        report.add("실패", "서버가 준비되지 않아 처리 시간을 재지 못했습니다.")
        return
    plan = [("ward", i) for i in range(args.ward_runs)] + [("sentences", i) for i in range(args.sentence_runs)]
    rows = []
    for number, (phrase_set, index) in enumerate(plan, start=1):
        clip = clips[index % len(clips)]
        row = send_clip(clip, phrase_set, args.timeout)
        rows.append(row)
        if row["ok"]:
            phrase = row.get("phrase") or {}
            kind = "명령" if phrase_set == "ward" else "문장"
            picked = (f"{kind} {phrase.get('id')} {'고름' if phrase.get('accepted') else '고르지 않음'}"
                      if phrase else "지정 문장 결과 없음")
            print(f"  {number:2d}. {phrase_set:9s} {clip.name:28s} {row['total_s']:6.2f}초  ({picked}, "
                  f"창 {row.get('windows')}개, 디코더 {row.get('decoder')})", flush=True)
        else:
            print(f"  {number:2d}. {phrase_set:9s} {clip.name:28s}  실패: {row.get('error')}", flush=True)
    report.data["latency_runs"] = rows
    summary: dict[str, Any] = {}
    for phrase_set in ("ward", "sentences"):
        done = [r for r in rows if r["phrase_set"] == phrase_set]
        if not done:
            continue
        times = [r["total_s"] for r in done if r["ok"]]
        failed = len(done) - len(times)
        stats = {"runs": len(done), "ok": len(times), "median_s": bc.median(times), "p90_s": bc.percentile(times, 90),
                 "max_s": max(times) if times else None,
                 "with_phrase": sum(1 for r in done if r.get("phrase"))}
        summary[phrase_set] = stats
        label = "병실(ward)" if phrase_set == "ward" else "문장(sentences)"
        if failed:
            report.add("실패", f"{label}: {len(done)}회 중 {failed}회 결과를 받지 못했습니다.")
        if times:
            line = (f"{label} {len(times)}회: 중앙값 {stats['median_s']:.2f}초, 90% {stats['p90_s']:.2f}초, "
                    f"최대 {stats['max_s']:.2f}초 (지정 문장 결과 {stats['with_phrase']}/{len(done)})")
            if phrase_set == "ward":
                level = "통과" if (stats["p90_s"] or 0) <= WARD_TARGET_SEC else "주의"
                report.add(level, line + f" — 참고 목표 {WARD_TARGET_SEC:.0f}초 이내")
                if stats["with_phrase"] < len(done):
                    report.add("주의", "병실 묶음인데 지정 문장 결과(phrase)가 없는 녹화가 있습니다(검출 실패·짧은 입력일 수 있음).")
            else:
                report.add("정보", line + " — 녹화 화면(받아쓰기 포함)용, 병실 시연과 무관")
    report.data["latency_summary"] = summary
    report.add("정보", "예시 영상은 8초 문장 녹화라 병실 녹화(5초)보다 조금 길게 걸릴 수 있습니다. 시간은 보내기 시작부터 "
                       "결과 도착까지이며 영상 업로드·입술 추출·인식·결과 확인 간격(0.3초)을 포함합니다.")


def _is_loadable(url: str, file_suffix: str, host: str) -> bool:
    """화면이 실제로 내려받을 수 있는 주소인지 추정한다.

    index.html·CSS 안의 주소는 모두 내려받는 것으로 본다. JS 안에서는 CDN 주소이거나 파일 확장자로 끝나는 주소만
    내려받는 것으로 본다(나머지는 오류 안내·설명 문서·이름공간 문자열).
    """
    if file_suffix in (".html", ".css"):
        return True
    if host in CDN_HOSTS:
        return True
    path = re.sub(r"^https?://[^/]+", "", url).split("?", 1)[0].split("#", 1)[0].lower()
    return path.endswith(RESOURCE_EXT)


def scan_dist(report: Report) -> None:
    """빌드한 화면 파일에 인터넷이 있어야 내려받을 수 있는 주소가 남았는지 정적 검사."""
    report.section("6. 화면 파일 오프라인 점검 (정적 검사)")
    if not bc.FRONTEND_DIST.is_dir():
        report.add("실패", f"화면 폴더가 없습니다: {bc.FRONTEND_DIST}")
        return
    loads: dict[str, dict[str, Any]] = {}
    docs: dict[str, int] = {}
    files = 0
    for path in bc.FRONTEND_DIST.rglob("*"):
        suffix = path.suffix.lower()
        if suffix not in (".js", ".mjs", ".css", ".html", ".json"):
            continue
        files += 1
        try:
            data = path.read_bytes()
        except OSError:
            continue
        relative = path.relative_to(bc.FRONTEND_DIST).as_posix()
        for match in URL_RE.finditer(data):
            url = match.group(0).decode("ascii", "replace")
            host = re.sub(r"^https?://", "", url).split("/", 1)[0].split(":", 1)[0].lower()
            if host in LOCAL_HOSTS:
                continue
            if _is_loadable(url, suffix, host):
                entry = loads.setdefault(host, {"count": 0, "files": set(), "examples": []})
                entry["count"] += 1
                entry["files"].add(relative)
                if url[:160] not in entry["examples"] and len(entry["examples"]) < 3:
                    entry["examples"].append(url[:160])
            else:
                docs[host] = docs.get(host, 0) + 1
    report.data["dist_scan"] = {
        "files": files, "doc_hosts": dict(sorted(docs.items())),
        "load_hosts": {h: {"count": v["count"], "files": sorted(v["files"]), "examples": v["examples"]}
                       for h, v in sorted(loads.items())}}
    if docs:
        shown = ", ".join(sorted(docs)[:8]) + (f" 외 {len(docs) - 8}곳" if len(docs) > 8 else "")
        report.add("정보", f"파일 {files}개 검사. 설명·오류 안내·주석 속 주소(내려받지 않음): {shown}")
    risky = 0
    for host, entry in sorted(loads.items()):
        examples = entry["examples"]
        if host == "storage.googleapis.com" and all("face_landmarker" in e for e in examples):
            report.add("정보", "storage.googleapis.com(얼굴 랜드마크 모델): 실시간 웹캠(compact) 화면 전용입니다. "
                               "병실(#ward)·간호사(#nurse)·녹화 화면은 쓰지 않아 오프라인 시연에 영향이 없습니다.")
            continue
        risky += 1
        report.add("주의", f"인터넷에서 내려받는 주소 {host} {entry['count']}곳 ({examples[0]}, 파일 "
                           f"{', '.join(sorted(entry['files']))}). 오프라인 리허설에서 이 기능이 되는지 확인하세요.")
    if not risky:
        report.add("통과", "병실·간호사 화면이 인터넷에서 내려받는 파일 주소가 없습니다.")


def manual_checklist(report: Report) -> None:
    """사람이 직접 확인할 항목."""
    report.section("7. 직접 확인할 것 (시연 전날·당일)")
    items = [
        "카메라 권한: 환자 창에서 처음 녹화할 때 '허용'을 눌렀는지 (주소창 왼쪽 아이콘 → 카메라 '허용')",
        "간호사 창에서 [소리 켜기]를 한 번 눌렀는지 (안 누르면 알림음이 나지 않습니다)",
        "두 창이 모두 http://localhost:5173 주소인지 (127.0.0.1로 열면 두 창이 연결되지 않습니다)",
        "프로젝터: Windows 키 + P → '확장', 간호사 창을 프로젝터 쪽으로 옮겼는지",
        "프로젝터(HDMI) 연결 뒤 작업 표시줄 스피커 아이콘 → 출력 장치를 노트북 스피커로 (프로젝터로 소리가 가면 알림음이 안 들립니다)",
        "전원 어댑터 연결, 설정 > 전원에서 화면 끄기·절전 '안 함'",
        "Windows 업데이트 '일시 중지', 알림 끄기(방해 금지 모드)",
        "인터넷을 끈 상태에서 시연_실행.bat부터 녹화→결과→간호사 창까지 한 번 해 보기 "
        "(Wi-Fi만 끄기. 블루투스 스위치·키보드를 쓰면 비행기 모드 금지, 켰다면 블루투스를 다시 켜기)",
        "인터넷을 끈 상태에서 음성 읽기(한국어 음성)가 되는지",
        "스위치(키보드·발판) 입력이 녹화를 시작하는지, 3초 안에 3번 누르면 긴급 호출이 되는지",
        "환자 화면의 '호출벨과 함께 쓰는 보조 수단' 안내가 보이는지",
        "실패 대비: 백업 영상(개발 PC에서 녹화)과 이 점검 결과 파일을 USB에 챙기기",
    ]
    for item in items:
        print(f"  [ ] {item}")
    report.data["manual_checklist"] = items


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점. 실패 항목이 없으면 0."""
    bc.setup_console()
    parser = argparse.ArgumentParser(description="Silent Talk 시연 점검")
    parser.add_argument("--ward-runs", type=int, default=7, help="병실 묶음으로 보낼 횟수 (기본 7)")
    parser.add_argument("--sentence-runs", type=int, default=3, help="문장 묶음으로 보낼 횟수 (기본 3)")
    parser.add_argument("--timeout", type=float, default=240.0, help="녹화 한 건 결과 대기 상한(초)")
    parser.add_argument("--samples", help="예시 영상 폴더(기본 deploy/samples)")
    parser.add_argument("--no-camera", action="store_true", help="카메라 확인을 건너뛴다")
    args = parser.parse_args(argv)
    report = Report()
    print("Silent Talk 병실 무음 도우미 — 시연 점검")
    try:
        computer_info(report)
        health = server_status(report)
        frontend_status(report)
        if args.no_camera:
            report.section("4. 카메라")
            report.add("정보", "옵션 --no-camera: 건너뜀")
        else:
            camera_check(report)
        latency_check(report, args, health is not None and bc.workers_ready(health))
        scan_dist(report)
        manual_checklist(report)
    except KeyboardInterrupt:
        print("\n중단했습니다.")
        return 1
    except Exception as exc:  # noqa: BLE001 - 점검 도중 예상 못 한 오류도 남긴다
        report.add("실패", f"점검 중 예상하지 못한 오류: {type(exc).__name__}: {exc}")
    report.data["items"] = report.items
    report.data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = bc.RUNTIME / "check_reports" / f"check_{bc.timestamp()}.json"
    try:
        bc.save_json(out, report.data)
    except bc.BundleError as exc:
        print(f"결과 파일 저장 실패: {exc.message}")
        out = None  # type: ignore[assignment]
    failed, warned = report.count("실패"), report.count("주의")
    print("\n" + "=" * 64)
    print(f" 점검 결과: 실패 {failed}개, 주의 {warned}개, 통과 {report.count('통과')}개")
    if out:
        print(f" 결과 파일: {out}")
    print(" 실패가 있으면 시연노트북_설치안내.md의 '문제 해결'을 보세요." if failed else " 자동 점검 항목은 모두 통과했습니다.")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
