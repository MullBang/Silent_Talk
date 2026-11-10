"""시연 노트북 묶음 스크립트(deploy/) 단위 테스트. 네트워크·GPU·실제 서버 없이 돈다.

실행: py -3.11 -m pytest deploy/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))

import bundle_common as bc  # noqa: E402
import build_bundle  # noqa: E402
import check  # noqa: E402
import git_shim  # noqa: E402
import install  # noqa: E402
import launch  # noqa: E402
import serve_frontend  # noqa: E402

SHA = "af2feda3e3ac91ff3fef2454fcee662366ce9f4b"


def _fake_repo(tmp_path: Path, head: str, refs: dict[str, str] | None = None, packed: str = "") -> Path:
    """HEAD·refs·packed-refs만 있는 가짜 저장소."""
    git_dir = tmp_path / "repo" / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text(head + "\n", encoding="utf-8")
    for name, value in (refs or {}).items():
        path = git_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
    if packed:
        (git_dir / "packed-refs").write_text(packed, encoding="utf-8")
    return tmp_path / "repo"


# ---------------------------------------------------------------------------
# git 대체 명령
# ---------------------------------------------------------------------------

def test_shim_reads_loose_ref(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "ref: refs/heads/main", {"refs/heads/main": SHA})
    assert git_shim.read_head_commit(repo) == SHA


def test_shim_reads_packed_ref(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "ref: refs/heads/main",
                      packed=f"# pack-refs with: peeled fully-peeled sorted\n{SHA} refs/heads/main\n")
    assert git_shim.read_head_commit(repo) == SHA


def test_shim_reads_detached_head(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, SHA)
    assert git_shim.read_head_commit(repo) == SHA


def test_shim_missing_ref_raises(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, "ref: refs/heads/none")
    with pytest.raises(ValueError):
        git_shim.read_head_commit(repo)


def test_shim_main_rev_parse_and_unsupported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _fake_repo(tmp_path, SHA)
    assert git_shim.main(["-C", str(repo), "rev-parse", "HEAD"]) == 0
    assert capsys.readouterr().out.strip() == SHA
    assert git_shim.main(["-C", str(repo), "status"]) == 128
    assert git_shim.main(["-C", str(tmp_path / "nothing"), "rev-parse", "HEAD"]) == 128


def test_shim_matches_real_upstream_if_present() -> None:
    """개발 PC 작업 폴더에 upstream이 있으면 진짜 git과 같은 값을 내는지 본다(없으면 건너뜀)."""
    if not (bc.UPSTREAM / ".git").exists():
        pytest.skip("upstream 없음")
    assert git_shim.read_head_commit(bc.UPSTREAM) == bc.core_constants()["upstream_commit"]


# ---------------------------------------------------------------------------
# 공용 도구
# ---------------------------------------------------------------------------

def test_core_constants_match_known_values() -> None:
    values = bc.core_constants()
    assert values["checkpoint_sha256"] == "f41d742380af11b72f9cafe178cd59d0a9016c02f5a976145f5f7595d44326a5"
    assert values["upstream_commit"] == SHA


@pytest.mark.parametrize("info, flavor", [
    ({"present": False}, "cpu"),
    ({"present": True, "cuda": "13.3", "driver": "610.88", "compute_cap": "8.6"}, "cu128"),
    ({"present": True, "cuda": "12.8", "driver": "572.16"}, "cu128"),
    ({"present": True, "cuda": "12.6", "driver": "560.94"}, "cu128"),  # CUDA 12 하위 호환으로 시도(실패하면 CPU판)
    ({"present": True, "cuda": "12.0", "driver": "528.49"}, "cu128"),
    ({"present": True, "cuda": "11.8", "driver": "522.06"}, "cpu"),
    ({"present": True, "cuda": None, "driver": "610.88"}, "cu128"),
    ({"present": True, "cuda": None, "driver": "560.94"}, "cu128"),
    ({"present": True, "cuda": None, "driver": "516.94"}, "cpu"),
    ({"present": True, "cuda": None, "driver": None}, "cpu"),
    ({"present": True, "cuda": "13.0", "driver": "580.97", "compute_cap": "6.1", "name": "GTX 1060"}, "cpu"),
    ({"present": True, "cuda": "13.0", "driver": "580.97", "compute_cap": "7.0"}, "cu128"),
    ({"present": True, "cuda": "13.0", "driver": "580.97", "compute_cap": "12.0"}, "cu128"),
])
def test_choose_torch_flavor(info: dict, flavor: str) -> None:
    chosen, reason = bc.choose_torch_flavor(info)
    assert chosen == flavor and reason


def test_choose_torch_flavor_explains_old_gpu() -> None:
    _, reason = bc.choose_torch_flavor({"present": True, "cuda": "12.4", "compute_cap": "6.1", "name": "GTX 1050"})
    assert "GTX 1050" in reason and "6.1" in reason


def test_space_requirements_same_and_split_drives(tmp_path: Path) -> None:
    same = bc.space_requirements("cu128", tmp_path / "bundle", [tmp_path / "temp", tmp_path / "cache"])
    assert [entry["need_gb"] for entry in same.values()] == [16.0]
    assert [entry["need_gb"] for entry in bc.space_requirements("cpu", tmp_path, [tmp_path]).values()] == [10.0]
    split = bc.space_requirements("cu128", Path("D:/SilentTalk"), [tmp_path])
    if Path("D:/").exists() and tmp_path.drive.upper() != "D:":
        assert split["D:"]["need_gb"] == 8.0 and split[tmp_path.drive.upper()]["need_gb"] == 8.0


def test_space_report_uses_free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc, "free_gb", lambda path: 12.0)
    assert all(free >= need for _, free, need in install.space_report("cpu"))
    assert any(free < need for _, free, need in install.space_report("cu128"))


@pytest.mark.parametrize("text, disk, dll", [
    ("ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device", True, False),
    ("OSError: [WinError 112] There is not enough space on the disk", True, False),
    ('OSError: [WinError 126] Error loading "c10.dll" or one of its dependencies.', False, True),
    ("ImportError: DLL load failed while importing _framework_bindings", False, True),
    ("OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.", False, True),
    ("ERROR: No matching distribution found for torch==2.10.0+cu128", False, False),
])
def test_error_text_classifiers(text: str, disk: bool, dll: bool) -> None:
    assert bc.text_mentions_disk_full(text) is disk
    assert bc.text_mentions_dll_problem(text) is dll


@pytest.mark.parametrize("header, version", [
    ("| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |", ("13", "3")),
    ("| NVIDIA-SMI 551.86   Driver Version: 551.86   CUDA Version: 12.4 |", ("12", "4")),
])
def test_cuda_header_regex(header: str, version: tuple[str, str]) -> None:
    match = bc._CUDA_HEADER.search(header)
    assert match and (match.group(1), match.group(2)) == version


def test_parse_lock_handles_comments_and_extras(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text("# 주석\nuvicorn[standard]==0.49.0\nPyYAML==6.0.3  # 끝 주석\n\ntorch==2.10.0+cpu\n", encoding="utf-8")
    assert bc.parse_lock(lock) == {"uvicorn": "0.49.0", "pyyaml": "6.0.3", "torch": "2.10.0+cpu"}
    lock.write_text("numpy>=1\n", encoding="utf-8")
    with pytest.raises(bc.BundleError):
        bc.parse_lock(lock)


def test_percentile_and_median() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert bc.percentile(values, 90) == 9.0
    assert bc.percentile([5.0], 90) == 5.0
    assert bc.percentile([], 90) is None
    assert bc.median(values) == 5.5
    assert bc.median([3.0, 1.0, 2.0]) == 2.0


def test_service_env_drops_lipnet_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bc, "WARD_THRESHOLDS", tmp_path / "no_thresholds.json")
    monkeypatch.setenv("LIPNET_OLKAVS_WARD_MIN_SCORE", "-9")
    monkeypatch.setenv("PYTHONPATH", "x")
    monkeypatch.setattr(bc, "GITSHIM_DIR", tmp_path / "gitshim")
    env = bc.service_env("cpu")
    assert "LIPNET_OLKAVS_WARD_MIN_SCORE" not in env and "PYTHONPATH" not in env
    assert env["LIPNET_OLKAVS_DEVICE"] == "cpu" and env["LIPNET_MODEL_BACKEND"] == "olkavs"
    (tmp_path / "gitshim").mkdir()
    (tmp_path / "gitshim" / "git.exe").write_bytes(b"x")
    assert bc.service_env()["PATH"].startswith(str(tmp_path / "gitshim"))
    assert "LIPNET_OLKAVS_DEVICE" not in bc.service_env()


def test_workers_ready_requires_startup_info() -> None:
    starting = {"model": {"roi_worker": "ready", "infer_worker": "ready", "device": None, "roi_profile": None}}
    ready = {"model": {"roi_worker": "ready", "infer_worker": "ready", "device": "cpu", "roi_profile": "p"}}
    assert not bc.workers_ready(starting)
    assert bc.workers_ready(ready)
    assert not bc.workers_ready(None)


def test_worker_states_read_as_korean() -> None:
    """서버 원래 상태(stopped/dead/ready, 'start_failed at ...')를 그대로 보이지 않는다."""
    starting = {"model": {"roi_worker": "dead", "infer_worker": "stopped", "device": None, "roi_profile": None,
                          "last_error": {"roi": None, "infer": None}}}
    assert bc.describe_workers(starting) == "입술 추출 켜는 중, 인식 대기"
    assert bc.worker_errors(starting) == {}
    broken = {"model": {"roi_worker": "ready", "infer_worker": "dead", "device": None, "roi_profile": "p",
                        "last_error": {"roi": None, "infer": "start_failed at 2026-11-10 01:02:03"}}}
    assert bc.describe_workers(broken) == "입술 추출 준비됨, 인식 시작 실패 (2026-11-10 01:02:03)"
    assert bc.worker_errors(broken) == {"infer": "시작 실패 (2026-11-10 01:02:03)"}
    assert bc.worker_error_text("weird") == "오류(weird)"
    for text in (bc.describe_workers(starting), bc.describe_workers(broken), bc.describe_workers(None)):
        assert not any(word in text for word in ("dead", "stopped", "ready", "start_failed", "{"))


def test_local_requests_skip_system_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """프록시가 설정돼 있어도 127.0.0.1 요청은 프록시를 거치지 않는다(기본 urlopen은 거친다)."""
    import http.server  # noqa: PLC0415

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - 표준 이름
            body = b'{"model_backend": "olkavs"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # 아무도 듣지 않는 프록시
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
        url = f"http://127.0.0.1:{server.server_address[1]}/health"
        with pytest.raises(OSError):
            urllib.request.build_opener().open(url, timeout=5)  # 대조군: 시스템 프록시를 쓰면 실패
        status, data, _ = bc.http_json(url, timeout=5)
        assert status == 200 and data == {"model_backend": "olkavs"}
        assert bc.http_bytes(url, timeout=5)[0] == 200
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# 고정 목록 파일
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lock", [bc.LOCK_SERVER, bc.LOCK_ROI, bc.LOCK_INFER])
def test_locks_fully_pinned(lock: Path) -> None:
    pins = bc.parse_lock(lock)
    assert pins
    if lock == bc.LOCK_SERVER:
        assert pins["torch"] == "2.10.0+cpu"
        assert pins["fastapi"] and pins["uvicorn"] and pins["python-multipart"]
        lock.read_bytes().decode("ascii")  # pip가 cp949로 읽어도 깨지지 않게 ASCII만
    if lock == bc.LOCK_INFER:
        assert pins["torch"].startswith("2.10.0+")
    if lock == bc.LOCK_ROI:
        assert pins["mediapipe"] == "0.10.21"


def test_derived_requirements_drop_torch_and_comments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bc, "RUNTIME", tmp_path / "runtime")
    lock = tmp_path / "lock.txt"
    lock.write_text("# 한글 주석\nnumpy==2.3.5\ntorch==2.10.0+cu128\nPyYAML==6.0.3\n", encoding="utf-8")
    spec = install.VenvSpec(".venv-x", "x", lock, "cpu")
    path, torch_version = install._derived_requirements(spec)
    assert torch_version == "2.10.0"
    assert path.read_text(encoding="ascii").split() == ["numpy==2.3.5", "PyYAML==6.0.3"]


def test_compare_with_lock_uses_installed_torch_flavor(tmp_path: Path) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text("numpy==2.3.5\ntorch==2.10.0+cu128\n", encoding="utf-8")
    spec = install.VenvSpec(".venv-x", "x", lock, "cpu")
    assert install._compare_with_lock(spec, {"numpy": "2.3.5", "torch": "2.10.0+cpu"}) == []
    problems = install._compare_with_lock(spec, {"numpy": "2.3.4", "torch": "2.10.0+cu128"})
    assert len(problems) == 2


def test_remove_venv_refuses_other_paths(tmp_path: Path) -> None:
    with pytest.raises(bc.BundleError):
        install._remove_venv(tmp_path / ".venv-server")


def test_ensure_venv_rebuilds_venv_without_pip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """가상환경 만들기가 끊겨 파이썬만 있고 pip가 없으면 지우고 다시 만든다(pip install에서 헛돌지 않게)."""
    monkeypatch.setattr(bc, "ROOT", tmp_path)
    monkeypatch.setattr(bc, "RUNTIME", tmp_path / "runtime")
    venv = tmp_path / ".venv-server"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_bytes(b"x")
    (venv / "bin").mkdir()
    (venv / "bin" / "python").write_bytes(b"x")
    monkeypatch.setattr(install, "venv_is_current", lambda spec: False)
    monkeypatch.setattr(install, "_python_works", lambda python: "3.11.9")
    monkeypatch.setattr(install, "_pip_works", lambda python: False)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *args: object, **kwargs: object) -> None:
        calls.append(cmd)
        raise bc.BundleError("멈춤")

    monkeypatch.setattr(install, "_run", fake_run)
    spec = install.VenvSpec(".venv-server", "서버 환경", tmp_path / "lock.txt", "cpu")
    with pytest.raises(bc.BundleError, match="멈춤"):
        install.ensure_venv(spec, install.Log(tmp_path / "log.txt"), lambda: None)
    assert not venv.exists()
    assert calls and calls[0][1:3] == ["-m", "venv"]


def test_pip_failure_points_to_disk_space(tmp_path: Path) -> None:
    pip_log = tmp_path / "pip.log"
    pip_log.write_text("old run: [Errno 28] No space left on device\n", encoding="utf-8")
    since = pip_log.stat().st_size
    with open(pip_log, "a", encoding="utf-8") as handle:
        handle.write("ERROR: Could not find a version that satisfies the requirement\n")
    error = install._pip_failure("설치 실패", pip_log, since, "인터넷을 확인하세요.")
    assert "디스크" not in error.message and "인터넷" in error.hint  # 지난 실행의 기록은 보지 않는다
    with open(pip_log, "a", encoding="utf-8") as handle:
        handle.write("OSError: [Errno 28] No space left on device\n")
    error = install._pip_failure("설치 실패", pip_log, since, "인터넷을 확인하세요.")
    assert "디스크" in error.message and "pip cache purge" in error.hint


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 전용")
def test_vc_runtime_check_on_this_pc(tmp_path: Path) -> None:
    info = install.check_vc_runtime(install.Log(tmp_path / "log.txt"))
    assert info["checked"] and info["msvcp140.dll"]["path"]
    assert info["msvcp140.dll"]["version"].startswith("14.")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 전용")
def test_vc_runtime_missing_gives_download_hint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ctypes  # noqa: PLC0415

    real = ctypes.WinDLL

    def fake(name: str, *args: object, **kwargs: object) -> object:
        if name == "msvcp140.dll":
            raise OSError("[WinError 126] 지정된 모듈을 찾을 수 없습니다")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", fake)
    with pytest.raises(bc.BundleError) as caught:
        install.check_vc_runtime(install.Log(tmp_path / "log.txt"))
    assert "msvcp140.dll" in caught.value.message and bc.VC_REDIST_URL in caught.value.hint


# ---------------------------------------------------------------------------
# 묶음 만들기
# ---------------------------------------------------------------------------

def test_pick_samples_refuses_p02(tmp_path: Path) -> None:
    folder = tmp_path / "웹캠 영상 P02"
    folder.mkdir()
    with pytest.raises(bc.BundleError):
        build_bundle.pick_samples(folder)


def test_pick_samples_takes_three_p01_silent(tmp_path: Path) -> None:
    folder = tmp_path / "P01"
    folder.mkdir()
    for name in ("P01_silent_A01_1.webm", "P01_silent_A01_2.webm", "P01_silent_A02_1.webm", "P01_silent_A03_1.webm",
                 "P01_silent_A04_1.webm", "P01_voiced_A01_1.webm", "P01_none_X01_1.webm"):
        (folder / name).write_bytes(b"x")
    picked = [p.name for p in build_bundle.pick_samples(folder)]
    assert picked == ["P01_silent_A01_1.webm", "P01_silent_A03_1.webm", "P01_silent_A04_1.webm"]


def test_templates_ascii_before_chcp_and_call_deploy() -> None:
    """chcp 65001 앞 줄은 ASCII여야 한다(UTF-8 .bat이 cp949로 읽히는 동안 한글이 명령으로 잘못 해석되지 않게)."""
    for name in build_bundle.ROOT_SCRIPTS:
        text = (DEPLOY / "templates" / name).read_text(encoding="utf-8")
        head, sep, _ = text.partition("chcp 65001")
        assert sep, name
        head.encode("ascii")
        assert "deploy\\" in text and "%~dp0" in text


def test_build_bundle_lists_existing_deploy_files() -> None:
    for name in build_bundle.DEPLOY_FILES:
        assert (DEPLOY / name).is_file(), name


# ---------------------------------------------------------------------------
# 화면 서버
# ---------------------------------------------------------------------------

def test_serve_frontend_mime_and_listing(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "assets" / "m.wasm").write_bytes(b"\0asm")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    from functools import partial

    server = serve_frontend.Server(("127.0.0.1", 0), partial(serve_frontend.DistHandler, directory=str(dist)))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        def get(path: str) -> tuple[int, str, bytes]:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                    return response.status, response.headers.get("Content-Type", ""), response.read()
            except urllib.error.HTTPError as exc:
                return exc.code, "", b""

        status, kind, body = get("/")
        assert status == 200 and "text/html" in kind and b'id="root"' in body
        assert get("/assets/app.js")[1].startswith("text/javascript")
        assert get("/assets/m.wasm")[1] == "application/wasm"
        assert get("/assets/")[0] == 404
        assert get("/../secret.txt")[0] == 404
        with pytest.raises(OSError):
            serve_frontend.Server(("127.0.0.1", port), partial(serve_frontend.DistHandler, directory=str(dist)))
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# 점검·실행
# ---------------------------------------------------------------------------

def test_multipart_body_has_fields() -> None:
    body, content_type = check._multipart({"phrase_set": "ward"}, "file", "a.webm", b"DATA", "video/webm")
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="phrase_set"\r\n\r\nward\r\n' in body
    assert b'filename="a.webm"\r\nContent-Type: video/webm\r\n\r\nDATA\r\n' in body


def test_scan_dist_classifies_hosts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>", encoding="utf-8")
    (dist / "assets" / "a.js").write_text(
        'a="http://www.w3.org/2000/svg";b="http://localhost:8000";'
        'c="https://storage.googleapis.com/mediapipe-models/face_landmarker/x.task";', encoding="utf-8")
    monkeypatch.setattr(bc, "FRONTEND_DIST", dist)
    report = check.Report()
    check.scan_dist(report)
    scan = report.data["dist_scan"]
    assert set(scan["doc_hosts"]) == {"www.w3.org"}
    assert set(scan["load_hosts"]) == {"storage.googleapis.com"}
    assert report.count("주의") == 0 and report.count("실패") == 0 and report.count("통과") == 1
    (dist / "assets" / "b.js").write_text(
        'fetch("https://fonts.googleapis.com/css");x="https://bugzil.la/1328882";y="https://example.org/m.wasm"',
        encoding="utf-8")
    report = check.Report()
    check.scan_dist(report)
    assert report.count("주의") == 2  # fonts.googleapis.com, example.org/m.wasm (bugzil.la는 설명 주소)
    assert "bugzil.la" in report.data["dist_scan"]["doc_hosts"]


def test_select_sample_clips_refuses_p02(tmp_path: Path) -> None:
    sealed = tmp_path / "웹캠 영상 P02"
    sealed.mkdir()
    (sealed / "P02_silent_A01_1.webm").write_bytes(b"x")
    clips, refused = check.select_sample_clips(sealed)
    assert clips == [] and refused
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    (mixed / "P01_silent_A01_1.webm").write_bytes(b"x")
    (mixed / "p02_silent_A02_1.mp4").write_bytes(b"x")
    clips, refused = check.select_sample_clips(mixed)
    assert clips == [] and "p02_silent_A02_1.mp4" in refused
    (mixed / "p02_silent_A02_1.mp4").unlink()
    (mixed / "notes.json").write_text("{}", encoding="utf-8")
    clips, refused = check.select_sample_clips(mixed)
    assert [p.name for p in clips] == ["P01_silent_A01_1.webm"] and refused is None


def test_latency_check_reports_p02_refusal(tmp_path: Path) -> None:
    sealed = tmp_path / "P02"
    sealed.mkdir()
    (sealed / "P02_silent_A01_1.webm").write_bytes(b"x")
    report = check.Report()
    args = check.argparse.Namespace(samples=str(sealed), ward_runs=7, sentence_runs=3, timeout=5.0)
    check.latency_check(report, args, health_ok=True)  # 서버로 보내기 전에 멈춘다(서버 없이도 통과)
    assert report.count("실패") == 1 and "latency_runs" not in report.data


def test_manual_checklist_mentions_audio_output_and_wifi(capsys: pytest.CaptureFixture[str]) -> None:
    report = check.Report()
    check.manual_checklist(report)
    items = " ".join(report.data["manual_checklist"])
    assert "출력 장치" in items and "Wi-Fi만" in items and "블루투스" in items


def test_start_lock_blocks_second_launcher(tmp_path: Path) -> None:
    import subprocess  # noqa: PLC0415

    lock = tmp_path / "launch.lock"
    assert launch.acquire_start_lock(lock)
    launch.release_start_lock(lock)
    assert not lock.exists()
    lock.write_text(str(os.getppid()), encoding="ascii")  # 살아 있는 다른 실행기(부모 프로세스)
    assert not launch.acquire_start_lock(lock)
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    lock.write_text(done.stdout.strip(), encoding="ascii")  # 이미 끝난 실행기 → 무시하고 잡는다
    assert launch.acquire_start_lock(lock)
    assert lock.read_text(encoding="ascii") == str(os.getpid())
    launch.release_start_lock(lock)
    assert launch._pid_alive(os.getpid()) and not launch._pid_alive(int(done.stdout.strip()))


def test_browser_particles() -> None:
    assert launch.BROWSER_PARTICLES["Chrome"] == ("Chrome으로", "Chrome을")
    assert launch.BROWSER_PARTICLES["Edge"] == ("Edge로", "Edge를")


def test_is_ours_matches_only_this_bundle() -> None:
    ours = f'cmd.exe /s /k "title X && "C:\\py.exe" "{bc.DEPLOY / "run_server.py"}""'
    assert launch._is_ours(ours)
    assert not launch._is_ours('"C:\\other\\deploy\\run_server.py"')
    assert not launch._is_ours(f'"{bc.DEPLOY / "check.py"}"')


def test_check_report_json_serializable() -> None:
    report = check.Report()
    report.add("통과", "a")
    report.add("주의", "b")
    json.dumps(report.data | {"items": report.items}, ensure_ascii=False)
    assert report.count("통과") == 1 and report.count("주의") == 1


def test_windows_name_reports_11_for_new_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    import platform  # noqa: PLC0415

    monkeypatch.setattr(platform, "release", lambda: "10")
    monkeypatch.setattr(platform, "version", lambda: "10.0.22631")
    assert bc.windows_name() == "Windows 11 (빌드 22631)"
    monkeypatch.setattr(platform, "version", lambda: "10.0.19045")
    assert bc.windows_name() == "Windows 10 (빌드 19045)"


def test_remove_old_bundle_handles_read_only_git_objects(tmp_path: Path) -> None:
    """upstream/.git의 객체 파일은 읽기 전용이다. --force가 반쯤 지운 채 멈추지 않아야 한다."""
    import stat  # noqa: PLC0415

    bundle = tmp_path / "SilentTalk"
    objects = bundle / "experiments" / "olkavs_official" / "upstream" / ".git" / "objects" / "ab"
    objects.mkdir(parents=True)
    packed = objects / "cdef"
    packed.write_bytes(b"x")
    os.chmod(packed, stat.S_IREAD)
    (bundle / "deploy").mkdir()
    (bundle / "deploy" / bc.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    build_bundle.remove_old_bundle(bundle)
    assert not bundle.exists() and not (tmp_path / "SilentTalk.removing").exists()


def test_remove_old_bundle_refuses_unknown_folder_and_clears_leftover(tmp_path: Path) -> None:
    bundle = tmp_path / "SilentTalk"
    bundle.mkdir()
    (bundle / "mine.txt").write_text("x", encoding="utf-8")
    with pytest.raises(bc.BundleError):
        build_bundle.remove_old_bundle(bundle)
    assert (bundle / "mine.txt").is_file()
    leftover = tmp_path / "SilentTalk.removing"
    (leftover / "backend").mkdir(parents=True)
    (bundle / "mine.txt").unlink()
    bundle.rmdir()
    build_bundle.remove_old_bundle(bundle)  # 지난번에 지우다 멈춘 폴더만 치운다
    assert not leftover.exists()


# ---------------------------------------------------------------------------
# 동결 병실 기준값 (build_bundle --freeze → deploy/ward_thresholds.json → service_env·시연 점검)
# ---------------------------------------------------------------------------

THRESHOLD_VALUES = {"min_margin": 0.45, "min_score": -3.3, "possible_min_score": -3.6, "enroll_max_distance": None}


def _write_thresholds(path: Path, values: dict | None = None, **override: object) -> Path:
    """묶음 기준값 파일을 쓴다 (override로 일부러 틀린 값)."""
    doc = bc.ward_thresholds_doc(values or THRESHOLD_VALUES, {"kind": "freeze", "freeze_path": "x/freeze.json"})
    doc.update(override)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_service_env_applies_bundle_thresholds_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """묶음 기준값 파일이 있으면 LIPNET_OLKAVS_WARD_* 4개만 그 값으로 넣고, 다른 LIPNET_* 값은 여전히 지운다."""
    monkeypatch.setattr(bc, "WARD_THRESHOLDS", _write_thresholds(tmp_path / "ward_thresholds.json"))
    monkeypatch.setattr(bc, "GITSHIM_DIR", tmp_path / "gitshim")
    monkeypatch.setenv("LIPNET_OLKAVS_WARD_MIN_SCORE", "-9")
    monkeypatch.setenv("LIPNET_WARD_PROFILE_DIR", "C:/elsewhere")
    env = bc.service_env()
    assert env["LIPNET_OLKAVS_WARD_MIN_MARGIN"] == "0.45" and env["LIPNET_OLKAVS_WARD_MIN_SCORE"] == "-3.3"
    assert env["LIPNET_OLKAVS_WARD_POSSIBLE_MIN_SCORE"] == "-3.6"
    assert env["LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE"] == "none"  # 서버는 'none'을 '등록 거리 안 씀'으로 읽는다
    assert "LIPNET_WARD_PROFILE_DIR" not in env
    monkeypatch.setattr(bc, "WARD_THRESHOLDS", tmp_path / "missing.json")
    assert not any(k.startswith("LIPNET_OLKAVS_WARD_") for k in bc.service_env())  # 파일이 없으면 코드 기본값


def test_service_env_values_are_accepted_by_server_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """묶음이 넣는 환경 변수 문자열을 서버 config.py가 같은 값으로 읽는다."""
    import subprocess  # noqa: PLC0415

    monkeypatch.setattr(bc, "WARD_THRESHOLDS", _write_thresholds(tmp_path / "ward_thresholds.json"))
    env = {k: v for k, v in bc.service_env().items() if k.startswith("LIPNET_OLKAVS_WARD_")}
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import config; "
            "print(json.dumps(config.ward_rule()))")
    base = {k: v for k, v in os.environ.items() if not k.upper().startswith("LIPNET_")}
    out = subprocess.run([sys.executable, "-c", code, str(bc.BACKEND)], capture_output=True, text=True,
                         env={**base, **env}, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == THRESHOLD_VALUES


@pytest.mark.parametrize("override, values", [
    ({"schema": "other"}, None),
    ({"env": {"LIPNET_OLKAVS_WARD_MIN_MARGIN": "9"}}, None),
    ({}, {**THRESHOLD_VALUES, "min_score": 0.5}),
])
def test_load_ward_thresholds_rejects_bad_files(tmp_path: Path, override: dict, values: dict | None) -> None:
    path = tmp_path / "ward_thresholds.json"
    if values is not None:
        path.write_text(json.dumps({"schema": bc.WARD_THRESHOLDS_SCHEMA, "values": values}), encoding="utf-8")
    else:
        _write_thresholds(path, **override)
    with pytest.raises(bc.BundleError):
        bc.load_ward_thresholds(path)
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(bc.BundleError):
        bc.load_ward_thresholds(path)
    assert bc.load_ward_thresholds(tmp_path / "none.json") is None


def test_ward_thresholds_doc_checks_ranges() -> None:
    with pytest.raises(bc.BundleError):
        bc.ward_thresholds_doc({**THRESHOLD_VALUES, "min_margin": None}, {})  # M·S는 비울 수 없다
    with pytest.raises(bc.BundleError):
        bc.ward_thresholds_doc({**THRESHOLD_VALUES, "enroll_max_distance": 0.0}, {})
    with pytest.raises(bc.BundleError):
        bc.ward_thresholds_doc({**THRESHOLD_VALUES, "possible_min_score": float("nan")}, {})
    doc = bc.ward_thresholds_doc({**THRESHOLD_VALUES, "enroll_max_distance": 0.25}, {"kind": "freeze"})
    assert doc["env"]["LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE"] == "0.25"


def test_thresholds_problems_compares_health_with_bundle() -> None:
    assert bc.thresholds_problems(dict(THRESHOLD_VALUES), THRESHOLD_VALUES) == []
    assert bc.thresholds_problems({**THRESHOLD_VALUES, "min_score": -3.3 + 1e-12}, THRESHOLD_VALUES) == []
    assert len(bc.thresholds_problems({**THRESHOLD_VALUES, "enroll_max_distance": 0.27}, THRESHOLD_VALUES)) == 1
    assert len(bc.thresholds_problems({**THRESHOLD_VALUES, "possible_min_score": None}, THRESHOLD_VALUES)) == 1
    assert len(bc.thresholds_problems({**THRESHOLD_VALUES, "min_margin": "0.45"}, THRESHOLD_VALUES)) == 1
    assert bc.thresholds_problems(None, THRESHOLD_VALUES)  # 옛 서버(thresholds 없음)


def test_check_reports_threshold_mismatch_and_enroll_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(bc, "WARD_THRESHOLDS", _write_thresholds(tmp_path / "ward_thresholds.json"))
    ok = check.Report()
    check.ward_thresholds_status(ok, {"thresholds": dict(THRESHOLD_VALUES), "enroll": {"profile_dir_writable": True}})
    assert [i["level"] for i in ok.items] == ["통과", "통과"]
    bad = check.Report()
    check.ward_thresholds_status(bad, {"thresholds": {**THRESHOLD_VALUES, "enroll_max_distance": 0.27},
                                       "enroll": {"profile_dir_writable": False}})
    assert [i["level"] for i in bad.items] == ["실패", "실패"]
    monkeypatch.setattr(bc, "WARD_THRESHOLDS", tmp_path / "missing.json")
    none = check.Report()
    check.ward_thresholds_status(none, {"thresholds": dict(THRESHOLD_VALUES)})
    assert [i["level"] for i in none.items] == ["주의", "주의"]  # 동결 값 없는 묶음, 옛 서버(등록 상태 없음)


def test_untracked_code_paths_reads_porcelain_z(monkeypatch: pytest.MonkeyPatch) -> None:
    out = "?? backend/api/new.py\0 M backend/old.py\0R  scripts/a.py\0scripts/b.py\0?? frontend/src/x.js\0"
    monkeypatch.setattr(build_bundle, "git", lambda *args: out)
    assert build_bundle.untracked_code_paths() == ["backend/api/new.py", "frontend/src/x.js"]


def _valid_freeze(status: str = "frozen") -> dict:
    """validate_freeze를 통과하는 최소 병실 동결 파일 (합성 값, 사람 자료 없음)."""
    if str(bc.ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(bc.ROOT / "scripts"))
    from ward_tools import common  # noqa: PLC0415

    values = {"min_margin": 0.45, "min_score": -3.3, "possible_min_score": -3.6, "enroll_max_distance": 0.25}
    ids = [f"W{i:02d}" for i in range(1, 21)]
    return {
        "schema": common.FREEZE_SCHEMA, "status": status, "phrase_set": "ward",
        "commands": {"ids": ids, "emergency_ids": ["W02", "W03", "W04", "W10"], "sha256": "a" * 64},
        "full_list": {"min_margin": 0.45, "min_score": -3.3, "env": common.server_env(values)},
        "decision_rule": {"possible_ranks": 2, "possible_min_score": -3.6, "floor_implemented_in_ui": True},
        "enrollment": {"max_distance": 0.25, "template": common.SERVICE_TEMPLATE, "feature": common.FEATURE_NAME,
                       "distance": common.DISTANCE_NAME, "unbounded": False, "source": "service"},
        "data": {"rows_files": [{"path": "rows/P01.json", "sha256": "b" * 64, "participants": ["P01"],
                                 "sessions": ["ward_tune"]}]},
        "code": {"files": {"scripts/ward_tools/metrics.py": "c" * 64}},
        "sealed_check": {"sealed_data_read": False},
    }


def _git(repo: Path, *args: str) -> str:
    """시험용 저장소 git (작성자는 이 명령에만 준다, 설정 파일은 바꾸지 않는다)."""
    import subprocess  # noqa: PLC0415

    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def test_freeze_thresholds_reads_committed_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "exp").mkdir(parents=True)
    _git(repo, "init", "-q")
    freeze = repo / "exp" / "freeze.json"
    freeze.write_text(json.dumps(_valid_freeze()), encoding="utf-8")
    draft = repo / "exp" / "draft.json"
    draft.write_text(json.dumps(_valid_freeze("draft")), encoding="utf-8")
    _git(repo, "add", "exp")
    _git(repo, "commit", "-q", "-m", "freeze")
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(build_bundle, "REPO", repo)
    doc = build_bundle.freeze_thresholds(str(freeze), commit)
    assert doc["values"] == {"min_margin": 0.45, "min_score": -3.3, "possible_min_score": -3.6, "enroll_max_distance": 0.25}
    assert doc["source"]["freeze_path"] == "exp/freeze.json" and doc["source"]["freeze_commit"] == commit
    assert doc["env"]["LIPNET_OLKAVS_WARD_ENROLL_MAX_DISTANCE"] == "0.25"
    with pytest.raises(bc.BundleError) as info:
        build_bundle.freeze_thresholds(str(draft), commit)  # 초안은 묶지 않는다
    assert "frozen" in info.value.hint
    freeze.write_text(json.dumps({**_valid_freeze(), "note": "edited"}), encoding="utf-8")
    with pytest.raises(bc.BundleError, match="수정"):
        build_bundle.freeze_thresholds(str(freeze), commit)  # 커밋 뒤 고친 동결 파일
    untracked = repo / "exp" / "new.json"
    untracked.write_text(json.dumps(_valid_freeze()), encoding="utf-8")
    with pytest.raises(bc.BundleError, match="커밋"):
        build_bundle.freeze_thresholds(str(untracked), commit)  # 커밋하지 않은 동결 파일
    with pytest.raises(bc.BundleError, match="저장소 밖"):
        build_bundle.freeze_thresholds(str(tmp_path / "outside.json"), commit)
