"""시연 노트북용 묶음(폴더 + zip)을 만든다 — 개발 PC에서 실행한다 (Node·git·저장소가 있는 곳).

  py -3.11 deploy/build_bundle.py [--freeze <커밋한 병실 동결 파일>] [--sample-clips] [--out 폴더] [--ref HEAD]
                                 [--force] [--no-zip] [--allow-dirty]

만드는 것 (기본 위치: 바탕화면 SilentTalk_시연패키지_<날짜>/):
  <out>/SilentTalk/            ← 노트북에서 실행하는 묶음. 폴더 이름은 영문이어야 한다(mediapipe가 한글 경로에서
                                  모델 파일을 못 여는 일이 있다). 노트북에서는 C:\\SilentTalk 처럼 영문 경로에 둔다.
  <out>.zip                    ← 위 폴더를 담은 zip (안에 SilentTalk/ 하나)

묶음 내용:
  - 저장소 코드: git archive <ref>의 backend/ frontend/ scripts/ experiments/olkavs_official/(추적 파일),
    README.md, requirements.txt. 커밋하지 않은 변경은 들어가지 않는다(경고로 알린다). 다른 실험 폴더·docs는 뺀다.
  - 공식 모델 코드 experiments/olkavs_official/upstream/(.git 포함, commit 확인용)과 가중치 weights/multiview_lip.pt
    (저장소에 없는 파일, 작업 폴더에서 복사, SHA-256 확인)
  - frontend/dist: 위 <ref> 코드로 새로 빌드한다(npm.cmd run build, node_modules는 작업 폴더 것을 연결해 쓴다).
    작업 폴더의 frontend/dist는 건드리지 않는다.
  - deploy/의 설치·실행·점검 스크립트와 고정 패키지 목록(작업 폴더에서 복사), 최상위 .bat 4개, 설치 안내 문서
  - --sample-clips: 점검용 예시 영상 3개(조정 참가자 P01의 무음 문장 녹화). P02(최종 평가 대상) 영상은 절대 넣지 않는다.
  - --freeze: 커밋한 병실 동결 파일의 기준값 4개(1·2등 차이·1등 점수·긴급 가능성 하한·등록 거리)를 deploy/ward_thresholds.json에
    넣는다. 노트북 서버는 이 값으로 켜지고(bundle_common.service_env), 시연 점검이 /health 기준값과 같은지 본다.
    주지 않으면 코드 기본값으로 판정한다(경고로 알린다).
  - 코드 폴더(backend·frontend/src·scripts)에 커밋하지 않은 새 파일이 있으면 만들지 않는다(--allow-dirty로 무시).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_common as bc  # noqa: E402

REPO = bc.ROOT
BUNDLE_NAME = "SilentTalk"
ARCHIVE_PATHS = ("backend", "frontend", "scripts", "experiments/olkavs_official", "README.md", "requirements.txt")
SCRIPTS_DIR = REPO / "scripts"
"""동결 파일 검사(ward_tools)를 가져오는 곳 (개발 PC 저장소)."""
CODE_DIRS = ("backend", "frontend/src", "scripts")
"""커밋하지 않은 새 파일이 있으면 묶음을 거부하는 코드 폴더(--allow-dirty로 무시)."""
DEPLOY_FILES = ("bundle_common.py", "git_shim.py", "install.py", "launch.py", "check.py", "serve_frontend.py",
                "run_server.py", "requirements-server.in", "requirements-server-lock.txt")
ROOT_SCRIPTS = ("설치.bat", "시연_실행.bat", "시연_점검.bat", "시연_중지.bat")
GUIDE_SOURCE = REPO / "docs" / "시연노트북_설치안내_2026-11-10.md"
GUIDE_NAME = "시연노트북_설치안내.md"
DEFAULT_SAMPLES = Path.home() / "Desktop" / "웹캠 영상 모음" / "웹캠 영상 P01"
SAMPLE_COUNT = 3
STORE_SUFFIXES = (".pt", ".webm", ".mp4", ".wasm", ".zip", ".png", ".jpg")
"""이미 압축된(또는 압축 효과가 거의 없는) 파일은 zip에 그대로 넣어 시간을 아낀다."""


def log(text: str) -> None:
    """진행 줄."""
    print(text, flush=True)


def git(*args: str) -> str:
    """저장소 git 명령 출력."""
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True,
                              encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise bc.BundleError(f"git 명령 실패: git {' '.join(args)} ({getattr(exc, 'stderr', exc)})") from exc


def _is_reparse(path: Path) -> bool:
    """정션·심볼릭 링크(재분석 지점)인지."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT) or stat.S_ISLNK(info.st_mode)


def _unlink_junction(link: Path) -> None:
    """정션만 지운다(가리키는 작업 폴더의 node_modules는 그대로 둔다)."""
    if _is_reparse(link):
        os.rmdir(link)


def _retry_writable(func: Any, path: str, error: BaseException) -> None:
    """읽기 전용 파일(upstream/.git 안의 객체 파일 등)은 쓰기 권한을 준 뒤 한 번 더 지운다."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise error from None


def _rmtree(path: Path) -> None:
    """읽기 전용 파일까지 지우는 rmtree (파이썬 3.11·3.12 모두)."""
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_retry_writable)  # type: ignore[call-arg]
    else:
        shutil.rmtree(path, onerror=lambda func, p, info: _retry_writable(func, p, info[1]))


def remove_old_bundle(bundle: Path) -> None:
    """이전에 만든 묶음 폴더만 지운다(묶음 표시 파일이 없는 폴더는 거부).

    먼저 '<이름>.removing'으로 이름을 바꾼 뒤 지운다. 지우다 멈춰도 묶음 폴더가 반쯤 지워진 채 남지 않고,
    남은 .removing 폴더는 다음 실행에서 지운다(이 스크립트가 만든 이름이라 안전하다).
    """
    trash = bundle.with_name(bundle.name + ".removing")
    try:
        if trash.exists():
            _unlink_junction(trash / "frontend" / "node_modules")
            _rmtree(trash)
        if not bundle.exists():
            return
        if not (bundle / "deploy" / bc.MANIFEST_NAME).is_file() and any(bundle.iterdir()):
            raise bc.BundleError(f"묶음 폴더가 아닌 것 같아 지우지 않습니다: {bundle}",
                                 "직접 확인한 뒤 지우거나 다른 --out을 주세요.")
        _unlink_junction(bundle / "frontend" / "node_modules")
        os.replace(bundle, trash)
        _rmtree(trash)
    except OSError as exc:
        raise bc.BundleError(f"이전 묶음 폴더를 지우지 못했습니다: {exc}",
                             "그 폴더의 서버·화면 창이나 탐색기 창을 닫은 뒤 다시 실행하세요.") from exc


def dirty_paths() -> list[str]:
    """묶음에 들어갈 경로 중 커밋하지 않은 변경(추적 파일 수정·새 파일)."""
    out = git("status", "--porcelain", "--", *ARCHIVE_PATHS)
    return [line[3:] for line in out.splitlines() if line.strip()]


def untracked_code_paths() -> list[str]:
    """코드 폴더(CODE_DIRS)의 커밋하지 않은 새 파일('??').

    커밋한 코드가 이 파일을 import하면 묶음에는 빠져 노트북에서 서버가 켜지지 않는다(git archive는 추적 파일만 꺼낸다).
    """
    out = git("status", "--porcelain", "-z", "--untracked-files=all", "--", *CODE_DIRS)
    try:
        entries = out.split("\0")
        found: list[str] = []
        i = 0
        while i < len(entries):
            entry = entries[i]
            i += 1
            if len(entry) < 4:
                continue
            code, path = entry[:2], entry[3:]
            if code[0] in "RC":
                i += 1  # 이름 바꾸기·복사는 다음 항목이 원래 경로다
            if code == "??":
                found.append(path)
        return found
    except (AttributeError, IndexError, TypeError) as exc:
        raise bc.BundleError(f"git status 출력을 읽지 못했습니다: {exc}") from exc


def freeze_thresholds(freeze: str, commit: str) -> dict[str, Any]:
    """커밋된 병실 동결 파일(<commit> 시점의 내용)에서 서버 기준값 4개를 꺼내 묶음 기준값 파일 내용을 만든다.

    동결 파일은 저장소 안 경로여야 하고 <commit>에 들어 있어야 한다. status frozen, 형식 검사(validate_freeze)와
    full_list.env 검사를 통과해야 한다. 작업 폴더의 내용과 다르면 거부한다(커밋하지 않은 수정).

    Raises:
        BundleError: 위 조건이 하나라도 맞지 않을 때
    """
    try:
        rel = Path(freeze).resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise bc.BundleError(f"동결 파일이 저장소 밖에 있습니다: {freeze}") from exc
    try:
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise bc.BundleError(f"동결 파일이 묶을 커밋 {commit[:12]}에 없습니다: {rel}",
                             "동결 파일을 커밋한 뒤 그 커밋(또는 HEAD)으로 묶음을 만드세요.") from exc
    try:
        working = (REPO / rel).read_bytes()
    except OSError:
        working = None
    if working is not None and working.replace(b"\r\n", b"\n") != raw.replace(b"\r\n", b"\n"):
        raise bc.BundleError(f"동결 파일이 커밋 뒤 수정됐습니다: {rel}", "수정을 되돌리거나 커밋한 뒤 다시 만드세요.")
    try:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        from ward_tools import calibrate as cal  # noqa: PLC0415  (개발 PC 저장소에서만 쓴다)
        from ward_tools import common as wcommon  # noqa: PLC0415

        data = json.loads(raw.decode("utf-8"))
        problems = cal.validate_freeze(data)
        if isinstance(data, dict) and data.get("status") != "frozen":
            problems.insert(0, "status가 frozen이 아닙니다(초안)")
        expected = wcommon.expected_server_rule(data) if not problems else None
    except (ImportError, UnicodeDecodeError, ValueError) as exc:
        raise bc.BundleError(f"동결 파일을 읽지 못했습니다: {rel} ({exc})") from exc
    if problems or expected is None:
        raise bc.BundleError(f"동결 파일 검사 실패: {rel}", "; ".join(problems or ["full_list가 없습니다"])[:600])
    changed = git("log", "-1", "--format=%H", commit, "--", rel).strip()
    return bc.ward_thresholds_doc(expected, {
        "kind": "freeze", "freeze_path": rel, "freeze_sha256": hashlib.sha256(raw).hexdigest(),
        "freeze_commit": changed or None, "bundle_commit": commit})


def extract_archive(ref: str, bundle: Path) -> int:
    """git archive로 추적 파일만 꺼낸다. 꺼낸 파일 수를 돌려준다."""
    tar_path = bundle.parent / f"_{BUNDLE_NAME}_source.tar"
    try:
        with open(tar_path, "wb") as handle:
            subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", ref, "--", *ARCHIVE_PATHS],
                           check=True, stdout=handle)
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            try:
                tar.extractall(bundle, filter="data")
            except TypeError:  # filter 인자가 없는 오래된 파이썬
                tar.extractall(bundle)  # noqa: S202 - 우리 저장소에서 만든 tar
        return len(members)
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        raise bc.BundleError(f"코드를 꺼내지 못했습니다: {exc}") from exc
    finally:
        try:
            tar_path.unlink()
        except OSError:
            pass


def copy_model_files(bundle: Path) -> dict[str, Any]:
    """공식 모델 코드(upstream, .git 포함)와 가중치를 복사하고 확인한다."""
    src_upstream, src_weights = bc.UPSTREAM, bc.WEIGHTS
    if not (src_upstream / ".git").exists() or not src_weights.is_file():
        raise bc.BundleError("작업 폴더에 공식 모델 코드(upstream) 또는 가중치가 없습니다.",
                             f"{src_upstream}, {src_weights} 를 확인하세요.")
    dst_olkavs = bundle / "experiments" / "olkavs_official"
    shutil.copytree(src_upstream, dst_olkavs / "upstream", symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst_olkavs / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_weights, dst_olkavs / "weights" / "multiview_lip.pt")
    check = bc.check_model_files(bundle)
    if not check["weights_ok"]:
        raise bc.BundleError(f"묶음의 가중치 SHA-256이 다릅니다: {check['weights_sha256']}")
    if not check["upstream_ok"]:
        raise bc.BundleError(f"묶음의 공식 코드 commit이 다릅니다: {check['upstream_commit']}")
    return check


def build_frontend(bundle: Path) -> dict[str, Any]:
    """묶음의 frontend(= <ref> 코드)로 화면을 빌드한다. node_modules는 작업 폴더 것을 정션으로 잠깐 연결한다."""
    front = bundle / "frontend"
    link = front / "node_modules"
    target = REPO / "frontend" / "node_modules"
    if not (target / ".bin").is_dir():
        raise bc.BundleError("작업 폴더에 frontend/node_modules가 없습니다.", "frontend 폴더에서 npm.cmd install을 먼저 하세요.")
    started = time.perf_counter()
    try:
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True,
                              errors="replace")
        if made.returncode != 0:
            raise bc.BundleError(f"node_modules 연결 실패: {made.stderr or made.stdout}")
        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if not npm:
            raise bc.BundleError("npm을 찾지 못했습니다(이 PC에 Node가 필요합니다).")
        result = subprocess.run([npm, "run", "build"], cwd=str(front), capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise bc.BundleError("화면 빌드(npm run build)에 실패했습니다.", (result.stdout + result.stderr)[-1500:])
    finally:
        _unlink_junction(link)
    if link.exists():
        raise bc.BundleError(f"node_modules 연결을 지우지 못했습니다: {link}")
    dist = front / "dist"
    if not (dist / "index.html").is_file() or not (dist / "mediapipe" / "wasm").is_dir():
        raise bc.BundleError("빌드 결과(dist/index.html, dist/mediapipe/wasm)가 없습니다.")
    files = [p for p in dist.rglob("*") if p.is_file()]
    return {"seconds": round(time.perf_counter() - started, 1), "files": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


def _write_crlf(src: Path, dst: Path) -> None:
    """.bat은 CRLF 줄바꿈이어야 cmd가 이름표(goto)를 확실히 찾는다."""
    text = src.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\n", "\r\n")
    dst.write_bytes(text.encode("utf-8"))


def copy_deploy(bundle: Path) -> None:
    """설치·실행·점검 스크립트, 최상위 .bat, 안내 문서를 넣는다."""
    target = bundle / "deploy"
    target.mkdir(parents=True, exist_ok=True)
    for name in DEPLOY_FILES:
        shutil.copy2(bc.DEPLOY / name, target / name)
    for name in ROOT_SCRIPTS:
        _write_crlf(bc.DEPLOY / "templates" / name, bundle / name)
    if GUIDE_SOURCE.is_file():
        shutil.copy2(GUIDE_SOURCE, bundle / GUIDE_NAME)
    else:
        log(f"  [주의] 설치 안내 문서가 없어 넣지 못했습니다: {GUIDE_SOURCE}")


def pick_samples(source: Path) -> list[Path]:
    """조정 참가자 P01의 무음 문장 녹화 중 3개를 고른다(정렬 후 처음·가운데·끝). P02는 거부한다."""
    if "p02" in str(source).lower():
        raise bc.BundleError("P02(최종 평가 대상) 영상은 묶음에 넣지 않습니다.", "--samples-from에 P01 폴더를 주세요.")
    clips = sorted(p for p in source.glob("P01_silent_*_1.webm") if "p02" not in p.name.lower())
    if len(clips) < SAMPLE_COUNT:
        raise bc.BundleError(f"예시 영상 후보가 {len(clips)}개뿐입니다: {source}")
    index = sorted({0, len(clips) // 2, len(clips) - 1})
    return [clips[i] for i in index][:SAMPLE_COUNT]


def copy_samples(bundle: Path, source: Path) -> list[dict[str, Any]]:
    """예시 영상만 복사한다(메타데이터 JSON은 넣지 않는다)."""
    target = bundle / "deploy" / "samples"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in pick_samples(source):
        shutil.copy2(clip, target / clip.name)
        rows.append({"name": clip.name, "bytes": clip.stat().st_size, "sha256": bc.sha256_file(clip)})
    return rows


def tree_size(path: Path) -> tuple[int, int]:
    """(파일 수, 바이트). 정션은 따라가지 않는다."""
    count = size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not _is_reparse(Path(dirpath) / d)]
        for name in filenames:
            count += 1
            size += (Path(dirpath) / name).stat().st_size
    return count, size


def make_zip(bundle: Path, zip_path: Path) -> dict[str, Any]:
    """묶음 폴더를 zip으로 만들고(안에 SilentTalk/ 하나), 다시 열어 CRC와 가중치 SHA-256을 확인한다."""
    started = time.perf_counter()
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for dirpath, dirnames, filenames in os.walk(bundle):
            dirnames[:] = sorted(d for d in dirnames if not _is_reparse(Path(dirpath) / d))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                arcname = Path(bundle.name) / path.relative_to(bundle)
                method = zipfile.ZIP_STORED if path.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED
                archive.write(path, arcname.as_posix(), compress_type=method)
    os.replace(tmp, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise bc.BundleError(f"zip 확인 실패(CRC): {bad}")
        name = f"{bundle.name}/experiments/olkavs_official/weights/multiview_lip.pt"
        import hashlib  # noqa: PLC0415

        digest = hashlib.sha256()
        with archive.open(name) as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        entries = len(archive.infolist())
    expected = bc.core_constants(bundle)["checkpoint_sha256"]
    if digest.hexdigest() != expected:
        raise bc.BundleError("zip 안 가중치의 SHA-256이 다릅니다.")
    return {"path": str(zip_path), "bytes": zip_path.stat().st_size, "entries": entries,
            "seconds": round(time.perf_counter() - started, 1), "weights_sha256_ok": True}


def run(args: argparse.Namespace) -> int:
    """묶음 만들기 전체 흐름."""
    started = time.perf_counter()
    commit = git("rev-parse", args.ref).strip()
    dirty = dirty_paths()
    out_dir = Path(args.out).resolve()
    bundle = out_dir / BUNDLE_NAME
    zip_path = out_dir.with_suffix(".zip") if not args.zip_path else Path(args.zip_path).resolve()
    log(f"묶음 만들기: {args.ref} = {commit[:12]} → {bundle}")
    if dirty:
        log(f"  [주의] 커밋하지 않은 변경 {len(dirty)}개는 묶음에 들어가지 않습니다(예: {', '.join(dirty[:4])}).")
    untracked = untracked_code_paths()
    if untracked and not args.allow_dirty:
        raise bc.BundleError(
            f"코드 폴더에 커밋하지 않은 새 파일 {len(untracked)}개가 있습니다(예: {', '.join(untracked[:4])}).",
            "커밋한 코드가 이 파일을 쓰면 노트북에서 서버가 켜지지 않습니다. 커밋하거나 지운 뒤 다시 만드세요"
            "(필요 없는 파일임을 확인했으면 --allow-dirty).")
    if untracked:
        log(f"  [주의] --allow-dirty: 커밋하지 않은 새 코드 파일 {len(untracked)}개를 빼고 만듭니다.")
    thresholds: dict[str, Any] | None = None
    if args.freeze:
        thresholds = freeze_thresholds(args.freeze, commit)
        log(f"  병실 기준값: 동결 파일 {thresholds['source']['freeze_path']} "
            f"({bc.describe_thresholds(thresholds['values'])})")
    else:
        log("  [주의] --freeze 없이 만듭니다. 노트북은 코드 기본값(임시 병실 기준값)으로 판정합니다. "
            "기준값을 동결한 뒤에는 --freeze <커밋한 동결 파일>로 다시 만드세요.")
    if bundle.exists():
        if not args.force:
            raise bc.BundleError(f"이미 있습니다: {bundle}", "--force로 다시 만들거나 다른 --out을 주세요.")
        remove_old_bundle(bundle)
    elif bundle.with_name(bundle.name + ".removing").exists():
        remove_old_bundle(bundle)  # 지난번에 지우다 멈춘 폴더만 치운다
    bundle.mkdir(parents=True)

    log("[1/6] 저장소 코드 꺼내기 (git archive)")
    archived = extract_archive(commit, bundle)
    log(f"  파일 {archived}개")
    log("[2/6] 공식 모델 코드·가중치 복사와 SHA-256 확인")
    model = copy_model_files(bundle)
    log(f"  가중치 {model['weights_bytes'] / 1e6:.0f}MB SHA-256 일치, upstream {model['upstream_commit'][:12]}")
    log("[3/6] 화면 빌드 (npm.cmd run build)")
    front = build_frontend(bundle)
    log(f"  dist 파일 {front['files']}개, {front['bytes'] / 1e6:.1f}MB, {front['seconds']}초")
    log("[4/6] 설치·실행·점검 스크립트")
    copy_deploy(bundle)
    if thresholds is not None:
        bc.save_json(bundle / "deploy" / bc.WARD_THRESHOLDS.name, thresholds)
        if bc.load_ward_thresholds(bundle / "deploy" / bc.WARD_THRESHOLDS.name) is None:
            raise bc.BundleError("병실 기준값 파일을 묶음에 쓰지 못했습니다.")
    samples: list[dict[str, Any]] = []
    if args.sample_clips:
        samples = copy_samples(bundle, Path(args.samples_from))
        log(f"  예시 영상 {len(samples)}개: {', '.join(s['name'] for s in samples)}")
    log("[5/6] 묶음 기록")
    count, size = tree_size(bundle)
    manifest = {
        "schema": "silenttalk-demo-bundle-v1", "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ref": args.ref, "source_commit": commit, "uncommitted_changes_not_included": dirty,
        "deploy_scripts_from": "working tree", "python_required": "3.11 (64-bit)",
        "weights_sha256": model["weights_sha256"], "upstream_commit": model["upstream_commit"],
        "frontend_build": front, "samples": samples, "files": count + 1, "bytes": size,
        "builder_python": sys.version.split()[0],
        "ward_thresholds": ({"values": thresholds["values"], **thresholds["source"]} if thresholds is not None
                            else {"kind": "code_defaults"}),
        "untracked_code_not_included": untracked,
    }
    bc.save_json(bundle / "deploy" / bc.MANIFEST_NAME, manifest)
    log(f"  파일 {count + 1}개, {size / 1e6:.0f}MB")
    zip_info = None
    if not args.no_zip:
        log(f"[6/6] zip 만들기와 확인 → {zip_path}")
        zip_info = make_zip(bundle, zip_path)
        log(f"  {zip_info['bytes'] / 1e6:.0f}MB, 항목 {zip_info['entries']}개, CRC·가중치 SHA-256 확인, "
            f"{zip_info['seconds']}초")
    else:
        log("[6/6] zip 건너뜀 (--no-zip)")
    log(f"\n완료 ({time.perf_counter() - started:.0f}초). 노트북에서는 zip을 C:\\ 에 풀어 C:\\{BUNDLE_NAME} 폴더를 만들고 "
        "설치.bat을 실행합니다.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """명령줄 진입점."""
    bc.setup_console()
    today = time.strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Silent Talk 시연 노트북 묶음 만들기")
    parser.add_argument("--out", default=str(Path.home() / "Desktop" / f"SilentTalk_시연패키지_{today}"),
                        help="묶음 출력 폴더(안에 SilentTalk/를 만든다). zip은 이 폴더 옆에 같은 이름.zip")
    parser.add_argument("--zip-path", help="zip 경로를 따로 정할 때")
    parser.add_argument("--ref", default="HEAD", help="꺼낼 커밋 (기본 HEAD)")
    parser.add_argument("--sample-clips", action="store_true", help="점검용 P01 무음 영상 3개를 넣는다")
    parser.add_argument("--samples-from", default=str(DEFAULT_SAMPLES), help="예시 영상 폴더(P01만 허용)")
    parser.add_argument("--force", action="store_true", help="이전에 만든 묶음 폴더를 지우고 다시 만든다")
    parser.add_argument("--no-zip", action="store_true", help="zip을 만들지 않는다")
    parser.add_argument("--freeze", help="커밋한 병실 동결 파일: 그 기준값 4개(M·S·긴급 가능성 하한·등록 거리)로 노트북 서버를 켠다")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="코드 폴더에 커밋하지 않은 새 파일이 있어도 만든다(그 파일은 묶음에 들어가지 않는다)")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except bc.BundleError as exc:
        print(f"\n[실패] {exc.message}")
        if exc.hint:
            print(f"       {exc.hint}")
        return 1
    except (OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"\n[실패] 묶음을 만들다 멈췄습니다: {type(exc).__name__}: {exc}")
        print("       원인을 고친 뒤 --force로 다시 만드세요.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
