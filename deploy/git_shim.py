#!python
"""git이 없는 시연 노트북을 위한 최소 대체 명령 (install.py가 git.exe로 감싸 만든다).

인식 워커(scripts/olkavs_transcribe.py)는 모델을 올리기 전에 `git -C <upstream> rev-parse HEAD`로
공식 코드 commit을 확인한다. 노트북에 Git for Windows가 없으면 이 확인이 실패해 인식 워커가 뜨지 않는다.
이 파일은 그 한 가지 명령만 .git 폴더의 HEAD·refs·packed-refs를 직접 읽어 같은 결과를 낸다.
다른 git 명령은 지원하지 않는다고 알리고 128로 끝낸다(진짜 git의 오류 종료 코드와 같다).

표준 라이브러리만 쓴다. bundle_common.py도 commit 확인에 이 파일의 read_head_commit을 쓴다.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_dir(repo: Path) -> Path:
    """저장소 폴더의 .git 경로. .git이 파일(gitdir: ...)이면 가리키는 폴더를 쓴다."""
    try:
        dot_git = repo / ".git"
        if dot_git.is_file():
            text = dot_git.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                target = Path(text[len("gitdir:"):].strip())
                return target if target.is_absolute() else (repo / target).resolve()
        return dot_git
    except OSError as exc:
        raise ValueError(f".git을 읽지 못했습니다: {exc}") from exc


def _resolve_ref(git_dir: Path, ref: str, depth: int = 0) -> str:
    """refs/heads/... 이름을 commit SHA로 바꾼다 (낱개 파일 → packed-refs 순서)."""
    if depth > 5:
        raise ValueError("참조가 너무 깊게 이어집니다")
    try:
        loose = git_dir / ref
        if loose.is_file():
            value = loose.read_text(encoding="utf-8").strip()
            if value.startswith("ref:"):
                return _resolve_ref(git_dir, value[4:].strip(), depth + 1)
            if _SHA_RE.match(value):
                return value
            raise ValueError(f"참조 내용이 SHA가 아닙니다: {ref}")
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(" ", 1)
                if len(parts) == 2 and parts[1] == ref and _SHA_RE.match(parts[0]):
                    return parts[0]
    except OSError as exc:
        raise ValueError(f"참조를 읽지 못했습니다: {exc}") from exc
    raise ValueError(f"참조를 찾지 못했습니다: {ref}")


def read_head_commit(repo: Path) -> str:
    """저장소 폴더의 HEAD commit SHA(소문자 40자)를 .git 파일만 읽어 구한다.

    Args:
        repo: 작업 폴더(.git을 담은 폴더).

    Returns:
        40자 commit SHA.

    Raises:
        ValueError: .git이 없거나 HEAD를 해석할 수 없을 때.
    """
    git_dir = _git_dir(Path(repo))
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"HEAD를 읽지 못했습니다: {exc}") from exc
    if head.startswith("ref:"):
        return _resolve_ref(git_dir, head[4:].strip())
    if _SHA_RE.match(head):
        return head
    raise ValueError("HEAD 내용을 해석하지 못했습니다")


def main(argv: list[str] | None = None) -> int:
    """`git [-C 폴더] rev-parse HEAD`와 `git --version`만 처리한다.

    Returns:
        종료 코드 (성공 0, 지원하지 않는 명령·읽기 실패 128).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args in (["--version"], ["version"]):
            print("git version 0.0.0 (Silent Talk rev-parse shim)")
            return 0
        repo = Path(os.getcwd())
        while len(args) >= 2 and args[0] == "-C":
            repo = (repo / args[1]) if not Path(args[1]).is_absolute() else Path(args[1])
            args = args[2:]
        if args == ["rev-parse", "HEAD"]:
            print(read_head_commit(repo))
            return 0
        # 콘솔 문자셋과 관계없이 읽히도록 오류 문구는 영어(ASCII)로 낸다.
        print(f"Silent Talk git shim supports only 'rev-parse HEAD': {' '.join(args)}", file=sys.stderr)
        return 128
    except ValueError as exc:
        print(f"fatal: {exc}", file=sys.stderr)  # 서버가 넘긴 PYTHONIOENCODING=utf-8로 워커 기록에 그대로 남는다
        return 128


if __name__ == "__main__":
    sys.exit(main())
