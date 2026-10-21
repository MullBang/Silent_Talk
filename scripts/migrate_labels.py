"""기존 ROI 캐시를 보존하고 새 디렉토리에 v2 정답 라벨을 생성한다.

python scripts/migrate_labels.py --source data_cache --out data_cache/jamo_v2
지원하지 않는 정답, 빈 문장, CTC 불가능 표본은 report에 남기고 학습 캐시에서 제외한다.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from models.cache import load_clip  # noqa: E402
from models.tokenizer import (  # noqa: E402
    ctc_min_input_len, decode_labels, encode_text, normalize_text, tokenizer_metadata,
)


def sha256(path: Path) -> str:
    """파일을 제한된 메모리로 해시한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrate(source: Path, destination: Path) -> dict:
    """빈 출력 폴더에만 변환하며, 원본 NPZ와 영상 배열 보존을 검증한다."""
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == destination or source.is_relative_to(destination):
        raise ValueError("출력은 원본 폴더 또는 원본의 상위 폴더일 수 없습니다.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"출력 폴더가 비어 있지 않습니다. 새 경로를 사용하세요: {destination}")
    files = sorted(p for p in source.rglob("*.npz") if not p.is_relative_to(destination))
    if not files:
        raise FileNotFoundError(f"NPZ 캐시가 없습니다: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, path in enumerate(files, 1):
        relative = path.relative_to(source)
        entry = {"source": relative.as_posix(), "source_sha256": sha256(path)}
        temporary = None
        try:
            with np.load(path, allow_pickle=False) as data:
                original = str(data["source_text"] if "source_text" in data else data["text"])
                entry["source_text"] = original
                text = normalize_text(original)
                entry["normalized_text"] = text
                if not text:
                    raise ValueError("정규화 후 빈 정답")
                labels = encode_text(text)
                assert decode_labels(labels) == text
                frames = data["frames"]
                if frames.ndim != 4 or frames.shape[1:] != (96, 96, 3) or frames.dtype != np.uint8:
                    raise ValueError("올바른 uint8 ROI 형상이 아닙니다.")
                entry.update(frames=len(frames), label_length=len(labels),
                             ctc_min_frames=ctc_min_input_len(labels))
                if len(frames) < ctc_min_input_len(labels):
                    raise ValueError("새 라벨의 CTC 정렬에 필요한 프레임 수 부족")
                # Retain any additional per-clip provenance supplied by the source.
                metadata_keys = {"label", "text", "source_text", "frames", *tokenizer_metadata()}
                extras = {key: data[key] for key in data.files if key not in metadata_keys}
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, frames=frames, label=np.asarray(labels, dtype=np.int64),
                                    text=text, source_text=original, **extras, **tokenizer_metadata())
            checked = load_clip(temporary)
            if not np.array_equal(frames, checked["frames"]):
                raise ValueError("변환 중 영상 배열 변경 감지")
            temporary.replace(target)
            entry.update(status="converted", output=relative.as_posix(), frames_unchanged=True)
        except (ValueError, KeyError, OSError) as exc:
            # Source files remain untouched. Only this script's exact temp file is removed.
            if temporary is not None and temporary.exists():
                temporary.unlink()
            entry.update(status="excluded", reason=str(exc))
        if sha256(path) != entry["source_sha256"]:
            raise RuntimeError(f"변환 중 원본 파일 변경 감지: {path}")
        entries.append(entry)
        if i % 25 == 0 or i == len(files):
            print(f"[변환] {i}/{len(files)}", flush=True)
    counts = Counter(e["status"] for e in entries)
    report = {**tokenizer_metadata(), "source_root": str(source), "output_root": str(destination),
              "total": len(files), "converted": counts["converted"], "excluded": counts["excluded"],
              "source_files_unchanged": True, "converted_frames_unchanged": True, "entries": entries}
    (destination / "migration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """명령행 변환 진입점."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data_cache"))
    parser.add_argument("--out", type=Path, default=Path("data_cache/jamo_v2"))
    args = parser.parse_args()
    report = migrate(args.source, args.out)
    print(f"완료: 변환 {report['converted']} / 검토 필요 {report['excluded']}. "
          f"상세: {args.out / 'migration_report.json'}")


if __name__ == "__main__":
    main()
