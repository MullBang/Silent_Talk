"""대량 학습 데이터 준비 — 여러 영상을 라벨 tar와 매칭해 일괄 ROI/라벨 캐시화.

원천 영상(*.mp4)을 순회하며, 동일 basename의 라벨 JSON을 라벨 tar에서 추출해
prepare_data.prepare_video로 문장별 .npz를 생성한다.

병목은 MediaPipe(CPU) ROI 추출이므로 --max-videos / --max-sentences로 규모를
제한한다. 더 큰 규모는 같은 명령을 상한만 늘려 (야간 등) 실행한다.

사용:
  python scripts/prepare_batch.py \
      --source-root "D:\\...\\1.Training\\원천데이터\\TS1\\TS1" \
      --label-tar   "D:\\...\\1.Training\\라벨링데이터\\TL1.tar" \
      --out data_cache/clips --max-videos 15 --max-sentences 6 --scale 480
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from prepare_data import prepare_video  # noqa: E402


def _build_label_index(label_tar: str) -> dict[str, str]:
    """라벨 tar의 {basename(json): member 경로} 인덱스를 만든다 (1회 스캔)."""
    index: dict[str, str] = {}
    with tarfile.open(label_tar, "r") as t:
        for m in t.getmembers():
            if m.name.endswith(".json"):
                index[os.path.basename(m.name)] = m.name
    return index


def main() -> None:
    p = argparse.ArgumentParser(description="대량 학습 데이터 준비")
    p.add_argument("--source-root", required=True, help="원천 영상 루트 디렉토리")
    p.add_argument("--label-tar", required=True, help="라벨 tar 경로(TL*.tar)")
    p.add_argument("--out", default="data_cache/clips", help="npz 출력 디렉토리")
    p.add_argument("--max-videos", type=int, default=15, help="처리 최대 영상 수")
    p.add_argument("--max-sentences", type=int, default=6, help="영상당 최대 문장 수")
    p.add_argument("--scale", type=int, default=480, help="다운스케일 폭(px)")
    args = p.parse_args()

    videos = sorted(glob.glob(os.path.join(args.source_root, "**", "*.mp4"), recursive=True))
    if not videos:
        print(f"[오류] 영상이 없습니다: {args.source_root}")
        sys.exit(1)
    videos = videos[: args.max_videos]

    print(f"[인덱스] 라벨 tar 스캔: {os.path.basename(args.label_tar)}")
    label_index = _build_label_index(args.label_tar)
    print(f"  라벨 {len(label_index)}개")

    os.makedirs(args.out, exist_ok=True)
    total_clips = 0
    processed = 0
    tmp_dir = tempfile.mkdtemp(prefix="labels_")
    try:
        with tarfile.open(args.label_tar, "r") as tar:
            for vi, video in enumerate(videos, 1):
                base = os.path.splitext(os.path.basename(video))[0]
                member = label_index.get(base + ".json")
                if member is None:
                    print(f"[{vi}/{len(videos)}] 라벨 없음 → skip: {base}")
                    continue
                tmp_json = os.path.join(tmp_dir, base + ".json")
                with open(tmp_json, "wb") as f:
                    f.write(tar.extractfile(member).read())
                print(f"[{vi}/{len(videos)}] {base}")
                n = prepare_video(
                    video, tmp_json, args.out,
                    max_sentences=args.max_sentences, scale_w=args.scale,
                )
                total_clips += n
                processed += 1
                os.remove(tmp_json)
    finally:
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    print(f"[완료] 영상 {processed}개 처리 · 문장 클립 {total_clips}개 → {args.out}")


if __name__ == "__main__":
    main()
