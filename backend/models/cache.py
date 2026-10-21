"""버전 및 정답 정합을 검사하는 NPZ 로더 (NumPy만 필요)."""

from __future__ import annotations

import os
import numpy as np

from .tokenizer import (
    ctc_min_input_len, decode_labels, encode_text, normalize_text,
    validate_tokenizer_metadata,
)


def load_clip(path: str | os.PathLike) -> dict:
    """구형/손상/CTC 불가능한 캐시를 거부하고 독립 배열을 반환한다."""
    with np.load(path, allow_pickle=False) as data:
        validate_tokenizer_metadata(data, str(path))
        frames, labels = data["frames"], data["label"]
        text, source_text = str(data["text"]), str(data["source_text"])
        if frames.ndim != 4 or frames.shape[1:] != (96, 96, 3) or frames.dtype != np.uint8:
            raise ValueError(f"{path}: 올바른 uint8 ROI 형상이 아닙니다.")
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"{path}: 라벨은 1차원 정수 배열이어야 합니다.")
        if not text or text != normalize_text(source_text):
            raise ValueError(f"{path}: 원문/정규화 텍스트 불일치 또는 빈 정답.")
        if labels.tolist() != encode_text(text) or decode_labels(labels) != text:
            raise ValueError(f"{path}: 정답 텍스트와 토큰 인덱스가 일치하지 않습니다.")
        if len(frames) < ctc_min_input_len(labels):
            raise ValueError(f"{path}: CTC 정렬에 필요한 프레임 수가 부족합니다.")
        return {"frames": frames, "label": labels.astype(np.int64, copy=False),
                "text": text, "source_text": source_text}
