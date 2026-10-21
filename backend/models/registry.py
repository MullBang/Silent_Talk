"""실험 모델 생성 및 구조 메타데이터에 따른 안전한 체크포인트 로드."""
from __future__ import annotations

import torch

from .baseline import ARCHITECTURE_VERSION, LipNetBaseline, ModelCompatibilityError
from .tokenizer import validate_tokenizer_metadata

ARCHITECTURES = (ARCHITECTURE_VERSION, "lipnet-residual-v1")


def model_class(architecture: str):
    """지원 구조만 선택한다. 기존 compact를 기본으로 유지한다."""
    if architecture == ARCHITECTURE_VERSION:
        return LipNetBaseline
    if architecture == "lipnet-residual-v1":
        from .residual import LipNetResidual
        return LipNetResidual
    raise ModelCompatibilityError(f"지원하지 않는 모델 구조: {architecture}")


def create_model(architecture: str = ARCHITECTURE_VERSION):
    """명시한 구조를 무작위 초기화한다."""
    return model_class(architecture)()


def load_model_checkpoint(path: str):
    """토큰/구조를 확인한 뒤 해당 모델의 엄격한 로더를 사용한다."""
    try:
        metadata = torch.load(path, map_location="cpu", weights_only=True)
    except OSError as exc:
        raise OSError(f"체크포인트를 읽을 수 없습니다: {path}") from exc
    validate_tokenizer_metadata(metadata, path)
    cls = model_class(metadata.get("architecture_version"))
    del metadata
    return cls.load_from_checkpoint(path)
