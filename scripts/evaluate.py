"""성능 평가 스크립트.

등록된 test_set_id 기준으로 모델을 추론하고 CER/WER 등 지표를 산출한다.
test_set_path 직접 전달 금지 → test_set_id로만 데이터셋을 참조한다.
"""

from __future__ import annotations

import argparse


def evaluate(test_set_id: str) -> dict:
    """테스트셋에 대해 모델을 평가하고 지표를 반환한다.

    Args:
        test_set_id: 등록된 테스트셋 식별자.

    Returns:
        {cer, wer, num_samples, ...} 지표 딕셔너리.
    """
    pass


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다 (--test-set-id).

    Returns:
        파싱된 인자 네임스페이스.
    """
    pass


def main() -> None:
    """엔트리포인트: 인자 파싱 → 평가 실행 → 결과 출력."""
    pass


if __name__ == "__main__":
    main()
