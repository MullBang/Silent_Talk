"""'내 입 모양 등록' 거리 계산 (서버용, numpy만).

scripts/ward_tools/enroll.py의 unit_rows·dtw_distance·template_distances를 글자 그대로 옮겼다. 오프라인 채점 도구와
서버가 같은 거리를 내야 하므로 한쪽을 고치면 다른 쪽도 똑같이 고친다(backend/tests/test_ward_enroll_service.py가
무작위 배열로 두 쪽 거리가 같은지, 함수 본문이 글자 그대로 같은지 확인한다).

특징: 인식 모델의 CTC 직전 은닉 특징(model.medium 출력, T x 256)을 프레임마다 L2 정규화한다.
거리: 코사인 프레임 거리(1 - cos), 단위 이동 (1,0)·(0,1)·(1,1) 가중치 1, 최적 경로 누적 비용 / 경로 칸 수.
명령별 거리 = 그 명령의 등록 템플릿들과의 거리 중 가장 작은 값.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def unit_rows(x: Any) -> np.ndarray:
    """프레임 벡터마다 L2 정규화 (float64).

    Raises:
        ValueError: 2차원이 아니거나 길이 0 벡터가 있을 때
    """
    try:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(f"(T, D) 배열이 아닙니다: {arr.shape}")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if not np.all(norms > 0):
            raise ValueError("길이가 0인 프레임 벡터")
        return arr / norms
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unit_rows 실패: {exc}") from exc


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """정규화된 두 특징열의 DTW 거리 (코사인 비용, 단위 이동, 경로 길이로 나눔).

    동점이면 대각 → 위 → 왼쪽 순으로 고른다(실험 코드와 같다). 반대각선 단위로 벡터화했다.
    """
    try:
        cost = 1.0 - a @ b.T
        np.clip(cost, 0.0, 2.0, out=cost)
        n, m = cost.shape
        acc = np.full((n + 1, m + 1), np.inf)
        length = np.zeros((n + 1, m + 1))
        acc[0, 0] = 0.0
        for k in range(n + m - 1):
            i = np.arange(max(0, k - m + 1), min(n - 1, k) + 1)
            j = k - i
            ii, jj = i + 1, j + 1
            cand = np.stack([acc[ii - 1, jj - 1], acc[ii - 1, jj], acc[ii, jj - 1]])
            lcand = np.stack([length[ii - 1, jj - 1], length[ii - 1, jj], length[ii, jj - 1]])
            arg = np.argmin(cand, axis=0)
            col = np.arange(len(i))
            acc[ii, jj] = cand[arg, col] + cost[i, j]
            length[ii, jj] = lcand[arg, col] + 1.0
        return float(acc[n, m] / length[n, m])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"DTW 실패: {exc}") from exc


def template_distances(test: np.ndarray, templates: dict[str, list[np.ndarray]]) -> dict[str, float | None]:
    """시험 특징열(정규화됨) → 명령별 최소 DTW 거리. 템플릿이 없는 명령은 None.

    Args:
        test: unit_rows를 거친 (T, D)
        templates: 명령 ID → unit_rows를 거친 템플릿 목록
    """
    out: dict[str, float | None] = {}
    for cid, tlist in templates.items():
        out[cid] = min((dtw_distance(test, t) for t in tlist), default=None) if tlist else None
    return out
