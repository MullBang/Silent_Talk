"""전역 설정 상수 모듈.

CLAUDE.md '절대 변경 금지 기준값'을 단일 진실 공급원(Single Source of Truth)으로
정의한다. 이 값들은 전처리 → 모델 입력 → 추론 파이프라인 전 구간의 계약(contract)이며
임의로 변경 시 학습 가중치와 입력 텐서 형상이 어긋난다.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 절대 변경 금지 기준값 (CLAUDE.md §절대 변경 금지 기준값)
# ---------------------------------------------------------------------------

FPS: int = 25
"""목표 프레임 레이트. cap.set(CAP_PROP_FPS) 금지 — 타임스탬프 기반 리샘플링 필수."""

WS_PROTOCOL_VERSION: int = 2
WS_CHUNK_FRAMES: int = 11
WS_MAX_UTTERANCE_FRAMES: int = 200
"""웹캠 전송 청크 상한 및 발화당 프레임 상한(8초). 프론트엔드와 동일하게 유지."""

SEQ_LEN: int = 75
"""시퀀스 길이 (3초 × 25fps)."""

ROI_SIZE: tuple[int, int] = (96, 96)
"""ROI 크롭 출력 크기 (H, W) 픽셀."""

ROI_MARGIN: float = 0.20
"""ROI 바운딩 박스 상하좌우 margin 비율. margin 추가 후 resize."""

PADDING: str = "last-frame"
"""부족한 프레임 패딩 전략. zero-padding 절대 금지 — 마지막 프레임 복제."""

PREPROC_SHAPE: tuple[str, ...] = ("B", "T", "H", "W", "C")
"""전처리 출력 텐서 형상 = (B, 75, 96, 96, 3)."""

MODEL_INPUT_SHAPE: tuple[str, ...] = ("B", "C", "T", "H", "W")
"""모델 입력 텐서 형상 = (B, 3, 75, 96, 96). permute((0, 4, 1, 2, 3)) 변환 필수."""

NORMALIZE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""ImageNet 정규화 평균 (R, G, B)."""

NORMALIZE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet 정규화 표준편차 (R, G, B)."""

# ---------------------------------------------------------------------------
# 랜드마크 인덱스 (CLAUDE.md §랜드마크 인덱스 / MediaPipe Face Mesh)
# ---------------------------------------------------------------------------

ROI_CROP_IDX: list[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
"""ROI 크롭 기준 외측 입술 랜드마크 20개."""

VVAD_IDX_TOP: list[int] = [13, 14, 312, 311, 310, 415]
"""윗입술 내측 랜드마크 — d_raw 상단 중심점 계산용."""

VVAD_IDX_BOT: list[int] = [17, 18, 84, 181, 180, 314]
"""아랫입술 내측 랜드마크 — d_raw 하단 중심점 계산용.

d_raw = 윗입술 평균좌표 ↔ 아랫입술 평균좌표 유클리드 거리.
mouth_ratio = d_raw / W (W: 좌우 광대 랜드마크 거리).
"""

# ---------------------------------------------------------------------------
# 업로드 / 추론 제약
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE_MB: int = 500
"""업로드 허용 최대 파일 크기 (MB). 설계문서 F-MVP-01 서비스 제한값.

검증 임계값은 500MB이며 초과 시 400 에러('500MB 이하' 안내, TC-03).
"""

DEMO_UPLOAD_SIZE_MB: int = 100
"""시연 합격 기준 파일 크기 (MB). 검증 임계값이 아닌 데모 성공 기준 지표."""

MAX_DURATION_SEC: int = 180
"""업로드 영상 허용 최대 길이 (초). 설계문서 F-MVP-01 = 3분.

초과 시 400 에러('3분 이하' 안내, TC-04). 장편 영상은 3초 세그먼트로 분할 처리.
"""

JOB_TIMEOUT_SEC: int = 300
"""비동기 Job 타임아웃 (초). 초과 시 status='timeout' 처리."""

TEST_SETS: dict[str, str] = {
    # "demo_word": r"D:\009.립리딩(입모양) 음성인식 데이터\01.데이터\2.Validation",
}
"""평가용 테스트셋 화이트리스트 (test_set_id → 경로).

설계문서 보안 규칙: test_set_path 직접 전달 금지 → 등록된 test_set_id로만 조회.
"""

POLL_INTERVAL_SEC: float = 1.0
"""GET /api/result 폴링 권장 간격 (초)."""

# ---------------------------------------------------------------------------
# CORS / 경로
# ---------------------------------------------------------------------------

CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
"""CORS 허용 오리진 화이트리스트. 와일드카드 '*' 금지."""

TMP_UPLOAD_DIR: str = os.path.join(os.getcwd(), "tmp", "uploads")
"""임시 업로드 파일 저장 디렉토리. 추론 완료 후 즉시 os.remove()."""

TMP_ROI_DIR: str = os.path.join(os.getcwd(), "tmp", "roi")
"""OLKAVS 입술 프레임(NPZ) 임시 디렉토리. 세션별 하위 폴더를 추론 직후 삭제한다.

TMP_UPLOAD_DIR 밖에 두어 `{session_id}_*` 업로드 파일 탐색과 섞이지 않게 한다."""

BACKEND_ROOT: str = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT: str = os.path.dirname(BACKEND_ROOT)

MODEL_WEIGHTS_PATH: str = os.environ.get("LIPNET_WEIGHTS") or os.path.join(
    BACKEND_ROOT, "models", "weights", "baseline_jamo_v2.pt"
)
"""compact(69클래스 v2) 가중치 경로. 환경변수 LIPNET_WEIGHTS로 바꿀 수 있다.

기존 baseline.pt(41클래스)는 보존하되 사용하지 않는다.
*.pt는 .gitignore 대상 — download_weights.sh로 별도 배포."""

LOG_DIR: str = os.path.join(os.getcwd(), "logs")
"""로그 출력 디렉토리."""

# ---------------------------------------------------------------------------
# 서비스 인식 모델 선택 (2026-10-20 공식 모델 전환 결정)
# ---------------------------------------------------------------------------
# 'olkavs'  : 공식 OLKAVS 영상 모델. 업로드 경로에서 상주 워커 2개(입술 추출/인식)를 쓴다.
#             입력 규약은 위 compact 규약과 별개다: 원본 약 30fps 프레임(WebM은 실제 PTS 기준 30fps
#             격자), 정사각형·1.5배 문맥·EMA 입술 상자(stable_context_square), BGR 96x96,
#             발화 단위 평균/표준편차 정규화, 72토큰. 위 '절대 변경 금지' 값은 compact 경로 전용이다.
# 'compact' : 자체 경량 모델(LipNetBaseline, 69토큰). 웹캠 실시간(WS v2)은 계속 이 경로를 쓴다.

MODEL_BACKEND: str = os.environ.get("LIPNET_MODEL_BACKEND", "olkavs").strip().lower()
if MODEL_BACKEND not in ("olkavs", "compact"):
    raise ValueError(f"LIPNET_MODEL_BACKEND는 olkavs 또는 compact여야 합니다: {MODEL_BACKEND!r}")


def _venv_python(name: str) -> str:
    """프로젝트 가상환경의 파이썬 경로 (Windows Scripts/ 또는 POSIX bin/)."""
    windows = os.path.join(PROJECT_ROOT, name, "Scripts", "python.exe")
    return windows if os.name == "nt" else os.path.join(PROJECT_ROOT, name, "bin", "python")


OLKAVS_ROI_PYTHON: str = os.environ.get("LIPNET_OLKAVS_ROI_PYTHON") or _venv_python(".venv-olkavs-roi")
"""입술 추출 워커 인터프리터 (mediapipe 0.10.21)."""

OLKAVS_INFER_PYTHON: str = os.environ.get("LIPNET_OLKAVS_INFER_PYTHON") or _venv_python(".venv-olkavs")
"""인식 워커 인터프리터 (torch, 공식 모델 코드)."""

OLKAVS_WORKER_SCRIPT: str = os.path.join(BACKEND_ROOT, "services", "olkavs_worker.py")

OLKAVS_DEVICE: str = os.environ.get("LIPNET_OLKAVS_DEVICE", "auto")
"""auto(CUDA 있으면 cuda:0) | cpu | cuda:0."""

OLKAVS_DECODER: str = os.environ.get("LIPNET_OLKAVS_DECODER", "hybrid")
"""hybrid(정확도 우선) | ctc(속도 우선, 1ms 미만).

hybrid 실측(RTX 3080): 10초 창 3~13초(출력 자모 수의 제곱에 가깝게 늘어남, 쉼 없는 발화면 약 16초).
180초 자연 발화 영상 1개: 입술 추출 약 51초 + 인식 약 136초 ≈ 187초."""
if OLKAVS_DECODER not in ("hybrid", "ctc"):
    raise ValueError(f"LIPNET_OLKAVS_DECODER는 hybrid 또는 ctc여야 합니다: {OLKAVS_DECODER!r}")

OLKAVS_SEGMENT_SECONDS: float = float(os.environ.get("LIPNET_OLKAVS_SEGMENT_SECONDS", "12"))
"""모델 입력 창의 최대 길이(초). 허용 범위 3~12초.

상한 12초는 hybrid 출력 상한(200자모)·처리 시간(자모 수의 제곱에 비례)을 고려한 값이다.
창은 [최대 길이 − 탐색 폭, 최대 길이]에서 입 움직임이 가장 작은 지점에서 자른다.
2차 검토 실측(C297 연속 녹화 302초·28문장, hybrid): 최대 10초·탐색 3초는 고정 10초 분할과
오류율이 같았다(19.86% 대 19.77%, 문장 중간 절단 14/34). 최대 12초·탐색 6초는 16.01%,
문장 중간 절단 6/32였다. 녹화 하나의 측정이므로 3단계 촬영 자료로 다시 확인한다."""
if not (3.0 <= OLKAVS_SEGMENT_SECONDS <= 12.0):  # NaN도 여기서 거부된다
    raise ValueError(f"LIPNET_OLKAVS_SEGMENT_SECONDS는 3~12초여야 합니다: {OLKAVS_SEGMENT_SECONDS!r}")

OLKAVS_CUT_SEARCH_SECONDS: float = OLKAVS_SEGMENT_SECONDS / 2
"""창을 자를 지점을 찾는 폭(초): 최대 길이의 절반. 기본 12초면 [6, 12]초 범위에서 자른다.

탐색 폭만 넓히고 최대 길이를 10초로 두면 효과가 없었다(문장 중간 절단 8/24). 두 값을 함께 움직인다."""

OLKAVS_MIN_WINDOW_SECONDS: float = 1.0
"""연속 검출 구간 끝에 남은 창이 이보다 짧으면 직전 창에 붙인다."""

OLKAVS_PRELOAD: bool = os.environ.get("LIPNET_OLKAVS_PRELOAD", "1") != "0"
"""서버 시작 시 워커를 백그라운드로 미리 띄운다(첫 요청 지연 방지)."""

OLKAVS_START_TIMEOUT_SEC: float = 180.0
"""워커 기동(라이브러리 import·모델 로드·예열) 대기 상한."""

OLKAVS_ROI_TIMEOUT_SEC: float = 240.0
"""영상 한 개의 입술 추출 상한 (180초 1080p 영상 약 51초 실측 기준 여유)."""

OLKAVS_INFER_TIMEOUT_SEC: float = 120.0
"""창 하나의 인식 상한."""

OLKAVS_FATAL_COOLDOWN_SEC: float = 30.0
"""워커 기동이 실패하면 이 시간 동안은 다시 띄우지 않고 바로 오류를 낸다(요청마다 반복 기동 방지)."""

OLKAVS_MAX_CONSECUTIVE_FAILURES: int = 3
"""같은 워커의 처리 실패가 연속 이 횟수에 이르면 워커를 재기동한다."""

OLKAVS_EXPECTED_OVERHEAD_SEC: float = 10.0
"""입장 제어용 예상 처리 시간의 고정 부분(워커 요청·창 분할 등)."""

OLKAVS_EXPECTED_SECONDS_PER_VIDEO_SECOND: float = 1.25
"""입장 제어용 영상 1초당 예상 처리 시간.

실측(RTX 3080, hybrid, 180초 1080p 자연 발화): 입술 추출 약 51초 + 인식 약 164초(12초 창) ≈ 215초.
10초 + 1.25 × 180 = 235초로 여유를 둔다. 시연 PC가 더 느리면 다시 잰다."""


def _env_unit_float(name: str, default: float) -> float:
    """환경변수 실수값을 읽어 0~1 범위를 검사한다 (NaN·무한대·범위 밖은 기동 시 거부).

    Args:
        name: 환경변수 이름.
        default: 미설정 시 기본값.

    Returns:
        0.0 이상 1.0 이하의 실수.

    Raises:
        ValueError: 숫자가 아니거나 범위를 벗어난 경우.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}는 0~1 사이 숫자여야 합니다: {raw!r}") from exc
    if not (0.0 <= value <= 1.0):  # NaN은 비교가 모두 거짓이라 여기서 거부된다
        raise ValueError(f"{name}는 0~1 사이 숫자여야 합니다: {raw!r}")
    return value


OLKAVS_MIN_CONFIDENCE: float = _env_unit_float("LIPNET_OLKAVS_MIN_CONFIDENCE", 0.75)
"""인식 결과 confidence(프레임별 최대 확률의 기하평균, 0~1)가 이 값보다 낮으면 '[인식 불가]'로 표시한다.

모델이 낸 문장은 결과의 candidate로 함께 남긴다. 0.0이면 판정을 끈다. 환경변수 LIPNET_OLKAVS_MIN_CONFIDENCE로 바꿀 수 있다.

임시값 0.75 (2026-10-20 보정, experiments/stage6_vad_calibration_20261020/calibration.json):
- AI Hub 업로드 창 204개(6명)와 val20 문장 20개 중 문자 오류율 30% 이하인 결과 144개는 하나도 숨기지 않는다.
- val20에서는 오류율 54~62%인 4문장을 숨긴다. 업로드 창은 모두 0.80 이상이라 숨기지 않는다.
- 말하지 않는 구간을 강제로 인식한 27개(C297 문장 사이)는 confidence 0.63~0.92이고 그중 16개를 숨긴다.
  정지 화면은 0.92로 높아 이 문턱으로는 못 거른다(입 움직임 판정이 먼저 '[발화 없음]'으로 거른다).
AI Hub는 모델 학습 분포에 가까운 자료라 웹캠·처음 보는 사람에서는 분포가 다를 수 있다. 3단계 팀 촬영 자료로 다시 정한다."""

# ---------------------------------------------------------------------------
# 웹캠 녹화 접수 (POST /api/record, 6단계)
# ---------------------------------------------------------------------------

RECORD_MAX_SECONDS: float = 30.0
"""녹화 길이 상한(초). 서버가 실제 프레임 시각으로 잰 길이가 이 값 '미만'일 때만 접수한다."""

RECORD_MIN_SECONDS: float = 0.5
"""녹화 길이 하한(초). 이보다 짧으면 인식할 발화가 없다고 보고 거부한다."""

RECORD_MAX_SIZE_MB: int = 100
"""녹화 파일 크기 상한(MB). 30초 미만 웹캠 녹화는 보통 수 MB라 충분한 여유다."""

RECORD_MAX_FRAMES: int = 3600
"""길이 측정 중 읽을 최대 프레임 수 (30초 × 120fps). 넘으면 비정상 파일로 거부한다."""

RECORD_PROBE_TIMEOUT_SEC: float = 30.0
"""녹화 길이 측정(프레임 시각 읽기)에 쓸 최대 시간(초). 넘으면 비정상 파일로 거부한다."""

REQUEST_BODY_OVERHEAD_BYTES: int = 1024 * 1024
"""POST /api/record·/api/upload 요청 본문 상한의 멀티파트 여유분(바이트).

본문 상한 = 파일 상한(RECORD_MAX_SIZE_MB / MAX_UPLOAD_SIZE_MB) + 이 값. 넘으면 받는 즉시 413으로 끊는다
(api/limits.py RequestBodyLimitMiddleware). 파일 자체의 정확한 상한 검사(400)는 각 라우트가 따로 한다."""
