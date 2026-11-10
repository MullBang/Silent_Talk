"""Pydantic 요청/응답 스키마 정의.

설계문서 5절 API 명세 기준. API 계층의 입출력 계약을 정의한다.
confidence는 0~1 정규화값(또는 null), raw_score는 CTC log-prob 원본.
status 문자열은 services.job_manager의 상태값(pending/processing/done/error/timeout)을
따른다.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class UploadResponse(BaseModel):
    """POST /api/upload 응답."""

    session_id: str
    file_name: str
    duration_sec: float


class InferRequest(BaseModel):
    """POST /api/infer 요청."""

    session_id: str


class InferStartResponse(BaseModel):
    """POST /api/infer 응답. results를 절대 포함하지 않는다 (폴링 전용 분리)."""

    job_id: str
    session_id: str
    status: str = "processing"


class RecordStartResponse(BaseModel):
    """POST /api/record 응답. results를 포함하지 않는다 (GET /api/result/{session_id} 폴링 전용).

    duration_sec는 서버가 실제 프레임 시각으로 잰 녹화 길이(초)다.
    """

    job_id: str
    session_id: str
    status: str = "processing"
    duration_sec: float


def _absent(value: Any) -> bool:
    """값이 없으면(None) 응답에서 키 자체를 뺀다 (문장 묶음 결과에 새 키가 생기지 않게)."""
    return value is None


class WardRule(BaseModel):
    """ward 결과의 판정 기준값 (서버가 이번 판정에 쓴 값). 화면이 이 값으로 판정표를 적용한다."""

    min_margin: float  # 1·2등 차이 하한 (accepted 계산에 쓴 값)
    min_score: float  # 1등 점수 하한 (accepted 계산에 쓴 값)
    possible_min_score: Optional[float] = None  # 긴급 가능성 점수 하한. null이면 쓰지 않음(이전 규칙)
    enroll_max_distance: Optional[float] = None  # 등록 거리 기준 D. null이면 등록으로 고르지 않음


class PhraseListRef(BaseModel):
    """내 목록으로 고른 결과의 목록 표시 (화면이 지금 불러 둔 목록과 같은지 확인한다)."""

    id: str  # 내 목록 ID
    sha256: str  # 녹화 요청 때 그 목록 commands의 SHA-256


class PhraseMatch(BaseModel):
    """지정 문장 보조 모드 결과 (7단계). 녹화 한 건 전체가 창 하나이고 모든 후보를 정렬할 수 있을 때만 있다.

    클라이언트는 accepted만 보고 표시 여부를 정한다. 서버는 명령의 긴급 여부를 해석하지 않는다.
    """

    id: str
    text: str  # 후보 중 입 모양에 가장 가까운 문장
    score: float  # 그 문장의 토큰당 CTC 강제 정렬 로그우도
    margin: float  # 1등과 2등의 score 차이
    runner_up_id: Optional[str] = None
    accepted: bool  # margin과 score가 모두 기준 이상이면 True. False면 화면에 '고르지 못함'으로 표시한다.
    # 후보 목록 묶음(config.PHRASE_SET_IDS): 'sentences'(기본 20문장) 또는 'ward'(병실 명령 20개).
    set: Optional[Literal["sentences", "ward"]] = None
    # ward 묶음에만 있다(판정 기준값). 문장 묶음 결과에는 키 자체가 없다.
    rule: Optional[WardRule] = Field(default=None, exclude_if=_absent)
    # 내 목록(command_list)으로 녹화했을 때만 있다. 기본 목록·문장 묶음 결과에는 키 자체가 없다.
    list: Optional[PhraseListRef] = Field(default=None, exclude_if=_absent)


EnrollReason = Literal["multi_window", "partial_detection", "no_speech", "no_phrase", "no_features", "error"]
EnrollMatchStatus = Literal["ok", "missing", "incomplete", "stale", "no_phrase", "error"]


class EnrollStoreResult(BaseModel):
    """내 입 모양 등록 — 저장 모드 결과 (POST /api/record에 enroll_profile·enroll_command를 보낸 ward 녹화)."""

    mode: Literal["store"]
    profile: str
    command: str
    stored: bool  # 템플릿을 저장했으면 True (창 하나·지정 문장 결과·은닉 특징이 모두 있을 때만)
    reason: Optional[EnrollReason] = None  # 저장하지 않은 사유. 저장했으면 null
    text_top: Optional[str] = None  # 글자 점수 1등 명령 ID
    text_top_matches: bool  # 글자 1등 == 찍은 명령. False면 화면이 경고하고 다시 찍기를 권한다
    count: int  # 이 녹화 뒤 저장된 템플릿 수
    complete: bool  # 명령 전체(20개)를 등록했는가


class EnrollMatchResult(BaseModel):
    """내 입 모양 등록 — 판정 모드 결과 (enroll_profile만 보낸 ward 녹화). ok일 때만 거리 항목이 있다."""

    mode: Literal["match"]
    profile: str
    status: EnrollMatchStatus
    count: int
    top_id: Optional[str] = None  # 등록 거리 1등 명령
    distance: Optional[float] = None
    runner_up_id: Optional[str] = None
    runner_up_distance: Optional[float] = None
    distances: Optional[dict[str, float]] = None  # 명령 ID → 거리
    max_distance: Optional[float] = None  # 서버 기준 D (null이면 등록으로 고르지 않음)
    within: bool  # status ok, D 있음, distance ≤ D
    agree: bool  # 등록 1등 == 글자 1등(phrase.id)
    accepted: bool  # within and agree


EnrollResult = Annotated[Union[EnrollStoreResult, EnrollMatchResult], Field(discriminator="mode")]


class ResultItem(BaseModel):
    """추론 결과 단일 세그먼트."""

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: str
    confidence: Optional[float] = None  # 0.0~1.0 정규화 신뢰도 또는 null
    raw_score: Optional[float] = None  # 출력 문장의 CTC 로그우도 원본(olkavs) 또는 null
    # 확신이 낮아 text를 '[인식 불가]'로 바꾼 경우 모델이 낸 원래 문장. 그 밖에는 null.
    candidate: Optional[str] = None
    phrase: Optional[PhraseMatch] = None  # 지정 문장 보조 모드 결과. 없으면 null.
    # 내 입 모양 등록 결과 (enroll_profile을 보낸 ward 녹화의 첫 결과에만). 그 밖에는 키 자체가 없다.
    enroll: Optional[EnrollResult] = Field(default=None, exclude_if=_absent)


class ResultResponse(BaseModel):
    """GET /api/result/{session_id} 응답."""

    session_id: str
    status: str
    progress: float = 0.0
    error_message: Optional[str] = None
    results: Optional[list[ResultItem]] = None
    warnings: Optional[list[str]] = None  # 결과 품질 경고(낮은 프레임률 등). 없으면 null.
    # 완료 시점의 인식 모델 정보 {backend, decoder, checkpoint_sha256, upstream_commit, roi_profile}.
    # compact 백엔드·미완료 Job은 null.
    model: Optional[dict[str, Any]] = None


class EvalRequest(BaseModel):
    """POST /api/evaluation/run 요청. 경로 대신 등록된 식별자만 받는다."""

    test_set_id: str
    eval_unit: Literal["word", "sentence"] = "word"


class EvalStartResponse(BaseModel):
    """POST /api/evaluation/run 응답."""

    eval_job_id: str
    status: str = "queued"


class EvalResult(BaseModel):
    """평가 지표 결과."""

    cer: float
    wer: float
    avg_latency_ms: float
    detection_rate: float


class EvalStatusResponse(BaseModel):
    """GET /api/evaluation/status/{eval_job_id} 응답."""

    status: str
    progress: float = 0.0
    result: Optional[EvalResult] = None


class ErrorResponse(BaseModel):
    """공통 에러 응답."""

    detail: str
    code: Optional[str] = None


WardProfileStatus = Literal["ok", "missing", "incomplete", "stale"]


class WardProfileSummary(BaseModel):
    """GET /api/ward/profiles 목록의 등록 하나."""

    profile: str
    count: int  # 현재 명령 목록에 있고 특징 파일도 있는 템플릿 수
    complete: bool  # count가 명령 수(20)에 이르렀는가
    status: WardProfileStatus  # ok(20/20, 목록·가중치 같음) | incomplete | stale(목록·가중치가 바뀜) | missing
    updated: Optional[str] = None  # 마지막 저장 시각(UTC ISO)


class WardProfileList(BaseModel):
    """GET /api/ward/profiles 응답."""

    profiles: list[WardProfileSummary]


class WardTemplateInfo(BaseModel):
    """등록 템플릿 하나의 요약 (특징 값은 내보내지 않는다)."""

    created: Optional[str] = None
    text_top: Optional[str] = None  # 찍을 때 글자 점수 1등 명령
    text_top_matches: bool = False  # 그 1등이 찍은 명령과 같았는가


class WardProfileDetail(BaseModel):
    """GET /api/ward/profiles/{profile} 응답. 없는 등록은 status 'missing'·빈 templates."""

    profile: str
    count: int
    complete: bool
    status: WardProfileStatus
    templates: dict[str, WardTemplateInfo]


class WardProfileDeleted(BaseModel):
    """DELETE /api/ward/profiles/{profile} 응답."""

    deleted: bool


# ---------------------------------------------------------------------------
# 병실 '내 목록' (services/ward_lists.py)
# ---------------------------------------------------------------------------

WardListCategory = Literal["emergency", "call", "care", "room", "answer"]


class WardListCommand(BaseModel):
    """내 목록의 명령 하나 (id는 서버가 저장할 때 U01부터 붙인다)."""

    id: str
    text: str  # 입 모양 문장 (인식용, 정규화됨)
    display: str  # 화면·간호사 카드 문장
    category: str  # emergency·call·care·room·answer (검사 결과의 normalized에는 틀린 값이 그대로 올 수 있다)
    emergency: bool
    cancel: bool  # '방금 요청 취소' 문장 (하나까지)


class WardListIssue(BaseModel):
    """검사 항목 하나. row·rows는 0부터 센 줄 번호다(목록 전체에 대한 항목은 row null, rows 빈 목록)."""

    code: str  # 고정 식별자 (예: text_chars, lip_same)
    level: Literal["error", "strong", "warning"]  # error는 저장을 막는다. strong·warning은 보여 주기만 한다
    message: str  # 사용자에게 보여 줄 한국어 문구
    row: Optional[int] = None
    rows: list[int] = Field(default_factory=list)  # 두 줄 사이 항목은 두 줄 모두
    field: Optional[str] = None  # title·commands·text·display·category·emergency·cancel


class WardListChecks(BaseModel):
    """지금 검사기로 본 오류·주의."""

    errors: list[WardListIssue]
    warnings: list[WardListIssue]


class WardListSummary(BaseModel):
    """GET /api/ward/lists의 목록 하나."""

    id: str
    title: str
    count: int  # 명령 수
    emergency_count: int
    updated: Optional[str] = None  # 마지막 저장 시각(UTC ISO)
    sha256: str  # commands의 SHA-256


class WardListIndex(BaseModel):
    """GET /api/ward/lists 응답."""

    lists: list[WardListSummary]


class WardListDetail(BaseModel):
    """GET /api/ward/lists/{id} 응답: 저장된 목록 + 지금 검사 결과(checks)."""

    model_config = ConfigDict(populate_by_name=True)

    schema_: str = Field(alias="schema")  # 'ward-custom-list-v1'
    id: str
    title: str
    created: Optional[str] = None
    updated: Optional[str] = None
    commands: list[WardListCommand]
    sha256: str
    checks: WardListChecks


class WardListSaved(WardListDetail):
    """PUT /api/ward/lists/{id} 응답: 저장한 목록 + 주의(warnings, checks.warnings와 같다)."""

    warnings: list[WardListIssue]


class WardListNormalized(BaseModel):
    """검사기가 정규화한 목록 (오류가 없으면 이대로 저장된다)."""

    title: str
    commands: list[WardListCommand]


class WardListRowStats(BaseModel):
    """줄 하나의 음절 수·OLKAVS 토큰 수 (입 모양 문장을 셀 수 없으면 null)."""

    syllables: Optional[int] = None
    tokens: Optional[int] = None


class WardListCheckResult(BaseModel):
    """POST /api/ward/lists/check 응답 (저장하지 않는다)."""

    errors: list[WardListIssue]
    warnings: list[WardListIssue]
    normalized: WardListNormalized
    stats: list[WardListRowStats]


class WardListDeleted(BaseModel):
    """DELETE /api/ward/lists/{id} 응답."""

    deleted: bool
