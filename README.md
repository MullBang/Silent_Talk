# 립리딩 프로젝트 (Lip Reading) — Silent Talk

영상 속 화자의 입 모양을 분석해 발화 내용을 텍스트로 복원하는 한국어 립리딩 시스템.

> 모든 핵심 기준값과 규칙은 [`CLAUDE.md`](./CLAUDE.md)를 단일 진실 공급원으로 따른다.

**진행 현황:** 1단계(프로젝트 뼈대) ✅ · 2단계(전처리 파이프라인) ✅ · 3단계(모델/추론) 예정

## 아키텍처

```
영상 업로드 → 전처리(25fps 리샘플 → 3초 세그먼트 → MediaPipe ROI 96×96 정규화)
           → 3D-CNN + LSTM + CTC 추론 → Beam Search + 자모 복원 → 결과
```

- **백엔드**: FastAPI. 추론은 비동기 Job으로 등록되며 결과는 폴링으로 조회한다
  (`POST /api/infer` → `GET /api/result/{session_id}`).
- **프론트엔드**: React + Vite. 1초 간격 폴링으로 진행률/결과를 표시한다.

## 프로젝트 구조

```
backend/
├── preprocessing/        # ✅ 전처리 파이프라인 (2단계 완료)
│   ├── resampler.py      #   FPS 타임스탬프 리샘플링 + 길이 검증
│   ├── segmenter.py      #   3초(75프레임) 세그먼트 분할 + last-frame 패딩
│   ├── roi_extractor.py  #   MediaPipe ROI 크롭/정규화/모델 텐서 변환
│   └── pipeline.py       #   위 3종을 연결하는 오케스트레이터
├── models/               # ⏳ 3D-CNN + LSTM + CTC, CTC 디코더 (뼈대)
├── api/                  # ⏳ upload / infer / evaluation 라우트 (뼈대)
├── schemas/              #   Pydantic 요청/응답 스키마
├── services/             # ⏳ Job 관리 / 파일 클리너 / WebSocket (뼈대)
├── config.py             #   CLAUDE.md 기준값 전역 상수
├── main.py               #   FastAPI 진입점
└── tests/                # ✅ 전처리 단위 테스트 (22 passed)
frontend/                 # ⏳ React + Vite UI (SCR-01 업로드 / SCR-02 결과)
scripts/                  #   가중치 다운로드, 성능 평가
```

## 전처리 파이프라인 (2단계 완료)

`run_preprocessing_pipeline(video_path)`가 아래 흐름으로 영상을 모델 입력 텐서로 변환한다.

| 단계 | 모듈 / 함수 | 입력 → 출력 |
|---|---|---|
| ① 길이 검증 | `resampler.get_video_duration_sec` | `MAX_DURATION_SEC`(30s) 초과 시 `ValueError` |
| ② 리샘플링 | `resampler.resample_to_fps` | 원본 → 25fps 프레임 (타임스탬프 nearest-neighbor) |
| ③ 세그먼트 분할 | `segmenter.split_into_segments` | 프레임 → 3초(75프레임) 세그먼트, 부족분 last-frame 패딩 |
| ④ ROI 추출 | `roi_extractor.extract_roi_from_segment` | 세그먼트 → `(75,96,96,3)` ROI, 연속 12프레임 미검출 시 `None` |
| ⑤ 정규화·텐서화 | `roi_extractor.normalize_roi` → `to_model_tensor` | ROI → `(1,3,75,96,96)` `torch.Tensor` |

**반환** `list[dict]`:
- 처리 성공: `{tensor: Tensor(1,3,75,96,96), start_ms, end_ms, skipped: False}`
- 검출 실패: `{tensor: None, start_ms, end_ms, skipped: True, text: '[검출 실패 구간]', confidence: None}`

핵심 규칙 준수: `cap.set(CAP_PROP_FPS)` 미사용(타임스탬프 기반), zero-padding 금지(last-frame),
여러 명 검출 시 bbox 면적 최대 1인만 추적, 전처리 `(B,75,96,96,3)` → 모델 입력 `(B,3,75,96,96)`.

## 핵심 불변 규칙 (CLAUDE.md 발췌)

- FPS=25, SEQ_LEN=75, ROI 96×96 — **절대 변경 금지**.
- `cap.set(CAP_PROP_FPS)` 금지 → 타임스탬프 기반 리샘플링.
- zero-padding 금지 → `last-frame` 패딩.
- 전처리 출력 `(B,75,96,96,3)` → 모델 입력 `permute((0,4,1,2,3))` → `(B,3,75,96,96)`.
- CORS 와일드카드 `*` 금지, MIME 2중 검증, `*.pt/*.pth/*.onnx` 커밋 금지.

## 개발 환경

### 백엔드
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

### 프론트엔드
```bash
cd frontend
npm install
npm run dev
```

### 테스트
```bash
pytest backend/tests
```

현재 전처리 단위 테스트 통과 현황:

| 모듈 | 테스트 | 상태 |
|---|---|---|
| `resampler` | 5 | ✅ |
| `segmenter` | 6 | ✅ |
| `roi_extractor` | 11 | ✅ |

## 로드맵

- **1단계 — 프로젝트 뼈대** ✅
  폴더 구조, 함수 시그니처, 타입 힌트, docstring, 의존성 파일.
- **2단계 — 전처리 파이프라인** ✅
  - 2-A: FPS 타임스탬프 리샘플러 + 길이 검증
  - 2-B: 3초 세그먼트 분할 + last-frame 패딩
  - 2-C: MediaPipe ROI 크롭 / ImageNet 정규화 / 모델 텐서 변환
  - 2-D: 오케스트레이터(`pipeline.py`)로 통합, `(1,3,75,96,96)` 출력 확인
- **3단계 — 모델 / 추론** ⏳
  3D-CNN + LSTM + CTC 모델, CTC Beam Search + 자모 복원 디코더.
- **4단계 — API / 비동기 Job** ⏳
  업로드 · 추론 등록 · 결과 폴링 · 평가, 파일 클리너, WebSocket.
- **5단계 — 프론트엔드** ⏳
  업로드/결과 화면, 1초 폴링, 보안 컨텍스트 검사.
