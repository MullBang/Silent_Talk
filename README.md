# 립리딩 프로젝트 (Lip Reading)

영상 속 화자의 입 모양을 분석해 발화 내용을 텍스트로 복원하는 한국어 립리딩 시스템.

> 모든 핵심 기준값과 규칙은 [`CLAUDE.md`](./CLAUDE.md)를 단일 진실 공급원으로 따른다.

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
backend/        FastAPI 앱, 전처리, 모델, 서비스, 테스트
frontend/       React + Vite UI (SCR-01 업로드 / SCR-02 결과)
scripts/        가중치 다운로드, 성능 평가
```

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

## 단계

- **1단계 (현재)**: 프로젝트 뼈대 — 폴더 구조, 함수 시그니처, 타입 힌트, docstring,
  의존성 파일. 실제 로직 미구현.
