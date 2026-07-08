# Silent Talk — 립리딩 기반 실시간 텍스트 변환 시스템

영상·웹캠 속 화자의 입 모양을 분석해 발화 내용을 한국어 텍스트로 복원하는 립리딩 웹 서비스.
소음·무음·보안 환경에서의 시각 기반 대체 입력(배리어프리)을 목표로 한다.

> 핵심 기준값/규칙은 [`CLAUDE.md`](./CLAUDE.md), 전체 명세는 근간 문서를 단일 진실 공급원으로 따른다:
> [기획서 완성본](./docs/기획서_완성본.md) · [설계문서 완성본](./docs/설계문서_완성본.md) (충돌 시 설계문서 기준)

## 프로젝트 개요

```
[업로드 영상 | 웹캠]
   → 전처리(25fps 타임스탬프 리샘플 → 3초 세그먼트 → MediaPipe 입술 ROI 96×96 정규화)
   → 3D-CNN + BiLSTM + CTC 추론
   → CTC Beam Search + 자모(python-jamo) 음절 복원
   → [타임스탬프 결과 화면 | 실시간 자막]
```

- **백엔드**: FastAPI. 업로드 추론은 비동기 Job(폴링), 웹캠은 WebSocket 스트리밍.
- **프론트엔드**: React + Vite + Tailwind. 업로드(SCR-01)/결과(SCR-02)/웹캠(SCR-03) 화면.
- **모델**: LipNet 구조 베이스라인(3D-CNN+BiLSTM+CTC), 자모 41클래스(자음19+모음21+blank).

**진행 현황**: 1~5단계 구현 완료(전처리·서비스·API·모델/디코더·웹캠·평가). 백엔드 75개 테스트 통과.
인식 텍스트는 **모델 가중치 학습 후** 실제 한국어가 출력된다(현재는 미학습 placeholder).

## 실행 방법

### 사전 준비
```bash
# 백엔드 (Python 3.10+)
pip install -r requirements.txt

# 프론트엔드 (Node 18+)
cd frontend && npm install
```

### 백엔드 실행
```bash
cd backend
uvicorn main:app --reload --port 8000
# Swagger UI: http://127.0.0.1:8000/docs   ·   헬스체크: /health
```

### 프론트엔드 실행
```bash
cd frontend
npm run dev
# http://localhost:5173
```

> Windows에서 백엔드 헬스체크가 `localhost`로 안 되면 `127.0.0.1:8000`을 사용한다(uvicorn 기본 IPv4 바인딩).
> 웹캠 모드는 `https` 또는 `localhost`(보안 컨텍스트)에서만 동작한다.

## 모델 학습

AI Hub 「립리딩(입모양) 음성인식 데이터」로 베이스라인을 학습한다. GPU 권장(설계 RTX 3060+).
CUDA 빌드 torch 필요: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.

```bash
# ① 데이터 준비: 영상+라벨(Sentence_info) → 문장별 ROI/자모 라벨 npz 캐시
python scripts/prepare_data.py \
    --video "<원천데이터>/lip_..._001.mp4" \
    --label "<라벨링데이터>/lip_..._001.json" \
    --out data_cache/clips

# ①-b 대량 준비: 여러 영상을 라벨 tar와 매칭해 일괄 처리 (규모는 상한으로 조절)
#     --max-sentences 기본값 None = 영상당 전체 51문장(긴 문장 포함)을 학습에 사용
python scripts/prepare_batch.py \
    --source-root "<원천데이터>/TS1/TS1" --label-tar "<라벨링데이터>/TL1.tar" \
    --out data_cache/clips --max-videos 15 --scale 480
#     짧게 스모크 테스트만 하려면 --max-sentences 6 처럼 상한을 준다.
#     메모리가 빠듯하면 --max-frames 250 (10초) 등으로 초과 클립을 스킵한다.

# ② 학습: 캐시 npz로 3D-CNN+BiLSTM+CTC 학습 (GPU 자동, train/val 분리·val CER·best 저장)
#     길이 버킷 배치 샘플러가 비슷한 길이끼리 묶어 가변 길이(짧은/긴 문장 혼합)를
#     효율적으로 학습한다. --max-frames 로 학습 시에도 초과 클립을 제외할 수 있다.
python scripts/train.py --data data_cache/clips --epochs 150 \
    --batch-size 2 --lr 1e-4 --val-split 0.15 --eval-every 10
# → backend/models/weights/baseline.pt 저장 (best val CER)
```

> 데이터 준비(MediaPipe)가 병목이다. 대량 준비는 상한/오프셋을 늘려 백그라운드/야간 실행을 권장한다.
> (`--offset N`으로 이미 처리한 앞 N개를 건너뛴다.)
> 립리딩은 본래 대량 데이터가 필요하므로, 소규모로는 train loss는 내려가도 val CER은 높게 머문다.

**긴 문장(가변 길이) 학습 구조**: 모델(3D-CNN+BiLSTM)은 시간축 T를 런타임에 읽어
처리하므로 문장별 클립의 프레임 수가 달라도(관측 42~294프레임, 최대 ~12초) 그대로
학습한다. `SEQ_LEN=75`는 명목 기준값일 뿐 하드코딩 가정이 아니다.
- 준비 단계에서 CTC 정렬 최소 길이(라벨 길이 + 인접 중복 자모 수)를 만족하지 못하는
  클립은 자동 스킵한다.
- 학습 단계는 길이 버킷 배치 샘플러로 last-frame 패딩 낭비를 최소화한다.

### 학습 모델 성능 테스트
```bash
# 검증셋(학습 제외분)으로 예측/정답/CER 출력
python scripts/test_model.py --data data_cache/clips

# 전체 또는 별도 준비한 테스트셋으로
python scripts/test_model.py --data data_cache/clips --all
```
- 각 클립의 정답·예측·CER과 평균 CER을 출력한다(낮을수록 정확).
- 웹 UI에서 직접 테스트하려면 백엔드+프론트를 띄우고 3분 이하 영상을 업로드한다
  (`baseline.pt`가 있으면 업로드 추론도 실제 모델을 사용).

- 라벨 텍스트는 자모(초성/중성/종성) 인덱스로 인코딩(41클래스 CTC).
- CTC 제약상 입력 프레임 길이 T ≥ (라벨 길이 + 인접 중복 자모 수)여야 하며, 문장이 길면
  충분한 프레임이 필요하다(준비 단계에서 자동 검사·스킵).
- `prepare_data.py`는 mediapipe만, `train.py`는 torch만 사용해 네이티브 충돌(segfault)을 피한다.
- 학습된 `baseline.pt`가 있으면 추론(API/웹캠/평가)이 자동으로 이를 로드한다.

## 모델 가중치 다운로드

가중치 파일(`*.pt`)은 GitHub 100MB 제한으로 저장소에 커밋하지 않는다(`.gitignore` 등록).
별도 스토리지에서 받아 `backend/models/weights/baseline.pt`에 배치한다.

```bash
bash scripts/download_weights.sh
# 또는 수동으로 backend/models/weights/baseline.pt 에 가중치 배치
```

가중치가 없으면 미학습 모델로 폴백하여 파이프라인 구조는 동작하지만, 의미 있는 텍스트는 학습 후 생성된다.

## 테스트 실행

```bash
# 백엔드 단위/통합 테스트 (75개)
pytest backend/tests

# 특정 모듈만
pytest backend/tests/test_api.py -v

# 프론트엔드 빌드 검증
cd frontend && npm run build
```

### 성능 평가 (CER/WER)
```bash
python scripts/evaluate.py \
    --test_set_id aihub_subset \
    --model_version baseline_v1 \
    --eval_unit word
```
- `test_set_id`는 `backend/config.py`의 `TEST_SETS`에 등록된 식별자만 허용(경로 직접 전달 금지).
- 테스트셋 디렉토리는 `manifest.json`(`[{"video","text"}, ...]`)과 영상 파일로 구성.
- 결과: `results/evaluation_log.csv`(eval_id, model_version, cer, wer, avg_latency_ms, detection_rate, …) + `results/eval_*.png` 그래프.

## 폴더 구조

```
backend/
├── main.py                  # FastAPI 진입점(CORS·라이프사이클·/ws/stream)
├── config.py                # CLAUDE.md 기준값 전역 상수 (FPS/SEQ_LEN/ROI/제약/CORS/TEST_SETS)
├── api/                     # REST 라우트
│   ├── upload.py            #   POST /api/upload (MIME 2중 검증·크기·길이)
│   ├── infer.py             #   POST /api/infer (비동기) · GET /api/result/{id}
│   └── evaluation.py        #   POST /api/evaluation/run · GET /api/evaluation/status/{id}
├── schemas/models.py        # Pydantic 요청/응답 스키마
├── preprocessing/           # 전처리 파이프라인
│   ├── resampler.py         #   타임스탬프 25fps 리샘플 + 길이 검증
│   ├── segmenter.py         #   3초(75프레임) 분할 + last-frame 패딩
│   ├── roi_extractor.py     #   MediaPipe ROI 크롭/정규화/모델 텐서
│   └── pipeline.py          #   오케스트레이터
├── models/                  # 모델/디코더
│   ├── __init__.py          #   자모 어휘(GRAPHEME_VOCAB, 41클래스)
│   ├── baseline.py          #   LipNetBaseline (3D-CNN+BiLSTM+CTC)
│   └── decoder.py           #   CTC beam search + 자모 복원 + 신뢰도
├── services/                # 서비스 계층
│   ├── job_manager.py       #   비동기 Job 상태(메모리)
│   ├── file_cleaner.py      #   APScheduler 임시 파일 정리
│   └── ws_handler.py        #   WebSocket 실시간 스트리밍 핸들러
└── tests/                   # pytest (75개)

frontend/
├── index.html · src/main.jsx · src/App.jsx   # 진입점·라우팅
├── src/pages/               # UploadPage(SCR-01) · ResultPage(SCR-02) · WebcamPage(SCR-03)
├── src/components/          # DropZone · ProgressBar · ResultList · ErrorModal
├── src/hooks/usePolling.js  # 1초 폴링 훅
├── src/api/client.js        # axios API 호출
├── src/constants.js         # CLAUDE.md 값 미러(랜드마크·임계값)
└── src/webcamUtils.js       # V-VAD/ROI 헬퍼

scripts/
├── download_weights.sh      # 가중치 다운로드
└── evaluate.py              # 성능 평가(CER/WER·CSV·그래프)

docs/                        # 근간 문서(기획서·설계문서 완성본)
```

## 핵심 불변 규칙 (CLAUDE.md 발췌)
- FPS=25, SEQ_LEN=75, ROI 96×96 — 절대 변경 금지. `cap.set(CAP_PROP_FPS)` 금지(타임스탬프 리샘플).
- zero-padding 금지 → `last-frame` 패딩. 전처리 `(B,75,96,96,3)` → 모델 입력 `(B,3,75,96,96)`.
- CORS 와일드카드 금지, MIME 2중 검증, `*.pt/*.pth/*.onnx` 커밋 금지.
- confidence = 0~1 정규화값 / raw_score = CTC log-prob 원본.

## 로드맵
- **1단계** 프로젝트 뼈대 ✅ · **2단계** 전처리 파이프라인 ✅
- **3단계** FastAPI 백엔드(Job·파일정리·REST API) ✅
- **4단계** React 프론트엔드(업로드/결과 화면, E2E) ✅
- **5단계** 모델·디코더 + 웹캠 실시간 + 성능 평가 ✅
- **이후** 모델 학습/가중치 확보 → 실제 인식 정확도(CER/WER) 개선, 개선 모델(3D-ResNet/Transformer) 실험
