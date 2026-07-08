# 립리딩 프로젝트 — 개발 규칙 (설계문서 완성본 기준)

## 절대 변경 금지 기준값
FPS = 25                          # cap.set(CAP_PROP_FPS) 금지 — 타임스탬프 기반 리샘플링 필수
SEQ_LEN = 75                      # 3초 × 25fps (추론 세그먼트 명목값)
ROI_SIZE = (96, 96)               # 픽셀
ROI_MARGIN = 0.20                 # 상하좌우 20% margin 추가 후 resize
PADDING = 'last-frame'            # zero-padding 절대 금지
PREPROC_SHAPE = (B,T,H,W,C)       # = (B,75,96,96,3) 전처리 출력
MODEL_INPUT_SHAPE = (B,C,T,H,W)   # = (B,3,75,96,96) permute((0,4,1,2,3)) 변환 필수
NORMALIZE = mean[0.485,0.456,0.406] std[0.229,0.224,0.225]
# ※ (B,75,...)의 75는 '추론 세그먼트' 명목값이다. 학습은 문장 단위 가변 길이 T를
#   그대로 쓴다(모델이 forward에서 T를 런타임 판독 — baseline.py). SEQ_LEN을 학습
#   코드에서 고정 상수로 쓰지 말 것. temporal 풀링 금지(MaxPool3d 커널 (1,2,2))로
#   출력 시퀀스 T'=T를 유지 → CTC 조건(T' ≥ 라벨 정렬 길이) 충족.

## 랜드마크 인덱스 (MediaPipe Face Mesh)
ROI_CROP_IDX  = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]  # 외측 20개
VVAD_IDX_TOP  = [13,14,312,311,310,415]   # 윗입술 내측 — d_raw 중심점
VVAD_IDX_BOT  = [17,18,84,181,180,314]    # 아랫입술 내측 — d_raw 중심점
# d_raw = 윗입술 평균좌표 ↔ 아랫입술 평균좌표 유클리드 거리
# mouth_ratio = d_raw / W (W: 좌우 광대 랜드마크 거리)

## API 핵심 규칙
# POST /api/infer → results 즉시 반환 절대 금지
#   반환: {job_id, session_id, status:'processing'}
#   결과: GET /api/result/{session_id} 폴링 전용
# CORS: allow_origins 명시 화이트리스트 (와일드카드 '*' 금지)
# test_set_path 직접 전달 금지 → test_set_id 사용
# 임시 파일: 추론 완료 후 즉시 os.remove()

## WebSocket 규격
# 업링크: JSON 제어 메시지 선행 → Binary ArrayBuffer (Uint8, 27,648 bytes)
# Float32 직접 전송 금지 (110,592 bytes — MVP 미사용)
# 제어: {type:'SESSION_START', session_id, client_ts}
#       {type:'CHUNK_START', chunk_id, window_start_ms, is_final}

## 보안
# MIME 2중 검증 (확장자만 믿지 말 것): python-magic 우선, 부재 시 파일 시그니처 폴백
#   (python-magic은 서브프로세스 probe 후에만 in-process import — 아래 환경·실행 참조)
# window.isSecureContext 검사 → false이면 HTTPS 모달
# .gitignore: *.pt, *.pth, *.onnx 반드시 등록

## 코드 품질
# Python: 타입 힌트 + docstring 필수
# 모든 함수: try/except 에러 핸들링
# confidence: 0~1 정규화값 / raw_score: CTC log-prob 원본

## 운영/구현 노트 (데모 검증 2026-06-10 · 학습 파이프라인 2026-07 반영)

### 환경 · 실행
# torch + mediapipe 동시 사용 시, 단독 스크립트(python -c / .py 직접 실행)에서
#   import 순서에 따라 segfault(exit 139) 발생 가능. pytest 환경에서는 정상.
#   → ML 라이브러리(torch, mediapipe)는 모듈 상단에서 일관된 순서로 먼저 import.
#     학습/추론 스크립트는 동일 import 순서를 유지하고, 검증은 pytest 경유 권장.
#   → prepare(mediapipe 전용) / train·test(torch 전용) 프로세스를 분리해 충돌 회피.
# python-magic은 Windows에서 libmagic DLL 미탑재 시 `import magic` 자체가 native
#   access violation을 던진다. cv2가 먼저 로드된 서버 프로세스에서는 이 크래시가
#   try/except로 못 잡히는 하드 segfault(exit 139)로 번져 uvicorn이 조용히 죽는다.
#   → upload.py는 격리 서브프로세스에서 magic import를 먼저 probe(_probe_magic_safe)
#     하고, 통과할 때만 in-process import한다. 실패 시 파일 시그니처(ftyp / RIFF-AVI)
#     폴백으로 '확장자 + 내용' 2중 검증을 유지한다. (본 프로세스에서 무방비 import 금지)
# 백그라운드 스크립트 로그가 안 보이면 stdout 버퍼링 문제 → python -u 또는 flush=True.
# Windows 콘솔(cp949)에서 한글 출력 크래시 → sys.stdout.reconfigure(encoding="utf-8").

### 성능
# MediaPipe Face Mesh는 1920x1080 프레임 CPU 처리 시 매우 느림(프레임당 비용 큼).
#   → 얼굴 검출 입력 프레임은 적정 해상도(예: 가로 640px)로 축소 후 처리 권장.
#   ★ 단, 학습·추론 전처리는 반드시 동일 기준(해상도 축소 포함)을 적용하고,
#     최종 ROI 출력은 96x96 불변. (전처리 불일치 시 모델 성능 저하)
# 학습 데이터 준비는 라벨 입술좌표 ROI(--roi bbox)가 MediaPipe보다 ~2.4배 빠르고
#   미검출이 없다(아래 데이터 항 참조). ROI margin/리사이즈 규약은 crop_roi_from_box로
#   MediaPipe 경로와 동일하게 맞춰 학습/추론 ROI 정의를 일치시킨다.

### 데이터 (AI Hub)
# 원본 영상 = 30fps · 1920x1080 · 약 5분(다문장). 그대로는 MAX_DURATION_SEC(180s) 초과.
#   → 라벨 Sentence_info[].start_time/end_time 기준으로 문장 단위 클립을 잘라 투입.
# 문장은 '가/안녕' 단어가 아니라 긴 문장: 영상당 51문장, 평균 5초·최대 ~12초,
#   자모 라벨 평균 66·최대 125. 대부분 3초(SEQ_LEN) 초과 → 학습은 가변 길이로 처리.
# ★ 좌표계/회전 함정: 라벨 JSON에 Bounding_box_info.Lip_bounding_box(원본 fps 프레임별
#   입술 박스)가 있다. 단, 영상은 세로(1080×1920) 촬영·라벨링이나 파일은 가로(1920×1080)
#   저장이라 라벨 y좌표가 프레임 높이(1080)를 초과하며 좌표계가 90° 어긋난다(회전 플래그
#   미적용). → cv2 프레임을 반시계(CCW) 90° 회전해야 라벨과 정합(A/B/C/D 각도 동일).
#   scripts/prepare_data._detect_rotation이 자동 감지. (bbox 미회전 크롭 시 눈·옷을 자름.)
#   기존 MediaPipe 학습분은 회전 미적용 '옆으로 누운 얼굴'이라 ROI 품질 저하 가능성 有.

### 학습 파이프라인 (2026-07)
# 문장 단위 절단·라벨 인코딩: scripts/prepare_data.py, 대량은 prepare_batch.py.
#   --roi {mediapipe|bbox|auto}, --max-sentences(기본 None=전체 51문장),
#   --max-frames(0=무제한, 초과 클립 스킵), --scale(mediapipe 다운스케일 폭).
# CTC 가용성: 준비 단계에서 T ≥ (라벨 길이 + 인접 중복 자모 수) 검사(ctc_min_input_len).
#   미달 클립은 스킵. CTCLoss는 blank=0, zero_infinity=True.
# 학습: scripts/train.py — 가변 길이 last-frame 패딩 collate, 길이 버킷 배치 샘플러
#   (LengthBucketBatchSampler)로 패딩 낭비 최소화, gradient clipping(--grad-clip 기본 1.0),
#   train/val 분리·val CER·best 체크포인트. torch 전용(mediapipe 미import).
# 성능 테스트: scripts/test_model.py(검증셋/전체 CER). 가중치 *.pt 커밋 금지.
# 주의: 검증/스모크 학습은 반드시 --out 임시 경로로 저장(기본값은 배포 baseline.pt를 덮음).

### Git 위생
# logs/ · tmp/ · *.log 는 하위 경로 포함 전역 gitignore (루트 전용 패턴은
#   backend/logs/ 등 하위 디렉토리를 놓침).
# *.pt / *.pth / *.onnx 커밋 금지 (GitHub 100MB 제한).
