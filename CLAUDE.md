# 립리딩 프로젝트 — 개발 규칙 (설계문서 완성본 기준)

## 절대 변경 금지 기준값
FPS = 25                          # cap.set 금지 — 타임스탬프 기반 리샘플링 필수
SEQ_LEN = 75                      # 3초 × 25fps
ROI_SIZE = (96, 96)               # 픽셀
ROI_MARGIN = 0.20                 # 상하좌우 20% margin 추가 후 resize
PADDING = 'last-frame'            # zero-padding 절대 금지
PREPROC_SHAPE = (B,T,H,W,C)       # = (B,75,96,96,3) 전처리 출력
MODEL_INPUT_SHAPE = (B,C,T,H,W)   # = (B,3,75,96,96) permute((0,4,1,2,3)) 변환 필수
NORMALIZE = mean[0.485,0.456,0.406] std[0.229,0.224,0.225]

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
# python-magic으로 MIME 2중 검증 (확장자만 믿지 말 것)
# window.isSecureContext 검사 → false이면 HTTPS 모달
# .gitignore: *.pt, *.pth, *.onnx 반드시 등록

## 코드 품질
# Python: 타입 힌트 + docstring 필수
# 모든 함수: try/except 에러 핸들링
# confidence: 0~1 정규화값 / raw_score: CTC log-prob 원본
