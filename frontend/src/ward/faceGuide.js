// faceGuide.js — 환자 화면 카메라 위치 안내의 판정 순수 함수 (node --test로 검사).
// 브라우저의 얼굴 검출(MediaPipe FaceDetector, 짧은 거리 모델) 결과를 받아 "좋아요 / 더 가까이 / 뒤로 / 가운데로 …" 안내를 고른다.
// 안내일 뿐이며 인식·판정에는 쓰지 않는다(서버 입술 추출과 기준값은 그대로).
//
// 기준은 실제로 인식이 잘 된 조정 참가자 영상에서 잰 범위보다 넉넉하게 잡았다 (2026-11-25, P01·P03 병실·문장 영상 47장,
// 1280×720): 얼굴 너비 0.145~0.344(중앙 0.213), 가로 중심 0.443~0.539, 세로 중심 0.603~0.694, 입 높이 0.659~0.796,
// 얼굴 아래 끝 0.749~0.946. 그래서 이 범위 안이면 늘 '좋아요'가 나온다.

/** 판정 기준 (화면 크기에 대한 비율). */
export const FACE_GUIDE = Object.freeze({
  minWidth: 0.11, // 얼굴 너비가 이보다 작으면 '더 가까이'
  maxWidth: 0.5, // 이보다 크면 '조금 뒤로'
  maxCenterOffsetX: 0.2, // 가로 중심이 0.5에서 이만큼 넘게 벗어나면 '가운데로'
  minCenterY: 0.25, // 세로 중심이 이보다 위면 '너무 위'
  maxMouthY: 0.9, // 입 위치가 이보다 아래면 '입이 화면 아래에 걸림'
  maxBottom: 1.08, // 얼굴 상자가 화면 아래로 크게 벗어나면 '입이 화면 아래에 걸림' (턱만 살짝 잘리는 것은 괜찮다)
  secondFaceRatio: 0.6, // 두 번째 얼굴이 첫 얼굴 너비의 이 비율 이상이면 '한 사람만'
  minScore: 0.5,
});

/** 안내 종류와 문구. tone: ok(초록) / adjust(노랑) / none(회색). */
export const FACE_ADVICE = Object.freeze({
  ok: Object.freeze({ state: 'ok', tone: 'ok', text: '얼굴 위치가 좋아요' }),
  // 짧은 거리 검출 모델은 너무 멀거나 작은 얼굴을 찾지 못하므로 '안 보임'에 '가까이'도 함께 안내한다.
  none: Object.freeze({ state: 'none', tone: 'none', text: '얼굴이 보이지 않아요 · 카메라 가까이, 화면 가운데로 와 주세요' }),
  multiple: Object.freeze({ state: 'multiple', tone: 'adjust', text: '한 사람만 화면에 보이게 해 주세요' }),
  far: Object.freeze({ state: 'far', tone: 'adjust', text: '조금 더 가까이 와 주세요' }),
  near: Object.freeze({ state: 'near', tone: 'adjust', text: '조금만 뒤로 가 주세요' }),
  low: Object.freeze({ state: 'low', tone: 'adjust', text: '입이 화면 아래에 걸려요 · 카메라를 조금 낮추거나 고개를 들어 주세요' }),
  high: Object.freeze({ state: 'high', tone: 'adjust', text: '얼굴이 너무 위에 있어요 · 카메라를 조금 올려 주세요' }),
  side: Object.freeze({ state: 'side', tone: 'adjust', text: '얼굴을 화면 가운데로 옮겨 주세요' }),
});

/**
 * 검출 결과 하나를 비율 상자로 바꾼다.
 * @param {{boundingBox?: {originX:number, originY:number, width:number, height:number}, keypoints?: Array<{x:number,y:number}>,
 *   categories?: Array<{score:number}>}} det
 * @param {number} width 영상 가로 픽셀.
 * @param {number} height 영상 세로 픽셀.
 * @returns {{x:number, y:number, w:number, h:number, cx:number, cy:number, mouthY:number|null, score:number}|null}
 */
export function normalizeDetection(det, width, height) {
  try {
    const b = det?.boundingBox;
    if (!b || !(width > 0) || !(height > 0)) return null;
    const x = b.originX / width;
    const y = b.originY / height;
    const w = b.width / width;
    const h = b.height / height;
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;
    // 짧은 거리 얼굴 검출의 점 6개: 오른눈·왼눈·코끝·입 가운데·오른귀·왼귀 (비율 좌표).
    const mouth = Array.isArray(det.keypoints) ? det.keypoints[3] : null;
    const mouthY = mouth && Number.isFinite(mouth.y) ? mouth.y : null;
    const score = Number(det.categories?.[0]?.score ?? 1);
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, mouthY, score: Number.isFinite(score) ? score : 0 };
  } catch {
    return null;
  }
}

/**
 * 얼굴 검출 결과로 안내 하나를 고른다. 순서: 얼굴 없음 → 여러 사람 → 거리 → 위아래 → 가로 → 좋아요.
 * @param {Array<object>} detections FaceDetector 결과의 detections.
 * @param {number} width 영상 가로 픽셀.
 * @param {number} height 영상 세로 픽셀.
 * @returns {{state:string, tone:string, text:string, box: object|null}}
 */
export function framingAdvice(detections, width, height) {
  try {
    const faces = (Array.isArray(detections) ? detections : [])
      .map((d) => normalizeDetection(d, width, height))
      .filter((f) => f && f.score >= FACE_GUIDE.minScore)
      .sort((a, b) => b.w - a.w);
    if (!faces.length) return { ...FACE_ADVICE.none, box: null };
    const face = faces[0];
    const pick = (advice) => ({ ...advice, box: face });
    if (faces.length > 1 && faces[1].w >= face.w * FACE_GUIDE.secondFaceRatio) return pick(FACE_ADVICE.multiple);
    if (face.w < FACE_GUIDE.minWidth) return pick(FACE_ADVICE.far);
    if (face.w > FACE_GUIDE.maxWidth) return pick(FACE_ADVICE.near);
    if ((face.mouthY !== null && face.mouthY > FACE_GUIDE.maxMouthY) || face.y + face.h > FACE_GUIDE.maxBottom) {
      return pick(FACE_ADVICE.low);
    }
    if (face.cy < FACE_GUIDE.minCenterY) return pick(FACE_ADVICE.high);
    if (Math.abs(face.cx - 0.5) > FACE_GUIDE.maxCenterOffsetX) return pick(FACE_ADVICE.side);
    return pick(FACE_ADVICE.ok);
  } catch {
    return { ...FACE_ADVICE.none, box: null };
  }
}

/**
 * 깜빡임을 줄인다: 새 안내가 연달아 `needed`번 나와야 화면 안내를 바꾼다 (상자 위치는 늘 최신).
 * @param {{shown: object|null, candidate: string|null, count: number}} prev
 * @param {{state:string}} next framingAdvice 결과.
 * @param {number} [needed]
 * @returns {{shown: object, candidate: string, count: number}}
 */
export function stabilizeAdvice(prev, next, needed = 2) {
  const p = prev || { shown: null, candidate: null, count: 0 };
  if (!p.shown) return { shown: next, candidate: next.state, count: needed };
  if (next.state === p.shown.state) return { shown: next, candidate: next.state, count: needed };
  const count = p.candidate === next.state ? p.count + 1 : 1;
  if (count >= needed) return { shown: next, candidate: next.state, count };
  return { shown: { ...p.shown, box: next.box }, candidate: next.state, count };
}

/**
 * 거울처럼 뒤집어 보이는 미리보기 위에 그릴 상자 위치 (CSS 백분율).
 * 미리보기는 object-fit: cover라 영상과 칸의 가로세로 비율이 다르면 영상 가장자리가 잘린다. view를 주면 그만큼 맞춰 그린다.
 * 화면 밖으로 나간 부분은 잘라 내고, 남는 부분이 없으면 null이다.
 * @param {{x:number, y:number, w:number, h:number}|null} box 원본 영상 기준 비율 상자(거울 뒤집기 전).
 * @param {{videoW:number, videoH:number, boxW:number, boxH:number}} [view] 영상 픽셀 크기와 미리보기 칸 크기.
 * @returns {{left:string, top:string, width:string, height:string}|null}
 */
export function mirroredBoxStyle(box, view) {
  try {
    if (!box) return null;
    // 거울 뒤집기: 원본 x 구간 [x, x+w] → 화면 [1-(x+w), 1-x].
    let x0 = 1 - (box.x + box.w);
    let x1 = 1 - box.x;
    let y0 = box.y;
    let y1 = box.y + box.h;
    const { videoW, videoH, boxW, boxH } = view || {};
    if ([videoW, videoH, boxW, boxH].every((v) => Number.isFinite(v) && v > 0)) {
      const scale = Math.max(boxW / videoW, boxH / videoH); // object-fit: cover
      const shownW = (videoW * scale) / boxW; // 칸 너비에 대한 영상 너비 비율 (1 이상)
      const shownH = (videoH * scale) / boxH;
      const offX = (1 - shownW) / 2;
      const offY = (1 - shownH) / 2;
      [x0, x1] = [offX + x0 * shownW, offX + x1 * shownW];
      [y0, y1] = [offY + y0 * shownH, offY + y1 * shownH];
    }
    const clamp = (v) => Math.max(0, Math.min(1, v));
    const left = clamp(x0);
    const right = clamp(x1);
    const top = clamp(y0);
    const bottom = clamp(y1);
    if (right - left <= 0 || bottom - top <= 0) return null;
    const pct = (v) => `${(v * 100).toFixed(2)}%`;
    return { left: pct(left), top: pct(top), width: pct(right - left), height: pct(bottom - top) };
  } catch {
    return null;
  }
}
