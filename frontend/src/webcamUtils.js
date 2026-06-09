// webcamUtils.js — V-VAD/ROI 순수 헬퍼 (WebcamPage에서 사용).

import {
  FACE_WIDTH_IDX,
  HEADER_BYTES,
  MSG_TYPE_FRAME_DATA,
  ROI_BYTES,
  ROI_CROP_IDX,
  ROI_MARGIN,
  ROI_SIZE,
  VVAD_IDX_BOT,
  VVAD_IDX_TOP,
} from './constants';

/** 랜드마크 idx 집합의 평균 좌표(정규화 [0,1]). */
function avgPoint(landmarks, idxs) {
  let x = 0;
  let y = 0;
  for (const i of idxs) {
    x += landmarks[i].x;
    y += landmarks[i].y;
  }
  return { x: x / idxs.length, y: y / idxs.length };
}

/**
 * mouth_ratio = d_raw / W.
 * d_raw = 윗입술 내측 평균 ↔ 아랫입술 내측 평균 유클리드 거리.
 * W = 좌우 광대 랜드마크 거리 (카메라 거리 정규화).
 * @param {Array<{x:number,y:number}>} landmarks - 정규화 랜드마크.
 * @returns {number}
 */
export function computeMouthRatio(landmarks) {
  const top = avgPoint(landmarks, VVAD_IDX_TOP);
  const bot = avgPoint(landmarks, VVAD_IDX_BOT);
  const dRaw = Math.hypot(top.x - bot.x, top.y - bot.y);
  const w = Math.abs(
    landmarks[FACE_WIDTH_IDX[0]].x - landmarks[FACE_WIDTH_IDX[1]].x,
  );
  return w > 0 ? dRaw / w : 0;
}

/**
 * ROI_CROP_IDX 랜드마크로 입술 bbox(px)를 구하고 margin 적용 후 클램프.
 * @returns {{x:number,y:number,w:number,h:number}}
 */
export function computeRoiBBox(landmarks, videoW, videoH, margin = ROI_MARGIN) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const i of ROI_CROP_IDX) {
    const p = landmarks[i];
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  let x1 = minX * videoW;
  let y1 = minY * videoH;
  let x2 = maxX * videoW;
  let y2 = maxY * videoH;
  const bw = x2 - x1;
  const bh = y2 - y1;
  x1 -= bw * margin;
  y1 -= bh * margin;
  x2 += bw * margin;
  y2 += bh * margin;
  x1 = Math.max(0, Math.min(x1, videoW - 1));
  y1 = Math.max(0, Math.min(y1, videoH - 1));
  x2 = Math.max(x1 + 1, Math.min(x2, videoW));
  y2 = Math.max(y1 + 1, Math.min(y2, videoH));
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/**
 * 비디오의 bbox 영역을 96×96으로 크롭하여 RGB Uint8(27,648B) 반환.
 * @returns {Uint8Array}
 */
export function extractRoiBytes(video, bbox, canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  canvas.width = ROI_SIZE;
  canvas.height = ROI_SIZE;
  ctx.drawImage(video, bbox.x, bbox.y, bbox.w, bbox.h, 0, 0, ROI_SIZE, ROI_SIZE);
  const { data } = ctx.getImageData(0, 0, ROI_SIZE, ROI_SIZE); // RGBA
  const out = new Uint8Array(ROI_BYTES);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    out[j++] = data[i];
    out[j++] = data[i + 1];
    out[j++] = data[i + 2];
  }
  return out;
}

/**
 * 바이너리 프레임 생성: [chunk_id(Uint32 BE)][0x01][reserved×3][ROI Uint8].
 * 서버(ws_handler)가 기대하는 8B 헤더 + 27,648B 포맷.
 * @returns {ArrayBuffer}
 */
export function buildBinaryFrame(chunkId, roiBytes) {
  const buf = new ArrayBuffer(HEADER_BYTES + roiBytes.length);
  const dv = new DataView(buf);
  dv.setUint32(0, chunkId >>> 0, false); // Big-Endian
  dv.setUint8(4, MSG_TYPE_FRAME_DATA);
  new Uint8Array(buf, HEADER_BYTES).set(roiBytes);
  return buf;
}

/** 평균. */
export function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/** 분산. */
export function variance(arr, m) {
  if (arr.length === 0) return 0;
  const mu = m === undefined ? mean(arr) : m;
  return arr.reduce((a, b) => a + (b - mu) * (b - mu), 0) / arr.length;
}

/** 표준편차. */
export function std(arr, m) {
  return Math.sqrt(variance(arr, m));
}
