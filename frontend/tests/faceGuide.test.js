// faceGuide.test.js — 환자 화면 카메라 위치 안내 판정 (node --test).
import assert from 'node:assert/strict';
import { test } from 'node:test';

import { FACE_ADVICE, FACE_GUIDE, framingAdvice, mirroredBoxStyle, normalizeDetection, stabilizeAdvice } from '../src/ward/faceGuide.js';

const W = 1280;
const H = 720;
/** 비율로 얼굴 하나를 만든다 (입 점은 얼굴 상자의 위에서 75% 지점). */
function face({ w = 0.21, cx = 0.5, cy = 0.64, score = 0.95, mouthY } = {}) {
  const bw = w * W;
  const bh = bw; // 짧은 거리 검출 상자는 거의 정사각형
  const x = cx * W - bw / 2;
  const y = cy * H - bh / 2;
  const my = mouthY ?? (y + bh * 0.75) / H;
  return {
    boundingBox: { originX: x, originY: y, width: bw, height: bh },
    keypoints: [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }, { x: cx, y: my }, { x: 0, y: 0 }, { x: 0, y: 0 }],
    categories: [{ score }],
  };
}

test('인식이 잘 된 조정 영상의 범위(얼굴 너비 0.145~0.344, 중심 0.44~0.54/0.60~0.69)는 늘 "좋아요"', () => {
  for (const w of [0.145, 0.213, 0.344]) {
    for (const cx of [0.443, 0.5, 0.539]) {
      for (const cy of [0.603, 0.64, 0.694]) {
        const out = framingAdvice([face({ w, cx, cy })], W, H);
        assert.equal(out.state, 'ok', JSON.stringify({ w, cx, cy, out: out.state }));
      }
    }
  }
});

test('얼굴 없음·여러 사람·거리·위아래·가로 안내', () => {
  assert.equal(framingAdvice([], W, H).state, 'none');
  assert.equal(framingAdvice([face({ score: 0.3 })], W, H).state, 'none', '확신 낮은 검출은 없는 것으로');
  assert.equal(framingAdvice([face(), face({ w: 0.18, cx: 0.2 })], W, H).state, 'multiple');
  assert.equal(framingAdvice([face(), face({ w: 0.05, cx: 0.2 })], W, H).state, 'ok', '멀리 작게 지나가는 얼굴은 무시');
  assert.equal(framingAdvice([face({ w: 0.08 })], W, H).state, 'far');
  assert.equal(framingAdvice([face({ w: 0.55, cy: 0.5 })], W, H).state, 'near');
  assert.equal(framingAdvice([face({ cy: 0.8, mouthY: 0.93 })], W, H).state, 'low');
  assert.equal(framingAdvice([face({ cy: 0.2 })], W, H).state, 'high');
  assert.equal(framingAdvice([face({ cx: 0.78 })], W, H).state, 'side');
  assert.equal(framingAdvice(null, W, H).state, 'none');
  for (const key of Object.keys(FACE_ADVICE)) assert.ok(FACE_ADVICE[key].text.length > 0);
  assert.ok(FACE_GUIDE.minWidth < 0.145 && FACE_GUIDE.maxWidth > 0.344, '기준은 잘 된 영상 범위보다 넉넉하다');
});

test('깜빡임 줄이기: 다른 안내가 두 번 이어져야 바뀐다', () => {
  const ok = framingAdvice([face()], W, H);
  const far = framingAdvice([face({ w: 0.08 })], W, H);
  let st = stabilizeAdvice(null, ok);
  assert.equal(st.shown.state, 'ok');
  st = stabilizeAdvice(st, far);
  assert.equal(st.shown.state, 'ok', '한 번은 그대로');
  assert.equal(st.shown.box, far.box, '상자 위치는 최신');
  st = stabilizeAdvice(st, far);
  assert.equal(st.shown.state, 'far');
  st = stabilizeAdvice(st, ok);
  st = stabilizeAdvice(st, far);
  assert.equal(st.shown.state, 'far', '번갈아 나오면 바꾸지 않는다');
});

test('거울 미리보기 위 상자 위치와 잘못된 입력', () => {
  assert.deepEqual(mirroredBoxStyle({ x: 0.1, y: 0.2, w: 0.3, h: 0.4 }), { left: '60.00%', top: '20.00%', width: '30.00%', height: '40.00%' });
  assert.equal(mirroredBoxStyle(null), null);
  const edge = mirroredBoxStyle({ x: -0.1, y: 0.9, w: 0.3, h: 0.4 });
  assert.equal(edge.height, '10.00%', '화면 밖은 잘라 낸다');
  assert.equal(normalizeDetection({}, W, H), null);
  assert.equal(normalizeDetection(face(), 0, H), null);
});

test('미리보기 상자: object-fit cover로 잘린 만큼 맞추고, 화면 밖은 잘라 낸다', () => {
  // 16:9 영상을 16:10 칸에 cover로 넣으면 좌우가 잘린다: 가운데 얼굴은 가운데에, 너비는 영상보다 넓게 보인다.
  const view = { videoW: 1280, videoH: 720, boxW: 320, boxH: 200 };
  const st = mirroredBoxStyle({ x: 0.4, y: 0.4, w: 0.2, h: 0.3 }, view);
  const pctNum = (s) => Number(s.replace('%', ''));
  assert.ok(Math.abs(pctNum(st.left) + pctNum(st.width) / 2 - 50) < 0.01, '가운데 얼굴은 가운데');
  assert.ok(pctNum(st.width) > 20, 'cover로 확대된 만큼 넓다');
  assert.equal(st.top, '40.00%', '세로는 잘리지 않음');
  // 원본 오른쪽 끝을 넘는 얼굴(x=0.9, w=0.3): 거울 화면에서는 왼쪽 끝 0~0.1만 보인다.
  const edge = mirroredBoxStyle({ x: 0.9, y: 0.2, w: 0.3, h: 0.3 });
  assert.equal(edge.left, '0.00%');
  assert.equal(edge.width, '10.00%');
  assert.equal(mirroredBoxStyle({ x: 1.2, y: 0.2, w: 0.1, h: 0.1 }), null, '완전히 화면 밖이면 그리지 않음');
});
