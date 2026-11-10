// recorderPreview.test.js — useRecorder 미리보기 연결 동작 검사 (node --test, DOM 라이브러리 없음).
// 등록 화면처럼 카메라를 먼저 열고 <video>를 나중에 그리는 순서에서 미리보기·프레임 기록이 살아 있는지 본다.
// React 대신 아주 작은 훅 실행기를 쓰려고, 훅 소스의 'react' import만 임시 파일의 가짜 모듈로 바꿔 불러온다.
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { pathToFileURL } from 'node:url';

const FAKE_REACT = `
let cur = null;
const same = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => Object.is(x, b[i]));
export function render(inst, fn) { cur = inst; inst.idx = 0; inst.effects = []; try { return fn(); } finally { cur = null; } }
export function commit(inst) { for (const s of inst.effects) { if (typeof s.destroy === 'function') s.destroy(); s.destroy = s.create(); } inst.effects = []; }
export function unmount(inst) { for (const s of inst.hooks) if (s && typeof s.destroy === 'function') { s.destroy(); s.destroy = undefined; } }
export function useRef(init) { const i = cur.idx++; if (!(i in cur.hooks)) cur.hooks[i] = { current: init }; return cur.hooks[i]; }
export function useState(init) {
  const inst = cur; const i = inst.idx++;
  if (!(i in inst.hooks)) { const s = { value: typeof init === 'function' ? init() : init }; s.set = (u) => { s.value = typeof u === 'function' ? u(s.value) : u; }; inst.hooks[i] = s; }
  return [inst.hooks[i].value, inst.hooks[i].set];
}
export function useCallback(fn, deps) { const i = cur.idx++; const p = cur.hooks[i]; if (p && same(p.deps, deps)) return p.fn; cur.hooks[i] = { fn, deps }; return fn; }
export function useEffect(create, deps) {
  const i = cur.idx++; const p = cur.hooks[i]; if (p && same(p.deps, deps)) return;
  const s = { deps, create, destroy: p ? p.destroy : undefined }; cur.hooks[i] = s; cur.effects.push(s);
}
`;

const dir = mkdtempSync(join(tmpdir(), 'recorder-preview-'));
after(() => rmSync(dir, { recursive: true, force: true }));
writeFileSync(join(dir, 'fakeReact.mjs'), FAKE_REACT);
const hookSrc = readFileSync(new URL('../src/hooks/useRecorder.js', import.meta.url), 'utf8')
  .replace(/from 'react';/, `from '${pathToFileURL(join(dir, 'fakeReact.mjs')).href}';`)
  .replace(/from '\.\/cameraErrors\.js';/, `from '${new URL('../src/hooks/cameraErrors.js', import.meta.url).href}';`);
assert.ok(hookSrc.includes('fakeReact.mjs') && !hookSrc.includes("from './cameraErrors.js'"), 'import 바꾸기 실패');
writeFileSync(join(dir, 'useRecorder.mjs'), hookSrc);
const fake = await import(pathToFileURL(join(dir, 'fakeReact.mjs')).href);
const { default: useRecorder } = await import(pathToFileURL(join(dir, 'useRecorder.mjs')).href);

const tick = () => new Promise((r) => setTimeout(r, 0));

/** 가짜 카메라 스트림. */
function makeStream(id) {
  const track = {
    readyState: 'live', kind: 'video', label: 'cam',
    addEventListener() {}, stop() { this.readyState = 'ended'; },
    getSettings: () => ({ width: 1280, height: 720, frameRate: 30 }),
  };
  return {
    id, get active() { return track.readyState !== 'ended'; }, addEventListener() {},
    getTracks: () => [track], getVideoTracks: () => [track], getAudioTracks: () => [],
  };
}

/** 가짜 <video>: srcObject·play·requestVideoFrameCallback만 흉내 낸다. */
function makeVideo() {
  return {
    srcObject: null, muted: false, plays: 0, frameCallbacks: 0,
    play() { this.plays += 1; return this.srcObject ? Promise.resolve() : Promise.reject(Object.assign(new Error('no src'), { name: 'NotSupportedError' })); },
    requestVideoFrameCallback() { this.frameCallbacks += 1; return this.frameCallbacks; },
    cancelVideoFrameCallback() {},
  };
}

/** 훅 하나를 띄운다. 매 호출마다 다시 그린 최신 반환값을 준다. */
function mountHook() {
  const inst = { hooks: [], idx: 0, effects: [] };
  const draw = () => {
    const r = fake.render(inst, () => useRecorder({ maxSeconds: 5 }));
    fake.commit(inst);
    return r;
  };
  return { draw, unmount: () => fake.unmount(inst) };
}

const saved = {};
for (const k of ['window', 'navigator', 'MediaRecorder']) saved[k] = Object.getOwnPropertyDescriptor(globalThis, k);
after(() => {
  for (const [k, d] of Object.entries(saved)) {
    if (d) Object.defineProperty(globalThis, k, d);
    else delete globalThis[k];
  }
});
let opened = [];
Object.defineProperty(globalThis, 'window', { value: { isSecureContext: true }, configurable: true, writable: true });
Object.defineProperty(globalThis, 'navigator', {
  value: { userAgent: 'test', mediaDevices: { getUserMedia: async () => { const s = makeStream(`S${opened.length + 1}`); opened.push(s); return s; } } },
  configurable: true, writable: true,
});
Object.defineProperty(globalThis, 'MediaRecorder', {
  value: class {
    static isTypeSupported(m) { return m === 'video/webm'; }
    constructor(stream) { this.stream = stream; this.state = 'inactive'; }
    start() { this.state = 'recording'; }
    stop() { this.state = 'inactive'; }
  },
  configurable: true, writable: true,
});

test('useRecorder: 카메라를 연 뒤에 <video>가 붙어도 미리보기·프레임 기록이 연결된다 (등록 화면 순서)', async () => {
  opened = [];
  const hook = mountHook();
  let rec = hook.draw();
  assert.equal(typeof rec.videoRef, 'function', 'videoRef는 콜백 ref');
  assert.equal(rec.videoRef.current, null);

  // 이름 화면: <video> 없이 카메라를 연다.
  await rec.open();
  rec = hook.draw();
  assert.equal(rec.status, 'ready');
  assert.equal(opened.length, 1);

  // 명령 화면: <video>가 나중에 붙는다 → 같은 스트림이 연결되고 재생된다.
  const v1 = makeVideo();
  rec.videoRef(v1);
  assert.equal(rec.videoRef.current, v1);
  assert.equal(v1.srcObject, opened[0], '늦게 붙은 <video>에도 열린 스트림이 연결돼야 한다');
  assert.equal(v1.muted, true);
  assert.equal(v1.plays, 1);
  assert.equal(opened.length, 1, '카메라를 다시 열지 않는다');

  // 녹화 시작 시 프레임 시각 기록이 그 <video>에 걸린다.
  rec.start();
  rec = hook.draw();
  assert.equal(rec.status, 'recording');
  assert.ok(v1.frameCallbacks >= 1, 'requestVideoFrameCallback이 걸려야 프레임 기록이 찬다');
  rec.stop();

  // '다른 등록 이름'으로 갔다가 다시 오면 <video>가 떨어졌다 새로 붙는다 → 다시 연결된다.
  rec.videoRef(null);
  assert.equal(rec.videoRef.current, null);
  const v2 = makeVideo();
  rec.videoRef(v2);
  assert.equal(v2.srcObject, opened[0]);
  assert.equal(v2.plays, 1);

  // 화면을 떠나면 카메라를 끄고 미리보기를 비운다.
  hook.unmount();
  assert.equal(opened[0].active, false);
  assert.equal(v2.srcObject, null);
  await tick();
});

test('useRecorder: <video>가 먼저 있는 화면(병실·녹화·촬영)도 예전처럼 open() 때 연결되고, 다시 열면 새 스트림으로 바뀐다', async () => {
  opened = [];
  const hook = mountHook();
  let rec = hook.draw();
  const v = makeVideo();
  rec.videoRef(v); // 카메라를 열기 전에 붙음: 스트림이 없으니 아무것도 하지 않는다.
  assert.equal(v.srcObject, null);
  assert.equal(v.plays, 0);
  await rec.open();
  rec = hook.draw();
  assert.equal(rec.status, 'ready');
  assert.equal(v.srcObject, opened[0]);
  // 카메라 다시 열기: 옛 스트림은 멈추고 붙어 있는 <video>에 새 스트림을 연결한다.
  await rec.open();
  rec = hook.draw();
  assert.equal(opened.length, 2);
  assert.equal(opened[0].active, false);
  assert.equal(v.srcObject, opened[1]);
  assert.equal(rec.status, 'ready');
  hook.unmount();
  await tick();
});
