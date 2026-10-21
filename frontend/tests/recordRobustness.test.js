// recordRobustness.test.js — 폴링 404 유예·카메라 오류 문구·포커스 해제 순수 함수 검사 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  NETWORK_ERROR, NOT_FOUND_GRACE_POLLS, POLL_ERROR, START_ERROR, errorMessageFrom, interpretPoll, isGraceNotFound,
  withNotFoundGrace,
} from '../src/api/pollOutcome.js';
import {
  CAMERA_BUSY_ERROR, CAMERA_LOST_ERROR, CAMERA_NOT_FOUND_ERROR, CAMERA_PERMISSION_ERROR, INSECURE_CONTEXT_ERROR,
  cameraErrorType, cameraOpenErrorMessage, clientCameraError, isStreamLost, logCameraError,
} from '../src/hooks/cameraErrors.js';
import { blurAfterPointerClick, isPointerClick } from '../src/record/pointerFocus.js';

const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const httpError = (status, detail) => Object.assign(new Error(`Request failed with status code ${status}`), {
  response: { status, data: detail === undefined ? {} : { detail } },
});
const named = (name, message = 'english browser text') => Object.assign(new Error(message), { name });

test('isGraceNotFound: 처음 2번의 404만 유예, 그 밖의 오류·순번은 유예하지 않는다', () => {
  assert.equal(NOT_FOUND_GRACE_POLLS, 2);
  const nf = httpError(404, '세션을 찾을 수 없습니다.');
  assert.equal(isGraceNotFound(nf, 0), true);
  assert.equal(isGraceNotFound(nf, 1), true);
  assert.equal(isGraceNotFound(nf, 2), false);
  assert.equal(isGraceNotFound(nf, -1), false);
  assert.equal(isGraceNotFound(nf, 0.5), false);
  assert.equal(isGraceNotFound(httpError(500), 0), false);
  assert.equal(isGraceNotFound(Object.assign(new Error('Network Error'), { request: {} }), 0), false);
  assert.equal(isGraceNotFound(null, 0), false);
  assert.equal(isGraceNotFound(nf, 2, 3), true, '유예 횟수 지정');
});

test('withNotFoundGrace: 초기 404는 pending, 3번째 404는 그대로 오류', async () => {
  const nf = httpError(404, '세션을 찾을 수 없습니다.');
  let calls = 0;
  const poll = withNotFoundGrace(async () => { calls += 1; throw nf; });
  assert.deepEqual(await poll(), { status: 'pending', progress: null });
  assert.deepEqual(await poll(), { status: 'pending', progress: null });
  await assert.rejects(poll(), (e) => e === nf);
  assert.equal(calls, 3);
  // 유예된 pending은 화면 동작상 진행 중(진행률 갱신 없음)이다.
  assert.deepEqual(interpretPoll({ status: 'pending', progress: null }), { kind: 'progress', progress: null });
  // 유예 뒤 404의 화면 문구는 서버의 한국어 detail.
  assert.equal(errorMessageFrom(nf, POLL_ERROR), '세션을 찾을 수 없습니다.');
});

test('withNotFoundGrace: 404 다음 정상 응답은 그대로, 404가 아닌 첫 오류는 즉시 오류', async () => {
  const replies = [httpError(404), { status: 'processing', progress: 10 }, httpError(404), { status: 'done', results: [] }];
  let i = 0;
  const poll = withNotFoundGrace(async () => {
    const r = replies[i]; i += 1;
    if (r instanceof Error) throw r;
    return r;
  });
  assert.deepEqual(await poll(), { status: 'pending', progress: null });
  assert.deepEqual(await poll(), { status: 'processing', progress: 10 });
  await assert.rejects(poll(), /404/, '3번째 폴링(순번2)의 404는 유예하지 않는다');
  assert.deepEqual(await poll(), { status: 'done', results: [] });

  const boom = httpError(500);
  const poll2 = withNotFoundGrace(async () => { throw boom; });
  await assert.rejects(poll2(), (e) => e === boom);
});

test('withNotFoundGrace: 순번은 요청 순서로 센다 (동시에 걸린 요청도 각자 순번)', async () => {
  const resolvers = [];
  const poll = withNotFoundGrace(() => new Promise((_res, rej) => { resolvers.push(rej); }));
  const p0 = poll(); const p1 = poll(); const p2 = poll();
  resolvers[2](httpError(404)); resolvers[1](httpError(404)); resolvers[0](httpError(404));
  assert.deepEqual(await p0, { status: 'pending', progress: null });
  assert.deepEqual(await p1, { status: 'pending', progress: null });
  await assert.rejects(p2, /404/);
});

test('UploadPage 오류 문구: 한국어 고정 문구만 (axios 영어 원문 없음)', () => {
  assert.equal(errorMessageFrom(httpError(404), START_ERROR), START_ERROR, 'detail 없는 404');
  assert.equal(errorMessageFrom(httpError(400, '지원하지 않는 형식입니다.'), START_ERROR), '지원하지 않는 형식입니다.');
  assert.equal(errorMessageFrom(Object.assign(new Error('Network Error'), { request: {} }), START_ERROR), NETWORK_ERROR);
  assert.equal(errorMessageFrom(new Error('missing session_id'), START_ERROR), START_ERROR);
  const check = Object.assign(new Error('3분 이하의 영상만 업로드 가능합니다.'), { isClientCheck: true });
  assert.equal(errorMessageFrom(check, START_ERROR), '3분 이하의 영상만 업로드 가능합니다.');
  assert.equal(errorMessageFrom(httpError(500), POLL_ERROR), POLL_ERROR);
  assert.doesNotMatch(POLL_ERROR + START_ERROR, /[A-Za-z]/);
});

test('UploadPage 소스 계약: startInfer 응답 뒤에 INFERRING(폴러 마운트), errorMessageFrom 사용', () => {
  const page = src('../src/pages/UploadPage.jsx');
  const awaitAt = page.indexOf('await startInfer(');
  const phaseAt = page.indexOf("setPhase('INFERRING')");
  const sidAt = page.indexOf('setSessionId(up.session_id)');
  assert.ok(awaitAt > 0 && phaseAt > awaitAt && sidAt > awaitAt, 'Job 생성 전에 폴링을 시작하지 않는다');
  assert.match(page, /errorMessageFrom\(err, START_ERROR\)/);
  assert.match(page, /errorMessageFrom\(err, message\)/);
  assert.doesNotMatch(page, /err\?\.message/);
  const poller = src('../src/components/ResultPoller.jsx');
  assert.match(poller, /withNotFoundGrace\(/);
  assert.match(poller, /errorMessageFrom\(error, POLL_ERROR\)/);
  assert.doesNotMatch(poller, /error\.message/);
  const client = src('../src/api/client.js');
  assert.equal((client.match(/throw clientCheckError\(/g) || []).length, 3, '업로드 1차 검증 문구는 isClientCheck');
});

test('cameraOpenErrorMessage: 오류 이름별 고정 한국어, 모르는 오류는 null', () => {
  assert.equal(cameraOpenErrorMessage(named('NotAllowedError')), CAMERA_PERMISSION_ERROR);
  assert.equal(CAMERA_PERMISSION_ERROR, '웹캠 접근 권한이 필요합니다.');
  assert.equal(cameraOpenErrorMessage(named('NotFoundError', 'Requested device not found')), '카메라를 찾을 수 없습니다.');
  assert.equal(cameraOpenErrorMessage(named('OverconstrainedError')), CAMERA_NOT_FOUND_ERROR);
  assert.equal(cameraOpenErrorMessage(named('NotReadableError', 'Could not start video source')), '다른 프로그램이 카메라를 사용 중입니다.');
  assert.equal(cameraOpenErrorMessage(named('AbortError')), CAMERA_BUSY_ERROR);
  assert.equal(cameraOpenErrorMessage(named('TypeError')), null);
  assert.equal(cameraOpenErrorMessage(named('SecurityError')), null);
  assert.equal(cameraOpenErrorMessage(new Error('anything')), null);
  assert.equal(cameraOpenErrorMessage(null), null);
  assert.equal(cameraOpenErrorMessage(undefined), null);
  assert.equal(cameraOpenErrorMessage(clientCameraError(INSECURE_CONTEXT_ERROR)), INSECURE_CONTEXT_ERROR);
  assert.equal(CAMERA_LOST_ERROR, '카메라 연결이 끊겼습니다. 카메라를 다시 열어 주세요.');
});

test('cameraErrorType: 권한 거부만 PERMISSION_DENIED', () => {
  assert.equal(cameraErrorType(named('NotAllowedError')), 'PERMISSION_DENIED');
  for (const n of ['NotFoundError', 'NotReadableError', 'OverconstrainedError', 'AbortError', 'TypeError']) {
    assert.equal(cameraErrorType(named(n)), 'DEVICE_ERROR', n);
  }
  assert.equal(cameraErrorType(null), 'DEVICE_ERROR');
});

test('logCameraError: 원문은 콘솔(logger)에만, 로거 오류는 삼킨다', () => {
  const lines = [];
  logCameraError('카메라 열기 실패', named('NotReadableError', 'Could not start video source'), { warn: (...a) => lines.push(a) });
  assert.deepEqual(lines, [['[camera] 카메라 열기 실패', 'NotReadableError', 'Could not start video source']]);
  assert.doesNotThrow(() => logCameraError('x', null, { warn: () => { throw new Error('no console'); } }));
});

test('isStreamLost: inactive 스트림·끝난 트랙·트랙 없음은 끊김', () => {
  const track = (readyState) => ({ readyState });
  assert.equal(isStreamLost({ active: true, getVideoTracks: () => [track('live')] }), false);
  assert.equal(isStreamLost({ active: false, getVideoTracks: () => [track('live')] }), true);
  assert.equal(isStreamLost({ active: true, getVideoTracks: () => [track('ended')] }), true);
  assert.equal(isStreamLost({ active: true, getVideoTracks: () => [] }), true);
  assert.equal(isStreamLost({ getVideoTracks: () => [track('live'), track('ended')] }), false);
  assert.equal(isStreamLost(null), true);
});

test('useRecorder 소스 계약: 트랙 ended·스트림 inactive 감시, MediaRecorder 생성·start 보호', () => {
  const hook = src('../src/hooks/useRecorder.js');
  assert.match(hook, /addEventListener\('ended', onLost\)/);
  assert.match(hook, /addEventListener\('inactive', onLost\)/);
  assert.match(hook, /try \{\s*rec = new MediaRecorder\(/);
  assert.match(hook, /rec\.start\(250\);\s*\} catch/);
  assert.match(hook, /cameraOpenErrorMessage\(e\)/);
  assert.doesNotMatch(hook, /setError\(e\.message\)|: e\.message\)/, '브라우저 영어 원문을 화면 문구로 쓰지 않는다');
  assert.match(src('../src/pages/CapturePage.jsx'), /errorSeq > 0\) setModal\(\{ type: errorType \|\| 'DEVICE_ERROR', message: error \|\| undefined \}\)/);
  // RecordPage: 업로드·분석 중이거나 업로드 오류 안내가 떠 있으면 카메라 안내로 덮지 않는다.
  const record = src('../src/pages/RecordPage.jsx');
  assert.match(record, /errorSeq <= 0 \|\| work === 'UPLOADING' \|\| work === 'INFERRING'\) return;/);
  assert.match(record, /current\?\.type === 'UPLOAD_ERROR'\s*\? current\s*: \{ type: errorType \|\| 'DEVICE_ERROR', message: error \|\| undefined \}/);
});

test('isPointerClick: 마우스 클릭(detail>0)만', () => {
  assert.equal(isPointerClick({ detail: 1 }), true);
  assert.equal(isPointerClick({ detail: 2 }), true);
  assert.equal(isPointerClick({ detail: 0 }), false, '키보드 Space/Enter 활성화');
  assert.equal(isPointerClick({}), false);
  assert.equal(isPointerClick(null), false);
});

test('blurAfterPointerClick: 마우스 클릭 뒤 컨테이너 안 포커스만 해제, 키보드는 유지', () => {
  const mk = () => {
    const el = { blurred: 0, blur() { this.blurred += 1; } };
    const outside = { blurred: 0, blur() { this.blurred += 1; } };
    const container = { contains: (n) => n === el };
    return { el, outside, container };
  };
  const queue = [];
  const schedule = (fn) => queue.push(fn);

  const a = mk();
  assert.equal(blurAfterPointerClick({ detail: 1, currentTarget: a.container }, { doc: { activeElement: a.el }, schedule }), true);
  assert.equal(a.el.blurred, 0, '기본 동작이 끝난 뒤(다음 작업)에 해제');
  queue.shift()();
  assert.equal(a.el.blurred, 1);

  const b = mk();
  assert.equal(blurAfterPointerClick({ detail: 0, currentTarget: b.container }, { doc: { activeElement: b.el }, schedule }), false);
  assert.equal(queue.length, 0, '키보드 활성화는 포커스 유지');

  const c = mk();
  blurAfterPointerClick({ detail: 1, currentTarget: c.container }, { doc: { activeElement: c.outside }, schedule });
  queue.shift()();
  assert.equal(c.outside.blurred, 0, '컨테이너 밖 포커스는 건드리지 않음');

  assert.equal(blurAfterPointerClick({ detail: 1 }, { doc: { activeElement: null }, schedule }), false);
  assert.equal(blurAfterPointerClick(null), false);
  const d = mk();
  blurAfterPointerClick({ detail: 1, currentTarget: d.container }, { doc: { activeElement: null }, schedule });
  assert.doesNotThrow(() => queue.shift()());
});

test('RecordPage 소스 계약: 녹화 길이 버튼·카운트다운 체크박스는 마우스 클릭 뒤 포커스 해제', () => {
  const page = src('../src/pages/RecordPage.jsx');
  assert.equal((page.match(/onClick=\{blurAfterPointerClick\}/g) || []).length, 2);
  assert.match(page, /role="radiogroup"[\s\S]{0,80}onClick=\{blurAfterPointerClick\}/);
  assert.match(page, /<label[^>]*onClick=\{blurAfterPointerClick\}>\s*<input\s+type="checkbox"/);
  // 페이지 단축키는 포커스된 컨트롤의 Space를 가로채지 않는다 (키보드 접근성).
  assert.match(page, /if \(e\.target !== document\.body && tag !== 'VIDEO' && tag !== 'DIV'\) return;/);
});
