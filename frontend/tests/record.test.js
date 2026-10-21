// record.test.js — 6단계 녹화→분석 화면의 순수 함수 검사 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  COUNTDOWN_SECONDS, DEFAULT_MODE_ID, MIN_RECORD_SECONDS, RECORD_MAX_SIZE_MB, RECORD_MODES,
  RECORD_UNAVAILABLE_ERROR, RECORD_UPLOAD_ERROR, SERVER_MAX_SECONDS, baseMimeType, buildRecordPayload,
  clockLabel, formatElapsed, modeById, phaseLabel, recordUploadErrorMessage, recordingFileName,
  recordingProgress, recordingTitle, validateBlob, validateRecording,
} from '../src/record/recordFlow.js';
import {
  NETWORK_ERROR, POLL_TIMEOUT_MS, REQUEST_TIMEOUT_ERROR, errorMessageFrom, interpretPoll, isTerminal,
} from '../src/api/pollOutcome.js';
import {
  NON_RECOGNITION_MARKERS, NO_SPEECH_TEXT, UNSEEN_PERSON_NOTICE, isNonRecognition, lowConfidenceCandidate,
  modelInfoLine,
} from '../src/results/markers.js';
import { LOCAL_FACE_LANDMARKER_PATH, resolveFaceLandmarkerModel, withBase } from '../src/mediapipeAssets.js';

const MB = 1024 * 1024;
const blob = (size) => ({ size });
const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');

test('녹화 모드: 기본 8초, 30초 실험 모드는 서버 상한(30초 미만)보다 짧은 29초에 자동 종료', () => {
  assert.equal(DEFAULT_MODE_ID, 'short');
  assert.equal(modeById('short').maxSeconds, 8);
  assert.equal(modeById('short').experimental, false);
  const long = modeById('long');
  assert.equal(long.maxSeconds, 29);
  assert.equal(long.experimental, true);
  assert.ok(long.maxSeconds < SERVER_MAX_SECONDS);
  assert.equal(modeById('nope').id, 'short', '모르는 id는 기본 모드');
  assert.equal(RECORD_MODES.length, 2);
  assert.ok(RECORD_MODES.every((m) => m.maxSeconds < SERVER_MAX_SECONDS));
  assert.equal(COUNTDOWN_SECONDS, 3);
});

test('validateRecording: 빈 녹화·1초 미만은 업로드하지 않는다', () => {
  assert.equal(validateRecording(blob(0), 5000).reason, 'empty');
  assert.equal(validateRecording(null, 5000).reason, 'empty');
  assert.equal(validateRecording(blob(10), 999).reason, 'too_short');
  assert.equal(validateRecording(blob(10), Number.NaN).reason, 'too_short');
  assert.equal(validateRecording(blob(10), MIN_RECORD_SECONDS * 1000).ok, true);
  assert.equal(validateRecording(blob(10), 8000).ok, true);
  assert.equal(validateRecording(blob(10), 29999).ok, true);
  assert.equal(validateRecording(blob(10), 30000).reason, 'too_long');
  assert.equal(validateRecording(blob(RECORD_MAX_SIZE_MB * MB + 1), 5000).reason, 'too_large');
  assert.equal(validateRecording(blob(RECORD_MAX_SIZE_MB * MB), 5000).ok, true);
  for (const r of [validateRecording(blob(0), 1), validateRecording(blob(1), 1)]) {
    assert.equal(typeof r.message, 'string');
    assert.ok(r.message.length > 0);
  }
});

test('validateBlob: 크기만 검사한다', () => {
  assert.equal(validateBlob(blob(0)).reason, 'empty');
  assert.equal(validateBlob(blob(Number.NaN)).reason, 'empty');
  assert.equal(validateBlob(blob(1)).ok, true);
  assert.equal(validateBlob(blob(101 * MB)).reason, 'too_large');
});

test('파일명·MIME: webm/mp4만, 코덱 매개변수 제거', () => {
  assert.equal(recordingFileName('video/webm;codecs=vp9'), 'recording.webm');
  assert.equal(recordingFileName('video/webm'), 'recording.webm');
  assert.equal(recordingFileName('video/mp4;codecs=avc1.42E01E'), 'recording.mp4');
  assert.equal(recordingFileName('VIDEO/MP4'), 'recording.mp4');
  assert.equal(recordingFileName(null), 'recording.webm');
  assert.equal(baseMimeType('video/webm;codecs=vp8'), 'video/webm');
  assert.equal(baseMimeType('video/mp4'), 'video/mp4');
  assert.equal(baseMimeType(''), 'video/webm');
});

test('시각·시간 표시', () => {
  assert.equal(clockLabel(new Date(2026, 9, 20, 9, 5, 7)), '09:05:07');
  assert.equal(recordingTitle(new Date(2026, 9, 20, 23, 59, 0)), '녹화 23:59:00');
  assert.match(clockLabel(new Date('invalid')), /^\d\d:\d\d:\d\d$/);
  assert.equal(formatElapsed(0), '0:00.0');
  assert.equal(formatElapsed(3249), '0:03.2');
  assert.equal(formatElapsed(29000), '0:29.0');
  assert.equal(formatElapsed(61500), '1:01.5');
  assert.equal(formatElapsed(-5), '0:00.0');
  assert.equal(formatElapsed(Number.NaN), '0:00.0');
});

test('recordingProgress: 경과/남은 시간과 진행률을 0~100으로 자른다', () => {
  assert.deepEqual(recordingProgress(0, 8), { elapsedMs: 0, remainingMs: 8000, percent: 0 });
  assert.deepEqual(recordingProgress(2000, 8), { elapsedMs: 2000, remainingMs: 6000, percent: 25 });
  assert.deepEqual(recordingProgress(9000, 8), { elapsedMs: 8000, remainingMs: 0, percent: 100 });
  assert.deepEqual(recordingProgress(-10, 29), { elapsedMs: 0, remainingMs: 29000, percent: 0 });
  assert.equal(recordingProgress(1000, 0).percent, 0);
});

test('buildRecordPayload: 결과 화면 계약 (origin record, 제목 녹화 HH:MM:SS)', () => {
  const at = new Date(2026, 9, 20, 14, 3, 9);
  const model = { backend: 'olkavs', decoder: 'hybrid' };
  const results = [{ start_ms: 0, end_ms: 1400, text: '안녕하세요', confidence: 0.8, raw_score: -3 }];
  assert.deepEqual(buildRecordPayload({ sessionId: 's1', recordedAt: at, results, warnings: ['w'], model }), {
    sessionId: 's1', fileName: '녹화 14:03:09', results, warnings: ['w'], model, origin: 'record',
  });
  const empty = buildRecordPayload({ sessionId: 's2', recordedAt: at });
  assert.deepEqual([empty.results, empty.warnings, empty.model, empty.origin], [[], [], null, 'record']);
});

test('interpretPoll: 진행/완료/시간초과/오류 해석', () => {
  assert.equal(POLL_TIMEOUT_MS, 315000);
  assert.deepEqual(interpretPoll(null), { kind: 'none' });
  assert.deepEqual(interpretPoll({ status: 'pending', progress: 0 }), { kind: 'progress', progress: 0 });
  assert.deepEqual(interpretPoll({ status: 'processing' }), { kind: 'progress', progress: null });
  const model = { backend: 'olkavs' };
  assert.deepEqual(interpretPoll({ status: 'done', progress: 100, results: null, warnings: null, model }), {
    kind: 'done', progress: 100, results: [], warnings: [], model,
  });
  assert.equal(interpretPoll({ status: 'done', results: [] }).model, null);
  assert.deepEqual(interpretPoll({ status: 'timeout', progress: 40 }), { kind: 'timeout', progress: 40 });
  assert.equal(interpretPoll({ status: 'error', error_message: '영상을 읽을 수 없습니다.' }).message, '영상을 읽을 수 없습니다.');
  assert.equal(interpretPoll({ status: 'error', error_message: '' }).message, '추론 중 오류가 발생했습니다.');
  assert.ok(isTerminal({ status: 'done' }) && isTerminal({ status: 'error' }) && isTerminal({ status: 'timeout' }));
  assert.ok(!isTerminal({ status: 'pending' }) && !isTerminal({ status: 'processing' }) && !isTerminal(null));
});

test('오류 문구: 서버 detail 우선, 내부 내용은 노출하지 않음', () => {
  const d400 = { response: { status: 400, data: { detail: '30초 미만의 녹화만 분석할 수 있습니다.' } } };
  assert.equal(recordUploadErrorMessage(d400), '30초 미만의 녹화만 분석할 수 있습니다.');
  assert.equal(recordUploadErrorMessage({ response: { status: 503, data: {} } }), RECORD_UNAVAILABLE_ERROR);
  assert.equal(recordUploadErrorMessage({ response: { status: 503, data: { detail: '모델 준비 중' } } }), '모델 준비 중');
  assert.equal(recordUploadErrorMessage({ response: { status: 422, data: { detail: [{ loc: ['body', 'file'] }] } } }), RECORD_UPLOAD_ERROR);
  assert.equal(recordUploadErrorMessage({ response: { status: 500, data: 'Traceback C:\\x' } }), RECORD_UPLOAD_ERROR);
  assert.equal(recordUploadErrorMessage({ request: {}, message: 'Network Error', code: 'ERR_NETWORK' }), NETWORK_ERROR);
  assert.equal(recordUploadErrorMessage({ request: {}, code: 'ECONNABORTED' }), REQUEST_TIMEOUT_ERROR);
  const client = Object.assign(new Error('녹화된 영상이 비어 있습니다. 다시 녹화해 주세요.'), { isClientCheck: true });
  assert.equal(recordUploadErrorMessage(client), client.message);
  assert.equal(recordUploadErrorMessage(new Error('missing session_id')), RECORD_UPLOAD_ERROR);
  assert.equal(errorMessageFrom(undefined, 'fb'), 'fb');
});

test('결과 마커: [발화 없음]도 비인식, 낮은 신뢰도 후보는 [인식 불가]에만', () => {
  assert.deepEqual(NON_RECOGNITION_MARKERS, ['[인식 불가]', '[검출 실패 구간]', '[발화 없음]']);
  assert.equal(NO_SPEECH_TEXT, '[발화 없음]');
  for (const t of ['[인식 불가]', '[검출 실패 구간]', ' [발화 없음] ']) assert.ok(isNonRecognition(t), t);
  for (const t of ['안녕하세요', '', null, undefined, '[발화 없음] 안녕']) assert.ok(!isNonRecognition(t), String(t));
  assert.equal(lowConfidenceCandidate({ text: '[인식 불가]', candidate: ' 안녕 ', confidence: 0.1 }), '안녕');
  assert.equal(lowConfidenceCandidate({ text: '[인식 불가]', candidate: null }), null);
  assert.equal(lowConfidenceCandidate({ text: '[인식 불가]', candidate: '  ' }), null);
  assert.equal(lowConfidenceCandidate({ text: '[발화 없음]', candidate: '안녕' }), null);
  assert.equal(lowConfidenceCandidate({ text: '안녕', candidate: '안녕' }), null);
  assert.equal(lowConfidenceCandidate(null), null);
  assert.match(UNSEEN_PERSON_NOTICE, /처음 보는 사람/);
  assert.match(UNSEEN_PERSON_NOTICE, /검증되지 않았습니다/);
});

test('modelInfoLine: 짧은 SHA 12자, commit 7자, 값 없으면 null', () => {
  const sha = 'a'.repeat(12) + 'b'.repeat(52);
  const line = modelInfoLine({
    backend: 'olkavs', decoder: 'hybrid', checkpoint_sha256: sha, upstream_commit: '1234567890abcdef', roi_profile: 'step04-bbox1.5',
  });
  assert.equal(line, 'OLKAVS 공식 모델 · 디코더 hybrid · 가중치 aaaaaaaaaaaa · 코드 1234567 · 입술 영역 step04-bbox1.5');
  assert.equal(modelInfoLine({ backend: 'olkavs', decoder: null, checkpoint_sha256: null }), 'OLKAVS 공식 모델');
  assert.equal(modelInfoLine({ backend: 'compact' }), 'compact');
  assert.equal(modelInfoLine({}), null);
  assert.equal(modelInfoLine(null), null);
  assert.equal(modelInfoLine('x'), null);
});

test('phaseLabel: 모든 화면 단계에 한국어 문구', () => {
  for (const p of ['IDLE', 'OPENING', 'READY', 'COUNTDOWN', 'RECORDING', 'UPLOADING', 'INFERRING', 'TIMEOUT', 'ERROR']) {
    assert.ok(phaseLabel(p).length > 0, p);
  }
  assert.equal(phaseLabel('???'), '');
});

test('MediaPipe 로컬 경로: 로컬 모델이 있으면 로컬, HTML 대체 응답·404·오류면 CDN', async () => {
  const CDN = 'https://example.invalid/face_landmarker.task';
  assert.equal(withBase(LOCAL_FACE_LANDMARKER_PATH), '/mediapipe/face_landmarker.task');
  assert.equal(withBase('mediapipe/wasm', '/app'), '/app/mediapipe/wasm');
  const fake = (ok, type) => async (url, opts) => {
    assert.equal(url, '/mediapipe/face_landmarker.task');
    assert.equal(opts.method, 'HEAD');
    return { ok, headers: { get: () => type } };
  };
  assert.deepEqual(await resolveFaceLandmarkerModel(CDN, { fetchFn: fake(true, 'application/octet-stream') }),
    { url: '/mediapipe/face_landmarker.task', source: 'local' });
  assert.deepEqual(await resolveFaceLandmarkerModel(CDN, { fetchFn: fake(true, 'text/html') }), { url: CDN, source: 'cdn' });
  assert.deepEqual(await resolveFaceLandmarkerModel(CDN, { fetchFn: fake(false, '') }), { url: CDN, source: 'cdn' });
  const boom = async () => { throw new Error('offline'); };
  assert.deepEqual(await resolveFaceLandmarkerModel(CDN, { fetchFn: boom }), { url: CDN, source: 'cdn' });
});

test('오프라인: wasm은 로컬 경로, RecordPage와 녹화 모듈은 외부 주소를 쓰지 않는다', () => {
  assert.match(src('../src/constants.js'), /MEDIAPIPE_WASM = `\$\{import\.meta\.env\?\.BASE_URL \|\| '\/'\}mediapipe\/wasm`/);
  assert.doesNotMatch(src('../src/constants.js'), /cdn\.jsdelivr\.net/);
  for (const rel of ['../src/pages/RecordPage.jsx', '../src/record/recordFlow.js', '../src/components/ResultPoller.jsx']) {
    assert.doesNotMatch(src(rel), /https?:\/\//, rel);
  }
  const cfg = src('../vite.config.js');
  assert.match(cfg, /mediapipeLocal\(\)/);
  assert.match(cfg, /tasks-vision', 'wasm'/);
});

test('라우팅 소스 계약: #record, 다시 녹화, 업로드 화면의 녹화 버튼', () => {
  const app = src('../src/App.jsx');
  assert.match(app, /'#record': 'record'/);
  assert.match(app, /origin === 'record'/);
  assert.match(src('../src/pages/ResultPage.jsx'), /다시 녹화/);
  assert.match(src('../src/pages/UploadPage.jsx'), /녹화해서 분석 →/);
  assert.match(src('../src/api/client.js'), /'\/api\/record'/);
});
