// capture.test.js — 파일럿 촬영 순서·파일명·녹화 통계 순수 함수 검사 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  SESSIONS, buildPlan, clipBaseName, clipQualityWarnings, nextUnsaved, nonspeechItems, normalizeSpoken, resumePoint, safeName,
  sessionFromMetas, spokenProblem,
} from '../src/capture/plan.js';
import { extensionFor, pickMimeType, publicTrackSettings, summarizeFrameLog } from '../src/hooks/useRecorder.js';

const prompts = JSON.parse(readFileSync(new URL('../src/capture/prompts.json', import.meta.url), 'utf8'));

test('prompts: 지정 20(AI Hub 10 + 새 문장 10), 자유 10, 무발화 10', () => {
  assert.equal(prompts.fixed.length, 20);
  assert.equal(prompts.fixed.filter((p) => p.id.startsWith('A')).length, 10);
  assert.equal(prompts.fixed.filter((p) => p.id.startsWith('N')).length, 10);
  assert.equal(prompts.free.length, 10);
  assert.equal(prompts.nonspeech.length, 10);
  assert.equal(new Set(prompts.nonspeech.map((p) => p.id)).size, 10);
  assert.equal(new Set(prompts.fixed.map((p) => p.id)).size, 20);
  assert.equal(new Set(prompts.fixed.map((p) => p.text)).size, 20);
});

test('prompts: 지정 문장은 한글·공백만, 새 문장(N)의 평균 음절 수는 AI Hub 문장(A)과 1음절 안쪽', () => {
  const syllables = (t) => [...t].filter((ch) => ch >= '가' && ch <= '힣').length;
  for (const p of prompts.fixed) assert.match(p.text, /^[가-힣]+( [가-힣]+)*$/, p.id);
  const mean = (prefix) => {
    const xs = prompts.fixed.filter((p) => p.id.startsWith(prefix)).map((p) => syllables(p.text));
    return xs.reduce((a, b) => a + b, 0) / xs.length;
  };
  assert.ok(Math.abs(mean('A') - mean('N')) <= 1, `A ${mean('A')} / N ${mean('N')}`);
  const firstWords = prompts.fixed.map((p) => p.text.split(' ')[0]);
  const nFirst = firstWords.slice(10);
  assert.equal(new Set(nFirst).size, 10, '새 문장끼리 첫 단어가 겹치지 않는다');
  for (const w of nFirst) assert.ok(!firstWords.slice(0, 10).includes(w), `A와 첫 단어가 같다: ${w}`);
});

test('buildPlan: 20문장 × 2회 × 2조건 + 자유 10 + 무발화 10 = 100클립', () => {
  const plan = buildPlan(prompts, 2);
  assert.equal(plan.length, 100);
  assert.equal(plan.filter((x) => x.cond === 'voiced' && x.kind === 'fixed').length, 40);
  assert.equal(plan.filter((x) => x.cond === 'silent').length, 40);
  assert.equal(plan.filter((x) => x.kind === 'free').length, 10);
  assert.equal(plan.filter((x) => x.kind === 'nonspeech').length, 10);
  assert.equal(plan[0].rep, 1);
  assert.equal(plan[20].rep, 2);
  assert.equal(new Set(plan.map((x) => clipBaseName('P01', x))).size, 100, '파일명은 겹치지 않아야 한다');
});

test('buildPlan: 최종 평가용 묶음은 지정 문장 3회 = 140클립, 파일명이 겹치지 않는다', () => {
  assert.deepEqual(SESSIONS.map((x) => [x.id, x.repeats]), [['pilot', 2], ['eval', 3]]);
  const plan = buildPlan(prompts, 3);
  assert.equal(plan.length, 140);
  assert.equal(plan.filter((x) => x.kind === 'fixed' && x.rep === 3).length, 40);
  assert.equal(new Set(plan.map((x) => clipBaseName('P05', x))).size, 140);
  assert.deepEqual(plan.filter((x) => x.kind === 'nonspeech').map((x) => x.sentenceId), prompts.nonspeech.map((x) => x.id));
});

test('nonspeechItems: 구형 v1(숫자)도 같은 ID 규칙으로 읽는다', () => {
  const items = nonspeechItems(3);
  assert.deepEqual(items.map((x) => x.id), ['X01', 'X02', 'X03']);
  assert.ok(items.every((x) => x.text.includes('말하지')));
  assert.deepEqual(nonspeechItems(undefined), []);
});

test('sessionFromMetas: 폴더의 기존 클립 기록으로 촬영 묶음을 정하고, 섞여 있으면 mixed', () => {
  assert.deepEqual(sessionFromMetas([]), { session: null, mixed: false, clips: 0 });
  assert.deepEqual(sessionFromMetas([{ kind: 'fixed', session: 'eval' }, { kind: 'free', session: 'eval' }, { other: 1 }, null]),
    { session: 'eval', mixed: false, clips: 2 });
  assert.deepEqual(sessionFromMetas([{ kind: 'fixed' }]), { session: 'pilot', mixed: false, clips: 1 }, 'v1 클립은 파일럿');
  assert.equal(sessionFromMetas([{ kind: 'fixed', session: 'pilot' }, { kind: 'fixed', session: 'eval' }]).mixed, true);
});

test('resumePoint: 저장된 클립 다음의 첫 빈 순번에서 이어 찍는다', () => {
  const plan = buildPlan(prompts, 2);
  const names = plan.slice(0, 3).map((x) => clipBaseName('P01', x));
  assert.deepEqual(resumePoint(plan, 'P01', names), { index: 3, saved: 3 });
  assert.deepEqual(resumePoint(plan, 'P02', names), { index: 0, saved: 0 }, '다른 가명 ID의 클립은 세지 않는다');
  const gap = [...names, clipBaseName('P01', plan[5])];
  assert.deepEqual(resumePoint(plan, 'P01', gap), { index: 3, saved: 4 }, '중간에 빠진 클립이 있으면 거기부터');
  const all = plan.map((x) => clipBaseName('P01', x));
  assert.deepEqual(resumePoint(plan, 'P01', all), { index: 100, saved: 100 });
});

test('nextUnsaved: 다시 찍은 뒤에는 이미 저장된 다음 클립을 건너뛰고 빈 순번으로 간다', () => {
  const plan = buildPlan(prompts, 2);
  const names = plan.slice(0, 10).map((x) => clipBaseName('P01', x));
  assert.equal(nextUnsaved(plan, 'P01', names, 4), 10, '5번을 다시 찍은 뒤 6~10번을 덮어쓰지 않는다');
  assert.equal(nextUnsaved(plan, 'P01', names, 10), 10);
  assert.equal(nextUnsaved(plan, 'P02', names, 4), 4, '다른 가명 ID는 바로 다음 칸');
  assert.equal(nextUnsaved(plan, 'P01', plan.map((x) => clipBaseName('P01', x)), 0), 100);
});

test('clipQualityWarnings: 너무 짧은 녹화·낮은 실제 fps·긴 멈춤을 알린다', () => {
  assert.deepEqual(clipQualityWarnings(null), []);
  assert.deepEqual(clipQualityWarnings({ durationMs: 3200, estimatedFps: 29.8, maxGapMs: 45 }), []);
  assert.deepEqual(clipQualityWarnings({ durationMs: 3000, estimatedFps: 22, maxGapMs: 120 }), [], '20fps 이상·300ms 이하는 경고 없음');
  const w = clipQualityWarnings({ durationMs: 400, estimatedFps: 14.9, maxGapMs: 350 });
  assert.equal(w.length, 3);
  assert.match(w[0], /1초보다 짧/);
  assert.match(w[1], /14\.9/);
  assert.match(w[2], /350ms/);
});

test('spokenProblem/normalizeSpoken: 자유 문장 정답은 한글·띄어쓰기만, 공백은 정리해 저장한다', () => {
  assert.equal(normalizeSpoken('  오늘은   집에 일찍 갈게 '), '오늘은 집에 일찍 갈게');
  assert.equal(spokenProblem('오늘은 집에 일찍 갈게'), null);
  assert.match(spokenProblem(''), /먼저 입력/);
  assert.match(spokenProblem('3시에 만나자'), /숫자/);
  assert.match(spokenProblem('카페 가서 coffee 마실래'), /영어/);
  assert.match(spokenProblem('정말 좋아요!'), /한글과 띄어쓰기만/);
  assert.match(spokenProblem('ㅋㅋ 재밌다'), /한글과 띄어쓰기만/);
});

test('publicTrackSettings: 카메라 기기 고유값과 이름은 저장하지 않는다', () => {
  const out = publicTrackSettings({ width: 1280, height: 720, frameRate: 30, deviceId: 'abc', groupId: 'g1', label: '홍길동의 iPhone 카메라' });
  assert.deepEqual(out, { width: 1280, height: 720, frameRate: 30 });
  assert.deepEqual(publicTrackSettings(undefined), {});
});

test('safeName/clipBaseName: 파일명에 안전한 문자만 남긴다', () => {
  assert.equal(safeName(' P 01/..\\x '), 'P01x');
  assert.equal(safeName(''), 'anon');
  assert.equal(clipBaseName('P01', { cond: 'silent', sentenceId: 'A03', rep: 2 }), 'P01_silent_A03_2');
});

test('summarizeFrameLog: 평균·최대 간격과 추정 fps', () => {
  assert.deepEqual(summarizeFrameLog([]), { frameCallbacks: 0, meanGapMs: null, maxGapMs: null, estimatedFps: null });
  const s = summarizeFrameLog([0, 33, 66, 133]);
  assert.equal(s.frameCallbacks, 4);
  assert.equal(s.maxGapMs, 67);
  assert.ok(Math.abs(s.meanGapMs - 133 / 3) < 1e-9);
  assert.ok(Math.abs(s.estimatedFps - 3000 / 133) < 1e-9);
});

test('pickMimeType/extensionFor: MediaRecorder가 없으면 null, mp4 확장자 판정', () => {
  assert.equal(pickMimeType(), null);
  assert.equal(extensionFor('video/mp4'), 'mp4');
  assert.equal(extensionFor('video/webm;codecs=vp9'), 'webm');
});
