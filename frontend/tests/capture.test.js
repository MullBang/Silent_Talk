// capture.test.js — 파일럿 촬영 순서·파일명·녹화 통계 순수 함수 검사 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { buildPlan, clipBaseName, safeName } from '../src/capture/plan.js';
import { extensionFor, pickMimeType, summarizeFrameLog } from '../src/hooks/useRecorder.js';

const prompts = JSON.parse(readFileSync(new URL('../src/capture/prompts.json', import.meta.url), 'utf8'));

test('prompts: 지정 20(AI Hub 10 + 새 문장 10), 자유 10, 무발화 10', () => {
  assert.equal(prompts.fixed.length, 20);
  assert.equal(prompts.fixed.filter((p) => p.id.startsWith('A')).length, 10);
  assert.equal(prompts.fixed.filter((p) => p.id.startsWith('N')).length, 10);
  assert.equal(prompts.free.length, 10);
  assert.equal(prompts.nonspeech, 10);
  assert.equal(new Set(prompts.fixed.map((p) => p.id)).size, 20);
  assert.equal(new Set(prompts.fixed.map((p) => p.text)).size, 20);
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
