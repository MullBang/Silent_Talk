// capturePackage.test.js — 조원용 촬영 패키지 만들기의 문서 변환·압축 검사 (node --test, 빌드는 하지 않음).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { inflateRawSync } from 'node:zlib';

import { leakedWardTexts, makeZip, mdToHtml } from '../scripts/build-capture-package.mjs';

test('mdToHtml: 제목·중첩 목록·표·인용·코드·체크 상자를 바꾸고 문서 사이 링크는 글자만 남긴다', () => {
  const html = mdToHtml([
    '# 제목', '', '> 인용 **굵게**', '', '1. 하나', '   1. 안쪽', '2. 둘', '   - [ ] 확인', '',
    '| 가 | 나 |', '|---|---|', '| `코드` | [문서](other.md) |', '', '```bash', 'npm run dev', '```', '', '---', '문단 <태그>',
  ].join('\n'));
  assert.match(html, /<h1>제목<\/h1>/);
  assert.match(html, /<blockquote>인용 <strong>굵게<\/strong><\/blockquote>/);
  assert.match(html, /<ol start="1">\n<li>하나\n<ol start="1">\n<li>안쪽\n<\/li><\/ol>\n<\/li>\n<li>둘/);
  assert.match(html, /<li>☐ 확인/);
  assert.match(html, /<td><code>코드<\/code><\/td><td>문서<\/td>/);
  assert.match(html, /<pre><code>npm run dev<\/code><\/pre>/);
  assert.match(html, /<p>문단 &lt;태그&gt;<\/p>/);
  assert.equal((html.match(/<ol/g) || []).length, (html.match(/<\/ol>/g) || []).length);
});

test('조원용 가이드·동의서 표시가 문서에 있다', () => {
  const guide = readFileSync(new URL('../../docs/공식모델_전환_웹캠촬영_따라하기_2026-10-28.md', import.meta.url), 'utf8');
  const consent = readFileSync(new URL('../../docs/공식모델_전환_웹캠촬영_기록지_2026-10-27.md', import.meta.url), 'utf8');
  assert.ok(guide.indexOf('<!-- 조원용 시작 -->') < guide.indexOf('<!-- 조원용 끝 -->'));
  assert.ok(consent.indexOf('<!-- 동의서 시작 -->') < consent.indexOf('<!-- 동의서 끝 -->'));
  const form = consent.slice(consent.indexOf('<!-- 동의서 시작 -->'), consent.indexOf('<!-- 동의서 끝 -->'));
  assert.ok(!form.includes('____'), '조원에게 가는 동의서에 빈칸 연락처가 없다');
  assert.ok(!form.includes('촬영 담당'), '혼자 찍으므로 촬영 담당 서명이 없다');
});

test('병실 명령 촬영 안내: 조원용 표시가 있고, 함정·짧은 말 문장은 없고, 명령 20개와 클립 수가 목록과 같다', () => {
  const doc = readFileSync(new URL('../../docs/병실명령_촬영안내_2026-11-09.md', import.meta.url), 'utf8');
  const ward = JSON.parse(readFileSync(new URL('../src/capture/ward_prompts.json', import.meta.url), 'utf8'));
  const commands = JSON.parse(readFileSync(new URL('../src/ward/commands.json', import.meta.url), 'utf8'));
  const start = doc.indexOf('<!-- 조원용 시작 -->');
  const end = doc.indexOf('<!-- 조원용 끝 -->');
  assert.ok(start >= 0 && start < end);
  const guide = doc.slice(start, end);
  const commandTexts = commands.fixed.map((p) => p.text);
  assert.deepEqual(leakedWardTexts(doc, ward, commandTexts), [], '평가 참가자가 읽는 문서에 함정·짧은 말 문장이 없다');
  for (const p of commands.fixed) assert.ok(guide.includes(`| ${p.id} | ${p.text} |`), `대사표에 ${p.id}`);
  assert.match(guide, /93클립/);
  assert.match(guide, /103클립/);
  assert.match(guide, /186개/);
  assert.match(guide, /206개/);
  assert.match(guide, /5초/);
  const html = mdToHtml(guide.split('\n').slice(1).join('\n'));
  assert.equal((html.match(/<table>/g) || []).length, (html.match(/<\/table>/g) || []).length);
  assert.match(html, /<h1>병실 명령 촬영 안내/);
});

test('leakedWardTexts: 띄어쓰기가 달라도 함정·짧은 말 문장을 찾는다', () => {
  const ward = { decoys: { tune: [{ id: 'D01', text: '물은 이제 괜찮아요' }], eval: [] }, short: [{ id: 'E02', text: '아파요' }] };
  assert.deepEqual(leakedWardTexts('안내: 물은이제 괜찮아요', ward), ['D01']);
  assert.deepEqual(leakedWardTexts('너무 아파요', ward), ['E02']);
  assert.deepEqual(leakedWardTexts('명령만 있는 안내', ward), []);
  // 짧은 말이 명령 문장의 일부일 때 명령(대사표)만으로는 걸리지 않고, 따로 적히면 걸린다.
  const withShort = { decoys: { tune: [], eval: [] }, short: [{ id: 'E03', text: '토할 것 같아' }] };
  const w10 = '구역질이 나서 토할 것 같아요';
  assert.deepEqual(leakedWardTexts(`| W10 | ${w10} |`, withShort, [w10]), []);
  assert.deepEqual(leakedWardTexts(`| W10 | ${w10} | 짧은 말: 토할 것 같아`, withShort, [w10]), ['E03']);
});

test('makeZip: 한글 이름에 UTF-8 표시를 켜고 내용이 그대로 풀린다', () => {
  const data = Buffer.from('안녕하세요 '.repeat(50), 'utf8');
  const zip = makeZip([{ name: '패키지/가이드.html', data }]);
  assert.equal(zip.readUInt32LE(0), 0x04034b50);
  assert.equal(zip.readUInt16LE(6) & 0x0800, 0x0800);
  const nameLen = zip.readUInt16LE(26);
  const compSize = zip.readUInt32LE(18);
  assert.equal(zip.subarray(30, 30 + nameLen).toString('utf8'), '패키지/가이드.html');
  assert.deepEqual(inflateRawSync(zip.subarray(30 + nameLen, 30 + nameLen + compSize)), data);
  const end = zip.subarray(zip.length - 22);
  assert.equal(end.readUInt32LE(0), 0x06054b50);
  assert.equal(end.readUInt16LE(10), 1);
});
