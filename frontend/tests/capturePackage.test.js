// capturePackage.test.js — 조원용 촬영 패키지 만들기의 문서 변환·압축 검사 (node --test, 빌드는 하지 않음).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { inflateRawSync } from 'node:zlib';

import { fillContact, makeZip, mdToHtml } from '../scripts/build-capture-package.mjs';

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

test('fillContact: 동의서 연락처 빈칸만 채운다', () => {
  const md = '| 보관 기한 | 지운다 |\n| 문의·철회 연락처 | ____________________ |\n';
  assert.match(fillContact(md, '팀장 010-0000-0000'), /\| 문의·철회 연락처 \| 팀장 010-0000-0000 \|/);
  assert.equal(fillContact(md, null), md);
});

test('조원용 가이드·동의서 표시가 문서에 있다', () => {
  const guide = readFileSync(new URL('../../docs/공식모델_전환_웹캠촬영_따라하기_2026-10-28.md', import.meta.url), 'utf8');
  const consent = readFileSync(new URL('../../docs/공식모델_전환_웹캠촬영_기록지_2026-10-27.md', import.meta.url), 'utf8');
  assert.ok(guide.indexOf('<!-- 조원용 시작 -->') < guide.indexOf('<!-- 조원용 끝 -->'));
  assert.ok(consent.indexOf('<!-- 동의서 시작 -->') < consent.indexOf('<!-- 동의서 끝 -->'));
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
