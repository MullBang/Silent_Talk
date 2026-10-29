// zipStore.test.js — 브라우저용 압축 없는 zip 묶기 검사 (node --test).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { crc32 as zlibCrc32 } from 'node:zlib';

import { crc32, zipStore } from '../src/capture/zipStore.js';

test('crc32: zlib 값과 같다', () => {
  for (const text of ['', 'a', '안녕하세요 웹캠 촬영', 'x'.repeat(10000)]) {
    const bytes = new TextEncoder().encode(text);
    assert.equal(crc32(bytes), zlibCrc32(bytes) >>> 0, text.slice(0, 10));
  }
});

test('zipStore: 한글 이름·UTF-8 표시·저장 방식, 내용과 목록이 맞다', () => {
  const a = new TextEncoder().encode('영상 바이트');
  const b = new Uint8Array([1, 2, 3, 255]);
  const zip = zipStore([{ name: 'P02_웹캠촬영/P02_voiced_A01_1.webm', bytes: b }, { name: 'P02_웹캠촬영/P02_voiced_A01_1.json', bytes: a }]);
  const view = new DataView(zip.buffer);
  assert.equal(view.getUint32(0, true), 0x04034b50);
  assert.equal(view.getUint16(6, true) & 0x0800, 0x0800);
  assert.equal(view.getUint16(8, true), 0, '저장(압축 없음)');
  const nameLen = view.getUint16(26, true);
  assert.equal(new TextDecoder().decode(zip.subarray(30, 30 + nameLen)), 'P02_웹캠촬영/P02_voiced_A01_1.webm');
  assert.deepEqual(zip.subarray(30 + nameLen, 30 + nameLen + b.length), b);
  const end = new DataView(zip.buffer, zip.length - 22);
  assert.equal(end.getUint32(0, true), 0x06054b50);
  assert.equal(end.getUint16(10, true), 2);
  const cdOffset = end.getUint32(16, true);
  assert.equal(view.getUint32(cdOffset, true), 0x02014b50);
});
