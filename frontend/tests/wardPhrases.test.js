// wardPhrases.test.js — 환자 화면 '말할 수 있는 문장' 칸의 묶음 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { buildCommandIndex, buildCustomCommandIndex, phraseGroups } from '../src/ward/wardCommands.js';
import { normalizeSettings } from '../src/ward/wardSettings.js';

const json = JSON.parse(readFileSync(new URL('../src/ward/commands.json', import.meta.url)));
const index = buildCommandIndex(json);
const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');

test('기본 목록: 20개 입 모양 문장(text)을 분류 순서대로, 긴급·취소 표시와 함께', () => {
  const groups = phraseGroups(index);
  assert.deepEqual(groups.map((g) => g.id), json.categories.map((c) => c.id));
  const all = groups.flatMap((g) => g.commands);
  assert.equal(all.length, 20);
  for (const c of all) {
    const raw = json.fixed.find((f) => f.id === c.id);
    assert.equal(c.text, raw.text, '말할 문장은 인식용 text');
    assert.equal(c.display, raw.display);
    assert.equal(c.emergency, raw.emergency);
  }
  assert.deepEqual(all.filter((c) => c.emergency).map((c) => c.id).sort(), ['W02', 'W03', 'W04', 'W10']);
  assert.deepEqual(all.filter((c) => c.cancel).map((c) => c.id), ['W20']);
});

test('내 목록: 그 목록의 문장만, 명령이 없는 분류는 빼고', () => {
  const custom = buildCustomCommandIndex({
    id: 'rehab', title: '재활 병동', sha256: 'a'.repeat(64),
    commands: [
      { id: 'U01', text: '다리가 저려서 주물러 주세요', display: '다리가 저려요', category: 'care', emergency: false, cancel: false },
      { id: 'U02', text: '가슴이 쥐어짜듯 아파요', display: '가슴이 아파요', category: 'emergency', emergency: true, cancel: false },
      { id: 'U03', text: '산책을 나가고 싶어요', display: '산책하고 싶어요', category: 'room', emergency: false, cancel: false },
      { id: 'U04', text: '아까 고른 건 없던 일로 할래', display: '방금 요청은 취소할게요', category: 'call', emergency: false, cancel: true },
    ],
  }, json);
  const groups = phraseGroups(custom);
  assert.equal(groups.flatMap((g) => g.commands).length, 4);
  assert.equal(groups.some((g) => g.id === 'answer'), false, '빈 분류는 없음');
  assert.equal(groups.flatMap((g) => g.commands).find((c) => c.id === 'U02').emergency, true);
  assert.equal(groups.flatMap((g) => g.commands).find((c) => c.id === 'U04').cancel, true);
  assert.deepEqual(phraseGroups(null), []);
  assert.deepEqual(phraseGroups({ categories: [{ id: 'x', label: 'x', commands: [] }] }), []);
});

test('설정: 문장 목록은 기본으로 펼침, false일 때만 접음', () => {
  assert.equal(normalizeSettings({}).showPhrases, true);
  assert.equal(normalizeSettings({ showPhrases: false }).showPhrases, false);
  assert.equal(normalizeSettings({ showPhrases: 'no' }).showPhrases, true);
});

test('화면 연결: 환자 화면이 지금 쓰는 목록으로 문장 칸을 그리고, 내 목록을 쓸 수 없으면 숨긴다', () => {
  const ward = src('../src/pages/WardPage.jsx');
  assert.match(ward, /!machine\.lipBlock && \(\s*<PhrasePanel\s+index=\{machine\.index\}/);
  const panel = src('../src/ward/PhrasePanel.jsx');
  assert.match(panel, /phraseGroups\(index\)/);
  assert.match(panel, /data-no-switch/, '접기 버튼은 스위치 누름이 아니다');
  assert.doesNotMatch(panel, /dispatch|SELECT_ITEM|onSelect/, '문장 칸은 보기만 한다(눌러서 고르지 않음)');
});
