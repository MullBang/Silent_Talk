// wardLists.test.js — '내 목록'(사용자가 만든 병실 명령 목록) 검사 (node --test, DOM 없음):
// 목록 편집 상태도·즉시 검사·서버 검사 결과 읽기, 내 목록 색인, 취소 명령 일반화, 목록 ID·지문이 같을 때만 결과 쓰기,
// 기본 목록으로 몰래 바꾸지 않기, 간호사 카드 합치기 열쇠, 설정, API 호출 모양.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  LIST_CHANGED_MESSAGE, commandListProblem, isListChangedError, listSaveHeaders,
} from '../src/api/client.js';
import {
  CANCEL_COMMAND_ID, buildCommandIndex, buildCustomCommandIndex, commandById, emptyListIndex, isCancelCommand,
} from '../src/ward/wardCommands.js';
import { ENROLL_TEXT, STORE_LIST_CHANGED, createEnrollState, stepEnroll, storeOutcome } from '../src/ward/wardEnroll.js';
import {
  LISTS_TEXT, MAX_COMMANDS, canSaveList, canUseList, checksFromError, createListEditor, defaultListRows, editorMessages,
  listBannerText, listIdProblem, listLoadErrorMessage, listLoadOutcome, normalizeListDisplay, normalizeListText,
  objectParticle, olkavsTokenCount, parseChecks, parseExamples, parseIssue, parseListSummaries, quickDisplayProblem,
  quickTextProblem, quickTitleProblem, reenrollSummary, requestBody, resultsFromOtherListVersion, rowLength, saveCondition,
  stepListEditor, suggestListId, syllableCount,
} from '../src/ward/wardLists.js';
import {
  WARD_TEXT, createWardState, listSwitchBlockReason, listsOpenBlockReason, stepWard, wardView,
} from '../src/ward/wardMachine.js';
import { classifyWardResults, phraseListMatches } from '../src/ward/wardResult.js';
import {
  DEFAULT_SETTINGS, SETTINGS_STORAGE_KEY, enrollProfileFor, enrollProfilePatch, loadSettings, normalizeCommandList,
  normalizeEnrollProfileByList, normalizeSettings, readStoredSettings, saveSettings, sharedSettingsPatch,
} from '../src/ward/wardSettings.js';
import { cardListNote, cardListTag, mergeKey, upsertNurseCard } from '../src/ward/wardStore.js';
import { analyzeWardRecording, wardRecordOptions } from '../src/ward/wardUpload.js';

const src = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const COMMANDS = JSON.parse(src('../src/ward/commands.json'));
const INDEX = buildCommandIndex(COMMANDS);
const SHA = 'a'.repeat(64);
const SHA2 = 'b'.repeat(64);
const T0 = 1_000_000;

/** 이 검사에서만 쓰는 내 목록 (서버 GET 응답 모양). 문장은 이 프로젝트에서 새로 쓴 예시다. */
function serverList(over = {}) {
  return {
    schema: 'ward-custom-list-v1',
    id: 'rehab',
    title: '재활 병동',
    created: '2026-11-11T09:00:00',
    updated: '2026-11-11T09:00:00',
    sha256: SHA,
    commands: [
      { id: 'U01', text: '창문을 조금만 열어 주실래요', display: '창문을 열어 주세요', category: 'room', emergency: false, cancel: false },
      { id: 'U02', text: '가슴이 갑자기 너무 조여요', display: '가슴이 조여요', category: 'emergency', emergency: true, cancel: false },
      { id: 'U03', text: '다리 운동을 도와주시겠어요', display: '다리 운동을 도와주세요', category: 'care', emergency: false, cancel: false },
      { id: 'U04', text: '아까 부탁한 일은 안 해도 돼요', display: '방금 요청은 취소할게요', category: 'call', emergency: false, cancel: true },
      { id: 'U05', text: '휠체어를 가까이 가져다줘요', display: '휠체어를 가져다주세요', category: 'care', emergency: false, cancel: false },
    ],
    checks: { errors: [], warnings: [] },
    ...over,
  };
}
const CUSTOM = buildCustomCommandIndex(serverList(), COMMANDS);

/** 서버 결과 한 건 (병실 묶음, 내 목록이면 list를 붙인다). */
function phraseResult(id, runnerUp, accepted, { list = { id: 'rehab', sha256: SHA }, index = CUSTOM } = {}) {
  const phrase = {
    id, text: index.byId.get(id)?.text || '', score: accepted ? -1.2 : -3.9, margin: accepted ? 1.2 : 0.1,
    runner_up_id: runnerUp, accepted, set: 'ward',
  };
  if (list) phrase.list = list;
  return [{ start_ms: 0, end_ms: 5000, text: '받아쓰기', confidence: 0.8, raw_score: -1, phrase }];
}

/** 이벤트 여러 개. */
function feed(state, events) {
  let s = state;
  const effects = [];
  for (const e of events) {
    const r = stepWard(s, e);
    s = r.state;
    effects.push(...r.effects);
  }
  return { state: s, effects };
}

/** 누름 → 녹화 끝 → 결과. */
function recordWith(state, at, results) {
  const r1 = feed(state, [{ type: 'PRESS', at }]);
  const run = r1.state.run;
  const r2 = feed(r1.state, [{ type: 'RECORDING_DONE', at: at + 5000, run }, { type: 'RESULT', at: at + 6000, run, results }]);
  return { state: r2.state, effects: [...r1.effects, ...r2.effects] };
}
const sent = (effects) => effects.filter((e) => e.type === 'SEND_CARD').map((e) => e.card);
const updated = (effects) => effects.filter((e) => e.type === 'UPDATE_CARD').map((e) => e.card);

// ---------------------------------------------------------------------------
// 문장 길이·즉시 검사
// ---------------------------------------------------------------------------

test('문장 다듬기와 길이: NFC·공백, 음절 수, OLKAVS 토큰 수(받침 없는 음절 2·받침 3·띄어쓰기 1)', () => {
  assert.equal(normalizeListText('  물   주세요 '), '물 주세요');
  assert.equal(normalizeListText('가'), '가', 'NFC로 합친다');
  assert.equal(syllableCount('물 주세요'), 4);
  assert.equal(olkavsTokenCount('물 주세요'), 3 + 1 + 2 + 2 + 2);
  assert.equal(olkavsTokenCount('가'), 2);
  assert.equal(olkavsTokenCount('각'), 3);
  assert.equal(olkavsTokenCount('물a'), null, '한글 음절·띄어쓰기 밖 글자');
  // 기본 목록 20개는 모두 토큰 23~34로 만들었다(commands.json note). 같은 셈법이어야 한다.
  for (const c of COMMANDS.fixed) {
    const n = olkavsTokenCount(c.text);
    assert.ok(n >= 23 && n <= 34, `${c.id} ${n}`);
  }
  const good = rowLength('창문을 조금만 열어 주실래요');
  assert.equal(good.syllables, 12);
  assert.equal(good.tokens, 34);
  assert.equal(good.good, true);
  assert.equal(rowLength('물').good, false);
});

test('즉시 검사: 한글·띄어쓰기만, 2~40자, 화면 문장 1~30자·제어 문자 없음, 목록 이름 1~30자', () => {
  assert.equal(quickTextProblem('창문을 열어 주세요'), null);
  assert.equal(quickTextProblem(''), LISTS_TEXT.textRequired);
  for (const bad of ['물 2잔 주세요', 'TV 켜 줘요', '물 주세요!', 'ㅋㅋ 좋아요']) assert.equal(quickTextProblem(bad), LISTS_TEXT.textHangulOnly, bad);
  assert.equal(quickTextProblem('가'), LISTS_TEXT.textLength);
  assert.equal(quickTextProblem('가'.repeat(41)), LISTS_TEXT.textLength);
  assert.equal(quickTextProblem('각'.repeat(16)), null, '토큰 48');
  assert.equal(quickTextProblem('각'.repeat(17)), LISTS_TEXT.textTokens, '17자(40자 안)여도 토큰 51 > 50');
  assert.equal(quickDisplayProblem('물을 주세요'), null);
  assert.equal(quickDisplayProblem('  '), LISTS_TEXT.displayRequired);
  assert.equal(quickDisplayProblem('가'.repeat(31)), LISTS_TEXT.displayLength);
  assert.equal(quickDisplayProblem('물\u0007'), LISTS_TEXT.displayControl);
  assert.equal(quickTitleProblem('재활 병동'), null);
  assert.equal(quickTitleProblem(''), LISTS_TEXT.titleRequired);
  assert.equal(quickTitleProblem('가'.repeat(31)), LISTS_TEXT.titleLength);
});

test('목록 ID: 규칙·Windows 예약 이름·대소문자만 다른 겹침, 제안 ID', () => {
  assert.equal(listIdProblem('rehab', []), null);
  assert.equal(listIdProblem('', []), LISTS_TEXT.idRequired);
  for (const bad of ['재활', 'a b', '../x', 'x'.repeat(33)]) assert.equal(listIdProblem(bad, []), LISTS_TEXT.idPattern, bad);
  for (const dev of ['CON', 'nul', 'Com1', 'LPT9']) assert.equal(listIdProblem(dev, []), LISTS_TEXT.idReserved, dev);
  assert.equal(listIdProblem('Rehab', ['rehab']), LISTS_TEXT.idTaken);
  assert.equal(listIdProblem('rehab', ['rehab'], 'rehab'), null, '자기 자신은 겹침이 아니다');
  assert.equal(suggestListId([]), 'mylist1');
  assert.equal(suggestListId(['MyList1', 'mylist2']), 'mylist3');
});

// ---------------------------------------------------------------------------
// 서버 검사 결과
// ---------------------------------------------------------------------------

test('서버 검사 항목 읽기: {code, level, message, row, rows, field}, 강한 주의, 문자열, 400 본문', () => {
  const pair = parseIssue({ code: 'lip_same', level: 'strong', message: '입 모양이 거의 같아서 구분하기 어려워요', row: 2, rows: [0, 2], field: 'text' }, 'warning');
  assert.deepEqual(pair.rows, [2, 0]);
  assert.equal(pair.strong, true);
  assert.equal(pair.level, 'warning');
  assert.equal(parseIssue({ code: 'too_few', level: 'error', message: '문장은 4개 이상이어야 해요', row: null, rows: [] }, 'error').rows.length, 0);
  assert.equal(parseIssue('그냥 문구', 'warning').message, '그냥 문구');
  assert.equal(parseIssue({ code: 'x' }, 'error'), null, '문구가 없으면 버린다');
  const checks = parseChecks({ errors: [{ message: 'e', row: 1 }], warnings: ['w', null] });
  assert.equal(checks.errors.length, 1);
  assert.equal(checks.warnings.length, 1);
  const top = checksFromError({ response: { status: 400, data: { detail: '내 목록에 오류가 있어요', errors: [{ message: 'e', row: 0 }], warnings: [] } } });
  assert.equal(top.detail, '내 목록에 오류가 있어요');
  assert.equal(top.errors[0].rows[0], 0);
  const nested = checksFromError({ response: { status: 400, data: { detail: { detail: '오류', errors: [{ message: 'e' }], warnings: [] } } } });
  assert.equal(nested.errors.length, 1);
  assert.equal(checksFromError({ response: { status: 400, data: { detail: '목록 ID 형식이 틀려요' } } }), null);
  assert.equal(checksFromError(new Error('x')), null);
});

test('목록 요약·예시 파일 읽기·기본 목록 표 (읽기 전용)', () => {
  const lists = parseListSummaries({
    lists: [
      { id: 'b2', title: '나 목록', count: 6, emergency_count: 1, updated: 'x', sha256: SHA },
      { id: 'a1', title: '가 목록', count: 4, emergency_count: 0, updated: 'x', sha256: SHA2 },
      { id: '../bad', title: 'x' },
    ],
  });
  assert.deepEqual(lists.map((l) => l.id), ['a1', 'b2'], '이름순, 이상한 ID는 버림');
  assert.equal(lists[1].emergencyCount, 1);
  assert.deepEqual(parseListSummaries(null), []);
  assert.deepEqual(parseExamples({ schema: 'other', examples: [] }), []);
  const ex = parseExamples({
    schema: 'ward-custom-list-examples-v1',
    examples: [{ title: '연습', commands: [{ text: '가나다', display: '가', category: 'care', emergency: false, cancel: false }, { text: 3 }] }],
  });
  assert.equal(ex.length, 1);
  assert.equal(ex[0].commands.length, 1);
  const rows = defaultListRows(COMMANDS);
  assert.equal(rows.length, 20);
  assert.deepEqual(rows.filter((r) => r.cancel).map((r) => r.id), ['W20']);
  assert.deepEqual(rows.filter((r) => r.emergency).map((r) => r.id), ['W02', 'W03', 'W04', 'W10']);
});

test('예시 목록 파일(있으면): 형식이 맞고 즉시 검사·분류·긴급·취소 규칙을 지킨다', { skip: !existsSync(new URL('../src/ward/custom_list_examples.json', import.meta.url)) }, () => {
  const raw = JSON.parse(src('../src/ward/custom_list_examples.json'));
  const examples = parseExamples(raw);
  assert.ok(examples.length >= 1);
  const catIds = COMMANDS.categories.map((c) => c.id);
  for (const ex of examples) {
    assert.ok(ex.commands.length >= 4 && ex.commands.length <= MAX_COMMANDS, ex.title);
    assert.equal(quickTitleProblem(ex.title), null);
    for (const c of ex.commands) {
      assert.equal(quickTextProblem(c.text), null, c.text);
      assert.equal(quickDisplayProblem(c.display), null, c.display);
      assert.ok(catIds.includes(c.category), c.category);
      assert.equal(c.emergency, c.category === 'emergency', c.text);
      assert.ok(!(c.cancel && c.emergency), c.text);
    }
    assert.ok(ex.commands.filter((c) => c.cancel).length <= 1);
    // 예시를 편집 상태로 불러오면 저장할 수 있는 모양이다(서버 검사 전 기준).
    let s = createListEditor({ categories: COMMANDS.categories, id: 'exam1' });
    s = stepListEditor(s, { type: 'LOAD_EXAMPLE', example: ex });
    assert.equal(canSaveList(s, []).ok, true, ex.title);
  }
});

// ---------------------------------------------------------------------------
// 편집 상태도
// ---------------------------------------------------------------------------

/** 줄을 채운 새 목록. */
function filledEditor() {
  let s = createListEditor({ categories: COMMANDS.categories, id: 'rehab' });
  s = stepListEditor(s, { type: 'SET_TITLE', value: '재활 병동' });
  const texts = ['창문을 조금만 열어 주실래요', '가슴이 갑자기 너무 조여요', '다리 운동을 도와주시겠어요', '아까 부탁한 일은 안 해도 돼요'];
  const displays = ['창문을 열어 주세요', '가슴이 조여요', '다리 운동을 도와주세요', '방금 요청은 취소할게요'];
  s.rows.forEach((r, i) => {
    s = stepListEditor(s, { type: 'SET_TEXT', key: r.key, value: texts[i] });
    s = stepListEditor(s, { type: 'SET_DISPLAY', key: r.key, value: displays[i] });
  });
  return s;
}

test('편집: 새 목록은 빈 줄 4개, 줄 추가는 20개까지, 지우기·위아래 옮기기', () => {
  let s = createListEditor({ categories: COMMANDS.categories, id: 'mylist1' });
  assert.equal(s.mode, 'new');
  assert.equal(s.rows.length, 4);
  assert.equal(s.dirty, false);
  for (let i = 0; i < 30; i += 1) s = stepListEditor(s, { type: 'ADD_ROW' });
  assert.equal(s.rows.length, MAX_COMMANDS);
  assert.equal(stepListEditor(s, { type: 'ADD_ROW' }), s, '20개면 그대로');
  const keys = s.rows.map((r) => r.key);
  s = stepListEditor(s, { type: 'REMOVE_ROW', key: keys[0] });
  assert.equal(s.rows.length, 19);
  s = stepListEditor(s, { type: 'MOVE_ROW', key: keys[2], delta: -1 });
  assert.deepEqual(s.rows.slice(0, 2).map((r) => r.key), [keys[2], keys[1]]);
  assert.equal(stepListEditor(s, { type: 'MOVE_ROW', key: keys[2], delta: -1 }), s, '맨 위에서 더 못 올림');
  assert.equal(s.dirty, true);
});

test('편집: 긴급 ⇔ 종류 긴급, 취소는 한 줄만이고 긴급일 수 없다', () => {
  let s = filledEditor();
  const [a, b, c] = s.rows.map((r) => r.key);
  s = stepListEditor(s, { type: 'SET_CATEGORY', key: a, value: 'room' });
  s = stepListEditor(s, { type: 'SET_EMERGENCY', key: a, value: true });
  let row = s.rows.find((r) => r.key === a);
  assert.deepEqual([row.category, row.emergency], ['emergency', true]);
  s = stepListEditor(s, { type: 'SET_EMERGENCY', key: a, value: false });
  row = s.rows.find((r) => r.key === a);
  assert.deepEqual([row.category, row.emergency], ['room', false], '긴급을 끄면 이전 종류로');
  s = stepListEditor(s, { type: 'SET_CATEGORY', key: b, value: 'emergency' });
  assert.equal(s.rows.find((r) => r.key === b).emergency, true);
  assert.equal(stepListEditor(s, { type: 'SET_CATEGORY', key: b, value: 'nope' }), s, '모르는 종류는 무시');
  s = stepListEditor(s, { type: 'SET_CANCEL', key: c, value: true });
  s = stepListEditor(s, { type: 'SET_CANCEL', key: b, value: true });
  assert.deepEqual(s.rows.filter((r) => r.cancel).map((r) => r.key), [b], '취소는 마지막으로 고른 한 줄');
  row = s.rows.find((r) => r.key === b);
  assert.equal(row.emergency, false, '취소로 고른 긴급 줄은 일반으로');
  assert.notEqual(row.category, 'emergency');
  s = stepListEditor(s, { type: 'SET_EMERGENCY', key: b, value: true });
  row = s.rows.find((r) => r.key === b);
  assert.deepEqual([row.emergency, row.cancel], [true, false], '긴급으로 바꾸면 취소가 풀린다');
  const body = requestBody(s);
  assert.deepEqual(Object.keys(body), ['title', 'commands']);
  assert.deepEqual(Object.keys(body.commands[0]), ['text', 'display', 'category', 'emergency', 'cancel']);
});

test('편집: 서버 검사는 지금 내용(editSeq)의 결과만 쓰고, 줄 위치를 검사할 때의 줄에 붙인다', () => {
  let s = filledEditor();
  const keys = s.rows.map((r) => r.key);
  const seq = s.editSeq;
  s = stepListEditor(s, { type: 'CHECK_START', seq });
  assert.equal(s.check.status, 'checking');
  const data = {
    errors: [{ code: 'dup', level: 'error', message: '같은 문장이 있어요', row: 3, rows: [1, 3], field: 'text' }],
    warnings: [{ code: 'emergency_many', level: 'warning', message: '긴급이 많아요', row: null, rows: [] }],
    normalized: { title: '재활 병동', commands: [] },
  };
  const done = stepListEditor(s, { type: 'CHECK_DONE', seq, data });
  let m = editorMessages(done);
  assert.equal(m.current, true);
  assert.equal(m.byKey[keys[1]].errors.length, 1);
  assert.equal(m.byKey[keys[3]].errors.length, 1);
  assert.equal(m.byKey[keys[0]].errors.length, 0);
  assert.equal(m.global.warnings.length, 1, '줄이 없는 항목은 전체 안내');
  assert.equal(canSaveList(done, []).ok, false, '지금 내용에 오류가 있으면 저장 안 함');
  assert.equal(canSaveList(done, []).reason, LISTS_TEXT.errorsBlock);
  // 옮긴 뒤에도 그 항목은 원래 줄(key)을 따라간다. 내용이 바뀌었으니 지금 결과는 아니다.
  const moved = stepListEditor(done, { type: 'MOVE_ROW', key: keys[3], delta: -1 });
  m = editorMessages(moved);
  assert.equal(m.current, false);
  assert.equal(m.byKey[keys[3]].errors.length, 1);
  // 늦게 온 예전 검사 결과는 버린다.
  assert.equal(stepListEditor(moved, { type: 'CHECK_DONE', seq, data: { errors: [], warnings: [] } }), moved);
  assert.equal(stepListEditor(moved, { type: 'CHECK_FAILED', seq, message: 'x' }), moved);
  const failed = stepListEditor(moved, { type: 'CHECK_FAILED', seq: moved.editSeq, message: '서버 없음' });
  assert.equal(failed.check.status, 'failed');
  assert.equal(canSaveList(failed, []).ok, true, '서버 검사를 못 해도 저장 때 서버가 다시 검사한다');
});

test('편집: 저장 가능 조건·저장 실패(400 검사 결과)·저장 성공·불러오기·예시·병실 화면에 쓰기 조건', () => {
  let s = filledEditor();
  assert.equal(canSaveList(s, []).ok, true);
  assert.equal(canSaveList(s, ['REHAB']).reason, LISTS_TEXT.idTaken);
  assert.equal(canSaveList({ ...s, title: '' }, []).reason, LISTS_TEXT.titleRequired);
  const three = stepListEditor(s, { type: 'REMOVE_ROW', key: s.rows[0].key });
  assert.equal(canSaveList(three, []).reason, LISTS_TEXT.rowsMin);
  const badText = stepListEditor(s, { type: 'SET_TEXT', key: s.rows[0].key, value: 'TV 켜 줘' });
  assert.equal(canSaveList(badText, []).reason, LISTS_TEXT.textHangulOnly);
  assert.equal(canUseList(s), false, '저장 전에는 병실 화면에 쓸 수 없다');

  s = stepListEditor(s, { type: 'SAVE_START' });
  assert.equal(canSaveList(s, []).ok, false, '저장 중');
  const checks = checksFromError({ response: { status: 400, data: { detail: '내 목록에 오류가 있어요', errors: [{ message: '두 줄이 같아요', rows: [0, 1] }], warnings: [] } } });
  const failed = stepListEditor(s, { type: 'SAVE_FAILED', seq: s.editSeq, checks, message: checks.detail });
  assert.equal(failed.saving, false);
  assert.equal(failed.saveMessage, '내 목록에 오류가 있어요');
  assert.equal(editorMessages(failed).byKey[failed.rows[1].key].errors.length, 1);

  const savedList = serverList({ commands: serverList().commands.slice(0, 4), warnings: [{ message: '긴급이 하나뿐이에요', level: 'warning' }] });
  delete savedList.checks;
  const saved = stepListEditor(s, { type: 'SAVED', list: savedList });
  assert.equal(saved.mode, 'edit');
  assert.equal(saved.savedId, 'rehab');
  assert.equal(saved.dirty, false);
  assert.equal(saved.savedSha, SHA);
  assert.equal(saved.saveMessage, LISTS_TEXT.saved);
  assert.equal(saved.check.warnings.length, 1, 'PUT 응답의 주의를 보인다');
  assert.equal(canUseList(saved), true);
  assert.equal(canSaveList(saved, ['rehab']).ok, false, '바뀐 것이 없으면 저장 버튼을 끈다');
  assert.equal(stepListEditor(saved, { type: 'SET_ID', value: 'other' }), saved, '저장한 목록의 ID는 바꾸지 않는다');
  const edited = stepListEditor(saved, { type: 'SET_TITLE', value: '재활 병동 2' });
  assert.equal(canUseList(edited), false, '고친 뒤 저장 전에는 병실 화면에 쓸 수 없다');
  assert.equal(canSaveList(edited, ['rehab']).ok, true);

  const loaded = stepListEditor(createListEditor({ categories: COMMANDS.categories }), { type: 'LOAD_LIST', list: serverList() });
  assert.equal(loaded.rows.length, 5);
  assert.deepEqual(loaded.rows.map((r) => [r.category, r.emergency, r.cancel]).slice(1, 4),
    [['emergency', true, false], ['care', false, false], ['call', false, true]]);
  assert.equal(canUseList(loaded), true);
  const withErrors = stepListEditor(loaded, { type: 'LOAD_LIST', list: serverList({ checks: { errors: [{ message: 'x' }], warnings: [] } }) });
  assert.equal(canUseList(withErrors), false, '서버 검사 오류가 있는 목록은 병실 화면에 쓰지 않는다');

  const ex = { title: '예시', commands: serverList().commands.map(({ id, ...rest }) => rest) };
  // 저장된 목록을 열어 둔 채 예시를 불러와도 그 목록을 덮어쓰지 않는다: 새 목록(새 ID)으로 연다.
  const fromEx = stepListEditor(loaded, { type: 'LOAD_EXAMPLE', example: ex, id: 'mylist2' });
  assert.equal(fromEx.mode, 'new', '예시는 언제나 새 목록');
  assert.equal(fromEx.savedId, null, '저장된 목록(rehab)과 이어지지 않는다');
  assert.equal(fromEx.savedSha, null);
  assert.equal(fromEx.id, 'mylist2');
  assert.equal(fromEx.dirty, true);
  assert.equal(fromEx.title, '예시');
  assert.equal(fromEx.rows.length, 5);
  assert.deepEqual(saveCondition(fromEx), { create: true }, '저장하면 없을 때만 만든다(PUT If-None-Match: *)');
  assert.ok(fromEx.editSeq > loaded.editSeq, '검사 짝이 맞게 editSeq는 계속 는다');
  // 새 목록 편집 중이면 적어 둔 ID를 그대로 둔다.
  const typed = stepListEditor(stepListEditor(createListEditor({ categories: COMMANDS.categories, id: 'mylist1' }),
    { type: 'SET_ID', value: 'ward7' }), { type: 'LOAD_EXAMPLE', example: ex, id: 'mylist3' });
  assert.equal(typed.id, 'ward7');
  assert.equal(typed.mode, 'new');
});

test('저장 조건: 새 목록은 없을 때만 만들기, 저장된 목록은 불러온 판일 때만 고치기 (다른 탭 저장을 덮어쓰지 않음)', () => {
  const fresh = createListEditor({ categories: COMMANDS.categories, id: 'mylist1' });
  assert.deepEqual(saveCondition(fresh), { create: true });
  const loaded = stepListEditor(fresh, { type: 'LOAD_LIST', list: serverList() });
  assert.deepEqual(saveCondition(loaded), { ifMatch: SHA });
  assert.deepEqual(saveCondition({ ...loaded, savedSha: null }), {}, '판을 모르면 조건 없이');
  assert.deepEqual(saveCondition(null), {});
  assert.deepEqual(listSaveHeaders({ create: true }), { 'If-None-Match': '*' });
  assert.deepEqual(listSaveHeaders({ ifMatch: SHA }), { 'If-Match': `"${SHA}"` });
  assert.deepEqual(listSaveHeaders({}), {});
  assert.deepEqual(listSaveHeaders(), {});
  assert.throws(() => listSaveHeaders({ create: true, ifMatch: SHA }), (e) => e.isClientCheck === true);
  assert.throws(() => listSaveHeaders({ ifMatch: 'x'.repeat(64) }), (e) => e.isClientCheck === true);
});

// ---------------------------------------------------------------------------
// 내 목록 색인
// ---------------------------------------------------------------------------

test('기본 목록 색인: 취소 명령은 W20 하나, customList 없음 (예전과 같은 명령·분류)', () => {
  assert.equal(INDEX.cancelId, CANCEL_COMMAND_ID);
  assert.equal(INDEX.customList, null);
  assert.deepEqual(INDEX.list.filter(isCancelCommand).map((c) => c.id), ['W20']);
  assert.equal(INDEX.categories.length, 5);
});

test('내 목록 색인: 일반은 action custom, 취소는 cancel_last, 빈 분류는 빼고, 목록 표시를 단다', () => {
  assert.equal(CUSTOM.list.length, 5);
  assert.equal(CUSTOM.cancelId, 'U04');
  assert.equal(commandById(CUSTOM, 'U04').action, 'cancel_last');
  assert.equal(commandById(CUSTOM, 'U01').action, 'custom');
  assert.equal(commandById(CUSTOM, 'W05'), null, '기본 목록 명령은 없다');
  assert.deepEqual(CUSTOM.categories.map((c) => c.id), ['emergency', 'call', 'care', 'room'], '대답 분류는 명령이 없어 뺀다');
  assert.equal(CUSTOM.categories[0].label, '긴급');
  assert.deepEqual(CUSTOM.customList, { id: 'rehab', title: '재활 병동', sha256: SHA, count: 5 });
  const bad = [
    serverList({ id: '../x' }),
    serverList({ sha256: 'xyz' }),
    serverList({ title: ' ' }),
    serverList({ commands: [{ ...serverList().commands[0], id: 'W01' }, serverList().commands[1]] }),
    serverList({ commands: [{ ...serverList().commands[0], emergency: true }, serverList().commands[1]] }),
    serverList({ commands: [{ ...serverList().commands[1], cancel: true }, serverList().commands[0]] }),
    serverList({ commands: [serverList().commands[3], { ...serverList().commands[0], cancel: true }] }),
    serverList({ commands: [serverList().commands[0], { ...serverList().commands[2], category: 'nope' }] }),
    serverList({ commands: [serverList().commands[0]] }),
  ];
  for (const b of bad) assert.throws(() => buildCustomCommandIndex(b, COMMANDS), /내 목록을 읽지 못했습니다/);
  const empty = emptyListIndex('rehab', '재활 병동');
  assert.equal(empty.list.length, 0);
  assert.equal(empty.customList.sha256, null);
});

test('내 목록 불러오기 결과: 오류가 있거나 형식이 틀리면 실패(기본 목록으로 바꾸지 않음), 404·연결 문구', () => {
  const ok = listLoadOutcome(serverList(), 'rehab', COMMANDS);
  assert.equal(ok.ok, true);
  assert.equal(ok.index.customList.id, 'rehab');
  assert.ok(ok.index.list.every((c) => c.id.startsWith('U')), '기본 목록 명령이 섞이지 않는다');
  assert.deepEqual(listLoadOutcome(serverList({ checks: { errors: [{ message: 'x', row: 0 }], warnings: [] } }), 'rehab', COMMANDS),
    { ok: false, message: LISTS_TEXT.hasErrors, title: '재활 병동' });
  assert.equal(listLoadOutcome(serverList(), 'other', COMMANDS).ok, false, '다른 목록 응답');
  assert.equal(listLoadOutcome(serverList({ schema: 'v9' }), 'rehab', COMMANDS).ok, false);
  assert.equal(listLoadOutcome(serverList({ commands: [] }), 'rehab', COMMANDS).message, LISTS_TEXT.broken);
  assert.equal(listLoadOutcome(null, 'rehab', COMMANDS).ok, false);
  assert.equal(listLoadErrorMessage({ response: { status: 404, data: { detail: '없음' } } }), LISTS_TEXT.notFound);
  assert.equal(listLoadErrorMessage({ response: { status: 400, data: { detail: '목록 ID 형식이 틀려요' } } }), '목록 ID 형식이 틀려요');
  assert.equal(listLoadErrorMessage(new Error('Network Error')), LISTS_TEXT.loadFailed);
  assert.equal(listBannerText(CUSTOM.customList), '내 목록 ‘재활 병동’ 사용 중 · 성능을 측정하지 않은 목록');
  assert.equal(listBannerText(null), null);
});

// ---------------------------------------------------------------------------
// 결과 받아들이기 (목록 ID + 지문)
// ---------------------------------------------------------------------------

test('결과는 목록 ID와 지문이 지금 목록과 같을 때만 쓴다. 기본 목록 결과에는 list가 없어야 한다', () => {
  assert.equal(classifyWardResults(phraseResult('U01', 'U03', true), CUSTOM).kind, 'phrase');
  assert.equal(classifyWardResults(phraseResult('U01', 'U03', true), CUSTOM).command.id, 'U01');
  const other = { kind: 'no_phrase', reason: 'other_set' };
  assert.deepEqual(classifyWardResults(phraseResult('U01', 'U03', true, { list: { id: 'rehab', sha256: SHA2 } }), CUSTOM), other, '목록을 고친 뒤 온 결과');
  assert.deepEqual(classifyWardResults(phraseResult('U01', 'U03', true, { list: { id: 'other', sha256: SHA } }), CUSTOM), other, '다른 목록');
  assert.deepEqual(classifyWardResults(phraseResult('U01', 'U03', true, { list: null }), CUSTOM), other, 'list 없는 결과');
  // 불러오는 중·실패(지문 없음) 색인은 어떤 결과도 받지 않는다.
  assert.deepEqual(classifyWardResults(phraseResult('U01', 'U03', true), emptyListIndex('rehab')), other);
  // 기본 목록: list가 없으면 예전과 같고, list가 붙은 결과는 쓰지 않는다.
  const w = phraseResult('W05', 'W06', true, { list: null, index: INDEX });
  assert.equal(classifyWardResults(w, INDEX).command.id, 'W05');
  assert.deepEqual(classifyWardResults(phraseResult('W05', 'W06', true, { index: INDEX }), INDEX), other);
  assert.equal(phraseListMatches(undefined, null), true);
  assert.equal(phraseListMatches({ id: 'rehab', sha256: SHA }, { id: 'rehab', sha256: null }), false);
  assert.equal(resultsFromOtherListVersion(phraseResult('U01', 'U03', true, { list: { id: 'rehab', sha256: SHA2 } }), CUSTOM.customList), true);
  assert.equal(resultsFromOtherListVersion(phraseResult('U01', 'U03', true), CUSTOM.customList), false);
  assert.equal(resultsFromOtherListVersion([{ text: '[발화 없음]' }], CUSTOM.customList), false);
});

// ---------------------------------------------------------------------------
// 상태도: 내 목록으로 보내기·취소 일반화·기본 목록으로 몰래 바꾸지 않기
// ---------------------------------------------------------------------------

test('내 목록 일반 명령: 확인 뒤 카드에 list {id, title}, 가상 병실은 움직이지 않는다', () => {
  const s0 = createWardState({ index: CUSTOM, sessionKey: 'c' });
  const r = recordWith(s0, T0, phraseResult('U01', 'U03', true));
  assert.equal(r.state.phase, 'confirm');
  const done = stepWard(r.state, { type: 'TICK', at: T0 + 30000 });
  const cards = sent(done.effects);
  assert.deepEqual(cards.map((c) => [c.kind, c.commandId, c.text]), [['request', 'U01', '창문을 열어 주세요']]);
  assert.deepEqual(cards[0].list, { id: 'rehab', title: '재활 병동', sha256: SHA }, '목록 판도 함께 (간호사 카드 합치기 열쇠)');
  assert.deepEqual(done.state.room, s0.room, '내 목록 동작(custom)은 가상 병실을 바꾸지 않는다');
  // 긴급 명령은 기본 목록과 같이 바로 빨간 카드.
  const e = recordWith(done.state, T0 + 40000, phraseResult('U02', 'U01', true));
  assert.deepEqual(sent(e.effects).map((c) => [c.kind, c.commandId, c.list?.id]), [['emergency', 'U02', 'rehab']]);
});

test('취소 일반화: 기본 목록 W20은 예전처럼, 내 목록은 취소로 고른 명령(U04)이 마지막 일반 카드에 취소 요청을 단다', () => {
  // 기본 목록: W05 보내기 → W20
  let d = createWardState({ index: INDEX, sessionKey: 'd' });
  d = stepWard(recordWith(d, T0, phraseResult('W05', 'W06', true, { list: null, index: INDEX })).state, { type: 'TICK', at: T0 + 20000 }).state;
  const dc = recordWith(d, T0 + 30000, phraseResult('W20', 'W16', true, { list: null, index: INDEX }));
  const dEnd = stepWard(dc.state, { type: 'TICK', at: T0 + 60000 });
  assert.deepEqual(updated(dEnd.effects).map((c) => [c.commandId, c.cancelRequested]), [['W05', true]]);
  assert.equal(dEnd.state.screen.kind, 'cancel_request');
  assert.equal(dEnd.effects.filter((e) => e.type === 'SEND_CARD').length, 0, 'W20은 새 카드를 만들지 않는다');
  assert.equal(updated(dEnd.effects)[0].list, undefined, '기본 목록 카드에는 list가 없다');
  // 내 목록: U03 보내기 → U04(취소)
  let c = createWardState({ index: CUSTOM, sessionKey: 'u' });
  c = stepWard(recordWith(c, T0, phraseResult('U03', 'U05', true)).state, { type: 'TICK', at: T0 + 20000 }).state;
  const cc = recordWith(c, T0 + 30000, phraseResult('U04', 'U01', true));
  assert.equal(cc.state.phase, 'confirm', '취소 명령도 일반 명령이라 확인 카운트다운');
  const cEnd = stepWard(cc.state, { type: 'TICK', at: T0 + 60000 });
  assert.deepEqual(updated(cEnd.effects).map((x) => [x.commandId, x.cancelRequested, x.note]), [['U03', true, WARD_TEXT.cancelRequestNote]]);
  assert.equal(sent(cEnd.effects).length, 0);
  // 내 목록에서 W20이라는 ID는 취소가 아니다(없는 명령).
  assert.equal(isCancelCommand(commandById(CUSTOM, 'W20')), false);
  assert.equal(isCancelCommand(commandById(CUSTOM, 'U01')), false);
});

test('기본 목록으로 몰래 바꾸지 않기: 불러오기 실패면 입 읽기 녹화를 막고, 스위치 3번 긴급 호출은 된다', () => {
  const blocked = createWardState({
    index: emptyListIndex('rehab', '재활 병동'), lipBlock: { reason: 'error', message: LISTS_TEXT.notFound }, sessionKey: 'b',
  });
  const one = stepWard(blocked, { type: 'PRESS', at: T0 });
  assert.equal(one.effects.some((e) => e.type === 'START_RECORDING'), false, '녹화를 시작하지 않는다');
  assert.equal(one.state.phase, 'idle');
  const v = wardView(one.state, T0);
  assert.equal(v.headline, WARD_TEXT.listBlocked);
  assert.match(v.sub, /내 목록을 찾을 수 없어요/);
  assert.match(v.sub, /스위치 3번/);
  assert.equal(wardView(blocked, T0).headline, WARD_TEXT.listBlocked, '대기 화면도 막힘을 보인다');
  const triple = feed(blocked, [{ type: 'PRESS', at: T0 }, { type: 'PRESS', at: T0 + 800 }, { type: 'PRESS', at: T0 + 1600 }]);
  const cards = sent(triple.effects);
  assert.deepEqual(cards.map((c) => c.kind), ['emergency_call']);
  assert.equal(triple.effects.some((e) => e.type === 'START_RECORDING'), false);
  // 막힌 동안 들어온 결과(연습 주입 포함)는 명령으로 해석하지 않는다(기본 목록 ID든 내 목록 ID든).
  for (const results of [phraseResult('W05', 'W06', true, { list: null, index: INDEX }), phraseResult('U01', 'U03', true), [{ text: '[발화 없음]' }]]) {
    const sim = stepWard(blocked, { type: 'SIMULATE', at: T0, results });
    assert.equal(sent(sim.effects).length, 0, '카드 없음');
    assert.equal(sim.state.phase, 'idle');
    assert.equal(sim.state.screen.kind, 'lip_blocked');
  }
  // 불러오는 중 표시
  const loading = createWardState({ index: emptyListIndex('rehab'), lipBlock: { reason: 'loading' }, sessionKey: 'l' });
  assert.equal(wardView(loading, T0).headline, WARD_TEXT.listLoading);
});

test('SET_INDEX: 목록을 바꾸면 진행 중인 흐름을 멈추고 늦은 결과는 무시, 막힘을 풀면 녹화가 된다. 카드는 남는다', () => {
  let s = createWardState({ index: emptyListIndex('rehab'), lipBlock: { reason: 'loading' }, sessionKey: 'i' });
  s = stepWard(s, { type: 'SET_INDEX', at: T0, index: CUSTOM, lipBlock: null }).state;
  assert.equal(s.lipBlock, null);
  assert.equal(s.index, CUSTOM);
  const same = stepWard(s, { type: 'SET_INDEX', at: T0, index: CUSTOM, lipBlock: null });
  assert.equal(same.state, s, '같은 목록이면 그대로');
  const rec = stepWard(s, { type: 'PRESS', at: T0 + 1000 });
  assert.equal(rec.state.phase, 'recording');
  const run = rec.state.run;
  // 녹화 중에 목록이 바뀜(예: 서버의 목록을 고침) → 흐름을 멈추고 실행 번호를 바꾼다.
  const changed = buildCustomCommandIndex(serverList({ sha256: SHA2 }), COMMANDS);
  const sw = stepWard(rec.state, { type: 'SET_INDEX', at: T0 + 2000, index: changed, lipBlock: null });
  assert.equal(sw.state.phase, 'idle');
  assert.notEqual(sw.state.run, run);
  const late = stepWard(sw.state, { type: 'RESULT', at: T0 + 7000, run, results: phraseResult('U01', 'U03', true) });
  assert.equal(late.state, sw.state, '예전 실행의 결과는 무시');
  // 보낸 카드는 목록을 바꿔도 남는다.
  const withCard = stepWard(recordWith(s, T0, phraseResult('U02', 'U01', true)).state, { type: 'SET_INDEX', at: T0 + 9000, index: INDEX, lipBlock: null });
  assert.equal(withCard.state.cardOrder.length, 1);
  assert.equal(withCard.state.index, INDEX);
  assert.equal(stepWard(withCard.state, { type: 'SET_INDEX', at: T0 + 9100, index: { nope: true } }).state.index, INDEX, '형식이 틀린 색인은 무시');
  // 목록을 바꿀 수 있는 때·목록 관리 화면을 열 수 있는 때
  assert.equal(listSwitchBlockReason(s), null);
  assert.equal(listSwitchBlockReason(rec.state), WARD_TEXT.listSwitchBusy);
  assert.equal(listsOpenBlockReason(rec.state), WARD_TEXT.listsBusy);
  assert.equal(listsOpenBlockReason(withCard.state), WARD_TEXT.listsAlarmOpen, '확인 안 된 긴급 카드');
  assert.equal(listsOpenBlockReason(s), null);
});

test('내 목록 분류 스캐닝: 명령이 있는 분류만, 고른 취소 명령은 취소 요청', () => {
  let s = createWardState({ index: CUSTOM, sessionKey: 's' });
  s = stepWard(recordWith(s, T0, phraseResult('U03', 'U05', true)).state, { type: 'TICK', at: T0 + 20000 }).state;
  // 결과 없음 → 분류 스캐닝
  const r = recordWith(s, T0 + 30000, [{ start_ms: 0, end_ms: 5000, text: 'x', confidence: 0.5, raw_score: -1 }]);
  assert.equal(r.state.phase, 'scan');
  const view = wardView(r.state, T0 + 36000);
  assert.deepEqual(view.items.map((x) => x.label), ['긴급', '호출·소통', '몸 돌봄', '병실', '다시 말하기', '닫기']);
  // 호출·소통(두 번째) → 그 안의 첫 명령이 U04(취소)
  const pick = feed(r.state, [
    { type: 'TICK', at: T0 + 38000 }, { type: 'PRESS', at: T0 + 38100 }, { type: 'PRESS', at: T0 + 38700 },
  ]);
  assert.deepEqual(updated(pick.effects).map((c) => [c.commandId, c.cancelRequested]), [['U03', true]]);
});

// ---------------------------------------------------------------------------
// 간호사 카드
// ---------------------------------------------------------------------------

test('간호사 카드: 기본 목록 열쇠는 예전 그대로, 내 목록 카드는 목록 ID를 붙여 W05·U05·다른 목록 U05와 합치지 않는다', () => {
  const base = {
    rev: 1, room: '305', kind: 'request', commandId: 'W05', text: '물을 마시고 싶어요', candidates: [], reason: null, hint: null,
    note: null, cancelRequested: false, selectedId: null, source: 'lip', createdAt: T0, updatedAt: T0, acked: false, ackedAt: null,
  };
  const w5 = { ...base, id: 'a-1' };
  const u5 = { ...base, id: 'a-2', commandId: 'U05', text: '휠체어를 가져다주세요', list: { id: 'rehab', title: '재활 병동', sha256: SHA } };
  const u5b = { ...u5, id: 'a-3' };
  const u5other = { ...u5, id: 'a-4', list: { id: 'night', title: '밤 목록', sha256: SHA } };
  assert.equal(mergeKey(w5), '305|request|W05');
  assert.equal(mergeKey(u5), `305|request|U05|list:rehab@${SHA}`);
  assert.equal(mergeKey({ ...u5, list: 'x' }), null, '형식이 틀린 list는 어떤 카드와도 합치지 않는다');
  assert.equal(mergeKey({ ...u5, list: { id: 'rehab', title: '재활 병동' } }), null, '판(sha256)이 없는 list도 합치지 않는다');
  assert.equal(mergeKey({ ...base, id: 'n', kind: 'check', commandId: null, reason: 'no_mouth', list: { id: 'rehab', title: 't', sha256: SHA } }),
    `305|check|no_mouth|list:rehab@${SHA}`);
  let cards = upsertNurseCard([], w5, T0).cards;
  cards = upsertNurseCard(cards, { ...w5, id: 'a-5', commandId: 'U05', list: { id: 'rehab', title: '재활 병동', sha256: SHA } }, T0 + 1).cards;
  assert.equal(cards.length, 2, '기본 W05와 내 목록 카드는 따로');
  cards = upsertNurseCard([], u5, T0).cards;
  const merged = upsertNurseCard(cards, u5b, T0 + 1);
  assert.equal(merged.alert, 'merged');
  assert.equal(merged.cards[0].count, 2);
  assert.equal(upsertNurseCard(merged.cards, u5other, T0 + 2).cards.length, 2, '다른 내 목록의 U05와는 따로');
  assert.deepEqual(cardListTag(u5), { id: 'rehab', title: '재활 병동', sha256: SHA });
  assert.deepEqual(cardListTag({ ...u5, list: { id: 'rehab', title: '재활 병동', sha256: 'zz' } }), { id: 'rehab', title: '재활 병동', sha256: null });
  assert.equal(cardListNote(u5), '내 목록: 재활 병동');
  assert.equal(cardListNote(w5), null);
  assert.equal(cardListNote({ ...u5, list: { id: 'rehab', title: '' } }), '내 목록: rehab');
  // 갱신으로 list가 바뀌지 않는다(CONTENT_FIELDS 밖).
  const up = upsertNurseCard(merged.cards, { ...u5b, rev: 2, list: { id: 'zzz', title: 'z' }, note: '환자가 취소 요청함' }, T0 + 3);
  assert.equal(up.cards[0].list.id, 'rehab');
});

test('간호사 카드: 목록을 고친 뒤(판이 다름) 같은 U 번호의 새 요청·긴급은 옛 카드에 합치지 않고 따로 보인다', () => {
  const base = {
    rev: 1, room: '305', candidates: [], reason: null, hint: null, note: null, cancelRequested: false, selectedId: null,
    source: 'lip', createdAt: T0, updatedAt: T0, acked: false, ackedAt: null,
  };
  const v1 = { id: 'rehab', title: '재활 병동', sha256: SHA };
  const v2 = { id: 'rehab', title: '재활 병동', sha256: SHA2 };
  for (const kind of ['request', 'emergency']) {
    const old = { ...base, id: `w1-1-${kind}`, kind, commandId: 'U01', text: '창문을 열어 주세요', list: v1 };
    const fresh = { ...base, id: `w2-1-${kind}`, kind, commandId: 'U01', text: '수술 부위에서 피가 나요', list: v2 };
    let cards = upsertNurseCard([], old, T0).cards;
    const r = upsertNurseCard(cards, fresh, T0 + 1);
    assert.equal(r.alert, 'new', kind);
    assert.deepEqual(r.cards.map((c) => [c.text, c.count]), [['수술 부위에서 피가 나요', 1], ['창문을 열어 주세요', 1]], kind);
    // 같은 판이면 예전처럼 합친다.
    cards = upsertNurseCard(r.cards, { ...fresh, id: `w2-2-${kind}` }, T0 + 2).cards;
    assert.deepEqual(cards.map((c) => [c.text, c.count]), [['수술 부위에서 피가 나요', 2], ['창문을 열어 주세요', 1]], kind);
  }
});

test('상태도+간호사 카드: 목록 쓰기 → 문장 고치기(SET_INDEX) → 같은 U 번호 요청이 따로 보인다 (시연 순서)', () => {
  const v2 = buildCustomCommandIndex(serverList({
    sha256: SHA2,
    commands: serverList().commands.map((c) => (c.id === 'U01' ? { ...c, text: '창밖 소리가 너무 시끄러워요', display: '창문을 닫아 주세요' } : c)),
  }), COMMANDS);
  let s = createWardState({ index: CUSTOM, baseIndex: INDEX, sessionKey: 'p' });
  const first = stepWard(recordWith(s, T0, phraseResult('U01', 'U03', true)).state, { type: 'TICK', at: T0 + 30000 });
  s = stepWard(first.state, { type: 'SET_INDEX', at: T0 + 31000, index: v2 }).state;
  const second = stepWard(recordWith(s, T0 + 40000, phraseResult('U01', 'U03', true, { list: { id: 'rehab', sha256: SHA2 }, index: v2 })).state,
    { type: 'TICK', at: T0 + 80000 });
  const patientCards = [...sent(first.effects), ...sent(second.effects)];
  assert.deepEqual(patientCards.map((c) => [c.commandId, c.text, c.list.sha256]),
    [['U01', '창문을 열어 주세요', SHA], ['U01', '창문을 닫아 주세요', SHA2]]);
  let nurse = [];
  for (const c of patientCards) nurse = upsertNurseCard(nurse, c, T0 + 90000).cards;
  assert.deepEqual(nurse.map((c) => [c.text, c.count]), [['창문을 닫아 주세요', 1], ['창문을 열어 주세요', 1]]);
});

test('카드 목록 꼬리표: 불러온 내 목록에서만 붙이고, 스위치 3번 긴급 호출·불러오는 중에는 붙이지 않는다', () => {
  const press3 = (state, at) => feed(state, [{ type: 'PRESS', at }, { type: 'PRESS', at: at + 700 }, { type: 'PRESS', at: at + 1400 }]);
  const loadedList = press3(createWardState({ index: CUSTOM, baseIndex: INDEX, sessionKey: 'e' }), T0);
  const call = sent(loadedList.effects).find((c) => c.kind === 'emergency_call');
  assert.ok(call, '스위치 3번 = 긴급 호출');
  assert.equal(call.list, undefined, '긴급 호출은 목록과 상관없다');
  const loading = press3(createWardState({
    index: emptyListIndex('rehab', '재활 병동'), baseIndex: INDEX, sessionKey: 'f', lipBlock: { reason: 'loading' },
  }), T0);
  const call2 = sent(loading.effects).find((c) => c.kind === 'emergency_call');
  assert.ok(call2, '불러오는 중에도 긴급 호출은 된다');
  assert.equal(call2.list, undefined, '불러오지 못한 목록 꼬리표를 붙이지 않는다');
  assert.equal(mergeKey(call2), mergeKey({ ...call2, list: undefined }), '기본 목록 긴급 호출과 같은 열쇠');
});

test('가상 병실: 기본 목록 W14(등받이)를 보낸 뒤 내 목록으로 바꾸고 간호사가 확인해도 침대가 움직인다', () => {
  let s = createWardState({ index: INDEX, baseIndex: INDEX, sessionKey: 'g' });
  const r = stepWard(recordWith(s, T0, phraseResult('W14', 'W13', true, { list: null, index: INDEX })).state, { type: 'TICK', at: T0 + 30000 });
  const card = sent(r.effects).find((c) => c.commandId === 'W14');
  assert.ok(card, 'W14 카드');
  const before = r.state.room;
  // 확인하지 않고 바꾸지 않은 경우와 내 목록으로 바꾼 경우의 침대가 같아야 한다.
  const ackSame = stepWard(r.state, { type: 'NURSE_ACK', at: T0 + 40000, cardIds: [card.id], room: '305' }).state.room;
  s = stepWard(r.state, { type: 'SET_INDEX', at: T0 + 35000, index: CUSTOM }).state;
  assert.equal(s.baseIndex, INDEX, '기본 목록 색인은 그대로 기억한다');
  const ackSwitched = stepWard(s, { type: 'NURSE_ACK', at: T0 + 40000, cardIds: [card.id], room: '305' }).state.room;
  assert.notDeepEqual(ackSame, before, '확인하면 등받이가 움직인다');
  assert.deepEqual(ackSwitched, ackSame, '목록을 바꾼 뒤에도 같은 동작');
  // baseIndex를 주지 않아도 처음 색인이 기본 목록이면 그것을 기억한다.
  assert.equal(createWardState({ index: INDEX }).baseIndex, INDEX);
  assert.equal(createWardState({ index: CUSTOM }).baseIndex, null);
});

test('목록 이름만 바뀜(같은 판): 흐름을 멈추지 않고 띠·카드 이름만 바꾼다', () => {
  const renamed = buildCustomCommandIndex(serverList({ title: '재활 병동 2층' }), COMMANDS);
  const s0 = createWardState({ index: CUSTOM, baseIndex: INDEX, sessionKey: 'h' });
  const rec = feed(s0, [{ type: 'PRESS', at: T0 }]);
  assert.equal(rec.state.phase, 'recording');
  const after = stepWard(rec.state, { type: 'SET_INDEX', at: T0 + 100, index: renamed });
  assert.equal(after.state.phase, 'recording', '녹화를 멈추지 않는다');
  assert.equal(after.state.run, rec.state.run, '결과를 버리지 않는다');
  assert.equal(after.state.index, renamed);
  assert.equal(listBannerText(after.state.index.customList), '내 목록 ‘재활 병동 2층’ 사용 중 · 성능을 측정하지 않은 목록');
  const r2 = feed(after.state, [{ type: 'RECORDING_DONE', at: T0 + 5000, run: after.state.run },
    { type: 'RESULT', at: T0 + 6000, run: after.state.run, results: phraseResult('U01', 'U03', true) }, { type: 'TICK', at: T0 + 30000 }]);
  assert.equal(sent(r2.effects)[0].list.title, '재활 병동 2층');
  // 판이 다르면 예전처럼 흐름을 멈춘다.
  const other = buildCustomCommandIndex(serverList({ sha256: SHA2 }), COMMANDS);
  const reset = stepWard(rec.state, { type: 'SET_INDEX', at: T0 + 100, index: other });
  assert.equal(reset.state.phase, 'idle');
  assert.equal(reset.state.run, rec.state.run + 1);
});

test('화면 즉시 검사가 서버와 같은 공백·정규화 규칙: BOM은 공백이 아님, NFD 붙여 넣기, NBSP', () => {
  assert.equal(normalizeListText('\ufeff보호자를 불러 주실래요').startsWith('\ufeff'), true, 'BOM은 지우지 않는다(서버도 쓸 수 없는 글자로 알림)');
  assert.equal(quickTextProblem('\ufeff보호자를 불러 주실래요'), LISTS_TEXT.textHangulOnly);
  assert.equal(normalizeListText('보호자를\u3000불러\u00a0주실래요\t'), '보호자를 불러 주실래요');
  assert.equal(normalizeListText('\u001c물\u0085주세요'), '물 주세요', '서버(Python)가 공백으로 보는 글자');
  const nfd = '화장실에 가고 싶어요 부탁해요'.normalize('NFD');
  assert.ok([...nfd].length > 30);
  assert.equal(quickDisplayProblem(nfd), null, 'NFD 한글도 NFC로 센다(서버와 같음)');
  assert.equal(quickTitleProblem('재활 병동 이층 오른쪽 방 목록'.normalize('NFD')), null);
  assert.equal(normalizeListDisplay('물\u00a0주세요'), '물 주세요');
  assert.equal(normalizeListDisplay('  물  \u2003주세요 '), '물 주세요');
  assert.equal(quickDisplayProblem('물\u200b주세요'), LISTS_TEXT.displayControl, '서식 문자(Cf)도 서버처럼 막는다');
  assert.equal(quickDisplayProblem('물\n주세요'), LISTS_TEXT.displayControl);
});

test('조사: 받침 있는 이름은 을, 없는 이름은 를, 한글로 끝나지 않으면 을(를)', () => {
  assert.equal(objectParticle('재활 병동'), '을');
  assert.equal(objectParticle('테스트'), '를');
  assert.equal(objectParticle('rehab'), '을(를)');
  assert.equal(objectParticle(''), '을(를)');
  assert.equal(LISTS_TEXT.deleteConfirm('재활 병동').startsWith('목록 “재활 병동”을 지울까요?'), true);
  assert.equal(LISTS_TEXT.deleted('테스트'), '목록 “테스트”를 지웠어요');
  assert.equal(LISTS_TEXT.deletedInUse('재활 병동'), '목록 “재활 병동”을 지웠어요. 병실 화면은 이제 기본 목록을 써요');
  assert.match(LISTS_TEXT.exampleLoaded('재활 병동'), /^예시 “재활 병동”을 새 목록으로 불러왔어요/);
  assert.equal(LISTS_TEXT.cannotUse('테스트', '내 목록을 찾을 수 없어요'), '내 목록 ‘테스트’를 쓸 수 없어요: 내 목록을 찾을 수 없어요');
  assert.equal(LISTS_TEXT.cannotUse('재활 병동', 'x'), '내 목록 ‘재활 병동’을 쓸 수 없어요: x');
  const ward = src('../src/pages/WardPage.jsx');
  assert.match(ward, /LISTS_TEXT\.cannotUse\(listStatus\.title, listStatus\.message\)/, '병실 화면 오류 띠도 같은 조사 규칙');
});

test('다시 찍을 문장 안내: 고친 문장 때문에 등록이 덜 찼으면 개수, 다 찼거나 한 번도 안 찍었으면 없음', () => {
  assert.equal(reenrollSummary(null), null);
  assert.equal(reenrollSummary({ profile: 'p01', count: 10, total: 10, status: 'ok' }), null);
  assert.equal(reenrollSummary({ profile: 'p01', count: 0, total: 10, status: 'missing' }), null);
  const part = reenrollSummary({ profile: 'p01', count: 9, total: 10, status: 'incomplete' });
  assert.equal(part.kind, 'partial');
  assert.equal(part.missing, 1);
  assert.match(part.text, /1개/);
  assert.match(part.text, /9\/10/);
  assert.equal(reenrollSummary({ profile: 'p01', count: 10, total: 10, status: 'stale' }).kind, 'stale');
  const page = src('../src/pages/ListsPage.jsx');
  assert.match(page, /reenrollSummary\(parseProfileInfo\(data, index, profile\)\)/, '목록 화면이 저장한 목록의 등록 상태를 읽는다');
  assert.match(page, /saveWardList\(id, requestBody\(st\), saveCondition\(st\)\)/, '저장은 조건부');
  assert.match(page, /lists\.status !== 'ready' \|\| lists\.items\.length >= MAX_LISTS/, '목록들을 못 불러오면 새 목록·예시를 막는다');
});

test('설정 공유: 저장소의 명령 목록·등록 이름만 다시 읽고, 저장된 값이 없으면 바꾸지 않는다', () => {
  const data = new Map();
  const store = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v) };
  assert.equal(readStoredSettings(store), null, '저장된 값 없음 → null(기본값으로 덮지 않음)');
  assert.equal(readStoredSettings(null), null);
  data.set(SETTINGS_STORAGE_KEY, '{broken');
  assert.equal(readStoredSettings(store), null);
  data.set(SETTINGS_STORAGE_KEY, JSON.stringify({ room: '412', commandList: 'rehab', enrollProfileByList: { rehab: 'p01' } }));
  const stored = readStoredSettings(store);
  assert.equal(stored.commandList, 'rehab');
  const current = normalizeSettings({ room: '305', tts: true });
  assert.deepEqual(sharedSettingsPatch(current, stored), { commandList: 'rehab', enrollProfileByList: { rehab: 'p01' } },
    '병실 번호·음성 같은 이 창의 설정은 가져오지 않는다');
  assert.equal(sharedSettingsPatch(normalizeSettings({ commandList: 'rehab', enrollProfileByList: { rehab: 'p01' } }), stored), null);
  assert.equal(sharedSettingsPatch(current, null), null);
  const page = src('../src/pages/WardPage.jsx');
  assert.match(page, /addEventListener\('storage', onStorage\)/, '병실 화면이 다른 탭의 설정 저장을 듣는다');
  assert.match(page, /saveSettings\(storage, \{ \.\.\.machineRef\.current\.settings, \.\.\.shared \}\)/, '저장할 때 다른 탭 값을 지키다');
});

test('등록 화면: 내 목록 판이 다른 저장 결과는 저장하지 않은 것으로 보이고, 서버의 목록 바뀜 거절을 알아본다', () => {
  const results = (list) => [{
    text: 'x', phrase: { id: 'U03', set: 'ward', accepted: true, ...(list ? { list } : {}) },
    enroll: { mode: 'store', profile: 'p01', command: 'U03', stored: true, text_top: 'U03', text_top_matches: true, count: 1, complete: false },
  }];
  assert.equal(storeOutcome(results({ id: 'rehab', sha256: SHA }), 'U03', CUSTOM).kind, 'stored');
  for (const list of [{ id: 'rehab', sha256: SHA2 }, { id: 'night', sha256: SHA }, null]) {
    const out = storeOutcome(results(list), 'U03', CUSTOM);
    assert.equal(out.kind, 'failed', JSON.stringify(list));
    assert.equal(out.reason, STORE_LIST_CHANGED);
    assert.equal(out.message, ENROLL_TEXT.listChanged);
  }
  // 기본 목록 등록은 예전 그대로 (phrase.list를 보지 않는다).
  const w = [{ text: 'x', phrase: { id: 'W05' }, enroll: { mode: 'store', command: 'W05', stored: true, text_top: 'W05', text_top_matches: true } }];
  assert.equal(storeOutcome(w, 'W05', INDEX).kind, 'stored');
  // 상태도: 저장하지 않은 것으로 남는다(개수에 넣지 않음).
  let st = createEnrollState({ index: CUSTOM, profile: 'p01' });
  st = stepEnroll(st, { type: 'PROFILE_LOADED', data: { profile: 'p01', count: 0, status: 'missing', templates: {} } });
  st = stepEnroll(st, { type: 'SELECT', commandId: 'U03' });
  st = stepEnroll(stepEnroll(st, { type: 'RECORD_START' }), { type: 'RECORD_DONE' });
  const after = stepEnroll(st, { type: 'STORE_RESULT', results: results({ id: 'rehab', sha256: SHA2 }) });
  assert.equal(after.phase, 'ready');
  assert.equal(after.info.count, 0);
  assert.equal(after.last.reason, STORE_LIST_CHANGED);
  // 서버 거절(400 MSG_LIST_CHANGED)을 알아본다.
  assert.equal(isListChangedError({ response: { status: 400, data: { detail: LIST_CHANGED_MESSAGE } } }), true);
  assert.equal(isListChangedError({ response: { status: 400, data: { detail: '다른 문구' } } }), false);
  assert.equal(isListChangedError({ response: { status: 409, data: { detail: LIST_CHANGED_MESSAGE } } }), false);
  assert.equal(isListChangedError(null), false);
  const page = src('../src/pages/EnrollPage.jsx');
  assert.match(page, /commandListSha256: listId \? \(index\.customList\?\.sha256 \|\| ''\) : ''/, '등록 요청에 불러온 판을 보낸다');
  assert.match(page, /key=\{loaded\.index\.customList\?\.sha256 \|\| 'default'\}/, '판이 바뀌면 등록 화면을 새로 연다');
  assert.match(page, /addEventListener\('focus', onFocus\)/, '창으로 돌아오면 목록을 다시 확인한다');
});

test('녹화 옵션: 목록 판은 내 목록과 함께만 보내고, 서버의 목록 바뀜 거절은 listChanged로 돌려준다', async () => {
  assert.deepEqual(wardRecordOptions({ commandList: 'rehab', commandListSha256: SHA, enrollProfile: 'p01', enrollCommand: 'U03' }),
    { phraseSet: 'ward', commandList: 'rehab', commandListSha256: SHA, enrollProfile: 'p01', enrollCommand: 'U03' });
  assert.deepEqual(wardRecordOptions({ commandListSha256: SHA }), { phraseSet: 'ward' }, '목록 없이 판만은 보내지 않는다');
  assert.deepEqual(wardRecordOptions({ commandList: 'rehab', commandListSha256: '' }), { phraseSet: 'ward', commandList: 'rehab' });
  assert.equal(commandListProblem('ward', 'rehab', 'U03', SHA), null);
  assert.equal(commandListProblem('ward', 'rehab', 'U03', 'A'.repeat(64)), '목록 지문 형식이 올바르지 않습니다.');
  assert.equal(commandListProblem('ward', '', '', SHA), '목록 지문은 내 목록과 함께 보내야 합니다.');
  assert.equal(commandListProblem('ward', '', 'W05', undefined), null, '기본 목록 등록은 예전 그대로');
  const blob = new Blob([new Uint8Array(4096)], { type: 'video/webm' });
  let seen = null;
  const rejected = await analyzeWardRecording({
    blob, mime: 'video/webm', durationMs: 5000, enrollProfile: 'p01', enrollCommand: 'U03', commandList: 'rehab', commandListSha256: SHA,
    deps: {
      upload: async (_b, _m, _p, opts) => {
        seen = opts;
        const err = new Error('400');
        err.response = { status: 400, data: { detail: LIST_CHANGED_MESSAGE } };
        throw err;
      },
    },
  });
  assert.equal(seen.commandListSha256, SHA);
  assert.deepEqual(rejected, { kind: 'error', message: LIST_CHANGED_MESSAGE, listChanged: true });
  const other = await analyzeWardRecording({
    blob, mime: 'video/webm', durationMs: 5000, commandList: 'rehab',
    deps: { upload: async () => { const err = new Error('400'); err.response = { status: 400, data: { detail: '내 목록을 찾을 수 없어요.' } }; throw err; } },
  });
  assert.deepEqual(other, { kind: 'error', message: '내 목록을 찾을 수 없어요.' }, '다른 거절에는 listChanged가 없다');
});

// ---------------------------------------------------------------------------
// 설정
// ---------------------------------------------------------------------------

test('설정: 명령 목록(규칙 밖이면 기본 목록), 목록별 등록 이름(기본 목록은 예전 enrollProfile)', () => {
  assert.equal(DEFAULT_SETTINGS.commandList, '');
  assert.deepEqual(normalizeSettings(null).enrollProfileByList, {});
  assert.equal(normalizeCommandList(' rehab '), 'rehab');
  for (const bad of ['../x', '재활', 'a b', 'x'.repeat(33), 3, null]) assert.equal(normalizeCommandList(bad), '', String(bad));
  assert.deepEqual(normalizeEnrollProfileByList({ rehab: 'bed305', '../x': 'p', night: '가나', ok_1: ' p1 ' }), { rehab: 'bed305', ok_1: 'p1' });
  assert.deepEqual(normalizeEnrollProfileByList(['x']), {});
  const s = normalizeSettings({ enrollProfile: 'bed305', commandList: 'rehab', enrollProfileByList: { rehab: 'p01' } });
  assert.equal(enrollProfileFor(s, ''), 'bed305', '기본 목록은 예전 키');
  assert.equal(enrollProfileFor(s, 'rehab'), 'p01');
  assert.equal(enrollProfileFor(s, 'night'), '');
  assert.equal(enrollProfileFor(s, '../x'), '');
  assert.deepEqual(enrollProfilePatch(s, '', 'new1'), { enrollProfile: 'new1' });
  assert.deepEqual(enrollProfilePatch(s, 'night', 'p02'), { enrollProfileByList: { rehab: 'p01', night: 'p02' } });
  assert.deepEqual(enrollProfilePatch(s, 'rehab', ''), { enrollProfileByList: {} }, "''이면 그 목록 등록을 지운다");
  const data = new Map();
  const store = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v) };
  assert.equal(saveSettings(store, { ...s, commandList: 'BAD/ID' }), true);
  assert.equal(loadSettings(store).commandList, '', '저장된 이상한 값은 기본 목록');
  saveSettings(store, s);
  assert.deepEqual(loadSettings(store), { ...DEFAULT_SETTINGS, enrollProfile: 'bed305', commandList: 'rehab', enrollProfileByList: { rehab: 'p01' } });
});

// ---------------------------------------------------------------------------
// API 호출 모양
// ---------------------------------------------------------------------------

test('녹화 옵션: 내 목록이면 command_list, 기본 목록이면 보내지 않는다(요청이 예전과 같다)', () => {
  assert.deepEqual(wardRecordOptions({ enrollProfile: 'bed305' }), { phraseSet: 'ward', enrollProfile: 'bed305' });
  assert.deepEqual(wardRecordOptions({ commandList: '' }), { phraseSet: 'ward' });
  assert.deepEqual(wardRecordOptions({ commandList: 'rehab', enrollProfile: 'p01', enrollCommand: 'U03' }),
    { phraseSet: 'ward', commandList: 'rehab', enrollProfile: 'p01', enrollCommand: 'U03' });
});

test('API: command_list 필드 규칙, 등록 이름공간 ?list=, 내 목록 CRUD 주소, 이상한 ID는 보내기 전에 거절', async () => {
  const {
    apiClient, ENROLL_COMMAND_RE, checkWardList, commandListProblem, deleteWardList, getWardList, getWardLists, getWardProfile,
    getWardProfiles, saveWardList, uploadRecording, wardProfileApi,
  } = await import('../src/api/client.js');
  const calls = [];
  const real = { post: apiClient.post, get: apiClient.get, put: apiClient.put, delete: apiClient.delete };
  apiClient.post = async (url, body) => {
    calls.push(['post', url, body instanceof FormData ? [...body.entries()].map(([k, v]) => (k === 'file' ? 'file' : `${k}=${v}`)) : body]);
    return { data: url === '/api/ward/lists/check' ? { errors: [], warnings: [], normalized: {} } : { job_id: 'j', session_id: 's', status: 'processing' } };
  };
  apiClient.get = async (url) => { calls.push(['get', url]); return { data: { lists: [], profiles: [] } }; };
  apiClient.put = async (url, body) => { calls.push(['put', url, body]); return { data: { id: 'rehab' } }; };
  apiClient.delete = async (url) => { calls.push(['delete', url]); return { data: { deleted: true } }; };
  try {
    assert.ok(ENROLL_COMMAND_RE.test('U01') && ENROLL_COMMAND_RE.test('W20') && !ENROLL_COMMAND_RE.test('A01'));
    const blob = new Blob([new Uint8Array([1, 2, 3])], { type: 'video/webm' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', commandList: 'rehab' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', commandList: 'rehab', enrollProfile: 'p01', enrollCommand: 'U02' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', commandList: '' });
    await uploadRecording(blob, 'video/webm', undefined, { phraseSet: 'ward', enrollProfile: 'bed305', enrollCommand: 'W05' });
    assert.deepEqual(calls.map((c) => c[2]), [
      ['file', 'phrase_set=ward', 'command_list=rehab'],
      ['file', 'phrase_set=ward', 'command_list=rehab', 'enroll_profile=p01', 'enroll_command=U02'],
      ['file', 'phrase_set=ward'],
      ['file', 'phrase_set=ward', 'enroll_profile=bed305', 'enroll_command=W05'],
    ]);
    const bad = [
      [{ phraseSet: 'sentences', commandList: 'rehab' }, /병실 명령 녹화에서만/],
      [{ commandList: 'rehab' }, /병실 명령 녹화에서만/],
      [{ phraseSet: 'ward', commandList: '../x' }, /내 목록 이름/],
      [{ phraseSet: 'ward', enrollProfile: 'p01', enrollCommand: 'U02' }, /내 목록과 함께/],
      [{ phraseSet: 'ward', commandList: 'rehab', enrollProfile: 'p01', enrollCommand: 'W05' }, /기본 목록 명령/],
    ];
    for (const [options, re] of bad) {
      await assert.rejects(uploadRecording(blob, 'video/webm', undefined, options), (e) => e.isClientCheck === true && re.test(e.message), JSON.stringify(options));
    }
    assert.equal(calls.length, 4, '거절한 요청은 보내지 않는다');
    assert.equal(commandListProblem('ward', '', ''), null);
    assert.equal(commandListProblem(undefined, undefined, undefined), null);

    calls.length = 0;
    await getWardProfiles();
    await getWardProfiles('rehab');
    await getWardProfile('p01', 'rehab');
    const api = wardProfileApi('rehab');
    await api.getWardProfile('p01');
    await api.deleteWardProfile('p01');
    await api.getWardProfiles();
    await wardProfileApi('').getWardProfile('bed305');
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), [
      'get /api/ward/profiles', 'get /api/ward/profiles?list=rehab', 'get /api/ward/profiles/p01?list=rehab',
      'get /api/ward/profiles/p01?list=rehab', 'delete /api/ward/profiles/p01?list=rehab', 'get /api/ward/profiles?list=rehab',
      'get /api/ward/profiles/bed305',
    ]);
    await assert.rejects(wardProfileApi('../x').getWardProfile('p01'), (e) => e.isClientCheck === true);
    await assert.rejects(getWardProfiles('a b'), (e) => e.isClientCheck === true);

    calls.length = 0;
    await getWardLists();
    await getWardList('rehab');
    const body = { title: '재활 병동', commands: [] };
    await saveWardList('rehab', body);
    await checkWardList(body);
    await deleteWardList('rehab');
    assert.deepEqual(calls.map((c) => `${c[0]} ${c[1]}`), [
      'get /api/ward/lists', 'get /api/ward/lists/rehab', 'put /api/ward/lists/rehab', 'post /api/ward/lists/check', 'delete /api/ward/lists/rehab',
    ]);
    assert.equal(calls[2][2], body);
    for (const badId of ['../etc', 'a/b', '', 'x'.repeat(33), null]) {
      await assert.rejects(getWardList(badId), (e) => e.isClientCheck === true, String(badId));
      await assert.rejects(saveWardList(badId, body), (e) => e.isClientCheck === true, String(badId));
      await assert.rejects(deleteWardList(badId), (e) => e.isClientCheck === true, String(badId));
    }
    assert.equal(calls.length, 5, '이상한 ID는 서버로 보내지 않는다');
  } finally {
    Object.assign(apiClient, real);
  }
});

// ---------------------------------------------------------------------------
// 화면 연결 (소스 계약)
// ---------------------------------------------------------------------------

test('화면 연결: #lists 경로, 병실 화면은 내 목록을 불러와 막고(기본 목록으로 바꾸지 않음) 연습 모드는 기본 목록', () => {
  const app = src('../src/App.jsx');
  assert.match(app, /'#lists': 'lists'/);
  assert.match(app, /<ListsPage \/>/);
  const ward = src('../src/pages/WardPage.jsx');
  assert.match(ward, /const activeList = practice \? '' : machine\.settings\.commandList;/, '연습 모드는 늘 기본 목록');
  assert.match(ward, /lipBlock: \{ reason: 'error', message \}/, '불러오기 실패 → 입 읽기 막기');
  assert.match(ward, /emptyListIndex\(activeList/);
  assert.match(ward, /commandList,\s*isCancelled: stale/, '녹화 요청에 목록 ID');
  assert.match(ward, /listBannerText\(/);
  assert.match(ward, /practiceResults\(scenarioId, INDEX/, '연습 결과는 기본 목록');
  const enroll = src('../src/pages/EnrollPage.jsx');
  assert.match(enroll, /enrollCommand: s\.recordingId, commandList: listId/);
  assert.match(enroll, /practice \? '' : loadSettings\(storage\)\.commandList/);
  assert.doesNotMatch(enroll, /명령 20개/);
  const nurse = src('../src/pages/NursePage.jsx');
  assert.match(nurse, /cardListNote\(card\)/);
  const lists = src('../src/pages/ListsPage.jsx');
  assert.match(lists, /checkWardList\(requestBody\(editorRef\.current\)\)/);
  assert.match(lists, /import\.meta\.glob\('\.\.\/ward\/custom_list_examples\.json'/, '예시 파일이 없어도 빌드된다');
  assert.match(lists, /globalThis\.confirm\?\.\(LISTS_TEXT\.deleteConfirm\(title\)\)/, '지우기 전에 확인');
  assert.doesNotMatch(lists, /localStorage\.setItem|sessionStorage/, '설정은 wardSettings로만 저장');
  const panel = src('../src/ward/WardSettingsPanel.jsx');
  assert.match(panel, /onChange\(enrollProfilePatch\(settings, activeList, profile\)\)/);
  assert.match(panel, /내 목록 관리/);
});
