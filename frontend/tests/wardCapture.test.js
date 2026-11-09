// wardCapture.test.js — 병실 명령 촬영 묶음(ward_tune·ward_eval)의 목록·순서·파일명·녹화 규칙 검사 (node --test).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  ALL_SESSIONS, SESSIONS, WARD_SESSIONS, WARD_START_COOLDOWN_MS, buildPlan, buildSessionPlan, buildWardPlan, clipBaseName,
  clipMeta, conditionInfo, findSession, isWardSession, kindLabel, nextUnsaved, recordToggleAction, recordingRule, repeatsFor,
  resumePoint, sessionFromMetas, wardPromptSet, zipFolderName,
} from '../src/capture/plan.js';

const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8'));
const prompts = readJson('../src/capture/prompts.json');
const wardPrompts = readJson('../src/capture/ward_prompts.json');
const commands = readJson('../src/ward/commands.json');
const ward = wardPromptSet(wardPrompts, commands);
const tune = findSession('ward_tune');
const evalSession = findSession('ward_eval');
const HANGUL = /^[가-힣]+( [가-힣]+)*$/;

test('ward_prompts: 명령 20개는 commands.json의 id·text와 같고, 촬영 화면은 commands.json을 쓴다', () => {
  assert.equal(wardPrompts.version, 'ward-prompts-v1');
  assert.equal(wardPrompts.commandsVersion, commands.version);
  const pairs = (list) => list.map(({ id, text }) => [id, text]);
  assert.deepEqual(pairs(wardPrompts.fixed), pairs(commands.fixed), 'ward_prompts.json의 fixed가 commands.json과 다르다');
  assert.deepEqual(ward.fixed, commands.fixed.map(({ id, text }) => ({ id, text })));
  assert.equal(ward.commandsVersion, 'ward-commands-v1');
  assert.equal(ward.fixed.length, 20);
});

test('ward_prompts: 함정 조정용 10·평가용 10은 서로 겹치지 않고, 짧은 말 3·말하지 않기 10·기댄 자세 명령 10', () => {
  const { tune: dTune, eval: dEval } = wardPrompts.decoys;
  assert.deepEqual(dTune.map((p) => p.id), ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10']);
  assert.deepEqual(dEval.map((p) => p.id), ['D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20']);
  const squash = (t) => t.replace(/ /g, '');
  const tuneTexts = new Set(dTune.map((p) => squash(p.text)));
  for (const p of dEval) assert.ok(!tuneTexts.has(squash(p.text)), `조정용과 같은 평가용 함정: ${p.id}`);
  assert.deepEqual(wardPrompts.short.map((p) => p.id), ['E01', 'E02', 'E03']);
  assert.equal(wardPrompts.nonspeech.length, 10);
  assert.ok(wardPrompts.nonspeech.every((p) => p.text.startsWith('(말하지 않기)')), '말하지 않는 동작은 지시문');
  assert.deepEqual(wardPrompts.reclined, ['W02', 'W03', 'W04', 'W05', 'W06', 'W10', 'W11', 'W13', 'W17', 'W18']);
  const commandIds = new Set(commands.fixed.map((p) => p.id));
  for (const id of wardPrompts.reclined) assert.ok(commandIds.has(id), id);
  const spoken = [...dTune, ...dEval, ...wardPrompts.short];
  for (const p of spoken) assert.match(p.text, HANGUL, p.id);
  const commandTexts = new Set(commands.fixed.map((p) => squash(p.text)));
  for (const p of spoken) assert.ok(!commandTexts.has(squash(p.text)), `명령과 같은 문장: ${p.id}`);
  const ids = [...commands.fixed, ...dTune, ...dEval, ...wardPrompts.short, ...wardPrompts.nonspeech].map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length, 'ID가 겹치지 않는다');
});

test('WARD_SESSIONS: 기존 문장 묶음은 그대로, 병실 묶음은 ward_prompts.json의 sessions 구성과 같다', () => {
  assert.deepEqual(SESSIONS.map((x) => [x.id, x.repeats]), [['pilot', 2], ['eval', 3]]);
  assert.deepEqual(ALL_SESSIONS.map((x) => x.id), ['pilot', 'eval', 'ward_tune', 'ward_eval']);
  for (const s of WARD_SESSIONS) {
    const spec = wardPrompts.sessions[s.id];
    assert.ok(spec, s.id);
    assert.deepEqual(s.repeats, spec.repeats, s.id);
    assert.equal(s.decoys, spec.decoys, s.id);
    assert.equal(s.reclined, spec.reclined, s.id);
    assert.equal(s.maxSeconds, spec.maxSeconds, s.id);
    assert.equal(buildWardPlan(ward, s).length, spec.clips, s.id);
  }
  assert.equal(findSession('없는묶음').id, 'pilot');
  assert.ok(isWardSession('ward_tune') && isWardSession(evalSession) && !isWardSession('eval') && !isWardSession(undefined));
});

test('ward_tune: 명령 소리 1회(등록) + 무음 2회 + 함정 10 + 짧은 말 3 + 말하지 않기 10 + 기댄 자세 10 = 93클립', () => {
  const plan = buildSessionPlan(tune, prompts, ward);
  assert.equal(plan.length, 93);
  const count = (f) => plan.filter(f).length;
  assert.equal(count((x) => x.kind === 'fixed' && x.cond === 'voiced'), 20);
  assert.equal(count((x) => x.kind === 'fixed' && x.cond === 'silent'), 40);
  assert.equal(count((x) => x.kind === 'fixed' && x.cond === 'silent' && x.rep === 2), 20);
  assert.equal(count((x) => x.kind === 'fixed' && x.cond === 'reclined'), 10);
  assert.equal(count((x) => x.kind === 'decoy' && x.cond === 'silent'), 10);
  assert.equal(count((x) => x.kind === 'short' && x.cond === 'silent'), 3);
  assert.equal(count((x) => x.kind === 'nonspeech' && x.cond === 'none'), 10);
  assert.deepEqual(plan.filter((x) => x.kind === 'decoy').map((x) => x.sentenceId), wardPrompts.decoys.tune.map((p) => p.id));
  // 순서: 등록(소리) 1~20 → 무음 21~60 → 함정 61~70 → 짧은 말 71~73 → 말하지 않기 74~83 → 기댄 자세 84~93
  assert.deepEqual([plan[0].cond, plan[19].cond, plan[20].cond, plan[59].cond], ['voiced', 'voiced', 'silent', 'silent']);
  assert.deepEqual([plan[60].kind, plan[70].kind, plan[73].kind, plan[83].kind], ['decoy', 'short', 'nonspeech', 'fixed']);
  assert.deepEqual(plan.slice(83).map((x) => x.sentenceId), wardPrompts.reclined);
  assert.ok(plan.slice(83).every((x) => x.posture === 'reclined' && x.cond === 'reclined' && x.rep === 1));
  assert.ok(plan.slice(0, 83).every((x) => x.posture === 'upright'));
  const w05 = plan.find((x) => x.cond === 'reclined' && x.sentenceId === 'W05');
  assert.equal(w05.text, commands.fixed.find((p) => p.id === 'W05').text, '기댄 자세 문장도 commands.json 문장');
  const names = plan.map((x) => clipBaseName('P01', x));
  assert.equal(new Set(names).size, 93, '파일명은 겹치지 않아야 한다');
  for (const n of ['P01_voiced_W01_1', 'P01_silent_W20_2', 'P01_silent_D01_1', 'P01_silent_E03_1', 'P01_none_X10_1', 'P01_reclined_W02_1']) {
    assert.ok(names.includes(n), n);
  }
});

test('ward_eval: 명령 소리 1회 + 무음 3회 + 평가용 함정 10 + 짧은 말 3 + 말하지 않기 10 = 103클립, 기댄 자세 없음', () => {
  const plan = buildSessionPlan(evalSession, prompts, ward);
  assert.equal(plan.length, 103);
  assert.equal(plan.filter((x) => x.kind === 'fixed' && x.cond === 'voiced').length, 20);
  assert.equal(plan.filter((x) => x.kind === 'fixed' && x.cond === 'silent').length, 60);
  assert.equal(plan.filter((x) => x.kind === 'fixed' && x.rep === 3).length, 20);
  assert.equal(plan.filter((x) => x.cond === 'reclined' || x.posture === 'reclined').length, 0);
  assert.deepEqual(plan.filter((x) => x.kind === 'decoy').map((x) => x.sentenceId), wardPrompts.decoys.eval.map((p) => p.id));
  assert.ok(!plan.some((x) => wardPrompts.decoys.tune.some((d) => d.id === x.sentenceId)), '조정용 함정이 평가용에 섞이지 않는다');
  assert.equal(plan.filter((x) => x.kind === 'short').length, 3);
  assert.equal(plan.filter((x) => x.kind === 'nonspeech').length, 10);
  assert.equal(new Set(plan.map((x) => clipBaseName('P02', x))).size, 103);
});

test('buildPlan: 조건별 반복 횟수(객체)와 숫자 하나가 같은 순서를 만들고, 파일럿·평가용 순서는 그대로다', () => {
  assert.deepEqual(buildPlan(prompts, { voiced: 2, silent: 2 }), buildPlan(prompts, 2));
  assert.deepEqual(buildPlan(prompts, { voiced: 3, silent: 3 }), buildPlan(prompts, 3));
  assert.deepEqual(buildSessionPlan(findSession('pilot'), prompts, ward), buildPlan(prompts, 2));
  assert.deepEqual(buildSessionPlan(findSession('eval'), prompts, ward), buildPlan(prompts, 3));
  const mixed = buildPlan(prompts, { voiced: 1, silent: 2 });
  assert.equal(mixed.length, 20 + 40 + 10 + 10);
  assert.equal(mixed.filter((x) => x.kind === 'fixed' && x.cond === 'voiced').length, 20);
  assert.ok(!buildPlan(prompts, 2).some((x) => 'posture' in x), '문장 촬영 항목에는 자세 필드를 붙이지 않는다');
  assert.equal(buildPlan({ fixed: prompts.fixed }, { silent: 1 }).length, 20, '자유·무발화가 없는 목록도 받는다');
  assert.equal(repeatsFor(2, 'voiced'), 2);
  assert.equal(repeatsFor({ silent: 3 }, 'voiced'), 0);
  assert.equal(repeatsFor({ silent: 3 }, 'silent'), 3);
  assert.equal(repeatsFor({ silent: -1 }, 'silent'), 0);
});

test('buildWardPlan: 없는 함정 목록·명령 ID는 조용히 넘기지 않는다', () => {
  assert.throws(() => buildWardPlan(ward, { ...tune, decoys: 'test' }), /함정 문장 목록 'test'/);
  assert.throws(() => buildWardPlan({ ...ward, reclined: ['W99'] }, tune), /W99/);
});

test('녹화 규칙: 문장 촬영은 8초·손으로 멈춤, 병실 명령 촬영은 5초 자동 정지만', () => {
  assert.deepEqual(recordingRule(findSession('pilot')), { maxSeconds: 8, manualStop: true, startCooldownMs: 0 });
  assert.deepEqual(recordingRule(findSession('eval')), { maxSeconds: 8, manualStop: true, startCooldownMs: 0 });
  assert.deepEqual(recordingRule(tune), { maxSeconds: 5, manualStop: false, startCooldownMs: 1000 });
  assert.deepEqual(recordingRule(evalSession), { maxSeconds: 5, manualStop: false, startCooldownMs: 1000 });
  assert.equal(WARD_START_COOLDOWN_MS, 1000);
  const base = { status: 'recording', done: false, saving: false, blocked: false };
  assert.equal(recordToggleAction({ ...base, manualStop: true }), 'stop');
  assert.equal(recordToggleAction({ ...base, manualStop: false }), null, '병실 명령 녹화는 중간에 멈추지 않는다');
  assert.equal(recordToggleAction({ ...base, status: 'ready', manualStop: false }), 'start');
  assert.equal(recordToggleAction({ ...base, status: 'ready', blocked: true, manualStop: false }), null);
  assert.equal(recordToggleAction({ ...base, status: 'ready', saving: true, manualStop: true }), null);
  assert.equal(recordToggleAction({ ...base, status: 'ready', done: true, manualStop: true }), null);
  assert.equal(recordToggleAction({ ...base, status: 'opening', manualStop: true }), null);
});

test('저장 직후 시작 막기: 병실 명령 클립 저장 뒤 잠깐은 스페이스바·버튼이 다음 녹화를 시작하지 않는다', () => {
  const ready = { status: 'ready', done: false, saving: false, blocked: false, manualStop: false };
  // 5초 자동 정지 직후 습관처럼 누른 스페이스바: 저장 중에도, 저장이 끝난 직후(cooldown)에도 시작하지 않는다.
  assert.equal(recordToggleAction({ ...ready, saving: true, cooldown: true }), null);
  assert.equal(recordToggleAction({ ...ready, cooldown: true }), null);
  assert.equal(recordToggleAction({ ...ready, cooldown: false }), 'start', '막는 시간이 지나면 시작한다');
  // cooldown은 시작만 막는다(문장 촬영의 정지, cooldown을 넘기지 않는 기존 호출은 그대로).
  assert.equal(recordToggleAction({ ...ready, status: 'recording', manualStop: true, cooldown: true }), 'stop');
  assert.equal(recordToggleAction({ status: 'ready', done: false, saving: false, blocked: false, manualStop: true }), 'start');
});

const RESULT_META = { extension: 'webm', mime: 'video/webm', bytes: 123456, durationMs: 4980, frameCallbacks: 148,
  estimatedFps: 29.7, maxGapMs: 41, trackSettings: { width: 1280, height: 720, frameRate: 30 } };
const metaFor = (session, item, extra = {}) => clipMeta({ session, item, participant: 'P01', resultMeta: RESULT_META,
  spoken: '', prompts, ward, ...extra });

test('clipMeta: 파일럿·평가용 클립 기록은 예전 모양 그대로 (필드·순서·값, 자세·명령 버전 없음)', () => {
  for (const [sid, reps] of [['pilot', 2], ['eval', 3]]) {
    const session = findSession(sid);
    const plan = buildSessionPlan(session, prompts, ward);
    const item = plan.find((x) => x.kind === 'fixed' && x.cond === 'silent');
    // 예전 CapturePage.persist가 만들던 기록 (MAX_SECONDS 8, prompts.json 버전)
    const legacy = { ...RESULT_META, participant: 'P01', condition: item.cond, sentenceId: item.sentenceId, text: item.text,
      kind: item.kind, rep: item.rep, maxSeconds: 8, file: `${clipBaseName('P01', item)}.webm`, session: sid, repeats: reps,
      promptsVersion: prompts.version };
    const meta = metaFor(session, item);
    assert.deepEqual(meta, legacy, sid);
    assert.deepEqual(Object.keys(meta), Object.keys(legacy), `${sid} 필드 순서`);
    assert.ok(!('posture' in meta) && !('commandsVersion' in meta), `${sid}에 병실 필드가 새지 않는다`);
    const free = plan.find((x) => x.kind === 'free');
    const freeMeta = metaFor(session, free, { spoken: '  오늘   점심은  비빔밥  ' });
    assert.deepEqual(freeMeta, { ...legacy, condition: free.cond, sentenceId: free.sentenceId, text: free.text, kind: 'free',
      rep: 1, file: `${clipBaseName('P01', free)}.webm`, spoken: '오늘 점심은 비빔밥' });
    const non = plan.find((x) => x.kind === 'nonspeech');
    assert.ok(!('spoken' in metaFor(session, non)), '자유 문장이 아니면 spoken 없음');
  }
});

test('clipMeta: 병실 명령 클립 기록은 ward_prompts 버전·명령 버전·자세·5초를 더한다', () => {
  for (const session of [tune, evalSession]) {
    const plan = buildSessionPlan(session, prompts, ward);
    for (const item of [plan[0], plan.find((x) => x.kind === 'decoy'), plan.find((x) => x.kind === 'nonspeech')]) {
      const meta = metaFor(session, item);
      assert.equal(meta.session, session.id);
      assert.equal(meta.promptsVersion, 'ward-prompts-v1');
      assert.equal(meta.commandsVersion, 'ward-commands-v1');
      assert.equal(meta.maxSeconds, 5);
      assert.equal(meta.posture, 'upright');
      assert.equal(meta.kind, item.kind);
      assert.deepEqual(meta.repeats, session.repeats);
      assert.equal(meta.file, `${clipBaseName('P01', item)}.webm`);
      assert.ok(!('spoken' in meta));
    }
  }
  const reclined = buildSessionPlan(tune, prompts, ward).find((x) => x.cond === 'reclined');
  const meta = metaFor(tune, reclined);
  assert.equal(meta.posture, 'reclined');
  assert.equal(meta.condition, 'reclined');
  assert.equal(meta.file, `P01_reclined_${reclined.sentenceId}_1.webm`);
  assert.deepEqual(Object.keys(meta).slice(-2), ['commandsVersion', 'posture']);
});

test('zipFolderName: 문장 촬영은 예전 이름, 병실 명령 묶음은 묶음 ID를 넣어 다른 폴더로 풀린다', () => {
  assert.equal(zipFolderName('P01', findSession('pilot')), 'P01_웹캠촬영');
  assert.equal(zipFolderName('P02', findSession('eval')), 'P02_웹캠촬영');
  assert.equal(zipFolderName('P01', tune), 'P01_ward_tune_웹캠촬영');
  assert.equal(zipFolderName('P02', evalSession), 'P02_ward_eval_웹캠촬영');
  assert.equal(zipFolderName('', findSession('pilot')), 'clips_웹캠촬영');
});

test('sessionFromMetas: 병실 명령 폴더에 문장 촬영(파일럿·평가·v1) 클립이 섞이면 mixed', () => {
  assert.deepEqual(sessionFromMetas([{ kind: 'fixed', session: 'ward_tune' }, { kind: 'decoy', session: 'ward_tune' }]),
    { session: 'ward_tune', mixed: false, clips: 2 });
  assert.equal(sessionFromMetas([{ kind: 'fixed', session: 'ward_tune' }, { kind: 'fixed', session: 'pilot' }]).mixed, true);
  assert.equal(sessionFromMetas([{ kind: 'fixed', session: 'ward_eval' }, { kind: 'fixed', session: 'eval' }]).mixed, true);
  assert.equal(sessionFromMetas([{ kind: 'fixed', session: 'ward_tune' }, { kind: 'fixed' }]).mixed, true, 'v1(묶음 없음)은 파일럿');
  assert.equal(sessionFromMetas([{ kind: 'fixed', session: 'ward_tune' }, { kind: 'short', session: 'ward_eval' }]).mixed, true);
});

test('resumePoint/nextUnsaved: 병실 명령 순서에서도 이어 찍기·다음 빈 순번이 맞다', () => {
  const plan = buildSessionPlan(tune, prompts, ward);
  const names = plan.slice(0, 61).map((x) => clipBaseName('P03', x));
  assert.deepEqual(resumePoint(plan, 'P03', names), { index: 61, saved: 61 });
  assert.equal(nextUnsaved(plan, 'P03', names, 5), 61, '앞 클립을 다시 찍은 뒤 저장된 클립을 덮어쓰지 않는다');
  const withReclined = [...names, clipBaseName('P03', plan[90])];
  assert.deepEqual(resumePoint(plan, 'P03', withReclined), { index: 61, saved: 62 });
  assert.equal(nextUnsaved(plan, 'P03', withReclined, 90), 91);
  const all = plan.map((x) => clipBaseName('P03', x));
  assert.deepEqual(resumePoint(plan, 'P03', all), { index: 93, saved: 93 });
  // 같은 명령의 무음(서 있는 자세)과 기댄 자세 클립은 이름이 달라 서로를 저장된 것으로 세지 않는다.
  assert.equal(resumePoint(plan, 'P03', ['P03_silent_W02_1']).saved, 1);
  assert.notEqual(clipBaseName('P03', plan[83]), 'P03_silent_W02_1');
});

test('조건·종류 표시: 기댄 자세 조건 안내, 병실 명령 종류 이름', () => {
  assert.equal(conditionInfo('reclined').id, 'reclined');
  assert.match(conditionInfo('reclined').hint, /기대어/);
  assert.equal(conditionInfo('silent').id, 'silent');
  assert.equal(conditionInfo('none'), null);
  assert.equal(kindLabel('fixed'), '지정 문장');
  assert.equal(kindLabel('fixed', true), '병실 명령');
  assert.equal(kindLabel('decoy', true), '목록 밖 문장');
  assert.equal(kindLabel('short', true), '짧은 말');
  assert.equal(kindLabel('nonspeech'), '무발화');
  assert.equal(kindLabel('free'), '자유 문장');
});
