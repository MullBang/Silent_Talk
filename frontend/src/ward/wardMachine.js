// wardMachine.js — 환자 화면(#ward) 상태도 순수 함수 (node --test로 검사, JSX·DOM·시계 없음).
// 입력 이벤트: 스위치 누름(PRESS, 시각 포함)·시간 경과(TICK)·녹화/분석 결과·간호사 확인. 출력: 새 상태 + 할 일(effects).
// 시각은 모두 호출하는 쪽이 넘기는 ms(Date.now 기준)다. 모든 이벤트는 먼저 시각 기반 처리(카운트다운 끝, 스캐닝 이동)를
// 따라잡은 뒤 처리한다. 그래서 TICK이 늦게 와도 누른 시각 기준으로 같은 결과가 나온다.
//
// effects (화면이 실행):
//   {type:'START_RECORDING', run, seconds}  녹화 시작 + 시작음 + 테두리 깜빡임 (카운트다운 없음)
//   {type:'SEND_CARD', card} / {type:'UPDATE_CARD', card}  간호사실로 카드 보내기·갱신
//   {type:'SPEAK', text}  음성 읽기(설정·밤 무음은 화면이 판단)
//   {type:'LOG', message}  예상하지 못한 오류 (상태는 바꾸지 않음)
//
// 명령 목록: 상태의 index가 지금 쓰는 목록이다(기본 목록 또는 내 목록, SET_INDEX로 바꾼다). 내 목록을 실제로 불러와 쓰는
// 동안 만든 카드에는 list {id, title, sha256}(목록 판)을 붙인다. 스위치 3번 긴급 호출 카드와 불러오는 중·실패 때의 카드에는
// 붙이지 않는다. 내 목록을 불러오는 중이거나 불러오지 못하면 lipBlock을 두어 입 읽기 녹화를 시작하지 않는다
// (기본 목록으로 몰래 바꾸지 않는다). 스위치 3번 긴급 호출은 그대로 된다.
// baseIndex는 기본 목록 색인이다. 목록을 바꾼 뒤 간호사가 기본 목록 카드(W07 자세·W14 등받이)를 확인해도 가상 침대가 움직이게 한다.

import {
  INITIAL_ROOM, TEMPERATURE_ACTIONS, TEMPERATURE_HINT, applyRoomAction, commandById, isCancelCommand, roomActionTiming,
} from './wardCommands.js';
import { NO_PHRASE_REASON_TEXT, classifyWardResults, decideWard } from './wardResult.js';
import { normalizeSettings } from './wardSettings.js';

/** 이 간격(ms)보다 짧게 이어진 누름은 하나로 친다 (마지막으로 받아들인 누름 기준). 키 반복(repeat)은 항상 무시. */
export const PRESS_MERGE_MS = 500;
/** 3초 안에 3번 누르면 입 읽기 없이 긴급 호출 (첫 누름 포함). */
export const TRIPLE_PRESS_COUNT = 3;
export const TRIPLE_PRESS_WINDOW_MS = 3000;
/** 스캐닝 한 칸 강조 시간과 최대 반복 바퀴 수. */
export const SCAN_STEP_MS = 2000;
export const SCAN_MAX_ROUNDS = 3;
/** "간호사가 확인했어요" 표시 시간. */
export const ACK_NOTICE_MS = 10000;
/** 녹화가 끝났다는 알림이 이 시간(녹화 길이 뒤)까지 없으면 녹화 실패로 본다. */
export const RECORD_GRACE_MS = 8000;
/** 분석 결과가 이 시간까지 없으면 분류 스캐닝으로 넘어간다 (화면 폴링 제한보다 길게). */
export const ANALYSIS_LIMIT_MS = 60000;
/**
 * [발화 없음]·검출 실패가 이 횟수만큼 이어지면 간호사실에도 '확인 필요(입 인식 안 됨)'.
 * 입을 본 분석 결과(지정 문장 결과·결과 없음)가 오면 0으로 돌아간다. 녹화 실패·분석 오류·분석 시간 초과는
 * 입이 보였다는 근거가 아니므로 연속을 끊지 않는다(그 사이 분류 스캐닝에서 고른 것도 끊지 않는다).
 */
export const NO_SPEECH_NURSE_STREAK = 2;
/** 환자 화면이 기억하는 카드 수 (간호사 확인·취소 요청 연결용). */
export const MAX_KEPT_CARDS = 50;

/** 카드 종류. emergency·emergency_call·emergency_possible은 간호사실에서 반복음이 울리는 경보 카드다. */
export const CARD_KINDS = Object.freeze(['request', 'emergency', 'emergency_call', 'emergency_possible', 'check']);
export const ALARM_KINDS = Object.freeze(['emergency', 'emergency_call', 'emergency_possible']);

export const WARD_TEXT = Object.freeze({
  welcome: '스위치를 누르고 말씀하세요',
  welcomeSub: (sec) => `스위치(아무 키·화면)를 한 번 누르면 ${sec}초 동안 녹화합니다 · 3초 안에 3번 = 긴급 호출`,
  recording: '말씀하세요',
  recordingStatus: (sec) => `● 말씀하세요 (${sec}초)`,
  analyzing: '분석 중이에요',
  analyzingSub: '잠시만 기다려 주세요',
  confirmSub: (sec) => `${sec}초 뒤 간호사실로 보냅니다 · 스위치 = 취소`,
  sent: '간호사실에 보냈어요',
  acked: '간호사가 확인했어요',
  cancelled: '보내지 않았어요',
  cancelledSub: '다시 누르면 새로 녹화합니다',
  emergencySub: '간호사실에 바로 알렸어요 · 이 화면에서는 취소할 수 없어요',
  emergencyCall: '긴급 호출(내용 미확인)',
  emergencyCallSub: '스위치 3번 — 간호사실에 긴급 호출을 보냈어요',
  emergencyPossible: '긴급일 수 있어 간호사실에 알렸어요',
  emergencyPossibleCard: '긴급 가능성, 확인 필요',
  checkCard: '확인 필요',
  noMouthCard: '확인 필요(입 인식 안 됨)',
  noSpeech: '입 움직임이 안 보였어요. 다시 눌러 주세요',
  noSpeechNurse: '간호사실에도 확인을 요청했어요',
  scanCandidates: '이 중에 있나요?',
  scanCategories: '분류를 고르세요',
  scanCommands: (label) => `${label} — 요청을 고르세요`,
  scanSub: '원하는 것이 강조될 때 스위치를 누르세요',
  scanTimeout: '선택하지 않았어요',
  scanTimeoutSub: '다시 누르면 새로 녹화합니다',
  scanTimeoutNurse: '간호사실에는 확인 필요로 알렸어요',
  closed: '선택을 닫았어요',
  cancelRequest: '방금 요청은 취소할게요',
  cancelRequestSent: '간호사실 카드에 “환자가 취소 요청함”을 표시했어요',
  cancelRequestNone: '취소할 요청이 없어요',
  cancelRequestNote: '환자가 취소 요청함',
  errorHeadline: '다시 눌러 주세요',
  recordStuck: '녹화가 끝나지 않았어요',
  cameraNotReady: '카메라가 준비되지 않았어요',
  retry: '다시 말하기',
  noneOfTwo: '둘 다 아님',
  noneOfOne: '이것 아님',
  back: '뒤로',
  close: '닫기',
  selectedNote: '환자가 직접 고름',
  retryNote: '환자가 다시 말하기를 고름',
  noneNote: '환자: 후보가 아님',
  closedNote: '환자가 선택을 닫음',
  timeoutNote: '환자 선택 없음',
  emergencyChosenNote: (text) => `환자 선택: ${text} (긴급 카드를 따로 보냄)`,
  enrollChosen: '내 입 모양 등록으로 골랐어요',
  enrollAlarm: '내 입 모양 등록과 긴급 명령이 비슷해요',
  enrollBusy: '진행 중인 녹화·확인·선택이 끝난 뒤 등록 화면을 열 수 있어요',
  enrollAlarmOpen: '확인하지 않은 긴급 알림이 있어요. 간호사 화면에서 확인한 뒤 등록 화면을 열 수 있어요',
  listsBusy: '진행 중인 녹화·확인·선택이 끝난 뒤 목록 관리 화면을 열 수 있어요',
  listsAlarmOpen: '확인하지 않은 긴급 알림이 있어요. 간호사 화면에서 확인한 뒤 목록 관리 화면을 열 수 있어요',
  listSwitchBusy: '진행 중인 녹화·확인·선택이 끝난 뒤 명령 목록을 바꿀 수 있어요',
  listLoading: '내 목록을 불러오는 중이에요',
  listBlocked: '입 읽기 녹화를 쓸 수 없어요',
  listBlockedSub: '스위치 3번 = 긴급 호출은 됩니다 · 설정에서 명령 목록을 확인해 주세요',
});

/** 입 읽기 녹화를 막는 사유 (lipBlock.reason). */
export const LIP_BLOCK_REASONS = Object.freeze(['loading', 'error']);

/**
 * lipBlock 값을 다듬는다. null·undefined면 막지 않음(null). 모르는 사유는 'error'로 본다(목록 문제를 놓치지 않게).
 * @param {any} raw {reason:'loading'|'error', message?:string} 또는 null.
 * @returns {{reason:string, message:string}|null}
 */
export function normalizeLipBlock(raw) {
  if (raw === null || raw === undefined) return null;
  const reason = LIP_BLOCK_REASONS.includes(raw?.reason) ? raw.reason : 'error';
  const message = typeof raw?.message === 'string' ? raw.message : '';
  return { reason, message };
}

/** 카드의 enroll 표시: 'chosen' = 등록한 입 모양으로 고름, 'alarm' = 등록한 입 모양 때문에만 긴급 가능성이 됨. */
export const CARD_ENROLL_TAGS = Object.freeze(['chosen', 'alarm']);

/** 화면 단계 이름 (간호사 화면에 함께 보인다). */
export const PHASE_LABELS = Object.freeze({
  idle: '대기', recording: '녹화 중', analyzing: '분석 중', confirm: '보내기 전 확인', scan: '선택 중',
});

/**
 * 처음 상태를 만든다.
 * @param {{index: object, baseIndex?: object|null, settings?: object, sessionKey?: string, room?: object,
 *   lipBlock?: object|null}} opts
 *   index는 buildCommandIndex(또는 buildCustomCommandIndex·emptyListIndex) 결과, sessionKey는 카드 ID 앞부분(창마다 다르게).
 *   baseIndex는 기본 목록 색인(buildCommandIndex(commands.json)). 없으면 index가 기본 목록일 때 그것을 쓴다.
 *   lipBlock은 내 목록을 불러오는 중·실패일 때 입 읽기 녹화를 막는 사유(기본 목록은 null).
 * @returns {object}
 */
export function createWardState({
  index, baseIndex = null, settings, sessionKey = 'w', room = INITIAL_ROOM, lipBlock = null,
} = {}) {
  return {
    index,
    baseIndex: isBaseIndex(baseIndex) ? baseIndex : (isBaseIndex(index) ? index : null),
    lipBlock: normalizeLipBlock(lipBlock),
    settings: normalizeSettings(settings),
    sessionKey: String(sessionKey),
    phase: 'idle', // idle | recording | analyzing | confirm | scan
    run: 0, // 녹화·분석 실행 번호: 늦게 온 이전 결과는 무시한다
    seq: 0, // 카드 번호
    scanSeq: 0, // 선택 단계 식별자. 이전 목록의 늦은 직접 클릭을 무시한다.
    lastPressAt: null,
    pressTimes: [],
    recording: null, // {run, startedAt, endsAt}
    analysis: null, // {run, startedAt}
    confirm: null, // {commandId, startedAt, endsAt}
    scan: null, // {mode, items, index, round, stepEndsAt, cardId, reason, detail, categoryId}
    screen: { kind: 'welcome' },
    noSpeechStreak: 0,
    lastNormalCardId: null,
    cards: {},
    cardOrder: [],
    ackNotice: null, // {cardIds, until}
    room,
  };
}

// ---------------------------------------------------------------------------
// 카드
// ---------------------------------------------------------------------------

/** 카드를 환자 쪽 기억에 넣는다 (오래된 것부터 버림). */
function keepCard(s, card) {
  const cards = { ...s.cards, [card.id]: card };
  let order = s.cardOrder.includes(card.id) ? s.cardOrder : [...s.cardOrder, card.id];
  while (order.length > MAX_KEPT_CARDS) {
    delete cards[order[0]];
    order = order.slice(1);
  }
  return { ...s, cards, cardOrder: order };
}

/** 기본 목록 색인인지 (명령 조회용 Map이 있고 내 목록 표시가 없음). */
function isBaseIndex(index) {
  return !!index && typeof index === 'object' && index.byId instanceof Map && !index.customList;
}

/**
 * 카드에 붙일 목록 표시 {id, title, sha256}. 기본 목록이면 null.
 * 내 목록을 실제로 불러왔을 때(목록 판 sha256이 있을 때)만 붙인다: 불러오는 중·실패한 동안의 카드는 그 목록의 문장에서 온 것이
 * 아니다. 스위치 3번 긴급 호출(emergency_call)은 목록과 상관없는 호출이라 붙이지 않는다.
 * @param {object} s
 * @param {string} kind 카드 종류.
 * @returns {{id:string, title:string, sha256:string}|null}
 */
function cardListTag(s, kind) {
  if (kind === 'emergency_call') return null;
  const meta = s.index?.customList;
  return meta && meta.id && meta.sha256 ? { id: meta.id, title: meta.title || meta.id, sha256: meta.sha256 } : null;
}

/** 새 카드를 만들어 보낸다. 내 목록을 쓰는 중이면 list {id, title, sha256}을 붙인다(기본 목록 카드는 예전 모양 그대로). */
function addCard(s, at, fields, effects) {
  const seq = s.seq + 1;
  const list = cardListTag(s, fields.kind);
  const card = {
    id: `${s.sessionKey}-${seq}`,
    rev: 1,
    room: s.settings.room,
    kind: fields.kind,
    commandId: null,
    text: '',
    candidates: [],
    reason: null,
    hint: null,
    note: null,
    cancelRequested: false,
    selectedId: null,
    source: 'lip',
    createdAt: at,
    updatedAt: at,
    acked: false,
    ackedAt: null,
    enroll: null,
    ...(list ? { list } : {}),
    ...fields,
  };
  effects.push({ type: 'SEND_CARD', card });
  return { s: keepCard({ ...s, seq }, card), card };
}

/** 보낸 카드를 고친다 (새 카드를 만들지 않음). 간호사가 다시 봐야 하므로 확인 표시는 지운다. */
function changeCard(s, at, id, patch, effects) {
  const card = id ? s.cards[id] : null;
  if (!card) return s;
  const updated = { ...card, ...patch, rev: card.rev + 1, updatedAt: at, acked: false, ackedAt: null };
  effects.push({ type: 'UPDATE_CARD', card: updated });
  return keepCard(s, updated);
}

/** 명령 → 카드 후보 표시 항목. */
function candidateOf(command) {
  return { id: command.id, text: command.display, emergency: command.emergency };
}

/** W08·W09 카드에 붙일 문구. */
function hintFor(command) {
  return TEMPERATURE_ACTIONS.includes(command?.action) ? TEMPERATURE_HINT : null;
}

/** 보낼 때 바로 바뀌는 가상 병실 동작을 적용한다. */
function applySendAction(s, command) {
  return roomActionTiming(command?.action) === 'send' ? { ...s, room: applyRoomAction(s.room, command.action) } : s;
}

/** 흐름을 끝내고 대기 화면으로. */
function toIdle(s, screen) {
  return { ...s, phase: 'idle', recording: null, analysis: null, confirm: null, scan: null, screen };
}

// ---------------------------------------------------------------------------
// 보내기
// ---------------------------------------------------------------------------

/** 일반 요청 카드를 보낸다 (확인 카운트다운이 끝났거나 스캐닝으로 직접 고름). enroll은 등록 표시(CARD_ENROLL_TAGS 또는 null). */
function sendRequest(s, at, command, source, effects, enroll = null) {
  const { s: s1, card } = addCard(s, at, {
    kind: 'request', commandId: command.id, text: command.display, hint: hintFor(command), source, enroll,
  }, effects);
  return toIdle(applySendAction({ ...s1, lastNormalCardId: card.id }, command),
    { kind: 'sent', commandId: command.id, cardId: card.id });
}

/** 긴급 명령: 확인 없이 빨간 카드를 바로 보낸다. 환자 쪽에서 취소할 수 없다. */
function sendEmergency(s, at, command, source, effects) {
  const { s: s1, card } = addCard(s, at, {
    kind: 'emergency', commandId: command.id, text: command.display, source,
  }, effects);
  effects.push({ type: 'SPEAK', text: command.display });
  return toIdle(s1, { kind: 'emergency', commandId: command.id, cardId: card.id });
}

/** 스위치 3번: 입 읽기 없는 긴급 호출. 녹화·분석 중이면 그대로 이어서 결과를 처리한다. */
function sendEmergencyCall(s, at, effects) {
  const { s: s1, card } = addCard(s, at, {
    kind: 'emergency_call', text: WARD_TEXT.emergencyCall, source: 'switch',
  }, effects);
  const screen = { kind: 'emergency_call', cardId: card.id };
  if (s1.phase === 'recording' || s1.phase === 'analyzing') return { ...s1, screen };
  return toIdle(s1, screen);
}

/** 취소 명령(기본 W20·내 목록의 취소 명령): 마지막 일반 카드를 지우지 않고 "환자가 취소 요청함"을 덧붙인다. 긴급 카드에는 쓰지 않는다. */
function requestCancelLast(s, at, effects) {
  const target = s.lastNormalCardId ? s.cards[s.lastNormalCardId] : null;
  if (!target || target.kind === 'emergency' || target.kind === 'emergency_call' || target.kind === 'emergency_possible') {
    return toIdle(s, { kind: 'cancel_request', ok: false });
  }
  const s1 = changeCard(s, at, target.id, { cancelRequested: true, note: WARD_TEXT.cancelRequestNote }, effects);
  return toIdle(s1, { kind: 'cancel_request', ok: true, cardId: target.id });
}

// ---------------------------------------------------------------------------
// 녹화·확인·스캐닝
// ---------------------------------------------------------------------------

/** 녹화 시작 (카운트다운 없음, 시작음·테두리 깜빡임은 화면이 한다). 내 목록을 쓸 수 없으면 시작하지 않고 안내만 한다. */
function startRecording(s, at, effects) {
  if (s.lipBlock) return toIdle(s, { kind: 'lip_blocked' });
  const run = s.run + 1;
  const seconds = s.settings.recordSeconds;
  effects.push({ type: 'START_RECORDING', run, seconds });
  return {
    ...s,
    run,
    phase: 'recording',
    recording: { run, startedAt: at, endsAt: at + seconds * 1000 },
    analysis: null,
    confirm: null,
    scan: null,
  };
}

/** 고른 일반 명령: 확인 카운트다운. enroll='chosen'이면 등록한 입 모양으로 고른 것(카드에 표시). */
function startConfirm(s, at, command, effects, enroll = null) {
  effects.push({ type: 'SPEAK', text: command.display });
  return {
    ...s,
    phase: 'confirm',
    recording: null,
    analysis: null,
    scan: null,
    confirm: { commandId: command.id, startedAt: at, endsAt: at + s.settings.confirmSeconds * 1000, enroll },
    screen: { kind: 'confirm', commandId: command.id },
  };
}

/** 카운트다운이 끝나면 보낸다 (취소 명령이면 취소 요청). */
function finishConfirm(s, at, effects) {
  const command = commandById(s.index, s.confirm?.commandId);
  if (!command) return toIdle(s, { kind: 'welcome' });
  if (isCancelCommand(command)) return requestCancelLast(s, at, effects);
  return sendRequest(s, at, command, 'lip', effects, s.confirm?.enroll || null);
}

/** 스캐닝 상태를 만든다. */
function scanState(at, mode, items, extra = {}) {
  return { mode, items, index: 0, round: 0, stepEndsAt: at + SCAN_STEP_MS, cardId: null, reason: null, detail: null, ...extra };
}

/** 고르지 못함(일반): 후보1 → 후보2 → 다시 말하기 → 둘 다 아님 스캐닝 + 간호사실 '확인 필요' 카드(추정 후보). */
function startCandidateScan(s, at, command, runnerUp, effects) {
  const candidates = [candidateOf(command)];
  if (runnerUp) candidates.push(candidateOf(runnerUp));
  const { s: s1, card } = addCard(s, at, { kind: 'check', text: WARD_TEXT.checkCard, candidates }, effects);
  const items = candidates.map((c) => ({ key: 'command', commandId: c.id, label: c.text, emergency: c.emergency }));
  items.push({ key: 'retry', label: WARD_TEXT.retry });
  items.push({ key: 'none', label: candidates.length > 1 ? WARD_TEXT.noneOfTwo : WARD_TEXT.noneOfOne });
  return {
    ...s1,
    scanSeq: s.scanSeq + 1,
    phase: 'scan',
    recording: null,
    analysis: null,
    confirm: null,
    scan: scanState(at, 'candidates', items, { cardId: card.id }),
    screen: { kind: 'scan' },
  };
}

/** 분류 5개 스캐닝 (+ 다시 말하기·닫기). cardId가 있으면 고른 결과로 그 카드를 갱신한다. */
function startCategoryScan(s, at, reason, { cardId = null, detail = null } = {}) {
  const items = s.index.categories.map((c) => ({ key: 'category', categoryId: c.id, label: c.label, emergency: c.id === 'emergency' }));
  items.push({ key: 'retry', label: WARD_TEXT.retry });
  items.push({ key: 'close', label: WARD_TEXT.close });
  return {
    ...s,
    scanSeq: s.scanSeq + 1,
    phase: 'scan',
    recording: null,
    analysis: null,
    confirm: null,
    scan: scanState(at, 'categories', items, { cardId, reason, detail }),
    screen: { kind: 'scan' },
  };
}

/** 분류 하나를 고르면 그 분류의 명령 스캐닝 (+ 뒤로). */
function startCommandScan(s, at, categoryId) {
  const cat = s.index.categories.find((c) => c.id === categoryId);
  if (!cat) return s;
  const items = cat.commands.map((c) => ({ key: 'command', commandId: c.id, label: c.display, emergency: c.emergency }));
  items.push({ key: 'back', label: WARD_TEXT.back });
  return { ...s, scanSeq: s.scanSeq + 1, scan: scanState(at, 'commands', items, { cardId: s.scan?.cardId ?? null, reason: s.scan?.reason ?? null, detail: s.scan?.detail ?? null, categoryId }) };
}

/** 화면 버튼 누름(PRESS intent)이 유효한 단계. */
export const PRESS_INTENT_PHASES = Object.freeze({ start: 'idle', cancel: 'confirm', select: 'scan' });

/**
 * 스캐닝이 고르지 않은 채 끝나는 시각 (SCAN_MAX_ROUNDS 바퀴). 직접 선택 모드에서 남은 시간 표시에 쓴다.
 * @param {object|null} scan
 * @returns {{endsAt:number, startedAt:number}|null}
 */
export function scanDeadline(scan) {
  try {
    const n = scan?.items?.length || 0;
    if (!n || !Number.isFinite(scan.stepEndsAt)) return null;
    const stepsLeft = (SCAN_MAX_ROUNDS - 1 - scan.round) * n + (n - 1 - scan.index);
    const endsAt = scan.stepEndsAt + stepsLeft * SCAN_STEP_MS;
    return { endsAt, startedAt: endsAt - SCAN_MAX_ROUNDS * n * SCAN_STEP_MS };
  } catch {
    return null;
  }
}

/** 직접 선택은 강조 위치와 독립적인 항목 키와 현재 목록 식별자를 사용한다. */
export function scanItemKey(item) {
  return `${item?.key || ''}:${item?.commandId || item?.categoryId || ''}`;
}

export function scanToken(state) {
  return state?.phase === 'scan' ? `${state.run}:${state.scanSeq}` : null;
}

/** 스캐닝으로 명령을 골랐을 때. 확인 카운트다운 없이 바로 보낸다(고른 것 자체가 확인). */
function chooseCommand(s, at, commandId, cardId, effects) {
  const command = commandById(s.index, commandId);
  if (!command) return s;
  if (command.emergency) {
    let s1 = s;
    if (cardId) s1 = changeCard(s1, at, cardId, { note: WARD_TEXT.emergencyChosenNote(command.display) }, effects);
    return sendEmergency(s1, at, command, 'scan', effects);
  }
  effects.push({ type: 'SPEAK', text: command.display });
  if (!cardId) {
    if (isCancelCommand(command)) return requestCancelLast(s, at, effects);
    return sendRequest(s, at, command, 'scan', effects);
  }
  // '확인 필요' 카드를 새로 만들지 않고 그 카드를 갱신한다.
  const s1 = changeCard(s, at, cardId, {
    selectedId: command.id, commandId: command.id, text: command.display, hint: hintFor(command), note: WARD_TEXT.selectedNote,
  }, effects);
  if (isCancelCommand(command)) return requestCancelLast(s1, at, effects);
  return toIdle(applySendAction({ ...s1, lastNormalCardId: cardId }, command),
    { kind: 'sent', commandId: command.id, cardId });
}

/** 스캐닝 중 누름: 강조된 항목을 고른다. */
function selectScanItem(s, at, effects) {
  const scan = s.scan;
  const item = scan?.items?.[scan.index];
  if (!item) return toIdle(s, { kind: 'welcome' });
  switch (item.key) {
    case 'command':
      return chooseCommand(s, at, item.commandId, scan.cardId, effects);
    case 'category':
      return startCommandScan(s, at, item.categoryId);
    case 'back':
      return { ...startCategoryScan(s, at, scan.reason, { cardId: scan.cardId, detail: scan.detail }) };
    case 'retry': {
      const s1 = scan.cardId ? changeCard(s, at, scan.cardId, { note: WARD_TEXT.retryNote }, effects) : s;
      return startRecording(s1, at, effects);
    }
    case 'none': {
      const s1 = changeCard(s, at, scan.cardId, { note: WARD_TEXT.noneNote }, effects);
      return startCategoryScan(s1, at, 'not_candidates', { cardId: scan.cardId });
    }
    case 'close': {
      const s1 = scan.cardId ? changeCard(s, at, scan.cardId, { note: WARD_TEXT.closedNote }, effects) : s;
      return toIdle(s1, { kind: 'closed' });
    }
    default:
      return s;
  }
}

/** 스캐닝에서 단계만 옮기는 항목 (분류 고르기·둘 다 아님·뒤로). 이 누름은 3번 누르기 긴급 호출에 세지 않는다. */
export const SCAN_MOVE_KEYS = Object.freeze(['category', 'none', 'back']);

/**
 * 지금 강조된 스캐닝 항목이 단계만 옮기는 항목인지.
 * @param {object} s
 * @returns {boolean}
 */
function highlightedIsScanMove(s) {
  const item = s.phase === 'scan' ? s.scan?.items?.[s.scan.index] : null;
  return !!item && SCAN_MOVE_KEYS.includes(item.key);
}

/** 스캐닝을 SCAN_MAX_ROUNDS 바퀴 돌 동안 고르지 않음. */
function endScanTimeout(s, at, effects) {
  const cardId = s.scan?.cardId || null;
  const s1 = cardId ? changeCard(s, at, cardId, { note: WARD_TEXT.timeoutNote }, effects) : s;
  return toIdle(s1, { kind: 'scan_timeout', nurse: !!cardId });
}

// ---------------------------------------------------------------------------
// 결과 처리 (9.3 판정 규칙)
// ---------------------------------------------------------------------------

/** '긴급 가능성, 확인 필요' 카드를 바로 보낸다. 환자 화면은 대기로 돌아가 후보를 보여 준다. */
function sendPossible(s, at, commands, enroll, effects) {
  const candidates = commands.map(candidateOf);
  const { s: s1, card } = addCard(s, at, {
    kind: 'emergency_possible', text: WARD_TEXT.emergencyPossibleCard, candidates, enroll,
  }, effects);
  return toIdle(s1, { kind: 'emergency_possible', cardId: card.id, candidates, enroll });
}

/**
 * 분석 결과를 처리한다 (판정표는 wardResult.js decideWard = metrics.py decide).
 * - 글자로 고름 + 일반 → 확인 카운트다운 / 글자로 고름 + 긴급 → 빨간 카드 즉시
 * - 등록으로 고름 + 일반 → 확인 카운트다운 / 등록으로 고름 + 긴급, 또는 일반이지만 글자 2등이 긴급
 *   → '긴급 가능성' 즉시(고른 명령이 후보 맨 앞)
 * - 고르지 못함 + (1·2등에 긴급 또는 거리 안의 등록 1등이 긴급) → '긴급 가능성, 확인 필요' 즉시
 * - 고르지 못함 일반 → 후보 스캐닝 + '확인 필요'
 * - 지정 문장 결과 없음 → 분류 스캐닝 + 사유 / 발화 없음·검출 실패 → 1회 환자 화면만, 연속 2회부터 간호사실에도
 */
function handleOutcome(s, at, outcome, effects) {
  // 내 목록을 쓸 수 없는 동안에는 어떤 결과도 명령으로 해석하지 않는다(고를 명령 목록이 없다).
  if (s.lipBlock) return toIdle(s, { kind: 'lip_blocked' });
  if (outcome.kind === 'no_speech') {
    const streak = s.noSpeechStreak + 1;
    let s1 = { ...s, noSpeechStreak: streak };
    const nurse = streak >= NO_SPEECH_NURSE_STREAK;
    if (nurse) {
      s1 = addCard(s1, at, { kind: 'check', text: WARD_TEXT.noMouthCard, reason: 'no_mouth' }, effects).s;
    }
    return toIdle(s1, { kind: 'no_speech', nurse, reason: outcome.reason });
  }
  const s0 = { ...s, noSpeechStreak: 0 };
  if (outcome.kind === 'phrase') {
    const d = decideWard(outcome);
    const enrollTag = d.via === 'enroll' ? 'chosen' : (d.enrollAlarm ? 'alarm' : null);
    switch (d.outcome) {
      case 'emergency':
        return sendEmergency(s0, at, d.chosen, 'lip', effects);
      case 'request':
        return startConfirm(s0, at, d.chosen, effects, enrollTag);
      case 'emergency_possible':
        return sendPossible(s0, at, d.candidates, enrollTag, effects);
      default:
        return startCandidateScan(s0, at, outcome.command, outcome.runnerUp, effects);
    }
  }
  return startCategoryScan(s0, at, outcome.reason || 'empty', { detail: outcome.detail || null });
}

// ---------------------------------------------------------------------------
// 시간 경과·누름
// ---------------------------------------------------------------------------

/** 시각 기반 처리를 따라잡는다 (카운트다운 끝, 스캐닝 이동, 안내 만료, 녹화·분석 감시). */
function advance(state, at, effects) {
  let s = state;
  if (s.ackNotice && at >= s.ackNotice.until) s = { ...s, ackNotice: null };
  if (s.phase === 'confirm' && s.confirm && at >= s.confirm.endsAt) {
    s = finishConfirm(s, s.confirm.endsAt, effects);
  }
  if (s.phase === 'scan' && s.scan) {
    let scan = s.scan;
    let timedOut = false;
    while (at >= scan.stepEndsAt) {
      let index = scan.index + 1;
      let round = scan.round;
      if (index >= scan.items.length) {
        index = 0;
        round += 1;
      }
      if (round >= SCAN_MAX_ROUNDS) {
        timedOut = true;
        break;
      }
      scan = { ...scan, index, round, stepEndsAt: scan.stepEndsAt + SCAN_STEP_MS };
    }
    if (timedOut) s = endScanTimeout(s, scan.stepEndsAt, effects);
    else if (scan !== s.scan) s = { ...s, scan };
  }
  if (s.phase === 'recording' && s.recording && at > s.recording.endsAt + RECORD_GRACE_MS) {
    s = toIdle(s, { kind: 'error', message: WARD_TEXT.recordStuck });
  }
  if (s.phase === 'analyzing' && s.analysis && at > s.analysis.startedAt + ANALYSIS_LIMIT_MS) {
    s = startCategoryScan(s, at, 'timeout');
  }
  return s;
}

/** 스위치 누름 하나. */
function handlePress(state, at, repeat, effects) {
  if (repeat) return state; // 키를 누르고 있을 때의 자동 반복
  const gap = state.lastPressAt === null ? Infinity : at - state.lastPressAt;
  if (gap >= 0 && gap < PRESS_MERGE_MS) return state; // 0.5초 안의 연속 입력은 하나로
  // 직접 선택 모드의 스캐닝: 강조를 그리지 않으므로 키(스위치)는 항목을 고르지 않는다. 3번 누르기 긴급 호출에만 센다.
  if (state.phase === 'scan' && state.settings?.inputMode === 'direct') {
    const pressed = [...state.pressTimes.filter((t) => at - t >= 0 && at - t <= TRIPLE_PRESS_WINDOW_MS), at];
    if (pressed.length >= TRIPLE_PRESS_COUNT) return sendEmergencyCall({ ...state, lastPressAt: at, pressTimes: [] }, at, effects);
    return { ...state, lastPressAt: at, pressTimes: pressed };
  }
  // 스캐닝에서 단계를 옮기는 누름(분류 고르기·둘 다 아님·뒤로)은 일부러 고른 것이다. 3번 누르기에 세지 않고 앞선 누름도
  // 지운다. 원하는 항목이 앞쪽에 있어 빨리 이어 누른 선택(예: 둘 다 아님 → 호출·소통 → W01)이 긴급 호출로 바뀌지 않게 한다.
  if (highlightedIsScanMove(state)) {
    return selectScanItem({ ...state, lastPressAt: at, pressTimes: [] }, at, effects);
  }
  const recent = [...state.pressTimes.filter((t) => at - t >= 0 && at - t <= TRIPLE_PRESS_WINDOW_MS), at];
  if (recent.length >= TRIPLE_PRESS_COUNT) {
    return sendEmergencyCall({ ...state, lastPressAt: at, pressTimes: [] }, at, effects);
  }
  const s = { ...state, lastPressAt: at, pressTimes: recent };
  switch (s.phase) {
    case 'idle':
      return startRecording(s, at, effects);
    case 'confirm':
      return toIdle(s, { kind: 'cancelled', commandId: s.confirm?.commandId || null });
    case 'scan':
      return selectScanItem(s, at, effects);
    default:
      return s; // 녹화·분석 중 한 번 누름은 녹화를 멈추지 않는다
  }
}

/**
 * 카드가 가리키는 명령 (확인 필요 카드는 환자가 고른 것만). 카드를 만든 목록에서 찾는다:
 * 기본 목록 카드(list 없음)는 기본 목록 색인(baseIndex)에서, 내 목록 카드는 지금 쓰는 내 목록이 같은 판(id·sha256)일 때만 찾는다.
 * 그래서 기본 목록에서 보낸 W14(등받이)를 내 목록으로 바꾼 뒤 확인해도 가상 침대가 움직이고, 고친 내 목록의 같은 U 번호를
 * 옛 카드의 명령으로 잘못 읽지 않는다.
 */
function cardCommand(s, card) {
  if (card.kind === 'check' && !card.selectedId) return null;
  const id = card.selectedId || card.commandId;
  const list = card.list;
  if (list !== undefined && list !== null) {
    const meta = s.index?.customList;
    const same = !!meta && !!list && meta.id === list.id && !!meta.sha256 && meta.sha256 === list.sha256;
    return same ? commandById(s.index, id) : null;
  }
  const base = s.index?.customList ? s.baseIndex : s.index;
  return base ? commandById(base, id) : null;
}

/**
 * 간호사 확인: 카드 확인 표시, "간호사가 확인했어요", 확인 뒤에만 움직이는 가상 침대.
 * 환자가 취소 요청한 카드는 확인해도 침대·자세를 움직이지 않는다(몸에 닿는 동작은 스스로 하지 않는다).
 * 간호사 화면이 합친 카드(같은 요청 여러 번)를 한 번에 확인했을 때 그중 하나라도 취소 요청이면 그 동작은 하지 않는다.
 */
function handleAck(state, at, cardIds, room) {
  let s = state;
  let known = false;
  const applied = new Set();
  const cancelledActions = new Set();
  for (const id of cardIds) {
    const card = s.cards[id];
    const command = card ? cardCommand(s, card) : null;
    if (card?.cancelRequested && command) cancelledActions.add(command.action);
  }
  for (const id of cardIds) {
    const card = s.cards[id];
    if (!card) continue;
    known = true;
    if (card.acked) continue;
    const updated = { ...card, acked: true, ackedAt: at };
    const command = cardCommand(s, card);
    if (command && !card.cancelRequested && !cancelledActions.has(command.action)) {
      if (roomActionTiming(command.action) === 'ack' && !card.roomApplied && !applied.has(command.action)) {
        s = { ...s, room: applyRoomAction(s.room, command.action) };
        applied.add(command.action);
        updated.roomApplied = true;
      }
    }
    s = keepCard(s, updated);
  }
  if (known || room === s.settings.room) s = { ...s, ackNotice: { cardIds: [...cardIds], until: at + ACK_NOTICE_MS } };
  return s;
}

/** 지금 흐름을 모두 멈추고 이전 녹화 결과를 무효로 한다 (연습 모드 결과 주입·명령 목록 바꿈). */
function resetFlow(s) {
  return { ...s, run: s.run + 1, phase: 'idle', recording: null, analysis: null, confirm: null, scan: null };
}

/** 두 lipBlock이 같은지. */
function sameLipBlock(a, b) {
  if (!a || !b) return a === b;
  return a.reason === b.reason && a.message === b.message;
}

/**
 * 명령 목록(색인)을 바꾼다. 진행 중인 녹화·확인·선택은 멈추고 늦게 온 결과는 무시한다(다른 목록 문장으로 잘못 보내지 않게).
 * 보낸 카드·가상 병실·긴급 알림은 그대로 둔다. 바뀐 것이 없으면 같은 상태. 같은 목록·같은 판에서 이름만 바뀌었으면
 * 흐름을 그대로 두고 색인만 바꾼다(명령은 sha256이 같으니 그대로다).
 * @param {object} s
 * @param {object} index 새 색인 (byId가 Map이 아니면 지금 색인 유지).
 * @param {object|null} lipBlockRaw
 * @returns {object}
 */
function setIndex(s, index, lipBlockRaw) {
  const next = index && typeof index === 'object' && index.byId instanceof Map ? index : s.index;
  const lipBlock = normalizeLipBlock(lipBlockRaw);
  if (next === s.index && sameLipBlock(lipBlock, s.lipBlock)) return s;
  const baseIndex = isBaseIndex(next) ? next : s.baseIndex;
  // 같은 목록의 같은 판(명령이 그대로, 이름만 바뀜): 흐름을 멈추지 않고 색인(띠·카드에 쓰는 이름)만 바꾼다.
  const a = s.index?.customList;
  const b = next.customList;
  if (next !== s.index && a && b && a.id === b.id && a.sha256 && a.sha256 === b.sha256 && sameLipBlock(lipBlock, s.lipBlock)) {
    return { ...s, index: next, baseIndex };
  }
  const keepScreen = s.phase === 'idle' && s.screen?.kind !== 'lip_blocked';
  return { ...resetFlow(s), index: next, baseIndex, lipBlock, screen: keepScreen ? s.screen : { kind: 'welcome' } };
}

/**
 * 상태도 한 단계.
 * @param {object} state createWardState 결과.
 * @param {{type:string, at:number, [key:string]: any}} event
 *   PRESS{repeat?} · SELECT_ITEM{scanToken, itemKey} · TICK · RECORDING_DONE{run} · RECORD_FAILED{run, message} · RESULT{run, results}
 *   · ANALYSIS_ERROR{run, message, reason?} · SIMULATE{results} · NURSE_ACK{cardIds, room?} · SETTINGS{settings}
 *   · SET_INDEX{index, lipBlock?} (명령 목록 바꿈·내 목록 불러오기 결과)
 * @returns {{state: object, effects: Array<object>}} 바뀐 것이 없으면 state는 같은 객체다.
 */
export function stepWard(state, event) {
  const effects = [];
  try {
    if (!state || !event || typeof event.type !== 'string') return { state, effects };
    const at = Number(event.at);
    if (!Number.isFinite(at)) return { state, effects };
    let s = advance(state, at, effects);
    const sameRun = (phases) => event.run === s.run && phases.includes(s.phase);
    switch (event.type) {
      case 'TICK':
        break;
      case 'PRESS':
        // 화면 버튼(녹화 시작·전송 취소·강조된 항목 선택)은 intent를 붙여 보낸다. 그사이 단계가 바뀌었으면(예: 카운트다운이 막
        // 끝나 이미 보냄) 글자와 다른 일을 하지 않도록 무시한다. 스위치·키 누름은 intent가 없어 예전과 같다.
        if (event.intent && PRESS_INTENT_PHASES[event.intent] !== s.phase) break;
        s = handlePress(s, at, event.repeat === true, effects);
        break;
      case 'SELECT_ITEM': {
        if (s.settings.inputMode !== 'direct' || s.phase !== 'scan' || event.scanToken !== scanToken(s)) break;
        // 두 번 누르기(더블 클릭)의 두 번째 누름이 바로 다음 목록의 같은 자리 항목을 고르지 않게 0.5초 안의 연속 선택은 하나로 본다.
        const gap = s.lastPressAt === null ? Infinity : at - s.lastPressAt;
        if (gap >= 0 && gap < PRESS_MERGE_MS) break;
        const selected = s.scan.items.findIndex((item) => scanItemKey(item) === event.itemKey);
        if (selected < 0) break;
        s = selectScanItem({ ...s, lastPressAt: at, pressTimes: [], scan: { ...s.scan, index: selected } }, at, effects);
        break;
      }
      case 'RECORDING_DONE':
        if (sameRun(['recording'])) s = { ...s, phase: 'analyzing', recording: null, analysis: { run: s.run, startedAt: at } };
        break;
      case 'RECORD_FAILED':
        if (sameRun(['recording', 'analyzing'])) {
          s = toIdle(s, { kind: 'error', message: String(event.message || WARD_TEXT.cameraNotReady) });
        }
        break;
      case 'RESULT':
        if (sameRun(['recording', 'analyzing'])) s = handleOutcome(s, at, classifyWardResults(event.results, s.index), effects);
        break;
      case 'ANALYSIS_ERROR':
        if (sameRun(['recording', 'analyzing'])) {
          s = startCategoryScan(s, at, event.reason === 'timeout' ? 'timeout' : 'error',
            { detail: typeof event.message === 'string' ? event.message : null });
        }
        break;
      case 'SIMULATE':
        s = handleOutcome(resetFlow(s), at, classifyWardResults(event.results, s.index), effects);
        break;
      case 'NURSE_ACK':
        s = handleAck(s, at, Array.isArray(event.cardIds) ? event.cardIds.map(String) : [], event.room);
        break;
      case 'SETTINGS':
        if (event.settings?.inputMode && event.settings.inputMode !== s.settings.inputMode) {
          s = { ...s, pressTimes: [], lastPressAt: null };
        }
        s = { ...s, settings: normalizeSettings({ ...s.settings, ...(event.settings || {}) }) };
        break;
      case 'SET_INDEX':
        s = setIndex(s, event.index, event.lipBlock ?? null);
        break;
      default:
        break;
    }
    return { state: s, effects };
  } catch (err) {
    return { state, effects: [{ type: 'LOG', message: String(err?.message || err) }] };
  }
}

// ---------------------------------------------------------------------------
// 화면 표시용 요약
// ---------------------------------------------------------------------------

/**
 * 확인하지 않은 가장 최근 경보 카드 (환자 화면 상단 긴급 띠).
 * @param {object} s
 * @returns {object|null}
 */
export function latestOpenAlarm(s) {
  try {
    let best = null;
    for (const id of s.cardOrder) {
      const c = s.cards[id];
      if (c && ALARM_KINDS.includes(c.kind) && !c.acked && (!best || c.createdAt >= best.createdAt)) best = c;
    }
    return best;
  } catch {
    return null;
  }
}

/**
 * 등록 화면(#enroll)을 지금 열면 안 되는 이유. 열어도 되면 null.
 * 환자 화면을 떠나면 확인 카운트다운·스캐닝 선택·분석 대기가 사라지고, 확인하지 않은 긴급 표시도 다시 열 때 사라진다.
 * @param {object} s 상태.
 * @returns {string|null}
 */
export function enrollOpenBlockReason(s) {
  try {
    if (!s) return null;
    if (s.phase !== 'idle') return WARD_TEXT.enrollBusy;
    if (latestOpenAlarm(s)) return WARD_TEXT.enrollAlarmOpen;
    return null;
  } catch {
    return WARD_TEXT.enrollBusy;
  }
}

/**
 * 목록 관리 화면(#lists)을 지금 열면 안 되는 이유. 열어도 되면 null (등록 화면과 같은 조건).
 * @param {object} s 상태.
 * @returns {string|null}
 */
export function listsOpenBlockReason(s) {
  try {
    if (!s) return null;
    if (s.phase !== 'idle') return WARD_TEXT.listsBusy;
    if (latestOpenAlarm(s)) return WARD_TEXT.listsAlarmOpen;
    return null;
  } catch {
    return WARD_TEXT.listsBusy;
  }
}

/**
 * 설정에서 명령 목록을 지금 바꾸면 안 되는 이유 (진행 중인 녹화·확인·선택이 있으면). 바꿔도 되면 null.
 * @param {object} s 상태.
 * @returns {string|null}
 */
export function listSwitchBlockReason(s) {
  try {
    return s && s.phase !== 'idle' ? WARD_TEXT.listSwitchBusy : null;
  } catch {
    return WARD_TEXT.listSwitchBusy;
  }
}

/**
 * 입 읽기 녹화를 쓸 수 없을 때(내 목록 불러오는 중·실패) 표시.
 * @param {object} view 기본 표시 값.
 * @param {{reason:string, message:string}} lipBlock
 * @returns {object}
 */
function lipBlockedView(view, lipBlock) {
  if (lipBlock.reason === 'loading') {
    return { ...view, tone: 'neutral', headline: WARD_TEXT.listLoading, sub: WARD_TEXT.listBlockedSub };
  }
  const sub = lipBlock.message ? `${lipBlock.message} · ${WARD_TEXT.listBlockedSub}` : WARD_TEXT.listBlockedSub;
  return { ...view, tone: 'warning', headline: WARD_TEXT.listBlocked, sub };
}

/**
 * 상태를 환자 화면 표시 값으로 바꾼다.
 * @param {object} s 상태.
 * @param {number} now 현재 시각(ms).
 * @returns {{phase:string, tone:string, label:string|null, headline:string, sub:string|null, status:string|null,
 *   items:Array<{label:string, highlighted:boolean, emergency:boolean}>|null, reason:string|null,
 *   candidates:Array|null, alarm:object|null, acked:boolean, flash:boolean}}
 */
export function wardView(s, now) {
  const t = Number.isFinite(now) ? now : 0;
  const view = {
    phase: s.phase, tone: 'neutral', label: null, headline: WARD_TEXT.welcome, sub: null, status: null,
    items: null, reason: null, candidates: null, alarm: latestOpenAlarm(s),
    acked: !!(s.ackNotice && t < s.ackNotice.until), flash: s.phase === 'recording',
  };
  try {
    const cmd = (id) => commandById(s.index, id);
    if (s.phase === 'recording' && s.recording) {
      const sec = Math.max(0, Math.min(s.settings.recordSeconds, Math.ceil((s.recording.endsAt - t) / 1000)));
      return { ...view, tone: 'recording', headline: WARD_TEXT.recording, status: WARD_TEXT.recordingStatus(sec) };
    }
    if (s.phase === 'analyzing') return { ...view, tone: 'neutral', headline: WARD_TEXT.analyzing, sub: WARD_TEXT.analyzingSub };
    if (s.phase === 'confirm' && s.confirm) {
      const c = cmd(s.confirm.commandId);
      // 화면의 시각(now)이 결과 처리 시각보다 조금 늦게 갱신돼도 설정보다 큰 숫자(예: 7초)를 보이지 않게 한다.
      const sec = Math.max(0, Math.min(s.settings.confirmSeconds, Math.ceil((s.confirm.endsAt - t) / 1000)));
      return {
        ...view, tone: 'confirm', headline: c?.display || '', sub: WARD_TEXT.confirmSub(sec),
        reason: s.confirm.enroll === 'chosen' ? WARD_TEXT.enrollChosen : null,
      };
    }
    if (s.phase === 'scan' && s.scan) {
      const sc = s.scan;
      let headline = WARD_TEXT.scanCategories;
      if (sc.mode === 'candidates') headline = WARD_TEXT.scanCandidates;
      if (sc.mode === 'commands') {
        const cat = s.index.categories.find((c) => c.id === sc.categoryId);
        headline = WARD_TEXT.scanCommands(cat?.label || '');
      }
      const reason = sc.reason ? NO_PHRASE_REASON_TEXT[sc.reason] || null : null;
      return {
        ...view,
        tone: 'scan',
        headline,
        sub: WARD_TEXT.scanSub,
        reason: reason && sc.detail ? `${reason} (${sc.detail})` : reason,
        items: sc.items.map((it, i) => ({ label: it.label, highlighted: i === sc.index, emergency: !!it.emergency })),
      };
    }
    const sc = s.screen || { kind: 'welcome' };
    const card = sc.cardId ? s.cards[sc.cardId] : null;
    switch (sc.kind) {
      case 'sent':
        return { ...view, tone: 'sent', headline: cmd(sc.commandId)?.display || '', sub: card?.acked ? WARD_TEXT.acked : WARD_TEXT.sent };
      case 'emergency':
        return { ...view, tone: 'emergency', label: '[긴급]', headline: cmd(sc.commandId)?.display || '', sub: card?.acked ? WARD_TEXT.acked : WARD_TEXT.emergencySub };
      case 'emergency_call':
        return { ...view, tone: 'emergency', label: '[긴급]', headline: WARD_TEXT.emergencyCall, sub: card?.acked ? WARD_TEXT.acked : WARD_TEXT.emergencyCallSub };
      case 'emergency_possible':
        return {
          ...view, tone: 'possible', label: '[긴급 가능성]', headline: WARD_TEXT.emergencyPossible, sub: card?.acked ? WARD_TEXT.acked : null,
          candidates: sc.candidates || [],
          reason: sc.enroll === 'chosen' ? WARD_TEXT.enrollChosen : (sc.enroll === 'alarm' ? WARD_TEXT.enrollAlarm : null),
        };
      case 'cancelled':
        return { ...view, tone: 'neutral', headline: WARD_TEXT.cancelled, sub: WARD_TEXT.cancelledSub };
      case 'no_speech':
        return { ...view, tone: 'warning', headline: WARD_TEXT.noSpeech, sub: sc.nurse ? WARD_TEXT.noSpeechNurse : null };
      case 'cancel_request':
        return { ...view, tone: 'neutral', headline: WARD_TEXT.cancelRequest, sub: sc.ok ? WARD_TEXT.cancelRequestSent : WARD_TEXT.cancelRequestNone };
      case 'scan_timeout':
        return { ...view, tone: 'neutral', headline: WARD_TEXT.scanTimeout, sub: sc.nurse ? WARD_TEXT.scanTimeoutNurse : WARD_TEXT.scanTimeoutSub };
      case 'closed':
        return { ...view, tone: 'neutral', headline: WARD_TEXT.closed, sub: WARD_TEXT.cancelledSub };
      case 'lip_blocked':
        return lipBlockedView(view, s.lipBlock || { reason: 'error', message: '' });
      case 'error':
        return { ...view, tone: 'warning', headline: WARD_TEXT.errorHeadline, sub: sc.message || null };
      default:
        if (s.lipBlock) return lipBlockedView(view, s.lipBlock);
        return { ...view, headline: WARD_TEXT.welcome, sub: WARD_TEXT.welcomeSub(s.settings.recordSeconds) };
    }
  } catch {
    return view;
  }
}
