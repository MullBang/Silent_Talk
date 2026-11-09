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

import {
  CANCEL_COMMAND_ID, INITIAL_ROOM, TEMPERATURE_ACTIONS, TEMPERATURE_HINT, applyRoomAction, commandById, roomActionTiming,
} from './wardCommands.js';
import { NO_PHRASE_REASON_TEXT, classifyWardResults } from './wardResult.js';
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
});

/** 화면 단계 이름 (간호사 화면에 함께 보인다). */
export const PHASE_LABELS = Object.freeze({
  idle: '대기', recording: '녹화 중', analyzing: '분석 중', confirm: '보내기 전 확인', scan: '선택 중',
});

/**
 * 처음 상태를 만든다.
 * @param {{index: object, settings?: object, sessionKey?: string, room?: object}} opts
 *   index는 buildCommandIndex 결과, sessionKey는 카드 ID 앞부분(창마다 다르게).
 * @returns {object}
 */
export function createWardState({ index, settings, sessionKey = 'w', room = INITIAL_ROOM } = {}) {
  return {
    index,
    settings: normalizeSettings(settings),
    sessionKey: String(sessionKey),
    phase: 'idle', // idle | recording | analyzing | confirm | scan
    run: 0, // 녹화·분석 실행 번호: 늦게 온 이전 결과는 무시한다
    seq: 0, // 카드 번호
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

/** 새 카드를 만들어 보낸다. */
function addCard(s, at, fields, effects) {
  const seq = s.seq + 1;
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

/** 일반 요청 카드를 보낸다 (확인 카운트다운이 끝났거나 스캐닝으로 직접 고름). */
function sendRequest(s, at, command, source, effects) {
  const { s: s1, card } = addCard(s, at, {
    kind: 'request', commandId: command.id, text: command.display, hint: hintFor(command), source,
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

/** W20: 마지막 일반 카드를 지우지 않고 "환자가 취소 요청함"을 덧붙인다. 긴급 카드에는 쓰지 않는다. */
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

/** 녹화 시작 (카운트다운 없음, 시작음·테두리 깜빡임은 화면이 한다). */
function startRecording(s, at, effects) {
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

/** 고른 일반 명령: 확인 카운트다운. */
function startConfirm(s, at, command, effects) {
  effects.push({ type: 'SPEAK', text: command.display });
  return {
    ...s,
    phase: 'confirm',
    recording: null,
    analysis: null,
    scan: null,
    confirm: { commandId: command.id, startedAt: at, endsAt: at + s.settings.confirmSeconds * 1000 },
    screen: { kind: 'confirm', commandId: command.id },
  };
}

/** 카운트다운이 끝나면 보낸다 (W20이면 취소 요청). */
function finishConfirm(s, at, effects) {
  const command = commandById(s.index, s.confirm?.commandId);
  if (!command) return toIdle(s, { kind: 'welcome' });
  if (command.id === CANCEL_COMMAND_ID) return requestCancelLast(s, at, effects);
  return sendRequest(s, at, command, 'lip', effects);
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
  return { ...s, scan: scanState(at, 'commands', items, { cardId: s.scan?.cardId ?? null, reason: s.scan?.reason ?? null, detail: s.scan?.detail ?? null, categoryId }) };
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
    if (command.id === CANCEL_COMMAND_ID) return requestCancelLast(s, at, effects);
    return sendRequest(s, at, command, 'scan', effects);
  }
  // '확인 필요' 카드를 새로 만들지 않고 그 카드를 갱신한다.
  const s1 = changeCard(s, at, cardId, {
    selectedId: command.id, commandId: command.id, text: command.display, hint: hintFor(command), note: WARD_TEXT.selectedNote,
  }, effects);
  if (command.id === CANCEL_COMMAND_ID) return requestCancelLast(s1, at, effects);
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

/**
 * 분석 결과를 처리한다.
 * - 고름 + 일반 → 확인 카운트다운 / 고름 + 긴급 → 빨간 카드 즉시
 * - 고르지 못함 + 1·2등에 긴급 → '긴급 가능성, 확인 필요' 즉시 / 고르지 못함 일반 → 후보 스캐닝 + '확인 필요'
 * - 지정 문장 결과 없음 → 분류 스캐닝 + 사유 / 발화 없음·검출 실패 → 1회 환자 화면만, 연속 2회부터 간호사실에도
 */
function handleOutcome(s, at, outcome, effects) {
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
    const { command, runnerUp, accepted } = outcome;
    if (accepted) {
      return command.emergency ? sendEmergency(s0, at, command, 'lip', effects) : startConfirm(s0, at, command, effects);
    }
    if (command.emergency || runnerUp?.emergency) {
      const candidates = [candidateOf(command)];
      if (runnerUp) candidates.push(candidateOf(runnerUp));
      const { s: s1, card } = addCard(s0, at, {
        kind: 'emergency_possible', text: WARD_TEXT.emergencyPossibleCard, candidates,
      }, effects);
      return toIdle(s1, { kind: 'emergency_possible', cardId: card.id, candidates });
    }
    return startCandidateScan(s0, at, command, runnerUp, effects);
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

/** 카드가 가리키는 명령 (확인 필요 카드는 환자가 고른 것만). */
function cardCommand(s, card) {
  if (card.kind === 'check' && !card.selectedId) return null;
  return commandById(s.index, card.selectedId || card.commandId);
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

/** 지금 흐름을 모두 멈추고 이전 녹화 결과를 무효로 한다 (연습 모드 결과 주입). */
function resetFlow(s) {
  return { ...s, run: s.run + 1, phase: 'idle', recording: null, analysis: null, confirm: null, scan: null };
}

/**
 * 상태도 한 단계.
 * @param {object} state createWardState 결과.
 * @param {{type:string, at:number, [key:string]: any}} event
 *   PRESS{repeat?} · TICK · RECORDING_DONE{run} · RECORD_FAILED{run, message} · RESULT{run, results}
 *   · ANALYSIS_ERROR{run, message, reason?} · SIMULATE{results} · NURSE_ACK{cardIds, room?} · SETTINGS{settings}
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
        s = handlePress(s, at, event.repeat === true, effects);
        break;
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
        s = { ...s, settings: normalizeSettings({ ...s.settings, ...(event.settings || {}) }) };
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
      return { ...view, tone: 'confirm', headline: c?.display || '', sub: WARD_TEXT.confirmSub(sec) };
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
        return { ...view, tone: 'possible', label: '[긴급 가능성]', headline: WARD_TEXT.emergencyPossible, sub: card?.acked ? WARD_TEXT.acked : null, candidates: sc.candidates || [] };
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
      case 'error':
        return { ...view, tone: 'warning', headline: WARD_TEXT.errorHeadline, sub: sc.message || null };
      default:
        return { ...view, headline: WARD_TEXT.welcome, sub: WARD_TEXT.welcomeSub(s.settings.recordSeconds) };
    }
  } catch {
    return view;
  }
}
