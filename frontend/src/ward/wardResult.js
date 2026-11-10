// wardResult.js — 녹화 분석 결과(GET /api/result의 results)를 병실 판정 입력으로 바꾸고 판정표를 적용하는 순수 함수 (node --test로 검사).
// 병실 화면은 받아쓰기 글자를 쓰지 않고 지정 문장 결과(phrase)와 내 입 모양 등록 결과(enroll, 첫 결과 항목에만)만 쓴다.
// 글자 규칙으로 고를지는 서버의 phrase.accepted만, 등록 거리로 고를지는 서버의 enroll.accepted·within만 믿는다.
// 긴급 여부는 화면이 commands.json으로 판단한다(서버는 긴급을 해석하지 않는다).
//
// 판정표(decideWard)는 scripts/ward_tools/metrics.py decide()와 같다 (require_agreement=True, possible_ranks=2):
//   글자 규칙 통과(phrase.accepted) + 긴급            → emergency          (빨간 카드 즉시)
//   글자 규칙 통과 + 일반                              → request            (확인 카운트다운 뒤 전송)
//   글자 규칙 실패, 등록으로 고름(enroll.accepted) + 긴급 → emergency_possible (긴급 가능성 카드, 고른 명령이 후보 맨 앞)
//   글자 규칙 실패, 등록으로 고름 + 일반 + 글자 2등이 긴급(하한이 있으면 그 점수 이상)
//                                                       → emergency_possible (고른 명령이 후보 맨 앞, 등록 없이도 나던 알림을 없애지 않음)
//   글자 규칙 실패, 등록으로 고름 + 일반 (그 밖)        → request            (확인 카운트다운, 글자로 고른 것과 같게)
//   그 밖: 글자 1·2등에 긴급(하한이 있으면 그 점수 이상) 또는 거리 안의 등록 1등이 긴급 → emergency_possible
//   그 밖                                               → not_chosen         (후보 스캐닝 + 확인 필요)
// 하한(phrase.rule.possible_min_score)이 없으면(옛 서버 포함) 점수와 관계없이 1·2등에 긴급이 있으면 긴급 가능성이다(오늘과 같음).
// 1등 점수는 phrase.score, 2등 점수는 phrase.score − phrase.margin이다.

import { DETECTION_FAILED_TEXT, NO_SPEECH_TEXT } from '../results/markers.js';
import { WARD_PHRASE_SET, commandById } from './wardCommands.js';

/**
 * 결과의 phrase.list가 지금 색인의 목록과 맞는지.
 * - 기본 목록 색인(customList 없음): phrase.list가 없어야 한다(서버는 command_list 없는 요청에 list를 붙이지 않는다).
 * - 내 목록 색인: phrase.list.id와 sha256이 불러온 목록과 같아야 한다(목록을 고친 뒤 온 결과는 쓰지 않는다).
 * @param {any} phraseList 서버 phrase.list.
 * @param {{id:string, sha256:string|null}|null|undefined} customList 색인의 customList.
 * @returns {boolean}
 */
export function phraseListMatches(phraseList, customList) {
  try {
    if (!customList) return phraseList === undefined || phraseList === null;
    if (!phraseList || typeof phraseList !== 'object') return false;
    return typeof customList.sha256 === 'string' && customList.sha256.length > 0
      && phraseList.id === customList.id && phraseList.sha256 === customList.sha256;
  } catch {
    return false;
  }
}

/** 지정 문장 결과가 없을 때의 사유 → 환자 화면 문구. */
export const NO_PHRASE_REASON_TEXT = Object.freeze({
  empty: '분석 결과가 비어 있어요',
  multi_window: '녹화가 길어 여러 구간으로 나뉘었어요',
  partial_detection: '일부 구간에서 입이 안 보였어요',
  unaligned: '말이 너무 짧거나 목록 문장과 맞춰 볼 수 없었어요',
  other_set: '병실 명령 목록으로 분석되지 않았어요',
  error: '분석하지 못했어요',
  timeout: '분석이 너무 오래 걸렸어요',
  not_candidates: '후보 중에 없다고 고르셨어요',
});

/** 판정 결과 종류 (metrics.py의 OUT_* 와 같은 이름). */
export const DECISION_OUTCOMES = Object.freeze([
  'emergency', 'request', 'emergency_possible', 'not_chosen', 'no_result', 'no_speech', 'lip_fail',
]);

/** 등록 결과 상태 (서버 enroll.status, match 모드). 'ok'일 때만 거리를 쓴다. */
export const ENROLL_MATCH_STATUSES = Object.freeze(['ok', 'missing', 'incomplete', 'stale', 'no_phrase', 'error']);

/**
 * 결과 항목 하나가 마커(발화 없음·검출 실패)인지.
 * @param {{text?: string}|null|undefined} item
 * @returns {'no_speech'|'not_detected'|null}
 */
function markerKind(item) {
  const text = String(item?.text ?? '').trim();
  if (text === NO_SPEECH_TEXT) return 'no_speech';
  if (text === DETECTION_FAILED_TEXT) return 'not_detected';
  return null;
}

/** 유한한 숫자면 그 값, 아니면 null. */
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/**
 * 서버 phrase.rule (병실 묶음에만 있다)을 읽는다. 없으면 null(옛 서버).
 * @param {any} raw
 * @returns {{minMargin:number|null, minScore:number|null, possibleMinScore:number|null, enrollMaxDistance:number|null}|null}
 */
export function parseWardRule(raw) {
  try {
    if (!raw || typeof raw !== 'object') return null;
    return {
      minMargin: num(raw.min_margin),
      minScore: num(raw.min_score),
      possibleMinScore: num(raw.possible_min_score),
      enrollMaxDistance: num(raw.enroll_max_distance),
    };
  } catch {
    return null;
  }
}

/**
 * 첫 결과 항목의 enroll(match 모드)을 읽는다. store 모드·형식이 틀린 값은 null(등록 없음으로 본다).
 * 상태가 'ok'이고 등록 1등이 명령 목록 안일 때만 within이 참일 수 있다. accepted는 within이고 서버 accepted·agree가 참이며
 * 등록 1등이 글자 1등과 같을 때만 참이다(서버 값이 서로 맞지 않으면 고르지 않는 쪽으로 본다).
 * @param {any} raw results[0].enroll
 * @param {{byId: Map<string, object>}} index
 * @param {string} phraseId 글자 1등 ID.
 * @returns {{profile:string|null, status:string, count:number|null, top:object|null, distance:number|null,
 *   runnerUp:object|null, runnerUpDistance:number|null, maxDistance:number|null, within:boolean, agree:boolean,
 *   accepted:boolean}|null}
 */
export function parseEnrollMatch(raw, index, phraseId) {
  try {
    if (!raw || typeof raw !== 'object' || raw.mode !== 'match') return null;
    const status = ENROLL_MATCH_STATUSES.includes(raw.status) ? raw.status : 'error';
    const ok = status === 'ok';
    const top = ok ? commandById(index, raw.top_id) : null;
    const within = ok && !!top && raw.within === true;
    const agree = within && raw.agree === true && top.id === phraseId;
    return {
      profile: typeof raw.profile === 'string' ? raw.profile : null,
      status,
      count: Number.isInteger(raw.count) ? raw.count : null,
      top,
      distance: ok ? num(raw.distance) : null,
      runnerUp: ok ? commandById(index, raw.runner_up_id) : null,
      runnerUpDistance: ok ? num(raw.runner_up_distance) : null,
      maxDistance: num(raw.max_distance),
      within,
      agree,
      accepted: agree && raw.accepted === true,
    };
  } catch {
    return null;
  }
}

/**
 * 결과 배열을 병실 판정 입력 하나로 요약한다.
 *
 * - 결과 한 건 + 병실 묶음 phrase → {kind:'phrase', accepted, command, runnerUp, score, margin, rule, floor, enroll}.
 *   accepted는 phrase.accepted === true일 때만. rule·enroll이 없으면(옛 서버) null이고 floor도 null이다.
 * - 전부 [발화 없음] 또는 [검출 실패 구간] → {kind:'no_speech'} (입 움직임이 안 보임).
 * - 여러 창·일부 검출 실패·정렬 불가(짧음)·다른 묶음·모르는 ID·다른 목록(목록 ID·지문이 다름) → {kind:'no_phrase', reason}.
 *
 * @param {Array<object>|null|undefined} results 서버 results.
 * @param {{byId: Map<string, object>}} index buildCommandIndex 결과.
 * @returns {{kind:'phrase', accepted:boolean, command:object, runnerUp:object|null, score:number|null, margin:number|null,
 *     rule:object|null, floor:number|null, enroll:object|null}
 *   |{kind:'no_speech', reason:'no_speech'|'not_detected'}
 *   |{kind:'no_phrase', reason:string}}
 */
export function classifyWardResults(results, index) {
  try {
    const items = Array.isArray(results) ? results.filter((x) => x && typeof x === 'object') : [];
    if (!items.length) return { kind: 'no_phrase', reason: 'empty' };
    const markers = items.map(markerKind);
    if (markers.every((m) => m !== null)) {
      return { kind: 'no_speech', reason: markers.every((m) => m === 'no_speech') ? 'no_speech' : 'not_detected' };
    }
    if (items.length > 1) {
      return { kind: 'no_phrase', reason: markers.includes('not_detected') ? 'partial_detection' : 'multi_window' };
    }
    const phrase = items[0].phrase;
    if (!phrase || typeof phrase !== 'object' || typeof phrase.id !== 'string') return { kind: 'no_phrase', reason: 'unaligned' };
    // 묶음 표시는 반드시 'ward'여야 한다. 표시가 없는 옛 서버 결과나 문장 모드 결과는 병실 명령으로 해석하지 않는다.
    if (phrase.set !== WARD_PHRASE_SET) return { kind: 'no_phrase', reason: 'other_set' };
    // 내 목록은 목록 ID와 지문이 지금 불러온 목록과 같을 때만 쓴다. 기본 목록 결과에는 list가 없어야 한다.
    if (!phraseListMatches(phrase.list, index?.customList)) return { kind: 'no_phrase', reason: 'other_set' };
    const command = commandById(index, phrase.id);
    if (!command) return { kind: 'no_phrase', reason: 'other_set' };
    const runnerUp = commandById(index, phrase.runner_up_id);
    const rule = parseWardRule(phrase.rule);
    return {
      kind: 'phrase',
      accepted: phrase.accepted === true,
      command,
      runnerUp: runnerUp && runnerUp.id !== command.id ? runnerUp : null,
      score: num(phrase.score),
      margin: num(phrase.margin),
      rule,
      floor: rule ? rule.possibleMinScore : null,
      enroll: parseEnrollMatch(items[0].enroll, index, command.id),
    };
  } catch {
    return { kind: 'no_phrase', reason: 'empty' };
  }
}

/**
 * 긴급 후보 점수가 하한 이상인지. 하한이 없으면 참(점수와 관계없이 긴급 가능성).
 * 하한이 있는데 점수를 모르면(형식이 틀린 결과) 놓치지 않도록 참으로 본다.
 * @param {number|null} value
 * @param {number|null} floor
 * @returns {boolean}
 */
function atOrAboveFloor(value, floor) {
  if (floor === null) return true;
  if (value === null) return true;
  return value >= floor;
}

/** 명령 목록에서 같은 ID를 한 번씩만 남긴다 (순서 유지). */
function dedupeCommands(list) {
  const seen = new Set();
  const out = [];
  for (const c of list) {
    if (c && !seen.has(c.id)) {
      seen.add(c.id);
      out.push(c);
    }
  }
  return out;
}

/**
 * 판정표를 적용한다 (metrics.py decide()와 같은 결과, 위 머리말 표 참고).
 *
 * @param {ReturnType<typeof classifyWardResults>} outcome
 * @returns {{outcome:string, chosenId:string|null, via:'text'|'enroll'|null, textPass:boolean, conflict:boolean,
 *   enrollAlarm:boolean, textAlarm:boolean, command:object|null, chosen:object|null, candidates:Array<object>,
 *   reason:string|null}}
 *   enrollAlarm은 metrics와 같게 '등록 1등 때문에만' 긴급 가능성이 된 경우에만 참이다. candidates는 긴급 가능성·후보 스캐닝에
 *   쓸 명령 목록이다(긴급 가능성: 고른 명령 또는 글자 1등, 2등, 거리 안의 긴급 등록 1등).
 */
export function decideWard(outcome) {
  const base = {
    outcome: 'no_result', chosenId: null, via: null, textPass: false, conflict: false, enrollAlarm: false, textAlarm: false,
    command: null, chosen: null, candidates: [], reason: null,
  };
  try {
    if (!outcome || typeof outcome !== 'object') return { ...base, reason: 'empty' };
    if (outcome.kind === 'no_speech') {
      return { ...base, outcome: outcome.reason === 'no_speech' ? 'no_speech' : 'lip_fail', reason: outcome.reason };
    }
    if (outcome.kind !== 'phrase' || !outcome.command) return { ...base, reason: outcome.reason || 'empty' };
    const { command, runnerUp, enroll } = outcome;
    const floor = num(outcome.floor);
    const score = num(outcome.score);
    const margin = num(outcome.margin);
    const textPass = outcome.accepted === true;
    const out = { ...base, command, textPass };
    if (textPass) {
      return {
        ...out, outcome: command.emergency ? 'emergency' : 'request', chosenId: command.id, via: 'text', chosen: command,
        candidates: dedupeCommands([command, runnerUp]),
      };
    }
    const within = !!(enroll && enroll.within && enroll.top);
    const rank2 = score !== null && margin !== null ? score - margin : null;
    if (enroll && enroll.accepted && within && enroll.top.id === command.id) {
      const chosen = enroll.top;
      // 등록으로 고른 일반 명령이라도 글자 2등(하한 이상)이 긴급이면 등록이 없을 때처럼 긴급 가능성으로 둔다(알림을 없애지 않음).
      const runnerAlarm = !!runnerUp?.emergency && atOrAboveFloor(rank2, floor);
      return {
        ...out, outcome: (chosen.emergency || runnerAlarm) ? 'emergency_possible' : 'request', chosenId: chosen.id, via: 'enroll',
        chosen, textAlarm: !chosen.emergency && runnerAlarm, candidates: dedupeCommands([chosen, runnerUp]),
      };
    }
    const conflict = within && enroll.top.id !== command.id;
    const textAlarm = (command.emergency && atOrAboveFloor(score, floor))
      || (!!runnerUp?.emergency && atOrAboveFloor(rank2, floor));
    const enrollAlarmRaw = within && enroll.top.emergency === true;
    if (textAlarm || enrollAlarmRaw) {
      return {
        ...out, outcome: 'emergency_possible', conflict, textAlarm, enrollAlarm: enrollAlarmRaw && !textAlarm,
        candidates: dedupeCommands([command, runnerUp, enrollAlarmRaw ? enroll.top : null]),
      };
    }
    return { ...out, outcome: 'not_chosen', conflict, candidates: dedupeCommands([command, runnerUp]) };
  } catch {
    return { ...base, reason: 'error' };
  }
}
