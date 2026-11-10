// wardEnroll.js — 내 입 모양 등록 화면(#enroll)의 순수 함수와 상태도 (node --test로 검사, JSX·DOM·시계 없음).
// 등록은 환자마다 명령 20개를 소리 내어 한 번씩 찍어 서버(POST /api/record, phrase_set='ward', enroll_profile·enroll_command)에
// 저장하는 일이다. 저장한 입 모양(인식 모델의 숨은 특징)은 서버의 등록 폴더에만 남고 화면·저장소에는 남기지 않는다.
// 병실 화면은 20개를 모두 찍고 명령 목록·모델이 그대로일 때(status 'ok')만 등록 거리를 판정에 쓴다.

import { commandById } from './wardCommands.js';
import { ENROLL_PROFILE_PATTERN, isPracticeMode, isValidProfileId, normalizeProfileId } from './wardSettings.js';

/** 등록 녹화 길이(초): 병실 명령 촬영·병실 화면 기본과 같은 5초 자동 정지. */
export const ENROLL_RECORD_SECONDS = 5;
/** 결과를 받은 직후 이 시간(ms) 동안은 스페이스바로 다음 녹화를 시작하지 않는다 (습관처럼 누른 키). */
export const ENROLL_START_COOLDOWN_MS = 1000;
/** 등록 상태 (서버 GET /api/ward/profiles/{id}의 status). */
export const PROFILE_STATUSES = Object.freeze(['ok', 'missing', 'incomplete', 'stale']);
/** 저장하지 못한 사유 (서버 enroll.reason, store 모드). */
export const STORE_REASONS = Object.freeze(['multi_window', 'partial_detection', 'no_speech', 'no_phrase', 'no_features', 'error']);

export { ENROLL_PROFILE_PATTERN, isValidProfileId, normalizeProfileId };

export const ENROLL_TEXT = Object.freeze({
  title: '내 입 모양 등록',
  intro: '명령 20개를 한 번씩 찍어 두면, 병실 화면이 글자만으로 고르지 못할 때 내 입 모양과 비교해서 고를 수 있어요.',
  privacy: '등록 이름은 실명 대신 별명으로 정해 주세요 (영문·숫자·_·-, 32자까지, 예: bed305).',
  profileLabel: '등록 이름',
  start: '등록 시작',
  existing: '이미 있는 등록',
  loading: '등록 상태를 불러오는 중이에요',
  sayThis: '이 문장을 말하세요',
  howTo: '소리 내어 평소처럼 한 번 말하세요',
  shownAs: (display) => `화면과 간호사실에는 “${display}”로 보여요`,
  record: (sec) => `녹화 시작 (스페이스바 · ${sec}초)`,
  recording: (sec) => `● 녹화 중 · ${sec}초 뒤 저절로 멈춰요`,
  analyzing: '저장하는 중이에요. 잠시만 기다려 주세요',
  retake: '다시 찍기',
  next: '다음 명령으로',
  backToWard: '병실 화면으로',
  otherProfile: '다른 등록 이름',
  progress: (count, total) => `등록 ${count}/${total}`,
  stored: (id) => `${id} 저장했어요`,
  mismatch: (id, top) => `${id} 저장은 했지만 ${top ? `“${top.display}”(${top.id}) 문장으로` : '다른 문장으로'} 알아들었어요. `
    + '화면의 큰 문장을 또박또박 다시 말해 찍어 주세요',
  doneTitle: '등록을 마쳤어요',
  doneSub: (id) => `병실 화면에서 이 등록(${id})을 씁니다`,
  doneNotInUse: (id) => `이 등록(${id})은 아직 병실 화면에서 쓰지 않아요`,
  useThis: '병실 화면에서 이 등록 쓰기',
  useAnyway: '그래도 이 등록 쓰기',
  inUse: (id) => `병실 화면이 이 등록(${id})을 씁니다`,
  uncertainTitle: (n) => `확인 필요 ${n}개: 다른 문장으로 알아들은 명령이 있어요`,
  uncertainSub: '다시 찍기를 권해요. 그대로 쓰면 이 명령은 틀린 입 모양과 비교될 수 있어요',
  uncertainItem: (c, heardAs) => `${c.id} “${c.display}”${c.emergency ? ' [긴급]' : ''} → `
    + `${heardAs ? `“${heardAs.display}”(${heardAs.id})로` : '다른 문장으로'} 알아들음`,
  pickAgain: '다시 찍을 명령은 아래 목록에서 고르세요',
  stale: '명령 목록이나 인식 모델이 바뀌어 이 등록은 지금 쓰지 않아요. 지우고 처음부터 다시 등록해 주세요',
  deleteAndRestart: '이 등록 지우고 처음부터',
  deleteConfirm: (id) => `등록 “${id}”를 지울까요? 찍어 둔 입 모양이 모두 지워지고 되돌릴 수 없어요.`,
  cameraFirst: '카메라가 준비되면 녹화할 수 있어요',
  unsupported: '서버가 입 모양 등록 결과를 보내지 않았어요. 서버를 최신으로 다시 켜 주세요',
  loadFailed: (msg) => `등록 상태를 가져오지 못했어요${msg ? `: ${msg}` : ''}`,
  badProfile: '등록 이름은 영문·숫자·_·- 1~32자로 적어 주세요',
  timeout: '서버가 아직 처리 중일 수 있어요. 잠시 뒤 목록을 확인하고, 아직이면 다시 찍어 주세요',
  staleNoRecord: '이 등록은 다시 찍을 수 없어요. 먼저 “이 등록 지우고 처음부터”를 눌러 주세요',
});

/** 저장하지 못한 사유 → 안내 문구. */
export const STORE_REASON_TEXT = Object.freeze({
  multi_window: '녹화가 여러 구간으로 나뉘었어요. 한 번만 이어서 말해 주세요',
  partial_detection: '중간에 입이 안 보였어요. 얼굴을 화면 가운데에 두고 다시 찍어 주세요',
  no_speech: '입 움직임이 안 보였어요. 입을 크게 움직여 다시 말해 주세요',
  no_phrase: '말이 너무 짧거나 명령 문장과 맞춰 볼 수 없었어요. 문장을 끝까지 말해 주세요',
  no_features: '입 모양 정보를 저장하지 못했어요. 다시 찍어 주세요',
  error: '저장하지 못했어요. 다시 찍어 주세요',
});

/**
 * 주소 해시에서 등록 이름을 읽는다 (#enroll?profile=bed305). 없거나 규칙에 맞지 않으면 ''.
 * @param {string|null|undefined} hash
 * @returns {string}
 */
export function profileFromHash(hash) {
  try {
    const h = String(hash || '');
    const q = h.includes('?') ? h.slice(h.indexOf('?') + 1) : '';
    return normalizeProfileId(new URLSearchParams(q).get('profile'));
  } catch {
    return '';
  }
}

/**
 * 등록 화면 주소 해시. practice면 연습 모드 표시(practice=1)를 함께 실어 병실 화면으로 돌아갈 때 되살린다.
 * @param {string} [profile]
 * @param {{practice?: boolean}} [opts]
 * @returns {string}
 */
export function enrollHash(profile = '', { practice = false } = {}) {
  try {
    const id = normalizeProfileId(profile);
    const q = new URLSearchParams();
    if (id) q.set('profile', id);
    if (practice) q.set('practice', '1');
    const text = q.toString();
    return text ? `#enroll?${text}` : '#enroll';
  } catch {
    return '#enroll';
  }
}

/**
 * 등록 화면에서 돌아갈 병실 화면 해시 (연습 모드로 왔으면 #ward?practice=1).
 * @param {{hash?: string, search?: string}|null|undefined} loc 지금 주소.
 * @returns {string}
 */
export function wardReturnHash(loc) {
  try {
    const hash = String(loc?.hash || '');
    const q = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : '';
    return new URLSearchParams(q).get('practice') === '1' ? '#ward?practice=1' : '#ward';
  } catch {
    return '#ward';
  }
}

export { isPracticeMode };

/**
 * 서버 등록 상태(GET /api/ward/profiles/{id})를 화면용으로 읽는다.
 * @param {any} data 서버 응답.
 * @param {{list: Array<object>}} index 명령 색인.
 * @param {string} [profile] 응답에 이름이 없을 때 쓸 이름.
 * @returns {{profile:string, count:number, total:number, complete:boolean, status:string,
 *   done: Record<string, {textTop:string|null, matches:boolean|null, created:string|null}>}}
 */
export function parseProfileInfo(data, index, profile = '') {
  const total = index?.list?.length || 0;
  try {
    const src = data && typeof data === 'object' ? data : {};
    const done = {};
    const templates = src.templates && typeof src.templates === 'object' ? src.templates : {};
    for (const [cid, t] of Object.entries(templates)) {
      if (!commandById(index, cid)) continue;
      done[cid] = {
        textTop: typeof t?.text_top === 'string' ? t.text_top : null,
        matches: typeof t?.text_top_matches === 'boolean' ? t.text_top_matches : null,
        created: typeof t?.created === 'string' ? t.created : null,
      };
    }
    const count = Number.isInteger(src.count) ? src.count : Object.keys(done).length;
    const status = PROFILE_STATUSES.includes(src.status) ? src.status : (count >= total && total > 0 ? 'ok' : 'incomplete');
    return {
      profile: typeof src.profile === 'string' && src.profile ? src.profile : profile,
      count,
      total,
      complete: src.complete === true || (src.complete === undefined && count >= total && total > 0),
      status,
      done,
    };
  } catch {
    return { profile, count: 0, total, complete: false, status: 'missing', done: {} };
  }
}

/**
 * 등록을 병실 판정에 쓸 수 있는지 (20/20, 명령 목록·모델 그대로).
 * @param {{complete:boolean, status:string}|null|undefined} info
 * @returns {boolean}
 */
export function profileUsable(info) {
  return !!info && info.complete === true && info.status === 'ok';
}

/**
 * 다음에 찍을(아직 없는) 명령 ID. afterId 다음부터 목록 순서로 찾고 끝에 닿으면 처음으로 돌아간다. 모두 있으면 null.
 * @param {{list: Array<{id:string}>}} index
 * @param {Record<string, any>} done 찍은 명령 ID → 정보.
 * @param {string|null} [afterId]
 * @returns {string|null}
 */
export function nextMissingId(index, done, afterId = null) {
  try {
    const ids = (index?.list || []).map((c) => c.id);
    const have = done && typeof done === 'object' ? done : {};
    const start = afterId && ids.includes(afterId) ? ids.indexOf(afterId) + 1 : 0;
    for (let i = 0; i < ids.length; i += 1) {
      const id = ids[(start + i) % ids.length];
      if (id !== afterId && !have[id]) return id;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * 등록 저장 결과(results[0].enroll, store 모드)를 읽는다.
 * @param {Array<object>|null|undefined} results 서버 results.
 * @param {string} commandId 찍은 명령.
 * @param {{byId: Map<string, object>}} [index] 명령 색인 (인식 문장이 다를 때 그 명령 문장을 보이려고 쓴다).
 * @returns {{kind:'stored'|'failed'|'unsupported', commandId:string, reason:string|null, textTop:string|null,
 *   matches:boolean, count:number|null, complete:boolean|null, message:string}}
 */
export function storeOutcome(results, commandId, index = null) {
  const base = { commandId, reason: null, textTop: null, matches: false, count: null, complete: null };
  try {
    const first = Array.isArray(results) ? results.find((x) => x && typeof x === 'object') : null;
    const e = first?.enroll;
    if (!e || typeof e !== 'object' || e.mode !== 'store') {
      return { ...base, kind: 'unsupported', message: ENROLL_TEXT.unsupported };
    }
    const count = Number.isInteger(e.count) ? e.count : null;
    const complete = typeof e.complete === 'boolean' ? e.complete : null;
    const textTop = typeof e.text_top === 'string' ? e.text_top : null;
    if (typeof e.command === 'string' && e.command !== commandId) {
      return { ...base, kind: 'failed', reason: 'error', count, complete, textTop, message: STORE_REASON_TEXT.error };
    }
    if (e.stored === true) {
      const matches = e.text_top_matches === true;
      return {
        ...base, kind: 'stored', textTop, matches, count, complete,
        message: matches ? ENROLL_TEXT.stored(commandId) : ENROLL_TEXT.mismatch(commandId, commandById(index, textTop)),
      };
    }
    const reason = STORE_REASONS.includes(e.reason) ? e.reason : 'error';
    return { ...base, kind: 'failed', reason, textTop, count, complete, message: STORE_REASON_TEXT[reason] };
  } catch {
    return { ...base, kind: 'failed', reason: 'error', message: STORE_REASON_TEXT.error };
  }
}

/**
 * 병실 화면·설정 창의 등록 상태 한 줄. 등록 이름이 없으면 null.
 * @param {string} profile 설정의 등록 이름.
 * @param {{count:number, total:number, complete:boolean, status:string}|null} info 서버 상태(불러오는 중이면 null).
 * @param {{loadError?: boolean}} [opts]
 * @returns {{text:string, tone:'ok'|'warn'|'neutral'}|null}
 */
export function enrollStatusLine(profile, info, { loadError = false } = {}) {
  try {
    if (!profile) return null;
    const head = `내 입 모양 등록: ${profile}`;
    if (loadError) return { text: `${head} — 서버에서 상태를 확인하지 못했어요`, tone: 'warn' };
    if (!info) return { text: `${head} (확인 중)`, tone: 'neutral' };
    const frac = `${info.count}/${info.total}`;
    if (profileUsable(info)) return { text: `${head} (${frac})`, tone: 'ok' };
    if (info.status === 'stale') {
      return { text: `${head} (${frac}) — 명령 목록이나 인식 모델이 바뀌어 지금은 쓰지 않아요. 다시 등록해 주세요`, tone: 'warn' };
    }
    if (info.status === 'missing') return { text: `${head} — 등록 기록이 없어요. 설정에서 등록해 주세요`, tone: 'warn' };
    return { text: `${head} (${frac}) — 다 찍지 않아 아직 쓰지 않아요`, tone: 'warn' };
  } catch {
    return null;
  }
}

/**
 * 등록 목록(GET /api/ward/profiles)에서 고를 이름들 (이름순, 지금 설정한 이름이 목록에 없어도 넣는다).
 * @param {any} data 서버 응답 {profiles:[{profile,...}]}.
 * @param {string} [current]
 * @returns {Array<{profile:string, count:number|null, complete:boolean, status:string|null}>}
 */
export function profileChoices(data, current = '') {
  try {
    const list = Array.isArray(data?.profiles) ? data.profiles : [];
    const byId = new Map();
    for (const p of list) {
      if (!p || !isValidProfileId(p.profile)) continue;
      byId.set(p.profile, {
        profile: p.profile,
        count: Number.isInteger(p.count) ? p.count : null,
        complete: p.complete === true,
        status: typeof p.status === 'string' ? p.status : null,
      });
    }
    if (isValidProfileId(current) && !byId.has(current)) byId.set(current, { profile: current, count: null, complete: false, status: null });
    return [...byId.values()].sort((a, b) => a.profile.localeCompare(b.profile));
  } catch {
    return [];
  }
}

/**
 * '확인 필요' 명령 (저장했지만 인식 1등이 다른 명령이었던 템플릿), 목록 순서.
 * @param {{list: Array<object>}} index
 * @param {{done: Record<string, {matches:boolean|null, textTop:string|null}>}|null} info
 * @returns {Array<{command:object, heardAs:object|null}>}
 */
export function uncertainCommands(index, info) {
  try {
    const done = info?.done || {};
    return (index?.list || [])
      .filter((c) => done[c.id] && done[c.id].matches === false)
      .map((c) => ({ command: c, heardAs: commandById(index, done[c.id].textTop) }));
  } catch {
    return [];
  }
}

/**
 * 끝 화면에서 병실 화면 설정의 등록 이름을 저절로 이 등록으로 바꿀지.
 * 이번 화면에서 실제로 찍어(저장) 20/20을 마쳤고 '확인 필요'가 없을 때만이다. 이미 끝난 등록을 열어 보기만 했거나
 * '확인 필요'가 있으면 사람이 [병실 화면에서 이 등록 쓰기]·[그래도 이 등록 쓰기]를 눌러야 바꾼다.
 * @param {{phase:string, storedHere?:boolean, info:object|null, index:object}} state
 * @returns {boolean}
 */
export function shouldAutoUseProfile(state) {
  try {
    return !!state && state.phase === 'done' && state.storedHere === true && profileUsable(state.info)
      && uncertainCommands(state.index, state.info).length === 0;
  } catch {
    return false;
  }
}

/**
 * 명령 목록 표시 행 (찍음·확인 필요·지금 찍는 것).
 * @param {{list: Array<object>}} index
 * @param {{done: Record<string, {matches:boolean|null}>}|null} info
 * @param {string|null} currentId
 * @returns {Array<{id:string, text:string, display:string, emergency:boolean, done:boolean, warn:boolean, current:boolean}>}
 */
export function commandRows(index, info, currentId) {
  try {
    const done = info?.done || {};
    return (index?.list || []).map((c) => ({
      id: c.id, text: c.text, display: c.display, emergency: c.emergency,
      done: !!done[c.id], warn: !!done[c.id] && done[c.id].matches === false, current: c.id === currentId,
    }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// 등록 화면 상태도
// ---------------------------------------------------------------------------

/**
 * 처음 상태.
 * @param {{index: object, profile?: string}} opts
 * @returns {object}
 *   phase: choose(이름 고르기) | loading | ready(찍을 준비) | recording | analyzing | review(저장했지만 인식 문장이 다름) | done
 */
export function createEnrollState({ index, profile = '' } = {}) {
  const id = normalizeProfileId(profile);
  return {
    index,
    phase: id ? 'loading' : 'choose',
    profile: id,
    info: null,
    currentId: null,
    recordingId: null, // 지금 찍는(결과를 기다리는) 명령
    last: null, // 마지막 저장 결과 storeOutcome 또는 {kind:'failed', message}
    error: null, // 이름 고르기·불러오기 오류 문구
    storedHere: false, // 이번 화면에서 이 이름으로 하나라도 저장했는지 (끝 화면 자동 사용 조건)
  };
}

/** 찍은 명령 하나를 상태에 반영한다 (서버가 준 개수·완료 여부를 우선). */
function markStored(info, index, cid, out) {
  const base = info || parseProfileInfo(null, index);
  // 명령 목록·모델이 바뀐 등록(stale)에 저장하면 서버는 예전 템플릿을 모두 지우고 이 하나부터 새로 시작한다.
  const kept = base.status === 'stale' ? {} : base.done;
  const done = { ...kept, [cid]: { textTop: out.textTop, matches: out.matches, created: null } };
  const count = Number.isInteger(out.count) ? out.count : Object.keys(done).length;
  const complete = typeof out.complete === 'boolean' ? out.complete : count >= base.total;
  return { ...base, done, count, complete, status: complete ? 'ok' : 'incomplete' };
}

/** 첫 화면에서 찍을 명령: 빠진 것 중 첫째, 모두 있으면 첫 명령. */
function firstCommandId(index, info) {
  return nextMissingId(index, info?.done || {}, null) || index?.list?.[0]?.id || null;
}

/**
 * 등록 화면 상태도 한 단계.
 * @param {object} state createEnrollState 결과.
 * @param {{type:string, [key:string]: any}} event
 *   CHOOSE{profile} · PROFILE_LOADED{data} · PROFILE_FAILED{message} · SELECT{commandId} · RECORD_START · RECORD_DONE
 *   · RECORD_FAILED{message} · STORE_RESULT{results} · ANALYSIS_ERROR{message} · RETAKE · NEXT · RESET
 * @returns {object} 바뀐 것이 없으면 같은 객체.
 */
export function stepEnroll(state, event) {
  try {
    if (!state || !event || typeof event.type !== 'string') return state;
    const s = state;
    const { index } = s;
    switch (event.type) {
      case 'CHOOSE': {
        if (!['choose', 'ready', 'review', 'done'].includes(s.phase)) return s;
        const id = normalizeProfileId(event.profile);
        if (!id) return { ...s, error: ENROLL_TEXT.badProfile };
        return { ...createEnrollState({ index, profile: id }) };
      }
      case 'PROFILE_LOADED': {
        const info = parseProfileInfo(event.data, index, s.profile);
        if (s.phase === 'loading') {
          return {
            ...s, info, error: null, currentId: firstCommandId(index, info), phase: profileUsable(info) ? 'done' : 'ready',
          };
        }
        if (s.phase === 'choose') return s;
        let phase = s.phase;
        if (phase === 'done' && !profileUsable(info)) phase = 'ready';
        let currentId = s.currentId || firstCommandId(index, info);
        // 방금 저장·일치한 명령에 머물러 있으면(서버가 개수를 늦게 알려 준 경우) 다음 빈 명령으로 넘어간다.
        if (phase === 'ready' && currentId && info.done[currentId] && s.last?.kind === 'stored' && s.last.matches
            && s.last.commandId === currentId) {
          currentId = nextMissingId(index, info.done, currentId) || currentId;
        }
        return { ...s, info, phase, currentId };
      }
      case 'PROFILE_FAILED':
        if (s.phase !== 'loading') return s;
        return { ...s, phase: 'choose', error: ENROLL_TEXT.loadFailed(event.message || '') };
      case 'SELECT': {
        if (!['ready', 'review', 'done'].includes(s.phase) || !commandById(index, event.commandId)) return s;
        return { ...s, phase: 'ready', currentId: event.commandId, last: null };
      }
      case 'RECORD_START':
        // 다시 등록 필요(stale)는 찍지 않는다: 하나를 찍으면 서버가 나머지 템플릿을 모두 지운다. 먼저 지우고 처음부터 한다.
        if (!['ready', 'review'].includes(s.phase) || !s.currentId || s.info?.status === 'stale') return s;
        return { ...s, phase: 'recording', recordingId: s.currentId, last: null };
      case 'RECORD_DONE':
        if (s.phase !== 'recording') return s;
        return { ...s, phase: 'analyzing' };
      case 'RECORD_FAILED':
        if (!['recording', 'analyzing'].includes(s.phase)) return s;
        return {
          ...s, phase: 'ready', recordingId: null,
          last: { kind: 'failed', commandId: s.recordingId, reason: 'record', message: String(event.message || STORE_REASON_TEXT.error) },
        };
      case 'ANALYSIS_ERROR':
        if (s.phase !== 'analyzing') return s;
        return {
          ...s, phase: 'ready', recordingId: null,
          last: { kind: 'failed', commandId: s.recordingId, reason: 'error', message: String(event.message || STORE_REASON_TEXT.error) },
        };
      case 'STORE_RESULT': {
        if (s.phase !== 'analyzing' || !s.recordingId) return s;
        const cid = s.recordingId;
        const out = storeOutcome(event.results, cid, index);
        if (out.kind !== 'stored') return { ...s, phase: 'ready', currentId: cid, recordingId: null, last: out };
        const info = markStored(s.info, index, cid, out);
        const t = { ...s, info, recordingId: null, last: out, storedHere: true };
        if (!out.matches) return { ...t, phase: 'review', currentId: cid };
        if (profileUsable(info)) return { ...t, phase: 'done', currentId: cid };
        const next = nextMissingId(index, info.done, cid);
        return { ...t, phase: 'ready', currentId: next || cid };
      }
      case 'RETAKE':
        if (!['ready', 'review'].includes(s.phase)) return s;
        return { ...s, phase: 'ready', last: null };
      case 'NEXT': {
        if (!['ready', 'review'].includes(s.phase)) return s;
        if (profileUsable(s.info)) return { ...s, phase: 'done', last: null };
        const next = nextMissingId(index, s.info?.done || {}, s.currentId);
        return { ...s, phase: 'ready', currentId: next || s.currentId, last: null };
      }
      case 'RESET':
        if (['recording', 'analyzing'].includes(s.phase)) return s;
        return createEnrollState({ index });
      default:
        return s;
    }
  } catch {
    return state;
  }
}

/**
 * 스페이스바·녹화 버튼이 지금 녹화를 시작할 수 있는지.
 * @param {{phase:string, currentId:string|null}} state
 * @param {{cameraReady:boolean, coolingDown:boolean}} opts
 * @returns {boolean}
 */
export function canStartEnrollRecording(state, { cameraReady, coolingDown }) {
  return !!state && ['ready', 'review'].includes(state.phase) && !!state.currentId && cameraReady === true && coolingDown !== true
    && state.info?.status !== 'stale';
}
