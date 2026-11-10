// wardLists.js — '내 목록'(사용자가 만든 병실 명령 목록) 화면(#lists)의 순수 함수와 편집 상태도 (node --test로 검사, JSX·DOM 없음).
// 검사의 기준은 서버(backend/services/ward_lists.py)다. 여기의 즉시 검사(한글만·길이)는 입력하는 동안 바로 보여 주는 안내일 뿐이고,
// 화면은 언제나 서버 검사 결과(POST /api/ward/lists/check, PUT 400)를 함께 보인다.
// 병실 화면·등록 화면이 내 목록을 불러올 때 쓰는 함수(listLoadOutcome·listLoadErrorMessage·listBannerText)도 여기에 둔다.

import { buildCustomCommandIndex, CANCEL_ACTION, EMERGENCY_CATEGORY } from './wardCommands.js';

/** 서버 목록 형식 이름. */
export const LIST_SCHEMA = 'ward-custom-list-v1';
/** 예시 목록 파일 형식 이름 (frontend/src/ward/custom_list_examples.json). */
export const EXAMPLES_SCHEMA = 'ward-custom-list-examples-v1';
/** 목록 ID 규칙 (서버와 같다). */
export const LIST_ID_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
/** 서버에 둘 수 있는 내 목록 수. */
export const MAX_LISTS = 20;
export const MIN_COMMANDS = 4;
export const MAX_COMMANDS = 20;
export const TEXT_MIN_CHARS = 2;
export const TEXT_MAX_CHARS = 40;
export const TEXT_MAX_TOKENS = 50;
export const DISPLAY_MAX_CHARS = 30;
export const TITLE_MAX_CHARS = 30;
/** 잘 읽히는 길이 (서버 주의 기준과 같다): 9~13글자, 입 모양 길이 23~34. */
export const GOOD_SYLLABLES = Object.freeze([9, 13]);
export const GOOD_TOKENS = Object.freeze([23, 34]);
/** 새 줄의 기본 종류. */
export const DEFAULT_ROW_CATEGORY = 'call';
/** 새 목록을 열 때 처음 넣는 빈 줄 수 (최소 개수와 같다). */
export const NEW_LIST_ROWS = MIN_COMMANDS;
/** Windows에서 파일 이름으로 쓸 수 없는 이름 (서버도 거절한다). */
export const RESERVED_LIST_IDS = Object.freeze([
  'CON', 'PRN', 'AUX', 'NUL',
  ...Array.from({ length: 10 }, (_, i) => `COM${i}`),
  ...Array.from({ length: 10 }, (_, i) => `LPT${i}`),
]);

/**
 * 목적격 조사 '을'/'를'. 마지막 글자가 받침 있는 한글이면 '을', 받침 없는 한글이면 '를', 한글로 끝나지 않으면 '을(를)'.
 * @param {any} word
 * @returns {'을'|'를'|'을(를)'}
 */
export function objectParticle(word) {
  try {
    const chars = [...String(word ?? '').trim()];
    const code = chars.length ? chars[chars.length - 1].codePointAt(0) - 0xac00 : -1;
    if (code < 0 || code > 11171) return '을(를)';
    return code % 28 === 0 ? '를' : '을';
  } catch {
    return '을(를)';
  }
}

/** “이름”을 / “이름”를 (조사를 이름에 맞춘다). */
function quotedObject(title) {
  return `“${title}”${objectParticle(title)}`;
}

export const LISTS_TEXT = Object.freeze({
  title: '내 명령 목록',
  intro: '직접 만든 문장으로 병실 화면을 쓸 수 있어요. 기본 목록은 그대로 남고, 고른 목록만 병실 화면에 쓰여요.',
  untested: '성능을 측정하지 않은 목록',
  defaultName: '기본 목록',
  defaultReadOnly: '기본 목록은 바꿀 수 없어요 (측정한 기준값이 이 목록에 맞춰져 있어요)',
  myLists: '내 목록',
  newList: '새 목록 만들기',
  noLists: '아직 만든 목록이 없어요',
  loadListsFailed: '서버에서 목록을 가져오지 못했어요',
  inUse: '사용 중',
  usingNow: (name) => `지금 병실 화면이 쓰는 목록: ${name}`,
  backToDefault: '기본 목록으로 되돌리기',
  backToWard: '병실 화면으로',
  listTitle: '목록 이름',
  listId: '목록 ID',
  listIdHint: '영문·숫자·_·- 32자까지 (예: rehab). 저장한 뒤에는 바꿀 수 없어요',
  colText: '입 모양 문장',
  colTextHint: '소리 없이 입 모양으로 말할 문장 · 한글과 띄어쓰기만 · 9~13글자가 좋아요',
  colDisplay: '화면 문장',
  colDisplayHint: '화면과 간호사실에 보일 짧은 존댓말',
  colCategory: '종류',
  colEmergency: '긴급',
  colCancel: '취소',
  colCancelHint: '“방금 요청 취소” 명령 (한 줄만)',
  addRow: '줄 추가',
  removeRow: '줄 지우기',
  moveUp: '위로',
  moveDown: '아래로',
  save: '저장',
  saving: '저장하는 중이에요',
  saved: '저장했어요',
  checking: '검사하는 중이에요',
  checkFailed: '서버 검사를 하지 못했어요. 저장할 때 다시 검사해요',
  noProblems: '오류 없음',
  summary: (errors, warnings) => `오류 ${errors}개 · 주의 ${warnings}개`,
  errorsBlock: '오류를 고쳐야 저장할 수 있어요',
  deleteList: '이 목록 지우기',
  deleteConfirm: (title) => `목록 ${quotedObject(title)} 지울까요? 이 목록으로 찍은 입 모양 등록도 함께 지워지고 되돌릴 수 없어요.`,
  deleted: (title) => `목록 ${quotedObject(title)} 지웠어요`,
  deletedInUse: (title) => `목록 ${quotedObject(title)} 지웠어요. 병실 화면은 이제 기본 목록을 써요`,
  loadExample: '예시 목록 불러오기',
  exampleLoaded: (title) => `예시 ${quotedObject(title)} 새 목록으로 불러왔어요. 저장해야 서버에 남아요`,
  replaceConfirm: '지금 고치던 내용을 버리고 예시를 새 목록으로 열까요?',
  listsNeedReload: '목록들을 불러와야 새 목록을 만들 수 있어요',
  reloadLists: '다시 불러오기',
  reloadList: '이 목록 다시 불러오기',
  reloadConfirm: '서버의 지금 목록을 다시 불러올까요? 여기서 고친 내용은 버려져요.',
  cannotUse: (title, message) => `내 목록 ‘${title}’${objectParticle(title)} 쓸 수 없어요: ${message}`,
  reenrollNeeded: (profile, missing, count, total) => `등록 ‘${profile}’: 아직 찍지 않은(고친) 문장 ${missing}개를 찍어야 `
    + `이 목록에서 입 모양 등록을 써요 (${count}/${total})`,
  reenrollStale: (profile) => `등록 ‘${profile}’: 인식 모델이 바뀌어 다시 등록해야 해요`,
  reenrollHowTo: '병실 화면에 이 목록을 쓰는 동안 [이 목록으로 입 모양 등록하기]에서 그 문장만 다시 찍으면 돼요',
  discardConfirm: '저장하지 않은 내용이 있어요. 버리고 넘어갈까요?',
  useThis: '이 목록으로 병실 화면 쓰기',
  usingThis: '병실 화면이 이 목록을 써요',
  useNeedsSave: '저장하고 오류가 없어야 병실 화면에 쓸 수 있어요',
  enrollThis: '이 목록으로 입 모양 등록하기',
  textRequired: '입 모양 문장을 적어 주세요',
  textHangulOnly: '한글 글자와 띄어쓰기만 쓸 수 있어요 (숫자·영문·문장부호·자모 낱자는 안 돼요)',
  textLength: `${TEXT_MIN_CHARS}~${TEXT_MAX_CHARS}자로 적어 주세요`,
  textTokens: `너무 길어요 (입 모양 길이 ${TEXT_MAX_TOKENS} 이하)`,
  displayRequired: '화면 문장을 적어 주세요',
  displayLength: `${DISPLAY_MAX_CHARS}자 이하로 적어 주세요`,
  displayControl: '쓸 수 없는 글자가 있어요',
  titleRequired: '목록 이름을 적어 주세요',
  titleLength: `${TITLE_MAX_CHARS}자 이하로 적어 주세요`,
  idRequired: '목록 ID를 적어 주세요',
  idPattern: '목록 ID는 영문·숫자·_·- 1~32자로 적어 주세요',
  idReserved: '이 ID(CON·NUL·COM1 같은 이름)는 쓸 수 없어요',
  idTaken: '같은 ID의 목록이 이미 있어요 (대소문자만 달라도 같은 ID예요)',
  tooMany: `목록은 ${MAX_LISTS}개까지 만들 수 있어요`,
  rowsMin: `명령은 ${MIN_COMMANDS}개 이상이어야 해요`,
  rowsMax: `명령은 ${MAX_COMMANDS}개까지예요`,
  normalizedAs: (text) => `저장될 문장: ${text}`,
  rowLength: (syllables) => `${syllables}글자`,
  notFound: '내 목록을 찾을 수 없어요',
  hasErrors: '내 목록에 오류가 있어요',
  broken: '내 목록의 형식이 올바르지 않아요',
  loadFailed: '서버에서 내 목록을 가져오지 못했어요',
  loading: '내 목록을 불러오는 중이에요',
  practiceDefaultOnly: '연습 모드는 기본 목록만 써요',
});

// ---------------------------------------------------------------------------
// 문장 다듬기·길이
// ---------------------------------------------------------------------------

/**
 * 서버(Python str.split·str.strip)가 공백으로 보는 글자. JS의 \s와 달리 U+FEFF(BOM)는 공백이 아니고(서버는 '쓸 수 없는 글자'로
 * 알린다), U+001C~U+001F·U+0085는 공백이다.
 */
const PY_SPACE_CLASS = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const PY_SPACES_RE = new RegExp(`[${PY_SPACE_CLASS}]+`, 'g');
const PY_EDGE_SPACES_RE = new RegExp(`^[${PY_SPACE_CLASS}]+|[${PY_SPACE_CLASS}]+$`, 'g');
/** 화면 문장·목록 이름에서 쓸 수 없는 글자 (서버 _bad_chars: 제어·서식·대리·줄/문단 구분). */
const BAD_DISPLAY_CHARS_RE = /[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}]/u;

/**
 * 입 모양 문장을 다듬는다 (서버와 같은 규칙: NFC, 공백 문자 묶음을 띄어쓰기 하나로, 앞뒤 공백 제거).
 * @param {any} text
 * @returns {string}
 */
export function normalizeListText(text) {
  try {
    return String(text ?? '').normalize('NFC').replace(PY_SPACES_RE, ' ').replace(PY_EDGE_SPACES_RE, '');
  } catch {
    return '';
  }
}

/**
 * 화면 문장·목록 이름을 다듬는다 (서버 normalize_display와 같다: NFC, 공백 문자(Zs)를 띄어쓰기로, 앞뒤 공백 제거,
 * 이어진 띄어쓰기를 하나로). 줄바꿈·탭은 가운데 있으면 남겨 둔다(검사에서 '쓸 수 없는 글자').
 * @param {any} value
 * @returns {string}
 */
export function normalizeListDisplay(value) {
  try {
    return String(value ?? '').normalize('NFC').replace(/\p{Zs}/gu, ' ').replace(PY_EDGE_SPACES_RE, '')
      .replace(/ {2,}/g, ' ');
  } catch {
    return '';
  }
}

/**
 * 한글 음절 수.
 * @param {string} text 다듬은 문장.
 * @returns {number}
 */
export function syllableCount(text) {
  return (String(text || '').match(/[가-힣]/g) || []).length;
}

/**
 * OLKAVS 토큰 수 (받침 없는 음절 2, 받침 있는 음절 3, 띄어쓰기 1). 한글 음절·띄어쓰기 밖의 글자가 있으면 null.
 * 서버(backend/services/ward_lists.py)가 기준이며, 이 값은 입력 중 안내용이다.
 * @param {string} text 다듬은 문장.
 * @returns {number|null}
 */
export function olkavsTokenCount(text) {
  try {
    let n = 0;
    for (const ch of String(text || '')) {
      if (ch === ' ') {
        n += 1;
        continue;
      }
      const code = ch.codePointAt(0) - 0xac00;
      if (code < 0 || code > 11171) return null;
      n += code % 28 === 0 ? 2 : 3;
    }
    return n;
  } catch {
    return null;
  }
}

/**
 * 입 모양 문장의 즉시 검사 (서버 검사 전 안내). 문제가 없으면 null.
 * @param {any} text
 * @returns {string|null}
 */
export function quickTextProblem(text) {
  const t = normalizeListText(text);
  if (!t) return LISTS_TEXT.textRequired;
  if (/[^가-힣 ]/.test(t)) return LISTS_TEXT.textHangulOnly;
  if (t.length < TEXT_MIN_CHARS || t.length > TEXT_MAX_CHARS) return LISTS_TEXT.textLength;
  const tokens = olkavsTokenCount(t);
  if (tokens === null) return LISTS_TEXT.textHangulOnly;
  if (tokens > TEXT_MAX_TOKENS) return LISTS_TEXT.textTokens;
  return null;
}

/**
 * 화면 문장의 즉시 검사. 문제가 없으면 null.
 * @param {any} display
 * @returns {string|null}
 */
export function quickDisplayProblem(display) {
  const d = normalizeListDisplay(display);
  if (!d) return LISTS_TEXT.displayRequired;
  if (BAD_DISPLAY_CHARS_RE.test(d)) return LISTS_TEXT.displayControl;
  if ([...d].length > DISPLAY_MAX_CHARS) return LISTS_TEXT.displayLength;
  return null;
}

/**
 * 목록 이름의 즉시 검사. 문제가 없으면 null.
 * @param {any} title
 * @returns {string|null}
 */
export function quickTitleProblem(title) {
  const t = normalizeListDisplay(title);
  if (!t) return LISTS_TEXT.titleRequired;
  if ([...t].length > TITLE_MAX_CHARS) return LISTS_TEXT.titleLength;
  return null;
}

/**
 * 목록 ID 검사 (규칙·Windows 예약 이름·대소문자만 다른 겹침). 문제가 없으면 null.
 * @param {any} id
 * @param {string[]} [existingIds] 서버에 있는 목록 ID들.
 * @param {string|null} [ownId] 지금 고치는 목록의 ID(겹침 검사에서 뺀다).
 * @returns {string|null}
 */
export function listIdProblem(id, existingIds = [], ownId = null) {
  try {
    if (typeof id !== 'string' || !id) return LISTS_TEXT.idRequired;
    if (!LIST_ID_PATTERN.test(id)) return LISTS_TEXT.idPattern;
    if (RESERVED_LIST_IDS.includes(id.toUpperCase())) return LISTS_TEXT.idReserved;
    const lower = id.toLowerCase();
    const own = typeof ownId === 'string' ? ownId.toLowerCase() : null;
    if ((existingIds || []).some((x) => typeof x === 'string' && x.toLowerCase() === lower && x.toLowerCase() !== own)) {
      return LISTS_TEXT.idTaken;
    }
    return null;
  } catch {
    return LISTS_TEXT.idPattern;
  }
}

/**
 * 새 목록 ID 제안 (mylist1, mylist2, … 중 아직 없는 것).
 * @param {string[]} existingIds
 * @returns {string}
 */
export function suggestListId(existingIds = []) {
  const taken = new Set((existingIds || []).filter((x) => typeof x === 'string').map((x) => x.toLowerCase()));
  for (let i = 1; i <= 99; i += 1) {
    const id = `mylist${i}`;
    if (!taken.has(id)) return id;
  }
  return '';
}

// ---------------------------------------------------------------------------
// 서버 검사 결과 읽기
// ---------------------------------------------------------------------------

/**
 * 서버 오류·주의 항목 하나를 화면용으로 읽는다. 문자열 또는 {message, row|index|rows|indices, field, code, strong|level}.
 * 줄 번호는 보낸 commands 배열의 0부터 세는 위치다.
 * @param {any} raw
 * @param {'error'|'warning'} level
 * @returns {{level:string, rows:number[], field:string|null, code:string|null, strong:boolean, message:string}|null}
 */
export function parseIssue(raw, level) {
  try {
    if (typeof raw === 'string') {
      const message = raw.trim();
      return message ? { level, rows: [], field: null, code: null, strong: false, message } : null;
    }
    if (!raw || typeof raw !== 'object') return null;
    const message = [raw.message, raw.msg, raw.text, raw.detail].find((x) => typeof x === 'string' && x.trim());
    if (!message) return null;
    const rows = [];
    for (const k of ['row', 'index', 'row_index']) if (Number.isInteger(raw[k])) rows.push(raw[k]);
    for (const k of ['rows', 'indices', 'row_indices', 'pair']) {
      if (Array.isArray(raw[k])) rows.push(...raw[k].filter((x) => Number.isInteger(x)));
    }
    const strong = raw.strong === true || raw.level === 'strong' || raw.severity === 'strong';
    return {
      level,
      rows: [...new Set(rows)].filter((r) => r >= 0),
      field: typeof raw.field === 'string' ? raw.field : null,
      code: typeof raw.code === 'string' ? raw.code : null,
      strong,
      message: message.trim(),
    };
  } catch {
    return null;
  }
}

/**
 * {errors, warnings}를 화면용으로 읽는다.
 * @param {any} raw
 * @returns {{errors: Array<object>, warnings: Array<object>}}
 */
export function parseChecks(raw) {
  try {
    const errors = (Array.isArray(raw?.errors) ? raw.errors : []).map((x) => parseIssue(x, 'error')).filter(Boolean);
    const warnings = (Array.isArray(raw?.warnings) ? raw.warnings : []).map((x) => parseIssue(x, 'warning')).filter(Boolean);
    return { errors, warnings };
  } catch {
    return { errors: [], warnings: [] };
  }
}

/**
 * 저장(PUT) 400 응답에서 검사 결과를 꺼낸다. 본문 {detail, errors, warnings} 또는 FastAPI식 {detail: {detail, errors, warnings}}.
 * 검사 결과가 없는 오류면 null.
 * @param {any} err axios 오류.
 * @returns {{detail:string|null, errors:Array<object>, warnings:Array<object>}|null}
 */
export function checksFromError(err) {
  try {
    const data = err?.response?.data;
    if (!data || typeof data !== 'object') return null;
    const box = Array.isArray(data.errors) || Array.isArray(data.warnings) ? data
      : (data.detail && typeof data.detail === 'object' && !Array.isArray(data.detail) ? data.detail : null);
    if (!box || !(Array.isArray(box.errors) || Array.isArray(box.warnings))) return null;
    const detail = [box.detail, data.detail].find((x) => typeof x === 'string' && x.trim()) || null;
    return { detail, ...parseChecks(box) };
  } catch {
    return null;
  }
}

/**
 * 서버 오류에서 보여 줄 문구 (문자열 detail은 그대로, 상태 코드별 고정 문구).
 * @param {any} err
 * @param {string} fallback
 * @returns {string}
 */
export function listRequestErrorMessage(err, fallback) {
  try {
    const data = err?.response?.data;
    const detail = typeof data?.detail === 'string' ? data.detail : (typeof data?.detail?.detail === 'string' ? data.detail.detail : null);
    if (detail && detail.trim()) return detail;
    const status = err?.response?.status;
    if (status === 404) return LISTS_TEXT.notFound;
    if (status === 409) return LISTS_TEXT.idTaken;
    if (status === 413) return '목록이 너무 커요. 문장을 줄여 주세요';
    if (status === 403) return '이 주소에서는 목록을 바꿀 수 없어요';
    if (!err?.response) return '서버에 연결할 수 없어요. 서버가 켜져 있는지 확인해 주세요';
    return fallback;
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// 목록들·예시
// ---------------------------------------------------------------------------

/**
 * GET /api/ward/lists 응답을 화면용으로 읽는다 (이름순).
 * @param {any} data
 * @returns {Array<{id:string, title:string, count:number|null, emergencyCount:number|null, updated:string|null, sha256:string|null}>}
 */
export function parseListSummaries(data) {
  try {
    const items = Array.isArray(data?.lists) ? data.lists : [];
    return items
      .filter((x) => x && typeof x.id === 'string' && LIST_ID_PATTERN.test(x.id))
      .map((x) => ({
        id: x.id,
        title: typeof x.title === 'string' && x.title.trim() ? x.title.trim() : x.id,
        count: Number.isInteger(x.count) ? x.count : null,
        emergencyCount: Number.isInteger(x.emergency_count) ? x.emergency_count : null,
        updated: typeof x.updated === 'string' ? x.updated : null,
        sha256: typeof x.sha256 === 'string' ? x.sha256 : null,
      }))
      .sort((a, b) => a.title.localeCompare(b.title, 'ko') || a.id.localeCompare(b.id));
  } catch {
    return [];
  }
}

/**
 * 예시 목록 파일(custom_list_examples.json)을 읽는다. 형식이 틀리거나 없으면 [].
 * @param {any} json
 * @returns {Array<{title:string, commands:Array<{text:string, display:string, category:string, emergency:boolean, cancel:boolean}>}>}
 */
export function parseExamples(json) {
  try {
    if (!json || json.schema !== EXAMPLES_SCHEMA || !Array.isArray(json.examples)) return [];
    return json.examples
      .filter((e) => e && typeof e.title === 'string' && e.title.trim() && Array.isArray(e.commands))
      .map((e) => ({
        title: e.title.trim(),
        commands: e.commands
          .filter((c) => c && typeof c.text === 'string' && typeof c.display === 'string')
          .slice(0, MAX_COMMANDS)
          .map((c) => ({
            text: c.text,
            display: c.display,
            category: typeof c.category === 'string' ? c.category : DEFAULT_ROW_CATEGORY,
            emergency: c.emergency === true,
            cancel: c.cancel === true,
          })),
      }))
      .filter((e) => e.commands.length > 0);
  } catch {
    return [];
  }
}

/**
 * 기본 목록(commands.json)을 읽기 전용 표로 (바꿀 수 없음).
 * @param {{categories: Array<{id:string,label:string}>, fixed: Array<object>}} json
 * @returns {Array<{id:string, text:string, display:string, category:string, categoryLabel:string, emergency:boolean, cancel:boolean}>}
 */
export function defaultListRows(json) {
  try {
    const labels = new Map((json?.categories || []).map((c) => [c.id, c.label]));
    return (json?.fixed || []).map((c) => ({
      id: c.id, text: c.text, display: c.display, category: c.category, categoryLabel: labels.get(c.category) || c.category,
      emergency: c.emergency === true, cancel: c.action === CANCEL_ACTION,
    }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// 병실 화면·등록 화면에서 내 목록 불러오기
// ---------------------------------------------------------------------------

/**
 * GET /api/ward/lists/{id} 응답을 병실 화면 색인으로 만든다. 오류가 있는 목록·형식이 틀린 목록은 쓰지 않는다
 * (기본 목록으로 바꾸지 않고 실패로 돌려준다).
 * @param {any} data 서버 응답.
 * @param {string} listId 쓰려던 목록 ID.
 * @param {{categories: Array<{id:string,label:string}>}} base 기본 목록(commands.json, 분류 이름).
 * @returns {{ok:true, index:object, title:string, warnings:Array<object>}|{ok:false, message:string, title:string}}
 */
export function listLoadOutcome(data, listId, base) {
  const title = typeof data?.title === 'string' && data.title.trim() ? data.title.trim() : String(listId || '');
  try {
    if (!data || typeof data !== 'object' || data.id !== listId) return { ok: false, message: LISTS_TEXT.broken, title };
    if (data.schema !== undefined && data.schema !== LIST_SCHEMA) return { ok: false, message: LISTS_TEXT.broken, title };
    const checks = parseChecks(data.checks);
    if (checks.errors.length) return { ok: false, message: LISTS_TEXT.hasErrors, title };
    const index = buildCustomCommandIndex(data, base);
    return { ok: true, index, title: index.customList.title, warnings: checks.warnings };
  } catch {
    return { ok: false, message: LISTS_TEXT.broken, title };
  }
}

/**
 * 내 목록을 불러오지 못한 이유 (404 = 없음, 400 = 서버 문구, 그 밖 = 연결 문제).
 * @param {any} err
 * @returns {string}
 */
export function listLoadErrorMessage(err) {
  try {
    const status = err?.response?.status;
    if (status === 404) return LISTS_TEXT.notFound;
    const detail = err?.response?.data?.detail;
    if (status === 400 && typeof detail === 'string' && detail.trim()) return detail;
    if (err?.isClientCheck && typeof err.message === 'string') return err.message;
    return LISTS_TEXT.loadFailed;
  } catch {
    return LISTS_TEXT.loadFailed;
  }
}

/**
 * 병실 화면 위쪽 띠 문구 (내 목록을 쓸 때만).
 * @param {{title?:string, id?:string}|null|undefined} customList 색인의 customList.
 * @returns {string|null}
 */
export function listBannerText(customList) {
  if (!customList || !customList.id) return null;
  const title = customList.title || customList.id;
  return `내 목록 ‘${title}’ 사용 중 · ${LISTS_TEXT.untested}`;
}

/**
 * 분석 결과가 지금 불러온 내 목록과 다른 판(지문)으로 나왔는지. 그렇다면 목록이 바뀐 것이니 다시 불러온다.
 * @param {Array<object>|null|undefined} results 서버 results.
 * @param {{id:string, sha256:string|null}|null|undefined} customList
 * @returns {boolean}
 */
export function resultsFromOtherListVersion(results, customList) {
  try {
    if (!customList || !Array.isArray(results)) return false;
    const list = results.find((x) => x && typeof x === 'object')?.phrase?.list;
    return !!list && list.id === customList.id && typeof list.sha256 === 'string' && list.sha256 !== customList.sha256;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// 편집 상태도
// ---------------------------------------------------------------------------

/** 빈 줄. */
function blankRow(key) {
  return { key, text: '', display: '', category: DEFAULT_ROW_CATEGORY, emergency: false, cancel: false, normalCategory: DEFAULT_ROW_CATEGORY };
}

/** 명령 하나(서버·예시)를 편집 줄로. */
function rowFrom(key, c, catIds) {
  const emergency = c?.emergency === true || c?.category === EMERGENCY_CATEGORY;
  let category = typeof c?.category === 'string' && catIds.includes(c.category) ? c.category : DEFAULT_ROW_CATEGORY;
  if (emergency) category = EMERGENCY_CATEGORY;
  else if (category === EMERGENCY_CATEGORY) category = DEFAULT_ROW_CATEGORY;
  return {
    key,
    text: typeof c?.text === 'string' ? c.text : '',
    display: typeof c?.display === 'string' ? c.display : '',
    category,
    emergency,
    cancel: c?.cancel === true && !emergency,
    normalCategory: emergency ? DEFAULT_ROW_CATEGORY : category,
  };
}

/** 빈 검사 상태. */
function idleCheck() {
  return { status: 'idle', seq: -1, errors: [], warnings: [], normalized: null, rowKeys: [], message: null };
}

/**
 * 편집 상태를 만든다.
 * @param {{categories: Array<{id:string,label:string}>, id?: string, rows?: number}} opts
 *   categories는 기본 목록의 분류(종류 선택지), id는 새 목록 ID 제안.
 * @returns {object}
 *   mode: 'new'(아직 저장 안 함) | 'edit'(서버에 있음). savedId는 서버에 있는 ID, editSeq는 내용이 바뀔 때마다 1씩 는다(검사 짝 맞추기).
 */
export function createListEditor({ categories = [], id = '', rows = NEW_LIST_ROWS } = {}) {
  const cats = (Array.isArray(categories) ? categories : []).map((c) => ({ id: String(c.id), label: String(c.label) }));
  const n = Math.max(0, Math.min(MAX_COMMANDS, Number.isInteger(rows) ? rows : NEW_LIST_ROWS));
  return {
    categories: cats,
    mode: 'new',
    id: typeof id === 'string' ? id : '',
    savedId: null,
    title: '',
    rows: Array.from({ length: n }, (_, i) => blankRow(i + 1)),
    nextKey: n + 1,
    editSeq: 0,
    dirty: false,
    savedSha: null,
    savedUpdated: null,
    saving: false,
    saveMessage: null,
    check: idleCheck(),
  };
}

/** 내용이 바뀐 새 상태 (검사를 다시 해야 한다). */
function edited(s, patch) {
  return { ...s, ...patch, editSeq: s.editSeq + 1, dirty: true, saveMessage: null };
}

/** 줄 하나를 바꾼다. */
function updateRow(s, key, fn) {
  const i = s.rows.findIndex((r) => r.key === key);
  if (i < 0) return s;
  const rows = [...s.rows];
  rows[i] = fn(rows[i]);
  return edited(s, { rows });
}

/** 서버 목록을 편집 상태로 (저장한 그대로, 바뀐 것 없음). */
function loadedFrom(s, list, extraWarnings = null) {
  const catIds = s.categories.map((c) => c.id);
  const cmds = Array.isArray(list?.commands) ? list.commands : [];
  const rows = cmds.slice(0, MAX_COMMANDS).map((c, i) => rowFrom(i + 1, c, catIds));
  const checks = parseChecks(list?.checks || (extraWarnings ? { errors: [], warnings: extraWarnings } : null));
  const editSeq = s.editSeq + 1;
  return {
    ...s,
    mode: 'edit',
    id: list.id,
    savedId: list.id,
    title: typeof list.title === 'string' ? list.title : '',
    rows,
    nextKey: rows.length + 1,
    editSeq,
    dirty: false,
    savedSha: typeof list.sha256 === 'string' ? list.sha256 : null,
    savedUpdated: typeof list.updated === 'string' ? list.updated : null,
    saving: false,
    check: {
      status: 'done', seq: editSeq, errors: checks.errors, warnings: checks.warnings, normalized: null,
      rowKeys: rows.map((r) => r.key), message: null,
    },
  };
}

/**
 * 편집 상태도 한 단계.
 * @param {object} state createListEditor 결과.
 * @param {{type:string, [key:string]: any}} event
 *   NEW{id} · LOAD_LIST{list} · LOAD_EXAMPLE{example, id?}(언제나 새 목록, 저장된 목록이 열려 있었으면 새 ID id) · SET_TITLE{value} · SET_ID{value} · SET_TEXT/SET_DISPLAY{key, value}
 *   · SET_CATEGORY{key, value} · SET_EMERGENCY{key, value} · SET_CANCEL{key, value} · ADD_ROW · REMOVE_ROW{key}
 *   · MOVE_ROW{key, delta} · CHECK_START{seq} · CHECK_DONE{seq, data} · CHECK_FAILED{seq, message}
 *   · SAVE_START · SAVED{list} · SAVE_FAILED{seq, message, checks}
 * @returns {object} 바뀐 것이 없으면 같은 객체.
 */
export function stepListEditor(state, event) {
  try {
    if (!state || !event || typeof event.type !== 'string') return state;
    const s = state;
    const catIds = s.categories.map((c) => c.id);
    switch (event.type) {
      case 'NEW': {
        const fresh = createListEditor({ categories: s.categories, id: typeof event.id === 'string' ? event.id : '' });
        return { ...fresh, editSeq: s.editSeq + 1 };
      }
      case 'LOAD_LIST':
        if (!event.list || typeof event.list.id !== 'string') return s;
        return loadedFrom(s, event.list);
      case 'LOAD_EXAMPLE': {
        const ex = event.example;
        if (!ex || !Array.isArray(ex.commands) || !ex.commands.length) return s;
        // 예시는 언제나 새 목록으로 연다: 저장된 목록을 열어 둔 채 불러와도 그 목록을 덮어쓰지 않는다(새 ID event.id).
        const base = s.mode === 'new'
          ? s
          : {
            ...createListEditor({ categories: s.categories, id: typeof event.id === 'string' ? event.id : '', rows: 0 }),
            editSeq: s.editSeq,
          };
        const rows = ex.commands.slice(0, MAX_COMMANDS).map((c, i) => rowFrom(base.nextKey + i, c, catIds));
        return edited(base, {
          title: typeof ex.title === 'string' ? ex.title : base.title,
          rows,
          nextKey: base.nextKey + rows.length,
        });
      }
      case 'SET_TITLE':
        if (typeof event.value !== 'string' || event.value === s.title) return s;
        return edited(s, { title: event.value });
      case 'SET_ID':
        if (s.mode !== 'new' || typeof event.value !== 'string' || event.value === s.id) return s;
        return { ...s, id: event.value.trim(), dirty: true, saveMessage: null };
      case 'SET_TEXT':
      case 'SET_DISPLAY': {
        if (typeof event.value !== 'string') return s;
        const field = event.type === 'SET_TEXT' ? 'text' : 'display';
        return updateRow(s, event.key, (r) => ({ ...r, [field]: event.value }));
      }
      case 'SET_CATEGORY': {
        if (!catIds.includes(event.value)) return s;
        return updateRow(s, event.key, (r) => (event.value === EMERGENCY_CATEGORY
          ? { ...r, category: EMERGENCY_CATEGORY, emergency: true, cancel: false }
          : { ...r, category: event.value, emergency: false, normalCategory: event.value }));
      }
      case 'SET_EMERGENCY':
        return updateRow(s, event.key, (r) => (event.value === true
          ? { ...r, category: EMERGENCY_CATEGORY, emergency: true, cancel: false }
          : { ...r, category: r.normalCategory || DEFAULT_ROW_CATEGORY, emergency: false }));
      case 'SET_CANCEL': {
        const i = s.rows.findIndex((r) => r.key === event.key);
        if (i < 0) return s;
        const on = event.value === true;
        const rows = s.rows.map((r) => {
          if (r.key !== event.key) return on && r.cancel ? { ...r, cancel: false } : r;
          if (!on) return { ...r, cancel: false };
          // 취소 명령은 긴급일 수 없다: 긴급이던 줄은 일반으로 돌린다.
          return r.emergency
            ? { ...r, cancel: true, emergency: false, category: r.normalCategory || DEFAULT_ROW_CATEGORY }
            : { ...r, cancel: true };
        });
        return edited(s, { rows });
      }
      case 'ADD_ROW':
        if (s.rows.length >= MAX_COMMANDS) return s;
        return edited(s, { rows: [...s.rows, blankRow(s.nextKey)], nextKey: s.nextKey + 1 });
      case 'REMOVE_ROW': {
        if (!s.rows.some((r) => r.key === event.key)) return s;
        return edited(s, { rows: s.rows.filter((r) => r.key !== event.key) });
      }
      case 'MOVE_ROW': {
        const i = s.rows.findIndex((r) => r.key === event.key);
        const j = i + (event.delta < 0 ? -1 : 1);
        if (i < 0 || j < 0 || j >= s.rows.length) return s;
        const rows = [...s.rows];
        [rows[i], rows[j]] = [rows[j], rows[i]];
        return edited(s, { rows });
      }
      case 'CHECK_START':
        if (event.seq !== s.editSeq) return s;
        return { ...s, check: { ...s.check, status: 'checking', seq: event.seq } };
      case 'CHECK_DONE': {
        if (event.seq !== s.editSeq) return s;
        const checks = parseChecks(event.data);
        return {
          ...s,
          check: {
            status: 'done', seq: event.seq, errors: checks.errors, warnings: checks.warnings,
            normalized: event.data?.normalized && typeof event.data.normalized === 'object' ? event.data.normalized : null,
            rowKeys: s.rows.map((r) => r.key), message: null,
          },
        };
      }
      case 'CHECK_FAILED':
        if (event.seq !== s.editSeq) return s;
        return { ...s, check: { ...idleCheck(), status: 'failed', seq: event.seq, message: String(event.message || LISTS_TEXT.checkFailed) } };
      case 'SAVE_START':
        if (s.saving) return s;
        return { ...s, saving: true, saveMessage: null };
      case 'SAVED':
        if (!event.list || typeof event.list.id !== 'string') return { ...s, saving: false };
        return { ...loadedFrom(s, event.list, Array.isArray(event.list.warnings) ? event.list.warnings : null), saveMessage: LISTS_TEXT.saved };
      case 'SAVE_FAILED': {
        const base = { ...s, saving: false, saveMessage: String(event.message || '저장하지 못했어요') };
        if (!event.checks || event.seq !== s.editSeq) return base;
        return {
          ...base,
          check: {
            status: 'done', seq: s.editSeq, errors: event.checks.errors || [], warnings: event.checks.warnings || [],
            normalized: null, rowKeys: s.rows.map((r) => r.key), message: null,
          },
        };
      }
      default:
        return s;
    }
  } catch {
    return state;
  }
}

/**
 * 저장(PUT) 조건: 새 목록은 '없을 때만 만들기'(If-None-Match: *), 저장된 목록은 '불러온 판일 때만 고치기'(If-Match: sha256).
 * 그래서 다른 탭이 같은 ID로 만들었거나 그사이 고친 목록을 말없이 덮어쓰지 않는다(서버가 409로 거절).
 * @param {object} state
 * @returns {{create: true}|{ifMatch: string}|{}}
 */
export function saveCondition(state) {
  if (!state) return {};
  if (state.mode === 'new') return { create: true };
  return typeof state.savedSha === 'string' && /^[0-9a-f]{64}$/.test(state.savedSha) ? { ifMatch: state.savedSha } : {};
}

/**
 * 저장한(또는 연) 목록의 입 모양 등록 상태 요약. 고친 문장 때문에 다시 찍어야 할 문장이 있으면 안내, 아니면 null.
 * 한 번도 찍지 않은 등록(count 0)·다 찍은 등록은 null.
 * @param {{profile:string, count:number, total:number, status:string}|null|undefined} info wardEnroll.parseProfileInfo 결과.
 * @returns {{kind:'partial'|'stale', profile:string, missing:number, count:number, total:number, text:string}|null}
 */
export function reenrollSummary(info) {
  try {
    if (!info || !info.profile || !Number.isInteger(info.total) || info.total <= 0) return null;
    const count = Number.isInteger(info.count) ? Math.max(0, Math.min(info.count, info.total)) : 0;
    if (info.status === 'stale') {
      return {
        kind: 'stale', profile: info.profile, missing: info.total, count, total: info.total,
        text: LISTS_TEXT.reenrollStale(info.profile),
      };
    }
    const missing = info.total - count;
    if (count === 0 || missing === 0) return null;
    return {
      kind: 'partial', profile: info.profile, missing, count, total: info.total,
      text: LISTS_TEXT.reenrollNeeded(info.profile, missing, count, info.total),
    };
  } catch {
    return null;
  }
}

/**
 * 저장·검사 요청 본문 (서버가 다듬고 ID를 붙인다).
 * @param {object} state
 * @returns {{title:string, commands:Array<{text:string, display:string, category:string, emergency:boolean, cancel:boolean}>}}
 */
export function requestBody(state) {
  return {
    title: String(state?.title ?? ''),
    commands: (state?.rows || []).map((r) => ({
      text: r.text, display: r.display, category: r.category, emergency: r.emergency === true, cancel: r.cancel === true,
    })),
  };
}

/**
 * 검사 결과가 지금 내용에 대한 것인지 (내용을 고치면 다음 검사까지 예전 결과를 믿지 않는다).
 * @param {object} state
 * @returns {boolean}
 */
export function checkIsCurrent(state) {
  return !!state && state.check?.status === 'done' && state.check.seq === state.editSeq;
}

/**
 * 줄별·전체 안내. 줄 위치(0부터)는 검사할 때의 줄 순서로 줄 key에 붙인다. 즉시 검사(한글만·길이)는 적은 칸에만 붙인다.
 * @param {object} state
 * @returns {{byKey: Record<string, {errors:Array<object>, warnings:Array<object>, quick:{text:string|null, display:string|null}}>,
 *   global: {errors:Array<object>, warnings:Array<object>}, current:boolean}}
 */
export function editorMessages(state) {
  const byKey = {};
  const global = { errors: [], warnings: [] };
  try {
    for (const r of state.rows) {
      byKey[r.key] = {
        errors: [],
        warnings: [],
        quick: {
          text: r.text.trim() ? quickTextProblem(r.text) : null,
          display: r.display.trim() ? quickDisplayProblem(r.display) : null,
        },
      };
    }
    const keys = state.check?.rowKeys || [];
    const put = (issue, bucket) => {
      const targets = issue.rows.map((i) => keys[i]).filter((k) => k !== undefined && byKey[k]);
      if (!targets.length) global[bucket].push(issue);
      for (const k of targets) byKey[k][bucket].push(issue);
    };
    for (const e of state.check?.errors || []) put(e, 'errors');
    for (const w of state.check?.warnings || []) put(w, 'warnings');
    return { byKey, global, current: checkIsCurrent(state) };
  } catch {
    return { byKey, global, current: false };
  }
}

/**
 * 저장할 수 있는지와 못 하는 이유.
 * @param {object} state
 * @param {string[]} [existingIds]
 * @returns {{ok:boolean, reason:string|null}}
 */
export function canSaveList(state, existingIds = []) {
  try {
    if (!state || state.saving) return { ok: false, reason: null };
    if (state.mode === 'edit' && !state.dirty) return { ok: false, reason: null };
    const idProblem = listIdProblem(state.id, existingIds, state.savedId);
    if (idProblem) return { ok: false, reason: idProblem };
    if (state.mode === 'new' && existingIds.length >= MAX_LISTS) return { ok: false, reason: LISTS_TEXT.tooMany };
    const titleProblem = quickTitleProblem(state.title);
    if (titleProblem) return { ok: false, reason: titleProblem };
    if (state.rows.length < MIN_COMMANDS) return { ok: false, reason: LISTS_TEXT.rowsMin };
    if (state.rows.length > MAX_COMMANDS) return { ok: false, reason: LISTS_TEXT.rowsMax };
    for (const r of state.rows) {
      const p = quickTextProblem(r.text) || quickDisplayProblem(r.display);
      if (p) return { ok: false, reason: p };
    }
    if (checkIsCurrent(state) && state.check.errors.length) return { ok: false, reason: LISTS_TEXT.errorsBlock };
    return { ok: true, reason: null };
  } catch {
    return { ok: false, reason: null };
  }
}

/**
 * 병실 화면에 쓸 수 있는지 (서버에 저장했고, 고친 것이 없고, 지금 검사에 오류가 없음).
 * @param {object} state
 * @returns {boolean}
 */
export function canUseList(state) {
  return !!state && state.mode === 'edit' && !!state.savedId && !state.dirty && !state.saving
    && checkIsCurrent(state) && state.check.errors.length === 0;
}

/**
 * 줄 길이 안내 (음절 수·입 모양 길이, 한글만일 때).
 * @param {string} text
 * @returns {{syllables:number, tokens:number|null, good:boolean}}
 */
export function rowLength(text) {
  const t = normalizeListText(text);
  const syllables = syllableCount(t);
  const tokens = /[^가-힣 ]/.test(t) ? null : olkavsTokenCount(t);
  const good = tokens !== null && syllables >= GOOD_SYLLABLES[0] && syllables <= GOOD_SYLLABLES[1]
    && tokens >= GOOD_TOKENS[0] && tokens <= GOOD_TOKENS[1];
  return { syllables, tokens, good };
}
