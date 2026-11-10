// wardSettings.js — 환자 화면 설정(녹화 길이·확인 시간·병실 번호·음성 읽기·밤 무음·명령 목록·내 입 모양 등록 이름)과 저장 순수 함수 (node --test로 검사).
// 브라우저 저장소는 사생활 보호 모드·차단으로 없거나 예외를 던질 수 있어 모든 읽기·쓰기를 try/catch로 감싼다.
// 명령 목록(commandList)은 ''이면 기본 목록(commands.json), 아니면 서버에 저장한 내 목록 ID다.
// 등록 이름은 목록마다 따로다: 기본 목록은 예전과 같은 enrollProfile, 내 목록은 enrollProfileByList[목록 ID].

export const RECORD_SECONDS_OPTIONS = Object.freeze([4, 5, 6, 8]);
export const DEFAULT_RECORD_SECONDS = 5;
export const CONFIRM_SECONDS_MIN = 5;
export const CONFIRM_SECONDS_MAX = 8;
export const DEFAULT_CONFIRM_SECONDS = 6;
export const DEFAULT_ROOM = '305';
/** 밤 무음 시간 (22시~6시, 음성 읽기를 자동으로 끈다). */
export const NIGHT_START_HOUR = 22;
export const NIGHT_END_HOUR = 6;
export const SETTINGS_STORAGE_KEY = 'silent-talk-ward-settings-v1';
/** 내 입 모양 등록 이름 (서버 프로필 ID 규칙과 같다: 영문·숫자·_·-, 1~32자). */
export const ENROLL_PROFILE_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
/** 내 목록 ID (서버 목록 ID 규칙과 같다: 영문·숫자·_·-, 1~32자). */
export const COMMAND_LIST_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
/** 목록별 등록 이름을 기억하는 최대 개수 (서버 목록은 최대 20개, 지운 목록 몫까지 넉넉히). */
export const MAX_PROFILE_LISTS = 40;

export const DEFAULT_SETTINGS = Object.freeze({
  recordSeconds: DEFAULT_RECORD_SECONDS,
  confirmSeconds: DEFAULT_CONFIRM_SECONDS,
  room: DEFAULT_ROOM,
  tts: false, // 다인실 사생활을 위해 기본으로 끈다.
  nightMute: true,
  enrollProfile: '', // 기본 목록의 내 입 모양 등록 이름. ''이면 등록을 쓰지 않는다.
  commandList: '', // 병실 화면이 쓰는 명령 목록. ''이면 기본 목록.
  enrollProfileByList: Object.freeze({}), // 내 목록 ID → 그 목록의 등록 이름.
  inputMode: 'switch', // 'switch': 아무 키·화면 누름 = 스위치(스캐닝), 'direct': 고를 항목을 직접 누름.
});

/**
 * 등록 이름이 서버 규칙에 맞는지.
 * @param {any} value
 * @returns {boolean}
 */
export function isValidProfileId(value) {
  return typeof value === 'string' && ENROLL_PROFILE_PATTERN.test(value);
}

/**
 * 등록 이름을 다듬는다 (앞뒤 공백 제거). 문자열이 아니거나 규칙에 맞지 않으면 ''(등록 안 씀).
 * @param {any} value
 * @returns {string}
 */
export function normalizeProfileId(value) {
  try {
    if (typeof value !== 'string') return '';
    const id = value.trim();
    return isValidProfileId(id) ? id : '';
  } catch {
    return '';
  }
}

/**
 * 내 목록 ID가 서버 규칙에 맞는지.
 * @param {any} value
 * @returns {boolean}
 */
export function isValidCommandListId(value) {
  return typeof value === 'string' && COMMAND_LIST_PATTERN.test(value);
}

/**
 * 명령 목록 설정을 다듬는다. 규칙에 맞지 않으면 ''(기본 목록).
 * @param {any} value
 * @returns {string}
 */
export function normalizeCommandList(value) {
  try {
    if (typeof value !== 'string') return '';
    const id = value.trim();
    return isValidCommandListId(id) ? id : '';
  } catch {
    return '';
  }
}

/**
 * 목록별 등록 이름을 다듬는다 (목록 ID·등록 이름이 규칙에 맞는 것만, 최대 MAX_PROFILE_LISTS개).
 * @param {any} raw
 * @returns {Record<string, string>}
 */
export function normalizeEnrollProfileByList(raw) {
  try {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {};
    const out = {};
    for (const [key, value] of Object.entries(raw)) {
      if (Object.keys(out).length >= MAX_PROFILE_LISTS) break;
      const profile = normalizeProfileId(value);
      if (isValidCommandListId(key) && profile) out[key] = profile;
    }
    return out;
  } catch {
    return {};
  }
}

/**
 * 목록 하나의 등록 이름. listId가 ''(기본 목록)이면 예전과 같은 enrollProfile이다.
 * @param {{enrollProfile?: string, enrollProfileByList?: Record<string, string>}|null|undefined} settings
 * @param {string} [listId]
 * @returns {string}
 */
export function enrollProfileFor(settings, listId = '') {
  try {
    if (!listId) return normalizeProfileId(settings?.enrollProfile);
    if (!isValidCommandListId(listId)) return '';
    return normalizeProfileId(settings?.enrollProfileByList?.[listId]);
  } catch {
    return '';
  }
}

/**
 * 목록 하나의 등록 이름을 바꾸는 설정 조각 (SETTINGS 이벤트·saveSettings에 그대로 넣는다).
 * 기본 목록은 {enrollProfile}, 내 목록은 {enrollProfileByList} 전체(그 목록 몫만 바꿈, ''이면 지움).
 * @param {object} settings 지금 설정.
 * @param {string} listId ''이면 기본 목록.
 * @param {string} profile 등록 이름(''이면 등록 안 씀).
 * @returns {{enrollProfile: string}|{enrollProfileByList: Record<string, string>}}
 */
export function enrollProfilePatch(settings, listId, profile) {
  const id = normalizeProfileId(profile);
  if (!listId) return { enrollProfile: id };
  const byList = { ...normalizeEnrollProfileByList(settings?.enrollProfileByList) };
  if (!isValidCommandListId(listId)) return { enrollProfileByList: byList };
  if (id) byList[listId] = id;
  else delete byList[listId];
  return { enrollProfileByList: byList };
}

/**
 * 병실 번호를 화면·카드용으로 다듬는다 (한글·영문·숫자·하이픈, 최대 10자). 비면 기본값.
 * @param {any} value
 * @returns {string}
 */
export function normalizeRoom(value) {
  const room = String(value ?? '').replace(/[^0-9A-Za-z가-힣-]/g, '').slice(0, 10);
  return room || DEFAULT_ROOM;
}

/**
 * 설정 값을 허용 범위로 맞춘다. 모르는 값은 기본값으로 바꾼다.
 * @param {object|null|undefined} raw
 * @returns {{recordSeconds:number, confirmSeconds:number, room:string, tts:boolean, nightMute:boolean, enrollProfile:string,
 *   commandList:string, enrollProfileByList:Record<string, string>}}
 */
export function normalizeSettings(raw) {
  try {
    const src = raw && typeof raw === 'object' ? raw : {};
    const rec = Number(src.recordSeconds);
    const conf = Number(src.confirmSeconds);
    return {
      recordSeconds: RECORD_SECONDS_OPTIONS.includes(rec) ? rec : DEFAULT_RECORD_SECONDS,
      confirmSeconds: Number.isInteger(conf) && conf >= CONFIRM_SECONDS_MIN && conf <= CONFIRM_SECONDS_MAX
        ? conf : DEFAULT_CONFIRM_SECONDS,
      room: src.room === undefined ? DEFAULT_ROOM : normalizeRoom(src.room),
      tts: src.tts === true,
      nightMute: src.nightMute !== false,
      enrollProfile: normalizeProfileId(src.enrollProfile),
      commandList: normalizeCommandList(src.commandList),
      enrollProfileByList: normalizeEnrollProfileByList(src.enrollProfileByList),
      inputMode: src.inputMode === 'direct' ? 'direct' : 'switch',
    };
  } catch {
    return { ...DEFAULT_SETTINGS, enrollProfileByList: {} };
  }
}

/**
 * 밤 무음 시간인지 (22:00 이상 또는 6:00 미만, 로컬 시각).
 * @param {Date} date
 * @returns {boolean}
 */
export function isNightTime(date) {
  const d = date instanceof Date && !Number.isNaN(date.getTime()) ? date : new Date();
  const h = d.getHours();
  return h >= NIGHT_START_HOUR || h < NIGHT_END_HOUR;
}

/**
 * 지금 음성 읽기를 해도 되는지 (설정 켜짐 + 밤 무음 시간이 아님).
 * @param {{tts:boolean, nightMute:boolean}} settings
 * @param {Date} date
 * @returns {boolean}
 */
export function speechAllowed(settings, date) {
  if (!settings?.tts) return false;
  return !(settings.nightMute !== false && isNightTime(date));
}

/**
 * 저장된 설정을 읽는다. 저장소가 없거나 내용이 깨졌으면 기본값.
 * @param {{getItem: (k:string) => string|null}|null|undefined} storage
 * @returns {ReturnType<typeof normalizeSettings>}
 */
export function loadSettings(storage) {
  try {
    const raw = storage?.getItem(SETTINGS_STORAGE_KEY);
    return normalizeSettings(raw ? JSON.parse(raw) : null);
  } catch {
    return normalizeSettings(null);
  }
}

/**
 * 저장된 설정을 읽는다. 저장된 값이 없거나 읽지 못하면 null (기본값으로 바꾸지 않는다: 다른 탭의 값을 지우지 않으려고 쓴다).
 * @param {{getItem: (k:string) => string|null}|null|undefined} storage
 * @returns {ReturnType<typeof normalizeSettings>|null}
 */
export function readStoredSettings(storage) {
  try {
    const raw = storage?.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? normalizeSettings(parsed) : null;
  } catch {
    return null;
  }
}

/** 다른 탭(목록 관리 #lists·등록 #enroll)이 바꾸는 설정. 병실 화면은 저장소가 바뀌면 이 값들을 다시 읽는다. */
export const SHARED_SETTING_KEYS = Object.freeze(['commandList', 'enrollProfile', 'enrollProfileByList']);

/** 목록별 등록 이름 두 개가 같은지 (키 순서 무관). */
function sameProfileMap(a, b) {
  const x = normalizeEnrollProfileByList(a);
  const y = normalizeEnrollProfileByList(b);
  const kx = Object.keys(x).sort();
  const ky = Object.keys(y).sort();
  return kx.length === ky.length && kx.every((k, i) => k === ky[i] && x[k] === y[k]);
}

/**
 * 저장소의 설정(다른 탭이 바꿨을 수 있음)과 지금 설정의 공유 값(SHARED_SETTING_KEYS) 차이. 다른 것이 없거나 저장된 값이 없으면 null.
 * @param {object} current 지금 화면의 설정.
 * @param {object|null} stored readStoredSettings 결과.
 * @returns {Partial<{commandList:string, enrollProfile:string, enrollProfileByList:Record<string,string>}>|null}
 */
export function sharedSettingsPatch(current, stored) {
  try {
    if (!stored || typeof stored !== 'object') return null;
    const cur = normalizeSettings(current);
    const next = normalizeSettings(stored);
    const patch = {};
    if (cur.commandList !== next.commandList) patch.commandList = next.commandList;
    if (cur.enrollProfile !== next.enrollProfile) patch.enrollProfile = next.enrollProfile;
    if (!sameProfileMap(cur.enrollProfileByList, next.enrollProfileByList)) patch.enrollProfileByList = next.enrollProfileByList;
    return Object.keys(patch).length ? patch : null;
  } catch {
    return null;
  }
}

/**
 * 설정을 저장한다. 실패해도 화면 동작에는 영향이 없다.
 * @param {{setItem: (k:string, v:string) => void}|null|undefined} storage
 * @param {object} settings
 * @returns {boolean} 저장했으면 true.
 */
export function saveSettings(storage, settings) {
  try {
    if (!storage) return false;
    storage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(normalizeSettings(settings)));
    return true;
  } catch {
    return false;
  }
}

/**
 * 연습 모드 여부: 주소의 #ward?practice=1 또는 ?practice=1#ward.
 * @param {{hash?: string, search?: string}|null|undefined} loc window.location 비슷한 객체.
 * @returns {boolean}
 */
export function isPracticeMode(loc) {
  try {
    const hash = String(loc?.hash || '');
    const q = hash.includes('?') ? hash.slice(hash.indexOf('?') + 1) : '';
    const fromHash = new URLSearchParams(q).get('practice');
    const fromSearch = new URLSearchParams(String(loc?.search || '')).get('practice');
    return fromHash === '1' || fromSearch === '1';
  } catch {
    return false;
  }
}

/**
 * 브라우저 localStorage를 안전하게 가져온다 (접근 자체가 예외를 던질 수 있음).
 * @returns {Storage|null}
 */
export function safeLocalStorage() {
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}
