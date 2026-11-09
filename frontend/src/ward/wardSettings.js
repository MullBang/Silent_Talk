// wardSettings.js — 환자 화면 설정(녹화 길이·확인 시간·병실 번호·음성 읽기·밤 무음)과 저장 순수 함수 (node --test로 검사).
// 브라우저 저장소는 사생활 보호 모드·차단으로 없거나 예외를 던질 수 있어 모든 읽기·쓰기를 try/catch로 감싼다.

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

export const DEFAULT_SETTINGS = Object.freeze({
  recordSeconds: DEFAULT_RECORD_SECONDS,
  confirmSeconds: DEFAULT_CONFIRM_SECONDS,
  room: DEFAULT_ROOM,
  tts: false, // 다인실 사생활을 위해 기본으로 끈다.
  nightMute: true,
});

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
 * @returns {{recordSeconds:number, confirmSeconds:number, room:string, tts:boolean, nightMute:boolean}}
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
    };
  } catch {
    return { ...DEFAULT_SETTINGS };
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
