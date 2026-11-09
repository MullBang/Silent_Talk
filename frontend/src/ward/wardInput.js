// wardInput.js — 스위치 입력 판정 순수 함수 (node --test로 검사).
// 환자 화면은 아무 키·포인터를 스위치로 받는다. 보조공학 스위치·발판은 보통 키보드 신호를 보낸다.
// 단, 설정·연습 버튼처럼 data-no-switch 안의 조작, 글자 입력 칸, 브라우저 단축키는 스위치로 치지 않는다.
// 간호사 창에서 환자 창으로 넘기는 키는 forwardIsSwitch로 한 번 더 거른다(목록 이동 키 제외).

/** 스위치로 치지 않는 키 (포커스 이동·보조 키·새로 고침·전체 화면·개발자 도구). */
export const NON_SWITCH_KEYS = Object.freeze([
  'Tab', 'Shift', 'Control', 'Alt', 'AltGraph', 'Meta', 'OS', 'CapsLock', 'NumLock', 'ScrollLock', 'Escape',
  'ContextMenu', 'F5', 'F11', 'F12', 'Unidentified',
]);

const INTERACTIVE_TAGS = ['BUTTON', 'A', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY', 'OPTION', 'LABEL'];

/**
 * 글자 입력 중인 칸인지.
 * @param {any} target
 * @returns {boolean}
 */
export function isEditableTarget(target) {
  try {
    const tag = String(target?.tagName || '').toUpperCase();
    return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || target?.isContentEditable === true;
  } catch {
    return false;
  }
}

/**
 * data-no-switch 영역 안인지.
 * @param {any} target
 * @returns {boolean}
 */
export function insideNoSwitch(target) {
  try {
    return typeof target?.closest === 'function' && !!target.closest('[data-no-switch]');
  } catch {
    return false;
  }
}

/**
 * 키 입력이 스위치 누름인지.
 * @param {{key?: string, ctrlKey?: boolean, metaKey?: boolean, altKey?: boolean, target?: any}} event
 * @returns {boolean}
 */
export function keyIsSwitch(event) {
  try {
    if (!event) return false;
    if (NON_SWITCH_KEYS.includes(event.key)) return false;
    if (event.ctrlKey || event.metaKey || event.altKey) return false; // 브라우저 단축키
    const target = event.target;
    if (isEditableTarget(target)) return false;
    const tag = String(target?.tagName || '').toUpperCase();
    if (insideNoSwitch(target) && INTERACTIVE_TAGS.includes(tag)) return false; // 설정 버튼은 키보드로도 쓸 수 있게
    return true;
  } catch {
    return false;
  }
}

/** 간호사 창에서 환자 창으로 넘기지 않는 키: 카드 목록을 키보드로 넘겨 보는 이동 키. */
export const NURSE_NON_FORWARD_KEYS = Object.freeze([
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End',
]);

/**
 * 간호사 창에 들어온 키를 환자 창의 스위치 누름으로 넘길지. 스위치 판정에 더해 이동 키는 넘기지 않는다
 * (간호사가 목록을 키보드로 넘기다가 환자 화면의 녹화·취소·선택이 일어나지 않게, 이동 키 스크롤도 그대로 둔다).
 * @param {{key?: string, ctrlKey?: boolean, metaKey?: boolean, altKey?: boolean, target?: any}} event
 * @returns {boolean}
 */
export function forwardIsSwitch(event) {
  try {
    if (!event || NURSE_NON_FORWARD_KEYS.includes(event.key)) return false;
    return keyIsSwitch(event);
  } catch {
    return false;
  }
}

/**
 * 포인터(마우스·터치·펜) 누름이 스위치 누름인지. 마우스는 왼쪽 버튼만.
 * @param {{button?: number, pointerType?: string, target?: any}} event
 * @returns {boolean}
 */
export function pointerIsSwitch(event) {
  try {
    if (!event) return false;
    if (event.pointerType === 'mouse' && event.button !== 0) return false;
    if (typeof event.button === 'number' && event.button > 0) return false;
    return !insideNoSwitch(event.target);
  } catch {
    return false;
  }
}
