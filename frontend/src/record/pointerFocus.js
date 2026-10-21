// pointerFocus.js — 마우스로 누른 보조 컨트롤(녹화 길이 버튼·카운트다운 체크박스)의 포커스를 푼다.
// 포커스가 남으면 스페이스바가 페이지 단축키(녹화 시작·정지) 대신 그 컨트롤을 다시 누른다.
// 키보드로 누른 경우(click detail 0)는 포커스를 그대로 둬 키보드 접근성을 유지한다.

/**
 * 마우스 클릭으로 생긴 포커스인지 (키보드 활성화의 click은 detail이 0).
 * @param {{detail?: number}|null|undefined} event click 이벤트.
 * @returns {boolean}
 */
export function isPointerClick(event) {
  return typeof event?.detail === 'number' && event.detail > 0;
}

/**
 * 마우스 클릭 직후, 포커스가 이 컨테이너 안에 있으면 풀어 준다.
 * 라벨 클릭이 체크박스로 넘어가는 기본 동작이 끝난 뒤 실행되도록 다음 작업으로 미룬다.
 * @param {{detail?: number, currentTarget?: any}} event click 이벤트 (컨테이너에 단 핸들러).
 * @param {{doc?: {activeElement?: any}, schedule?: (fn: () => void) => any}} [deps] 테스트용 주입.
 * @returns {boolean} 포커스 해제를 예약했으면 true.
 */
export function blurAfterPointerClick(event, deps = {}) {
  try {
    if (!isPointerClick(event)) return false;
    const container = event.currentTarget;
    if (!container) return false;
    const doc = deps.doc || (typeof document !== 'undefined' ? document : null);
    const schedule = deps.schedule || ((fn) => setTimeout(fn, 0));
    if (!doc) return false;
    schedule(() => {
      try {
        const active = doc.activeElement;
        if (!active || typeof active.blur !== 'function') return;
        const inside = active === container || (typeof container.contains === 'function' && container.contains(active));
        if (inside) active.blur();
      } catch {
        /* 포커스 해제 실패는 무시 (단축키만 영향) */
      }
    });
    return true;
  } catch {
    return false;
  }
}
