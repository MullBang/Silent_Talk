// plan.js — 파일럿 촬영 순서와 파일명 규칙 (순수 함수, node 테스트 가능).

export const CONDITIONS = [
  { id: 'voiced', label: '소리 내어 말하기', hint: '평소처럼 소리 내어 읽습니다 (영상만 저장됩니다).' },
  { id: 'silent', label: '소리 없이 입만 움직이기', hint: '목소리를 내지 않고 입 모양만 또렷하게 움직입니다.' },
];

/** 파일명에 쓸 수 없는 문자를 제거한다. */
export function safeName(value) {
  return String(value || '').replace(/[^0-9A-Za-z가-힣_-]/g, '').slice(0, 32) || 'anon';
}

/**
 * 촬영 묶음: 파일럿·조정용은 지정 문장 2회, 처음 보는 사람의 최종 평가용은 3회.
 * 두 묶음은 저장 폴더를 나누고, 클립 .json의 session으로도 구분한다.
 */
export const SESSIONS = [
  { id: 'pilot', label: '파일럿·조정용 (지정 문장 2회, 100클립)', repeats: 2 },
  { id: 'eval', label: '최종 평가용 (지정 문장 3회, 140클립)', repeats: 3 },
];

/**
 * 저장 폴더에 이미 있는 클립 기록(.json)으로 그 폴더의 촬영 묶음을 정한다.
 * session이 없는 클립 기록(v1)은 파일럿으로 찍은 것으로 본다. 묶음이 둘 이상이면 mixed.
 * @param {Array<object>} metas 폴더 안 .json을 읽은 객체 목록(클립 기록이 아닌 것은 kind가 없다)
 * @returns {{session: (string|null), mixed: boolean, clips: number}}
 */
export function sessionFromMetas(metas) {
  const clipMetas = (metas || []).filter((m) => m && typeof m === 'object' && m.kind);
  const found = new Set(clipMetas.map((m) => m.session || 'pilot'));
  if (found.size > 1) return { session: null, mixed: true, clips: clipMetas.length };
  return { session: found.size ? [...found][0] : null, mixed: false, clips: clipMetas.length };
}

const DEFAULT_NONSPEECH_TEXT = '(말하지 않고 카메라를 보세요)';

/** 무발화 지시 목록: 배열이면 그대로, 숫자면(구형 v1) 같은 지시를 그 수만큼 만든다. */
export function nonspeechItems(nonspeech) {
  if (Array.isArray(nonspeech)) return nonspeech;
  return Array.from({ length: nonspeech || 0 }, (_, i) => ({ id: `X${String(i + 1).padStart(2, '0')}`, text: DEFAULT_NONSPEECH_TEXT }));
}

/**
 * 촬영 순서: 조건별로 지정 문장 전체를 반복 횟수만큼 돈 뒤 자유 문장(유성)·무발화를 붙인다.
 * @param {{fixed:Array, free:Array, nonspeech:(number|Array<{id:string,text:string}>)}} prompts
 * @param {number} [repeats=2]
 * @returns {Array<{cond:string, sentenceId:string, text:string, rep:number, kind:string}>}
 */
export function buildPlan(prompts, repeats = 2) {
  const plan = [];
  for (const cond of CONDITIONS) {
    for (let rep = 1; rep <= repeats; rep += 1) {
      for (const p of prompts.fixed) plan.push({ cond: cond.id, sentenceId: p.id, text: p.text, rep, kind: 'fixed' });
    }
  }
  for (const p of prompts.free) plan.push({ cond: 'voiced', sentenceId: p.id, text: p.text, rep: 1, kind: 'free' });
  for (const p of nonspeechItems(prompts.nonspeech)) {
    plan.push({ cond: 'none', sentenceId: p.id, text: p.text, rep: 1, kind: 'nonspeech' });
  }
  return plan;
}

/** 클립 파일 기본 이름: {가명ID}_{조건}_{문장ID}_{회차}. */
export function clipBaseName(participant, item) {
  return `${safeName(participant)}_${item.cond}_${item.sentenceId}_${item.rep}`;
}
