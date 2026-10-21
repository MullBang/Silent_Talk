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
 * 촬영 순서: 조건별로 지정 문장 전체를 반복 횟수만큼 돈 뒤 자유 문장(유성)·무발화를 붙인다.
 * @param {{fixed:Array, free:Array, nonspeech:number}} prompts
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
  for (let i = 1; i <= (prompts.nonspeech || 0); i += 1) {
    plan.push({ cond: 'none', sentenceId: `X${String(i).padStart(2, '0')}`, text: '(말하지 않고 카메라를 보세요)', rep: 1, kind: 'nonspeech' });
  }
  return plan;
}

/** 클립 파일 기본 이름: {가명ID}_{조건}_{문장ID}_{회차}. */
export function clipBaseName(participant, item) {
  return `${safeName(participant)}_${item.cond}_${item.sentenceId}_${item.rep}`;
}
