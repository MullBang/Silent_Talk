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
  { id: 'pilot', label: '파일럿·조정용 (지정 문장 2회, 100클립)', short: '파일럿·조정용', repeats: 2 },
  { id: 'eval', label: '최종 평가용 (지정 문장 3회, 140클립)', short: '최종 평가용', repeats: 3 },
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

/**
 * 폴더에 이미 있는 클립 이름으로 이 참가자가 이어 찍을 위치를 찾는다 (새로고침·창 닫힘 뒤 이어 찍기).
 * @param {Array<object>} plan buildPlan 결과
 * @param {string} participant 가명 ID
 * @param {Iterable<string>} existingNames 폴더 안 영상 파일의 기본 이름(확장자 제외)
 * @returns {{index:number, saved:number}} index = 아직 없는 첫 순번(모두 있으면 plan.length), saved = 이미 있는 클립 수
 */
export function resumePoint(plan, participant, existingNames) {
  const names = new Set(existingNames || []);
  let first = -1;
  let savedCount = 0;
  plan.forEach((item, i) => {
    if (names.has(clipBaseName(participant, item))) savedCount += 1;
    else if (first < 0) first = i;
  });
  return { index: first < 0 ? plan.length : first, saved: savedCount };
}

/**
 * from 순번부터 이 참가자의 아직 저장되지 않은 첫 순번 (없으면 plan.length).
 * 저장 뒤 다음 칸으로 갈 때 쓴다: '이전'으로 돌아가 다시 찍은 뒤 이미 저장된 다음 클립을 덮어쓰지 않게 한다.
 * @param {Array<object>} plan buildPlan 결과
 * @param {string} participant 가명 ID
 * @param {Iterable<string>} existingNames 폴더 안 영상 파일의 기본 이름(확장자 제외)
 * @param {number} from 찾기 시작할 순번
 * @returns {number}
 */
export function nextUnsaved(plan, participant, existingNames, from) {
  const names = new Set(existingNames || []);
  for (let i = Math.max(0, from); i < plan.length; i += 1) {
    if (!names.has(clipBaseName(participant, plan[i]))) return i;
  }
  return plan.length;
}

/**
 * 방금 저장한 클립의 녹화 통계로 다시 찍을 필요가 있는지 알려 준다.
 * 화면 옆의 카메라 fps는 요청값이라 어두울 때 실제로 줄어든 프레임을 보여 주지 않으므로 실제 프레임 간격으로 본다.
 * @param {{durationMs?:number, estimatedFps?:number|null, maxGapMs?:number|null}|null} meta
 * @returns {string[]} 경고 문구 (없으면 빈 배열)
 */
export function clipQualityWarnings(meta) {
  const out = [];
  if (!meta) return out;
  if (typeof meta.durationMs === 'number' && meta.durationMs < 1000) {
    out.push("녹화가 1초보다 짧습니다. 스페이스바를 두 번 누르지 않았는지 보고 '이전 (다시 찍기)'로 다시 찍으세요.");
  }
  if (typeof meta.estimatedFps === 'number' && meta.estimatedFps < 24) {
    out.push(`실제 프레임이 초당 ${meta.estimatedFps.toFixed(1)}개로 적습니다. 조명을 밝게 하고 카메라를 쓰는 다른 프로그램을 끈 뒤 다시 찍으세요.`);
  }
  if (typeof meta.maxGapMs === 'number' && meta.maxGapMs > 200) {
    out.push(`영상이 ${meta.maxGapMs.toFixed(0)}ms 동안 멈췄습니다. 다른 프로그램을 끄고 다시 찍으세요.`);
  }
  return out;
}
