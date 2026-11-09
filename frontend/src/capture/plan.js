// plan.js — 파일럿·병실 명령 촬영 순서와 파일명 규칙 (순수 함수, node 테스트 가능).

export const CONDITIONS = [
  { id: 'voiced', label: '소리 내어 말하기', hint: '평소처럼 소리 내어 읽습니다 (영상만 저장됩니다).' },
  { id: 'silent', label: '소리 없이 입만 움직이기', hint: '목소리를 내지 않고 입 모양만 또렷하게 움직입니다.' },
];

/** 병실 명령 조정용의 기댄 자세 조건(소리 없이). 촬영 순서를 도는 CONDITIONS에는 넣지 않는다. */
export const RECLINED_CONDITION = {
  id: 'reclined',
  label: '등받이에 기대어 소리 없이',
  hint: '등받이에 기대어 앉고 노트북은 침상 테이블(낮은 받침) 위에 둔 채, 목소리 없이 입 모양만 움직입니다.',
};

/** 조건 ID의 표시 정보. 무발화(none)처럼 목록에 없으면 null. */
export function conditionInfo(id) {
  return CONDITIONS.find((c) => c.id === id) || (id === RECLINED_CONDITION.id ? RECLINED_CONDITION : null);
}

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
 * 병실 명령 묶음에서 클립을 저장한 뒤 녹화 시작을 막는 시간(ms). 문장 촬영처럼 끝에 스페이스바를 습관대로 누르면
 * 5초 자동 정지 직후 다음 클립이 준비 없이 시작되고, 그 클립은 멈출 수도 없으므로 잠깐 시작을 받지 않는다.
 */
export const WARD_START_COOLDOWN_MS = 1000;

/**
 * 병실 명령 촬영 묶음 (목록 ward_prompts.json + 명령 commands.json). 실제 사용과 같이 5초에 저절로 멈추고 중간에 멈출 수 없다.
 * repeats는 조건별 명령 반복 횟수(소리 내어 1회는 등록용), decoys는 ward_prompts.decoys의 어느 목록을 찍을지,
 * reclined는 기댄 자세 무음 클립을 찍을지다. 구성·클립 수는 ward_prompts.json의 sessions와 테스트로 맞춘다.
 */
export const WARD_SESSIONS = [
  { id: 'ward_tune', label: '병실 명령 조정용 (5초 자동 정지, 93클립)', short: '병실 명령 조정용', promptSet: 'ward',
    repeats: { voiced: 1, silent: 2 }, decoys: 'tune', reclined: true, maxSeconds: 5, manualStop: false,
    startCooldownMs: WARD_START_COOLDOWN_MS },
  { id: 'ward_eval', label: '병실 명령 평가용 (5초 자동 정지, 103클립)', short: '병실 명령 평가용', promptSet: 'ward',
    repeats: { voiced: 1, silent: 3 }, decoys: 'eval', reclined: false, maxSeconds: 5, manualStop: false,
    startCooldownMs: WARD_START_COOLDOWN_MS },
];

/** 촬영 화면에서 고를 수 있는 모든 묶음 (문장 촬영 + 병실 명령 촬영). */
export const ALL_SESSIONS = [...SESSIONS, ...WARD_SESSIONS];

/** 묶음 ID로 묶음을 찾는다. 모르는 ID면 파일럿·조정용. */
export function findSession(id) {
  return ALL_SESSIONS.find((x) => x.id === id) || SESSIONS[0];
}

/** 병실 명령 묶음인지 (클립 .json의 session이 ward_로 시작). */
export function isWardSession(session) {
  const id = typeof session === 'string' ? session : session?.id;
  return typeof id === 'string' && id.startsWith('ward_');
}

/** 문장 촬영의 한 클립 최대 길이(초). 스페이스바로 멈추고, 이 시간이 지나면 저절로 멈춘다. */
export const SENTENCE_MAX_SECONDS = 8;

/**
 * 묶음의 녹화 규칙: 최대 길이(초), 손으로 멈출 수 있는지, 저장 뒤 시작을 막는 시간(ms).
 * 문장 촬영은 8초까지 스페이스바로 멈추고(시작 막음 없음), 병실 명령 촬영은 실제 사용처럼 5초에 저절로만 멈춘다.
 * @param {object} session SESSIONS 또는 WARD_SESSIONS의 항목
 * @returns {{maxSeconds:number, manualStop:boolean, startCooldownMs:number}}
 */
export function recordingRule(session) {
  return {
    maxSeconds: Number.isFinite(session?.maxSeconds) ? session.maxSeconds : SENTENCE_MAX_SECONDS,
    manualStop: session?.manualStop !== false,
    startCooldownMs: Number.isFinite(session?.startCooldownMs) ? session.startCooldownMs : 0,
  };
}

/**
 * 스페이스바·녹화 버튼을 눌렀을 때 할 일.
 * @param {{status:string, done:boolean, saving:boolean, blocked:boolean, manualStop:boolean, cooldown?:boolean}} state
 *   cooldown: 병실 명령 묶음에서 방금 클립을 저장해 아직 시작을 받지 않는 동안
 * @returns {('start'|'stop'|null)} 시작·정지 또는 아무것도 하지 않음(녹화 중 손으로 멈출 수 없는 묶음 포함)
 */
export function recordToggleAction({ status, done, saving, blocked, manualStop, cooldown = false }) {
  if (done || saving) return null;
  if (status === 'recording') return manualStop ? 'stop' : null;
  if (status === 'ready' && !blocked && !cooldown) return 'start';
  return null;
}

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
 * 조건별 반복 횟수. 숫자 하나면 모든 조건에 같은 횟수(파일럿·평가용), 객체면 조건 ID별 횟수(없는 조건은 0회).
 * @param {number|Object<string,number>} repeats
 * @param {string} condId
 * @returns {number}
 */
export function repeatsFor(repeats, condId) {
  const n = typeof repeats === 'number' ? repeats : (repeats || {})[condId];
  return Number.isInteger(n) && n > 0 ? n : 0;
}

/**
 * 촬영 순서: 조건별로 지정 문장 전체를 반복 횟수만큼 돈 뒤 자유 문장(유성)·무발화를 붙인다.
 * @param {{fixed:Array, free?:Array, nonspeech?:(number|Array<{id:string,text:string}>)}} prompts
 * @param {number|Object<string,number>} [repeats=2] 모든 조건의 반복 횟수, 또는 {voiced, silent} 조건별 횟수
 * @returns {Array<{cond:string, sentenceId:string, text:string, rep:number, kind:string}>}
 */
export function buildPlan(prompts, repeats = 2) {
  const plan = [];
  for (const cond of CONDITIONS) {
    for (let rep = 1; rep <= repeatsFor(repeats, cond.id); rep += 1) {
      for (const p of prompts.fixed) plan.push({ cond: cond.id, sentenceId: p.id, text: p.text, rep, kind: 'fixed' });
    }
  }
  for (const p of prompts.free || []) plan.push({ cond: 'voiced', sentenceId: p.id, text: p.text, rep: 1, kind: 'free' });
  for (const p of nonspeechItems(prompts.nonspeech)) {
    plan.push({ cond: 'none', sentenceId: p.id, text: p.text, rep: 1, kind: 'nonspeech' });
  }
  return plan;
}

/**
 * 병실 명령 촬영 순서: 명령(소리 내어 등록 1회 → 소리 없이 n회) → 목록 밖 문장(소리 없이) → 짧은 긴급 표현(소리 없이)
 * → 말하지 않는 동작 → (조정용만) 등받이에 기댄 자세로 명령 일부(소리 없이). 자세를 바꾸는 클립은 맨 끝에 모았다.
 * 모든 항목에 posture('upright'|'reclined')를 붙인다. 파일명 규칙은 파일럿과 같다(기댄 자세는 조건 reclined).
 * @param {{fixed:Array<{id:string,text:string}>, decoys:Object<string,Array>, short:Array, nonspeech:Array, reclined:string[]}} ward
 *   ward_prompts.json (fixed는 commands.json에서 가져온 명령)
 * @param {object} session WARD_SESSIONS의 항목
 * @returns {Array<{cond:string, sentenceId:string, text:string, rep:number, kind:string, posture:string}>}
 */
export function buildWardPlan(ward, session) {
  const upright = (item) => ({ ...item, posture: 'upright' });
  const plan = buildPlan({ fixed: ward.fixed }, session.repeats).map(upright);
  const decoys = (ward.decoys || {})[session.decoys];
  if (session.decoys && !Array.isArray(decoys)) throw new Error(`ward_prompts.json에 함정 문장 목록 '${session.decoys}'가 없습니다`);
  for (const p of decoys || []) plan.push(upright({ cond: 'silent', sentenceId: p.id, text: p.text, rep: 1, kind: 'decoy' }));
  for (const p of ward.short || []) plan.push(upright({ cond: 'silent', sentenceId: p.id, text: p.text, rep: 1, kind: 'short' }));
  for (const p of nonspeechItems(ward.nonspeech)) {
    plan.push(upright({ cond: 'none', sentenceId: p.id, text: p.text, rep: 1, kind: 'nonspeech' }));
  }
  if (session.reclined) {
    for (const id of ward.reclined || []) {
      const cmd = ward.fixed.find((p) => p.id === id);
      if (!cmd) throw new Error(`기댄 자세 명령 ${id}가 명령 목록에 없습니다`);
      plan.push({ cond: RECLINED_CONDITION.id, sentenceId: cmd.id, text: cmd.text, rep: 1, kind: 'fixed', posture: 'reclined' });
    }
  }
  return plan;
}

/**
 * 묶음에 맞는 촬영 순서. 병실 명령 묶음은 buildWardPlan, 그 밖은 기존 buildPlan(prompts, repeats)과 같다.
 * @param {object} session ALL_SESSIONS의 항목
 * @param {object} prompts prompts.json
 * @param {object} ward ward_prompts.json (fixed는 commands.json의 명령)
 */
export function buildSessionPlan(session, prompts, ward) {
  return isWardSession(session) ? buildWardPlan(ward, session) : buildPlan(prompts, session.repeats);
}

/**
 * 병실 촬영 목록: ward_prompts.json에 commands.json의 명령(id·text)을 fixed로 넣는다 (명령 문장은 commands.json 하나만 기준).
 * @param {object} wardPrompts ward_prompts.json
 * @param {{version:string, fixed:Array<{id:string,text:string}>}} commands commands.json
 */
export function wardPromptSet(wardPrompts, commands) {
  return { ...wardPrompts, commandsVersion: commands.version, fixed: commands.fixed.map(({ id, text }) => ({ id, text })) };
}

/** 촬영 화면 오른쪽 위에 보여 줄 클립 종류 이름. */
export function kindLabel(kind, ward = false) {
  if (kind === 'fixed') return ward ? '병실 명령' : '지정 문장';
  return { free: '자유 문장', decoy: '목록 밖 문장', short: '짧은 말', nonspeech: '무발화' }[kind] || kind;
}

/** 클립 파일 기본 이름: {가명ID}_{조건}_{문장ID}_{회차}. */
export function clipBaseName(participant, item) {
  return `${safeName(participant)}_${item.cond}_${item.sentenceId}_${item.rep}`;
}

/**
 * 클립 영상 옆에 저장할 .json 기록.
 * 문장 촬영(파일럿·평가용)은 예전과 같은 필드·순서·값이다(maxSeconds 8, prompts.json 버전, 자세·명령 버전 없음).
 * 병실 명령 묶음만 promptsVersion을 ward_prompts.json 버전으로 쓰고 commandsVersion·posture를 더한다.
 * @param {{session:object, item:object, participant:string, resultMeta:object, spoken?:string,
 *   prompts:{version:string}, ward:{version:string, commandsVersion:string}}} args
 *   resultMeta는 useRecorder가 준 녹화 통계(extension 포함), spoken은 자유 문장에 적은 정답 문장
 * @returns {object}
 */
export function clipMeta({ session, item, participant, resultMeta, spoken, prompts, ward }) {
  const isWard = isWardSession(session);
  return {
    ...resultMeta, participant: safeName(participant), condition: item.cond, sentenceId: item.sentenceId, text: item.text,
    kind: item.kind, rep: item.rep, maxSeconds: recordingRule(session).maxSeconds,
    file: `${clipBaseName(participant, item)}.${resultMeta.extension}`,
    session: session.id, repeats: session.repeats, promptsVersion: isWard ? ward.version : prompts.version,
    ...(isWard ? { commandsVersion: ward.commandsVersion, posture: item.posture } : {}),
    ...(item.kind === 'free' ? { spoken: normalizeSpoken(spoken) } : {}),
  };
}

/**
 * 폴더에 저장할 수 없을 때 내려받는 zip과 그 안 폴더의 이름.
 * 문장 촬영은 예전 이름({가명ID}_웹캠촬영) 그대로, 병실 명령 묶음은 묶음 ID를 넣어 문장 촬영 zip과 같은 폴더로 풀리지 않게 한다.
 * @param {string} participant 가명 ID
 * @param {object} session ALL_SESSIONS의 항목
 * @returns {string}
 */
export function zipFolderName(participant, session) {
  const pid = safeName(participant || 'clips');
  return isWardSession(session) ? `${pid}_${session.id}_웹캠촬영` : `${pid}_웹캠촬영`;
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
  if (typeof meta.estimatedFps === 'number' && meta.estimatedFps < 20) {
    out.push(`실제 프레임이 초당 ${meta.estimatedFps.toFixed(1)}개로 적습니다. 조명을 밝게 하고 카메라를 쓰는 다른 프로그램을 꺼 보세요. 그래도 같으면 이 카메라의 한계이니 그대로 계속 찍어도 됩니다.`);
  }
  if (typeof meta.maxGapMs === 'number' && meta.maxGapMs > 300) {
    out.push(`영상이 ${meta.maxGapMs.toFixed(0)}ms 동안 멈췄습니다. 다른 프로그램을 끄고 '이전 (다시 찍기)'로 다시 찍으세요.`);
  }
  return out;
}

/**
 * 자유 문장으로 입력한 정답 문장을 정리한다 (앞뒤 공백 제거, 연속 공백 하나로).
 * @param {string} text
 * @returns {string}
 */
export function normalizeSpoken(text) {
  return String(text || '').trim().replace(/\s+/g, ' ');
}

/**
 * 자유 문장 정답으로 쓸 수 없는 입력이면 고칠 방법을 알려 준다. 쓸 수 있으면 null.
 * 채점 사전(한글 자모)이 숫자·영어·기호를 받지 않으므로 소리 나는 대로 한글로 적게 한다.
 * @param {string} text
 * @returns {string|null}
 */
export function spokenProblem(text) {
  const t = normalizeSpoken(text);
  if (!t) return '말할 문장을 먼저 입력하세요.';
  if (/[0-9]/.test(t)) return '숫자는 소리 나는 대로 한글로 적으세요 (예: 3시 → 세 시).';
  if (/[A-Za-z]/.test(t)) return '영어는 쓰지 않습니다. 한국어로만 말하고 적으세요.';
  if (!/^[가-힣]+( [가-힣]+)*$/.test(t)) return '한글과 띄어쓰기만 적으세요 (문장부호·특수문자·자음만 쓰기 없이).';
  return null;
}
