// wardPractice.js — 연습 모드(#ward?practice=1)용 가짜 서버 결과. 카메라 없이 리허설·검증할 때 쓴다 (node --test로 검사).
// 결과 모양은 GET /api/result의 results와 같다. 기본 화면에서는 이 버튼들을 숨긴다.

import { DETECTION_FAILED_TEXT, NO_SPEECH_TEXT } from '../results/markers.js';
import { WARD_PHRASE_SET, commandById } from './wardCommands.js';

/** 연습 장면. needsCommand가 true면 고른 명령으로 만든다. */
export const PRACTICE_SCENARIOS = Object.freeze([
  Object.freeze({ id: 'accepted_normal', label: '고름 · 일반', needsCommand: true }),
  Object.freeze({ id: 'accepted_emergency', label: '고름 · 긴급(숨)' }),
  Object.freeze({ id: 'uncertain_emergency', label: '못 고름 · 2등 긴급' }),
  Object.freeze({ id: 'uncertain_normal', label: '못 고름 · 일반' }),
  Object.freeze({ id: 'enroll_normal', label: '글자로 못 고름 · 등록으로 고름', needsCommand: true }),
  Object.freeze({ id: 'enroll_emergency', label: '글자로 못 고름 · 등록으로 긴급(아파요)' }),
  Object.freeze({ id: 'no_phrase', label: '문장 결과 없음' }),
  Object.freeze({ id: 'no_speech', label: '발화 없음' }),
  Object.freeze({ id: 'not_detected', label: '입 검출 실패' }),
]);

/** 연습 결과의 서버 기준 (phrase.rule 모양, 값은 서버 기본값 = 2026-11-10 고정값과 같은 예시). */
export const PRACTICE_RULE = Object.freeze({
  min_margin: 0.7, min_score: -2.9, possible_min_score: null, enroll_max_distance: 0.26,
});

/**
 * 서버 phrase 모양을 만든다.
 * @param {object} index buildCommandIndex 결과.
 * @param {string} id 1등 명령 ID.
 * @param {string|null} runnerUpId 2등 명령 ID.
 * @param {boolean} accepted
 * @returns {object}
 */
function phraseOf(index, id, runnerUpId, accepted) {
  const command = commandById(index, id);
  return {
    id,
    text: command ? command.text : '',
    score: accepted ? -1.25 : -3.9,
    margin: accepted ? 1.4 : 0.18,
    runner_up_id: runnerUpId,
    accepted,
    set: WARD_PHRASE_SET,
    rule: { ...PRACTICE_RULE },
  };
}

/**
 * 등록 비교 결과(enroll, match 모드)가 고른 모양을 만든다.
 * @param {string} id 등록 1등 = 글자 1등 명령 ID.
 * @param {string} runnerUpId 등록 2등.
 * @returns {object}
 */
function enrollAccepted(id, runnerUpId) {
  return {
    mode: 'match', profile: 'practice', status: 'ok', count: 20, top_id: id, distance: 0.18, runner_up_id: runnerUpId,
    runner_up_distance: 0.31, distances: { [id]: 0.18, [runnerUpId]: 0.31 }, max_distance: PRACTICE_RULE.enroll_max_distance,
    within: true, agree: true, accepted: true,
  };
}

/** 결과 항목 하나 (5초 녹화 한 창). */
function item(text, phrase, enroll = null) {
  return {
    start_ms: 0, end_ms: 5000, text, confidence: phrase ? 0.8 : null, raw_score: null, ...(phrase ? { phrase } : {}),
    ...(enroll ? { enroll } : {}),
  };
}

/**
 * 연습 장면의 가짜 결과 배열.
 * @param {string} scenarioId PRACTICE_SCENARIOS의 id.
 * @param {object} index buildCommandIndex 결과.
 * @param {{commandId?: string}} [opts] accepted_normal에서 고를 명령 (기본 W05 물).
 * @returns {Array<object>}
 */
export function practiceResults(scenarioId, index, { commandId = 'W05' } = {}) {
  try {
    switch (scenarioId) {
      case 'accepted_normal': {
        const id = commandById(index, commandId) ? commandId : 'W05';
        return [item('연습 결과', phraseOf(index, id, id === 'W06' ? 'W05' : 'W06', true))];
      }
      case 'accepted_emergency':
        return [item('연습 결과', phraseOf(index, 'W02', 'W04', true))];
      case 'uncertain_emergency':
        return [item('연습 결과', phraseOf(index, 'W05', 'W04', false))];
      case 'uncertain_normal':
        return [item('연습 결과', phraseOf(index, 'W11', 'W13', false))];
      case 'enroll_normal': {
        const id = commandById(index, commandId) ? commandId : 'W05';
        const runner = id === 'W06' ? 'W05' : 'W06';
        return [item('연습 결과', phraseOf(index, id, runner, false), enrollAccepted(id, runner))];
      }
      case 'enroll_emergency':
        return [item('연습 결과', phraseOf(index, 'W03', 'W05', false), enrollAccepted('W03', 'W05'))];
      case 'no_phrase':
        return [item('연습 결과', null)];
      case 'no_speech':
        return [item(NO_SPEECH_TEXT, null)];
      case 'not_detected':
        return [item(DETECTION_FAILED_TEXT, null)];
      default:
        return [];
    }
  } catch {
    return [];
  }
}
