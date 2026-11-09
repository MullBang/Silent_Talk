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
  Object.freeze({ id: 'no_phrase', label: '문장 결과 없음' }),
  Object.freeze({ id: 'no_speech', label: '발화 없음' }),
  Object.freeze({ id: 'not_detected', label: '입 검출 실패' }),
]);

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
  };
}

/** 결과 항목 하나 (5초 녹화 한 창). */
function item(text, phrase) {
  return { start_ms: 0, end_ms: 5000, text, confidence: phrase ? 0.8 : null, raw_score: null, ...(phrase ? { phrase } : {}) };
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
