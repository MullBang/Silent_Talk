// markers.js — 결과 마커·모델 정보 표시용 순수 함수 (node --test로 검사, JSX 없음).
// 백엔드 결과 마커: '[인식 불가]'(확신 낮음/빈 결과), '[검출 실패 구간]'(입술 미검출),
// '[발화 없음]'(입 움직임 없음, 6단계 녹화 경로). 셋 다 '인식하지 못한 구간'으로 센다.

export const UNRECOGNIZED_TEXT = '[인식 불가]';
export const DETECTION_FAILED_TEXT = '[검출 실패 구간]';
export const NO_SPEECH_TEXT = '[발화 없음]';

/** 비인식(마커) 텍스트 목록. 취소선·글자 수 제외 대상. */
export const NON_RECOGNITION_MARKERS = [UNRECOGNIZED_TEXT, DETECTION_FAILED_TEXT, NO_SPEECH_TEXT];

/** 처음 보는 사람 성능 미검증 안내 (결과 화면에 항상 표시). */
export const UNSEEN_PERSON_NOTICE = '처음 보는 사람의 입 모양 인식 정확도는 아직 검증되지 않았습니다.';

/**
 * 비인식(마커) 텍스트 여부.
 * @param {string|null|undefined} text
 * @returns {boolean}
 */
export function isNonRecognition(text) {
  return NON_RECOGNITION_MARKERS.includes(String(text ?? '').trim());
}

/**
 * '[인식 불가]' 항목에 서버가 남긴 낮은 신뢰도 추정 문장을 돌려준다. 없으면 null.
 * @param {{text?: string, candidate?: string|null}} item
 * @returns {string|null}
 */
export function lowConfidenceCandidate(item) {
  if (!item || String(item.text ?? '').trim() !== UNRECOGNIZED_TEXT) return null;
  const cand = typeof item.candidate === 'string' ? item.candidate.trim() : '';
  return cand ? cand : null;
}

/**
 * 지정 문장 보조 모드 결과를 화면용 값으로 바꾼다. 결과가 없으면 null.
 * @param {{phrase?: {text?: string, accepted?: boolean}|null}|null|undefined} item
 * @returns {{accepted: boolean, text: string}|null}
 */
export function phraseResult(item) {
  const phrase = item && typeof item.phrase === 'object' ? item.phrase : null;
  const text = phrase && typeof phrase.text === 'string' ? phrase.text.trim() : '';
  if (!text) return null;
  return { accepted: phrase.accepted === true, text };
}

/**
 * '전체 텍스트 복사'에서 항목 뒤에 붙일 지정 문장 줄. 고른 문장이 있을 때만 붙이고, 고르지 못한 후보는 복사하지 않는다.
 * @param {{phrase?: {text?: string, accepted?: boolean}|null}|null|undefined} item
 * @returns {string}
 */
export function phraseCopySuffix(item) {
  const phrase = phraseResult(item);
  return phrase && phrase.accepted ? `\n  지정 문장 모드: ${phrase.text}` : '';
}

/**
 * 결과 응답의 model 객체를 한 줄 설명으로 바꾼다. 표시할 값이 없으면 null.
 * 체크포인트 SHA는 앞 12자, upstream commit은 앞 7자만 보인다.
 * @param {{backend?:string, decoder?:string|null, checkpoint_sha256?:string|null,
 *   upstream_commit?:string|null, roi_profile?:string|null}|null|undefined} model
 * @returns {string|null}
 */
export function modelInfoLine(model) {
  if (!model || typeof model !== 'object') return null;
  const str = (v) => (typeof v === 'string' && v.trim() ? v.trim() : null);
  const parts = [];
  const backend = str(model.backend);
  if (backend) parts.push(backend === 'olkavs' ? 'OLKAVS 공식 모델' : backend);
  const decoder = str(model.decoder);
  if (decoder) parts.push(`디코더 ${decoder}`);
  const sha = str(model.checkpoint_sha256);
  if (sha) parts.push(`가중치 ${sha.slice(0, 12)}`);
  const commit = str(model.upstream_commit);
  if (commit) parts.push(`코드 ${commit.slice(0, 7)}`);
  const roi = str(model.roi_profile);
  if (roi) parts.push(`입술 영역 ${roi}`);
  return parts.length ? parts.join(' · ') : null;
}
