// wardSpeech.js — 음성 읽기(선택 기능). 오프라인에서 되는 한국어 음성(lang ko-KR, localService true)만 쓴다.
// 그런 음성이 없으면 읽지 않고 화면에 "음성 읽기 없음"을 표시한다. 고르는 함수는 node --test로 검사한다.

export const NO_VOICE_TEXT = '음성 읽기 없음';

/**
 * 오프라인 한국어 음성을 고른다. 없으면 null (온라인 음성은 쓰지 않는다).
 * @param {Array<{lang?: string, localService?: boolean, name?: string, default?: boolean}>|null|undefined} voices
 * @returns {object|null}
 */
export function pickKoreanLocalVoice(voices) {
  try {
    const list = Array.isArray(voices) ? voices : [];
    const ok = list.filter((v) => String(v?.lang || '').replace('_', '-').toLowerCase() === 'ko-kr' && v?.localService === true);
    return ok.find((v) => v.default) || ok[0] || null;
  } catch {
    return null;
  }
}

/**
 * 음성 상태 표시 문구.
 * @param {object|null} voice
 * @returns {string}
 */
export function voiceLabel(voice) {
  return voice ? `음성: ${voice.name || '한국어'}` : NO_VOICE_TEXT;
}

/**
 * 문장을 읽는다. 앞서 읽던 것은 멈춘다.
 * @param {{speak: Function, cancel: Function}|null|undefined} synth window.speechSynthesis
 * @param {object|null} voice pickKoreanLocalVoice 결과.
 * @param {string} text
 * @returns {boolean} 읽기를 시작했으면 true.
 */
export function speakText(synth, voice, text) {
  try {
    if (!synth || !voice || !text || typeof globalThis.SpeechSynthesisUtterance !== 'function') return false;
    synth.cancel();
    const u = new globalThis.SpeechSynthesisUtterance(text);
    u.voice = voice;
    u.lang = 'ko-KR';
    u.rate = 0.95;
    synth.speak(u);
    return true;
  } catch {
    return false;
  }
}
