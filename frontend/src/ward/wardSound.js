// wardSound.js — 병실 화면 알림음 (Web Audio, 외부 파일 없음). 패턴은 순수 데이터라 node --test로 검사한다.
// 브라우저 자동 재생 정책 때문에 사용자 조작(클릭·키) 안에서 enable()을 한 번 불러야 소리가 난다.

/** 알림음 패턴: [주파수(Hz, 0이면 쉼), 길이(초)]. */
export const BEEP_PATTERNS = Object.freeze({
  start: Object.freeze([[880, 0.15]]), // 환자: 녹화 시작
  normal: Object.freeze([[660, 0.16], [0, 0.06], [880, 0.2]]), // 간호사: 일반 카드 (한 번)
  emergency: Object.freeze([[988, 0.14], [0, 0.07], [988, 0.14], [0, 0.07], [988, 0.14]]), // 간호사: 긴급 (반복)
});

/** 긴급 카드 반복음 간격(ms). */
export const ALARM_REPEAT_MS = 2000;

/**
 * 패턴 전체 길이(초).
 * @param {Array<[number, number]>} pattern
 * @returns {number}
 */
export function patternSeconds(pattern) {
  return (Array.isArray(pattern) ? pattern : []).reduce((sum, [, d]) => sum + (Number.isFinite(d) ? d : 0), 0);
}

/**
 * 알림음 재생기를 만든다.
 * @param {() => any} [factory] AudioContext 생성 (테스트용 주입).
 * @returns {{enable: () => Promise<boolean>, isEnabled: () => boolean, play: (kind: string) => boolean, close: () => void}}
 */
export function createBeeper(factory) {
  let ctx = null;
  const make = factory || (() => {
    const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
    return Ctor ? new Ctor() : null;
  });
  return {
    async enable() {
      try {
        if (!ctx) ctx = make();
        if (!ctx) return false;
        if (ctx.state === 'suspended') await ctx.resume();
        return ctx.state === 'running';
      } catch {
        return false;
      }
    },
    isEnabled() {
      return !!ctx && ctx.state === 'running';
    },
    play(kind) {
      try {
        const pattern = BEEP_PATTERNS[kind];
        if (!ctx || ctx.state !== 'running' || !pattern) return false;
        let t = ctx.currentTime + 0.01;
        for (const [freq, dur] of pattern) {
          if (freq > 0) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.0001, t);
            gain.gain.exponentialRampToValueAtTime(0.35, t + 0.015);
            gain.gain.exponentialRampToValueAtTime(0.0001, t + dur);
            osc.connect(gain).connect(ctx.destination);
            osc.start(t);
            osc.stop(t + dur + 0.02);
          }
          t += dur;
        }
        return true;
      } catch {
        return false;
      }
    },
    close() {
      try {
        ctx?.close();
      } catch {
        /* 이미 닫힘 */
      }
      ctx = null;
    },
  };
}
