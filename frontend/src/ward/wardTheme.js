// wardTheme.js — 병실 화면 색 (명암비 7:1 이상). 색 짝은 node --test에서 WCAG 명암비로 검사한다.
// 긴급은 색만으로 구분하지 않는다: [긴급] 글자와 아이콘을 함께 쓴다.

export const WARD_COLORS = Object.freeze({
  ink: '#111827', // 기본 글자 (gray-900)
  paper: '#ffffff',
  inkSoft: '#374151', // 보조 글자 (gray-700)
  estimate: '#4b5563', // 간호사 화면 회색 '추정' 후보 (gray-600, 흰 바탕 7.5:1)
  line: '#6b7280',
  emergencyBg: '#991b1b', // red-800 (흰 글자 8.3:1)
  emergencyInk: '#ffffff',
  possibleBg: '#fde68a', // amber-200 (검정 글자)
  possibleInk: '#111827',
  checkBg: '#e5e7eb', // gray-200 (검정 글자)
  checkInk: '#111827',
  infoBg: '#1e3a8a', // blue-900 (흰 글자)
  infoInk: '#ffffff',
  okBg: '#14532d', // green-900 (흰 글자)
  okInk: '#ffffff',
  warnBg: '#fef3c7', // amber-100 (검정 글자)
  warnInk: '#111827',
  bannerBg: '#111827',
  bannerInk: '#fde68a',
  overdueBg: '#fef08a', // yellow-200 (60초 넘게 확인 안 됨)
  overdueInk: '#111827',
});

/** 결과 영역 색 (wardView의 tone). */
export const TONE_STYLES = Object.freeze({
  neutral: Object.freeze({ background: WARD_COLORS.paper, color: WARD_COLORS.ink }),
  recording: Object.freeze({ background: WARD_COLORS.infoBg, color: WARD_COLORS.infoInk }),
  confirm: Object.freeze({ background: WARD_COLORS.paper, color: WARD_COLORS.ink }),
  sent: Object.freeze({ background: WARD_COLORS.paper, color: WARD_COLORS.ink }),
  emergency: Object.freeze({ background: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyInk }),
  possible: Object.freeze({ background: WARD_COLORS.possibleBg, color: WARD_COLORS.possibleInk }),
  warning: Object.freeze({ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }),
  scan: Object.freeze({ background: WARD_COLORS.paper, color: WARD_COLORS.ink }),
});

/** 화면에서 실제로 겹쳐 쓰는 글자/바탕 짝 (검사 대상). */
export const CONTRAST_PAIRS = Object.freeze([
  ['ink', 'paper'], ['inkSoft', 'paper'], ['estimate', 'paper'], ['emergencyInk', 'emergencyBg'],
  ['possibleInk', 'possibleBg'], ['checkInk', 'checkBg'], ['infoInk', 'infoBg'], ['okInk', 'okBg'],
  ['warnInk', 'warnBg'], ['bannerInk', 'bannerBg'], ['paper', 'bannerBg'], ['overdueInk', 'overdueBg'],
  ['infoBg', 'paper'], // 환자 화면 녹화 상태 글자 (흰 바탕 위 파랑)
]);

/**
 * '#rrggbb'의 상대 휘도 (WCAG 2).
 * @param {string} hex
 * @returns {number}
 */
export function relativeLuminance(hex) {
  const m = /^#([0-9a-f]{6})$/i.exec(String(hex));
  if (!m) throw new Error(`색 형식 오류: ${hex}`);
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(m[1].slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/**
 * 두 색의 명암비 (1~21).
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
export function contrastRatio(a, b) {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** 녹화 중 화면 테두리 깜빡임 (녹화 파랑 infoBg, 긴급 빨강과 구분). */
export const WARD_KEYFRAMES = '@keyframes ward-flash { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0.15; } }';
