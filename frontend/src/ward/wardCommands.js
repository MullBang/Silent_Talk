// wardCommands.js — 병실 명령 목록(commands.json) 조회와 가상 병실 동작 순수 함수 (node --test로 검사, JSX·DOM 없음).
// commands.json은 동결 파일이다. 여기서는 읽기만 하고, 화면·카드에는 존댓말 display를 쓴다(text는 인식용).

/** 서버 지정 문장 묶음 ID (POST /api/record phrase_set). */
export const WARD_PHRASE_SET = 'ward';
/** "방금 말한 건 취소할게요": 마지막 일반 카드에 취소 요청을 덧붙인다. */
export const CANCEL_COMMAND_ID = 'W20';
/** 체온 확인을 권하는 카드 문구 (W08 추워요·W09 더워요). */
export const TEMPERATURE_HINT = '체온 확인 권장';
/** 카드에 '체온 확인 권장'을 붙이는 동작. */
export const TEMPERATURE_ACTIONS = Object.freeze(['request_blanket', 'report_hot']);
/** 환자 몸에 닿는 동작: 요청만 보내고, 간호사 확인 뒤에만 가상 침대가 움직인다 (W07 자세·W14 등받이). */
export const ACK_ONLY_ACTIONS = Object.freeze(['request_reposition', 'request_backrest']);
/** 카드를 보낼 때 바로 바뀌는 가상 병실 동작. */
export const SEND_ROOM_ACTIONS = Object.freeze(['light_on', 'sleep_mode', 'curtain_close', 'tv_on', 'request_blanket']);

/** 가상 병실 처음 상태. 조명은 꺼진 상태에서 시작한다(시연: "불 좀 켜 줘"). */
export const INITIAL_ROOM = Object.freeze({
  light: 'off', // 'off' | 'on' | 'night'(수면등)
  curtain: 'open', // 'open' | 'closed'
  bedAngle: 30, // 등받이 각도(도)
  position: '바로 누움',
  tv: false,
  temperature: 24, // 가상 실내 온도(℃)
});

export const BED_ANGLE_STEP = 15;
export const BED_ANGLE_MAX = 60;
export const TEMPERATURE_MAX = 28;
export const POSITIONS = Object.freeze(['바로 누움', '왼쪽으로 누움', '오른쪽으로 누움']);

/**
 * commands.json을 조회용 색인으로 만든다. 형식이 틀린 항목이 있으면 오류를 던진다(동결 파일 손상 감지).
 * @param {{categories: Array<{id:string,label:string}>, fixed: Array<{id:string,text:string,display:string,
 *   category:string,emergency:boolean,action:string}>}} json
 * @returns {{list: Array<object>, byId: Map<string, object>, categories: Array<{id:string,label:string,commands:Array<object>}>}}
 */
export function buildCommandIndex(json) {
  try {
    const fixed = Array.isArray(json?.fixed) ? json.fixed : [];
    const categories = Array.isArray(json?.categories) ? json.categories : [];
    const list = fixed.map((c) => {
      const ok = c && typeof c.id === 'string' && typeof c.text === 'string' && typeof c.display === 'string'
        && typeof c.category === 'string' && typeof c.emergency === 'boolean' && typeof c.action === 'string';
      if (!ok) throw new Error(`명령 형식 오류: ${JSON.stringify(c)}`);
      return Object.freeze({ ...c });
    });
    if (list.length < 2) throw new Error('명령이 2개 미만입니다');
    const byId = new Map(list.map((c) => [c.id, c]));
    if (byId.size !== list.length) throw new Error('명령 ID가 겹칩니다');
    const cats = categories.map((cat) => Object.freeze({
      id: String(cat.id),
      label: String(cat.label),
      commands: Object.freeze(list.filter((c) => c.category === cat.id)),
    }));
    return Object.freeze({ list: Object.freeze(list), byId, categories: Object.freeze(cats) });
  } catch (err) {
    throw new Error(`병실 명령 목록을 읽지 못했습니다: ${err.message}`);
  }
}

/**
 * ID로 명령을 찾는다. 없으면 null.
 * @param {{byId: Map<string, object>}} index
 * @param {string|null|undefined} id
 * @returns {object|null}
 */
export function commandById(index, id) {
  try {
    if (typeof id !== 'string') return null;
    return index?.byId?.get(id) || null;
  } catch {
    return null;
  }
}

/**
 * 가상 병실 동작이 언제 적용되는지.
 * @param {string|null|undefined} action
 * @returns {'send'|'ack'|null} 'send'는 카드를 보낼 때, 'ack'는 간호사가 확인했을 때.
 */
export function roomActionTiming(action) {
  if (SEND_ROOM_ACTIONS.includes(action)) return 'send';
  if (ACK_ONLY_ACTIONS.includes(action)) return 'ack';
  return null;
}

/**
 * 가상 병실에 동작 하나를 적용한다. 바뀌는 것이 없으면 같은 객체를 돌려준다.
 * 시점(보낼 때/간호사 확인 때)은 호출하는 쪽이 roomActionTiming으로 정한다.
 * @param {typeof INITIAL_ROOM} room
 * @param {string} action commands.json의 action.
 * @returns {typeof INITIAL_ROOM}
 */
export function applyRoomAction(room, action) {
  try {
    const r = room && typeof room === 'object' ? room : INITIAL_ROOM;
    switch (action) {
      case 'light_on':
        return r.light === 'on' ? r : { ...r, light: 'on' };
      case 'sleep_mode':
        // 조명은 끄고 수면등은 유지한다. TV도 끈다.
        return r.light === 'night' && !r.tv ? r : { ...r, light: 'night', tv: false };
      case 'curtain_close':
        return r.curtain === 'closed' ? r : { ...r, curtain: 'closed' };
      case 'tv_on':
        return r.tv ? r : { ...r, tv: true };
      case 'request_blanket':
        return r.temperature >= TEMPERATURE_MAX ? r : { ...r, temperature: r.temperature + 1 };
      case 'request_backrest':
        return r.bedAngle >= BED_ANGLE_MAX ? r : { ...r, bedAngle: Math.min(BED_ANGLE_MAX, r.bedAngle + BED_ANGLE_STEP) };
      case 'request_reposition': {
        const i = POSITIONS.indexOf(r.position);
        return { ...r, position: POSITIONS[(i + 1) % POSITIONS.length] };
      }
      default:
        return r;
    }
  } catch {
    return room;
  }
}

/**
 * 가상 병실 상태를 화면 표시용 항목으로 바꾼다.
 * @param {typeof INITIAL_ROOM} room
 * @returns {Array<{key:string, label:string, value:string, on:boolean}>}
 */
export function describeRoom(room) {
  try {
    const r = room && typeof room === 'object' ? room : INITIAL_ROOM;
    const light = { on: '켜짐', off: '꺼짐', night: '수면등' }[r.light] || '꺼짐';
    return [
      { key: 'light', label: '조명', value: light, on: r.light !== 'off' },
      { key: 'curtain', label: '커튼', value: r.curtain === 'closed' ? '닫힘' : '열림', on: r.curtain === 'closed' },
      { key: 'bed', label: '침대', value: `등받이 ${r.bedAngle}° · ${r.position}`, on: true },
      { key: 'tv', label: 'TV', value: r.tv ? '켜짐' : '꺼짐', on: !!r.tv },
      { key: 'temperature', label: '온도', value: `${r.temperature}℃`, on: true },
    ];
  } catch {
    return [];
  }
}

/**
 * 가상 병실 한 줄 요약 (간호사 화면).
 * @param {typeof INITIAL_ROOM} room
 * @returns {string}
 */
export function roomSummary(room) {
  return describeRoom(room).map((x) => `${x.label} ${x.value}`).join(' · ');
}
