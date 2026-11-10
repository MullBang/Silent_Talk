// wardCommands.js — 병실 명령 목록(commands.json) 조회와 가상 병실 동작 순수 함수 (node --test로 검사, JSX·DOM 없음).
// commands.json은 동결 파일이다. 여기서는 읽기만 하고, 화면·카드에는 존댓말 display를 쓴다(text는 인식용).
// 사용자가 만든 '내 목록'(서버 GET /api/ward/lists/{id})도 같은 모양의 색인으로 만든다(buildCustomCommandIndex).
// 내 목록 명령은 가상 병실을 움직이지 않는다(action 'custom'). 취소 명령만 기본 목록 W20과 같은 'cancel_last'다.

/** 서버 지정 문장 묶음 ID (POST /api/record phrase_set). */
export const WARD_PHRASE_SET = 'ward';
/** "방금 말한 건 취소할게요": 마지막 일반 카드에 취소 요청을 덧붙인다 (기본 목록의 취소 명령 ID). */
export const CANCEL_COMMAND_ID = 'W20';
/** 취소 명령의 동작 이름. 상태도는 ID가 아니라 이 동작으로 취소 명령을 알아본다. */
export const CANCEL_ACTION = 'cancel_last';
/** 내 목록의 일반 명령 동작 (가상 병실·체온 안내 없음). */
export const CUSTOM_ACTION = 'custom';
/** 내 목록 명령 ID (서버가 저장할 때 U01, U02, … 순서로 붙인다). */
export const CUSTOM_COMMAND_ID_PATTERN = /^U\d{2}$/;
/** 내 목록 ID (서버 규칙과 같다). */
export const COMMAND_LIST_ID_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
/** 긴급 분류 ID (긴급 명령은 반드시 이 분류다). */
export const EMERGENCY_CATEGORY = 'emergency';
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
 * 기본 목록 색인은 customList가 null이고, cancelId는 취소 동작(cancel_last) 명령의 ID다(기본 목록은 W20).
 * @param {{categories: Array<{id:string,label:string}>, fixed: Array<{id:string,text:string,display:string,
 *   category:string,emergency:boolean,action:string}>}} json
 * @returns {{list: Array<object>, byId: Map<string, object>, categories: Array<{id:string,label:string,commands:Array<object>}>,
 *   cancelId: string|null, customList: null}}
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
    const cancels = list.filter((c) => c.action === CANCEL_ACTION);
    return Object.freeze({
      list: Object.freeze(list), byId, categories: Object.freeze(cats), cancelId: cancels.length === 1 ? cancels[0].id : null,
      customList: null,
    });
  } catch (err) {
    throw new Error(`병실 명령 목록을 읽지 못했습니다: ${err.message}`);
  }
}

/**
 * 서버의 내 목록(GET /api/ward/lists/{id})을 기본 목록과 같은 모양의 색인으로 만든다.
 * 분류 이름은 기본 목록(commands.json)의 분류를 쓰고, 명령이 하나도 없는 분류는 스캐닝에서 빼려고 넣지 않는다.
 * 일반 명령의 action은 'custom'(가상 병실을 움직이지 않음), 취소 명령은 'cancel_last'다.
 * 형식이 틀리면 오류를 던진다(호출하는 쪽이 오류를 보이고 기본 목록으로 바꾸지 않는다).
 * @param {{id:string, title:string, sha256:string, commands: Array<{id:string, text:string, display:string, category:string,
 *   emergency:boolean, cancel:boolean}>}} list 서버 응답.
 * @param {{categories: Array<{id:string,label:string}>}} base 분류 이름을 가져올 기본 목록(commands.json).
 * @returns {{list: Array<object>, byId: Map<string, object>, categories: Array<object>, cancelId: string|null,
 *   customList: {id:string, title:string, sha256:string, count:number}}}
 */
export function buildCustomCommandIndex(list, base) {
  try {
    if (!list || typeof list !== 'object') throw new Error('목록이 비어 있습니다');
    if (typeof list.id !== 'string' || !COMMAND_LIST_ID_PATTERN.test(list.id)) throw new Error('목록 ID 형식 오류');
    if (typeof list.title !== 'string' || !list.title.trim()) throw new Error('목록 이름이 없습니다');
    if (typeof list.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(list.sha256)) throw new Error('목록 지문(sha256) 형식 오류');
    const baseCats = Array.isArray(base?.categories) ? base.categories : [];
    const catIds = new Set(baseCats.map((c) => String(c.id)));
    const raw = Array.isArray(list.commands) ? list.commands : [];
    const commands = raw.map((c) => {
      const ok = c && typeof c.id === 'string' && CUSTOM_COMMAND_ID_PATTERN.test(c.id)
        && typeof c.text === 'string' && c.text.trim() && typeof c.display === 'string' && c.display.trim()
        && typeof c.category === 'string' && catIds.has(c.category) && typeof c.emergency === 'boolean'
        && c.emergency === (c.category === EMERGENCY_CATEGORY) && (c.cancel === undefined || typeof c.cancel === 'boolean')
        && !(c.cancel === true && c.emergency);
      if (!ok) throw new Error(`명령 형식 오류: ${JSON.stringify(c)}`);
      return Object.freeze({
        id: c.id, text: c.text, display: c.display, category: c.category, emergency: c.emergency,
        action: c.cancel === true ? CANCEL_ACTION : CUSTOM_ACTION,
      });
    });
    if (commands.length < 2) throw new Error('명령이 2개 미만입니다');
    const byId = new Map(commands.map((c) => [c.id, c]));
    if (byId.size !== commands.length) throw new Error('명령 ID가 겹칩니다');
    const cancels = commands.filter((c) => c.action === CANCEL_ACTION);
    if (cancels.length > 1) throw new Error('취소 명령이 둘 이상입니다');
    const cats = baseCats
      .map((cat) => Object.freeze({
        id: String(cat.id),
        label: String(cat.label),
        commands: Object.freeze(commands.filter((c) => c.category === cat.id)),
      }))
      .filter((cat) => cat.commands.length > 0);
    return Object.freeze({
      list: Object.freeze(commands),
      byId,
      categories: Object.freeze(cats),
      cancelId: cancels.length ? cancels[0].id : null,
      customList: Object.freeze({ id: list.id, title: list.title.trim(), sha256: list.sha256, count: commands.length }),
    });
  } catch (err) {
    throw new Error(`내 목록을 읽지 못했습니다: ${err.message}`);
  }
}

/**
 * 명령이 하나도 없는 색인 (내 목록을 불러오는 중이거나 불러오지 못했을 때). 기본 목록으로 바꾸지 않으려고 쓴다.
 * @param {string} listId 쓰려던 내 목록 ID.
 * @param {string} [title]
 * @returns {{list: Array<object>, byId: Map<string, object>, categories: Array<object>, cancelId: null,
 *   customList: {id:string, title:string, sha256:null, count:0}}}
 */
export function emptyListIndex(listId, title = '') {
  return Object.freeze({
    list: Object.freeze([]),
    byId: new Map(),
    categories: Object.freeze([]),
    cancelId: null,
    customList: Object.freeze({ id: String(listId || ''), title: String(title || ''), sha256: null, count: 0 }),
  });
}

/**
 * 취소 명령("방금 말한 건 취소")인지. 기본 목록에서는 W20만 해당한다.
 * @param {{action?: string}|null|undefined} command
 * @returns {boolean}
 */
export function isCancelCommand(command) {
  return !!command && command.action === CANCEL_ACTION;
}

/**
 * 환자 화면 '말할 수 있는 문장' 칸에 보일 묶음. 지금 쓰는 목록(기본 목록 또는 내 목록)의 분류 순서대로,
 * 명령이 없는 분류는 뺀다. 문장은 인식용 입 모양 문장(text)이고, 화면용 문장(display)은 함께 넘겨 보조로 쓴다.
 * @param {{list?: Array<object>, categories?: Array<{id:string, label:string, commands:Array<object>}>}|null} index
 * @returns {Array<{id:string, label:string, commands:Array<{id:string, text:string, display:string, emergency:boolean, cancel:boolean}>}>}
 */
export function phraseGroups(index) {
  try {
    const cats = Array.isArray(index?.categories) ? index.categories : [];
    return cats
      .map((cat) => ({
        id: String(cat.id),
        label: String(cat.label),
        commands: (Array.isArray(cat.commands) ? cat.commands : []).map((c) => ({
          id: c.id, text: c.text, display: c.display, emergency: c.emergency === true, cancel: isCancelCommand(c),
        })),
      }))
      .filter((cat) => cat.commands.length > 0);
  } catch {
    return [];
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
