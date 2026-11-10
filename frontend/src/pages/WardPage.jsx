// WardPage.jsx — 병실 무음 도우미 환자 화면 (#ward). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 스위치(아무 키·화면 누름) 한 번 → 시작음과 테두리 깜빡임으로 N초 녹화(카운트다운 없음) → POST /api/record(phrase_set='ward',
// 설정에 내 입 모양 등록 이름이 있으면 enroll_profile도) → 약 300ms 폴링 → 상태도(ward/wardMachine.js)가 결과를 처리한다.
// 고를지 말지는 서버 phrase.accepted와 등록 비교 결과(enroll.accepted·within)만 믿는다(판정표는 ward/wardResult.js).
// 간호사 화면(#nurse)과는 같은 주소의 BroadcastChannel로 카드·확인·스위치 입력·병실 상태를 주고받는다.
// 명령 목록: 설정의 commandList가 비어 있으면 기본 목록(commands.json)을 쓰고 요청도 예전과 같다. 내 목록이면 서버에서 불러와
// 그 문장들로 판정하고(command_list를 함께 보냄), 결과는 목록 ID·지문이 같을 때만 쓴다. 불러오지 못하면 입 읽기 녹화를 막고
// (기본 목록으로 몰래 바꾸지 않음) 스위치 3번 긴급 호출과 설정은 그대로 둔다. 연습 모드는 늘 기본 목록이다.
// 다른 탭(목록 관리·등록)이 명령 목록·등록 이름을 바꾸면 저장소 알림(storage)·창으로 돌아올 때(focus) 다시 읽는다.
// 녹화·확인·선택 중이면 끝난 뒤에 바꾼다. 이 화면에서 설정을 바꿔 저장할 때도 다른 탭이 바꾼 그 값들을 지우지 않는다.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getWardList, wardProfileApi } from '../api/client';
import useRecorder from '../hooks/useRecorder';
import { blurAfterPointerClick } from '../record/pointerFocus';
import COMMANDS from '../ward/commands.json';
import PracticePanel from '../ward/PracticePanel';
import PhrasePanel from '../ward/PhrasePanel';
import RoomPanel from '../ward/RoomPanel';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import WardSettingsPanel from '../ward/WardSettingsPanel';
import { ROOM_HEARTBEAT_MS, openWardChannel } from '../ward/wardChannel';
import { buildCommandIndex, emptyListIndex } from '../ward/wardCommands';
import { enrollHash, enrollStatusLine, parseProfileInfo } from '../ward/wardEnroll';
import { keyIsSwitch, pointerIsSwitch } from '../ward/wardInput';
import {
  LISTS_TEXT, listBannerText, listLoadErrorMessage, listLoadOutcome, resultsFromOtherListVersion,
} from '../ward/wardLists';
import {
  PHASE_LABELS, WARD_TEXT, createWardState, enrollOpenBlockReason, listSwitchBlockReason, listsOpenBlockReason, scanDeadline,
  scanItemKey, scanToken, stepWard, wardView,
} from '../ward/wardMachine';
import { practiceResults } from '../ward/wardPractice';
import {
  SETTINGS_STORAGE_KEY, enrollProfileFor, isNightTime, isPracticeMode, loadSettings, readStoredSettings, safeLocalStorage,
  saveSettings, sharedSettingsPatch, speechAllowed,
} from '../ward/wardSettings';
import { createBeeper } from '../ward/wardSound';
import { NO_VOICE_TEXT, pickKoreanLocalVoice, speakText } from '../ward/wardSpeech';
import { TONE_STYLES, WARD_COLORS, WARD_KEYFRAMES } from '../ward/wardTheme';
import { analyzeWardRecording } from '../ward/wardUpload';
import '../ward/patient-ui.css';

const INDEX = buildCommandIndex(COMMANDS);
const CAMERA = { width: 1280, height: 720, frameRate: 30 };
const TICK_MS = 200;
const SAFETY_TEXT = '숨이 막히면 호출벨을 먼저 누르세요';
const PROTOTYPE_TEXT = '시연용 연구 시제품 · 실제 환자 미검증';

/**
 * 창마다 다른 카드 ID 앞부분 (간호사 화면 기록과 겹치지 않게).
 * @returns {string}
 */
function newSessionKey() {
  try {
    if (globalThis.crypto?.randomUUID) return `w${globalThis.crypto.randomUUID().slice(0, 8)}`;
  } catch {
    /* 아래 대체 */
  }
  return `w${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

/**
 * 카메라 상태 문구.
 * @param {{status: string, error: string|null}} rec
 * @returns {string}
 */
function cameraText(rec) {
  if (!globalThis.isSecureContext) return 'localhost 주소로 열어야 카메라를 쓸 수 있습니다';
  if (rec.status === 'opening') return '카메라를 여는 중...';
  if (rec.status === 'error') return rec.error || '카메라를 쓸 수 없습니다';
  return '카메라가 꺼져 있습니다';
}

/**
 * 환자 화면.
 * @returns {JSX.Element}
 */
export default function WardPage() {
  const storage = useMemo(() => safeLocalStorage(), []);
  const practice = useMemo(() => isPracticeMode(globalThis.location), []);
  const machineRef = useRef(null);
  if (!machineRef.current) {
    const saved = loadSettings(storage);
    // 내 목록을 고른 경우: 불러올 때까지 명령 없는 색인으로 시작하고 입 읽기 녹화를 막는다(기본 목록으로 시작하지 않음).
    const startList = practice ? '' : saved.commandList;
    machineRef.current = createWardState({
      index: startList ? emptyListIndex(startList) : INDEX,
      baseIndex: INDEX,
      lipBlock: startList ? { reason: 'loading', message: '' } : null,
      settings: saved,
      sessionKey: newSessionKey(),
    });
  }
  const [machine, setMachine] = useState(machineRef.current);
  const [now, setNow] = useState(() => Date.now());
  const [showSettings, setShowSettings] = useState(false);
  const showSettingsRef = useRef(false);
  showSettingsRef.current = showSettings;
  const [soundReady, setSoundReady] = useState(false);
  const [voice, setVoice] = useState(null);
  const [practiceCommand, setPracticeCommand] = useState('W05');
  const [channelOk, setChannelOk] = useState(true);
  const [enrollState, setEnrollState] = useState({ key: '', info: null, loadError: false });
  // 내 목록 상태 (기본 목록이면 null): {listId, state:'loading'|'ready'|'error', title, count, message}
  const [listStatus, setListStatus] = useState(null);
  const [listReload, setListReload] = useState(0);

  const beeperRef = useRef(null);
  if (!beeperRef.current) beeperRef.current = createBeeper();
  const channelRef = useRef(null);
  const mountedRef = useRef(true);
  const scanListRef = useRef(null); // 스위치 모드에서 강조된 항목을 화면 안으로 스크롤할 때 쓴다
  const activeRunRef = useRef(0);
  // 녹화기가 지금 찍고 있는(결과를 아직 넘기지 않은) 녹화의 실행 번호. 녹화를 실제로 시작할 때 정하고 결과에서 지운다.
  // 결과가 올 때의 activeRunRef가 아니라 이 값으로 판단해야, 멈춘 옛 녹화가 새 실행의 결과로 올라가지 않는다.
  const recordingRunRef = useRef(null);
  const effectRef = useRef(() => {});

  /** 상태도에 이벤트 하나를 넣고 나온 할 일을 실행한다. */
  const dispatch = useCallback((event) => {
    const { state, effects } = stepWard(machineRef.current, { at: Date.now(), ...event });
    if (state !== machineRef.current) {
      machineRef.current = state;
      setMachine(state);
    }
    for (const effect of effects) effectRef.current(effect);
  }, []);

  /** 녹화가 끝나면(자동 정지) 올리고 폴링해서 결과를 상태도에 넣는다. */
  const handleRecorded = useCallback(async ({ blob, mime, meta }) => {
    const run = recordingRunRef.current;
    recordingRunRef.current = null;
    if (run === null) return;
    const stale = () => !mountedRef.current || machineRef.current.run !== run;
    if (stale()) return; // 연습 결과 주입 등으로 무효가 된 녹화는 보내지 않는다
    dispatch({ type: 'RECORDING_DONE', run });
    // 내 목록을 쓰면 command_list(목록 ID)를 보낸다. 기본 목록이면 보내지 않아 요청이 예전과 같다.
    // 등록 이름이 있으면 enroll_profile을 함께 보낸다(목록의 명령을 다 찍지 않았으면 서버가 거리를 계산하지 않아 오늘과 같은 판정이다).
    const usedList = machineRef.current.index.customList;
    const commandList = usedList?.id || '';
    const outcome = await analyzeWardRecording({
      blob, mime, durationMs: meta?.durationMs, enrollProfile: enrollProfileFor(machineRef.current.settings, commandList), commandList,
      isCancelled: stale,
    });
    if (outcome.kind === 'invalid') dispatch({ type: 'RECORD_FAILED', run, message: outcome.message });
    else if (stale()) return;
    else if (outcome.kind === 'done') {
      dispatch({ type: 'RESULT', run, results: outcome.results });
      // 서버의 목록이 그사이 바뀌었으면(지문이 다름) 결과는 쓰지 않았으니 목록을 다시 불러온다.
      if (resultsFromOtherListVersion(outcome.results, usedList)) setListReload((n) => n + 1);
    }
    else if (outcome.kind === 'error') dispatch({ type: 'ANALYSIS_ERROR', run, message: outcome.message });
    else if (outcome.kind === 'timeout') dispatch({ type: 'ANALYSIS_ERROR', run, reason: 'timeout' });
  }, [dispatch]);

  const rec = useRecorder({ maxSeconds: machine.settings.recordSeconds, video: CAMERA, onResult: handleRecorded });
  const recRef = useRef(rec);
  recRef.current = rec;
  const { open, videoRef, status: camStatus } = rec;

  /** 소리(시작음)·음성 읽기는 사용자 조작 안에서 한 번 열어야 한다 (브라우저 자동 재생 정책). */
  const unlockAudio = useCallback(() => {
    beeperRef.current.enable().then((ok) => { if (mountedRef.current) setSoundReady(ok); }).catch(() => {});
  }, []);

  /** 카드 다시 보내기 + 병실 상태 (간호사 창이 새로 열렸을 때). */
  const resendAll = useCallback(() => {
    const ch = channelRef.current;
    const s = machineRef.current;
    if (!ch) return;
    ch.post('ROOM_STATE', { room: s.settings.room, state: s.room, phase: s.phase, windowId: s.sessionKey });
    for (const id of s.cardOrder) {
      const card = s.cards[id];
      if (card && !card.acked) ch.post('CARD_UPSERT', { card });
    }
  }, []);

  // 상태도 할 일 실행기 (렌더마다 최신 값으로 바꾼다).
  effectRef.current = (effect) => {
    try {
      switch (effect.type) {
        case 'START_RECORDING': {
          activeRunRef.current = effect.run;
          const r = recRef.current;
          // 앞 녹화가 아직 결과를 넘기지 않았으면(멈추는 중) 새로 시작하지 않는다. 녹화기 하나가 두 녹화를 섞지 않게 한다.
          if (r.status === 'ready' && recordingRunRef.current === null) {
            recordingRunRef.current = effect.run;
            beeperRef.current.play('start');
            r.start();
          } else {
            setTimeout(() => dispatch({ type: 'RECORD_FAILED', run: effect.run, message: WARD_TEXT.cameraNotReady }), 0);
          }
          break;
        }
        case 'SEND_CARD':
        case 'UPDATE_CARD':
          channelRef.current?.post('CARD_UPSERT', { card: effect.card });
          break;
        case 'SPEAK':
          if (speechAllowed(machineRef.current.settings, new Date())) speakText(globalThis.speechSynthesis, voice, effect.text);
          break;
        case 'LOG':
          console.warn('[ward]', effect.message);
          break;
        default:
          break;
      }
    } catch (err) {
      console.warn('[ward] 할 일 실행 실패', err);
    }
  };

  // 카메라 열기 (localhost/HTTPS에서만). 닫기는 useRecorder가 화면 이탈 때 한다.
  useEffect(() => {
    mountedRef.current = true;
    if (globalThis.isSecureContext) open();
    return () => { mountedRef.current = false; };
  }, [open]);

  // 녹화 중 카메라가 끊기거나 녹화를 시작하지 못하면 그 녹화를 실패로 끝낸다 (이때 녹화기는 결과를 넘기지 않는다).
  useEffect(() => {
    if (camStatus !== 'error') return;
    recordingRunRef.current = null;
    if (machineRef.current.phase === 'recording') {
      dispatch({ type: 'RECORD_FAILED', run: activeRunRef.current, message: recRef.current.error || WARD_TEXT.cameraNotReady });
    }
  }, [camStatus, dispatch]);

  // 간호사 창과 연결.
  useEffect(() => {
    const ch = openWardChannel((msg) => {
      if (msg.type === 'CARD_ACK') dispatch({ type: 'NURSE_ACK', cardIds: msg.cardIds, room: msg.room });
      else if (msg.type === 'SWITCH_PRESS') dispatch({ type: 'PRESS', repeat: msg.repeat === true });
      else if (msg.type === 'HELLO') resendAll();
    }, { from: 'ward' });
    channelRef.current = ch;
    setChannelOk(ch.available);
    resendAll();
    // 간호사 창이 이 창의 병실 줄을 오래 남기지 않도록: 주기적으로 알리고, 닫히거나 새로 고침될 때 지우라고 알린다.
    const postRoom = () => {
      const s = machineRef.current;
      ch.post('ROOM_STATE', { room: s.settings.room, state: s.room, phase: s.phase, windowId: s.sessionKey });
    };
    const heartbeat = setInterval(postRoom, ROOM_HEARTBEAT_MS);
    const onPageHide = () => ch.post('ROOM_CLOSED', { windowId: machineRef.current.sessionKey });
    const onPageShow = (e) => { if (e.persisted) postRoom(); };
    window.addEventListener('pagehide', onPageHide);
    window.addEventListener('pageshow', onPageShow);
    return () => {
      clearInterval(heartbeat);
      window.removeEventListener('pagehide', onPageHide);
      window.removeEventListener('pageshow', onPageShow);
      // 해시 이동(등록 화면 등)으로 이 화면이 내려갈 때는 pagehide가 없으므로 여기서도 병실 줄을 지우라고 알린다.
      ch.post('ROOM_CLOSED', { windowId: machineRef.current.sessionKey });
      ch.close();
      channelRef.current = null;
    };
  }, [dispatch, resendAll]);

  // 병실 상태·화면 단계가 바뀌면 간호사 창에 알린다.
  useEffect(() => {
    channelRef.current?.post('ROOM_STATE', {
      room: machine.settings.room, state: machine.room, phase: machine.phase, windowId: machine.sessionKey,
    });
  }, [machine.room, machine.phase, machine.settings.room, machine.sessionKey]);

  // 시간 경과 (카운트다운·스캐닝·안내 만료).
  useEffect(() => {
    const id = setInterval(() => {
      const t = Date.now();
      setNow(t);
      dispatch({ type: 'TICK', at: t });
    }, TICK_MS);
    return () => clearInterval(id);
  }, [dispatch]);

  // 아무 키·포인터 = 스위치 (설정·연습 버튼과 글자 입력 칸은 제외). 설정 창이 열려 있는 동안 이 창의 키·포인터는
  // 스위치로 받지 않는다(보호자가 설정을 고르다 누른 키가 창 뒤에서 녹화를 시작하지 않게). 간호사 창이 넘긴 입력은 받는다.
  useEffect(() => {
    const onKey = (e) => {
      if (showSettingsRef.current || !keyIsSwitch(e)) return;
      e.preventDefault();
      unlockAudio();
      dispatch({ type: 'PRESS', repeat: e.repeat === true });
    };
    const onPointer = (e) => {
      if (showSettingsRef.current || machineRef.current.settings.inputMode === 'direct' || !pointerIsSwitch(e)) return;
      unlockAudio();
      dispatch({ type: 'PRESS' });
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onPointer);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('pointerdown', onPointer);
    };
  }, [dispatch, unlockAudio]);

  // 오프라인 한국어 음성 찾기 (목록은 늦게 채워질 수 있다).
  useEffect(() => {
    const synth = globalThis.speechSynthesis;
    if (!synth) return undefined;
    const update = () => {
      try { setVoice(pickKoreanLocalVoice(synth.getVoices())); } catch { setVoice(null); }
    };
    update();
    synth.addEventListener?.('voiceschanged', update);
    return () => synth.removeEventListener?.('voiceschanged', update);
  }, []);

  // 다른 탭이 바꾼 명령 목록·등록 이름을 아직 쓰지 못했는지 (녹화·확인·선택 중이었음). 대기로 돌아오면 쓴다.
  const pendingSyncRef = useRef(false);

  /** 저장소의 명령 목록·등록 이름(다른 탭이 바꿨을 수 있음)을 읽어 이 화면에 쓴다. 대기 중이 아니면 끝난 뒤에 쓴다. */
  const syncSharedSettings = useCallback(() => {
    const patch = sharedSettingsPatch(machineRef.current.settings, readStoredSettings(storage));
    if (!patch) {
      pendingSyncRef.current = false;
      return;
    }
    if (machineRef.current.phase !== 'idle') {
      pendingSyncRef.current = true;
      return;
    }
    pendingSyncRef.current = false;
    dispatch({ type: 'SETTINGS', settings: patch });
  }, [dispatch, storage]);

  /**
   * 설정 창의 변경을 쓰고 저장한다. 다른 탭이 저장소에 바꿔 둔 명령 목록·등록 이름은 지우지 않는다(대기 중이면 함께 쓰고,
   * 아니면 저장만 지키고 끝난 뒤에 쓴다).
   */
  const changeSettings = useCallback((patch) => {
    const shared = { ...(sharedSettingsPatch(machineRef.current.settings, readStoredSettings(storage)) || {}) };
    for (const key of Object.keys(patch || {})) delete shared[key];
    const idle = machineRef.current.phase === 'idle';
    dispatch({ type: 'SETTINGS', settings: idle ? { ...shared, ...patch } : patch });
    saveSettings(storage, { ...machineRef.current.settings, ...shared });
    if (!idle && Object.keys(shared).length) pendingSyncRef.current = true;
  }, [dispatch, storage]);

  // 다른 탭이 설정을 저장하면(storage 알림) 또는 이 창으로 돌아오면 명령 목록·등록 이름을 다시 읽는다.
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key === null || e.key === SETTINGS_STORAGE_KEY) syncSharedSettings();
    };
    const onFocus = () => syncSharedSettings();
    window.addEventListener('storage', onStorage);
    window.addEventListener('focus', onFocus);
    return () => {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('focus', onFocus);
    };
  }, [syncSharedSettings]);

  // 녹화·확인·선택이 끝나 대기로 돌아오면 미뤄 둔 설정을 쓴다.
  useEffect(() => {
    if (machine.phase === 'idle' && pendingSyncRef.current) syncSharedSettings();
  }, [machine.phase, syncSharedSettings]);

  const closeSettings = useCallback(() => setShowSettings(false), []);

  /**
   * 내 입 모양 등록 화면으로 (등록 이름·연습 모드를 주소에 넣는다). 이 화면을 떠나면 카메라는 useRecorder가 닫는다.
   * 진행 중인 녹화·확인·선택이나 확인하지 않은 긴급 알림이 있으면 열지 않는다(화면을 떠나면 그 상태가 사라진다).
   */
  const openEnroll = useCallback((profile) => {
    if (enrollOpenBlockReason(machineRef.current)) return;
    setShowSettings(false);
    try {
      globalThis.location.hash = enrollHash(profile, { practice });
    } catch {
      /* 주소를 바꾸지 못하면 그대로 둔다 */
    }
  }, [practice]);

  /** 목록 관리 화면(#lists)으로. 등록 화면과 같은 조건(대기 중·확인 안 된 긴급 알림 없음)에서만 연다. */
  const openLists = useCallback(() => {
    if (listsOpenBlockReason(machineRef.current)) return;
    setShowSettings(false);
    try {
      globalThis.location.hash = practice ? '#lists?practice=1' : '#lists';
    } catch {
      /* 주소를 바꾸지 못하면 그대로 둔다 */
    }
  }, [practice]);

  // 지금 쓰는 명령 목록: 연습 모드는 늘 기본 목록(''), 아니면 설정의 commandList.
  const activeList = practice ? '' : machine.settings.commandList;

  // 내 목록 불러오기. 처음 불러올 때는 입 읽기 녹화를 막고, 실패하면 막은 채로 오류를 보인다(기본 목록으로 바꾸지 않음).
  // 설정 창을 열고 닫을 때·창으로 돌아올 때·결과의 목록 지문이 다를 때 다시 불러와, 목록이 바뀌었으면 새 목록으로 바꾼다.
  useEffect(() => {
    const stopRecording = () => {
      const r = recRef.current;
      if (r.status === 'recording') r.stop(); // 결과는 실행 번호가 바뀌어 무시된다
    };
    if (!activeList) {
      setListStatus(null);
      if (machineRef.current.index !== INDEX) {
        stopRecording();
        dispatch({ type: 'SET_INDEX', index: INDEX, lipBlock: null });
      }
      return undefined;
    }
    let alive = true;
    const before = machineRef.current.index.customList;
    const loadedBefore = !!before && before.id === activeList && !!before.sha256 && !machineRef.current.lipBlock;
    const block = (message, title) => {
      stopRecording();
      dispatch({ type: 'SET_INDEX', index: emptyListIndex(activeList, title), lipBlock: { reason: 'error', message } });
      setListStatus({ listId: activeList, state: 'error', title, count: 0, message });
    };
    if (!loadedBefore) {
      if (!before || before.id !== activeList || !machineRef.current.lipBlock) {
        stopRecording();
        dispatch({ type: 'SET_INDEX', index: emptyListIndex(activeList), lipBlock: { reason: 'loading', message: '' } });
      }
      setListStatus((prev) => (prev?.listId === activeList && prev.state === 'error' ? prev
        : { listId: activeList, state: 'loading', title: activeList, count: 0, message: '' }));
    }
    getWardList(activeList)
      .then((data) => {
        if (!alive) return;
        const out = listLoadOutcome(data, activeList, COMMANDS);
        if (!out.ok) {
          block(out.message, out.title);
          return;
        }
        const now = machineRef.current.index.customList;
        const loaded = out.index.customList;
        // 같은 판(명령 그대로)이면 녹화를 멈추지 않는다. 이름만 바뀌었으면 띠·카드 이름만 바꾼다(상태도가 흐름을 그대로 둔다).
        const sameCommands = !!now && now.id === activeList && now.sha256 === loaded.sha256 && !machineRef.current.lipBlock;
        if (!sameCommands) {
          stopRecording();
          dispatch({ type: 'SET_INDEX', index: out.index, lipBlock: null });
        } else if (now.title !== loaded.title) {
          dispatch({ type: 'SET_INDEX', index: out.index, lipBlock: null });
        }
        setListStatus({ listId: activeList, state: 'ready', title: out.title, count: out.index.list.length, message: '' });
      })
      .catch((err) => {
        if (!alive) return;
        // 이미 불러온 목록을 다시 읽다가 연결만 끊긴 경우는 그대로 둔다(서버가 없으면 분석도 오류로 끝난다).
        if (loadedBefore && !err?.response) return;
        block(listLoadErrorMessage(err), machineRef.current.index.customList?.title || activeList);
      });
    return () => { alive = false; };
  }, [activeList, listReload, showSettings, dispatch]);

  // 다른 창(목록 관리 등)에서 돌아오면 내 목록을 다시 확인한다.
  useEffect(() => {
    if (!activeList) return undefined;
    const onFocus = () => setListReload((n) => n + 1);
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [activeList]);

  // 설정의 등록 이름이 바뀌거나 설정 창을 닫으면 등록 상태(n/n·다시 등록 필요)를 서버에서 다시 읽는다. 등록은 목록마다 따로다.
  const enrollProfile = enrollProfileFor(machine.settings, activeList);
  const profileApi = useMemo(() => wardProfileApi(activeList), [activeList]);
  const enrollKey = `${activeList}|${enrollProfile}`;
  const listReady = !machine.lipBlock;
  const machineIndex = machine.index;
  useEffect(() => {
    if (!enrollProfile || !listReady) {
      setEnrollState({ key: '', info: null, loadError: false });
      return undefined;
    }
    let alive = true;
    setEnrollState((prev) => (prev.key === enrollKey ? prev : { key: enrollKey, info: null, loadError: false }));
    profileApi.getWardProfile(enrollProfile)
      .then((data) => { if (alive) setEnrollState({ key: enrollKey, info: parseProfileInfo(data, machineIndex, enrollProfile), loadError: false }); })
      .catch(() => { if (alive) setEnrollState({ key: enrollKey, info: null, loadError: true }); });
    return () => { alive = false; };
  }, [enrollProfile, enrollKey, profileApi, listReady, machineIndex, showSettings]);

  const inject = useCallback((scenarioId) => {
    const r = recRef.current;
    if (r.status === 'recording') r.stop(); // 결과는 실행 번호가 바뀌어 무시된다
    dispatch({ type: 'SIMULATE', results: practiceResults(scenarioId, INDEX, { commandId: practiceCommand }) });
  }, [dispatch, practiceCommand]);

  const view = wardView(machine, now);
  const tone = TONE_STYLES[view.tone] || TONE_STYLES.neutral;
  const emergencyTone = view.tone === 'emergency' || view.tone === 'possible';
  const settings = machine.settings;
  const nightMuted = settings.tts && settings.nightMute && isNightTime(new Date(now));
  const enrollLine = listReady
    ? enrollStatusLine(enrollProfile, enrollState.key === enrollKey ? enrollState.info : null,
      { loadError: enrollState.key === enrollKey && enrollState.loadError })
    : null;
  const banner = listReady ? listBannerText(machine.index.customList) : null;
  const pickedList = machine.settings.commandList;

  const direct = settings.inputMode === 'direct';
  const busy = machine.phase === 'recording' || machine.phase === 'analyzing';
  // 남은 시간 막대: 녹화·보내기 전 확인, 그리고 직접 선택 모드의 스캐닝(강조가 없으니 목록이 닫히는 시각을 보여 준다).
  const scanTimed = direct && machine.phase === 'scan' ? scanDeadline(machine.scan) : null;
  const timed = machine.phase === 'recording' ? machine.recording : machine.phase === 'confirm' ? machine.confirm : scanTimed;
  const secondsLeft = timed ? Math.max(0, Math.min(Math.ceil((timed.endsAt - timed.startedAt) / 1000), Math.ceil((timed.endsAt - now) / 1000))) : null;
  const timeFraction = timed ? Math.max(0, Math.min(1, (timed.endsAt - now) / (timed.endsAt - timed.startedAt))) : 0;
  const analysisSeconds = machine.analysis ? Math.max(0, Math.floor((now - machine.analysis.startedAt) / 1000)) : 0;
  const selectionToken = scanToken(machine);
  // 첫 화면 문구는 입 읽기를 쓸 수 있을 때만 바꾼다 (내 목록을 불러오는 중·실패 안내를 덮지 않는다).
  const welcome = machine.phase === 'idle' && machine.screen?.kind === 'welcome' && !machine.lipBlock;
  const displayedHeadline = welcome ? '어떤 도움이 필요하세요?' : view.headline;
  const displayedSub = machine.phase === 'scan'
    ? (direct ? '원하는 항목을 직접 누르세요' : '파란 항목이 원하는 요청일 때 스위치를 누르세요')
    : welcome ? `${settings.recordSeconds}초 동안 입 모양을 촬영합니다. 준비되면 시작해 주세요.${settings.showPhrases ? ' 말할 문장은 아래 목록에 있어요.' : ''}`
      : machine.phase === 'idle' && machine.screen?.kind === 'sent' && !machine.cards[machine.screen.cardId]?.acked ? '요청 전송 · 간호사 확인 대기' : view.sub;
  const press = (intent) => { unlockAudio(); dispatch({ type: 'PRESS', intent }); };
  const scanHighlight = machine.phase === 'scan' && !direct ? `${scanToken(machine)}:${machine.scan?.index}` : null;

  // 스위치 모드에서 강조가 화면 아래로 내려가면 (작은 화면·긴급 띠가 있을 때) 그 항목이 보이게 스크롤한다.
  useEffect(() => {
    if (!scanHighlight) return;
    try {
      const li = scanListRef.current?.querySelector('li[aria-current="true"]');
      li?.scrollIntoView?.({ block: 'nearest' });
    } catch {
      /* 스크롤하지 못해도 강조는 바뀌어 있다 */
    }
  }, [scanHighlight]);

  return (
    <div className="patient-page select-none">
      <style>{WARD_KEYFRAMES}</style>
      {/* 녹화 중 테두리 깜빡임은 녹화 파랑(긴급 빨강은 긴급에만 쓴다) */}
      {view.flash && (
        <div
          aria-hidden="true"
          className="pointer-events-none fixed inset-0 z-40"
          style={{ border: `6px solid ${WARD_COLORS.infoBg}`, animation: 'ward-flash 0.8s infinite' }}
        />
      )}

      {/* 늘 보이는 안전 안내 */}
      <header className="patient-safety">
        <p className="flex items-center gap-2 font-semibold">
          <AlertIcon title="안내" /> {SAFETY_TEXT}
        </p>
        <span className="patient-prototype">{PROTOTYPE_TEXT}</span>
        <div data-no-switch className="ml-auto flex items-center gap-2" onClick={blurAfterPointerClick}>
          <span className="patient-room-badge">{settings.room}호</span>
          {!soundReady && (
            <button
              type="button"
              onClick={unlockAudio}
              className="patient-small-button"
            >
              시작음 켜기
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="patient-small-button"
          >
            설정
          </button>
        </div>
      </header>

      {view.alarm && (
        <div role="alert" className="patient-alert" style={{ background: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyInk }}>
          <AlertIcon /> [긴급] {view.alarm.kind === 'emergency_possible' ? WARD_TEXT.emergencyPossibleCard : view.alarm.text} — 간호사실에 알렸어요 · 간호사 확인을 기다리는 중
        </div>
      )}
      {view.acked && (
        <div role="status" className="patient-alert" style={{ background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }}>
          <CheckIcon /> {WARD_TEXT.acked}
        </div>
      )}
      {!channelOk && (
        <p role="alert" className="patient-band patient-band-warn">
          <AlertIcon /> 이 브라우저는 간호사 화면과 연결할 수 없습니다 (BroadcastChannel 없음). 다른 최신 브라우저로 열어 주세요.
        </p>
      )}
      {banner && (
        <p data-list-banner={machine.index.customList?.id} className="patient-band patient-band-list">
          {banner}
        </p>
      )}
      {activeList && listStatus?.state === 'error' && (
        <p role="alert" data-list-error className="patient-band patient-band-warn">
          <AlertIcon /> {LISTS_TEXT.cannotUse(listStatus.title, listStatus.message)} · 입 읽기 녹화를 쓸 수 없어요 (스위치 3번 = 긴급 호출은 됩니다)
        </p>
      )}
      {practice && pickedList && (
        <p data-practice-list-note className="patient-band patient-band-info">
          {LISTS_TEXT.practiceDefaultOnly} (설정에서 고른 내 목록은 연습 모드 밖 병실 화면에서 써요)
        </p>
      )}

      <main className="patient-main">
        {/* 카메라 미리보기 */}
        <section aria-label="카메라" className="patient-camera-panel">
          <div className="patient-section-title"><span>카메라 미리보기</span><span className={`patient-dot ${camStatus === 'ready' || camStatus === 'recording' ? 'is-ready' : ''}`} /> </div>
          <div className="patient-preview" style={{ background: WARD_COLORS.ink }}>
            <video ref={videoRef} className="h-full w-full object-cover" style={{ transform: 'scaleX(-1)' }} playsInline muted />
            {camStatus !== 'ready' && camStatus !== 'recording' && (
              <div className="patient-preview-msg" style={{ color: WARD_COLORS.paper }}>
                {cameraText(rec)}
              </div>
            )}
          </div>
          {camStatus === 'error' && globalThis.isSecureContext && (
            <div data-no-switch onClick={blurAfterPointerClick}>
              <button type="button" onClick={() => open()} className="patient-small-button">
                카메라 다시 열기
              </button>
            </div>
          )}
          <p className="patient-camera-hint">얼굴과 입이 화면 가운데에 보이도록 맞춰 주세요.</p>
          <p className="patient-camera-detail" style={{ color: WARD_COLORS.inkSoft }}>
            녹화 {settings.recordSeconds}초 · 확인 {settings.confirmSeconds}초
            {settings.tts ? ` · 음성 읽기 켬${voice ? '' : ` (${NO_VOICE_TEXT})`}${nightMuted ? ' · 밤 무음 중' : ''}` : ' · 음성 읽기 끔'}
          </p>
          {enrollLine && (
            <p data-enroll-status={enrollLine.tone} className={`patient-enroll-line is-${enrollLine.tone}`}>
              {enrollLine.text}
            </p>
          )}
        </section>

        {/* 결과와 조작. 같은 video DOM은 화면 폭이 변해도 유지한다. */}
        <section
          aria-live="assertive"
          className={`patient-result tone-${view.tone} ${machine.phase === 'scan' ? 'is-scanning' : ''}`}
          style={emergencyTone || machine.phase === 'recording' || view.tone === 'warning' ? { ...tone, borderColor: emergencyTone ? WARD_COLORS.emergencyBg : undefined } : undefined}
        >
          <div className="patient-result-top">
            <span className="patient-phase" data-phase={machine.phase}>{PHASE_LABELS[machine.phase]}</span>
            <div className="patient-input-choice" role="group" aria-label="입력 방식" data-no-switch onClick={blurAfterPointerClick}>
              <button type="button" disabled={busy} aria-pressed={!direct} onClick={() => changeSettings({ inputMode: 'switch' })}>스위치</button>
              <button type="button" disabled={busy} aria-pressed={direct} onClick={() => changeSettings({ inputMode: 'direct' })}>직접 선택</button>
            </div>
          </div>
          {view.label && (
            <p className="patient-result-label">
              <AlertIcon /> {view.label}
            </p>
          )}
          {view.reason && <p className="patient-reason">{view.reason}</p>}
          <p className="patient-headline">
            {displayedHeadline}
          </p>
          {view.candidates && view.candidates.length > 0 && (
            <p className="patient-subline">추정: {view.candidates.map((c) => c.text).join(' / ')}</p>
          )}
          {displayedSub && <p className="patient-subline">{displayedSub}</p>}
          {timed && <div className="patient-timing" aria-live="off">
            <div><strong>{secondsLeft}초</strong><span>{machine.phase === 'recording' ? '뒤 녹화 종료' : machine.phase === 'scan' ? '안에 골라 주세요' : '뒤 자동 전송'}</span></div>
            <div className="patient-time-track" role="progressbar" aria-label={machine.phase === 'recording' ? '남은 녹화 시간' : machine.phase === 'scan' ? '고를 수 있는 남은 시간' : '전송까지 남은 시간'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(timeFraction * 100)}><span style={{ width: `${timeFraction * 100}%` }} /></div>
          </div>}
          {machine.phase === 'analyzing' && <div className="patient-analysis" aria-live="off"><span className="patient-spinner" /><span>{analysisSeconds}초 경과 · 결과를 기다리고 있어요</span></div>}
          {view.items && (
            <ol ref={scanListRef} className="patient-scan-list" data-no-switch={direct ? '' : undefined} onClick={direct ? blurAfterPointerClick : undefined}>
              {view.items.map((it, i) => (
                <li
                  key={scanItemKey(machine.scan.items[i])}
                  aria-current={!direct && it.highlighted ? 'true' : undefined}
                  className={!direct && it.highlighted ? 'is-highlighted' : ''}
                >
                  {direct ? <button type="button" onClick={() => { unlockAudio(); dispatch({ type: 'SELECT_ITEM', scanToken: selectionToken, itemKey: scanItemKey(machine.scan.items[i]) }); }}>
                    <span className="patient-choice-number" aria-hidden="true">{i + 1}</span><span>{it.emergency && machine.scan.items[i]?.key !== 'category' && <span className="patient-emergency-tag">긴급 </span>}{it.label}</span><span aria-hidden="true" className="patient-choice-arrow">→</span>
                  </button> : <div className="patient-switch-option"><span className="patient-choice-number" aria-hidden="true">{it.highlighted ? '▶' : i + 1}</span><span>{it.emergency && machine.scan.items[i]?.key !== 'category' && '[긴급] '}{it.label}</span></div>}
                </li>
              ))}
            </ol>
          )}
          <div className="patient-actions">
            {/* 화면 버튼은 단계(intent)를 붙여 보낸다: 그사이 단계가 바뀌었으면 상태도가 무시한다(예: 카운트다운이 막 끝난 뒤의 전송 취소). */}
            <span className="patient-action-buttons" data-no-switch onClick={blurAfterPointerClick}>
              {machine.phase === 'idle' && <button type="button" className="patient-primary" onClick={() => press('start')} disabled={camStatus !== 'ready' || !!machine.lipBlock}>{welcome ? '녹화 시작' : '새 요청 시작'} <span aria-hidden="true">→</span></button>}
              {machine.phase === 'confirm' && <button type="button" className="patient-secondary" onClick={() => press('cancel')}>전송 취소</button>}
              {machine.phase === 'scan' && !direct && <button type="button" className="patient-primary" onClick={() => press('select')}>강조된 항목 선택</button>}
            </span>
            <span className="patient-key-hint">
              {direct ? '항목은 화면에서 직접 누르세요 · 키(스위치)는 항목을 고르지 않아요' : '아무 키 또는 화면 누름 = 스위치'}
              <br /><strong>3초 안에 {direct ? '키(스위치)' : '스위치'} 3번 = 긴급 호출</strong>
            </span>
          </div>
          {machine.phase === 'idle' && camStatus !== 'ready' && !machine.lipBlock && <p className="patient-camera-hint">카메라가 준비되면 녹화할 수 있어요.{practice ? ' 아래 시연 도구로 카메라 없이 화면 흐름을 확인할 수 있어요.' : ''}</p>}
        </section>
      </main>

      <div className="patient-secondary-content">
        {/* 말할 수 있는 입 모양 문장 (지금 쓰는 목록, 연습 모드는 기본 목록). 내 목록을 쓸 수 없는 동안에는 보이지 않는다. */}
        {!machine.lipBlock && (
          <PhrasePanel
            index={machine.index}
            open={settings.showPhrases}
            onToggle={() => changeSettings({ showPhrases: !settings.showPhrases })}
          />
        )}
        <RoomPanel room={machine.room} />
        {practice && (
          <PracticePanel
            commands={INDEX.list}
            commandId={practiceCommand}
            onCommandChange={setPracticeCommand}
            onInject={inject}
            onPress={() => { unlockAudio(); dispatch({ type: 'PRESS' }); }}
          />
        )}
      </div>

      {showSettings && (
        <WardSettingsPanel
          settings={settings}
          onChange={changeSettings}
          onClose={closeSettings}
          voice={voice}
          onOpenEnroll={openEnroll}
          enrollBlocked={enrollOpenBlockReason(machine)}
          activeList={activeList}
          practice={practice}
          index={machine.index}
          listStatus={activeList ? listStatus : null}
          listSwitchBlocked={listSwitchBlockReason(machine)}
          onOpenLists={openLists}
          listsBlocked={listsOpenBlockReason(machine)}
          inputModeBlocked={busy}
        />
      )}
    </div>
  );
}
