// WardPage.jsx — 병실 무음 도우미 환자 화면 (#ward). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 스위치(아무 키·화면 누름) 한 번 → 시작음과 테두리 깜빡임으로 N초 녹화(카운트다운 없음) → POST /api/record(phrase_set='ward',
// 설정에 내 입 모양 등록 이름이 있으면 enroll_profile도) → 약 300ms 폴링 → 상태도(ward/wardMachine.js)가 결과를 처리한다.
// 고를지 말지는 서버 phrase.accepted와 등록 비교 결과(enroll.accepted·within)만 믿는다(판정표는 ward/wardResult.js).
// 간호사 화면(#nurse)과는 같은 주소의 BroadcastChannel로 카드·확인·스위치 입력·병실 상태를 주고받는다.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getWardProfile } from '../api/client';
import useRecorder from '../hooks/useRecorder';
import { blurAfterPointerClick } from '../record/pointerFocus';
import COMMANDS from '../ward/commands.json';
import PracticePanel from '../ward/PracticePanel';
import RoomPanel from '../ward/RoomPanel';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import WardSettingsPanel from '../ward/WardSettingsPanel';
import { ROOM_HEARTBEAT_MS, openWardChannel } from '../ward/wardChannel';
import { buildCommandIndex } from '../ward/wardCommands';
import { enrollHash, enrollStatusLine, parseProfileInfo } from '../ward/wardEnroll';
import { keyIsSwitch, pointerIsSwitch } from '../ward/wardInput';
import {
  PHASE_LABELS, WARD_TEXT, createWardState, enrollOpenBlockReason, stepWard, wardView,
} from '../ward/wardMachine';
import { practiceResults } from '../ward/wardPractice';
import {
  isNightTime, isPracticeMode, loadSettings, safeLocalStorage, saveSettings, speechAllowed,
} from '../ward/wardSettings';
import { createBeeper } from '../ward/wardSound';
import { NO_VOICE_TEXT, pickKoreanLocalVoice, speakText } from '../ward/wardSpeech';
import { TONE_STYLES, WARD_COLORS, WARD_KEYFRAMES } from '../ward/wardTheme';
import { analyzeWardRecording } from '../ward/wardUpload';

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
    machineRef.current = createWardState({ index: INDEX, settings: loadSettings(storage), sessionKey: newSessionKey() });
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
  const [enrollState, setEnrollState] = useState({ profile: '', info: null, loadError: false });

  const beeperRef = useRef(null);
  if (!beeperRef.current) beeperRef.current = createBeeper();
  const channelRef = useRef(null);
  const mountedRef = useRef(true);
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
    // 등록 이름이 있으면 enroll_profile을 함께 보낸다(등록이 20/20이 아니면 서버가 거리를 계산하지 않아 오늘과 같은 판정이다).
    const outcome = await analyzeWardRecording({
      blob, mime, durationMs: meta?.durationMs, enrollProfile: machineRef.current.settings.enrollProfile, isCancelled: stale,
    });
    if (outcome.kind === 'invalid') dispatch({ type: 'RECORD_FAILED', run, message: outcome.message });
    else if (stale()) return;
    else if (outcome.kind === 'done') dispatch({ type: 'RESULT', run, results: outcome.results });
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
      if (showSettingsRef.current || !pointerIsSwitch(e)) return;
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

  const changeSettings = useCallback((patch) => {
    dispatch({ type: 'SETTINGS', settings: patch });
    saveSettings(storage, machineRef.current.settings);
  }, [dispatch, storage]);

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

  // 설정의 등록 이름이 바뀌거나 설정 창을 닫으면 등록 상태(20/20·다시 등록 필요)를 서버에서 다시 읽는다.
  const enrollProfile = machine.settings.enrollProfile;
  useEffect(() => {
    if (!enrollProfile) {
      setEnrollState({ profile: '', info: null, loadError: false });
      return undefined;
    }
    let alive = true;
    setEnrollState((prev) => (prev.profile === enrollProfile ? prev : { profile: enrollProfile, info: null, loadError: false }));
    getWardProfile(enrollProfile)
      .then((data) => { if (alive) setEnrollState({ profile: enrollProfile, info: parseProfileInfo(data, INDEX, enrollProfile), loadError: false }); })
      .catch(() => { if (alive) setEnrollState({ profile: enrollProfile, info: null, loadError: true }); });
    return () => { alive = false; };
  }, [enrollProfile, showSettings]);

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
  const enrollLine = enrollStatusLine(enrollProfile, enrollState.profile === enrollProfile ? enrollState.info : null,
    { loadError: enrollState.profile === enrollProfile && enrollState.loadError });

  return (
    <div className="min-h-screen select-none" style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}>
      <style>{WARD_KEYFRAMES}</style>
      {/* 녹화 중 테두리 깜빡임은 녹화 파랑(긴급 빨강은 긴급에만 쓴다) */}
      {view.flash && (
        <div
          aria-hidden="true"
          className="pointer-events-none fixed inset-0 z-40"
          style={{ border: `18px solid ${WARD_COLORS.infoBg}`, animation: 'ward-flash 0.8s infinite' }}
        />
      )}

      {/* 늘 보이는 안전 안내 */}
      <header className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ background: WARD_COLORS.bannerBg, color: WARD_COLORS.bannerInk }}>
        <p className="flex items-center gap-2 text-2xl font-bold sm:text-3xl">
          <AlertIcon title="안내" /> {SAFETY_TEXT}
        </p>
        <p className="text-base font-semibold" style={{ color: WARD_COLORS.paper }}>{PROTOTYPE_TEXT}</p>
        <div data-no-switch className="ml-auto flex items-center gap-2" onClick={blurAfterPointerClick}>
          <span className="text-lg font-bold" style={{ color: WARD_COLORS.paper }}>{settings.room}호</span>
          {!soundReady && (
            <button
              type="button"
              onClick={unlockAudio}
              className="rounded-lg border-2 px-3 py-1 font-semibold"
              style={{ borderColor: WARD_COLORS.bannerInk, color: WARD_COLORS.bannerInk }}
            >
              시작음 켜기
            </button>
          )}
          <button
            type="button"
            onClick={() => setShowSettings(true)}
            className="rounded-lg px-3 py-1 font-semibold"
            style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}
          >
            설정
          </button>
        </div>
      </header>

      {view.alarm && (
        <div role="alert" className="flex items-center gap-3 px-4 py-3 text-2xl font-bold" style={{ background: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyInk }}>
          <AlertIcon /> [긴급] {view.alarm.kind === 'emergency_possible' ? WARD_TEXT.emergencyPossibleCard : view.alarm.text} — 간호사실에 알렸어요 · 간호사 확인을 기다리는 중
        </div>
      )}
      {view.acked && (
        <div role="status" className="flex items-center gap-3 px-4 py-3 text-3xl font-bold" style={{ background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }}>
          <CheckIcon /> {WARD_TEXT.acked}
        </div>
      )}
      {!channelOk && (
        <p role="alert" className="px-4 py-2 font-semibold" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
          이 브라우저는 간호사 화면과 연결할 수 없습니다 (BroadcastChannel 없음)
        </p>
      )}

      <main className="grid gap-4 p-4 lg:grid-cols-[minmax(0,340px)_minmax(0,1fr)]">
        {/* 카메라 미리보기 */}
        <section aria-label="카메라" className="flex flex-col gap-2">
          <div className="relative aspect-video w-full overflow-hidden rounded-xl" style={{ background: WARD_COLORS.ink }}>
            <video ref={videoRef} className="h-full w-full object-cover" style={{ transform: 'scaleX(-1)' }} playsInline muted />
            {camStatus !== 'ready' && camStatus !== 'recording' && (
              <div className="absolute inset-0 flex items-center justify-center p-3 text-center text-base" style={{ color: WARD_COLORS.paper }}>
                {cameraText(rec)}
              </div>
            )}
          </div>
          <p className="text-2xl font-bold" aria-live="polite" style={view.status ? { color: WARD_COLORS.infoBg } : undefined}>
            {view.status || `상태: ${PHASE_LABELS[view.phase] || ''}`}
          </p>
          <p className="text-base" style={{ color: WARD_COLORS.inkSoft }}>
            녹화 {settings.recordSeconds}초 · 확인 {settings.confirmSeconds}초
            {settings.tts ? ` · 음성 읽기 켬${voice ? '' : ` (${NO_VOICE_TEXT})`}${nightMuted ? ' · 밤 무음 중' : ''}` : ' · 음성 읽기 끔'}
          </p>
          {enrollLine && (
            <p
              data-enroll-status={enrollLine.tone}
              className="rounded-md px-2 py-1 text-base font-semibold"
              style={enrollLine.tone === 'warn'
                ? { background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }
                : { color: enrollLine.tone === 'ok' ? WARD_COLORS.ink : WARD_COLORS.inkSoft }}
            >
              {enrollLine.text}
            </p>
          )}
        </section>

        {/* 결과 (화면 높이 1/10 이상 글씨) */}
        <section
          aria-live="assertive"
          className="flex min-h-[40vh] flex-col justify-center gap-3 rounded-2xl border-4 p-5"
          style={{ ...tone, borderColor: emergencyTone ? WARD_COLORS.emergencyBg : WARD_COLORS.line }}
        >
          {view.label && (
            <p className="flex items-center gap-2 text-4xl font-black">
              <AlertIcon /> {view.label}
            </p>
          )}
          {view.reason && <p className="text-2xl font-semibold">{view.reason}</p>}
          <p className="font-black" style={{ fontSize: 'max(10vh, 2.25rem)', lineHeight: 1.15, wordBreak: 'keep-all' }}>
            {view.headline}
          </p>
          {view.candidates && view.candidates.length > 0 && (
            <p className="text-3xl font-semibold">추정: {view.candidates.map((c) => c.text).join(' / ')}</p>
          )}
          {view.sub && <p className="text-3xl font-semibold">{view.sub}</p>}
          {view.items && (
            <ol className="mt-2 grid gap-2">
              {view.items.map((it, i) => (
                <li
                  key={`${it.label}-${i}`}
                  aria-current={it.highlighted ? 'true' : undefined}
                  className="flex items-center gap-3 rounded-xl border-4 px-4 py-2 font-bold"
                  style={{
                    fontSize: 'max(5vh, 1.5rem)',
                    background: it.highlighted ? WARD_COLORS.infoBg : WARD_COLORS.paper,
                    color: it.highlighted ? WARD_COLORS.infoInk : WARD_COLORS.ink,
                    borderColor: it.highlighted ? WARD_COLORS.infoBg : WARD_COLORS.line,
                  }}
                >
                  <span aria-hidden="true" style={{ width: '1.2em' }}>{it.highlighted ? '▶' : ''}</span>
                  {it.emergency && <><AlertIcon /> [긴급]</>}
                  {it.label}
                </li>
              ))}
            </ol>
          )}
        </section>
      </main>

      <div className="flex flex-col gap-4 px-4 pb-6">
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
        />
      )}
    </div>
  );
}
