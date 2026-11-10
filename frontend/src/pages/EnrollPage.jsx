// EnrollPage.jsx — 내 입 모양 등록 화면 (#enroll, 병실 화면 설정에서 연다). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 등록 이름을 고르고 병실 명령 20개를 하나씩 소리 내어 한 번 말해 찍는다. 녹화는 병실 화면과 같은 5초 자동 정지이고,
// POST /api/record(phrase_set='ward', enroll_profile, enroll_command)로 보내 서버 등록 폴더에 저장한다(결과는 폴링).
// 상태도는 ward/wardEnroll.js(순수 함수)다. 이 화면에서 20개를 모두 찍고 '확인 필요'가 없으면 설정의 등록 이름을 이 이름으로
// 바꾼다. 이미 끝난 등록을 열었거나 '확인 필요'가 있으면 [병실 화면에서 이 등록 쓰기]를 눌러야 바꾼다.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { deleteWardProfile, getWardProfile, getWardProfiles } from '../api/client';
import { errorMessageFrom } from '../api/pollOutcome';
import useRecorder from '../hooks/useRecorder';
import { blurAfterPointerClick } from '../record/pointerFocus';
import COMMANDS from '../ward/commands.json';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import { buildCommandIndex, commandById } from '../ward/wardCommands';
import {
  ENROLL_RECORD_SECONDS, ENROLL_START_COOLDOWN_MS, ENROLL_TEXT, canStartEnrollRecording, commandRows, createEnrollState,
  enrollHash, enrollStatusLine, isPracticeMode, profileChoices, profileFromHash, shouldAutoUseProfile, stepEnroll,
  uncertainCommands, wardReturnHash,
} from '../ward/wardEnroll';
import { loadSettings, safeLocalStorage, saveSettings } from '../ward/wardSettings';
import { createBeeper } from '../ward/wardSound';
import { WARD_COLORS } from '../ward/wardTheme';
import { analyzeWardRecording } from '../ward/wardUpload';

const INDEX = buildCommandIndex(COMMANDS);
const CAMERA = { width: 1280, height: 720, frameRate: 30 };
const PROTOTYPE_TEXT = '시연용 연구 시제품 · 실제 환자 미검증';

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
 * 주소 해시를 바꾼다 (hashchange 없이).
 * @param {string} hash
 */
function replaceHash(hash) {
  try {
    const loc = globalThis.location;
    if (!loc || loc.hash === hash) return;
    globalThis.history?.replaceState(null, '', `${loc.pathname}${loc.search}${hash}`);
  } catch {
    /* 주소 갱신 실패는 화면 동작에 영향 없음 */
  }
}

/**
 * 병실 화면으로 돌아간다 (연습 모드로 왔으면 연습 모드 그대로).
 */
function goWard() {
  try {
    globalThis.location.hash = wardReturnHash(globalThis.location);
  } catch {
    /* 주소를 바꾸지 못하면 그대로 둔다 */
  }
}

/**
 * 큰 버튼 공통 모양.
 * @param {'primary'|'secondary'} kind
 * @returns {object}
 */
function buttonStyle(kind) {
  return kind === 'primary'
    ? { background: WARD_COLORS.infoBg, color: WARD_COLORS.infoInk, borderColor: WARD_COLORS.infoBg }
    : { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.ink };
}

/**
 * 내 입 모양 등록 화면.
 * @returns {JSX.Element}
 */
export default function EnrollPage() {
  const storage = useMemo(() => safeLocalStorage(), []);
  const practice = useMemo(() => isPracticeMode(globalThis.location), []);
  const stateRef = useRef(null);
  if (!stateRef.current) {
    stateRef.current = createEnrollState({ index: INDEX, profile: profileFromHash(globalThis.location?.hash) });
  }
  const [state, setState] = useState(stateRef.current);
  const [profileDraft, setProfileDraft] = useState(() => state.profile || loadSettings(storage).enrollProfile || '');
  const [choices, setChoices] = useState([]);
  const [coolingDown, setCoolingDown] = useState(false);
  const coolRef = useRef(false);
  coolRef.current = coolingDown;
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [inUseProfile, setInUseProfile] = useState(() => loadSettings(storage).enrollProfile || '');
  const mountedRef = useRef(true);
  const loadSeqRef = useRef(0);
  const coolTimerRef = useRef(null);
  const beeperRef = useRef(null);
  if (!beeperRef.current) beeperRef.current = createBeeper();

  /** 상태도에 이벤트 하나를 넣는다. */
  const dispatch = useCallback((event) => {
    const next = stepEnroll(stateRef.current, event);
    if (next !== stateRef.current) {
      stateRef.current = next;
      setState(next);
    }
  }, []);

  /** 등록 상태를 서버에서 읽는다 (마지막 요청의 응답만 쓴다). */
  const loadProfile = useCallback((profile) => {
    loadSeqRef.current += 1;
    const seq = loadSeqRef.current;
    const current = () => mountedRef.current && seq === loadSeqRef.current;
    getWardProfile(profile)
      .then((data) => { if (current()) dispatch({ type: 'PROFILE_LOADED', data }); })
      .catch((err) => { if (current()) dispatch({ type: 'PROFILE_FAILED', message: errorMessageFrom(err, '') }); });
  }, [dispatch]);

  /** 결과를 받은 직후 잠깐 스페이스바 시작을 막는다. */
  const startCooldown = useCallback(() => {
    setCoolingDown(true);
    clearTimeout(coolTimerRef.current);
    coolTimerRef.current = setTimeout(() => { if (mountedRef.current) setCoolingDown(false); }, ENROLL_START_COOLDOWN_MS);
  }, []);

  /** 녹화가 끝나면(5초 자동 정지) 올리고 저장 결과를 기다린다. */
  const handleRecorded = useCallback(async ({ blob, mime, meta }) => {
    const s = stateRef.current;
    if (!mountedRef.current || s.phase !== 'recording' || !s.recordingId) return;
    const profile = s.profile;
    dispatch({ type: 'RECORD_DONE' });
    const outcome = await analyzeWardRecording({
      blob, mime, durationMs: meta?.durationMs, enrollProfile: profile, enrollCommand: s.recordingId,
      isCancelled: () => !mountedRef.current,
    });
    if (!mountedRef.current) return;
    if (outcome.kind === 'invalid') dispatch({ type: 'RECORD_FAILED', message: outcome.message });
    else if (outcome.kind === 'done') {
      dispatch({ type: 'STORE_RESULT', results: outcome.results });
      loadProfile(profile);
    } else if (outcome.kind === 'error' || outcome.kind === 'timeout') {
      dispatch({ type: 'ANALYSIS_ERROR', message: outcome.kind === 'timeout' ? ENROLL_TEXT.timeout : outcome.message });
      loadProfile(profile); // 서버가 늦게라도 저장했을 수 있으니 목록을 서버 상태로 맞춘다
    }
    startCooldown();
  }, [dispatch, loadProfile, startCooldown]);

  const rec = useRecorder({ maxSeconds: ENROLL_RECORD_SECONDS, video: CAMERA, onResult: handleRecorded });
  const recRef = useRef(rec);
  recRef.current = rec;
  const { open, videoRef, status: camStatus } = rec;

  // 카메라 열기 (localhost/HTTPS에서만). 닫기는 useRecorder가 화면 이탈 때 한다.
  useEffect(() => {
    mountedRef.current = true;
    if (globalThis.isSecureContext) open();
    return () => {
      mountedRef.current = false;
      clearTimeout(coolTimerRef.current);
    };
  }, [open]);

  // 녹화 중 카메라가 끊기면 그 녹화를 실패로 끝낸다 (녹화기는 결과를 넘기지 않는다).
  useEffect(() => {
    if (camStatus === 'error' && stateRef.current.phase === 'recording') {
      dispatch({ type: 'RECORD_FAILED', message: recRef.current.error || ENROLL_TEXT.cameraFirst });
    }
  }, [camStatus, dispatch]);

  // 이름을 고르면 서버에서 등록 상태를 읽고, 주소에도 이름을 넣는다(새로 고침해도 이어서).
  useEffect(() => {
    if (state.phase === 'loading' && state.profile) loadProfile(state.profile);
  }, [state.phase, state.profile, loadProfile]);
  useEffect(() => {
    replaceHash(enrollHash(state.profile, { practice }));
  }, [state.profile, practice]);

  // 이름 고르기 화면: 이미 있는 등록 목록.
  useEffect(() => {
    if (state.phase !== 'choose') return undefined;
    let alive = true;
    getWardProfiles()
      .then((data) => { if (alive) setChoices(profileChoices(data, '')); })
      .catch(() => { if (alive) setChoices([]); });
    return () => { alive = false; };
  }, [state.phase]);

  // 주소의 등록 이름을 직접 바꾸면 그 이름으로 다시 연다.
  useEffect(() => {
    const onHash = () => {
      const id = profileFromHash(globalThis.location?.hash);
      if (id && id !== stateRef.current.profile) dispatch({ type: 'CHOOSE', profile: id });
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [dispatch]);

  /** 병실 화면 설정의 등록 이름을 이 등록으로 바꾼다. */
  const applyProfile = useCallback((profile) => {
    if (!profile) return;
    const current = loadSettings(storage);
    if (current.enrollProfile !== profile) saveSettings(storage, { ...current, enrollProfile: profile });
    setInUseProfile(profile);
  }, [storage]);

  // 이 화면에서 20개를 모두 찍었고(명령 목록·모델 그대로) '확인 필요'가 없으면 병실 화면이 이 등록을 쓰도록 설정을 바꾼다.
  // 이미 끝난 등록을 열어 보기만 했거나 '확인 필요'가 있으면 끝 화면의 버튼을 눌러야 바꾼다.
  useEffect(() => {
    if (state.phase !== 'done' || !state.profile) return;
    if (shouldAutoUseProfile(state)) applyProfile(state.profile);
  }, [state, applyProfile]);

  /** 녹화 시작 (스페이스바·버튼). */
  const startRecording = useCallback(() => {
    const r = recRef.current;
    if (!canStartEnrollRecording(stateRef.current, { cameraReady: r.status === 'ready', coolingDown: coolRef.current })) return;
    beeperRef.current.enable().then((ok) => { if (ok) beeperRef.current.play('start'); }).catch(() => {});
    dispatch({ type: 'RECORD_START' });
    r.start();
  }, [dispatch]);

  // 스페이스바 = 녹화 시작. 글자 입력 칸·선택 상자에서는 원래 동작을 둔다.
  useEffect(() => {
    const typing = (t) => Boolean(t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName)));
    const onKeyDown = (e) => {
      if (e.code !== 'Space' || typing(e.target)) return;
      e.preventDefault();
      if (!e.repeat) startRecording();
    };
    const onKeyUp = (e) => {
      if (e.code === 'Space' && !typing(e.target)) e.preventDefault();
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [startRecording]);

  const choose = useCallback((profile) => {
    setNotice(null);
    setProfileDraft(profile);
    dispatch({ type: 'CHOOSE', profile });
  }, [dispatch]);

  /** 명령 목록·모델이 바뀐 등록을 지우고 같은 이름으로 처음부터. */
  const deleteAndRestart = useCallback(async () => {
    const profile = stateRef.current.profile;
    if (!profile || deleteBusy) return;
    if (!globalThis.confirm?.(ENROLL_TEXT.deleteConfirm(profile))) return;
    setDeleteBusy(true);
    try {
      await deleteWardProfile(profile);
      setNotice(`등록 “${profile}”를 지웠어요. 처음부터 다시 찍어 주세요`);
      dispatch({ type: 'CHOOSE', profile }); // 같은 이름으로 다시 불러온다(빈 등록)
    } catch (err) {
      setNotice(errorMessageFrom(err, '등록을 지우지 못했어요'));
    } finally {
      if (mountedRef.current) setDeleteBusy(false);
    }
  }, [deleteBusy, dispatch]);

  const command = commandById(INDEX, state.currentId);
  const rows = commandRows(INDEX, state.info, state.currentId);
  const total = INDEX.list.length;
  const count = state.info?.count ?? 0;
  const line = state.info ? enrollStatusLine(state.profile, state.info) : null;
  const busy = state.phase === 'recording' || state.phase === 'analyzing';
  const cameraReady = camStatus === 'ready';
  const canStart = canStartEnrollRecording(state, { cameraReady, coolingDown });
  const remaining = Math.max(0, Math.ceil((ENROLL_RECORD_SECONDS * 1000 - (rec.elapsedMs || 0)) / 1000));
  const last = state.last && state.last.commandId === state.currentId ? state.last
    : (state.last && state.last.kind === 'stored' && state.last.matches ? state.last : null);
  const uncertain = uncertainCommands(INDEX, state.info);
  const inUse = inUseProfile === state.profile;
  const stale = state.info?.status === 'stale';

  return (
    <div className="min-h-screen" style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}>
      <header data-no-switch className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ background: WARD_COLORS.bannerBg, color: WARD_COLORS.paper }} onClick={blurAfterPointerClick}>
        <h1 className="text-2xl font-bold">{ENROLL_TEXT.title}</h1>
        <p className="text-base font-semibold" style={{ color: WARD_COLORS.bannerInk }}>{PROTOTYPE_TEXT}</p>
        <div className="ml-auto flex flex-wrap gap-2">
          {state.phase !== 'choose' && !busy && (
            <button
              type="button"
              onClick={() => { setNotice(null); dispatch({ type: 'RESET' }); }}
              className="rounded-lg border-2 px-3 py-1 font-semibold"
              style={{ borderColor: WARD_COLORS.paper, color: WARD_COLORS.paper }}
            >
              {ENROLL_TEXT.otherProfile}
            </button>
          )}
          <button
            type="button"
            onClick={goWard}
            disabled={busy}
            className="rounded-lg px-3 py-1 font-semibold"
            style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}
          >
            {ENROLL_TEXT.backToWard}
          </button>
        </div>
      </header>

      {notice && <p role="status" className="px-4 py-2 text-lg font-semibold" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>{notice}</p>}

      {state.phase === 'choose' && (
        <main className="mx-auto flex max-w-2xl flex-col gap-4 p-4">
          <p className="text-xl">{ENROLL_TEXT.intro}</p>
          <p className="text-base" style={{ color: WARD_COLORS.inkSoft }}>{ENROLL_TEXT.privacy}</p>
          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(e) => { e.preventDefault(); choose(profileDraft.trim()); }}
          >
            <label className="flex flex-col gap-1">
              <span className="text-lg font-semibold">{ENROLL_TEXT.profileLabel}</span>
              <input
                type="text"
                value={profileDraft}
                maxLength={32}
                autoComplete="off"
                spellCheck={false}
                onChange={(e) => setProfileDraft(e.target.value)}
                className="w-64 rounded-lg border-2 px-3 py-2 text-2xl"
                style={{ borderColor: WARD_COLORS.line }}
              />
            </label>
            <button type="submit" className="rounded-xl border-4 px-6 py-2 text-2xl font-bold" style={buttonStyle('primary')}>
              {ENROLL_TEXT.start}
            </button>
          </form>
          {state.error && <p role="alert" className="rounded-md px-3 py-2 text-lg font-semibold" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>{state.error}</p>}
          {choices.length > 0 && (
            <section aria-label={ENROLL_TEXT.existing}>
              <h2 className="mb-2 text-lg font-semibold">{ENROLL_TEXT.existing}</h2>
              <div className="flex flex-wrap gap-2">
                {choices.map((c) => (
                  <button
                    key={c.profile}
                    type="button"
                    onClick={() => choose(c.profile)}
                    className="rounded-lg border-2 px-4 py-2 text-lg font-semibold"
                    style={buttonStyle('secondary')}
                  >
                    {c.profile}{c.count !== null ? ` (${c.count}/${total})` : ''}{c.status === 'stale' ? ' · 다시 등록 필요' : ''}
                  </button>
                ))}
              </div>
            </section>
          )}
        </main>
      )}

      {state.phase === 'loading' && <p className="p-6 text-2xl font-semibold">{ENROLL_TEXT.loading}</p>}

      {!['choose', 'loading'].includes(state.phase) && (
        <main className="grid gap-4 p-4 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
          <section aria-label="카메라" className="flex flex-col gap-2">
            <div className="relative aspect-video w-full overflow-hidden rounded-xl" style={{ background: WARD_COLORS.ink }}>
              <video ref={videoRef} className="h-full w-full object-cover" style={{ transform: 'scaleX(-1)' }} playsInline muted />
              {camStatus !== 'ready' && camStatus !== 'recording' && (
                <div className="absolute inset-0 flex items-center justify-center p-3 text-center text-base" style={{ color: WARD_COLORS.paper }}>
                  {cameraText(rec)}
                </div>
              )}
              {state.phase === 'recording' && (
                <div aria-hidden="true" className="pointer-events-none absolute inset-0" style={{ border: `10px solid ${WARD_COLORS.infoBg}` }} />
              )}
            </div>
            <p className="text-2xl font-bold">{state.profile} · {ENROLL_TEXT.progress(count, total)}</p>
            {line && line.tone === 'warn' && state.info?.status === 'stale' && (
              <div role="alert" className="rounded-lg px-3 py-2" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
                <p className="flex items-center gap-2 text-lg font-bold"><AlertIcon /> {ENROLL_TEXT.stale}</p>
                <button
                  type="button"
                  onClick={deleteAndRestart}
                  disabled={deleteBusy || busy}
                  className="mt-2 rounded-lg border-2 px-3 py-1 text-lg font-semibold"
                  style={{ borderColor: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyBg, background: WARD_COLORS.paper }}
                >
                  {ENROLL_TEXT.deleteAndRestart}
                </button>
              </div>
            )}
          </section>

          <section aria-live="polite" className="flex flex-col gap-3 rounded-2xl border-4 p-5" style={{ borderColor: state.phase === 'recording' ? WARD_COLORS.infoBg : WARD_COLORS.line }}>
            {state.phase === 'done' ? (
              <>
                <p className="flex items-center gap-3 text-4xl font-black"><CheckIcon /> {ENROLL_TEXT.doneTitle} ({count}/{total})</p>
                {uncertain.length > 0 && (
                  <div role="alert" data-uncertain-count={uncertain.length} className="flex flex-col gap-2 rounded-lg px-3 py-2" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
                    <p className="flex items-center gap-2 text-2xl font-bold"><AlertIcon /> {ENROLL_TEXT.uncertainTitle(uncertain.length)}</p>
                    <p className="text-lg">{ENROLL_TEXT.uncertainSub}</p>
                    <ul className="flex flex-col gap-2" data-no-switch onClick={blurAfterPointerClick}>
                      {uncertain.map(({ command: c, heardAs }) => (
                        <li key={c.id} className="flex flex-wrap items-center gap-2 text-lg font-semibold">
                          <span>{ENROLL_TEXT.uncertainItem(c, heardAs)}</span>
                          <button
                            type="button"
                            data-retake-id={c.id}
                            onClick={() => dispatch({ type: 'SELECT', commandId: c.id })}
                            className="rounded-lg border-2 px-3 py-1 text-lg font-bold"
                            style={buttonStyle('secondary')}
                          >
                            {ENROLL_TEXT.retake}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <p className="text-2xl font-semibold" data-enroll-in-use={inUse ? 'true' : 'false'}>
                  {inUse ? ENROLL_TEXT.doneSub(state.profile) : ENROLL_TEXT.doneNotInUse(state.profile)}
                </p>
                <div data-no-switch className="flex flex-wrap gap-3" onClick={blurAfterPointerClick}>
                  {!inUse && (
                    <button
                      type="button"
                      onClick={() => applyProfile(state.profile)}
                      className="rounded-xl border-4 px-6 py-3 text-2xl font-bold"
                      style={uncertain.length > 0 ? buttonStyle('secondary') : buttonStyle('primary')}
                    >
                      {uncertain.length > 0 ? ENROLL_TEXT.useAnyway : ENROLL_TEXT.useThis}
                    </button>
                  )}
                  <button type="button" onClick={goWard} className="rounded-xl border-4 px-6 py-3 text-2xl font-bold" style={inUse ? buttonStyle('primary') : buttonStyle('secondary')}>
                    {ENROLL_TEXT.backToWard}
                  </button>
                </div>
                <p className="text-lg" style={{ color: WARD_COLORS.inkSoft }}>{ENROLL_TEXT.pickAgain}</p>
              </>
            ) : command && (
              <>
                <p className="text-xl font-semibold" style={{ color: WARD_COLORS.inkSoft }}>
                  {command.id}{command.emergency ? ' · [긴급]' : ''} · {ENROLL_TEXT.sayThis}
                </p>
                <p className="font-black" style={{ fontSize: 'max(7vh, 2.25rem)', lineHeight: 1.2, wordBreak: 'keep-all' }}>
                  “{command.text}”
                </p>
                <p className="text-2xl font-semibold">{ENROLL_TEXT.howTo}</p>
                <p className="text-lg" style={{ color: WARD_COLORS.inkSoft }}>{ENROLL_TEXT.shownAs(command.display)}</p>

                {last && (
                  <div
                    role={last.kind === 'stored' && last.matches ? 'status' : 'alert'}
                    data-store-result={last.kind}
                    className="flex items-start gap-2 rounded-lg px-3 py-2 text-xl font-semibold"
                    style={last.kind === 'stored' && last.matches
                      ? { background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }
                      : { background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}
                  >
                    {last.kind === 'stored' && last.matches ? <CheckIcon /> : <AlertIcon />}
                    <span>{last.message}</span>
                  </div>
                )}

                <div data-no-switch className="flex flex-wrap gap-3" onClick={blurAfterPointerClick}>
                  {state.phase === 'recording' && (
                    <p className="rounded-xl px-6 py-3 text-2xl font-bold" style={buttonStyle('primary')}>{ENROLL_TEXT.recording(remaining)}</p>
                  )}
                  {state.phase === 'analyzing' && <p className="py-3 text-2xl font-bold">{ENROLL_TEXT.analyzing}</p>}
                  {(state.phase === 'ready' || state.phase === 'review') && (
                    <button
                      type="button"
                      onClick={startRecording}
                      disabled={!canStart}
                      className="rounded-xl border-4 px-6 py-3 text-2xl font-bold disabled:opacity-60"
                      style={buttonStyle('primary')}
                    >
                      {state.phase === 'review' || (last && last.kind !== 'stored' && last.commandId === command.id)
                        ? `${ENROLL_TEXT.retake} (스페이스바)` : ENROLL_TEXT.record(ENROLL_RECORD_SECONDS)}
                    </button>
                  )}
                  {state.phase === 'review' && (
                    <button
                      type="button"
                      onClick={() => dispatch({ type: 'NEXT' })}
                      className="rounded-xl border-4 px-6 py-3 text-2xl font-bold"
                      style={buttonStyle('secondary')}
                    >
                      {ENROLL_TEXT.next}
                    </button>
                  )}
                </div>
                {stale && !busy && <p className="text-lg font-semibold">{ENROLL_TEXT.staleNoRecord}</p>}
                {!stale && !cameraReady && !busy && <p className="text-lg" style={{ color: WARD_COLORS.inkSoft }}>{ENROLL_TEXT.cameraFirst}</p>}
              </>
            )}
          </section>

          <section aria-label="명령 목록" className="lg:col-span-2" data-no-switch onClick={blurAfterPointerClick}>
            <ol className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
              {rows.map((row) => {
                const mark = row.warn ? '확인 필요' : (row.done ? '찍음' : '아직');
                return (
                  <li key={row.id}>
                    <button
                      type="button"
                      data-command-id={row.id}
                      data-done={row.done ? 'true' : 'false'}
                      aria-current={row.current ? 'true' : undefined}
                      onClick={() => dispatch({ type: 'SELECT', commandId: row.id })}
                      disabled={busy}
                      className="flex w-full items-center gap-2 rounded-xl border-4 px-3 py-2 text-left text-lg font-semibold"
                      style={{
                        background: row.current ? WARD_COLORS.infoBg : WARD_COLORS.paper,
                        color: row.current ? WARD_COLORS.infoInk : WARD_COLORS.ink,
                        borderColor: row.current ? WARD_COLORS.infoBg : (row.done ? WARD_COLORS.ink : WARD_COLORS.line),
                        borderStyle: row.done ? 'solid' : 'dashed',
                      }}
                    >
                      <span className="w-20 shrink-0 font-black">{row.done && !row.warn ? '✓ ' : ''}{mark}</span>
                      <span>{row.id} {row.display}{row.emergency ? ' [긴급]' : ''}</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </section>
        </main>
      )}
    </div>
  );
}
