// EnrollPage.jsx — 내 입 모양 등록 화면 (#enroll, 병실 화면 설정에서 연다). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 등록 이름을 고르고 지금 쓰는 명령 목록(기본 목록 20개 또는 내 목록)의 명령을 하나씩 소리 내어 한 번 말해 찍는다. 녹화는 병실 화면과
// 같은 5초 자동 정지이고, POST /api/record(phrase_set='ward', enroll_profile, enroll_command, 내 목록이면 command_list)로 보내
// 서버 등록 폴더(목록마다 따로)에 저장한다(결과는 폴링).
// 상태도는 ward/wardEnroll.js(순수 함수)다. 이 화면에서 목록의 명령을 모두 찍고 '확인 필요'가 없으면 설정의 등록 이름(그 목록 몫)을
// 이 이름으로 바꾼다. 이미 끝난 등록을 열었거나 '확인 필요'가 있으면 [병실 화면에서 이 등록 쓰기]를 눌러야 바꾼다.
// 등록할 목록은 병실 화면이 쓰는 목록(설정의 commandList, 연습 모드는 기본 목록)이다. 내 목록을 불러오지 못하면 등록하지 않는다
// (기본 목록으로 바꾸지 않는다). 내 목록 등록 요청에는 불러온 목록 판(command_list_sha256)을 함께 보낸다: 다른 탭에서 목록을
// 고쳤으면 서버가 저장하지 않고 거절하고, 이 화면은 목록을 다시 불러온다. 창으로 돌아올 때도 목록이 바뀌었는지 다시 확인한다.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getWardList, wardProfileApi } from '../api/client';
import { errorMessageFrom } from '../api/pollOutcome';
import useRecorder from '../hooks/useRecorder';
import { blurAfterPointerClick } from '../record/pointerFocus';
import COMMANDS from '../ward/commands.json';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import { buildCommandIndex, commandById } from '../ward/wardCommands';
import {
  ENROLL_RECORD_SECONDS, ENROLL_START_COOLDOWN_MS, ENROLL_TEXT, STORE_LIST_CHANGED, canStartEnrollRecording, commandRows,
  createEnrollState, enrollHash, enrollStatusLine, isPracticeMode, profileChoices, profileFromHash, shouldAutoUseProfile,
  stepEnroll, storeOutcome, uncertainCommands, wardReturnHash,
} from '../ward/wardEnroll';
import { LISTS_TEXT, listLoadErrorMessage, listLoadOutcome } from '../ward/wardLists';
import { enrollProfileFor, enrollProfilePatch, loadSettings, safeLocalStorage, saveSettings } from '../ward/wardSettings';
import { createBeeper } from '../ward/wardSound';
import { analyzeWardRecording } from '../ward/wardUpload';
import '../ward/enroll-ui.css';

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
/**
 * 아래 목록에서 문장을 고른 뒤, 바뀐 '이 문장을 말하세요' 칸이 화면 밖이면 그 칸으로 올려 보여 준다.
 * @param {HTMLElement|null} el
 */
function revealTaskPanel(el) {
  try {
    globalThis.requestAnimationFrame?.(() => {
      if (!el) return;
      const top = el.getBoundingClientRect().top;
      if (top >= 0 && top <= globalThis.innerHeight * 0.5) return;
      const reduce = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
      el.scrollIntoView({ block: 'start', behavior: reduce ? 'auto' : 'smooth' });
    });
  } catch {
    /* 스크롤하지 못해도 고른 문장은 바뀌어 있다 */
  }
}

/**
 * 목록 관리 화면으로 (연습 모드로 왔으면 연습 모드 표시를 함께).
 * @param {boolean} practice
 */
function goLists(practice) {
  try {
    globalThis.location.hash = practice ? '#lists?practice=1' : '#lists';
  } catch {
    /* 주소를 바꾸지 못하면 그대로 둔다 */
  }
}

/**
 * 내 목록을 불러오는 중이거나 불러오지 못했을 때의 화면 (등록하지 않는다).
 * @param {{state: {state:string, message?:string, title?:string}, listId: string, practice: boolean}} props
 * @returns {JSX.Element}
 */
function ListGate({ state, listId, practice }) {
  const failed = state.state === 'error';
  return (
    <div className="enroll-ui">
      <header className="enroll-page-heading" data-no-switch>
        <div>
          <p className="enroll-eyebrow">나에게 맞춘 인식 준비</p>
          <h1>{ENROLL_TEXT.title}</h1>
          <p className="enroll-page-description">{PROTOTYPE_TEXT}</p>
        </div>
      </header>
      <main className="enroll-start-panel">
        {failed ? (
          <div role="alert" data-enroll-list-error className="enroll-feedback enroll-feedback-warning">
            <h3><AlertIcon /> 내 목록 ‘{state.title || listId}’: {state.message}</h3>
            <p>{ENROLL_TEXT.listLoadFailed}</p>
          </div>
        ) : (
          <div className="enroll-gate-loading" role="status"><span className="enroll-spinner" aria-hidden="true" />{LISTS_TEXT.loading}</div>
        )}
        <div className="enroll-task-actions enroll-gate-actions">
          <button type="button" onClick={goWard} className="enroll-btn enroll-btn-primary">{ENROLL_TEXT.backToWard}</button>
          <button type="button" onClick={() => goLists(practice)} className="enroll-btn enroll-btn-outline">{ENROLL_TEXT.manageLists}</button>
        </div>
      </main>
    </div>
  );
}

/**
 * 내 입 모양 등록 화면. 지금 쓰는 명령 목록을 정한 뒤(내 목록이면 서버에서 불러온 뒤) 등록 화면을 연다.
 * @returns {JSX.Element}
 */
export default function EnrollPage() {
  const storage = useMemo(() => safeLocalStorage(), []);
  const practice = useMemo(() => isPracticeMode(globalThis.location), []);
  // 등록할 목록: 병실 화면이 쓰는 목록(설정의 commandList). 연습 모드는 병실 화면처럼 기본 목록이다.
  const listId = useMemo(() => (practice ? '' : loadSettings(storage).commandList), [practice, storage]);
  const [loaded, setLoaded] = useState(() => (listId ? { state: 'loading' } : { state: 'ready', index: INDEX, notice: null }));
  const loadedRef = useRef(loaded);
  loadedRef.current = loaded;
  const [reload, setReload] = useState(0);

  // 내 목록 불러오기. 다시 불러온 목록이 같은 판(sha256)이면 등록 화면을 그대로 두고, 판이 바뀌었으면 새 목록으로 화면을 다시 연다.
  useEffect(() => {
    if (!listId) return undefined;
    let alive = true;
    getWardList(listId)
      .then((data) => {
        if (!alive) return;
        const out = listLoadOutcome(data, listId, COMMANDS);
        if (!out.ok) {
          setLoaded({ state: 'error', message: out.message, title: out.title });
          return;
        }
        const prev = loadedRef.current;
        const prevSha = prev.state === 'ready' ? prev.index.customList?.sha256 : null;
        if (prevSha && prevSha === out.index.customList.sha256) return;
        setLoaded({ state: 'ready', index: out.index, notice: prevSha ? ENROLL_TEXT.listReloaded : null });
      })
      .catch((err) => {
        if (!alive) return;
        // 이미 불러온 목록을 다시 확인하다 연결만 끊긴 경우는 그대로 둔다(저장 요청은 서버가 판을 다시 확인한다).
        if (loadedRef.current.state === 'ready' && !err?.response) return;
        setLoaded({ state: 'error', message: listLoadErrorMessage(err), title: listId });
      });
    return () => { alive = false; };
  }, [listId, reload]);

  // 다른 창(목록 관리 등)에서 돌아오면 목록이 바뀌었는지 다시 확인한다.
  useEffect(() => {
    if (!listId) return undefined;
    const onFocus = () => setReload((n) => n + 1);
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [listId]);

  const reloadList = useCallback(() => setReload((n) => n + 1), []);

  if (loaded.state !== 'ready') return <ListGate state={loaded} listId={listId} practice={practice} />;
  return (
    <EnrollScreen
      key={loaded.index.customList?.sha256 || 'default'}
      index={loaded.index}
      listId={listId}
      storage={storage}
      practice={practice}
      initialNotice={loaded.notice || null}
      onListChanged={reloadList}
    />
  );
}

/**
 * 등록 화면 본체 (목록 하나의 한 판에 묶임. 목록 판이 바뀌면 부모가 새로 연다).
 * @param {{index: object, listId: string, storage: Storage|null, practice: boolean, initialNotice?: string|null,
 *   onListChanged?: () => void}} props
 *   index는 등록할 목록의 명령 색인, listId는 ''(기본 목록) 또는 내 목록 ID. onListChanged는 서버가 '목록이 바뀜'을 알렸을 때
 *   목록을 다시 불러오게 한다.
 * @returns {JSX.Element}
 */
function EnrollScreen({ index, listId, storage, practice, initialNotice = null, onListChanged = () => {} }) {
  const api = useMemo(() => wardProfileApi(listId), [listId]);
  const listName = index.customList ? `내 목록 ‘${index.customList.title}’` : LISTS_TEXT.defaultName;
  const stateRef = useRef(null);
  if (!stateRef.current) {
    stateRef.current = createEnrollState({ index, profile: profileFromHash(globalThis.location?.hash) });
  }
  const [state, setState] = useState(stateRef.current);
  const [profileDraft, setProfileDraft] = useState(() => state.profile || enrollProfileFor(loadSettings(storage), listId) || '');
  const [choices, setChoices] = useState([]);
  const [coolingDown, setCoolingDown] = useState(false);
  const coolRef = useRef(false);
  coolRef.current = coolingDown;
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [notice, setNotice] = useState(initialNotice);
  const [inUseProfile, setInUseProfile] = useState(() => enrollProfileFor(loadSettings(storage), listId) || '');
  const [listFilter, setListFilter] = useState('missing');
  const [analysisElapsed, setAnalysisElapsed] = useState(0);
  const analysisStartedRef = useRef(null);
  const taskPanelRef = useRef(null);
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
    api.getWardProfile(profile)
      .then((data) => { if (current()) dispatch({ type: 'PROFILE_LOADED', data }); })
      .catch((err) => { if (current()) dispatch({ type: 'PROFILE_FAILED', message: errorMessageFrom(err, '') }); });
  }, [api, dispatch]);

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
    analysisStartedRef.current = Date.now();
    dispatch({ type: 'RECORD_DONE' });
    // 내 목록이면 불러온 판(sha256)을 함께 보낸다: 그사이 다른 탭에서 목록을 고쳤으면 서버가 저장하지 않고 거절한다.
    const outcome = await analyzeWardRecording({
      blob, mime, durationMs: meta?.durationMs, enrollProfile: profile, enrollCommand: s.recordingId, commandList: listId,
      commandListSha256: listId ? (index.customList?.sha256 || '') : '', isCancelled: () => !mountedRef.current,
    });
    if (!mountedRef.current) return;
    if (outcome.kind === 'invalid') dispatch({ type: 'RECORD_FAILED', message: outcome.message });
    else if (outcome.kind === 'done') {
      const listChanged = storeOutcome(outcome.results, s.recordingId, index).reason === STORE_LIST_CHANGED;
      dispatch({ type: 'STORE_RESULT', results: outcome.results });
      loadProfile(profile);
      if (listChanged) onListChanged();
    } else if (outcome.kind === 'error' && outcome.listChanged) {
      dispatch({ type: 'ANALYSIS_ERROR', message: ENROLL_TEXT.listChanged });
      onListChanged(); // 저장하지 않았다. 새 목록으로 화면을 다시 연다
    } else if (outcome.kind === 'error' || outcome.kind === 'timeout') {
      dispatch({ type: 'ANALYSIS_ERROR', message: outcome.kind === 'timeout' ? ENROLL_TEXT.timeout : outcome.message });
      loadProfile(profile); // 서버가 늦게라도 저장했을 수 있으니 목록을 서버 상태로 맞춘다
    }
    startCooldown();
  }, [dispatch, index, listId, loadProfile, onListChanged, startCooldown]);

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
    api.getWardProfiles()
      .then((data) => { if (alive) setChoices(profileChoices(data, '')); })
      .catch(() => { if (alive) setChoices([]); });
    return () => { alive = false; };
  }, [api, state.phase]);

  // 주소의 등록 이름을 직접 바꾸면 그 이름으로 다시 연다.
  useEffect(() => {
    const onHash = () => {
      const id = profileFromHash(globalThis.location?.hash);
      if (id && id !== stateRef.current.profile) dispatch({ type: 'CHOOSE', profile: id });
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [dispatch]);

  /** 병실 화면 설정의 등록 이름(이 목록 몫)을 이 등록으로 바꾼다. 기본 목록은 예전과 같은 enrollProfile이다. */
  const applyProfile = useCallback((profile) => {
    if (!profile) return;
    const current = loadSettings(storage);
    if (!listId) {
      if (current.enrollProfile !== profile) saveSettings(storage, { ...current, enrollProfile: profile });
    } else if (enrollProfileFor(current, listId) !== profile) {
      saveSettings(storage, { ...current, ...enrollProfilePatch(current, listId, profile) });
    }
    setInUseProfile(profile);
  }, [listId, storage]);

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
      await api.deleteWardProfile(profile);
      setNotice(`등록 “${profile}”를 지웠어요. 처음부터 다시 찍어 주세요`);
      dispatch({ type: 'CHOOSE', profile }); // 같은 이름으로 다시 불러온다(빈 등록)
    } catch (err) {
      setNotice(errorMessageFrom(err, '등록을 지우지 못했어요'));
    } finally {
      if (mountedRef.current) setDeleteBusy(false);
    }
  }, [api, deleteBusy, dispatch]);

  const command = commandById(index, state.currentId);
  const rows = commandRows(index, state.info, state.currentId);
  const total = index.list.length;
  const count = state.info?.count ?? 0;
  const line = state.info ? enrollStatusLine(state.profile, state.info) : null;
  const busy = state.phase === 'recording' || state.phase === 'analyzing';
  const cameraReady = camStatus === 'ready';
  const canStart = canStartEnrollRecording(state, { cameraReady, coolingDown });
  const remaining = Math.max(0, Math.ceil((ENROLL_RECORD_SECONDS * 1000 - (rec.elapsedMs || 0)) / 1000));
  const last = state.last && state.last.commandId === state.currentId ? state.last
    : (state.last && state.last.kind === 'stored' && state.last.matches ? state.last : null);
  const uncertain = uncertainCommands(index, state.info);
  const inUse = inUseProfile === state.profile;
  const stale = state.info?.status === 'stale';
  const missingCount = rows.filter((row) => !row.done).length;
  const visibleRows = rows.filter((row) => listFilter === 'all' || (listFilter === 'review' ? row.warn : !row.done));
  const progress = total ? Math.min(100, Math.max(0, count / total * 100)) : 0;
  const priorSuccess = last?.kind === 'stored' && last.matches && last.commandId !== state.currentId;
  const stageLabel = state.phase === 'recording' ? '녹화 중' : state.phase === 'analyzing' ? '결과 기다리는 중'
    : state.phase === 'review' ? '다시 확인해 주세요' : state.phase === 'done' ? '저장 완료' : '촬영 준비';

  // 표시용 시간만 갱신한다. 기존 업로드·폴링·저장 상태 전환은 변경하지 않는다.
  useEffect(() => {
    if (state.phase !== 'analyzing') {
      setAnalysisElapsed(0);
      return undefined;
    }
    const started = analysisStartedRef.current ?? Date.now();
    const tick = () => setAnalysisElapsed(Math.max(0, Math.floor((Date.now() - started) / 1000)));
    tick();
    const timer = setInterval(tick, 500);
    return () => clearInterval(timer);
  }, [state.phase]);

  // 공통 화면 이동도 기존 등록 페이지의 녹화·저장 중 이동 제한을 따른다.
  useEffect(() => {
    const announce = (locked) => window.dispatchEvent(new CustomEvent('ward-ui-navigation-lock', {
      detail: { locked, reason: '등록 녹화·저장 중입니다' },
    }));
    announce(busy);
    return () => announce(false);
  }, [busy]);

  return (
    <div className="enroll-ui">
      <header className="enroll-page-heading" data-no-switch onClick={blurAfterPointerClick}>
        <div>
          <p className="enroll-eyebrow">나에게 맞춘 인식 준비</p>
          <h1>{ENROLL_TEXT.title}</h1>
          <p className="enroll-page-description">{PROTOTYPE_TEXT}</p>
        </div>
        <div className="enroll-heading-actions">
          {state.phase !== 'choose' && !busy && (
            <button type="button" className="enroll-btn enroll-btn-quiet" onClick={() => { setNotice(null); dispatch({ type: 'RESET' }); }}>
              {ENROLL_TEXT.otherProfile}
            </button>
          )}
          <button type="button" className="enroll-btn enroll-btn-outline" onClick={goWard} disabled={busy}>
            {ENROLL_TEXT.backToWard}
          </button>
        </div>
      </header>

      {notice && <p role="status" className="enroll-feedback enroll-feedback-warning">{notice}</p>}

      {state.phase === 'choose' && (
        <main className="enroll-start-panel">
          <div className="enroll-start-icon" aria-hidden="true">01</div>
          <h2>등록 이름부터 정해 주세요</h2>
          <p>{ENROLL_TEXT.intro}</p>
          <p data-enroll-list={listId || 'default'} className="enroll-list-line">
            {ENROLL_TEXT.listLine(listName, total)}{index.customList ? ` · ${LISTS_TEXT.untested}` : ''}
          </p>
          <form className="enroll-name-form" onSubmit={(e) => { e.preventDefault(); choose(profileDraft.trim()); }}>
            <label>
              <span>{ENROLL_TEXT.profileLabel}</span>
              <input type="text" value={profileDraft} maxLength={32} autoComplete="off" spellCheck={false}
                placeholder="예: bed305" aria-describedby="enroll-name-help"
                onChange={(e) => setProfileDraft(e.target.value)} />
            </label>
            <button type="submit" className="enroll-btn enroll-btn-primary">{ENROLL_TEXT.start}</button>
          </form>
          <p id="enroll-name-help" className="enroll-help">{ENROLL_TEXT.privacy}</p>
          {state.error && <p role="alert" className="enroll-feedback enroll-feedback-warning">{state.error}</p>}
          {choices.length > 0 && (
            <section className="enroll-existing" aria-label={ENROLL_TEXT.existing}>
              <h3>{ENROLL_TEXT.existing}</h3>
              <div className="enroll-profile-choices">
                {choices.map((c) => (
                  <button key={c.profile} type="button" className="enroll-existing-choice" onClick={() => choose(c.profile)}>
                    <strong>{c.profile}</strong>
                    <span>{c.count !== null ? `저장 ${c.count}/${total}` : '등록 상태 확인'}
                      {c.status === 'stale' ? ' · 다시 등록 필요' : ''}</span>
                    <span aria-hidden="true">→</span>
                  </button>
                ))}
              </div>
            </section>
          )}
        </main>
      )}

      {state.phase === 'loading' && (
        <div className="enroll-loading-panel" role="status"><span className="enroll-spinner" aria-hidden="true" />{ENROLL_TEXT.loading}</div>
      )}

      {!['choose', 'loading'].includes(state.phase) && (
        <>
          <section className="enroll-progress-panel" aria-label="등록 진행">
            <div className="enroll-progress-heading">
              <div className="enroll-profile-label">
                <span className="enroll-label">등록 이름</span><strong>{state.profile}</strong>
                <span className="enroll-list-name" data-enroll-list={listId || 'default'}>
                  {listName}{index.customList ? ` · ${LISTS_TEXT.untested}` : ''}
                </span>
              </div>
              <div className="enroll-counts">
                <span className="enroll-count-main">저장 <strong>{count}<small>/{total}</small></strong></span>
                <span className={uncertain.length ? 'enroll-chip enroll-chip-warning' : 'enroll-chip'}>확인 필요 {uncertain.length}</span>
                <span className="enroll-chip">미등록 {missingCount}</span>
              </div>
            </div>
            <div className="enroll-progress-track" role="progressbar" aria-label="저장 진행" aria-valuemin={0} aria-valuemax={total} aria-valuenow={count}>
              <span style={{ width: `${progress}%` }} />
            </div>
            <p className="enroll-help">확인 필요 항목도 저장 개수에 포함됩니다. 원하는 항목은 아래 목록에서 다시 찍을 수 있어요.</p>
          </section>

          <main className="enroll-workspace">
            <section aria-label="카메라" className="enroll-camera-panel">
              <div className="enroll-camera-heading"><h2>카메라 미리보기</h2><span className="enroll-camera-state">{cameraReady ? '준비됨' : camStatus === 'recording' ? '녹화 중' : camStatus === 'error' ? '카메라를 쓸 수 없음' : '준비 확인'}</span></div>
              <div className={`enroll-camera-preview${state.phase === 'recording' ? ' is-recording' : ''}`}>
                <video ref={videoRef} playsInline muted />
                {camStatus !== 'ready' && camStatus !== 'recording' && (
                  <div className="enroll-camera-placeholder">{cameraText(rec)}</div>
                )}
              </div>
              <p className="enroll-help">얼굴과 입이 화면 안에 보이도록 맞춰 주세요.</p>
              {camStatus === 'error' && globalThis.isSecureContext && (
                <div data-no-switch onClick={blurAfterPointerClick}>
                  <button type="button" onClick={() => open()} className="enroll-btn enroll-btn-outline">카메라 다시 열기</button>
                </div>
              )}
              {line && line.tone === 'warn' && stale && (
                <div role="alert" className="enroll-feedback enroll-feedback-warning">
                  <p><AlertIcon /> {ENROLL_TEXT.stale}</p>
                  <button type="button" className="enroll-btn enroll-btn-danger" onClick={deleteAndRestart} disabled={deleteBusy || busy}>
                    {deleteBusy ? '지우는 중…' : ENROLL_TEXT.deleteAndRestart}
                  </button>
                </div>
              )}
            </section>

            <section ref={taskPanelRef} aria-live="polite" className={`enroll-task-panel phase-${state.phase}`}>
              <div className="enroll-task-topline"><span className="enroll-stage-label">{stageLabel}</span>{command && state.phase !== 'done' && <span className="enroll-current-id">현재 {command.id} / {total}개</span>}</div>
              {state.phase === 'done' ? (
                <>
                  <div className="enroll-completion-title"><CheckIcon /><h2>{ENROLL_TEXT.doneTitle} · {total}개 문장 저장</h2></div>
                  {uncertain.length > 0 && (
                    <div role="alert" data-uncertain-count={uncertain.length} className="enroll-feedback enroll-feedback-warning">
                      <h3><AlertIcon /> {ENROLL_TEXT.uncertainTitle(uncertain.length)}</h3>
                      <p>{ENROLL_TEXT.uncertainSub}</p>
                      <ul className="enroll-uncertain-list" data-no-switch onClick={blurAfterPointerClick}>
                        {uncertain.map(({ command: c, heardAs }) => (
                          <li key={c.id}>
                            <span>{ENROLL_TEXT.uncertainItem(c, heardAs)}</span>
                            <button type="button" data-retake-id={c.id} onClick={() => dispatch({ type: 'SELECT', commandId: c.id })} className="enroll-btn enroll-btn-outline">
                              {ENROLL_TEXT.retake}
                            </button>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  <p className={`enroll-use-status${inUse ? ' is-active' : ''}`} data-enroll-in-use={inUse ? 'true' : 'false'}>
                    {inUse ? ENROLL_TEXT.doneSub(state.profile) : ENROLL_TEXT.doneNotInUse(state.profile)}
                  </p>
                  <div className="enroll-task-actions" data-no-switch onClick={blurAfterPointerClick}>
                    {!inUse && (
                      <button type="button" onClick={() => applyProfile(state.profile)} className={`enroll-btn ${uncertain.length ? 'enroll-btn-outline' : 'enroll-btn-primary'}`}>
                        {uncertain.length > 0 ? ENROLL_TEXT.useAnyway : ENROLL_TEXT.useThis}
                      </button>
                    )}
                    <button type="button" onClick={goWard} className={`enroll-btn ${inUse ? 'enroll-btn-primary' : 'enroll-btn-outline'}`}>{ENROLL_TEXT.backToWard}</button>
                  </div>
                  <p className="enroll-help">{ENROLL_TEXT.pickAgain}</p>
                </>
              ) : command && (
                <>
                  <p className="enroll-prompt-label">{ENROLL_TEXT.sayThis}{command.emergency && <span className="enroll-emergency-label">긴급 문장</span>}</p>
                  <h2 className="enroll-prompt">“{command.text}”</h2>
                  <p className="enroll-speaking-tip">{ENROLL_TEXT.howTo}</p>
                  <p className="enroll-display-note">{ENROLL_TEXT.shownAs(command.display)}</p>
                  {last && (
                    <div role={last.kind === 'stored' && last.matches ? 'status' : 'alert'} data-store-result={last.kind}
                      className={`enroll-feedback ${last.kind === 'stored' && last.matches ? 'enroll-feedback-success' : 'enroll-feedback-warning'}`}>
                      {last.kind === 'stored' && last.matches ? <CheckIcon /> : <AlertIcon />}
                      <span>{priorSuccess ? `${last.commandId} 저장 완료 → 이제 ${command.id}를 찍으세요` : last.message}</span>
                    </div>
                  )}
                  <div className="enroll-task-actions" data-no-switch onClick={blurAfterPointerClick}>
                    {state.phase === 'recording' && (
                      <div className="enroll-recording-status" role="status">
                        <p><span className="enroll-recording-dot" aria-hidden="true" />녹화 중 <strong>{remaining}초</strong> 남음</p>
                        <div className="enroll-recording-track" aria-hidden="true"><span style={{ width: `${Math.max(0, remaining / ENROLL_RECORD_SECONDS * 100)}%` }} /></div>
                        <span className="enroll-help">시간이 끝나면 자동으로 저장을 시작합니다.</span>
                      </div>
                    )}
                    {state.phase === 'analyzing' && (
                      <div className="enroll-analysis-status" role="status">
                        <span className="enroll-spinner" aria-hidden="true" />
                        <div><p>등록 결과를 기다리는 중 <strong>{analysisElapsed}초</strong></p>
                          <span className="enroll-help">{analysisElapsed >= 15 ? '평소보다 오래 걸리고 있어요. 서버 처리 결과를 기다리고 있습니다.' : '녹화한 영상을 보내고 입 모양을 확인하고 있어요.'}</span></div>
                      </div>
                    )}
                    {(state.phase === 'ready' || state.phase === 'review') && (
                      <button type="button" onClick={startRecording} disabled={!canStart} className="enroll-btn enroll-btn-primary enroll-record-button">
                        {state.phase === 'review' || (last && last.kind !== 'stored' && last.commandId === command.id) ? ENROLL_TEXT.retake : '녹화 시작'}
                        <span className="enroll-button-shortcut">스페이스바 · {ENROLL_RECORD_SECONDS}초</span>
                      </button>
                    )}
                    {state.phase === 'review' && (
                      <button type="button" onClick={() => dispatch({ type: 'NEXT' })} className="enroll-btn enroll-btn-outline">확인 필요로 남기고 다음</button>
                    )}
                  </div>
                  {coolingDown && !busy && <p className="enroll-help">잠시 뒤 다음 녹화를 시작할 수 있어요.</p>}
                  {stale && !busy && <p className="enroll-help">{ENROLL_TEXT.staleNoRecord}</p>}
                  {!stale && !cameraReady && !busy && <p className="enroll-help">{ENROLL_TEXT.cameraFirst}</p>}
                  {busy && <p className="enroll-help">녹화·저장 중에는 등록 이름과 촬영 항목을 바꿀 수 없어요.</p>}
                </>
              )}
            </section>

            <section aria-label="명령 목록" className="enroll-command-panel" data-no-switch onClick={blurAfterPointerClick}>
              <div className="enroll-list-heading">
                <div><h2>촬영할 문장</h2><p className="enroll-help">현재 문장은 위에 표시됩니다. 다른 문장을 선택하면 다시 찍을 수 있어요.</p></div>
                <div className="enroll-filters" role="group" aria-label="명령 목록 필터">
                  {[['missing', '미등록', missingCount], ['review', '확인 필요', uncertain.length], ['all', '전체', total]].map(([value, label, n]) => (
                    <button key={value} type="button" aria-pressed={listFilter === value} onClick={() => setListFilter(value)}>
                      {label} <span>{n}</span>
                    </button>
                  ))}
                </div>
              </div>
              {visibleRows.length === 0 ? (
                <div className="enroll-empty-list">
                  <CheckIcon /><p>{listFilter === 'review' ? '확인 필요한 문장이 없어요.' : '미등록 문장이 없어요.'}</p>
                  <button type="button" className="enroll-btn enroll-btn-quiet" onClick={() => setListFilter('all')}>전체 문장 보기</button>
                </div>
              ) : (
                <ol className="enroll-command-grid">
                  {visibleRows.map((row) => (
                    <li key={row.id}>
                      <button type="button" data-command-id={row.id} data-done={row.done ? 'true' : 'false'}
                        aria-current={row.current ? 'true' : undefined}
                        onClick={() => { dispatch({ type: 'SELECT', commandId: row.id }); revealTaskPanel(taskPanelRef.current); }} disabled={busy}
                        className={`enroll-command ${row.current ? 'is-current' : ''}`}>
                        <div className="enroll-command-meta"><span>{row.id}</span><span className={`enroll-command-mark ${row.warn ? 'is-warning' : row.done ? 'is-saved' : ''}`}>
                          {row.warn ? '확인 필요' : row.done ? '✓ 저장됨' : '미등록'}
                        </span></div>
                        <span className="enroll-command-title">{row.display}</span>
                        {row.emergency && <span className="enroll-command-emergency"><AlertIcon />긴급</span>}
                        {row.current && <span className="enroll-current-note">현재 촬영할 문장</span>}
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </main>
        </>
      )}
    </div>
  );
}
