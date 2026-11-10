// ListsPage.jsx — 내 명령 목록 화면 (#lists, 병실 화면 설정의 [내 목록 관리]에서 연다). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 기본 목록(commands.json)은 읽기만 하고, 사용자가 만든 '내 목록'을 서버(/api/ward/lists)에 만들고 고치고 지운다.
// 검사의 기준은 서버다. 입력하는 동안 잠깐 쉬면 POST /api/ward/lists/check로 검사해 줄 옆에 오류(빨강)·주의(노랑)를 보이고,
// 저장(PUT)할 때 서버가 다시 검사한다. 편집 상태도는 ward/wardLists.js(순수 함수)다.
// [이 목록으로 병실 화면 쓰기]는 병실 화면 설정의 명령 목록(commandList)만 바꾼다. 기본 목록과 그 기준값은 그대로다.
// 저장은 조건부다: 새 목록은 '없을 때만 만들기', 저장된 목록은 '불러온 판일 때만 고치기'라서 다른 탭이 만든·고친 목록을
// 말없이 덮어쓰지 않는다(서버 409). [예시 목록 불러오기]는 언제나 새 목록으로 연다. 목록들을 불러오지 못했으면 새 목록을 막는다.
// 이 목록에 입 모양 등록이 있으면 고친 문장 때문에 다시 찍어야 할 개수를 알려 준다.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { checkWardList, deleteWardList, getWardList, getWardLists, saveWardList, wardProfileApi } from '../api/client';
import COMMANDS from '../ward/commands.json';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import { buildCustomCommandIndex } from '../ward/wardCommands';
import { parseProfileInfo, wardReturnHash } from '../ward/wardEnroll';
import {
  DISPLAY_MAX_CHARS, LISTS_TEXT, MAX_COMMANDS, MAX_LISTS, TITLE_MAX_CHARS, canSaveList, canUseList, checksFromError,
  createListEditor, defaultListRows, editorMessages, listIdProblem, listRequestErrorMessage, parseExamples, parseListSummaries,
  quickTitleProblem, reenrollSummary, requestBody, rowLength, saveCondition, stepListEditor, suggestListId,
} from '../ward/wardLists';
import { SETTINGS_STORAGE_KEY, enrollProfileFor, loadSettings, safeLocalStorage, saveSettings } from '../ward/wardSettings';
import { WARD_COLORS } from '../ward/wardTheme';

// 예시 목록 파일은 있으면 읽는다(없어도 빌드·화면이 동작하도록 glob으로 읽는다).
const EXAMPLE_FILES = import.meta.glob('../ward/custom_list_examples.json', { eager: true, import: 'default' });
const EXAMPLES = parseExamples(Object.values(EXAMPLE_FILES)[0]);
const CATEGORIES = COMMANDS.categories;
const DEFAULT_ROWS = defaultListRows(COMMANDS);
const PROTOTYPE_TEXT = '시연용 연구 시제품 · 실제 환자 미검증';
/** 입력을 멈추고 이 시간(ms)이 지나면 서버 검사를 한다. */
const CHECK_DEBOUNCE_MS = 500;
const NEW_KEY = '__new__';

/**
 * 큰 버튼 공통 모양.
 * @param {'primary'|'secondary'|'danger'} kind
 * @returns {object}
 */
function buttonStyle(kind) {
  if (kind === 'primary') return { background: WARD_COLORS.infoBg, color: WARD_COLORS.infoInk, borderColor: WARD_COLORS.infoBg };
  if (kind === 'danger') return { background: WARD_COLORS.paper, color: WARD_COLORS.emergencyBg, borderColor: WARD_COLORS.emergencyBg };
  return { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.ink };
}

/**
 * 주소 해시를 바꾼다.
 * @param {string} hash
 */
function goHash(hash) {
  try {
    globalThis.location.hash = hash;
  } catch {
    /* 주소를 바꾸지 못하면 그대로 둔다 */
  }
}

/**
 * 오류·주의 한 줄.
 * @param {{issue: {level:string, strong:boolean, message:string}}} props
 * @returns {JSX.Element}
 */
function IssueLine({ issue }) {
  if (issue.level === 'error') {
    return (
      <li data-issue="error" className="flex items-start gap-2 rounded-md border-2 px-2 py-1 text-base font-semibold" style={{ borderColor: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyBg, background: WARD_COLORS.paper }}>
        <AlertIcon title="오류" /> <span>오류: {issue.message}</span>
      </li>
    );
  }
  return (
    <li
      data-issue={issue.strong ? 'strong-warning' : 'warning'}
      className="flex items-start gap-2 rounded-md px-2 py-1 text-base font-semibold"
      style={{ background: issue.strong ? WARD_COLORS.possibleBg : WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}
    >
      <AlertIcon title="주의" /> <span>{issue.strong ? '강한 주의' : '주의'}: {issue.message}</span>
    </li>
  );
}

/**
 * 기본 목록 (읽기 전용).
 * @param {{inUse: boolean, onUseDefault: () => void}} props
 * @returns {JSX.Element}
 */
function DefaultListView({ inUse, onUseDefault }) {
  return (
    <section aria-label={LISTS_TEXT.defaultName} className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-3xl font-black">{LISTS_TEXT.defaultName} ({DEFAULT_ROWS.length}개)</h2>
        {inUse ? (
          <span className="flex items-center gap-1 rounded-md px-2 py-1 text-lg font-bold" style={{ background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }}>
            <CheckIcon /> {LISTS_TEXT.inUse}
          </span>
        ) : (
          <button type="button" onClick={onUseDefault} className="rounded-xl border-4 px-4 py-2 text-lg font-bold" style={buttonStyle('secondary')}>
            {LISTS_TEXT.backToDefault}
          </button>
        )}
      </div>
      <p className="text-lg" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.defaultReadOnly}</p>
      <ol className="flex flex-col gap-2" data-default-list>
        {DEFAULT_ROWS.map((r) => (
          <li key={r.id} className="grid gap-x-4 gap-y-1 rounded-xl border-2 px-3 py-2 md:grid-cols-[4rem_minmax(0,1.2fr)_minmax(0,1fr)_7rem_auto]" style={{ borderColor: WARD_COLORS.line }}>
            <span className="font-black">{r.id}</span>
            <span className="text-lg font-semibold">{r.text}</span>
            <span className="text-lg">{r.display}</span>
            <span className="text-base" style={{ color: WARD_COLORS.inkSoft }}>{r.categoryLabel}</span>
            <span className="flex flex-wrap gap-2 text-base font-bold">
              {r.emergency && <span className="flex items-center gap-1" style={{ color: WARD_COLORS.emergencyBg }}><AlertIcon /> 긴급</span>}
              {r.cancel && <span>취소</span>}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/**
 * 편집 줄 하나.
 * @param {object} props
 * @returns {JSX.Element}
 */
function EditorRow({ row, position, total, messages, normalizedText, dispatch, disabled }) {
  const len = rowLength(row.text);
  const hasError = messages.errors.length > 0 || !!messages.quick.text || !!messages.quick.display;
  const hasWarn = messages.warnings.length > 0;
  const n = position + 1;
  const showNormalized = normalizedText && row.text.trim() && normalizedText !== row.text;
  const borderColor = hasError ? WARD_COLORS.emergencyBg : (hasWarn ? WARD_COLORS.bannerInk : WARD_COLORS.line);
  return (
    <li data-row-key={row.key} className="rounded-xl border-4 p-3" style={{ borderColor }}>
      {/* 첫 줄: 번호 · 입 모양 문장 · 화면 문장 (문장 칸이 넓어야 9~13글자가 다 보인다). 둘째 줄: 종류 · 긴급 · 취소 · 순서 버튼. */}
      <div className="grid items-start gap-3 md:grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)]">
        <span className="pt-2 text-xl font-black">{n}</span>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold md:hidden">{LISTS_TEXT.colText}</span>
          <input
            type="text"
            lang="ko"
            value={row.text}
            maxLength={60}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colText}`}
            onChange={(e) => dispatch({ type: 'SET_TEXT', key: row.key, value: e.target.value })}
            className="w-full rounded-lg border-2 px-3 py-2 text-xl"
            style={{ borderColor: messages.quick.text ? WARD_COLORS.emergencyBg : WARD_COLORS.line }}
          />
          <span className="text-sm" style={{ color: WARD_COLORS.inkSoft }}>
            {row.text.trim() ? `${LISTS_TEXT.rowLength(len.syllables)}${len.good ? ' · 길이 좋음' : ''}` : ' '}
          </span>
          {messages.quick.text && <span className="text-base font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{messages.quick.text}</span>}
          {showNormalized && <span className="text-sm" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.normalizedAs(normalizedText)}</span>}
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm font-semibold md:hidden">{LISTS_TEXT.colDisplay}</span>
          <input
            type="text"
            lang="ko"
            value={row.display}
            maxLength={DISPLAY_MAX_CHARS + 10}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colDisplay}`}
            onChange={(e) => dispatch({ type: 'SET_DISPLAY', key: row.key, value: e.target.value })}
            className="w-full rounded-lg border-2 px-3 py-2 text-xl"
            style={{ borderColor: messages.quick.display ? WARD_COLORS.emergencyBg : WARD_COLORS.line }}
          />
          {messages.quick.display && <span className="text-base font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{messages.quick.display}</span>}
        </label>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-2 md:pl-[3.25rem]">
        <label className="flex items-center gap-2">
          <span className="text-base font-semibold">{LISTS_TEXT.colCategory}</span>
          <select
            value={row.category}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colCategory}`}
            onChange={(e) => dispatch({ type: 'SET_CATEGORY', key: row.key, value: e.target.value })}
            className="w-40 rounded-lg border-2 px-2 py-2 text-lg"
            style={{ borderColor: WARD_COLORS.line }}
          >
            {CATEGORIES.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-lg font-bold" style={row.emergency ? { color: WARD_COLORS.emergencyBg } : undefined}>
          <input
            type="checkbox"
            checked={row.emergency}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colEmergency}`}
            onChange={(e) => dispatch({ type: 'SET_EMERGENCY', key: row.key, value: e.target.checked })}
            className="h-6 w-6"
          />
          <span>{LISTS_TEXT.colEmergency}</span>
          {row.emergency && <AlertIcon title="긴급" />}
        </label>
        <label className="flex items-center gap-2 text-lg font-bold">
          <input
            type="checkbox"
            checked={row.cancel}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colCancel}`}
            onChange={(e) => dispatch({ type: 'SET_CANCEL', key: row.key, value: e.target.checked })}
            className="h-6 w-6"
          />
          <span>{LISTS_TEXT.colCancel}</span>
        </label>
        <div className="ml-auto flex flex-wrap gap-1">
          <button type="button" disabled={disabled || position === 0} onClick={() => dispatch({ type: 'MOVE_ROW', key: row.key, delta: -1 })} aria-label={`${n}번 ${LISTS_TEXT.moveUp}`} className="rounded-lg border-2 px-3 py-1 text-lg font-bold disabled:opacity-40" style={buttonStyle('secondary')}>↑</button>
          <button type="button" disabled={disabled || position === total - 1} onClick={() => dispatch({ type: 'MOVE_ROW', key: row.key, delta: 1 })} aria-label={`${n}번 ${LISTS_TEXT.moveDown}`} className="rounded-lg border-2 px-3 py-1 text-lg font-bold disabled:opacity-40" style={buttonStyle('secondary')}>↓</button>
          <button type="button" disabled={disabled} onClick={() => dispatch({ type: 'REMOVE_ROW', key: row.key })} aria-label={`${n}번 ${LISTS_TEXT.removeRow}`} className="rounded-lg border-2 px-3 py-1 text-lg font-bold disabled:opacity-40" style={buttonStyle('danger')}>✕</button>
        </div>
      </div>
      {(messages.errors.length > 0 || messages.warnings.length > 0) && (
        <ul className="mt-2 flex flex-col gap-1">
          {messages.errors.map((m, i) => <IssueLine key={`e${i}`} issue={m} />)}
          {messages.warnings.map((m, i) => <IssueLine key={`w${i}`} issue={m} />)}
        </ul>
      )}
    </li>
  );
}

/**
 * 내 명령 목록 화면.
 * @returns {JSX.Element}
 */
export default function ListsPage() {
  const storage = useMemo(() => safeLocalStorage(), []);
  const [settings, setSettings] = useState(() => loadSettings(storage));
  const [lists, setLists] = useState({ status: 'loading', items: [], message: null });
  // 고른 것: ''(기본 목록 보기) · NEW_KEY(새 목록) · 내 목록 ID
  const [selected, setSelected] = useState('');
  const editorRef = useRef(null);
  if (!editorRef.current) editorRef.current = createListEditor({ categories: CATEGORIES });
  const [editor, setEditor] = useState(editorRef.current);
  const [loadingList, setLoadingList] = useState(false);
  const [notice, setNotice] = useState(null); // {tone:'ok'|'warn', text}
  const [justUsed, setJustUsed] = useState(false);
  // 저장 조건이 맞지 않음(다른 곳에서 바뀜): 이 목록을 다시 불러오는 버튼을 보인다.
  const [conflict, setConflict] = useState(false);
  // 이 목록의 입 모양 등록에서 다시 찍어야 할 문장 (reenrollSummary 결과, 없으면 null)
  const [reenroll, setReenroll] = useState(null);
  const mountedRef = useRef(true);
  const openSeqRef = useRef(0);

  /** 편집 상태도에 이벤트 하나를 넣는다. */
  const dispatch = useCallback((event) => {
    const next = stepListEditor(editorRef.current, event);
    if (next !== editorRef.current) {
      editorRef.current = next;
      setEditor(next);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  /** 서버 목록들을 다시 읽는다. */
  const refreshLists = useCallback(() => {
    getWardLists()
      .then((data) => { if (mountedRef.current) setLists({ status: 'ready', items: parseListSummaries(data), message: null }); })
      .catch((err) => {
        if (mountedRef.current) setLists({ status: 'error', items: [], message: listRequestErrorMessage(err, LISTS_TEXT.loadListsFailed) });
      });
  }, []);

  useEffect(() => { refreshLists(); }, [refreshLists]);

  // 다른 탭(병실 화면 설정·등록 화면)이 설정을 바꾸면 '지금 병실 화면이 쓰는 목록'·등록 이름을 다시 읽는다.
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key === null || e.key === SETTINGS_STORAGE_KEY) setSettings(loadSettings(storage));
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [storage]);

  const ids = useMemo(() => lists.items.map((l) => l.id), [lists.items]);
  const editing = selected !== '';

  /** 저장하지 않은 내용이 있으면 버릴지 묻는다. */
  const confirmDiscard = useCallback(() => {
    const st = editorRef.current;
    if (selected === '' || !st.dirty) return true;
    return !!globalThis.confirm?.(LISTS_TEXT.discardConfirm);
  }, [selected]);

  /** 서버 목록 하나를 연다. */
  const openList = useCallback((id) => {
    openSeqRef.current += 1;
    const seq = openSeqRef.current;
    setSelected(id);
    setJustUsed(false);
    setConflict(false);
    setNotice(null);
    setLoadingList(true);
    getWardList(id)
      .then((data) => {
        if (!mountedRef.current || seq !== openSeqRef.current) return;
        dispatch({ type: 'LOAD_LIST', list: { ...data, id: typeof data?.id === 'string' ? data.id : id } });
      })
      .catch((err) => {
        if (!mountedRef.current || seq !== openSeqRef.current) return;
        setSelected('');
        setNotice({ tone: 'warn', text: listRequestErrorMessage(err, LISTS_TEXT.loadFailed) });
      })
      .finally(() => { if (mountedRef.current && seq === openSeqRef.current) setLoadingList(false); });
  }, [dispatch]);

  // 처음에는 병실 화면이 지금 쓰는 내 목록을 연다(없으면 기본 목록).
  const openedInitialRef = useRef(false);
  useEffect(() => {
    if (openedInitialRef.current || lists.status !== 'ready') return;
    openedInitialRef.current = true;
    if (settings.commandList && ids.includes(settings.commandList)) openList(settings.commandList);
  }, [lists.status, ids, settings.commandList, openList]);

  /** 목록 고르기 (기본 목록 · 새 목록 · 내 목록). */
  const choose = useCallback((key) => {
    if (key === selected && key !== NEW_KEY) return;
    if (!confirmDiscard()) return;
    openSeqRef.current += 1;
    setLoadingList(false);
    setJustUsed(false);
    setConflict(false);
    setNotice(null);
    if (key === '') {
      setSelected('');
    } else if (key === NEW_KEY) {
      dispatch({ type: 'NEW', id: suggestListId(ids) });
      setSelected(NEW_KEY);
    } else {
      openList(key);
    }
  }, [confirmDiscard, dispatch, ids, openList, selected]);

  /**
   * 예시 목록을 새 목록으로 불러온다 (저장 전까지 서버에 남지 않는다). 저장된 목록을 열어 두었어도 그 목록을 바꾸지 않는다.
   * 서버 목록들을 불러오지 못했으면 새 ID가 겹치는지 알 수 없어 열지 않는다.
   */
  const loadExample = useCallback((example) => {
    if (lists.status !== 'ready') return;
    const st = editorRef.current;
    if (selected !== '' && st.dirty && !globalThis.confirm?.(LISTS_TEXT.replaceConfirm)) return;
    openSeqRef.current += 1;
    setLoadingList(false);
    const id = suggestListId(ids);
    if (selected === '' || st.mode !== 'new') dispatch({ type: 'NEW', id });
    setSelected(NEW_KEY);
    dispatch({ type: 'LOAD_EXAMPLE', example, id });
    setJustUsed(false);
    setConflict(false);
    setNotice({ tone: 'ok', text: LISTS_TEXT.exampleLoaded(example.title) });
  }, [dispatch, ids, lists.status, selected]);

  // 입력을 멈추면 서버 검사 (새 목록은 무언가 적은 뒤부터). 오래된 검사 결과는 상태도가 버린다(editSeq).
  useEffect(() => {
    if (!editing || loadingList) return undefined;
    const st = editorRef.current;
    if (st.mode === 'new' && !st.dirty) return undefined;
    if (st.check.seq === st.editSeq && ['done', 'checking'].includes(st.check.status)) return undefined;
    const seq = st.editSeq;
    const timer = setTimeout(() => {
      dispatch({ type: 'CHECK_START', seq });
      checkWardList(requestBody(editorRef.current))
        .then((data) => { if (mountedRef.current) dispatch({ type: 'CHECK_DONE', seq, data }); })
        .catch((err) => {
          if (mountedRef.current) dispatch({ type: 'CHECK_FAILED', seq, message: listRequestErrorMessage(err, LISTS_TEXT.checkFailed) });
        });
    }, CHECK_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [editing, loadingList, editor.editSeq, editor.dirty, dispatch]);

  // 저장하지 않은 내용이 있으면 창을 닫기 전에 묻는다.
  useEffect(() => {
    const onBeforeUnload = (e) => {
      if (selected !== '' && editorRef.current.dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [selected]);

  /** 설정을 바꾸고 다시 읽는다. 저장하지 못하면 알린다. */
  const updateSettings = useCallback((patch) => {
    const current = loadSettings(storage);
    const ok = saveSettings(storage, { ...current, ...patch });
    setSettings(loadSettings(storage));
    if (!ok) setNotice({ tone: 'warn', text: '이 브라우저에 설정을 저장하지 못했어요 (사생활 보호 모드 등)' });
    return ok;
  }, [storage]);

  const save = useCallback(async () => {
    const st = editorRef.current;
    if (!canSaveList(st, ids).ok) return;
    if (st.mode === 'new' && lists.status !== 'ready') return;
    const seq = st.editSeq;
    const id = st.id;
    dispatch({ type: 'SAVE_START' });
    try {
      // 새 목록은 '없을 때만 만들기', 저장된 목록은 '불러온 판일 때만 고치기' (다른 탭이 만든·고친 목록을 덮어쓰지 않는다)
      const data = await saveWardList(id, requestBody(st), saveCondition(st));
      if (!mountedRef.current) return;
      dispatch({ type: 'SAVED', list: { ...data, id: typeof data?.id === 'string' ? data.id : id } });
      setSelected(typeof data?.id === 'string' ? data.id : id);
      setJustUsed(false);
      setConflict(false);
      setNotice({ tone: 'ok', text: LISTS_TEXT.saved });
      refreshLists();
    } catch (err) {
      if (!mountedRef.current) return;
      const checks = checksFromError(err);
      const status = err?.response?.status;
      dispatch({
        type: 'SAVE_FAILED', seq, checks,
        message: checks?.detail || listRequestErrorMessage(err, '저장하지 못했어요'),
      });
      if (status === 409) {
        setConflict(st.mode === 'edit');
        refreshLists(); // 새 목록 ID가 이미 있으면 ID 칸이 바로 '같은 ID가 있어요'를 보인다
      }
    }
  }, [dispatch, ids, lists.status, refreshLists]);

  /** 저장 조건이 맞지 않았을 때: 서버의 지금 목록을 다시 불러온다(여기서 고친 내용은 버린다). */
  const reloadCurrent = useCallback(() => {
    const st = editorRef.current;
    if (st.mode !== 'edit' || !st.savedId) return;
    if (st.dirty && !globalThis.confirm?.(LISTS_TEXT.reloadConfirm)) return;
    openList(st.savedId);
    refreshLists();
  }, [openList, refreshLists]);

  const remove = useCallback(async () => {
    const st = editorRef.current;
    if (st.mode !== 'edit' || !st.savedId) return;
    const title = st.title || st.savedId;
    if (!globalThis.confirm?.(LISTS_TEXT.deleteConfirm(title))) return;
    try {
      await deleteWardList(st.savedId);
      if (!mountedRef.current) return;
      const current = loadSettings(storage);
      const byList = { ...current.enrollProfileByList };
      delete byList[st.savedId];
      const inUse = current.commandList === st.savedId;
      // 지운 목록을 병실 화면이 쓰고 있었다면 기본 목록으로 바꾸고 그렇게 알린다(목록이 없으면 병실 화면은 입 읽기를 막는다).
      updateSettings(inUse ? { commandList: '', enrollProfileByList: byList } : { enrollProfileByList: byList });
      setSelected('');
      setJustUsed(false);
      dispatch({ type: 'NEW', id: '' });
      setNotice({ tone: inUse ? 'warn' : 'ok', text: inUse ? LISTS_TEXT.deletedInUse(title) : LISTS_TEXT.deleted(title) });
      refreshLists();
    } catch (err) {
      if (mountedRef.current) setNotice({ tone: 'warn', text: listRequestErrorMessage(err, '목록을 지우지 못했어요') });
    }
  }, [dispatch, refreshLists, storage, updateSettings]);

  // 저장된(바뀐 것 없는) 목록에 입 모양 등록이 있으면, 고친 문장 때문에 다시 찍어야 할 개수를 서버에서 읽어 알려 준다.
  const reenrollKey = editor.mode === 'edit' && editor.savedId && !editor.dirty && !editor.saving && editor.savedSha
    ? `${editor.savedId}|${editor.savedSha}|${enrollProfileFor(settings, editor.savedId)}` : '';
  useEffect(() => {
    setReenroll(null);
    if (!reenrollKey) return undefined;
    const st = editorRef.current;
    const profile = enrollProfileFor(settings, st.savedId);
    if (!profile) return undefined;
    let alive = true;
    let index = null;
    try {
      index = buildCustomCommandIndex({ id: st.savedId, title: st.title || st.savedId, sha256: st.savedSha,
        commands: st.rows.map((r, i) => ({ id: `U${String(i + 1).padStart(2, '0')}`, text: r.text, display: r.display,
          category: r.category, emergency: r.emergency, cancel: r.cancel })) }, COMMANDS);
    } catch {
      return undefined;
    }
    wardProfileApi(st.savedId).getWardProfile(profile)
      .then((data) => { if (alive && mountedRef.current) setReenroll(reenrollSummary(parseProfileInfo(data, index, profile))); })
      .catch(() => { if (alive && mountedRef.current) setReenroll(null); });
    return () => { alive = false; };
  }, [reenrollKey, settings]);

  const applyThisList = useCallback(() => {
    const st = editorRef.current;
    if (!canUseList(st)) return;
    if (updateSettings({ commandList: st.savedId })) setJustUsed(true);
  }, [updateSettings]);

  const applyDefault = useCallback(() => {
    updateSettings({ commandList: '' });
    setJustUsed(false);
  }, [updateSettings]);

  const leaveTo = useCallback((hash) => {
    if (!confirmDiscard()) return;
    goHash(hash);
  }, [confirmDiscard]);

  const usedList = lists.items.find((l) => l.id === settings.commandList) || null;
  const usingName = settings.commandList ? `내 목록 ‘${usedList?.title || settings.commandList}’` : LISTS_TEXT.defaultName;
  const msgs = editorMessages(editor);
  const saveState = canSaveList(editor, ids);
  const canUse = canUseList(editor);
  const usingThis = editor.mode === 'edit' && editor.savedId === settings.commandList && !!settings.commandList;
  const idProblem = editor.mode === 'new' ? listIdProblem(editor.id, ids, null) : null;
  const titleProblem = editor.title.trim() || editor.dirty ? quickTitleProblem(editor.title) : null;
  const checkLine = (() => {
    if (editor.check.status === 'failed') return { tone: 'warn', text: editor.check.message || LISTS_TEXT.checkFailed };
    if (msgs.current) {
      const e = editor.check.errors.length;
      const w = editor.check.warnings.length;
      return { tone: e ? 'error' : (w ? 'warn' : 'ok'), text: e || w ? LISTS_TEXT.summary(e, w) : LISTS_TEXT.noProblems };
    }
    if (editor.mode === 'new' && !editor.dirty) return null;
    return { tone: 'neutral', text: LISTS_TEXT.checking };
  })();
  const normalizedRows = Array.isArray(editor.check.normalized?.commands) && msgs.current ? editor.check.normalized.commands : null;
  const busy = editor.saving || loadingList;

  return (
    <div className="min-h-screen" style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}>
      <header className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ background: WARD_COLORS.bannerBg, color: WARD_COLORS.paper }}>
        <h1 className="text-2xl font-bold">{LISTS_TEXT.title}</h1>
        <p className="text-base font-semibold" style={{ color: WARD_COLORS.bannerInk }}>{PROTOTYPE_TEXT}</p>
        <button
          type="button"
          onClick={() => leaveTo(wardReturnHash(globalThis.location))}
          className="ml-auto rounded-lg px-3 py-1 font-semibold"
          style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}
        >
          {LISTS_TEXT.backToWard}
        </button>
      </header>

      <div className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
        <p data-using-list={settings.commandList || 'default'} className="text-xl font-bold">
          {LISTS_TEXT.usingNow(usingName)}{settings.commandList ? ` · ${LISTS_TEXT.untested}` : ''}
        </p>
        {settings.commandList && (
          <button type="button" onClick={applyDefault} className="rounded-lg border-2 px-3 py-1 text-lg font-semibold" style={buttonStyle('secondary')}>
            {LISTS_TEXT.backToDefault}
          </button>
        )}
      </div>

      {notice && (
        <p
          role="status"
          className="px-4 py-2 text-lg font-semibold"
          style={notice.tone === 'ok' ? { background: WARD_COLORS.okBg, color: WARD_COLORS.okInk } : { background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}
        >
          {notice.text}
        </p>
      )}

      <main className="grid gap-4 p-4 lg:grid-cols-[minmax(0,300px)_minmax(0,1fr)]">
        <aside aria-label="목록 고르기" className="flex flex-col gap-3">
          <p className="text-base" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.intro}</p>
          <button
            type="button"
            aria-current={selected === '' ? 'true' : undefined}
            onClick={() => choose('')}
            className="flex flex-col items-start rounded-xl border-4 px-3 py-2 text-left"
            style={selected === '' ? buttonStyle('primary') : buttonStyle('secondary')}
          >
            <span className="text-xl font-bold">{LISTS_TEXT.defaultName} ({DEFAULT_ROWS.length}개)</span>
            <span className="text-sm">바꿀 수 없음{!settings.commandList ? ` · ${LISTS_TEXT.inUse}` : ''}</span>
          </button>
          <h2 className="mt-2 text-lg font-bold">{LISTS_TEXT.myLists} ({lists.items.length}/{MAX_LISTS})</h2>
          {lists.status === 'loading' && <p className="text-base">불러오는 중이에요</p>}
          {lists.status === 'error' && (
            <div role="alert" className="flex flex-col gap-2">
              <p className="text-base font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{lists.message}</p>
              <p className="text-base">{LISTS_TEXT.listsNeedReload}</p>
              <button type="button" onClick={refreshLists} className="self-start rounded-lg border-2 px-3 py-1 text-lg font-semibold" style={buttonStyle('secondary')}>
                {LISTS_TEXT.reloadLists}
              </button>
            </div>
          )}
          {lists.status === 'ready' && lists.items.length === 0 && <p className="text-base" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.noLists}</p>}
          <ul className="flex flex-col gap-2">
            {lists.items.map((l) => (
              <li key={l.id}>
                <button
                  type="button"
                  data-list-id={l.id}
                  aria-current={selected === l.id ? 'true' : undefined}
                  onClick={() => choose(l.id)}
                  className="flex w-full flex-col items-start rounded-xl border-4 px-3 py-2 text-left"
                  style={selected === l.id ? buttonStyle('primary') : buttonStyle('secondary')}
                >
                  <span className="text-xl font-bold">{l.title}</span>
                  <span className="text-sm">
                    {l.id}{l.count !== null ? ` · ${l.count}개` : ''}{l.emergencyCount !== null ? ` · 긴급 ${l.emergencyCount}` : ''}
                    {settings.commandList === l.id ? ` · ${LISTS_TEXT.inUse}` : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            onClick={() => choose(NEW_KEY)}
            disabled={lists.status !== 'ready' || lists.items.length >= MAX_LISTS}
            className="rounded-xl border-4 px-3 py-2 text-xl font-bold disabled:opacity-50"
            style={buttonStyle('secondary')}
          >
            + {LISTS_TEXT.newList}
          </button>
          {lists.items.length >= MAX_LISTS && <p className="text-base">{LISTS_TEXT.tooMany}</p>}
          {EXAMPLES.map((ex) => (
            <button
              key={ex.title}
              type="button"
              data-example={ex.title}
              onClick={() => loadExample(ex)}
              disabled={busy || lists.status !== 'ready' || lists.items.length >= MAX_LISTS}
              className="rounded-xl border-2 border-dashed px-3 py-2 text-lg font-semibold disabled:opacity-50"
              style={buttonStyle('secondary')}
            >
              {LISTS_TEXT.loadExample} ({ex.title})
            </button>
          ))}
        </aside>

        {!editing && <DefaultListView inUse={!settings.commandList} onUseDefault={applyDefault} />}

        {editing && loadingList && <p className="text-2xl font-semibold">{LISTS_TEXT.loading}</p>}

        {editing && !loadingList && (
          <section aria-label="목록 고치기" className="flex flex-col gap-4">
            <div className="flex flex-wrap items-end gap-4">
              <label className="flex flex-col gap-1">
                <span className="text-lg font-semibold">{LISTS_TEXT.listTitle}</span>
                <input
                  type="text"
                  lang="ko"
                  value={editor.title}
                  maxLength={TITLE_MAX_CHARS + 10}
                  disabled={busy}
                  onChange={(e) => dispatch({ type: 'SET_TITLE', value: e.target.value })}
                  className="w-72 max-w-full rounded-lg border-2 px-3 py-2 text-2xl font-bold"
                  style={{ borderColor: titleProblem ? WARD_COLORS.emergencyBg : WARD_COLORS.line }}
                />
                {titleProblem && <span className="text-base font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{titleProblem}</span>}
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-lg font-semibold">{LISTS_TEXT.listId}</span>
                {editor.mode === 'new' ? (
                  <input
                    type="text"
                    value={editor.id}
                    maxLength={32}
                    autoComplete="off"
                    spellCheck={false}
                    disabled={busy}
                    onChange={(e) => dispatch({ type: 'SET_ID', value: e.target.value })}
                    className="w-56 max-w-full rounded-lg border-2 px-3 py-2 text-xl"
                    style={{ borderColor: idProblem ? WARD_COLORS.emergencyBg : WARD_COLORS.line }}
                  />
                ) : (
                  <span className="px-1 py-2 text-xl font-semibold">{editor.savedId}</span>
                )}
                {editor.mode === 'new' && <span className="text-sm" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.listIdHint}</span>}
                {idProblem && editor.id && <span className="text-base font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{idProblem}</span>}
              </label>
              {usingThis && (
                <span className="flex items-center gap-1 rounded-md px-2 py-1 text-lg font-bold" style={{ background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }}>
                  <CheckIcon /> {LISTS_TEXT.usingThis}
                </span>
              )}
            </div>

            {checkLine && (
              <p
                data-check-status={checkLine.tone}
                aria-live="polite"
                className="flex items-center gap-2 text-xl font-bold"
                style={{ color: checkLine.tone === 'error' ? WARD_COLORS.emergencyBg : WARD_COLORS.ink }}
              >
                {checkLine.tone === 'ok' ? <CheckIcon /> : (checkLine.tone === 'neutral' ? null : <AlertIcon />)} {checkLine.text}
              </p>
            )}
            {(msgs.global.errors.length > 0 || msgs.global.warnings.length > 0) && (
              <ul className="flex flex-col gap-1" aria-label="목록 전체 안내">
                {msgs.global.errors.map((m, i) => <IssueLine key={`ge${i}`} issue={m} />)}
                {msgs.global.warnings.map((m, i) => <IssueLine key={`gw${i}`} issue={m} />)}
              </ul>
            )}

            <div className="hidden gap-3 px-3 text-base font-bold md:grid md:grid-cols-[2.5rem_minmax(0,1.3fr)_minmax(0,1fr)]" style={{ color: WARD_COLORS.inkSoft }}>
              <span>#</span>
              <span>{LISTS_TEXT.colText}</span>
              <span>{LISTS_TEXT.colDisplay}</span>
            </div>
            <p className="text-sm" style={{ color: WARD_COLORS.inkSoft }}>
              {LISTS_TEXT.colTextHint} · {LISTS_TEXT.colDisplayHint} · 취소: {LISTS_TEXT.colCancelHint}
            </p>
            <ol className="flex flex-col gap-3">
              {editor.rows.map((row, i) => (
                <EditorRow
                  key={row.key}
                  row={row}
                  position={i}
                  total={editor.rows.length}
                  messages={msgs.byKey[row.key]}
                  normalizedText={normalizedRows?.[i]?.text || null}
                  dispatch={dispatch}
                  disabled={busy}
                />
              ))}
            </ol>

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => dispatch({ type: 'ADD_ROW' })}
                disabled={busy || editor.rows.length >= MAX_COMMANDS}
                className="rounded-xl border-4 px-5 py-2 text-xl font-bold disabled:opacity-50"
                style={buttonStyle('secondary')}
              >
                + {LISTS_TEXT.addRow} ({editor.rows.length}/{MAX_COMMANDS})
              </button>
              <button
                type="button"
                data-save
                onClick={save}
                disabled={!saveState.ok || (editor.mode === 'new' && lists.status !== 'ready')}
                className="rounded-xl border-4 px-6 py-2 text-2xl font-black disabled:opacity-50"
                style={buttonStyle('primary')}
              >
                {editor.saving ? LISTS_TEXT.saving : LISTS_TEXT.save}
              </button>
              {editor.mode === 'edit' && (
                <button type="button" onClick={remove} disabled={busy} className="rounded-xl border-4 px-5 py-2 text-xl font-bold disabled:opacity-50" style={buttonStyle('danger')}>
                  {LISTS_TEXT.deleteList}
                </button>
              )}
            </div>
            {saveState.reason && editor.dirty && <p className="text-lg font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{saveState.reason}</p>}
            {editor.saveMessage && editor.saveMessage !== LISTS_TEXT.saved && (
              <p role="alert" className="text-lg font-semibold" style={{ color: WARD_COLORS.emergencyBg }}>{editor.saveMessage}</p>
            )}
            {conflict && editor.mode === 'edit' && (
              <button type="button" data-reload-list onClick={reloadCurrent} disabled={busy} className="self-start rounded-xl border-4 px-5 py-2 text-xl font-bold disabled:opacity-50" style={buttonStyle('secondary')}>
                {LISTS_TEXT.reloadList}
              </button>
            )}
            {reenroll && (
              <div role="status" data-reenroll={reenroll.kind} className="flex flex-col gap-1 rounded-lg px-3 py-2" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
                <p className="flex items-center gap-2 text-xl font-bold"><AlertIcon /> {reenroll.text}</p>
                <p className="text-base">{LISTS_TEXT.reenrollHowTo}</p>
              </div>
            )}

            <div className="flex flex-col gap-3 rounded-xl border-4 p-4" style={{ borderColor: WARD_COLORS.infoBg }}>
              {usingThis ? (
                <p className="flex items-center gap-2 text-xl font-bold"><CheckIcon /> {LISTS_TEXT.usingThis} · {LISTS_TEXT.untested}</p>
              ) : (
                <button
                  type="button"
                  data-use-list
                  onClick={applyThisList}
                  disabled={!canUse}
                  className="self-start rounded-xl border-4 px-6 py-3 text-2xl font-bold disabled:opacity-50"
                  style={buttonStyle('primary')}
                >
                  {LISTS_TEXT.useThis}
                </button>
              )}
              {!canUse && !usingThis && <p className="text-base" style={{ color: WARD_COLORS.inkSoft }}>{LISTS_TEXT.useNeedsSave}</p>}
              {(usingThis || justUsed) && (
                <div className="flex flex-wrap gap-3">
                  <button type="button" onClick={() => leaveTo('#enroll')} className="rounded-xl border-4 px-5 py-2 text-xl font-bold" style={buttonStyle('primary')}>
                    {LISTS_TEXT.enrollThis}
                  </button>
                  <button type="button" onClick={() => leaveTo('#ward')} className="rounded-xl border-4 px-5 py-2 text-xl font-bold" style={buttonStyle('secondary')}>
                    {LISTS_TEXT.backToWard}
                  </button>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
