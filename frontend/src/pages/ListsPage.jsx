// ListsPage.jsx — 내 명령 목록 화면 (#lists, 병실 화면 설정의 [내 목록 관리]에서 연다). 시연용 연구 시제품이며 실제 환자에게 검증하지 않았다.
// 기본 목록(commands.json)은 읽기만 하고, 사용자가 만든 '내 목록'을 서버(/api/ward/lists)에 만들고 고치고 지운다.
// 검사의 기준은 서버다. 입력하는 동안 잠깐 쉬면 POST /api/ward/lists/check로 검사해 줄 옆에 오류(빨강)·주의(노랑)를 보이고,
// 저장(PUT)할 때 서버가 다시 검사한다. 편집 상태도는 ward/wardLists.js(순수 함수)다.
// [이 목록으로 병실 화면 쓰기]는 병실 화면 설정의 명령 목록(commandList)만 바꾼다. 기본 목록과 그 기준값은 그대로다.
// 저장은 조건부다: 새 목록은 '없을 때만 만들기', 저장된 목록은 '불러온 판일 때만 고치기'라서 다른 탭이 만든·고친 목록을
// 말없이 덮어쓰지 않는다(서버 409). [예시 목록 불러오기]는 언제나 새 목록으로 연다. 목록들을 불러오지 못했으면 새 목록을 막는다.
// 이 목록에 입 모양 등록이 있으면 고친 문장 때문에 다시 찍어야 할 개수를 알려 준다.
// 모양은 ward/lists-ui.css(.lists-ui 아래, 병실 공통 틀 WardShell 안)이며 1280·768·390px 폭에서 가로 스크롤 없이 쓴다.

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
import '../ward/lists-ui.css';

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
      <li data-issue="error" className="lists-issue is-error">
        <AlertIcon title="오류" /> <span>오류: {issue.message}</span>
      </li>
    );
  }
  return (
    <li
      data-issue={issue.strong ? 'strong-warning' : 'warning'}
      className={`lists-issue ${issue.strong ? 'is-strong' : 'is-warning'}`}
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
    <section aria-label={LISTS_TEXT.defaultName} className="lists-panel lists-default">
      <div className="lists-section-head">
        <h2 className="lists-section-title">{LISTS_TEXT.defaultName} ({DEFAULT_ROWS.length}개)</h2>
        {inUse ? (
          <span className="lists-chip lists-chip-ok lists-chip-lg">
            <CheckIcon /> {LISTS_TEXT.inUse}
          </span>
        ) : (
          <button type="button" onClick={onUseDefault} className="lists-btn lists-btn-outline">
            {LISTS_TEXT.backToDefault}
          </button>
        )}
      </div>
      <p className="lists-help">{LISTS_TEXT.defaultReadOnly}</p>
      {/* 표 머리 (넓은 화면에서만 보인다. 좁은 화면은 줄마다 카드로 쌓는다) */}
      <div className="lists-default-columns" aria-hidden="true">
        <span>#</span>
        <span>{LISTS_TEXT.colText}</span>
        <span>{LISTS_TEXT.colDisplay}</span>
        <span>{LISTS_TEXT.colCategory}</span>
        <span>{LISTS_TEXT.colEmergency} · {LISTS_TEXT.colCancel}</span>
      </div>
      <ol className="lists-default-rows" data-default-list>
        {DEFAULT_ROWS.map((r) => (
          <li key={r.id} className="lists-default-row">
            <span className="lists-default-id">{r.id}</span>
            <span className="lists-default-text">{r.text}</span>
            <span className="lists-default-display">{r.display}</span>
            <span className="lists-default-category">{r.categoryLabel}</span>
            <span className="lists-default-marks">
              {r.emergency && <span className="lists-mark lists-mark-emergency"><AlertIcon /> 긴급</span>}
              {r.cancel && <span className="lists-mark">취소</span>}
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
  const tone = hasError ? ' is-error' : (hasWarn ? ' is-warn' : '');
  return (
    <li data-row-key={row.key} className={`lists-row${tone}`}>
      {/* 첫 줄: 번호 · 입 모양 문장 · 화면 문장 (문장 칸이 넓어야 9~13글자가 다 보인다). 둘째 줄: 종류 · 긴급 · 취소 · 순서 버튼.
          좁은 화면(768px 미만)에서는 칸을 위아래로 쌓고 칸 이름을 줄마다 보인다. */}
      <div className="lists-row-main">
        <span className="lists-row-number">{n}</span>
        <label className="lists-field">
          <span className="lists-field-label">{LISTS_TEXT.colText}</span>
          <input
            type="text"
            lang="ko"
            value={row.text}
            maxLength={60}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colText}`}
            onChange={(e) => dispatch({ type: 'SET_TEXT', key: row.key, value: e.target.value })}
            className={`lists-input${messages.quick.text ? ' has-error' : ''}`}
          />
          <span className="lists-field-hint">
            {row.text.trim() ? `${LISTS_TEXT.rowLength(len.syllables)}${len.good ? ' · 길이 좋음' : ''}` : ' '}
          </span>
          {messages.quick.text && <span className="lists-field-error">{messages.quick.text}</span>}
          {showNormalized && <span className="lists-field-hint">{LISTS_TEXT.normalizedAs(normalizedText)}</span>}
        </label>
        <label className="lists-field">
          <span className="lists-field-label">{LISTS_TEXT.colDisplay}</span>
          <input
            type="text"
            lang="ko"
            value={row.display}
            maxLength={DISPLAY_MAX_CHARS + 10}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colDisplay}`}
            onChange={(e) => dispatch({ type: 'SET_DISPLAY', key: row.key, value: e.target.value })}
            className={`lists-input${messages.quick.display ? ' has-error' : ''}`}
          />
          {messages.quick.display && <span className="lists-field-error">{messages.quick.display}</span>}
        </label>
      </div>
      <div className="lists-row-controls">
        <label className="lists-category">
          <span>{LISTS_TEXT.colCategory}</span>
          <select
            value={row.category}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colCategory}`}
            onChange={(e) => dispatch({ type: 'SET_CATEGORY', key: row.key, value: e.target.value })}
            className="lists-select"
          >
            {CATEGORIES.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
        </label>
        <label className={`lists-toggle lists-toggle-emergency${row.emergency ? ' is-on' : ''}`}>
          <input
            type="checkbox"
            checked={row.emergency}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colEmergency}`}
            onChange={(e) => dispatch({ type: 'SET_EMERGENCY', key: row.key, value: e.target.checked })}
          />
          <span>{LISTS_TEXT.colEmergency}</span>
          {row.emergency && <AlertIcon title="긴급" />}
        </label>
        <label className={`lists-toggle${row.cancel ? ' is-on' : ''}`}>
          <input
            type="checkbox"
            checked={row.cancel}
            disabled={disabled}
            aria-label={`${n}번 ${LISTS_TEXT.colCancel}`}
            onChange={(e) => dispatch({ type: 'SET_CANCEL', key: row.key, value: e.target.checked })}
          />
          <span>{LISTS_TEXT.colCancel}</span>
        </label>
        <div className="lists-row-order">
          <button type="button" disabled={disabled || position === 0} onClick={() => dispatch({ type: 'MOVE_ROW', key: row.key, delta: -1 })} aria-label={`${n}번 ${LISTS_TEXT.moveUp}`} className="lists-icon-btn">↑</button>
          <button type="button" disabled={disabled || position === total - 1} onClick={() => dispatch({ type: 'MOVE_ROW', key: row.key, delta: 1 })} aria-label={`${n}번 ${LISTS_TEXT.moveDown}`} className="lists-icon-btn">↓</button>
          <button type="button" disabled={disabled} onClick={() => dispatch({ type: 'REMOVE_ROW', key: row.key })} aria-label={`${n}번 ${LISTS_TEXT.removeRow}`} className="lists-icon-btn lists-icon-btn-danger">✕</button>
        </div>
      </div>
      {(messages.errors.length > 0 || messages.warnings.length > 0) && (
        <ul className="lists-issues">
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
    <div className="lists-ui">
      <header className="lists-heading">
        <div className="lists-heading-text">
          <p className="lists-eyebrow">병실 화면 설정</p>
          <h1>{LISTS_TEXT.title}</h1>
          <p className="lists-description">{LISTS_TEXT.intro}</p>
          <p className="lists-prototype">{PROTOTYPE_TEXT}</p>
        </div>
        <div className="lists-heading-actions">
          <button
            type="button"
            onClick={() => leaveTo(wardReturnHash(globalThis.location))}
            className="lists-btn lists-btn-outline"
          >
            {LISTS_TEXT.backToWard}
          </button>
        </div>
      </header>

      {/* 지금 병실 화면이 쓰는 목록 (내 목록이면 노란 띠, 기본 목록이면 파란 띠) */}
      <div className={`lists-using${settings.commandList ? ' is-custom' : ''}`}>
        <p data-using-list={settings.commandList || 'default'} className="lists-using-text">
          {LISTS_TEXT.usingNow(usingName)}{settings.commandList ? ` · ${LISTS_TEXT.untested}` : ''}
        </p>
        {settings.commandList && (
          <button type="button" onClick={applyDefault} className="lists-btn lists-btn-outline">
            {LISTS_TEXT.backToDefault}
          </button>
        )}
      </div>

      {notice && (
        <p
          role="status"
          className={`lists-notice ${notice.tone === 'ok' ? 'is-ok' : 'is-warn'}`}
        >
          {notice.text}
        </p>
      )}

      <main className="lists-layout">
        <aside aria-label="목록 고르기" className="lists-sidebar">
          <button
            type="button"
            aria-current={selected === '' ? 'true' : undefined}
            onClick={() => choose('')}
            className="lists-card"
          >
            <span className="lists-card-top">
              <span className="lists-card-title">{LISTS_TEXT.defaultName} ({DEFAULT_ROWS.length}개)</span>
              {!settings.commandList && <span className="lists-chip lists-chip-ok">{LISTS_TEXT.inUse}</span>}
            </span>
            <span className="lists-card-meta">바꿀 수 없음</span>
          </button>
          <h2 className="lists-sidebar-title">{LISTS_TEXT.myLists} ({lists.items.length}/{MAX_LISTS})</h2>
          {lists.status === 'loading' && <p className="lists-help">불러오는 중이에요</p>}
          {lists.status === 'error' && (
            <div role="alert" className="lists-alert">
              <p className="lists-alert-title">{lists.message}</p>
              <p>{LISTS_TEXT.listsNeedReload}</p>
              <button type="button" onClick={refreshLists} className="lists-btn lists-btn-outline">
                {LISTS_TEXT.reloadLists}
              </button>
            </div>
          )}
          {lists.status === 'ready' && lists.items.length === 0 && <p className="lists-empty">{LISTS_TEXT.noLists}</p>}
          <ul className="lists-cards">
            {lists.items.map((l) => (
              <li key={l.id}>
                <button
                  type="button"
                  data-list-id={l.id}
                  aria-current={selected === l.id ? 'true' : undefined}
                  onClick={() => choose(l.id)}
                  className="lists-card"
                >
                  <span className="lists-card-top">
                    <span className="lists-card-title">{l.title}</span>
                    {settings.commandList === l.id && <span className="lists-chip lists-chip-ok">{LISTS_TEXT.inUse}</span>}
                  </span>
                  <span className="lists-card-meta">
                    {l.id}{l.count !== null ? ` · ${l.count}개` : ''}{l.emergencyCount !== null ? ` · 긴급 ${l.emergencyCount}` : ''}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <button
            type="button"
            onClick={() => choose(NEW_KEY)}
            disabled={lists.status !== 'ready' || lists.items.length >= MAX_LISTS}
            className="lists-btn lists-btn-new"
          >
            + {LISTS_TEXT.newList}
          </button>
          {lists.items.length >= MAX_LISTS && <p className="lists-help">{LISTS_TEXT.tooMany}</p>}
          {EXAMPLES.map((ex) => (
            <button
              key={ex.title}
              type="button"
              data-example={ex.title}
              onClick={() => loadExample(ex)}
              disabled={busy || lists.status !== 'ready' || lists.items.length >= MAX_LISTS}
              className="lists-btn lists-btn-example"
            >
              {LISTS_TEXT.loadExample} ({ex.title})
            </button>
          ))}
        </aside>

        {!editing && <DefaultListView inUse={!settings.commandList} onUseDefault={applyDefault} />}

        {editing && loadingList && <p className="lists-loading"><span className="lists-spinner" aria-hidden="true" />{LISTS_TEXT.loading}</p>}

        {editing && !loadingList && (
          <section aria-label="목록 고치기" className="lists-editor">
            <div className="lists-panel">
              <div className="lists-meta-fields">
                <label className="lists-field lists-field-title">
                  <span className="lists-field-name">{LISTS_TEXT.listTitle}</span>
                  <input
                    type="text"
                    lang="ko"
                    value={editor.title}
                    maxLength={TITLE_MAX_CHARS + 10}
                    disabled={busy}
                    onChange={(e) => dispatch({ type: 'SET_TITLE', value: e.target.value })}
                    className={`lists-input lists-input-title${titleProblem ? ' has-error' : ''}`}
                  />
                  {titleProblem && <span className="lists-field-error">{titleProblem}</span>}
                </label>
                <label className="lists-field lists-field-id">
                  <span className="lists-field-name">{LISTS_TEXT.listId}</span>
                  {editor.mode === 'new' ? (
                    <input
                      type="text"
                      value={editor.id}
                      maxLength={32}
                      autoComplete="off"
                      spellCheck={false}
                      disabled={busy}
                      onChange={(e) => dispatch({ type: 'SET_ID', value: e.target.value })}
                      className={`lists-input${idProblem ? ' has-error' : ''}`}
                    />
                  ) : (
                    <span className="lists-id-value">{editor.savedId}</span>
                  )}
                  {editor.mode === 'new' && <span className="lists-field-hint">{LISTS_TEXT.listIdHint}</span>}
                  {idProblem && editor.id && <span className="lists-field-error">{idProblem}</span>}
                </label>
                {usingThis && (
                  <span className="lists-chip lists-chip-ok lists-chip-lg">
                    <CheckIcon /> {LISTS_TEXT.usingThis}
                  </span>
                )}
              </div>

              {checkLine && (
                <p
                  data-check-status={checkLine.tone}
                  aria-live="polite"
                  className={`lists-check is-${checkLine.tone}`}
                >
                  {checkLine.tone === 'ok' ? <CheckIcon /> : (checkLine.tone === 'neutral' ? null : <AlertIcon />)} {checkLine.text}
                </p>
              )}
              {(msgs.global.errors.length > 0 || msgs.global.warnings.length > 0) && (
                <ul className="lists-issues" aria-label="목록 전체 안내">
                  {msgs.global.errors.map((m, i) => <IssueLine key={`ge${i}`} issue={m} />)}
                  {msgs.global.warnings.map((m, i) => <IssueLine key={`gw${i}`} issue={m} />)}
                </ul>
              )}
            </div>

            <div className="lists-rows-head">
              <div className="lists-columns">
                <span>#</span>
                <span>{LISTS_TEXT.colText}</span>
                <span>{LISTS_TEXT.colDisplay}</span>
              </div>
              <p className="lists-help">
                {LISTS_TEXT.colTextHint} · {LISTS_TEXT.colDisplayHint} · 취소: {LISTS_TEXT.colCancelHint}
              </p>
            </div>
            <ol className="lists-rows">
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

            <div className="lists-actions">
              <button
                type="button"
                onClick={() => dispatch({ type: 'ADD_ROW' })}
                disabled={busy || editor.rows.length >= MAX_COMMANDS}
                className="lists-btn lists-btn-outline"
              >
                + {LISTS_TEXT.addRow} ({editor.rows.length}/{MAX_COMMANDS})
              </button>
              <button
                type="button"
                data-save
                onClick={save}
                disabled={!saveState.ok || (editor.mode === 'new' && lists.status !== 'ready')}
                className="lists-btn lists-btn-primary lists-btn-save"
              >
                {editor.saving ? LISTS_TEXT.saving : LISTS_TEXT.save}
              </button>
              {editor.mode === 'edit' && (
                <button type="button" onClick={remove} disabled={busy} className="lists-btn lists-btn-danger">
                  {LISTS_TEXT.deleteList}
                </button>
              )}
            </div>
            {saveState.reason && editor.dirty && <p className="lists-error-text">{saveState.reason}</p>}
            {editor.saveMessage && editor.saveMessage !== LISTS_TEXT.saved && (
              <p role="alert" className="lists-error-text">{editor.saveMessage}</p>
            )}
            {conflict && editor.mode === 'edit' && (
              <button type="button" data-reload-list onClick={reloadCurrent} disabled={busy} className="lists-btn lists-btn-outline lists-btn-start">
                {LISTS_TEXT.reloadList}
              </button>
            )}
            {reenroll && (
              <div role="status" data-reenroll={reenroll.kind} className="lists-reenroll">
                <p className="lists-reenroll-title"><AlertIcon /> {reenroll.text}</p>
                <p className="lists-reenroll-howto">{LISTS_TEXT.reenrollHowTo}</p>
              </div>
            )}

            <div className="lists-use-panel">
              {usingThis ? (
                <p className="lists-use-status"><CheckIcon /> {LISTS_TEXT.usingThis} · {LISTS_TEXT.untested}</p>
              ) : (
                <button
                  type="button"
                  data-use-list
                  onClick={applyThisList}
                  disabled={!canUse}
                  className="lists-btn lists-btn-primary lists-btn-lg"
                >
                  {LISTS_TEXT.useThis}
                </button>
              )}
              {!canUse && !usingThis && <p className="lists-help">{LISTS_TEXT.useNeedsSave}</p>}
              {(usingThis || justUsed) && (
                <div className="lists-use-actions">
                  <button type="button" onClick={() => leaveTo('#enroll')} className="lists-btn lists-btn-primary">
                    {LISTS_TEXT.enrollThis}
                  </button>
                  <button type="button" onClick={() => leaveTo('#ward')} className="lists-btn lists-btn-outline">
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
