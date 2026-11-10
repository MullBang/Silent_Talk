// WardSettingsPanel.jsx — 환자 화면 설정 (입력 방식·녹화 길이·확인 시간·병실 번호·음성 읽기·밤 무음·명령 목록·내 입 모양 등록). 보호자·간호사가 마우스로 조작한다.
// 전체를 data-no-switch로 덮어 설정 조작이 스위치 누름(녹화 시작)으로 잡히지 않게 한다.
// 명령 목록은 기본 목록(측정한 목록) 또는 서버에 저장한 내 목록이다. 등록은 목록마다 따로라서, 등록 고르기는 지금 쓰는 목록의 등록만 보인다.
// 모양은 settings-ui.css (흰 창 + 묶음 카드, 좁은 화면에서는 아래에서 올라오는 시트).

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getWardLists, wardProfileApi } from '../api/client';
import { errorMessageFrom } from '../api/pollOutcome';
import { blurAfterPointerClick } from '../record/pointerFocus';
import COMMANDS from './commands.json';
import { buildCommandIndex } from './wardCommands';
import { ENROLL_TEXT, enrollStatusLine, parseProfileInfo, profileChoices } from './wardEnroll';
import { LISTS_TEXT, parseListSummaries } from './wardLists';
import {
  CONFIRM_SECONDS_MAX, CONFIRM_SECONDS_MIN, NIGHT_END_HOUR, NIGHT_START_HOUR, RECORD_SECONDS_OPTIONS, enrollProfileFor, enrollProfilePatch,
} from './wardSettings';
import { voiceLabel } from './wardSpeech';
import './settings-ui.css';

const CONFIRM_OPTIONS = Array.from({ length: CONFIRM_SECONDS_MAX - CONFIRM_SECONDS_MIN + 1 }, (_, i) => CONFIRM_SECONDS_MIN + i);
const INDEX = buildCommandIndex(COMMANDS);
const LIST_ERROR = '서버에서 등록 목록을 가져오지 못했어요';

/**
 * 명령 목록 고르기 (기본 목록 / 내 목록들) + 내 목록 관리 화면 열기.
 * @param {object} props
 * @param {string} props.current 설정의 명령 목록(''이면 기본 목록).
 * @param {(listId: string) => void} props.onSelect
 * @param {boolean} props.practice 연습 모드(기본 목록만 씀).
 * @param {{state:string, title?:string, count?:number, message?:string}|null} props.status 병실 화면의 목록 상태.
 * @param {string|null} [props.switchBlocked] 지금 목록을 바꿀 수 없는 이유.
 * @param {() => void} [props.onOpenLists] 내 목록 관리 화면 열기.
 * @param {string|null} [props.listsBlocked] 지금 목록 관리 화면을 열 수 없는 이유.
 * @returns {JSX.Element}
 */
function CommandListSection({ current, onSelect, practice, status, switchBlocked = null, onOpenLists, listsBlocked = null }) {
  const [lists, setLists] = useState([]);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    let alive = true;
    getWardLists()
      .then((data) => { if (alive) { setLists(parseListSummaries(data)); setLoadError(null); } })
      .catch((err) => { if (alive) { setLists([]); setLoadError(errorMessageFrom(err, LISTS_TEXT.loadListsFailed)); } });
    return () => { alive = false; };
  }, []);

  const known = lists.some((l) => l.id === current);
  let line = null;
  if (practice) {
    line = { tone: 'warn', text: `${LISTS_TEXT.practiceDefaultOnly}. 여기서 고른 목록은 연습 모드 밖 병실 화면에서 써요` };
  } else if (status?.state === 'ready') {
    line = { tone: 'warn', text: `내 목록 ‘${status.title}’ ${status.count}개 · ${LISTS_TEXT.untested}` };
  } else if (status?.state === 'loading') {
    line = { tone: 'neutral', text: LISTS_TEXT.loading };
  } else if (status?.state === 'error') {
    line = { tone: 'warn', text: `${status.message || LISTS_TEXT.loadFailed} — 입 읽기 녹화를 쓸 수 없어요` };
  }

  return (
    <fieldset className="settings-card">
      <legend className="settings-card-title">명령 목록</legend>
      <label className="settings-field">
        <span className="settings-sublabel">병실 화면이 쓸 문장 목록</span>
        <select
          data-command-list-select
          value={current}
          disabled={!!switchBlocked}
          onChange={(e) => onSelect(e.target.value)}
          className="settings-select"
        >
          <option value="">{LISTS_TEXT.defaultName} ({INDEX.list.length}개)</option>
          {lists.map((l) => (
            <option key={l.id} value={l.id}>
              내 목록: {l.title}{l.count !== null ? ` (${l.count}개)` : ''}
            </option>
          ))}
          {current && !known && <option value={current}>내 목록: {current} (목록에서 찾지 못함)</option>}
        </select>
      </label>
      {line && (
        <p
          data-list-status={practice ? 'practice' : status?.state}
          className={`settings-status ${line.tone === 'warn' ? 'is-warn' : 'is-neutral'}`}
        >
          {line.text}
        </p>
      )}
      {loadError && <p className="settings-help">{loadError}</p>}
      {switchBlocked && <p className="settings-status is-warn">{switchBlocked}</p>}
      {listsBlocked && !switchBlocked && (
        <p className="settings-status is-warn">{listsBlocked}</p>
      )}
      <div className="settings-actions">
        <button
          type="button"
          onClick={() => onOpenLists?.()}
          disabled={!!listsBlocked}
          className="settings-btn settings-btn-outline"
        >
          내 목록 관리
        </button>
      </div>
    </fieldset>
  );
}

/**
 * 내 입 모양 등록 고르기·상태·지우기 (지금 쓰는 목록의 등록만).
 * @param {object} props
 * @param {string} props.current 이 목록의 등록 이름(''이면 쓰지 않음).
 * @param {(profile: string) => void} props.onSelect
 * @param {(profile: string) => void} [props.onOpenEnroll] 등록 화면 열기.
 * @param {string|null} [props.blocked] 지금 등록 화면을 열 수 없는 이유(없으면 null).
 * @param {string} [props.listId] 등록 이름공간(''이면 기본 목록).
 * @param {string} [props.listName] 화면에 보일 목록 이름.
 * @param {object} [props.index] 이 목록의 명령 색인(개수 표시). 명령이 없으면(불러오기 전·실패) 개수를 보이지 않는다.
 * @returns {JSX.Element}
 */
function EnrollSection({ current, onSelect, onOpenEnroll, blocked = null, listId = '', listName = '', index = INDEX }) {
  const api = useMemo(() => wardProfileApi(listId), [listId]);
  const total = index?.list?.length || 0;
  const [choices, setChoices] = useState(() => profileChoices(null, current));
  const [listError, setListError] = useState(null);
  const [info, setInfo] = useState(null);
  const [infoError, setInfoError] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let alive = true;
    api.getWardProfiles()
      .then((data) => { if (alive) { setChoices(profileChoices(data, current)); setListError(null); } })
      .catch(() => { if (alive) { setChoices(profileChoices(null, current)); setListError(LIST_ERROR); } });
    return () => { alive = false; };
  }, [api, current, reload]);

  useEffect(() => {
    let alive = true;
    setInfo(null);
    setInfoError(false);
    if (!current || !total) return undefined;
    api.getWardProfile(current)
      .then((data) => { if (alive) setInfo(parseProfileInfo(data, index, current)); })
      .catch(() => { if (alive) setInfoError(true); });
    return () => { alive = false; };
  }, [api, current, reload, index, total]);

  const remove = useCallback(async () => {
    if (!current || busy) return;
    if (!globalThis.confirm?.(ENROLL_TEXT.deleteConfirm(current))) return;
    setBusy(true);
    setMessage(null);
    try {
      await api.deleteWardProfile(current);
      setMessage(`등록 “${current}”를 지웠어요`);
      onSelect('');
    } catch (err) {
      setMessage(errorMessageFrom(err, '등록을 지우지 못했어요'));
    } finally {
      setBusy(false);
      setReload((n) => n + 1);
    }
  }, [api, busy, current, onSelect]);

  const line = enrollStatusLine(current, info, { loadError: infoError });
  return (
    <fieldset className="settings-card">
      <legend className="settings-card-title">{ENROLL_TEXT.title}{listName ? ` · ${listName}` : ''}</legend>
      <label className="settings-field">
        <span className="settings-sublabel">병실 화면이 비교할 등록</span>
        <select
          value={current}
          onChange={(e) => { setMessage(null); onSelect(e.target.value); }}
          className="settings-select"
        >
          <option value="">등록 안 씀</option>
          {choices.map((c) => (
            <option key={c.profile} value={c.profile}>
              {c.profile}{c.count !== null ? (total ? ` (${c.count}/${total})` : ` (${c.count}개)`) : ''}
            </option>
          ))}
        </select>
      </label>
      {line && (
        <p
          data-enroll-status={line.tone}
          className={`settings-status is-${line.tone === 'warn' || line.tone === 'ok' ? line.tone : 'neutral'}`}
        >
          {line.text}
        </p>
      )}
      {listError && <p className="settings-help">{listError}</p>}
      {message && <p role="status" className="settings-message">{message}</p>}
      {blocked && (
        <p data-enroll-blocked className="settings-status is-warn">
          {blocked}
        </p>
      )}
      <div className="settings-actions">
        <button
          type="button"
          onClick={() => onOpenEnroll?.(current)}
          disabled={!!blocked}
          className="settings-btn settings-btn-primary"
        >
          {current ? '등록 화면 열기 (이어 찍기·다시 찍기)' : '새로 등록하기'}
        </button>
        {current && (
          <button
            type="button"
            onClick={remove}
            disabled={busy}
            className="settings-btn settings-btn-danger"
          >
            이 등록 지우기
          </button>
        )}
      </div>
    </fieldset>
  );
}

/**
 * 설정 창.
 * @param {object} props
 * @param {{recordSeconds:number, confirmSeconds:number, room:string, tts:boolean, nightMute:boolean, commandList:string, inputMode:'switch'|'direct'}} props.settings
 * @param {(patch: object) => void} props.onChange
 * @param {() => void} props.onClose
 * @param {object|null} props.voice 오프라인 한국어 음성(없으면 null).
 * @param {(profile: string) => void} [props.onOpenEnroll] 내 입 모양 등록 화면 열기.
 * @param {string|null} [props.enrollBlocked] 지금 등록 화면을 열 수 없는 이유(진행 중인 요청 등, 없으면 null).
 * @param {string} [props.activeList] 병실 화면이 지금 쓰는 목록(''이면 기본 목록, 연습 모드는 늘 '').
 * @param {boolean} [props.practice] 연습 모드.
 * @param {object} [props.index] 지금 쓰는 목록의 명령 색인.
 * @param {object|null} [props.listStatus] 내 목록 상태 {state:'loading'|'ready'|'error', title, count, message}.
 * @param {string|null} [props.listSwitchBlocked] 지금 명령 목록을 바꿀 수 없는 이유.
 * @param {() => void} [props.onOpenLists] 내 목록 관리 화면 열기.
 * @param {string|null} [props.listsBlocked] 지금 목록 관리 화면을 열 수 없는 이유.
 * @param {boolean} [props.inputModeBlocked] 녹화·분석 중이라 입력 방식을 바꿀 수 없음 (환자 화면 토글과 같은 조건).
 * @returns {JSX.Element}
 */
export default function WardSettingsPanel({
  settings, onChange, onClose, voice, onOpenEnroll, enrollBlocked = null, activeList = '', practice = false, index = INDEX,
  listStatus = null, listSwitchBlocked = null, onOpenLists, listsBlocked = null, inputModeBlocked = false,
}) {
  // 병실 번호는 입력 중에 기본값으로 되돌아가지 않도록 초안으로 두고, 칸을 벗어나거나 닫을 때 반영한다.
  const [roomDraft, setRoomDraft] = useState(settings.room);
  const draftRef = useRef(roomDraft);
  draftRef.current = roomDraft;
  const commitRoom = useCallback(() => {
    if (draftRef.current !== settings.room) onChange({ room: draftRef.current });
  }, [onChange, settings.room]);
  const close = useCallback(() => {
    commitRoom();
    onClose();
  }, [commitRoom, onClose]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close]);

  const direct = settings.inputMode === 'direct';

  return (
    <div
      data-no-switch
      className="settings-overlay"
      onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="환자 화면 설정"
        className="settings-dialog"
        onClick={blurAfterPointerClick}
      >
        <div className="settings-header">
          <h2 className="settings-title">설정</h2>
          <p className="settings-note">변경사항은 자동 저장됩니다. 병실 번호는 입력을 마치거나 창을 닫을 때 저장합니다.</p>
        </div>

        <div className="settings-body">
          <section className="settings-card">
            <h3 className="settings-card-title">입력과 녹화</h3>

            <fieldset className="settings-field">
              <legend className="settings-label">입력 방식</legend>
              <div className="settings-segment">
                <button
                  type="button"
                  aria-pressed={!direct}
                  disabled={inputModeBlocked}
                  onClick={() => onChange({ inputMode: 'switch' })}
                  className="settings-segment-button"
                >
                  스위치
                </button>
                <button
                  type="button"
                  aria-pressed={direct}
                  disabled={inputModeBlocked}
                  onClick={() => onChange({ inputMode: 'direct' })}
                  className="settings-segment-button"
                >
                  직접 선택
                </button>
              </div>
              <p className="settings-help">스위치: 아무 키·화면 누름으로 고릅니다 · 직접 선택: 고를 항목을 바로 누릅니다 (키 3번은 두 방식 모두 긴급 호출)</p>
              {inputModeBlocked && <p className="settings-help">녹화·분석 중에는 입력 방식을 바꿀 수 없어요</p>}
            </fieldset>

            <fieldset className="settings-field">
              <legend className="settings-label">녹화 길이</legend>
              <div className="settings-segment">
                {RECORD_SECONDS_OPTIONS.map((sec) => (
                  <button
                    key={sec}
                    type="button"
                    aria-pressed={settings.recordSeconds === sec}
                    onClick={() => onChange({ recordSeconds: sec })}
                    className="settings-segment-button"
                  >
                    {sec}초
                  </button>
                ))}
              </div>
            </fieldset>

            <label className="settings-field">
              <span className="settings-label">보내기 전 확인 시간 (스위치 = 취소)</span>
              <select
                value={settings.confirmSeconds}
                onChange={(e) => onChange({ confirmSeconds: Number(e.target.value) })}
                className="settings-select settings-select-short"
              >
                {CONFIRM_OPTIONS.map((sec) => <option key={sec} value={sec}>{sec}초</option>)}
              </select>
            </label>
          </section>

          <section className="settings-card">
            <h3 className="settings-card-title">병실과 음성</h3>

            <label className="settings-field">
              <span className="settings-label">병실 번호</span>
              <input
                type="text"
                value={roomDraft}
                maxLength={10}
                onChange={(e) => setRoomDraft(e.target.value)}
                onBlur={commitRoom}
                onKeyDown={(e) => { if (e.key === 'Enter') commitRoom(); }}
                className="settings-input settings-input-short"
              />
            </label>

            <div className="settings-check-group">
              <label className="settings-check">
                <input type="checkbox" checked={settings.tts} onChange={(e) => onChange({ tts: e.target.checked })} className="settings-check-input" />
                <span>음성 읽기 (다인실 사생활을 위해 기본 꺼짐)</span>
              </label>
              <p className="settings-help settings-check-help">
                {voiceLabel(voice)} · 오프라인 한국어 음성만 씁니다
              </p>
            </div>

            <label className="settings-check">
              <input type="checkbox" checked={settings.nightMute} onChange={(e) => onChange({ nightMute: e.target.checked })} className="settings-check-input" />
              <span>밤 무음 ({NIGHT_START_HOUR}시~{NIGHT_END_HOUR}시 음성 읽기 끔)</span>
            </label>
          </section>

          <CommandListSection
            current={settings.commandList || ''}
            onSelect={(listId) => onChange({ commandList: listId })}
            practice={practice}
            status={practice ? null : listStatus}
            switchBlocked={listSwitchBlocked}
            onOpenLists={() => { commitRoom(); onOpenLists?.(); }}
            listsBlocked={listsBlocked}
          />

          <EnrollSection
            key={activeList || '(default)'}
            current={enrollProfileFor(settings, activeList)}
            onSelect={(profile) => onChange(enrollProfilePatch(settings, activeList, profile))}
            onOpenEnroll={(profile) => { commitRoom(); onOpenEnroll?.(profile); }}
            blocked={enrollBlocked}
            listId={activeList}
            listName={activeList ? `내 목록 ‘${listStatus?.title || activeList}’` : LISTS_TEXT.defaultName}
            index={index}
          />
        </div>

        <div className="settings-footer">
          <button
            type="button"
            onClick={close}
            className="settings-btn settings-btn-primary settings-close"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}
