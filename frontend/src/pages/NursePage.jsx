// NursePage.jsx — 병실 무음 도우미 간호사실 화면 (#nurse, 시연용 모의 화면). 실제 병원 시스템과 연결하지 않는다.
// 환자 화면(#ward)과 같은 주소의 BroadcastChannel로 카드를 받고 [확인]을 돌려준다. 카드는 localStorage에 남아 새로 고침 뒤에도 보인다.
// 긴급 카드는 [확인]까지 반복음, 일반 카드는 한 번 울린다. 브라우저 자동 재생 정책 때문에 [소리 켜기]를 한 번 눌러야 한다.
// 이 창에 들어온 스위치(키) 입력은 환자 창으로 넘긴다 (발표 중 간호사 창에 포커스가 있어도 스위치가 동작하게).
// 단, 목록을 넘겨 보는 이동 키(화살표·Page·Home·End)는 넘기지 않는다.
// 환자 화면이 '내 목록'을 쓰는 동안 온 카드에는 작은 표시 "내 목록: <이름>"을 붙인다(성능을 측정하지 않은 목록).

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import { AlertIcon, CheckIcon } from '../ward/WardIcons';
import { openWardChannel } from '../ward/wardChannel';
import { roomSummary } from '../ward/wardCommands';
import { forwardIsSwitch } from '../ward/wardInput';
import { PHASE_LABELS } from '../ward/wardMachine';
import { safeLocalStorage } from '../ward/wardSettings';
import { ALARM_REPEAT_MS, createBeeper } from '../ward/wardSound';
import {
  ackNurseCard, cardEnrollNote, cardHeadline, cardLabel, cardListNote, clearAcked, dropRoom, hasOpenAlarm, hhmm, isAlarmKind, isOverdue, loadNurseCards,
  loadNursePrefs, noteRoomState, pruneRooms, saveNurseCards, saveNursePrefs, upsertNurseCard,
  NURSE_STORAGE_KEY,
} from '../ward/wardStore';
import { WARD_COLORS } from '../ward/wardTheme';
import { NURSE_FILTERS, ackClickAllowed, nurseSummary, visibleNurseCards, waitingText } from '../ward/nursePresentation';
import '../ward/nurse-ui.css';

/**
 * 카드 색 (색만으로 구분하지 않고 글자 표시와 아이콘을 함께 쓴다).
 * @param {object} card
 * @param {boolean} overdue
 * @returns {object}
 */
function cardStyle(card, overdue) {
  const ring = overdue ? { boxShadow: `0 0 0 3px ${WARD_COLORS.overdueBg}, 0 6px 20px rgba(20,45,78,.05)` } : {};
  if (card.acked) return { background: '#f8fafc', color: WARD_COLORS.inkSoft, borderColor: '#d7dfeb' };
  if (card.kind === 'emergency' || card.kind === 'emergency_call') {
    return { background: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyInk, borderColor: WARD_COLORS.emergencyBg, ...ring };
  }
  if (card.kind === 'emergency_possible') {
    return { background: WARD_COLORS.possibleBg, color: WARD_COLORS.possibleInk, borderColor: WARD_COLORS.emergencyBg, ...ring };
  }
  if (card.kind === 'check') {
    return { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.line, borderStyle: 'dashed', ...ring };
  }
  return { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: '#d7dfeb', ...ring };
}

/**
 * 간호사실 화면.
 * @returns {JSX.Element}
 */
export default function NursePage() {
  const storage = useMemo(() => safeLocalStorage(), []);
  const [cards, setCards] = useState(() => loadNurseCards(storage));
  const cardsRef = useRef(cards);
  const [prefs, setPrefs] = useState(() => loadNursePrefs(storage));
  const prefsRef = useRef(prefs);
  prefsRef.current = prefs;
  const [soundOn, setSoundOn] = useState(false);
  const soundOnRef = useRef(false);
  soundOnRef.current = soundOn;
  const [now, setNow] = useState(() => Date.now());
  const [rooms, setRooms] = useState({});
  const [channelOk, setChannelOk] = useState(true);
  const [filter, setFilter] = useState('open');
  const [notice, setNotice] = useState(null);
  const beeperRef = useRef(null);
  if (!beeperRef.current) beeperRef.current = createBeeper();
  const channelRef = useRef(null);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(timer);
  }, [notice]);

  /** 카드 목록 바꾸기 + 저장. */
  const commit = useCallback((next) => {
    cardsRef.current = next;
    setCards(next);
    saveNurseCards(storage, next);
  }, [storage]);

  // 환자 창과 연결. 열리면 HELLO로 병실 상태·확인 안 된 카드를 다시 받는다.
  useEffect(() => {
    const ch = openWardChannel((msg) => {
      if (msg.type === 'CARD_UPSERT') {
        const r = upsertNurseCard(cardsRef.current, msg.card, Date.now());
        if (!r.alert) return;
        commit(r.cards);
        if (soundOnRef.current) beeperRef.current.play(r.alarm ? 'emergency' : 'normal');
      } else if (msg.type === 'ROOM_STATE') {
        // 환자 창마다 한 줄 (병실 번호를 바꾸면 그 창의 줄이 바뀐다).
        setRooms((prev) => noteRoomState(prev, msg, Date.now()));
      } else if (msg.type === 'ROOM_CLOSED') {
        // 환자 창을 닫거나 새로 고침하면 그 창의 줄을 바로 지운다 (새 창은 새 줄로 다시 알린다).
        setRooms((prev) => dropRoom(prev, msg.windowId));
      }
    }, { from: 'nurse' });
    channelRef.current = ch;
    setChannelOk(ch.available);
    ch.post('HELLO', {});
    return () => {
      ch.close();
      channelRef.current = null;
    };
  }, [commit]);

  // 다른 간호사 창에서 바뀐 기록을 따라간다.
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key !== NURSE_STORAGE_KEY) return;
      const next = loadNurseCards(storage);
      cardsRef.current = next;
      setCards(next);
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [storage]);

  // 1초마다: 60초 강조 갱신, 소식이 끊긴 환자 창의 병실 줄 지우기. 긴급 카드가 확인 전이면 반복음.
  useEffect(() => {
    const tick = setInterval(() => {
      const t = Date.now();
      setNow(t);
      setRooms((prev) => pruneRooms(prev, t));
    }, 1000);
    const alarm = setInterval(() => {
      if (soundOnRef.current && hasOpenAlarm(cardsRef.current)) beeperRef.current.play('emergency');
    }, ALARM_REPEAT_MS);
    return () => {
      clearInterval(tick);
      clearInterval(alarm);
    };
  }, []);

  // 이 창에 들어온 스위치 입력을 환자 창으로 넘긴다 (버튼 조작·글자 입력·목록 이동 키는 제외).
  useEffect(() => {
    const onKey = (e) => {
      if (!prefsRef.current.forwardSwitch || !forwardIsSwitch(e)) return;
      e.preventDefault();
      channelRef.current?.post('SWITCH_PRESS', { repeat: e.repeat === true });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const toggleSound = useCallback(async () => {
    if (soundOnRef.current) {
      setSoundOn(false);
      return;
    }
    const ok = await beeperRef.current.enable();
    setSoundOn(ok);
    if (ok) beeperRef.current.play('normal');
  }, []);

  // 확인한 카드는 목록에서 빠지거나 자리를 옮겨 다음 카드가 같은 자리로 온다. 두 번 누르기의 두 번째 누름이
  // 보지 못한 다음 카드(긴급일 수도 있음)를 확인하지 않게, 방금 확인한 직후의 누름은 무시한다.
  const lastAckAtRef = useRef(null);
  const ack = useCallback((cardId, event) => {
    const at = Date.now();
    if (!ackClickAllowed(lastAckAtRef.current, at, event?.detail)) return;
    const r = ackNurseCard(cardsRef.current, cardId, at);
    if (!r.ackedIds.length) return;
    lastAckAtRef.current = at;
    commit(r.cards);
    channelRef.current?.post('CARD_ACK', { cardIds: r.ackedIds, room: r.room });
    setNotice({ text: `${r.room}호 카드를 확인 완료로 이동했습니다.`, id: Date.now() });
  }, [commit]);

  const clearDone = useCallback(() => {
    if (!globalThis.confirm?.('확인한 카드를 이 화면 기록에서 지울까요?')) return;
    commit(clearAcked(cardsRef.current));
    setNotice({ text: '확인한 카드 기록을 지웠습니다.', id: Date.now() });
  }, [commit]);

  const setForward = useCallback((value) => {
    const next = { ...prefsRef.current, forwardSwitch: value };
    setPrefs(next);
    saveNursePrefs(storage, next);
  }, [storage]);

  const sorted = useMemo(() => visibleNurseCards(cards, filter), [cards, filter]);
  const summary = useMemo(() => nurseSummary(cards), [cards]);
  const openAlarm = hasOpenAlarm(cards);
  const hiddenAlarms = summary.alarms - sorted.filter((card) => !card.acked && isAlarmKind(card.kind)).length;
  const roomList = Object.entries(rooms);

  return (
    <div className="nurse-page">
      <header data-no-switch className="nurse-heading" onClick={blurAfterPointerClick}>
        <div>
          <p className="nurse-eyebrow">NURSE STATION</p>
          <h1>간호사실</h1>
          <p className="nurse-description">도착한 요청을 한눈에 확인하세요. <span className="nurse-prototype-inline">시연용 모의 화면 · 실제 환자 미검증</span></p>
        </div>
        <div className="nurse-heading-actions">
          <button type="button" onClick={toggleSound} aria-pressed={soundOn}
            className={`nurse-button nurse-sound ${soundOn ? 'is-on' : ''}`}>
            <span aria-hidden="true">♪</span> {soundOn ? '소리 켜짐 · 끄기' : '소리 켜기'}
          </button>
          <a href="#ward" target="_blank" rel="noopener noreferrer" className="nurse-button nurse-button-secondary">
            환자 화면 열기 <span aria-hidden="true">↗</span>
          </a>
        </div>
      </header>

      {!soundOn && (
        <p role={openAlarm ? 'alert' : 'status'} className={`nurse-sound-note ${openAlarm ? 'has-alarm' : ''}`}>
          {openAlarm && <AlertIcon />} 소리가 꺼져 있습니다. 위의 ‘소리 켜기’를 눌러 주세요.
          {openAlarm && <strong> 미확인 경보 카드 {summary.alarms}개가 있습니다.</strong>}
        </p>
      )}
      {!channelOk && (
        <p role="alert" className="nurse-sound-note has-alarm">
          이 브라우저는 환자 화면과 연결할 수 없습니다. 다른 최신 브라우저로 열어 주세요.
        </p>
      )}

      <section className="nurse-summary" aria-label="요청 카드 요약">
        <div className="nurse-stat"><span>미확인 카드</span><strong data-summary="open">{summary.open}<small>개</small></strong></div>
        <div className="nurse-stat nurse-stat-alarm"><span>그중 경보 카드</span><strong data-summary="alarms">{summary.alarms}<small>개</small></strong><p>긴급 · 긴급 호출 · 긴급 가능성</p></div>
        <div className="nurse-stat"><span>확인 완료</span><strong data-summary="acked">{summary.acked}<small>개</small></strong></div>
      </section>

      <main className="nurse-main">
        <div className="nurse-list-heading">
          <div><h2>요청 목록</h2><p>같은 요청은 카드 하나에 횟수로 표시합니다.</p></div>
          <div data-no-switch className="nurse-filters" role="group" aria-label="카드 목록 필터" onClick={blurAfterPointerClick}>
            {NURSE_FILTERS.map((item) => (
              <button key={item.id} type="button" data-filter={item.id} aria-pressed={filter === item.id}
                onClick={() => setFilter(item.id)} className={filter === item.id ? 'is-active' : ''}>
                {item.label} <span>{item.id === 'open' ? summary.open : item.id === 'acked' ? summary.acked : summary.total}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="nurse-feedback" role="status" aria-live="polite">{notice?.text || ''}</div>
        {hiddenAlarms > 0 && (
          <p className="nurse-hidden-alarm" data-no-switch onClick={blurAfterPointerClick}>
            <AlertIcon /> 미확인 경보 카드 {hiddenAlarms}개가 다른 목록에 있습니다.
            <button type="button" onClick={() => setFilter('open')}>미확인 보기</button>
          </p>
        )}
        {sorted.length === 0 ? (
          <section className="nurse-empty" data-no-switch aria-label="빈 요청 목록" onClick={blurAfterPointerClick}>
            <span className="nurse-empty-icon" aria-hidden="true"><CheckIcon /></span>
            <h3>{summary.total === 0 ? '아직 도착한 요청이 없어요' : filter === 'acked' ? '확인 완료한 카드가 없어요' : '미확인 요청을 모두 확인했어요'}</h3>
            <p>{summary.total === 0 ? '환자 화면에서 요청을 보내면 이곳에 카드가 나타납니다. 환자 화면은 이 화면과 같은 주소로 열어 주세요.' : filter === 'acked' ? '요청 카드의 ‘확인’을 누르면 이 목록에서 다시 볼 수 있습니다.' : '확인한 카드는 ‘확인 완료’ 목록에 보관되어 있습니다.'}</p>
            {summary.total === 0 ? (
              <div className="nurse-empty-actions">
                <a href="#ward" target="_blank" rel="noopener noreferrer" className="nurse-button">환자 화면 새 창으로 열기 ↗</a>
                <a href="#ward?practice=1" target="_blank" rel="noopener noreferrer" className="nurse-button nurse-button-secondary">연습 화면 열기</a>
              </div>
            ) : filter === 'open' && <button type="button" className="nurse-button nurse-button-secondary" onClick={() => setFilter('acked')}>확인 완료 보기</button>}
          </section>
        ) : (
          <ul className="nurse-card-list">
            {sorted.map((card) => {
              const overdue = isOverdue(card, now);
              const alarm = isAlarmKind(card.kind);
              const label = cardLabel(card);
              const headline = cardHeadline(card);
              const style = cardStyle(card, overdue);
              const estimateColor = card.acked || card.kind === 'check' ? WARD_COLORS.estimate : style.color;
              const enrollNote = cardEnrollNote(card);
              const listNote = cardListNote(card);
              return (
                <li key={card.id} data-card-id={card.id} data-kind={card.kind} data-acked={card.acked ? 'true' : 'false'} className={`nurse-card ${card.acked ? 'is-acked' : ''}`} style={style}>
                  <div className="nurse-card-top">
                    <strong className="nurse-room">{card.room}호</strong>
                    {label && (
                      <span className="nurse-kind">
                        {alarm && <AlertIcon />} {label}
                      </span>
                    )}
                    {!label && <span className="nurse-kind">일반 요청</span>}
                    {card.count > 1 && <span className="nurse-count">같은 요청 ×{card.count}</span>}
                    {listNote && <span data-list-tag={card.list?.id} className="nurse-list-tag">{listNote}</span>}
                    {alarm && !card.acked && <span className="nurse-alarm-label">{soundOn ? '반복음 켜짐' : '알림음 꺼짐'}</span>}
                  </div>
                  <div className="nurse-card-content">
                    {headline && <h3>{headline}</h3>}
                    {card.candidates?.length > 0 && !card.selectedId && (
                      <p className="nurse-candidates" style={{ color: estimateColor }}>
                        추정: {card.candidates.map((c) => c.text).join(' / ')}
                      </p>
                    )}
                    {card.cancelRequested && <p className="nurse-cancel-request"><AlertIcon /> 환자가 취소 요청함</p>}
                    {card.hint && <p className="nurse-card-hint">{card.hint}</p>}
                    {card.note && !card.cancelRequested && <p className="nurse-card-note">{card.note}</p>}
                    {enrollNote && <p data-enroll-note={card.enroll} className="nurse-card-note nurse-enroll-note">{enrollNote}</p>}
                  </div>
                  {((card.source === 'scan' && card.kind !== 'check') || (card.selectedId && card.candidates?.length > 0)) && (
                    <details className="nurse-card-details" data-no-switch onClick={blurAfterPointerClick}>
                      <summary>선택 정보 보기</summary>
                      {card.selectedId && card.candidates?.length > 0 && <p>추정이었던 후보: {card.candidates.map((c) => c.text).join(' / ')}</p>}
                      {card.source === 'scan' && card.kind !== 'check' && <p>환자가 항목을 직접 고름</p>}
                    </details>
                  )}
                  <div className="nurse-card-footer">
                    <div className="nurse-card-time">
                      <span>도착 {hhmm(card.firstAt)}{card.count > 1 || card.lastAt !== card.firstAt ? ` · 최근 ${hhmm(card.lastAt)}` : ''}</span>
                      {!card.acked && <strong className={overdue ? 'is-overdue' : ''}>{waitingText(card, now)}{overdue ? ' · 60초 넘게 미확인' : ''}</strong>}
                    </div>
                    <span data-no-switch onClick={blurAfterPointerClick}>
                      {card.acked ? (
                        <span className="nurse-acknowledged"><CheckIcon /> 확인함 {hhmm(card.ackedAt)}</span>
                      ) : (
                        <button
                          type="button"
                          onClick={(e) => ack(card.id, e)}
                          className="nurse-ack-button"
                        >
                          확인
                        </button>
                      )}
                    </span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </main>
      <details className="nurse-tools" data-no-switch onClick={blurAfterPointerClick}>
        <summary>시연 도구 <span>병실 상태 · 키 입력 전달 · 기록 정리</span></summary>
        <div className="nurse-tools-body">
          <label className="nurse-forward-option">
            <input type="checkbox" checked={prefs.forwardSwitch} onChange={(e) => setForward(e.target.checked)} />
            이 창의 스위치(키) 입력을 환자 화면으로 넘기기
          </label>
          {roomList.length > 0 && (
            <section className="nurse-room-states" aria-label="병실 상태">
              {roomList.map(([key, info]) => <p key={key}><strong>{info.room}호</strong> 환자 화면: {PHASE_LABELS[info.phase] || '알 수 없음'}<br /><span>가상 병실: {roomSummary(info.state)}</span></p>)}
            </section>
          )}
          <div className="nurse-tools-clear">
            <p>확인한 카드만 지웁니다. 미확인 카드는 유지됩니다.</p>
            <button type="button" className="nurse-button nurse-button-danger" onClick={clearDone}>확인한 카드 지우기</button>
          </div>
          <p className="nurse-prototype-note">시연용 모의 화면 · 실제 환자 미검증</p>
        </div>
      </details>
    </div>
  );
}
