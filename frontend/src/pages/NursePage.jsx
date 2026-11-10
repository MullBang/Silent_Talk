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
  loadNursePrefs, noteRoomState, pruneRooms, saveNurseCards, saveNursePrefs, sortNewestFirst, upsertNurseCard,
  NURSE_STORAGE_KEY,
} from '../ward/wardStore';
import { WARD_COLORS } from '../ward/wardTheme';

/**
 * 카드 색 (색만으로 구분하지 않고 글자 표시와 아이콘을 함께 쓴다).
 * @param {object} card
 * @param {boolean} overdue
 * @returns {object}
 */
function cardStyle(card, overdue) {
  const ring = overdue ? { boxShadow: `0 0 0 6px ${WARD_COLORS.overdueBg}, 0 0 0 9px ${WARD_COLORS.ink}` } : {};
  if (card.acked) return { background: WARD_COLORS.paper, color: WARD_COLORS.inkSoft, borderColor: WARD_COLORS.line };
  if (card.kind === 'emergency' || card.kind === 'emergency_call') {
    return { background: WARD_COLORS.emergencyBg, color: WARD_COLORS.emergencyInk, borderColor: WARD_COLORS.emergencyBg, ...ring };
  }
  if (card.kind === 'emergency_possible') {
    return { background: WARD_COLORS.possibleBg, color: WARD_COLORS.possibleInk, borderColor: WARD_COLORS.emergencyBg, ...ring };
  }
  if (card.kind === 'check') {
    return { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.line, borderStyle: 'dashed', ...ring };
  }
  return { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.ink, ...ring };
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
  const beeperRef = useRef(null);
  if (!beeperRef.current) beeperRef.current = createBeeper();
  const channelRef = useRef(null);

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

  const ack = useCallback((cardId) => {
    const r = ackNurseCard(cardsRef.current, cardId, Date.now());
    if (!r.ackedIds.length) return;
    commit(r.cards);
    channelRef.current?.post('CARD_ACK', { cardIds: r.ackedIds, room: r.room });
  }, [commit]);

  const clearDone = useCallback(() => {
    if (!globalThis.confirm?.('확인한 카드를 이 화면 기록에서 지울까요?')) return;
    commit(clearAcked(cardsRef.current));
  }, [commit]);

  const setForward = useCallback((value) => {
    const next = { ...prefsRef.current, forwardSwitch: value };
    setPrefs(next);
    saveNursePrefs(storage, next);
  }, [storage]);

  const sorted = sortNewestFirst(cards);
  const openAlarm = hasOpenAlarm(cards);
  const roomList = Object.entries(rooms);

  return (
    <div className="min-h-screen" style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}>
      <header data-no-switch className="flex flex-wrap items-center gap-3 px-4 py-3" style={{ background: WARD_COLORS.bannerBg, color: WARD_COLORS.paper }} onClick={blurAfterPointerClick}>
        <h1 className="text-2xl font-bold">간호사실 <span className="text-base font-semibold" style={{ color: WARD_COLORS.bannerInk }}>(시연용 모의 화면 · 실제 환자 미검증)</span></h1>
        <button
          type="button"
          onClick={toggleSound}
          aria-pressed={soundOn}
          className="rounded-lg px-4 py-2 text-lg font-bold"
          style={soundOn
            ? { background: WARD_COLORS.okBg, color: WARD_COLORS.okInk }
            : { background: WARD_COLORS.bannerInk, color: WARD_COLORS.ink }}
        >
          {soundOn ? '소리 켜짐 (끄기)' : '소리 켜기'}
        </button>
        <label className="flex items-center gap-2 text-base">
          <input type="checkbox" checked={prefs.forwardSwitch} onChange={(e) => setForward(e.target.checked)} className="h-5 w-5" />
          이 창의 스위치(키) 입력을 환자 화면으로 넘기기
        </label>
        <button
          type="button"
          onClick={clearDone}
          className="ml-auto rounded-lg border-2 px-3 py-1 text-base font-semibold"
          style={{ borderColor: WARD_COLORS.paper, color: WARD_COLORS.paper }}
        >
          확인한 카드 지우기
        </button>
      </header>

      {!soundOn && (
        <p role="alert" className="px-4 py-2 text-lg font-bold" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
          소리가 꺼져 있습니다. [소리 켜기]를 한 번 눌러 주세요{openAlarm ? ' — 확인하지 않은 긴급 카드가 있습니다' : ''}.
        </p>
      )}
      {!channelOk && (
        <p role="alert" className="px-4 py-2 font-semibold" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
          이 브라우저는 환자 화면과 연결할 수 없습니다 (BroadcastChannel 없음)
        </p>
      )}

      {roomList.length > 0 && (
        <section aria-label="병실 상태" className="px-4 pt-3 text-base" style={{ color: WARD_COLORS.inkSoft }}>
          {roomList.map(([key, info]) => (
            <p key={key}>
              <strong style={{ color: WARD_COLORS.ink }}>{info.room}호</strong> 환자 화면: {PHASE_LABELS[info.phase] || '알 수 없음'} · 가상 병실: {roomSummary(info.state)}
            </p>
          ))}
        </section>
      )}

      <main className="p-4">
        {sorted.length === 0 ? (
          <p className="text-xl" style={{ color: WARD_COLORS.inkSoft }}>아직 받은 요청이 없습니다. 환자 화면(#ward)을 같은 주소(localhost:5173)로 열어 두세요.</p>
        ) : (
          <ul className="flex flex-col gap-3">
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
                <li key={card.id} data-card-id={card.id} data-kind={card.kind} className="rounded-xl border-4 px-4 py-3" style={style}>
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
                    <span className="text-2xl font-bold">{card.room}호</span>
                    {label && (
                      <span className="flex items-center gap-1 text-2xl font-black">
                        {alarm && <AlertIcon />} {label}
                      </span>
                    )}
                    {headline && <span className="text-3xl font-bold">{headline}</span>}
                    {card.candidates?.length > 0 && !card.selectedId && (
                      <span className="text-2xl font-semibold" style={{ color: estimateColor }}>
                        추정: {card.candidates.map((c) => c.text).join(' / ')}
                      </span>
                    )}
                    {card.count > 1 && <span className="text-2xl font-black">×{card.count}</span>}
                    {listNote && (
                      <span
                        data-list-tag={card.list?.id}
                        className="rounded-md border-2 px-2 py-0.5 text-base font-semibold"
                        style={{ borderColor: 'currentColor' }}
                      >
                        {listNote}
                      </span>
                    )}
                    <span className="text-xl">{hhmm(card.firstAt)}{card.count > 1 || card.lastAt !== card.firstAt ? ` · 최근 ${hhmm(card.lastAt)}` : ''}</span>
                    {alarm && !card.acked && <span className="text-lg font-bold">반복음</span>}
                    <span data-no-switch className="ml-auto" onClick={blurAfterPointerClick}>
                      {card.acked ? (
                        <span className="flex items-center gap-1 text-lg font-semibold"><CheckIcon /> 확인함 {hhmm(card.ackedAt)}</span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => ack(card.id)}
                          className="rounded-xl px-6 py-2 text-2xl font-black"
                          style={{ background: WARD_COLORS.ink, color: WARD_COLORS.paper, outline: `3px solid ${WARD_COLORS.paper}` }}
                        >
                          확인
                        </button>
                      )}
                    </span>
                  </div>
                  {card.cancelRequested && (
                    <p className="mt-2 inline-block rounded-md px-2 py-1 text-2xl font-black" style={{ background: WARD_COLORS.warnBg, color: WARD_COLORS.warnInk }}>
                      환자가 취소 요청함
                    </p>
                  )}
                  {card.hint && <p className="mt-1 text-xl font-bold">{card.hint}</p>}
                  {card.selectedId && card.candidates?.length > 0 && (
                    <p className="mt-1 text-lg" style={{ color: estimateColor }}>추정이었던 후보: {card.candidates.map((c) => c.text).join(' / ')}</p>
                  )}
                  {card.note && !card.cancelRequested && <p className="mt-1 text-lg font-semibold">{card.note}</p>}
                  {card.source === 'scan' && card.kind !== 'check' && <p className="mt-1 text-lg">환자가 스캐닝으로 직접 고름</p>}
                  {enrollNote && <p data-enroll-note={card.enroll} className="mt-1 text-lg font-semibold">{enrollNote}</p>}
                  {overdue && <p className="mt-1 text-lg font-black">60초 넘게 확인하지 않았습니다</p>}
                </li>
              );
            })}
          </ul>
        )}
      </main>
    </div>
  );
}
