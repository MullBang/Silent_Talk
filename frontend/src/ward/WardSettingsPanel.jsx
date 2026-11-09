// WardSettingsPanel.jsx — 환자 화면 설정 (녹화 길이·확인 시간·병실 번호·음성 읽기·밤 무음). 보호자·간호사가 마우스로 조작한다.
// 전체를 data-no-switch로 덮어 설정 조작이 스위치 누름(녹화 시작)으로 잡히지 않게 한다.

import React, { useCallback, useEffect, useRef, useState } from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import {
  CONFIRM_SECONDS_MAX, CONFIRM_SECONDS_MIN, NIGHT_END_HOUR, NIGHT_START_HOUR, RECORD_SECONDS_OPTIONS,
} from './wardSettings';
import { voiceLabel } from './wardSpeech';
import { WARD_COLORS } from './wardTheme';

const CONFIRM_OPTIONS = Array.from({ length: CONFIRM_SECONDS_MAX - CONFIRM_SECONDS_MIN + 1 }, (_, i) => CONFIRM_SECONDS_MIN + i);

/**
 * 설정 창.
 * @param {object} props
 * @param {{recordSeconds:number, confirmSeconds:number, room:string, tts:boolean, nightMute:boolean}} props.settings
 * @param {(patch: object) => void} props.onChange
 * @param {() => void} props.onClose
 * @param {object|null} props.voice 오프라인 한국어 음성(없으면 null).
 * @returns {JSX.Element}
 */
export default function WardSettingsPanel({ settings, onChange, onClose, voice }) {
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

  return (
    <div
      data-no-switch
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(17, 24, 39, 0.6)' }}
      onPointerDown={(e) => { if (e.target === e.currentTarget) close(); }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="환자 화면 설정"
        className="w-full max-w-lg rounded-2xl p-5 shadow-xl"
        style={{ background: WARD_COLORS.paper, color: WARD_COLORS.ink }}
        onClick={blurAfterPointerClick}
      >
        <h2 className="mb-4 text-2xl font-bold">설정</h2>

        <fieldset className="mb-4">
          <legend className="mb-1 font-semibold">녹화 길이</legend>
          <div className="flex flex-wrap gap-2">
            {RECORD_SECONDS_OPTIONS.map((sec) => (
              <button
                key={sec}
                type="button"
                aria-pressed={settings.recordSeconds === sec}
                onClick={() => onChange({ recordSeconds: sec })}
                className="rounded-lg border-2 px-4 py-2 text-lg font-semibold"
                style={settings.recordSeconds === sec
                  ? { background: WARD_COLORS.infoBg, color: WARD_COLORS.infoInk, borderColor: WARD_COLORS.infoBg }
                  : { background: WARD_COLORS.paper, color: WARD_COLORS.ink, borderColor: WARD_COLORS.line }}
              >
                {sec}초
              </button>
            ))}
          </div>
        </fieldset>

        <label className="mb-4 block">
          <span className="mb-1 block font-semibold">보내기 전 확인 시간 (스위치 = 취소)</span>
          <select
            value={settings.confirmSeconds}
            onChange={(e) => onChange({ confirmSeconds: Number(e.target.value) })}
            className="rounded-lg border-2 px-3 py-2 text-lg"
            style={{ borderColor: WARD_COLORS.line }}
          >
            {CONFIRM_OPTIONS.map((sec) => <option key={sec} value={sec}>{sec}초</option>)}
          </select>
        </label>

        <label className="mb-4 block">
          <span className="mb-1 block font-semibold">병실 번호</span>
          <input
            type="text"
            value={roomDraft}
            maxLength={10}
            onChange={(e) => setRoomDraft(e.target.value)}
            onBlur={commitRoom}
            onKeyDown={(e) => { if (e.key === 'Enter') commitRoom(); }}
            className="w-40 rounded-lg border-2 px-3 py-2 text-lg"
            style={{ borderColor: WARD_COLORS.line }}
          />
        </label>

        <label className="mb-2 flex items-center gap-2 text-lg">
          <input type="checkbox" checked={settings.tts} onChange={(e) => onChange({ tts: e.target.checked })} className="h-5 w-5" />
          음성 읽기 (다인실 사생활을 위해 기본 꺼짐)
        </label>
        <p className="mb-3 pl-7 text-sm" style={{ color: WARD_COLORS.inkSoft }}>
          {voiceLabel(voice)} · 오프라인 한국어 음성만 씁니다
        </p>

        <label className="mb-4 flex items-center gap-2 text-lg">
          <input type="checkbox" checked={settings.nightMute} onChange={(e) => onChange({ nightMute: e.target.checked })} className="h-5 w-5" />
          밤 무음 ({NIGHT_START_HOUR}시~{NIGHT_END_HOUR}시 음성 읽기 끔)
        </label>

        <button
          type="button"
          onClick={close}
          className="w-full rounded-xl py-3 text-lg font-bold"
          style={{ background: WARD_COLORS.ink, color: WARD_COLORS.paper }}
        >
          닫기
        </button>
      </div>
    </div>
  );
}
