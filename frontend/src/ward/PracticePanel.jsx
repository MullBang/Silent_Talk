// PracticePanel.jsx — 연습 모드(#ward?practice=1) 버튼: 카메라 없이 가짜 서버 결과를 넣어 흐름을 리허설·검증한다.
// 기본 화면에서는 보이지 않는다. 전체를 data-no-switch로 덮어 버튼 조작이 스위치 누름으로 잡히지 않게 한다.

import React from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import { PRACTICE_SCENARIOS } from './wardPractice';
import { WARD_COLORS } from './wardTheme';

/**
 * 연습 모드 패널.
 * @param {object} props
 * @param {Array<{id:string, display:string}>} props.commands 명령 목록 (고름·일반 장면에 쓸 명령 선택).
 * @param {string} props.commandId 선택한 명령 ID.
 * @param {(id: string) => void} props.onCommandChange
 * @param {(scenarioId: string) => void} props.onInject
 * @param {() => void} props.onPress 화면 스위치 누름 (포인터 대신 버튼으로).
 * @returns {JSX.Element}
 */
export default function PracticePanel({ commands, commandId, onCommandChange, onInject, onPress }) {
  return (
    <section
      data-no-switch
      aria-label="연습 모드"
      className="rounded-xl border-2 border-dashed p-3 text-sm"
      style={{ borderColor: WARD_COLORS.line }}
      onClick={blurAfterPointerClick}
    >
      <h2 className="mb-2 font-bold">연습 모드 — 가짜 분석 결과 넣기 (카메라 없이 흐름 확인)</h2>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1">
          고를 명령
          <select
            value={commandId}
            onChange={(e) => onCommandChange(e.target.value)}
            className="rounded border px-2 py-1"
            style={{ borderColor: WARD_COLORS.line }}
          >
            {commands.map((c) => <option key={c.id} value={c.id}>{c.id} {c.display}</option>)}
          </select>
        </label>
        <button
          type="button"
          onClick={onPress}
          className="rounded border-2 px-3 py-1 font-semibold"
          style={{ borderColor: WARD_COLORS.ink }}
        >
          스위치 누르기
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {PRACTICE_SCENARIOS.map((s) => (
          <button
            key={s.id}
            type="button"
            data-scenario={s.id}
            onClick={() => onInject(s.id)}
            className="rounded px-3 py-1 font-semibold"
            style={{ background: WARD_COLORS.checkBg, color: WARD_COLORS.checkInk }}
          >
            {s.label}{s.needsCommand ? ` (${commandId})` : ''}
          </button>
        ))}
      </div>
    </section>
  );
}
