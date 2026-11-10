// PracticePanel.jsx — 연습 모드(#ward?practice=1) 버튼: 카메라 없이 가짜 서버 결과를 넣어 흐름을 리허설·검증한다.
// 기본 화면에서는 보이지 않는다. 전체를 data-no-switch로 덮어 버튼 조작이 스위치 누름으로 잡히지 않게 한다.

import React, { useState } from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import { PRACTICE_SCENARIOS } from './wardPractice';

const LABELS = {
  accepted_normal: '일반 요청 성공', accepted_emergency: '긴급 요청', uncertain_emergency: '긴급 가능성',
  uncertain_normal: '후보에서 선택', enroll_normal: '개인 등록으로 선택', enroll_emergency: '개인 등록 · 긴급',
  no_phrase: '문장 결과 없음', no_speech: '입 움직임 없음', not_detected: '입 검출 실패',
};

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
  const [expanded, setExpanded] = useState(true);
  return (
    <details
      data-no-switch
      aria-label="연습 모드"
      className="patient-practice"
      open={expanded}
      onToggle={(e) => setExpanded(e.currentTarget.open)}
      onClick={blurAfterPointerClick}
    >
      <summary><span>시연 도구 <span aria-hidden="true">{expanded ? '−' : '+'}</span></span><span>가짜 분석 결과로 화면 흐름 확인 · 카메라 불필요</span></summary>
      <div className="patient-practice-body">
      <div className="patient-practice-controls">
        <label>
          시연할 명령
          <select
            value={commandId}
            onChange={(e) => onCommandChange(e.target.value)}
          >
            {commands.map((c) => <option key={c.id} value={c.id}>{c.id} {c.display}</option>)}
          </select>
        </label>
        <button
          type="button"
          onClick={onPress}
          className="patient-small-button"
        >
          스위치 누르기
        </button>
      </div>
      <div className="patient-scenarios">
        {PRACTICE_SCENARIOS.map((s) => (
          <button
            key={s.id}
            type="button"
            data-scenario={s.id}
            onClick={() => onInject(s.id)}
            title={s.label}
          >
            {LABELS[s.id] || s.label}{s.needsCommand ? ` · ${commandId}` : ''}
          </button>
        ))}
      </div>
      </div>
    </details>
  );
}
