// RoomPanel.jsx — 가상 병실 (조명·커튼·침대·TV·온도). 명령 동작으로 바뀌고, 침대는 간호사 확인 뒤에만 움직인다.

import React from 'react';

import { describeRoom } from './wardCommands';

const ICON_PATHS = {
  light: 'M9 18h6m-5 3h4M8.5 14.5a6 6 0 1 1 7 0L15 17H9z',
  curtain: 'M3 4h18M5 4v16l7-5 7 5V4M12 4v11',
  bed: 'M3 18v3m18-3v3M3 18h18v-7H3zm0-7V6m4 5V8h5v3',
  tv: 'M4 5h16v12H4zm4 15h8m-4-3v3',
  temperature: 'M10 14V5a2 2 0 0 1 4 0v9a4 4 0 1 1-4 0m2-7v10',
};

/**
 * 가상 병실 표시.
 * @param {{room: object}} props
 * @returns {JSX.Element}
 */
export default function RoomPanel({ room }) {
  const items = describeRoom(room);
  return (
    <section aria-label="가상 병실" className="patient-room-panel">
      <div className="patient-panel-heading"><h2>가상 병실</h2><p>요청에 따라 바뀌는 시연 화면 · 실제 기기 연동 없음</p></div>
      <ul className="patient-room-grid" aria-live="polite">
        {items.map((it) => (
          <li
            key={`${it.key}:${it.value}`}
            className={`patient-room-item ${it.on ? 'is-on' : ''}`}
          >
            <svg className="patient-room-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={ICON_PATHS[it.key] || ICON_PATHS.temperature} /></svg>
            <div><div className="patient-room-label">{it.label}</div><div className="patient-room-value">{it.value}</div></div>
          </li>
        ))}
      </ul>
    </section>
  );
}
