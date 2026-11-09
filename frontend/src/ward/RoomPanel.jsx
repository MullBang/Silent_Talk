// RoomPanel.jsx — 가상 병실 (조명·커튼·침대·TV·온도). 명령 동작으로 바뀌고, 침대는 간호사 확인 뒤에만 움직인다.

import React from 'react';

import { describeRoom } from './wardCommands';
import { WARD_COLORS } from './wardTheme';

/**
 * 가상 병실 표시.
 * @param {{room: object}} props
 * @returns {JSX.Element}
 */
export default function RoomPanel({ room }) {
  const items = describeRoom(room);
  return (
    <section aria-label="가상 병실" className="rounded-xl border-2 p-3" style={{ borderColor: WARD_COLORS.line }}>
      <h2 className="mb-2 text-lg font-bold">가상 병실 <span className="text-sm font-normal" style={{ color: WARD_COLORS.inkSoft }}>(시연용 · 실제 기기 연동 없음)</span></h2>
      <ul className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {items.map((it) => (
          <li
            key={it.key}
            className="rounded-lg border-2 px-3 py-2"
            style={{
              borderColor: it.on && (it.key === 'light' || it.key === 'tv' || it.key === 'curtain') ? WARD_COLORS.infoBg : WARD_COLORS.line,
              background: WARD_COLORS.paper,
            }}
          >
            <div className="text-sm font-semibold" style={{ color: WARD_COLORS.inkSoft }}>{it.label}</div>
            <div className="text-xl font-bold">{it.value}</div>
          </li>
        ))}
      </ul>
    </section>
  );
}
