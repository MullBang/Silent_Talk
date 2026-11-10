// WardShell.jsx — 병실 도우미 화면(#ward·#nurse·#enroll·#lists)의 공통 틀: 위쪽 머리띠(이름·지금 화면·다른 화면 새 탭으로 열기).
// 업로드·녹화·촬영 같은 다른 화면은 이 틀을 쓰지 않는다. 버튼 묶음(nav)만 data-no-switch로 덮어 버튼 조작이 스위치 누름으로 잡히지 않게 하고,
// 이름 부분을 누르는 것은 환자 화면에서 스위치 누름으로 친다.
// 등록 화면이 녹화·저장 중이면 'ward-ui-navigation-lock' 이벤트로 알려 오고, 그동안 다른 화면 열기 버튼을 막는다.

import React, { useEffect, useState } from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import './ui.css';

/** 화면 이름 (머리띠 가운데 글자). */
const VIEW_LABELS = Object.freeze({ ward: '환자 화면', nurse: '간호사 화면', enroll: '입 모양 등록', lists: '내 목록 관리' });
/** 등록 화면이 화면 이동을 막고 있음을 알리는 이벤트 이름. */
export const NAVIGATION_LOCK_EVENT = 'ward-ui-navigation-lock';

/**
 * 같은 주소의 다른 병실 화면을 새 탭으로 연다 (BroadcastChannel은 같은 주소끼리만 통하므로 경로를 그대로 쓴다).
 * @param {string} hash '#nurse' 같은 해시.
 */
function openInNewTab(hash) {
  try {
    const loc = globalThis.location;
    globalThis.open?.(`${loc.pathname}${hash}`, '_blank', 'noopener');
  } catch {
    /* 새 탭을 열지 못하면 그대로 둔다 */
  }
}

/**
 * 병실 화면 공통 틀.
 * @param {{view: 'ward'|'nurse'|'enroll'|'lists', children: React.ReactNode}} props
 * @returns {JSX.Element}
 */
export default function WardShell({ view, children }) {
  const [lock, setLock] = useState({ locked: false, reason: '' });

  useEffect(() => {
    const onLock = (event) => {
      const detail = event?.detail || {};
      setLock({ locked: detail.locked === true, reason: String(detail.reason || '') });
    };
    window.addEventListener(NAVIGATION_LOCK_EVENT, onLock);
    return () => window.removeEventListener(NAVIGATION_LOCK_EVENT, onLock);
  }, []);
  // 화면이 바뀌면 이전 화면의 막음을 풀어 둔다 (등록 화면은 내릴 때도 풀어 준다).
  useEffect(() => { setLock({ locked: false, reason: '' }); }, [view]);

  const otherHash = view === 'nurse' ? '#ward' : '#nurse';
  return (
    <div className="ward-app">
      <div className="app-bar">
        <div className="app-brand">
          <svg width="30" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <rect x="2" y="2" width="28" height="24" rx="8" fill="currentColor" />
            <path d="M9 24v6l7-6" fill="currentColor" />
            <path d="M10 14h12M16 8v12" stroke="white" strokeWidth="3" strokeLinecap="round" />
          </svg>
          <div><strong>병실도우미</strong><span>입 모양으로 전하는 한마디</span></div>
        </div>
        {/* 마우스로 누른 버튼의 포커스를 풀어 준다: 남아 있으면 뒤이은 스위치(키) 누름이 그 버튼으로 가서 긴급 호출 3번 누르기도 막힌다. */}
        <nav aria-label="다른 시연 화면 열기" title={lock.locked ? lock.reason : undefined} data-no-switch onClick={blurAfterPointerClick}>
          <span className="app-current">{VIEW_LABELS[view] || ''}</span>
          <button type="button" disabled={lock.locked} onClick={() => openInNewTab(otherHash)}>
            {view === 'nurse' ? '환자 화면 열기' : '간호사 화면 열기'} <span aria-hidden="true">↗</span>
          </button>
          <button type="button" disabled={lock.locked} onClick={() => openInNewTab('#ward?practice=1')}>
            연습 화면 열기 <span aria-hidden="true">↗</span>
          </button>
        </nav>
      </div>
      {children}
    </div>
  );
}
