// PhrasePanel.jsx — 환자 화면 '말할 수 있는 문장' 칸: 지금 쓰는 명령 목록의 입 모양 문장을 분류별로 보여 준다.
// 환자가 20개(내 목록은 4~20개) 문장을 외우지 않고 보고 말할 수 있게 한다. 보기만 하는 칸이라 문장을 눌러 고르지는 않는다
// (입 모양 인식을 건너뛰지 않는다). 접기 버튼만 data-no-switch이고, 나머지를 누르는 것은 환자 화면의 스위치 누름이다.

import React from 'react';

import { blurAfterPointerClick } from '../record/pointerFocus';
import { AlertIcon } from './WardIcons';
import { phraseGroups } from './wardCommands';

/**
 * 말할 수 있는 문장 칸.
 * @param {{index: object, open: boolean, onToggle: () => void, inline?: boolean}} props
 *   index는 지금 쓰는 명령 색인(연습 모드는 기본 목록), open은 펼침 여부(설정 showPhrases), onToggle은 접기·펼치기,
 *   inline은 결과 칸 안에 넣을 때(테두리 없이 촘촘하게, 노트북 화면에서도 스크롤 없이 보이게).
 * @returns {JSX.Element|null}
 */
export default function PhrasePanel({ index, open, onToggle, inline = false }) {
  const groups = phraseGroups(index);
  if (!groups.length) return null;
  const count = groups.reduce((n, g) => n + g.commands.length, 0);
  const custom = index?.customList;
  return (
    // 결과 칸(aria-live)에 넣을 때는 대기 화면으로 돌아올 때마다 25줄을 읽지 않게 이 칸만 알림을 끈다.
    <section aria-label="말할 수 있는 문장" aria-live={inline ? 'off' : undefined} className={`patient-phrases${open ? ' is-open' : ''}${inline ? ' is-inline' : ''}`} data-phrase-list={custom?.id || 'default'}>
      <div className="patient-phrases-heading">
        <div>
          <h2>말할 수 있는 문장 <span>{custom ? `내 목록 ‘${custom.title}’ · ` : ''}{count}개</span></h2>
          {open && <p>녹화가 시작되면 이 중 하나를 그대로 입 모양으로 말하세요</p>}
        </div>
        <span data-no-switch onClick={blurAfterPointerClick}>
          <button type="button" className="patient-small-button" aria-expanded={open} onClick={onToggle}>
            {open ? '문장 접기' : '문장 보기'}
          </button>
        </span>
      </div>
      {open && (
        <div className="patient-phrase-groups">
          {groups.map((g) => (
            <div key={g.id} className="patient-phrase-group">
              <h3>{g.label}</h3>
              <ul>
                {g.commands.map((c) => (
                  <li key={c.id} data-phrase-id={c.id} className={c.emergency ? 'is-emergency' : ''} title={`화면·간호사실에는 “${c.display}”`}>
                    {/* 긴급은 색만으로 구분하지 않는다: 같은 줄 맨 앞에 경고 아이콘(읽기: 긴급)을 붙인다. */}
                    <span className="patient-phrase-text">
                      {c.emergency && <span className="patient-phrase-icon"><AlertIcon /> </span>}{c.text}
                      {c.cancel && <span className="patient-phrase-tag">취소</span>}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
