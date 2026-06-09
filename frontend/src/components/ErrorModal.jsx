// ErrorModal.jsx — 에러/HTTPS 안내 모달.

import React from 'react';

/**
 * 에러 모달 컴포넌트.
 * window.isSecureContext === false 시 HTTPS 요구 안내 등에 사용한다.
 * @param {{ open: boolean, message: string, onClose: () => void }} props
 * @returns {JSX.Element}
 */
export default function ErrorModal({ open, message, onClose }) {
  // TODO: 모달 표시/닫기 처리
  return null;
}
