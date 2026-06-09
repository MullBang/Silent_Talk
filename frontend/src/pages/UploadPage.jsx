// UploadPage.jsx — SCR-01 업로드 화면.
// DropZone으로 영상 업로드, window.isSecureContext 검사 후 추론을 시작한다.

import React from 'react';

/**
 * 업로드 페이지 (SCR-01).
 * 보안 컨텍스트(window.isSecureContext) 검사 → false이면 HTTPS 모달 노출.
 * 업로드 성공 시 session_id를 받아 추론을 등록하고 결과 페이지로 전환한다.
 * @returns {JSX.Element}
 */
export default function UploadPage() {
  // TODO: 업로드 폼, 보안 컨텍스트 검사, 추론 등록 흐름
  return null;
}
