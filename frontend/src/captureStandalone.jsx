// captureStandalone.jsx — 조원이 각자 PC에서 여는 단독 촬영 화면의 진입점 (서버·모델 없이 동작).
// scripts/build-capture-package.mjs가 이 화면을 HTML 파일 하나로 묶는다(웹캠촬영.html).

import React from 'react';
import ReactDOM from 'react-dom/client';

import CapturePage from './pages/CapturePage';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <CapturePage />
  </React.StrictMode>,
);
