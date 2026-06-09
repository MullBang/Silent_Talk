// App.jsx — 루트 컴포넌트.
// SCR-01(UploadPage) ↔ SCR-02(ResultPage) 상태 기반 라우팅.

import React, { useState } from 'react';

import ResultPage from './pages/ResultPage';
import UploadPage from './pages/UploadPage';
import WebcamPage from './pages/WebcamPage';

/**
 * 애플리케이션 루트.
 * 업로드 ↔ 결과 ↔ 웹캠(/webcam) 화면을 상태 기반으로 전환한다.
 * @returns {JSX.Element}
 */
export default function App() {
  const [view, setView] = useState('upload'); // 'upload' | 'result' | 'webcam'
  const [payload, setPayload] = useState(null); // { sessionId, fileName, results }

  const handleComplete = (result) => {
    setPayload(result);
    setView('result');
  };

  const handleNewFile = () => {
    setPayload(null);
    setView('upload');
  };

  let screen;
  if (view === 'webcam') {
    screen = <WebcamPage onBack={() => setView('upload')} />;
  } else if (view === 'result' && payload) {
    screen = (
      <ResultPage
        fileName={payload.fileName}
        results={payload.results}
        onNewFile={handleNewFile}
      />
    );
  } else {
    screen = (
      <UploadPage
        onComplete={handleComplete}
        onWebcam={() => setView('webcam')}
      />
    );
  }

  return <div className="min-h-screen bg-white text-gray-900">{screen}</div>;
}
