// App.jsx — 루트 컴포넌트.
// SCR-01(UploadPage) ↔ SCR-02(ResultPage) 상태 기반 라우팅 + 녹화→분석(RecordPage, 6단계).

import React, { useEffect, useState } from 'react';

import CapturePage from './pages/CapturePage';
import RecordPage from './pages/RecordPage';
import ResultPage from './pages/ResultPage';
import UploadPage from './pages/UploadPage';
import WebcamPage from './pages/WebcamPage';

// URL 해시로 바로 여는 화면. #capture: 파일럿 촬영, #record: 녹화→분석.
const HASH_VIEWS = { '#capture': 'capture', '#record': 'record' };

/** 현재 URL 해시에 해당하는 화면 (없으면 업로드). */
function viewFromHash() {
  return HASH_VIEWS[globalThis.location?.hash] || 'upload';
}

/**
 * 주소창 해시를 화면과 맞춘다 (hashchange를 일으키지 않도록 replaceState 사용).
 * @param {string} hash '#record' 같은 해시 또는 ''.
 */
function syncHash(hash) {
  try {
    const loc = globalThis.location;
    if (!loc || loc.hash === hash) return;
    globalThis.history?.replaceState(null, '', `${loc.pathname}${loc.search}${hash}`);
  } catch {
    // 해시 갱신 실패는 화면 전환에 영향 없음
  }
}

/**
 * 애플리케이션 루트.
 * 업로드 ↔ 결과 ↔ 웹캠 ↔ 촬영 ↔ 녹화 화면을 상태 기반으로 전환한다.
 * @returns {JSX.Element}
 */
export default function App() {
  // 'upload' | 'result' | 'webcam' | 'capture' | 'record'
  const [view, setView] = useState(viewFromHash);
  const [payload, setPayload] = useState(null); // { sessionId, fileName, results, warnings, model, origin }

  // 주소창에서 #capture / #record 를 붙이거나 지우면 화면을 바꾼다.
  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  const goUpload = () => {
    syncHash('');
    setView('upload');
  };

  const goRecord = () => {
    syncHash('#record');
    setView('record');
  };

  const handleComplete = (result) => {
    setPayload(result);
    setView('result');
  };

  const handleNewFile = () => {
    const origin = payload?.origin;
    setPayload(null);
    if (origin === 'record') goRecord();
    else goUpload();
  };

  let screen;
  if (view === 'capture') {
    screen = <CapturePage onBack={goUpload} />;
  } else if (view === 'record') {
    screen = <RecordPage onComplete={handleComplete} onBack={goUpload} />;
  } else if (view === 'webcam') {
    screen = <WebcamPage onBack={goUpload} />;
  } else if (view === 'result' && payload) {
    screen = (
      <ResultPage
        fileName={payload.fileName}
        results={payload.results}
        warnings={payload.warnings}
        model={payload.model}
        origin={payload.origin}
        onNewFile={handleNewFile}
      />
    );
  } else {
    screen = (
      <UploadPage
        onComplete={handleComplete}
        onWebcam={() => setView('webcam')}
        onCapture={() => setView('capture')}
        onRecord={goRecord}
      />
    );
  }

  return <div className="min-h-screen bg-white text-gray-900">{screen}</div>;
}
