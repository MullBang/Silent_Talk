// App.jsx — 루트 컴포넌트.
// SCR-01(UploadPage) ↔ SCR-02(ResultPage) 상태 기반 라우팅.

import React, { useState } from 'react';

import ResultPage from './pages/ResultPage';
import UploadPage from './pages/UploadPage';

/**
 * 애플리케이션 루트.
 * 추론 완료 시 결과(payload)를 받아 결과 화면으로 전환하고,
 * '새 파일' 클릭 시 업로드 화면으로 되돌린다.
 * @returns {JSX.Element}
 */
export default function App() {
  const [view, setView] = useState('upload'); // 'upload' | 'result'
  const [payload, setPayload] = useState(null); // { sessionId, fileName, results }

  const handleComplete = (result) => {
    setPayload(result);
    setView('result');
  };

  const handleNewFile = () => {
    setPayload(null);
    setView('upload');
  };

  return (
    <div className="min-h-screen bg-white text-gray-900">
      {view === 'result' && payload ? (
        <ResultPage
          fileName={payload.fileName}
          results={payload.results}
          onNewFile={handleNewFile}
        />
      ) : (
        <UploadPage onComplete={handleComplete} />
      )}
    </div>
  );
}
