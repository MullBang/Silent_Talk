// ResultPage.jsx — SCR-02 추론 결과 화면.
// Tailwind CSS만 사용.

import React, { useMemo, useState } from 'react';

import ResultList, { msToClock } from '../components/ResultList';
import { UNSEEN_PERSON_NOTICE, isNonRecognition, modelInfoLine } from '../results/markers';

/**
 * 결과 페이지 (SCR-02).
 *
 * @param {object} props
 * @param {string} [props.fileName] - 분석한 파일명.
 * @param {Array<{start_ms:number,end_ms:number,text:string,confidence:number|null}>} [props.results]
 * @param {string[]} [props.warnings] - 서버가 보낸 결과 품질 경고(낮은 프레임률 등).
 * @param {object|null} [props.model] - 결과를 만든 모델 정보 (GET /api/result 의 model). 없으면 표시하지 않는다.
 * @param {'upload'|'record'} [props.origin] - 'record'이면 돌아가기 버튼이 '다시 녹화'가 된다.
 * @param {() => void} [props.onNewFile] - '새 파일'/'다시 녹화' 클릭 시 이전 입력 화면으로 이동.
 * @returns {JSX.Element}
 */
export default function ResultPage({ fileName, results, warnings, model, origin, onNewFile }) {
  const items = useMemo(() => results || [], [results]);
  const notices = useMemo(
    () => (Array.isArray(warnings) ? warnings.filter((w) => typeof w === 'string' && w.trim()) : []),
    [warnings],
  );
  const [toast, setToast] = useState(null);
  const modelLine = useMemo(() => modelInfoLine(model), [model]);
  const backLabel = origin === 'record' ? '다시 녹화' : '새 파일';

  // 요약 집계
  const summary = useMemo(() => {
    const recognized = items.filter((i) => !isNonRecognition(i.text));
    const totalChars = recognized.reduce((acc, i) => acc + (i.text?.length || 0), 0);
    const confs = items
      .map((i) => i.confidence)
      .filter((c) => typeof c === 'number');
    const avgConf = confs.length
      ? (confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(2)
      : '—';
    return { segments: items.length, totalChars, avgConf };
  }, [items]);

  const handleCopy = async () => {
    const text = items
      .map((i) => `[${msToClock(i.start_ms)}~${msToClock(i.end_ms)}] ${i.text}`)
      .join('\n');
    try {
      if (!navigator.clipboard?.writeText) throw new Error('clipboard unsupported');
      await navigator.clipboard.writeText(text);
      setToast('복사됨 ✓');
    } catch {
      setToast('이 브라우저는 복사를 지원하지 않습니다');
    }
    setTimeout(() => setToast(null), 1500);
  };

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      {/* 헤더 */}
      <header className="mb-6 flex items-center gap-3">
        <button
          type="button"
          onClick={onNewFile}
          aria-label={backLabel}
          className="shrink-0 rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-50"
        >
          ← {backLabel}
        </button>
        <span className="truncate text-sm text-gray-600">{fileName || '영상'}</span>
        <span className="ml-auto shrink-0 text-sm font-medium text-green-600">
          분석 완료 ✅
        </span>
      </header>

      {/* 성능 미검증 안내 (항상 표시) + 모델 정보 */}
      <div className="mb-4 rounded-xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm text-blue-800">
        <p>ℹ {UNSEEN_PERSON_NOTICE}</p>
        {modelLine && <p className="mt-1 break-words text-xs text-blue-600">모델: {modelLine}</p>}
      </div>

      {/* 품질 경고 (낮은 프레임률 등) */}
      {notices.length > 0 && (
        <div
          role="status"
          className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800"
        >
          {notices.map((w) => (
            <p key={w}>⚠ {w}</p>
          ))}
        </div>
      )}

      {/* 요약 카드 3개 */}
      <div className="mb-6 grid grid-cols-3 gap-3">
        <SummaryCard label="총 구간" value={summary.segments} />
        <SummaryCard label="총 글자" value={summary.totalChars} />
        <SummaryCard label="평균 신뢰도" value={summary.avgConf} />
      </div>

      {/* 결과 리스트 */}
      <div className="rounded-2xl border border-gray-100 px-4 py-2">
        <ResultList results={items} />
      </div>

      {/* 버튼 바 */}
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={handleCopy}
          aria-label="전체 텍스트 복사"
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-700"
        >
          전체 텍스트 복사 📋
        </button>
        <button
          type="button"
          disabled
          title="준비 중 (확장 기능)"
          aria-label="TXT 저장 (준비 중)"
          className="cursor-not-allowed rounded-lg border border-gray-200 px-4 py-2 text-sm text-gray-400"
        >
          TXT 저장
        </button>
        <button
          type="button"
          disabled
          title="준비 중 (확장 기능)"
          aria-label="SRT 저장 (준비 중)"
          className="cursor-not-allowed rounded-lg border border-gray-200 px-4 py-2 text-sm text-gray-400"
        >
          SRT 저장
        </button>
      </div>

      {/* 복사 토스트 */}
      {toast && (
        <div
          role="status"
          className="fixed bottom-6 left-1/2 -translate-x-1/2 rounded-lg bg-gray-900 px-4 py-2 text-sm text-white shadow-lg"
        >
          {toast}
        </div>
      )}
    </div>
  );
}

/** 요약 카드 1개. */
function SummaryCard({ label, value }) {
  return (
    <div className="rounded-xl bg-gray-50 px-3 py-4 text-center">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="mt-1 text-lg font-semibold text-gray-900">{value}</p>
    </div>
  );
}
