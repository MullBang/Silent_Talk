// DropZone.jsx — 드래그앤드롭 / 클릭 파일 선택 영역 (화면명세서 SCR-01 UI-01-02/03).
// Tailwind CSS만 사용.

import React, { useEffect, useRef, useState } from 'react';

const ALLOWED_EXT = ['.mp4', '.avi', '.mov'];

/** 확장자 허용 여부. */
function hasAllowedExt(name) {
  const lower = (name || '').toLowerCase();
  return ALLOWED_EXT.some((ext) => lower.endsWith(ext));
}

/** 바이트 → "N.N MB". */
function formatSize(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

/** 초 → "HH:MM:SS". */
function formatDuration(sec) {
  if (!Number.isFinite(sec)) return '--:--:--';
  const s = Math.floor(sec % 60);
  const m = Math.floor((sec / 60) % 60);
  const h = Math.floor(sec / 3600);
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

/**
 * 영상 파일 드롭존.
 *
 * @param {object} props
 * @param {File|null} props.file - 현재 선택된 파일.
 * @param {(file: File|null) => void} props.onFileSelect - 파일 선택 콜백.
 * @param {boolean} [props.disabled] - 비활성화(업로드/추론 중) 여부.
 * @returns {JSX.Element}
 */
export default function DropZone({ file, onFileSelect, disabled = false }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [duration, setDuration] = useState(null);
  const [localError, setLocalError] = useState('');

  // 파일 길이(초) 메타데이터 추출
  useEffect(() => {
    if (!file) {
      setDuration(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      setDuration(video.duration);
      URL.revokeObjectURL(url);
    };
    video.onerror = () => {
      setDuration(null);
      URL.revokeObjectURL(url);
    };
    video.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pick = (f) => {
    if (!f) return;
    if (!hasAllowedExt(f.name)) {
      setLocalError('지원하지 않는 형식입니다. MP4 / AVI / MOV만 가능합니다.');
      return;
    }
    setLocalError('');
    onFileSelect(f);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    pick(e.dataTransfer.files?.[0]);
  };

  const handleClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const handleKeyDown = (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  const base =
    'flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition';
  const stateCls = disabled
    ? 'cursor-not-allowed border-gray-200 bg-gray-50 opacity-60'
    : dragOver
      ? 'cursor-pointer border-blue-500 bg-blue-50'
      : 'cursor-pointer border-gray-300 hover:border-blue-400';

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="영상 파일 선택"
        aria-disabled={disabled}
        className={`${base} ${stateCls}`}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="text-sm text-gray-700">
            <p className="mb-1 font-medium text-gray-900">📄 {file.name}</p>
            <p className="text-gray-500">
              크기: {formatSize(file.size)} &nbsp;|&nbsp; 길이: {formatDuration(duration)}
            </p>
            <p className="mt-2 text-xs text-blue-500">다른 파일을 선택하려면 클릭하세요</p>
          </div>
        ) : (
          <div className="text-gray-500">
            <p className="mb-1 text-base">📁 파일을 드래그하거나 클릭하여 선택</p>
            <p className="text-xs">MP4 / AVI / MOV · 최대 500MB · 최대 3분</p>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.avi,.mov,video/mp4,video/x-msvideo,video/quicktime"
          className="hidden"
          onChange={(e) => pick(e.target.files?.[0])}
          disabled={disabled}
        />
      </div>

      {localError && (
        <p className="mt-2 text-sm text-red-600" role="alert">
          {localError}
        </p>
      )}
    </div>
  );
}
