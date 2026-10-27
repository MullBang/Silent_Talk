// CapturePage.jsx — 웹캠 촬영 화면 (모델 없이 동작). 3단계 파일럿·조정용과 8단계 최종 평가용을 같이 쓴다.
// 문장 프롬프트를 띄우고 스페이스바/버튼으로 영상만 녹화해 저장소 밖 폴더에 저장한다.
// 저장: Chrome/Edge는 File System Access API(폴더 선택), 그 밖에는 다운로드로 대체.
// 파일명: {가명ID}_{조건}_{문장ID}_{회차}.{webm|mp4} + 같은 이름의 .json(실제 해상도·fps·프레임 시각 로그).

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ErrorModal from '../components/ErrorModal';
import useRecorder from '../hooks/useRecorder';
import PROMPTS from '../capture/prompts.json';
import { CONDITIONS, SESSIONS, buildPlan, clipBaseName, safeName, sessionFromMetas } from '../capture/plan';

const MAX_SECONDS = 8;
const KIND_HINT = {
  free: '영어·숫자·사람 이름·사는 곳 없이 3초 안쪽의 한 문장으로 소리 내어 말합니다. 실제로 말한 문장을 기록지에 적어 주세요.',
  nonspeech: '정지할 때까지 말은 하지 않고 위 지시대로만 합니다.',
};

export default function CapturePage({ onBack }) {
  const [participant, setParticipant] = useState('');
  const [sessionId, setSessionId] = useState(SESSIONS[0].id);
  const [dirHandle, setDirHandle] = useState(null);
  const [index, setIndex] = useState(0);
  const [saved, setSaved] = useState([]);
  const [modal, setModal] = useState(null);
  const [lastMeta, setLastMeta] = useState(null);
  const [saving, setSaving] = useState(false);
  // 고른 폴더에 이미 있는 클립의 촬영 묶음 {session, clips}. 있으면 묶음을 그 값으로 고정한다.
  const [folderSession, setFolderSession] = useState(null);
  const session = SESSIONS.find((x) => x.id === sessionId) || SESSIONS[0];
  const plan = useMemo(() => buildPlan(PROMPTS, session.repeats), [session.repeats]);
  const item = plan[Math.min(index, plan.length - 1)];
  const done = index >= plan.length;
  const participantRef = useRef(participant);
  participantRef.current = participant;
  const itemRef = useRef(item);
  itemRef.current = item;
  const dirRef = useRef(dirHandle);
  dirRef.current = dirHandle;
  const sessionRef = useRef(session);
  sessionRef.current = session;

  const persist = useCallback(async (result) => {
    setSaving(true);
    const cur = itemRef.current;
    const base = clipBaseName(participantRef.current, cur);
    const videoName = `${base}.${result.meta.extension}`;
    const meta = { ...result.meta, participant: safeName(participantRef.current), condition: cur.cond,
      sentenceId: cur.sentenceId, text: cur.text, kind: cur.kind, rep: cur.rep, maxSeconds: MAX_SECONDS, file: videoName,
      session: sessionRef.current.id, repeats: sessionRef.current.repeats, promptsVersion: PROMPTS.version };
    const metaBlob = new Blob([JSON.stringify(meta, null, 2)], { type: 'application/json' });
    try {
      if (dirRef.current) {
        for (const [name, blob] of [[videoName, result.blob], [`${base}.json`, metaBlob]]) {
          const fh = await dirRef.current.getFileHandle(name, { create: true });
          const w = await fh.createWritable();
          await w.write(blob);
          await w.close();
        }
      } else {
        for (const [name, blob] of [[videoName, result.blob], [`${base}.json`, metaBlob]]) {
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = name;
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 2000);
        }
      }
      setSaved((s) => [...s, { file: videoName, bytes: result.blob.size, frames: result.meta.frameCallbacks, fps: result.meta.estimatedFps }]);
      setLastMeta(meta);
      setIndex((i) => i + 1);
    } catch (e) {
      setModal({ type: 'UPLOAD_ERROR', message: `저장 실패: ${e.message}` });
    } finally {
      setSaving(false);
    }
  }, []);

  const rec = useRecorder({ maxSeconds: MAX_SECONDS, onResult: persist });
  const { status, error, errorType, errorSeq, videoRef, settings, open, close, start, stop, elapsedMs } = rec;

  // 카메라 오류마다(errorSeq 증가) 모달. error가 null이면 모달의 기본 한국어 문구를 쓴다.
  useEffect(() => {
    if (errorSeq > 0) setModal({ type: errorType || 'DEVICE_ERROR', message: error || undefined });
  }, [errorSeq]); // eslint-disable-line react-hooks/exhaustive-deps

  const pickFolder = async () => {
    try {
      if (!window.showDirectoryPicker) { setModal({ type: 'UPLOAD_ERROR', message: '이 브라우저는 폴더 저장을 지원하지 않습니다. 다운로드 폴더에 저장됩니다.' }); return; }
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
      const metas = [];
      for await (const entry of handle.values()) {
        if (entry.kind !== 'file' || !entry.name.endsWith('.json')) continue;
        try { metas.push(JSON.parse(await (await entry.getFile()).text())); } catch { /* 클립 기록이 아닌 파일은 건너뛴다 */ }
      }
      const found = sessionFromMetas(metas);
      if (found.mixed) {
        setModal({ type: 'UPLOAD_ERROR', message: '이 폴더에는 파일럿·조정용과 최종 평가용 클립이 섞여 있습니다. 묶음마다 다른 폴더를 고르세요.' });
        return;
      }
      const startedHere = index > 0 || saved.length > 0;
      if (found.session && startedHere && found.session !== sessionRef.current.id) {
        setModal({ type: 'UPLOAD_ERROR', message: '이 폴더의 클립은 지금 고른 촬영 묶음과 다릅니다. 다른 폴더를 고르세요.' });
        return;
      }
      setDirHandle(handle);
      setFolderSession(found.session ? { session: found.session, clips: found.clips } : null);
      if (found.session) setSessionId(found.session);
    } catch (e) {
      if (e.name !== 'AbortError') setModal({ type: 'UPLOAD_ERROR', message: e.message });
    }
  };

  const toggle = useCallback(() => {
    if (done || saving) return;
    if (status === 'recording') stop();
    else if (status === 'ready') start();
  }, [done, saving, status, start, stop]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); toggle(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [toggle]);

  const remaining = Math.max(0, MAX_SECONDS - elapsedMs / 1000);
  // 첫 클립을 저장하기 시작한 뒤, 또는 폴더에 이미 클립이 있으면 촬영 묶음을 바꿀 수 없다 ('이전'으로 처음에 돌아가도 유지).
  const sessionLocked = index > 0 || saved.length > 0 || saving || folderSession !== null || status === 'recording';
  const cond = CONDITIONS.find((c) => c.id === item?.cond);

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <header className="mb-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">🎥 웹캠 촬영 (모델 없음)</h1>
        <button type="button" onClick={() => { close(); onBack(); }} className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50">
          ← 업로드 모드
        </button>
      </header>
      <p className="mb-3 text-xs text-amber-600">
        영상만 녹화합니다(음성 트랙 없음). 정면, 50~70cm, 고정 조명. 한 클립은 최대 {MAX_SECONDS}초이며 스페이스바로 시작·정지합니다.
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <label className="flex items-center gap-2">
          가명 ID
          <input value={participant} onChange={(e) => setParticipant(e.target.value)} placeholder="예: P01"
            className="w-24 rounded border border-gray-300 px-2 py-1" disabled={status === 'recording'} />
        </label>
        <label className="flex items-center gap-2">
          촬영 묶음
          <select value={sessionId} onChange={(e) => setSessionId(e.target.value)} disabled={sessionLocked}
            className="rounded border border-gray-300 px-2 py-1" title="첫 클립을 찍은 뒤에는 바꿀 수 없습니다">
            {SESSIONS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <button type="button" onClick={pickFolder} disabled={status === 'recording' || saving}
          className="rounded border border-gray-300 px-2 py-1 hover:bg-gray-50 disabled:opacity-40">
          {dirHandle ? `저장 폴더: ${dirHandle.name}` : '저장 폴더 선택'}
        </button>
        {status === 'idle' || status === 'error' ? (
          <button type="button" onClick={open} disabled={!participant} className="rounded bg-blue-600 px-3 py-1 text-white disabled:opacity-40">카메라 열기</button>
        ) : (
          <button type="button" onClick={close} className="rounded border border-gray-300 px-3 py-1 hover:bg-gray-50">카메라 닫기</button>
        )}
        {settings && <span className="text-xs text-gray-500">{settings.width}×{settings.height} @ {settings.frameRate}fps · {settings.label}</span>}
      </div>
      {folderSession && (
        <p className="mb-3 text-xs text-gray-600">
          이 폴더에 '{(SESSIONS.find((x) => x.id === folderSession.session) || SESSIONS[0]).label}' 클립 기록 {folderSession.clips}개가 있어 촬영 묶음을 고정했습니다. 같은 가명 ID로 이어 찍으면 같은 이름의 클립은 덮어씁니다.
        </p>
      )}

      <div className="relative mb-3 overflow-hidden rounded-lg bg-black">
        <video ref={videoRef} className="w-full" style={{ transform: 'scaleX(-1)' }} playsInline muted />
        {status === 'recording' && (
          <div className="absolute left-3 top-3 rounded bg-red-600 px-2 py-0.5 text-sm text-white">● REC {remaining.toFixed(1)}s</div>
        )}
      </div>

      {!done ? (
        <div className="mb-3 rounded-lg border border-gray-200 bg-gray-50 p-4">
          <div className="mb-1 flex items-center justify-between text-xs text-gray-500">
            <span>{index + 1} / {plan.length} · {cond ? cond.label : '무발화'} · {item.sentenceId} · {item.rep}회차</span>
            <span>{item.kind === 'fixed' ? '지정 문장' : item.kind === 'free' ? '자유 문장' : '무발화'}</span>
          </div>
          <p className="text-2xl font-semibold leading-relaxed text-gray-900">{item.text}</p>
          {cond && <p className="mt-1 text-sm text-gray-600">{cond.hint}</p>}
          {KIND_HINT[item.kind] && <p className="mt-1 text-sm text-gray-600">{KIND_HINT[item.kind]}</p>}
          <div className="mt-3 flex gap-2">
            <button type="button" onClick={toggle} disabled={saving || (status !== 'ready' && status !== 'recording')}
              className={`rounded px-4 py-2 text-white disabled:opacity-40 ${status === 'recording' ? 'bg-red-600' : 'bg-green-600'}`}>
              {status === 'recording' ? '정지 (Space)' : '녹화 시작 (Space)'}
            </button>
            <button type="button" onClick={() => setIndex((i) => i + 1)} disabled={status === 'recording'}
              className="rounded border border-gray-300 px-3 py-2 text-sm hover:bg-gray-50 disabled:opacity-40">건너뛰기</button>
            <button type="button" onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={status === 'recording' || index === 0}
              className="rounded border border-gray-300 px-3 py-2 text-sm hover:bg-gray-50 disabled:opacity-40">이전 (다시 찍기)</button>
          </div>
        </div>
      ) : (
        <div className="mb-3 rounded-lg bg-green-50 p-4 text-green-800">촬영이 끝났습니다. 저장된 클립 {saved.length}개.</div>
      )}

      {lastMeta && (
        <p className="mb-2 text-xs text-gray-500">
          마지막 저장: {lastMeta.file} · {(lastMeta.bytes / 1024).toFixed(0)}KB · 프레임 {lastMeta.frameCallbacks}개 · 추정 {lastMeta.estimatedFps ? lastMeta.estimatedFps.toFixed(1) : '?'}fps · 최대 간격 {lastMeta.maxGapMs ? lastMeta.maxGapMs.toFixed(0) : '?'}ms
        </p>
      )}
      {saved.length > 0 && (
        <details className="text-xs text-gray-600">
          <summary>저장 목록 ({saved.length})</summary>
          <ul className="mt-1 max-h-40 overflow-auto">{saved.map((s) => <li key={s.file}>{s.file} · {(s.bytes / 1024).toFixed(0)}KB</li>)}</ul>
        </details>
      )}

      {modal && <ErrorModal type={modal.type} message={modal.message} onClose={() => setModal(null)} />}
    </div>
  );
}
