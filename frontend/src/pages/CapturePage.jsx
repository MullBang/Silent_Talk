// CapturePage.jsx — 웹캠 촬영 화면 (모델 없이 동작). 3단계 파일럿·조정용과 8단계 최종 평가용을 같이 쓴다.
// 문장 프롬프트를 띄우고 스페이스바/버튼으로 영상만 녹화해 저장소 밖 폴더에 저장한다.
// 저장: Chrome/Edge는 File System Access API(폴더 선택). 쓸 수 없으면 클립을 화면에 모아 두었다가 zip 하나로 내려받는다.
// 조원이 각자 PC에서 찍는 단독 실행판(capture.html → 웹캠촬영.html)도 이 화면을 쓴다(onBack 없음: 뒤로 가기·다음 사람 버튼을 숨김).
// 파일명: {가명ID}_{조건}_{문장ID}_{회차}.{webm|mp4} + 같은 이름의 .json(실제 해상도·fps·프레임 시각 로그).

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ErrorModal from '../components/ErrorModal';
import useRecorder from '../hooks/useRecorder';
import PROMPTS from '../capture/prompts.json';
import {
  CONDITIONS, SESSIONS, buildPlan, clipBaseName, clipQualityWarnings, nextUnsaved, normalizeSpoken, resumePoint, safeName,
  sessionFromMetas, spokenProblem,
} from '../capture/plan';
import { zipStore } from '../capture/zipStore';

const MAX_SECONDS = 8;
const KIND_HINT = {
  free: '영어·숫자·사람 이름·사는 곳 없이 3초 안쪽의 한 문장으로 소리 내어 말합니다. 말할 문장을 아래 칸에 먼저 적고 Enter를 누른 뒤, 적은 그대로 말하며 녹화합니다.',
  nonspeech: '정지할 때까지 말은 하지 않고 위 지시대로만 합니다.',
};
const HAS_FOLDER_API = typeof window !== 'undefined' && Boolean(window.showDirectoryPicker);

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
  // 저장된 클립의 기본 이름(확장자 제외): 폴더를 고를 때 있던 것 + 이번에 저장한 것. 다음 빈 순번·덮어쓰기 표시에 쓴다.
  const [existingNames, setExistingNames] = useState(() => new Set());
  // 폴더를 고를 때 이미 있던 클립만. 이어 찍기 안내는 이것으로만 띄운다(처음 찍는 중에는 뜨지 않게).
  const [preexistingNames, setPreexistingNames] = useState(() => new Set());
  // 자유 문장: 녹화 전에 적은 정답 문장(.json의 spoken으로 저장). 순번이 바뀌면 비운다.
  const [spokenText, setSpokenText] = useState('');
  // 폴더에 저장할 수 없으면 클립을 화면(메모리)에 모아 두고 zip으로 내려받는다.
  const [memoryMode, setMemoryMode] = useState(!HAS_FOLDER_API);
  const [memCount, setMemCount] = useState(0);
  const [memVersion, setMemVersion] = useState(0);
  const [downloadedVersion, setDownloadedVersion] = useState(0);
  const [jumpText, setJumpText] = useState('');
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
  const planRef = useRef(plan);
  planRef.current = plan;
  const indexRef = useRef(index);
  indexRef.current = index;
  const existingRef = useRef(existingNames);
  existingRef.current = existingNames;
  const modalOpenRef = useRef(false);
  modalOpenRef.current = modal !== null;
  const spokenRef = useRef(spokenText);
  spokenRef.current = spokenText;
  const memRef = useRef(new Map()); // 기본 이름 → [[파일 이름, Blob], ...] (다시 찍으면 바뀐다)

  useEffect(() => { setSpokenText(''); }, [index]);

  const unsavedMemory = memoryMode && memVersion > downloadedVersion;
  useEffect(() => {
    // 내려받지 않은 클립이 있으면 창을 닫거나 새로고침할 때 브라우저가 한 번 묻게 한다.
    if (!unsavedMemory) return undefined;
    const onBeforeUnload = (e) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [unsavedMemory]);

  const persist = useCallback(async (result) => {
    setSaving(true);
    const cur = itemRef.current;
    const base = clipBaseName(participantRef.current, cur);
    const videoName = `${base}.${result.meta.extension}`;
    const meta = { ...result.meta, participant: safeName(participantRef.current), condition: cur.cond,
      sentenceId: cur.sentenceId, text: cur.text, kind: cur.kind, rep: cur.rep, maxSeconds: MAX_SECONDS, file: videoName,
      session: sessionRef.current.id, repeats: sessionRef.current.repeats, promptsVersion: PROMPTS.version,
      ...(cur.kind === 'free' ? { spoken: normalizeSpoken(spokenRef.current) } : {}) };
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
        // 다시 찍으면 같은 이름을 바꿔 넣으므로 zip 안에 같은 클립이 둘 생기지 않는다.
        memRef.current.set(base, [[videoName, result.blob], [`${base}.json`, metaBlob]]);
        setMemCount(memRef.current.size);
        setMemVersion((v) => v + 1);
      }
      setSaved((s) => [...s, { file: videoName, bytes: result.blob.size, frames: result.meta.frameCallbacks, fps: result.meta.estimatedFps }]);
      setLastMeta(meta);
      // 다음 빈 순번으로 간다('이전'으로 다시 찍은 뒤 이미 저장된 다음 클립을 덮어쓰지 않게).
      const names = new Set(existingRef.current).add(base);
      existingRef.current = names;
      setExistingNames(names);
      if (dirRef.current) setFolderSession({ session: sessionRef.current.id, clips: names.size });
      setIndex(nextUnsaved(planRef.current, participantRef.current, names, indexRef.current + 1));
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
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
      const metas = [];
      const names = new Set();
      for await (const entry of handle.values()) {
        if (entry.kind !== 'file' || entry.name.startsWith('._')) continue;
        const video = entry.name.match(/^(.*)\.(webm|mp4)$/i);
        if (video) { names.add(video[1]); continue; }
        if (!entry.name.endsWith('.json')) continue;
        try { metas.push(JSON.parse(await (await entry.getFile()).text())); } catch { /* 클립 기록이 아닌 파일은 건너뛴다 */ }
      }
      const found = sessionFromMetas(metas);
      if (found.mixed) {
        setModal({ type: 'UPLOAD_ERROR', message: '이 폴더에는 파일럿·조정용과 최종 평가용 클립이 섞여 있습니다. 묶음마다 다른 폴더를 고르세요.' });
        return;
      }
      if (found.session && found.session !== sessionRef.current.id) {
        const label = (id) => (SESSIONS.find((x) => x.id === id) || SESSIONS[0]).short;
        setModal({ type: 'UPLOAD_ERROR', message: `이 폴더에는 '${label(found.session)}' 클립이 있는데 지금 고른 촬영 묶음은 '${label(sessionRef.current.id)}'입니다. 촬영 묶음과 폴더를 다시 확인하세요.` });
        return;
      }
      setDirHandle(handle);
      setMemoryMode(false);
      setExistingNames(names);
      setPreexistingNames(names);
      setFolderSession(found.session ? { session: found.session, clips: found.clips } : null);
    } catch (e) {
      if (e.name === 'AbortError') return;
      // 파일로 연 화면 등에서 폴더 저장이 막히면 화면에 모아 두었다가 zip으로 내려받는다(녹화가 막히지 않게).
      setMemoryMode(true);
      setModal({ type: 'UPLOAD_ERROR', message: `폴더에 저장할 수 없습니다(${e.message}). 찍은 클립을 화면에 모아 두었다가 zip 파일 하나로 내려받습니다.` });
    }
  };

  const downloadZip = async () => {
    try {
      const folder = `${safeName(participant || 'clips')}_웹캠촬영`;
      const version = memVersion;
      const entries = [];
      for (const files of memRef.current.values()) {
        for (const [name, blob] of files) entries.push({ name: `${folder}/${name}`, bytes: new Uint8Array(await blob.arrayBuffer()) });
      }
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([zipStore(entries)], { type: 'application/zip' }));
      a.download = `${folder}.zip`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      setDownloadedVersion(version);
    } catch (e) {
      setModal({ type: 'UPLOAD_ERROR', message: `내려받기 실패: ${e.message}` });
    }
  };

  // Chrome·Edge에서는 폴더를 고를 때까지 녹화를 막는다(폴더 저장이 막히면 화면에 모아 두는 방식으로 바뀐다).
  const needFolder = HAS_FOLDER_API && !dirHandle && !memoryMode;
  // 가명 ID가 비면 파일 이름이 'anon'이 되어 사람을 구분할 수 없으므로 녹화를 막는다('다음 사람 촬영' 직후 등).
  const needId = participant.trim() === '';
  // 자유 문장은 정답 문장을 먼저 적어야 녹화할 수 있다(소리를 저장하지 않으므로 이 글이 정답이다).
  const needSpoken = !done && item?.kind === 'free' && spokenProblem(spokenText) !== null;
  const blocked = needFolder || needId || needSpoken;

  const toggle = useCallback(() => {
    if (done || saving) return;
    if (status === 'recording') stop();
    else if (status === 'ready' && !blocked) start();
  }, [done, saving, status, start, stop, blocked]);

  useEffect(() => {
    // 스페이스바는 언제나 녹화 시작·정지다. 마우스로 누른 버튼(이전·카메라 열기 등)에 포커스가 남아도 그 버튼을
    // 다시 누르지 않도록 keydown·keyup 기본 동작을 막는다. 글자 입력칸·선택 상자·오류 창에서는 원래 동작을 둔다.
    const typing = (t) => Boolean(t && (t.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName)));
    const onKeyDown = (e) => {
      if (e.code !== 'Space' || modalOpenRef.current || typing(e.target)) return;
      e.preventDefault();
      if (!e.repeat) toggle();
    };
    const onKeyUp = (e) => {
      if (e.code === 'Space' && !modalOpenRef.current && !typing(e.target)) e.preventDefault();
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [toggle]);

  const startNextPerson = () => {
    // 같은 폴더·촬영 묶음으로 다음 사람을 처음부터 찍는다. 가명 ID는 새로 입력한다.
    if (unsavedMemory) {
      setModal({ type: 'UPLOAD_ERROR', message: '먼저 지금 사람의 클립을 zip으로 내려받으세요.' });
      return;
    }
    memRef.current = new Map();
    setMemCount(0);
    setParticipant('');
    setIndex(0);
    setSaved([]);
    setLastMeta(null);
  };

  const jumpTo = () => {
    const n = parseInt(jumpText, 10);
    if (Number.isFinite(n) && n >= 1 && n <= plan.length) { setIndex(n - 1); setJumpText(''); }
  };

  const remaining = Math.max(0, MAX_SECONDS - elapsedMs / 1000);
  // 첫 클립을 저장하기 시작한 뒤, 또는 폴더·화면에 이미 클립이 있으면 촬영 묶음을 바꿀 수 없다 ('이전'으로 처음에 돌아가도 유지).
  const sessionLocked = index > 0 || saved.length > 0 || saving || folderSession !== null || memCount > 0 || status === 'recording';
  const cond = CONDITIONS.find((c) => c.id === item?.cond);
  const preexisting = useMemo(() => resumePoint(plan, participant, preexistingNames), [plan, participant, preexistingNames]);
  const resume = useMemo(() => resumePoint(plan, participant, existingNames), [plan, participant, existingNames]);
  // 이어 찍기 버튼은 폴더를 고를 때 이 가명 ID의 클립이 이미 있었을 때만(창을 닫았다 다시 연 경우) 띄운다.
  const canResume = preexisting.saved > 0 && resume.index !== index && status !== 'recording' && !saving;
  const currentSaved = !done && existingNames.has(clipBaseName(participant, item));
  const warnings = clipQualityWarnings(lastMeta);
  const idle = status !== 'recording' && !saving;

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <header className="mb-3 flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900">🎥 웹캠 촬영 (모델 없음)</h1>
        {onBack && (
          <button type="button" onClick={() => { close(); onBack(); }} className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50">
            ← 업로드 모드
          </button>
        )}
      </header>
      <p className="mb-3 text-xs text-amber-600">
        영상만 녹화합니다(음성 트랙 없음). 정면, 50~70cm, 고정 조명. 한 클립은 최대 {MAX_SECONDS}초이며 스페이스바로 시작·정지합니다.
      </p>

      <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
        <label className="flex items-center gap-2">
          가명 ID
          <input value={participant} onChange={(e) => setParticipant(e.target.value.toUpperCase())} placeholder="예: P01"
            className="w-24 rounded border border-gray-300 px-2 py-1" disabled={status === 'recording'} />
        </label>
        <label className="flex items-center gap-2">
          촬영 묶음
          <select value={sessionId} onChange={(e) => setSessionId(e.target.value)} disabled={sessionLocked}
            className="rounded border border-gray-300 px-2 py-1" title="첫 클립을 찍은 뒤에는 바꿀 수 없습니다">
            {SESSIONS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        {HAS_FOLDER_API && (
          <button type="button" onClick={pickFolder} disabled={!idle}
            className="rounded border border-gray-300 px-2 py-1 hover:bg-gray-50 disabled:opacity-40">
            {dirHandle ? `저장 폴더: ${dirHandle.name}` : '저장 폴더 선택'}
          </button>
        )}
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
      {preexisting.saved > 0 && (
        <p className="mb-2 text-xs text-amber-700">
          폴더를 고를 때 가명 ID '{safeName(participant)}'로 저장된 클립이 이미 {preexisting.saved}개 있었습니다. 같은 사람이 이어 찍을 때만 계속하세요. 새 사람이면 다른 가명 ID를 입력하세요.
        </p>
      )}
      {needId && (
        <p className="mb-2 text-xs text-amber-700">가명 ID를 먼저 입력하세요. 입력하기 전에는 녹화되지 않습니다.</p>
      )}
      {needFolder && (
        <p className="mb-2 text-xs text-amber-700">저장 폴더를 먼저 고르세요. 폴더를 고르기 전에는 녹화되지 않습니다.</p>
      )}
      {memoryMode && (
        <div className="mb-3 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          <p>
            이 브라우저에서는 폴더에 바로 저장할 수 없어, 찍은 클립을 이 화면에 모아 둡니다. 20클립마다, 그리고 다 찍은 뒤
            아래 버튼으로 zip 파일 하나를 내려받으세요. 내려받기 전에 창을 닫거나 새로고침하면 그 사이 클립이 사라집니다.
            가장 마지막에 받은 zip 하나만 보내면 됩니다.
          </p>
          <button type="button" onClick={downloadZip} disabled={memCount === 0 || !idle}
            className="mt-2 rounded bg-amber-600 px-3 py-1.5 text-white disabled:opacity-40">
            찍은 클립 {memCount}개 내려받기 (zip){unsavedMemory ? ' · 아직 안 받은 클립 있음' : ''}
          </button>
        </div>
      )}
      {canResume && (
        <button type="button" onClick={() => setIndex(resume.index)}
          className="mb-3 rounded border border-blue-300 bg-blue-50 px-3 py-1.5 text-sm text-blue-800 hover:bg-blue-100">
          {resume.index >= plan.length
            ? `이 가명 ID의 클립 ${resume.saved}개가 모두 저장돼 있습니다 (끝으로 이동)`
            : `이어 찍기: ${resume.index + 1}번부터 (이 가명 ID로 저장된 클립 ${resume.saved}개)`}
        </button>
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
            <span>{session.short} · {index + 1} / {plan.length} · {cond ? cond.label : '무발화'} · {item.sentenceId} · {item.rep}회차{currentSaved ? ' · 이미 저장됨(녹화하면 덮어씀)' : ''}</span>
            <span>{item.kind === 'fixed' ? '지정 문장' : item.kind === 'free' ? '자유 문장' : '무발화'}</span>
          </div>
          <p className="text-2xl font-semibold leading-relaxed text-gray-900">{item.text}</p>
          {cond && <p className="mt-1 text-sm text-gray-600">{cond.hint}</p>}
          {KIND_HINT[item.kind] && <p className="mt-1 text-sm text-gray-600">{KIND_HINT[item.kind]}</p>}
          {item.kind === 'free' && (
            <div className="mt-2">
              <input value={spokenText} onChange={(e) => setSpokenText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur(); }}
                disabled={status === 'recording'} placeholder="말할 문장을 먼저 적고 Enter (한글·띄어쓰기만)"
                className="w-full rounded border border-gray-300 px-2 py-1.5 text-base" />
              {spokenText.trim() !== '' && spokenProblem(spokenText) && (
                <p className="mt-1 text-xs text-amber-700">{spokenProblem(spokenText)}</p>
              )}
            </div>
          )}
          <div className="mt-3 flex gap-2">
            <button type="button" onClick={toggle} disabled={saving || (status === 'ready' && blocked) || (status !== 'ready' && status !== 'recording')}
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
        <div className="mb-3 rounded-lg bg-green-50 p-4 text-green-800">
          <p>촬영이 끝났습니다. 이번에 저장한 클립 {saved.length}개.{memoryMode ? ' 위의 내려받기 버튼으로 zip을 꼭 받으세요.' : ''}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            <button type="button" onClick={() => setIndex(plan.length - 1)} disabled={!idle}
              className="rounded border border-green-300 bg-white px-3 py-1.5 text-sm text-green-800 hover:bg-green-100 disabled:opacity-40">
              마지막 클립 다시 찍기
            </button>
            {onBack && (
              <button type="button" onClick={startNextPerson} disabled={!idle}
                className="rounded border border-green-300 bg-white px-3 py-1.5 text-sm text-green-800 hover:bg-green-100 disabled:opacity-40">
                다음 사람 촬영 (같은 폴더·촬영 묶음, 가명 ID 새로 입력)
              </button>
            )}
          </div>
        </div>
      )}

      <div className="mb-3 flex items-center gap-2 text-xs text-gray-600">
        <label className="flex items-center gap-1">
          순번으로 가기
          <input type="number" min={1} max={plan.length} value={jumpText} onChange={(e) => setJumpText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { jumpTo(); e.currentTarget.blur(); } }}
            disabled={!idle} className="w-20 rounded border border-gray-300 px-2 py-0.5" />
        </label>
        <button type="button" onClick={jumpTo} disabled={!idle}
          className="rounded border border-gray-300 px-2 py-0.5 hover:bg-gray-50 disabled:opacity-40">이동</button>
        <span>(다시 찍을 클립의 순번, 1~{plan.length})</span>
      </div>

      {lastMeta && (
        <p className="mb-2 text-xs text-gray-500">
          마지막 저장: {lastMeta.file} · {(lastMeta.bytes / 1024).toFixed(0)}KB · 프레임 {lastMeta.frameCallbacks}개 · 추정 {lastMeta.estimatedFps ? lastMeta.estimatedFps.toFixed(1) : '?'}fps · 최대 간격 {lastMeta.maxGapMs ? lastMeta.maxGapMs.toFixed(0) : '?'}ms
        </p>
      )}
      {warnings.map((w) => <p key={w} className="mb-2 text-sm font-medium text-amber-700">⚠ {w}</p>)}
      {saved.length > 0 && (
        <details className="text-xs text-gray-600">
          <summary>저장 목록 ({saved.length})</summary>
          <ul className="mt-1 max-h-40 overflow-auto">{saved.map((s, k) => <li key={`${s.file}-${k}`}>{s.file} · {(s.bytes / 1024).toFixed(0)}KB</li>)}</ul>
        </details>
      )}

      {modal && <ErrorModal type={modal.type} message={modal.message} onClose={() => setModal(null)} />}
    </div>
  );
}
