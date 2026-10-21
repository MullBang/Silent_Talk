// useRecorder.js — 브라우저 MediaRecorder 녹화 훅 (영상 전용, 음성 트랙 없음).
// CapturePage(3단계 파일럿 촬영)와 RecordPage(6단계 녹화→분석)가 함께 쓴다.
// 카메라 실제 설정(getSettings)과 프레임 시각(requestVideoFrameCallback mediaTime)을 함께 기록한다.
// 카메라 트랙이 끊기면(USB 분리·다른 앱 점유·절전 등) status를 'error'로 바꾸고 고정 한국어 문구를 낸다.

import { useCallback, useEffect, useRef, useState } from 'react';

import {
  AUDIO_TRACK_ERROR, CAMERA_LOST_ERROR, INSECURE_CONTEXT_ERROR, RECORDER_UNSUPPORTED_ERROR, RECORD_START_ERROR,
  cameraErrorType, cameraOpenErrorMessage, clientCameraError, isStreamLost, logCameraError,
} from './cameraErrors.js';

/** 선호 순서대로 지원되는 MIME을 고른다. 없으면 null. */
export function pickMimeType(candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']) {
  if (typeof MediaRecorder === 'undefined') return null;
  return candidates.find((m) => MediaRecorder.isTypeSupported(m)) || null;
}

/** MIME에서 파일 확장자를 정한다. */
export function extensionFor(mime) {
  return mime && mime.startsWith('video/mp4') ? 'mp4' : 'webm';
}

/** 프레임 시각 로그에서 요약 통계를 만든다 (순수 함수, 테스트 가능). */
export function summarizeFrameLog(log) {
  if (!log || log.length < 2) return { frameCallbacks: log ? log.length : 0, meanGapMs: null, maxGapMs: null, estimatedFps: null };
  let maxGap = 0;
  for (let i = 1; i < log.length; i += 1) maxGap = Math.max(maxGap, log[i] - log[i - 1]);
  const span = log[log.length - 1] - log[0];
  const meanGap = span / (log.length - 1);
  return { frameCallbacks: log.length, meanGapMs: meanGap, maxGapMs: maxGap, estimatedFps: meanGap > 0 ? 1000 / meanGap : null };
}

/**
 * 영상 전용 녹화 훅.
 *
 * 오류 상태: status==='error'이면 error(고정 한국어 문구 또는 null), errorType(모달 유형),
 * errorSeq(오류가 날 때마다 1씩 증가)를 함께 준다. error가 null이면 화면은 모달 기본 문구를 쓴다.
 * 녹화 중 카메라가 끊기면 잘린 녹화는 onResult로 넘기지 않고 버린다.
 *
 * @param {object} opts
 * @param {number} [opts.maxSeconds=8] 자동 종료 시간(초).
 * @param {{width:number,height:number,frameRate:number}} [opts.video] getUserMedia 제약.
 * @param {(result:{blob:Blob,mime:string,meta:object}) => void} [opts.onResult] 녹화가 끝날 때(수동·자동 모두) 호출.
 */
export default function useRecorder({ maxSeconds = 8, video = { width: 1280, height: 720, frameRate: 30 }, onResult } = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recRef = useRef(null);
  const chunksRef = useRef([]);
  const frameLogRef = useRef([]);
  const vfcRef = useRef(null);
  const startedRef = useRef(0);
  const timerRef = useRef(null);
  const tickRef = useRef(null);
  // 현재 스트림의 카메라 끊김 여부 (open마다 초기화).
  const lostRef = useRef(false);
  // open/close 세대 번호: 카메라 열기 도중 close(화면 이탈·StrictMode 재마운트)되면 늦게 열린 스트림을 즉시 닫는다.
  const genRef = useRef(0);
  const onResultRef = useRef(onResult);
  onResultRef.current = onResult;
  const [status, setStatus] = useState('idle'); // idle | opening | ready | recording | error
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null); // 'PERMISSION_DENIED' | 'DEVICE_ERROR' | null
  const [errorSeq, setErrorSeq] = useState(0);
  const [settings, setSettings] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(0);

  const stopFrameLog = useCallback(() => {
    try {
      const v = videoRef.current;
      if (vfcRef.current != null && v?.cancelVideoFrameCallback) v.cancelVideoFrameCallback(vfcRef.current);
    } catch {
      /* 이미 해제된 video */
    }
    vfcRef.current = null;
  }, []);

  /** 스트림의 모든 트랙을 멈추고 미리보기를 비운다. */
  const releaseStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => {
      try { t.stop(); } catch { /* 이미 종료 */ }
    });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  /** 오류 상태로 전환한다 (message가 null이면 화면이 모달 기본 문구를 쓴다). */
  const fail = useCallback((message, type = 'DEVICE_ERROR') => {
    setError(message);
    setErrorType(type);
    setErrorSeq((s) => s + 1);
    setStatus('error');
  }, []);

  /** 녹화 타이머·프레임 기록을 멈춘다. */
  const clearRecordingTimers = useCallback(() => {
    clearTimeout(timerRef.current);
    clearInterval(tickRef.current);
    stopFrameLog();
  }, [stopFrameLog]);

  /** 카메라 끊김 마무리: 녹화 정리·스트림 해제·오류 상태. */
  const finishLost = useCallback((stream) => {
    lostRef.current = true;
    clearRecordingTimers();
    recRef.current = null;
    if (streamRef.current === stream) releaseStream();
    fail(CAMERA_LOST_ERROR);
  }, [clearRecordingTimers, releaseStream, fail]);

  const close = useCallback(() => {
    genRef.current += 1;
    clearRecordingTimers();
    if (recRef.current && recRef.current.state !== 'inactive') {
      recRef.current.onstop = null;
      try { recRef.current.stop(); } catch { /* 이미 정지됨 */ }
    }
    recRef.current = null;
    releaseStream();
    setStatus('idle');
  }, [clearRecordingTimers, releaseStream]);

  const open = useCallback(async () => {
    genRef.current += 1;
    const gen = genRef.current;
    const stale = () => gen !== genRef.current;
    // 이미 열린 스트림이 있으면 먼저 닫는다 (다시 열기).
    releaseStream();
    lostRef.current = false;
    setError(null);
    setErrorType(null);
    setStatus('opening');
    let stream = null;
    try {
      if (!window.isSecureContext) throw clientCameraError(INSECURE_CONTEXT_ERROR);
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: video.width }, height: { ideal: video.height }, frameRate: { ideal: video.frameRate } },
      });
      if (stale()) { stream.getTracks().forEach((t) => t.stop()); return; }
      if (stream.getAudioTracks().length) {
        stream.getTracks().forEach((t) => t.stop());
        throw clientCameraError(AUDIO_TRACK_ERROR);
      }
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const opened = stream;
      // 트랙 종료(장치 분리·점유·절전) 감시. close/다시 열기로 바뀐 스트림의 늦은 이벤트는 무시한다.
      const onLost = () => {
        if (stale() || streamRef.current !== opened || lostRef.current) return;
        lostRef.current = true;
        logCameraError('카메라 트랙 종료', { name: 'ended', message: 'video track ended or stream inactive' });
        const r = recRef.current;
        if (r && r.state !== 'inactive') {
          // 녹화 중이면 recorder를 멈추고 onstop에서 마무리한다 (잘린 녹화는 버림).
          try { r.stop(); return; } catch { /* 아래에서 바로 마무리 */ }
        }
        finishLost(opened);
      };
      if (track && typeof track.addEventListener === 'function') track.addEventListener('ended', onLost);
      if (typeof stream.addEventListener === 'function') stream.addEventListener('inactive', onLost);
      setSettings({ ...track.getSettings(), label: track.label });
      const v = videoRef.current;
      if (v) { v.srcObject = stream; v.muted = true; await v.play(); }
      if (stale()) return;
      if (lostRef.current) return; // 열기 도중 끊김: onLost가 이미 오류 상태로 바꿨다.
      if (isStreamLost(stream)) { finishLost(stream); return; }
      setStatus('ready');
    } catch (e) {
      if (stale()) return;
      logCameraError('카메라 열기 실패', e);
      if (stream && streamRef.current === stream) releaseStream();
      fail(cameraOpenErrorMessage(e), cameraErrorType(e));
    }
  }, [video.width, video.height, video.frameRate, releaseStream, finishLost, fail]);

  /** 녹화를 시작하지 못했을 때: 끊긴 스트림이면 끊김 처리, 아니면 시작 실패 문구. */
  const failStart = useCallback((stream, err) => {
    logCameraError('녹화 시작/정지 실패', err);
    clearRecordingTimers();
    recRef.current = null;
    if (isStreamLost(stream)) { finishLost(stream); return; }
    if (streamRef.current === stream) releaseStream();
    fail(RECORD_START_ERROR);
  }, [clearRecordingTimers, finishLost, releaseStream, fail]);

  const start = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || (recRef.current && recRef.current.state === 'recording')) return;
    if (lostRef.current || isStreamLost(stream)) { finishLost(stream); return; }
    const mime = pickMimeType();
    if (!mime) { fail(RECORDER_UNSUPPORTED_ERROR); return; }
    chunksRef.current = [];
    frameLogRef.current = [];
    let rec;
    try {
      rec = new MediaRecorder(stream, { mimeType: mime });
    } catch (e) {
      failStart(stream, e);
      return;
    }
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunksRef.current.push(e.data); };
    rec.onstop = () => {
      clearRecordingTimers();
      if (recRef.current === rec) recRef.current = null;
      // 카메라가 끊겨 멈춘 녹화는 잘린 영상이므로 결과로 넘기지 않는다.
      if (lostRef.current || isStreamLost(stream)) {
        if (streamRef.current === stream || !lostRef.current) finishLost(stream);
        return;
      }
      const blob = new Blob(chunksRef.current, { type: mime });
      const log = frameLogRef.current;
      const meta = {
        mime, extension: extensionFor(mime), bytes: blob.size,
        startedAt: new Date(Date.now() - (performance.now() - startedRef.current)).toISOString(),
        durationMs: performance.now() - startedRef.current,
        trackSettings: settings,
        ...summarizeFrameLog(log),
        mediaTimesMs: log,
        userAgent: navigator.userAgent,
        audioTracks: stream.getAudioTracks().length,
      };
      setStatus('ready');
      onResultRef.current?.({ blob, mime, meta });
    };
    recRef.current = rec;
    startedRef.current = performance.now();
    setElapsedMs(0);
    const v = videoRef.current;
    try {
      if (v?.requestVideoFrameCallback) {
        const onFrame = (_now, metadata) => {
          frameLogRef.current.push(Math.round(metadata.mediaTime * 1e6) / 1e3);
          if (recRef.current === rec && rec.state === 'recording') vfcRef.current = v.requestVideoFrameCallback(onFrame);
        };
        vfcRef.current = v.requestVideoFrameCallback(onFrame);
      }
      rec.start(250);
    } catch (e) {
      rec.onstop = null;
      rec.ondataavailable = null;
      failStart(stream, e);
      return;
    }
    setStatus('recording');
    tickRef.current = setInterval(() => setElapsedMs(performance.now() - startedRef.current), 100);
    timerRef.current = setTimeout(() => { if (recRef.current === rec && rec.state === 'recording') rec.stop(); }, maxSeconds * 1000);
  }, [settings, maxSeconds, clearRecordingTimers, finishLost, failStart, fail]);

  const stop = useCallback(() => {
    const rec = recRef.current;
    if (rec && rec.state === 'recording') {
      try { rec.stop(); } catch (e) { failStart(streamRef.current, e); }
    }
  }, [failStart]);

  useEffect(() => () => close(), [close]);

  return { status, error, errorType, errorSeq, videoRef, settings, open, close, start, stop, elapsedMs };
}
