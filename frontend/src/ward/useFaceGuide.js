// useFaceGuide.js — 환자 화면 카메라 미리보기의 얼굴 위치 안내 (브라우저 안 얼굴 검출, 1초에 약 4번).
// 모델은 frontend/public/models/face_detection_short_range.tflite(약 230KB, MediaPipe 0.10.21 파이썬 패키지와 같은 파일,
// Apache-2.0)를 쓰고, wasm은 기존처럼 /mediapipe/wasm 에서 읽는다. 인터넷 없이 동작한다.
// 모델이나 wasm을 불러오지 못하면 안내만 쉬고(status 'unavailable') 녹화·인식에는 영향이 없다.

import { useEffect, useRef, useState } from 'react';

import { MEDIAPIPE_WASM } from '../constants';
import { withBase } from '../mediapipeAssets';
import { framingAdvice, stabilizeAdvice } from './faceGuide';

/** 얼굴 검출 모델 경로 (앱 기준). */
export const FACE_DETECTOR_MODEL_PATH = 'models/face_detection_short_range.tflite';
/** 검출 간격(ms). 입술 녹화와 서버 인식을 방해하지 않게 넉넉히 둔다. */
export const FACE_GUIDE_INTERVAL_MS = 250;

/**
 * 검출기를 닫는다 (이미 닫혔거나 없으면 그대로).
 * @param {any} detector
 */
function closeQuietly(detector) {
  try {
    detector?.close?.();
  } catch {
    /* 이미 닫힘 */
  }
}

/**
 * 카메라 미리보기 <video>에서 얼굴 위치 안내를 만든다.
 * @param {{current: HTMLVideoElement|null}} videoRef 미리보기 video ref (useRecorder의 videoRef).
 * @param {boolean} active 카메라가 켜져 있고 안내가 필요할 때 true.
 * @returns {{status: 'off'|'loading'|'ready'|'unavailable', advice: object|null}}
 */
export default function useFaceGuide(videoRef, active) {
  const [status, setStatus] = useState('off');
  const [advice, setAdvice] = useState(null);
  const detectorRef = useRef(null);
  const loadingRef = useRef(null);
  const stableRef = useRef(null);
  const disposedRef = useRef(false);

  // 화면을 내릴 때 검출기를 닫는다. 아직 불러오는 중이면 다 불러온 뒤 바로 닫는다(아래 load). 다시 붙으면 새로 불러온다.
  useEffect(() => {
    disposedRef.current = false;
    return () => {
      disposedRef.current = true;
      closeQuietly(detectorRef.current);
      detectorRef.current = null;
      loadingRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!active) {
      setAdvice(null);
      stableRef.current = null;
      return undefined;
    }
    let alive = true;
    let timer = null;

    const load = async () => {
      if (detectorRef.current) return detectorRef.current;
      if (!loadingRef.current) {
        const pending = (async () => {
          const { FaceDetector, FilesetResolver } = await import('@mediapipe/tasks-vision');
          const resolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
          const detector = await FaceDetector.createFromOptions(resolver, {
            baseOptions: { modelAssetPath: withBase(FACE_DETECTOR_MODEL_PATH), delegate: 'CPU' },
            runningMode: 'VIDEO',
            minDetectionConfidence: 0.5,
          });
          if (disposedRef.current || loadingRef.current !== pending) {
            closeQuietly(detector); // 그사이 화면을 내렸다
            throw new Error('disposed');
          }
          detectorRef.current = detector;
          return detector;
        })();
        loadingRef.current = pending;
      }
      return loadingRef.current;
    };

    const tick = () => {
      if (!alive) return;
      try {
        const video = videoRef?.current;
        const detector = detectorRef.current;
        if (detector && video && video.readyState >= 2 && video.videoWidth > 0 && !document.hidden) {
          const result = detector.detectForVideo(video, performance.now());
          const next = framingAdvice(result?.detections, video.videoWidth, video.videoHeight);
          stableRef.current = stabilizeAdvice(stableRef.current, next);
          setAdvice(stableRef.current.shown);
        }
      } catch {
        /* 한 번 실패는 다음 차례에 다시 시도 */
      }
      timer = setTimeout(tick, FACE_GUIDE_INTERVAL_MS);
    };

    setStatus((s) => (s === 'ready' ? s : 'loading'));
    load()
      .then(() => {
        if (!alive) return;
        setStatus('ready');
        tick();
      })
      .catch(() => {
        if (disposedRef.current) return;
        loadingRef.current = null;
        if (alive) setStatus('unavailable');
      });
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [active, videoRef]);

  return { status: active ? status : 'off', advice: active ? advice : null };
}

/**
 * 미리보기 칸과 영상 크기 (상자를 object-fit: cover에 맞춰 그리는 데 쓴다).
 * @param {HTMLVideoElement|null} video
 * @returns {{videoW:number, videoH:number, boxW:number, boxH:number}|undefined}
 */
export function previewGeometry(video) {
  try {
    if (!video) return undefined;
    return { videoW: video.videoWidth, videoH: video.videoHeight, boxW: video.clientWidth, boxH: video.clientHeight };
  } catch {
    return undefined;
  }
}
