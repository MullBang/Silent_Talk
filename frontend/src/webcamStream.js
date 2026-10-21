// 시간축·발화·전송 상태는 React 렌더링과 분리해 같은 코드로 재현 검사한다.
import {
  ACK_TIMEOUT_MS, CHUNK_FRAMES, CLOSE_HOLD_FRAMES, DEFAULT_THRESHOLDS,
  EMA_ALPHA, FRAME_MS, MAX_PENDING_FRAMES, MAX_SOURCE_GAP_MS,
  MAX_UTTERANCE_FRAMES, MIN_SPEECH_FRAMES, MISS_WARN_FRAMES,
  PRE_ROLL_FRAMES, VAR_WINDOW,
} from './constants.js';
import { buildBinaryFrame, variance } from './webcamUtils.js';

/** 실제 영상 시각의 양옆 프레임 중 가까운 것을 선택해 균일한 40ms 축을 만든다. */
export class TimestampResampler {
  constructor() { this.previous = null; this.nextMs = null; }

  push(timestampMs, value) {
    if (!Number.isFinite(timestampMs) || timestampMs < 0) throw new Error('잘못된 영상 시각');
    if (this.previous && timestampMs <= this.previous.timestampMs) return { samples: [], discontinuity: false };
    const discontinuity = !!this.previous && timestampMs - this.previous.timestampMs > MAX_SOURCE_GAP_MS;
    if (!this.previous || discontinuity) {
      this.previous = { timestampMs, value };
      this.nextMs = Math.ceil(timestampMs / FRAME_MS) * FRAME_MS;
    }
    const samples = [];
    while (this.nextMs <= timestampMs + 1e-6) {
      const source = this.nextMs - this.previous.timestampMs < timestampMs - this.nextMs
        ? this.previous : { timestampMs, value };
      samples.push({ timestampMs: this.nextMs, value: source.value });
      this.nextMs += FRAME_MS;
    }
    this.previous = { timestampMs, value };
    return { samples, discontinuity };
  }
}

/** 25fps 입력에서 V-VAD·시작 프레임 포함 8프레임 버퍼·유한 길이 발화를 구성한다. */
export class SpeechSegmenter {
  constructor({ onChunk, onAbort = () => {}, onSpeaking = () => {}, thresholds = DEFAULT_THRESHOLDS }) {
    Object.assign(this, { onChunk, onAbort, onSpeaking, thresholds });
    this.utteranceId = 0;
    this.ema = 0;
    this.ratios = [];
    this.preRoll = [];
    this.active = false;
    this.pending = [];
    this.lastGood = null;
    this.misses = 0;
  }

  push({ timestampMs, value }) {
    if (!value?.roi || !Number.isFinite(value.ratio)) {
      this.misses += 1;
      if (this.misses >= MISS_WARN_FRAMES || !this.lastGood) { this.abort(); return; }
      // 1~2개의 짧은 검출 누락은 마지막 유효 ROI로 25fps 시간축을 유지한다.
      value = this.lastGood;
    } else { this.misses = 0; this.lastGood = value; }
    this.ema = EMA_ALPHA * value.ratio + (1 - EMA_ALPHA) * this.ema;
    this.ratios.push(this.ema);
    if (this.ratios.length > VAR_WINDOW) this.ratios.shift();
    const sample = { timestampMs, roi: value.roi };
    if (!this.active) {
      this.preRoll.push(sample);
      if (this.preRoll.length > PRE_ROLL_FRAMES) this.preRoll.shift();
      if (this.ema <= this.thresholds.tOpen || variance(this.ratios) <= this.thresholds.tVar) return;
      this.active = true;
      this.utteranceId += 1;
      this.pending = this.preRoll;
      this.preRoll = [];
      this.totalFrames = this.pending.length;
      this.speechFrames = 1;
      this.closeCounter = 0;
      this.onSpeaking(true);
    } else {
      this.pending.push(sample);
      this.totalFrames += 1;
      this.speechFrames += 1;
    }
    this.nextMs = timestampMs + FRAME_MS;
    this.closeCounter = this.ema < this.thresholds.tClose ? this.closeCounter + 1 : 0;
    if (this.closeCounter >= CLOSE_HOLD_FRAMES || this.totalFrames >= MAX_UTTERANCE_FRAMES) {
      this.finish();
    } else if (this.speechFrames >= MIN_SPEECH_FRAMES) {
      while (this.pending.length >= CHUNK_FRAMES) this.emit(CHUNK_FRAMES, false);
    }
  }

  emit(count, final) {
    const samples = this.pending.splice(0, count);
    const start = samples.length ? samples[0].timestampMs : this.nextMs;
    this.onChunk({ utterance_id: this.utteranceId, frames: samples.map(s => s.roi),
      window_start_ms: start, window_end_ms: start + count * FRAME_MS, is_final: final });
  }

  finish() {
    if (!this.active) return;
    if (this.speechFrames >= MIN_SPEECH_FRAMES) {
      while (this.pending.length > CHUNK_FRAMES) this.emit(CHUNK_FRAMES, false);
      this.emit(this.pending.length, true); // 빈 final도 서버의 누적 발화를 확정한다.
    }
    this.pending = [];
    this.preRoll = [];
    this.active = false;
    this.onSpeaking(false);
  }

  abort() {
    if (this.active) this.onAbort(this.utteranceId);
    this.active = false;
    this.pending = [];
    this.preRoll = [];
    this.ratios = [];
    this.ema = 0;
    this.lastGood = null;
    this.onSpeaking(false);
  }
}

/** 서버 응답 전 다음 청크를 보내지 않으며, 대기 메모리와 응답 시간을 제한한다. */
export class ChunkTransport {
  constructor(ws, { onResult = () => {}, onError = () => {}, onEnd = () => {} } = {}) {
    Object.assign(this, { ws, onResult, onError, onEnd });
    this.queue = [];
    this.pending = null;
    this.sentUtterances = new Set();
    this.chunkId = 0;
    this.ready = false;
    this.ending = false;
    this.closed = false;
  }

  enqueue(chunk) {
    if (this.closed || this.ending) return;
    const queued = this.queue.reduce((n, item) => n + (item.frames?.length || 0), 0);
    if (queued + (this.pending?.frames?.length || 0) + chunk.frames.length > MAX_PENDING_FRAMES) {
      this.fail('처리 속도가 촬영 속도를 따라가지 못해 중단했습니다. 모델 처리 속도를 개선한 뒤 다시 시도하세요.');
      return;
    }
    this.queue.push({ kind: 'chunk', ...chunk });
    this.pump();
  }

  abort(utteranceId) {
    this.queue = this.queue.filter(item => item.utterance_id !== utteranceId);
    if (this.sentUtterances.has(utteranceId)) this.queue.push({ kind: 'abort', utterance_id: utteranceId });
    this.pump();
  }

  end() { this.ending = true; this.pump(); }

  pump() {
    if (this.closed || !this.ready || this.pending) return;
    const item = this.queue.shift() || (this.ending ? { kind: 'end' } : null);
    if (!item) return;
    this.pending = item;
    try {
      if (this.ws.readyState !== 1) throw new Error('연결이 끊겼습니다. 다시 시작하세요.');
      if (item.kind === 'chunk') {
        item.chunk_id = ++this.chunkId;
        const { utterance_id, window_start_ms, window_end_ms, is_final, chunk_id } = item;
        this.ws.send(JSON.stringify({ type: 'CHUNK_START', utterance_id, window_start_ms,
          window_end_ms, is_final, chunk_id, frame_count: item.frames.length }));
        for (const roi of item.frames) this.ws.send(buildBinaryFrame(chunk_id, roi));
        this.sentUtterances.add(utterance_id);
      } else if (item.kind === 'abort') {
        this.ws.send(JSON.stringify({ type: 'UTTERANCE_ABORT', utterance_id: item.utterance_id }));
      } else this.ws.send(JSON.stringify({ type: 'SESSION_END' }));
      this.timer = setTimeout(() => this.fail('서버 응답 시간이 초과되었습니다. 서버와 모델 상태를 확인하세요.'), ACK_TIMEOUT_MS);
    } catch (error) { this.fail(error.message); }
  }

  receive(message) {
    if (this.closed) return;
    if (message.type === 'ERROR') { this.fail(message.message); return; }
    if (message.type === 'SESSION_READY') { this.ready = true; this.pump(); return; }
    const pending = this.pending;
    if (message.type === 'PARTIAL' && pending?.kind === 'chunk' &&
        message.chunk_id === pending.chunk_id && message.utterance_id === pending.utterance_id &&
        message.is_final === pending.is_final) {
      if (message.is_final) this.sentUtterances.delete(message.utterance_id);
      this.onResult(message);
    } else if (message.type === 'ABORTED' && pending?.kind === 'abort' && message.utterance_id === pending.utterance_id) {
      this.sentUtterances.delete(message.utterance_id);
    } else if (message.type === 'SESSION_END' && pending?.kind === 'end') {
      this.dispose(); this.onEnd(message); return;
    } else { this.fail('서버 응답 순서가 일치하지 않습니다. 다시 시작하세요.'); return; }
    clearTimeout(this.timer);
    this.pending = null;
    this.pump();
  }

  fail(message) { if (!this.closed) { this.dispose(); this.onError(new Error(message)); } }
  dispose() { this.closed = true; clearTimeout(this.timer); this.queue = []; this.pending = null; this.sentUtterances.clear(); }
}
