// scripts/verify_webcam_wire.py가 로컬 검증 서버 주소를 전달한다.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { ChunkTransport } from '../src/webcamStream.js';
import { ROI_BYTES } from '../src/constants.js';

const ws = new WebSocket(process.argv[2]);
const timeout = setTimeout(() => { ws.close(); throw new Error('wire probe timed out'); }, 15000);
const clips = [[11, 0], [11, 5]]; // 빈 final 및 여러 프레임 final을 모두 실제 송수신한다.
const expected = new Map();
const finals = [];
const transport = new ChunkTransport(ws, {
  onResult: m => { if (m.is_final) {
    assert.equal(m.text, expected.get(m.utterance_id));
    finals.push({ utterance_id: m.utterance_id, frames: m.frame_count, hash_matches: true });
  } },
  onError: error => { clearTimeout(timeout); ws.close(); throw error; },
  onEnd: m => {
    assert.equal(m.total_segments, 2); assert.equal(m.total_frames, 27);
    assert.equal(finals.length, 2);
    clearTimeout(timeout); ws.close();
    console.log(JSON.stringify({ passed: true, model: 'sha256_stub', final_results: finals, session_end: m }));
  },
});
ws.onopen = () => ws.send(JSON.stringify({ type: 'SESSION_START', session_id: 'node-wire-probe', client_ts: Date.now(), protocol_version: 2 }));
ws.onmessage = event => {
  const message = JSON.parse(event.data);
  transport.receive(message);
  if (message.type === 'SESSION_READY') {
    let index = 0;
    for (const [i, sizes] of clips.entries()) {
      let start = i * 2000;
      const digest = createHash('sha256');
      for (const [j, count] of sizes.entries()) {
        const frames = Array.from({ length: count }, () => new Uint8Array(ROI_BYTES).fill(++index));
        frames.forEach(frame => digest.update(frame));
        transport.enqueue({ utterance_id: i + 1, frames, window_start_ms: start,
          window_end_ms: start + count * 40, is_final: j === sizes.length - 1 });
        start += count * 40;
      }
      expected.set(i + 1, digest.digest('hex'));
    }
    transport.end();
  }
};
