import test from 'node:test';
import assert from 'node:assert/strict';
import { ChunkTransport, SpeechSegmenter, TimestampResampler } from '../src/webcamStream.js';
import { buildBinaryFrame, computeMouthRatio, computeRoiBBox } from '../src/webcamUtils.js';
import { ACK_TIMEOUT_MS, FACE_WIDTH_IDX, ROI_BYTES, ROI_CROP_IDX, VVAD_IDX_BOT, VVAD_IDX_TOP } from '../src/constants.js';

for (const fps of [15, 25, 30, 60]) {
  test(`${fps}fps source produces a uniform 25fps timeline, with nearest samples`, () => {
    const sampler = new TimestampResampler();
    const output = [];
    for (let n = 0; n <= fps * 3; n++) output.push(...sampler.push(n * 1000 / fps, n).samples);
    assert.deepEqual(output.map(s => s.timestampMs), Array.from({ length: 76 }, (_, i) => i * 40));
    for (const s of output) assert.ok(Math.abs(s.value * 1000 / fps - s.timestampMs) <= 500 / fps + 0.001);
  });
}

test('duplicate decoded frames are ignored; long gaps do not fabricate a long frozen video', () => {
  const sampler = new TimestampResampler();
  assert.equal(sampler.push(0, 'a').samples.length, 1);
  assert.equal(sampler.push(0, 'b').samples.length, 0);
  const result = sampler.push(2000, 'c');
  assert.equal(result.discontinuity, true);
  assert.deepEqual(result.samples, [{ timestampMs: 2000, value: 'c' }]);
});

function speechRig() {
  const chunks = [], aborted = [];
  const segment = new SpeechSegmenter({ onChunk: c => chunks.push(c), onAbort: id => aborted.push(id),
    thresholds: { tOpen: 0.01, tClose: 0.005, tVar: 0.000001 } });
  let index = 0;
  const feed = (ratio, count = 1, missing = false) => {
    for (let i = 0; i < count; i++) {
      segment.push({ timestampMs: index * 40, value: missing ? null : { ratio, roi: new Uint8Array(ROI_BYTES).fill(index % 256) } });
      index++;
    }
  };
  return { segment, chunks, aborted, feed };
}

test('speech start retains preroll and triggering frame, stop retains the tail without duplication', () => {
  const { feed, segment, chunks } = speechRig();
  feed(0, 10); feed(0.2, 20); segment.finish();
  const values = chunks.flatMap(c => c.frames.map(f => f[0]));
  assert.deepEqual(values, Array.from({ length: 27 }, (_, i) => i + 3));
  assert.equal(chunks[0].window_start_ms, 120);
  assert.equal(chunks.at(-1).window_end_ms, 1200);
  assert.equal(chunks.at(-1).is_final, true);
  assert.equal(chunks.filter(c => c.is_final).length, 1);
  for (let i = 1; i < chunks.length; i++) assert.equal(chunks[i].window_start_ms, chunks[i-1].window_end_ms);
});

test('a stop at an exact chunk boundary sends an empty final control', () => {
  const { feed, segment, chunks } = speechRig();
  feed(0, 10); feed(0.2, 15); segment.finish(); // 7 preroll + 15 speech = 22
  assert.equal(chunks.at(-1).frames.length, 0);
  assert.equal(chunks.at(-1).is_final, true);
  assert.equal(chunks.flatMap(c => c.frames).length, 22);
});

test('short false starts never enter the server buffer', () => {
  const { feed, segment, chunks } = speechRig();
  feed(0, 10); feed(0.2, 3); segment.finish();
  assert.equal(chunks.length, 0);
});

test('sustained mouth closure finishes speech exactly once', () => {
  const { feed, chunks } = speechRig();
  feed(0, 10); feed(0.2, 20); feed(0, 50);
  assert.equal(chunks.filter(c => c.is_final).length, 1);
});

test('tracking loss aborts sent speech and removes previous preroll', () => {
  const { feed, segment, chunks, aborted } = speechRig();
  feed(0, 10); feed(0.2, 20); feed(0, 3, true);
  assert.deepEqual(aborted, [1]);
  assert.equal(segment.active, false);
  feed(0, 10); feed(0.2, 20); segment.finish();
  const next = chunks.filter(c => c.utterance_id === 2);
  assert.ok(next.length);
  assert.ok(next[0].window_start_ms >= 33 * 40);
});

test('one utterance never exceeds 200 frames', () => {
  const { feed, chunks } = speechRig();
  feed(0, 10); feed(0.2, 240);
  const first = chunks.filter(c => c.utterance_id === 1);
  assert.equal(first.flatMap(c => c.frames).length, 200);
  assert.equal(first.at(-1).is_final, true);
});

function transportRig() {
  const sent = [], errors = [], results = [], ends = [];
  const ws = { readyState: 1, send: data => sent.push(typeof data === 'string' ? JSON.parse(data) : data) };
  const transport = new ChunkTransport(ws, { onError: e => errors.push(e.message), onResult: m => results.push(m), onEnd: m => ends.push(m) });
  transport.receive({ type: 'SESSION_READY' });
  const makeChunk = (start, count, final = false, id = 1) => ({ utterance_id: id,
    window_start_ms: start, window_end_ms: start + count * 40, is_final: final,
    frames: Array.from({ length: count }, () => new Uint8Array(ROI_BYTES)) });
  const ack = () => transport.receive({ type: 'PARTIAL', chunk_id: transport.pending.chunk_id,
    utterance_id: transport.pending.utterance_id, is_final: transport.pending.is_final });
  return { transport, sent, errors, results, ends, makeChunk, ack };
}

test('one chunk in flight, and SESSION_END follows the final acknowledgement', () => {
  const r = transportRig();
  r.transport.enqueue(r.makeChunk(0, 3)); r.transport.enqueue(r.makeChunk(120, 2, true)); r.transport.end();
  assert.equal(r.sent.length, 4); // control + all 3 binaries
  assert.equal(r.sent[0].frame_count, 3);
  r.ack();
  assert.equal(r.sent[4].is_final, true);
  assert.equal(r.sent.length, 7);
  r.ack();
  assert.equal(r.sent.at(-1).type, 'SESSION_END');
  r.transport.receive({ type: 'SESSION_END' });
  assert.equal(r.ends.length, 1);
  assert.equal(r.errors.length, 0);
});

test('empty final sends no binary but still waits for final response', () => {
  const r = transportRig();
  r.transport.enqueue(r.makeChunk(0, 11)); r.transport.enqueue(r.makeChunk(440, 0, true));
  r.ack();
  assert.equal(r.sent.length, 13);
  assert.equal(r.sent.at(-1).frame_count, 0);
  r.ack(); r.transport.dispose();
});

test('slow inference has bounded pending memory and gives a visible error', () => {
  const r = transportRig();
  for (let i = 0; i < 20; i++) r.transport.enqueue(r.makeChunk(i * 440, 11));
  assert.equal(r.transport.closed, true);
  assert.equal(r.errors.length, 1);
  assert.equal(r.transport.queue.length, 0);
});

test('abort removes unsent frames and clears sent server state before next speech', () => {
  const r = transportRig();
  r.transport.enqueue(r.makeChunk(0, 11)); r.transport.enqueue(r.makeChunk(440, 11));
  r.transport.abort(1); r.transport.enqueue(r.makeChunk(1000, 1, true, 2));
  r.ack();
  assert.equal(r.sent.at(-1).type, 'UTTERANCE_ABORT');
  r.transport.receive({ type: 'ABORTED', utterance_id: 1 });
  assert.equal(r.sent.at(-2).chunk_id, 2); // discarded unsent chunk does not consume an ID
  r.ack(); r.transport.dispose();
});

test('out-of-order acknowledgement terminates the stream', () => {
  const r = transportRig(); r.transport.enqueue(r.makeChunk(0, 1));
  r.transport.receive({ type: 'PARTIAL', chunk_id: 99, utterance_id: 1, is_final: false });
  assert.equal(r.errors.length, 1);
});

test('missing server response times out without leaving a queue', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const r = transportRig(); r.transport.enqueue(r.makeChunk(0, 1));
  t.mock.timers.tick(ACK_TIMEOUT_MS + 1);
  assert.equal(r.errors.length, 1);
  assert.equal(r.transport.closed, true);
});

test('binary header and RGB byte count exactly match the server contract', () => {
  const binary = buildBinaryFrame(257, new Uint8Array(ROI_BYTES).fill(7));
  assert.equal(binary.byteLength, ROI_BYTES + 8);
  assert.equal(new DataView(binary).getUint32(0, false), 257);
  assert.deepEqual([...new Uint8Array(binary).slice(4, 9)], [1, 0, 0, 0, 7]);
  assert.throws(() => buildBinaryFrame(1, new Float32Array(ROI_BYTES)));
});

test('mouth ratio uses pixel aspect ratio and crop bounds use floor/ceil', () => {
  const landmarks = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
  for (const i of VVAD_IDX_TOP) landmarks[i] = { x: 0.5, y: 0.45 };
  for (const i of VVAD_IDX_BOT) landmarks[i] = { x: 0.5, y: 0.55 };
  landmarks[FACE_WIDTH_IDX[0]] = { x: 0.25, y: 0.5 };
  landmarks[FACE_WIDTH_IDX[1]] = { x: 0.75, y: 0.5 };
  assert.ok(Math.abs(computeMouthRatio(landmarks, 640, 480) - 0.15) < 1e-9);
  for (const i of ROI_CROP_IDX) landmarks[i] = { x: 0.2, y: 0.2 };
  landmarks[ROI_CROP_IDX[0]] = { x: 0.3, y: 0.3 };
  assert.deepEqual(computeRoiBBox(landmarks, 640, 480), { x: 115, y: 86, w: 90, h: 68 });
});
