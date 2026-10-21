// Windows: PLAYWRIGHT_PATH를 사용 가능한 Playwright 패키지 경로로 설정해 실행.
// 카메라·랜드마크·서버 응답은 명시적 대체물. 실제 React 화면/영상 callback/정지 수명주기는 실행한다.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../..');
const checks = [];

const visionStub = `
export const FilesetResolver = { forVisionTasks: async () => ({}) };
export const FaceLandmarker = { createFromOptions: async () => {
  await new Promise(r => setTimeout(r, window.__landmarkDelay || 0));
  window.__created++;
  return { close() { window.__closed++; }, detectForVideo() {
    const p = Array.from({length:478},()=>({x:.5,y:.6}));
    const roi = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146];
    roi.forEach((id,i)=>{p[id]={x:.4+(i%2)*.2,y:.55+(i%3)*.05}});
    [13,14,312,311,310,415].forEach(id=>{p[id]={x:.5,y:.6-(window.__mouth?.1:0)}});
    [17,18,84,181,180,314].forEach(id=>{p[id]={x:.5,y:.6+(window.__mouth?.1:0)}});
    p[234]={x:.1,y:.5};p[454]={x:.9,y:.5};
    return {faceLandmarks:window.__faceMissing?[]:[p]};
  }};
}};`;

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.route('**/*mediapipe_tasks-vision*', route => route.fulfill({ contentType: 'application/javascript', body: visionStub }));
  await page.addInitScript(() => {
    window.__mode = 'no-model'; window.__mouth = false; window.__streams = [];
    window.__ws = []; window.__created = 0; window.__closed = 0;
    class FakeSocket {
      constructor() {
        this.readyState = 0; this.records = []; window.__ws.push(this);
        setTimeout(() => { if (this.readyState === 3) return; this.readyState = 1; this.onopen?.(); }, 5);
      }
      emit(data) { setTimeout(() => this.onmessage?.({ data: JSON.stringify(data) }), 15); }
      send(data) {
        if (typeof data !== 'string') {
          this.records.push({ type: 'BINARY', chunk: new DataView(data).getUint32(0, false) });
          this.received++;
          if (this.received === this.chunk.frame_count) this.complete();
          return;
        }
        const m = JSON.parse(data); this.records.push(m);
        if (m.type === 'SESSION_START') {
          if (window.__mode === 'no-model') this.emit({type:'ERROR',code:'MODEL_NOT_READY',message:'학습된 v2 모델이 아직 없습니다. 모델 학습을 완료한 뒤 웹캠 인식을 시작하세요.'});
          else this.emit({ type:'SESSION_READY',protocol_version:2,fps:25,chunk_frames:11,max_utterance_frames:200 });
        } else if (m.type === 'CHUNK_START') {
          this.chunk = m; this.received = 0;
          if (m.frame_count === 0) this.complete();
        } else if (m.type === 'UTTERANCE_ABORT') this.emit({type:'ABORTED',utterance_id:m.utterance_id});
        else if (m.type === 'SESSION_END') this.emit({type:'SESSION_END',total_segments:1});
      }
      complete() {
        const c = this.chunk;
        this.emit({type:'PARTIAL',chunk_id:c.chunk_id,utterance_id:c.utterance_id,is_final:c.is_final,
          text:c.is_final?'검증 완료':'검증 중',confidence:c.is_final?1:null});
      }
      close() { this.readyState = 3; setTimeout(() => this.onclose?.(), 0); }
    }
    FakeSocket.OPEN = 1; window.WebSocket = FakeSocket;
    navigator.mediaDevices.getUserMedia = async () => {
      window.__cameraRequested = true;
      await new Promise(resolve => setTimeout(resolve, window.__cameraDelay || 0));
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
      const context = canvas.getContext('2d'); let index = 0;
      const timer = setInterval(() => { context.fillStyle = `rgb(${index++%255},90,120)`; context.fillRect(0,0,640,480); }, 30);
      const stream = canvas.captureStream(30); window.__streams.push(stream);
      for (const track of stream.getTracks()) { const stop = track.stop.bind(track); track.stop = () => { clearInterval(timer); stop(); }; }
      return stream;
    };
  });
  try {
    await page.goto(process.env.UI_TEST_URL || 'http://127.0.0.1:5175/');
    await page.getByRole('button', { name: '웹캠 모드로 전환', exact: true }).click();
    await page.getByRole('button', { name: '시작', exact: true }).click();
    await page.getByRole('dialog').waitFor();
    assert.match(await page.getByRole('dialog').innerText(), /학습된 v2 모델이 아직 없습니다/);
    assert.equal(await page.evaluate(() => !!window.__cameraRequested), false);
    checks.push('missing-model reason displayed before camera request');
    await page.getByRole('button', { name: '닫기', exact: true }).last().click();
    await page.evaluate(() => { window.__mode = 'ready'; });
    await page.getByRole('button', { name: '시작', exact: true }).click();
    await page.getByText('캘리브레이션 완료 ✓', { exact: true }).waitFor();
    await page.evaluate(() => { window.__mouth = true; });
    await page.getByText('검증 중', { exact: true }).waitFor();
    await page.getByRole('button', { name: '정지', exact: true }).click();
    await page.getByText('검증 완료', { exact: true }).waitFor();
    await page.getByRole('button', { name: '시작', exact: true }).waitFor();
    const first = await page.evaluate(() => {
      const records = window.__ws.at(-1).records;
      return { records, stopped: window.__streams.every(s=>s.getTracks().every(t=>t.readyState==='ended')), created:window.__created,closed:window.__closed };
    });
    assert.equal(first.stopped, true); assert.equal(first.created, first.closed);
    assert.equal(first.records.at(-1).type, 'SESSION_END');
    assert.equal(first.records.filter(m=>m.type==='CHUNK_START').at(-1).is_final, true);
    checks.push('calibration, partial subtitle, graceful stop, final subtitle, resource cleanup');
    await page.evaluate(() => { window.__mouth = false; });
    await page.getByRole('button', { name: '시작', exact: true }).click();
    await page.getByText('캘리브레이션 완료 ✓', { exact: true }).waitFor();
    await page.evaluate(() => { window.__mouth = true; });
    await page.getByText('검증 중', { exact: true }).waitFor();
    await page.evaluate(() => window.__ws.at(-1).onclose());
    await page.getByRole('dialog').waitFor();
    assert.match(await page.getByRole('dialog').innerText(), /연결이 끊겨/);
    assert.equal(await page.getByText('검증 중', { exact: true }).count(), 0);
    checks.push('restart resets state; disconnect removes unfinished subtitle and displays reason');
    await page.getByRole('button', { name: '닫기', exact: true }).last().click();
    await page.evaluate(() => { window.__cameraDelay = 400; window.__cameraRequested = false; });
    await page.getByRole('button', { name: '시작', exact: true }).click();
    await page.waitForFunction(() => window.__cameraRequested);
    await page.getByRole('button', { name: '정지', exact: true }).click();
    await page.waitForTimeout(550);
    assert.equal(await page.evaluate(() => window.__streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))), true);
    checks.push('stop during permission setup closes late-arriving camera tracks');
    await page.evaluate(() => { window.__cameraDelay = 0; window.__landmarkDelay = 500; });
    await page.getByRole('button', { name: '시작', exact: true }).click();
    await page.waitForFunction(() => window.__streams.at(-1)?.getTracks()[0].readyState==='live');
    await page.waitForTimeout(150);
    await page.getByRole('button', { name: '영상 업로드 모드로 전환', exact: true }).click();
    await page.waitForTimeout(650);
    assert.equal(await page.evaluate(() => window.__streams.every(s=>s.getTracks().every(t=>t.readyState==='ended'))), true);
    assert.equal(await page.evaluate(() => window.__created===window.__closed), true);
    checks.push('unmount during model setup closes late-created detector and camera');
    assert.deepEqual(pageErrors, []);
    fs.writeFileSync(path.join(root,'tmp/step3-ui-result.json'), JSON.stringify({passed:true,checks,page_errors:pageErrors,kind:'headless Edge; camera, landmark detector and responses mocked'},null,2));
    console.log(JSON.stringify({passed:true,checks}));
  } catch (error) {
    await page.screenshot({path:path.join(root,'tmp/step3-ui-failure.png'),fullPage:true});
    console.error('Page errors:', pageErrors); throw error;
  } finally { await browser.close(); }
})();
