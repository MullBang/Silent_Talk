// build-capture-package.mjs — 조원에게 보낼 웹캠 촬영 패키지(압축 파일)를 만든다.
//
// 만드는 것 (frontend/dist-capture/):
//   웹캠촬영_패키지/웹캠촬영.html      촬영 화면을 HTML 파일 하나로 묶은 것 (서버·인터넷 없이 Chrome/Edge로 연다)
//   웹캠촬영_패키지/촬영가이드.html    docs 따라하기 가이드의 조원용 부분
//   웹캠촬영_패키지/동의서.html        docs 동의서·기록지 문서의 동의서 부분 (인쇄용)
//   웹캠촬영_패키지/먼저_읽어주세요.txt
//   웹캠촬영_패키지_<날짜>.zip       위 폴더를 압축한 것
//
// 사용: cd frontend && npm.cmd run build:capture-package

import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { crc32, deflateRawSync } from 'node:zlib';

import { build } from 'vite';

const FRONTEND = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = resolve(FRONTEND, '..');
const OUT = join(FRONTEND, 'dist-capture');
const BUILD_DIR = join(OUT, 'build');
const PACKAGE_NAME = '웹캠촬영_패키지';
const GUIDE_DOC = join(REPO, 'docs', '공식모델_전환_웹캠촬영_따라하기_2026-10-28.md');
const CONSENT_DOC = join(REPO, 'docs', '공식모델_전환_웹캠촬영_기록지_2026-10-27.md');
const PROMPTS = JSON.parse(readFileSync(join(FRONTEND, 'src', 'capture', 'prompts.json'), 'utf8'));

/** HTML 특수문자를 이스케이프한다. */
function esc(text) {
  return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** 한 줄 안의 강조·코드·링크를 HTML로 바꾼다. 문서 사이 링크는 패키지에 없으므로 글자만 남긴다. */
function inline(text) {
  const codes = [];
  let s = esc(text).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, t, u) => (/^https?:/.test(u) ? `<a href="${u}">${t}</a>` : t));
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/^\[ \] /, '☐ ');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
}

/**
 * 이 프로젝트 문서에 쓰는 마크다운 일부(제목, 문단, 목록·중첩 목록, 표, 인용, 코드 블록, 구분선)를 HTML로 바꾼다.
 * @param {string} md
 * @returns {string}
 */
export function mdToHtml(md) {
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const out = [];
  const stack = []; // 열린 목록: {type, indent, li}
  let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join(' ')}</p>`); para = []; } };
  const closeTop = () => { const top = stack.pop(); out.push(`${top.li ? '</li>' : ''}</${top.type}>`); };
  const closeLists = () => { while (stack.length) closeTop(); };
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    const trimmed = line.trim();
    if (/^```/.test(trimmed)) {
      flushPara(); closeLists();
      const body = [];
      for (i += 1; i < lines.length && !/^```/.test(lines[i].trim()); i += 1) body.push(lines[i].trim());
      out.push(`<pre><code>${esc(body.join('\n'))}</code></pre>`);
      continue;
    }
    if (!trimmed) { flushPara(); continue; }
    if (/^<!--.*-->$/.test(trimmed)) continue;
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (heading && !/^\s/.test(line)) {
      flushPara(); closeLists();
      out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      continue;
    }
    if (/^-{3,}$/.test(trimmed)) { flushPara(); closeLists(); out.push('<hr>'); continue; }
    if (/^>/.test(trimmed)) {
      flushPara(); closeLists();
      const quote = [];
      for (; i < lines.length && /^>/.test(lines[i].trim()); i += 1) quote.push(inline(lines[i].trim().replace(/^>\s?/, '')));
      i -= 1;
      out.push(`<blockquote>${quote.join('<br>')}</blockquote>`);
      continue;
    }
    if (/^\|/.test(trimmed)) {
      flushPara(); closeLists();
      const rows = [];
      for (; i < lines.length && /^\|/.test(lines[i].trim()); i += 1) rows.push(lines[i].trim());
      i -= 1;
      const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      const body = rows.filter((r, k) => k !== 1 || !/^\|[\s:|-]+\|$/.test(r));
      const [head, ...rest] = body;
      out.push(`<table><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>`
        + rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('') + '</tbody></table>');
      continue;
    }
    const item = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(line);
    if (item) {
      flushPara();
      const indent = item[1].length;
      const type = /\d/.test(item[2]) ? 'ol' : 'ul';
      while (stack.length && stack[stack.length - 1].indent > indent) closeTop();
      let top = stack[stack.length - 1];
      if (top && top.indent === indent && top.type !== type) { closeTop(); top = stack[stack.length - 1]; }
      if (!top || top.indent < indent) {
        out.push(type === 'ol' ? `<ol start="${parseInt(item[2], 10)}">` : '<ul>');
        stack.push({ type, indent, li: false });
        top = stack[stack.length - 1];
      }
      if (top.li) out.push('</li>');
      out.push(`<li>${inline(item[3])}`);
      top.li = true;
      continue;
    }
    if (stack.length && /^\s+/.test(line)) { out.push(` ${inline(trimmed)}`); continue; }
    closeLists();
    para.push(trimmed);
  }
  flushPara(); closeLists();
  return out.join('\n');
}

/** 문서에서 <!-- name 시작 --> 와 <!-- name 끝 --> 사이를 꺼낸다. */
function section(path, name) {
  const text = readFileSync(path, 'utf8');
  const start = text.indexOf(`<!-- ${name} 시작 -->`);
  const end = text.indexOf(`<!-- ${name} 끝 -->`);
  if (start < 0 || end < start) throw new Error(`${path}에 '${name}' 표시가 없습니다`);
  return text.slice(start, end).split('\n').slice(1).join('\n');
}

/** 읽기·인쇄용 HTML 문서 틀. */
function page(title, bodyHtml) {
  return `<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>
:root { --fg:#1f2937; --muted:#6b7280; --line:#d1d5db; --bg:#ffffff; --soft:#f3f4f6; --accent:#1d4ed8; }
@media (prefers-color-scheme: dark) { :root:not([data-print]) { --fg:#e5e7eb; --muted:#9ca3af; --line:#4b5563; --bg:#111827; --soft:#1f2937; --accent:#93c5fd; } }
body { margin:0 auto; max-width:860px; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
  font-family:'Malgun Gothic','Apple SD Gothic Neo',system-ui,sans-serif; line-height:1.65; font-size:16px; }
h1 { font-size:26px; } h2 { font-size:21px; margin-top:36px; border-bottom:1px solid var(--line); padding-bottom:4px; } h3 { font-size:18px; margin-top:24px; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:15px; } th, td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { background:var(--soft); } blockquote { margin:12px 0; padding:8px 12px; background:var(--soft); border-left:4px solid var(--accent); color:var(--fg); }
code { background:var(--soft); padding:1px 4px; border-radius:4px; font-size:14px; } pre { background:var(--soft); padding:10px; overflow-x:auto; }
a { color:var(--accent); } hr { border:0; border-top:1px solid var(--line); margin:28px 0; }
@media print { :root { --fg:#000000; --muted:#333333; --line:#999999; --bg:#ffffff; --soft:#eeeeee; --accent:#000000; }
  body { max-width:none; padding:0; font-size:12pt; } h2 { break-after:avoid; } table, blockquote { break-inside:avoid; } }
</style></head><body>
${bodyHtml}
</body></html>
`;
}

/** 빌드한 capture.html의 JS·CSS를 HTML 안에 넣어 파일 하나로 만든다 (파일로 열어도 모듈을 불러오지 않게). */
function inlineBuild() {
  let html = readFileSync(join(BUILD_DIR, 'capture.html'), 'utf8');
  html = html.replace(/<script type="module" crossorigin src="\.\/(assets\/[^"]+\.js)"><\/script>/g, (_, src) => {
    const js = readFileSync(join(BUILD_DIR, src), 'utf8').replace(/<\/script/gi, '<\\/script');
    return `<script type="module">${js}</script>`;
  });
  html = html.replace(/<link rel="stylesheet" crossorigin href="\.\/(assets\/[^"]+\.css)">/g, (_, href) => `<style>${readFileSync(join(BUILD_DIR, href), 'utf8')}</style>`);
  if (/\.\/assets\//.test(html)) throw new Error('HTML 안에 넣지 못한 파일 참조가 남았습니다');
  return html;
}

/** DOS 형식 날짜·시각 (zip 머리). */
function dosTime(d) {
  return { time: (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2),
    date: ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate() };
}

/**
 * 파일 목록을 zip으로 묶는다. 한글 이름이 깨지지 않게 UTF-8 표시(0x0800)를 켠다.
 * @param {{name:string, data:Buffer}[]} files
 * @returns {Buffer}
 */
export function makeZip(files) {
  const { time, date } = dosTime(new Date());
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const f of files) {
    const name = Buffer.from(f.name, 'utf8');
    const comp = deflateRawSync(f.data, { level: 9 });
    const crc = crc32(f.data) >>> 0;
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x0800, 6); local.writeUInt16LE(8, 8);
    local.writeUInt16LE(time, 10); local.writeUInt16LE(date, 12); local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(comp.length, 18); local.writeUInt32LE(f.data.length, 22); local.writeUInt16LE(name.length, 26); local.writeUInt16LE(0, 28);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6); central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(8, 10); central.writeUInt16LE(time, 12); central.writeUInt16LE(date, 14); central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(comp.length, 20); central.writeUInt32LE(f.data.length, 24); central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, name, comp);
    centrals.push(central, name);
    offset += local.length + name.length + comp.length;
  }
  const cd = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(files.length, 8); end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, end]);
}

async function main() {
  rmSync(OUT, { recursive: true, force: true });
  await build({ configFile: join(FRONTEND, 'vite.capture.config.js'), root: FRONTEND, logLevel: 'warn' });
  const today = new Date().toISOString().slice(0, 10);
  const readme = [
    'Silent Talk 웹캠 촬영 패키지',
    `만든 날: ${today} · 촬영 문장 ${PROMPTS.version}`,
    '',
    '1. 이 압축 파일을 "압축 풀기"로 풀어 주세요 (압축 파일 안에서 바로 열지 마세요).',
    '2. 촬영가이드.html 을 열어 처음부터 순서대로 따라 하세요.',
    '3. 촬영은 웹캠촬영.html 을 Chrome 또는 Edge로 열어서 합니다. 인터넷·설치가 필요 없습니다.',
    '4. 동의서.html 은 인쇄하거나 화면으로 읽고 서명합니다.',
    '',
    '찍은 영상은 팀장이 정한 방법으로만 보내고, 단톡방·공개 링크에는 올리지 마세요.',
    '',
  ].join('\r\n');
  const files = [
    { name: '웹캠촬영.html', data: Buffer.from(inlineBuild(), 'utf8') },
    { name: '촬영가이드.html', data: Buffer.from(page('웹캠 촬영 가이드', mdToHtml(section(GUIDE_DOC, '조원용'))), 'utf8') },
    { name: '동의서.html', data: Buffer.from(page('웹캠 촬영 동의서', mdToHtml(section(CONSENT_DOC, '동의서'))), 'utf8') },
    { name: '먼저_읽어주세요.txt', data: Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), Buffer.from(readme, 'utf8')]) },
  ];
  const dir = join(OUT, PACKAGE_NAME);
  mkdirSync(dir, { recursive: true });
  for (const f of files) writeFileSync(join(dir, f.name), f.data);
  const zipPath = join(OUT, `${PACKAGE_NAME}_${today}.zip`);
  writeFileSync(zipPath, makeZip(files.map((f) => ({ name: `${PACKAGE_NAME}/${f.name}`, data: f.data }))));
  rmSync(BUILD_DIR, { recursive: true, force: true });
  console.log(`패키지: ${zipPath}`);
  for (const name of readdirSync(dir)) console.log(`  ${name}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { console.error(`패키지 만들기 실패: ${e.message}`); process.exit(1); });
}
