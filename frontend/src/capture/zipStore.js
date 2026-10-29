// zipStore.js — 브라우저에서 클립 파일을 압축 없이 zip 하나로 묶는다 (폴더 저장을 쓸 수 없을 때 내려받기용).
// 영상(webm/mp4)은 이미 압축돼 있으므로 저장(store) 방식만 쓴다. 한글 이름이 깨지지 않게 UTF-8 표시를 켠다.

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/**
 * CRC-32 (zip 표준).
 * @param {Uint8Array} bytes
 * @returns {number} 부호 없는 32비트 값
 */
export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

/**
 * 파일 목록을 압축 없는 zip으로 묶는다.
 * @param {{name: string, bytes: Uint8Array}[]} entries 이름은 '폴더/파일' 형식 가능
 * @param {Date} [when] 파일 시각
 * @returns {Uint8Array}
 */
export function zipStore(entries, when = new Date()) {
  const enc = new TextEncoder();
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | Math.floor(when.getSeconds() / 2);
  const date = ((when.getFullYear() - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  const parts = [];
  const central = [];
  let offset = 0;
  for (const e of entries) {
    const name = enc.encode(e.name);
    const crc = crc32(e.bytes);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true); local.setUint16(4, 20, true); local.setUint16(6, 0x0800, true);
    local.setUint16(8, 0, true); local.setUint16(10, time, true); local.setUint16(12, date, true);
    local.setUint32(14, crc, true); local.setUint32(18, e.bytes.length, true); local.setUint32(22, e.bytes.length, true);
    local.setUint16(26, name.length, true); local.setUint16(28, 0, true);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true);
    cd.setUint16(10, 0, true); cd.setUint16(12, time, true); cd.setUint16(14, date, true); cd.setUint32(16, crc, true);
    cd.setUint32(20, e.bytes.length, true); cd.setUint32(24, e.bytes.length, true); cd.setUint16(28, name.length, true);
    cd.setUint32(42, offset, true);
    parts.push(new Uint8Array(local.buffer), name, e.bytes);
    central.push(new Uint8Array(cd.buffer), name);
    offset += 30 + name.length + e.bytes.length;
  }
  const cdSize = central.reduce((n, p) => n + p.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, entries.length, true); end.setUint16(10, entries.length, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  const all = [...parts, ...central, new Uint8Array(end.buffer)];
  const out = new Uint8Array(all.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of all) { out.set(p, pos); pos += p.length; }
  return out;
}
