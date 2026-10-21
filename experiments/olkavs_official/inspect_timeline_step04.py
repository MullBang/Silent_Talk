"""Read only MP4 metadata boxes to audit movie/media timeline offsets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

BASE = Path(__file__).resolve().parent


def boxes(data: bytes, start: int = 0, end: int | None = None):
    end = len(data) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from('>I4s', data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from('>Q', data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise ValueError(f'Invalid {kind!r} at {pos}')
        yield kind.decode('ascii', 'replace'), data[pos+header:pos+size]
        pos += size
    if pos != end:
        raise ValueError('Trailing incomplete box')


def clock_box(payload: bytes) -> dict:
    version = payload[0]
    offset = 20 if version == 1 else 12
    timescale = struct.unpack_from('>I', payload, offset)[0]
    duration = struct.unpack_from('>Q' if version == 1 else '>I', payload, offset+4)[0]
    return dict(version=version, timescale=timescale, duration=duration,
                duration_seconds=duration/timescale)


def edit_list(payload: bytes) -> list[dict]:
    version = payload[0]
    n = struct.unpack_from('>I', payload, 4)[0]
    offset, result = 8, []
    fmt = '>Qqhh' if version == 1 else '>Iihh'
    for _ in range(n):
        duration, media_time, rate_i, rate_f = struct.unpack_from(fmt, payload, offset)
        result.append(dict(segment_duration=duration, media_time=media_time,
                           media_rate_integer=rate_i, media_rate_fraction=rate_f))
        offset += struct.calcsize(fmt)
    if offset != len(payload):
        raise ValueError('Unexpected elst bytes')
    return result


def inspect(path: Path) -> dict:
    length = path.stat().st_size
    moov = None
    read_bytes = 0
    with path.open('rb') as fh:
        pos = 0
        while pos < length:
            fh.seek(pos)
            head = fh.read(8)
            read_bytes += len(head)
            size, kind = struct.unpack('>I4s', head)
            header = 8
            if size == 1:
                size = struct.unpack('>Q', fh.read(8))[0]
                header = 16
                read_bytes += 8
            elif size == 0:
                size = length-pos
            if size < header or pos+size > length:
                raise ValueError('Invalid top-level box')
            if kind == b'moov':
                if size > 16*1024*1024:
                    raise ValueError('Metadata exceeds bounded read limit')
                moov = fh.read(size-header)
                read_bytes += len(moov)
            pos += size
    if moov is None:
        raise ValueError('No moov metadata')
    movie, tracks = None, []
    for kind, data in boxes(moov):
        if kind == 'mvhd':
            movie = clock_box(data)
        if kind != 'trak':
            continue
        track = {}
        for child, body in boxes(data):
            if child == 'edts':
                for edit, payload in boxes(body):
                    if edit == 'elst':
                        track['edit_list'] = edit_list(payload)
            if child == 'mdia':
                for media, payload in boxes(body):
                    if media == 'mdhd':
                        track['media_clock'] = clock_box(payload)
                    if media == 'hdlr':
                        track['handler'] = payload[8:12].decode('ascii')
        tracks.append(track)
    for track in tracks:
        elapsed = 0
        for entry in track.get('edit_list', []):
            entry['movie_start_seconds'] = elapsed/movie['timescale']
            entry['movie_duration_seconds'] = entry['segment_duration']/movie['timescale']
            if entry['media_time'] >= 0:
                entry['media_start_seconds'] = entry['media_time']/track['media_clock']['timescale']
                entry['movie_minus_media_start_seconds'] = entry['movie_start_seconds']-entry['media_start_seconds']
            elapsed += entry['segment_duration']
    return dict(path=str(path), size_bytes=length, metadata_bytes_read=read_bytes,
                moov_sha256=hashlib.sha256(moov).hexdigest(), movie_clock=movie, tracks=tracks)


def main() -> None:
    plan = json.loads((BASE/'step03_source_plan.json').read_text(encoding='utf-8'))
    result = {'schema':'step04-mp4-timeline-audit-v1',
              'sources':[inspect(Path(x['source_video']['path'])) for x in plan['sources']],
              'limits':'Metadata only; rate1 edit mapping is descriptive. Actual frame timestamp origin still requires decoder verification.'}
    target = BASE/'results/step04_timeline_metadata.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
