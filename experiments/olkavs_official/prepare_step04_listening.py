"""Extract small original audio excerpts for independent label-content review.

Audio is never passed to the lipreading model. Labels and recordings are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
IDS = ('lip_J_1_F_02_C025_A_006_001', 'lip_J_1_F_02_C025_A_006_005',
       'lip_J_1_F_02_C025_A_007_033')


def main() -> None:
    sys.path.insert(0, str(BASE/'audio_tools'))
    import imageio_ffmpeg
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    plan = json.loads((BASE/'step03_source_plan.json').read_text(encoding='utf-8'))
    inputs = json.loads((BASE/'inputs_step03_recovery02/manifest.json').read_text(encoding='utf-8'))
    source_by_stem = {x['stem']:x['source_video']['path'] for x in plan['sources']}
    records = {x['id']:x for x in plan['records']}
    out = BASE/'step04_listening'
    out.mkdir(exist_ok=False)
    excerpts = []
    for sid in IDS:
        row = records[sid]
        item = next(x for x in inputs['records'] if x['id']==sid and x['arm']=='annotation_bbox')
        source = source_by_stem[sid.rsplit('_',1)[0]]
        start, duration = item['start_frame']/item['native_fps'], item['frames']/item['native_fps']
        dest = out/f'{sid}.wav'
        cmd = [executable,'-nostdin','-v','error','-ss',str(start),'-i',source,'-t',str(duration),
               '-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le','-n',str(dest)]
        result = subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr)
        excerpts.append(dict(id=sid,reference=row['text'],source=source,start_seconds=start,
                             duration_seconds=duration,path=str(dest),bytes=dest.stat().st_size,
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    manifest = {'purpose':'manual source-content review only; no ASR or lipreading inference',
                'audio_used_for_lipreading':False,'excerpts':excerpts,'ffmpeg':executable,
                'ffmpeg_version':subprocess.check_output([executable,'-version'],text=True).splitlines()[0]}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
