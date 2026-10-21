"""Build an immutable current review page, leaving previous snapshots intact."""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from prepare_q01_revision import BASE, PREVIOUS, bind, read, write


def main(version: str) -> None:
    """Use explicit summary state and bind each displayed audio and video."""
    try:
        assert version.startswith('r') and version[1:].isdigit()
        summary_path = BASE / 'results' / f'stage5g_summary_{version}.json'
        summary = read(summary_path)
        old = {r['review_id']: r for r in read(PREVIOUS / 'phrase_review_manifest.json')['records']}
        revision = read(BASE / 'revisions/Q01_end_parent/record.json')
        visual = read(BASE / 'visual_Q01_r1/record.json')
        rows, cards, routes = [], [], {}
        output = BASE / 'ui' / version
        output.mkdir(parents=True, exist_ok=False)
        for item in summary['records']:
            qid = item['review_id']
            original = old[qid]
            media = dict(original['media'])
            if qid == 'Q01':
                media = {'phrase_audio': revision['media']['phrase'], 'context_audio': revision['media']['context'],
                         'phrase_video': visual['videos']['phrase'], 'phrase_context_video': visual['videos']['context']}
            if 'review_media_override' in item:
                override_path = Path(item['review_media_override']['path'])
                assert bind(override_path) == item['review_media_override']
                override = read(override_path)
                media = {'phrase_audio': override['media']['phrase'], 'context_audio': override['media']['context']}
            if 'review_visual_override' in item:
                visual_path = Path(item['review_visual_override']['path'])
                assert bind(visual_path) == item['review_visual_override']
                visual_override = read(visual_path)
                assert visual_override['review_id'] == qid
                media['phrase_video'] = visual_override['videos']['phrase']
                media['phrase_context_video'] = visual_override['videos']['context']
            assert bind(Path(media['phrase_audio']['path'])) == item['selected_audio']
            urls = {}
            for key, value in media.items():
                binding = bind(Path(value['path']))
                assert binding['sha256'] == value['sha256']
                url = f'/media/{qid}/{key}/{binding["sha256"][:16]}{Path(value["path"]).suffix}'
                routes[url] = binding
                urls[key] = url
            confirmed = item['human_phrase_audio_boundary_approved']
            state = '음성 확인 완료 · 입술 검수 대기' if confirmed else '음성 확인 대기'
            videos = ''.join(f'<label>{"구절만 무음 영상" if key == "phrase_video" else "앞뒤 문맥 무음 영상"}</label><video controls muted playsinline preload="none" src="{urls[key]}"></video>' for key in ['phrase_video', 'phrase_context_video'] if key in urls)
            if 'review_media_override' in item and 'review_visual_override' not in item:
                videos = '<p>확인된 수정 음성에 맞춘 입술 영상은 다음 검수 전에 준비합니다.</p>'
            audio_duration = media['phrase_audio']['duration_seconds']
            cards.append(f'''<section id="{qid}"><div class="tag">{qid}<span>{state}</span></div><h2>{html.escape(item['text'])}</h2>
<p>{'확인된 수정 구절입니다.' if item['audio_status'] == 'confirmed_revised_audio' else '잘림·혼입 여부를 확인할 구절입니다.'}</p>
<label>구절 음성 · {audio_duration:.4f}초</label><audio controls preload="metadata" src="{urls['phrase_audio']}"></audio>
<details><summary>앞뒤 0.20초 음성과 무음 영상 비교</summary><label>문맥 음성</label><audio controls preload="none" src="{urls['context_audio']}"></audio>{videos}<p>음성과 영상은 각각 재생합니다. 영상 표시 구간은 음성보다 조금 넓을 수 있습니다.</p></details></section>''')
            rows.append({'review_id': qid, 'text': item['text'], 'state': state,
                         'phrase_audio_confirmed': confirmed, 'phrase_visual_confirmed': False,
                         'media': {k: bind(Path(v['path'])) for k, v in media.items()}, 'urls': urls})
        confirmed_ids = [r['review_id'] for r in rows if r['phrase_audio_confirmed']]
        nav = ''.join(f'<a href="#{r["review_id"]}">{r["review_id"]}</a>' for r in rows)
        page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5g · 구절 검수 진행 현황</title><style>body{margin:0;font-family:Malgun Gothic,system-ui,sans-serif;background:#f2f5f8;color:#173348}main{max-width:820px;margin:auto;padding:28px 22px 70px}h1{font-size:26px}h2{font-size:23px}p{font-size:14px;line-height:1.8}section{background:white;border:1px solid #d8e1e8;border-radius:14px;padding:24px;margin:20px 0;scroll-margin-top:75px}.tag{font-weight:bold;color:#185771}.tag span{float:right;font-weight:normal;font-size:12px;background:#edf2ea;padding:5px 8px;border-radius:8px}nav{position:sticky;top:0;display:flex;gap:8px;background:#f2f5f8ee;padding:12px 0;z-index:2;flex-wrap:wrap}nav a{background:#214f68;color:white;text-decoration:none;border-radius:6px;padding:8px 12px}audio{width:100%;margin:12px 0}video{display:block;width:100%;max-width:320px;border-radius:8px;margin:10px 0;background:#122532}label{display:block;margin-top:20px;font-size:14px;font-weight:bold}summary{cursor:pointer;color:#245d7c}.notice{background:#e1eee7;border-radius:9px;padding:12px 18px}</style><main><h1>구절 검수 진행 현황</h1>''' + f'<p class="notice">음성 확인 완료: {", ".join(confirmed_ids)} · 입술 시각 검수: 아직 확인 전</p>' + '<p>답변은 채팅에 남겨 주세요. 미응답 구절은 확인 대기로 유지합니다. 재생만으로 확인 처리하지 않습니다.</p><nav>' + nav + '</nav>' + '\n'.join(cards) + '''</main><script>document.addEventListener('play',e=>{if(e.target.matches('audio,video'))document.querySelectorAll('audio,video').forEach(x=>{if(x!==e.target)x.pause()})},true)</script></html>'''
        page_path = output / 'index.html'
        with page_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(page)
        routes['/'] = bind(page_path)
        manifest_path = output / 'manifest.json'
        write(manifest_path, {'schema': 'offline-stage5g-current-review-ui-v1', 'version': version,
                             'summary': bind(summary_path), 'builder': bind(Path(__file__)),
                             'records': rows, 'routes': routes,
                             'range_handler': read(PREVIOUS / 'phrase_server_plan.json')['range_handler'],
                             'current_url': 'http://127.0.0.1:8775/', 'training_ready': False})
        # This explicit pointer is the only mutable UI file; the versioned page stays immutable.
        pointer = BASE / 'ui/current.json'
        import json
        pointer.write_text(json.dumps({'manifest': bind(manifest_path)}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Prepared {version} review UI for {len(rows)} phrases')
    except Exception as exc:
        raise RuntimeError('Cannot build current review page from incomplete or inconsistent evidence') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    main(parser.parse_args().version)
