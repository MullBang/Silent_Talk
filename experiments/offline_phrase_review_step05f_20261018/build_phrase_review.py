"""Create a separate phrase-review page after actual whole-parent review."""
from __future__ import annotations

import html
from pathlib import Path

from prepare_policy import BASE, ROOT, binding, load, save


def main() -> None:
    """Bind exact questions and media; retain the earlier parent-page snapshot."""
    try:
        selection_path = BASE / 'phrase_media_selected.json'
        selected = load(selection_path)
        source = Path(selected['manifest']['path'])
        assert binding(source) == selected['manifest']
        media_root = source.parent
        parents_path = BASE / 'effective_parent_inputs.json'
        parents = {r['review_id']: r for r in load(parents_path)['records']}
        rows, cards = [], []
        for candidate in load(source)['records']:
            qid, phrase = candidate['review_id'], candidate['text']
            parent = parents[qid]
            assert parent['parent_transcript_confirmed'] and parent['parent_audio_qualitatively_confirmed']
            media = {'phrase_audio': candidate['phrase_audio'],
                     'context_audio': candidate['phrase_context_audio']}
            for role, spec in candidate['muted_video'].items():
                media[role + '_video'] = spec
            urls = {}
            for key, spec in media.items():
                path = Path(spec['path'])
                assert path.resolve().is_relative_to(media_root)
                assert binding(path)['sha256'] == spec['sha256']
                urls[key] = f'/phrase/{qid}/{path.name}'
            question = (f'{qid} 짧게 자른 구절 음성에서 “{phrase}”가 시작·끝 잘림이나 '
                        '다른 말의 혼입 없이 온전히 들리나요?')
            rows.append({'review_id': qid, 'parent_id': candidate['parent_id'],
                         'candidate_id': candidate['candidate_id'], 'text': phrase,
                         'question': question, 'media': media, 'urls': urls,
                         'candidate_record': binding(media_root / qid / 'record.json'),
                         'parent_transcript_confirmed': True,
                         'phrase_audio_boundary_status': 'pending',
                         'phrase_visual_boundary_status': 'pending', 'training_ready': False})
            videos = ''.join(f'<label>{"구절 영상" if key == "phrase_video" else "앞뒤 문맥 영상"}</label><video controls muted playsinline preload="none" src="{url}"></video>'
                             for key, url in urls.items() if key.endswith('_video'))
            cards.append(f'''<section id="{qid}"><div class="eyebrow">{qid} · {html.escape(parent['speaker_id'])}<span>구절 확인 대기</span></div>
<h2>{html.escape(phrase)}</h2><p class="parent">전체 문장 확인 완료 · {html.escape(parent['parent_text'])}</p>
<label>① 구절만 음성 · {candidate['phrase_audio']['duration_seconds']:.2f}초</label>
<audio controls preload="metadata" src="{urls['phrase_audio']}"></audio>
<p>목표 구절이 처음부터 끝까지 들리는지, 다른 말이 섞이지 않았는지 확인해 주세요.</p>
<details><summary>앞뒤 0.20초 음성과 입술 영상 비교</summary><label>② 앞뒤 문맥 음성</label>
<audio controls preload="none" src="{urls['context_audio']}"></audio>
<p>문맥 음성에는 다른 말이 포함될 수 있습니다. 각 음성과 영상은 따로 재생합니다.</p>{videos}
<p class="note">영상은 기존 검수 영상에서 잘라낸 미리보기입니다. 영상의 시작·끝은 음성과 조금 다를 수 있습니다.</p></details></section>''')
        assert [r['review_id'] for r in rows] == [f'Q{i:02}' for i in range(1, 9)]
        nav = ''.join(f'<a href="#{r["review_id"]}">{r["review_id"]}</a>' for r in rows)
        page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5f · 새 8구절 경계 확인</title>
<style>body{margin:0;background:#f2f5f8;color:#173348;font-family:Malgun Gothic,system-ui,sans-serif}main{max-width:860px;margin:auto;padding:28px 22px 70px}h1{font-size:27px}h2{font-size:23px;line-height:1.6}p{line-height:1.8;font-size:15px}.notice{background:#e0eee8;padding:14px 18px;border-radius:9px}nav{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#f2f5f8ee;padding:12px 0;z-index:2}nav a{background:#214f68;color:white;padding:8px 12px;border-radius:6px;text-decoration:none}section{background:white;border:1px solid #d8e1e8;padding:24px;border-radius:14px;margin:20px 0;scroll-margin-top:80px}.eyebrow{font-weight:bold;color:#185771}.eyebrow span{float:right;font-size:12px;color:#796123;background:#fff3cf;border-radius:12px;padding:4px 8px}.parent,.note{color:#617383;font-size:13px}audio{width:100%;margin:10px 0}label{display:block;font-size:14px;font-weight:bold;margin-top:20px}video{width:100%;max-width:350px;display:block;background:#142532;border-radius:7px;margin-top:10px}summary{cursor:pointer;color:#245d7c}</style></head><body><main>
<h1>새 8구절 · 잘린 소리 확인</h1><p class="notice">전체 8문장 청취 확인을 마쳤습니다. Q02 전체 음성은 끝을 0.12초 늘린 수정본으로 확인했습니다. 이번에는 아래의 짧은 구절을 확인합니다.</p>
<p>자동으로 추정한 구간이므로 마지막 음절이 잘리거나 앞말이 섞일 수 있습니다. 답변은 채팅에 남겨 주세요.</p><nav>''' + nav + '</nav>' + '\n'.join(cards) + '''</main><script>document.addEventListener('play',e=>{if(e.target.matches('audio,video'))document.querySelectorAll('audio,video').forEach(x=>{if(x!==e.target)x.pause()})},true)</script></body></html>'''
        page_path = BASE / 'index_phrase.html'
        with page_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(page)
        save(BASE / 'phrase_review_manifest.json', {
            'schema': 'offline-stage5f-phrase-question-snapshot-v1',
            'status': 'parent8_confirmed_phrase8_pending',
            'sources': [binding(selection_path), binding(source), binding(parents_path), binding(Path(__file__))],
            'media_root': str(media_root),
            'page': binding(page_path), 'records': rows,
            'review_url': 'http://127.0.0.1:8773/',
            'actual_phrase_answers': 0, 'actual_visual_answers': 0, 'training_ready': False})
        helper = ROOT / 'experiments/offline_boundary_review_step02b_20261013/serve_review.py'
        save(BASE / 'phrase_server_plan.json', {
            'range_handler': binding(helper), 'server': binding(BASE / 'serve_phrase_review.py'),
            'manifest': binding(BASE / 'phrase_review_manifest.json'), 'page': binding(page_path),
            'listen': '127.0.0.1:8773', 'read_only': True})
        print('Created 8 phrase questions and separate review page')
    except Exception as exc:
        raise RuntimeError('Phrase page creation failed; preserve earlier review evidence') from exc


if __name__ == '__main__':
    main()
