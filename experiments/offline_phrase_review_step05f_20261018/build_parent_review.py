"""Build an exact-audio parent review page for the fixed Q01-Q08 only."""
from __future__ import annotations
import html
import json
from pathlib import Path
from typing import Any
import wave

from prepare_policy import BASE, ROOT, binding, load, save


def audio_spec(path: Path) -> dict[str, Any]:
    """Verify review PCM headers and bind exact audio bytes."""
    try:
        with wave.open(str(path), 'rb') as stream:
            rate, channels, width, samples = (stream.getframerate(), stream.getnchannels(),
                                             stream.getsampwidth(), stream.getnframes())
        if (rate, channels, width) != (16000, 1, 2) or samples <= 0:
            raise ValueError('Unexpected review audio format')
        return binding(path) | {'sample_rate': rate, 'samples': samples,
                                'duration_seconds': samples / rate}
    except (OSError, ValueError, wave.Error) as exc:
        raise RuntimeError(f'Cannot bind review PCM {path}') from exc


def main() -> None:
    """Show original reference text without recording or implying user approval."""
    try:
        selected_path = BASE / 'selected_inputs.json'
        selected = load(selected_path)['records']
        assert len(selected) == 8
        rows, cards = [], []
        for pair in selected:
            item = pair['review']
            qid, pid = item['review_id'], item['parent_id']
            folder = BASE / 'media' / pid
            sentence = audio_spec(folder / 'sentence.wav')
            context = audio_spec(folder / 'context.wav')
            lips_path = folder / 'context_lips.mp4'
            record_path = folder / 'record.json'
            assert record_path.exists() and lips_path.exists()
            media = {'sentence_audio': sentence, 'context_audio': context,
                     'context_lips': binding(lips_path)}
            urls = {'sentence_audio': f'/media/{qid}/sentence.wav',
                    'context_audio': f'/media/{qid}/context.wav',
                    'context_lips': f'/media/{qid}/context_lips.mp4'}
            question = (f'{qid} {item["speaker_id"]} 전체 음성이 “{item["parent_text"]}”와 일치하고, '
                        '첫·마지막 소리가 잘리거나 다른 말이 섞이지 않고 온전히 들리나요?')
            rows.append({'review_id': qid, 'parent_id': pid,
                         'candidate_id': item['candidate_id'], 'speaker_id': item['speaker_id'],
                         'parent_text': item['parent_text'], 'original_source_text': item['parent_source_text'],
                         'candidate_phrase_text': item['text'], 'question': question,
                         'media_record': binding(record_path), 'media': media, 'urls': urls,
                         'parent_transcript_status': 'pending', 'parent_audio_boundary_status': 'pending',
                         'phrase_audio_boundary_status': 'pending', 'phrase_visual_boundary_status': 'pending',
                         'new_human_approval_added': False, 'training_ready': False})
            title = html.escape(f'{qid} · {item["speaker_id"]}')
            parent_text = html.escape(item['parent_text'])
            phrase = html.escape(item['text'])
            cards.append(f'''<section class="card" id="{qid}">
<div class="eyebrow">{title} <span class="status">확인 대기</span></div>
<h2>{parent_text}</h2>
<p class="help">위 문장이 실제 음성과 일치하는지, 시작·끝이 잘리거나 다른 말이 섞이지 않았는지 확인해 주세요.</p>
<label for="audio-{qid}">① 전체 문장 음성 · {sentence['duration_seconds']:.2f}초</label>
<audio id="audio-{qid}" controls preload="metadata" src="{urls['sentence_audio']}"></audio>
<details><summary>앞뒤 문맥과 입술 영상 보기</summary>
<p>문맥 음성에는 앞뒤 다른 말이 포함될 수 있습니다. 위의 전체 문장 음성과 구분해 들어 주세요.</p>
<audio controls preload="none" src="{urls['context_audio']}"></audio>
<video controls muted playsinline preload="none" src="{urls['context_lips']}"></video>
<p class="help">영상과 문맥 음성은 시작 시각이 서로 조금 다를 수 있어 각각 재생합니다. 이 화면의 확인은 구절 경계 승인과 별개입니다.</p>
</details><p class="phrase">이후 확인할 구절: <strong>{phrase}</strong></p>
</section>''')
        manifest = {'schema': 'offline-stage5f-parent-review-manifest-v1',
                    'status': 'full_parent_review_pending',
                    'sources': [binding(selected_path), binding(BASE / 'execution_policy.json'),
                                binding(Path(__file__))], 'records': rows,
                    'actual_answers': 0, 'phrase_approval_implied': False,
                    'review_server': 'http://127.0.0.1:8772/', 'training_ready': False}
        save(BASE / 'parent_review_manifest.json', manifest)
        nav = ''.join(f'<a href="#{row["review_id"]}">{row["review_id"]}</a>' for row in rows)
        page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>5f · 첫 8문장 음성 검수</title>
<style>
:root{font-family:Malgun Gothic,system-ui,sans-serif;color:#172c3d;background:#f2f5f8}body{margin:0}main{max-width:850px;margin:auto;padding:28px 22px 70px}header{margin-bottom:22px}h1{font-size:27px;line-height:1.4;margin:0 0 10px}p{line-height:1.75}nav{position:sticky;top:0;display:flex;gap:8px;flex-wrap:wrap;background:#f2f5f8ed;padding:12px 0;z-index:2}nav a{background:#183f58;color:white;text-decoration:none;padding:8px 12px;border-radius:7px;font-size:14px}.card{background:white;border:1px solid #dbe4eb;border-radius:14px;padding:24px;margin:20px 0;scroll-margin-top:75px}h2{font-size:21px;line-height:1.7;font-weight:600;margin:15px 0}.eyebrow{font-weight:700;color:#1a5875}.status{float:right;color:#735b24;font-size:12px;background:#fff4d5;padding:4px 9px;border-radius:12px}.help{color:#526779;font-size:14px}label{display:block;font-weight:600;font-size:14px;margin-top:22px}audio{width:100%;margin:12px 0}details{margin:18px 0}summary{cursor:pointer;color:#245d7b}video{width:100%;max-width:360px;image-rendering:auto;background:#0b1820;border-radius:8px}details p,.phrase{font-size:14px}.phrase{border-top:1px solid #e5ebf0;padding-top:15px;color:#536978}.notice{background:#e6eff6;border-radius:10px;padding:12px 16px;font-size:14px}
</style></head><body><main><header><h1>첫 8문장 · 전체 음성 확인</h1>
<p>먼저 전체 문장을 확인합니다. 잘리는 소리나 전사와 다른 표현이 있으면 문장 번호와 함께 채팅에 알려 주세요.</p>
<p class="notice">페이지 열기나 재생은 확인 답변으로 기록되지 않습니다. 구절 경계는 이후 따로 확인합니다.</p>
</header><nav>''' + nav + '</nav>' + '\n'.join(cards) + '''</main>
<script>document.addEventListener('play',function(e){if(e.target.matches('audio,video')){document.querySelectorAll('audio,video').forEach(x=>{if(x!==e.target)x.pause()})}},true)</script></body></html>'''
        with (BASE / 'index.html').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(page)
        save(BASE / 'results/parent_page_build.json', {
            'status': 'created_parent_review_page', 'record_count': 8,
            'manifest': binding(BASE / 'parent_review_manifest.json'), 'page': binding(BASE / 'index.html'),
            'human_answers_added': 0, 'media_route_count': 24})
        print('Created exact-audio parent review page for Q01-Q08')
    except Exception as exc:
        raise RuntimeError('Cannot build parent review page from incomplete or invalid media') from exc


if __name__ == '__main__':
    main()
