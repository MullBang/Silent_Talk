"""Preserve Q01's actual clipping report and prepare a separately reviewable end."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import wave

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PREVIOUS = ROOT / 'experiments/offline_phrase_review_step05f_20261018'


def bind(path: Path) -> dict:
    """Bind exact file bytes."""
    try:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind {path}') from exc


def read(path: Path) -> dict:
    """Read declared metadata without treating it as instructions."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read {path}') from exc


def write(path: Path, value: dict) -> None:
    """Create a stage-local immutable evidence file."""
    try:
        assert path.resolve().is_relative_to(BASE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'Cannot create {path}') from exc


def main() -> None:
    """Extend only Q01's end using stored PCM; do not infer a new approval."""
    try:
        frozen_path = PREVIOUS / 'results/completion_manifest.json'
        assert bind(frozen_path)['sha256'] == '153d12419b93790be72d1fdb91230d40510d00bd54e3f7dbcbf791f82a042281'
        frozen = {r['path']: r for r in read(frozen_path)['artifacts']}
        source_paths = [PREVIOUS / name for name in (
            'results/next_step_handoff.json', 'effective_parent_inputs.json',
            'user_evidence/parent_answers.json', 'phrase_review_manifest.json',
            'phrase_media_r2/Q01/record.json', 'phrase_media_r2/Q01/phrase.wav')]
        for path in source_paths:
            assert bind(path) == frozen[str(path)]
        questions = read(PREVIOUS / 'phrase_review_manifest.json')
        old_question = next(r for r in questions['records'] if r['review_id'] == 'Q01')
        candidate = read(PREVIOUS / 'phrase_media_r2/Q01/record.json')
        parent = next(r for r in read(PREVIOUS / 'effective_parent_inputs.json')['records'] if r['review_id'] == 'Q01')
        source_record_path = Path(parent['media_record']['path'])
        assert bind(source_record_path) == parent['media_record'] == frozen[str(source_record_path)]
        source_record = read(source_record_path)
        context = source_record['context_audio']
        context_path = Path(context['path'])
        assert bind(context_path) == frozen[str(context_path)]
        original_audio = candidate['phrase_audio']
        evidence_path = BASE / 'user_evidence/001_Q01_original_end_clipping.json'
        evidence = {
            'schema': 'offline-stage5g-actual-answer-v1', 'review_id': 'Q01',
            'parent_id': candidate['parent_id'], 'candidate_id': candidate['candidate_id'],
            'question_item_id': '["request_user_input_async","call_c6b912f45a194f748b5c9e8dc2e486de",0]',
            'question': 'Q01 짧게 자른 구절 음성에서 “너무 속상해”가 시작·끝 잘림이나 다른 말의 혼입 없이 온전히 들리나요?',
            'answer': '마지막 "해" 부분이 거의 들리지 않을정도로 끝이 잘림',
            'source': 'actual_user_chat_reply', 'question_manifest': bind(PREVIOUS / 'phrase_review_manifest.json'),
            'question_audio': bind(Path(original_audio['path'])),
            'interpretation': 'Reported clipping of final 해 on this exact original candidate only.',
            'phrase_audio_boundary_approved': False, 'phrase_visual_boundary_approved': False,
            'new_revision_approved': False, 'training_ready': False,
        }
        assert evidence['question'] == old_question['question']
        write(evidence_path, evidence)
        left = original_audio['source_context_first_sample_16k']
        old_right = original_audio['source_context_end_sample_16k']
        new_right_exact = Fraction(parent['original_recording_end_sample_48k'] - context['source_first_sample'], 3)
        assert new_right_exact.denominator == 1
        new_right = int(new_right_exact)
        assert 0 <= left - 3200 < old_right < new_right < new_right + 3200 <= context['samples']
        assert new_right - old_right == 3288
        with wave.open(str(context_path), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            source_pcm = wav.readframes(wav.getnframes())
        with wave.open(original_audio['path'], 'rb') as wav:
            old_pcm = wav.readframes(wav.getnframes())
        assert old_pcm == source_pcm[2 * left:2 * old_right]
        output = BASE / 'revisions/Q01_end_parent'
        output.mkdir(parents=True, exist_ok=False)
        created = {}
        for role, start, end in [('phrase', left, new_right), ('context', left - 3200, new_right + 3200)]:
            path = output / (role + '.wav')
            payload = source_pcm[2 * start:2 * end]
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000); wav.writeframes(payload)
            with wave.open(str(path), 'rb') as wav:
                assert wav.getnframes() == end - start and wav.readframes(wav.getnframes()) == payload
            created[role] = bind(path) | {
                'sample_rate': 16000, 'samples': end - start, 'duration_seconds': (end - start) / 16000,
                'source_context_first_sample_16k': start, 'source_context_end_sample_16k': end,
                'source_original_wav_first_sample_48k': context['source_first_sample'] + 3 * start,
                'source_original_wav_end_sample_48k': context['source_first_sample'] + 3 * end,
                'source_context': bind(context_path), 'new_resampling': False, 'padding': False, 'fade': False,
            }
        assert source_pcm[2 * left:2 * new_right].startswith(old_pcm)
        record_path = output / 'record.json'
        write(record_path, {
            'schema': 'offline-stage5g-Q01-parent-end-revision-v1',
            'review_id': 'Q01', 'revision_id': 'Q01_end_parent_r1',
            'parent_id': candidate['parent_id'], 'candidate_id': candidate['candidate_id'],
            'text': candidate['text'], 'trigger': bind(evidence_path),
            'previous_candidate': bind(PREVIOUS / 'phrase_media_r2/Q01/record.json'),
            'effective_parent_inputs': bind(PREVIOUS / 'effective_parent_inputs.json'),
            'parent_confirmation_evidence_id': parent['actual_answer_evidence_id'],
            'original_question_audio': bind(Path(original_audio['path'])),
            'media': created, 'start_unchanged': True, 'text_unchanged': True,
            'extension_samples_16k': new_right - old_right,
            'extension_seconds': (new_right - old_right) / 16000,
            'extension_is_not_measured_boundary_error': True,
            'old_audio_is_exact_prefix': True,
            'boundary_choice': 'Include the previously confirmed whole-parent ending; phrase confirmation remains separate.',
            'phrase_audio_boundary_approved': False, 'phrase_visual_boundary_approved': False,
            'training_ready': False, 'actual_split': None, 'new_video_created': False,
            'new_neural_calls': 0, 'new_tar_payload_bytes': 0, 'new_training_updates': 0,
            'code': bind(Path(__file__)),
        })
        question = 'Q01 전체 문장 끝까지 포함한 새 구절 음성에서 “너무 속상해”의 마지막 ‘해’까지 온전히 들리고, 시작 잘림이나 다른 말의 혼입은 없나요?'
        page_path = BASE / 'index_q01_r1.html'
        page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Q01 · 끝부분 수정 음성</title><style>body{font-family:Malgun Gothic,system-ui,sans-serif;background:#f2f5f8;color:#173348;margin:0}main{max-width:760px;margin:35px auto;padding:24px}section{background:white;padding:25px;border-radius:14px;border:1px solid #d8e1e8}p{line-height:1.8}audio{width:100%;margin:12px 0}h1{font-size:25px}h2{font-size:22px}.note{color:#5b6f7c;font-size:14px}</style><main><h1>Q01 · 끝부분 수정 음성</h1><section><h2>너무 속상해</h2><p>기존 구절은 마지막 ‘해’가 잘렸다는 답변을 기록했습니다. 시작과 문구는 그대로 두고, 앞서 확인한 전체 문장 끝까지 포함했습니다.</p><p><strong>① 새 구절 음성 · 1.4855초</strong></p><audio controls preload="metadata" src="/q01/phrase.wav"></audio><p>마지막 ‘해’까지 온전히 들리는지, 시작이 잘리거나 다른 말이 섞이지 않았는지 채팅으로 알려 주세요.</p><details><summary>앞뒤 0.20초 문맥 음성</summary><audio controls preload="metadata" src="/q01/context.wav"></audio><p class="note">문맥에는 다른 말이 포함될 수 있습니다.</p></details><p class="note">끝을 0.2055초 늘린 후보이며 아직 확인 전입니다. Q02·Q03은 기존 질문의 답변을 기다리고 있습니다.</p><a href="http://127.0.0.1:8773/#Q02">기존 Q02·Q03 구절 화면</a></section></main><script>document.addEventListener('play',e=>{if(e.target.matches('audio'))document.querySelectorAll('audio').forEach(x=>{if(x!==e.target)x.pause()})},true)</script></html>'''
        with page_path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(page)
        write(BASE / 'review_manifest_r1.json', {
            'schema': 'offline-stage5g-review-question-r1', 'review_id': 'Q01',
            'revision': bind(record_path), 'question': question, 'media': created,
            'page': bind(page_path), 'question_audio': created['phrase'],
            'review_url': 'http://127.0.0.1:8774/', 'actual_revision_answer': None,
            'phrase_audio_boundary_approved': False, 'phrase_visual_boundary_approved': False,
            'Q02_Q03_original_answers': 'pending', 'training_ready': False,
        })
        handler = read(PREVIOUS / 'phrase_server_plan.json')['range_handler']
        write(BASE / 'server_plan_r1.json', {
            'manifest': bind(BASE / 'review_manifest_r1.json'), 'range_handler': handler,
            'server': bind(BASE / 'serve_revision.py'), 'listen': '127.0.0.1:8774',
            'routes': {'/': bind(page_path), '/q01/phrase.wav': bind(Path(created['phrase']['path'])),
                       '/q01/context.wav': bind(Path(created['context']['path']))},
        })
        for path in source_paths + [context_path, source_record_path]:
            assert bind(path) == frozen[str(path)]
        print(json.dumps({'actual_answers': 1, 'approved_phrases': 0, 'extension_seconds': .2055, 'revised_samples': 23768, 'revision_answer': 'pending'}))
    except Exception as exc:
        raise RuntimeError('Q01 revision preparation failed; preserve original evidence') from exc


if __name__ == '__main__':
    main()
