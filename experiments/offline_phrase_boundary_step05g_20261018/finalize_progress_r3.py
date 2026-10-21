"""Save actual review answers and a new progress snapshot without rewriting history."""
from pathlib import Path

from prepare_q01_revision import BASE, bind, read, write


def main():
    batch_path = BASE / 'question_batches/004_Q06_plus40ms.json'
    question = read(batch_path)['records'][0]
    write(BASE / 'user_evidence/008_Q06_plus40ms_audio_confirmed.json', {
        'schema': 'offline-stage5g-actual-answer-v1', 'review_id': 'Q06',
        'revision_id': 'Q06_end_plus40ms_r1',
        'question_item_id': '["request_user_input_async","call_8767739030ba4feeb8192d672083921e",0]',
        'question': question['question'], 'answer': '온전히 들림·잘림·혼입 없음',
        'source': 'actual_user_chat_reply', 'question_manifest': bind(batch_path),
        'question_audio': question['question_audio'], 'revision_record': question['revision'],
        'scope': 'This exact +40ms revised phrase audio only; original, +80ms and visual candidates are not approved.',
        'phrase_audio_boundary_approved': True, 'phrase_visual_boundary_approved': False,
        'original_candidate_approved': False, 'training_ready': False,
    })
    summary = read(BASE / 'results/stage5g_summary_r2.json')
    summary.update({
        'status': 'in_progress_Q01_Q04_Q05_Q06_audio_confirmed_remaining_review_pending',
        'actual_phrase_answers': 8, 'actual_phrase_answer_units_note': 'Includes 2 item-specific reports from one free-text user message.',
        'original_end_clipping_reports': 3, 'revised_other_speech_mixing_reports': 1,
        'selected_phrase_audio_confirmed': 4,
        'evidence': [bind(p) for p in sorted((BASE / 'user_evidence').glob('*.json'))],
        'next_question_batch': None, 'latest_question_batch': bind(batch_path),
        'latest_question_batch_answered': True,
        'next': 'Collect pending original Q02/Q03 phrase audio answers and ask Q07/Q08; prepare matching Q05/Q06 visual previews before separate qualitative lip boundary review.',
        'snapshot_note': 'Question batches and revision records retain their original pre-answer pending state. This new summary binds later actual answers.',
        'new_review_videos_this_revision': 0,
    })
    dirs = {'Q05': 'Q05_end_confirmed_parent', 'Q06': 'Q06_end_plus40ms'}
    for row in summary['records']:
        qid = row['review_id']
        if qid in dirs:
            revision_path = BASE / 'revisions' / dirs[qid] / 'record.json'
            revision = read(revision_path)
            row.update({'original_audio': row['selected_audio'],
                        'selected_audio': bind(Path(revision['media']['phrase']['path'])),
                        'audio_status': 'confirmed_revised_audio',
                        'human_phrase_audio_boundary_approved': True,
                        'review_media_override': bind(revision_path),
                        'matching_revised_visual_preview_ready': False})
    assert sum(x['human_phrase_audio_boundary_approved'] for x in summary['records']) == 4
    summary_path = BASE / 'results/stage5g_summary_r3.json'
    write(summary_path, summary)
    handoff = read(BASE / 'results/next_step_handoff_r2.json')
    handoff.update({'schema': 'offline-stage5g-handoff-r3', 'summary': bind(summary_path),
                    'current_audio_confirmed': ['Q01', 'Q04', 'Q05', 'Q06'],
                    'pending_existing_questions': handoff['pending_existing_questions'][:2],
                    'Q05_Q06_audio_revisions': {r['review_id']: r['review_media_override'] for r in summary['records'] if r['review_id'] in dirs},
                    'matching_visual_preview_not_yet_prepared': ['Q05', 'Q06'],
                    'Q05_Q06_old_videos_must_not_represent_revised_boundaries': True,
                    'next': summary['next']})
    write(BASE / 'results/next_step_handoff_r3.json', handoff)

    # Create a new builder version; the existing r2 builder and pages are immutable.
    code = (BASE / 'build_progress_ui.py').read_text(encoding='utf-8')
    anchor = "            assert bind(Path(media['phrase_audio']['path'])) == item['selected_audio']"
    replacement = """            if 'review_media_override' in item:
                override_path = Path(item['review_media_override']['path'])
                assert bind(override_path) == item['review_media_override']
                override = read(override_path)
                media = {'phrase_audio': override['media']['phrase'], 'context_audio': override['media']['context']}
            assert bind(Path(media['phrase_audio']['path'])) == item['selected_audio']"""
    assert code.count(anchor) == 1
    code = code.replace(anchor, replacement)
    code = code.replace("for key in ['phrase_video', 'phrase_context_video'])", "for key in ['phrase_video', 'phrase_context_video'] if key in urls)")
    code = code.replace("            audio_duration =", "            if 'review_media_override' in item:\n                videos = '<p>확인된 수정 음성에 맞춘 입술 영상은 다음 검수 전에 준비합니다.</p>'\n            audio_duration =")
    code = code.replace("{'전체 문장 끝까지 포함한 수정 구절입니다.' if qid == 'Q01' else '잘림·혼입 여부를 확인할 구절입니다.'}", "{'확인된 수정 구절입니다.' if item['audio_status'] == 'confirmed_revised_audio' else '잘림·혼입 여부를 확인할 구절입니다.'}")
    with (BASE / 'build_progress_ui_r3.py').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(code)
    print('Recorded Q06 confirmation; r3 summary and handoff saved with 4/8 audio confirmed.')


if __name__ == '__main__':
    main()
