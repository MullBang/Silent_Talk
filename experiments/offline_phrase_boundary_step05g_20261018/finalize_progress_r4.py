"""Bind eight actual audio confirmations to matching, still-unapproved visual previews."""
from pathlib import Path

from prepare_q01_revision import BASE, PREVIOUS, bind, read, write


def main():
    summary = read(BASE / 'results/stage5g_summary_r3.json')
    q08_revision_path = BASE / 'revisions/Q08_end_plus80ms/record.json'
    q08_revision = read(q08_revision_path)
    evidence = [bind(p) for p in sorted((BASE / 'user_evidence').glob('*.json'))]
    assert len(evidence) == 13
    originals = {r['review_id']: r for r in read(PREVIOUS / 'phrase_review_manifest.json')['records']}
    questions = []
    for row in summary['records']:
        qid = row['review_id']
        if qid in ['Q02', 'Q03', 'Q07']:
            row['audio_status'] = 'confirmed_original_audio'
            row['human_phrase_audio_boundary_approved'] = True
        if qid == 'Q08':
            row.update({'original_audio': row['selected_audio'],
                        'selected_audio': bind(Path(q08_revision['media']['phrase']['path'])),
                        'audio_status': 'confirmed_revised_audio',
                        'human_phrase_audio_boundary_approved': True,
                        'review_media_override': bind(q08_revision_path)})
        if qid in ['Q05', 'Q06', 'Q08']:
            visual_path = BASE / f'visual_{qid}_r1/record.json'
            visual = read(visual_path)
            assert visual['review_id'] == qid and not visual['phrase_visual_boundary_approved']
            row.update({'review_visual_override': bind(visual_path),
                        'matching_revised_visual_preview_ready': True})
        if qid in ['Q01', 'Q05', 'Q06', 'Q08']:
            visual_path = BASE / f'visual_{qid}_r1/record.json'
            visual = read(visual_path)
            videos = {key: bind(Path(visual['videos'][role]['path'])) for key, role in [('phrase_video', 'phrase'), ('context_video', 'context')]}
        else:
            videos = {key: bind(Path(originals[qid]['media'][role]['path'])) for key, role in [('phrase_video', 'phrase_video'), ('context_video', 'phrase_context_video')]}
        assert row['human_phrase_audio_boundary_approved'] and not row['human_phrase_visual_boundary_approved']
        confirmations = [e for e in evidence if (read(Path(e['path'])).get('review_id') == qid and read(Path(e['path'])).get('phrase_audio_boundary_approved') and read(Path(e['path'])).get('question_audio') == row['selected_audio'])]
        assert len(confirmations) == 1
        row['audio_confirmation'] = confirmations[0]
        questions.append({'review_id': qid, 'candidate_id': row['candidate_id'], 'text': row['text'],
                          'audio': row['selected_audio'], 'audio_confirmation': confirmations[0],
                          **videos, 'human_visual_answer': None, 'training_ready': False})
    displays = sum(read(BASE / f'visual_{qid}_r1/record.json')['videos'][role]['frames'] for qid in ['Q01', 'Q05', 'Q06', 'Q08'] for role in ['phrase', 'context'])
    question_path = BASE / 'question_batches/008_all8_visual.json'
    write(question_path, {'schema': 'offline-stage5g-qualitative-visual-question-v1',
                         'question': '오른쪽 최신 검수 화면의 Q01~Q08에서 ‘구절만 무음 영상’과 ‘앞뒤 문맥 무음 영상’을 비교했을 때, 입 움직임이 시작이나 끝에서 뚜렷하게 잘리는 구절이 있나요? 있으면 번호와 시작/끝을 적어 주세요. 구분이 어려운 경우도 그대로 알려 주세요.',
                         'records': questions, 'actual_answer': None,
                         'scope': 'Qualitative visible clipping assessment only; not exact articulatory timestamps, original training pixel approval, or training readiness.',
                         'training_ready': False})
    summary.pop('question_batch_issued_in_chat', None)
    summary.update({'status': 'in_progress_all8_audio_confirmed_visual_review_pending',
                    'actual_phrase_answers': 13, 'original_end_clipping_reports': 4,
                    'selected_phrase_audio_confirmed': 8, 'evidence': evidence,
                    'next_question_batch': bind(question_path), 'next_question_batch_issued_in_chat_at_snapshot': False,
                    'latest_question_batch': bind(BASE / 'question_batches/007_Q08_plus80ms.json'),
                    'latest_question_batch_answered': True,
                    'new_review_videos': 8, 'new_review_display_frames': displays,
                    'new_review_videos_this_revision': 6,
                    'next': 'Obtain actual qualitative visual boundary review for the exact eight currently selected phrase/context videos; preserve audio confirmation scope and all rejected candidates.',
                    'stage5g_complete': False})
    summary_path = BASE / 'results/stage5g_summary_r4.json'
    write(summary_path, summary)
    handoff = read(BASE / 'results/next_step_handoff_r3.json')
    handoff.update({'schema': 'offline-stage5g-handoff-r4', 'summary': bind(summary_path),
                    'current_audio_confirmed': [r['review_id'] for r in summary['records']],
                    'pending_existing_questions': [], 'not_yet_asked': [],
                    'matching_visual_preview_not_yet_prepared': [],
                    'pending_visual_question': bind(question_path),
                    'next': summary['next']})
    write(BASE / 'results/next_step_handoff_r4.json', handoff)
    print(f'r4: audio 8/8, qualitative visual 0/8; revised preview total {displays} display frames.')


if __name__ == '__main__':
    main()
