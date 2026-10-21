"""Bind actual revision answers and prepare an unapproved intermediate endpoint."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

from prepare_q01_revision import BASE, bind, read, write
from prepare_q05_q06_revisions import pcm, rational, save_slice


def main():
    batch_path = BASE / 'question_batches/003_Q05_Q06_revisions.json'
    questions = {r['review_id']: r for r in read(batch_path)['records']}
    for qid, number, index, answer, approved, suffix in [
        ('Q05', '006', 0, '온전히 들림·잘림·혼입 없음', True, 'revised_audio_confirmed'),
        ('Q06', '007', 1, '뒤의 다른 말이 섞임', False, 'plus80ms_mixing_reported'),
    ]:
        row = questions[qid]
        record = read(Path(row['revision']['path']))
        assert bind(Path(row['question_audio']['path'])) == row['question_audio']
        write(BASE / f'user_evidence/{number}_{qid}_{suffix}.json', {
            'schema': 'offline-stage5g-actual-answer-v1', 'review_id': qid,
            'revision_id': record['revision_id'],
            'question_item_id': f'["request_user_input_async","call_24356cfdf7134061b0212ea4e81d6b88",{index}]',
            'question': row['question'], 'answer': answer, 'source': 'actual_user_chat_reply',
            'question_manifest': bind(batch_path), 'question_audio': row['question_audio'],
            'revision_record': row['revision'],
            'scope': 'This exact revised phrase audio only; no original-candidate or visual approval.',
            'phrase_audio_boundary_approved': approved, 'phrase_visual_boundary_approved': False,
            'original_candidate_approved': False, 'training_ready': False,
        })
    old_path = Path(questions['Q06']['revision']['path'])
    old = read(old_path)
    record = deepcopy(old)
    start, endpoint = 40, 19560
    assert endpoint == (old['old_parent_local_end_sample_16k'] + old['parent_local_end_sample_16k']) // 2
    directory = BASE / 'revisions/Q06_end_plus40ms'
    directory.mkdir(exist_ok=False)
    phrase = save_slice(Path(old['media']['phrase']['source']['path']), start, endpoint, directory / 'phrase.wav')
    context = save_slice(Path(old['media']['context']['source']['path']), 2440, 28360, directory / 'context.wav')
    assert pcm(Path(old['media']['phrase']['path'])).startswith(pcm(Path(phrase['path'])))
    assert pcm(Path(phrase['path'])) == pcm(Path(context['path']))[6400:6400 + phrase['samples'] * 2]
    record.update({
        'revision_id': 'Q06_end_plus40ms_r1',
        'trigger': bind(BASE / 'user_evidence/007_Q06_plus80ms_mixing_reported.json'),
        'previous_rejected_revision': bind(old_path),
        'media': {'phrase': phrase, 'context': context},
        'parent_local_end_sample_16k': endpoint, 'extension_samples_16k': 640,
        'extension_seconds': 0.04,
        'choice': 'Midpoint between the user-reported clipped original and the +80ms candidate reported to contain other speech. This is an unverified trial, not a measured boundary.',
        'diagnostics': {'source_alignment': old['diagnostics']['source_alignment'],
                        'next_word': '싶으면', 'next_character_ctc_start_sample_16k': 20520,
                        'candidate_to_next_ctc_occupancy_samples': 960,
                        'this_is_not_a_verified_speech_gap': True,
                        'actual_intruding_word_identified_by_user': False},
        'audio_end_seconds_exact': rational(Fraction(old['audio_end_seconds_exact']['numerator'], old['audio_end_seconds_exact']['denominator']) - Fraction(1, 25)),
        'video_end_seconds_exact': rational(Fraction(old['video_end_seconds_exact']['numerator'], old['video_end_seconds_exact']['denominator']) - Fraction(1, 25)),
        'phrase_within_context_samples_16k': [3200, 3200 + phrase['samples']],
        'code': bind(Path(__file__)),
    })
    write(directory / 'record.json', record)
    write(BASE / 'question_batches/004_Q06_plus40ms.json', {
        'schema': 'offline-stage5g-revision-questions-v1',
        'records': [{'review_id': 'Q06', 'text': old['text'],
                     'question': 'Q06 원래보다 끝을 0.04초 늘린 새 음성에서는 “전동 킥보드를 타고”의 ‘고’까지 온전히 들리고, 뒤의 다른 말은 섞이지 않았나요?',
                     'revision': bind(directory / 'record.json'), 'question_audio': bind(directory / 'phrase.wav'),
                     'actual_answer': None, 'phrase_audio_boundary_approved': False,
                     'phrase_visual_boundary_approved': False, 'training_ready': False}],
        'training_ready': False,
    })
    print(f'Actual answers recorded; Q06 midpoint prepared: {phrase["samples"]} samples, {phrase["duration_seconds"]} seconds')


if __name__ == '__main__':
    main()
