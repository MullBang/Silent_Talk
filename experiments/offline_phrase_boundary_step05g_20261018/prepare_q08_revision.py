"""Prepare a bounded Q08 audio endpoint trial after an actual clipping report."""
from array import array
from fractions import Fraction
from pathlib import Path

from prepare_q01_revision import BASE, PREVIOUS, bind, read, write
from prepare_q05_q06_revisions import pcm, rational, save_slice


def main():
    original_path = PREVIOUS / 'phrase_media_r2/Q08/record.json'
    original = read(original_path)
    parents_path = PREVIOUS / 'effective_parent_inputs.json'
    parent = next(r for r in read(parents_path)['records'] if r['review_id'] == 'Q08')
    parent_path = Path(parent['audio']['path'])
    parent_media_path = Path(parent['media_record']['path'])
    media = read(parent_media_path)
    context_path = Path(media['context_audio']['path'])
    old_audio_path = Path(original['phrase_audio']['path'])
    alignment_path = PREVIOUS / 'alignment/execution/Q08_alignment.json'
    frozen = {r['path']: r for r in read(PREVIOUS / 'results/completion_manifest.json')['artifacts']}
    used = [original_path, parents_path, parent_path, parent_media_path, context_path, old_audio_path, alignment_path]
    for path in used:
        assert bind(path) == frozen[str(path)]
    origin = parent['original_recording_first_sample_48k']
    shift = Fraction(origin - media['context_audio']['source_first_sample'], 3)
    assert shift == 5600
    start = original['aligned_proposal']['local_audio_start_sample_16k']
    old_end = original['aligned_proposal']['local_audio_end_sample_16k']
    assert (start, old_end) == (1000, 22760)
    endpoint = old_end + 1280
    alignment = read(alignment_path)
    assert alignment['canonical_parent_text'][11] == '강'
    next_spike = alignment['alignment']['token_spans'][11]['start_frame'] * 320 + 40
    assert next_spike == 26600 and next_spike - endpoint == 2560
    assert pcm(parent_path)[start * 2:old_end * 2] == pcm(old_audio_path)
    output = BASE / 'revisions/Q08_end_plus80ms'
    output.mkdir(exist_ok=False)
    phrase = save_slice(parent_path, start, endpoint, output / 'phrase.wav')
    context = save_slice(context_path, start + 5600 - 3200, endpoint + 5600 + 3200, output / 'context.wav')
    assert pcm(Path(phrase['path'])).startswith(pcm(old_audio_path))
    reference = pcm(context_path)[(start + 5600) * 2:(endpoint + 5600) * 2]
    diff = [abs(a - b) for a, b in zip(array('h', pcm(Path(phrase['path']))), array('h', reference), strict=True)]
    record = {
        'schema': 'offline-stage5g-end-revision-v1', 'review_id': 'Q08', 'revision_id': 'Q08_end_plus80ms_r1',
        'parent_id': original['parent_id'], 'candidate_id': original['candidate_id'], 'text': '전동 킥보드를 타고',
        'trigger': bind(BASE / 'user_evidence/012_Q08_original_end_clipping.json'),
        'original_candidate': bind(original_path), 'effective_parent_inputs': bind(parents_path),
        'media': {'phrase': phrase, 'context': context}, 'parent_origin_sample_48k': origin,
        'parent_local_start_sample_16k': start, 'parent_local_end_sample_16k': endpoint,
        'old_parent_local_end_sample_16k': old_end, 'extension_samples_16k': 1280, 'extension_seconds': .08,
        'start_unchanged': True, 'text_unchanged': True, 'old_audio_is_exact_prefix': True,
        'choice': 'Try +80ms after the reported missing 고; ends 160ms before next 강 CTC occupancy. Occupancy does not certify actual speech onset or a silent gap.',
        'diagnostics': {'source_alignment': bind(alignment_path), 'next_word': '강변의',
                        'next_character_ctc_start_sample_16k': next_spike,
                        'candidate_to_next_ctc_occupancy_samples': next_spike - endpoint,
                        'this_is_not_a_verified_speech_gap': True},
        'audio_start_seconds_exact': rational(Fraction(origin, 48000) + Fraction(start, 16000)),
        'audio_end_seconds_exact': rational(Fraction(origin, 48000) + Fraction(endpoint, 16000)),
        'video_start_seconds_exact': rational(Fraction(origin + parent['video_offset_native_samples'], 48000) + Fraction(start, 16000)),
        'video_end_seconds_exact': rational(Fraction(origin + parent['video_offset_native_samples'], 48000) + Fraction(endpoint, 16000)),
        'video_offset_applied_count': 1,
        'parent_vs_context_pcm': {'unequal_samples': sum(d != 0 for d in diff),
                                 'max_absolute_int16_difference': max(diff),
                                 'question_uses_confirmed_parent_pcm': True, 'context_is_separately_resampled': True},
        'phrase_within_context_samples_16k': [3200, 3200 + phrase['samples']],
        'extension_is_not_measured_boundary_error': True, 'phrase_audio_boundary_approved': False,
        'phrase_visual_boundary_approved': False, 'actual_split': None, 'frame_count': None,
        'ctc_feasible': None, 'training_ready': False, 'new_video_created': False,
        'new_tar_payload_bytes': 0, 'new_neural_calls': 0, 'training_updates': 0, 'code': bind(Path(__file__)),
    }
    write(output / 'record.json', record)
    write(BASE / 'question_batches/007_Q08_plus80ms.json', {
        'schema': 'offline-stage5g-revision-questions-v1',
        'records': [{'review_id': 'Q08', 'text': record['text'],
                     'question': 'Q08 끝을 0.08초 늘린 새 구절 음성에서 “전동 킥보드를 타고”의 ‘고’까지 온전히 들리고, 뒤의 “강변의”나 다른 말은 섞이지 않았나요?',
                     'revision': bind(output / 'record.json'), 'question_audio': bind(output / 'phrase.wav'),
                     'actual_answer': None, 'phrase_audio_boundary_approved': False,
                     'phrase_visual_boundary_approved': False, 'training_ready': False}],
        'training_ready': False,
    })
    for path in used:
        assert bind(path) == frozen[str(path)]
    print(f'Q08 +80ms: {phrase["samples"]} samples / {phrase["duration_seconds"]} seconds; PCM differences {sum(d != 0 for d in diff)}')


if __name__ == '__main__':
    main()
