"""Summarize completed automatic evidence while preserving manual review pending."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_review import BASE, bind, read, require, sha


def main():
    source = read(BASE / 'source_plan.json')
    media = read(BASE / 'results/media_manifest.json')
    asr = read(BASE / 'results/asr_results.json')
    differences = read(BASE / 'results/asr_difference_review.json')
    visual = read(BASE / 'results/visual_review.json')
    review = read(BASE / 'results/review_package.json')
    media_audit = read(BASE / 'results/independent_media_audit.json')
    asr_audit = read(BASE / 'results/independent_asr_audit.json')
    require(media_audit['status'] == asr_audit['status'] == 'passed', 'Both independent audits must pass')
    media_ref, asr_ref = bind(BASE / 'results/media_manifest.json'), bind(BASE / 'results/asr_results.json')
    require(media_audit['media_manifest'] == media_ref and asr_audit['inputs']['results/asr_results.json'] == asr_ref,
            'Independent audits refer to different evidence')
    require(asr['media_manifest'] == media_ref and review['asr_results'] == asr_ref and
            review['media_manifest'] == visual['media_manifest'] == media_ref,
            'Review/evidence bindings differ')
    ids = [row['id'] for row in source['records']]
    require(len(ids) == len(set(ids)) == 120 and
            ids == [row['id'] for row in media['clips']] == [row['id'] for row in asr['records']] ==
            [row['id'] for row in visual['records']], 'Final summary lost original120 cohort')
    for spec in review.values():
        if isinstance(spec, dict) and set(spec) == {'path', 'sha256'}:
            require(sha(spec['path']) == spec['sha256'], 'Review package dependency changed')
    sources = [read(ref['path']) for ref in media['sources']]
    global_scores = [anchor['score'] for item in sources for anchor in item['anchors']]
    local_scores = [clip['local_check']['score'] for clip in media['clips']]
    offsets = Counter(f"{item['offset_summary']['median_offset_seconds']*1000:.6f}" for item in sources)
    diagnostics = read(BASE / 'diagnosis/lip_J_6_M_03_C646_A_004/result.json')
    selected_bytes = sum(member['size'] for item in source['sources'] for member in item['members'].values())
    duration = sum(row['end_time']-row['start_time'] for row in source['records'])
    alternatives = [{'id': row['id'], 'split': row['split'], 'source_text': row['source_text'],
                     'current_normalized_text': row['text'], 'resolution': 'Pending a separately versioned transcript policy; do not concatenate alternatives as confirmed speech'}
                    for row in source['records'] if ')/(' in row['source_text']]
    paths = ['source_plan.json', 'media_plan.json', 'asr_setup.json', 'asr_plan.json', 'review_policy.json',
             'results/media_manifest.json', 'results/asr_results.json', 'results/asr_difference_review.json',
             'results/independent_media_audit.json', 'results/independent_asr_audit.json',
             'results/visual_review.json', 'results/review_package.json',
             'results/user_review_001.json',
             'diagnosis/lip_J_6_M_03_C646_A_004/result.json', 'ui_verification.json']
    result = {
        'schema': 'lipnet-new-data-step06-summary-v1',
        'status': 'automated_audit_complete_human_listening_pending',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'code': bind(Path(__file__)),
        'evidence': {path: bind(BASE / path) for path in paths},
        'counts': {'recordings': 30, 'sentences': 120, 'train': 96, 'val': 24, 'speakers': 15,
                   'train_speakers': 12, 'val_speakers': 3, 'timing_verified': 120,
                   'global_waveform_checks': 90, 'local_waveform_checks': 120,
                   'roi_samples_visually_reviewed': 360, 'asr_completed': 120,
                   'manual_listening_confirmed_in_original_results': 0,
                   'actual_user_scoped_listening_findings': 1,
                   'automatically_supported_without_review_flags': review['counts']['automatic_support_only'],
                   'original_rows_requiring_review': review['counts']['human_listening_required']},
        'timing': {'offset_ms_recording_counts': dict(offsets),
                   'minimum_global_correlation': min(score['absolute_correlation'] for score in global_scores),
                   'minimum_local_correlation': min(score['absolute_correlation'] for score in local_scores),
                   'maximum_absolute_local_residual_seconds': max(abs(clip['local_check']['offset_residual_seconds']) for clip in media['clips']),
                   'formula': 'MP4 time = original WAV/label time + recording offset',
                   'interpretation': 'Audio-clock correspondence, not proof of lips/audio sync'},
        'transcript': {'category_counts': differences['category_counts'], 'asr_summary': asr['summary'],
                       'user_scoped_finding': read(BASE / 'results/user_review_001.json'),
                       'alternative_notation_cases': alternatives,
                       'content_or_number_review_ids': [row['id'] for row in differences['records'] if row['category'] in
                                                       ('possible_content_difference', 'possible_number_expression_difference')],
                       'warning': 'No ASR mismatch is automatically a label error. All120 carry a conservative start-boundary cue; it is not a proven clipped word.'},
        'resources': {'selected_original_member_bytes_read': selected_bytes,
                      'main_source_materializations': 30, 'additional_diagnostic_video_materializations': 1,
                      'maximum_single_source_temporary_payload_bytes': max(sum(m['size'] for m in s['members'].values()) for s in source['sources']),
                      'media_artifact_bytes': media_audit['artifact_bytes_checked'],
                      'retained_asr_model_bytes': sum(Path(s['path']).stat().st_size for s in read(BASE / 'asr_setup.json')['model_files']),
                      'selected_audio_duration_seconds': duration,
                      'source_processing_seconds_sum': sum(item['elapsed_seconds'] for item in sources),
                      'asr_seconds': asr['elapsed_seconds_this_run'],
                      'temporary_originals_cleaned': media['temporary_originals_cleaned'] and diagnostics['temporary_originals_cleaned']},
        'scope': {'new_student_training': False, 'service_weights_changed': False, 'old512_frozen_artifacts_changed': False,
                  'final_test50_used': False, 'additional_aihub_download': False, 'G_drive_used': False,
                  'audio_uploaded': False, 'original_labels_modified': False, 'training_ready': False},
        'next_actions': ['Listen to the27 possible content and5 possible numeric differences first; retain original ID/split and source references.',
                         'Review remaining spacing and boundary cues without treating coarse ASR timestamps as proof of clipping.',
                         'Resolve the alternative transcript notation in a separate reviewed normalization plan; keep original labels intact.',
                         'Freeze a new downstream data/training plan only after declaring how unresolved records are held or handled.'],
        'limits': ['Auxiliary audio ASR scores do not measure visual lipreading accuracy.',
                   'The48kHz lag grid interpolates saved8kHz PCM and is not a new48kHz measurement.',
                   'First/middle/last ROI inspection does not verify all15109 target frames.',
                   'One diagnostic native-frame re-extraction confirms pixel reproducibility only for that frame.',
                   'No human listening judgments were fabricated; future actual user reviews must be stored separately and bound to these hashes.']}
    output = BASE / 'results/step06_summary.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'summary': bind(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
