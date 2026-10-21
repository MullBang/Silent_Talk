"""Record completed diagnostics and explicit unfinished human review."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(ref):
    if bind(ref['path']) != ref:
        raise ValueError('Changed bound artifact: ' + ref['path'])


def save(path, doc):
    if path.exists():
        raise FileExistsError('Preserve existing summary/check record')
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    labels = read(BASE / 'results/label_targets.json')
    context = read(BASE / 'results/context_asr_results.json')
    package = read(BASE / 'results/review_package.json')
    core_audit = read(BASE / 'results/independent_stage07_audit.json')
    review_audit = read(BASE / 'results/review_integration_audit.json')
    if core_audit['status'] != 'passed' or review_audit['status'] != 'passed':
        raise ValueError('Independent audit not passed')
    for name in ('builder', 'template', 'data', 'html'):
        verify(package[name])
    for ref in package['inputs'].values():
        verify(ref)
    for ref in core_audit['inputs'].values():
        verify(ref)
    for key in ('code', 'upstream_independent_audit', 'review_package', 'review_data', 'review_html', 'builder', 'template'):
        verify(review_audit[key])
    previous = BASE.parent / 'new_data_step06_20261007/results/step06_summary.json'
    if bind(previous)['sha256'] != '36cfe8ab969cccc8434533a73ac28df6d7813f51c5be50b4008bff4641ac3ff1':
        raise ValueError('Previous summary changed')
    now = datetime.now(timezone.utc).isoformat()
    ui_check = {
        'schema': 'lipnet-step07-browser-review-check-v1', 'created_utc': now,
        'method': 'actual_browser_DOM_filter_search_media_and_screenshot_inspection',
        'html': package['html'], 'screenshot': bind(BASE / 'results/review_screenshot.png'),
        'filter_counts': {'both_differ': 29, 'one_supported': 6, 'both_supported': 85, 'all': 120, 'scoped': 2},
        'search_id': 'lip_J_1_F_05_C057_A_005_005', 'search_results': 1,
        'checked_transcript_or_boundary_radios': 0, 'human_records_created_by_agent': 0,
        'loaded_media_sample': {'id': 'lip_J_1_F_05_C057_A_005_005', 'audio_durations': [5.290688, 6.290688],
                               'video_duration': 5.3053, 'media_ready_state': 4, 'media_errors': 0, 'roi_images_loaded': 3},
        'captured_console_warnings_errors': 0,
        'all120_human_listening_verified': False, 'human_export_download_tested': False,
        'limits': ['Media loading and UI behavior checks do not establish speech content or complete playback of all media.']}
    save(BASE / 'results/review_ui_check.json', ui_check)
    artifacts = [
        'workflow_policy.json', 'label_plan.json', 'context_plan.json', 'label_overrides.json', 'user_alternative_choice.json',
        'results/label_targets.json', 'results/context_asr_results.json', 'results/independent_stage07_audit.json',
        'results/review_integration_audit.json', 'results/review_package.json', 'results/review_ui_check.json',
        'results/tests.log', 'research/official_transcription_policy.json', 'summarize_stage07.py']
    summary = {
        'schema': 'lipnet-step07-summary-v1', 'created_utc': now,
        'status': 'normalization_and_context_diagnosis_complete_human_review_pending',
        'previous_step_summary': bind(previous),
        'artifacts': {p: bind(BASE / p) for p in artifacts},
        'report': bind(ROOT / 'docs/실행계획_2026-10-07/07_전사정규화_및_문맥검수.md'),
        'cohort': {'records': 120, 'train': 96, 'val': 24, 'speakers': 15, 'ids_order_roles_preserved': True},
        'label_targets': {'ordinary_targets_unchanged': 119, 'scoped_user_resolved_alternatives': 1,
                          'target_changes': labels['target_changes'], 'ctc_feasible_records': 120},
        'context_asr': context['summary'],
        'canonical_reference_comparisons_posthoc': package['canonical_comparison_metrics'],
        'review_groups': package['group_counts'], 'priority_review_count': 35,
        'word_time_diagnostic': package['word_time_diagnostic'],
        'tests': {'label_policy': 24, 'context_asr': 15, 'independent_audit': 17, 'total_passed': 56},
        'audits': {'core': core_audit['status'], 'review_integration': review_audit['status']},
        'human_evidence': {'alternative_choice_only': 1, 'prior_extra_greeting_absent_only': 1,
                           'full_transcript_approvals_added': 0, 'boundary_approvals_added': 0},
        'source_labels_modified': False, 'existing_tokenizer_modified': False,
        'training_ready': False, 'student_training_updates': 0, 'service_weights_changed': False,
        'additional_dataset_downloaded': False, 'tar_extraction_this_step': False, 'audio_uploaded': False,
        'next_actions': [
            'Listen to the35 priority differences and save actual transcript and boundary decisions separately.',
            'Review unresolved boundaries using exact and context audio; do not use ASR word timings alone to change labels.',
            'Keep reviewed and held rows with original split roles; declare actual inclusion counts and policy before a new training plan.',
            'Prepare student inputs and training-only teacher features only after the chosen dataset plan is frozen.'],
        'limits': ['Same-model context diagnosis is not independent ASR validation or visual lipreading evaluation.',
                   'Full-context reference mismatches can include context effects; word-midpoint selection failed here.',
                   'Display priorities and machine consistency are not human ground truth or training authorization.',
                   'Dataset-specific paired-notation convention remains unverified; the one actual user choice is scoped.']}
    save(BASE / 'results/step07_summary.json', summary)
    print(json.dumps({'status': summary['status'], 'summary': bind(BASE / 'results/step07_summary.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
