"""Publish the completed metadata-only stage and freeze its reviewable evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from prepare_inventory import HERE, ROOT, read, ref, save

REPORT = ROOT / 'docs/녹화형_단계학습_2026-10-12/05e_구절다양성_조사와_검수설계.md'


def publish() -> None:
    """Connect actual inventory, selection, audits and first review batch."""
    try:
        protection = read(HERE / 'results/protection_summary.json')
        inventory = read(HERE / 'inventory/summary.json')
        contexts = read(HERE / 'inventory/shared_phrase_context_diversity.json')
        selection = read(HERE / 'design/selection_summary.json')
        queue = read(HERE / 'design/review_queue.json')
        tests = [HERE / 'results/protection_tests.json',
                 HERE / 'inventory/unit_test_results_r3.json',
                 HERE / 'design/test_selection_results.json']
        test_values = [read(path) for path in tests]
        assert test_values[0]['passed'] == 6 and test_values[0]['failed'] == 0
        assert test_values[1]['passed'] is True and test_values[1]['tests_run'] == 16
        assert test_values[2]['status'] == 'passed' and test_values[2]['exit_code'] == 0
        audits = [HERE / 'audit' / name for name in [
            'source_preflight.json', 'protection_audit.json', 'inventory_audit.json',
            'shared_context_audit.json', 'design_audit.json']]
        assert all(read(path)['status'] == 'passed' for path in audits)
        assert len(queue['records']) == 20 and not queue['training_ready']
        assert selection['queue_role_counts'] == {
            'proposed_train': 13, 'same_expression_probe': 3, 'new_phrase_probe': 4}
        assert selection['first_batch']['speaker_count'] == 8
        assert 'pending final selection' not in REPORT.read_text(encoding='utf-8')
        source_paths = [HERE / 'results/protection_summary.json', HERE / 'inventory/summary.json',
                        HERE / 'inventory/coverage.json',
                        HERE / 'inventory/shared_phrase_context_diversity.json',
                        HERE / 'design/selection_policy.json', HERE / 'design/review_queue.json',
                        HERE / 'design/role_proposal.json', HERE / 'design/selection_summary.json',
                        HERE / 'data/protected_scope.json', Path(__file__)]
        summary = {
            'schema': 'offline-stage5e-summary-v1', 'client_date': '2026-10-18',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'completed_metadata_diversity_inventory_and_review_design',
            'report': ref(REPORT), 'source_bindings': [ref(p) for p in source_paths],
            'protection': protection,
            'inventory_counts': {key: inventory[key] for key in [
                'parent_count', 'speaker_count', 'all_angle_recording_count',
                'candidate_count', 'single_eojeol_candidate_count', 'phrase_candidate_count',
                'frequency_counts']},
            'shared_phrase_context_counts': {key: contexts[key] for key in [
                'shared_phrase_surface_count',
                'shared_phrase_surfaces_with_at_least_two_distinct_parent_transcripts',
                'shared_phrase_surfaces_with_one_repeated_full_parent_transcript']},
            'selection': selection,
            'tests': {'reports': [ref(p) for p in tests],
                      'passed_count': 6 + 16 + test_values[2]['tests_run'],
                      'whole_project_tests': False,
                      'earlier_inventory_test_fixture_failures_preserved': True},
            'independent_audits': [ref(p) for p in audits],
            'scope': {'metadata_only': True, 'raw_media_extractions': 0,
                      'new_neural_calls': 0, 'new_optimizer_updates': 0,
                      'new_clips': 0, 'new_splits': 0, 'new_human_approvals': 0,
                      'downloads': 0, 'service_changes': 0,
                      'checkpoint_tensor_deserializations': 0},
            'source_training_ready': False, 'source_provisional': True,
            'prior5c_strict_fit_gate_passed': False,
            'preserved_boundary_flags': {'existing_thirty_cache_clips': 2,
                                         'existing_train_phrase_subset': 1},
            'interpretation': [
                'The existing pool supports a small review design; recognition improvement has not been measured.',
                'Overlapping transcript windows are not independent videos or approved training clips.',
                'Only one of four proposed new-phrase probes has every eojeol observed in proposed training.',
                'Historical candidate exposure and familiar speakers prevent an untouched-final-test claim.',
                'Proposed roles and recording reservations do not change the existing execution split.'],
            'next_stage': '5f', 'next_stage_status': 'not_executed',
            'immediate_additional_download_needed': False,
            'automatic_sentence_expansion': False,
        }
        save(HERE / 'results/stage5e_summary.json', summary)
        handoff = {
            'schema': 'offline-stage5e-next-step-handoff-v1', 'client_date': '2026-10-18',
            'status': 'metadata_design_complete_first_review_batch_pending',
            'summary': ref(HERE / 'results/stage5e_summary.json'), 'report': ref(REPORT),
            'queue': ref(HERE / 'design/review_queue.json'),
            'role_proposal': ref(HERE / 'design/role_proposal.json'),
            'protected_scope': ref(HERE / 'data/protected_scope.json'),
            'first_batch': selection['first_batch'],
            'next_stage': '5f', 'next_scope': 'Separate source binding, parent transcript/audio review and phrase alignment candidates for Q01-Q08 only.',
            'next_tasks': [
                'Use the exact eight parent/candidate identities; retain the full twenty-candidate queue without filling quotas automatically.',
                'Verify existing TAR member, label, measured offset and recording identities before limited temporary source extraction.',
                'Prepare exact full-parent audio and visible source transcripts for actual user review; preserve old labels and record any correction separately.',
                'Use acoustic alignment only for annotation assistance; handle OOV and clipped candidate boundaries explicitly.',
                'Review exact phrase audio with surrounding context and mouth motion before approving extraction bounds.',
                'Keep original-grid PTS/25fps/offset-once rules, measured frame counts and CTC feasibility for a subsequent explicit cache preparation stage.'],
            'future_model_input': 'Lip video only; annotation audio is never student input.',
            'new_training_allowed_in_next_review_stage': False,
            'new_cache_auto_promotion': False,
            'reserved_probe_recordings': selection['reserved_recording_count'],
            'reserved_eligible_parent_count': selection['reserved_eligible_parent_count'],
            'reserved_unselected_sibling_count': selection['reserved_unselected_sibling_count'],
            'actual_new_clips': 0, 'actual_new_human_approvals': 0,
            'training_ready': False, 'source_provisional': True,
            'original_strict_gate_unchanged': True,
            'current_dev3_and_all_historical_protection_preserved': True,
            'held_acoustic_OOV_parent_still_held': 'lip_J_1_F_03_C293_A_012_019',
            'pending_old_listening_answers_remain_pending': True,
            'immediate_additional_download_needed': False, 'immediate_colab_needed': False,
            'if_sources_or_coverage_insufficient': 'Explain specific missing source/phrase/speaker evidence to the user before requesting additional data.',
            'automatic_sentence_stage6': False, 'service_changes': 0,
        }
        save(HERE / 'results/next_step_handoff.json', handoff)
        print(json.dumps({'status': summary['status'], 'tests': summary['tests']['passed_count'],
                          'review_parents': 20, 'next_batch': 8}, ensure_ascii=True))
    except Exception as exc:
        raise RuntimeError('Cannot publish an incomplete metadata stage') from exc


def freeze() -> None:
    """Bind finalized stage files and report, then verify every stored digest."""
    try:
        read(HERE / 'results/stage5e_summary.json')
        assert read(HERE / 'audit/final_source_preservation.json')['status'] == 'passed'
        assert read(HERE / 'audit/final_report_audit.json')['status'] == 'passed'
        destination = HERE / 'results/completion_manifest.json'
        paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p != destination
                       and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)
        paths.append(REPORT)
        artifacts = [ref(p) for p in paths]
        save(destination, {'schema': 'offline-stage5e-completion-manifest-v1',
                           'status': 'completed_frozen_metadata_design',
                           'created_utc': datetime.now(timezone.utc).isoformat(),
                           'artifact_count': len(artifacts), 'artifacts': artifacts,
                           'exclusions': ['self', 'Python runtime caches', 'root AGENTS.md rolling handoff'],
                           'future_edits': 'Use a new version; preserve these code, plan and result bytes.'})
        for item in read(destination)['artifacts']:
            assert ref(Path(item['path'])) == item
        print(json.dumps(ref(destination) | {'artifact_count': len(artifacts)}, ensure_ascii=True))
    except Exception as exc:
        raise RuntimeError('Cannot freeze unverified metadata artifacts') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    arguments = parser.parse_args()
    freeze() if arguments.freeze else publish()
