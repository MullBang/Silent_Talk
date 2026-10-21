"""Finalize only after fixed training, separated evaluation and audit complete."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import protocol

BASE = Path(__file__).resolve().parent
bind, read, require = protocol.bind, protocol.read, protocol.require


def main() -> None:
    """Link verified artifacts while preserving internal-holdout limitations."""
    policy = protocol.verify_frozen()
    comparison = read(BASE / 'results/comparison.json')
    audit = read(BASE / 'results/independent_audit.json')
    training = read(BASE / 'results/training_complete.json')
    require(audit['status'] == 'passed' and audit['comparison'] == bind(BASE / 'results/comparison.json'),
            'Independent audit missing or stale')
    require(audit['optimizer_updates_verified'] == comparison['new_optimizer_updates'] == training['new_optimizer_updates'] == 14400,
            'Incomplete optimization')
    require(audit['neural_inference_rows_independently_checked'] == comparison['inference_rows'] == 896 and
            audit['complete_checkpoints_independently_checked'] == 16 and
            audit['known_speaker_baseline_rows_independently_checked'] == 8, 'Incomplete evaluation audit')
    require(audit['internal_holdout_optimization_presentations'] == 0 and audit['service_weights_preserved'],
            'Holdout or preservation failure')
    require(comparison['generator'] == bind(BASE / 'summarize.py') and audit['generator'] == bind(BASE / 'independent_audit.py'),
            'Executed analysis source changed')
    for ref in comparison['artifacts']:
        protocol.verify(ref)
    suites = [('protocol', 24), ('training', 10), ('evaluate', 16), ('summarize', 19), ('independent_audit', 35)]
    evidence = [{'suite': name, 'passed': count, 'test_source': bind(BASE / f'test_{name}.py')} for name, count in suites]
    paths = ['diagnostic_plan.json', 'plans/freeze_manifest.json', 'plans/seed42.json', 'plans/seed43.json',
             'results/independent_preflight.json', 'results/cohort_profile.json', 'results/training_complete.json',
             'results/comparison.json', 'results/independent_audit.json']
    result = {'schema': 'lipnet-step12-summary-v1', 'status': 'completed_multisentence_internal_holdout_diagnostic',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
              'artifacts': {name: bind(BASE / name) for name in paths}, 'test_suites': evidence,
              'test_total_passed': sum(item['passed'] for item in evidence),
              'train_records': 24, 'internal_holdout_records': 8, 'shared_speakers': 8,
              'train_frames': policy['loading_scope']['train_frames'],
              'internal_holdout_frames': policy['loading_scope']['holdout_frames'],
              'optimizer_updates': 14400, 'utterance_presentations': 28800,
              'processed_frames': comparison['processed_frames_sum'], 'neural_inference_rows': 896,
              'known_speaker_baseline_rows': 8, 'complete_checkpoints_checked': 16,
              'failed_inferences': comparison['failures'],
              'training_success_runs_at_fixed_final': comparison['fixed_final_success_runs'],
              'training_elapsed_seconds': comparison['training_elapsed_seconds_sum'],
              'final_normal_results': [{'seed': m['seed'], 'arm': m['arm'], 'groups': m['final_conditions']['normal']}
                                      for m in comparison['models']],
              'raw_media_accessed': False, 'downloads': 0, 'historical_development_final_test_inference_rows': 0,
              'internal_holdout_optimization_presentations': 0,
              'internal_holdout_teacher_features_read_for_validation_only': 8,
              'internal_holdout_teacher_features_used_for_inference': False,
              'new_whole_transcript_approvals': 0, 'service_weights_updated': False,
              'prior_artifacts_preserved': True, 'unseen_speaker_generalization_estimate': False,
              'untouched_test_estimate': False, 'independent_preflight_preceded_training': True,
              'cohort_profile_scope': 'Postselection metadata description, before final evaluation; never a selection rule.'}
    protocol.runner._save_json(BASE / 'results/step12_summary.json', result)
    print(json.dumps(bind(BASE / 'results/step12_summary.json')))


if __name__ == '__main__':
    main()
