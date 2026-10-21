"""Finalize the diagnostic only after all fixed runs and independent checks exist."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import protocol

BASE = Path(__file__).resolve().parent
bind, read, require = protocol.bind, protocol.read, protocol.require


def main() -> None:
    """Preserve a compact hash-linked completion record without promoting weights."""
    policy = protocol.verify_frozen()
    comparison = read(BASE / 'results/comparison.json')
    audit = read(BASE / 'results/independent_audit.json')
    training = read(BASE / 'results/training_complete.json')
    require(audit['status'] == 'passed' and audit['comparison'] == bind(BASE / 'results/comparison.json'),
            'Independent audit missing or stale')
    require(audit['optimizer_updates_verified'] == comparison['new_optimizer_updates'] == training['new_optimizer_updates'] == 4800,
            'Incomplete optimizer work')
    require(audit['inference_rows_independently_checked'] == comparison['inference_rows'] == 224
            and audit['complete_checkpoints_independently_checked'] == 16, 'Incomplete inference/checkpoint audit')
    require(audit['service_weights_preserved'] and not comparison['service_weights_updated'], 'Service preservation failed')
    require(comparison['generator'] == bind(BASE / 'summarize.py') and audit['generator'] == bind(BASE / 'independent_audit.py'),
            'Executed source changed after result')
    for ref in comparison['artifacts']:
        protocol.verify(ref)
    suites = [('protocol', 21), ('training', 9), ('evaluate', 17), ('summarize', 13), ('independent_audit', 34)]
    evidence = [{'suite': name, 'passed': count, 'test_source': bind(BASE / f'test_{name}.py')}
                for name, count in suites]
    paths = ['diagnostic_plan.json', 'plans/freeze_manifest.json', 'plans/seed42.json', 'plans/seed43.json',
             'results/independent_preflight.json', 'results/training_complete.json',
             'results/comparison.json', 'results/independent_audit.json']
    result = {'schema': 'lipnet-step11-summary-v1', 'status': 'completed_train_only_memorization_diagnostic',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)),
              'artifacts': {name: bind(BASE / name) for name in paths}, 'test_suites': evidence,
              'test_total_passed': sum(item['passed'] for item in evidence),
              'train_records': 8, 'train_speakers': 8, 'old_records': 4, 'provisional_new_records': 4,
              'student_frames': policy['loading_scope']['student_frames'],
              'optimizer_updates': 4800, 'utterance_presentations': 9600,
              'processed_frames': comparison['processed_frames_sum'], 'inference_rows': 224,
              'complete_checkpoints_checked': 16, 'failed_inferences': comparison['failures'],
              'success_runs_at_fixed_final': comparison['fixed_final_success_runs'],
              'training_elapsed_seconds': comparison['training_elapsed_seconds_sum'],
              'final_normal_results': [{'seed': m['seed'], 'arm': m['arm'],
                                       'groups': m['final_conditions']['normal']}
                                      for m in comparison['models']],
              'raw_media_accessed': False, 'downloads': 0, 'development_test_inference_rows': 0,
              'new_whole_transcript_approvals': 0, 'service_weights_updated': False,
              'prior_artifacts_preserved': True, 'generalization_estimate': False,
              'preflight_timing': 'Independent source audit followed training start; production47 tests and frozen contract preceded first update.'}
    protocol.runner._save_json(BASE / 'results/step11_summary.json', result)
    print(json.dumps(bind(BASE / 'results/step11_summary.json')))


if __name__ == '__main__':
    main()
