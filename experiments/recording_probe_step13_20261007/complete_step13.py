"""Finalize only after the fixed recording probe and independent audit pass."""
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import protocol

BASE = Path(__file__).resolve().parent
read, bind, require = protocol.read, protocol.bind, protocol.require


def main():
    policy = protocol.verify_frozen()
    comparison = read(BASE / 'results/comparison.json')
    audit = read(BASE / 'results/independent_audit.json')
    require(comparison['status'] == 'completed' and audit['status'] == 'passed'
            and audit['comparison'] == bind(BASE / 'results/comparison.json'), 'Missing/stale independent audit')
    require(audit['new_neural_inference_rows_independently_checked'] == comparison['new_inference_rows'] == 72
            and audit['reused_neural_inference_rows_independently_checked'] == comparison['reused_inference_rows'] == 72,
            'Incomplete or mixed new/reused inference accounting')
    require(audit['new_optimizer_updates'] == comparison['new_optimizer_updates'] == 0
            and audit['probe_optimization_presentations'] == 0 and audit['service_weights_preserved'],
            'Inference-only scope or preservation failure')
    require(audit['known_speaker_baseline_rows_independently_checked'] == comparison['baseline_rows'] == 12
            and audit['complete_checkpoints_independently_checked'] == 4, 'Incomplete model/baseline audit')
    require(comparison['generator'] == bind(BASE / 'summarize.py') and
            audit['generator'] == bind(BASE / 'independent_audit.py'), 'Executed analysis source changed')
    for reference in comparison['artifacts']:
        protocol.verify(reference)
    test_path = BASE / 'results/test_results.xml'
    suites = ET.parse(test_path).getroot().findall('testsuite')
    require(suites and sum(int(suite.attrib['tests']) for suite in suites) > 0, 'No executed integration tests')
    for suite in suites:
        require(all(int(suite.attrib.get(key, '0')) == 0 for key in ('errors', 'failures', 'skipped')),
                'Integration tests did not all pass')
    records = {row['id']: row for row in policy['selection']['records']}
    counts = {kind: sum(records[sid]['frames'] for sid in policy['selection'][kind+'_ids']) for kind in ('same', 'different')}
    paths = ['diagnostic_plan.json', 'plans/freeze_manifest.json', 'results/test_results.xml',
             'results/independent_preflight.json', 'results/comparison.json', 'results/independent_audit.json']
    result = {'schema': 'lipnet-step13-summary-v1', 'status': 'completed_matched_speaker_recording_probe',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(__file__),
              'artifacts': {path: bind(BASE / path) for path in paths},
              'test_total_passed': sum(int(suite.attrib['tests']) for suite in suites),
              'test_sources': [bind(BASE / name) for name in ('test_protocol.py', 'test_evaluate.py',
                                                           'test_summarize.py', 'test_independent_audit.py')],
              'matched_speakers': comparison['matched_speakers'], 'same_recording_records': 6,
              'different_recording_records': 6, 'student_frames_by_cohort': counts,
              'new_inference_rows': 72, 'reused_inference_rows': 72, 'baseline_rows': 12,
              'new_optimizer_updates': 0, 'probe_optimization_presentations': 0,
              'new_inference_failures': audit['new_inference_failures'],
              'reused_inference_failures': audit['reused_inference_failures'],
              'final_normal_results': [{'seed': m['seed'], 'arm': m['arm'], 'groups': m['final_conditions']['normal']}
                                       for m in comparison['models']],
              'new_human_approvals_added': 0, 'downloads': 0, 'raw_media_accessed': False,
              'student_input_npz_decoded': 6, 'teacher_feature_npz_opened': 0, 'teacher_pixel_arrays_decoded': 0,
              'paired_teacher_compressed_bytes_read_for_whole_file_hash': True,
              'historical_development_held27_test_payloads_opened': 0,
              'service_weights_updated': False, 'prior_artifacts_preserved': True,
              'unseen_speaker_generalization_estimate': False, 'untouched_test_estimate': False,
              'pure_recording_environment_causal_effect': False,
              'independent_preflight_preceded_inference': audit['preflight_precedes_all_inference_contracts']}
    path = BASE / 'results/step13_summary.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(path)))


if __name__ == '__main__':
    main()
