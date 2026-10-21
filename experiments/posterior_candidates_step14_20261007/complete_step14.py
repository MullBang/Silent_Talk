"""Finalize the posterior-only diagnostic after independent full-score verification."""
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
    scores = read(BASE / 'results/candidate_scores.json')
    audit = read(BASE / 'results/independent_audit.json')
    plan = bind(BASE / 'diagnostic_plan.json')
    require(comparison['status'] == scores['status'] == 'completed' and audit['status'] == 'passed',
            'Missing completed scores, summary or independent audit')
    require(all(value['experiment_plan'] == plan for value in (comparison, scores, audit)), 'Different plans')
    require(audit['comparison'] == bind(BASE / 'results/comparison.json') and
            audit['candidate_scores'] == comparison['candidate_scores'] == bind(BASE / 'results/candidate_scores.json'),
            'Missing or stale independent audit')
    require(audit['source_rows_independently_checked'] == comparison['source_rows'] == 144 and
            audit['main_candidate_pairs_independently_checked'] == comparison['main_candidate_pairs'] == 3600 and
            audit['auxiliary_greedy_pairs_independently_checked'] == comparison['auxiliary_greedy_pairs'] == 144 and
            audit['total_ctc_pairs_independently_checked'] == comparison['total_ctc_pairs'] == 3744,
            'Incomplete independent score audit')
    require(audit['failed_source_rows'] == comparison['failed_source_rows'] == scores['failed_source_rows'] == 0 and
            audit['source_failed_candidate_pairs'] == comparison['source_failed_candidate_pairs'] == 0,
            'Unresolved scoring failures')
    require(audit['new_model_inferences'] == comparison['new_model_inferences'] == scores['new_model_inferences'] == 0 and
            audit['new_optimizer_updates'] == comparison['new_optimizer_updates'] == scores['new_optimizer_updates'] == 0 and
            audit['checkpoint_tensors_deserialized'] == audit['student_input_npz_opened'] ==
            audit['teacher_feature_npz_opened'] == audit['raw_audio_or_video_opened'] == 0 and
            audit['service_weights_preserved'] and not comparison['service_weights_updated'],
            'Posterior-only scope or preservation failure')
    require(audit['preflight_precedes_scoring_contract'] and audit['preflight_source_snapshots_verified'] and
            audit['source_and_code_hashes_unchanged_before_and_after_audit'] and
            audit['original_greedy_transcripts_and_metrics_preserved'], 'Provenance failure')
    require(comparison['generator'] == bind(BASE / 'summarize.py') and
            audit['generator'] == bind(BASE / 'independent_audit.py'), 'Analysis source changed')
    require(not audit['pytorch_imported'] and comparison['reference_candidate_is_oracle'] and
            not comparison['ranking_is_accuracy'] and comparison['length_normalization_is_heuristic'],
            'Independence or interpretation failure')
    suites = ET.parse(BASE / 'results/test_results.xml').getroot().findall('testsuite')
    require(suites and sum(int(suite.attrib['tests']) for suite in suites) > 0, 'No executed tests')
    for suite in suites:
        require(all(int(suite.attrib.get(key, '0')) == 0 for key in ('errors', 'failures', 'skipped')),
                'Tests did not all pass')
    paths = ['diagnostic_plan.json', 'plans/freeze_manifest.json', 'results/test_results.xml',
             'results/independent_preflight.json', 'results/scoring.contract.json',
             'results/candidate_scores.json', 'results/comparison.json', 'results/independent_audit.json']
    result = {'schema': 'lipnet-step14-summary-v1', 'status': 'completed_posterior_candidate_diagnostic',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(__file__),
              'artifacts': {path: bind(BASE / path) for path in paths},
              'test_total_passed': sum(int(suite.attrib['tests']) for suite in suites),
              'test_sources': [bind(BASE / name) for name in ('test_protocol.py', 'test_scoring.py',
                                                           'test_summarize.py', 'test_independent_audit.py')],
              'source_rows': 144, 'main_candidate_pairs': 3600, 'auxiliary_greedy_pairs': 144, 'total_ctc_pairs': 3744,
              'failed_source_rows': 0, 'infeasible_candidate_pairs': comparison['infeasible_candidate_pairs'],
              'saved_posterior_archives_opened': 8, 'saved_posterior_arrays_decoded': 144,
              'saved_posterior_frames_decoded': audit['saved_posterior_frames_decoded'],
              'new_model_inferences': 0, 'new_optimizer_updates': 0, 'new_human_approvals_added': 0,
              'checkpoint_tensors_deserialized': 0, 'raw_media_accessed': False, 'downloads': 0,
              'teacher_features_or_pixel_arrays_opened': 0, 'student_input_npz_opened': 0,
              'source_predictions_changed': False, 'service_weights_updated': False, 'prior_artifacts_preserved': True,
              'reference_candidate_is_oracle': True, 'ranking_is_accuracy': False,
              'length_normalization_is_heuristic': True, 'unseen_speaker_generalization_estimate': False,
              'untouched_test_estimate': False, 'independent_preflight_preceded_scoring': True,
              'independent_float64_ctc_max_abs_total_score_difference':
                  audit['independent_float64_ctc_max_abs_total_score_difference'],
              'normal_results': [{'seed': model['seed'], 'arm': model['arm'], 'groups': model['conditions']['normal']}
                                 for model in comparison['models']],
              'across_models_condition_cohort_counts': comparison['across_models_condition_cohort_counts'],
              'next_step_decision_rule': 'Use the reference-versus-greedy total probabilities and training-candidate '
                                        'ranks to distinguish search limitations from weak transcript separation. '
                                        'Any later recognizer must exclude oracle references and preserve the '
                                        'original greedy comparison and fixed evaluation denominators.'}
    protocol.verify_frozen()
    output = BASE / 'results/step14_summary.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(bind(output)))


if __name__ == '__main__':
    main()
