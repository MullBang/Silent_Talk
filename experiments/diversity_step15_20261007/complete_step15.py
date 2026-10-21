"""Record prepared datasets and a bounded design, without claiming model training."""
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import data_plan
import experiment_plan

BASE = Path(__file__).resolve().parent
read, bind, require = data_plan.read, data_plan.bind, data_plan.require


def main():
    data = data_plan.verify_plan()
    design = read(BASE / 'training_design.json')
    experiment_plan.validate_plan(design, data, read(experiment_plan.PRIOR))
    require(design['data_plan'] == bind(BASE / 'data_plan.json'), 'Design dataset reference changed')
    audit = read(BASE / 'results/independent_audit.json')
    require(audit['schema'] == 'lipnet-step15-independent-audit-v1' and audit['status'] == 'passed' and
            audit['data_plan'] == bind(BASE / 'data_plan.json') and
            audit['training_design'] == bind(BASE / 'training_design.json'), 'Missing or stale independent audit')
    require(audit['student_input_npz_decoded'] == 88 and audit['training_teacher_feature_npz_decoded'] == 72 and
            audit['development_teacher_feature_npz_opened'] == audit['teacher_pixel_arrays_decoded'] == 0,
            'Incomplete or out-of-scope cache audit')
    require(audit['planned_runs'] == 4 and audit['planned_optimizer_updates'] == 14400 and
            audit['planned_presentations'] == 28800 and audit['planned_student_frames'] ==
            design['budget']['student_frames_presented'], 'Planned budget mismatch')
    require(audit['optimizer_updates_executed'] == audit['new_model_inferences'] == audit['checkpoint_tensors_loaded'] ==
            audit['raw_media_opened'] == 0 and audit['service_weights_preserved'] and
            audit['source_and_code_hashes_unchanged_before_and_after_audit'], 'Preparation scope or preservation failure')
    suites = ET.parse(BASE / 'results/test_results.xml').getroot().findall('testsuite')
    require(suites and sum(int(suite.attrib['tests']) for suite in suites) > 0, 'Missing test evidence')
    require(all(all(int(suite.attrib.get(key, '0')) == 0 for key in ('failures', 'errors', 'skipped')) for suite in suites),
            'Tests did not all pass')
    rows = {row['id']: row for row in data['selection']['records']}
    listing = BASE / 'results/selected_utterances.txt'
    lines = ['15단계 선정 문장 목록 — 학습 실행 전',
             '72문장 확대군에 24문장 대조군이 포함됩니다. 검증16문장은 별도 녹화입니다.',
             '이 목록은 새 사람 검수 승인이나 인식 성공을 뜻하지 않습니다.', '']
    for group, label in [('train72', '학습72'), ('dev16', '검증16')]:
        lines.append('['+label+']')
        for sid in data['selection'][group+'_ids']:
            row = rows[sid]
            marker = '대조24 포함' if row['in_control24'] else '확대72 추가' if group == 'train72' else '학습 사용 금지'
            lines.extend([f"{sid} | {marker}", f"녹화: {row['source_stem']} | {row['start_time']:.6f}~{row['end_time']:.6f}초",
                          row['text'], ''])
    with listing.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    artifacts = ['data_plan.json', 'training_design.json', 'results/test_results.xml',
                 'results/independent_audit.json', 'results/selected_utterances.txt']
    result = {'schema': 'lipnet-step15-summary-v1', 'status': 'completed_data_and_training_design_preparation',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(__file__),
              'artifacts': {path: bind(BASE / path) for path in artifacts},
              'tests_passed': sum(int(suite.attrib['tests']) for suite in suites),
              'selected_train24': 24, 'selected_train72': 72, 'selected_dev16': 16, 'selected_unique_records': 88,
              'training_speakers': data['selection']['speaker_order'],
              'recording_counts': {'train24': 16, 'train72': 16, 'dev16': 8},
              'student_frames_by_cohort': {group: data['selection']['statistics'][group+'_frames']
                                         for group in ('train24', 'train72', 'dev16')},
              'post_selection_corpus_profile': design['post_selection_corpus_profile'],
              'planned_runs': 4, 'planned_optimizer_updates': 14400, 'planned_evaluation_rows': 824,
              'planned_presentations': 28800, 'planned_student_frames': design['budget']['student_frames_presented'],
              'same_updates_not_same_frames_flops_or_wall_time': True,
              'new_optimizer_updates': 0, 'new_model_inferences': 0, 'new_teacher_forwards': 0,
              'downloads': 0, 'raw_media_read': False, 'checkpoint_tensors_loaded': 0,
              'service_weights_updated': False, 'new_human_approvals_added': 0,
              'measured_recognition_improvement': None,
              'step16_execution_required': True, 'training_runtime_and_resume_compatibility_verified': False,
              'next_step': 'Implement and validate the new frozen executor including LR boundaries and resume, then run four bounded fresh B_feature jobs and all fixed evaluations.'}
    data_plan.verify_plan()
    output = BASE / 'results/step15_summary.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'summary': bind(output), 'selected_utterances': bind(listing), 'new_optimizer_updates': 0}))


if __name__ == '__main__':
    main()
