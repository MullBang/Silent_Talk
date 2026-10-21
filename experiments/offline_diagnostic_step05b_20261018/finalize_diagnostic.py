"""Bind verified descriptive findings and the unexecuted next design."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re

from diagnose_ctc import HERE, ROOT, PLAN, MANIFEST, MANIFEST_SHA, binding, check, read, save, sha


def main():
    output = HERE / 'results'
    assert not (output / 'completion_manifest.json').exists()
    assert sha(MANIFEST) == MANIFEST_SHA
    plan = read(PLAN)
    check(plan['code']); check(plan['tests'])
    paths = {
        'ctc': output / 'ctc_candidates.json',
        'loss_output': HERE / 'training_analysis/loss_output_analysis.json',
        'ctc_audit': HERE / 'audit/candidate_audit.json',
        'loss_audit': HERE / 'audit/loss_output_audit.json',
        'preservation': HERE / 'audit/final_preservation.json',
        'ctc_calibration': HERE / 'audit/ctc_calibration.json',
        'analysis_verification': HERE / 'training_analysis/verification_r2.json',
        'design': HERE / 'design/proposed_phrase_only_design.json',
        'design_verification': HERE / 'design/design_verification.json',
    }
    values = {key: read(path) for key, path in paths.items()}
    for key in ['ctc_audit', 'loss_audit', 'preservation', 'ctc_calibration', 'analysis_verification']:
        assert values[key]['status'] == 'passed', (key, values[key]['status'])
    assert values['design_verification']['status'].startswith('passed_')
    check(values['ctc_audit']['source']); check(values['ctc_audit']['plan'])
    ctc = values['ctc']
    assert [ctc[key] for key in ['posterior_rows', 'candidate_pairs', 'greedy_pairs',
                                'ctc_computations', 'impossible_candidate_pairs']] == [360, 4032, 360, 4392, 528]
    B420 = [r for r in ctc['rows'] if r['arm'] == 'B_curriculum' and r['update'] == 420
            and r['split'] == 'train' and r['kind'] == 'phrase']
    final_dev = [r for r in ctc['rows'] if r['update'] == 1260 and r['split'] == 'dev'
                 and r['kind'] == 'full_sentence']
    assert len(B420) == 24 and len(final_dev) == 12
    def observations(rows):
        return {
            'repeated_observations': len(rows),
            'greedy_exact': sum(r['greedy_exact'] for r in rows),
            'reference_more_probable_than_greedy': sum(r['reference_minus_greedy_nll'] < -1e-9 for r in rows),
            'reference_equal_to_greedy': sum(abs(r['reference_minus_greedy_nll']) <= 1e-9 for r in rows),
            'reference_less_probable_than_greedy': sum(r['reference_minus_greedy_nll'] > 1e-9 for r in rows),
            'reference_top_total': sum(r['total_score_ranking']['reference_rank'] == 1 for r in rows),
            'reference_top_per_token': sum(r['per_token_ranking']['reference_rank'] == 1 for r in rows),
        }
    first, last = observations(B420), observations(final_dev)
    assert [first[k] for k in ['reference_more_probable_than_greedy', 'reference_equal_to_greedy',
                               'reference_less_probable_than_greedy']] == [2, 1, 21]
    assert last['reference_less_probable_than_greedy'] == 12 and last['reference_top_total'] == 0
    design = values['design']
    assert design['planned_accounting']['main_updates'] == 3360
    assert design['evaluation']['total_planned_forwards'] == 144
    schedule_checks = []
    for source in design['schedule_bindings']:
        check(source)
        schedule = read(source['path'])
        updates = schedule['updates']
        assert [r['update'] for r in updates] == list(range(1, 1681))
        allowed = {r['clip_id']: r for r in design['train_records']}
        counts = Counter()
        true = dense = tokens = 0
        for row in updates:
            assert row['kind'] == 'phrase' and len(row['clip_ids']) == 2
            for clip_id in row['clip_ids']:
                assert clip_id in allowed
                counts[clip_id] += 1
            lengths = [allowed[clip_id]['T'] for clip_id in row['clip_ids']]
            true += sum(lengths); dense += 2 * max(lengths)
            tokens += sum(allowed[clip_id]['label_length'] for clip_id in row['clip_ids'])
            assert row['lr'] == (0.001 if row['update'] <= 840 else 0.0003 if row['update'] <= 1050 else 0.0001)
        assert len(counts) == 12 and set(counts.values()) == {280}
        assert true == 134680 and tokens == 76160
        assert dense == schedule['accounting']['dense_frames']
        schedule_checks.append({'seed': schedule['seed'], 'updates': len(updates),
                                'true_frames': true, 'dense_frames': dense, 'target_tokens': tokens})
    assert sum(r['dense_frames'] for r in schedule_checks) == 315946
    save(output / 'root_design_check.json', {'status': 'passed', 'checks': schedule_checks,
                                             'scope': 'Metadata arithmetic only, no training'})
    save(output / 'ctc_unit_tests.json', {
        'status': 'passed', 'tests_passed': 14, 'exit_code': 0,
        'command': '.venv-tokenizer/Scripts/python.exe -m pytest experiments/offline_diagnostic_step05b_20261018/test_diagnose_ctc.py -q',
        'observed_output': '14 passed in 0.30s',
        'code': binding(HERE / 'diagnose_ctc.py'), 'tests': binding(HERE / 'test_diagnose_ctc.py'),
    })
    loss = values['loss_output']
    summary = {
        'schema': 'offline-stage5b-completion-summary-v1', 'client_date': '2026-10-18',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'completed_saved_output_diagnosis_and_followup_design',
        'previous_primary_result_unchanged': True,
        'analysis_scope': loss['scope'],
        'CTC_analysis': {k: ctc[k] for k in ['posterior_rows', 'posterior_frames', 'candidate_pairs',
                                            'greedy_pairs', 'ctc_computations', 'impossible_candidate_pairs']},
        'B420_train_phrase': first, 'final_dev_sentences': last,
        'B_phrase_output_groups': [r for r in loss['normal_output_groups']
                                  if r['arm'] == 'B_curriculum' and r['split'] == 'train' and r['kind'] == 'phrase'],
        'last_training_loss_windows': [{'seed': r['seed'], 'arm': r['arm'],
                                        'final60_mean': r['final60']['loss']['mean'],
                                        'difference_from_previous60': r['final60_minus_penultimate60_mean_loss']}
                                       for r in loss['training_runs']],
        'findings_ko': [
            '420회 구절 학습 종료 시 정확1/12·0/12로 충분한 구절 학습 전제가 충족되지 않았다.',
            'B420의24관측 중21개에서 정답 전체CTC확률이 실제greedy보다 낮았고, 최종dev12관측은 모두 낮았다.',
            'blank감소·출력길이증가 이후에도 구절오류가 남아 blank편향만으로 설명할 수 없다.',
            '정답30개는 CTC길이조건을 통과했으며 손실·기울기 수치 오류를 발견하지 못했다.',
            '마지막60평균손실은 모두 낮아졌지만 추가학습효과나 실패의 단일원인을 확정하지 않는다.',
        ],
        'new_optimizer_updates': 0, 'new_model_forward_calls': 0, 'checkpoint_tensor_loads': 0,
        'original_media_reads': 0, 'new_downloads': 0, 'service_changes': 0,
        'source_training_ready': False, 'source_provisional': True,
        'precise_visual_boundary_approval': False, 'existing_boundary_flags_preserved': 2,
        'OOV_parent_exclusion_preserved': 'C293_A012_019',
        'candidate_rank_is_recognition_accuracy': False,
        'original_stage6_expansion_gate_met': False,
        'future_training_executed': False,
        'sources': {key: binding(path) for key, path in paths.items()},
    }
    save(output / 'stage5b_summary.json', summary)
    report = ROOT / 'docs/녹화형_단계학습_2026-10-12/05b_학습정체_진단과_구절학습_후속설계.md'
    save(output / 'next_step_handoff.json', {
        'schema': 'offline-stage5b-next-step-handoff-v1', 'client_date': '2026-10-18',
        'status': 'ready_for_separate_phrase_only_execution_preparation',
        'summary': binding(output / 'stage5b_summary.json'), 'report': binding(report),
        'proposed_design': binding(paths['design']),
        'next_work': 'Stage5c: prepare a separate train-phrase-only execution runner, policy and state-recovery checks, then execute the fixed diagnostic when continued.',
        'planned_main_updates': 3360, 'planned_train_evaluations': 144,
        'planned_separate_preflight_max_updates': 8,
        'fixed_per_seed_updates': 1680,
        'fit_gate': design['selection_policy'],
        'pre_execution_requirements': design['pre_execution_requirements'],
        'do_not_reuse_frozen_main_runner_without_new_scope_and_evaluation_contract': True,
        'source_training_ready_remains_false': True,
        'immediate_additional_download_needed': False, 'immediate_colab_needed': False,
        'design_training_executed': False,
        'no_automatic_stage6_expansion': True,
        'conditional_sentence_comparison_executed': False,
        'new_version_required': True,
    })
    excluded = {'completion_manifest.json'}
    artifacts = sorted(path for path in HERE.rglob('*') if path.is_file()
                       and path.name not in excluded and path.suffix != '.pyc'
                       and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts)
    artifacts.append(report)
    entries = [binding(path) for path in artifacts]
    save(output / 'completion_manifest.json', {
        'schema': 'offline-stage5b-completion-snapshot-v1', 'client_date': '2026-10-18',
        'status': summary['status'], 'artifacts': entries, 'artifact_count': len(entries),
        'bytes': sum(item['bytes'] for item in entries), 'new_training': 0, 'new_neural_inference': 0,
        'policy': 'Preserve completed analysis/design artifacts. Any execution or revision uses a separate experiment/version; root AGENTS and runtime caches excluded.',
    })
    for entry in entries:
        check(entry)
    for destination in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        if not re.match(r'^[a-z]+://', destination):
            assert (report.parent / destination).resolve().is_file(), destination
    print(json.dumps({'status': 'passed', 'artifacts': len(entries),
                      'manifest': binding(output / 'completion_manifest.json'),
                      'summary': binding(output / 'stage5b_summary.json')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
