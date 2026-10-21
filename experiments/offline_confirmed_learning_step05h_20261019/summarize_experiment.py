"""Summarize the fixed four completed runs without selecting a winning checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prepare_experiment import BASE, ROOT, read, ref, save


def summarize() -> None:
    """Report fitting, unseen-phrase evaluation, controls and actual resources."""
    try:
        plan = read(BASE / 'plan.json')
        data = read(BASE / 'data_plan.json')
        pre = read(BASE / 'results/preflight/result.json')
        assert pre['status'] == 'passed' and pre['physical_updates'] == 5
        runs = []
        all_rows = []
        for seed in [42, 43]:
            initial = None
            for arm in ['A12', 'B14']:
                folder = BASE / f'results/{arm}_seed{seed}'
                result = read(folder / 'result.json')
                assert result['status'] == 'completed' and result['physical_updates'] == result['total_history'] == 1680
                assert result['seconds'] <= plan['limits']['wall_seconds_per_run']
                if initial is None:
                    initial = result['initial_state_hashes']
                else:
                    assert initial == result['initial_state_hashes']
                history = [json.loads(line) for line in (folder / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
                assert len(history) == 1680 and all(r['optimizer_updated'] for r in history)
                final = read(folder / 'eval_1680/summary.json')
                middle = read(folder / 'eval_0840/summary.json')
                assert final['count'] == 51 and final['failures'] == middle['failures'] == 0
                assert final['state_isolation_passed'] and middle['state_isolation_passed']
                expected = plan['planned_accounting'][f'{arm}_seed{seed}']
                assert result['true_frames'] == expected['true_frames'] and result['dense_frames'] == expected['dense_frames']
                all_rows.extend(middle['rows'] + final['rows'])
                actual_train_exact = final['cohorts']['train_common12/normal']['exact_count']
                if arm == 'B14':
                    actual_train_exact += final['cohorts']['new2/normal']['exact_count']
                runs.append({'arm': arm, 'seed': seed, 'result': ref(folder / 'result.json'),
                    'final': final['cohorts'], 'midpoint': middle['cohorts'],
                    'train_exact': actual_train_exact, 'train_count': len(data['arms'][arm]),
                    'initial_state_hashes': initial, 'seconds': result['seconds'],
                    'true_frames': result['true_frames'], 'dense_frames': result['dense_frames'],
                    'cuda_peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in history),
                    'cuda_peak_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in history),
                    'final_dev_predictions': [{'clip_id': r['clip_id'], 'reference': r['reference'], 'prediction': r['raw_text']}
                        for r in final['rows'] if r['cohort'] == 'dev3' and r['condition'] == 'normal']})
        assert len(all_rows) == 256
        means = {arm: sum(r['final']['dev3/normal']['CER'] for r in runs if r['arm'] == arm) / 2 for arm in ['A12', 'B14']}
        differences = {str(seed): next(r for r in runs if r['seed'] == seed and r['arm'] == 'B14')['final']['dev3/normal']['CER'] -
            next(r for r in runs if r['seed'] == seed and r['arm'] == 'A12')['final']['dev3/normal']['CER'] for seed in [42, 43]}
        criterion = all(v < 0 for v in differences.values()) and means['B14'] - means['A12'] <= -.05
        source = read(BASE / 'data/cache_handoff.json')
        validation = read(BASE / 'data/cache_validation.json')
        audit_path = BASE / 'audit/final_saved_results_check.json'
        audit = read(audit_path)
        assert audit['status'] == 'passed'
        summary = {'schema': 'offline-stage5h-completed-confirmed-learning-v1', 'client_date': '2026-10-19',
            'status': 'completed_bounded_confirmed_phrase_comparison', 'plan': ref(BASE / 'plan.json'),
            'data_plan': ref(BASE / 'data_plan.json'), 'runs': runs, 'mean_final_dev_normal_CER': means,
            'per_seed_dev_B_minus_A_CER': differences, 'mean_dev_B_minus_A_percentage_points': 100 * (means['B14'] - means['A12']),
            'predefined_exploratory_improvement_criterion_met': criterion,
            'actual_main_optimizer_updates': 6720, 'separate_preflight_updates': 5, 'total_optimizer_updates': 6725,
            'main_checkpoint_count': 8, 'preflight_checkpoint_count': 1,
            'evaluation_rows': len(all_rows), 'evaluation_frames': sum(r['T'] for r in all_rows), 'evaluation_failures': 0,
            'main_true_frames': sum(r['true_frames'] for r in runs), 'main_dense_frames': sum(r['dense_frames'] for r in runs),
            'main_seconds_sum': sum(r['seconds'] for r in runs),
            'cuda_peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in runs),
            'cuda_peak_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in runs),
            'new_source_cache_handoff': ref(BASE / 'data/cache_handoff.json'),
            'cache_validation': ref(BASE / 'data/cache_validation.json'), 'saved_output_audit': ref(audit_path),
            'held_review_ids': ['Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08'],
            'new_review_questions': 0, 'audio_model_inputs': 0, 'teacher_calls': 0,
            'new_downloads': 0, 'service_changes': 0, 'source_training_ready': False,
            'stage5g_all8_review_completed': False, 'stage6_automatically_authorized': False,
            'temporal_control_observation': 'Both B14 seeds recognize both added phrases under normal, static and shuffled input. This supports fitting but does not establish useful temporal-motion recognition; appearance, duration and other cues remain possible.',
            'limitations': plan['caveats'] + ['Only three reused internal development phrases; no statistically powered or independent final generalization claim.',
                'The new2 A/B comparison changes whether they were trained; use it as fit diagnostics only.',
                'Source qualitative approvals remain unchanged; controlled-experiment permission does not certify exact visual articulation boundaries.']}
        save(BASE / 'results/stage5h_summary.json', summary)
        lines = ['# 확인된 구절로 수행한 제한 비교 학습', '', '2026-10-19 · 녹화형 5h', '',
            '사용자의 검수 축소·보류 후 학습 요청에 따라 Q03~Q08을 보류하고, 기존 학습 12구절에 음성·정성적 입술 검수가 확인된 Q01·Q02만 추가했다. 추가 청취·시각 질문은 하지 않았다.', '',
            '기존 12구절(A12)과 확대 14구절(B14)을 두 초기값에서 각각 새로 1,680회 학습했다. 구조는 compact 시각 모델과 한글 자모 69토큰 CTC이며, 음성·교사 모델은 입력하지 않았다. 학습률은 840회까지 0.001, 1,050회까지 0.0003, 이후 0.0001이다.', '',
            '| 초기값 | 자료 | 학습 구절 완전 일치 | 내부 미학습 3구절 CER | 내부 미학습 완전 일치 |',
            '|---|---|---:|---:|---:|']
        for r in runs:
            score = r['final']['dev3/normal']
            lines.append(f"| {r['seed']} | {r['arm']} | {r['train_exact']}/{r['train_count']} | {100*score['CER']:.2f}% | {score['exact_count']}/3 |")
        lines += ['', f"미학습 구절 평균 CER: A {100*means['A12']:.2f}% → B {100*means['B14']:.2f}%, 차이(B−A) {summary['mean_dev_B_minus_A_percentage_points']:+.2f}%p. CER는 오류율이며 1−CER를 인식 정확도로 환산하지 않는다.",
            f"두 초기값 모두 개선되고 평균 5%p 이상 낮아지는 사전 탐색 기준 충족: {'예' if criterion else '아니오'}. 이 기준은 실용화나 다음 문장 단계의 자동 진입 기준이 아니다.", '',
            '## 추가한 두 구절의 학습 여부', '',
            'Q01은 “너무 속상해”(38프레임), Q02는 “정규직으로 전환이 돼”(31프레임)다. A는 이 둘을 학습하지 않았고 B는 학습했다. 따라서 아래 차이는 추가 구절을 익혔는지 보는 결과이며 새 문장 일반화 개선으로 합산하지 않는다.', '',
            '| 초기값 | A 추가2 CER / 일치 | B 추가2 CER / 일치 |', '|---|---:|---:|']
        for seed in [42, 43]:
            a,b = [next(r for r in runs if r['seed']==seed and r['arm']==arm)['final']['new2/normal'] for arm in ['A12','B14']]
            lines.append(f"| {seed} | {100*a['CER']:.2f}% / {a['exact_count']}/2 | {100*b['CER']:.2f}% / {b['exact_count']}/2 |")
        lines += ['', '## 시간 순서 대조: 내부 미학습 3구절', '',
            '| 자료·초기값 | 정상 CER | 정지 CER | 순서 섞기 CER |', '|---|---:|---:|---:|']
        for r in runs:
            values = [100*r['final'][f'dev3/{c}']['CER'] for c in ['normal','static','shuffled']]
            lines.append(f"| {r['arm']}·{r['seed']} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
        if all(r['final']['new2/static']['exact_count'] == r['final']['new2/shuffled']['exact_count'] == 2 for r in runs if r['arm'] == 'B14'):
            lines += ['', '확대군 두 초기값 모두 새 2구절을 정상·정지·순서 섞기 조건에서 2/2 맞혔다. 추가 자료를 학습할 수 있다는 결과이지만, 입 움직임의 시간 순서를 유용하게 읽는 능력의 개선을 입증하지는 않는다. 외양·길이 등 다른 단서를 활용했을 가능성이 남아 있으며 특정 원인 하나로 확정하지 않는다.']
        lines += ['', '## 실제 출력', '', '| 자료·초기값 | 정답 | 정상 출력 |', '|---|---|---|']
        for r in runs:
            for prediction in r['final_dev_predictions']:
                lines.append(f"| {r['arm']}·{r['seed']} | {prediction['reference']} | {prediction['prediction'] or '(빈 출력)'} |")
        lines += ['', '## 해석 범위와 비용', '',
            '- 확대군은 14부모·13녹화·8명·13개 구절 표현이다. Q01은 기존 학습 녹화와 같고 Q02만 새 녹화이므로 새로운 화자가 추가된 실험은 아니다.',
            '- A는 구절당 280회, B는 240회 노출되었다. 양쪽 업데이트는 같지만 실제 프레임 수와 배치가 달라 순수한 다양성 효과로 단정하지 않는다.',
            '- 평가 3개는 반복 사용한 같은 화자의 내부 개발 구절이다. 독립 최종시험·실제 웹캠 성능은 아직 측정하지 않았다.',
            '- 기존 C002 경계 주의 표시와 원자료의 provisional/training_ready=false는 보존했다. 이는 이번 사용자 승인에 따른 제한 실험 수행과 구분된다.',
            f"- 본 학습 6,720회 + 저장·재개 확인 5회 = 실제 6,725회. 본 체크포인트 8개, 사전 확인 1개, 평가 256건·실패 0건. 본 구간 합계 {summary['main_seconds_sum']:.2f}초.",
            f"- 본 실제 프레임 {summary['main_true_frames']:,}, 패딩 포함 {summary['main_dense_frames']:,}. 최대 GPU 할당 {summary['cuda_peak_allocated_bytes']/1024**2:.2f}MiB / 예약 {summary['cuda_peak_reserved_bytes']/1024**2:.2f}MiB. 웹캠 처리 속도 측정은 아니다.",
            '- 새 캐시는 69프레임·1,218,667bytes이며 원본 영상·라벨 4멤버 735,347,388bytes만 읽었다. 임시 최대 377,753,938bytes, 잔여 0. 추가 다운로드·클라우드 이동·서비스 가중치 교체는 없다.',
            '- 학습 중단이 생기면 저장 지점 이후 물리 업데이트와 미완료 평가를 먼저 정산해야 한다. 이번 본 실행은 중단·재개 없이 완료했다.', '',
            '이번 실험의 결과를 기준으로 다음 자료 규모와 모델 비교의 우선순위를 정한다. 보류한 6구절의 반복 검수를 자동 재개하거나 새 학습을 자동 연장하지 않는다.']
        path = ROOT / 'docs/녹화형_단계학습_2026-10-12/05h_확인구절_추가학습과_평가.md'
        with path.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
        save(BASE / 'results/next_step_handoff.json', {'schema':'offline-stage5h-handoff-v1',
            'summary': ref(BASE / 'results/stage5h_summary.json'), 'report': ref(path),
            'current_user_direction': 'Keep review bounded, hold ambiguous samples, prioritize fixed-budget training and evaluation.',
            'held_review_ids': summary['held_review_ids'], 'new_review_question_pending': False,
            'current_comparison_complete': True, 'practical_visual_speech_recognition_complete': False,
            'next': 'Use the fixed comparison results to choose a bounded larger-expression or model-feature experiment; preserve the held6 and independent final-test separation. Do not automatically extend this training or claim stage6 readiness.'})
        print(json.dumps({'mean_dev_CER': means, 'B_minus_A_pp':summary['mean_dev_B_minus_A_percentage_points'],
            'criterion':criterion,'report':str(path)},ensure_ascii=False))
    except Exception as exc:
        raise RuntimeError('Summary requires all fixed runs and independent saved-output validation') from exc


if __name__ == '__main__':
    summarize()
