"""Report the six fixed runs after the independent saved-output audit passes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prepare_experiment import BASE, ROOT, read, ref, save

ARMS = ['A14', 'B26', 'C26_static']
SEEDS = [42, 43]
CONDITIONS = ['normal', 'static_mid', 'shuffled', 'reversed']
DEV_COHORTS = ['dev_phrase3', 'dev_sentence3']
REPORT = ROOT / 'docs/녹화형_단계학습_2026-10-12/05i_표현확대와_움직임대조_학습.md'


def require(ok: bool, message: str) -> None:
    """Reject incomplete or inconsistent evidence before publishing results."""
    try:
        if not ok:
            raise ValueError(message)
    except Exception as exc:
        exc.add_note('summarize_experiment.py:require failed while processing experiment evidence')
        raise


def cell(value: Any) -> str:
    """Keep saved predictions literal and safe inside a Markdown table."""
    try:
        return str(value).replace('|', '\\|').replace('\n', '<br>') or '(빈 출력)'
    except Exception as exc:
        exc.add_note('summarize_experiment.py:cell failed while processing experiment evidence')
        raise


def score(run: dict, cohort: str, condition: str = 'normal') -> dict:
    """Select one predeclared final cohort and input condition."""
    try:
        return run['final'][f'{cohort}/{condition}']
    except Exception as exc:
        exc.add_note('summarize_experiment.py:score failed while processing experiment evidence')
        raise


def train_fit(final: dict, arm: str, condition: str) -> dict:
    """Aggregate only IDs actually optimized by this arm, without mixing dev."""
    try:
        names = ['train_phrase14'] + ([] if arm == 'A14' else ['train_sentence12'])
        scores = [final[f'{name}/{condition}'] for name in names]
        edits = sum(s['character_edits'] for s in scores)
        characters = sum(s['reference_characters'] for s in scores)
        return {'count': sum(s['count'] for s in scores),
                'exact_count': sum(s['exact_count'] for s in scores),
                'character_edits': edits, 'reference_characters': characters,
                'CER': edits / characters, 'condition': condition}
    except Exception as exc:
        exc.add_note('summarize_experiment.py:train_fit failed while processing experiment evidence')
        raise


def load_runs(plan: dict, data: dict) -> tuple[list[dict], list[dict]]:
    """Read all fixed endpoints, actual logs and model-state isolation evidence."""
    try:
        runs: list[dict] = []
        all_rows: list[dict] = []
        for seed in SEEDS:
            initial = None
            for arm in ARMS:
                folder = BASE / f'results/{arm}_seed{seed}'
                result = read(folder / 'result.json')
                require(result['status'] == 'completed', f'Incomplete run {arm}/{seed}')
                require(result['physical_updates'] == result['total_history'] == 1820,
                        f'Unexpected physical updates {arm}/{seed}')
                require(result['seconds'] <= plan['limits']['wall_seconds_per_run'], 'Wall budget exceeded')
                if initial is None:
                    initial = result['initial_state_hashes']
                require(initial == result['initial_state_hashes'], f'Paired initial state differs {arm}/{seed}')
                history = [json.loads(line) for line in (folder / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
                require(len(history) == 1820 and all(r['optimizer_updated'] for r in history), 'Incomplete update log')
                require([r['update'] for r in history] == list(range(1, 1821)), 'Update order differs')
                view = plan['training_conditions'][arm]
                require(all(r['training_condition'] == view for r in history), 'Training view differs')
                final = read(folder / 'eval_1820/summary.json')
                middle = read(folder / 'eval_0910/summary.json')
                require(final['count'] == 128 and middle['count'] == len(data['arms'][arm]), 'Evaluation count differs')
                require(final['failures'] == middle['failures'] == 0, 'Failed evaluation exists')
                require(final['state_isolation_passed'] and middle['state_isolation_passed'], 'Evaluation state changed')
                expected = plan['planned_accounting'][f'{arm}_seed{seed}']
                require(result['true_frames'] == expected['true_frames'] and result['dense_frames'] == expected['dense_frames'],
                        'Actual frame accounting differs')
                require(len(result['checkpoints']) == 2, 'Wrong main checkpoint count')
                all_rows.extend(middle['rows'] + final['rows'])
                normal_fit = train_fit(final['cohorts'], arm, 'normal')
                training_view_fit = train_fit(final['cohorts'], arm, view)
                require(training_view_fit['count'] == len(data['arms'][arm]), 'Fit cohort differs')
                runs.append({'arm': arm, 'seed': seed, 'result': ref(folder / 'result.json'),
                    'final_summary': ref(folder / 'eval_1820/summary.json'), 'final': final['cohorts'],
                    'midpoint': middle['cohorts'], 'training_condition': view,
                    'training_view_fit': training_view_fit, 'normal_view_on_optimizer_ids': normal_fit,
                    'initial_state_hashes': result['initial_state_hashes'], 'seconds': result['seconds'],
                    'true_frames': result['true_frames'], 'dense_frames': result['dense_frames'],
                    'cuda_peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in history),
                    'cuda_peak_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in history),
                    'final_dev_predictions': [{'clip_id': r['clip_id'], 'cohort': r['cohort'],
                        'reference': r['reference'], 'prediction': r['raw_text']}
                        for r in final['rows'] if r['cohort'] in DEV_COHORTS and r['condition'] == 'normal']})
        require(len(all_rows) == plan['evaluation']['planned_rows'] == 900, 'Wrong total evaluations')
        return runs, all_rows
    except Exception as exc:
        exc.add_note('summarize_experiment.py:load_runs failed while processing experiment evidence')
        raise


def comparisons(runs: list[dict]) -> dict:
    """Compute descriptive fixed-seed contrasts; never select a best result."""
    try:
        by_run = {(r['arm'], r['seed']): r for r in runs}
        out: dict = {}
        for cohort in DEV_COHORTS:
            means = {arm: sum(score(by_run[arm, seed], cohort)['CER'] for seed in SEEDS) / 2 for arm in ARMS}
            effects = []
            for seed in SEEDS:
                a, b, c = [by_run[arm, seed] for arm in ARMS]
                effects.append({'seed': seed,
                    'B_minus_A_normal_CER_pp': 100 * (score(b, cohort)['CER'] - score(a, cohort)['CER']),
                    'C_minus_B_normal_CER_pp': 100 * (score(c, cohort)['CER'] - score(b, cohort)['CER']),
                    'B_ablated_minus_normal_CER_pp': {condition: 100 * (score(b, cohort, condition)['CER'] - score(b, cohort)['CER'])
                        for condition in CONDITIONS if condition != 'normal'}})
            out[cohort] = {'mean_normal_CER': means, 'per_seed_effects': effects,
                'mean_B_minus_A_normal_CER_pp': 100 * (means['B26'] - means['A14']),
                'mean_C_minus_B_normal_CER_pp': 100 * (means['C26_static'] - means['B26']),
                'mean_B_ablated_minus_normal_CER_pp': {condition: sum(e['B_ablated_minus_normal_CER_pp'][condition] for e in effects) / 2
                    for condition in CONDITIONS if condition != 'normal'}}
        return out
    except Exception as exc:
        exc.add_note('summarize_experiment.py:comparisons failed while processing experiment evidence')
        raise


def observed_interpretation(summary: dict) -> list[str]:
    """Describe the measured contrasts without attributing a unique cause."""
    try:
        contrast = summary['comparisons']['dev_phrase3']
        means = contrast['mean_normal_CER']
        delta = contrast['mean_B_minus_A_normal_CER_pp']
        direction = '낮아졌다' if delta < 0 else '높아졌다' if delta > 0 else '같았다'
        expanded = [r for r in summary['runs'] if r['arm'] == 'B26']
        temporal = {condition: sum(score(r, 'dev_phrase3', condition)['CER'] for r in expanded)/len(expanded)
                    for condition in CONDITIONS}
        normal_order_consistent = all(value > 0 for item in contrast['per_seed_effects']
            for value in item['B_ablated_minus_normal_CER_pp'].values())
        order_statement = ('두 초기값의 모든 시간 변형 대조에서 정상 입력의 오류가 더 낮았지만, 이 소규모 결과만으로 실제 조음 이해를 인증할 수는 없다.'
            if normal_order_consistent else '새 구절에서 정상적인 입술 움직임·시간 순서가 일관되게 유리하다는 결과는 확인되지 않았다.')
        full_fits = ', '.join(f"초기값 {r['seed']}: {score(r, 'train_sentence12')['exact_count']}/12"
                             for r in expanded)
        full_complete = all(score(r, 'train_sentence12')['exact_count'] == 12 for r in expanded)
        full_statement = ('확대군의 전체 문장 학습은 두 초기값에서 완전 일치했다.' if full_complete else
                          '확대군은 학습한 전체 문장도 아직 모두 맞히지 못했다.')
        motion_by_seed = ', '.join(f"{item['seed']}: {item['C_minus_B_normal_CER_pp']:+.2f}%p"
                                  for item in contrast['per_seed_effects'])
        sentence_motion_delta = summary['comparisons']['dev_sentence3']['mean_C_minus_B_normal_CER_pp']
        return [
            f"이번 고정 비교에서 내부 미학습 구절의 평균 CER는 A {100*means['A14']:.2f}% → B {100*means['B26']:.2f}%로 {abs(delta):.2f}%p {direction}. 이는 자료·문맥 확대군의 기술적인 관측 결과이며 길이·노출 횟수 등의 차이도 포함한다.",
            f"B의 평균 CER는 정상 {100*temporal['normal']:.2f}%, 정지 {100*temporal['static_mid']:.2f}%, 섞기 {100*temporal['shuffled']:.2f}%, 역순 {100*temporal['reversed']:.2f}%다. {order_statement} 정지 영상으로 학습한 C의 정상 입력 평균 CER는 {100*means['C26_static']:.2f}%이며 C−B는 {contrast['mean_C_minus_B_normal_CER_pp']:+.2f}%p다. 이 차이의 초기값별 값은 {motion_by_seed}이고, 대응 전체 문장의 평균 C−B는 {sentence_motion_delta:+.2f}%p이므로 함께 해석해야 한다.",
            f"{full_statement} 전체 문장 완전 일치는 {full_fits}다. 평가 구절은 반복 사용한 3개뿐이므로, 학습 적합도·표현 범위·움직임 활용 중 하나만을 유일한 실패 원인으로 단정하지 않는다.",
        ]
    except Exception as exc:
        exc.add_note('summarize_experiment.py:observed_interpretation failed while interpreting fixed contrasts')
        raise


def report_lines(summary: dict) -> list[str]:
    """Render measured outcomes, actual predictions and interpretation limits."""
    try:
        runs = summary['runs']
        lines = ['# 표현 확대와 입술 움직임 대조 학습', '', '2026-10-19 · 녹화형 5i', '',
            '애매했던 Q03~Q08은 계속 보류하고, 기존에 전사·음성 경계를 확인한 부모 문장 12개를 활용했다. 추가 청취·시각 검수나 원본 다운로드는 하지 않았다. 이번 작업은 확인 자료 안에서 표현을 넓히는 제한 비교 실험이다.', '',
            summary['observed_interpretation'][0], '', summary['observed_interpretation'][1], '', summary['observed_interpretation'][2], '',
            '| 비교군 | 학습 자료 | 학습 시 입력 |', '|---|---|---|',
            '| A14 | 기존 구절 14개 | 정상 입술 영상 |',
            '| B26 | 같은 구절 14개 + 대응 전체 문장 12개 | 정상 입술 영상 |',
            '| C26_static | B와 동일한 26개·동일 순서·동일 길이 | 가운데 한 프레임을 원래 길이만큼 반복 |', '',
            '두 초기값(42·43)에서 각 군을 새로 1,820회 학습했다. B/C는 구절과 전체 문장을 처음부터 섞어 학습했다. compact 시각 모델·한글 자모 69토큰 CTC를 사용하며 음성·교사 모델은 입력하지 않았다. B/C는 같은 초기 상태·배치 순서·프레임 수를 사용한다. A/B는 자료 길이와 노출 횟수가 달라 표현 수 하나의 순수한 효과로 해석하지 않는다.', '',
            'A14는 이번 1,820회 예산·학습률·새 배치 순서로 다시 학습한 대조군이다. 이전 5h의 1,680회 결과(평균 CER 77.42%)와 수치를 직접 빼서 개선·악화를 판단하지 않는다.', '',
            '전체 학습 표현은 13→24종, 어절은 35→76종, 음절은 64→121종으로 늘었다. 부모 14개·녹화 13개·화자 8명은 그대로이며, 같은 부모의 구절과 문장은 겹친다.', '',
            '## 학습한 자료의 적합도', '',
            '| 초기값 | 군 | 학습 입력 조건에서 완전 일치 | 같은 학습 ID에 정상 영상을 넣었을 때 |',
            '|---|---|---:|---:|']
        for r in runs:
            a, b = r['training_view_fit'], r['normal_view_on_optimizer_ids']
            lines.append(f"| {r['seed']} | {r['arm']} | {a['exact_count']}/{a['count']} · CER {100*a['CER']:.2f}% | {b['exact_count']}/{b['count']} · CER {100*b['CER']:.2f}% |")
        lines += ['', 'C의 왼쪽은 정지 입력을 학습한 결과이고 오른쪽은 정상 영상으로 바꿨을 때의 입력 변화 진단이다. 두 결과를 같은 학습 적합도로 혼동하지 않는다. A의 추가 전체 문장 12개는 같은 부모에서 나온 미학습 길이·문맥 진단으로, 독립 미학습 문장 평가에 합치지 않는다.', '',
            '## 내부 미학습 구절 3개', '',
            '| 초기값 | 군 | 정상 CER | 완전 일치 |', '|---|---|---:|---:|']
        for r in runs:
            s = score(r, 'dev_phrase3')
            lines.append(f"| {r['seed']} | {r['arm']} | {100*s['CER']:.2f}% | {s['exact_count']}/3 |")
        for cohort, label in [('dev_phrase3', '구절'), ('dev_sentence3', '대응 전체 문장')]:
            c = summary['comparisons'][cohort]; m = c['mean_normal_CER']
            if cohort == 'dev_sentence3':
                lines += ['', '## 같은 내부 평가 부모의 전체 문장 3개', '',
                    '| 초기값 | 군 | 정상 CER | 완전 일치 |', '|---|---|---:|---:|']
                for r in runs:
                    s = score(r, cohort)
                    lines.append(f"| {r['seed']} | {r['arm']} | {100*s['CER']:.2f}% | {s['exact_count']}/3 |")
            lines += ['', f"{label} 평균 CER: A {100*m['A14']:.2f}%, B {100*m['B26']:.2f}%, C {100*m['C26_static']:.2f}%. 표현·문맥 확대 비교 B−A는 {c['mean_B_minus_A_normal_CER_pp']:+.2f}%p, 정상 영상 학습과 정지 영상 학습 비교 C−B는 {c['mean_C_minus_B_normal_CER_pp']:+.2f}%p다."]
        lines += ['', 'CER는 문자 오류율로 낮을수록 좋으며 1−CER를 인식 정확도로 바꾸지 않는다. 구절 3개와 전체 문장 3개는 같은 부모를 공유하므로 독립 평가 6개가 아니다. 모두 기존 화자의 반복 사용한 내부 개발 자료이며 최종 시험·웹캠 성능을 나타내지 않는다.', '',
            '## 시간 정보 대조', '',
            '모든 최종 모델에 정상·중앙 프레임 정지·순서 섞기·역순 영상을 입력했다. 정지 입력은 외양·길이 단서를 남기고 움직임을 없애며, 섞기와 역순은 원래의 시간 순서를 바꾼다.', '',
            '| 평가 | 초기값 | 군 | 정상 CER | 정지 CER | 섞기 CER | 역순 CER |',
            '|---|---|---|---:|---:|---:|---:|']
        for cohort, label in [('dev_phrase3', '구절3'), ('dev_sentence3', '문장3')]:
            for r in runs:
                values = ' | '.join(f"{100*score(r, cohort, condition)['CER']:.2f}%" for condition in CONDITIONS)
                lines.append(f"| {label} | {r['seed']} | {r['arm']} | {values} |")
        lines += ['', '| 평가 | 초기값 | B−A 정상 CER | C−B 정상 CER | B 정지−정상 | B 섞기−정상 | B 역순−정상 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for cohort, label in [('dev_phrase3', '구절3'), ('dev_sentence3', '문장3')]:
            for e in summary['comparisons'][cohort]['per_seed_effects']:
                gaps = e['B_ablated_minus_normal_CER_pp']
                lines.append(f"| {label} | {e['seed']} | {e['B_minus_A_normal_CER_pp']:+.2f}%p | {e['C_minus_B_normal_CER_pp']:+.2f}%p | {gaps['static_mid']:+.2f}%p | {gaps['shuffled']:+.2f}%p | {gaps['reversed']:+.2f}%p |")
        lines += ['', 'B−A가 음수이면 확대군 오류가 낮다. C−B와 B의 변형−정상이 양수이면 정상 영상 학습·입력 쪽 오류가 낮다. 다만 입력 변형에 민감하다는 사실만으로 유용한 조음 정보를 읽었다고 인증할 수 없다. 같은 소규모 평가에서 두 초기값·인식 오류·정지 학습 적합도를 함께 판단한다.', '',
            '## 늘어난 표현이 평가 어휘를 얼마나 포함하는가', '',
            '고정된 자료·학습 계획을 변경하지 않고 학습 정답에서 평가 어휘가 관측되었는지 집계했다. 각 칸은 “출현 횟수 기준 / 고유 종류 기준”이며 학습 성공률은 아니다. C는 B와 같은 정답을 사용한다.', '',
            '| 평가 | 학습 자료 | 어절 관측 | 음절 관측 | 자모·공백 토큰 관측 |',
            '|---|---|---|---|---|']
        for item in summary['vocabulary_coverage']['rows']:
            fields = []
            for dimension in ['eojeol', 'hangul_syllable', 'jamo_space_token_id']:
                coverage = item['coverage'][dimension]
                fields.append(f"{coverage['covered_occurrences']}/{coverage['occurrences']} / {coverage['covered_unique_types']}/{coverage['unique_types']}")
            label = '구절3' if item['cohort'] == 'dev_phrase3' else '문장3'
            lines.append(f"| {label} | {item['arm']} | {' | '.join(fields)} |")
        missing = next(item for item in summary['vocabulary_coverage']['rows'] if item['cohort'] == 'dev_phrase3' and item['arm'] == 'B26')['coverage']['eojeol']['uncovered_types']
        lines += ['', '구절 평가의 미관측 어절은 확대 후에도 ' + '·'.join(f'“{word}”' for word in missing) + '이다. 전체 어휘 수의 증가가 평가 표현의 충분한 관측으로 곧바로 이어지지는 않았다. 자모를 관측했어도 새 음절·어절 조합을 영상으로 인식할 수 있다는 뜻은 아니며, 이 집계만으로 오류의 원인을 확정하지 않는다.', '',
            '## 실제 정상 입력 출력', '', '| 평가 | 군·초기값 | 정답 | 출력 |', '|---|---|---|---|']
        for r in runs:
            for p in r['final_dev_predictions']:
                label = '구절' if p['cohort'] == 'dev_phrase3' else '문장'
                lines.append(f"| {label} | {r['arm']}·{r['seed']} | {cell(p['reference'])} | {cell(p['prediction'])} |")
        lines += ['', '## 실행 검증과 한계', '',
            f"- 본 학습 {summary['actual_main_optimizer_updates']:,}회 + 별도 저장·재개 검증 {summary['separate_preflight_updates']}회 = 실제 {summary['total_optimizer_updates']:,}회. 본 체크포인트 12개·사전 1개, 평가 {summary['evaluation_rows']}건·실패 0건.",
            f"- 본 구간 합계 {summary['main_seconds_sum']:.2f}초. 실제 프레임 {summary['main_true_frames']:,}, 패딩 포함 {summary['main_dense_frames']:,}. 최대 GPU 할당 {summary['cuda_peak_allocated_bytes']/1024**2:.2f}MiB / 예약 {summary['cuda_peak_reserved_bytes']/1024**2:.2f}MiB. 웹캠 처리 속도 측정은 아니다.",
            '- A는 클립당 260회, B/C는 140회 노출했다. A/B는 업데이트 수가 같아도 자료 길이·문맥·배치·프레임 양이 다르다. B/C는 학습 입력의 시간 정보만 달리한 같은 일정이다.',
            '- 저장 출력에서 입력 변형·복원 문자열·CER·일정·원본 해시·보류 대상을 독립 검산했다. 모델의 학습 상태는 평가 전후 보존되었다.',
            '- 문장별 실제 전사·음성 경계 확인과 원본 캐시 기술 검증을 이용했다. 구절 시각 답변을 전체 문장의 시각 승인으로 전이하지 않았다. 정밀 조음 경계 미인증·C002 경계 주의·provisional/training_ready=false는 보존했다.',
            '- Q03~Q08 제외, 원본/TAR 읽기·새 캐시 추출·추가 검수·다운로드·음성 모델 입력·교사·서비스 가중치 교체 모두 0이다.',
            '- 결과가 좋은 초기값이나 체크포인트를 골라 대표값으로 보고하지 않았다. 정해진 두 초기값과 마지막 시점을 모두 기록했으며 자동 학습 연장은 하지 않는다.', '',
            '후속 작업은 이번 결과에 따라 표현 확대와 시각 특징·학습 구성 중 우선순위를 정한다. 보류 6구절을 재질문하거나 6단계·실용화 성공으로 자동 전환하지 않는다.']
        return lines
    except Exception as exc:
        exc.add_note('summarize_experiment.py:report_lines failed while processing experiment evidence')
        raise


def summarize() -> None:
    """Publish only after all planned runs and the independent audit complete."""
    try:
        plan, data = read(BASE / 'plan.json'), read(BASE / 'data_plan.json')
        pre = read(BASE / 'results/preflight/result.json')
        audit_path = BASE / 'audit/final_saved_results_check.json'
        require(read(audit_path)['status'] == 'passed', 'Independent saved-output audit has not passed')
        require(pre['status'] == 'passed' and pre['physical_updates'] == 5, 'Preflight incomplete')
        require(plan['steps'] == 1820 and plan['seeds'] == SEEDS, 'Plan differs')
        runs, rows = load_runs(plan, data)
        inventory = read(BASE / 'inventory/verified_expansion.json')
        require(inventory['status'] == 'passed', 'Source inventory incomplete')
        coverage_path = BASE / 'inventory/posthoc_coverage.json'
        vocabulary = read(coverage_path)
        require(vocabulary['status'] == 'completed_descriptive_only' and vocabulary['source_inventory'] == ref(BASE / 'inventory/verified_expansion.json'), 'Coverage provenance differs')
        summary = {'schema': 'offline-stage5i-completed-diversity-motion-v1', 'client_date': '2026-10-19',
            'status': 'completed_bounded_diversity_and_motion_comparison', 'plan': ref(BASE / 'plan.json'),
            'data_plan': ref(BASE / 'data_plan.json'), 'source_inventory': ref(BASE / 'inventory/verified_expansion.json'),
            'runs': runs, 'comparisons': comparisons(runs), 'source_statistics': inventory['statistics'],
            'vocabulary_coverage_source': ref(coverage_path), 'vocabulary_coverage': vocabulary,
            'actual_main_optimizer_updates': 10920, 'separate_preflight_updates': 5, 'total_optimizer_updates': 10925,
            'main_checkpoint_count': 12, 'preflight_checkpoint_count': 1,
            'evaluation_rows': len(rows), 'evaluation_frames': sum(r['T'] for r in rows), 'evaluation_failures': 0,
            'main_true_frames': sum(r['true_frames'] for r in runs), 'main_dense_frames': sum(r['dense_frames'] for r in runs),
            'main_seconds_sum': sum(r['seconds'] for r in runs),
            'cuda_peak_allocated_bytes': max(r['cuda_peak_allocated_bytes'] for r in runs),
            'cuda_peak_reserved_bytes': max(r['cuda_peak_reserved_bytes'] for r in runs),
            'saved_output_audit': ref(audit_path), 'held_review_ids': ['Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08'],
            'new_raw_media_reads': 0, 'new_cache_extractions': 0, 'new_review_questions': 0,
            'audio_model_inputs': 0, 'teacher_calls': 0, 'new_downloads': 0, 'service_changes': 0,
            'source_training_ready': False, 'source_approvals_changed': False, 'stage6_automatically_authorized': False,
            'no_best_seed_or_checkpoint_selection': True, 'descriptive_only_no_new_success_threshold': True,
            'limitations': plan['caveats'] + inventory['caveats'] + [
                'Fresh stage5i A14 uses this stage1820 budget, LR and shuffled schedule. Do not subtract stage5h B14 mean77.42% to infer causal improvement/regression.']}
        summary['observed_interpretation'] = observed_interpretation(summary)
        summary_path = BASE / 'results/stage5i_summary.json'
        handoff_path = BASE / 'results/next_step_handoff.json'
        require(not summary_path.exists() and not handoff_path.exists() and not REPORT.exists(), 'Reporting output already exists; preserve it')
        save(summary_path, summary)
        with REPORT.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(report_lines(summary)) + '\n')
        save(handoff_path, {'schema': 'offline-stage5i-handoff-v1', 'summary': ref(summary_path), 'report': ref(REPORT),
            'current_comparison_complete': True, 'practical_visual_speech_recognition_complete': False,
            'held_review_ids': summary['held_review_ids'], 'new_review_question_pending': False,
            'next': 'Choose a separate bounded experiment from the fixed context-expansion and static-training contrasts. Keep uncertain6 held and preserve independent final-test separation; do not extend these six runs or promote to stage6 automatically.'})
        print(json.dumps({'comparisons': summary['comparisons'], 'report': str(REPORT)}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as exc:
        raise RuntimeError('Summary requires all six fixed runs and passing independent validation; preserve any existing output') from exc


if __name__ == '__main__':
    summarize()
