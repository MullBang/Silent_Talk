"""Freeze completed bounded experiment artifacts and append the current handoff."""
from __future__ import annotations

from pathlib import Path

from prepare_experiment import BASE, ROOT, read, ref, save


def finalize() -> None:
    """Preserve all prior status text and publish only the measured current status."""
    try:
        summary = read(BASE / 'results/stage5h_summary.json')
        handoff = read(BASE / 'results/next_step_handoff.json')
        assert summary['status'] == 'completed_bounded_confirmed_phrase_comparison'
        assert summary['actual_main_optimizer_updates'] == 6720
        assert read(BASE / 'audit/final_saved_results_check.json')['status'] == 'passed'
        plan = read(BASE / 'plan.json')
        for binding in plan['bindings']:
            assert ref(Path(binding['path']))['sha256'] == binding['sha256']
        files = [p for p in BASE.rglob('*') if p.is_file() and not any(part in {'__pycache__', '.pytest_cache', 'runtime'} for part in p.parts)]
        assert not any(p.suffix == '.partial' for p in files)
        doc = Path(handoff['report']['path'])
        files.append(doc)
        artifacts = [ref(p) for p in sorted(set(files))]
        save(BASE / 'results/completion_manifest.json', {
            'schema': 'offline-stage5h-completion-manifest-v1', 'client_date': '2026-10-19',
            'status': 'completed_bounded_training_and_evaluation', 'artifacts': artifacts,
            'held_review_ids': summary['held_review_ids'], 'old_source_flags_preserved': True,
            'source_training_ready': False, 'service_changes': 0,
            'main_updates': 6720, 'preflight_updates': 5, 'evaluation_rows': 256,
            'no_automatic_further_training': True})
        by_run = {(r['arm'], r['seed']): r for r in summary['runs']}
        values = []
        fits = []
        for seed in [42, 43]:
            a, b = [by_run[(arm, seed)] for arm in ['A12', 'B14']]
            values.append(f"seed{seed} {100*a['final']['dev3/normal']['CER']:.2f}→{100*b['final']['dev3/normal']['CER']:.2f}%")
            fits.append(f"{seed} A{a['train_exact']}/12·B{b['train_exact']}/14")
        mean = summary['mean_final_dev_normal_CER']
        criterion = '충족' if summary['predefined_exploratory_improvement_criterion_met'] else '미충족'
        paragraph = (
            '> **2026-10-19 녹화형 5h 완료: 검수 범위를 줄이고 확인 구절 추가 학습·평가 실행.** '
            '사용자 명시 지시에 따라 Q03~Q08은 보류하고 Q01/Q02만 기존train12에 추가했다. '
            '`experiments/offline_confirmed_learning_step05h_20261019/`에서 원본D격자 신규캐시69프레임·1,218,667bytes를 준비하고 '
            '새 A12/B14×seed42/43 각1680회=본6720회·별도저장재개5회, 본체크포인트8·사전1, 평가256건을 완료했다. '
            f"최종학습 정확 {'; '.join(fits)}. 내부미학습dev3 CER {'; '.join(values)}, "
            f"평균{100*mean['A12']:.2f}→{100*mean['B14']:.2f}%·두seed개선+평균5pp 기준{criterion}. "
            '개별dev완전일치와정지/섞기는요약참조. A의추가2는미학습진단·B는학습적합도로일반화에합산금지. '
            'dev3은재사용내부평가/기존화자이며최종시험아님. A280/B240노출·프레임량차이, 추가2중Q01은기존녹화여서 '
            '확대14부모/13녹화/8명/13표현·균형화자표현실험아님. 8개guard검사·독립저장posterior/입력변형/점수/일정/초기상태 검산 통과. '
            '새원본735,347,388bytes·임시최대377,753,938bytes·잔여0, 새다운로드/추가검수질문/음성모델입력/교사/서비스교체0. '
            '원검수근거/기존경계주의/provisional·training_readyfalse 유지하며 이번제한실험사용과구분한다. '
            '최신 `results/stage5h_summary.json`/`next_step_handoff.json`, 전체는 `results/completion_manifest.json`으로동결한다. '
            '다음은 결과에 따른 자료규모/시각특징·모델 비교 우선순위 설정이며 보류6반복검수/학습연장/6단계 자동진행금지. '
            '[보고서](docs/녹화형_단계학습_2026-10-12/05h_확인구절_추가학습과_평가.md). 아래는 이전 시점이다.\n\n'
        )
        agents = ROOT / 'AGENTS.md'
        original = agents.read_bytes()
        marker = original.find(b'> **')
        assert marker >= 0 and '2026-10-19 녹화형 5h 완료'.encode() not in original
        agents.write_bytes(original[:marker] + paragraph.encode('utf-8') + original[marker:])
        print(f'Frozen {len(artifacts)} current artifacts; AGENTS current handoff appended, old paragraphs preserved.')
    except Exception as exc:
        raise RuntimeError('Completion freeze failed; preserve existing output before diagnosing') from exc


if __name__ == '__main__':
    finalize()
