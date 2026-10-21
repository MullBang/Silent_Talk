"""Freeze the completed 5i comparison and prepend measured project status."""
from __future__ import annotations

from pathlib import Path

from prepare_experiment import BASE, ROOT, read, ref, save


def require(ok: bool, message: str) -> None:
    """Stop publication if source bindings or completion evidence disagree."""
    try:
        if not ok:
            raise ValueError(message)
    except Exception as exc:
        exc.add_note('finalize_experiment.py:require failed while processing experiment evidence')
        raise


def finalize() -> None:
    """Freeze every current artifact and preserve previous AGENTS bytes verbatim."""
    try:
        summary = read(BASE / 'results/stage5i_summary.json')
        handoff = read(BASE / 'results/next_step_handoff.json')
        require(summary['status'] == 'completed_bounded_diversity_and_motion_comparison', 'Summary incomplete')
        require(summary['actual_main_optimizer_updates'] == 10920 and summary['evaluation_rows'] == 900, 'Budget differs')
        require(read(BASE / 'audit/final_saved_results_check.json')['status'] == 'passed', 'Independent audit incomplete')
        require(ref(handoff['summary']['path']) == handoff['summary'], 'Handoff summary changed')
        require(ref(handoff['report']['path']) == handoff['report'], 'Handoff report changed')
        for binding in read(BASE / 'plan.json')['bindings']:
            require(ref(binding['path']) == binding, f"Bound source changed: {binding['path']}")
        agents = ROOT / 'AGENTS.md'
        original = agents.read_bytes()
        marker = original.find(b'> **')
        title = '2026-10-19 녹화형 5i 완료'
        require(marker >= 0 and title.encode('utf-8') not in original, 'AGENTS status already exists or marker missing')
        manifest = BASE / 'results/completion_manifest.json'
        require(not manifest.exists(), 'Completion manifest already exists; preserve it')
        files = [p for p in BASE.rglob('*') if p.is_file() and not any(part in {'__pycache__', '.pytest_cache', 'runtime'} for part in p.parts)]
        require(not any(p.suffix == '.partial' for p in files), 'Partial artifacts remain')
        files.append(Path(handoff['report']['path']))
        artifacts = [ref(p) for p in sorted(set(files))]
        means = summary['comparisons']
        phr = means['dev_phrase3']; sent = means['dev_sentence3']
        p = phr['mean_normal_CER']; s = sent['mean_normal_CER']
        paragraph = (
            f'> **{title}: 확인 자료의 표현 확대와 정지 학습 대조.** '
            'Q03~Q08 보류를 유지하고 기존 구절14에 별도 부모 전사·음성 경계가 확인된 문장12를 추가했다. '
            '`experiments/offline_diversity_motion_step05i_20261019/`에서 A14 정상/B26 정상/C26_static 중앙프레임 반복 × seed42/43 각1820회, '
            '본10920+저장재개검증5=실제10925업데이트·본체크포인트12/사전1·고정평가900건을 완료했다. '
            f"재사용 내부dev구절3 평균 CER A/B/C={100*p['A14']:.2f}/{100*p['B26']:.2f}/{100*p['C26_static']:.2f}%, "
            f"B−A {phr['mean_B_minus_A_normal_CER_pp']:+.2f}pp·C−B {phr['mean_C_minus_B_normal_CER_pp']:+.2f}pp. "
            f"동일부모 dev문장3 평균 A/B/C={100*s['A14']:.2f}/{100*s['B26']:.2f}/{100*s['C26_static']:.2f}%; "
            '구절과문장은 독립6표본아님. 각seed·완전일치·정상/정지/섞기/역순 출력은 요약참조. '
            'C의 정지학습 적합도와 정상입력 전이를 분리했고 정지/순서변형 차이만으로 실제조음이해를 인증하지 않는다. '
            '표현13→24·어절35→76·음절64→121이나14부모/13녹화/8명유지·구절문장중첩, A260/B·C140노출로 A/B는 순수다양성효과아님. '
            'B/C일정·초기상태·길이 동일. 독립 저장출력/변형/점수/일정/출처보호 검산통과. '
            '새원본/TAR·캐시추출·검수질문·다운로드·음성모델입력·교사·서비스교체0. '
            '문장 시각승인전이0·원경계주의/provisional·training_readyfalse 보존. '
            '최신 `results/stage5i_summary.json`/`next_step_handoff.json`, `results/completion_manifest.json` 동결. '
            '다음은 고정결과에 따른 별도 후속실험설계이며 보류6반복검수/학습연장/6단계 자동진행금지. '
            '[보고서](docs/녹화형_단계학습_2026-10-12/05i_표현확대와_움직임대조_학습.md). 아래는 이전 시점이다.\n\n'
        )
        save(manifest, {'schema': 'offline-stage5i-completion-manifest-v1', 'client_date': '2026-10-19',
            'status': 'completed_bounded_diversity_and_motion_comparison', 'artifacts': artifacts,
            'held_review_ids': summary['held_review_ids'], 'old_source_flags_preserved': True,
            'source_training_ready': False, 'service_changes': 0, 'main_updates': 10920,
            'preflight_updates': 5, 'evaluation_rows': 900, 'no_automatic_further_training': True})
        agents.write_bytes(original[:marker] + paragraph.encode('utf-8') + original[marker:])
        require(agents.read_bytes() == original[:marker] + paragraph.encode('utf-8') + original[marker:], 'AGENTS write verification failed')
        print(f'Frozen {len(artifacts)} current artifacts; prepended measured status with all previous AGENTS bytes preserved.')
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as exc:
        raise RuntimeError('Completion freeze failed; preserve existing evidence and diagnose before retry') from exc


if __name__ == '__main__':
    finalize()
