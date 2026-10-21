"""Seal completed stage5 results; stdlib-only, no inference or training."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPORT = ROOT / 'docs/녹화형_단계학습_2026-10-12/05_구절문장_비교학습.md'
MANIFEST = HERE / 'results/completion_manifest.json'
HANDOFF = HERE / 'results/next_step_handoff.json'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path: Path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write_new(path: Path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def verify_references(entries):
    for entry in entries:
        path = Path(entry['path'])
        assert path.is_file(), path
        assert sha(path) == entry['sha256'], path
        if 'bytes' in entry:
            assert path.stat().st_size == entry['bytes'], path


def main():
    assert not MANIFEST.exists() and not HANDOFF.exists(), 'Do not overwrite completed results'
    plan_path = HERE / 'execution_plan.json'
    assert sha(plan_path) == '516c631d10bf397a51fc0b2e5fd1a454e8784556db0c625378eca8f2261a9983'
    assert sha(HERE / 'controlled_pilot_policy.json') == 'a2a960b444f835e3d710c6c89e4036c9131bfdc4e15ee4c22146ea8635307094'
    verify_references(read(plan_path)['bindings'])
    previous = []
    for folder, count in [('offline_dataset_step03_20261013', 114),
                          ('offline_training_step04_20261013', 101)]:
        path = ROOT / 'experiments' / folder / 'results/completion_manifest.json'
        entries = read(path)['artifacts']
        assert len(entries) == count
        verify_references(entries)
        previous.append({'manifest': ref(path), 'verified_artifacts': count})
    audit_paths = sorted((HERE / 'audit').glob('*.json'))
    assert len(audit_paths) == 8
    assert all(read(path)['status'] == 'passed' for path in audit_paths)
    tests = [read(HERE / 'results/execution_unit_tests.json')['tests_passed'],
             read(HERE / 'results/evaluation_unit_tests.json')['tests_passed'],
             read(HERE / 'data/loader_test_results.json')['tests_run']]
    assert tests == [11, 21, 18]
    summary_path = HERE / 'results/stage5_summary.json'
    summary = read(summary_path)
    assert summary['status'] == 'completed_fixed_curriculum_comparison'
    aggregate_path = HERE / 'results/aggregate_metrics.json'
    assert sha(aggregate_path) == '7f9a6d524735fa36befdeb11d7c3f35e2c016fbc0f45ed0547770ebce8ed19da'
    plot_paths = [HERE / 'reporting' / name for name in (
        'sentence_cer_trajectory.png', 'sentence_cer_plotted_points.json',
        'plot_verification.json', 'visual_qa.json')]
    assert all(path.is_file() for path in plot_paths)
    assert not list(HERE.rglob('*.partial')), 'Incomplete writes must be investigated'
    created = datetime.now(timezone.utc).isoformat()
    handoff = {
        'schema': 'offline-stage5-next-step-handoff-v1',
        'created_utc': created,
        'status': 'comparison_complete_followup_diagnostic_proposed',
        'completed_stage': 5,
        'summary': ref(summary_path),
        'aggregate': ref(aggregate_path),
        'interpretation': ref(HERE / 'analysis/result_interpretation.json'),
        'resources': ref(HERE / 'results/resource_accounting.json'),
        'report': ref(REPORT),
        'plots': [ref(path) for path in plot_paths],
        'audits': [ref(path) for path in audit_paths],
        'primary_checkpoint': 'fixed_final_1260_only',
        'decision': {
            'A_mean_CER_percent': 78.37837837837837,
            'B_mean_CER_percent': 81.75675675675676,
            'B_minus_A_pp': 3.3783783783783825,
            'all_final_dev_exact': '0/3',
            'prespecified_research_criterion_met': False,
            'do_not_infer_curriculum_universally_ineffective': True,
            'original_stage6_expansion_gate_met': False,
        },
        'proposed_next_work_ko': [
            '5단계 후속 진단: 저장된 학습 손실과 평가 확률을 분석해 구절420회 학습 실패 및 초기값별 차이를 점검한다.',
            '정답과 학습 문장 후보의 CTC 확률을 비교하되 정답 포함 후보 순위를 실제 인식 정확도로 보고하지 않는다.',
            '결과에 따라 구절 학습 정도의 기준과 문장 노출을 맞춘 후속 대조를 별도 계획으로 고정한다.',
            '새 발화 인식 개선을 확인한 뒤 문장·여러 문장 확대와 녹화 프로그램 단계로 진행한다.',
        ],
        'immediate_additional_AIHub_download_needed': False,
        'immediate_colab_needed': False,
        'next_work_executed': False,
        'source_provisional': True,
        'source_training_ready': False,
        'source_precise_visual_boundary_approval': False,
        'boundary_flag_count_preserved': 2,
        'oov_parent_excluded': 'C293_A012_019',
        'additional_254_metadata_not_promoted_to_training': True,
        'new_training_or_service_changes_by_finalizer': 0,
        'rules': [
            'Keep all completed plans, code, checkpoints, predictions and reports immutable; use a new experiment/version.',
            'Current dev3 is internal historical-speaker pilot data, not untouched final or webcam performance.',
            'Do not automatically continue optimization, change the primary endpoint, or select a better intermediate checkpoint.',
            'Future actual interruption recovery must complete any missing fixed evaluation separately and count physical replay updates.',
        ],
    }
    write_new(HANDOFF, handoff)
    files = sorted(path for path in HERE.rglob('*') if path.is_file()
                   and '__pycache__' not in path.parts and '.pytest_cache' not in path.parts
                   and path.suffix != '.pyc' and path != MANIFEST)
    files.append(REPORT)
    entries = [ref(path) for path in files]
    manifest = {
        'schema': 'offline-stage5-completion-snapshot-v1',
        'created_utc': created,
        'status': 'completed_fixed_curriculum_comparison',
        'artifacts': entries,
        'artifact_count': len(entries),
        'total_artifact_bytes': sum(entry['bytes'] for entry in entries),
        'previous_manifests_verified': previous,
        'main_optimizer_updates': 5040,
        'separate_preflight_updates': 8,
        'evaluation_rows': 1080,
        'tests_passed': 50,
        'source_training_ready': False,
        'service_changes': 0,
        'policy': 'Immutable snapshot. Changes must use a separate version. Manifest excludes itself, mutable root AGENTS.md, and interpreter/test caches.',
    }
    write_new(MANIFEST, manifest)
    verify_references(read(MANIFEST)['artifacts'])
    for destination in re.findall(r'\]\(([^)]+)\)', REPORT.read_text(encoding='utf-8')):
        if not re.match(r'^[a-z]+://', destination):
            assert (REPORT.parent / destination).resolve().is_file(), destination
    print(json.dumps({'status': 'passed', 'artifacts': len(entries),
                      'bytes': manifest['total_artifact_bytes'],
                      'manifest_sha256': sha(MANIFEST),
                      'handoff_sha256': sha(HANDOFF),
                      'report_sha256': sha(REPORT)}, ensure_ascii=True))


if __name__ == '__main__':
    main()
