"""Freeze the completed stage5c evidence and a separate, unexecuted next-step outline."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPORT = ROOT / 'docs/녹화형_단계학습_2026-10-12/05c_구절전용_학습과_재현검증.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    audit_names = ['source_preflight', 'preflight_plan_audit', 'preflight_checkpoint_audit',
                   'replay420_seed42', 'replay420_seed43', 'main_checkpoint_audit',
                   'evaluation_audit', 'aggregate_audit', 'final_source_preservation', 'report_audit',
                   'analysis_audit']
    for name in audit_names:
        assert read(HERE / f'audit/{name}.json')['status'] == 'passed', name
    summary = read(HERE / 'results/stage5c_summary.json')
    assert summary['fit_gate']['passed'] is False
    assert read(HERE / 'results/resource_accounting.json')['total_physical_optimizer_updates'] == 3368
    old_eval = ROOT / 'experiments/offline_training_step04_20261013/protocol/evaluation_plan.json'
    targets = [item for item in read(old_eval)['items'] if item['split'] == 'dev' and item['kind'] == 'phrase']
    assert len(targets) == 3 and sum(r['T'] for r in targets) == 96
    train_texts = {r['reference_text'] for r in summary['normal_trajectory'][0]['predictions']}
    assert not train_texts.intersection(item['text'] for item in targets)
    handoff = {
        'schema': 'offline-stage5c-next-step-handoff-v1', 'client_date': '2026-10-18',
        'status': 'completed_phrase_only_diagnostic_next_frozen_model_probe_proposed',
        'summary': ref(HERE / 'results/stage5c_summary.json'), 'report': ref(REPORT),
        'fixed_fit_gate_passed': False,
        'reason': 'Both final1680 models fit12/12; seed43 at1260 retained one character error. Static8/12 for both seeds leaves shortcut/memorization concerns.',
        'next_stage': '5d', 'next_status': 'outline_only_not_executed',
        'question': 'Can the fixed phrase-trained final models decode three previously untrained phrase clips, and how do temporal controls change their output?',
        'checkpoints': [ref(HERE / f'runs/seed{seed}/update_1680.pt') for seed in [42, 43]],
        'source_evaluation_plan': ref(old_eval),
        'candidate_targets': [{k: item[k] for k in ['clip_id', 'parent_id', 'split', 'kind', 'text', 'T']} for item in targets],
        'conditions': ['normal', 'static', 'shuffled'],
        'proposed_neural_evaluations': 18, 'proposed_evaluation_frames': 576,
        'proposed_optimizer_updates': 0, 'actual_next_stage_neural_evaluations': 0,
        'pre_execution_work': [
            'Create separate stage5d evaluation-only plan/policy/output; current stage5c policy excludes dev inputs.',
            'Bind both final1680 checkpoint hashes and all three existing cache/label hashes; do not select a checkpoint or seed by dev results.',
            'Preserve the prior split, canonical labels, preprocessing, decoder, exact static/shuffle indices and source flags.',
            'Validate scoped input access and inference-only state isolation; no optimizer, new labels, teacher or audio input.',
            'Run all18 fixed evaluations, retain every prediction/posterior and independently audit metrics/inputs/CTC.',
            'Report this reused three-clip development probe as exploratory; it is not an untouched test or new-speaker/webcam performance.',
        ],
        'future_data_need': 'An independent reviewed recording/transcript-disjoint set is needed before generalization claims; inventory existing material first. The unreviewed254 metadata are not automatically admitted.',
        'strict_gate_remains_failed': True, 'extra_phrase_updates': 0,
        'automatic_sentence_expansion': False, 'conditional_equal_sentence_exposure_comparison': 'on_hold_not_executed',
        'new_download_needed_for_immediate_next_probe': False, 'colab_needed_for_immediate_next_probe': False,
        'source_training_ready': False, 'source_provisional': True, 'service_changes': 0,
    }
    save(HERE / 'results/next_step_handoff.json', handoff)
    manifest_path = HERE / 'results/completion_manifest.json'
    excluded_dirs = {'__pycache__', '.pytest_cache'}
    paths = [p for p in HERE.rglob('*') if p.is_file() and p != manifest_path
             and not excluded_dirs.intersection(p.parts) and p.suffix not in {'.pyc', '.partial'}]
    assert not list(HERE.rglob('*.partial'))
    paths.append(REPORT)
    artifacts = [ref(p) for p in sorted(paths)]
    manifest = {
        'schema': 'offline-stage5c-completion-snapshot-v1', 'client_date': '2026-10-18',
        'status': summary['status'], 'artifacts': artifacts, 'artifact_count': len(artifacts),
        'excluded': ['this manifest itself', '__pycache__', '.pytest_cache', '*.pyc', 'mutable repository AGENTS.md'],
        'frozen_production_code_and_plans': True, 'future_changes_require_separate_version': True,
        'main_optimizer_updates': 3360, 'separate_preflight_updates': 8, 'evaluation_rows': 144,
        'fit_gate_passed': False, 'new_stage5d_evaluations': 0,
        'source_training_ready': False, 'automatic_stage6_expansion': False,
    }
    save(manifest_path, manifest)
    assert all(ref(row['path']) == row for row in read(manifest_path)['artifacts'])
    print(json.dumps({'status': 'frozen_and_hash_verified', 'artifacts': len(artifacts),
                      'manifest': ref(manifest_path), 'handoff': ref(HERE / 'results/next_step_handoff.json')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
