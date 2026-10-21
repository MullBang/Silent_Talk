"""Publish a completed-stage summary only after training, evaluation and independent audit."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    target = BASE / 'results/step10_summary.json'
    require(not target.exists(), 'Preserve existing summary')
    files = {
        'experiment_plan': BASE / 'comparison_plan.json',
        'freeze_manifest': BASE / 'plans/freeze_manifest.json',
        'mixed_loader_validation': BASE / 'results/mixed_loader_validation.json',
        'training_complete': BASE / 'results/training_complete.json',
        'comparison': BASE / 'results/comparison.json',
        'independent_audit': BASE / 'results/independent_audit.json',
        'posthoc_output_diagnostics': BASE / 'results/output_diagnostics.json',
    }
    files.update({f'evaluation_{seed}_{variant}': BASE / f'results/evaluation_seed{seed}_{variant}.json'
                  for seed in (42, 43) for variant in ('control_old441', 'expanded510')})
    refs = {key: bind(path) for key, path in files.items()}
    docs = {key: read(path) for key, path in files.items()}
    plan, train, comparison, audit = (docs[k] for k in ('experiment_plan', 'training_complete', 'comparison', 'independent_audit'))
    require(docs['freeze_manifest']['experiment_plan'] == refs['experiment_plan'] and train['experiment_plan'] == refs['experiment_plan'], 'Plan binding differs')
    require(train['status'] == comparison['status'] == 'completed' and audit['status'] == 'passed', 'Incomplete execution/audit')
    for key in ('experiment_plan', 'training_complete', 'comparison'):
        require(audit[key] == refs[key], 'Independent audit is bound to another artifact: ' + key)
    require(train['new_optimizer_updates'] == 6400 and train['reused_historical_optimizer_updates'] == 6400, 'Update accounting changed')
    require(comparison['optimizer_accounting'] == {'new_updates_executed_in_step10': 6400,
            'historical_control_updates_reused': 6400, 'updates_executed_by_summary': 0}, 'Comparison update accounting differs')
    for ref in plan['execution_code']:
        require(bind(ref['path']) == ref, 'Frozen code changed')
    for ref in train['run_results']:
        require(bind(ref['path']) == ref, 'Training result changed')
    evaluations = []
    for seed in (42, 43):
        for variant in ('control_old441', 'expanded510'):
            key = f'evaluation_{seed}_{variant}'
            doc = docs[key]
            require(doc['status'] == 'completed' and doc['experiment_plan'] == refs['experiment_plan'], 'Evaluation incomplete/unbound')
            require(doc['seed'] == seed and doc['variant'] == variant and doc['completed_updates'] == 3200, 'Wrong evaluation target')
            require(doc['inferred_condition_records'] == 285 and doc['inferred_distinct_records'] == 95, 'Evaluation coverage changed')
            require(doc['new_optimizer_updates'] == 0 and not doc['service_weights_updated'], 'Unexpected evaluation mutation')
            if variant == 'control_old441':
                require(doc['old_control_replay']['status'] == 'passed', 'Historical control failed to reproduce')
            evaluations.append(refs[key])
    require({(r['path'], r['sha256']) for r in evaluations} ==
            {(r['evaluation']['path'], r['evaluation']['sha256']) for r in audit['evaluations']}, 'Audited evaluation cohort differs')
    diagnostics = docs['posthoc_output_diagnostics']
    require(diagnostics['status'] == 'completed' and diagnostics['posthoc'] is True and
            bind(diagnostics['code']['path']) == diagnostics['code'], 'Output diagnostic status/code differs')
    require(diagnostics['evaluations'] == evaluations, 'Output diagnostic evaluations differ')
    summary = {
        'schema': 'lipnet-step10-summary-v1', 'status': 'completed_provisional_data_expansion_comparison',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'generator': bind(Path(__file__)), 'artifacts': refs,
        'seeds': [42, 43], 'method': 'B_feature', 'historical_control_training_records': 441,
        'expanded_training_records': 510, 'new_provisional_training_records': 69, 'new_training_speakers': 12,
        'old_development_records': 71, 'new_development_records': 24, 'supplemental_new_development_records': 23,
        'held_training_records': 27, 'new_optimizer_updates': 6400, 'historical_control_updates_reused': 6400,
        'new_trained_models': 2, 'historical_control_models_reused': 2, 'evaluation_model_runs': 4,
        'unique_model_sentence_condition_evaluations': 1140, 'supplemental_reaggregated_rows': 276,
        'normal_two_seed_means': {group: comparison['two_seed_arithmetic_means'][group]['normal'] for group in ('old71', 'new24', 'new23')},
        'observed_conclusion': 'Both expanded seeds worsened normal CER on old71 and new24; retain historical service weights. Repetitive output and weak temporal-control advantage require targeted diagnosis.',
        'next_step_proposal': 'Freeze a small train-only diagnostic set; check memorization, CTC output diversity, and feature-loss influence in matched ablations before claiming a remedy. Cause not established by this comparison.',
        'pending_listening': plan['pending_listening'], 'source_training_ready_flags_promoted': False,
        'human_full_transcript_approvals_added': 0, 'service_weights_updated': False,
        'teacher_model_trained': False, 'new_data_downloaded': False, 'webcam_accuracy_established': False,
        'limits': plan['limitations'] + [
            'This compares data expansion with the same feature-transfer method, not a new architecture or a fresh CTC-only ablation.',
            'Matched optimizer updates and utterance presentations do not imply equal processed frames, epochs, FLOPs or wall time.',
            'Two seeds on small reviewed development groups do not establish statistical significance or deployment readiness.',
        ],
    }
    with target.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'summary': bind(target), 'status': summary['status'],
                      'normal_two_seed_means': summary['normal_two_seed_means']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
