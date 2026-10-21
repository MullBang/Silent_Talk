"""Bind a separate evaluation-only scope before any stage5d inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
S3 = ROOT / 'experiments/offline_dataset_step03_20261013'
S4 = ROOT / 'experiments/offline_training_step04_20261013'
S5 = ROOT / 'experiments/offline_comparison_step05_20261013'
S5B = ROOT / 'experiments/offline_diagnostic_step05b_20261018'
S5C = ROOT / 'experiments/offline_phrase_only_step05c_20261018'


def read(path: Path) -> dict:
    """Read existing evidence or fail without substituting defaults."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'Cannot read {path}') from exc


def ref(path: Path) -> dict:
    """Bind local bytes; no deserialization of checkpoint tensors."""
    try:
        path = path.resolve()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
    except OSError as exc:
        raise RuntimeError(f'Cannot bind {path}') from exc


def save(path: Path, value: dict) -> None:
    """Create scoped JSON without overwriting earlier evidence."""
    try:
        assert path.resolve().is_relative_to(HERE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, AssertionError) as exc:
        raise RuntimeError(f'Cannot create {path}') from exc


def data_plan() -> None:
    """Freeze three existing development phrases and all temporal conditions."""
    try:
        handoff_path = S5C / 'results/next_step_handoff.json'
        handoff = read(handoff_path)
        assert handoff['fixed_fit_gate_passed'] is False
        source_plan = Path(handoff['source_evaluation_plan']['path'])
        assert ref(source_plan) == handoff['source_evaluation_plan']
        items = [r for r in read(source_plan)['items'] if r['split'] == 'dev' and r['kind'] == 'phrase']
        assert len(items) == 3 and sum(r['T'] for r in items) == 96
        assert [r['clip_id'] for r in items] == [r['clip_id'] for r in handoff['candidate_targets']]
        assert all(ref(Path(row['path'])) == row for row in handoff['checkpoints'])
        plan = {
            'schema': 'offline-stage5d-unseen-phrase-evaluation-plan-v1', 'client_date': '2026-10-18',
            'items': items, 'checkpoints': [1680], 'seeds': [42, 43],
            'conditions_by_update': {'1680': ['normal', 'static', 'shuffled']},
            'fixed_checkpoints': handoff['checkpoints'], 'source_handoff': ref(handoff_path),
            'source_evaluation_plan': ref(source_plan),
            'planned_neural_evaluations': 18, 'planned_frames': 576, 'optimizer_updates': 0,
            'primary': 'Each final model: corpus NFC character CER and exact phrase count on normal dev3.',
            'metrics': 'Preserve spaces and U+FFFD in predictions; frozen greedy decoder/metric contract.',
            'selection': 'Exactly the prior handoff targets; no score-based clip, seed or checkpoint selection.',
            'reuse_limits': 'Known speakers and development data used in previous experiments/design; exploratory, not independent test performance.',
            'source_training_ready': False, 'source_provisional': True,
            'model_updates_allowed': False, 'full_sentence_input_allowed': False,
            'teacher_or_audio_input_allowed': False, 'automatic_sentence_expansion': False,
            'resource_caps': {'cuda_reserved_bytes': 2*1024**3, 'seconds_per_model': 120},
        }
        save(HERE / 'evaluation_plan.json', plan)
        policy = {
            'schema': 'lipnet-stage5d-evaluation-only-data-policy-v1', 'mode': 'evaluation_only',
            'authorization': {'source': 'current_user_request', 'request_text': '다음 진행',
                              'scope': 'stage5d_fixed_unseen_phrase_evaluation'},
            'evaluation_plan': ref(HERE / 'evaluation_plan.json'),
            'handoff': ref(S3 / 'results/cache_handoff.json'),
            'stage3_frozen_manifest': ref(S3 / 'results/completion_manifest.json'),
            'allowed_evaluation_clip_ids': sorted(r['clip_id'] for r in items),
            'allowed_optimizer_clip_ids': [], 'allow_provisional_cache_for_controlled_evaluation': True,
            'preserve_provisional_flags': True, 'optimizer_updates_authorized': False,
            'service_change_authorized': False,
        }
        save(HERE / 'controlled_pilot_policy.json', policy)
        print(json.dumps({'status': 'evaluation_scope_prepared', 'clips': 3, 'planned_forwards': 18, 'optimizer_updates': 0}))
    except Exception as exc:
        raise RuntimeError('Stage5d data plan preparation failed; preserve partial artifacts') from exc


def freeze_execution() -> None:
    """Bind completed production code and the prior numerical runtime."""
    try:
        code = [HERE / p for p in ['prepare.py', 'run_probe.py', 'test_probe.py',
                'data/loader.py', 'data/test_loader.py', 'evaluation/evaluator.py',
                'evaluation/test_evaluator.py', 'evaluation/aggregate.py']]
        assert all(p.is_file() for p in code)
        policy = read(HERE / 'controlled_pilot_policy.json')
        plan = {
            'schema': 'offline-stage5d-inference-only-execution-v1', 'client_date': '2026-10-18',
            'seeds': [42, 43], 'checkpoint_update': 1680, 'evaluation_rows': 18,
            'evaluation_frames': 576, 'optimizer_updates': 0, 'optimizer_objects': 0,
            'runtime': read(S5C / 'execution_plan.json')['runtime'],
            'bindings': [ref(p) for p in code + [HERE / 'evaluation_plan.json', HERE / 'controlled_pilot_policy.json',
                         S4 / 'engine.py', S5 / 'evaluation/evaluator.py',
                         S5C / 'results/next_step_handoff.json', S5C / 'execution_plan.json']],
            'previous_manifests': [ref(p / 'results/completion_manifest.json') for p in [S3, S4, S5, S5B, S5C]],
            'checkpoints': read(HERE / 'evaluation_plan.json')['fixed_checkpoints'],
            'data_policy': ref(HERE / 'controlled_pilot_policy.json'),
            'dataset_class': 'Dataset', 'evaluation_plan': policy['evaluation_plan'],
            'checkpoint_load': 'Restricted weights_only CPU load, validate full saved state and1680 identity, instantiate model only, restore model and RNG; Adam is never instantiated.',
            'state_isolation': 'Exact checkpoint model hash before/after all9 evaluations; unchanged model/gradient/flags, restored RNG; source checkpoint bytes preserved.',
            'failure_policy': 'Preserve failed/partial outputs and stop. Completed outputs may be hash-verified/reused, with new versus historical forward calls accounted separately.',
            'no_new_training': True, 'no_source_media_or_audio': True, 'no_service_change': True,
            'pre_execution_gate_required': True,
        }
        save(HERE / 'execution_plan.json', plan)
        print(json.dumps({'status': 'execution_plan_frozen', 'plan': ref(HERE / 'execution_plan.json')}))
    except Exception as exc:
        raise RuntimeError('Stage5d execution freeze failed') from exc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['data', 'freeze'])
    action = parser.parse_args().action
    data_plan() if action == 'data' else freeze_execution()
