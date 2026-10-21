"""Evaluate two fixed models without constructing an optimizer or updating weights."""
from __future__ import annotations
import gc
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from prepare import read, ref, save


def require(value: bool, message: str) -> None:
    """Reject a scope or identity discrepancy without a fallback."""
    if not value:
        raise ValueError(message)


def module(path: Path, name: str) -> Any:
    """Load a local module without invoking its CLI."""
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        require(spec is not None and spec.loader is not None, 'Missing module')
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    except Exception as exc:
        raise RuntimeError(f'Cannot import {path}') from exc


ENGINE = ROOT / 'experiments/offline_training_step04_20261013/engine.py'
require(ref(ENGINE)['sha256'] == 'fcd64be0d8731a75c10a3f7dfc41d58c814bda51b08ec2032c88ac50e80a2b2f', 'Frozen engine changed')
old = module(ENGINE, 'stage5d_frozen_model_state_primitives')
torch = old.torch


def verify_plan(plan: dict) -> None:
    """Verify frozen code and all existing snapshots before any inference."""
    try:
        require(plan['schema'] == 'offline-stage5d-inference-only-execution-v1', 'Wrong stage scope')
        require(plan['seeds'] == [42, 43] and plan['checkpoint_update'] == 1680, 'Wrong fixed models')
        require(plan['evaluation_rows'] == 18 and plan['evaluation_frames'] == 576, 'Changed evaluation budget')
        require(plan['optimizer_updates'] == plan['optimizer_objects'] == 0, 'Optimization is prohibited')
        handoff = read(ROOT / 'experiments/offline_phrase_only_step05c_20261018/results/next_step_handoff.json')
        require(plan['checkpoints'] == handoff['checkpoints'], 'Checkpoint selection changed')
        for binding in plan['bindings'] + plan['checkpoints'] + plan['previous_manifests']:
            require(ref(Path(binding['path'])) == binding, 'Changed artifact: ' + binding['path'])
        for binding in plan['previous_manifests']:
            for artifact in read(Path(binding['path']))['artifacts']:
                require(ref(Path(artifact['path'])) == artifact, 'Frozen source changed: ' + artifact['path'])
    except Exception as exc:
        raise RuntimeError('Stage5d pre-inference plan validation failed') from exc


def validate_payload(payload: dict, seed: int, runtime: dict) -> None:
    """Check the complete saved state, final cursor and known training identity."""
    try:
        require(seed in [42, 43] and type(seed) is int, 'Unknown seed')
        identity = payload['identity']
        require(payload['schema'] == 'offline-stage5c-complete-checkpoint-v1', 'Wrong checkpoint schema')
        require(payload['completed_updates'] == 1680 and type(payload['completed_updates']) is int, 'Not final1680')
        require(identity['seed'] == seed and identity['mode'] == 'main' and identity['arm'] == 'P_phrase', 'Wrong source model')
        require(identity['runtime'] == runtime, 'Numerical runtime changed')
        require(payload['state_hashes'] == old.hashes(payload['state']), 'Saved complete-state hash changed')
        require(len(payload['history']) == 1680 and payload['next_task'] is None, 'Wrong final cursor')
        require(all(r['update'] == i and r['kind'] == 'phrase' and r['optimizer_updated'] is True
                    for i, r in enumerate(payload['history'], 1)), 'Wrong saved training history')
        require(all(float(r['step']) == 1680 for r in payload['state']['optimizer']['state'].values()), 'Wrong historical Adam step')
    except Exception as exc:
        raise RuntimeError('Refusing mismatched final checkpoint') from exc


def run() -> None:
    """Run the sole18-forward probe, preserving all source state and failed evidence."""
    began = time.perf_counter()
    try:
        plan_path, policy_path = HERE / 'execution_plan.json', HERE / 'controlled_pilot_policy.json'
        plan = read(plan_path)
        verify_plan(plan)
        gate = read(HERE / 'results/pre_execution_gate.json')
        require(gate['status'] == 'passed' and gate['execution_plan'] == ref(plan_path), 'Missing exact pre-execution gate')
        for binding in gate['evidence']:
            require(ref(Path(binding['path'])) == binding, 'Pre-execution evidence changed')
        require(old.setup() == plan['runtime'], 'Runtime differs from prior models')
        loader = module(HERE / 'data/loader.py', 'stage5d_data')
        evaluator = module(HERE / 'evaluation/evaluator.py', 'stage5d_evaluation')
        dataset = loader.Dataset(HERE / 'evaluation_plan.json', policy_path,
                                 expected_policy_sha256=ref(policy_path)['sha256'])
        inputs = [dataset.get(item['clip_id'], purpose='evaluation') for item in read(HERE / 'evaluation_plan.json')['items']]
        require(len(inputs) == 3 and sum(item['length'] for item in inputs) == 96, 'Unexpected development inputs')
        attempt = HERE / 'results/attempt.json'
        require(not attempt.exists(), 'Existing attempt: inspect before any new neural calls')
        save(attempt, {'schema': 'stage5d-attempt-v1', 'plan': ref(plan_path), 'planned_forwards': 18,
                       'optimizer_objects': 0, 'optimizer_updates': 0, 'input_audits': [r['audit'] for r in inputs]})
        runs = []
        for seed, source in zip(plan['seeds'], plan['checkpoints']):
            require(ref(Path(source['path'])) == source, 'Checkpoint bytes changed')
            payload = old.runner._unpack_safe(torch.load(source['path'], map_location='cpu', weights_only=True))
            validate_payload(payload, seed, plan['runtime'])
            old.core.seed_all(seed)
            model = old.pool.configure_model(old.LipNetBaseline()).to('cuda:0').eval()
            model.load_state_dict(payload['state']['model'], strict=True)
            old.core.restore_rng(payload['state']['rng'])
            before = old.runner.nested_hash(old.core.cpu_clone(model.state_dict()))
            require(before == payload['state_hashes']['model'], 'Loaded model is not the saved model')
            rng_before = old.runner.nested_hash(old.core.capture_rng())
            identity = {'plan_sha256': ref(plan_path)['sha256'], 'policy_sha256': ref(policy_path)['sha256'],
                        'runtime': plan['runtime'], 'mode': 'evaluation_only', 'seed': seed, 'arm': 'P_phrase',
                        'completed_update': 1680, 'checkpoint': source}
            result = evaluator.evaluate(model, dataset, identity, HERE / f'runs/seed{seed}')
            after = old.runner.nested_hash(old.core.cpu_clone(model.state_dict()))
            rng_after = old.runner.nested_hash(old.core.capture_rng())
            require(before == after and rng_before == rng_after, 'Evaluation mutated model or RNG')
            require(result['count'] == result['new_forward_calls'] == 9 and result['failures'] == 0, 'Evaluation incomplete')
            require(ref(Path(source['path'])) == source, 'Source checkpoint changed during evaluation')
            receipt = {'seed': seed, 'checkpoint': source, 'complete_saved_state_hashes': payload['state_hashes'],
                       'loaded_model_hash': before, 'after_model_hash': after,
                       'rng_before': rng_before, 'rng_after': rng_after,
                       'optimizer_objects_created': 0, 'optimizer_updates': 0, 'evaluation': result}
            save(HERE / f'results/seed{seed}_receipt.json', receipt)
            runs.append(receipt)
            print(f'seed{seed}: completed9 fixed dev-phrase evaluations; weights/RNG unchanged', flush=True)
            del model, payload
            gc.collect()
            torch.cuda.empty_cache()
        require(sum(r['evaluation']['frames'] for r in runs) == 576, 'Unexpected frame count')
        save(HERE / 'results/execution_summary.json', {
            'schema': 'offline-stage5d-execution-summary-v1', 'status': 'completed',
            'plan': ref(plan_path), 'policy': ref(policy_path), 'runs': runs,
            'neural_forward_calls': 18, 'evaluation_frames': 576, 'failed_rows': 0,
            'optimizer_updates': 0, 'optimizer_objects': 0, 'checkpoint_tensor_loads': 2,
            'train_forward_calls': 0, 'full_sentence_forward_calls': 0,
            'source_media_reads': 0, 'audio_inputs': 0, 'teacher_calls': 0,
            'downloads': 0, 'service_changes': 0, 'seconds': time.perf_counter() - began,
        })
    except Exception as exc:
        failure = HERE / 'results/failure.json'
        if not failure.exists():
            save(failure, {'status': 'failed', 'error': repr(exc), 'seconds': time.perf_counter() - began,
                           'note': 'Preserve all artifacts and count any completed/partial forwards before retry.'})
        raise


if __name__ == '__main__':
    run()
