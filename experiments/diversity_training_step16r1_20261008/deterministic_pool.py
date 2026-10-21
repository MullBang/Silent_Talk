"""Parameter-free framewise pooling equivalents for strict CUDA determinism.

This experiment-only replacement is restricted to the compact student's 96px
input geometry. No existing model file or service checkpoint is modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent


def pooling_contract():
    return {'schema': 'step16r1-framewise-pooling-v1', 'strict_deterministic_algorithms': True,
            'warn_only': False, 'input_spatial_shape': [96, 96],
            'max_pool': {'original': 'MaxPool3d((1,2,2))', 'replacement': 'framewise MaxPool2d(2,2)',
                         'blocks': 3, 'time_pooling': False, 'ties': 'first spatial maximum'},
            'spatial_pool': {'original': 'AdaptiveAvgPool3d((None,4,4))',
                             'input': [6, 6], 'output': [4, 4],
                             'replacement': 'framewise AvgPool2d(2,1), select rows/columns [0,1,3,4]',
                             'selected_indices': [0, 1, 3, 4]},
            'trainable_parameter_change': 0, 'persistent_buffer_change': 0,
            'state_dict_keys_unchanged': True, 'rng_consumption_by_configuration': 0}


def setup_runtime(core):
    core.setup_runtime()
    torch.use_deterministic_algorithms(True, warn_only=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Unexpected CUBLAS workspace configuration')


def amend_runtime(runtime):
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise ValueError('Strict deterministic algorithms must be enabled')
    return {**runtime, 'strict_deterministic_algorithms': True, 'deterministic_algorithms': True,
            'deterministic_warn_only': False}


def _frames(x):
    if x.ndim != 5:
        raise ValueError('Expected B,C,T,H,W tensor')
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, c, t)


def _video(x, dimensions):
    b, c, t = dimensions
    return x.reshape(b, t, c, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)


class FramewiseMaxPool(nn.Module):
    def forward(self, x):
        frames, dimensions = _frames(x)
        return _video(F.max_pool2d(frames, kernel_size=2, stride=2), dimensions)


class FramewiseAdaptive6to4(nn.Module):
    def forward(self, x):
        if x.shape[-2:] != (6, 6):
            raise ValueError('Fixed adaptive pooling requires 6x6 input')
        frames, dimensions = _frames(x)
        pooled = F.avg_pool2d(frames, kernel_size=2, stride=1)
        indices = torch.tensor([0, 1, 3, 4], device=x.device, dtype=torch.long)
        return _video(pooled.index_select(-2, indices).index_select(-1, indices), dimensions)


def configure_model(model):
    if not hasattr(model, 'conv') or len(model.conv) != 3:
        raise ValueError('Expected the three-block compact student')
    for block in model.conv:
        pool = block.pool
        if isinstance(pool, FramewiseMaxPool):
            continue
        if not (isinstance(pool, nn.MaxPool3d) and pool.kernel_size == (1, 2, 2) and
                pool.stride == (1, 2, 2) and pool.padding == 0 and pool.dilation == 1 and
                not pool.ceil_mode and not pool.return_indices):
            raise ValueError('Unexpected original max pooling configuration')
    if not (isinstance(model.spatial_pool, FramewiseAdaptive6to4) or
            isinstance(model.spatial_pool, nn.AdaptiveAvgPool3d) and model.spatial_pool.output_size == (None, 4, 4)):
        raise ValueError('Unexpected original adaptive pooling configuration')
    for block in model.conv:
        if not isinstance(block.pool, FramewiseMaxPool):
            block.pool = FramewiseMaxPool().train(block.pool.training)
    if not isinstance(model.spatial_pool, FramewiseAdaptive6to4):
        model.spatial_pool = FramewiseAdaptive6to4().train(model.spatial_pool.training)
    return model


def cpu_equivalence(include_arrays=False):
    """Compare reference operators and gradients, including overlapping bins/ties."""
    generator = torch.Generator(device='cpu').manual_seed(1615)
    rows, arrays = [], {}
    for dtype in (torch.float32, torch.float64):
        for kind in ('max_random', 'max_ties', 'average_random', 'average_impulse'):
            shape = (2, 3, 4, 6, 6)
            value = torch.randn(shape, generator=generator, dtype=dtype)
            if kind == 'max_ties':
                value = value.round()
                value[..., :2, :2] = 2
            elif kind == 'average_impulse':
                value.zero_()
                value[..., 1, 1] = 1
            left, right = value.clone().requires_grad_(), value.clone().requires_grad_()
            original = nn.MaxPool3d((1, 2, 2)) if kind.startswith('max') else nn.AdaptiveAvgPool3d((None, 4, 4))
            replacement = FramewiseMaxPool() if kind.startswith('max') else FramewiseAdaptive6to4()
            a, b = original(left), replacement(right)
            weights = torch.randn(a.shape, generator=generator, dtype=dtype)
            (a * weights).sum().backward(); (b * weights).sum().backward()
            atol = 1e-6 if dtype == torch.float32 else 1e-12
            passed = torch.allclose(a, b, atol=atol, rtol=atol) and torch.allclose(left.grad, right.grad, atol=atol, rtol=atol)
            rows.append({'kind': kind, 'dtype': str(dtype), 'shape': list(shape), 'passed': bool(passed),
                         'atol': atol, 'rtol': atol, 'output_max_abs_difference': (a-b).abs().max().item(),
                         'input_gradient_max_abs_difference': (left.grad-right.grad).abs().max().item(),
                         'output_bitwise_equal': torch.equal(a, b), 'input_gradient_bitwise_equal': torch.equal(left.grad, right.grad)})
            prefix = f'case{len(rows)-1}'
            rows[-1]['array_prefix'] = prefix
            for name, tensor in {'input': value, 'upstream_gradient': weights, 'original_output': a,
                                 'original_input_gradient': left.grad, 'new_output': b,
                                 'new_input_gradient': right.grad}.items():
                arrays[f'{prefix}_{name}'] = tensor.detach().numpy()
    result = {'passed': all(row['passed'] for row in rows), 'cases': rows}
    return (result, arrays) if include_arrays else result


def verify():
    """Synthetic CPU equivalence and two CUDA forward/backward calls; zero steps."""
    import engine
    core, runner = engine.core, engine.runner
    output = BASE / 'results/deterministic_pool_verification.json'
    arrays_path = BASE / 'results/deterministic_pool_cpu_arrays.npz'
    if output.exists() or arrays_path.exists():
        raise ValueError('Pooling evidence already exists; no selective rerun')
    setup_runtime(core)
    cpu, arrays = cpu_equivalence(include_arrays=True)
    if not cpu['passed']:
        raise ValueError('CPU pooling equivalence failed')
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for this verification')
    outer = core.capture_rng()
    repeats = []
    try:
        core.seed_all(42)
        model = core.LipNetBaseline().to(device).train()
        before_state, before_rng = runner.nested_hash(model.state_dict()), runner.nested_hash(core.capture_rng())
        configure_model(model)
        unchanged_state = runner.nested_hash(model.state_dict()) == before_state
        unchanged_rng = runner.nested_hash(core.capture_rng()) == before_rng
        sample = torch.randn((1, 3, 5, 96, 96), generator=torch.Generator(device='cpu').manual_seed(1616)).to(device)
        replay_rng = core.capture_rng()
        for _ in range(2):
            core.restore_rng(replay_rng)
            model.zero_grad(set_to_none=True)
            x = sample.detach().clone().requires_grad_()
            prediction = model(x)
            prediction.square().mean().backward()
            torch.cuda.synchronize(device)
            gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
            finite = all(value is not None and torch.isfinite(value).all().item() for value in gradients.values())
            repeats.append({'output_sha256': runner.nested_hash(prediction.detach()),
                            'input_gradient_sha256': runner.nested_hash(x.grad),
                            'parameter_gradients_sha256': runner.nested_hash(gradients),
                            'parameter_gradient_arrays': len(gradients),
                            'parameter_gradient_array_sha256': {name: runner.nested_hash(value) for name, value in gradients.items()},
                            'all_finite': bool(finite and torch.isfinite(prediction).all().item() and torch.isfinite(x.grad).all().item()),
                            'state_sha256': runner.nested_hash(model.state_dict()),
                            'post_backward_rng_sha256': runner.nested_hash(core.capture_rng())})
        state_unchanged_after_backward = all(row['state_sha256'] == before_state for row in repeats)
        passed = unchanged_state and unchanged_rng and state_unchanged_after_backward and all(row['all_finite'] for row in repeats) and repeats[0] == repeats[1]
        value = {'schema': 'step16r1-deterministic-pool-verification-v1', 'status': 'passed' if passed else 'failed',
                 'created_utc': datetime.now(timezone.utc).isoformat(), 'contract': pooling_contract(),
                 'execution_code': engine.execution_code(), 'runtime': engine.runtime_for(device),
                 'cpu_equivalence': cpu, 'synthetic_input_shape': [1, 3, 5, 96, 96], 'synthetic_input_seed': 1616,
                 'student_seed': 42, 'training_mode': True, 'gpu_forward_calls': 2, 'gpu_backward_calls': 2,
                 'optimizer_updates': 0, 'training_or_evaluation_data_loaded': False, 'teacher_calls': 0,
                 'configure_state_unchanged': unchanged_state, 'configure_rng_unchanged': unchanged_rng,
                 'state_unchanged_after_backward': state_unchanged_after_backward, 'repeats': repeats,
                 'bitwise_equal_repeats': repeats[0] == repeats[1]}
    finally:
        core.restore_rng(outer)
    value['caller_rng_restored_exactly'] = runner.nested_hash(core.capture_rng()) == runner.nested_hash(outer)
    if not value['caller_rng_restored_exactly']:
        value['status'] = 'failed'
    output.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open('xb') as stream:
        engine.np.savez_compressed(stream, **arrays)
    value['cpu_equivalence_arrays'] = {'path': str(arrays_path.resolve()), 'sha256': engine.file_sha(arrays_path)}
    runner._save_json(output, value)
    if value['status'] != 'passed':
        raise ValueError('Strict pooling verification failed')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify'])
    parser.parse_args()
    result = verify()
    print({'status': result['status'], 'optimizer_updates': result['optimizer_updates'],
           'output': str(BASE / 'results/deterministic_pool_verification.json')})
