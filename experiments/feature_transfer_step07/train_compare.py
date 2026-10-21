"""Paired fresh compact students, CTC versus CTC plus fixed visual features."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT/'experiments/olkavs_official')]
from models.baseline import LipNetBaseline
from models.tokenizer import decode_labels, ctc_min_input_len
from evaluation_metrics import counts, aggregate

ARMS = ('A_ctc', 'B_feature')
MEAN = np.asarray([.485,.456,.406], dtype=np.float32)
STD = np.asarray([.229,.224,.225], dtype=np.float32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path | str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bind(path: Path) -> dict:
    return {'path':str(path.resolve()), 'sha256':sha(path)}


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path: Path, value: dict, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if replace:
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(encoded, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(encoded)


def verify(spec: dict) -> None:
    require(sha(spec['path']) == spec['sha256'], f"Changed evidence: {spec['path']}")


def setup_runtime() -> None:
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # AdaptiveAvgPool3d's overlapping bins may use atomic CUDA backwards.
    # Saved states/RNG are exact; GPU next-update replay is separately measured.
    torch.use_deterministic_algorithms(False)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    return {'python':random.getstate(), 'numpy':np.random.get_state(), 'torch':torch.get_rng_state().clone(),
            'cuda':[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def make_objects(seed: int, arm: str, device: str | torch.device) -> tuple:
    require(arm in ARMS, 'Unknown arm')
    seed_all(seed)
    model = LipNetBaseline().to(device).train()
    adapter = nn.Linear(256,512).to(device).train()
    # Construct the same adapter even in A so its initialization does not alter
    # the paired student's subsequent dropout RNG. It is unused/unoptimized in A.
    parameters = list(model.parameters()) + (list(adapter.parameters()) if arm == 'B_feature' else [])
    optimizer = torch.optim.Adam(parameters, lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    return model, adapter, optimizer


def frame_distillation_loss(h: torch.Tensor, target: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    require(h.ndim == target.ndim == 3 and h.shape[:2] == target.shape[:2] and h.shape[-1] == 256 and target.shape[-1] == 512,
            'Feature shape mismatch')
    require(not target.requires_grad, 'Teacher target must be detached')
    require(torch.isfinite(h).all().item() and torch.isfinite(target).all().item(), 'Nonfinite feature')
    require((torch.linalg.vector_norm(target, dim=-1) > 1e-8).all().item(), 'Zero teacher feature')
    projected = adapter(h)
    require(torch.isfinite(projected).all().item(), 'Nonfinite projected feature')
    return ((F.normalize(projected, dim=-1, eps=1e-8)-F.normalize(target, dim=-1, eps=1e-8))**2).sum(-1).mean()


def prepare_student(pixels: np.ndarray, device: str | torch.device) -> torch.Tensor:
    require(pixels.dtype == np.uint8 and pixels.ndim == 4 and pixels.shape[1:] == (96,96,3) and len(pixels)>0,
            'Wrong student pixels')
    normalized = (pixels.astype(np.float32)/255.-MEAN)/STD
    return torch.from_numpy(normalized).permute(3,0,1,2).unsqueeze(0).contiguous().to(device)


def optimizer_update(model: nn.Module, adapter: nn.Module, optimizer: torch.optim.Optimizer,
                     batch: list[dict], arm: str, device: str | torch.device) -> dict:
    require(arm in ARMS and batch, 'Empty or unknown update')
    model.train()
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    ctc_values, feature_values = [], []
    for clip in batch:
        pixels, tokens = clip['pixels'], clip['tokens']
        require(tokens and len(pixels) >= ctc_min_input_len(tokens), 'CTC infeasible clip')
        captured = []
        hook = model.projection[2].register_forward_hook(lambda module, args, out: captured.append(out)) if arm == 'B_feature' else None
        try:
            lp = model(prepare_student(pixels, device))
            require(lp.shape == (len(pixels),1,69) and torch.isfinite(lp).all().item(), 'Invalid CTC posterior')
            # CPU FP32 CTC retains autograd through the device copy and avoids
            # the native CUDA CTC backward's nondeterministic accumulation.
            ctc = F.ctc_loss(lp.float().cpu(), torch.tensor(tokens, dtype=torch.long),
                             torch.tensor([len(pixels)]), torch.tensor([len(tokens)]),
                             blank=0, reduction='mean', zero_infinity=False).to(device)
            if arm == 'B_feature':
                require(len(captured) == 1 and captured[0].requires_grad, 'Detached or missing student feature')
                target_np = clip['teacher_features']
                require(target_np.dtype == np.float32 and target_np.shape == (len(pixels),512), 'Wrong teacher feature cache')
                target = torch.from_numpy(target_np).unsqueeze(0).to(device)
                feature = frame_distillation_loss(captured[0], target, adapter)
            else:
                feature = ctc.new_zeros(())
            loss = ctc+feature
            require(torch.isfinite(loss).item(), 'Nonfinite training loss; no skipped updates allowed')
            (loss/len(batch)).backward()
            ctc_values.append(float(ctc.detach().cpu()))
            feature_values.append(float(feature.detach().cpu()))
        finally:
            if hook is not None:
                hook.remove()
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters), 'Nonfinite gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters,1.0, error_if_nonfinite=True).cpu())
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in parameters), 'Nonfinite updated parameter')
    ctc_mean, feature_mean = float(np.mean(ctc_values)), float(np.mean(feature_values))
    return {'ids':[r['id'] for r in batch], 'ctc_loss':ctc_mean, 'feature_loss':feature_mean,
            'lambda':1.0 if arm == 'B_feature' else 0.0, 'loss':ctc_mean+feature_mean,
            'gradient_norm_before_clipping':norm, 'optimizer_updated':True}


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_clone(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def runtime_state(model: nn.Module, adapter: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    return {'model':cpu_clone(model.state_dict()), 'adapter':cpu_clone(adapter.state_dict()),
            'optimizer':cpu_clone(optimizer.state_dict()), 'rng':capture_rng()}


def restore_state(state: dict, model: nn.Module, adapter: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model.load_state_dict(state['model'], strict=True)
    adapter.load_state_dict(state['adapter'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape,str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def perturb_pixels(pixels: np.ndarray, condition: str, sample_id: str) -> np.ndarray:
    if condition == 'normal':
        return pixels
    if condition == 'static':
        return np.repeat(pixels[len(pixels)//2:len(pixels)//2+1],len(pixels),axis=0)
    if condition == 'shuffled':
        seed = int.from_bytes(hashlib.sha256(('step07-order-v1:'+sample_id).encode()).digest()[:8], 'little')
        return pixels[np.random.default_rng(seed).permutation(len(pixels))]
    raise ValueError('Unknown perturbation')


def greedy(lp: np.ndarray) -> str:
    ids = lp.argmax(-1).tolist()
    collapsed, previous = [], None
    for token in ids:
        if token != previous and token != 0:
            collapsed.append(token)
        previous = token
    return decode_labels(collapsed)


def runtime_environment() -> dict:
    return {'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'precision':'FP32 GPU student, FP32 CPU CTC with autograd',
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'tf32':False,'strict_deterministic_algorithms':False}


def fixed_schedule(ids: list[str], seed: int, updates: int, batch_size: int = 2) -> list[list[str]]:
    require(len(ids) == len(set(ids)) and len(ids)>0 and updates>0, 'Invalid schedule')
    generator = random.Random(seed+17000)
    stream = []
    while len(stream) < updates*batch_size:
        epoch = sorted(ids)
        generator.shuffle(epoch)
        stream.extend(epoch)
    return [stream[i*batch_size:(i+1)*batch_size] for i in range(updates)]


# Artifact-loading, measured preflight and fixed-budget execution are defined
# below once the immutable preparation manifests are available.
