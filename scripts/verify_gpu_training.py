"""가장 긴 실제 train 클립 두 개로 GPU 학습/저장/CPU 재로드를 검증한다.

정확도 평가가 아니다. val/test는 학습/추론에 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from train import (ClipDataset, collate, ctc_training_loss, resolve_device_precision,
                   LipNetBaseline, DEFAULT_MANIFEST, load_manifest, split_files)
from models.baseline import ARCHITECTURE_VERSION


def verify(args: argparse.Namespace) -> dict:
    """4회 역전파 및 장치간 가중치 호환성을 실측하고 JSON을 남긴다."""
    device, use_amp = resolve_device_precision("cuda", "amp")
    document, summary = load_manifest(args.manifest)
    paths = split_files(document, args.manifest, "train")
    ds = ClipDataset(paths)
    # 명세의 순서 대신 실제 캐시 길이로 선택하되, train 소속만 사용한다.
    lengths = []
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            lengths.append((len(data["frames"]), i))
    chosen = sorted(lengths, reverse=True)[:2]
    batch = collate([ds[i] for _, i in chosen])
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = LipNetBaseline().to(device).train()
    original = model.fc.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    measurements = []
    for index in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        loss = ctc_training_loss(model, batch, device, use_amp)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError("AMP 기울기 overflow로 갱신을 건너뛰었습니다.")
        torch.cuda.synchronize(device)
        measurement = {"step": index + 1, "warmup": index == 0,
                       "loss": loss.item(), "grad_norm": norm.item(),
                       "seconds": time.perf_counter() - start}
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    if torch.equal(original, model.fc.weight):
        raise RuntimeError("가중치가 갱신되지 않았습니다.")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("갱신된 가중치에 NaN/Inf가 있습니다.")
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    checkpoint = Path(args.report).with_suffix(".pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(checkpoint), training_data={
        "mode": "smoke_only", "manifest_sha256": summary["manifest_sha256"],
        "generalization_evaluation": False,
    }, training_run={"bounded_smoke": True, "steps": 4})
    model.eval()
    with torch.no_grad():
        cuda_prediction = model(batch[0][:1].to(device), batch[2][:1]).cpu()
    loaded = LipNetBaseline.load_from_checkpoint(str(checkpoint))
    with torch.no_grad():
        cpu_prediction = loaded(batch[0][:1], batch[2][:1])
    torch.testing.assert_close(cpu_prediction, cuda_prediction, atol=2e-4, rtol=2e-4)
    result = {
        "purpose": "GPU execution smoke test; no recognition accuracy evaluation",
        "architecture_version": ARCHITECTURE_VERSION,
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()),
        "old_parameters": 115714213,
        "parameter_reduction_percent": (1 - sum(p.numel() for p in model.parameters()) / 115714213) * 100,
        "manifest_sha256": summary["manifest_sha256"], "split": "train",
        "files": [str(paths[i]) for _, i in chosen], "actual_lengths": batch[2].tolist(),
        "input_shape": list(batch[0].shape), "output_shape": [batch[0].shape[2], 2, 69],
        "precision": "AMP FP16; log-softmax/CTC FP32", "optimizer": "Adam", "steps": measurements,
        "weights_changed": True, "all_weights_finite": True,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        "mean_step_seconds_after_warmup": sum(s["seconds"] for s in measurements[1:]) / 3,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "cpu_reload_max_abs_difference": (cpu_prediction - cuda_prediction).abs().max().item(),
        "cpu_reload_passed": True, "checkpoint": str(checkpoint.resolve()),
        "additional_training_data_required_for_this_check": False,
    }
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """명세/결과 경로를 받아 검증한다. 결과 체크포인트는 .pt로 나란히 저장한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report", default="tmp/step4-gpu.json")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
