"""장치 선택과 CTC 학습 수치/실제 가중치 갱신을 검증한다."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
from train import ctc_training_loss, resolve_device_precision
from models.baseline import LipNetBaseline


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device_precision("cuda", "auto")
    assert resolve_device_precision(None, "auto") == (torch.device("cpu"), False)
    with pytest.raises(ValueError, match="AMP"):
        resolve_device_precision("cpu", "amp")


@pytest.mark.parametrize("device_name,use_amp", [
    ("cpu", False),
    pytest.param("cuda", True, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 전용")),
])
def test_ctc_backward_updates_weights_with_variable_lengths(device_name, use_amp):
    torch.manual_seed(19)
    device = torch.device(device_name)
    model = LipNetBaseline(lstm_hidden=16).to(device).train()
    before = model.fc.weight.detach().clone()
    batch = (torch.randn(2, 3, 7, 96, 96), torch.tensor([1, 20, 2, 21]),
             torch.tensor([5, 7]), torch.tensor([2, 2]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024)
    loss = ctc_training_loss(model, batch, device, use_amp)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(opt)
    scaler.update()
    assert not torch.equal(before, model.fc.weight)


def test_impossible_ctc_alignment_is_not_silently_zeroed():
    model = LipNetBaseline(lstm_hidden=8)
    batch = (torch.randn(1, 3, 1, 96, 96), torch.tensor([1, 20]),
             torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(FloatingPointError, match="CTC"):
        ctc_training_loss(model, batch, torch.device("cpu"), False)


def test_accumulation_matches_full_groups_including_partial_last_group(tmp_path, monkeypatch):
    """2+2개 누적 후 남은 1개도 갱신하며, 각 묶음의 평균 손실과 같아야 한다."""
    import train
    from types import SimpleNamespace

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.5]))

        def save_checkpoint(self, *args, **kwargs):
            self.saved_weight = self.weight.detach().clone()

    model = TinyModel()
    expected = TinyModel()
    values = torch.tensor([1., 2., 3., 4., 5.])
    reference_opt = torch.optim.Adam(expected.parameters(), lr=1e-3)
    for group in [values[:4], values[4:]]:
        reference_opt.zero_grad()
        ((expected.weight * group - 1) ** 2).mean().backward()
        reference_opt.step()
    sampler = train.LengthBucketBatchSampler
    monkeypatch.setattr(train, "training_files", lambda args: (["cache"] * 5, [], {"mode": "smoke_only"}))
    monkeypatch.setattr(train, "clip_length", lambda path: 1)
    monkeypatch.setattr(train, "ClipDataset", lambda files: values)
    monkeypatch.setattr(train, "LengthBucketBatchSampler", lambda lengths, size, shuffle: sampler(lengths, size, False))
    monkeypatch.setattr(train, "create_model", lambda architecture: model)
    monkeypatch.setattr(train, "collate", lambda items: (torch.stack(items), None, torch.ones(len(items), dtype=torch.long), None))
    monkeypatch.setattr(train, "ctc_training_loss", lambda m, b, d, a: ((m.weight * b[0] - 1) ** 2).mean())
    args = SimpleNamespace(check_data=False, max_steps=3, smoke_data="cache", out=str(tmp_path / "smoke.pt"),
                           epochs=1, batch_size=2, eval_every=1, accum_steps=2, lr=1e-3, grad_clip=0,
                           device="cpu", precision="fp32", seed=42, report=None)
    report = train.train(args)
    assert report["optimizer_steps"] == 2 and len(report["steps"]) == 3
    assert report["training_data"]["mode"] == "smoke_only"
    assert not report["training_data"]["generalization_evaluation"]
    torch.testing.assert_close(model.saved_weight, expected.weight)
