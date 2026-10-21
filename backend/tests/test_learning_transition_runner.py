"""Stage12 runner checks on fake evaluation posteriors and synthetic CPU/CUDA data.

No project cache or checkpoint is read. The CUDA equivalence test uses one real
compact-model optimizer attempt per copy, never the experiment's data or weights.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import diagnose_learning_transition as runner  # noqa: E402
import temporal_diagnostics  # noqa: E402
from models.baseline import LipNetBaseline  # noqa: E402
from transition_diagnostics import parameter_groups  # noqa: E402


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    assert torch.equal(actual[2], expected[2])
    assert len(actual[3]) == len(expected[3])
    assert all(torch.equal(a,b) for a,b in zip(actual[3],expected[3]))


def consume_rng():
    random.random()
    np.random.random(9)
    torch.rand(9)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.rand(9,device=f"cuda:{device}")


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("raise_error", [False, True])
def test_preserve_rng_and_mode_on_normal_and_exception_paths(training,raise_error):
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.5)).train(training)
    before=rng_state()
    def action():
        with runner.preserve_rng_and_mode(model):
            assert not model.training and all(not child.training for child in model.children())
            consume_rng()
            if raise_error:
                raise RuntimeError("synthetic evaluation failure")
    if raise_error:
        with pytest.raises(RuntimeError,match="synthetic evaluation"):
            action()
    else:
        action()
    assert model.training is training
    assert all(child.training is training for child in model.children())
    assert_rng_equal(rng_state(),before)


@pytest.mark.parametrize("split", ["val","test"])
def test_nontrain_cache_is_rejected_before_hash_loader_or_npz_access(monkeypatch,split):
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(args)
        raise AssertionError("Filesystem access before split guard")
    monkeypatch.setattr(runner,"sha256_file",forbidden)
    monkeypatch.setattr(runner,"load_clip",forbidden)
    monkeypatch.setattr(runner.np,"load",forbidden)
    # Deliberately omit cache_path: even trying to construct its path is too late.
    with pytest.raises(ValueError,match="Only train"):
        runner.load_train_cache({"split":split})
    assert calls == []


def synthetic_evaluation():
    ids=["old_synthetic","additional_synthetic"]
    records={sample_id:{"id":sample_id,"split":"train","speaker_id":"fake",
                       "text":"간","source_text":"간","frames":12} for sample_id in ids}
    cache={sample_id:{"frames":np.arange(36,dtype=np.uint8).reshape(12,1,1,3),
                     "label":np.asarray([1,20,44],dtype=np.int64)} for sample_id in ids}
    logits=np.full((12,69),-12.,dtype=np.float64)
    path=[0,1,0,20,0,44,0,0,0,0,0,0]
    logits[np.arange(12),path]=4.
    lp=(logits-np.log(np.exp(logits).sum(axis=1,keepdims=True))).astype(np.float32)
    plan={"cohorts":{"expanded64":ids,"short16":ids[:1]}}
    return plan,cache,records,lp


@pytest.mark.parametrize("step,expected_rows", [(1,2),(100,6)])
def test_intermediate_evaluation_does_not_run_or_fabricate_beam(monkeypatch,tmp_path,step,expected_rows):
    plan,cache,records,lp=synthetic_evaluation()
    model=torch.nn.Linear(1,1).train()
    def forward(model_arg,frames,capture):
        assert model_arg is model and not model.training and capture is False
        assert not torch.is_grad_enabled()
        consume_rng()
        return lp.copy(),{}
    def forbidden_beam(*args,**kwargs):
        raise AssertionError("Intermediate evaluation must not perform beam search")
    monkeypatch.setattr(runner,"forward",forward)
    monkeypatch.setattr(runner,"ctc_beam_search_decode",forbidden_beam)
    monkeypatch.setattr(temporal_diagnostics,"ctc_beam_search_decode",forbidden_beam)
    before=rng_state()
    folder=tmp_path/f"step_{step:04d}"
    summary=runner.evaluate_snapshot(model,plan,cache,records,step,folder,"synthetic-checkpoint")
    assert summary["beam_performed"] is False and summary["rows"] == expected_rows
    rows=[json.loads(line) for line in (folder/"rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==expected_rows
    for row in rows:
        assert row["beam"] is None
        assert not any(name.startswith("beam") for name in row["ctc"])
        assert row["greedy"]["prediction"] == "간"
        assert row["ctc"]["greedy_collapsed_token_ids"] == [1,20,44]
    assert model.training
    assert_rng_equal(rng_state(),before)
    with np.load(folder/"posteriors.npz",allow_pickle=False) as arrays:
        assert len(arrays.files)==expected_rows


def test_failed_evaluation_restores_training_mode_and_rng(monkeypatch,tmp_path):
    plan,cache,records,_=synthetic_evaluation()
    model=torch.nn.Linear(1,1).train()
    def failing_forward(*args,**kwargs):
        consume_rng()
        raise RuntimeError("fake forward failed")
    monkeypatch.setattr(runner,"forward",failing_forward)
    before=rng_state()
    with pytest.raises(RuntimeError,match="fake forward"):
        runner.evaluate_snapshot(model,plan,cache,records,1,tmp_path/"failed","fake")
    assert model.training
    assert_rng_equal(rng_state(),before)


class TinyGroups(torch.nn.Module):
    """Four scalar parameter groups; no forward or dataset involved."""
    def __init__(self):
        super().__init__()
        for name in ("conv","projection","lstm","fc"):
            module=torch.nn.Module()
            module.weight=torch.nn.Parameter(torch.ones(1))
            setattr(self,name,module)


@pytest.mark.parametrize("inject_inf", [False,True])
def test_optimizer_attempt_cpu_scaler_finite_and_skip_telemetry(monkeypatch,inject_inf):
    model=TinyGroups()
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    scaler=torch.amp.GradScaler("cpu",init_scale=1024)
    def synthetic_loss(m,batch,device,use_amp):
        assert m is model and device.type=="cpu" and use_amp is True
        return sum(p.square().sum() for p in m.parameters())
    monkeypatch.setattr(runner,"ctc_training_loss",synthetic_loss)
    handle=model.fc.weight.register_hook(lambda gradient: torch.full_like(gradient,float("inf"))) if inject_inf else None
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    try:
        report=runner.optimizer_attempt(model,optimizer,scaler,None,parameter_groups(model),torch.device("cpu"))
    finally:
        if handle is not None: handle.remove()
    assert report["loss"]==4
    assert report["skipped"] is inject_inf
    assert report["clip_applied"] is not inject_inf
    assert report["preclip"]["total"]["missing_grad_tensors"]==0
    if inject_inf:
        assert report["preclip"]["total"]["l2_norm"] is None
        assert report["preclip"]["total"]["nonfinite_grad_elements"]==1
        assert report["parameter_update"]["total"]["absolute_l2"]==0
        assert report["scale_after"]==report["scale_before"]/2==512
        assert not optimizer.state
        assert all(torch.equal(p,before[name]) for name,p in model.named_parameters())
    else:
        assert report["preclip"]["total"]["l2_norm"]==4
        assert report["postclip"]["total"]["l2_norm"]==pytest.approx(4/(4+1e-6))
        assert report["parameter_update"]["total"]["absolute_l2"]>0
        assert report["scale_after"]==report["scale_before"]==1024
    json.dumps(report,allow_nan=False)


def synthetic_training_batch():
    rng=np.random.default_rng(125)
    return runner.collate([
        {"frames":rng.integers(0,256,size=(length,96,96,3),dtype=np.uint8),
         "label":np.asarray(tokens,dtype=np.int64)}
        for length,tokens in [(12,[1,20,44]),(15,[3,24,48])]])


def assert_nested_exact(actual,expected):
    if isinstance(actual,torch.Tensor):
        assert torch.equal(actual,expected)
    elif isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual: assert_nested_exact(actual[key],expected[key])
    elif isinstance(actual,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): assert_nested_exact(a,b)
    else:
        assert actual==expected


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA compact AMP equivalence requires GPU")
def test_cuda_instrumentation_is_exactly_equal_to_plain_compact_amp_update():
    torch.set_num_threads(2)
    device=torch.device("cuda")
    runner.seed_training(341)
    plain=LipNetBaseline().to(device).train()
    measured=copy.deepcopy(plain)
    optimizer_plain=torch.optim.Adam(plain.parameters(),lr=.0003)
    optimizer_measured=torch.optim.Adam(measured.parameters(),lr=.0003)
    scaler_plain=torch.amp.GradScaler("cuda",init_scale=1024)
    scaler_measured=torch.amp.GradScaler("cuda",init_scale=1024)
    batch=synthetic_training_batch()
    assert batch[0].shape==(2,3,15,96,96)
    assert batch[2].tolist()==[12,15] and batch[3].tolist()==[3,3]
    runner.seed_training(342)
    optimizer_plain.zero_grad(set_to_none=True)
    loss=runner.ctc_training_loss(plain,batch,device,use_amp=True)
    scaler_plain.scale(loss).backward()
    scaler_plain.unscale_(optimizer_plain)
    norm=torch.nn.utils.clip_grad_norm_(plain.parameters(),1.,error_if_nonfinite=True)
    scaler_plain.step(optimizer_plain)
    scaler_plain.update()
    torch.cuda.synchronize()
    after_plain_rng=rng_state()
    runner.seed_training(342)
    report=runner.optimizer_attempt(measured,optimizer_measured,scaler_measured,batch,
                                    parameter_groups(measured),device)
    assert report["loss"]==loss.item()
    assert report["skipped"] is False
    assert report["scale_before"]==report["scale_after"]==1024
    assert report["clip_coefficient"]==min(1.,1./(norm.item()+1e-6))
    assert report["parameter_update"]["total"]["absolute_l2"]>0
    assert_nested_exact(measured.state_dict(),plain.state_dict())
    assert_nested_exact(optimizer_measured.state_dict(),optimizer_plain.state_dict())
    assert scaler_measured.state_dict()==scaler_plain.state_dict()
    assert_rng_equal(rng_state(),after_plain_rng)
    json.dumps(report,allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA compact AMP skip requires GPU")
def test_cuda_inf_gradient_skips_optimizer_and_preserves_every_parameter():
    torch.set_num_threads(2)
    runner.seed_training(443)
    model=LipNetBaseline().to("cuda").train()
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    handle=model.fc.weight.register_hook(lambda gradient: torch.full_like(gradient,float("inf")))
    try:
        report=runner.optimizer_attempt(model,optimizer,scaler,synthetic_training_batch(),
                                        parameter_groups(model),torch.device("cuda"))
    finally:
        handle.remove()
    assert report["skipped"] is True and report["clip_applied"] is False
    assert report["clip_coefficient"] is None
    assert report["preclip"]["total"]["l2_norm"] is None
    assert report["preclip"]["groups"]["output"]["nonfinite_grad_elements"]==model.fc.weight.numel()
    assert report["parameter_update"]["total"]["absolute_l2"]==0
    assert report["parameter_update"]["total"]["changed_finite_elements"]==0
    assert report["scale_before"]==1024 and report["scale_after"]==512
    assert not optimizer.state
    assert all(torch.equal(p,before[name]) for name,p in model.named_parameters())
    json.dumps(report,allow_nan=False)
