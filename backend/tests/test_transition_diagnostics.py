"""Train-only selection, exposure plans and gradient/update math on tiny CPU arrays."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from transition_diagnostics import (  # noqa: E402
    gradient_metrics, make_training_schedule, parameter_groups,
    parameter_update_metrics, select_transition_cohorts, snapshot_parameters,
)


def records() -> list[dict]:
    return [{"id": f"clip_{speaker}_{bucket}_{index:02d}", "speaker_id": speaker,
             "split": "train", "frames": bucket*50+index+1}
            for speaker in ("a", "b", "c") for bucket in range(4) for index in range(8)]


def test_selection_preserves_short16_and_balances_available_cells_without_leakage():
    data = records()
    result = select_transition_cohorts(data)
    expected = [row["id"] for row in sorted(data, key=lambda row: (row["frames"],row["id"]))[:16]]
    assert result["short16"] == expected
    assert len(result["additional48"]) == len(set(result["additional48"])) == 48
    assert not set(result["short16"]) & set(result["additional48"])
    assert result["expanded64"] == expected + result["additional48"]
    counts = [cell["selected"] for cell in result["strata"] if cell["available_excluding_short16"] >= 5]
    assert max(counts)-min(counts) <= 1
    assert sum(cell["selected"] for cell in result["strata"]) == 48
    for cell in result["strata"]:
        pool = [row["id"] for row in data if row["id"] not in expected
                and row["speaker_id"] == cell["speaker"]
                and (row["frames"]-1)//50 == (int(cell["length_bin"][:3])-1)//50]
        ordered = sorted(pool, key=lambda x:(hashlib.sha256(f"step12-selection-42:{x}".encode()).hexdigest(),x))
        assert cell["selected_ids"] == ordered[:cell["selected"]]
    assert result["distributions"]["expanded64"]["clips"] == 64
    assert sum(result["distributions"]["additional48"]["speaker"].values()) == 48


def test_selection_ignores_record_order_validation_test_and_all_performance_fields():
    original = records()
    expected = select_transition_cohorts(original)
    contaminated = copy.deepcopy(original)
    for index, row in enumerate(contaminated):
        row.update(cer=index/100, loss=999-index, text="unused", prediction="unused", label=[index])
    contaminated += [{"id": "val_super_short", "split": "val", "frames": 0, "speaker_id": "z"},
                     {"id": "test_super_short", "split": "test", "frames": -1, "speaker_id": "z"}]
    random.Random(731).shuffle(contaminated)
    assert select_transition_cohorts(contaminated) == expected
    assert original == records()
    random.seed(987)
    state = random.getstate()
    select_transition_cohorts(original)
    assert random.getstate() == state
    assert select_transition_cohorts(original,seed=137)["additional48"] != expected["additional48"]


@pytest.mark.parametrize("corruption", ["insufficient", "duplicate", "outside", "frames", "speaker"])
def test_selection_rejects_broken_train_identity_and_bounds(corruption):
    data = records()
    if corruption == "insufficient": data = data[:63]
    elif corruption == "duplicate": data.append(copy.deepcopy(data[0]))
    elif corruption == "outside": data.append({"id": data[0]["id"], "split": "test"})
    elif corruption == "frames": data[0]["frames"] = 201
    else: data[0]["speaker_id"] = ""
    with pytest.raises(ValueError):
        select_transition_cohorts(data)


@pytest.mark.parametrize("size,epochs", [(16,200),(64,50)])
def test_default_schedule_visits_every_sample_once_each_epoch(size,epochs):
    ids = [f"sample_{index:02d}" for index in range(size)]
    plan = make_training_schedule(ids)
    assert len(plan["batches"]) == 1600
    assert plan["completed_epochs"] == epochs and plan["partial_final_epoch_batches"] == 0
    assert plan["total_presentations"] == 3200
    assert set(plan["exposures"].values()) == {epochs}
    grouped = defaultdict(list)
    for expected_attempt, batch in enumerate(plan["batches"],start=1):
        assert batch["attempt"] == expected_attempt
        assert len(batch["indices"]) == 2 and len(set(batch["indices"])) == 2
        assert batch["ids"] == [ids[i] for i in batch["indices"]]
        grouped[batch["epoch"]].extend(batch["ids"])
    assert len(grouped) == epochs
    assert all(Counter(epoch_ids) == Counter(ids) for epoch_ids in grouped.values())
    assert grouped[1] != grouped[2]
    assert len(plan["batches_sha256"]) == 64


def test_schedule_is_independent_of_global_python_and_torch_rng_and_call_order():
    ids = [str(index) for index in range(16)]
    torch_state = torch.random.get_rng_state().clone()
    random.seed(22)
    python_state = random.getstate()
    first = make_training_schedule(ids,attempt_updates=10)
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(),torch_state)
    make_training_schedule(ids,attempt_updates=25,seed=13)
    assert make_training_schedule(ids,attempt_updates=10) == first
    assert make_training_schedule(ids,attempt_updates=10,seed=91)["batches"] != first["batches"]
    assert first["completed_epochs"] == 1 and first["partial_final_epoch_batches"] == 2
    assert sorted(first["exposures"].values()) == [1]*12+[2]*4


@pytest.mark.parametrize("ids,attempts,batch", [(["a","a"],1,2),(["a","b","c"],1,2),
                                               (["a","b"],0,2),(["a","b"],1,0),([],1,2)])
def test_schedule_rejects_duplicate_or_incomplete_batches(ids,attempts,batch):
    with pytest.raises(ValueError):
        make_training_schedule(ids,attempt_updates=attempts,batch_size=batch)


class TinyGroups(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("conv","projection","lstm","fc"):
            setattr(self,name,torch.nn.Linear(2,1,bias=False,dtype=torch.float64))
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[3.,4.]]))
            self.projection.weight.zero_()
            self.lstm.weight.zero_()
            self.fc.weight.copy_(torch.tensor([[1.,-1.]]))


def gradients(model):
    model.conv.weight.grad = torch.tensor([[3.,4.]],dtype=torch.float64)
    model.projection.weight.grad = torch.zeros((1,2),dtype=torch.float64)
    model.fc.weight.grad = torch.tensor([[0.,12.]],dtype=torch.float64)


def test_parameter_group_partition_rejects_missing_unknown_or_shared_parameters():
    model = TinyGroups()
    groups = parameter_groups(model)
    assert list(groups) == ["cnn","projection","lstm","output"]
    assert {id(p) for pairs in groups.values() for _,p in pairs} == {id(p) for p in model.parameters()}
    model.extra = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError,match="Unclassified"):
        parameter_groups(model)
    model.extra.requires_grad_(False)
    assert sum(len(x) for x in parameter_groups(model).values()) == 4
    model.conv.alias = model.fc.weight
    with pytest.raises(ValueError,match="Shared"):
        parameter_groups(model)
    del model.conv.alias
    model.lstm.weight.requires_grad_(False)
    with pytest.raises(ValueError,match="four"):
        parameter_groups(model)


def test_gradient_norm_known_three_four_twelve_and_missing_versus_zero():
    model=TinyGroups(); gradients(model); groups=parameter_groups(model)
    before={name:None if p.grad is None else p.grad.clone() for name,p in model.named_parameters()}
    result=gradient_metrics(groups)
    assert result["groups"]["cnn"]["l2_norm"] == 5
    assert result["groups"]["projection"]["l2_norm"] == 0
    assert result["groups"]["projection"]["zero_grad_tensors"] == 1
    assert result["groups"]["lstm"]["missing_grad_tensors"] == 1
    assert result["groups"]["lstm"]["zero_grad_tensors"] == 0
    assert result["total"]["l2_norm"] == 13
    assert result["total"]["zero_grad_elements"] == 3
    assert result["total"]["present_grad_elements"] == 6
    for name,p in model.named_parameters():
        if before[name] is None: assert p.grad is None
        else: torch.testing.assert_close(p.grad,before[name],rtol=0,atol=0)
    json.dumps(result,allow_nan=False)


def test_postclip_norm_is_measured_separately_without_parameter_mutation():
    model=TinyGroups(); gradients(model); groups=parameter_groups(model)
    before=snapshot_parameters(groups)
    pre=gradient_metrics(groups)
    returned=torch.nn.utils.clip_grad_norm_(model.parameters(),6.5)
    post=gradient_metrics(groups)
    assert returned.item() == 13 and pre["total"]["l2_norm"] == 13
    assert post["total"]["l2_norm"] == pytest.approx(13*6.5/(13+1e-6))
    assert post["groups"]["cnn"]["l2_norm"] == pytest.approx(5*6.5/(13+1e-6))
    assert parameter_update_metrics(groups,before)["total"]["absolute_l2"] == 0


def test_nonfinite_gradients_report_null_norm_and_explicit_counts_with_valid_json():
    model=TinyGroups(); gradients(model); groups=parameter_groups(model)
    model.conv.weight.grad.copy_(torch.tensor([[float('nan'),float('inf')]]))
    result=gradient_metrics(groups)
    assert result["total"]["l2_norm"] is None
    assert result["total"]["finite_l2_norm"] == 12
    assert result["total"]["nonfinite_grad_tensors"] == 1
    assert result["total"]["nonfinite_grad_elements"] == 2
    assert not result["total"]["all_present_gradients_finite"]
    assert torch.isnan(model.conv.weight.grad[0,0])
    json.dumps(result,allow_nan=False)


def test_all_missing_gradients_do_not_masquerade_as_present_zero_gradients():
    result=gradient_metrics(parameter_groups(TinyGroups()))
    assert result["total"]["l2_norm"] == 0
    assert result["total"]["missing_grad_tensors"] == 4
    assert result["total"]["present_grad_tensors"] == result["total"]["zero_grad_tensors"] == 0


def test_real_parameter_delta_uses_snapshot_not_gradient_and_zero_denominator_is_explicit():
    model=TinyGroups(); gradients(model); groups=parameter_groups(model)
    before=snapshot_parameters(groups)
    with torch.no_grad():
        model.conv.weight.add_(torch.tensor([[0.,12.]]))
        model.projection.weight.add_(torch.tensor([[3.,4.]]))
    result=parameter_update_metrics(groups,before)
    assert result["groups"]["cnn"]["absolute_l2"] == 12
    assert result["groups"]["cnn"]["relative_l2"] == 12/5
    assert result["groups"]["projection"]["absolute_l2"] == 5
    assert result["groups"]["projection"]["relative_l2"] == 5e12
    assert result["total"]["absolute_l2"] == 13
    assert result["total"]["relative_l2"] == pytest.approx(13/(27**.5))
    assert result["total"]["changed_finite_elements"] == 3
    assert before["conv.weight"][0,1].item() == 4
    assert not before["conv.weight"].requires_grad
    json.dumps(result,allow_nan=False)


def test_nonfinite_parameter_updates_do_not_hide_corrupt_values():
    model=TinyGroups(); groups=parameter_groups(model); before=snapshot_parameters(groups)
    with torch.no_grad(): model.fc.weight[0,0]=float('nan')
    result=parameter_update_metrics(groups,before)
    assert result["total"]["absolute_l2"] is None and result["total"]["relative_l2"] is None
    assert result["total"]["after_nonfinite_elements"] == 1
    assert result["total"]["before_nonfinite_elements"] == 0
    json.dumps(result,allow_nan=False)


def test_snapshot_contract_rejects_alias_and_missing_parameter():
    model=TinyGroups(); groups=parameter_groups(model)
    aliased={name:p.detach() for name,p in model.named_parameters()}
    with pytest.raises(ValueError,match="aliases"):
        parameter_update_metrics(groups,aliased)
    before=snapshot_parameters(groups);before.pop("conv.weight")
    with pytest.raises(ValueError,match="names"):
        parameter_update_metrics(groups,before)
