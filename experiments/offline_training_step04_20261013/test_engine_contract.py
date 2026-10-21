"""Focused negative controls for checkpoint identity, cursor and replay comparison."""
from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import engine
import run_smoke


def payload_fixture() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build a synthetic completed-update2 checkpoint, without neural execution."""
    identity={"plan_sha256":"fixed","policy_sha256":"fixed","seed":42,"arm":"B_curriculum"}
    schedule=[{"clip_ids":["a","b"],"lr":.001 if i<2 else .0003,"kind":"phrase" if i<2 else "full_sentence"} for i in range(6)]
    state={"model":{"w":torch.ones(2)},"optimizer":{"state":{0:{"step":torch.tensor(2.),"exp_avg":torch.zeros(2),"exp_avg_sq":torch.ones(2)}},"param_groups":[{"lr":.001}]},"rng":{"cpu":torch.arange(4,dtype=torch.uint8)}}
    measured=engine.hashes(state)
    history=[{"update":i+1,"clip_ids":["a","b"],"kind":"phrase","learning_rate":.001,"optimizer_updated":True,"state_hashes":measured} for i in range(2)]
    return {"schema":"offline-stage4-complete-checkpoint-v1","identity":identity,"completed_updates":2,"history":history,
            "state":state,"state_hashes":measured,"next_schedule_row":schedule[2]},identity,schedule


def test_valid_synthetic_resume_contract() -> None:
    """A complete update2 state declares the different update3 phase/LR."""
    payload,identity,schedule=payload_fixture()
    engine.validate_payload(payload,identity,schedule)


@pytest.mark.parametrize("fault",["identity","cursor","state","history","adam_step","LR","skipped_update"])
def test_corrupt_resume_contract_rejected(fault: str) -> None:
    """Reject incorrect run identity or concealed cursor/optimizer changes."""
    payload,identity,schedule=payload_fixture()
    payload=copy.deepcopy(payload)
    if fault=="identity": payload["identity"]["seed"]=43
    elif fault=="cursor": payload["next_schedule_row"]=schedule[1]
    elif fault=="state": payload["state"]["model"]["w"][0]=2.
    elif fault=="history": payload["history"][-1]["clip_ids"]=["b","a"]
    elif fault=="skipped_update": payload["history"][-1]["optimizer_updated"]=False
    else:
        if fault=="adam_step": payload["state"]["optimizer"]["state"][0]["step"].fill_(1)
        if fault=="LR": payload["state"]["optimizer"]["param_groups"][0]["lr"]=.0003
        payload["state_hashes"]=engine.hashes(payload["state"])
        payload["history"][-1]["state_hashes"]=payload["state_hashes"]
    with pytest.raises(ValueError):
        engine.validate_payload(payload,identity,schedule)


def test_main_comparison_cannot_enter_smoke_engine() -> None:
    """A future1260-update plan cannot use this bounded stage4 entry point."""
    with pytest.raises(ValueError):
        engine.verify_files({"schema":"offline-stage4-six-update-smoke-plan-v1","updates":1260})


def test_full_state_replay_checks_optimizer_and_rng() -> None:
    """A matching model alone is insufficient: changed moments/RNG must fail."""
    payload,_,_=payload_fixture()
    source=payload["state"]
    tolerances={key:{"atol":1e-7,"rtol":1e-5} for key in ("model","optimizer_exp_avg","optimizer_exp_avg_sq")}
    assert run_smoke.compare_state(source,copy.deepcopy(source),tolerances)["all_component_hashes_exact"]
    for target in ("moment","rng"):
        other=copy.deepcopy(source)
        if target=="moment": other["optimizer"]["state"][0]["exp_avg"][0]=.5
        else: other["rng"]["cpu"][0]=255
        with pytest.raises(ValueError):
            run_smoke.compare_state(source,other,tolerances)
