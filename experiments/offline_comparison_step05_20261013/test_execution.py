"""Checkpoint and actual frozen-schedule seam checks without neural execution."""
from __future__ import annotations
import copy
import pytest
import execution as e


def fixture():
    """A tiny complete CPU state with a changed next phase and LR."""
    tasks=[{"kind":"phrase","lr":.001,"clip_ids":["a","b"]},{"kind":"full_sentence","lr":.0003,"clip_ids":["c","d"]}]
    state={"model":{"w":e.torch.ones(2)},"optimizer":{"state":{0:{"step":e.torch.tensor(1.),"exp_avg":e.torch.ones(2),"exp_avg_sq":e.torch.ones(2)}},"param_groups":[{"lr":.001}]},"rng":{"cpu":e.torch.arange(3)}}
    identity={"mode":"preflight","seed":42,"arm":"B_curriculum"}
    payload={"schema":"offline-stage5-complete-checkpoint-v1","identity":identity,"state":state,"state_hashes":e.old.hashes(state),
        "completed_updates":1,"history":[{"update":1,"kind":"phrase","learning_rate":.001,"clip_ids":["a","b"],"optimizer_updated":True}],"next_task":tasks[1]}
    return payload,identity,tasks


def test_valid_resume():
    p,i,t=fixture()
    e.validate_checkpoint(p,i,t)


@pytest.mark.parametrize("fault",["identity","cursor","weights","adam_step","lr","inputs","kind","update_skipped"])
def test_wrong_checkpoint_rejected(fault):
    p,i,t=fixture();p=copy.deepcopy(p)
    if fault=="identity":p["identity"]["mode"]="main"
    elif fault=="cursor":p["next_task"]=t[0]
    elif fault=="weights":p["state"]["model"]["w"][0]=99
    elif fault=="inputs":p["history"][0]["clip_ids"]=["b","a"]
    elif fault=="kind":p["history"][0]["kind"]="full_sentence"
    elif fault=="update_skipped":p["history"][0]["optimizer_updated"]=False
    else:
        if fault=="adam_step":p["state"]["optimizer"]["state"][0]["step"].fill_(2)
        else:p["state"]["optimizer"]["param_groups"][0]["lr"]=.0003
        p["state_hashes"]=e.old.hashes(p["state"])
    with pytest.raises(ValueError):e.validate_checkpoint(p,i,t)


@pytest.mark.parametrize("seed",[42,43])
def test_actual_curriculum_and_lr_seams(seed):
    plan={"schedules":[{"seed":seed,"path":str(e.PREVIOUS/f"protocol/schedule_seed{seed}.json")} ]}
    a=e.scheduled_tasks(plan,seed,"A_sentence","main")
    b=e.scheduled_tasks(plan,seed,"B_curriculum","main")
    assert len(a)==len(b)==1260
    assert all(x["kind"]=="full_sentence" for x in a)
    assert b[419]["kind"]=="phrase" and b[420]["kind"]=="full_sentence"
    for rows in (a,b):
        assert rows[839]["lr"]==.001 and rows[840]["lr"]==.0003
        assert rows[1049]["lr"]==.0003 and rows[1050]["lr"]==.0001
    assert max(sum(x["lengths"]) for x in a)==316
