"""Independently compare new update420 numerical state with frozen stage5 B."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import torch
from audit_common import unpack, state_hashes
from audit_sources import binding, check_binding, read, S5

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def exact_tree(left,right):
    assert type(left)==type(right),(type(left),type(right))
    if isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
        return 1
    if isinstance(left,np.ndarray):
        assert left.dtype==right.dtype and left.shape==right.shape and np.array_equal(left,right)
        return 1
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        return sum(exact_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        return sum(exact_tree(a,b) for a,b in zip(left,right))
    assert left==right,(left,right)
    return 0

def inspect(checkpoint,seed):
    torch.set_num_threads(4)
    checkpoint=Path(checkpoint).resolve()
    assert checkpoint.is_relative_to(BASE) and checkpoint.suffix==".pt"
    prior_result=read(S5/f"runs/seed{seed}/B_curriculum/segment_result.json")
    candidates=[]
    for ref in prior_result["checkpoints"]:
        if "0420" in Path(ref["path"]).stem:
            candidates.append(ref)
    assert len(candidates)==1,candidates
    priorref=candidates[0];check_binding(priorref)
    current=unpack(torch.load(checkpoint,map_location="cpu",weights_only=True))
    prior=unpack(torch.load(priorref["path"],map_location="cpu",weights_only=True))
    assert current["completed_updates"]==prior["completed_updates"]==420
    assert current["identity"]["seed"]==prior["identity"]["seed"]==seed
    assert current["identity"]["runtime"]==prior["identity"]["runtime"]
    assert current["initial_state_hashes"]==prior["initial_state_hashes"]
    curhash,oldhash=state_hashes(current["state"]),state_hashes(prior["state"])
    assert curhash==current["state_hashes"] and oldhash==prior["state_hashes"] and curhash==oldhash
    components={key:exact_tree(current["state"][key],prior["state"][key]) for key in ["model","optimizer","rng"]}
    fields=["update","kind","clip_ids","learning_rate","lengths","true_frames","dense_frames","padding_frames","target_tokens","loss","gradient_norm_before_clip","gradient_norm_after_clip","optimizer_updated","nonzero_gradient_tensors","parameter_tensors"]
    assert len(current["history"])==len(prior["history"])==420
    for left,right in zip(current["history"],prior["history"]):
        for field in fields:
            assert left[field]==right[field],(left["update"],field,left[field],right[field])
    return {"schema":"offline-stage5c-independent-update420-replay-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","seed":seed,"completed_updates":420,"checkpoint":binding(checkpoint),"previous_checkpoint":priorref,"runtime_exact":True,"fresh_initial_state_exact":True,"numerical_state_bitwise_exact":True,"direct_tensor_array_comparisons":components,"state_hashes":curhash,"history_updates_compared":420,"history_fields_compared":fields,"producer_imports":0,"new_network_forwards":0,"optimizer_updates_by_audit":0,"CPU_checkpoint_tensor_loads":2}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",required=True);parser.add_argument("--seed",type=int,required=True,choices=[42,43]);args=parser.parse_args()
    output=HERE/f"replay420_seed{args.seed}.json";assert not output.exists()
    result=inspect(args.checkpoint,args.seed);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","seed","numerical_state_bitwise_exact","direct_tensor_array_comparisons"]}))
