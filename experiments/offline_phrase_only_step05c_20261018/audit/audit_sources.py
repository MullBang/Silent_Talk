"""Independent fixed-source and phrase-only schedule audit; no producer imports."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S3 = ROOT / "experiments/offline_dataset_step03_20261013"
S4 = ROOT / "experiments/offline_training_step04_20261013"
S5 = ROOT / "experiments/offline_comparison_step05_20261013"
S5B = ROOT / "experiments/offline_diagnostic_step05b_20261018"

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def binding(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}

def check_binding(row):
    path = Path(row["path"])
    if row.get("exists") is False:
        assert not path.exists(), str(path)
        return
    assert path.is_file(), str(path)
    assert digest(path) == row["sha256"], str(path)
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"], str(path)

def inspect():
    manifests = []
    for folder, count in [(S3,114),(S4,101),(S5,1238),(S5B,45)]:
        path = folder / "results/completion_manifest.json"
        if folder == S5B:
            assert digest(path) == "845dfd0f0dc93d2626729f779cc00c4c50f0a310370c45338687798e48ea8d12"
        entries = read(path)["artifacts"]
        assert len(entries) == len({r["path"] for r in entries}) == count
        for row in entries:
            check_binding(row)
        manifests.append({**binding(path),"verified_artifacts":count})
    scopepath = S3 / "results/scope_snapshot.json"
    scope = read(scopepath)["artifacts"]
    assert len(scope) == 10
    for row in scope:
        check_binding(row)
    cachepath = S3 / "results/cache_handoff.json"
    records = read(cachepath)["records"]
    assert len(records)==30
    assert Counter(r["split"] for r in records)=={"train":24,"dev":6}
    assert all(r["provisional"] and not r["training_ready"] for r in records)
    assert sum(r["boundary_followup_flag"] for r in records)==2
    lookup={r["clip_id"]:r for r in records if r["split"]=="train" and r["kind"]=="phrase"}
    assert len(lookup)==12 and len({r["text"] for r in lookup.values()})==11
    assert len({r["speaker_id"] for r in lookup.values()})==8
    assert sum(r["T"] for r in lookup.values())==481
    assert not any("C293_A_012_019" in r["parent_id"] for r in records)
    designpath = S5B / "design/proposed_phrase_only_design.json"
    design=read(designpath)
    for r in design["sources"]+design["schedule_bindings"]:
        check_binding(r)
    assert {r["clip_id"] for r in design["train_records"]}==set(lookup)
    for r in design["train_records"]:
        for key in ["parent_id","rank","speaker_id","T","label_length","ctc_min_frames","text","provisional","training_ready","boundary_followup_flag"]:
            assert r[key]==lookup[r["clip_id"]][key], (r["clip_id"],key)
    parents=sorted(r["parent_id"] for r in lookup.values())
    totals=Counter()
    schedules=[]
    for seed in [42,43]:
        schedulepath=S5B/f"design/proposed_phrase_schedule_seed{seed}.json"
        schedule=read(schedulepath)
        assert schedule["seed"]==seed and len(schedule["updates"])==1680
        old=read(S4/f"protocol/schedule_seed{seed}.json")["updates"]
        rng=random.Random(seed)
        independent=[]
        for _ in range(210):
            order=parents.copy();rng.shuffle(order)
            independent.extend(order[i:i+2] for i in range(0,12,2))
        pairs=independent+independent[:420]
        counts=Counter();sums=Counter();maxtrue=maxdense=0
        for n,(row,pair) in enumerate(zip(schedule["updates"],pairs),1):
            source_n=(n-1)%1260+1
            assert row["parent_ids"]==pair==old[source_n-1]["parent_ids"]
            assert row["update"]==n and row["kind"]=="phrase"
            assert row["source_schedule_update"]==source_n
            assert row["source_schedule_replayed_for_order_only"]==(n>1260)
            ids=[p+"__phrase" for p in pair]
            assert row["clip_ids"]==ids
            picked=[lookup[c] for c in ids]
            lengths=[r["T"] for r in picked]
            labels=[r["label_length"] for r in picked]
            true,dense=sum(lengths),2*max(lengths)
            assert row["lengths"]==lengths and row["target_lengths"]==labels
            assert row["true_frames"]==true and row["dense_frames"]==dense
            assert row["lr"]==(.001 if n<=840 else .0003 if n<=1050 else .0001)
            if n<=420:
                prior=old[n-1]["arms"]["B_curriculum"]
                assert prior["clip_ids"]==ids and old[n-1]["lr"]==row["lr"]
            counts.update(pair);sums.update(updates=1,true_frames=true,dense_frames=dense,target_tokens=sum(labels))
            maxtrue,maxdense=max(maxtrue,true),max(maxdense,dense)
        assert len(counts)==12 and set(counts.values())=={280}
        expected={"seed":seed,"logical_updates":1680,"parent_visits_each":280,"true_frames":134680,"dense_frames":158332 if seed==42 else 157614,"target_tokens":76160,"max_true_batch_frames":114,"max_dense_batch_frames":134}
        assert schedule["accounting"]==expected
        assert (sums["updates"],sums["true_frames"],sums["dense_frames"],sums["target_tokens"],maxtrue,maxdense)==(1680,134680,expected["dense_frames"],76160,114,134)
        totals.update(sums)
        schedules.append({"schedule":binding(schedulepath),**expected,"first420_prior_B_order_lr_exact":True})
    assert totals=={"updates":3360,"true_frames":269360,"dense_frames":315946,"target_tokens":152320}
    oldeval=read(S4/"protocol/evaluation_plan.json")
    items=[r for r in oldeval["items"] if r["clip_id"] in lookup]
    assert len(items)==12
    for item in items:
        assert item["split"]=="train" and item["kind"]=="phrase"
        assert item["static_frame_index"]==item["T"]//2
        expected=list(range(item["T"]));random.Random(item["shuffle_seed"]).shuffle(expected)
        assert item["shuffled_frame_indices"]==expected
    assert design["evaluation"]["total_planned_forwards"]==144
    return {"schema":"offline-stage5c-independent-source-audit-v1","created_utc":datetime.now(timezone.utc).isoformat(),"status":"passed","producer_imports":0,"network_forward_calls":0,"optimizer_updates":0,"checkpoint_tensor_loads":0,"frozen_manifests":manifests,"verified_frozen_artifact_entries":1498,"scope":{"binding":binding(scopepath),"verified_artifacts":10},"cache_handoff":binding(cachepath),"design":binding(designpath),"train_phrase_clips":12,"unique_phrase_texts":11,"train_speakers":8,"train_phrase_frames":481,"source_flags_preserved":True,"excluded_OOV_parent_preserved":True,"schedules":schedules,"planned_main_totals":dict(totals),"planned_evaluation_rows":144,"planned_posterior_frames":481*12,"planned_preflight_update_cap":8,"planned_main_checkpoints":8}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="source_preflight.json");args=parser.parse_args()
    output=(HERE/args.output).resolve();assert output.parent==HERE and not output.exists()
    result=inspect();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["status","verified_frozen_artifact_entries","train_phrase_clips","planned_main_totals","planned_evaluation_rows"]}))
