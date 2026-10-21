"""Independent scalar identities and fixture checks for saved-output diagnostics."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
spec=importlib.util.spec_from_file_location("analyzer",OUT/"analyze_saved.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks=[]
def record(name):
    checks.append(name)
def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))

# Analytical categorical distributions, including an intentionally invalid posterior.
p=np.full(69,0.5/68,dtype=np.float64)
p[0]=.5
known=module.posterior_statistics(np.stack([np.log(p)]*3))
entropy=-sum(float(x)*math.log(float(x)) for x in p)
assert abs(known["entropy_nats_sum"]-3*entropy)<1e-12
assert abs(known["blank_probability_sum"]-1.5)<1e-12
assert known["blank_argmax_frames"]==3
record("analytical_half_blank_distribution_entropy_and_blank")
uniform=module.posterior_statistics(np.full((2,69),-math.log(69)))
assert abs(uniform["entropy_nats_sum"]-2*math.log(69))<1e-12
record("analytical_uniform_entropy_log69")
try:
    module.posterior_statistics(np.zeros((2,69)))
except AssertionError:
    record("unnormalized_posterior_rejected")
else:
    raise AssertionError("Invalid posterior accepted")

d=load("loss_output_analysis.json")
for source in d["sources"]+load("posterior_sources.json"):
    path=Path(source["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source["sha256"]
record("all_1098_consumed_file_hashes_verified" if len(d["sources"])+len(load("posterior_sources.json"))==1098 else "all_consumed_file_hashes_verified")

allrows=[]
for run in d["training_runs"]:
    path=ROOT/f'experiments/offline_comparison_step05_20261013/runs/seed{run["seed"]}/{run["arm"]}/progress.jsonl'
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    allrows.extend(rows)
    for key,a,b in [("first60",0,60),("pre_transition60",360,420),("post_transition60",420,480),("penultimate60",1140,1200),("final60",1200,1260)]:
        values=[r["loss"] for r in rows[a:b]]
        assert abs(sum(values)/len(values)-run[key]["loss"]["mean"])<1e-12
    assert run["all_updates"]["updates_with_zero_gradient_tensor"]==0
assert len(allrows)==5040
assert sum(r["true_frames"] for r in allrows)==911540
assert sum(r["dense_frames"] for r in allrows)==1085994
record("scalar_loss_window_and_5040_update_frame_identities")

rows=load("posterior_row_statistics.json")
groups=load("output_groups.json")
assert len(rows)==1080 and sum(r["T"] for r in rows)==72828
assert len({(r["seed"],r["arm"],r["update"],r["clip_id"],r["condition"]) for r in rows})==1080
assert len(groups)==144 and sum(g["clips"] for g in groups)==1080 and sum(g["frames"] for g in groups)==72828
assert all(g["clips"]==(12 if g["split"]=="train" else 3) for g in groups)
for g in groups:
    subset=[r for r in rows if all(r[k]==g[k] for k in ("seed","arm","update","split","kind","condition"))]
    assert abs(g["CER_percent"]-100*sum(r["character_edits"] for r in subset)/sum(r["reference_characters"] for r in subset))<1e-12
    assert abs(g["mean_entropy_nats_per_frame"]-sum(r["entropy_nats_sum"] for r in subset)/sum(r["T"] for r in subset))<1e-12
    assert abs(g["blank_argmax_frame_percent"]-100*sum(r["blank_argmax_frames"] for r in subset)/sum(r["T"] for r in subset))<1e-12
record("all_144_groups_scalar_weighted_aggregation_and_unique_rows")

# Check each plot point against source tables; no chart-derived numeric reading.
points=load("plot_points_r2.json")["points"]
windows=load("loss_windows.json")
for p in points:
    if p["panel"]=="training_loss":
        src=next(r for r in windows if all(r[k]==p[k] for k in ("seed","arm","first_update","last_update")))
        assert src["loss"]["mean"]==p["value"]
    else:
        src=next(r for r in groups if all(r[k]==p[k] for k in ("seed","arm","update")) and r["split"]=="train" and r["kind"]=="phrase" and r["condition"]=="normal")
        field={"phrase_cer":"CER_percent","blank_share":"blank_argmax_frame_percent","token_length_ratio":"predicted_to_reference_token_ratio"}[p["panel"]]
        assert src[field]==p["value"]
assert len(points)==102
record("all_102_plot_points_match_json_sources")
result={"status":"passed","checks":checks,"check_count":len(checks),"new_model_calls":0,"source_mutations":0,"producer_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (OUT/"verification_r2.json").open("x",encoding="utf-8") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result))
