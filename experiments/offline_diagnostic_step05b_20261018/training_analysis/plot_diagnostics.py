"""Plot fixed saved-log diagnostics without importing any training/model code."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
def load(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))
def ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
windows=load("loss_windows.json")
groups=load("output_groups.json")
selected=[r for r in groups if r["arm"]=="B_curriculum" and r["split"]=="train" and r["kind"]=="phrase" and r["condition"]=="normal"]
fig,axes=plt.subplots(2,2,figsize=(14,8.8),layout="constrained")
colors={"A_sentence":"#1f77b4","B_curriculum":"#dd6b20"}
styles={42:"-",43:"--"}
points=[]
for seed in (42,43):
    for arm in ("A_sentence","B_curriculum"):
        rows=[r for r in windows if r["seed"]==seed and r["arm"]==arm]
        x=[r["last_update"] for r in rows]
        y=[r["loss"]["mean"] for r in rows]
        axes[0,0].plot(x,y,styles[seed],color=colors[arm],label=f'{arm[0]} / seed {seed}',linewidth=2)
        points.extend({"panel":"training_loss","seed":seed,"arm":arm,"first_update":r["first_update"],"last_update":r["last_update"],"value":r["loss"]["mean"]} for r in rows)
axes[0,0].axvline(420,color="#777777",linewidth=1)
axes[0,0].axvline(840,color="#aaaaaa",linewidth=1,linestyle=":")
axes[0,0].axvline(1050,color="#aaaaaa",linewidth=1,linestyle=":")
axes[0,0].set(title="Training-mode CTC loss (60-update means)",xlabel="Last update in each 60-update window",ylabel="Target-length-normalized CTC loss")
axes[0,0].legend(fontsize=9,ncol=2)
for seed,color in [(42,"#9445a0"),(43,"#168e70")]:
    rows=[r for r in selected if r["seed"]==seed]
    x=[r["update"] for r in rows]
    for ax,field,panel in [(axes[0,1],"CER_percent","phrase_cer"),(axes[1,0],"blank_argmax_frame_percent","blank_share"),(axes[1,1],"predicted_to_reference_token_ratio","token_length_ratio")]:
        y=[r[field] for r in rows]
        ax.plot(x,y,styles[seed],color=color,marker="o",linewidth=2,label=f'B / seed {seed}')
        ax.set_xticks([420,840,1260])
        ax.set_xlabel("Fixed evaluation update")
        points.extend({"panel":panel,"seed":seed,"arm":"B_curriculum","update":r["update"],"value":r[field],"clips":r["clips"],"frames":r["frames"],"reference_characters":r["reference_characters"],"reference_tokens":r["reference_tokens"]} for r in rows)
axes[0,1].set(title="B: same 12 training phrases, eval mode",ylabel="Corpus character error rate (%)",ylim=(50,70))
axes[0,1].legend(["Seed 42: exact 1 / 0 / 0 of 12", "Seed 43: exact 0 / 0 / 0 of 12"],fontsize=9,loc="upper right")
axes[1,0].set(title="B: blank argmax share on training phrases",ylabel="Blank argmax frames / all frames (%)",ylim=(0,70))
axes[1,0].legend(fontsize=9)
axes[1,1].set(title="B: prediction length on training phrases",ylabel="Predicted / reference CTC tokens",ylim=(.45,1.35))
axes[1,1].axhline(1,color="#888888",linestyle=":")
axes[1,1].legend(fontsize=9)
for ax in axes.flat:
    ax.grid(alpha=.2)
fig.suptitle("Saved-output diagnosis: phrase fit is incomplete; blank share alone is insufficient",fontsize=15)
fig.supxlabel("B trains phrases through update 420, then full sentences. LR changes after 840 and 1050.\nTrain loss and eval CER are different metrics. Repeated internal-pilot observations; no causal conclusion.",fontsize=10)
target=OUT/"loss_and_phrase_output_r2.png"
if target.exists():
    raise FileExistsError(target)
fig.savefig(target,dpi=150)
plt.close(fig)
with (OUT/"plot_points_r2.json").open("x",encoding="utf-8") as f:
    json.dump({"points":points,"sources":[ref(OUT/"loss_windows.json"),ref(OUT/"output_groups.json")],"producer":ref(Path(__file__))},f,ensure_ascii=False,indent=2)
print(json.dumps({"path":str(target),"points":len(points)}))
