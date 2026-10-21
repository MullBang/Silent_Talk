"""Early compatibility check of the first saved checkpoint evaluation schema."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from audit_common import collapsed_ids, compose_ids, evaluation_digest, score, unpack
from audit_sources import binding, check_binding, read

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
path = BASE / "runs/seed42/A_sentence/evaluations/update_0420/evaluation_summary.json"
value = read(path)
assert value["count"] == 90 and value["failures"] == 0 and value["posterior_count"] == 90
assert value["identity"]["completed_update"] == 420 and value["primary_result"] is None
row = value["rows"][0]
check_binding(row["posterior"])
with np.load(row["posterior"]["path"], allow_pickle=False) as data:
    assert set(data.files) == {"log_probs", "raw_argmax_ids", "collapsed_ids", "frame_indices", "length", "metadata_json"}
    log_probs = data["log_probs"]
    assert log_probs.dtype == np.float32 and log_probs.shape == (row["T"], 69)
    raw = log_probs.argmax(axis=1).tolist()
    collapsed = collapsed_ids(raw)
    assert row["raw_argmax_ids"] == raw and row["collapsed_ids"] == collapsed
    assert row["raw_text"] == compose_ids(collapsed)
    assert row["metrics"] == score(row["reference_text"], row["raw_text"])
check_binding(value["identity"]["checkpoint"])
payload = unpack(torch.load(value["identity"]["checkpoint"]["path"], weights_only=True, map_location="cpu"))
isolation = read(Path(value["state_isolation"]["path"]))
assert isolation["before"] == isolation["after"]
assert isolation["before"]["model"] == evaluation_digest(payload["state"]["model"])
rng = payload["state"]["rng"]
assert isolation["before"]["rng"] == evaluation_digest({"python":rng["python"], "numpy":rng["numpy"], "torch_cpu":rng["torch"], "torch_cuda":rng["cuda"]})
report = {"schema":"offline-stage5-first-evaluation-schema-precheck-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
          "status":"passed", "summary":binding(path), "scope":"First420 summary schema and one posterior/metric; not the final1080-row audit",
          "summary_rows":90, "sampled_posteriors":1, "sampled_clip":row["clip_id"], "checkpoint_model_rng_bound":True,
          "model_forward_calls":0, "producer_imports":0, "optimizer_updates":0}
output = HERE / "first_checkpoint_schema_precheck.json"
assert not output.exists()
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status":"passed", "summary_rows":90, "sampled_posteriors":1}))
