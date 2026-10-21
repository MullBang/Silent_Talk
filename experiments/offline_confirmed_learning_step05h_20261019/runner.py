"""Bounded 5h visual-only A12/B14 comparison; source approvals stay provisional.

Only the bound 17 phrase caches are read. Original audio/media are provenance,
never model inputs. This module does not execute training when imported.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound(binding):
    path = Path(binding["path"]).resolve()
    require(sha(path) == binding["sha256"], f"Bound bytes changed: {path}")
    if "bytes" in binding:
        require(path.stat().st_size == binding["bytes"], "Bound length changed")
    return path


def save(path, obj):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE), "Output outside 5h")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


PRIOR = ROOT / "experiments/offline_comparison_step05_20261013/execution.py"
CONTRACT = ROOT / "experiments/offline_training_step04_20261013/evaluation_contract.py"
require(sha(PRIOR) == "623e4a21d38bb8d32bc90e8949cada112fe19fd70d24c6734c9b5ccfadebce93", "Frozen update source changed")
require(sha(CONTRACT) == "654d83a6958622ebcec310bed5e97a9e5001a32bc4fbc1592a16dcefb7619dc3", "Frozen metrics changed")
prior = module(PRIOR, "stage5h_frozen_update")
contract = module(CONTRACT, "stage5h_frozen_metrics")
old, torch, np = prior.old, prior.torch, prior.np
tokenizer = module(ROOT / "backend/models/tokenizer.py", "stage5h_tokenizer")
CACHE_SCHEMA = "lipnet-recording-grid-D-rgb-provisional-v1"
KEYS = {"frames_rgb", "label", "length", "metadata_json", "schema", "training_ready", "provisional"}


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_plan(path):
    plan = read(path)
    require(plan["schema"] == "offline-stage5h-confirmed-learning-plan-v1", "Wrong plan")
    require(plan["steps"] == 1680 and plan["seeds"] == [42, 43] and plan["batch_size"] == 2,
            "Training budget changed")
    require(plan["evaluations"] == [840, 1680], "Evaluation schedule changed")
    bindings = {str(bound(b)): b for b in plan["bindings"]}
    for required in (Path(__file__).resolve(), PRIOR.resolve(), CONTRACT.resolve(), ROOT / "backend/models/tokenizer.py"):
        require(str(required) in bindings, f"Missing code binding: {required}")
    require(tokenizer.NUM_CLASSES == 69 and tokenizer.VOCAB_SHA256 ==
            "3ab81291a1892a4aece2d41bce634ff54663c4b93e3d56aef81f6783c822ff62", "Tokenizer changed")
    bound(plan["data_plan"])
    return plan


class Dataset:
    """Small explicit cache allowlist; no implicit scans or raw-media reads."""
    def __init__(self, binding):
        self.plan = read(bound(binding))
        records = self.plan["records"]
        self.rows = {r["clip_id"]: r for r in records}
        require(len(records) == len(self.rows) == 17, "Expected exactly 17 unique phrases")
        self.arms, self.dev = self.plan["arms"], self.plan["dev"]
        a, b, dev = self.arms["A12"], self.arms["B14"], self.dev
        require(len(a) == len(set(a)) == 12 and len(b) == len(set(b)) == 14 and
                len(dev) == len(set(dev)) == 3 and set(a) < set(b), "Wrong arm scope")
        require(not set(b) & set(dev) and set(b) | set(dev) == set(self.rows), "Train/dev scope overlaps")
        self.new = sorted(set(b) - set(a))
        self.overrides = self.plan.get("experimental_split_overrides", {})
        require(set(self.overrides) <= set(self.new), "Only the explicitly new two may override source split")
        for sid, row in self.rows.items():
            require(row["kind"] == "phrase" and row["split"] == ("dev" if sid in dev else "train"), "Nonphrase/wrong split")
            require(row["training_ready"] is False and row["provisional"] is True, "Source approval promoted")
        self.items = {sid: self._load(row) for sid, row in self.rows.items()}

    def _load(self, row):
        ident = row["clip_id"]
        meta = read(bound({"path": row["metadata_path"], "sha256": row["metadata_sha256"]}))
        path = bound({"path": row["npz_path"], "sha256": row["npz_sha256"], "bytes": row["npz_bytes"]})
        require(path.suffix == ".npz", "Only a visual NPZ may be loaded")
        if meta["split"] != row["split"]:
            require(self.overrides.get(ident) == {"source_split":meta["split"],"experiment_split":row["split"]},
                    "Unauthorized source split override")
        raw = path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(len(archive.infolist()) == 7 and set(archive.namelist()) == {k + ".npy" for k in KEYS},
                    "Unexpected payload/audio in visual cache")
            require(sum(m.file_size for m in archive.infolist()) < row["T"] * 96*96*3 + len(json.dumps(meta))*8 + 100000,
                    "Oversized NPZ payload")
        with np.load(io.BytesIO(raw), allow_pickle=False) as packed:
            require(packed["schema"].item() == meta["schema"] == CACHE_SCHEMA, "Wrong cache schema")
            require(packed["training_ready"].item() is False and packed["provisional"].item() is True and
                    meta["training_ready"] is False and meta["provisional"] is True, "Source flags changed")
            require(json.loads(packed["metadata_json"].item()) == meta, "NPZ/sidecar mismatch")
            require(meta["clip_id"] == ident and meta["text"] == row["text"], "Text/identity mismatch")
            rgb, labels = packed["frames_rgb"], packed["label"]
            length = packed["length"]
            require(rgb.dtype == np.uint8 and rgb.shape == (row["T"],96,96,3), "RGB shape/dtype mismatch")
            require(length.dtype == np.int64 and length.shape == () and int(length) == row["T"], "Length mismatch")
            require(labels.dtype == np.int64 and labels.shape == (row["label_length"],) and
                    labels.tolist() == meta["label"] == tokenizer.encode_text(row["text"]), "69-token target mismatch")
            require(tokenizer.ctc_min_input_len(labels.tolist()) <= row["T"], "CTC infeasible")
            require(array_sha(rgb) == meta["rgb_uint8_sha256"], "Raw RGB hash mismatch")
            require(not np.any(np.all(rgb == 0, axis=(1,2,3))), "Zero placeholder frame")
            norm = (rgb.astype(np.float32) / np.float32(255) - np.asarray([.485,.456,.406],np.float32)) / np.asarray([.229,.224,.225],np.float32)
            pixels = np.ascontiguousarray(norm.transpose(3,0,1,2))
            labels = labels.copy()
        require(np.isfinite(pixels).all(), "Nonfinite normalized pixels")
        pixels.setflags(write=False)
        labels.setflags(write=False)
        return {"input": pixels, "labels": labels, "length": row["T"], "row": row,
                "audit": {"clip_id": ident, "npz": ref(path), "metadata_sha256": row["metadata_sha256"],
                          "source_rgb_uint8_sha256": meta["rgb_uint8_sha256"], "normalized_input_sha256": array_sha(pixels),
                          "labels_sha256": array_sha(labels), "source_split": meta["split"], "experiment_split": row["split"],
                          "source_provisional": True, "source_training_ready": False, "approval_changed": False}}

    def batch(self, ids, arm):
        require(arm in self.arms and len(ids) == 2 and all(s in self.arms[arm] for s in ids), "Optimizer outside arm allowlist")
        items = [self.items[s] for s in ids]
        lengths = np.asarray([i["length"] for i in items], np.int64)
        x = np.empty((2,3,int(lengths.max()),96,96), np.float32)
        for index, item in enumerate(items):
            require(array_sha(item["input"]) == item["audit"]["normalized_input_sha256"] and
                    array_sha(item["labels"]) == item["audit"]["labels_sha256"], "Loaded arrays changed")
            length = item["length"]
            x[index,:,:length] = item["input"]
            x[index,:,length:] = item["input"][:,-1:]
        return {"inputs": x, "lengths": lengths, "labels": np.concatenate([i["labels"] for i in items]),
                "label_lengths": np.asarray([len(i["labels"]) for i in items],np.int64),
                "clip_ids": list(ids), "splits": ["train", "train"], "purpose": "optimizer"}


def schedule(plan, data, arm, seed):
    tasks = read(bound(plan["schedules"][arm][str(seed)]))["updates"]
    require(len(tasks) == 1680, "Schedule length differs")
    for index, task in enumerate(tasks, 1):
        lr = .001 if index <= 840 else (.0003 if index <= 1050 else .0001)
        require(task["update"] == index and task["lr"] == lr and len(task["clip_ids"]) == 2 and
                all(s in data.arms[arm] for s in task["clip_ids"]), "Schedule changed/outside arm")
        require(task["lengths"] == [data.rows[s]["T"] for s in task["clip_ids"]], "Schedule lengths differ")
    return tasks


def checkpoint(path, payload):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE) and not path.exists(), "Unsafe/existing checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.partial")
    with temporary.open("xb") as stream:
        torch.save(old.runner._pack_safe(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)
    binding = ref(path)
    restored = load_checkpoint(binding)
    require(old.hashes(restored["state"]) == payload["state_hashes"], "Checkpoint round-trip differs")
    return binding


def load_checkpoint(binding):
    path = bound(binding)
    require(path.is_relative_to(BASE), "Checkpoint outside 5h")
    return old.runner._unpack_safe(torch.load(path, map_location="cpu", weights_only=True))


def conditions(length, clip_id):
    # Local generator never consumes model/training RNG. Same permutation in all runs.
    seed = int(hashlib.sha256(("stage5h/" + clip_id).encode()).hexdigest()[:16],16)
    return {"normal": np.arange(length,dtype=np.int64), "static": np.full(length,length//2,dtype=np.int64),
            "shuffled": np.random.default_rng(seed).permutation(length).astype(np.int64)}


def evaluate(model, optimizer, data, arm, seed, update, out):
    out.mkdir(parents=True, exist_ok=False)
    (out/"posteriors").mkdir()
    initial = old.hashes(old.state(model,optimizer))
    rng = old.core.capture_rng()
    modes = {name:m.training for name,m in model.named_modules()}
    gradients = old.runner.nested_hash({n:p.grad for n,p in model.named_parameters()})
    ids = data.arms[arm] if update == 840 else sorted(data.rows)
    requested = ["normal"] if update == 840 else ["normal","static","shuffled"]
    rows = []
    started = time.perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for sid in ids:
                item = data.items[sid]
                source = item["row"]
                cohort = "dev3" if sid in data.dev else ("new2" if sid in data.new else "train_common12")
                for condition in requested:
                    order = conditions(item["length"],sid)[condition]
                    pixels = np.ascontiguousarray(item["input"][:,order])
                    row = {"clip_id":sid,"cohort":cohort,"condition":condition,"arm":arm,"seed":seed,"update":update,
                           "seen_in_optimizer":sid in data.arms[arm],"reference":source["text"],"T":item["length"],
                           "frame_indices":order.tolist(),"input_sha256":array_sha(pixels),"input_audit":item["audit"],
                           "reference_label_ids":item["labels"].tolist(),"audio_input":False}
                    try:
                        values = model(torch.from_numpy(pixels[None]).to("cuda:0"),
                                       torch.tensor([item["length"]],dtype=torch.long)).detach().cpu().numpy()[:,0,:]
                        require(values.shape == (item["length"],69) and values.dtype == np.float32 and
                                np.isfinite(values).all(), "Invalid posterior")
                        require(np.allclose(np.exp(values.astype(np.float64)).sum(1),1.,atol=1e-5,rtol=1e-5), "Unnormalized posterior")
                        require(torch.cuda.max_memory_reserved() <= 6*1024**3, "Evaluation CUDA budget exceeded")
                        decoded = contract.decode_prediction(values.argmax(1))
                        row.update({**decoded,"status":"success","error":None,
                                    "metrics":contract.score_prediction(source["text"],decoded["raw_text"])})
                    except Exception as exc:
                        row.update({"status":"failed","error":repr(exc),"posterior":None,"raw_text":"",
                                    "metrics":contract.score_prediction(source["text"],"",failed=True)})
                    if row["status"] == "success":
                        path = out/"posteriors"/f"{sid}__{condition}.npz"
                        with path.open("xb") as stream:
                            np.savez_compressed(stream,log_probs=values,frame_indices=order,
                                reference_labels=item["labels"],raw_argmax_ids=np.asarray(decoded["raw_argmax_ids"],np.int64))
                        row["posterior"] = ref(path)
                        row["posterior_array_sha256"] = array_sha(values)
                    rows.append(row)
                    with (out/"rows.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    finally:
        for name,m in model.named_modules():
            m.training = modes[name]
        old.core.restore_rng(rng)
        isolation = {"state_before":initial,"state_after":old.hashes(old.state(model,optimizer)),
                     "gradients_before":gradients,"gradients_after":old.runner.nested_hash({n:p.grad for n,p in model.named_parameters()}),
                     "modes_before":modes,"modes_after":{n:m.training for n,m in model.named_modules()}}
        save(out/"isolation.json",isolation)
    require(isolation["state_before"] == isolation["state_after"] and isolation["gradients_before"] == isolation["gradients_after"] and
            isolation["modes_before"] == isolation["modes_after"], "Evaluation mutated caller state")
    require(len(rows) == len(ids)*len(requested), "Missing evaluation rows")
    scores = {f"{cohort}/{condition}":contract.corpus_scores([r["metrics"] for r in rows if r["cohort"] == cohort and r["condition"] == condition])
              for cohort,condition in sorted({(r["cohort"],r["condition"]) for r in rows})}
    result = {"rows":rows,"cohorts":scores,"count":len(rows),"failures":sum(r["status"]=="failed" for r in rows),
              "seconds":time.perf_counter()-started,"frames":sum(r["T"] for r in rows),"state_isolation_passed":True,
              "new2_interpretation":"untrained diagnostic" if arm=="A12" else "training fit",
              "dev3_interpretation":"reused same-speaker internal pilot, not independent final test",
              "optimizer_updates":0,"audio_inputs":0,"teacher_calls":0}
    binding = save(out/"summary.json",result)
    return {"summary":binding,"cohorts":scores,"failures":result["failures"],"count":len(rows),"frames":result["frames"]}


def preflight(plan_path):
    plan = verify_plan(plan_path)
    data = Dataset(plan["data_plan"])
    runtime = old.setup()
    require(runtime == plan["runtime"], "Runtime differs")
    out = BASE/"results/preflight"
    out.mkdir(parents=True,exist_ok=False)
    ids = sorted(data.arms["B14"],key=lambda s:(-data.rows[s]["T"],s))[:2]
    batch = data.batch(ids,"B14")
    model,optimizer = old.objects(42)
    reports = []
    try:
        # Two continuous updates, persisted resume point, and reference third update.
        for _ in range(2):
            reports.append(prior.update(model,optimizer,batch,.001))
        saved = old.state(model,optimizer)
        binding = checkpoint(out/"after2.pt",{"state":saved,"state_hashes":old.hashes(saved),"completed_updates":2,"mode":"preflight"})
        reference = prior.update(model,optimizer,batch,.0003)
        reports.append(reference)
        expected = old.hashes(old.state(model,optimizer))
        # A separately initialized optimizer performs a real update before restore,
        # demonstrating replacement of nonempty Adam state. Total physical count=5.
        del model,optimizer
        gc.collect()
        model,optimizer = old.objects(43)
        reports.append(prior.update(model,optimizer,batch,.001))
        restored = load_checkpoint(binding)
        old.restore(restored["state"],model,optimizer)
        require(old.hashes(old.state(model,optimizer)) == restored["state_hashes"], "Restore not exact")
        replay = prior.update(model,optimizer,batch,.0003)
        reports.append(replay)
        require(old.hashes(old.state(model,optimizer)) == expected, "Resume next update not exact")
        for key in ("loss","gradient_norm_before_clip","gradient_norm_after_clip","nonzero_gradient_tensors"):
            require(reference[key] == replay[key], f"Resume {key} differs")
        require(all(r["nonzero_gradient_tensors"] == r["parameter_tensors"] for r in reports), "Preflight gradient coverage incomplete")
        return save(out/"result.json",{"status":"passed","plan":ref(plan_path),"runtime":runtime,"physical_updates":5,
                    "maxdense_pair":ids,"reports":reports,"checkpoint":binding,"state_exact":True,"next_update_exact":True,
                    "reference_final_hashes":expected,"restored_final_hashes":old.hashes(old.state(model,optimizer)),
                    "audio_inputs":0,"source_approval_changed":False})
    except Exception:
        save(out/"failure.json",{"error":traceback.format_exc(),"physical_updates_returned":len(reports),"reports":reports})
        raise
    finally:
        del model,optimizer
        gc.collect()
        torch.cuda.empty_cache()


def train(plan_path, arm, seed, resume=None):
    plan = verify_plan(plan_path)
    pre = read(BASE/"results/preflight/result.json")
    require(pre["status"] == "passed" and pre["plan"] == ref(plan_path), "Missing matching preflight")
    runtime = old.setup()
    require(runtime == plan["runtime"], "Runtime differs")
    data = Dataset(plan["data_plan"])
    require(arm in ("A12","B14") and seed in (42,43), "Run outside scope")
    tasks = schedule(plan,data,arm,seed)
    identity = {"plan_sha256":sha(plan_path),"arm":arm,"seed":seed,"runtime":runtime,
                "schedule_sha256":plan["schedules"][arm][str(seed)]["sha256"]}
    out = BASE/"results"/f"{arm}_seed{seed}"
    # Resume is explicit into a new segment; failed attempts and logs stay intact.
    if resume is not None:
        out = out.with_name(out.name+"_resume")
    out.mkdir(parents=True,exist_ok=False)
    model,optimizer = old.objects(seed)
    initial = old.hashes(old.state(model,optimizer))
    history, checkpoints, evaluations = [],[],[]
    started = time.perf_counter()
    executed = 0
    try:
        if resume is not None:
            payload = load_checkpoint(resume)
            require(payload["identity"] == identity and payload["initial_state_hashes"] == initial, "Resume identity changed")
            require(payload["completed_updates"] == 840 and len(payload["history"]) == 840, "Only bound840 resume allowed")
            require(old.hashes(payload["state"]) == payload["state_hashes"], "Checkpoint state differs")
            old.restore(payload["state"],model,optimizer)
            require(old.hashes(old.state(model,optimizer)) == payload["state_hashes"], "Resume restore differs")
            history = payload["history"]
        save(out/"run_contract.json",{"identity":identity,"initial_state_hashes":initial,"resume":resume,
                                      "inputs":[data.items[s]["audit"] for s in sorted(data.items)],"optimizer_ids":data.arms[arm]})
        for task in tasks[len(history):]:
            require(time.perf_counter()-started < plan.get("wall_seconds_per_run",3600), "Run wall budget exceeded")
            measured = prior.update(model,optimizer,data.batch(task["clip_ids"],arm),task["lr"])
            executed += 1
            row = {"update":task["update"],**measured}
            history.append(row)
            with (out/"progress.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            require(measured["cuda_peak_reserved_bytes"] <= 6*1024**3, "CUDA budget exceeded")
            if task["update"] in (840,1680):
                state = old.state(model,optimizer)
                payload = {"schema":"offline-stage5h-complete-checkpoint-v1","identity":identity,
                           "initial_state_hashes":initial,"completed_updates":task["update"],"history":list(history),
                           "state":state,"state_hashes":old.hashes(state),
                           "next_task":tasks[task["update"]] if task["update"]<1680 else None}
                binding = checkpoint(out/f"update_{task['update']:04d}.pt",payload)
                checkpoints.append(binding)
                result = evaluate(model,optimizer,data,arm,seed,task["update"],out/f"eval_{task['update']:04d}")
                evaluations.append(result)
                require(result["failures"] == 0, "Evaluation failures recorded; stop")
            if task["update"]%120 == 0:
                print(json.dumps({"arm":arm,"seed":seed,"update":task["update"],"loss":row["loss"],"seconds":time.perf_counter()-started}),flush=True)
        return save(out/"result.json",{"status":"completed","identity":identity,"physical_updates":executed,"total_history":len(history),
                    "initial_state_hashes":initial,"checkpoints":checkpoints,"evaluations":evaluations,
                    "seconds":time.perf_counter()-started,"true_frames":sum(r["true_frames"] for r in history),
                    "dense_frames":sum(r["dense_frames"] for r in history),"audio_inputs":0,"teacher_calls":0,
                    "service_changes":0,"source_approval_changed":False,"source_training_ready":False})
    except Exception:
        save(out/"failure.json",{"error":traceback.format_exc(),"physical_updates_returned":executed,
                    "completed_history":len(history),"last_checkpoint":checkpoints[-1] if checkpoints else resume})
        raise
    finally:
        del model,optimizer,data
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,required=True)
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("train")
    run.add_argument("--arm",choices=["A12","B14"],required=True)
    run.add_argument("--seed",type=int,choices=[42,43],required=True)
    run.add_argument("--resume",type=Path)
    args = parser.parse_args()
    result = preflight(args.plan) if args.command == "preflight" else train(args.plan,args.arm,args.seed,read(args.resume) if args.resume else None)
    print(json.dumps(result,ensure_ascii=False),flush=True)
