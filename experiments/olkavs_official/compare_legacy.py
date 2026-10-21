"""Same-four-utterance diagnostic of the fixed stage17B and official pathways.

This intentionally compares whole model/input pathways, not isolated networks.
Two selected clips were trained by stage17B. This is not a held-out benchmark.
Run in .venv-gpu; only the four frozen training caches are opened for inference.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from config import FPS, NORMALIZE_MEAN, NORMALIZE_STD, ROI_MARGIN, ROI_SIZE
from models.cache import load_clip
from models.decoder import ctc_greedy_decode, ctc_beam_search_decode
from models.registry import load_model_checkpoint
from diagnostic_utils import text_counts, canonical_text

MANIFEST = ROOT / "data_cache/datasets/aihub_timed_v4/manifest.json"
STAGE17 = ROOT / "results/step17-expansion/run01"
CHECKPOINT = STAGE17 / "B_expand512/step_6400.pt"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def pooled(records, name):
    counts = [r["outputs"][name]["metrics"] for r in records]
    edits = sum(c["character_edits_with_spaces"] for c in counts)
    refs = sum(c["reference_characters_with_spaces"] for c in counts)
    edits_no_space = sum(c["character_edits_without_spaces"] for c in counts)
    refs_no_space = sum(c["reference_characters_without_spaces"] for c in counts)
    return dict(clips=len(counts), edits=edits, reference_characters=refs,
                pooled_cer_percent=100 * edits / refs,
                edits_no_spaces=edits_no_space, reference_characters_no_spaces=refs_no_space,
                pooled_cer_no_spaces_percent=100 * edits_no_space / refs_no_space,
                exact=sum(c["exact_sentence"] for c in counts))


def main():
    plan = read_json(HERE / "step02_plan.json")
    prior_plan = read_json(STAGE17 / "plan.json")
    prior_done = read_json(STAGE17 / "B_expand512/completed.json")
    old_summary_path = STAGE17 / "B_expand512/step_6400/summary.json"
    old_summary = read_json(old_summary_path)
    endpoint = next(r for r in prior_done["snapshots"] if r["step"] == 6400)
    manifest_sha = sha256(MANIFEST)
    if manifest_sha != prior_plan["manifest_sha256"]:
        raise ValueError("Stage17 manifest changed")
    checkpoint_sha = sha256(CHECKPOINT)
    if checkpoint_sha != endpoint["checkpoint_sha256"]:
        raise ValueError("Stage17 fixed endpoint checkpoint changed")
    if sha256(old_summary_path) != endpoint["summary_sha256"]:
        raise ValueError("Saved stage17 summary changed")
    old_rows_path = STAGE17 / "B_expand512/step_6400/rows.jsonl"
    if sha256(old_rows_path) != old_summary["rows_sha256"]:
        raise ValueError("Saved stage17 prediction rows changed")
    old_rows = {r["id"]: r for r in
                (json.loads(line) for line in old_rows_path.read_text(encoding="utf-8").splitlines())
                if r["condition"] == "normal"}
    records = {r["id"]: r for r in read_json(MANIFEST)["records"]}
    train_ids = set(prior_plan["branches"]["B_expand512"]["train_ids"])
    trained_recordings = {records[sid]["recording_id"] for sid in train_ids}
    trained_speakers = {records[sid]["speaker_id"] for sid in train_ids}
    official_path = HERE / "results/step02.json"
    official = read_json(official_path)
    if official["status"] != "completed" or official["plan"]["source_ids"] != plan["source_ids"]:
        raise ValueError("Completed official comparison of the same frozen clips is required")
    official_rows = {(r["id"], r["size"]): r for r in official["records"]}
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint["architecture_version"] != "lipnet-compact-v1"
            or checkpoint["num_classes"] != 69
            or set(checkpoint["training_data"]["sample_ids"]) != train_ids):
        raise ValueError("Unexpected legacy model/training contract")
    if not torch.cuda.is_available():
        raise RuntimeError("This run is configured for the original GPU evaluation path")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model_checkpoint(str(CHECKPOINT)).to("cuda").eval()
    # The project loader uses load_state_dict's strict=True default. Verify the
    # loaded tensors exactly match all checkpoint keys before inference as well.
    actual = model.state_dict()
    if set(actual) != set(checkpoint["state_dict"]):
        raise ValueError("Loaded model keys differ")
    for key, expected in checkpoint["state_dict"].items():
        if not torch.equal(actual[key].cpu(), expected):
            raise ValueError("Loaded model tensor differs: " + key)
    result = {
        "schema": "olkavs-step02-legacy-comparison-v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Same-utterance diagnostic of model plus native input pipeline; two legacy-trained clips; not held-out model ranking",
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha,
                       "bytes": CHECKPOINT.stat().st_size, "strict_load": True,
                       "all_loaded_tensors_exact": True, "architecture": checkpoint["architecture_version"],
                       "num_classes": checkpoint["num_classes"],
                       "source_training_run": checkpoint["training_run"]},
        "source_plan_sha256": sha256(STAGE17 / "plan.json"),
        "manifest_sha256": manifest_sha, "step02_plan_sha256": sha256(HERE / "step02_plan.json"),
        "official_result_sha256": sha256(official_path),
        "source_rows_sha256": old_summary["rows_sha256"], "runner_sha256": sha256(Path(__file__)),
        "input_contract": {"fps": FPS, "rgb": True, "roi_size": list(ROI_SIZE),
                           "roi_margin": ROI_MARGIN, "normalization_mean": NORMALIZE_MEAN,
                           "normalization_std": NORMALIZE_STD, "resampling": "timestamp-based v4 cache",
                           "contrasted_official": "BGR, nominal30fps, no-margin bounding boxes, whole-clip normalization, mp4v crop"},
        "audio_used": False, "training_run": False, "test_cache_opened": False,
        "evaluation": {"mode": "eval", "dtype": "float32", "batch_size": 1, "padding": False,
                       "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
                       "cudnn_deterministic": True, "tf32": False},
        "records": [],
    }
    prior_posteriors_path = STAGE17 / "B_expand512/step_6400/posteriors.npz"
    if sha256(prior_posteriors_path) != old_summary["posteriors_sha256"]:
        raise ValueError("Saved stage17 posteriors changed")
    with np.load(prior_posteriors_path, allow_pickle=False) as saved_posteriors:
        for sid in plan["source_ids"]:
            source = records[sid]
            if source["split"] != "train":
                raise ValueError("Only frozen train clips are allowed")
            cache_path = MANIFEST.parent / source["cache_path"]
            if sha256(cache_path) != source["cache_sha256"]:
                raise ValueError("Cache hash mismatch")
            clip = load_clip(cache_path)
            if clip["text"] != source["text"] or len(clip["frames"]) != source["frames"]:
                raise ValueError("Cache metadata mismatch")
            reference = official_rows[(sid, 96)]["reference_for_metrics_only"]
            if canonical_text(reference) != canonical_text(clip["text"]):
                raise ValueError("Reference text differs between pathways")
            values = clip["frames"].astype(np.float32) / 255.0 - np.asarray(NORMALIZE_MEAN, dtype=np.float32)
            values /= np.asarray(NORMALIZE_STD, dtype=np.float32)
            tensor = torch.from_numpy(values).permute(3, 0, 1, 2).unsqueeze(0).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                lp = model(tensor).squeeze(1).float().cpu().numpy()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if lp.shape != (source["frames"], 69) or not np.isfinite(lp).all():
                raise ValueError("Invalid legacy output")
            greedy, beam = ctc_greedy_decode(lp), ctc_beam_search_decode(lp, beam_width=10)
            outputs = {name: {"prediction": prediction, "metrics": text_counts(reference, prediction)}
                       for name, prediction in (("legacy_greedy", greedy), ("legacy_beam10", beam))}
            for size in (224, 96):
                for method, value in official_rows[(sid, size)]["outputs"].items():
                    outputs[f"official_{size}_{method}"] = {
                        "prediction": value["prediction"], "metrics": text_counts(reference, value["prediction"])}
            reproduction = None
            if sid in old_rows:
                old = old_rows[sid]
                previous = saved_posteriors[old["posterior_key"]]
                reproduction = {"greedy_identical": old["greedy"]["prediction"] == greedy,
                                "beam10_identical": old["beam"]["prediction"] == beam,
                                "posterior_exact": np.array_equal(previous, lp),
                                "posterior_max_absolute_difference": float(np.max(np.abs(previous - lp)))}
                if not all(reproduction[k] for k in ("greedy_identical", "beam10_identical", "posterior_exact")):
                    raise ValueError("Legacy saved output reproduction failed: " + sid)
            item = {"id": sid, "reference": reference, "cache_path": str(cache_path),
                    "cache_sha256": source["cache_sha256"], "frames": source["frames"],
                    "legacy_training_member": sid in train_ids,
                    "legacy_training_cohort": [k for k, ids in prior_plan["cohorts"].items() if sid in ids],
                    "recording_seen_in_legacy_training": source["recording_id"] in trained_recordings,
                    "speaker_seen_in_legacy_training": source["speaker_id"] in trained_speakers,
                    "outputs": outputs, "forward_seconds": elapsed,
                    "saved_stage17_reproduction": reproduction}
            result["records"].append(item)
            print(json.dumps({"id": sid, "trained": item["legacy_training_member"],
                              "greedy": greedy, "saved_reproduction": reproduction}, ensure_ascii=False), flush=True)
            if sha256(cache_path) != source["cache_sha256"]:
                raise ValueError("Cache changed during inference")
    result["aggregate"] = {name: pooled(result["records"], name) for name in result["records"][0]["outputs"]}
    if sha256(CHECKPOINT) != checkpoint_sha or sha256(MANIFEST) != manifest_sha:
        raise ValueError("Source checkpoint/manifest changed")
    result["status"] = "completed"
    target = HERE / "results/step02_legacy_comparison.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
