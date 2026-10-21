"""Evaluate timestamp-aligned development20 geometry variants without fitting the model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch
import yaml

from verify_model import ROOT, UPSTREAM, EXPECTED_COMMIT, sha256, save_json, checked_state
from diagnose_step02 import read_tensor, timed_call
from corrected_search import CorrectedSearchSequence
from diagnostic_utils import collapse_ctc
from evaluation_metrics import counts, aggregate


class ObservedSearch(CorrectedSearchSequence):
    """Observe search depth; all scoring and candidate selection stay unchanged."""

    def _beam_search(self, features, current_hypothesis, length, *args, **kwargs):
        self.observed_max_search_length = max(getattr(self, "observed_max_search_length", 0), length)
        return super()._beam_search(features, current_hypothesis, length, *args, **kwargs)


def summarize(records: list[dict], plan: dict) -> dict:
    """Report paired arm metrics on the full frozen cohort and each speaker."""
    result = {}
    for arm in plan["arms"]:
        selected = [r for r in records if r["arm"] == arm]
        if len(selected) != plan["cohort_size"]:
            raise ValueError("Full-cohort denominator not available")
        for method in plan["methods"]:
            entries = [dict(id=r["id"], **r["outputs"][method]) for r in selected]
            key = f"{arm}/{method}"
            result[key] = aggregate(entries)
            result[key]["per_speaker"] = {
                speaker: aggregate([dict(id=r["id"], **r["outputs"][method])
                                    for r in selected if r["speaker_id"] == speaker])
                for speaker in sorted({r["speaker_id"] for r in selected})}
            times = [r["feature_timing"]["seconds"] + r["outputs"][method]["decoder_timing"]["seconds"]
                     for r in selected if r["outputs"][method]["status"] == "ok"]
            result[key]["model_timing"] = ({"n": len(times), "min_seconds": min(times),
                                              "median_seconds": statistics.median(times),
                                              "max_seconds": max(times), "mean_seconds": statistics.mean(times)}
                                             if times else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "inputs_step04_aligned" / "manifest.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "step04_aligned_eval_plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "step04.json")
    args = parser.parse_args()
    output, manifest_path = args.output.resolve(), args.manifest.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous runs: choose a new output path")
    plan_path, source_path = args.plan.resolve(), ROOT / "step03_source_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(source_path) != plan["source_plan_sha256"]:
        raise ValueError("Source plan changed")
    if sha256(ROOT / "reference_profile_96_v1.json") != plan["reference_profile_sha256"]:
        raise ValueError("Reference profile changed")
    if sha256(ROOT / "corrected_search.py") != plan["corrected_search_sha256"]:
        raise ValueError("Frozen corrected decoder changed")
    if (manifest["source_plan_sha256"] != plan["source_plan_sha256"]
            or manifest["reference_profile_sha256"] != plan["reference_profile_sha256"]
            or manifest["evaluation_plan"]["sha256"] != sha256(plan_path)):
        raise ValueError("Prepared inputs are not linked to the frozen evaluation plan")
    auto_profile = manifest["automatic_roi_profile"]
    if (auto_profile["mediapipe_version"] != "0.10.21" or auto_profile["margin_each_side"] != 0.2
            or auto_profile["detection_width"] != 640 or auto_profile["annotation_fallback"] is not False):
        raise ValueError("Unexpected automatic crop profile")
    old_plan = json.loads((ROOT / "step03_eval_plan.json").read_text(encoding="utf-8"))
    required_arms = ["annotation_bbox", "rectangle_control", "square", "context_square", "stable_context_square"]
    if (plan["source_ids"] != old_plan["source_ids"] or plan["cohort_size"] != 20
            or plan["arms"] != required_arms):
        raise ValueError("Frozen development20 and five comparison arms are required")
    for key in ("source_plan_sha256", "reference_profile_sha256", "corrected_search_sha256", "checkpoint_sha256"):
        if plan[key] != old_plan[key]:
            raise ValueError(f"Model/source contract changed: {key}")
    if (plan["hybrid"] != old_plan["hybrid"] or plan["common_input"] != old_plan["common_input"]
            or plan["methods"] != old_plan["methods"]):
        raise ValueError("Only geometry and independently measured timestamp offsets may change")
    alignment_path = Path(plan["alignment_sidecar"]["path"])
    if sha256(alignment_path) != plan["alignment_sidecar"]["sha256"]:
        raise ValueError("Alignment sidecar changed")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if manifest["alignment_sidecar"]["sha256"] != plan["alignment_sidecar"]["sha256"]:
        raise ValueError("Input alignment provenance mismatch")
    offsets = {s["source_name"]:s["label_to_video_offset_seconds"] for s in alignment["sources"]}
    expected_sources = {sid.rsplit("_",1)[0]+".mp4" for sid in plan["source_ids"]}
    if (len(alignment["sources"]) != 4 or set(offsets) != expected_sources
            or any(not math.isfinite(x) for x in offsets.values()) or alignment["status"] != "validated"):
        raise ValueError("Expected one independently validated finite offset for each source")
    for evidence in alignment["evidence_files"]:
        if sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("Independent audio alignment evidence changed")
    if sha256(ROOT / "roi_geometry_step04.py") != plan["geometry_script_sha256"]:
        raise ValueError("Frozen geometry changed")
    if manifest["geometry_script_sha256"] != plan["geometry_script_sha256"]:
        raise ValueError("Prepared geometry provenance mismatch")
    previous_manifest = json.loads((ROOT / "inputs_step03_recovery02" / "manifest.json").read_text(encoding="utf-8"))
    native_fps = {r["id"]:r["native_fps"] for r in previous_manifest["records"]}
    expected = {(sid, arm) for sid in plan["source_ids"] for arm in plan["arms"]}
    records = manifest["records"]
    if len(records) != len(expected) or {(r["id"], r["arm"]) for r in records} != expected:
        raise ValueError("Expected all 20 utterances in each arm, including failed crops")
    source_records = {r["id"]: r for r in sources["records"]}
    if any(r["split"] != "val" for r in source_records.values()):
        raise ValueError("Evaluation scope must be the existing validation split")
    for r in records:
        if r["text"] != source_records[r["id"]]["text"]:
            raise ValueError("Reference mismatch")
        if r["temporal_method"] != "native-cfr-audio-aligned-v1":
            raise ValueError("Prepared input used the wrong timestamp selection")
        ref = source_records[r["id"]]
        source_name = r["id"].rsplit("_",1)[0] + ".mp4"
        offset = offsets[source_name]
        if (r["native_fps"] != native_fps[r["id"]]
                or r["label_to_video_offset_seconds"] != offset):
            raise ValueError("Native FPS or fixed source offset changed across geometry arms")
        first = math.ceil((ref["start_time"]+offset)*r["native_fps"]-1e-9)
        stop = math.ceil((ref["end_time"]+offset)*r["native_fps"]-1e-9)
        if first < 0 or (r["start_frame"],r["stop_frame_exclusive"]) != (first,stop):
            raise ValueError("Input interval differs from independently locked time correction")
        if r["status"] == "ok" and (not r["valid_for_inference"]
                or r["frames"] != r["expected_frames"] or r["frames"] != stop-first or stop<=first):
            raise ValueError("Successful input must contain every selected frame")
    for sid in plan["source_ids"]:
        hashes = {r["source_selected_frames_sha256"] for r in records if r["id"]==sid}
        if len(hashes) != 1:
            raise ValueError("Geometry arms used different original frames")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise ValueError("Official code must remain at the clean fixed commit")
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    from avsr.utils.model_builder import build_model
    from avsr.utils.korean_funcs import grp2char
    from vocabulary.utils import KsponSpeechVocabulary

    config = yaml.safe_load((UPSTREAM / "configs" / "v_inference.yaml").read_text(encoding="utf-8"))
    vocab = KsponSpeechVocabulary(unit="grapheme")
    if (len(vocab), vocab.unk_id, vocab.eos_id) != (72, 3, 2):
        raise ValueError("Official vocabulary mismatch")
    torch.manual_seed(2207)
    np.random.seed(2207)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    kwargs = {k: config[k] for k in inspect.signature(build_model).parameters if k in config}
    model = build_model(vocab_size=72, pad_id=0, pass_visual_frontend=False, **kwargs)
    checkpoint = ROOT / "weights" / "multiview_lip.pt"
    if sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    state_contract = checked_state(model, torch.load(checkpoint, map_location="cpu", weights_only=True))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the frozen evaluation")
    device = torch.device("cuda:0")
    model.to(device).eval()
    if type(model.encoder).__name__ != "VisualConformerEncoder":
        raise ValueError("Video-only architecture required")
    result = {"schema": "olkavs-step04-results-v1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "plan_sha256": sha256(plan_path), "source_plan_sha256": sha256(source_path),
              "input_manifest_sha256": sha256(manifest_path), "runner_sha256": sha256(__file__),
              "input_manifest_path":str(manifest_path), "alignment_sidecar":plan["alignment_sidecar"],
              "geometry_script_sha256":sha256(ROOT / "roi_geometry_step04.py"),
              "metrics_code_sha256": sha256(ROOT / "evaluation_metrics.py"),
              "read_tensor_code_sha256": sha256(ROOT / "diagnose_step02.py"),
              "checkpoint_contract": state_contract, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "audio_used": False, "training_run": False,
              "test50_used": False, "records": [], "audio_used_only_for_timeline_audit":True,
              "timing_scope": "GPU-ready video -> shared encoder+medium plus decoder; excludes crop/source decode/transfer/real camera; first call not pre-warmed"}
    save_json(output, result)
    lookup = {(r["id"], r["arm"]): r for r in records}
    search_args = dict(model=model, vocab_size=72, pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                       method="hybrid", max_len=200, ctc_rate=0.6, mp_num=1)

    for arm in plan["arms"]:
        for i, sid in enumerate(plan["source_ids"]):
            record = lookup[(sid, arm)]
            item = {"id": sid, "speaker_id": source_records[sid]["speaker_id"], "arm": arm,
                    "input_status": record["status"], "reference": record["text"],
                    "input_failure_reason": record.get("failure_reason"),
                    "frames": record["frames"], "outputs": {}}
            print(f"{arm} {i+1}/20 {sid}", flush=True)
            if record["status"] != "ok":
                for method in plan["methods"]:
                    item["outputs"][method] = {"status": "failed_input", "prediction": "",
                                              "error": record.get("failure_reason", "Input preparation failed"),
                                              "metrics": counts(record["text"], ""), "decoder_timing": None}
            else:
                try:
                    video, norm = read_tensor(record, 96)
                    video = video.to(device)
                    lengths = torch.tensor([video.shape[2]], device=device)
                    item.update(input_sha256=record["sha256"], normalization=norm)
                    with torch.inference_mode():
                        features, timing = timed_call(lambda: model.medium(model.encoder(video, lengths, None, None)))
                        item["feature_timing"] = timing
                        item["feature_shape"] = list(features.shape)
                        if not torch.isfinite(features).all():
                            raise ValueError("Nonfinite features")
                        def ctc_forward():
                            logits = model.decoder.ctcdecoder(inputs=features)
                            return logits, logits.argmax(-1)

                        logits = None
                        for method in plan["methods"]:
                            try:
                                extra = {}
                                if method == "corrected_hybrid":
                                    search = ObservedSearch(**search_args)
                                    (ids, lens), decoder_time = timed_call(lambda: search.decode_features(features, beam_size=1, D_end=-14, M_end=16))
                                    tokens = ids[0, :int(lens[0])].cpu().tolist()
                                    if not tokens or tokens[-1] != 2 or tokens.count(2) != 1 or 0 in tokens or 1 in tokens:
                                        raise ValueError("Corrected output boundary contract failed")
                                    extra = {"max_search_length": search.observed_max_search_length,
                                             "search_cap_reached": search.observed_max_search_length >= search.max_len - 1}
                                else:
                                    (logits, raw_path), decoder_time = timed_call(ctc_forward)
                                    if not torch.isfinite(logits).all():
                                        raise ValueError("Nonfinite CTC logits")
                                    raw_path = raw_path[0].cpu().tolist()
                                    tokens = collapse_ctc(raw_path, blank_id=3)
                                    item["ctc_frame_path"] = raw_path
                                    item["ctc_stats"] = {"blank_fraction": raw_path.count(3) / len(raw_path),
                                                         "token_counts": dict(Counter(raw_path)),
                                                         "first_eos_frame": raw_path.index(2) if 2 in raw_path else None}
                                graphemes = vocab.label_to_string(torch.tensor(tokens, dtype=torch.long))
                                prediction = grp2char(graphemes).strip()
                                item["outputs"][method] = {"status": "ok", "token_ids": tokens,
                                                           "eos_index": tokens.index(2) if 2 in tokens else None,
                                                           "graphemes": graphemes, "prediction": prediction,
                                                           "metrics": counts(record["text"], prediction),
                                                           "decoder_timing": decoder_time, **extra}
                            except (ValueError, RuntimeError, OSError) as exc:
                                item["outputs"][method] = {"status": "failed_decoder", "prediction": "",
                                                          "error": {"type": type(exc).__name__, "message": str(exc)},
                                                          "metrics": counts(record["text"], ""), "decoder_timing": None}
                                torch.cuda.empty_cache()
                        feature_file = output.parent / f"{output.stem}_features" / f"{arm}_{sid}.npz"
                        feature_file.parent.mkdir(exist_ok=True)
                        saved_arrays = {"features": features[0].cpu().numpy()}
                        if logits is not None:
                            saved_arrays["ctc_logits"] = logits[0].cpu().numpy()
                        np.savez_compressed(feature_file, **saved_arrays)
                        item.update(feature_file=str(feature_file), feature_file_sha256=sha256(feature_file))
                    del video, features, logits
                except (ValueError, RuntimeError, OSError) as exc:
                    # Keep failures in the cohort; never substitute a label-based crop.
                    item["runtime_error"] = {"type": type(exc).__name__, "message": str(exc)}
                    for method in plan["methods"]:
                        if method not in item["outputs"]:
                            item["outputs"][method] = {"status": "failed_runtime", "prediction": "",
                                                      "metrics": counts(record["text"], ""), "decoder_timing": None}
                    torch.cuda.empty_cache()
            result["records"].append(item)
            save_json(output, result)
            print(json.dumps({"arm": arm, "id": sid, "predictions": {m: v["prediction"] for m, v in item["outputs"].items()}}, ensure_ascii=False), flush=True)
    result["aggregate"] = summarize(result["records"], plan)
    result["status"] = "completed"
    save_json(output, result)


if __name__ == "__main__":
    main()
