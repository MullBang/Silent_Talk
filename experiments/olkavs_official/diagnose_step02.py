"""Frozen 2-resolution x 3-decoder diagnostic; no training or service integration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import yaml

from diagnostic_utils import collapse_ctc, text_counts
from verify_model import ROOT, UPSTREAM, EXPECTED_COMMIT, sha256, save_json, checked_state


def read_tensor(record: dict, size: int):
    """Validate the independently encoded input and apply the published normalization."""
    path = Path(record["path"])
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Input hash changed: {path}")
    cap, frames = cv2.VideoCapture(str(path)), []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != record["frames"]:
        raise ValueError("Decoded frame count differs from frozen manifest")
    pixels = np.stack(frames)
    if pixels.shape[1:] != (size, size, 3) or pixels.dtype != np.uint8:
        raise ValueError("Wrong input profile")
    x = torch.from_numpy(pixels).float()
    raw_mean = x.mean().item()
    x -= x.mean()
    raw_std = x.std(correction=1)
    if not torch.isfinite(raw_std) or raw_std <= 0:
        raise ValueError("Invalid input variance")
    x /= raw_std
    return x.permute(3, 0, 1, 2).unsqueeze(0).contiguous(), {
        "raw_mean": raw_mean, "raw_std": raw_std.item(), "normalized_std": x.std().item()}


def timed_call(function):
    """Time CUDA work synchronously; allocation excludes other processes."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, {"seconds": time.perf_counter() - start,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "inputs_step02" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "step02.json")
    args = parser.parse_args()
    manifest_path, output = args.manifest.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose another output path; preserve earlier diagnostic runs")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise ValueError("Expected pinned clean official code")
    plan_path = ROOT / "step02_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_path = ROOT / "results" / "step01.json"
    if sha256(old_path) != plan["step01_result_sha256"]:
        raise ValueError("Step01 result changed")
    old = {r["id"]: r for r in json.loads(old_path.read_text(encoding="utf-8"))["records"]}
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    from avsr.utils.model_builder import build_model
    from avsr.utils.search import SearchSequence
    from avsr.utils.korean_funcs import grp2char
    from vocabulary.utils import KsponSpeechVocabulary
    from corrected_search import CorrectedSearchSequence

    config = yaml.safe_load((UPSTREAM / "configs" / "v_inference.yaml").read_text(encoding="utf-8"))
    vocab = KsponSpeechVocabulary(unit="grapheme")
    if (len(vocab), vocab.unk_id, vocab.eos_id) != (72, 3, 2):
        raise ValueError("Wrong official token mapping")
    torch.manual_seed(2207)
    np.random.seed(2207)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    kwargs = {k: config[k] for k in inspect.signature(build_model).parameters if k in config}
    model = build_model(vocab_size=72, pad_id=0, pass_visual_frontend=False, **kwargs)
    checkpoint = ROOT / "weights" / "multiview_lip.pt"
    if sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("Official checkpoint changed")
    contract = checked_state(model, torch.load(checkpoint, map_location="cpu", weights_only=True))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the declared experiment")
    device = torch.device("cuda:0")
    model.to(device).eval()
    if type(model.encoder).__name__ != "VisualConformerEncoder":
        raise ValueError("Video-only encoder required")
    search_args = dict(model=model, vocab_size=72, pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                       method="hybrid", max_len=200, ctc_rate=0.6, mp_num=1)
    result = {"schema": "olkavs-step02-results-v1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan,
              "plan_sha256": sha256(plan_path), "input_manifest_sha256": sha256(manifest_path),
              "runner_sha256": sha256(__file__), "corrected_search_sha256": sha256(ROOT / "corrected_search.py"),
              "helpers_sha256": sha256(ROOT / "diagnostic_utils.py"), "upstream_commit": commit,
              "checkpoint_contract": contract, "checkpoint_sha256": sha256(checkpoint),
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "audio_used": False, "training_run": False, "records": [],
              "timing_scope": "One shared encoder+medium pass per input; decoder-only times are separate and not total webcam latency"}
    save_json(output, result)
    try:
        # Preparation manifest stores one record for each id/resolution pair.
        records = manifest["records"]
        for size in plan["profiles"]:
            for clip_id in plan["source_ids"]:
                matches = [r for r in records if r["id"] == clip_id and r["resolution"] == size]
                if len(matches) != 1:
                    raise ValueError(f"Expected one manifest record for {size}/{clip_id}")
                record = matches[0]
                if record["frames"] != old[clip_id]["frames"] or record["text"] != old[clip_id]["reference_for_trace_only"]:
                    raise ValueError("Frame count/reference changed between steps")
                print(f"Running {size}px {clip_id}", flush=True)
                video, norm = read_tensor(record, size)
                video = video.to(device)
                lengths = torch.tensor([video.shape[2]], device=device)
                with torch.inference_mode():
                    features, feature_timing = timed_call(lambda: model.medium(model.encoder(video, lengths, None, None)))
                    logits = model.decoder.ctcdecoder(inputs=features)
                    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite model outputs")
                    logits_cpu = logits[0].cpu()
                    probs = logits_cpu.softmax(-1)
                    path = logits_cpu.argmax(-1).tolist()
                    collapsed = collapse_ctc(path, blank_id=3)
                    stats = {"argmax_blank_fraction": path.count(3) / len(path),
                             "mean_blank_posterior": probs[:, 3].mean().item(),
                             "argmax_token_counts": dict(Counter(path)),
                             "first_eos_frame": path.index(2) if 2 in path else None,
                             "mean_entropy_nats": (-(probs * probs.clamp_min(1e-30).log()).sum(-1)).mean().item()}
                    outputs = {}
                    for method in plan["methods"]:
                        if method == "ctc_greedy":
                            tokens, timing = collapsed, None
                        else:
                            search = (SearchSequence if method == "upstream_hybrid" else CorrectedSearchSequence)(**search_args)
                            if method == "upstream_hybrid":
                                ids = torch.full((1, 200), 0, device=device, dtype=torch.long)
                                counts = torch.zeros(1, device=device, dtype=torch.long)
                                def decode_original():
                                    search._search(features, beam_size=1, D_end=-14, M_end=16, device=device,
                                                   batch_idx=0, batch_size=1, shared_outputs=[ids, counts])
                                    return ids, counts
                                (ids, counts), timing = timed_call(decode_original)
                            else:
                                (ids, counts), timing = timed_call(lambda: search.decode_features(features, beam_size=1, D_end=-14, M_end=16))
                            tokens = ids[0, :int(counts[0])].cpu().tolist()
                        token_tensor = torch.tensor(tokens, dtype=torch.long)
                        graphemes = vocab.label_to_string(token_tensor)
                        prediction = grp2char(graphemes).strip()
                        first_eos = tokens.index(2) if 2 in tokens else None
                        outputs[method] = {"token_ids": tokens, "first_eos_index": first_eos,
                                           "tokens_after_eos": tokens[first_eos + 1:] if first_eos is not None else [],
                                           "graphemes": graphemes, "prediction": prediction,
                                           "metrics": text_counts(record["text"], prediction), "decoder_timing": timing}
                    features_cpu = features[0].cpu().numpy()
                    feature_file = output.parent / f"{output.stem}_features" / f"{size}_{clip_id}.npz"
                    feature_file.parent.mkdir(exist_ok=True)
                    np.savez_compressed(feature_file, features=features_cpu, ctc_logits=logits_cpu.numpy())
                item = {"id": clip_id, "size": size, "input_sha256": record["sha256"],
                        "frames": record["frames"], "reference_for_metrics_only": record["text"],
                        "normalization": norm, "feature_timing": feature_timing, "feature_shape": list(features.shape),
                        "features_sha256": sha256(feature_file), "feature_file": str(feature_file),
                        "ctc_frame_path": path, "ctc_stats": stats, "outputs": outputs}
                if size == 224:
                    item["step01_upstream_token_ids_identical"] = outputs["upstream_hybrid"]["token_ids"] == old[clip_id]["output_token_ids"]
                result["records"].append(item)
                save_json(output, result)
                print(json.dumps({"size": size, "id": clip_id, "predictions": {m: o["prediction"] for m, o in outputs.items()}, "ctc_blank_fraction": stats["argmax_blank_fraction"]}, ensure_ascii=False), flush=True)
                del video, features, logits
        aggregate = defaultdict(lambda: dict(edits=0, reference_characters=0, edits_no_spaces=0, reference_characters_no_spaces=0, exact=0, clips=0))
        for item in result["records"]:
            for method, out in item["outputs"].items():
                entry = aggregate[f"{item['size']}/{method}"]
                metric = out["metrics"]
                entry["edits"] += metric["character_edits_with_spaces"]
                entry["reference_characters"] += metric["reference_characters_with_spaces"]
                entry["edits_no_spaces"] += metric["character_edits_without_spaces"]
                entry["reference_characters_no_spaces"] += metric["reference_characters_without_spaces"]
                entry["exact"] += int(metric["exact_sentence"])
                entry["clips"] += 1
        for entry in aggregate.values():
            entry["pooled_cer_percent"] = 100 * entry["edits"] / entry["reference_characters"]
            entry["pooled_cer_no_spaces_percent"] = 100 * entry["edits_no_spaces"] / entry["reference_characters_no_spaces"]
        result["aggregate"] = dict(aggregate)
        result["status"] = "completed"
        save_json(output, result)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(output, result)
        raise


if __name__ == "__main__":
    main()
