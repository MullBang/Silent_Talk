"""Isolated OLKAVS raw-video smoke test; never imports project service/audio code.

Run with .venv-olkavs/Scripts/python.exe from any directory. Upstream is pinned
and unmodified. This is a feasibility runner, not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
EXPECTED_COMMIT = "af2feda3e3ac91ff3fef2454fcee662366ce9f4b"


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_state(model, state):
    """Reject partial/incompatible checkpoints before any model mutation."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint must be a nonempty direct state dictionary")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Checkpoint contains non-tensor state entries")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatch = {k: [list(state[k].shape), list(expected[k].shape)]
                for k in set(expected) & set(state) if state[k].shape != expected[k].shape}
    if missing or unexpected or mismatch:
        raise ValueError(json.dumps(dict(missing=missing, unexpected=unexpected, shape_mismatch=mismatch)))
    # No prefix stripping, allowlisting custom pickle globals, or strict=False.
    model.load_state_dict(state, strict=True)
    return {"strict": True, "state_entries": len(state), "missing": [], "unexpected": [], "shape_mismatch": {}}


def normalize_video(frames):
    if frames.ndim != 4 or frames.shape[1:] != (224, 224, 3) or frames.dtype != np.uint8:
        raise ValueError("Expected T x 224 x 224 x 3 uint8 BGR video")
    video = torch.from_numpy(frames).float()
    mean = video.mean()
    video -= mean
    std = video.std(correction=1)
    if not torch.isfinite(std) or std <= 0:
        raise ValueError("Invalid or constant video")
    video /= std
    tensor = video.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    # T is dimension 2 after B,C,T,H,W conversion (upstream single_infer uses C).
    lengths = torch.tensor([tensor.shape[2]], dtype=torch.long)
    return tensor, lengths, {"raw_mean": mean.item(), "raw_std_ddof1": std.item(),
                             "normalized_mean": video.mean().item(), "normalized_std": video.std().item()}


def read_video(record):
    path = Path(record["path"])
    if record["split"] != "train" or sha256(path) != record["sha256"]:
        raise ValueError("Input split/hash mismatch")
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != record["frames"]:
        raise ValueError("Decoded frame count mismatch")
    return normalize_video(np.stack(frames))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "step01.json")
    args = parser.parse_args()
    if not 1 <= args.limit <= 4:
        parser.error("limit must be 1..4")
    output = args.output.resolve()
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if commit != EXPECTED_COMMIT or dirty:
        raise ValueError("Expected pinned, unmodified upstream repository")
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)  # Official vocabulary uses a relative CSV path.
    from avsr.utils.model_builder import build_model
    from avsr.utils.search import SearchSequence
    from avsr.utils.korean_funcs import grp2char
    from vocabulary.utils import KsponSpeechVocabulary

    config_path = UPSTREAM / "configs" / "v_inference.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["architecture"] == "video" and config["raw_video"] is True
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    vocab = KsponSpeechVocabulary(unit="grapheme")
    assert (len(vocab), vocab.pad_id, vocab.sos_id, vocab.eos_id, vocab.unk_id) == (72, 0, 1, 2, 3)
    kwargs = {name: config[name] for name in inspect.signature(build_model).parameters if name in config}
    model = build_model(vocab_size=len(vocab), pad_id=vocab.pad_id,
                        pass_visual_frontend=False, **kwargs)
    weight = ROOT / "weights" / "multiview_lip.pt"
    result = {"schema": "olkavs-step01-runtime-v1", "status": "checking", "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "video-only runtime feasibility; not a generalization/accuracy evaluation",
              "upstream_commit": commit, "upstream_clean": True,
              "checkpoint": {"path": str(weight), "size_bytes": weight.stat().st_size, "sha256": sha256(weight),
                             "source": "https://drive.google.com/file/d/1nFLZHVwKgYrds-JJbFDJPlu5DJ1jeubC/view",
                             "published_author_checksum_available": False},
              "config_sha256": sha256(config_path), "config": config,
              "runner_sha256": sha256(__file__), "vocab_sha256": sha256(UPSTREAM / "vocabulary" / "kor_grapheme.csv"),
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "numpy", "PyYAML", "opencv-python-headless"]},
              "python": sys.version, "audio_used": False, "training_run": False,
              "actual_num_mp": 1, "dtype": "float32", "batch_size": 1, "records": []}
    try:
        state = torch.load(weight, map_location="cpu", weights_only=True)
        result["checkpoint_contract"] = checked_state(model, state)
        result["parameter_count"] = sum(p.numel() for p in model.parameters())
        result["encoder_class"] = type(model.encoder).__name__
        del state
        if args.check_only:
            result["status"] = "strict_load_pass"
            save_json(output, result)
            print(json.dumps({k: result[k] for k in ["status", "parameter_count", "checkpoint_contract"]}, indent=2), flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; do not silently substitute CPU timing")
        device = torch.device("cuda:0")
        result["gpu"] = {"name": torch.cuda.get_device_name(0), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                         "cuda_runtime": torch.version.cuda}
        model.to(device).eval()
        assert type(model.encoder).__name__ == "VisualConformerEncoder"
        assert not any("audio" in n.lower() for n, _ in model.named_modules())
        search = SearchSequence(model, len(vocab), method=config["search_method"], pad_id=vocab.pad_id,
                                sos_id=vocab.sos_id, eos_id=vocab.eos_id, unk_id=vocab.unk_id,
                                max_len=config["max_len"], ctc_rate=config["ctc_rate"], mp_num=1)
        manifest_path = ROOT / "inputs" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result["input_manifest_sha256"] = sha256(manifest_path)
        result["public_pretraining_overlap_unknown"] = True
        result["timing_scope"] = "GPU-ready tensor to hybrid token sequence; excludes source crop, video decode, host transfer, model load; first call recorded separately"

        def infer_once(video, lengths):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                # No waveform/dummy audio, audio transform, microphone, or transcript passed.
                ids, out_lengths = search(video, lengths, None, None, beam_size=config["beam_size"],
                                          D_end=config["EndDetect_D"], M_end=config["EndDetect_M"], device=device, mp_num=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            return ids.cpu(), out_lengths.cpu(), {"seconds": elapsed,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

        for i, record in enumerate(manifest["records"][:args.limit]):
            print(f"Running {i+1}/{args.limit}: {record['id']}", flush=True)
            start = time.perf_counter()
            video, lengths, stats = read_video(record)
            read_seconds = time.perf_counter() - start
            video, lengths = video.to(device), lengths.to(device)
            if i == 0:
                cold_ids, cold_lengths, cold_timing = infer_once(video, lengths)
                result["first_call"] = {"input_id": record["id"], **cold_timing}
            ids, out_lengths, timing = infer_once(video, lengths)
            count = int(out_lengths[0])
            if not 0 <= count <= config["max_len"]:
                raise ValueError("Decoder returned invalid length")
            selected = ids[0, :count]
            graphemes = vocab.label_to_string(selected)
            text = grp2char(graphemes).strip()
            item = {"id": record["id"], "input_sha256": record["sha256"], "frames": int(lengths[0]),
                    "clip_seconds": record["frames"] / record["fps"], "tensor_shape": list(video.shape),
                    "normalization": stats, "decode_and_normalize_seconds": read_seconds,
                    "output_token_length": count, "output_token_ids": selected.tolist(), "graphemes": graphemes,
                    "eos_present_in_selected_ids": bool((selected == vocab.eos_id).any()),
                    "prediction": text, "reference_for_trace_only": record["text"], "timing": timing,
                    "model_real_time_factor": timing["seconds"] / (record["frames"] / record["fps"])}
            if i == 0:
                item["repeat_token_ids_identical"] = bool(torch.equal(cold_ids, ids) and torch.equal(cold_lengths, out_lengths))
            result["records"].append(item)
            result["status"] = "running"
            save_json(output, result)
            print(json.dumps({"id": item["id"], "prediction": text, "timing": timing}, ensure_ascii=False), flush=True)
            del video, lengths
        result["status"] = "completed"
        result["nonempty_outputs"] = sum(bool(r["prediction"].strip()) for r in result["records"])
        save_json(output, result)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(output, result)
        raise


if __name__ == "__main__":
    main()
