"""Prepared-only bounded acoustic runner; requires a later explicit bound plan.

This file does not create its own execution authorization. Parent transcripts and
their audio bounds must have actual review evidence before the future caller
constructs the execution plan described in preparation.json. No current neural
execution is implied by this file existing.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import wave

from protocol import (BASE, ROOT, OLD, RATE, HOP, FIELD, acoustic_oov, binding,
                      checked_text, fraction, load_ctc, read, selected_phrase,
                      sha, verify_binding, write)


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject missing evidence, scope expansion and changed sources before torch."""
    if plan.get("schema") != "offline-stage5f-authorized-acoustic-plan-v1":
        raise ValueError("A separate authorized acoustic execution plan is required")
    if plan.get("neural_execution_authorized") is not True:
        raise ValueError("Neural execution has not been authorized by the bound policy")
    policy = read(verify_binding(plan["execution_policy"]))
    if policy.get("acoustic_alignment_neural_calls_allowed") != 8:
        raise ValueError("Separate policy must explicitly permit exactly eight acoustic calls")
    if plan.get("model_input") != "annotation_audio_only" or plan.get("training_updates") != 0:
        raise ValueError("This runner only permits annotation audio inference")
    queue_path = verify_binding(plan["fixed_queue"])
    if sha(queue_path) != "f0c6813d5ea8d433328f301804820180dc5ea27f7f9a76f28c49af04ae0c746c":
        raise ValueError("Expected the frozen stage5e twenty-item queue")
    queue = read(queue_path)["records"][:8]
    rows = plan["records"]
    if len(rows) != 8 or [r["review_id"] for r in rows] != [r["review_id"] for r in queue]:
        raise ValueError("Only fixed Q01-Q08 are allowed, in their original order")
    for expected, row in zip(queue, rows, strict=True):
        for key in ("candidate_id", "parent_id", "speaker_id"):
            if row[key] != expected[key]:
                raise ValueError("A candidate identity changed")
        if row.get("human_parent_transcript_confirmed") is not True or row.get("human_parent_audio_boundary_confirmed") is not True:
            raise ValueError("Actual parent transcript and audio-boundary evidence is required")
        evidence = read(verify_binding(row["parent_review_evidence"]))
        if not evidence:
            raise ValueError("Empty evidence does not confirm a parent")
        checked_text(row["canonical_parent_text"])
        checked_text(row["canonical_phrase_text"])
        words = row["canonical_parent_text"].split()
        start, end = row["word_start"], row["word_end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(words):
            raise ValueError("Canonical phrase has invalid word indices")
        if " ".join(words[start:end]) != row["canonical_phrase_text"]:
            raise ValueError("Canonical phrase does not match its explicit word span")
        audio = row["parent_audio"]
        verify_binding(audio)
        if audio["sample_rate"] != RATE or type(audio["sample_count"]) is not int:
            raise ValueError("Only exact unpadded 16kHz PCM samples are accepted")
        if fraction(audio["source_origin_seconds_exact"]) < 0:
            raise ValueError("A source origin cannot be negative")
        fraction(row["video_offset_seconds_exact"])
        verify_binding(row["media_record_binding"])
    checked_code = {verify_binding(spec).resolve() for spec in plan["execution_code"]}
    required_code = {BASE / "run_alignment.py", BASE / "protocol.py", OLD / "ctc_alignment.py"}
    if not required_code <= checked_code:
        raise ValueError("The runner, geometry and frozen DP helper must all be bound")
    return rows


def main(plan_path: Path) -> None:
    """Run at most eight forward calls after complete source/evidence validation."""
    attempts = 0
    started = time.perf_counter()
    output_dir = BASE / "execution"
    if output_dir.exists():
        raise FileExistsError("Existing execution evidence will not be overwritten or resumed implicitly")
    plan = read(plan_path)
    rows = validate_plan(plan)
    setup_path = verify_binding(plan["model_setup"])
    setup = read(setup_path)
    for spec in setup["files"]:
        verify_binding(spec)
    model_dir = Path(setup["model_directory"])
    vocab, config = read(model_dir / "vocab.json"), read(model_dir / "config.json")
    if len(vocab) != 1205 or config["pad_token_id"] != vocab["[PAD]"] or config["add_adapter"]:
        raise ValueError("Unexpected cached acoustic model contract")
    hop, field = 1, 1
    for kernel, stride in zip(config["conv_kernel"], config["conv_stride"], strict=True):
        field += (kernel - 1) * hop
        hop *= stride
    if (hop, field) != (HOP, FIELD):
        raise ValueError("Unexpected acoustic convolution geometry")
    if any(acoustic_oov(r["canonical_parent_text"], vocab) for r in rows):
        raise ValueError("Unsupported acoustic syllable: hold before inference; never drop it")
    for row in rows:
        with wave.open(row["parent_audio"]["path"], "rb") as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) != (RATE, 1, 2, row["parent_audio"]["sample_count"]):
                raise ValueError("PCM header differs from the bound exact audio input")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1")
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    ctc = load_ctc()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed CUDA runtime is unavailable; do not substitute silently")
    output_dir.mkdir()
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True, use_safetensors=True,
                                             attn_implementation="eager").to("cuda:0").eval()
        reverse = {value: key for key, value in vocab.items()}
        outputs: list[dict[str, Any]] = []
        for row in rows:
            spec = row["parent_audio"]
            with wave.open(spec["path"], "rb") as stream:
                audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
            inputs = processor(audio, sampling_rate=RATE, return_tensors="pt", padding=False)
            attempts += 1
            with torch.inference_mode():
                logits = model(**{k: v.to("cuda:0") for k, v in inputs.items()}).logits[0]
                probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            if probs.shape != ((len(audio) - FIELD) // HOP + 1, len(vocab)) or not np.isfinite(probs).all():
                raise ValueError("Unexpected acoustic probability shape or values")
            if not np.allclose(np.exp(probs.astype(np.float64)).sum(axis=1), 1, atol=2e-5, rtol=0):
                raise ValueError("Posterior rows are not normalized")
            posterior = output_dir / (row["review_id"] + "_posterior.npz")
            np.savez_compressed(posterior, log_probs=probs)
            text = row["canonical_parent_text"]
            target = [vocab[ch] for ch in text.replace(" ", "|")]
            aligned = ctc.forced_align(probs.astype(np.float64), target, vocab["[PAD]"])
            greedy_ids = ctc.collapse_ctc(probs.argmax(axis=-1).tolist(), vocab["[PAD]"])
            phrase = selected_phrase(text, row["canonical_phrase_text"], row["word_start"], row["word_end"],
                                     aligned, len(audio), fraction(spec["source_origin_seconds_exact"]),
                                     fraction(row["video_offset_seconds_exact"]))
            result = {"review_id": row["review_id"], "parent_id": row["parent_id"],
                      "candidate_id": row["candidate_id"], "canonical_parent_text": text,
                      "plan": binding(plan_path), "audio": binding(Path(spec["path"])),
                      "posterior": binding(posterior), "posterior_shape": list(probs.shape),
                      "acoustic_greedy_ids": greedy_ids,
                      "acoustic_greedy_text": "".join(reverse[i] for i in greedy_ids).replace("|", " ").strip(),
                      "greedy_is_independent_human_confirmation": False,
                      "alignment": aligned, "selected_phrase": phrase,
                      "parent_review_evidence": row["parent_review_evidence"],
                      "human_approvals_added_by_this_execution": 0, "training_ready": False,
                      "source_provisional": True, "actual_split": None}
            write(output_dir / (row["review_id"] + "_alignment.json"), result)
            outputs.append(result)
        write(output_dir / "alignment_results.json", {
            "schema": "offline-stage5f-acoustic-results-v1", "status": "automatic_phrase_proposals_pending_review",
            "records": outputs, "neural_calls": attempts, "training_updates": 0, "visual_model_calls": 0,
            "human_approvals_added": 0, "training_ready": False, "new_training_clips": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy", "safetensors")},
        })
    except Exception as exc:
        write(output_dir / "failure.json", {"status": "failed_preserved", "neural_call_attempts": attempts,
                                            "error_type": type(exc).__name__, "error": str(exc),
                                            "training_updates": 0, "training_ready": False})
        raise RuntimeError("Acoustic candidate generation stopped; preserve evidence and do not approve boundaries") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    main(parser.parse_args().plan)
