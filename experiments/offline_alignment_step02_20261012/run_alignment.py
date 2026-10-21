"""Propose Korean acoustic word spans; never approve them as visual training labels."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import wave

import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from ctc_alignment import forced_align, collapse_ctc
from transcript_audit import sha, save, comparison_text

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
RATE = 16000


def tokenizer() -> Any:
    """Load the unchanged visual tokenizer without importing the service model."""
    try:
        spec = importlib.util.spec_from_file_location("visual_tokenizer", ROOT / "backend/models/tokenizer.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise RuntimeError("Cannot load visual label contract") from exc


def grid(config: dict[str, Any]) -> tuple[int, int]:
    """Calculate convolution stride and receptive field in audio samples."""
    try:
        step, field = 1, 1
        for kernel, stride in zip(config["conv_kernel"], config["conv_stride"], strict=True):
            field += (kernel - 1) * step
            step *= stride
        if (step, field) != (320, 400) or config["add_adapter"]:
            raise ValueError("Unexpected acoustic time grid")
        return step, field
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Unsupported acoustic output geometry") from exc


def make_segment(row: dict[str, Any], text: str, start: float, end: float,
                 visual_tokenizer: Any) -> dict[str, Any]:
    """Map a provisional acoustic interval to proposed video frame indices."""
    try:
        origin = row["sentence_audio"]["actual_source_start_seconds"]
        wav_start, wav_end = origin + start, origin + end
        video_start, video_end = wav_start + row["offset_seconds"], wav_end + row["offset_seconds"]
        frames = round((end - start) * 25)
        times = [video_start + index / 25 for index in range(frames)]
        if frames <= 0 or times[-1] >= video_end:
            raise ValueError("Invalid proposal frame times")
        native = [round(value * row["video"]["native_fps"]) for value in times]
        if len(set(native)) != len(native) or native[-1] >= row["video"]["native_frame_count"]:
            raise ValueError("Invalid proposal native frame indices")
        labels = visual_tokenizer.encode_text(text)
        minimum = visual_tokenizer.ctc_min_input_len(labels)
        context_origin = row["context_audio"]["actual_source_start_seconds"]
        return {"text": text, "local_audio_start": start, "local_audio_end": end,
                "wav_start": wav_start, "wav_end": wav_end,
                "video_start": video_start, "video_end": video_end,
                "start_in_context": wav_start - context_origin, "end_in_context": wav_end - context_origin,
                "proposed_frames_25fps": frames, "proposed_native_indices": native,
                "visual_token_ids": labels, "visual_ctc_min_frames": minimum,
                "ctc_feasible_proposal": frames >= minimum,
                "actual_training_clip_created": False, "human_transcript_approved": False,
                "human_audio_boundary_approved": False, "human_visual_context_approved": False,
                "manual_boundary_error_ms": None, "training_ready": False}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid provisional segment") from exc


def words_and_phrases(row: dict[str, Any], alignment: dict[str, Any], log_probs: np.ndarray,
                      step: int, field: int, visual_tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group character spans without inventing or expanding missing word boundaries."""
    try:
        words = []
        spans = alignment["token_spans"]
        # Bin edges are centered on the convolution receptive-field center.
        edge_offset = (field - step) / (2 * RATE)
        for index, match in enumerate(re.finditer(r"\S+", row["text"])):
            first, last = spans[match.start()], spans[match.end() - 1]
            start = first["start_frame"] * step / RATE + edge_offset
            end = last["end_frame"] * step / RATE + edge_offset
            if not 0 <= start < end <= row["sentence_audio"]["duration_seconds"]:
                raise ValueError("Word proposal outside sentence audio")
            chosen = [float(log_probs[t, span["token_id"]])
                      for span in spans[match.start():match.end()]
                      for t in range(span["start_frame"], span["end_frame"])]
            score = math.exp(sum(chosen) / len(chosen))
            segment = make_segment(row, match.group(), start, end, visual_tokenizer)
            flags = []
            if not segment["ctc_feasible_proposal"]:
                flags.append("insufficient_frames_for_visual_ctc")
            if score < .5:
                flags.append("low_acoustic_path_score")
            if start < .06 or row["sentence_audio"]["duration_seconds"] - end < .06:
                flags.append("near_sentence_crop_edge")
            words.append({"index": index, "surface": match.group(), "char_start": match.start(),
                          "char_end": match.end(), "acoustic_start_frame": first["start_frame"],
                          "acoustic_end_frame": last["end_frame"], "acoustic_path_score": score,
                          "score_is_calibrated_confidence": False, "flags": flags, **segment})
        phrases = []
        for size in (2, 3, 4):
            for index in range(len(words) - size + 1):
                group = words[index:index + size]
                text = row["text"][group[0]["char_start"]:group[-1]["char_end"]]
                phrases.append({"first_word_index": index, "word_count": size,
                                "char_start": group[0]["char_start"], "char_end": group[-1]["char_end"],
                                **make_segment(row, text, group[0]["local_audio_start"],
                                               group[-1]["local_audio_end"], visual_tokenizer)})
        return words, phrases
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot derive word/phrase proposals") from exc


def main() -> None:
    """Run fixed local acoustic inference and exact CTC Viterbi on supported labels."""
    try:
        media_path = BASE / "results/media_manifest.json"
        media = json.loads(media_path.read_text(encoding="utf-8"))
        if media["completed_records"] != 16:
            raise ValueError("Expected all sixteen media records")
        setup_path = BASE / "aligner_model_setup.json"
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        for spec in setup["files"]:
            if sha(Path(spec["path"])) != spec["sha256"]:
                raise ValueError("Acoustic model evidence changed")
        model_dir = Path(setup["model_directory"])
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        blank = config["pad_token_id"]
        if len(vocab) != config["vocab_size"] or blank != vocab["[PAD]"] or blank != 1204:
            raise ValueError("Acoustic vocabulary mismatch")
        step, field = grid(config)
        visual_tokenizer = tokenizer()
        plan = {"schema": "offline-word-alignment-plan-v1", "media_sha256": sha(media_path),
                "model_setup_sha256": sha(setup_path),
                "execution_code": {name: sha(BASE / name) for name in
                                   ("run_alignment.py", "ctc_alignment.py", "transcript_audit.py")},
                "visual_tokenizer_sha256": sha(ROOT / "backend/models/tokenizer.py"),
                "input": "sentence_audio only; no context-only neighboring speech",
                "record_ids": [r["id"] for r in media["records"]], "sample_rate": RATE,
                "acoustic_blank_id": blank, "acoustic_vocab_size": len(vocab),
                "hop_samples": step, "receptive_field_samples": field,
                "time_rule": "bin edges: t*320/16000 + (400-320)/(2*16000); these are proposals, not measured articulatory boundaries",
                "oov_policy": "preserve text; hold whole sentence, no wildcard or deletion",
                "device": "cuda:0", "dtype": "float32", "attention": "eager",
                "padding": False, "reference_text_supplied_to_acoustic_neural_network": False,
                "reference_supplied_to_viterbi_only": True, "visual_model_inference": False,
                "versions": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "numpy", "safetensors")},
                "human_approvals_added": 0, "training_updates": 0}
        plan_path = BASE / "alignment_execution_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise ValueError("Alignment execution plan changed")
        else:
            save(plan_path, plan)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        feature = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True,
                                             use_safetensors=True, attn_implementation="eager").to("cuda").eval()
        reverse = {value: key for key, value in vocab.items()}
        records = []
        for row in media["records"]:
            result_path = BASE / "results/alignment_records" / (row["id"] + ".json")
            if result_path.exists():
                prior = json.loads(result_path.read_text(encoding="utf-8"))
                if prior["plan_sha256"] != sha(plan_path):
                    raise ValueError("Existing alignment record changed")
                records.append(prior)
                continue
            spec = row["sentence_audio"]
            if sha(Path(spec["path"])) != spec["sha256"]:
                raise ValueError("Sentence audio changed")
            with wave.open(spec["path"], "rb") as stream:
                if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (RATE, 1, 2):
                    raise ValueError("Expected unpadded PCM16")
                audio = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
            started = time.perf_counter()
            inputs = feature(audio, sampling_rate=RATE, return_tensors="pt", padding=False)
            with torch.inference_mode():
                logits = model(**{key: value.to("cuda") for key, value in inputs.items()}).logits[0]
                log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            if log_probs.shape != ((len(audio) - field) // step + 1, len(vocab)) or not np.isfinite(log_probs).all():
                raise ValueError("Unexpected acoustic posterior shape/value")
            posterior = BASE / "results/posteriors" / (row["id"] + ".npz")
            posterior.parent.mkdir(parents=True, exist_ok=True)
            if posterior.exists():
                raise ValueError("Refusing to replace existing posterior")
            np.savez_compressed(posterior, log_probs=log_probs)
            greedy_ids = collapse_ctc(log_probs.argmax(axis=-1).tolist(), blank)
            greedy = "".join(reverse[index] for index in greedy_ids).replace("|", " ").strip()
            target_chars = row["text"].replace(" ", "|")
            oov = sorted({ch for ch in target_chars if ch not in vocab})
            output = {"id": row["id"], "text": row["text"], "plan_sha256": sha(plan_path),
                      "audio_sha256": spec["sha256"], "posterior": {"path": str(posterior), "sha256": sha(posterior)},
                      "posterior_shape": list(log_probs.shape), "acoustic_greedy_ids": greedy_ids,
                      "acoustic_greedy_text": greedy,
                      "greedy_no_space_match": comparison_text(greedy).replace(" ", "") == row["text"].replace(" ", ""),
                      "oov_characters": oov, "human_transcript_approved": False,
                      "training_ready": False, "manual_boundary_error_ms": None,
                      "words": [], "phrases": [], "alignment": None}
            if oov:
                output["status"] = "held_unsupported_acoustic_vocabulary"
            else:
                target = [vocab[ch] for ch in target_chars]
                output["alignment"] = forced_align(log_probs.astype(np.float64), target, blank)
                output["words"], output["phrases"] = words_and_phrases(row, output["alignment"], log_probs, step, field, visual_tokenizer)
                output["status"] = "automatic_alignment_proposed_not_human_verified"
            output["elapsed_seconds"] = time.perf_counter() - started
            save(result_path, output)
            records.append(output)
            print(json.dumps({"id": row["id"], "status": output["status"], "greedy": greedy,
                              "words": len(output["words"]), "feasible_words": sum(w["ctc_feasible_proposal"] for w in output["words"])}, ensure_ascii=False), flush=True)
        save(BASE / "results/alignment_results.json", {"status": "completed_automatic_alignment_proposals",
             "plan_sha256": sha(plan_path), "records": records, "human_approvals_added": 0,
             "training_ready": False, "training_updates": 0,
             "maximum_gpu_allocated_bytes": torch.cuda.max_memory_allocated()})
        del model
    except Exception as exc:
        raise RuntimeError("Automatic acoustic alignment failed; labels remain unapproved") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
