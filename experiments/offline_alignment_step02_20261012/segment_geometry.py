"""Integer-sample geometry for saved CTC proposals; no neural inference."""
from __future__ import annotations
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
import re
from typing import Any
import numpy as np
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
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
        duration_samples = round((end - start) * RATE)
        if abs(duration_samples - (end - start) * RATE) > 1e-7:
            raise ValueError("Expected an integer acoustic sample interval")
        frames = round(Fraction(duration_samples * 25, RATE))
        times = [video_start + index / 25 for index in range(frames)]
        if frames < 0 or (times and times[-1] >= video_end):
            raise ValueError("Invalid proposal frame times")
        native = [round(value * row["video"]["native_fps"]) for value in times]
        if len(set(native)) != len(native) or (native and native[-1] >= row["video"]["native_frame_count"]):
            raise ValueError("Invalid proposal native frame indices")
        labels = visual_tokenizer.encode_text(text)
        minimum = visual_tokenizer.ctc_min_input_len(labels)
        context_origin = row["context_audio"]["actual_source_start_seconds"]
        return {"text": text, "local_audio_start": start, "local_audio_end": end,
                "wav_start": wav_start, "wav_end": wav_end,
                "video_start": video_start, "video_end": video_end,
                "start_in_context": wav_start - context_origin, "end_in_context": wav_end - context_origin,
                "duration_samples_16k": duration_samples, "frame_rounding": "exact rational half-even",
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

