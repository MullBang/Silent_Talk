"""Run unprompted local Whisper on pilot audio as auxiliary transcript evidence."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import unicodedata
import wave

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def sha(path: Path) -> str:
    """Hash an input or output in bounded memory."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def save(path: Path, data: Any) -> None:
    """Create a new evidence file without overwriting previous results."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot save {path}") from exc


def comparison_text(value: str) -> str:
    """Normalize punctuation and whitespace for comparison only."""
    try:
        return " ".join("".join(" " if unicodedata.category(ch).startswith("P") else ch
                                for ch in unicodedata.normalize("NFC", value)).split())
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid comparison string") from exc


def main() -> None:
    """Audit both exact sentence and review context without a reference prompt."""
    try:
        media_path = BASE / "results/media_manifest.json"
        media = json.loads(media_path.read_text(encoding="utf-8"))
        if len(media["records"]) != 16 or media["completed_records"] != 16:
            raise ValueError("All sixteen media records must be ready")
        old = ROOT / "experiments/new_data_step06_20261007"
        setup = json.loads((old / "asr_setup.json").read_text(encoding="utf-8"))
        old_plan = json.loads((old / "asr_plan.json").read_text(encoding="utf-8"))
        options = old_plan["transcribe_options"]
        if any(options[key] is not None for key in ("initial_prompt", "prefix", "hotwords")) or options["word_timestamps"]:
            raise ValueError("Unprompted transcription contract changed")
        for spec in setup["model_files"]:
            if sha(Path(spec["path"])) != spec["sha256"]:
                raise ValueError("Existing ASR model changed")
        plan = {"schema": "offline-alignment-asr-plan-v1", "media_sha256": sha(media_path),
                "producer_sha256": sha(Path(__file__)), "options": options,
                "model_setup_sha256": sha(old / "asr_setup.json"), "record_ids": [r["id"] for r in media["records"]],
                "input_types": ["sentence_audio", "context_audio"], "source_audio_uploaded": False,
                "reference_text_supplied_to_model": False, "word_timing_used": False}
        plan_path = BASE / "asr_execution_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
                raise ValueError("ASR plan changed")
        else:
            save(plan_path, plan)
        dll_dir = ROOT / ".venv-gpu/Lib/site-packages/torch/lib"
        dll_handle = os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
        from faster_whisper import WhisperModel
        model = WhisperModel(setup["model_directory"], device="cuda", device_index=0,
                             compute_type="int8_float16", cpu_threads=4, num_workers=1, local_files_only=True)
        records = []
        for row in media["records"]:
            result_path = BASE / "results/asr_records" / (row["id"] + ".json")
            if result_path.exists():
                prior = json.loads(result_path.read_text(encoding="utf-8"))
                if prior["plan_sha256"] != sha(plan_path):
                    raise ValueError("Existing ASR record plan changed")
                records.append(prior)
                continue
            output = {"id": row["id"], "reference_text": row["text"], "plan_sha256": sha(plan_path),
                      "human_transcript_approved": False, "training_ready": False, "transcriptions": {}}
            for kind in plan["input_types"]:
                spec = row[kind]
                if sha(Path(spec["path"])) != spec["sha256"]:
                    raise ValueError("Input audio changed")
                with wave.open(spec["path"], "rb") as stream:
                    if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (16000, 1, 2):
                        raise ValueError("Expected 16 kHz mono PCM16")
                    samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
                started = time.perf_counter()
                segments, info = model.transcribe(samples, **options)
                values = [asdict(segment) for segment in segments]
                text = "".join(segment["text"] for segment in values).strip()
                ref, pred = comparison_text(row["text"]), comparison_text(text)
                output["transcriptions"][kind] = {"audio_sha256": spec["sha256"], "text": text,
                    "normalized_text": pred, "exact_normalized_match": pred == ref,
                    "no_space_match": pred.replace(" ", "") == ref.replace(" ", ""),
                    "segments": values, "elapsed_seconds": time.perf_counter() - started,
                    "language": info.language, "language_probability": info.language_probability}
            save(result_path, output)
            records.append(output)
            print(json.dumps({"id": row["id"], "sentence": output["transcriptions"]["sentence_audio"]["text"],
                              "context": output["transcriptions"]["context_audio"]["text"]}, ensure_ascii=False), flush=True)
        save(BASE / "results/asr_results.json", {"status": "completed_automatic_transcript_audit",
            "plan_sha256": sha(plan_path), "records": records, "human_approvals_added": 0,
            "training_updates": 0, "source_audio_uploaded": False})
        del model
        dll_handle.close()
    except Exception as exc:
        raise RuntimeError("Unprompted transcript audit failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
