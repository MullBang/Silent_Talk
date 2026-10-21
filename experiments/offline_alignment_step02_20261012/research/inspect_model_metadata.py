"""Read small public aligner metadata and check the fixed Step 1 text vocabulary."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MODEL = "kresnik/wav2vec2-large-xlsr-korean"
QUEUE = ROOT / "experiments/offline_curriculum_step01_20261012/results/alignment_review_queue.json"


def read_remote(url: str) -> bytes:
    """Fetch at most one MiB of public JSON metadata, never model weights."""
    try:
        with urlopen(Request(url, headers={"User-Agent": "lipnet-alignment-metadata-audit"}), timeout=30) as response:
            data = response.read(1048577)
        if len(data) > 1048576:
            raise ValueError("Metadata exceeds the one MiB safety limit")
        json.loads(data)
        return data
    except Exception as exc:
        raise RuntimeError(f"Could not read public metadata: {url}") from exc


def save_new(path: Path, data: bytes) -> None:
    """Preserve existing evidence and save only absent or byte-identical metadata."""
    try:
        if path.exists():
            if path.read_bytes() != data:
                raise FileExistsError(f"Refusing to replace different evidence: {path}")
            return
        path.write_bytes(data)
    except Exception as exc:
        raise RuntimeError(f"Could not preserve metadata: {path}") from exc


def main() -> None:
    """Pin one revision and report every queue character not in the aligner vocabulary."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        api_url = f"https://huggingface.co/api/models/{MODEL}?blobs=true"
        api_bytes = read_remote(api_url)
        api = json.loads(api_bytes)
        revision = api["sha"]
        files: dict[str, dict] = {}
        payloads: dict[str, dict] = {}
        for name in ("vocab.json", "config.json", "preprocessor_config.json", "tokenizer_config.json"):
            url = f"https://huggingface.co/{MODEL}/resolve/{revision}/{name}"
            data = read_remote(url)
            save_new(OUT / name, data)
            payloads[name] = json.loads(data)
            files[name] = {"url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        save_new(OUT / "model_api.json", api_bytes)
        vocab = payloads["vocab.json"]
        queue_bytes = QUEUE.read_bytes()
        records = json.loads(queue_bytes)
        per_record = []
        for row in records:
            text = row["text"]
            align_text = text.replace(" ", "|")
            missing = sorted(set(align_text) - set(vocab))
            per_record.append({"id": row["id"], "text": text, "alignment_text": align_text,
                               "characters": len(align_text), "oov_characters": missing,
                               "oov_positions": [{"index": i, "character": c} for i, c in enumerate(align_text) if c not in vocab],
                               "token_ids": [vocab[c] for c in align_text] if not missing else None})
        summary = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL, "revision": revision, "license": api.get("cardData", {}).get("license"),
            "api_url": api_url, "api_sha256": hashlib.sha256(api_bytes).hexdigest(),
            "metadata_files": files,
            "weight_metadata_only": [s for s in api.get("siblings", []) if s.get("rfilename") == "model.safetensors"],
            "queue_path": str(QUEUE.relative_to(ROOT)), "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
            "record_count": len(per_record), "records_with_oov": sum(bool(r["oov_characters"]) for r in per_record),
            "oov_characters_all": sorted({c for r in per_record for c in r["oov_characters"]}),
            "vocab_size": len(vocab),
            "nfc_hangul_syllable_tokens": sum(len(c) == 1 and "\uac00" <= c <= "\ud7a3" for c in vocab),
            "non_syllable_tokens": {c: i for c, i in vocab.items() if not (len(c) == 1 and "\uac00" <= c <= "\ud7a3")},
            "config_blank_id": payloads["config.json"]["pad_token_id"],
            "sample_rate": payloads["preprocessor_config.json"]["sampling_rate"],
            "records": per_record, "model_weights_downloaded": False, "model_inference_executed": False,
            "human_approval_added": False, "training_ready": False,
        }
        save_new(OUT / "vocabulary_audit.json", (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        print(json.dumps({k: summary[k] for k in ("revision", "license", "record_count", "records_with_oov", "oov_characters_all", "vocab_size", "nfc_hangul_syllable_tokens", "non_syllable_tokens", "config_blank_id", "sample_rate", "weight_metadata_only")}, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise RuntimeError("Aligner metadata audit did not complete") from exc


if __name__ == "__main__":
    main()
