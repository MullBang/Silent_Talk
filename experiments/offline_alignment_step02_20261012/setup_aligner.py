"""Fetch pinned public alignment artifacts; source audio is never uploaded."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MODEL_ROOT = ROOT / "models/alignment_audit"
MODEL = MODEL_ROOT / "wav2vec2-large-xlsr-korean"
REPO = "kresnik/wav2vec2-large-xlsr-korean"
FILES = ["model.safetensors", "config.json", "preprocessor_config.json", "vocab.json",
         "tokenizer_config.json", "special_tokens_map.config", "README.md"]
os.environ["HF_HOME"] = str(MODEL_ROOT / "hf_home")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def sha(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def save(path: Path, value: Any) -> None:
    """Create immutable setup evidence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Cannot write {path}") from exc


def main() -> None:
    """Download only safetensors and small processor metadata, then verify hashes."""
    try:
        from huggingface_hub import HfApi, snapshot_download
        repo_path = BASE / "aligner_repository.json"
        if repo_path.exists():
            metadata = json.loads(repo_path.read_text(encoding="utf-8"))
            if metadata["repo_id"] != REPO:
                raise ValueError("Repository identity changed")
        else:
            info = HfApi(token=False).model_info(REPO, files_metadata=True)
            files = [{"name": item.rfilename, "size": item.size, "git_blob_id": item.blob_id,
                      "lfs_sha256": item.lfs.sha256 if item.lfs else None}
                     for item in info.siblings if item.rfilename in FILES]
            if {item["name"] for item in files} != set(FILES):
                raise ValueError("Required public model files missing")
            metadata = {"repo_id": REPO, "revision": info.sha, "files": files,
                        "purpose": "audio alignment proposals only; not the visual student",
                        "source_audio_uploaded": False, "token": False}
            save(repo_path, metadata)
        print(json.dumps({"revision": metadata["revision"],
                          "download_bytes": sum(row["size"] for row in metadata["files"])}), flush=True)
        snapshot_download(REPO, revision=metadata["revision"], allow_patterns=FILES,
                          local_dir=MODEL, cache_dir=MODEL_ROOT / "hf_cache", token=False, max_workers=2)
        verified = []
        for item in metadata["files"]:
            path = MODEL / item["name"]
            value = sha(path)
            if path.stat().st_size != item["size"]:
                raise ValueError("Downloaded model size mismatch")
            if item["lfs_sha256"]:
                if value != item["lfs_sha256"]:
                    raise ValueError("Downloaded model LFS SHA mismatch")
            else:
                raw = path.read_bytes()
                if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != item["git_blob_id"]:
                    raise ValueError("Downloaded metadata Git object mismatch")
            verified.append({"path": str(path), "sha256": value, "bytes": item["size"]})
        save(BASE / "aligner_model_setup.json", {"status": "download_verified", "repository": metadata,
                                               "model_directory": str(MODEL), "files": verified,
                                               "original_environments_modified": False,
                                               "source_audio_uploaded": False})
        print("Verified alignment artifacts", flush=True)
    except Exception as exc:
        raise SystemExit(f"Alignment setup failed: {exc}") from exc


if __name__ == "__main__":
    main()
