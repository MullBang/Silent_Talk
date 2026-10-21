"""Download only public ASR software/model artifacts; never read source audio."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MODEL_ROOT = ROOT / "models/asr_audit"
MODEL = MODEL_ROOT / "faster-whisper-large-v3"
REPO_ID = "Systran/faster-whisper-large-v3"
FILES = ["config.json", "model.bin", "tokenizer.json", "preprocessor_config.json", "vocabulary.json", "README.md"]
os.environ["HF_HOME"] = str(MODEL_ROOT / "hf_home")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    from huggingface_hub import HfApi, snapshot_download
    metadata_path = BASE / "asr_model_repository.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["repo_id"] != REPO_ID:
            raise ValueError("Existing repository binding differs")
    else:
        info = HfApi(token=False).model_info(REPO_ID, files_metadata=True)
        files = []
        for item in info.siblings:
            if item.rfilename not in FILES:
                continue
            files.append({"name": item.rfilename, "size": item.size, "git_blob_id": item.blob_id,
                          "lfs_sha256": item.lfs.sha256 if item.lfs else None})
        if {item["name"] for item in files} != set(FILES):
            raise ValueError("Required public model files are missing")
        metadata = {"schema": "step06-public-asr-repository-v1", "repo_id": REPO_ID, "revision": info.sha,
                    "recorded_utc": datetime.now(timezone.utc).isoformat(), "files": files,
                    "source": "https://huggingface.co/Systran/faster-whisper-large-v3",
                    "authentication": False, "source_audio_uploaded": False}
        save(metadata_path, metadata)
    print(json.dumps({"downloading_public_model_revision": metadata["revision"],
                      "bytes": sum(item["size"] for item in metadata["files"])}), flush=True)
    snapshot_download(REPO_ID, revision=metadata["revision"], allow_patterns=FILES, local_dir=MODEL,
                      cache_dir=MODEL_ROOT / "hf_cache", token=False, max_workers=2)
    verified = []
    for item in metadata["files"]:
        path = MODEL / item["name"]
        if path.stat().st_size != item["size"]:
            raise ValueError("Model file size mismatch: " + item["name"])
        digest = sha(path)
        if item["lfs_sha256"]:
            if digest != item["lfs_sha256"]:
                raise ValueError("Public LFS SHA mismatch: " + item["name"])
        else:
            content = path.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != item["git_blob_id"]:
                raise ValueError("Public Git blob mismatch: " + item["name"])
        verified.append({**bind(path), "bytes": item["size"], "public_digest_verified": True})
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    dll_dir = ROOT / ".venv-gpu/Lib/site-packages/torch/lib"
    handle = os.add_dll_directory(str(dll_dir))
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
    import ctranslate2
    supported = sorted(ctranslate2.get_supported_compute_types("cuda", 0))
    if "int8_float16" not in supported:
        raise ValueError("GPU does not support requested int8_float16")
    # Constructor validation only. No transcribe call and no audio is read.
    from faster_whisper import WhisperModel
    model = WhisperModel(str(MODEL), device="cuda", device_index=0, compute_type="int8_float16",
                         cpu_threads=4, num_workers=1, local_files_only=True)
    dlls = sorted(path for path in dll_dir.glob("*.dll") if path.name.startswith(("cublas", "cudnn", "cudart")))
    result = {"schema": "step06-local-asr-setup-v1", "status": "ready", "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "executable": sys.executable, "packages": packages,
              "repository": bind(metadata_path), "revision": metadata["revision"], "model_directory": str(MODEL),
              "model_files": verified, "model_bytes": sum(item["bytes"] for item in verified),
              "installation_report": bind(BASE / "asr_install_report.json"), "setup_code": bind(Path(__file__)),
              "cuda_dll_directory": str(dll_dir), "cuda_dlls": [bind(path) for path in dlls],
              "cuda_device_count": ctranslate2.get_cuda_device_count(), "supported_compute_types": supported,
              "loaded_compute_type": model.model.compute_type, "loaded_device": model.model.device,
              "model_loaded_successfully": True, "audio_read": False, "transcription_executed": False,
              "source_audio_uploaded": False, "original_environments_modified": False}
    save(BASE / "asr_setup.json", result)
    print(json.dumps({"status": result["status"], "revision": result["revision"], "model_bytes": result["model_bytes"],
                      "device": result["loaded_device"], "compute_type": result["loaded_compute_type"]}), flush=True)
    del model
    handle.close()


if __name__ == "__main__":
    main()
