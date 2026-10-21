"""Capture the preparation scope and unchanged deployment/tokenizer contracts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    """Write a new snapshot without changing original experiment or service files."""
    try:
        stage = Path(__file__).resolve().parent
        root = stage.parents[1]
        output = stage / "results" / "scope_snapshot.json"
        if output.exists():
            raise FileExistsError(output)
        paths = [stage / "scope.json", Path(__file__).resolve()]
        paths += [root / item for item in (
            "backend/config.py", "backend/models/tokenizer.py",
            "backend/preprocessing/resampler.py", "backend/models/baseline.py",
            "backend/models/weights/baseline_jamo_v2.pt",
            "backend/api/infer.py", "backend/services/ws_handler.py",
            "frontend/src/pages/WebcamPage.jsx",
        )]
        entries = []
        for path in paths:
            if not path.is_file():
                if path == root / "backend/models/weights/baseline_jamo_v2.pt" and not path.exists():
                    entries.append({"path": str(path), "exists": False,
                                    "note": "Configured default checkpoint absent before this task."})
                    continue
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            entries.append({"path": str(path), "exists": True, "bytes": path.stat().st_size,
                            "sha256": digest.hexdigest()})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "schema": "offline-stage3-preparation-scope-snapshot-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": entries,
            "no_training_or_service_mutation_authorized_in_this_stage": True,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"snapshot": str(output), "artifacts": len(entries)}))
    except Exception as exc:
        raise RuntimeError("Unable to freeze stage3 scope") from exc


if __name__ == "__main__":
    main()
