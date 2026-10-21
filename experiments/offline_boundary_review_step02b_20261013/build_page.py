"""Build the latest hash-bound boundary review page, never user decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from validate_review import load_json, manifest_index

BASE = Path(__file__).resolve().parent


def main() -> None:
    """Bind a template to the revised manifest and record exact identities."""
    try:
        manifest_path = BASE / "review_manifest_r1.json"
        payload, sha = load_json(manifest_path)
        manifest_index(payload)
        for path, expected in payload["sources"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed source: {path}")
        payload["review_manifest_sha256"] = sha
        template_path = BASE / "review_template.html"
        template = template_path.read_text(encoding="utf-8")
        if template.count("__REVIEW_DATA__") != 1:
            raise ValueError("Expected one JSON placeholder")
        embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        output = BASE / "index.html"
        output.write_text(template.replace("__REVIEW_DATA__", embedded), encoding="utf-8")
        receipt = {"schema": "offline-boundary-review-build-v1", "manifest_sha256": sha,
                   "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
                   "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "training_ready": False, "user_decisions_created": 0}
        (BASE / "results/build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(receipt))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Unable to build review page") from exc


if __name__ == "__main__":
    main()
