"""Build a hash-bound Stage2 review page without making review decisions.

Only the new stage's manifest and template are read. No audio, model, old label,
or prior review state is opened or changed by this builder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / "results" / "revision2" / "review_manifest.json"


def sha256(raw: bytes) -> str:
    """Return the exact byte identity used to bind local review evidence."""
    try:
        return hashlib.sha256(raw).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("Unable to hash review input") from exc


def finite_seconds(value: Any, field: str, nullable: bool = False) -> None:
    """Reject malformed or negative timestamps without inventing boundaries."""
    try:
        if value is None and nullable:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def media_url(value: Any, field: str) -> None:
    """Accept only same-server media paths, never remote or arbitrary files."""
    try:
        if value is None or value == "":
            return
        if not isinstance(value, str):
            raise ValueError("URL is not text")
        parsed = urlsplit(value)
        decoded = unquote(parsed.path)
        if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment
                or not decoded.startswith("/media/") or "\\" in decoded
                or "\x00" in decoded or any(p in {".", ".."} for p in decoded.split("/"))):
            raise ValueError("URL must be a simple /media/ path")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc


def validated_payload(raw: bytes) -> dict[str, Any]:
    """Validate display contracts and add hashes while retaining all source data."""
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError("Expected an object with records")
        if len(payload["records"]) != 16:
            raise ValueError("Stage2 expects exactly sixteen parent records")
        identifiers: set[str] = set()
        for row in payload["records"]:
            identity = row.get("id")
            if not isinstance(identity, str) or not identity or identity in identifiers:
                raise ValueError("Missing or duplicate parent ID")
            identifiers.add(identity)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Missing target text: {identity}")
            for field in ("context_audio_url", "context_video_url", "roi_video_url"):
                media_url(row.get(field), field)
            for field in ("sentence_start_in_context", "sentence_end_in_context"):
                finite_seconds(row.get(field), field)
            if row["sentence_end_in_context"] <= row["sentence_start_in_context"]:
                raise ValueError(f"Invalid sentence interval: {identity}")
            if not isinstance(row.get("words"), list):
                raise ValueError(f"Missing words: {identity}")
            surfaces = row["text"].split()
            if len(row["words"]) != len(surfaces):
                raise ValueError(f"Word count does not match target: {identity}")
            for index, (word, surface) in enumerate(zip(row["words"], surfaces)):
                if word.get("index") != index or word.get("surface") != surface:
                    raise ValueError(f"Word identity differs: {identity}/{index}")
                left, right = word.get("start_in_context"), word.get("end_in_context")
                finite_seconds(left, "start_in_context", nullable=True)
                finite_seconds(right, "end_in_context", nullable=True)
                if (left is None) != (right is None):
                    raise ValueError(f"One-sided boundary: {identity}/{index}")
                if left is not None and right < left:
                    raise ValueError(f"Reversed boundary: {identity}/{index}")
                # Zero-duration and overlapping candidates must remain visible for review.
            row["review_target_sha256"] = sha256(row["text"].encode("utf-8"))
        payload["review_manifest_sha256"] = sha256(raw)
        payload["review_interface_schema"] = "lipnet-offline-alignment-review-ui-v1"
        return payload
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Review manifest validation failed: {exc}") from exc


def main() -> None:
    """Write a review-only HTML artifact and a reproducible build receipt."""
    try:
        raw = MANIFEST.read_bytes()
        payload = validated_payload(raw)
        template_path = BASE / "review_template_r2.html"
        template_raw = template_path.read_bytes()
        template = template_raw.decode("utf-8")
        if template.count("__REVIEW_DATA__") != 1:
            raise ValueError("Template must contain exactly one data placeholder")
        embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
        output = BASE / "review_r2" / "index.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        html = template.replace("__REVIEW_DATA__", embedded)
        output.write_text(html, encoding="utf-8", newline="\n")
        receipt = {
            "schema": "lipnet-offline-alignment-review-build-v1",
            "manifest_path": str(MANIFEST), "manifest_sha256": sha256(raw),
            "template_sha256": sha256(template_raw),
            "builder_sha256": sha256(Path(__file__).read_bytes()),
            "html_path": str(output), "html_sha256": sha256(output.read_bytes()),
            "sentence_count": len(payload["records"]),
            "word_count": sum(len(r["words"]) for r in payload["records"]),
            "human_decisions_created_by_builder": 0, "training_ready": False,
            "origin_note": "Review export declarations require independently established user submission provenance.",
        }
        (BASE / "results" / "review_build_receipt_r2.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not build Stage2 review page: {exc}") from exc


if __name__ == "__main__":
    main()
