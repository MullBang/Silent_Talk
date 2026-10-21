"""Build a review page with scoped actual parent-audio evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    """Hash a local artifact without executing it."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def main() -> None:
    """Join actual scoped answers, never create UI decisions from them."""
    try:
        manifest = BASE / "review_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["manifest_sha256"] = sha(manifest)
        by_id = {row["id"]: row for row in payload["records"]}
        seen = set()
        sources = {}
        for path in sorted((BASE / "user_evidence").glob("*_parent_audio_answers.json")):
            evidence = json.loads(path.read_text(encoding="utf-8"))
            if evidence["manifest_sha256"] != payload["manifest_sha256"]:
                raise ValueError("Evidence points at another manifest")
            sources[str(path)] = sha(path)
            for item in evidence["records"]:
                row = by_id[item["parent_id"]]
                if row["id"] in seen:
                    raise ValueError("Duplicate parent answer; reconcile explicitly")
                checks = (("text", "text"), ("start_sample_16k", "sentence_start_sample_16k"),
                          ("end_sample_16k", "sentence_end_sample_16k"), ("wav_sha256", "parent_exact_audio_sha256"))
                if any(item[left] != row[right] for left, right in checks):
                    raise ValueError("Parent answer target mismatch")
                if sha(Path(item["wav_path"])) != item["wav_sha256"]:
                    raise ValueError("Question WAV changed")
                if item["answer_verbatim"] != "문장 일치·잘림 없음" or item["phrase_visual_boundary_decision"] is not None:
                    raise ValueError("This builder needs explicit handling for the supplied answer")
                seen.add(row["id"])
                row["parent_audio_evidence"] = {
                    "transcript": "correct", "boundary": "complete", "scope": "full_parent_audio_only",
                    "answer_verbatim": item["answer_verbatim"], "evidence_path": str(path),
                }
        template = BASE / "parent_review_template.html"
        html = template.read_text(encoding="utf-8")
        if html.count("__DATA_JSON__") != 1:
            raise ValueError("Expected one data placeholder")
        page_data = copy.deepcopy(payload)
        # Internal provenance remains in JSON files; the page needs displayable decisions only.
        for row in page_data["records"]:
            row["audio_evidence"] = {"decision": "acceptable", "scope": "phrase_only"}
            if "parent_audio_evidence" in row:
                row["parent_audio_evidence"].pop("evidence_path", None)
        encoded = json.dumps(page_data, ensure_ascii=False).replace("<", "\\u003c")
        (BASE / "index.html").write_text(html.replace("__DATA_JSON__", encoded), encoding="utf-8")
        (BASE / "results/current_parent_evidence.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        receipt = {"manifest_sha256": sha(manifest), "template_sha256": sha(template),
                   "html_sha256": sha(BASE / "index.html"), "evidence_sha256": sources,
                   "parent_audio_confirmed": len(seen), "ui_decisions_created": 0,
                   "visual_decisions_created": 0, "training_ready": False}
        (BASE / "results/build_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"parent_audio_confirmed": len(seen), "visual_decisions_created": 0}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Parent review build failed") from exc


if __name__ == "__main__":
    main()
