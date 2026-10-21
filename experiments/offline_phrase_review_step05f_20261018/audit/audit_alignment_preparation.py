"""Audit prepared acoustic annotation metadata without loading model tensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import unicodedata

from audit_preflight import HERE, bind, check_binding, read_json


def inspect() -> dict[str, Any]:
    base = HERE.parent
    path = base / "alignment/preparation.json"
    preparation = read_json(path)
    for binding in preparation["sources"] + preparation["model_files_verified"] + preparation["source_code"]:
        check_binding(binding)
    check_binding(preparation["reused_helper"])
    check_binding(preparation["visual_tokenizer_unchanged"])
    model_dir = Path(preparation["model_directory"])
    config = read_json(model_dir / "config.json")
    vocab = read_json(model_dir / "vocab.json")
    assert len(vocab) == 1205
    hop = receptive = 1
    for kernel, stride in zip(config["conv_kernel"], config["conv_stride"], strict=True):
        receptive += (kernel - 1) * hop
        hop *= stride
    assert (hop, receptive) == (320, 400)
    assert config["pad_token_id"] == vocab["[PAD]"]
    assert preparation["geometry"]["hop_samples"] == hop
    assert preparation["geometry"]["receptive_field_samples"] == receptive
    queue = read_json(base / "selected_inputs.json")["records"]
    rows = preparation["records"]
    assert len(rows) == len(queue) == 8
    for row, pair in zip(rows, queue, strict=True):
        expected = pair["review"]
        assert row["review_id"] == expected["review_id"]
        assert row["parent_id"] == expected["parent_id"]
        assert row["candidate_id"] == expected["candidate_id"]
        assert row["canonical_parent_text_provisional"] == expected["parent_text"]
        assert row["selected_phrase_text_provisional"] == expected["text"]
        for text, key in [(expected["parent_text"], "parent_acoustic_oov"), (expected["text"], "phrase_acoustic_oov")]:
            assert text == unicodedata.normalize("NFC", text)
            assert sorted(set(text.replace(" ", "|")) - set(vocab)) == row[key] == []
        assert row["parent_transcript_confirmed"] is False
        assert row["parent_audio_boundary_confirmed"] is False
        assert row["phrase_boundary_confirmed"] is False
    assert preparation["geometry"]["frame_count"] is None
    assert preparation["geometry"]["ctc_feasible"] is None
    assert preparation["neural_calls"] == preparation["model_tensor_loads"] == preparation["audio_reads"] == 0
    assert preparation["training_updates"] == preparation["actual_training_clips"] == preparation["new_human_approvals"] == 0
    assert preparation["training_ready"] is False
    return {"schema": "offline-stage5f-independent-alignment-preparation-audit-v1", "status": "passed",
            "preparation": bind(path), "provisional_queue_texts_verified": 8,
            "acoustic_vocabulary_size": 1205, "unsupported_characters_in_fixed_original_eight": [],
            "hop_samples": hop, "receptive_field_samples": receptive,
            "original_pending_snapshot_preserved": True,
            "model_files_hashed_as_opaque_bytes": True,
            "new_model_tensor_loads": 0, "new_neural_calls": 0,
            "claim_limit": "Acoustic vocabulary support and executable preparation do not imply approved text, correct boundaries, successful alignment or visual CTC feasibility. Later canonical changes require rechecking.",
            "future_plan_requirement": "Bind actual human answers and exact selected original or revised audio/global source origin before any separately authorized forward calls."}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "alignment_preparation_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "rows": 8, "neural_calls": 0}))
