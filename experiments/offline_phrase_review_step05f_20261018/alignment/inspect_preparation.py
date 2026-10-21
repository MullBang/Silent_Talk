"""Read-only local acoustic resource and fixed eight-text coverage inspection."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

from protocol import BASE, ROOT, OLD, acoustic_oov, binding, read, verify_binding, write


def main() -> None:
    """Create a non-executable preparation record without model or audio loading."""
    queue_path = ROOT / "experiments/offline_phrase_design_step05e_20261018/design/review_queue.json"
    handoff_path = ROOT / "experiments/offline_phrase_design_step05e_20261018/results/next_step_handoff.json"
    setup_path = OLD / "aligner_model_setup.json"
    queue, handoff, setup = read(queue_path), read(handoff_path), read(setup_path)
    rows = queue["records"][:8]
    if [r["candidate_id"] for r in rows] != handoff["first_batch"]["candidate_ids"]:
        raise ValueError("Frozen first-batch candidate identities changed")
    verified = []
    for spec in setup["files"]:
        verified.append(binding(verify_binding(spec)))
    model_dir = Path(setup["model_directory"])
    vocab = read(model_dir / "vocab.json")
    config = read(model_dir / "config.json")
    hop, field = 1, 1
    for kernel, stride in zip(config["conv_kernel"], config["conv_stride"], strict=True):
        field += (kernel - 1) * hop
        hop *= stride
    if (hop, field) != (320, 400) or config["add_adapter"] or len(vocab) != 1205 or vocab["[PAD]"] != 1204:
        raise ValueError("The cached acoustic model contract changed")
    import torch
    result = {
        "schema": "offline-stage5f-alignment-preparation-v1",
        "status": "prepared_not_executed_parent_reviews_and_bound_policy_required",
        "client_date": "2026-10-18", "sources": [binding(p) for p in (queue_path, handoff_path, setup_path)],
        "model_files_verified": verified, "model_directory": str(model_dir),
        "acoustic_model_repository": setup["repository"]["repo_id"],
        "acoustic_model_revision": setup["repository"]["revision"],
        "runtime": {"python": sys.executable, "cuda_available": torch.cuda.is_available(),
                    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "safetensors", "pytest")}},
        "records": [{"review_id": r["review_id"], "parent_id": r["parent_id"], "candidate_id": r["candidate_id"],
                     "canonical_parent_text_provisional": r["parent_text"], "selected_phrase_text_provisional": r["text"],
                     "parent_acoustic_oov": acoustic_oov(r["parent_text"], vocab),
                     "phrase_acoustic_oov": acoustic_oov(r["text"], vocab), "parent_transcript_confirmed": False,
                     "parent_audio_boundary_confirmed": False, "phrase_boundary_confirmed": False} for r in rows],
        "geometry": {"sample_rate": 16000, "hop_samples": hop, "receptive_field_samples": field,
                     "occupancy_bin_edge_sample": "320 * emission_index + 40",
                     "audio_origin": "exact native first_sample_48k / 48000 before separate resampling",
                     "video_time": "origin + local sample / 16000 + measured rational offset, exactly once",
                     "frame_count": None, "ctc_feasible": None,
                     "deferred_visual_extraction": "recording-origin 25fps rational D-grid, actual PTS nearest with ties earlier",
                     "legacy_segment_geometry_imported_or_executed": False},
        "future_input_contract": {
            "schema": "offline-stage5f-authorized-acoustic-plan-v1",
            "requires_separate_root_bound_policy": "acoustic_alignment_neural_calls_allowed = 8",
            "per_record": ["review_id", "candidate_id", "parent_id", "speaker_id", "canonical_parent_text", "canonical_phrase_text", "word_start", "word_end",
                           "human_parent_transcript_confirmed", "human_parent_audio_boundary_confirmed", "parent_review_evidence(path/sha256)",
                           "parent_audio(path/sha256/bytes/sample_rate/sample_count/source_origin_seconds_exact)",
                           "video_offset_seconds_exact(numerator/denominator)", "media_record_binding(path/sha256)"],
            "human_evidence_scope": "The trusted caller must bind actual answers to exact IDs/text/audio. JSON reviewer names alone are not authentication.",
            "canonical_correction_policy": "Preserve previous labels; explicitly bind corrected text and word span to actual human evidence in a new plan.",
        },
        "source_code": [binding(BASE / name) for name in ("protocol.py", "run_alignment.py", "inspect_preparation.py", "test_protocol.py")],
        "reused_helper": binding(OLD / "ctc_alignment.py"),
        "visual_tokenizer_unchanged": binding(ROOT / "backend/models/tokenizer.py"),
        "recommendation": "Collect actual parent transcript/audio-boundary answers before acoustic inference. Later transcript corrections may re-use identical saved posteriors with new bound DP-only policy.",
        "training_ready": False, "source_provisional": True, "new_human_approvals": 0,
        "neural_calls": 0, "model_tensor_loads": 0, "audio_reads": 0, "downloads": 0,
        "training_updates": 0, "actual_training_clips": 0,
    }
    write(BASE / "preparation.json", result)
    print("Alignment preparation saved; Q01-Q08 acoustic OOV = 0; neural calls = 0.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
