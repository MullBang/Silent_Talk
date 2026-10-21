"""Strict evaluation-only access to the three frozen development phrase caches.

An execution-validated new policy binds the fixed stage5d development probe
and its separate plan. Source approvals remain unchanged; this module executes
no model. Raw media/audio/TAR paths are provenance only. This is a new copy of
the frozen stage5c loader with a new constructor and explicit evaluation-only scope.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata
import zipfile

import numpy as np

HANDOFF_SCHEMA = "offline-stage3-paired-cache-handoff-v1"
CACHE_SCHEMA = "lipnet-recording-grid-D-rgb-provisional-v1"
POLICY_SCHEMA = "lipnet-stage5d-evaluation-only-data-policy-v1"
SPLIT_SCHEMA = "lipnet-offline-dataset-stage3-split-plan-v1"
MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)
NPZ_KEYS = {"frames_rgb", "label", "length", "metadata_json", "schema", "training_ready", "provisional"}


class DataContractError(ValueError):
    """A bound file or requested use violates the controlled pilot contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataContractError(message)


def file_sha256(path: Path) -> str:
    """Hash a local artifact with bounded memory; never follow a media reference."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise DataContractError(f"Cannot hash input: {path}") from exc


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: str | bytes) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(DataContractError(f"Nonfinite JSON: {value}")))
    except (ValueError, UnicodeError) as exc:
        raise DataContractError("Invalid strict JSON") from exc


def _checked_json(path: Path, expected: str) -> dict[str, Any]:
    require(path.suffix.lower() == ".json", "A JSON metadata input is required")
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
            "An external exact SHA256 is required")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DataContractError(f"Cannot read metadata: {path}") from exc
    require(hashlib.sha256(raw).hexdigest() == expected, f"Metadata SHA mismatch: {path}")
    value = _json(raw)
    require(isinstance(value, dict), "Metadata root must be an object")
    return value


def _ref(ref: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(ref, dict) and {"path", "sha256"} <= set(ref), "Missing bound file reference")
    path = Path(ref["path"]).resolve()
    if "bytes" in ref:
        require(path.stat().st_size == ref["bytes"], f"Metadata file size differs: {path}")
    return _checked_json(path, ref["sha256"])


def _text_key(text: str) -> str:
    return "".join(unicodedata.normalize("NFC", text).split())


def _ctc_min(labels: Sequence[int]) -> int:
    return len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))


def _normalization(value: dict[str, Any]) -> None:
    require(value["stored"] == "uint8 RGB without normalization", "Wrong stored normalization")
    require(value["mean"] == [.485, .456, .406] and value["std"] == [.229, .224, .225],
            "Normalization parameters differ from the fixed RGB contract")


class Dataset:
    """Validated CPU arrays for one explicit split and externally bound policy.

    Constructor hashes stage3 frozen artifacts and checks all30 sidecars. Each
    load rereads and hashes only its NPZ and sidecar. Existing ready=false and
    provisional=true remain intact. Only three fixed dev phrases may be
    evaluated. Train and full-sentence NPZ arrays are never loaded.
    """

    def __init__(self, plan_path: str | Path, policy_path: str | Path,
                 expected_policy_sha256: str) -> None:
        self.split = "dev"
        plan_path, policy_path = Path(plan_path).resolve(), Path(policy_path).resolve()
        # The runner supplies its independently fixed policy hash.
        self.policy = _checked_json(policy_path, expected_policy_sha256)
        self.policy_sha256 = expected_policy_sha256
        require(Path(self.policy["evaluation_plan"]["path"]).resolve() == plan_path,
                "Requested execution plan differs from policy binding")
        self.handoff_path = Path(self.policy["handoff"]["path"]).resolve()
        expected_handoff_sha256 = self.policy["handoff"]["sha256"]
        self.handoff_sha256 = expected_handoff_sha256
        policy = self.policy
        require(policy["schema"] == POLICY_SCHEMA and policy["mode"] == "evaluation_only", "Wrong controlled-pilot policy")
        authorization = policy["authorization"]
        require(authorization["source"] == "current_user_request" and
                isinstance(authorization["request_text"], str) and bool(authorization["request_text"].strip()) and
                authorization["scope"] == "stage5d_fixed_unseen_phrase_evaluation", "Missing scoped user-request provenance")
        for key in ("allow_provisional_cache_for_controlled_evaluation", "preserve_provisional_flags"):
            require(policy[key] is True, f"Explicit controlled permission is missing: {key}")
        require(policy["optimizer_updates_authorized"] is False, "Optimizer updates are forbidden in stage5d")
        require(policy["service_change_authorized"] is False, "Service changes exceed stage5d scope")
        require(Path(policy["handoff"]["path"]).resolve() == self.handoff_path and
                policy["handoff"]["sha256"] == expected_handoff_sha256, "Policy/handoff binding differs")
        self.evaluation_plan = _ref(policy["evaluation_plan"])
        self.handoff = _checked_json(self.handoff_path, expected_handoff_sha256)
        handoff = self.handoff
        require(handoff["schema"] == HANDOFF_SCHEMA and handoff["cache_schema"] == CACHE_SCHEMA, "Wrong handoff/cache schema")
        require(handoff["clip_count"] == 30 and handoff["parent_count"] == 15, "Expected fixed paired30/parent15 scope")
        require(handoff["training_ready"] is False and handoff["new_original_pixels_human_approved"] is False,
                "Do not rewrite provisional source approval flags")
        _normalization(handoff["normalization"])
        tokenizer = handoff["tokenizer"]
        require(tokenizer["num_classes"] == 69 and tokenizer["version"] == "ko-jamo-v2" and
                tokenizer["normalization_version"] == "ko-text-v1" and
                tokenizer["vocab_sha256"] == "3ab81291a1892a4aece2d41bce634ff54663c4b93e3d56aef81f6783c822ff62",
                "Wrong frozen 69-class tokenizer")
        require(file_sha256(Path(tokenizer["path"])) == tokenizer["sha256"], "Tokenizer source hash changed")
        frozen = _ref(policy["stage3_frozen_manifest"])
        require(frozen["schema"] == "offline-stage3-completion-snapshot-v1" and frozen["training_ready"] is False,
                "Wrong frozen stage3 completion manifest")
        frozen_by_path = {}
        for artifact in frozen["artifacts"]:
            path = Path(artifact["path"]).resolve()
            require(str(path) not in frozen_by_path, "Duplicate frozen artifact path")
            require(path.stat().st_size == artifact["bytes"] and file_sha256(path) == artifact["sha256"],
                    f"Frozen stage3 artifact changed: {path}")
            frozen_by_path[str(path)] = artifact["sha256"]
        require(frozen_by_path.get(str(self.handoff_path)) == expected_handoff_sha256,
                "Handoff is not in the externally bound frozen snapshot")
        split_plan = _ref(handoff["split_plan"])
        require(split_plan["schema"] == SPLIT_SCHEMA and split_plan["training_ready"] is False, "Wrong paired split plan")
        self.split_plan = split_plan
        self.extraction = _ref(handoff["extraction"])
        for evidence in handoff["audits"] + [handoff["technical_visual_evidence"]]:
            _ref(evidence)
        parents = {row["parent_id"]: row for row in split_plan["records"]}
        require(len(parents) == 15 and len(split_plan["records"]) == 15, "Duplicate/missing split parent")
        require({r["parent_id"]: r["split"] for r in parents.values()} == split_plan["parent_to_split"], "Split map differs")
        require(sum(r["split"] == "train" for r in parents.values()) == 12 and
                sum(r["split"] == "dev" for r in parents.values()) == 3, "Expected parent12/3 split")
        # Recheck leakage keys instead of trusting a passed-audit flag alone.
        key_splits = {}
        for parent in parents.values():
            for key in [(name, parent[name]) for name in ("source_stem", "angle_independent_recording_key", "video_sha256", "label_sha256")] + [
                ("text", _text_key(text)) for text in (parent["parent_text"], parent["historical_parent_text"], parent["phrase"]["text"])]:
                require(key not in key_splits or key_splits[key] == parent["split"], "Cross-split identity/full-text leakage")
                key_splits[key] = parent["split"]
        self._rows = {}
        self._metadata = {}
        pairs = defaultdict(list)
        cache_root = (self.handoff_path.parent.parent / "extraction/cache").resolve()
        for row in handoff["records"]:
            ident = row["clip_id"]
            require(ident not in self._rows, "Duplicate handoff clip")
            require(row["parent_id"] in parents and row["kind"] in {"phrase", "full_sentence"}, "Unknown parent/clip kind")
            parent = parents[row["parent_id"]]
            require(ident == row["parent_id"] + "__" + row["kind"], "Clip identity convention changed")
            require(row["split"] == parent["split"], "Handoff split differs from paired parent plan")
            require(row["provisional"] is True and row["training_ready"] is False, "Provisional source flag changed")
            require(row["cache_integrity_audit_passed"] is True and row["technical_sample_QC_passed"] is True,
                    "Required technical checks are absent")
            for path_key, hash_key, suffix in [("npz_path", "npz_sha256", ".npz"), ("metadata_path", "metadata_sha256", ".json")]:
                path = Path(row[path_key]).resolve()
                require(path.parent == cache_root and path.name == ident + suffix, "Cache path escapes fixed cache directory")
                require(frozen_by_path.get(str(path)) == row[hash_key], "Cache is not bound by frozen snapshot")
            meta = _ref({"path": row["metadata_path"], "sha256": row["metadata_sha256"]})
            self._validate_metadata(row, meta, parent)
            self._rows[ident], self._metadata[ident] = row, meta
            pairs[row["parent_id"]].append(row)
        require(len(self._rows) == 30 and set(pairs) == set(parents), "Handoff must contain all30 paired clips")
        require(all(len(group) == 2 and {r["kind"] for r in group} == {"phrase", "full_sentence"} and
                    len({r["split"] for r in group}) == 1 for group in pairs.values()), "Incomplete or split parent/phrase pair")
        optimizer = policy["allowed_optimizer_clip_ids"]
        evaluation = policy["allowed_evaluation_clip_ids"]
        require(isinstance(optimizer, list) and isinstance(evaluation, list) and len(set(optimizer)) == len(optimizer) and
                len(set(evaluation)) == len(evaluation), "Allowed IDs must be unique explicit lists")
        permitted = {sid for sid, row in self._rows.items() if row["split"] == "dev" and row["kind"] == "phrase"}
        require(len(permitted) == 3, "Expected exactly three development phrases")
        require(optimizer == [], "Optimizer allowlist must be empty")
        require(set(evaluation) == permitted, "Evaluation policy must contain exactly the fixed3 dev phrases")
        self._allowed_optimizer, self._allowed_evaluation = frozenset(), frozenset(evaluation)
        self.clip_ids = tuple(row["clip_id"] for row in handoff["records"] if row["clip_id"] in permitted)
        require(self.evaluation_plan["schema"] == "offline-stage5d-unseen-phrase-evaluation-plan-v1",
                "Wrong evaluation plan schema")
        planned = self.evaluation_plan["items"]
        require([row["clip_id"] for row in planned] == list(self.clip_ids), "Evaluation plan clip identities/order differ")
        for item in planned:
            row = self._rows[item["clip_id"]]
            require(all(item[key] == row[key] for key in ("parent_id", "split", "kind", "T", "text")),
                    "Evaluation plan changes source text, split or shape")
        self.audit = {"schema": "lipnet-stage5d-loader-construction-audit-v1", "policy_sha256": expected_policy_sha256,
            "handoff_sha256": expected_handoff_sha256, "evaluation_plan_sha256": policy["evaluation_plan"]["sha256"],
            "frozen_stage3_artifacts_verified": len(frozen_by_path), "paired_parent_count": 15,
            "paired_clip_count": 30, "dataset_split": self.split, "permitted_kind": "phrase", "permitted_dataset_clips": len(self.clip_ids),
            "original_training_ready": False, "original_provisional": True,
            "evaluation_use_only": True, "source_approval_changed": False}

    def _validate_metadata(self, row: dict[str, Any], meta: dict[str, Any], parent: dict[str, Any]) -> None:
        require(meta["schema"] == CACHE_SCHEMA, "Wrong cache metadata schema")
        for key in ("clip_id", "kind", "parent_id", "split", "speaker_id", "recording_group", "T", "text", "label_length", "ctc_min_frames"):
            require(meta[key] == row[key], f"Handoff/cache field mismatch: {key}")
        require(meta["training_ready"] is False and meta["provisional"] is True and
                meta["new_original_pixels_human_approved"] is False, "Unapproved cache flags changed")
        require(meta["audio_included"] is False and meta["raw_padding_frames"] == 0 and
                set(meta["source_members"]) == {"video", "label"}, "Audio or raw padding is forbidden")
        require(row["display_neighbor_intersections"] == meta["neighbor_label_intersections"] and
                row["boundary_followup_flag"] is bool(meta["neighbor_label_intersections"]) and
                row["exact_visual_boundary_certified"] is False, "Boundary scope differs from preserved handoff")
        require(meta["shape"] == [row["T"], 96, 96, 3] and meta["dtype"] == "uint8" and meta["color"] == "RGB", "Wrong RGB cache shape")
        require(type(row["T"]) is int and 0 < row["T"] < 750, "Clip exceeds less-than30s cache scope")
        require(meta["normalization"] == self.handoff["normalization"] and meta["tokenizer"] == self.handoff["tokenizer"], "Preprocessing/tokenizer drift")
        require(meta["plan_sha256"] == self.extraction["plan_sha256"], "Extraction plan provenance differs")
        for name in ("video", "label"):
            require(meta["source_members"][name]["sha256"] == parent[name + "_sha256"], "Source member provenance differs")
        require(meta["recording_group"] == parent["source_stem"] and meta["speaker_id"] == parent["speaker_id"], "Parent recording/speaker differs")
        if row["kind"] == "phrase":
            expected = parent["phrase"]
            require(meta["text"] == expected["text"] and meta["start_sample_16k"] == expected["start_sample_16k"] and
                    meta["end_sample_16k"] == expected["end_sample_16k"], "Phrase semantic span/text differs")
        else:
            require(meta["text"] == parent["parent_text"] and meta["start_sample_16k"] == parent["sentence_start_sample_16k"] and
                    meta["end_sample_16k"] == parent["sentence_end_sample_16k"], "Parent semantic span/text differs")
        labels = meta["label"]
        require(isinstance(labels, list) and labels and all(type(v) is int and 0 < v < 69 for v in labels), "Invalid 69-class nonblank CTC labels")
        require(len(labels) == meta["label_length"] and _ctc_min(labels) == meta["ctc_min_frames"] <= row["T"] and
                meta["ctc_feasible"] is True, "CTC target/length infeasible")

    def __len__(self) -> int:
        return len(self.clip_ids)

    def load_item(self, index: int, *, purpose: str = "evaluation") -> dict[str, Any]:
        require(type(index) is int and 0 <= index < len(self), "Dataset index outside explicit range")
        return self.get(self.clip_ids[index], purpose=purpose)

    def load_clip(self, clip_id: str, *, purpose: str = "evaluation") -> dict[str, Any]:
        """Compatibility alias with the same strict evaluation-only access check."""
        return self.get(clip_id, purpose=purpose)

    def get(self, clip_id: str, *, purpose: str = "evaluation") -> dict[str, Any]:
        """Read one bound visual NPZ; returns an auditable readonly CTHW float array."""
        require(purpose == "evaluation", "Only evaluation is permitted; optimizer and validation aliases are forbidden")
        require(self.split == "dev" and clip_id in self.clip_ids, "Clip is not permitted for this split/purpose")
        allowed = self._allowed_evaluation
        require(clip_id in allowed, "Clip is absent from scoped policy allowlist")
        row, meta = self._rows[clip_id], self._metadata[clip_id]
        require(row["split"] == "dev" and row["kind"] == "phrase", "Only development phrase pixels may be read")
        require(_checked_json(Path(row["metadata_path"]), row["metadata_sha256"]) == meta, "Sidecar changed after construction")
        path = Path(row["npz_path"])
        require(path.stat().st_size == row["npz_bytes"], "NPZ file length changed")
        try:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == row["npz_sha256"], "NPZ SHA mismatch")
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                require(len(members) == len(NPZ_KEYS) and {m.filename for m in members} == {key + ".npy" for key in NPZ_KEYS},
                        "Unexpected NPZ payload; only visual arrays and exact metadata are allowed")
                limits = {"frames_rgb": row["T"]*96*96*3+512, "label": row["label_length"]*8+512,
                          "length": 520, "schema": 1024, "training_ready": 513, "provisional": 513,
                          "metadata_json": 4*len(json.dumps(meta, ensure_ascii=False))+4096}
                require(all(not(m.flag_bits & 1) and m.file_size <= limits[m.filename[:-4]] for m in members), "Oversized/encrypted NPZ entry")
            with np.load(io.BytesIO(raw), allow_pickle=False) as packed:
                rgb, labels, length = packed["frames_rgb"], packed["label"], packed["length"]
                require(packed["schema"].shape == () and packed["schema"].item() == CACHE_SCHEMA, "NPZ schema differs")
                require(packed["training_ready"].shape == () and packed["training_ready"].dtype == np.bool_ and
                        packed["training_ready"].item() is False, "NPZ source approval changed")
                require(packed["provisional"].shape == () and packed["provisional"].dtype == np.bool_ and
                        packed["provisional"].item() is True, "NPZ provisional flag changed")
                require(packed["metadata_json"].shape == () and _json(packed["metadata_json"].item()) == meta, "NPZ/sidecar metadata differs")
                require(rgb.dtype == np.uint8 and rgb.shape == (row["T"],96,96,3), "NPZ pixels shape/dtype differs")
                require(length.shape == () and length.dtype == np.int64 and int(length) == row["T"], "NPZ actual length differs")
                require(labels.dtype == np.int64 and labels.shape == (row["label_length"],) and
                        labels.tolist() == meta["label"], "NPZ labels differ")
                require(array_sha256(rgb) == meta["rgb_uint8_sha256"], "Decoded RGB content hash differs")
                require(not np.any(np.all(rgb == 0, axis=(1,2,3))), "Raw zero/filler frame is forbidden")
                require(_ctc_min(labels.tolist()) <= len(rgb), "Actual CTC input length is infeasible")
                normalized = (rgb.astype(np.float32) / np.float32(255) - MEAN) / STD
                pixels = np.ascontiguousarray(normalized.transpose(3,0,1,2), dtype=np.float32)
                labels = labels.copy()
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            if isinstance(exc, DataContractError):
                raise
            raise DataContractError(f"Invalid visual NPZ: {clip_id}") from exc
        require(np.isfinite(pixels).all(), "Normalized input contains nonfinite values")
        pixels.setflags(write=False)
        labels.setflags(write=False)
        audit = {"schema": "lipnet-stage5d-loaded-clip-audit-v1", "source_clip_id": clip_id, "source_split": row["split"],
            "source_T": row["T"], "source_npz_sha256": row["npz_sha256"], "source_metadata_sha256": row["metadata_sha256"],
            "source_rgb_uint8_sha256": meta["rgb_uint8_sha256"], "normalized_input_sha256": array_sha256(pixels),
            "labels_sha256": array_sha256(labels), "policy_sha256": self.policy_sha256, "handoff_sha256": self.handoff_sha256,
            "evaluation_plan_sha256": self.policy["evaluation_plan"]["sha256"], "loaded_for": purpose,
            "allowlisted_clip_ids_for_purpose": sorted(allowed), "normalization": "float32 (RGB/255-mean)/std, then CTHW",
            "source_training_ready": False, "source_provisional": True, "approval_changed": False,
            "technical_use_scope": "controlled_stage5d_fixed_unseen_phrase_evaluation", "source_kind": "phrase",
            "boundary_followup_flag": row["boundary_followup_flag"],
            "display_neighbor_intersections": row["display_neighbor_intersections"],
            "boundary_followup_scope": row["boundary_followup_scope"],
            "exact_visual_boundary_certified": row["exact_visual_boundary_certified"],
            "new_original_pixels_human_approved": meta["new_original_pixels_human_approved"],
            "source_root_QC_status": meta["root_QC_status"]}
        return {"input": pixels, "labels": labels, "length": row["T"], "T": row["T"], "clip_id": clip_id,
            "text": row["text"], "parent_id": row["parent_id"], "kind": row["kind"], "split": row["split"], "purpose": purpose,
            "policy_sha256": self.policy_sha256, "handoff_sha256": self.handoff_sha256, "audit": audit}
