"""Meaningful live-cache checks plus invalid-input fixtures under data/ only.

Pass the externally fixed policy SHA explicitly. This suite never executes a
network and never changes stage3. Private copied loader state is used only to
exercise malformed but consistently hash-bound NPZ payloads in temporary files.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from loader import ControlledPilotDataset, DataContractError, array_sha256, collate, file_sha256, validate_batch

BASE = Path(__file__).resolve().parent
CONFIG: dict = {}
LOADED_AUDITS: list[dict] = []


class LoaderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = ControlledPilotDataset(CONFIG["handoff"], CONFIG["handoff_sha"], CONFIG["policy"], CONFIG["policy_sha"], split="train")
        cls.dev = ControlledPilotDataset(CONFIG["handoff"], CONFIG["handoff_sha"], CONFIG["policy"], CONFIG["policy_sha"], split="dev")
        cls.short = cls.train.load_clip(min(cls.train.clip_ids, key=lambda sid: cls.train._rows[sid]["T"]))
        cls.long = cls.train.load_clip(max(cls.train.clip_ids, key=lambda sid: cls.train._rows[sid]["T"]))
        cls.dev_item = cls.dev.load_item(0, purpose="validation")

    def test_all30_live_caches_preserve_flags_and_exact_float_normalization(self) -> None:
        LOADED_AUDITS.clear()
        for dataset, purpose in [(self.train, "optimizer"), (self.dev, "validation")]:
            for sid in dataset.clip_ids:
                item = dataset.load_clip(sid, purpose=purpose)
                with np.load(dataset._rows[sid]["npz_path"], allow_pickle=False) as packed:
                    raw = packed["frames_rgb"]
                    # Independent scalar-channel arithmetic, not producer transpose.
                    expected = np.empty((3,len(raw),96,96), dtype=np.float32)
                    for channel, (mean, std) in enumerate(zip((.485,.456,.406),(.229,.224,.225))):
                        expected[channel] = (raw[:,:,:,channel].astype(np.float32)/np.float32(255)-np.float32(mean))/np.float32(std)
                    np.testing.assert_array_equal(item["input"], expected)
                    np.testing.assert_array_equal(item["labels"], packed["label"])
                    self.assertEqual(item["length"], len(raw))
                self.assertFalse(item["input"].flags.writeable)
                self.assertFalse(item["labels"].flags.writeable)
                self.assertFalse(item["audit"]["source_training_ready"])
                self.assertTrue(item["audit"]["source_provisional"])
                LOADED_AUDITS.append(item["audit"])
        self.assertEqual(len(LOADED_AUDITS), 30)
        self.assertEqual(sum(a["boundary_followup_flag"] for a in LOADED_AUDITS), 2)
        self.assertEqual(sum(a["source_T"] for a in LOADED_AUDITS), 2023)

    def test_batch_true_lengths_and_last_frame_padding(self) -> None:
        batch = collate([self.short, self.long])
        self.assertEqual(batch["inputs"].shape, (2,3,self.long["length"],96,96))
        np.testing.assert_array_equal(batch["lengths"], [self.short["length"], self.long["length"]])
        np.testing.assert_array_equal(batch["label_lengths"], [len(self.short["labels"]),len(self.long["labels"])])
        np.testing.assert_array_equal(batch["labels"], np.concatenate([self.short["labels"],self.long["labels"]]))
        for t in range(self.short["length"], self.long["length"]):
            np.testing.assert_array_equal(batch["inputs"][0,:,t], self.short["input"][:,-1])
        validate_batch(batch)

    def test_dev_never_optimizer_input(self) -> None:
        with self.assertRaises(DataContractError): self.dev.load_item(0, purpose="optimizer")
        with self.assertRaises(DataContractError): self.train.load_clip(self.dev.clip_ids[0], purpose="optimizer")
        with self.assertRaises(DataContractError): collate([self.short, self.dev_item], purpose="optimizer")
        batch = collate([self.dev_item], purpose="validation")
        self.assertEqual(batch["splits"], ["dev"])

    def test_unknown_split_and_purpose_rejected(self) -> None:
        with self.assertRaises(DataContractError):
            ControlledPilotDataset(CONFIG["handoff"], CONFIG["handoff_sha"], CONFIG["policy"], CONFIG["policy_sha"], split="test")
        with self.assertRaises(DataContractError): self.train.load_item(0, purpose="train")
        with self.assertRaises(DataContractError): self.train.load_item(-1)

    def test_external_policy_and_handoff_hashes_are_mandatory(self) -> None:
        for policy_sha, handoff_sha in [("0"*64, CONFIG["handoff_sha"]), (CONFIG["policy_sha"], "0"*64), (None, CONFIG["handoff_sha"])]:
            with self.assertRaises(DataContractError):
                ControlledPilotDataset(CONFIG["handoff"], handoff_sha, CONFIG["policy"], policy_sha)

    def _changed_policy(self, change) -> None:
        with tempfile.TemporaryDirectory(prefix="policy-negative-", dir=BASE) as folder:
            policy = copy.deepcopy(self.train.policy)
            change(policy)
            path = Path(folder)/"policy.json"
            path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(DataContractError):
                ControlledPilotDataset(CONFIG["handoff"], CONFIG["handoff_sha"], path, file_sha256(path))

    def test_explicit_provisional_and_scope_gates(self) -> None:
        self._changed_policy(lambda p: p.update(allow_provisional_cache_for_controlled_pilot=False))
        self._changed_policy(lambda p: p.update(full_training_authorized=True))
        self._changed_policy(lambda p: p["authorization"].update(scope="any_training"))
        self._changed_policy(lambda p: p["training_plan"].update(sha256="0"*64))
        self._changed_policy(lambda p: p["stage3_frozen_manifest"].update(sha256="0"*64))

    def test_dev_in_optimizer_allowlist_rejected(self) -> None:
        self._changed_policy(lambda p: p["allowed_optimizer_clip_ids"].append(self.dev.clip_ids[0]))

    def test_changed_item_identity_and_arrays_rejected(self) -> None:
        for key, value in [("clip_id", self.long["clip_id"]), ("split", "dev"), ("length", self.short["length"]+1)]:
            item = dict(self.short); item[key] = value
            with self.assertRaises(DataContractError): collate([item])
        item = dict(self.short); item["input"] = self.short["input"].copy(); item["input"][0,0,0,0] += .1
        with self.assertRaises(DataContractError): collate([item])

    def test_batch_zero_padding_wrong_length_and_mask_rejected(self) -> None:
        batch = collate([self.short, self.long]); batch["inputs"][0,:,self.short["length"]:] = 0
        with self.assertRaisesRegex(DataContractError, "Padding must repeat"): validate_batch(batch)
        batch = collate([self.short, self.long]); batch["lengths"][0] += 1
        with self.assertRaises(DataContractError): validate_batch(batch)
        batch = collate([self.short, self.long]); batch["padding_mask"][:] = False
        with self.assertRaises(DataContractError): validate_batch(batch)

    def test_blank_bool_float_and_ctc_infeasible_labels_rejected(self) -> None:
        for labels in [np.asarray([0],dtype=np.int64), np.asarray([True],dtype=np.bool_),
                       np.asarray([1],dtype=np.float32), np.ones(self.short["length"],dtype=np.int64)]:
            item = dict(self.short); item["labels"] = labels; item["audit"] = dict(self.short["audit"])
            item["audit"]["labels_sha256"] = array_sha256(labels)
            with self.assertRaises(DataContractError): collate([item])

    def test_nan_and_boundary_approval_promotion_rejected(self) -> None:
        item = dict(self.short); item["input"] = self.short["input"].copy(); item["input"][0,0,0,0] = np.nan
        with self.assertRaises(DataContractError): collate([item])
        item = dict(self.short); item["audit"] = dict(self.short["audit"]); item["audit"]["exact_visual_boundary_certified"] = True
        with self.assertRaises(DataContractError): collate([item])

    def _invalid_npz(self, mutate, message: str | None = None, rehash: bool = True) -> None:
        # All mutated files live under this temporary data/ child. Original arrays
        # are copied before mutation; source metadata/NPZ bytes are never edited.
        sid = self.short["clip_id"]
        with tempfile.TemporaryDirectory(prefix="npz-negative-", dir=BASE) as folder:
            original = self.train._rows[sid]
            with np.load(original["npz_path"], allow_pickle=False) as packed:
                arrays = {key: packed[key].copy() for key in packed.files}
            metadata = copy.deepcopy(self.train._metadata[sid])
            mutate(arrays, metadata)
            arrays["metadata_json"] = np.asarray(json.dumps(metadata,ensure_ascii=False))
            path, sidecar = Path(folder)/"invalid.npz", Path(folder)/"invalid.json"
            np.savez_compressed(path, **arrays)
            sidecar.write_text(json.dumps(metadata,ensure_ascii=False),encoding="utf-8")
            cloned = copy.copy(self.train)
            cloned._rows = dict(self.train._rows); cloned._metadata = dict(self.train._metadata)
            row = dict(original); row.update(npz_path=str(path),npz_bytes=path.stat().st_size,
                metadata_path=str(sidecar),metadata_sha256=file_sha256(sidecar))
            if rehash: row["npz_sha256"] = file_sha256(path)
            cloned._rows[sid], cloned._metadata[sid] = row, metadata
            expectation = self.assertRaisesRegex(DataContractError,message) if message else self.assertRaises(DataContractError)
            with expectation: cloned.load_clip(sid)

    def test_wrong_npz_hash_rejected(self) -> None:
        self._invalid_npz(lambda a,m: a.update(length=np.asarray(self.short["length"]+1,dtype=np.int64)), "NPZ SHA mismatch", rehash=False)

    def test_audio_array_rejected_even_with_matching_file_hash(self) -> None:
        self._invalid_npz(lambda a,m: a.update(audio=np.zeros(5,dtype=np.float32)), "Unexpected NPZ payload")

    def test_raw_zero_frame_rejected_even_with_matching_pixel_hash(self) -> None:
        def change(arrays, metadata):
            arrays["frames_rgb"][0] = 0
            metadata["rgb_uint8_sha256"] = array_sha256(arrays["frames_rgb"])
        self._invalid_npz(change, "Raw zero/filler")

    def test_npz_length_dtype_and_approval_rejected(self) -> None:
        self._invalid_npz(lambda a,m: a.update(length=np.asarray(self.short["length"]+1,dtype=np.int64)), "actual length")
        self._invalid_npz(lambda a,m: a.update(label=a["label"].astype(np.int32)), "labels differ")
        self._invalid_npz(lambda a,m: a.update(training_ready=np.asarray(True)), "source approval")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--handoff-sha", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--result", type=Path, default=BASE/"loader_test_results.json")
    args = parser.parse_args()
    CONFIG.update({"handoff": args.handoff, "handoff_sha": args.handoff_sha,
                   "policy": args.policy, "policy_sha": args.policy_sha})
    require_output = args.result.resolve().is_relative_to(BASE.resolve()) and not args.result.exists()
    if not require_output: raise ValueError("Test output must be a new file under data/")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LoaderContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"schema":"lipnet-stage4-loader-test-report-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "passed":result.wasSuccessful(),"tests_run":result.testsRun,
        "failures":[{"test":str(test),"traceback":trace} for test,trace in result.failures],
        "errors":[{"test":str(test),"traceback":trace} for test,trace in result.errors],
        "policy_sha256":args.policy_sha,"handoff_sha256":args.handoff_sha,
        "loader_sha256":file_sha256(BASE/"loader.py"),"test_code_sha256":file_sha256(Path(__file__)),
        "loaded_clip_audits":LOADED_AUDITS,"source_flags_unchanged":True,
        "new_original_media_reads":0,"new_model_calls":0,"new_training_updates":0}
    args.result.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
