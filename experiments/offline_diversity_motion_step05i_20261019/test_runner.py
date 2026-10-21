"""Guard the expanded source allowlist and actual temporal-control pixels; no model forwards."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_5i_runner",BASE/"runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
np = r.np


class Guards(unittest.TestCase):
    def setUp(self) -> None:
        """Construct isolated synthetic visual caches with changing frame values."""
        try:
            self.temp = tempfile.TemporaryDirectory(prefix="5i_guard_")
            self.root = Path(self.temp.name)
            self.rows = []
            self.cohorts = {"train_phrase14":[],"train_sentence12":[],"dev_phrase3":[],"dev_sentence3":[]}
            for cohort, count in [("train_phrase14",14),("train_sentence12",12),("dev_phrase3",3),("dev_sentence3",3)]:
                split = "dev" if cohort.startswith("dev_") else "train"
                kind = "phrase" if "phrase" in cohort else "full_sentence"
                for index in range(count):
                    parent = f"parent{index+(100 if split=='dev' else 0):03d}"
                    sid = parent + "__" + kind
                    T = 4 + index%3
                    rgb = np.empty((T,96,96,3),np.uint8)
                    for frame in range(T):
                        rgb[frame] = 80 + index + frame*5
                    text = "나" if split=="dev" else "가"
                    labels = np.asarray(r.tokenizer.encode_text(text),np.int64)
                    meta = {"schema":r.CACHE_SCHEMA,"clip_id":sid,"kind":kind,"text":text,"split":split,
                            "label":labels.tolist(),"training_ready":False,"provisional":True,"rgb_uint8_sha256":r.array_sha(rgb)}
                    mp,cp = self.root/(sid+".json"),self.root/(sid+".npz")
                    mp.write_text(json.dumps(meta,ensure_ascii=False),encoding="utf-8")
                    np.savez_compressed(cp,frames_rgb=rgb,label=labels,length=np.asarray(T,np.int64),
                        metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)),schema=np.asarray(r.CACHE_SCHEMA),
                        training_ready=np.asarray(False),provisional=np.asarray(True))
                    self.rows.append({"clip_id":sid,"parent_id":parent,"recording_group":parent,"kind":kind,"split":split,
                        "T":T,"text":text,"label_length":len(labels),"npz_path":str(cp),"npz_sha256":r.sha(cp),"npz_bytes":cp.stat().st_size,
                        "metadata_path":str(mp),"metadata_sha256":r.sha(mp),"training_ready":False,"provisional":True})
                    self.cohorts[cohort].append(sid)
            source = self.root/"source.json"
            source.write_text("{}",encoding="utf-8")
            expanded = sorted(self.cohorts["train_phrase14"]+self.cohorts["train_sentence12"])
            self.data = {"records":self.rows,"arms":{"A14":self.cohorts["train_phrase14"],"B26":expanded,"C26_static":expanded},
                         "cohorts":self.cohorts,"held_parent_ids":["held00"],"source_handoffs":[r.ref(source),r.ref(source)],
                         "experimental_split_overrides":{}}
        except Exception as exc:
            exc.add_note("stage5i setUp")
            raise

    def tearDown(self) -> None:
        """Clean up the temporary synthetic guard fixtures."""
        try:
            self.temp.cleanup()
        except Exception as exc:
            exc.add_note("stage5i tearDown")
            raise

    def binding(self) -> dict:
        """Bind the current synthetic data plan to its exact saved bytes."""
        try:
            path = self.root/"data.json"
            path.write_text(json.dumps(self.data),encoding="utf-8")
            return r.ref(path)
        except Exception as exc:
            exc.add_note("stage5i binding")
            raise

    def test_dev_sentence_and_unlisted_ids_rejected_by_optimizer(self) -> None:
        """Test dev sentence and unlisted ids rejected by optimizer."""
        try:
            data = r.Dataset(self.binding())
            for denied in [data.dev[0],data.dev[-1],data.cohorts["train_sentence12"][0],"held00__phrase"]:
                with self.assertRaisesRegex(ValueError,"allowlist"):
                    data.batch([data.arms["A14"][0],denied],"A14")
            batch = data.batch(data.arms["A14"][:2],"A14")
            self.assertEqual(batch["lengths"].tolist(),[4,5])
            self.assertTrue(np.array_equal(batch["inputs"][0,:,4],batch["inputs"][0,:,3]))
        except Exception as exc:
            exc.add_note("stage5i test_dev_sentence_and_unlisted_ids_rejected_by_optimizer")
            raise

    def test_expanded_sentences_and_static_actual_pixels_keep_labels_lengths(self) -> None:
        """Test expanded sentences and static actual pixels keep labels lengths."""
        try:
            data = r.Dataset(self.binding())
            pair = data.cohorts["train_sentence12"][:2]
            normal,static = data.batch(pair,"B26"),data.batch(pair,"C26_static")
            self.assertEqual(normal["lengths"].tolist(),static["lengths"].tolist())
            self.assertTrue(np.array_equal(normal["labels"],static["labels"]))
            self.assertTrue(np.any(np.diff(normal["inputs"],axis=2)))
            self.assertFalse(np.any(np.diff(static["inputs"],axis=2)))
            for i,sid in enumerate(pair):
                T = data.items[sid]["length"]
                self.assertTrue(np.array_equal(static["inputs"][i,:,0],normal["inputs"][i,:,T//2]))
                self.assertEqual(static["input_audits"][i]["input_sha256"],r.array_sha(static["inputs"][i,:,:T]))
        except Exception as exc:
            exc.add_note("stage5i test_expanded_sentences_and_static_actual_pixels_keep_labels_lengths")
            raise

    def test_held_parent_and_cross_parent_leakage_rejected(self) -> None:
        """Test held parent and cross parent leakage rejected."""
        try:
            self.data["held_parent_ids"] = [self.rows[0]["parent_id"]]
            with self.assertRaisesRegex(ValueError,"Held source"):
                r.Dataset(self.binding())
            self.data["held_parent_ids"] = ["held00"]
            next(row for row in self.rows if row["split"]=="dev")["recording_group"] = self.rows[0]["recording_group"]
            with self.assertRaisesRegex(ValueError,"recording_group leakage"):
                r.Dataset(self.binding())
        except Exception as exc:
            exc.add_note("stage5i test_held_parent_and_cross_parent_leakage_rejected")
            raise

    def test_source_npz_byte_mutation_rejected(self) -> None:
        """Test source npz byte mutation rejected."""
        try:
            binding = self.binding()
            with Path(self.rows[0]["npz_path"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError,"bytes changed"):
                r.Dataset(binding)
        except Exception as exc:
            exc.add_note("stage5i test_source_npz_byte_mutation_rejected")
            raise

    def test_even_bound_audio_payload_rejected(self) -> None:
        """Test even bound audio payload rejected."""
        try:
            row = self.rows[0]
            path = Path(row["npz_path"])
            with np.load(path,allow_pickle=False) as z:
                packed = {k:z[k] for k in z.files}
            np.savez_compressed(path,**packed,audio=np.zeros(16,np.float32))
            row.update(npz_sha256=r.sha(path),npz_bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError,"payload/audio"):
                r.Dataset(self.binding())
        except Exception as exc:
            exc.add_note("stage5i test_even_bound_audio_payload_rejected")
            raise

    def test_source_approval_and_split_override_not_promoted(self) -> None:
        """Test source approval and split override not promoted."""
        try:
            self.rows[0]["training_ready"] = True
            with self.assertRaisesRegex(ValueError,"approval promoted"):
                r.Dataset(self.binding())
            self.rows[0]["training_ready"] = False
            self.data["experimental_split_overrides"] = {self.rows[0]["clip_id"]:{"source_split":"dev","experiment_split":"train"}}
            with self.assertRaisesRegex(ValueError,"No source split overrides"):
                r.Dataset(self.binding())
        except Exception as exc:
            exc.add_note("stage5i test_source_approval_and_split_override_not_promoted")
            raise

    def test_loaded_array_mutation_rejected(self) -> None:
        """Test loaded array mutation rejected."""
        try:
            data = r.Dataset(self.binding())
            item = data.items[data.arms["A14"][0]]
            item["input"] = item["input"].copy()
            item["input"][0,0,0,0] += 1
            with self.assertRaisesRegex(ValueError,"arrays changed"):
                data.batch(data.arms["A14"][:2],"A14")
        except Exception as exc:
            exc.add_note("stage5i test_loaded_array_mutation_rejected")
            raise

    def test_conditions_keep_rng_and_reversal_is_bijection(self) -> None:
        """Test conditions keep rng and reversal is bijection."""
        try:
            before = r.old.runner.nested_hash(np.random.get_state())
            result = r.conditions(31,"clip")
            self.assertEqual(before,r.old.runner.nested_hash(np.random.get_state()))
            self.assertEqual(sorted(result["shuffled"].tolist()),list(range(31)))
            self.assertEqual(result["reversed"].tolist(),list(range(30,-1,-1)))
            self.assertTrue(np.array_equal(result["reversed"][result["reversed"]],result["normal"]))
            self.assertTrue(np.all(result["static_mid"] == 15))
            self.assertTrue(np.array_equal(result["shuffled"],r.conditions(31,"clip")["shuffled"]))
        except Exception as exc:
            exc.add_note("stage5i test_conditions_keep_rng_and_reversal_is_bijection")
            raise

    def test_normal_static_arm_ids_cannot_differ(self) -> None:
        """Test normal static arm ids cannot differ."""
        try:
            self.data["arms"]["C26_static"] = list(reversed(self.data["arms"]["B26"]))
            with self.assertRaisesRegex(ValueError,"scope differs"):
                r.Dataset(self.binding())
        except Exception as exc:
            exc.add_note("stage5i test_normal_static_arm_ids_cannot_differ")
            raise

    def test_invalid_jamo_and_failed_rows_are_scored(self) -> None:
        """Test invalid jamo and failed rows are scored."""
        try:
            decoded = r.contract.decode_prediction([1,0])
            self.assertEqual(decoded["raw_text"],"\ufffd")
            scored = r.contract.score_prediction("가",decoded["raw_text"])
            self.assertEqual(scored["character_edits"],1)
            self.assertEqual(scored["replacement_characters"],1)
            failed = r.contract.score_prediction("가","",failed=True)
            self.assertEqual(r.contract.corpus_scores([scored,failed])["failure_count"],1)
        except Exception as exc:
            exc.add_note("stage5i test_invalid_jamo_and_failed_rows_are_scored")
            raise


if __name__ == "__main__":
    unittest.main()
