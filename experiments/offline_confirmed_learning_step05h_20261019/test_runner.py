"""Small guard tests: source mutation, hidden audio, dev use, and metric fidelity."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_5h_runner",BASE/"runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
np = r.np


class Guards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="5h_guard_")
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(17):
            sid = f"parent{index:02d}__phrase"
            T = 4 + index%3
            rgb = np.full((T,96,96,3),80+index,np.uint8)
            labels = np.asarray(r.tokenizer.encode_text("가"),np.int64)
            meta = {"schema":r.CACHE_SCHEMA,"clip_id":sid,"text":"가","split":"train" if index<14 else "dev",
                    "label":labels.tolist(),"training_ready":False,"provisional":True,"rgb_uint8_sha256":r.array_sha(rgb)}
            mp,cp = self.root/(sid+".json"),self.root/(sid+".npz")
            mp.write_text(json.dumps(meta,ensure_ascii=False),encoding="utf-8")
            np.savez_compressed(cp,frames_rgb=rgb,label=labels,length=np.asarray(T,np.int64),
                    metadata_json=np.asarray(json.dumps(meta,ensure_ascii=False)),schema=np.asarray(r.CACHE_SCHEMA),
                    training_ready=np.asarray(False),provisional=np.asarray(True))
            self.rows.append({"clip_id":sid,"parent_id":f"parent{index:02d}","kind":"phrase","split":meta["split"],
                    "T":T,"text":"가","label_length":2,"npz_path":str(cp),"npz_sha256":r.sha(cp),"npz_bytes":cp.stat().st_size,
                    "metadata_path":str(mp),"metadata_sha256":r.sha(mp),"training_ready":False,"provisional":True})
        ids = [row["clip_id"] for row in self.rows]
        self.data = {"records":self.rows,"arms":{"A12":ids[:12],"B14":ids[:14]},"dev":ids[14:]}

    def tearDown(self):
        self.temp.cleanup()

    def binding(self):
        path = self.root/"data.json"
        path.write_text(json.dumps(self.data),encoding="utf-8")
        return r.ref(path)

    def test_dev_and_new_untrained_are_rejected_by_optimizer(self):
        data = r.Dataset(self.binding())
        for denied in [data.dev[0],data.new[0]]:
            with self.assertRaisesRegex(ValueError,"allowlist"):
                data.batch([data.arms["A12"][0],denied],"A12")
        batch = data.batch(data.arms["A12"][:2],"A12")
        self.assertEqual(batch["lengths"].tolist(),[4,5])
        self.assertTrue(np.array_equal(batch["inputs"][0,:,4],batch["inputs"][0,:,3]))

    def test_source_npz_byte_mutation_rejected(self):
        binding = self.binding()
        path = Path(self.rows[0]["npz_path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            r.Dataset(binding)

    def test_even_bound_audio_payload_rejected(self):
        row = self.rows[0]
        path = Path(row["npz_path"])
        with np.load(path,allow_pickle=False) as z:
            packed = {k:z[k] for k in z.files}
        np.savez_compressed(path,**packed,audio=np.zeros(16,np.float32))
        row.update(npz_sha256=r.sha(path),npz_bytes=path.stat().st_size)
        with self.assertRaisesRegex(ValueError,"payload/audio"):
            r.Dataset(self.binding())

    def test_source_approval_not_promoted(self):
        self.rows[0]["training_ready"] = True
        with self.assertRaisesRegex(ValueError,"approval promoted"):
            r.Dataset(self.binding())

    def test_old_split_override_rejected(self):
        self.data["experimental_split_overrides"] = {self.rows[0]["clip_id"]:{"source_split":"dev","experiment_split":"train"}}
        with self.assertRaisesRegex(ValueError,"new two"):
            r.Dataset(self.binding())

    def test_loaded_array_mutation_rejected(self):
        data = r.Dataset(self.binding())
        item = data.items[data.arms["A12"][0]]
        item["input"] = item["input"].copy()
        item["input"][0,0,0,0] += 1
        with self.assertRaisesRegex(ValueError,"arrays changed"):
            data.batch(data.arms["A12"][:2],"A12")

    def test_conditions_keep_rng_and_pixels_scope(self):
        before = r.old.runner.nested_hash(np.random.get_state())
        result = r.conditions(31,"clip")
        self.assertEqual(before,r.old.runner.nested_hash(np.random.get_state()))
        self.assertEqual(sorted(result["shuffled"].tolist()),list(range(31)))
        self.assertTrue(np.all(result["static"] == 15))
        self.assertTrue(np.array_equal(result["shuffled"],r.conditions(31,"clip")["shuffled"]))

    def test_invalid_jamo_and_failed_rows_are_scored(self):
        decoded = r.contract.decode_prediction([1,0])
        self.assertEqual(decoded["raw_text"],"\ufffd")
        scored = r.contract.score_prediction("가",decoded["raw_text"])
        self.assertEqual(scored["character_edits"],1)
        self.assertEqual(scored["replacement_characters"],1)
        failed = r.contract.score_prediction("가","",failed=True)
        self.assertEqual(r.contract.corpus_scores([scored,failed])["failure_count"],1)


if __name__ == "__main__":
    unittest.main()
