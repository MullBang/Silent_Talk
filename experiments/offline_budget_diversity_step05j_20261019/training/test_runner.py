"""Small pre-execution guards for true continuation, not repeated training."""
import copy
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("extension_runner_test", BASE/"runner.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class ContinuationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = r.read(r.OLD_BASE/"data_plan.json")
        cls.data = SimpleNamespace(arms=data["arms"], rows={row["clip_id"]: row for row in data["records"]})

    def test_schedule_prefix_and_new_epochs_both_seeds(self):
        old_plan = r.read(r.OLD_BASE/"plan.json")
        for seed in (42, 43):
            tasks = r.expected_continuation(self.data, seed)
            self.assertEqual(tasks[:1820], r.read(old_plan["schedules"]["B26"][str(seed)]["path"])["updates"])
            self.assertEqual((tasks[1820]["epoch"], tasks[-1]["epoch"]), (141, 280))
            self.assertEqual((tasks[1820]["update"], tasks[-1]["update"]), (1821, 3640))
            self.assertTrue(all(t["lr"] == .0001 for t in tasks[1820:]))

    def test_schedule_generation_does_not_consume_global_rng(self):
        before = random.getstate()
        r.expected_continuation(self.data, 42)
        self.assertEqual(before, random.getstate())

    def test_changed_clip_lengths_rejected_by_original_prefix(self):
        data = copy.deepcopy(self.data)
        first = data.arms["B26"][0]
        data.rows[first]["T"] += 1
        source = r.read(r.OLD_BASE/"schedules/B26_seed42.json")["updates"]
        self.assertNotEqual(r.expected_continuation(data, 42)[:1820], source)

    def fixture_state(self):
        return {"model": {}, "rng": {}, "optimizer": {
            "state": {0: {"step": 1820}, 1: {"step": 1820}},
            "param_groups": [{"params": [0, 1], "lr": .0001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0.}]}}

    def test_reject_adam_reset_or_missing_parameter_state(self):
        with patch.object(r.old, "hashes", return_value={"complete": "fixed"}):
            original = self.fixture_state()
            r.validate_state(original, {"complete": "fixed"}, 1820)
            reset = copy.deepcopy(original)
            reset["optimizer"]["state"][0]["step"] = 0
            with self.assertRaisesRegex(ValueError, "counters"):
                r.validate_state(reset, {"complete": "fixed"}, 1820)
            missing = copy.deepcopy(original)
            del missing["optimizer"]["state"][1]
            with self.assertRaisesRegex(ValueError, "Adam tensors"):
                r.validate_state(missing, {"complete": "fixed"}, 1820)

    def test_reject_missing_rng_or_changed_state_digest(self):
        with patch.object(r.old, "hashes", return_value={"complete": "fixed"}):
            state = self.fixture_state()
            with self.assertRaisesRegex(ValueError, "Incomplete/corrupt"):
                r.validate_state(state, {"complete": "wrong"}, 1820)
            del state["rng"]
            with self.assertRaisesRegex(ValueError, "Incomplete/corrupt"):
                r.validate_state(state, {"complete": "fixed"}, 1820)

    def test_reject_learning_rate_reset(self):
        with patch.object(r.old, "hashes", return_value={"complete": "fixed"}):
            state = self.fixture_state()
            state["optimizer"]["param_groups"][0]["lr"] = .001
            with self.assertRaisesRegex(ValueError, "parameters"):
                r.validate_state(state, {"complete": "fixed"}, 1820)


if __name__ == "__main__":
    unittest.main()
