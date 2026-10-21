"""Recovery must work without a completed-run manifest or an expanded budget."""
from pathlib import Path
from unittest.mock import patch
import unittest

import resume_segment as subject


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"updates": 4}
        self.saved = {"completed_updates": 3, "identity": {"arm": "B_feature"}}

    def test_incomplete_run_needs_only_checkpoint_not_completed_run_manifest(self):
        path = subject.BASE / "runs/B_feature/interrupted/update_0003.pt"
        with patch.object(subject.train, "verify_plan", return_value=(self.plan, {"clips": {}})), \
             patch.object(subject.runner, "load_checkpoint", return_value=self.saved), \
             patch.object(subject.runner, "run_segment", return_value={"arm": "B_feature", "completed_updates": 4}) as run, \
             patch.object(subject.runner.core, "bind", return_value={"path": "bound", "sha256": "digest"}), \
             patch.object(subject.runner.core, "save_json"):
            result = subject.recover(path, "recovery-test")
            self.assertEqual(result["completed_updates"], 4)
            self.assertEqual(run.call_args.kwargs["resume_checkpoint"], path.resolve())
            self.assertEqual(run.call_args.kwargs["stop_update"], 4)

    def test_cannot_repeat_or_extend_budget(self):
        with patch.object(subject.train, "verify_plan", return_value=(self.plan, {"clips": {}})), \
             patch.object(subject.runner, "load_checkpoint", return_value=self.saved), \
             patch.object(subject.runner, "run_segment") as run:
            for stop in (3, 5):
                with self.assertRaises(RuntimeError):
                    subject.recover(subject.BASE / "runs/update_0003.pt", "recovery-test", stop)
            run.assert_not_called()

    def test_external_path_and_unsafe_output_rejected_before_loading(self):
        with patch.object(subject.train, "verify_plan") as verify:
            for path, name in ((subject.BASE.parent / "unrelated.pt", "test"),
                               (subject.BASE / "runs/test.pt", "../escape")):
                with self.assertRaises(RuntimeError):
                    subject.recover(path, name)
            verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
