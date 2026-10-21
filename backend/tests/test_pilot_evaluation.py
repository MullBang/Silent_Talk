"""파일럿 평가의 분모·데이터 구분 및 best/last 저장을 검증한다."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]
import train
import test_model as evaluation
from models.tokenizer import encode_text


def test_macro_and_corpus_errors_are_distinct_and_not_clamped():
    rows = [
        {"cer": 2., "wer": 2., "reference_jamo_count": 2, "reference_word_count": 1,
         "jamo_edit_distance": 4, "word_edit_distance": 2, "exact": False, "prediction": "����"},
        {"cer": 0., "wer": 0., "reference_jamo_count": 8, "reference_word_count": 3,
         "jamo_edit_distance": 0, "word_edit_distance": 0, "exact": True, "prediction": "가 나 다"},
    ]
    result = evaluation.aggregate_predictions(rows)
    assert result["mean_sentence_cer"] == 1.
    assert result["corpus_cer"] == .4
    assert result["corpus_wer"] == .5
    assert result["exact_count"] == result["invalid_jamo_sentence_count"] == 1
    assert evaluation.aggregate_predictions(rows[:1])["corpus_cer"] == 2.
    with pytest.raises(ValueError, match="평가 문장"):
        evaluation.aggregate_predictions([])


def test_evaluation_reports_wrong_prediction_without_using_labels(monkeypatch):
    sample = {"frames": np.full((4, 96, 96, 3), 70, dtype=np.uint8),
              "label": np.array(encode_text("나")), "text": "나"}
    monkeypatch.setattr(evaluation, "load_clip", lambda path: sample)
    class Fixed(torch.nn.Module):
        def forward(self, x):
            assert x.shape == (1, 3, 4, 96, 96)
            logits = torch.full((4, 1, 69), -20.)
            for t, token in enumerate([*encode_text("가"), 0, 0]):
                logits[t, 0, token] = 0
            return logits.log_softmax(-1)
    result = evaluation.evaluate_files(Fixed(), ["sample.npz"], "cpu", decoder="greedy")
    assert result["metrics"]["exact_count"] == 0
    assert result["metrics"]["corpus_cer"] == .5
    assert result["predictions"][0]["prediction"] == "가"
    assert result["control"] is None


def test_training_keeps_best_validation_and_final_epoch_separate(tmp_path, monkeypatch):
    saved = {}
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.5]))
        def save_checkpoint(self, path, **metadata):
            saved[str(path)] = {"weight": self.weight.detach().clone(), **metadata}
        @classmethod
        def load_from_checkpoint(cls, path):
            result = cls()
            result.loaded_path = str(path)
            return result
    validation_calls = []
    def validate(model, files, device):
        assert files == ["val"]
        if hasattr(model, "loaded_path"):
            assert model.loaded_path == str(tmp_path / "best.pt")
            return .3, []
        score = [.9, .3, .8][len(validation_calls)]
        validation_calls.append(score)
        return score, [("가", "나")]
    monkeypatch.setattr(train, "training_files", lambda args: (["train"], ["val"], {"mode": "speaker_disjoint"}))
    monkeypatch.setattr(train, "clip_length", lambda path: 1)
    monkeypatch.setattr(train, "ClipDataset", lambda files: [torch.tensor(2.)])
    monkeypatch.setattr(train, "collate", lambda items: (torch.stack(items), None, torch.ones(len(items), dtype=torch.long), None))
    monkeypatch.setattr(train, "create_model", lambda architecture: Tiny())
    monkeypatch.setattr(train, "load_model_checkpoint", Tiny.load_from_checkpoint)
    monkeypatch.setattr(train, "ctc_training_loss", lambda m, b, d, a: ((m.weight * b[0] - 1) ** 2).mean())
    monkeypatch.setattr(train, "evaluate_val", validate)
    args = SimpleNamespace(check_data=False, max_steps=0, smoke_data=None, out=str(tmp_path / "best.pt"),
                           last_out=str(tmp_path / "last.pt"), epochs=3, batch_size=1, eval_every=1,
                           accum_steps=1, lr=.001, grad_clip=0, device="cpu", precision="fp32", seed=42,
                           report=str(tmp_path / "run.json"), snapshot_epochs=[1, 1])
    result = train.train(args)
    assert result["best_epoch"] == 2 and result["best_val_cer"] == .3
    assert result["epochs_completed"] == 3 and result["status"] == "completed"
    assert saved[args.out]["training_run"]["epoch"] == 2
    assert saved[args.last_out]["training_run"]["epoch"] == 3
    assert not torch.equal(saved[args.out]["weight"], saved[args.last_out]["weight"])
    assert json.loads(Path(args.report).read_text(encoding="utf-8"))["epochs"][2]["val_cer"] == .8
    snapshot = str(tmp_path / "epoch_0001.pt")
    assert saved[snapshot]["training_run"]["epoch"] == 1
    assert saved[snapshot]["training_run"]["selection"] == "fixed_epoch"
    assert saved[snapshot]["training_run"]["optimizer_steps"] == 1
    assert not torch.equal(saved[snapshot]["weight"], saved[args.out]["weight"])
    assert result["snapshots"] == [{"epoch": 1, "optimizer_steps": 1, "checkpoint": snapshot}]
    assert saved[args.out]["training_run"]["optimizer_steps"] == 2
    assert saved[args.last_out]["training_run"]["optimizer_steps"] == 3


@pytest.mark.parametrize("case", ["outside", "bounded", "existing", "collision"])
def test_snapshot_validation_precedes_model_creation(tmp_path, monkeypatch, case):
    monkeypatch.setattr(train, "training_files", lambda args: (["train"], ["val"], {"mode": "speaker_disjoint"}))
    monkeypatch.setattr(train, "LipNetBaseline", lambda: pytest.fail("잘못된 저장 조건에서 모델 생성"))
    args = SimpleNamespace(check_data=False, max_steps=1 if case == "bounded" else 0, smoke_data=None,
                           out=str(tmp_path / "best.pt"), last_out=None, epochs=3, batch_size=1,
                           eval_every=1, accum_steps=1, lr=.001, grad_clip=1,
                           snapshot_epochs=[4] if case == "outside" else [1])
    if case == "existing":
        (tmp_path / "epoch_0001.pt").write_bytes(b"preserve")
    if case == "collision":
        args.out = str(tmp_path / "epoch_0001.pt")
    with pytest.raises((ValueError, FileExistsError), match="중간 저장"):
        train.train(args)
    if case == "existing":
        assert (tmp_path / "epoch_0001.pt").read_bytes() == b"preserve"
