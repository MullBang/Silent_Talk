"""암기 검증의 데이터 격리와 평가 경로가 잘못된 성공을 보고하지 않도록 검사한다."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
from overfit_small_set import evaluate, run, select_records
from models.tokenizer import encode_text
from test_model import validate_evaluation_provenance


def test_selection_uses_train_only_and_is_order_independent():
    rows = [{"id": str(i), "frames": length, "split": split, "text": text}
            for i, (length, split, text) in enumerate([
                (1, "test", "가"), (2, "val", "나"), (20, "train", "다"),
                (15, "train", "라"), (15, "train", "마"), (99, "train", "바")])]
    chosen = select_records({"records": rows}, 3)
    assert [r["id"] for r in chosen] == ["3", "4", "2"]
    assert chosen == select_records({"records": list(reversed(rows))}, 3)
    assert all(r["split"] == "train" for r in chosen)


def test_selection_rejects_single_sample_and_duplicate_targets():
    doc = {"records": [{"id": str(i), "frames": 10, "split": "train", "text": "가"}
                       for i in range(2)]}
    with pytest.raises(ValueError, match="2 이상"):
        select_records(doc, 1)
    with pytest.raises(ValueError, match="같은 정답"):
        select_records(doc, 2)


def test_existing_experiment_is_never_overwritten(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("original", encoding="utf-8")
    args = SimpleNamespace(max_steps=1, eval_every=1, batch_size=2, lr=0.001, out_dir=str(tmp_path))
    with pytest.raises(FileExistsError, match="덮어쓰지"):
        run(args)
    assert result.read_text(encoding="utf-8") == "original"


def test_evaluation_does_not_feed_labels_to_model():
    class FixedModel(torch.nn.Module):
        def forward(self, x):
            assert x.shape == (1, 3, 4, 96, 96)
            # 입력과 정답에 관계없이 가를 출력한다. 평가는 정답 나를 수정하면 안 된다.
            logits = torch.full((4, 1, 69), -20.)
            for t, token in enumerate([*encode_text("가"), 0, 0]):
                logits[t, 0, token] = 0
            return logits.log_softmax(-1)
    sample = {"frames": np.full((4, 96, 96, 3), 80, dtype=np.uint8),
              "label": np.array(encode_text("나")), "text": "나"}
    result = evaluate(FixedModel(), [sample], torch.device("cpu"), beam=True)
    assert result["greedy_exact_count"] == result["beam_exact_count"] == 0
    assert result["samples"][0]["greedy"] == "가"
    assert result["mean_greedy_cer"] > 0
    assert sample["text"] == "나"


def test_overfit_checkpoint_is_ineligible_for_fixed_evaluation():
    with pytest.raises(ValueError, match="학습 분할"):
        validate_evaluation_provenance({"mode": "smoke_only", "purpose": "train_subset_overfit",
                                       "manifest_sha256": "same"}, {"manifest_sha256": "same"})
