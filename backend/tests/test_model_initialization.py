"""학습 초기 가중치의 출처 제한과 구조별 로드를 실제 CPU 파일로 검사한다."""
from __future__ import annotations

import copy
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "backend")]

import train  # noqa: E402
from models.baseline import LipNetBaseline, ModelCompatibilityError  # noqa: E402
from models.registry import create_model, load_model_checkpoint, model_class  # noqa: E402
from models.residual import LipNetResidual  # noqa: E402
from models.tokenizer import TokenizerMismatchError  # noqa: E402

MANIFEST_HASH = "a" * 64


def _target_provenance() -> dict:
    """본 학습의 고정 명세 출처를 만든다."""
    return {"mode": "speaker_disjoint", "manifest_sha256": MANIFEST_HASH}


def _source_provenance() -> dict:
    """동일 명세 train의 일부만 사용한 진단 가중치 출처."""
    return {"mode": "smoke_only", "purpose": "train_subset_overfit",
            "manifest_sha256": MANIFEST_HASH, "sample_ids": ["train_a", "train_b"],
            "generalization_evaluation": False}


def _source_run() -> dict:
    """이미 수행된 갱신 수가 이후 학습에 복원되지 않는지 확인한다."""
    return {"overfit_only": True, "optimizer_steps": 123, "epoch": 9}


def _train_files(tmp_path: Path) -> list[str]:
    """initialize_model이 확인하는 실제 train 파일 이름을 제공한다."""
    return [str(tmp_path / "train_a.npz"), str(tmp_path / "train_b.npz")]


@pytest.mark.parametrize("model_type", [LipNetBaseline, LipNetResidual])
def test_valid_subset_initialization_preserves_all_weights_and_records_origin(tmp_path, model_type) -> None:
    torch.manual_seed(42)
    original = model_type(lstm_hidden=8)
    checkpoint = tmp_path / "subset.pt"
    original.save_checkpoint(str(checkpoint), _source_provenance(), _source_run())
    loaded, metadata = train.initialize_model(
        original.architecture_version, str(checkpoint), _target_provenance(), _train_files(tmp_path),
    )
    assert type(loaded) is model_type
    assert loaded.lstm.hidden_size == 8
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, atol=0, rtol=0)
    assert metadata["initialization"] == "train_subset_weights"
    assert metadata["initial_checkpoint"] == str(checkpoint.resolve())
    assert metadata["initial_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["initial_training_data"] == _source_provenance()
    assert metadata["initial_training_run"] == _source_run()
    assert metadata["optimizer_state_restored"] is False
    optimizer = torch.optim.Adam(loaded.parameters(), lr=0.002)
    assert not optimizer.state
    assert optimizer.param_groups[0]["lr"] == 0.002


@pytest.mark.parametrize("case", [
    "missing_source", "missing_run", "non_mapping_source", "non_mapping_run",
    "wrong_manifest", "missing_manifest",
    "wrong_source_mode", "wrong_purpose", "not_overfit", "truthy_not_boolean_overfit",
    "validation_id", "test_id", "duplicate_ids", "empty_ids", "empty_id",
    "non_list_ids", "non_string_id", "smoke_target", "target_missing_manifest",
])
def test_invalid_initialization_provenance_is_rejected(tmp_path, case: str) -> None:
    source = _source_provenance()
    source_run = _source_run()
    target = _target_provenance()
    if case == "missing_source":
        source = None
    elif case == "missing_run":
        source_run = None
    elif case == "non_mapping_source":
        source = "invalid source"
    elif case == "non_mapping_run":
        source_run = "invalid run"
    elif case == "wrong_manifest":
        source["manifest_sha256"] = "b" * 64
    elif case == "missing_manifest":
        del source["manifest_sha256"]
    elif case == "wrong_source_mode":
        source["mode"] = "speaker_disjoint"
    elif case == "wrong_purpose":
        source["purpose"] = "validation_tuning"
    elif case == "not_overfit":
        source_run["overfit_only"] = False
    elif case == "truthy_not_boolean_overfit":
        source_run["overfit_only"] = 1
    elif case == "validation_id":
        source["sample_ids"] = ["train_a", "val_a"]
    elif case == "test_id":
        source["sample_ids"] = ["test_a"]
    elif case == "duplicate_ids":
        source["sample_ids"] = ["train_a", "train_a"]
    elif case == "empty_ids":
        source["sample_ids"] = []
    elif case == "empty_id":
        source["sample_ids"] = [""]
    elif case == "non_list_ids":
        source["sample_ids"] = "train_a"
    elif case == "non_string_id":
        source["sample_ids"] = [7]
    elif case == "smoke_target":
        target["mode"] = "smoke_only"
    elif case == "target_missing_manifest":
        del target["manifest_sha256"]
    path = tmp_path / "invalid_source.pt"
    model = LipNetBaseline(lstm_hidden=8)
    model.save_checkpoint(str(path), source, source_run)
    with pytest.raises(ValueError, match="초기 가중치"):
        train.initialize_model(model.architecture_version, str(path), target, _train_files(tmp_path))


def test_initializer_rejects_architecture_mismatch(tmp_path) -> None:
    path = tmp_path / "compact.pt"
    LipNetBaseline(lstm_hidden=8).save_checkpoint(str(path), _source_provenance(), _source_run())
    with pytest.raises(ValueError, match="모델 구조"):
        train.initialize_model("lipnet-residual-v1", str(path), _target_provenance(), _train_files(tmp_path))


def test_initializer_rejects_wrong_tokenizer_before_training(tmp_path) -> None:
    path = tmp_path / "wrong_tokens.pt"
    model = LipNetBaseline(lstm_hidden=8)
    model.save_checkpoint(str(path), _source_provenance(), _source_run())
    document = torch.load(path, map_location="cpu", weights_only=True)
    document["vocab"] = list(reversed(document["vocab"]))
    torch.save(document, path)
    with pytest.raises(TokenizerMismatchError):
        train.initialize_model(model.architecture_version, str(path), _target_provenance(), _train_files(tmp_path))


@pytest.mark.parametrize("model_type", [LipNetBaseline, LipNetResidual])
def test_registry_auto_loads_existing_supported_architectures(tmp_path, model_type) -> None:
    original = model_type(lstm_hidden=8)
    path = tmp_path / "existing.pt"
    # 기존 compact 파일에도 없는 새 registry 전용 필드를 요구하면 안 된다.
    original.save_checkpoint(str(path))
    loaded = load_model_checkpoint(str(path))
    assert type(loaded) is model_type
    assert loaded.architecture_version == original.architecture_version
    for key, expected in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, atol=0, rtol=0)
    other = LipNetResidual if model_type is LipNetBaseline else LipNetBaseline
    with pytest.raises(ModelCompatibilityError):
        other.load_from_checkpoint(str(path))


@pytest.mark.parametrize("architecture", [None, "unrecognized-model-v9"])
def test_registry_rejects_unknown_architecture(tmp_path, architecture) -> None:
    path = tmp_path / "unknown.pt"
    LipNetBaseline(lstm_hidden=8).save_checkpoint(str(path))
    document = torch.load(path, map_location="cpu", weights_only=True)
    document["architecture_version"] = architecture
    torch.save(document, path)
    with pytest.raises(ModelCompatibilityError, match="지원하지 않는 모델 구조"):
        load_model_checkpoint(str(path))


def test_default_registry_and_uninitialized_training_keep_compact_model() -> None:
    assert type(create_model()) is LipNetBaseline
    assert model_class("lipnet-residual-v1") is LipNetResidual
    model, metadata = train.initialize_model("lipnet-compact-v1", None, {}, [])
    assert type(model) is LipNetBaseline
    assert metadata == {"initialization": "random"}


@pytest.mark.parametrize("model_type", [LipNetBaseline, LipNetResidual])
def test_training_creates_new_optimizer_and_does_not_resume_subset_steps(tmp_path, monkeypatch, model_type) -> None:
    """실제 작은 모델 한 번 학습으로 초기 출처와 새 optimizer/갱신 수를 검사한다."""
    source_model = model_type(lstm_hidden=8)
    source_path = tmp_path / "subset.pt"
    source_model.save_checkpoint(str(source_path), _source_provenance(), _source_run())
    source_hash_before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    train_files = _train_files(tmp_path)
    sample = {"frames": np.full((4, 96, 96, 3), 80, dtype=np.uint8),
              "label": np.array([1, 20], dtype=np.int64), "text": "가"}
    monkeypatch.setattr(train, "training_files", lambda args: (train_files, ["val_a.npz"], _target_provenance()))
    monkeypatch.setattr(train, "ClipDataset", lambda files: [copy.deepcopy(sample) for _ in files])
    monkeypatch.setattr(train, "clip_length", lambda path: 4)
    monkeypatch.setattr(train, "evaluate_val", lambda model, files, device: (0.5, []))
    observed = []
    real_adam = torch.optim.Adam

    def fresh_adam(parameters, **kwargs):
        optimizer = real_adam(parameters, **kwargs)
        observed.append({"initial_state_count": len(optimizer.state), "optimizer": optimizer})
        return optimizer

    monkeypatch.setattr(train.torch.optim, "Adam", fresh_adam)
    args = SimpleNamespace(
        check_data=False, max_steps=0, smoke_data=None, out=str(tmp_path / "best.pt"),
        last_out=str(tmp_path / "last.pt"), epochs=1, batch_size=2, eval_every=1,
        accum_steps=1, lr=0.002, grad_clip=1.0, device="cpu", precision="fp32", seed=42,
        report=str(tmp_path / "run.json"), snapshot_epochs=None,
        architecture=source_model.architecture_version, init_weights=str(source_path),
    )
    report = train.train(args)
    assert len(observed) == 1 and observed[0]["initial_state_count"] == 0
    assert observed[0]["optimizer"].param_groups[0]["lr"] == 0.002
    assert report["optimizer_steps"] == 1
    assert report["epochs_completed"] == 1
    assert report["initial_training_run"]["optimizer_steps"] == 123
    assert report["optimizer_state_restored"] is False
    assert report["architecture_version"] == source_model.architecture_version
    trained = load_model_checkpoint(args.out)
    assert trained.training_data == _target_provenance()
    assert trained.training_run["optimizer_steps"] == 1
    assert trained.training_run["optimizer_state_restored"] is False
    assert trained.training_run["initial_training_run"]["optimizer_steps"] == 123
    assert not torch.equal(trained.fc.weight, source_model.fc.weight)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash_before
    assert report["initial_checkpoint_sha256"] == source_hash_before


@pytest.mark.parametrize("collision", ["best", "last", "snapshot", "report", "relative_alias", "hard_link"])
def test_training_rejects_initial_checkpoint_output_collision_before_model_use(tmp_path, monkeypatch, collision) -> None:
    """실제 초기 파일을 best/last/snapshot/JSON 출력 및 파일 별칭으로부터 보존한다."""
    source = tmp_path / ("epoch_0001.pt" if collision == "snapshot" else "subset.pt")
    LipNetBaseline(lstm_hidden=8).save_checkpoint(str(source), _source_provenance(), _source_run())
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    args = SimpleNamespace(
        check_data=False, max_steps=0, smoke_data=None, out=str(tmp_path / "best.pt"),
        last_out=str(tmp_path / "last.pt"), epochs=1, batch_size=2, eval_every=1,
        accum_steps=1, lr=0.002, grad_clip=1.0, device="cpu", precision="fp32", seed=42,
        report=str(tmp_path / "run.json"), snapshot_epochs=None,
        architecture="lipnet-compact-v1", init_weights=str(source),
    )
    if collision == "best":
        args.out = str(source)
    elif collision == "last":
        args.last_out = str(source)
    elif collision == "snapshot":
        args.snapshot_epochs = [1]
    elif collision == "report":
        args.report = str(source)
    elif collision == "relative_alias":
        monkeypatch.chdir(tmp_path)
        args.out = "./subset.pt"
    elif collision == "hard_link":
        alias = tmp_path / "source_alias.pt"
        os.link(source, alias)
        args.report = str(alias)
    monkeypatch.setattr(train, "training_files", lambda args: (_train_files(tmp_path), [], _target_provenance()))
    monkeypatch.setattr(train, "resolve_device_precision", lambda *args: pytest.fail("출력 충돌은 장치/모델 준비 전에 거부해야 함"))
    files_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="초기 가중치 원본을 덮어쓸 수 없습니다"):
        train.train(args)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()} == files_before
