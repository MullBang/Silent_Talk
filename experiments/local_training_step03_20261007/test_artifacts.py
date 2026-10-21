"""Negative controls for the step03 fixed-pilot artifact boundary."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("step03_artifacts", Path(__file__).with_name("artifacts.py"))
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


@pytest.fixture
def clip():
    row = {"id": "sample", "source_stem": "recording", "speaker_id": "speaker", "split": "train",
           "frames": 3, "text": "가", "token_ids": [1, 20]}
    source = np.array([10, 11, 12], dtype=np.int64)
    teacher = np.arange(8, 15, dtype=np.int64)
    gather = np.array([2, 3, 4], dtype=np.int64)
    arrays = {"student_rgb": np.ones((3, 96, 96, 3), np.uint8),
              "metadata": {"id": row["id"], "record": copy.deepcopy(row), "plan_sha256": artifacts.PARENT_SHA256},
              "student_source_indices": source, "teacher_source_indices": teacher, "gather_indices": gather}
    payload = {"features": np.ones((3, 512), np.float32), "student_source_indices": source.copy(),
               "teacher_source_indices": teacher.copy(), "gather_indices": gather.copy()}
    feature = {key: row[key] for key in ("id", "source_stem", "speaker_id", "split")}
    feature.update(data_record_sha256=artifacts.canonical_sha(row), state_sha256_before="state", state_sha256_after="state",
                   feature_values_sha256=hashlib.sha256(payload["features"].tobytes()).hexdigest(),
                   audio_used=False, decoder_instantiated=False, transcript_used_for_teacher_forward=False,
                   optimization_steps=0, trainable_parameters=0)
    return row, arrays, payload, feature


def test_exact_pair_passes(clip):
    artifacts.validate_clip(*clip)


@pytest.mark.parametrize("kind", ["source_indices", "gather", "dtype", "nonfinite", "zero", "values", "text", "tokens", "speaker", "dev", "state", "audio", "optimization"])
def test_semantic_tampering_rejected_even_without_outer_hash(clip, kind):
    row, arrays, payload, feature = clip
    if kind == "source_indices": payload["student_source_indices"][0] += 1
    elif kind == "gather": payload["gather_indices"][0] += 1
    elif kind == "dtype": payload["features"] = payload["features"].astype(np.float64)
    elif kind == "nonfinite": payload["features"][0, 0] = np.nan
    elif kind == "zero": payload["features"][0] = 0
    elif kind == "values": payload["features"][0, 0] += 1
    elif kind == "text": arrays["metadata"]["record"]["text"] = "나"
    elif kind == "tokens": row["token_ids"] = [0]
    elif kind == "speaker": feature["speaker_id"] = "other"
    elif kind == "dev": feature["split"] = "dev"
    elif kind == "state": feature["state_sha256_after"] = "changed"
    elif kind == "audio": feature["audio_used"] = True
    elif kind == "optimization": feature["optimization_steps"] = 1
    with pytest.raises(ValueError):
        artifacts.validate_clip(*clip)


def test_binding_rejects_changed_bytes_and_outside_root(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    item = allowed / "a.json"
    item.write_text('{}')
    binding = {"path": str(item), "sha256": artifacts.sha256(item)}
    ledger = artifacts.LocalBindings(allowed)
    assert ledger.read(binding) == {}
    item.write_text('{"x":1}')
    with pytest.raises(ValueError): ledger.verify(binding)
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    with pytest.raises(ValueError): ledger.verify({"path": str(outside), "sha256": artifacts.sha256(outside)})


@pytest.mark.parametrize("kind", ["duplicate", "reorder", "expand", "dev", "changed_parent"])
def test_cohort_rejects_changes(kind):
    parent = json.loads((artifacts.ROOT / 'experiments/feature_transfer_step07/data_plan.json').read_text(encoding='utf-8'))
    source = json.loads((artifacts.ROOT / 'experiments/local_source_step01_20261007/d_source_plan.json').read_text(encoding='utf-8'))
    inputs = json.loads((artifacts.PILOT / 'inputs/pilot_manifest.json').read_text(encoding='utf-8'))
    features = json.loads((artifacts.PILOT / 'teacher_results/pilot_manifest.json').read_text(encoding='utf-8'))
    teacher = json.loads((artifacts.PILOT / 'teacher_plan.json').read_text(encoding='utf-8'))
    if kind == "duplicate": features['records'][1] = copy.deepcopy(features['records'][0])
    elif kind == "reorder": features['records'].reverse()
    elif kind == "expand": inputs['records'].append(copy.deepcopy(inputs['records'][0]))
    elif kind == "dev": features['records'][0]['split'] = 'dev'
    elif kind == "changed_parent": source['records'][0]['text'] = 'changed'
    with pytest.raises(ValueError):
        artifacts.validate_cohort(parent, source, inputs, features, teacher)


def test_real_pilot_loads_without_cv2_or_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: pytest.fail('loader deserialized a model'))
    monkeypatch.setattr(torch.cuda, 'init', lambda *a, **kw: pytest.fail('loader initialized CUDA'))
    loaded = artifacts.load_pilot()
    assert loaded['counts']['clips'] == 8
    assert loaded['counts']['student_frames'] == 1543
    assert loaded['counts']['dev'] == loaded['counts']['test'] == 0
    assert all(Path(b['path']).is_relative_to(artifacts.ROOT) for b in loaded['bindings'])
    assert all(c['teacher_features'].shape == (len(c['pixels']), 512) for c in loaded['clips'].values())
