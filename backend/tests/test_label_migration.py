"""원본 보존, 캐시 버전 거부, 검토가 필요한 정답의 제외를 검증한다."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
from migrate_labels import migrate, sha256
from models.cache import load_clip
from models.tokenizer import TokenizerMismatchError, encode_text, tokenizer_metadata


def test_migration_preserves_frames_and_original_files(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    frames = np.random.default_rng(0).integers(0, 256, size=(30, 96, 96, 3), dtype=np.uint8)
    old = source / "clip.npz"
    np.savez_compressed(old, frames=frames, label=np.array([1, 2]), text="값이 없었어.")
    before = sha256(old)
    out = source / "v2"
    report = migrate(source, out)
    loaded = load_clip(out / "clip.npz")
    assert report["converted"] == 1 and report["excluded"] == 0
    assert sha256(old) == before
    assert np.array_equal(loaded["frames"], frames)
    assert loaded["text"] == "값이 없었어"
    assert loaded["source_text"] == "값이 없었어."
    with pytest.raises(TokenizerMismatchError):
        load_clip(old)
    with pytest.raises(FileExistsError):
        migrate(source, out)


def test_unknown_empty_and_ctc_short_samples_are_reported(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    for name, text, length in [("unknown", "X 이유", 30), ("empty", "...", 30), ("short", "많다고", 1)]:
        np.savez_compressed(source / f"{name}.npz", frames=np.zeros((length, 96, 96, 3), dtype=np.uint8),
                            label=np.array([1]), text=text)
    out = tmp_path / "v2"
    report = migrate(source, out)
    assert report["converted"] == 0 and report["excluded"] == 3
    assert not list(out.glob("*.npz"))
    assert all(e["reason"] for e in report["entries"])


def test_loader_rejects_mislabelled_v2_cache(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, frames=np.zeros((30, 96, 96, 3), dtype=np.uint8),
                        label=np.array(encode_text("마다고")), text="많다고", source_text="많다고",
                        **tokenizer_metadata())
    with pytest.raises(ValueError, match="토큰 인덱스"):
        load_clip(path)


def test_migration_cannot_target_source_or_parent(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    with pytest.raises(ValueError):
        migrate(source, source)
    with pytest.raises(ValueError):
        migrate(source, tmp_path)
