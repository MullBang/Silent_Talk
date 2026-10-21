"""교정된 시간축 캐시와 과거 라벨 FPS 캐시가 조용히 섞이지 않게 한다."""
import numpy as np
import pytest

from .test_dataset_manifest import dataset
from models.dataset_manifest import TEMPORAL_METHOD, sha256_file, validate_manifest


def upgrade_fixture(path, document):
    """테스트 캐시에 시간축 출처를 추가한다."""
    document["policy"]["temporal_method"] = TEMPORAL_METHOD
    for row in document["records"]:
        cached_path = path.parent / row["cache_path"]
        with np.load(cached_path, allow_pickle=False) as cached:
            values = {k: cached[k] for k in cached.files}
        metadata = {"temporal_method": TEMPORAL_METHOD, "video_fps": 30000 / 1001,
                    "label_fps": 30.0}
        np.savez_compressed(cached_path, **values, **metadata)
        row.update(metadata, cache_sha256=sha256_file(cached_path))


def test_corrected_cache_metadata_is_verified(dataset):
    path, document = dataset
    upgrade_fixture(path, document)
    summary = validate_manifest(document, path.parent, verify_files=True)
    assert summary["splits"]["train"]["clips"] == 1
    document["records"][0]["video_fps"] = 30.0
    with pytest.raises(ValueError, match="캐시 video_fps"):
        validate_manifest(document, path.parent, verify_files=True)


@pytest.mark.parametrize("location", ["policy", "row", "cache"])
def test_legacy_and_corrected_timebase_cannot_be_mixed(dataset, location):
    path, document = dataset
    upgrade_fixture(path, document)
    if location == "policy":
        document["policy"].pop("temporal_method")
    elif location == "row":
        document["records"][0].pop("temporal_method")
    else:
        # 해시까지 새로 계산해도 필수 출처 누락은 검출해야 한다.
        row = document["records"][0]
        cached_path = path.parent / row["cache_path"]
        with np.load(cached_path, allow_pickle=False) as cached:
            values = {k: cached[k] for k in cached.files if k != "temporal_method"}
        np.savez_compressed(cached_path, **values)
        row["cache_sha256"] = sha256_file(cached_path)
    with pytest.raises(ValueError, match="시간축"):
        validate_manifest(document, path.parent, verify_files=True)


@pytest.mark.parametrize("fps", [None, 0, -30, float("nan"), float("inf"), True])
def test_completed_rows_require_valid_measured_fps(dataset, fps):
    path, document = dataset
    upgrade_fixture(path, document)
    document["records"][0]["video_fps"] = fps
    with pytest.raises(ValueError, match="video_fps"):
        validate_manifest(document, path.parent)


def test_selection_can_precede_video_probe_but_never_claim_cache_ready(dataset):
    path, document = dataset
    document["policy"]["temporal_method"] = TEMPORAL_METHOD
    for row in document["records"]:
        row["temporal_method"] = TEMPORAL_METHOD
        for key in ("cache_path", "cache_sha256", "frames"):
            row.pop(key)
    validate_manifest(document, path.parent, verify_files=False)
    with pytest.raises(ValueError, match="아직 준비되지"):
        validate_manifest(document, path.parent, verify_files=True)
