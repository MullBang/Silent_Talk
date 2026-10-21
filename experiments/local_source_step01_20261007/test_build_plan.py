"""Regression checks for source identity, immutable splits and failure behavior."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.local_source_step01_20261007 import build_plan as bp


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Use real frozen local evidence; archive stat behavior has isolated tests."""
    monkeypatch.setattr(bp, "stat_archives", lambda archives: None)
    return bp.collect_plan()


def save_json(path: Path, value: dict) -> str:
    """Write a test-owned candidate and return its current fingerprint."""
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return bp.sha256(path)


def test_selected_contract_and_recording_coverage(plan: dict) -> None:
    """All inherited records map once, without training/dev leakage or C025."""
    parent = bp.read_json(bp.PARENT)
    assert plan["records"] == parent["records"]
    assert len(plan["sources"]) == 59
    assert {source["speaker_id"] for source in plan["sources"]} == {
        "C002", "C004", "C017", "C032", "C282", "C293", "C297", "C312"}
    mapped = [record_id for source in plan["sources"] for record_id in source["record_ids"]]
    assert len(mapped) == len(set(mapped)) == 512
    assert set(mapped) == {row["id"] for row in parent["records"]}
    for source in plan["sources"]:
        assert set(source["members"]) == {"video", "label", "audio"}
        rows = [row for row in plan["records"] if row["source_stem"] == source["stem"]]
        assert {row["split"] for row in rows} == {source["split"]}
        for role, extension in (("video", ".mp4"), ("label", ".json"), ("audio", ".wav")):
            assert source["members"][role]["member_name"].endswith(source["stem"] + extension)


def test_roundtrip_checks_without_opening_g_drive(plan: dict, tmp_path: Path) -> None:
    """Historical G: strings remain inert while a candidate is fully validated."""
    path = tmp_path / "plan.json"
    digest = bp.write_new_plan(plan, path)
    assert bp.load_plan(path, digest) == plan


@pytest.mark.parametrize("change", ["split", "offset", "text", "member", "audio_sha"])
def test_rehashed_candidate_cannot_change_frozen_contract(
        plan: dict, tmp_path: Path, change: str) -> None:
    """Even a freshly hashed edited plan must agree with independent evidence."""
    candidate = deepcopy(plan)
    if change == "split":
        candidate["records"][0]["split"] = "train"
    elif change == "offset":
        candidate["records"][0]["start_time"] += 0.1
    elif change == "text":
        candidate["records"][0]["text"] = "바뀐 정답"
    elif change == "member":
        candidate["sources"][0]["members"]["video"]["offset"] += 512
    else:
        candidate["sources"][0]["members"]["audio"]["sha256"] = "0" * 64
    path = tmp_path / "edited.json"
    digest = save_json(path, candidate)
    with pytest.raises(bp.PlanError, match="content differs"):
        bp.load_plan(path, digest)


def test_plan_hash_is_required(plan: dict, tmp_path: Path) -> None:
    """An unapproved byte change is rejected before parsing/evidence access."""
    path = tmp_path / "plan.json"
    digest = bp.write_new_plan(plan, path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(bp.PlanError, match="plan SHA mismatch"):
        bp.load_plan(path, digest)


def test_existing_plan_is_not_overwritten(plan: dict, tmp_path: Path) -> None:
    """Accidentally rerunning creation preserves the accepted bytes."""
    path = tmp_path / "plan.json"
    digest = bp.write_new_plan(plan, path)
    with pytest.raises(bp.PlanError, match="Cannot create new plan"):
        bp.write_new_plan({"replacement": True}, path)
    assert bp.sha256(path) == digest


@pytest.mark.parametrize("name", ["../clip.mp4", "/clip.mp4", "TS1/../../clip.mp4", "TS1\\clip.mp4", "C:/clip.mp4"])
def test_unsafe_tar_paths_rejected(name: str) -> None:
    """A member reference cannot address absolute paths or parent directories."""
    member = {"member_name": name, "is_file": True, "is_dir": False,
              "type": "0", "size": 10, "offset_data": 512}
    with pytest.raises(bp.PlanError, match="Unsafe or wrong"):
        bp.validate_member(member, "clip.mp4")


def test_archive_changed_after_audit_is_rejected(tmp_path: Path) -> None:
    """A changed TAR snapshot cannot reuse earlier decoding fingerprints."""
    path = tmp_path / "source.tar"
    path.write_bytes(b"audited")
    stat = path.stat()
    snapshot = [{"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}]
    bp.stat_archives(snapshot)
    path.write_bytes(b"changed-size")
    with pytest.raises(bp.PlanError, match="Archive changed since audit"):
        bp.stat_archives(snapshot)


def test_ambiguous_basename_is_rejected() -> None:
    """Duplicate audited identities never silently select the last occurrence."""
    with pytest.raises(bp.PlanError, match="Duplicate stem"):
        bp.unique_index([{"stem": "a"}, {"stem": "a"}], "stem")


@pytest.mark.parametrize("change", ["errors", "label_sha", "label_speaker"])
def test_selected_label_evidence_must_agree(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    """A label with a failed audit, wrong payload digest or speaker is refused."""
    original_read = bp.read_json

    def modified_read(path: Path | str) -> object:
        """Inject one bad audit row without editing any protected evidence."""
        data = original_read(path)
        if Path(path).name == "label_records.json":
            row = next(row for row in data if row["stem"] == "lip_J_1_F_02_C002_A_004")
            if change == "errors":
                row["errors"] = ["sentence outside video"]
            elif change == "label_sha":
                row["sha256"] = "0" * 64
            else:
                row["detail"]["identity"]["speaker"] = "C025"
        return data

    monkeypatch.setattr(bp, "read_json", modified_read)
    monkeypatch.setattr(bp, "stat_archives", lambda archives: None)
    with pytest.raises(bp.PlanError, match="Unready selected|Label SHA mismatch|Label source identity"):
        bp.collect_plan()


def test_binding_rejects_modified_local_evidence(tmp_path: Path) -> None:
    """Changed local reports cannot be silently used with an accepted plan."""
    path = tmp_path / "evidence.json"
    path.write_text("{}", encoding="utf-8")
    binding = {"path": str(path), "sha256": bp.sha256(path)}
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(bp.PlanError, match="Evidence SHA mismatch"):
        bp.verify_binding(binding)
