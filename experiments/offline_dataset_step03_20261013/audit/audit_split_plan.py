"""Independent reference/graph/seed replay for the frozen stage3 split."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from independent_clock import assert_partition

BASE = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compact(text):
    return "".join(unicodedata.normalize("NFC", text).split())


def identities(row, extra=()):
    stem = row["source_stem"]
    keys = {("stem", stem), ("recording", re.sub(r"_[ABC]_(\d+)$", r"_\1", stem)),
            ("video", row["video_sha256"]), ("label", row["label_sha256"])}
    keys.update(("text", compact(t)) for t in (row["text"], *extra))
    return keys


def components(keymap):
    # Graph flood fill independently reconstructs the producer's union classes.
    members = defaultdict(set)
    for rid, keys in keymap.items():
        for key in keys:
            members[key].add(rid)
    remaining = set(keymap)
    groups = []
    while remaining:
        pending = [min(remaining)]
        group = set()
        while pending:
            rid = pending.pop()
            if rid in group:
                continue
            group.add(rid)
            for key in keymap[rid]:
                pending.extend(members[key] - group)
        groups.append(sorted(group))
        remaining -= group
    return sorted(groups)


def main():
    output = BASE / "audit/split_plan_audit.json"
    assert not output.exists(), "Preserve prior audits"
    pp = BASE / "split/split_plan.json"
    plan = load(pp)
    assert sha(pp) == "5804761ea36d2db9debcce3695fb29923dea69126e47113fd4360a2680d7a686"
    sources = {str(pp): sha(pp)}
    docs = {}
    for name, spec in plan["inputs"].items():
        assert sha(spec["path"]) == spec["sha256"]
        sources[spec["path"]] = spec["sha256"]
        docs[name] = load(spec["path"])
    for name in ("selection_protocol", "generator"):
        spec = plan[name]
        assert sha(spec["path"]) == spec["sha256"]
        sources[spec["path"]] = spec["sha256"]
    protocol = load(plan["selection_protocol"]["path"])
    assert protocol["seed"] == 42 and protocol["parents_total"] == 15
    refpath = BASE / "audit/source_time_audit.json"
    ref = load(refpath)
    sources[str(refpath)] = sha(refpath)
    parents = {r["parent_id"]: r for r in ref["records"]}
    pool = {r["id"]: r for r in docs["analysis293"]}
    phrases = {r["parent_id"]: r for r in docs["phrase_handoff"]["records"]}
    rows = {r["parent_id"]: r for r in plan["records"]}
    assert set(parents) == set(rows) == set(plan["parent_to_split"])
    assert len(parents) == 15 and len(pool) == 293
    keymap = {}
    for pid in sorted(parents):
        pr, r, ph, hist = parents[pid], rows[pid], phrases[pid], pool[pid]
        assert r["historical_parent_text"] == hist["text"]
        assert r["parent_text"] == pr["canonical_parent_text"]
        assert r["source_stem"] == pr["recording_group"] == hist["source_stem"]
        assert r["speaker_id"] == pr["speaker_id"]
        for kind in ("video", "label"):
            assert r[kind + "_sha256"] == pr["source_members_from_frozen_metadata"][kind]["sha256"] == hist[kind + "_sha256"]
        assert r["angle_independent_recording_key"] == re.sub(r"_[ABC]_(\d+)$", r"_\1", r["source_stem"])
        assert r["full_text_keys"] == sorted({compact(t) for t in [hist["text"], pr["canonical_parent_text"], ph["text"]]})
        assert r["split"] == plan["parent_to_split"][pid]
        assert r["training_ready"] is False
        keymap[pid] = identities(hist, [pr["canonical_parent_text"], ph["text"]])
    groups = components(keymap)
    assert groups == plan["leakage_components"] and len(groups) == 14
    allspeakers = {r["speaker_id"] for r in rows.values()}
    legal = []
    for count in range(len(groups) + 1):
        for chosen in combinations(range(len(groups)), count):
            dev = {pid for index in chosen for pid in groups[index]}
            if len(dev) != 3:
                continue
            train = set(rows) - dev
            if len({rows[x]["speaker_id"] for x in dev}) != 3:
                continue
            if {rows[x]["speaker_id"] for x in train} != allspeakers:
                continue
            legal.append(list(chosen))
    legal.sort()
    assert len(legal) == 220 == plan["selection"]["legal_partition_count"]
    assert digest(legal) == plan["selection"]["all_legal_partitions_sha256"]
    pick = random.Random(42).randrange(len(legal))
    assert pick == 163 == plan["selection"]["selected_zero_based_index"]
    assert legal[pick] == plan["selection"]["selected_component_indices"]
    dev = sorted(pid for i in legal[pick] for pid in groups[i])
    assert dev == plan["dev_parent_ids"]
    assert sorted(set(rows) - set(dev)) == plan["train_parent_ids"]
    adapted = []
    for r in plan["planned_clips"]:
        p = rows[r["parent_id"]]
        adapted.append({"parent_id": r["parent_id"], "kind": r["role"], "split": r["split"],
            "text": r["text"], "canonical_parent_text": p["parent_text"], "recording_group": p["source_stem"],
            "video_sha256": p["video_sha256"], "label_sha256": p["label_sha256"],
            "start_sample_16k": r["start_sample_16k"], "end_sample_16k": r["end_sample_16k"]})
        assert r["training_ready"] is False
    role_result = assert_partition(adapted, ref["clip_references_for_partition_validation"])
    # Rebuild expansion connectivity over the entire frozen metadata survey.
    allkeys = {pid: identities(r, [parents[pid]["canonical_parent_text"], phrases[pid]["text"]] if pid in parents else [])
               for pid, r in pool.items()}
    expanded_groups = components(allkeys)
    blocked = {pid for group in expanded_groups if set(group) & set(dev) for pid in group}
    held = "lip_J_1_F_03_C293_A_012_019"
    outside = set(pool) - set(rows) - {held}
    excluded, remaining = sorted(outside & blocked), sorted(outside - blocked)
    expansion = plan["expansion"]
    assert expansion["transitive_dev_component_ids"] == sorted(blocked)
    assert expansion["excluded_additional_ids"] == excluded and expansion["remaining_metadata_ids"] == remaining
    assert len(outside) == 277 and len(excluded) == 23 and len(remaining) == 254
    assert not blocked & set(plan["train_parent_ids"])
    assert len({pool[x]["source_stem"] for x in remaining}) == expansion["remaining_recordings"] == 32
    train_stems = {rows[x]["source_stem"] for x in plan["train_parent_ids"]}
    assert sum(pool[x]["source_stem"] in train_stems for x in remaining) == expansion["remaining_on_current_train_recordings"] == 92
    # Reconstruct all historical logical rows and their source identifiers, without opening caches.
    logical = {r["id"]: r for name in ("old_plan", "new_plan") for r in docs[name]["records"]}
    historical = {r["id"]: r for r in docs["historical_handoff"]["records"]}
    assert len(logical) == len(historical) == 632
    for pid, r in logical.items():
        assert digest(r) == historical[pid]["logical_record_sha256"]
    oldsrc = {r["stem"]: r["members"] for r in docs["d_source_plan"]["sources"]}
    newsrc = {r["stem"]: r["members"] for r in docs["new_plan"]["sources"]}
    logicalkeys = {}
    for pid, r in logical.items():
        members = (oldsrc if historical[pid]["generation"] == "old" else newsrc)[r["source_stem"]]
        logicalkeys[pid] = identities({"source_stem": r["source_stem"], "text": r["text"],
            "video_sha256": members["video"]["sha256"], "label_sha256": members["label"]["sha256"]})
    protection = {}
    for name in ("protected_historical_ids", "protected_probe_ids", "protected_dev16_ids"):
        ids = set(docs["stage1_summary"][name])
        keys = set().union(*(logicalkeys[pid] for pid in ids))
        hits = [pid for pid, k in keymap.items() if pid in ids or k & keys]
        assert not hits
        protection[name] = {"protected_id_count": len(ids), "current15_collisions": hits}
    inherited_blocked = set(docs["step15_plan"]["pool_profile"]["blocked_normalized_targets"])
    assert not [v for keys in keymap.values() for kind, v in keys if kind == "text" and v in inherited_blocked]
    assert plan["training_ready"] is False and plan["held_parent_excluded_from_both_roles"] == held
    summary = {"schema": "stage3-independent-split-plan-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_reference_roles_seeded_components_and_metadata_protections", "issues": [],
        "source_sha256": sources, "independent_components": groups, "independent_legal_partitions": len(legal),
        "seed": 42, "independent_random_index": pick, "dev_parent_ids": dev,
        "paired_role_validation": role_result, "historical_logical_rows": 632,
        "historical_protection": protection,
        "expansion_recount": {"unreviewed_additional": len(outside), "excluded_by_dev_component": len(excluded),
            "metadata_only_remaining": len(remaining), "remaining_recordings": 32,
            "on_current_train_recordings": 92, "other_recordings": 162},
        "limits": ["Dev speakers are already in train; three dev parents cannot establish robust or unseen-speaker performance.",
            "Protected-role checks use stored metadata only; raw media, historical NPZ pixels and model weights were not opened.",
            "254 expansion rows remain unreviewed metadata proposals, not train assignments or boundary-approved examples.",
            "The current paired references have overlapping phrase/parent pixels and are not 30 independent utterances."],
        "training_ready": False, "new_training_updates": 0, "production_imports": 0,
        "raw_TAR_reads": 0, "new_human_approvals": 0}
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "components": len(groups), "partitions": len(legal),
        "dev": dev, "expanded_metadata": len(remaining)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
