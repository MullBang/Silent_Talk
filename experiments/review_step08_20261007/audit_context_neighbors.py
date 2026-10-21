"""Independent all-row metadata audit using interval-union lengths.

Does not import context_neighbor_check.py or listen to/interpret audio.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def union_based_intersection(first, second):
    ordered = sorted((first, second))
    union_length = ordered[0][1] - ordered[0][0]
    if ordered[1][0] >= ordered[0][1]:
        union_length += ordered[1][1] - ordered[1][0]
    elif ordered[1][1] > ordered[0][1]:
        union_length += ordered[1][1] - ordered[0][1]
    value = (first[1] - first[0]) + (second[1] - second[0]) - union_length
    return 0.0 if abs(value) < 1e-10 else value


def main():
    output = read(BASE / "context_neighbors.json")
    require(output["status"] == "completed_metadata_only", "Output scope changed")
    require(digest(BASE / "context_neighbor_check.py") == output["code"]["sha256"], "Builder changed")
    fixed = {
        "source_plan": "5187f0ccbf84ea099d2f52d0a0e8d94db11d75deed50da1ad093e1a032465929",
        "media_manifest": "fc74e5664fbfc835a9b10173334e397be4abe51dfc7dac6d04c871ba30cdfdf8",
        "review_data": "7efa45a3154ce78f9c11da717966406eb9ce3aef883fc939496b5a5ecfbbaa42",
    }
    source = {}
    for name, expected in fixed.items():
        ref = output["inputs"][name]
        require(ref["sha256"] == expected == digest(ref["path"]), "Upstream hash differs: " + name)
        source[name] = read(ref["path"])
    require([r["id"] for r in output["records"]] == [r["id"] for r in source["source_plan"]["records"]] ==
            [r["id"] for r in source["review_data"]["records"]] == [r["id"] for r in source["media_manifest"]["clips"]],
            "IDs/order differ")
    counts = {"records": 120, "label_files": len(output["label_files"]), "context_with_labeled_neighbor": 0,
              "priority_context_with_labeled_neighbor": 0, "target_with_labeled_overlap": 0}
    checked_pairs = 0
    checked_neighbors = 0
    label_cache = {}
    actual_label_refs = {}
    for result, original, clip, review in zip(output["records"], source["source_plan"]["records"],
                                              source["media_manifest"]["clips"], source["review_data"]["records"]):
        require(result["split"] == original["split"] == review["split"] and
                result["speaker_id"] == original["speaker_id"], "Role/speaker differs")
        require(not result["actual_neighbor_speech_confirmed"] and not result["label_boundary_modified"],
                "Audio truth or modification was inferred")
        label_path = original["label_path"]
        if label_path not in label_cache:
            require(digest(label_path) == original["label_sha256"], "Original label hash differs")
            label_cache[label_path] = read(label_path)[0]["Sentence_info"]
        actual_label_refs[label_path] = original["label_sha256"]
        require(result["label"] == {"path": label_path, "sha256": original["label_sha256"]}, "Output label differs")
        require(result["comparison_group"] == review["comparison_group"], "Comparison group changed")
        require(result["context_audio"] == clip["context_audio"], "Context metadata changed")
        target = (original["start_time"], original["end_time"])
        context = (clip["context_audio"]["start_seconds"], clip["context_audio"]["end_seconds"])
        require(target == (result["target_start"], result["target_end"]), "Target clock changed")
        require(context == (result["context_start"], result["context_end"]), "Context clock changed")
        expected_neighbors = []
        for other in label_cache[label_path]:
            if other["ID"] == original["sentence_id"]:
                continue
            checked_pairs += 1
            other_interval = (other["start_time"], other["end_time"])
            context_overlap = union_based_intersection(context, other_interval)
            if context_overlap > 0:
                target_overlap = union_based_intersection(target, other_interval)
                side = "overlaps_target_label" if target_overlap > 0 else "before" if other_interval[0] < target[0] else "after"
                expected_neighbors.append({"sentence_id": other["ID"], "text": other["sentence_text"],
                                           "start_time": other_interval[0], "end_time": other_interval[1], "side": side,
                                           "overlap_with_context_seconds": context_overlap,
                                           "overlap_with_target_seconds": target_overlap})
        require(len(expected_neighbors) == len(result["neighbors"]), "Neighbor set size differs")
        for expected, actual in zip(expected_neighbors, result["neighbors"]):
            checked_neighbors += 1
            for field in expected:
                if field.endswith("_seconds"):
                    require(abs(expected[field] - actual[field]) < 1e-9, "Overlap duration differs")
                else:
                    require(expected[field] == actual[field], "Neighbor data differs")
        counts["context_with_labeled_neighbor"] += bool(expected_neighbors)
        counts["priority_context_with_labeled_neighbor"] += bool(expected_neighbors) and review["comparison_group"] != "both_supported"
        counts["target_with_labeled_overlap"] += any(n["overlap_with_target_seconds"] > 0 for n in expected_neighbors)
    require(actual_label_refs == {r["path"]: r["sha256"] for r in output["label_files"]}, "Label file list differs")
    require(len(label_cache) == counts["label_files"] and len(output["records"]) == counts["records"], "Cohort size differs")
    require(counts == output["counts"], "Aggregate counts differ")
    result = {"schema": "lipnet-step08-context-neighbors-independent-audit-v1", "status": "passed",
              "code": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
              "input": {"path": str((BASE / "context_neighbors.json").resolve()), "sha256": digest(BASE / "context_neighbors.json")},
              "counts": counts, "all_neighbor_pairs_compared": checked_pairs, "positive_neighbors_verified": checked_neighbors,
              "formula": "intersection length = first length + second length - merged union length",
              "actual_audio_listening": False, "training_approval_added": False}
    destination = BASE / "results/context_neighbors_independent_audit.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
