"""Check saved schedule invariants and independent exposure formulas without training."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent


def read(name: str) -> Any:
    """Read one prepared JSON artifact."""
    try:
        return json.loads((BASE/name).read_text(encoding="utf-8"))
    except (ValueError,OSError) as exc:
        raise RuntimeError(f"Cannot read {name}") from exc


def require(condition: bool, message: str) -> None:
    """Abort instead of accepting a mismatched schedule."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Hash only declared metadata and code artifacts."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def main() -> None:
    """Validate 5040 arm-updates and all fixed evaluation conditions."""
    try:
        protocol,data,account,eval_plan = read("comparison_protocol.json"),read("data_bindings.json"),read("schedule_accounting.json"),read("evaluation_plan.json")
        for path,expected in protocol["source_hashes"].items():
            require(digest(Path(path)) == expected,"Bound source changed")
        for bound in protocol["schedule_files"]+[protocol[k] for k in ("data","accounting","evaluation")]:
            require(digest(Path(bound["path"])) == bound["sha256"],"Bound prepared output changed")
        cache = {r["clip_id"]:r for r in data["records"]}
        train = {r["parent_id"] for r in cache.values() if r["split"] == "train"}
        require(len(train) == 12 and len(cache) == 30,"Cohort mismatch")
        frame_totals = {kind:sum(r["T"] for r in cache.values() if r["split"] == "train" and r["kind"] == kind) for kind in ("phrase","full_sentence")}
        expected_frames = {"A_sentence":frame_totals["full_sentence"]*210,"B_curriculum":frame_totals["phrase"]*70+frame_totals["full_sentence"]*140}
        saved_runs = {r["run_id"]:r for r in account["runs"]}
        verified = []
        for seed in (42,43):
            schedule = read(f"schedule_seed{seed}.json")
            rows = schedule["updates"]
            require(len(rows) == 1260,"Update count mismatch")
            for start in range(0,1260,6):
                parent_cycle = [p for row in rows[start:start+6] for p in row["parent_ids"]]
                require(len(set(parent_cycle)) == 12 and set(parent_cycle) == train,"Cycle missing/repeating a parent")
            for arm in ("A_sentence","B_curriculum"):
                count = Counter()
                exposure = Counter()
                for update,row in enumerate(rows,1):
                    batch = row["arms"][arm]
                    expected_kind = "phrase" if arm == "B_curriculum" and update <= 420 else "full_sentence"
                    require(row["update"] == update and row["round_index"] == (update-1)//6+1,"Update/round indexing differs")
                    require(row["lr"] == (.001 if update <= 840 else .0003 if update <= 1050 else .0001),"LR differs")
                    require(batch["kind"] == expected_kind,"Curriculum phase differs")
                    clips = [cache[c] for c in batch["clip_ids"]]
                    require(len(clips) == 2 and [c["parent_id"] for c in clips] == row["parent_ids"],"A/B paired parent order differs")
                    require(all(c["split"] == "train" and c["kind"] == expected_kind for c in clips),"Training uses wrong split/role")
                    lengths = [c["T"] for c in clips]
                    labels = [c["label_length"] for c in clips]
                    actual,dense = sum(lengths),2*max(lengths)
                    require(batch["lengths"] == lengths and batch["target_lengths"] == labels,"Length metadata differs")
                    require((batch["actual_frames"],batch["dense_processed_frames"],batch["padding_frames"]) == (actual,dense,dense-actual),"Batch frame accounting differs")
                    count.update({"updates":1,"clip_presentations":2,"actual_frames":actual,"dense_processed_frames":dense,"padding_frames":dense-actual,"target_tokens":sum(labels)})
                    exposure.update((c["parent_id"],c["kind"]) for c in clips)
                run = saved_runs[f"{arm}__seed{seed}"]
                require(dict(count) == run["totals"],"Saved total accounting differs")
                require(count["actual_frames"] == expected_frames[arm],"Closed-form true-frame exposure differs")
                for parent in train:
                    required = (0,210) if arm == "A_sentence" else (70,140)
                    require((exposure[parent,"phrase"],exposure[parent,"full_sentence"]) == required,"Per-parent role exposure differs")
                verified.append({"run_id":run["run_id"],"totals":dict(count)})
        for item in eval_plan["items"]:
            source = cache[item["clip_id"]]
            require(item["T"] == source["T"] and item["static_frame_index"] == item["T"]//2,"Evaluation length/static frame differs")
            require(sorted(item["shuffled_frame_indices"]) == list(range(item["T"])) and item["shuffled_frame_indices"] != list(range(item["T"])),"Invalid/identity evaluation shuffle")
        require(len(eval_plan["items"]) == 30 and sum(x["primary_at_final_normal_only"] for x in eval_plan["items"]) == 3,"Evaluation scope differs")
        require(eval_plan["total_evaluations_four_runs"] == 30*3*3*4 == 1080,"Evaluation count differs")
        result = {"schema":"offline-stage4-protocol-verification-v1","status":"passed","protocol_sha256":digest(BASE/"comparison_protocol.json"),"verifier_sha256":digest(Path(__file__)),
            "verified_source_hashes":len(protocol["source_hashes"]),"verified_arm_updates":5040,"verified_runs":verified,"verified_evaluation_items":30,"planned_evaluations":1080,"NPZ_payload_reads":0,"neural_calls":0,"training_updates":0}
        target = BASE/"protocol_verification.json"
        require(not target.exists(),"Preserve previous verification")
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":"passed","source_hashes":len(protocol["source_hashes"]),"arm_updates":5040,"evaluation_items":30,"training_updates":0}))
    except Exception as exc:
        raise RuntimeError("Saved protocol verification failed") from exc


if __name__ == "__main__":
    main()
