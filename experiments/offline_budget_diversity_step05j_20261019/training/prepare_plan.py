"""Freeze a new bounded continuation policy without starting model work."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD = ROOT/"experiments/offline_diversity_motion_step05i_20261019"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ref(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return ref(path)


def main():
    old_plan = read(OLD/"plan.json")
    data = read(OLD/"data_plan.json")
    mapping = {r["clip_id"]: r for r in data["records"]}
    ids = sorted(data["arms"]["B26"])
    assert len(ids) == 26 and len(mapping) == 32
    assert len(data["held_parent_ids"]) == 6
    schedules, sources, accounting = {}, {}, {}
    for seed in [42, 43]:
        rng = random.Random(seed)
        tasks = []
        for epoch in range(1, 281):
            order = list(ids)
            rng.shuffle(order)
            for start in range(0, 26, 2):
                pair = order[start:start+2]
                update = len(tasks)+1
                lengths = [mapping[s]["T"] for s in pair]
                tasks.append({"update": update, "epoch": epoch, "clip_ids": pair, "lengths": lengths,
                              "lr": .001 if update <= 910 else .0003 if update <= 1365 else .0001,
                              "true_frames": sum(lengths), "dense_frames": 2*max(lengths)})
        old_schedule = read(old_plan["schedules"]["B26"][str(seed)]["path"])
        assert tasks[:1820] == old_schedule["updates"]
        continuation = tasks[1820:]
        counts = Counter(s for t in continuation for s in t["clip_ids"])
        assert len(continuation) == 1820 and set(counts.values()) == {140}
        schedules[str(seed)] = save(BASE/f"schedules/B26_extended_seed{seed}.json", {
            "schema": "offline-stage5j-continuation-schedule-v1", "seed": seed, "arm": "B26_extended",
            "rule": "Local Random(seed), independent copies of sorted26 shuffled for280epochs. Prefix140epochs exactly equals frozen5i; continue epochs141-280, no RNG reset of learned model.",
            "original_schedule": old_plan["schedules"]["B26"][str(seed)], "updates": continuation})
        sources[str(seed)] = ref(OLD/f"results/B26_seed{seed}/update_1820.pt")
        assert sources[str(seed)] == read(OLD/f"results/B26_seed{seed}/result.json")["checkpoints"][-1]
        accounting[str(seed)] = {"new_optimizer_updates": 1820, "original_updates": 1820, "final_total_updates": 3640,
            "new_exposures": dict(counts), "final_exposures_per_clip": 280,
            "new_true_frames": sum(t["true_frames"] for t in continuation),
            "new_dense_frames": sum(t["dense_frames"] for t in continuation)}
    policy = save(BASE/"controlled_pilot_policy.json", {
        "schema": "offline-stage5j-budget-extension-policy-v1", "client_date": "2026-10-19",
        "actual_user_instruction": "다음 단계 진행해줘. 애매한 자료는 계속 보류하고, 다른 녹화의 짧은 표현을 추가하면서 확대군의 학습량이 부족한지도 확인해보자.",
        "scope": "This branch tests learning budget alone by continuing only frozen5iB26 for1820additionalupdates perseed; short-new-recording data is a separate branch.",
        "held_parent_ids": data["held_parent_ids"], "optimizer_ids": ids,
        "data_plan": ref(OLD/"data_plan.json"), "no_source_approval_transfer": True,
        "source_training_ready": False, "source_provisional": True, "new_sources_in_this_branch": 0,
        "audio_model_input": False, "teacher": False, "downloads": False, "service_weight_replacement": False,
        "no_new_review_questions": True, "no_automatic_extension_past3640": True,
        "source_full_state_mandatory": ["model", "Adam", "Python/NumPy/Torch/CUDA RNG"],
        "dev_reuse": "Three internal existing-speaker parent recordings, phrase and full-sentence views correlated, not final test",
    })
    code = {str(Path(b["path"]).resolve()) for b in old_plan["bindings"]}
    code.update(str(p.resolve()) for p in [BASE/"runner.py", BASE/"prepare_plan.py", BASE/"test_runner.py",
                OLD/"runner.py", OLD/"plan.json", OLD/"data_plan.json", OLD/"results/completion_manifest.json"])
    for seed in [42, 43]:
        code.add(str((OLD/f"results/B26_seed{seed}/result.json").resolve()))
        code.add(str((OLD/f"results/B26_seed{seed}/eval_1820/summary.json").resolve()))
    plan = {
        "schema": "offline-stage5j-budget-extension-plan-v1", "client_date": "2026-10-19",
        "seeds": [42, 43], "start_updates": 1820, "additional_steps": 1820, "final_updates": 3640,
        "evaluations": [2730, 3640], "batch_size": 2, "old_plan": ref(OLD/"plan.json"),
        "old_data_plan": ref(OLD/"data_plan.json"), "policy": policy, "source_checkpoints": sources,
        "schedules": schedules, "bindings": [ref(p) for p in sorted(code)], "runtime": old_plan["runtime"],
        "limits": {"wall_seconds_per_run": 1200, "cuda_reserved_bytes": 6*1024**3},
        "planned_accounting": accounting, "new_main_updates": 3640, "preflight_physical_updates": 4,
        "maximum_preflight_updates": 5, "source_reproduction_forward_calls": 8,
        "evaluation_rows": 512,
        "evaluation": {"cohorts": list(data["cohorts"]), "conditions": ["normal", "static_mid", "shuffled", "reversed"],
            "fixed_points": [2730, 3640], "baseline": "frozen5iB26@1820; same source pixels and deterministic temporal orders",
            "fit": "Normal training phrase14 and full_sentence12 reported separately at1820/2730/3640; mean/final60trainingloss descriptive",
            "generalization": "Normal devphrase3 primary, devsentence3 secondary; individual seeds and arithmetic mean; no best checkpointselection",
            "motion": "AblatedCER-normalCER plus perclip predictions; neither difference nor lack of difference alone certifies lip-motion understanding",
            "claim_limit": "Additional optimization exposure underfixedLR differs only along a continued trajectory; no arbitrary successful-fit or real-world-generalization certification"},
        "caveats": ["Source32allowlist and all6heldparents unchanged; no newdata in this budget-only branch.",
            "140additionalexposures eachclip, total280 versus original140; source14parents/13recordings/8speakers unchanged.",
            "Existingdev3 reused for many experiments; descriptive internaldiagnostic, no fresh finaltest.",
            "MaincontinuesoriginalAdam/RNG; preflightisolated and discarded, never usedasinitialweights.",
            "No learning-rate reset, no teacher/audioinput, no source approvalpromotion, no service deployment."],
    }
    binding = save(BASE/"plan.json", plan)
    print(json.dumps({"plan": binding, "main_updates": 3640, "preflight_updates": 4,
                      "evaluation_rows": 512, "source_verification_forwards": 8}, ensure_ascii=False))


if __name__ == "__main__":
    main()
