"""Freeze a metadata-only paired sentence versus phrase-to-sentence pilot plan.

No NPZ pixels, checkpoint tensors, neural models or original TARs are read.
Data order uses private RNGs and is written explicitly for both future arms.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
STAGE3 = ROOT/"experiments/offline_dataset_step03_20261013"
HANDOFF = STAGE3/"results/cache_handoff.json"
SPLIT = STAGE3/"split/split_plan.json"
UPDATES = 1260
PHRASE_UPDATES = 420
SEEDS = (42,43)
ARMS = ("A_sentence","B_curriculum")
CHECKPOINTS = (420,840,1260)


def require(condition: bool, message: str) -> None:
    """Reject metadata or schedule mismatches before saving a comparison plan."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Hash small source metadata and prepared protocol artifacts."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash metadata {path}") from exc


def read(path: Path) -> Any:
    """Read JSON evidence; never deserialize cache pixels or model weights."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def save(path: Path, value: Any) -> None:
    """Create new protocol outputs without overwriting earlier evidence."""
    require(path.resolve().is_relative_to(BASE.resolve()) and not path.exists(), "Unsafe or existing protocol output")
    try:
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {path}") from exc


def bind(path: Path) -> dict[str,Any]:
    """Record the path, byte length and digest of an existing artifact."""
    return {"path":str(path.resolve()),"sha256":sha(path),"bytes":path.stat().st_size}


def learning_rate(update: int) -> float:
    """Return the paired fixed LR without observing any validation result."""
    require(1 <= update <= UPDATES,"Invalid update")
    return .001 if update <= 840 else .0003 if update <= 1050 else .0001


def input_records() -> tuple[list[dict[str,Any]],dict[str,str],dict[str,Any]]:
    """Validate thirty metadata sidecars and retain audited NPZ identities only."""
    handoff = read(HANDOFF)
    require(handoff["schema"] == "offline-stage3-paired-cache-handoff-v1" and len(handoff["records"]) == 30,"Unexpected stage3 handoff")
    require(handoff["split_plan"]["sha256"] == sha(SPLIT),"Fixed split changed")
    parent_to_split = read(SPLIT)["parent_to_split"]
    hashes = {str(p.resolve()):sha(p) for p in (HANDOFF,SPLIT,Path(__file__).resolve())}
    records = []
    for record in handoff["records"]:
        path = Path(record["metadata_path"])
        require(sha(path) == record["metadata_sha256"],"Cache metadata changed")
        hashes[str(path)] = record["metadata_sha256"]
        metadata = read(path)
        for key in ("clip_id","parent_id","kind","split","T","text","label_length","ctc_min_frames"):
            require(record[key] == metadata[key],f"Handoff/metadata differs: {key}")
        require(record["split"] == parent_to_split[record["parent_id"]],"Parent split differs")
        require(record["cache_integrity_audit_passed"] is True and record["technical_sample_QC_passed"] is True,"Missing stage3 technical QC")
        require(metadata["training_ready"] is False and metadata["provisional"] is True,"Preserve provisional source state")
        labels = metadata["label"]
        require(labels and all(type(v) is int and 0 < v < 69 for v in labels),"Invalid 69-class target")
        minimum = len(labels)+sum(a == b for a,b in zip(labels,labels[1:]))
        require(len(labels) == metadata["label_length"] and minimum == metadata["ctc_min_frames"] <= metadata["T"],"CTC mismatch")
        require(metadata["normalization"] == handoff["normalization"] and metadata["tokenizer"] == handoff["tokenizer"],"Input preprocessing/tokenizer differs")
        records.append({**record,"label":labels})
    require(len({r["clip_id"] for r in records}) == 30,"Duplicate clip")
    pairs = {}
    for record in records:
        pairs.setdefault(record["parent_id"],{})[record["kind"]] = record
    require(len(pairs) == 15 and all(set(v) == {"phrase","full_sentence"} for v in pairs.values()),"Missing paired role")
    require(Counter(r["split"] for r in records) == {"train":24,"dev":6},"Unexpected clip split")
    return records,hashes,handoff


def schedule(seed: int, records: list[dict[str,Any]]) -> dict[str,Any]:
    """Fix 210 complete shuffled parent rounds, paired identically across arms."""
    by_id = {r["clip_id"]:r for r in records}
    parent_ids = sorted({r["parent_id"] for r in records if r["split"] == "train"})
    require(len(parent_ids) == 12,"Expected twelve training parents")
    rng = random.Random(seed)
    updates = []
    for round_index in range(1,211):
        order = list(parent_ids)
        rng.shuffle(order)
        for pair_start in range(0,12,2):
            update = len(updates)+1
            parents = order[pair_start:pair_start+2]
            arms = {}
            for arm in ARMS:
                kind = "phrase" if arm == "B_curriculum" and update <= PHRASE_UPDATES else "full_sentence"
                clips = [by_id[parent+"__"+kind] for parent in parents]
                lengths = [r["T"] for r in clips]
                actual = sum(lengths)
                dense = 2*max(lengths)
                arms[arm] = {"kind":kind,"clip_ids":[r["clip_id"] for r in clips],"lengths":lengths,
                    "target_lengths":[r["label_length"] for r in clips],"actual_frames":actual,"dense_processed_frames":dense,"padding_frames":dense-actual}
            updates.append({"update":update,"round_index":round_index,"parent_ids":parents,"lr":learning_rate(update),"arms":arms})
    require(len(updates) == UPDATES,"Wrong update count")
    return {"schema":"offline-curriculum-paired-schedule-v1","seed":seed,"order_rng":"private Python random.Random(seed); shuffle sorted 12 parent IDs independently once per round; recorded order is authoritative",
        "batch_size":2,"rounds":210,"updates_per_round":6,"updates":updates,"training_execution_performed":False}


def accounting(schedules: list[dict[str,Any]], records: list[dict[str,Any]]) -> dict[str,Any]:
    """Count all actual/dense/padding frames, target tokens and per-parent exposures."""
    lookup = {r["clip_id"]:r for r in records}
    runs = []
    for saved in schedules:
        for arm in ARMS:
            parent_counts: dict[str,Counter] = {}
            phases: dict[str,Counter] = {}
            total = Counter()
            for row in saved["updates"]:
                batch = row["arms"][arm]
                phase = "early_1_420" if row["update"] <= 420 else "late_421_1260"
                counts = {"updates":1,"clip_presentations":2,"actual_frames":batch["actual_frames"],"dense_processed_frames":batch["dense_processed_frames"],"padding_frames":batch["padding_frames"],"target_tokens":sum(batch["target_lengths"])}
                total.update(counts)
                phases.setdefault(phase,Counter()).update(counts)
                for clip_id in batch["clip_ids"]:
                    record = lookup[clip_id]
                    parent_counts.setdefault(record["parent_id"],Counter())[record["kind"]] += 1
            exposures = {parent:{"phrase":counts["phrase"],"full_sentence":counts["full_sentence"],"total":sum(counts.values())} for parent,counts in sorted(parent_counts.items())}
            require(all(v["total"] == 210 for v in exposures.values()),"Unequal parent exposure")
            if arm == "B_curriculum":
                require(all(v["phrase"] == 70 and v["full_sentence"] == 140 for v in exposures.values()),"Unbalanced curriculum phases")
            else:
                require(all(v["phrase"] == 0 and v["full_sentence"] == 210 for v in exposures.values()),"Unexpected direct-sentence exposure")
            runs.append({"run_id":f"{arm}__seed{saved['seed']}","seed":saved["seed"],"arm":arm,"totals":dict(total),"phases":{p:dict(c) for p,c in phases.items()},"per_parent_exposures":exposures,
                "padding_fraction_of_dense":total["padding_frames"]/total["dense_processed_frames"]})
    sums = {key:sum(r["totals"][key] for r in runs) for key in runs[0]["totals"]}
    pairs = []
    for seed in SEEDS:
        a,b = [next(r for r in runs if r["seed"] == seed and r["arm"] == arm) for arm in ARMS]
        pairs.append({"seed":seed,"B_minus_A_actual_frames":b["totals"]["actual_frames"]-a["totals"]["actual_frames"],
            "B_minus_A_dense_processed_frames":b["totals"]["dense_processed_frames"]-a["totals"]["dense_processed_frames"],
            "B_minus_A_padding_frames":b["totals"]["padding_frames"]-a["totals"]["padding_frames"],
            "B_actual_frame_reduction_percent":100*(a["totals"]["actual_frames"]-b["totals"]["actual_frames"])/a["totals"]["actual_frames"]})
    return {"schema":"offline-curriculum-exposure-accounting-v1","runs":runs,"totals_all_four_runs":sums,"paired_differences":pairs,
        "interpretation":"Equal optimizer updates and parent exposure; token count, visible source frames, dense work and full-sentence exposure differ. This is a curriculum-design comparison, not an isolated temporal-order causal estimate.","raw_clip_padding":0,"training_execution_performed":False}


def evaluation_plan(records: list[dict[str,Any]]) -> dict[str,Any]:
    """Fix all thirty clips and one frame permutation shared across all model runs."""
    items = []
    for record in sorted(records,key=lambda r:(r["split"],r["kind"],r["parent_id"])):
        seed_text = "offline-stage4-eval-shuffle-v1|"+record["clip_id"]
        seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],"big")
        permutation = list(range(record["T"]))
        random.Random(seed).shuffle(permutation)
        require(permutation != list(range(record["T"])),"Identity shuffle is not a useful diagnostic")
        items.append({"clip_id":record["clip_id"],"parent_id":record["parent_id"],"split":record["split"],"kind":record["kind"],"T":record["T"],"text":record["text"],
            "static_frame_index":record["T"]//2,"shuffled_frame_indices":permutation,"shuffle_seed":seed,
            "primary_at_final_normal_only":record["split"] == "dev" and record["kind"] == "full_sentence"})
    reference = [r for r in records if r["split"] == "dev" and r["kind"] == "full_sentence"]
    return {"schema":"offline-curriculum-evaluation-plan-v1","checkpoints":list(CHECKPOINTS),"conditions":["normal","static","shuffled"],"items":items,
        "batch_size":1,"clips_per_condition":30,"evaluations_per_checkpoint_per_run":90,"evaluations_per_run":270,"total_evaluations_four_runs":1080,
        "actual_frames_per_checkpoint_per_run":sum(r["T"] for r in records)*3,"total_evaluation_frames":sum(r["T"] for r in records)*3*3*4,
        "primary_final_reference_characters":sum(len(r["text"].strip()) for r in reference),"primary_final_reference_words":sum(len(r["text"].split()) for r in reference),
        "condition_contract":{"normal":"Original chronological cached frames.","static":"Repeat exactly the middle frame T//2 for T positions; true length remains T.","shuffled":"Apply recorded full permutation once; same permutation for both arms, both initialization seeds and all checkpoints."},
        "metrics":{"primary":"Final update1260 dev full_sentence3 normal corpus CER (sum edits/sum reference characters).","secondary":["WER and exact3 on the same primary cohort","Per-seed and two-seed paired B-A differences; report both seeds individually","Training sentence12 and training phrase12; dev phrase3","Normal/static/shuffled diagnostics at all fixed checkpoints"],
            "CER":"NFC normalization and outer strip only; retain internal spaces and all prediction error characters. Corpus aggregate, not mean clip CER.",
            "WER":"Levenshtein on whitespace-separated words; aggregate edit/reference-word totals.","exact":"NFC+outer-strip prediction equals reference exactly.",
            "failure":"Report every frozen clip and count its failure. Empty prediction contributes full deletions; do not silently exclude or report failure as successful blank decoding.",
            "prediction_decode":"Argmax -> CTC collapse adjacent repeats -> remove blank0 -> compose_jamos(strict=False), retain U+FFFD; save raw_argmax_ids and collapsed_ids. Frozen core.greedy using strict label decoding is prohibited."},
        "checkpoint_selection":"Final state at1260 is primary. 420/840 diagnostic outputs cannot choose best weights, tune the LR or change the planned order.","execution_performed":False}


def main() -> None:
    """Produce deterministic schedules, accounting and a comparison protocol only."""
    try:
        records,hashes,handoff = input_records()
        schedules = [schedule(seed,records) for seed in SEEDS]
        schedule_files = []
        for saved in schedules:
            path = BASE/f"schedule_seed{saved['seed']}.json"
            save(path,saved)
            schedule_files.append(bind(path))
        counts = accounting(schedules,records)
        save(BASE/"schedule_accounting.json",counts)
        evaluation = evaluation_plan(records)
        save(BASE/"evaluation_plan.json",evaluation)
        data = {"schema":"offline-stage4-bound-cache-metadata-v1","handoff":bind(HANDOFF),"records":records,"source_hashes":hashes,
            "NPZ_payloads_read":0,"NPZ_hash_verification_scope":"Inherit the stage3 audited NPZ digest; metadata only read here. The stage4/5 loader must verify actual NPZ bytes and tensor contract independently."}
        save(BASE/"data_bindings.json",data)
        protocol = {"schema":"offline-stage4-direct-versus-curriculum-protocol-v1","status":"comparison_design_prepared_no_main_training",
            "objective":"Compare direct full-sentence learning with a phrase-to-sentence curriculum from identical fresh compact visual initializations in a small fixed paired pilot.",
            "data":bind(BASE/"data_bindings.json"),"source_hashes":hashes,"schedule_files":schedule_files,"accounting":bind(BASE/"schedule_accounting.json"),"evaluation":bind(BASE/"evaluation_plan.json"),
            "arms":{"A_sentence":{"updates_1_1260":"full_sentence"},"B_curriculum":{"updates_1_420":"phrase","updates_421_1260":"full_sentence"}},
            "initialization":{"seeds":[42,43],"fresh_initialization":True,"A_B_initial_state_equal_within_seed":True,"old_model_weights_used":False,"pretrained_weights_used":False,
                "reason":"Isolate this curriculum design from teacher feature transfer. Existing teacher features belong to earlier sampling grids and are not aligned to the new recording-anchored D inputs."},
            "model":{"family":"compact visual CTC","classes":69,"input":"RGB 96x96, native D-grid25fps, variable T","normalization":handoff["normalization"],"teacher_model_or_features_used":False,
                "audio_input":False,"architecture_contract":"Same compact parameterization and initialization in A/B. True-length-aware temporal computation and deterministic equivalent spatial pooling must be verified in the separate stage4 engine binding."},
            "optimization":{"runs":4,"updates_per_run":1260,"total_planned_main_updates":5040,"batch_size":2,"dense_batch":True,"gradient_accumulation":False,
                "raw_clip_padding":False,"batch_padding":"Repeat each shorter clip's last valid frame to the batch max T. Preserve true lengths through all temporal computation and CTC.",
                "loss":"FP32 CPU CTC reduction=mean: each example CTC divided by its target length, averaged over two examples; GPU autograd retained. No feature loss.",
                "optimizer":{"name":"Adam","betas":[.9,.999],"eps":1e-8,"weight_decay":0.,"gradient_clip_norm":1.},
                "lr_schedule":[{"start_update":1,"end_update":840,"lr":.001},{"start_update":841,"end_update":1050,"lr":.0003},{"start_update":1051,"end_update":1260,"lr":.0001}],
                "precision":"FP32, AMP off, TF32 off, strict deterministic algorithms; separately verified equivalent deterministic spatial pooling", "augmentation":"None during training; static/shuffle are evaluation-only diagnostics."},
            "controls":{"same_parent_order_across_arms":True,"same_parent_exposures":210,"B_phrase_exposures_per_parent":70,"B_sentence_exposures_per_parent":140,"A_sentence_exposures_per_parent":210,
                "controlled_budget":"Optimizer update count and parent visits; not equal visible frames, tokens, dense FLOPs or wall time.","data_order_independent_of_model_RNG":True,"shuffle_order_is_recorded_not_regenerated_at_resume":True,
                "causal_limit":"Both input duration/content and curriculum phase differ. Results cannot identify a pure ordering effect or prove that smaller units alone are better."},
            "state_and_evaluation":{"checkpoints":[420,840,1260],"primary_checkpoint":1260,"best_checkpoint_selection":False,"checkpoint_fields":["model","optimizer","Python/NumPy/torch CPU and all CUDA RNG states","completed update and next schedule row","data/protocol/engine/runtime digests"],
                "required_stage4_checks":["A/B initialization equality per seed","Variable-length valid logits insensitive to paired clip length/padding","Backward finite and optimizer changes genuine parameters","Exact restored state and declared deterministic continuation checks","Prediction decoder preserves invalid jamo as U+FFFD"]},
            "execution_binding":"Stage4 root binds actual engine code/runtime and separate 48-step smoke plan. Main stage5 invocation additionally needs separate resource/time-budget record based on measured smoke; this protocol does not launch it.",
            "limitations":["12 training parents and 3 internal development parents, previously used speakers; no unseen-speaker or final-test claim","Phrase/full-sentence pairs share source pixels and are not independent examples","CTC-feasible, technically reviewed pilot inputs retain provisional source flags","Rank9 display support overlaps the next label by2.6667ms, but selected original frame support audit found no intersection; preserve flag","Acoustic OOV parent C293_A012_019 remains excluded from both arms","No direct numerical comparison to older feature-transfer experiments with different sampling/model initialization/data"],
            "training_ready":False,"main_training_updates_executed_here":0,"neural_calls_here":0,"service_changes":0}
        save(BASE/"comparison_protocol.json",protocol)
        print(json.dumps({"planned_runs":4,"planned_updates":5040,"actual_training_updates":0,"actual_frames":counts["totals_all_four_runs"]["actual_frames"],"dense_frames":counts["totals_all_four_runs"]["dense_processed_frames"],"padding_frames":counts["totals_all_four_runs"]["padding_frames"],"planned_evaluations":1080}))
    except Exception as exc:
        raise RuntimeError("Metadata-only comparison preparation failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
