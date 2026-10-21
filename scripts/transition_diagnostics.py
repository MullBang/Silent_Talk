"""Deterministic train-only transition plans and non-mutating gradient telemetry.

The caller owns training, AMP unscale, clipping, optimizer steps and evaluation.
Gradient norms must be captured after unscale and again after clipping. Forward
passes, optimizer operations, checkpoint loading and dataset I/O are absent here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random

import torch

GROUP_PREFIXES = {"cnn": "conv.", "projection": "projection.", "lstm": "lstm.", "output": "fc."}
ParameterGroups = dict[str, list[tuple[str, torch.nn.Parameter]]]
METRIC_DEFINITIONS = {
    "gradient_timing": "caller invokes after AMP unscale before clipping, and again after clipping; helper never unscales or clips",
    "l2_norm": "sqrt(sum of squares over all present gradient elements); null if any element is nonfinite",
    "finite_l2_norm": "norm of only finite elements, for diagnostics; not a substitute for l2_norm when nonfinite elements exist",
    "missing_grad_tensors": "trainable parameter tensors whose grad is None; distinct from zero gradients",
    "zero_grad_tensors": "present gradient tensors with all elements exactly zero",
    "zero_grad_elements": "elements exactly zero among present gradients; missing elements are excluded",
    "nonfinite_grad_elements": "NaN or positive/negative infinity among present gradient elements",
    "parameter_update_absolute_l2": "norm of actual parameters after minus snapshot before the attempted optimizer step",
    "parameter_update_relative_l2": "absolute L2 update / max(before parameter L2 norm,1e-12); null when inputs are nonfinite",
    "schedule": "one shuffled permutation of every branch sample per epoch, contiguous full batches; independent local Python RNG",
    "exposures": "planned sample presentations per attempted update; not successful AMP optimizer updates",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: int, name: str, minimum: int = 0) -> None:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{name} must be an integer >= {minimum}")


def length_bin(frames: int) -> str:
    _integer(frames, "frames", 1)
    require(frames <= 200, "frames must be <=200")
    start = ((frames - 1) // 50) * 50 + 1
    return f"{start:03d}-{start+49:03d}"


def _distribution(ids: list[str], by_id: dict[str, dict]) -> dict:
    rows = [by_id[sample_id] for sample_id in ids]
    return {"clips": len(rows),
            "speaker": dict(sorted(Counter(row["speaker_id"] for row in rows).items())),
            "length_bin": dict(sorted(Counter(length_bin(row["frames"]) for row in rows).items())),
            "speaker_length_bin": dict(sorted(Counter(
                f"{row['speaker_id']}/{length_bin(row['frames'])}" for row in rows).items()))}


def select_transition_cohorts(records: list[dict], seed: int = 42) -> dict:
    """Preserve shortest16, then round-robin48 from metadata-only train strata.

    Cells first sort by (speaker,length_bin), then shuffle once with a local
    Random(seed). Within each cell use SHA256('step12-selection-{seed}:'+id),
    with ID as a deterministic tie break. Exhausted cells are skipped. No val,
    test, text, label, score, loss or prediction participates in selection.
    """
    _integer(seed, "seed")
    train = [row for row in records if row.get("split") == "train"]
    require(len(train) >= 64, "At least64 train clips required")
    for row in train:
        require(isinstance(row.get("id"), str) and bool(row["id"]), "Nonempty train ID required")
        require(isinstance(row.get("speaker_id"), str) and bool(row["speaker_id"]), "Train speaker required")
        length_bin(row.get("frames"))
    by_id = {row["id"]: row for row in train}
    require(len(by_id) == len(train), "Duplicate train IDs")
    nontrain_ids = {row.get("id") for row in records if row.get("split") != "train"}
    require(not set(by_id) & nontrain_ids, "Train ID also appears outside train")
    short = [row["id"] for row in sorted(train, key=lambda row: (row["frames"], row["id"]))[:16]]
    pools = defaultdict(list)
    for row in train:
        if row["id"] not in short:
            pools[(row["speaker_id"], length_bin(row["frames"]))].append(row["id"])
    for cell, ids in pools.items():
        pools[cell] = sorted(ids, key=lambda sample_id: (
            hashlib.sha256(f"step12-selection-{seed}:{sample_id}".encode("utf-8")).hexdigest(), sample_id))
    sorted_cells = sorted(pools)
    cells = list(sorted_cells)
    random.Random(seed).shuffle(cells)
    offsets = Counter()
    selected = []
    while len(selected) < 48:
        added = False
        for cell in cells:
            if offsets[cell] < len(pools[cell]):
                selected.append(pools[cell][offsets[cell]])
                offsets[cell] += 1
                added = True
                if len(selected) == 48:
                    break
        require(added, "Insufficient additional train clips")
    require(len(set(selected)) == 48 and not set(short) & set(selected), "Cohort overlap")
    expanded = short + selected
    strata = [{"speaker": speaker, "length_bin": bucket,
               "available_excluding_short16": len(pools[(speaker, bucket)]),
               "selected": offsets[(speaker, bucket)],
               "selected_ids": pools[(speaker, bucket)][:offsets[(speaker, bucket)]]}
              for speaker, bucket in sorted_cells]
    return {"selection_version": "train-strata-round-robin-v1", "seed": seed,
            "short16": short, "additional48": selected, "expanded64": expanded,
            "cell_order": [{"speaker": speaker, "length_bin": bucket} for speaker, bucket in cells],
            "strata": strata,
            "distributions": {name: _distribution(ids, by_id) for name, ids in (
                ("short16", short), ("additional48", selected), ("expanded64", expanded))},
            "selection": "short16=(frames,id) ascending; candidate train-only speaker x50-frame cells; sorted cells shuffled once by local seed; SHA256 within cell; round-robin48",
            "performance_used": False, "validation_used": False, "test_used": False}


def make_training_schedule(sample_ids: list[str], attempt_updates: int = 1600,
                           batch_size: int = 2, seed: int = 42) -> dict:
    """Return epoch-shuffled index batches without consuming global RNG state.

    Every completed epoch visits every sample exactly once. Require full batches
    so a partial final batch cannot change loss weighting. The last epoch may be
    truncated only by the caller's attempted-update budget. Default16/64 branch
    plans finish200/50 epochs, respectively, with3200 sample presentations each.
    """
    _integer(attempt_updates, "attempt_updates", 1)
    _integer(batch_size, "batch_size", 1)
    _integer(seed, "seed")
    require(bool(sample_ids) and all(isinstance(x, str) and x for x in sample_ids), "Nonempty IDs required")
    require(len(set(sample_ids)) == len(sample_ids), "Duplicate schedule IDs")
    require(len(sample_ids) >= batch_size and len(sample_ids) % batch_size == 0, "Full batches required")
    rng = random.Random(seed)
    batches, epoch = [], 0
    exposures = Counter({sample_id: 0 for sample_id in sample_ids})
    batches_per_epoch = len(sample_ids) // batch_size
    while len(batches) < attempt_updates:
        epoch += 1
        indices = list(range(len(sample_ids)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            pair = indices[offset:offset+batch_size]
            ids = [sample_ids[index] for index in pair]
            batches.append({"attempt": len(batches)+1, "epoch": epoch,
                            "batch_in_epoch": offset // batch_size + 1, "indices": pair, "ids": ids})
            exposures.update(ids)
            if len(batches) == attempt_updates:
                break
    encoded = json.dumps(batches, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"schedule_version": "epoch-full-index-shuffle-v1", "seed": seed, "sample_ids": list(sample_ids),
            "sample_count": len(sample_ids), "attempt_updates": attempt_updates, "batch_size": batch_size,
            "batches_per_epoch": batches_per_epoch, "completed_epochs": attempt_updates // batches_per_epoch,
            "partial_final_epoch_batches": attempt_updates % batches_per_epoch,
            "total_presentations": attempt_updates * batch_size, "exposures": dict(exposures),
            "batches_sha256": hashlib.sha256(encoded).hexdigest(), "batches": batches}


def parameter_groups(model: torch.nn.Module) -> ParameterGroups:
    """Partition all trainable parameters; reject unclassified or shared weights."""
    groups: ParameterGroups = {name: [] for name in GROUP_PREFIXES}
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        matches = [group for group, prefix in GROUP_PREFIXES.items() if name.startswith(prefix)]
        require(len(matches) == 1, f"Unclassified trainable parameter: {name}")
        require(id(parameter) not in seen, f"Shared parameter would be counted twice: {name}")
        require(parameter.is_floating_point(), f"Floating parameter required: {name}")
        seen.add(id(parameter))
        groups[matches[0]].append((name, parameter))
    require(all(groups.values()), "All four trainable parameter groups required")
    return groups


def _validated_groups(groups: ParameterGroups) -> list[tuple[str, torch.nn.Parameter]]:
    require(set(groups) == set(GROUP_PREFIXES) and all(groups.values()), "Expected four nonempty parameter groups")
    flat = [pair for group in groups.values() for pair in group]
    require(len({name for name, _ in flat}) == len(flat), "Duplicate parameter names")
    require(len({id(parameter) for _, parameter in flat}) == len(flat), "Duplicate parameter objects")
    devices = {parameter.device for _, parameter in flat}
    require(len(devices) == 1, "Parameter groups must share one device")
    for group, pairs in groups.items():
        require(all(name.startswith(GROUP_PREFIXES[group]) and parameter.requires_grad
                    and parameter.is_floating_point() for name, parameter in pairs), "Invalid parameter group")
    return flat


def _reduce(parts: list[torch.Tensor]) -> float:
    return float(torch.stack(parts).sum().item()) if parts else 0.0


def _gradient_stats(pairs: list[tuple[str, torch.nn.Parameter]]) -> dict:
    missing = 0
    present_elements = 0
    squares, nonfinites, zeros, zero_tensors, nonfinite_tensors = [], [], [], [], []
    for _, parameter in pairs:
        grad = parameter.grad
        if grad is None:
            missing += 1
            continue
        require(grad.layout == torch.strided, "Sparse gradients unsupported")
        values = grad.detach().to(dtype=torch.float64)
        finite = torch.isfinite(values)
        count = (~finite).sum()
        nonfinites.append(count)
        nonfinite_tensors.append((count > 0).to(torch.int64))
        zeros.append((values == 0).sum())
        zero_tensors.append((values == 0).all().to(torch.int64))
        squares.append(torch.where(finite, values, 0.0).square().sum())
        present_elements += values.numel()
    nonfinite = int(_reduce(nonfinites))
    finite_norm = math.sqrt(_reduce(squares))
    return {"parameter_tensors": len(pairs), "parameter_elements": sum(p.numel() for _, p in pairs),
            "present_grad_tensors": len(pairs)-missing, "present_grad_elements": present_elements,
            "missing_grad_tensors": missing, "zero_grad_tensors": int(_reduce(zero_tensors)),
            "zero_grad_elements": int(_reduce(zeros)), "nonfinite_grad_tensors": int(_reduce(nonfinite_tensors)),
            "nonfinite_grad_elements": nonfinite, "all_present_gradients_finite": nonfinite == 0,
            "l2_norm": finite_norm if nonfinite == 0 else None, "finite_l2_norm": finite_norm}


@torch.no_grad()
def gradient_metrics(groups: ParameterGroups) -> dict:
    """Inspect current gradients without changing values, storage or grad state."""
    flat = _validated_groups(groups)
    return {"groups": {name: _gradient_stats(pairs) for name, pairs in groups.items()},
            "total": _gradient_stats(flat)}


@torch.no_grad()
def snapshot_parameters(groups: ParameterGroups) -> dict[str, torch.Tensor]:
    """Independent, detached snapshots on the parameters' current device."""
    return {name: parameter.detach().clone() for name, parameter in _validated_groups(groups)}


def _update_stats(pairs: list[tuple[str, torch.nn.Parameter]], before: dict[str, torch.Tensor]) -> dict:
    old_squares, new_squares, delta_squares, old_bad, new_bad, changed = [], [], [], [], [], []
    for name, parameter in pairs:
        old, new = before[name].to(torch.float64), parameter.detach().to(torch.float64)
        old_finite, new_finite = torch.isfinite(old), torch.isfinite(new)
        old_bad.append((~old_finite).sum())
        new_bad.append((~new_finite).sum())
        old_squares.append(torch.where(old_finite, old, 0.0).square().sum())
        new_squares.append(torch.where(new_finite, new, 0.0).square().sum())
        valid = old_finite & new_finite
        delta_squares.append(torch.where(valid, new-old, 0.0).square().sum())
        changed.append((valid & (new != old)).sum())
    old_nonfinite, new_nonfinite = int(_reduce(old_bad)), int(_reduce(new_bad))
    before_norm, after_norm, delta = (math.sqrt(_reduce(values)) for values in (old_squares, new_squares, delta_squares))
    finite = old_nonfinite == new_nonfinite == 0
    return {"parameter_tensors": len(pairs), "parameter_elements": sum(p.numel() for _, p in pairs),
            "before_l2": before_norm if old_nonfinite == 0 else None,
            "after_l2": after_norm if new_nonfinite == 0 else None,
            "absolute_l2": delta if finite else None,
            "relative_l2": delta/max(before_norm, 1e-12) if finite else None,
            "before_nonfinite_elements": old_nonfinite, "after_nonfinite_elements": new_nonfinite,
            "changed_finite_elements": int(_reduce(changed)), "all_parameters_finite": finite}


@torch.no_grad()
def parameter_update_metrics(groups: ParameterGroups, before: dict[str, torch.Tensor]) -> dict:
    """Measure actual deltas, including exactly zero when an AMP step was skipped.

    A zero delta alone does not prove AMP skip: the caller must record scaler and
    optimizer-step status separately. Snapshots are not optimizer momentum/state.
    """
    flat = _validated_groups(groups)
    require(set(before) == {name for name, _ in flat}, "Snapshot parameter names mismatch")
    for name, parameter in flat:
        snapshot = before[name]
        require(isinstance(snapshot, torch.Tensor) and snapshot.shape == parameter.shape
                and snapshot.dtype == parameter.dtype and snapshot.device == parameter.device,
                f"Snapshot shape/dtype/device mismatch: {name}")
        require(snapshot.data_ptr() != parameter.data_ptr(), f"Snapshot aliases current parameter: {name}")
    return {"groups": {name: _update_stats(pairs, before) for name, pairs in groups.items()},
            "total": _update_stats(flat, before)}
