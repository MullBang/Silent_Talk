"""Independent CPU CTC oracle and exhaustive tiny-path calibration.

Only saved probability arrays are consumed. No network module is constructed,
no checkpoint is deserialized, and no gradient/backward call is performed.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def collapse(path):
    out, previous = [], None
    for token in path:
        if token != previous and token != 0:
            out.append(int(token))
        previous = token
    return tuple(out)


def minimum_frames(target):
    return len(target) + sum(a == b for a, b in zip(target, target[1:]))


def torch_nll(log_probs, target):
    matrix = np.asarray(log_probs, dtype=np.float64)
    target = tuple(int(i) for i in target)
    assert matrix.ndim == 2 and len(matrix) > 0 and np.isfinite(matrix).all()
    assert all(0 < i < matrix.shape[1] for i in target)
    with torch.inference_mode():
        value = torch.nn.functional.ctc_loss(
            torch.from_numpy(np.array(matrix, copy=True)).unsqueeze(1),
            torch.tensor(target, dtype=torch.int64),
            torch.tensor([len(matrix)], dtype=torch.int64),
            torch.tensor([len(target)], dtype=torch.int64),
            blank=0, reduction="sum", zero_infinity=False,
        )
    return float(value.item())


def check_tiny():
    torch.set_num_threads(1)
    targets = [target for length in range(4) for target in itertools.product([1, 2], repeat=length)]
    evidence, maximum_difference, enumerated_paths = [], 0.0, 0
    feasible = impossible = 0
    for length in range(1, 6):
        logits = np.asarray([[0.13 * (i + 1), -0.17 * (i + 2), 0.08 * (i % 2)] for i in range(length)], dtype=np.float64)
        exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exponent / exponent.sum(axis=1, keepdims=True)
        mass = defaultdict(float)
        for path in itertools.product(range(3), repeat=length):
            product = math.prod(float(probs[i, token]) for i, token in enumerate(path))
            mass[collapse(path)] += product
            enumerated_paths += 1
        assert abs(sum(mass.values()) - 1.0) < 1e-12
        for target in targets:
            expected = -math.log(mass[target]) if mass[target] else math.inf
            actual = torch_nll(np.log(probs), target)
            predicted_feasibility = length >= minimum_frames(target)
            assert math.isfinite(actual) == math.isfinite(expected) == predicted_feasibility
            if math.isfinite(expected):
                difference = abs(actual - expected)
                assert difference < 1e-12
                maximum_difference = max(maximum_difference, difference)
                feasible += 1
            else:
                assert actual == math.inf
                difference = None
                impossible += 1
            evidence.append({"T": length, "target_ids": list(target), "minimum_frames": minimum_frames(target),
                             "feasible": predicted_feasibility, "enumeration_NLL": expected if predicted_feasibility else None,
                             "torch_NLL": actual if predicted_feasibility else None, "absolute_difference": difference})
    assert collapse([1, 1, 0, 1]) == (1, 1)
    assert collapse([0, 0, 0]) == ()
    assert collapse([1, 1, 2, 2]) == (1, 2)
    return {"schema": "offline-stage5b-independent-ctc-calibration-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "implementation": "torch.nn.functional.ctc_loss; CPU float64; no neural model",
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "enumerated_raw_paths": enumerated_paths, "target_cases": len(evidence),
            "finite_cases": feasible, "impossible_cases": impossible,
            "empty_target_cases": 5, "maximum_absolute_difference": maximum_difference,
            "checks": evidence, "neural_forward_calls": 0, "optimizer_updates": 0,
            "checkpoint_tensor_loads": 0, "GPU_calls": 0}


if __name__ == "__main__":
    path = HERE / "ctc_calibration.json"
    assert not path.exists()
    result = check_tiny()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "enumerated_raw_paths", "target_cases", "finite_cases", "impossible_cases", "maximum_absolute_difference"]}))
