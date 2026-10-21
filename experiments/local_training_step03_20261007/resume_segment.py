"""Recover from a complete checkpoint even if a run result was never written.

The frozen preflight plan remains the authority. This recovery entry point does
not require continuous/run_result.json and never extends the approved budget.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import runner
import train

BASE = Path(__file__).resolve().parent


def recover(checkpoint: Path, name: str, stop_update: int | None = None) -> dict:
    """Load an intact local snapshot into a new isolated continuation directory."""
    try:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(BASE) or checkpoint.suffix != ".pt":
            raise ValueError("Recovery requires this experiment's local checkpoint")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("Use a simple lowercase recovery name")
        plan, loaded = train.verify_plan()
        saved = runner.load_checkpoint(checkpoint)
        stop = plan["updates"] if stop_update is None else stop_update
        if not saved["completed_updates"] < stop <= plan["updates"]:
            raise ValueError("Recovery must advance within the frozen update budget")
        arm = saved["identity"]["arm"]
        output = BASE / "recoveries" / name
        result = runner.run_segment(plan, loaded["clips"], arm, output,
                                    stop_update=stop, resume_checkpoint=checkpoint, device="cuda:0")
        result["recovery_entrypoint"] = runner.core.bind(Path(__file__))
        result["plan_file"] = runner.core.bind(train.PLAN_PATH)
        runner.core.save_json(output / "recovery_result.json", result)
        return result
    except Exception as exc:
        raise RuntimeError("Checkpoint recovery failed without accepting a new result") from exc


def main() -> None:
    """Run a recovery only when explicitly invoked with a checkpoint and name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stop-update", type=int)
    args = parser.parse_args()
    result = recover(args.checkpoint, args.name, args.stop_update)
    print(f"Recovered {result['arm']} through update {result['completed_updates']}", flush=True)


if __name__ == "__main__":
    main()
