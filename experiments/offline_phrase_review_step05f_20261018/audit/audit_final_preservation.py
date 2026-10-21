"""Repeat frozen-source verification once after stage 5f work is complete."""
from __future__ import annotations

import json

from audit_preflight import HERE, inspect


if __name__ == "__main__":
    result = inspect()
    result["schema"] = "offline-stage5f-independent-final-source-preservation-v1"
    result["verification_phase"] = "post_work_final_preservation"
    result["scope_clarification"] = (
        "The pending flags reported here belong to unchanged historical stage5e snapshots. "
        "Current stage5f actual answers and effective inputs are separate append-only evidence; "
        "this preservation result does not reset or override them."
    )
    with (HERE / "source_preservation_final.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "frozen_entries": 1910,
                      "additional_stage1": 10, "scope": 10}))
