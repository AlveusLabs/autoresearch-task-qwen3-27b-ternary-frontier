from __future__ import annotations

from typing import Any

from competition_packs.qwen3_27b_shared import shared


def build_submission(*, seed: int, time_budget_seconds: int, debug_dataset_name: str | None = None) -> dict[str, Any]:
    artifact = shared.default_submission(quant_mode="ternary", kernel_task=False)
    for row in artifact["layers"]:
        if row["name"] == "attn_out":
            row["high_precision_fraction"] = 0.02
            row["threshold_multiplier"] = 0.65
    shared.attach_q4_rescue_values(artifact, quant_mode="ternary")
    return artifact
