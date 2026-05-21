from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition_packs.qwen3_27b_shared.shared import (
    DEFAULT_RESCUE_BITS_PER_PARAMETER,
    QWEN3_36_27B_MODEL_ID,
    REFERENCE_QWEN36_27B_PARAMETER_COUNT,
    BenchmarkConfig,
    write_prepare_report,
)


CONFIG = BenchmarkConfig(
    pack_name="qwen3_27b_ternary_frontier",
    quant_mode="ternary",
    secondary_metric_name=None,
    quality_floor=0.78,
    allow_runtime_patch=False,
    reference_model_id=QWEN3_36_27B_MODEL_ID,
    reference_parameter_count=REFERENCE_QWEN36_27B_PARAMETER_COUNT,
    native_bits_per_parameter=2.0,
    rescue_bits_per_parameter=DEFAULT_RESCUE_BITS_PER_PARAMETER,
)


def main() -> int:
    manifest = write_prepare_report(CONFIG, pack_dir=Path(__file__).resolve().parent)
    print(f"baseline_manifest={manifest.name}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
