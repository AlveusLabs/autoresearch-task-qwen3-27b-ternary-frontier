from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from competition_packs.qwen3_27b_shared import shared


DEFAULT_SEED = 20260603
DEFAULT_ZERO_FRACTION = 0.70
DEFAULT_TERNARY_SCALE = 0.02
DEFAULT_OUTPUT = Path("/tmp/autoresearch-qwen3-27b-ternary-random/artifact.zip")

VOCAB_SIZE = 257
N_POSITIONS = 512
N_EMBD = 64
N_LAYER = 2
N_HEAD = 4
TERNARY_BUDGET_BITS_PER_PARAMETER = (0.90 * 2.0) + (0.10 * (4.0 + (16.0 / 128.0)))


def build_submission(*, seed: int, time_budget_seconds: int, debug_dataset_name: str | None = None) -> dict[str, Any]:
    del seed, time_budget_seconds, debug_dataset_name
    artifact = shared.default_submission(quant_mode="ternary", kernel_task=False)
    for row in artifact["layers"]:
        if row["name"] == "attn_out":
            row["high_precision_fraction"] = 0.02
            row["threshold_multiplier"] = 0.65
    shared.attach_q4_rescue_values(artifact, quant_mode="ternary")
    return artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_deterministic_zip(source_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(
        output_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(source_dir).as_posix())
            info.date_time = (2026, 6, 3, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def create_random_ternary_artifact(
    *,
    output_path: Path,
    seed: int = DEFAULT_SEED,
    zero_fraction: float = DEFAULT_ZERO_FRACTION,
    ternary_scale: float = DEFAULT_TERNARY_SCALE,
) -> dict[str, Any]:
    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception as exc:  # pragma: no cover - depends on local training image
        raise RuntimeError("artifact generation requires torch and transformers") from exc

    if not 0.0 <= zero_fraction < 1.0:
        raise ValueError("zero_fraction must be in [0, 1)")
    if ternary_scale <= 0.0 or not math.isfinite(float(ternary_scale)):
        raise ValueError("ternary_scale must be positive and finite")

    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=N_POSITIONS,
        n_ctx=N_POSITIONS,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    ternary_count = 0
    zero_count = 0
    nonzero_count = 0
    with torch.no_grad():
        for parameter in model.parameters():
            shape = parameter.shape
            zeros = torch.rand(shape, generator=generator, dtype=torch.float32) < float(zero_fraction)
            signs = torch.where(
                torch.rand(shape, generator=generator, dtype=torch.float32) < 0.5,
                -float(ternary_scale),
                float(ternary_scale),
            )
            values = torch.where(zeros, torch.zeros((), dtype=torch.float32), signs)
            parameter.copy_(values.to(dtype=parameter.dtype))
            ternary_count += int(parameter.numel())
            zero_count += int(zeros.sum().item())
            nonzero_count += int(parameter.numel()) - int(zeros.sum().item())

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    with tempfile.TemporaryDirectory(prefix="autoresearch-random-ternary-model-") as temp_dir:
        model_dir = Path(temp_dir) / "model"
        model.save_pretrained(model_dir, safe_serialization=True, max_shard_size="10GB")
        _write_deterministic_zip(model_dir, output_path)

    artifact_size_bytes = int(output_path.stat().st_size)
    artifact_bits_per_parameter = float((artifact_size_bytes * 8) / max(1, parameter_count))
    manifest = {
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": _sha256(output_path),
        "artifact_size_bytes": artifact_size_bytes,
        "artifact_bits_per_parameter": artifact_bits_per_parameter,
        "max_artifact_bits_per_parameter": TERNARY_BUDGET_BITS_PER_PARAMETER,
        "parameter_count": parameter_count,
        "vocab_size": VOCAB_SIZE,
        "n_positions": N_POSITIONS,
        "seed": int(seed),
        "zero_fraction_target": float(zero_fraction),
        "ternary_scale": float(ternary_scale),
        "ternary_parameter_count": ternary_count,
        "ternary_fraction": float(ternary_count / max(1, parameter_count)),
        "zero_count": zero_count,
        "nonzero_ternary_count": nonzero_count,
        "nonzero_ternary_fraction": float(nonzero_count / max(1, parameter_count)),
        "non_ternary_fraction": 0.0,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a random mostly-ternary HF artifact for local replay.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--zero-fraction", type=float, default=DEFAULT_ZERO_FRACTION)
    parser.add_argument("--ternary-scale", type=float, default=DEFAULT_TERNARY_SCALE)
    args = parser.parse_args()

    manifest = create_random_ternary_artifact(
        output_path=args.output,
        seed=args.seed,
        zero_fraction=args.zero_fraction,
        ternary_scale=args.ternary_scale,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
