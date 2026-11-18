#!/usr/bin/env python
"""Fine-tune the pinned PhayaThaiBERT base checkpoint for binary classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.binary_training import TransformerConfig, train_transformer

MODEL_ID = "clicknext/phayathaibert"
MODEL_REVISION = "a81ece95beb7a24271905d30c23887d343193c62"
DEFAULT_DATA = ROOT / "datasets" / "processed" / "binary_insult_defamation_v1"
DEFAULT_OUTPUT = ROOT / "models" / "experiments" / "binary_insult_defamation_v1" / "phayathaibert_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--optimizer", choices=("Adafactor", "AdamW"), default="Adafactor")
    args = parser.parse_args()
    config = TransformerConfig(epochs=args.epochs, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation, learning_rate=5e-5, optimizer=args.optimizer)
    train_transformer(model_name="PhayaThaiBERT", model_id=MODEL_ID, revision=MODEL_REVISION, optimizer_name=args.optimizer, data_dir=args.data_dir, output_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()
