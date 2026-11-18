#!/usr/bin/env python
"""Fine-tune the pinned WangchanBERTa checkpoint for binary Insult/Defamation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.binary_training import TransformerConfig, train_transformer

MODEL_ID = "airesearch/wangchanberta-base-att-spm-uncased"
MODEL_REVISION = "b81d38df6b4755dbedec0bfea863c9956cbb963e"
DEFAULT_DATA = ROOT / "datasets" / "processed" / "binary_insult_defamation_v1"
DEFAULT_OUTPUT = ROOT / "models" / "experiments" / "binary_insult_defamation_v1" / "wangchanberta_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    args = parser.parse_args()
    config = TransformerConfig(epochs=args.epochs, batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation, learning_rate=2e-5, optimizer="AdamW")
    train_transformer(model_name="WangchanBERTa", model_id=MODEL_ID, revision=MODEL_REVISION, optimizer_name="AdamW", data_dir=args.data_dir, output_dir=args.output_dir, config=config)


if __name__ == "__main__":
    main()
