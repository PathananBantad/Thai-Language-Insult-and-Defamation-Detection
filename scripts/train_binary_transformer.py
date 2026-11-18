#!/usr/bin/env python
"""Shared validation-only checkpoint selection loop for binary transformers."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.thai_mod_api.binary_training import _TextDataset, _collate


def predict_transformer_frame(
    model: Any, tokenizer: Any, frame: pd.DataFrame, *, max_length: int,
    device: torch.device, batch_size: int = 4,
) -> tuple[list[int], list[list[float]]]:
    loader = DataLoader(
        _TextDataset(frame), batch_size=batch_size, shuffle=False,
        collate_fn=_collate(tokenizer, max_length),
    )
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu()
            predictions.extend(probs.argmax(dim=-1).tolist())
            probabilities.extend(probs.tolist())
    return predictions, probabilities
