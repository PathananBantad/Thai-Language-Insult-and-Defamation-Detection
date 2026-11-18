"""Shared, validation-selected transformer fine-tuning loop for binary experiment."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import Adafactor

from .binary_metrics import calculate_binary_metrics
from .binary_preprocessing import PREPROCESSING_VERSION

LABEL_MAPPING = {"0": "Insult", "1": "Defamation"}


@dataclass(frozen=True)
class TransformerConfig:
    max_length: int = 128
    batch_size: int = 1
    gradient_accumulation: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    seed: int = 42
    fp16: bool = True
    class_weighting: str = "balanced"
    optimizer: str = "AdamW"


class _TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.texts = frame["text_norm"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def _load_binary_split(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"text_raw": "string", "text_norm": "string", "source": "string"})
    required = {"sample_id", "split", "source_row", "text_raw", "text_norm", "label", "label_name", "source", "is_synthetic"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split} is missing columns: {sorted(missing)}")
    if frame.empty or set(frame["split"].astype(str)) != {split}:
        raise ValueError(f"{split} data has no rows or invalid split membership")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{split} contains duplicate sample IDs")
    if frame[["text_raw", "text_norm", "label_name", "source"]].isna().any().any():
        raise ValueError(f"{split} contains missing required values")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{split} has non-binary labels")
    expected_names = frame["label"].map({0: "Insult", 1: "Defamation"})
    if not frame["label_name"].astype(str).eq(expected_names).all():
        raise ValueError(f"{split} label names do not match binary mapping")
    return frame


def _collate(tokenizer: Any, max_length: int):
    def collate(rows: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*rows)
        batch = tokenizer(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
    return collate


def _predict(model: Any, tokenizer: Any, frame: pd.DataFrame, config: TransformerConfig, device: torch.device) -> tuple[list[int], list[list[float]]]:
    loader = DataLoader(_TextDataset(frame), batch_size=config.batch_size, shuffle=False, collate_fn=_collate(tokenizer, config.max_length))
    preds: list[int] = []
    probs: list[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda" and config.fp16):
                logits = model(**batch).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu()
            preds.extend(probabilities.argmax(dim=-1).tolist())
            probs.extend(probabilities.tolist())
    return preds, probs


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_transformer(
    *, model_name: str, model_id: str, revision: str, optimizer_name: str,
    data_dir: Path, output_dir: Path, config: TransformerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TransformerConfig(optimizer=optimizer_name)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite model output: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured transformer training run; no CPU fallback is configured")
    free_memory, total_memory = torch.cuda.mem_get_info()
    if total_memory < 5 * 1024**3 or free_memory < 4 * 1024**3:
        raise RuntimeError(f"Insufficient GPU memory before training: free={free_memory} total={total_memory} bytes")
    train = _load_binary_split(data_dir / "train.csv", "train")
    validation = _load_binary_split(data_dir / "validation.csv", "validation")
    if set(train.label.unique()) != {0, 1} or set(validation.label.unique()) != {0, 1}:
        raise ValueError("Train and Validation must each contain both binary classes")
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("binary_training")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda")
    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=False)
        model_config = AutoConfig.from_pretrained(model_id, revision=revision)
        model_config.num_labels = 2
        model_config.id2label = {0: "Insult", 1: "Defamation"}
        model_config.label2id = {"Insult": 0, "Defamation": 1}
        model_config.problem_type = "single_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, config=model_config, ignore_mismatched_sizes=True
        )
        model.to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        labels = train["label"].to_numpy(dtype=np.int64)
        class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=labels)
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        if optimizer_name == "Adafactor":
            optimizer = Adafactor(
                model.parameters(), lr=cfg.learning_rate, scale_parameter=False,
                relative_step=False, warmup_init=False, weight_decay=cfg.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        train_loader = DataLoader(
            _TextDataset(train), batch_size=cfg.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed), collate_fn=_collate(tokenizer, cfg.max_length),
        )
        updates_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation)
        total_updates = updates_per_epoch * cfg.epochs
        warmup_steps = int(math.ceil(total_updates * cfg.warmup_ratio))
        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps:
                return max((step + 1) / warmup_steps, 1e-8)
            return max((total_updates - step) / max(total_updates - warmup_steps, 1), 0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)
        best_macro_f1 = -1.0
        best_epoch = 0
        history: list[dict[str, Any]] = []
        logger.info("Model=%s id=%s revision=%s device=%s GPU free=%d/%d", model_name, model_id, revision, device, free_memory, total_memory)
        logger.info("Train=%d validation=%d class_weights=%s config=%s", len(train), len(validation), class_weights.tolist(), asdict(cfg))
        for epoch in range(1, cfg.epochs + 1):
            started = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            optimizer_steps = 0
            skipped_optimizer_steps = 0
            for step, batch in enumerate(train_loader):
                group_start = (step // cfg.gradient_accumulation) * cfg.gradient_accumulation
                group_size = min(cfg.gradient_accumulation, len(train_loader) - group_start)
                labels_batch = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.fp16):
                    logits = model(**batch).logits
                    loss = torch.nn.functional.cross_entropy(logits.float(), labels_batch, weight=class_weight_tensor)
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()
                running_loss += float(loss.detach().cpu())
                group_end = (step + 1) % cfg.gradient_accumulation == 0 or step + 1 == len(train_loader)
                if group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.get_scale() >= scale_before:
                        scheduler.step()
                        optimizer_steps += 1
                    else:
                        skipped_optimizer_steps += 1
            val_pred, val_probs = _predict(model, tokenizer, validation, cfg, device)
            val_metrics = calculate_binary_metrics(
                validation["label"].astype(int).tolist(), val_pred, val_probs,
                validation["source"].astype(str).tolist(),
            )
            record = {
                "epoch": epoch,
                "mean_train_loss": running_loss / max(len(train_loader), 1),
                "validation": val_metrics,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_steps": optimizer_steps,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "seconds": round(time.time() - started, 2),
            }
            history.append(record)
            logger.info("Epoch %d/%d train_loss=%.5f validation_macro_f1=%.5f optimizer_steps=%d skipped=%d seconds=%s", epoch, cfg.epochs, record["mean_train_loss"], val_metrics["macro_f1"], optimizer_steps, skipped_optimizer_steps, record["seconds"])
            _write_json(output_dir / "validation_history.json", history)
            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                model.save_pretrained(output_dir / "model", safe_serialization=True)
                tokenizer.save_pretrained(output_dir / "model")
        metadata = {
            "model_type": "transformer",
            "model_name": model_name,
            "model_id": model_id,
            "model_revision": revision,
            "label_mapping": LABEL_MAPPING,
            "preprocessing_version": PREPROCESSING_VERSION,
            "input_column": "text_norm",
            "max_length": cfg.max_length,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "train_label_counts": {str(int(k)): int(v) for k, v in train.label.value_counts().items()},
            "validation_label_counts": {str(int(k)): int(v) for k, v in validation.label.value_counts().items()},
            "class_weights": {str(i): float(value) for i, value in enumerate(class_weights)},
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_macro_f1,
            "selection_metric": "validation macro_f1",
            "optimizer": optimizer_name,
            "training_config": asdict(cfg),
            "device": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": int(total_memory),
            "fp16": cfg.fp16,
            "synthetic_defamation_train": int(((train.label == 1) & (train.source.str.casefold() == "synthetic")).sum()),
            "real_or_other_defamation_train": int(((train.label == 1) & (train.source.str.casefold() != "synthetic")).sum()),
            "test_used_for_training_or_selection": False,
        }
        _write_json(output_dir / "metadata.json", metadata)
        _write_json(output_dir / "validation_selection.json", {"best_epoch": best_epoch, "best_validation_macro_f1": best_macro_f1, "history": history})
        logger.info("Training completed; best checkpoint saved at epoch %d", best_epoch)
        return metadata
    except Exception as exc:
        _write_json(output_dir / "run_error.json", {"model_name": model_name, "model_id": model_id, "revision": revision, "error_type": type(exc).__name__, "error": str(exc)})
        logger.exception("Training failed")
        raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
