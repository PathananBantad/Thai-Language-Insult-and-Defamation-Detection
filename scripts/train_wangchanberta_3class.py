from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "airesearch/wangchanberta-base-att-spm-uncased"
DEFAULT_DATA_DIR = ROOT / "datasets" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "candidates" / "wangchanberta_3class"
DEFAULT_TRAINING_RUNS_DIR = ROOT / "models" / "training_runs"

LABEL_NAMES = {0: "Normal", 1: "Insult", 2: "Defamation"}
LABELS = tuple(LABEL_NAMES)
LABEL2ID = {name: label for label, name in LABEL_NAMES.items()}
REQUIRED_COLUMNS = {"text_raw", "text_norm", "label", "label_name", "source"}
RANDOM_SEED = 42
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TrainingConfig:
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation: int = 8
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    fp16: bool = True


DEFAULT_CONFIG = TrainingConfig()


def make_training_identity(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, Any]:
    def frame_digest(frame: pd.DataFrame) -> str:
        columns = sorted(REQUIRED_COLUMNS)
        payload = frame.loc[:, columns].to_json(
            orient="records",
            force_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    identity = {
        "model_id": MODEL_ID,
        "training_config": asdict(config),
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "train_sha256": frame_digest(train),
        "validation_sha256": frame_digest(validation),
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return identity


def create_training_run_dir(run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return DEFAULT_TRAINING_RUNS_DIR / f"wangchanberta_3class_{timestamp}"


def prepare_training_run_dir(run_dir: Path, *, resume: bool) -> None:
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Training run directory not found: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=False)


def validate_run_output_paths(run_dir: Path, output_dir: Path) -> None:
    resolved_run = run_dir.resolve()
    resolved_output = output_dir.resolve()
    if (
        resolved_run == resolved_output
        or resolved_run in resolved_output.parents
        or resolved_output in resolved_run.parents
    ):
        raise ValueError("Training run directory must be separate from final artifacts.")


def configure_training_logger(run_dir: Path, *, resume: bool = False) -> logging.Logger:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run directory not found: {run_dir}")

    logger_key = hashlib.sha256(str(run_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    logger = logging.getLogger(f"{__name__}.run.{logger_key}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_training_logger(logger)

    log_path = run_dir / "training.log"
    mode = "a" if resume or log_path.exists() else "x"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("Training run started in %s", run_dir)
    return logger


def close_training_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def atomic_torch_save(payload: dict[str, Any], checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_training_checkpoint(
    checkpoint_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"Unable to load training checkpoint: {checkpoint_path}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if checkpoint.get("kind") != "training_state":
        raise ValueError(f"Checkpoint is not resumable: {checkpoint_path}")
    if checkpoint.get("identity") != expected_identity:
        raise ValueError(
            "Checkpoint configuration or train/validation dataset does not match "
            f"this run: {checkpoint_path}"
        )

    required_keys = {
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "best_epoch",
        "best_validation_score",
        "best_validation_metrics",
        "best_model_state",
        "history",
        "train_loader_generator_state",
        "cpu_rng_state",
        "cuda_rng_state",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(
            f"Training checkpoint is incomplete; missing: {sorted(missing_keys)}"
        )
    return checkpoint


def validate_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} is missing required columns: {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(f"{split_name} must contain at least one row.")

    for column in ("text_norm", "text_raw", "label_name", "source"):
        values = frame[column]
        missing = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        if missing.any():
            raise ValueError(
                f"{split_name} contains missing or empty values in '{column}'."
            )

    for column in ("text_norm", "text_raw", "source"):
        if not frame[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{split_name} column '{column}' must contain text values.")

    if frame["label"].isna().any():
        raise ValueError(f"{split_name} contains missing values in 'label'.")

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.mod(1).eq(0).all():
        raise ValueError(f"{split_name} labels must be integers in {LABELS}.")
    labels = numeric_labels.astype(int)
    invalid_labels = set(labels.unique()) - set(LABELS)
    if invalid_labels:
        raise ValueError(
            f"{split_name} contains unsupported labels: {sorted(invalid_labels)}"
        )

    expected_names = labels.map(LABEL_NAMES)
    actual_names = frame["label_name"].astype(str).str.strip().str.lower()
    if not actual_names.eq(expected_names.str.lower()).all():
        raise ValueError(f"{split_name} label_name values do not match label mapping.")

    validated = frame.copy()
    validated["label"] = labels
    return validated


def load_split(data_dir: Path, split_name: str) -> pd.DataFrame:
    if split_name not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split_name}")
    path = data_dir / f"{split_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    frame = pd.read_csv(
        path,
        dtype={
            "text_raw": "string",
            "text_norm": "string",
            "label_name": "string",
            "source": "string",
        },
    )
    return validate_split(frame, split_name)


def compute_class_weights(train_labels: Sequence[int] | pd.Series) -> torch.Tensor:
    try:
        numeric_labels = np.asarray(train_labels, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Train labels must be integers in {LABELS}.") from error
    if numeric_labels.size == 0:
        raise ValueError("Train labels must not be empty.")
    if (
        not np.isfinite(numeric_labels).all()
        or not np.equal(numeric_labels, np.floor(numeric_labels)).all()
        or not np.isin(numeric_labels, LABELS).all()
    ):
        raise ValueError(f"Train labels must be in {LABELS}.")
    labels = numeric_labels.astype(np.int64)

    counts = np.bincount(labels, minlength=len(LABELS))
    if (counts == 0).any():
        missing = [LABEL_NAMES[index] for index, count in enumerate(counts) if count == 0]
        raise ValueError(f"Train split must contain every class; missing: {missing}.")

    weights = np.sqrt(labels.size / (len(LABELS) * counts.astype(np.float64)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def calculate_effective_batch_size(
    batch_size: int, gradient_accumulation: int
) -> int:
    if batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    return batch_size * gradient_accumulation


def calculate_optimizer_updates(
    number_of_batches: int, gradient_accumulation: int, epochs: int
) -> int:
    if number_of_batches <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("Batches, gradient accumulation, and epochs must be positive.")
    updates_per_epoch = math.ceil(number_of_batches / gradient_accumulation)
    return updates_per_epoch * epochs


def calculate_warmup_steps(total_updates: int, warmup_ratio: float) -> int:
    if total_updates < 0:
        raise ValueError("Total optimizer updates cannot be negative.")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("Warmup ratio must be in the range [0, 1).")
    return min(total_updates, math.ceil(total_updates * warmup_ratio))


def calculate_truncation_stats(
    texts: Sequence[str],
    tokenizer: Any,
    max_length: int,
    batch_size: int = 512,
) -> dict[str, int | float]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if batch_size <= 0:
        raise ValueError("Truncation batch size must be positive.")
    text_list = list(texts)
    if len(text_list) == 0:
        raise ValueError("At least one text is required to calculate truncation.")

    lengths: list[int] = []
    for offset in range(0, len(text_list), batch_size):
        encoded = tokenizer(
            text_list[offset : offset + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], (int, np.integer)):
            input_ids = [input_ids]
        lengths.extend(len(sequence) for sequence in input_ids)
    if len(lengths) != len(text_list):
        raise ValueError("Tokenizer returned an unexpected number of encoded texts.")

    truncated_count = sum(length > max_length for length in lengths)
    return {
        "max_length": max_length,
        "total_texts": len(lengths),
        "truncated_count": int(truncated_count),
        "truncated_fraction": float(truncated_count / len(lengths)),
    }


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same number of rows.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        self.texts = list(texts)
        self.labels = [int(label) for label in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoding = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        item = {
            key: list(encoding[key])
            for key in ("input_ids", "attention_mask")
            if key in encoding
        }
        item["labels"] = self.labels[index]
        return item


def make_collate_fn(tokenizer: Any):
    def collate(features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        labels = torch.tensor(
            [feature["labels"] for feature in features], dtype=torch.long
        )
        model_inputs = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch

    return collate


def create_dataloader(
    frame: pd.DataFrame,
    tokenizer: Any,
    config: TrainingConfig,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset = TextClassificationDataset(
        frame["text_norm"].tolist(),
        frame["label"].tolist(),
        tokenizer,
        config.max_length,
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=make_collate_fn(tokenizer),
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


def calculate_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, Any]:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Metrics require equal, non-empty true and predicted labels.")
    if set(y_true) - set(LABELS) or set(y_pred) - set(LABELS):
        raise ValueError(f"Metric labels must be in {LABELS}.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(LABELS),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "per_class": {
            str(label): {
                "name": LABEL_NAMES[label],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(LABELS)
        ).tolist(),
        "labels": list(LABELS),
        "total_support": int(len(y_true)),
    }


def evaluate_model(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    *,
    fp16: bool,
) -> dict[str, Any]:
    model.eval()
    true_labels: list[int] = []
    predictions: list[int] = []
    use_fp16 = fp16 and device.type == "cuda"
    with torch.inference_mode():
        for batch in dataloader:
            labels = batch["labels"].to(device, non_blocking=use_fp16)
            model_inputs = {
                key: value.to(device, non_blocking=use_fp16)
                for key, value in batch.items()
                if key != "labels"
            }
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_fp16
                else nullcontext()
            )
            with context:
                logits = model(**model_inputs).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    return calculate_metrics(true_labels, predictions)


def train_epoch(
    model: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    gradient_accumulation: int,
    fp16: bool,
) -> float:
    number_of_batches = len(dataloader)
    if number_of_batches == 0:
        raise ValueError("Training DataLoader must not be empty.")
    if gradient_accumulation <= 0:
        raise ValueError("Gradient accumulation must be positive.")
    if getattr(loss_fn, "reduction", None) != "sum":
        raise ValueError("Training loss must use reduction='sum' for accumulation.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_loss_denominator = 0.0
    use_fp16 = fp16 and device.type == "cuda"
    pending_batches: list[dict[str, torch.Tensor]] = []

    def train_accumulation_group(batches: list[dict[str, torch.Tensor]]) -> None:
        nonlocal total_loss, total_loss_denominator
        labels_by_batch = [
            batch["labels"].to(device, non_blocking=use_fp16) for batch in batches
        ]
        class_weights = loss_fn.weight
        if class_weights is None:
            group_denominator = float(sum(labels.numel() for labels in labels_by_batch))
        else:
            group_denominator = float(
                sum(
                    class_weights.index_select(0, labels).sum().item()
                    for labels in labels_by_batch
                )
            )
        if group_denominator <= 0:
            raise ValueError("Accumulation group must contain weighted examples.")

        for batch, labels in zip(batches, labels_by_batch):
            model_inputs = {
                key: value.to(device, non_blocking=use_fp16)
                for key, value in batch.items()
                if key != "labels"
            }
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_fp16
                else nullcontext()
            )
            with context:
                logits = model(**model_inputs).logits
                loss_sum = loss_fn(logits, labels)

            total_loss += float(loss_sum.detach().item())
            scaler.scale(loss_sum / group_denominator).backward()

        total_loss_denominator += group_denominator
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        step_optimizer_with_scaler(optimizer, scheduler, scaler)
        optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        pending_batches.append(batch)
        if len(pending_batches) == gradient_accumulation:
            train_accumulation_group(pending_batches)
            pending_batches = []

    if pending_batches:
        train_accumulation_group(pending_batches)

    return total_loss / total_loss_denominator


def step_optimizer_with_scaler(
    optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any
) -> bool:
    scaler_enabled = scaler.is_enabled()
    scale_before = scaler.get_scale() if scaler_enabled else None
    scaler.step(optimizer)
    scaler.update()

    optimizer_updated = (
        not scaler_enabled or scaler.get_scale() >= scale_before
    )
    if optimizer_updated:
        scheduler.step()
    return optimizer_updated


def fit_model(
    model: Any,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    train_labels: Sequence[int] | pd.Series,
    config: TrainingConfig,
    device: torch.device,
    *,
    checkpoint_dir: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_identity: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if (checkpoint_dir is not None or resume_from is not None) and (
        checkpoint_identity is None or checkpoint_dir is None
    ):
        raise ValueError(
            "Checkpointing requires both a checkpoint directory and identity."
        )

    class_weights = compute_class_weights(train_labels)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        reduction="sum",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_updates = calculate_optimizer_updates(
        len(train_loader), config.gradient_accumulation, config.epochs
    )
    warmup_steps = calculate_warmup_steps(total_updates, config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.fp16 and device.type == "cuda",
    )

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    start_epoch = 0

    if resume_from is not None:
        checkpoint = load_training_checkpoint(resume_from, checkpoint_identity)
        checkpoint_epoch = checkpoint["epoch"]
        if not isinstance(checkpoint_epoch, int) or not 1 <= checkpoint_epoch <= config.epochs:
            raise ValueError(
                f"Invalid completed epoch in checkpoint: {checkpoint_epoch!r}"
            )
        best_epoch = checkpoint["best_epoch"]
        best_macro_f1 = checkpoint["best_validation_score"]
        if (
            not isinstance(best_epoch, int)
            or not 1 <= best_epoch <= checkpoint_epoch
            or not isinstance(best_macro_f1, (int, float))
        ):
            raise ValueError("Invalid best-validation state in training checkpoint.")
        if not isinstance(checkpoint["history"], list):
            raise ValueError("Invalid epoch history in training checkpoint.")
        if not isinstance(checkpoint["best_model_state"], dict):
            raise ValueError("Missing best model state in training checkpoint.")
        if not isinstance(checkpoint["best_validation_metrics"], dict):
            raise ValueError("Invalid best-validation metrics in training checkpoint.")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_state = checkpoint["best_model_state"]
        best_validation_metrics = checkpoint["best_validation_metrics"]
        history = checkpoint["history"]
        start_epoch = checkpoint_epoch

        generator = getattr(train_loader, "generator", None)
        generator_state = checkpoint["train_loader_generator_state"]
        if generator_state is not None:
            if generator is None:
                raise ValueError(
                    "Checkpoint requires a train DataLoader generator to resume."
                )
            generator.set_state(generator_state)

        torch.set_rng_state(checkpoint["cpu_rng_state"])
        cuda_rng_state = checkpoint["cuda_rng_state"]
        if (device.type == "cuda") != (cuda_rng_state is not None):
            raise ValueError("CUDA device configuration does not match checkpoint.")
        if device.type == "cuda":
            if len(cuda_rng_state) != torch.cuda.device_count():
                raise ValueError("CUDA device configuration does not match checkpoint.")
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if checkpoint_dir is not None:
            atomic_torch_save(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "kind": "best_model",
                    "identity": checkpoint_identity,
                    "epoch": best_epoch,
                    "best_validation_score": best_macro_f1,
                    "model_state": best_state,
                    "validation_metrics": best_validation_metrics,
                },
                checkpoint_dir / "best.pt",
            )
        if logger is not None:
            logger.info(
                "Resumed from epoch %d; best validation macro-F1 %.6f at epoch %d",
                start_epoch,
                best_macro_f1,
                best_epoch,
            )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        if logger is not None:
            logger.info("Epoch %d/%d started", epoch, config.epochs)
        training_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            loss_fn,
            device,
            gradient_accumulation=config.gradient_accumulation,
            fp16=config.fp16,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            fp16=config.fp16,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation": validation_metrics,
            }
        )

        best_improved = validation_metrics["macro_f1"] > best_macro_f1
        if best_improved:
            best_epoch = epoch
            best_macro_f1 = validation_metrics["macro_f1"]
            best_validation_metrics = validation_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if checkpoint_dir is not None:
            generator = getattr(train_loader, "generator", None)
            atomic_torch_save(
                {
                    "format_version": CHECKPOINT_FORMAT_VERSION,
                    "kind": "training_state",
                    "identity": checkpoint_identity,
                    "epoch": epoch,
                    "model_state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_score": best_macro_f1,
                    "best_validation_metrics": best_validation_metrics,
                    "best_model_state": best_state,
                    "history": history,
                    "train_loader_generator_state": (
                        generator.get_state() if generator is not None else None
                    ),
                    "cpu_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                    ),
                },
                checkpoint_dir / "latest.pt",
            )
            if best_improved:
                atomic_torch_save(
                    {
                        "format_version": CHECKPOINT_FORMAT_VERSION,
                        "kind": "best_model",
                        "identity": checkpoint_identity,
                        "epoch": best_epoch,
                        "best_validation_score": best_macro_f1,
                        "model_state": best_state,
                        "validation_metrics": best_validation_metrics,
                    },
                    checkpoint_dir / "best.pt",
                )
        if logger is not None:
            logger.info(
                "Epoch %d/%d complete: train_loss=%.6f validation_macro_f1=%.6f "
                "best_validation_macro_f1=%.6f best_epoch=%d",
                epoch,
                config.epochs,
                training_loss,
                validation_metrics["macro_f1"],
                best_macro_f1,
                best_epoch,
            )

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "class_weights": class_weights.tolist(),
        "total_optimizer_updates": total_updates,
        "warmup_steps": warmup_steps,
    }


def resolve_cache_dir(cache_dir: Path | None = None) -> Path | None:
    if cache_dir is not None:
        return Path(cache_dir)
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return None


def load_model_and_tokenizer(
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        cache_dir=resolved_cache_dir,
        local_files_only=local_files_only,
    )
    config.num_labels = len(LABELS)
    config.label2id = LABEL2ID
    config.id2label = LABEL_NAMES
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=False,
        cache_dir=resolved_cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Checkpoint tokenizer has no padding token: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        cache_dir=resolved_cache_dir,
        local_files_only=local_files_only,
    )
    return tokenizer, model


def save_artifacts(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing candidate artifacts: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        model.save_pretrained(staging_dir)
        tokenizer.save_pretrained(staging_dir)
        (staging_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts()
    return {str(label): int(counts.get(label, 0)) for label in LABELS}


def run_training(
    data_dir: Path,
    output_dir: Path,
    device: torch.device | None,
    config: TrainingConfig = DEFAULT_CONFIG,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    run_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    run_dir = create_training_run_dir(run_dir)
    validate_run_output_paths(run_dir, output_dir)
    prepare_training_run_dir(run_dir, resume=resume)
    logger = configure_training_logger(run_dir, resume=resume)
    checkpoint_dir = run_dir / "checkpoints"
    try:
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing candidate artifacts: {output_dir}"
            )
        if device is None:
            device = get_cuda_device()

        logger.info("Final artifact directory: %s", output_dir)
        logger.info("Training configuration: %s", asdict(config))
        logger.info("Training data directory: %s", data_dir)
        train = load_split(data_dir, "train")
        validation = load_split(data_dir, "validation")
        identity = make_training_identity(train, validation, config)
        logger.info(
            "Loaded train rows=%d, validation rows=%d; identity=%s",
            len(train),
            len(validation),
            identity["fingerprint"],
        )
        if local_files_only:
            tokenizer, model = load_model_and_tokenizer(
                cache_dir,
                local_files_only=True,
            )
        else:
            tokenizer, model = load_model_and_tokenizer(cache_dir)
        model.to(device)

        truncation = {
            "train": calculate_truncation_stats(
                train["text_norm"].tolist(), tokenizer, config.max_length
            ),
            "validation": calculate_truncation_stats(
                validation["text_norm"].tolist(), tokenizer, config.max_length
            ),
        }
        train_loader = create_dataloader(
            train, tokenizer, config, shuffle=True, device=device
        )
        validation_loader = create_dataloader(
            validation, tokenizer, config, shuffle=False, device=device
        )
        fit_result = fit_model(
            model,
            train_loader,
            validation_loader,
            train["label"],
            config,
            device,
            checkpoint_dir=checkpoint_dir,
            resume_from=checkpoint_dir / "latest.pt" if resume else None,
            checkpoint_identity=identity,
            logger=logger,
        )

        # Test remains outside fitting and is read only after validation selection.
        test = load_split(data_dir, "test")
        truncation["test"] = calculate_truncation_stats(
            test["text_norm"].tolist(), tokenizer, config.max_length
        )
        test_loader = create_dataloader(
            test, tokenizer, config, shuffle=False, device=device
        )
        test_metrics = evaluate_model(
            fit_result["model"],
            test_loader,
            device,
            fp16=config.fp16,
        )

        metrics = {
            "selection_metric": "validation macro_f1",
            "best_epoch": fit_result["best_epoch"],
            "best_validation_macro_f1": fit_result["best_validation_metrics"]["macro_f1"],
            "validation": fit_result["best_validation_metrics"],
            "test": test_metrics,
            "history": fit_result["history"],
            "truncation": truncation,
        }
        metadata = {
            "model_name": "WangchanBERTa 3-Class Sequence Classification",
            "model_id": MODEL_ID,
            "num_labels": len(LABELS),
            "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
            "input_column": "text_norm",
            "text_norm_source": (
                "datasets/processed/{split}.csv:text_norm; generated by "
                "scripts/prepare_dataset.py via "
                "src/thai_mod_api/text_processing.py::preprocess."
            ),
            "tokenizer": "AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)",
            "training_config": asdict(config),
            "best_epoch": fit_result["best_epoch"],
            "best_validation_macro_f1": fit_result["best_validation_metrics"]["macro_f1"],
            "validation_metrics": metrics["validation"],
            "test_metrics": metrics["test"],
            "effective_batch_size": calculate_effective_batch_size(
                config.batch_size, config.gradient_accumulation
            ),
            "class_weight_method": "mean-normalized square-root inverse frequency from train labels",
            "class_weights": fit_result["class_weights"],
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_label_counts": _label_counts(train),
            "validation_label_counts": _label_counts(validation),
            "test_label_counts": _label_counts(test),
            "total_optimizer_updates": fit_result["total_optimizer_updates"],
            "warmup_steps": fit_result["warmup_steps"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        save_artifacts(output_dir, fit_result["model"], tokenizer, metadata, metrics)
        logger.info("Training completed; final artifact saved to %s", output_dir)
        return {
            "artifact_dir": str(output_dir),
            "training_run_dir": str(run_dir),
            "metadata": metadata,
            **metrics,
        }
    except (Exception, KeyboardInterrupt):
        logger.exception("Training failed or was interrupted")
        raise
    finally:
        close_training_logger(logger)


def get_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured FP16 training run.")
    return torch.device("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a three-class WangchanBERTa candidate."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training logs and resumable checkpoints (defaults to a unique run directory).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --run-dir/checkpoints/latest.pt; configuration and datasets must match.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face Hub cache directory (defaults to HF_HUB_CACHE or HF_HOME/hub).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load all checkpoint files from cache only; do not contact the Hub.",
    )
    args = parser.parse_args()
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")

    result = run_training(
        args.data_dir,
        args.output_dir,
        None,
        DEFAULT_CONFIG,
        args.cache_dir,
        args.local_files_only,
        args.run_dir,
        args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
