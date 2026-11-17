from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import train_wangchanberta_3class as trainer


def make_frame(labels=(0, 1, 2), prefix="text") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_raw": [f"raw-{prefix}-{index}" for index in range(len(labels))],
            "text_norm": [f"{prefix}-{index}" for index in range(len(labels))],
            "label": list(labels),
            "label_name": [trainer.LABEL_NAMES[label] for label in labels],
            "source": ["fixture"] * len(labels),
        }
    )


def test_label_mapping_and_valid_input():
    assert trainer.LABEL_NAMES == {0: "Normal", 1: "Insult", 2: "Defamation"}
    assert trainer.LABEL2ID == {"Normal": 0, "Insult": 1, "Defamation": 2}
    validated = trainer.validate_split(make_frame(), "train")
    assert validated["label"].tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("text_norm", None),
        ("text_norm", "  "),
        ("text_raw", None),
        ("label", None),
        ("label_name", None),
        ("source", "  "),
    ],
)
def test_validation_rejects_missing_or_empty_input(column, value):
    invalid = make_frame()
    invalid.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        trainer.validate_split(invalid, "train")


@pytest.mark.parametrize("value", [3, 1.5, "invalid"])
def test_validation_rejects_invalid_labels(value):
    invalid = make_frame()
    invalid["label"] = invalid["label"].astype(object)
    invalid.loc[0, "label"] = value
    with pytest.raises(ValueError, match="label"):
        trainer.validate_split(invalid, "train")


def test_validation_rejects_label_name_mismatch():
    invalid = make_frame()
    invalid.loc[0, "label_name"] = "insult"
    with pytest.raises(ValueError, match="label_name"):
        trainer.validate_split(invalid, "train")


def test_load_split_reads_requested_dataset_only(tmp_path):
    frame = make_frame(prefix="loaded")
    frame.to_csv(tmp_path / "train.csv", index=False)

    loaded = trainer.load_split(tmp_path, "train")

    assert loaded["text_norm"].tolist() == frame["text_norm"].tolist()
    assert loaded["label"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="Unknown split"):
        trainer.load_split(tmp_path, "validation-malicious")


def test_class_weights_are_calculated_from_supplied_train_labels():
    weights = trainer.compute_class_weights([0, 0, 0, 1, 2])
    expected = np.sqrt(5 / (3 * np.array([3, 1, 1], dtype=float)))
    expected /= expected.mean()
    np.testing.assert_allclose(weights.numpy(), expected.astype(np.float32))


def test_class_weights_reject_missing_train_class():
    with pytest.raises(ValueError, match="every class"):
        trainer.compute_class_weights([0, 0, 1])


@pytest.mark.parametrize("labels", [[0, 1.5, 2], [0, 1, 3], [0, "bad", 2]])
def test_class_weights_reject_invalid_train_labels(labels):
    with pytest.raises(ValueError, match="Train labels"):
        trainer.compute_class_weights(labels)


def test_metrics_include_all_three_classes_when_not_predicted():
    metrics = trainer.calculate_metrics([0, 1, 2], [0, 0, 0])

    assert metrics["labels"] == [0, 1, 2]
    assert metrics["per_class"]["1"]["support"] == 1
    assert metrics["per_class"]["2"]["support"] == 1
    assert metrics["per_class"]["2"]["f1"] == 0.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]


class FakeModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def parameters(self):
        return [self.weight]

    def state_dict(self):
        return {"weight": self.weight.detach().clone()}

    def load_state_dict(self, state):
        self.weight.data.copy_(state["weight"])

    def to(self, device):
        self.weight.data = self.weight.data.to(device)
        return self

    def train(self):
        return self

    def eval(self):
        return self


class FakeLoader:
    def __len__(self):
        return 2


def test_fit_model_selects_checkpoint_by_validation_macro_f1(monkeypatch):
    model = FakeModel()
    validation_scores = iter([0.2, 0.8, 0.5])
    epoch_number = 0

    def fake_train_epoch(model, *args, **kwargs):
        nonlocal epoch_number
        epoch_number += 1
        model.weight.data.fill_(epoch_number)
        return float(epoch_number)

    def fake_evaluate(model, *args, **kwargs):
        score = next(validation_scores)
        return {
            "macro_f1": score,
            "accuracy": score,
            "labels": [0, 1, 2],
        }

    monkeypatch.setattr(trainer, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate)
    config = trainer.TrainingConfig(
        batch_size=1,
        gradient_accumulation=1,
        epochs=3,
        fp16=False,
    )

    result = trainer.fit_model(
        model,
        FakeLoader(),
        FakeLoader(),
        [0, 1, 2],
        config,
        torch.device("cpu"),
    )

    assert result["best_epoch"] == 2
    assert result["best_validation_metrics"]["macro_f1"] == 0.8
    assert model.weight.item() == 2.0
    assert [entry["validation"]["macro_f1"] for entry in result["history"]] == [
        0.2,
        0.8,
        0.5,
    ]


def test_fit_model_keeps_first_epoch_when_validation_scores_tie(monkeypatch):
    model = FakeModel()
    validation_scores = iter([0.8, 0.8, 0.5])
    epoch_number = 0

    def fake_train_epoch(model, *args, **kwargs):
        nonlocal epoch_number
        epoch_number += 1
        model.weight.data.fill_(epoch_number)
        return float(epoch_number)

    monkeypatch.setattr(trainer, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(
        trainer,
        "evaluate_model",
        lambda *args, **kwargs: {"macro_f1": next(validation_scores)},
    )
    config = trainer.TrainingConfig(
        batch_size=1,
        gradient_accumulation=1,
        epochs=3,
        fp16=False,
    )

    result = trainer.fit_model(
        model,
        FakeLoader(),
        FakeLoader(),
        [0, 1, 2],
        config,
        torch.device("cpu"),
    )

    assert result["best_epoch"] == 1
    assert model.weight.item() == 1.0


def test_run_training_loads_test_only_after_validation_selection(monkeypatch, tmp_path):
    events = []
    train = make_frame(prefix="train")
    validation = make_frame(prefix="validation")
    test = make_frame(prefix="test")
    model = FakeModel()
    tokenizer = object()
    saved = {}

    def fake_load_split(data_dir, split_name):
        events.append(("load", split_name))
        return {"train": train, "validation": validation, "test": test}[split_name]

    def fake_fit(
        model_arg, train_loader, validation_loader, train_labels, *args, **kwargs
    ):
        events.append(("fit", list(train_labels)))
        return {
            "model": model_arg,
            "history": [{"epoch": 1, "validation": {"macro_f1": 0.5}}],
            "best_epoch": 1,
            "best_validation_metrics": {"macro_f1": 0.5},
            "class_weights": [1.0, 1.0, 1.0],
            "total_optimizer_updates": 1,
            "warmup_steps": 1,
        }

    def fake_evaluate(model_arg, loader, device, **kwargs):
        events.append(("evaluate_test", loader))
        return {"macro_f1": 0.4, "labels": [0, 1, 2]}

    monkeypatch.setattr(trainer, "load_split", fake_load_split)
    monkeypatch.setattr(
        trainer,
        "load_model_and_tokenizer",
        lambda cache_dir=None: (tokenizer, model),
    )
    monkeypatch.setattr(
        trainer,
        "calculate_truncation_stats",
        lambda texts, tokenizer_arg, max_length: {
            "max_length": max_length,
            "total_texts": len(texts),
            "truncated_count": 0,
            "truncated_fraction": 0.0,
        },
    )
    monkeypatch.setattr(
        trainer,
        "create_dataloader",
        lambda frame, *args, **kwargs: frame,
    )
    monkeypatch.setattr(trainer, "fit_model", fake_fit)
    monkeypatch.setattr(trainer, "evaluate_model", fake_evaluate)
    def fake_save_artifacts(output_dir, model_arg, tokenizer_arg, metadata, metrics):
        events.append(("save", output_dir))
        saved["metadata"] = metadata
        saved["metrics"] = metrics

    monkeypatch.setattr(trainer, "save_artifacts", fake_save_artifacts)

    result = trainer.run_training(
        tmp_path,
        tmp_path / "candidate",
        torch.device("cpu"),
        trainer.TrainingConfig(fp16=False),
        run_dir=tmp_path / "run",
    )

    event_names = [event[0] for event in events]
    assert event_names.index("fit") < event_names.index("load", 2)
    assert events[event_names.index("fit")][1] == train["label"].tolist()
    assert event_names.count("load") == 3
    assert events[event_names.index("load", 2)] == ("load", "test")
    assert result["test"]["macro_f1"] == 0.4
    assert event_names.index("evaluate_test") > event_names.index("load", 2)
    assert saved["metadata"]["best_epoch"] == saved["metrics"]["best_epoch"] == 1
    assert (
        saved["metadata"]["best_validation_macro_f1"]
        == saved["metrics"]["best_validation_macro_f1"]
        == 0.5
    )
    assert saved["metadata"]["validation_metrics"] == saved["metrics"]["validation"]
    assert saved["metadata"]["test_metrics"] == saved["metrics"]["test"]
    assert saved["metadata"]["training_config"]["epochs"] == 3
    assert saved["metadata"]["class_weights"] == [1.0, 1.0, 1.0]
    assert result["training_run_dir"] == str(tmp_path / "run")


def test_effective_batch_and_scheduler_steps():
    assert trainer.calculate_effective_batch_size(2, 8) == 16
    assert trainer.calculate_optimizer_updates(10, 8, 3) == 6
    assert trainer.calculate_warmup_steps(6, 0.06) == 1


class CountingOptimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls += 1
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.step_calls += 1


class CountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class FakeGradScaler:
    def __init__(self, overflow=False):
        self.overflow = overflow
        self.scale_value = 8.0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.scale_value

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        if self.overflow:
            self.scale_value /= 2


def test_grad_scaler_overflow_skips_scheduler_step():
    optimizer = CountingOptimizer([])
    scheduler = CountingScheduler()

    did_update = trainer.step_optimizer_with_scaler(
        optimizer,
        scheduler,
        FakeGradScaler(overflow=True),
    )

    assert not did_update
    assert optimizer.step_calls == 0
    assert scheduler.step_calls == 0


def test_optimizer_and_scheduler_step_counts_match_accumulation_groups():
    optimizer = CountingOptimizer([])
    scheduler = CountingScheduler()

    did_update = trainer.step_optimizer_with_scaler(
        optimizer,
        scheduler,
        FakeGradScaler(),
    )

    assert did_update
    assert optimizer.step_calls == scheduler.step_calls == 1


class TrainingProbeModel(FakeModel):
    def __call__(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]
        positive = self.weight.expand(batch_size)
        logits = torch.stack((positive, -positive, positive * 0), dim=1)
        return SimpleNamespace(logits=logits)


class DisabledFakeGradScaler:
    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def test_train_epoch_handles_partial_batch_and_accumulation_remainder():
    model = TrainingProbeModel()
    optimizer = CountingOptimizer(model.parameters())
    scheduler = CountingScheduler()
    labels = ([0, 1], [1, 2], [0, 2], [2])
    batches = [
        {
            "input_ids": torch.zeros(len(batch_labels), 1, dtype=torch.long),
            "attention_mask": torch.ones(len(batch_labels), 1, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
        for batch_labels in labels
    ]
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, reduction="sum")

    loss = trainer.train_epoch(
        model,
        batches,
        optimizer,
        scheduler,
        DisabledFakeGradScaler(),
        loss_fn,
        torch.device("cpu"),
        gradient_accumulation=3,
        fp16=False,
    )

    all_labels = torch.tensor([label for group in labels for label in group])
    all_logits = model(
        input_ids=torch.zeros(len(all_labels), 1, dtype=torch.long),
        attention_mask=torch.ones(len(all_labels), 1, dtype=torch.long),
    ).logits
    expected_loss = torch.nn.functional.cross_entropy(
        all_logits,
        all_labels,
        weight=weights,
    ).item()
    assert loss == pytest.approx(expected_loss)
    assert trainer.calculate_optimizer_updates(4, 3, 1) == 2
    assert optimizer.step_calls == scheduler.step_calls == 2


class FakeLengthTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        lengths = {"short": 4, "at-limit": 8, "long": 9}
        return {
            "input_ids": [[1] * lengths[text] for text in texts],
        }


class FakePaddingTokenizer:
    def __call__(self, text, **kwargs):
        length = len(text)
        return {
            "input_ids": list(range(length)),
            "attention_mask": [1] * length,
        }

    def pad(self, features, **kwargs):
        assert kwargs["padding"] is True
        assert kwargs["pad_to_multiple_of"] == 8
        target_length = 8
        return {
            key: torch.tensor(
                [
                    feature[key] + [0] * (target_length - len(feature[key]))
                    for feature in features
                ],
                dtype=torch.long,
            )
            for key in ("input_ids", "attention_mask")
        }


def test_collator_uses_dynamic_padding_and_keeps_labels():
    dataset = trainer.TextClassificationDataset(
        ["a", "abcde"],
        [0, 1],
        FakePaddingTokenizer(),
        max_length=128,
    )
    batch = trainer.make_collate_fn(FakePaddingTokenizer())(
        [dataset[0], dataset[1]]
    )

    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["labels"].tolist() == [0, 1]


def test_truncation_stats_counts_texts_over_maximum():
    stats = trainer.calculate_truncation_stats(
        ["short", "at-limit", "long"],
        FakeLengthTokenizer(),
        max_length=8,
    )

    assert stats == {
        "max_length": 8,
        "total_texts": 3,
        "truncated_count": 1,
        "truncated_fraction": pytest.approx(1 / 3),
    }


def test_save_artifacts_refuses_existing_output_directory(tmp_path):
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        trainer.save_artifacts(output_dir, object(), object(), {}, {})

    assert sentinel.read_text(encoding="utf-8") == "existing"


def test_cli_help_runs_without_loading_checkpoint_or_training():
    script = trainer.ROOT / "scripts" / "train_wangchanberta_3class.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=trainer.ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--data-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--cache-dir" in result.stdout


def test_cache_dir_resolution_prefers_cli_then_hub_then_home(monkeypatch):
    monkeypatch.setenv("HF_HOME", r"D:\huggingface-cache")
    monkeypatch.setenv("HF_HUB_CACHE", r"D:\huggingface-cache\hub")

    assert trainer.resolve_cache_dir() == Path(r"D:\huggingface-cache\hub")
    assert trainer.resolve_cache_dir(Path(r"E:\explicit-cache")) == Path(
        r"E:\explicit-cache"
    )

    monkeypatch.delenv("HF_HUB_CACHE")
    assert trainer.resolve_cache_dir() == Path(r"D:\huggingface-cache\hub")


def test_model_loader_passes_cache_dir_and_slow_tokenizer(monkeypatch):
    calls = {}

    class FakeTokenizer:
        pad_token_id = 1

    def fake_tokenizer_from_pretrained(model_id, **kwargs):
        calls["tokenizer"] = (model_id, kwargs)
        return FakeTokenizer()

    def fake_model_from_pretrained(model_id, **kwargs):
        calls["model"] = (model_id, kwargs)
        return object()

    monkeypatch.setattr(
        trainer.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        trainer.AutoModelForSequenceClassification,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    cache_dir = Path(r"D:\huggingface-cache\hub")

    trainer.load_model_and_tokenizer(cache_dir)

    assert calls["tokenizer"][0] == trainer.MODEL_ID
    assert calls["tokenizer"][1]["use_fast"] is False
    assert calls["tokenizer"][1]["cache_dir"] == cache_dir
    assert calls["model"][1]["cache_dir"] == cache_dir


def test_training_logger_persists_exception_traceback(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logger = trainer.configure_training_logger(run_dir)

    try:
        raise RuntimeError("training failure for logger test")
    except RuntimeError:
        logger.exception("Training failed")
    finally:
        trainer.close_training_logger(logger)

    log = (run_dir / "training.log").read_text(encoding="utf-8")
    assert "training failure for logger test" in log
    assert "Traceback (most recent call last)" in log


def test_training_identity_changes_with_train_validation_or_config():
    train = make_frame(prefix="train")
    validation = make_frame(prefix="validation")
    config = trainer.TrainingConfig(fp16=False)
    original = trainer.make_training_identity(train, validation, config)

    changed_train = train.copy()
    changed_train.loc[0, "text_norm"] = "changed"
    changed_validation = validation.copy()
    changed_validation.loc[0, "text_norm"] = "changed"
    changed_config = trainer.TrainingConfig(fp16=False, learning_rate=1e-5)

    assert (
        trainer.make_training_identity(changed_train, validation, config)["fingerprint"]
        != original["fingerprint"]
    )
    assert (
        trainer.make_training_identity(train, changed_validation, config)["fingerprint"]
        != original["fingerprint"]
    )
    assert (
        trainer.make_training_identity(train, validation, changed_config)["fingerprint"]
        != original["fingerprint"]
    )


def test_run_training_logs_and_reraises_training_exception(monkeypatch, tmp_path):
    def fail_load_split(data_dir, split_name):
        raise ValueError(f"failure loading {split_name}")

    monkeypatch.setattr(trainer, "load_split", fail_load_split)
    run_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="failure loading train"):
        trainer.run_training(
            tmp_path,
            tmp_path / "candidate",
            torch.device("cpu"),
            trainer.TrainingConfig(fp16=False),
            run_dir=run_dir,
        )

    log = (run_dir / "training.log").read_text(encoding="utf-8")
    assert "failure loading train" in log
    assert "Traceback (most recent call last)" in log


def make_training_checkpoint(identity):
    return {
        "format_version": trainer.CHECKPOINT_FORMAT_VERSION,
        "kind": "training_state",
        "identity": identity,
        "epoch": 1,
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "best_epoch": 1,
        "best_validation_score": 0.5,
        "best_validation_metrics": {"macro_f1": 0.5},
        "best_model_state": {"weight": torch.tensor([1.0])},
        "history": [{"epoch": 1}],
        "train_loader_generator_state": None,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": None,
    }


def test_checkpoint_atomic_save_and_identity_validation(tmp_path):
    identity = {"fingerprint": "fixture"}
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    trainer.atomic_torch_save(make_training_checkpoint(identity), checkpoint_path)

    loaded = trainer.load_training_checkpoint(checkpoint_path, identity)

    assert loaded["epoch"] == 1
    assert loaded["best_validation_score"] == 0.5
    with pytest.raises(ValueError, match="does not match"):
        trainer.load_training_checkpoint(
            checkpoint_path,
            {"fingerprint": "different"},
        )


def test_atomic_checkpoint_save_preserves_previous_file_on_write_failure(
    monkeypatch, tmp_path
):
    checkpoint_path = tmp_path / "latest.pt"
    original = make_training_checkpoint({"fingerprint": "original"})
    trainer.atomic_torch_save(original, checkpoint_path)

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(trainer.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated write failure"):
        trainer.atomic_torch_save(
            make_training_checkpoint({"fingerprint": "replacement"}),
            checkpoint_path,
        )

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert loaded["identity"] == {"fingerprint": "original"}
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_corrupt_resume_checkpoint_fails_without_fallback(tmp_path):
    checkpoint_path = tmp_path / "corrupt.pt"
    checkpoint_path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(RuntimeError, match="Unable to load training checkpoint"):
        trainer.load_training_checkpoint(
            checkpoint_path,
            {"fingerprint": "fixture"},
        )


def test_fit_model_resumes_from_last_epoch_and_keeps_best_validation_checkpoint(
    monkeypatch, tmp_path
):
    model = FakeModel()
    checkpoint_dir = tmp_path / "checkpoints"
    identity = {"fingerprint": "resume-fixture"}
    config = trainer.TrainingConfig(
        batch_size=1,
        gradient_accumulation=1,
        epochs=2,
        fp16=False,
    )
    train_calls = 0

    def interrupted_train(model_arg, *args, **kwargs):
        nonlocal train_calls
        train_calls += 1
        if train_calls == 2:
            raise RuntimeError("simulated interruption")
        model_arg.weight.grad = torch.ones_like(model_arg.weight)
        args[1].step()
        args[2].step()
        args[1].zero_grad(set_to_none=True)
        model_arg.weight.data.add_(1.0)
        return 1.0

    monkeypatch.setattr(trainer, "train_epoch", interrupted_train)
    monkeypatch.setattr(
        trainer,
        "evaluate_model",
        lambda *args, **kwargs: {"macro_f1": 0.4},
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        trainer.fit_model(
            model,
            FakeLoader(),
            FakeLoader(),
            [0, 1, 2],
            config,
            torch.device("cpu"),
            checkpoint_dir=checkpoint_dir,
            checkpoint_identity=identity,
        )

    saved = trainer.load_training_checkpoint(
        checkpoint_dir / "latest.pt",
        identity,
    )
    assert saved["epoch"] == 1
    assert (checkpoint_dir / "best.pt").is_file()
    assert saved["optimizer_state"]
    assert "last_epoch" in saved["scheduler_state"]
    assert isinstance(saved["scaler_state"], dict)

    def resumed_train(model_arg, *args, **kwargs):
        model_arg.weight.grad = torch.ones_like(model_arg.weight)
        args[1].step()
        args[2].step()
        args[1].zero_grad(set_to_none=True)
        model_arg.weight.data.add_(1.0)
        return 2.0

    monkeypatch.setattr(trainer, "train_epoch", resumed_train)
    monkeypatch.setattr(
        trainer,
        "evaluate_model",
        lambda *args, **kwargs: {"macro_f1": 0.8},
    )
    result = trainer.fit_model(
        model,
        FakeLoader(),
        FakeLoader(),
        [0, 1, 2],
        config,
        torch.device("cpu"),
        checkpoint_dir=checkpoint_dir,
        resume_from=checkpoint_dir / "latest.pt",
        checkpoint_identity=identity,
    )

    assert result["best_epoch"] == 2
    assert result["best_validation_metrics"]["macro_f1"] == 0.8
    assert result["model"].weight.item() == pytest.approx(2.0, abs=1e-4)
    assert [entry["epoch"] for entry in result["history"]] == [1, 2]
