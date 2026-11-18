from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.thai_mod_api.text_processing import preprocess

MODEL_DIR = ROOT / "models" / "candidates" / "wangchanberta_3class"
MAX_LENGTH = 128
LABELS = ("Normal", "Insult", "Defamation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Thai text with the local WangchanBERTa 3-class model."
    )
    parser.add_argument("text", nargs="+", help="Text to classify.")
    args = parser.parse_args()

    processed_text = preprocess(" ".join(args.text))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=False,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model.to(torch.device("cpu"))
    model.eval()

    inputs = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    print(f"Prediction: {LABELS[predicted_id]}")
    for label, probability in zip(LABELS, probabilities.tolist()):
        print(f"{label}: {probability:.4%}")


if __name__ == "__main__":
    main()
