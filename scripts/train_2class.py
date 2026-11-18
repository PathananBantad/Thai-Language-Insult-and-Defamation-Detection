"""Train the proposal's two-class task: insult (0) vs defamation (1).

Reuses the three-class training scripts unchanged and only swaps their label
constants, so model, hyper-parameters, selection on validation and the single
test evaluation stay identical.

Usage (after `python scripts/prepare_dataset.py --two-class --out-dir datasets/processed_2class`):
    python scripts/train_2class.py nb
    python scripts/train_2class.py bert [--run-dir models/training_runs/wcb_2class]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR = ROOT / "datasets" / "processed_2class"


def patch(module, names: dict[int, str]) -> None:
    module.LABEL_NAMES = names
    module.LABELS = tuple(names)
    if hasattr(module, "LABEL2ID"):
        module.LABEL2ID = {name: label for label, name in names.items()}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:]
    if which == "nb":
        import train_complement_nb_3class as nb

        patch(nb, {0: "insult", 1: "defamation"})
        out = ROOT / "models" / "candidates" / "complement_nb_2class"
        sys.argv = [sys.argv[0], "--data-dir", str(DATA_DIR), "--output-dir", str(out), *rest]
        nb.main()
    elif which == "bert":
        import train_wangchanberta_3class as bert

        patch(bert, {0: "Insult", 1: "Defamation"})
        out = ROOT / "models" / "candidates" / "wangchanberta_2class"
        sys.argv = [sys.argv[0], "--data-dir", str(DATA_DIR), "--output-dir", str(out), *rest]
        bert.main()
    else:
        raise SystemExit("usage: train_2class.py nb|bert [extra args]")


if __name__ == "__main__":
    main()
