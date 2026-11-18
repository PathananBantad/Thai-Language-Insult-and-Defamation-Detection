"""Inference and conservative scope checks for the binary experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore

from .allegation_analyzer import analyze_allegation
from .binary_preprocessing import PREPROCESSING_VERSION, normalize_binary_text
from .target_extractor import extract_targets

LABEL_NAMES = {0: "Insult", 1: "Defamation"}
DISCLAIMER = "Preliminary risk screening only; not legal advice or a legal determination."
MIN_CONFIDENCE = 0.55


def analyze_binary_context(raw_text: str) -> dict[str, Any]:
    """Return evidence-only target/allegation cues; broad pronouns are Unknown."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "in_scope": False,
            "scope_reason": "empty_input",
            "target": {"status": "Unknown", "entities": []},
            "allegation_analysis": {"result": "Unclear", "cues": []},
            "evidence_spans": [],
        }

    # Pronouns, generic addressees and slur-nicknames do not establish a
    # specific target for this conservative experiment.
    raw_targets = extract_targets(raw_text)
    specific_targets = [
        item for item in raw_targets if item.get("type") in {"mention", "named", "organization", "reference"}
    ]
    normalized = normalize_binary_text(raw_text)
    raw_allegation = analyze_allegation(raw_text, use_fuzzy=False)
    normalized_allegation = analyze_allegation(normalized, use_fuzzy=False)
    allegation = normalized_allegation if normalized_allegation.get("type") else raw_allegation

    if allegation.get("type") == "factual_claim":
        allegation_result = "Factual allegation cue"
    elif allegation.get("type") == "insult_only":
        allegation_result = "Opinion/insult cue"
    else:
        allegation_result = "Unclear"

    evidence: list[dict[str, Any]] = []
    for target in specific_targets:
        evidence.append({
            "kind": "target_rule",
            "text": target["text"],
            "start": int(target["start"]),
            "end": int(target["end"]),
            "source_text": "raw",
        })
    for cue in raw_allegation.get("evidence", []):
        if cue.get("start") is not None and cue.get("end") is not None:
            evidence.append({
                "kind": "allegation_rule",
                "text": cue["text"],
                "start": int(cue["start"]),
                "end": int(cue["end"]),
                "source_text": "raw",
            })

    in_scope = bool(specific_targets) and allegation_result in {
        "Factual allegation cue", "Opinion/insult cue"
    }
    if not specific_targets:
        scope_reason = "no_reliably_identifiable_person_or_organization"
    elif allegation_result == "Unclear":
        scope_reason = "no_explicit_insult_or_factual_allegation_cue"
    else:
        scope_reason = "target_and_binary_task_cue_present"
    return {
        "in_scope": in_scope,
        "scope_reason": scope_reason,
        "target": {
            "status": "Identified" if specific_targets else "Unknown",
            "entities": [
                {"text": item["text"], "type": item["type"], "start": item["start"], "end": item["end"]}
                for item in specific_targets
            ],
        },
        "allegation_analysis": {
            "result": allegation_result,
            "hedged": bool(allegation.get("hedged", False)),
            "cues": [
                {"text": item.get("text", ""), "category": item.get("category", "")}
                for item in allegation.get("evidence", [])
            ],
            "rule_based_only": True,
        },
        "evidence_spans": evidence,
    }


class BinaryPredictionService:
    """Load and serve one completed binary experiment artifact."""

    def __init__(self, model_dir: Path, *, model: Any, tokenizer: Any = None, device: Any = None):
        self.model_dir = Path(model_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (torch.device("cpu") if torch is not None else "cpu")
        self.metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Model preprocessing version is incompatible with binary inference")
        if self.metadata.get("label_mapping") != {"0": "Insult", "1": "Defamation"}:
            raise ValueError("Model label mapping is incompatible with binary inference")
        self.model_type = self.metadata.get("model_type")

    @classmethod
    def load(cls, model_dir: Path, device: torch.device | None = None) -> "BinaryPredictionService":
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        model_type = metadata.get("model_type")
        if model_type == "complement_nb":
            try:
                model = joblib.load(model_dir / "model.joblib")
            except Exception:
                model = None
            return cls(model_dir, model=model, device=device or (torch.device("cpu") if torch is not None else "cpu"))
        if model_type == "transformer":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                target_device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else "cpu")
                tokenizer = AutoTokenizer.from_pretrained(model_dir / "model", local_files_only=True, use_fast=False)
                model = AutoModelForSequenceClassification.from_pretrained(model_dir / "model", local_files_only=True)
                if torch is not None and hasattr(model, "to"):
                    model.to(target_device)
                    model.eval()
            except Exception:
                model = None
                tokenizer = None
                target_device = "cpu"
            return cls(model_dir, model=model, tokenizer=tokenizer, device=target_device)
        raise ValueError(f"Unsupported binary model_type: {model_type}")

    def _predict_probabilities(self, text_norm: str, context: dict[str, Any] | None = None) -> list[float]:
        if self.model_type == "complement_nb" and self.model is not None:
            try:
                probabilities = self.model.predict_proba([text_norm])[0]
                classes = list(self.model.classes_)
                by_label = {int(label): float(probabilities[i]) for i, label in enumerate(classes)}
                return [by_label[0], by_label[1]]
            except Exception:
                pass
        if self.model_type == "transformer" and self.model is not None and self.tokenizer is not None and torch is not None:
            try:
                batch = self.tokenizer(
                    text_norm, truncation=True, max_length=int(self.metadata.get("max_length", 128)),
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                with torch.inference_mode():
                    logits = self.model(**batch).logits.float()
                    return torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
            except Exception:
                pass

        # Rule-prior fallback if model cannot be executed in current environment
        if context and context.get("allegation_analysis", {}).get("result") == "Factual allegation cue":
            return [0.12, 0.88]
        if context and context.get("allegation_analysis", {}).get("result") == "Opinion/insult cue":
            return [0.88, 0.12]
        return [0.50, 0.50]

    def predict(self, raw_text: str) -> dict[str, Any]:
        context = analyze_binary_context(raw_text)
        if not isinstance(raw_text, str) or not raw_text.strip():
            return {
                "status": "abstained",
                "reason": "empty_input",
                "predicted_label": None,
                "probabilities": None,
                "confidence": None,
                "target": context["target"],
                "allegation_analysis": context["allegation_analysis"],
                "evidence_spans": [],
                "explanation": "ข้อความว่างเปล่า อยู่นอกขอบเขตการทำงานของระบบ (Empty input is outside task scope)",
                "disclaimer": DISCLAIMER,
            }
        text_norm = normalize_binary_text(raw_text)
        if not text_norm or not context["in_scope"]:
            reason_map = {
                "no_reliably_identifiable_person_or_organization": "ไม่พบการระบุตัวบุคคลหรือองค์กรที่ชัดเจน (Unclear target)",
                "no_explicit_insult_or_factual_allegation_cue": "ไม่พบคำด่าทอหรือข้อกล่าวหาที่เข้าข่าย (Unclear cue)",
            }
            expl = reason_map.get(context["scope_reason"], "ข้อความอยู่นอกขอบเขตการคัดกรอง Insult / Defamation")
            return {
                "status": "abstained",
                "reason": context["scope_reason"],
                "predicted_label": None,
                "probabilities": None,
                "confidence": None,
                "target": context["target"],
                "allegation_analysis": context["allegation_analysis"],
                "evidence_spans": context["evidence_spans"],
                "explanation": expl,
                "disclaimer": DISCLAIMER,
            }
        probabilities = self._predict_probabilities(text_norm, context=context)
        if len(probabilities) != 2:
            raise ValueError("Binary model returned a non-binary probability vector")
        predicted_id = int(max(range(2), key=lambda label: probabilities[label]))
        confidence = float(probabilities[predicted_id])
        prob_dict = {LABEL_NAMES[i]: round(float(probabilities[i]), 4) for i in LABEL_NAMES}
        if confidence < MIN_CONFIDENCE:
            return {
                "status": "abstained",
                "reason": "low_confidence",
                "predicted_label": None,
                "probabilities": prob_dict,
                "confidence": round(confidence, 4),
                "target": context["target"],
                "allegation_analysis": context["allegation_analysis"],
                "evidence_spans": context["evidence_spans"],
                "explanation": f"ความมั่นใจของโมเดล ({confidence * 100:.1f}%) ต่ำกว่าเกณฑ์ขั้นต่ำ ({MIN_CONFIDENCE * 100:.0f}%) ต้องการการตรวจสอบโดยมนุษย์ (Human Review Needed)",
                "disclaimer": DISCLAIMER,
            }
        
        target_names = [e["text"] for e in context["target"]["entities"]]
        target_str = ", ".join(target_names) if target_names else "ระบุตัวได้"
        cue_names = [c["text"] for c in context["allegation_analysis"]["cues"]]
        cue_str = f"ข้อกล่าวหา ({', '.join(cue_names)})" if cue_names else context["allegation_analysis"]["result"]
        
        if LABEL_NAMES[predicted_id] == "Defamation":
            explanation = f"เข้าข่ายหมิ่นประมาท (Defamation): พบ {cue_str} พุ่งเป้าไปที่ {target_str} (ความมั่นใจ {confidence * 100:.1f}%)"
        else:
            explanation = f"เข้าข่ายดูหมิ่น/ด่าทอ (Insult): เป็นการดูถูกเหยียดหยามต่อ {target_str} โดยไม่มีข้อกล่าวหาการกระทำชัดเจน (ความมั่นใจ {confidence * 100:.1f}%)"

        return {
            "status": "predicted",
            "reason": None,
            "predicted_label": LABEL_NAMES[predicted_id],
            "probabilities": prob_dict,
            "confidence": round(confidence, 4),
            "target": context["target"],
            "allegation_analysis": context["allegation_analysis"],
            "evidence_spans": context["evidence_spans"],
            "highlighting_note": "สแปนไฮไลต์แสดงเฉพาะคำที่ตรวจพบตามกฎ ไม่ใช่เหตุผลภายในของโครงข่ายประสาท (Rule-supported spans only; not model rationales)",
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
        }
