# modules/nlu.py
# NLU helpers: simple heuristic + optional HF pipeline loader
import os
import logging
import re
from transformers import pipeline

logger = logging.getLogger("nlu")
_pipeline = None


def load_nlu_pipeline(model_dir=None):
    global _pipeline
    # If you have a saved trained NLU model folder, provide model_dir in env
    model_dir = model_dir or os.getenv("NLU_MODEL_DIR")
    # If not provided via env or arg, try the local workspace folder `models/nlu`
    if not model_dir:
        default_dir = os.path.join(os.getcwd(), "models", "nlu")
        if os.path.exists(default_dir):
            model_dir = default_dir

    if model_dir and os.path.exists(model_dir):
        try:
            logger.info(f"Loading NLU model from {model_dir}")
            device = 0 if os.getenv("CUDA_AVAILABLE") else -1
            _pipeline = pipeline(
                "text-classification",
                model=model_dir,
                tokenizer=model_dir,
                return_all_scores=True,
                device=device,
            )
            return _pipeline
        except Exception as e:
            logger.exception(f"Failed loading NLU pipeline from {model_dir}: {e}")
            _pipeline = None

    logger.info("No NLU model dir set or not found; using heuristics.")
    return None


# Use your earlier mapping set
label2id = {"complaint": 0, "escalation": 1, "order": 2, "other": 3, "refund": 4}
id2label = {v: k for k, v in label2id.items()}


def get_predicted_intent(nlu_result):
    if not nlu_result:
        return "other"
    preds = nlu_result
    if isinstance(nlu_result, list) and isinstance(nlu_result[0], list):
        preds = nlu_result[0]
    best = max(preds, key=lambda x: x.get("score", 0.0))
    label = best.get("label", "")
    if isinstance(label, str) and label.upper().startswith("LABEL_"):
        try:
            idx = int(label.split("_", 1)[1])
            return id2label.get(idx, "other")
        except Exception:
            return "other"
    if label in label2id:
        return label
    return "other"


# ---------------------------
# Heuristic NLU helpers
# ---------------------------
_arabic_diacritics_re = re.compile(
    r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]+'
)


def normalize_arabic(text: str) -> str:
    """
    Basic Arabic normalization:
    - remove diacritics
    - normalize Alef variants to ا
    - normalize taa marbuta to ه/ة handling (keep as is)
    - convert ى to ي
    - remove tatweel
    - lowercase
    """
    if not text:
        return ""
    t = text.strip().lower()
    # remove diacritics
    t = _arabic_diacritics_re.sub("", t)
    # normalize alef variants
    t = re.sub(r"[إأآا]", "ا", t)
    # normalize hamza forms
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    # normalize alef maqsura to ya
    t = t.replace("ى", "ي")
    # remove tatweel
    t = t.replace("ـ", "")
    # normalize multiple spaces
    t = re.sub(r"\s+", " ", t)
    return t


def heuristic_intent_from_text(text: str) -> str:
    """
    Improved heuristic matcher for Arabic colloquial forms.
    Returns one of: "order", "refund", "complaint", "escalation", "other"
    """
    if not text:
        return "other"
    t = normalize_arabic(text)

    # Common order expressions (Arabic colloquial + English translit)
    order_terms = [
        "اوردر",
        "اوردِر",
        "اردر",
        "اردر",  # different spellings
        "أردر",
        "اطلب",
        "اطلبوا",
        "عايز اطلب",
        "عايزة اطلب",
        "عايزه اطلب",
        "عايز",
        "عايزة",
        "عايزه",
        "اعمل",
        "أعمل",
        "عايزة اعمل",
        "عايزة أعمل",
        "عمل",
        "order",
        "اوردر",  # transliteration
    ]
    # check order
    for w in order_terms:
        if w in t:
            return "order"

    # refund expressions
    if any(w in t for w in ["استرجاع", "فلوس", "ارجاع", "refund", "استرجاع فلوس"]):
        return "refund"

    # complaint expressions
    if any(
        w in t
        for w in [
            "شكوى",
            "مشكلة",
            "عيب",
            "مش عاجبني",
            "مش راضي",
            "محصلش",
            "مش تمام",
            "مش كويس",
            "مش مظبوط",
        ]
    ):
        return "complaint"

    # escalation expressions
    if any(w in t for w in ["مدير", "تصعيد", "مسؤول", "حولني", "حولني للمدير", "حوّلني"]):
        return "escalation"

    return "other"