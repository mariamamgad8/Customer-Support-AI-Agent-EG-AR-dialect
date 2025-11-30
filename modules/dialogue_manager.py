# modules/dialogue_manager.py
from typing import Dict, Any, Optional

label2id = {"complaint": 0, "escalation": 1, "order": 2, "other": 3, "refund": 4}
id2label = {v: k for k, v in label2id.items()}


class DialogueManager:
    def __init__(self):
        self.state = "idle"
        self.last_order = None

    def handle_intent(self, intent: str, text: Optional[str] = None) -> Dict[str, Any]:
        if intent not in label2id:
            intent = "other"
        if intent == "order":
            return self.handle_order(text)
        if intent == "refund":
            return self.handle_refund(text)
        if intent == "complaint":
            return self.handle_complaint(text)
        if intent == "escalation":
            return self.handle_escalation(text)
        return self.handle_other(text)

    def handle_order(self, text):
        self.state = "ordering"
        return {"response": "تمام يا فندم إتفضل قولّي تحب تطلب ايه؟", "need_llm": False}

    def handle_refund(self, text):
        self.state = "refund_process"
        return {
            "response": "معلش يا فندم على اللي حصل ممكن تقولّي رقم الأوردر أو المشكلة بالظبط؟",
            "need_llm": False,
        }

    def handle_complaint(self, text):
        self.state = "complaint"
        return {
            "response": "آسفين جدًا إن التجربة مكنتش كويسة قولّي إيه اللي ضايقك؟",
            "need_llm": True,
        }

    def handle_escalation(self, text):
        self.state = "escalation"
        return {
            "response": "تمام هحوّل حضرتك للإدارة في أسرع وقت. ممكن تديني رقم التواصل؟",
            "need_llm": False,
        }

    def handle_other(self, text):
        self.state = "idle"
        return {"response": "ممكن توضّحلي أكتر يا فندم؟", "need_llm": True}
