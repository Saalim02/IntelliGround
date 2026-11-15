"""Deterministic voice emotion state selection and persisted controls."""
from dataclasses import dataclass
import json
from pathlib import Path

from app.core.config import STORAGE_DIR

SETTINGS_PATH = STORAGE_DIR / "emotion_settings.json"
DEFAULT_SETTINGS = {
    "enabled": True,
    "intensity": 0.65,
    "progress_enabled": True,
    "progress_threshold_seconds": 2.0,
    "playfully_annoyed_intensity": 0.35,
}

DELIVERY = {
    "HAPPY": ("warm and natural", "happy"),
    "EXCITED": ("briefly enthusiastic, then answer", "excited"),
    "THINKING": ("calm progress update", "thinking"),
    "CONFIDENT": ("clear and assured", "confident"),
    "EMPATHETIC": ("empathetic and transparent", "empathetic"),
    "PLAYFULLY_ANNOYED": ("light, playful refusal without aggression", "playfully-annoyed"),
    "APOLOGETIC": ("apologetic and concise", "apologetic"),
    "NEUTRAL": ("clear and neutral", "neutral"),
}

RESPONSE_EMOTIONS = frozenset({"AUTO", "HAPPY", "SAD", "ANGRY", "CALM", "PROFESSIONAL", "FRIENDLY"})
MANUAL_DELIVERY = {
    "HAPPY": ("warm and positive", "happy"),
    "SAD": ("gentle and empathetic", "empathetic"),
    "ANGRY": ("firm and assertive, never insulting", "confident"),
    "CALM": ("calm and reassuring", "empathetic"),
    "PROFESSIONAL": ("clear, formal, and composed", "confident"),
    "FRIENDLY": ("warm, natural, and approachable", "happy"),
}

PROGRESS_MESSAGES = {
    "en": "Just a moment! I'm checking your document carefully.",
    "kn": "ಒಂದು ಕ್ಷಣ! ನಿಮ್ಮ ದಾಖಲೆಯನ್ನು ಜಾಗರೂಕತೆಯಿಂದ ಪರಿಶೀಲಿಸುತ್ತಿದ್ದೇನೆ.",
    "hi": "एक क्षण! मैं आपके दस्तावेज़ को ध्यान से जाँच रहा हूँ।",
    "te": "ఒక్క క్షణం! మీ పత్రాన్ని జాగ్రత్తగా పరిశీలిస్తున్నాను.",
    "ta": "ஒரு கணம்! உங்கள் ஆவணத்தை கவனமாகச் சரிபார்க்கிறேன்.",
    "ml": "ഒരു നിമിഷം! നിങ്ങളുടെ രേഖ ശ്രദ്ധാപൂർവ്വം പരിശോധിക്കുകയാണ്.",
}

@dataclass(frozen=True)
class EmotionState:
    emotion: str
    intensity: float
    trigger: str
    delivery: str
    avatar: str

    def as_dict(self):
        return {"emotion": self.emotion, "intensity": self.intensity, "trigger": self.trigger, "delivery": self.delivery, "avatar": self.avatar}

def get_settings():
    try:
        values = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_SETTINGS, **values}
    except (OSError, ValueError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)

def save_settings(values):
    current = get_settings()
    next_values = {**current, **values}
    if not 0 <= float(next_values["intensity"]) <= 1 or not 0 <= float(next_values["playfully_annoyed_intensity"]) <= 1 or float(next_values["progress_threshold_seconds"]) < 0:
        raise ValueError("Emotion intensity must be 0–1 and the announcement threshold must be non-negative.")
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(next_values, indent=2), encoding="utf-8")
    return next_values


def progress_announcement(language: str, elapsed_seconds: float, *, settings=None) -> str | None:
    """Return one honest delayed announcement, or None before the threshold."""
    cfg = {**DEFAULT_SETTINGS, **(settings or get_settings())}
    if not cfg["enabled"] or not cfg["progress_enabled"] or elapsed_seconds < float(cfg["progress_threshold_seconds"]):
        return None
    return PROGRESS_MESSAGES.get(language, PROGRESS_MESSAGES["en"])

def normalize_response_emotion(value: str | None) -> str:
    normalized = str(value or "AUTO").strip().upper()
    if normalized not in RESPONSE_EMOTIONS:
        return "AUTO"
    return normalized


def choose_emotion(*, intent: str | None = None, evidence_count: int = 0, confidence: float | None = None, processing_seconds: float = 0, failed: bool = False, settings=None, selected_emotion: str | None = None):
    cfg = {**DEFAULT_SETTINGS, **(settings or get_settings())}
    selected = normalize_response_emotion(selected_emotion)
    if selected != "AUTO":
        delivery, avatar = MANUAL_DELIVERY[selected]
        return EmotionState(selected, float(cfg["intensity"]), "user-selected response emotion", delivery, avatar)
    if not cfg["enabled"]:
        return EmotionState("NEUTRAL", 0, "emotional responses disabled", "clear and neutral", "neutral")
    if failed:
        state, trigger, intensity = "APOLOGETIC", "provider or processing failure", cfg["intensity"]
    elif intent in {"OUT_OF_SCOPE", "UNRELATED"}:
        state, trigger, intensity = "PLAYFULLY_ANNOYED", "unrelated question", cfg["playfully_annoyed_intensity"]
    elif intent in {"GREETING", "CONVERSATIONAL", "GENERAL_CONVERSATION"}:
        state, trigger, intensity = "HAPPY", "normal greeting or conversation", cfg["intensity"]
    elif evidence_count == 0:
        state, trigger, intensity = "EMPATHETIC", "no grounded evidence", cfg["intensity"]
    elif processing_seconds >= float(cfg["progress_threshold_seconds"]):
        state, trigger, intensity = "THINKING", "processing exceeded announcement threshold", cfg["intensity"]
    elif confidence is not None and confidence >= .85:
        state, trigger, intensity = "CONFIDENT", "strong validated evidence", cfg["intensity"]
    else:
        state, trigger, intensity = "EXCITED", "fast successful answer", cfg["intensity"]
    delivery, avatar = DELIVERY[state]
    return EmotionState(state, float(intensity), trigger, delivery, avatar)
