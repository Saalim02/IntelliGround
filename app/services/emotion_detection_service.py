"""Conservative per-utterance emotion detection; no persistent user profile."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EmotionDetection:
    emotion: str = "neutral"
    confidence: float = 0.0
    response_tone: str = "neutral"
    source: str = "lexical"

    def as_dict(self) -> dict:
        return {"detected_emotion": self.emotion, "emotion_confidence": self.confidence, "response_tone": self.response_tone}


_PATTERNS = {
    "distressed": (r"\b(help me|i feel hopeless|i can't cope|suicide|hurt myself)\b", r"நான் மிகவும் கவலை|నాకు చాలా బాధగా|ನನಗೆ ತುಂಬಾ ದುಃಖ"),
    "sad": (r"\b(i feel sad|i am sad|feeling down|lonely|depressed)\b", r"मुझे दुखी|मैं उदास|ನಾನು ದುಃಖಿತ|நான் சோகமாக|నేను బాధగా|എനിക്ക് സങ്കടം|আমি দুঃখিত|મને દુઃખી|ਮੈਂ ਉਦਾਸ"),
    "frustrated": (r"\b(frustrated|frustrating|this is not working|annoyed|ridiculous|useless)\b", r"बहुत परेशान|ತೊಂದರೆ|குழப்பம்|చిరాకు|പ്രശ്നം|বিরক্ত|પરેશાન|ਪਰੇਸ਼ਾਨ"),
    "angry": (r"\b(angry|furious|hate this|shut up)\b", r"गुस्सा|ಕೋಪ|கோபம்|కోపం|കോപം|রাগ|ગુસ્સો|ਗੁੱਸਾ"),
    "confused": (r"\b(confused|i don't understand|what does that mean|explain simply)\b", r"समझ नहीं|ಭ್ರಮೆ|புரியவில்லை|అర్థం కావడం లేదు|മനസ്സിലാകുന്നില്ല|বুঝতে পারছি না|સમજાતું નથી|ਸਮਝ ਨਹੀਂ"),
    "excited": (r"\b(excited|amazing|wonderful|great news)\b", r"उत्साहित|ಅದ್ಭುತ|அருமை|అద్భుతం|ആവേശം|দারুণ|ઉત્સાહ|ਉਤਸ਼ਾਹ"),
    "happy": (r"\b(happy|glad|thankful|thank you so much)\b", r"खुश|संतोष|ಸಂತೋಷ|மகிழ்ச்சி|సంతోషం|സന്തോഷം|খুশি|ખુશી|ਖੁਸ਼"),
    "curious": (r"\b(i wonder|curious|tell me more|i'd like to know)\b", r"जानना चाहता|ಕುತೂಹಲ|தெரிந்து கொள்ள|తెలుసుకోవాలి|അറിയാൻ|জানতে চাই|જાણવું છે|ਜਾਣਨਾ"),
}
_TONES = {"angry": "calm", "frustrated": "reassuring", "sad": "empathetic", "distressed": "supportive", "confused": "simplified", "happy": "warm", "excited": "enthusiastic", "curious": "encouraging"}


def detect_emotion(text: str) -> EmotionDetection:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    matches = [emotion for emotion, patterns in _PATTERNS.items() if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns)]
    if len(matches) != 1:
        return EmotionDetection()
    emotion = matches[0]
    return EmotionDetection(emotion, 0.78, _TONES[emotion])


def is_emotional_conversation(text: str) -> bool:
    return detect_emotion(text).emotion != "neutral" and not re.search(r"\b(pdf|document|page|chapter|section|file|uploaded)\b", (text or "").lower())
