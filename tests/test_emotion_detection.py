import pytest

from app.services.emotion_detection_service import detect_emotion, is_emotional_conversation


@pytest.mark.parametrize("text,emotion,tone", [
    ("I feel sad today", "sad", "empathetic"),
    ("This is frustrating and not working", "frustrated", "reassuring"),
    ("I am happy, thank you so much", "happy", "warm"),
    ("I am confused, explain simply", "confused", "simplified"),
    ("ನನಗೆ ತುಂಬಾ ದುಃಖ", "distressed", "supportive"),
])
def test_emotion_detection_is_per_utterance(text, emotion, tone):
    result = detect_emotion(text)
    assert result.emotion == emotion
    assert result.response_tone == tone
    assert result.confidence > 0


def test_uncertain_emotion_defaults_to_neutral():
    result = detect_emotion("Could you tell me about the document?")
    assert result.emotion == "neutral"
    assert result.response_tone == "neutral"


def test_document_mention_prevents_emotion_from_bypassing_grounding():
    assert not is_emotional_conversation("I am frustrated that the PDF explanation is unclear")
    assert is_emotional_conversation("I feel sad today")
