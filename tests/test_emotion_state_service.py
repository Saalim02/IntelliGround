import pytest

from app.services.emotion_state_service import choose_emotion, progress_announcement


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"intent": "GREETING"}, "HAPPY"),
        ({"intent": "DOCUMENT_QUERY", "evidence_count": 2}, "EXCITED"),
        ({"intent": "DOCUMENT_QUERY", "evidence_count": 2, "processing_seconds": 2.1}, "THINKING"),
        ({"intent": "DOCUMENT_QUERY", "evidence_count": 2, "confidence": .9}, "CONFIDENT"),
        ({"intent": "DOCUMENT_QUERY"}, "EMPATHETIC"),
        ({"intent": "OUT_OF_SCOPE"}, "PLAYFULLY_ANNOYED"),
        ({"intent": "DOCUMENT_QUERY", "failed": True}, "APOLOGETIC"),
    ],
)
def test_state_machine_uses_deterministic_events(kwargs, expected):
    result = choose_emotion(**kwargs)
    assert result.emotion == expected
    assert result.trigger
    assert result.delivery
    assert result.avatar


def test_disabled_emotion_is_neutral():
    assert choose_emotion(evidence_count=2, settings={"enabled": False}).emotion == "NEUTRAL"


def test_progress_announcement_waits_for_threshold_and_localizes():
    settings = {"enabled": True, "progress_enabled": True, "progress_threshold_seconds": 2}
    assert progress_announcement("hi", 1.99, settings=settings) is None
    assert progress_announcement("hi", 2, settings=settings).startswith("एक क्षण")
    assert progress_announcement("ml", 3, settings=settings)


def test_progress_announcement_can_be_disabled():
    settings = {"enabled": True, "progress_enabled": False, "progress_threshold_seconds": 0}
    assert progress_announcement("en", 10, settings=settings) is None
