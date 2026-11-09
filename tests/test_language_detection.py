
from app.services.language_detection_service import detect_text_language, detect_voice_response_language


def test_detects_kannada_script():
    result = detect_text_language(
        "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಗ್ರಾಮ ಪಂಚಾಯಿತಿ ಅಭಿವೃದ್ಧಿ"
    )
    assert result["language"] == "kn"


def test_detects_tamil_script():
    result = detect_text_language(
        "தமிழ்நாடு அரசு ஊராட்சி வளர்ச்சி"
    )
    assert result["language"] == "ta"


def test_detects_telugu_script():
    result = detect_text_language(
        "ఆంధ్రప్రదేశ్ ప్రభుత్వం గ్రామ పంచాయతీ"
    )
    assert result["language"] == "te"


def test_detects_english_latin_script():
    result = detect_text_language(
        "The government has issued a new development order."
    )
    assert result["language"] == "en"


def test_detects_mixed_kannada_and_english():
    result = detect_text_language(
        "ಕರ್ನಾಟಕ ಸರ್ಕಾರ Karnataka Government"
    )
    assert result["language"] == "mixed"


def test_empty_text_is_undetermined():
    result = detect_text_language("   123 -- !!!")
    assert result["language"] == "und"


def test_voice_response_language_supports_all_required_languages():
    assert detect_voice_response_language("The government issued a new order.", "hi") == "en"
    assert detect_voice_response_language("सरकार ने नया आदेश जारी किया है।", "en") == "hi"
    assert detect_voice_response_language("ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಹೊಸ ಆದೇಶ ಹೊರಡಿಸಿದೆ.", "en") == "kn"
    assert detect_voice_response_language("தமிழ்நாடு அரசு புதிய உத்தரவை வெளியிட்டது.", "en") == "ta"
    assert detect_voice_response_language("ఆంధ్రప్రదేశ్ ప్రభుత్వం కొత్త ఆదేశం జారీ చేసింది.", "en") == "te"


def test_voice_language_continuity_ignores_short_technical_term():
    assert detect_voice_response_language("API", "kn") == "kn"
    assert detect_voice_response_language("ಕರ್ನಾಟಕ ಸರ್ಕಾರ Karnataka", "kn") == "kn"


def test_short_non_latin_utterance_can_switch_language():
    assert detect_voice_response_language("நன்றி", "en") == "ta"


def test_voice_language_switches_on_dominant_new_language():
    assert detect_voice_response_language(
        "This is a complete English sentence asking about the document.",
        "hi",
    ) == "en"


def test_urdu_detection_is_preserved_but_provider_capability_remains_explicit():
    from app.services.language_registry import provider_capability

    assert detect_voice_response_language("آپ کیسے ہیں؟", "en") == "ur"
    assert provider_capability("prisma_stt", "ur") == "unsupported"
    assert provider_capability("timbre_tts", "ur") == "unsupported"
