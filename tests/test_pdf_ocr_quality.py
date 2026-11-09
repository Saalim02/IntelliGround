
from types import SimpleNamespace

import pytest

from app.services import pdf_service


def make_page(fonts=None):
    """Create a lightweight mock PDF page."""

    return SimpleNamespace(
        get_fonts=lambda full=True: fonts or []
    )


# --------------------------------------------------
# LEGACY FONT DETECTION
# --------------------------------------------------

def test_tamil_legacy_font_is_detected():
    page = make_page([
        (
            1,
            "ttf",
            "TrueType",
            "AAAAAJ+Sun-TommyTamilNormal",
            "F1",
            "WinAnsiEncoding",
            0,
        )
    ])

    assert pdf_service._get_legacy_font_language(page) == "tam+eng"


def test_kannada_nudi_font_is_detected():
    page = make_page([
        (
            1,
            "ttf",
            "TrueType",
            "ABCDEE+Nudi 01 k",
            "F1",
            "WinAnsiEncoding",
            0,
        )
    ])

    assert pdf_service._get_legacy_font_language(page) == "kan+eng"


def test_unknown_legacy_font_returns_none():
    page = make_page([
        (
            1,
            "ttf",
            "TrueType",
            "Arial",
            "F1",
            "WinAnsiEncoding",
            0,
        )
    ])

    assert pdf_service._get_legacy_font_language(page) is None


# --------------------------------------------------
# UNICODE CORRUPTION DETECTION
# --------------------------------------------------

def test_private_use_corruption_is_detected():
    text = "\uf0b6" * 576 + "தமிழ்"

    assert pdf_service._detect_unicode_script(text) == "Tamil"
    assert pdf_service._has_corrupted_text(text) is True


def test_replacement_character_corruption_is_detected():
    text = "தமிழ்" * 90 + "\uFFFD" * 10

    assert pdf_service._has_corrupted_text(text) is True


def test_clean_tamil_text_is_not_marked_as_corrupted():
    text = "தமிழ் இயற்பியல் மற்றும் அறிவியல்"

    assert pdf_service._has_corrupted_text(text) is False
    assert pdf_service._detect_unicode_script(text) == "Tamil"


def test_empty_text_is_not_marked_as_corrupted():
    assert pdf_service._has_corrupted_text("") is False


# --------------------------------------------------
# OCR LANGUAGE CONFIGURATION
# --------------------------------------------------

@pytest.mark.parametrize(
    ("language_code", "expected"),
    [
        ("eng", "eng"),
        ("hin", "hin+eng"),
        ("kan", "kan+eng"),
        ("tam", "tam+eng"),
        ("tel", "tel+eng"),
        ("mal", "mal+eng"),
        ("ben", "ben+eng"),
        ("guj", "guj+eng"),
        ("mar", "mar+eng"),
        ("pan", "pan+eng"),
        ("urd", "urd+eng"),
    ],
)
def test_supported_ocr_language_mapping(language_code, expected):
    assert (
        pdf_service.SUPPORTED_OCR_LANGUAGES[language_code]
        == expected
    )


def test_auto_is_default_language():
    assert pdf_service.SUPPORTED_OCR_LANGUAGES["auto"] is None


def test_explicit_language_selection_overrides_script_detection():
    page = make_page()

    result = pdf_service._detect_ocr_language(
        page,
        "हिन्दी मराठी देवनागरी",
        language_hint="mar",
    )

    assert result == "mar+eng"


def test_explicit_kannada_selection():
    page = make_page()

    result = pdf_service._detect_ocr_language(
        page,
        "",
        language_hint="kan",
    )

    assert result == "kan+eng"


def test_explicit_urdu_selection():
    page = make_page()

    result = pdf_service._detect_ocr_language(
        page,
        "",
        language_hint="urd",
    )

    assert result == "urd+eng"


def test_legacy_font_detection_takes_priority_over_explicit_selection():
    page = make_page([
        (
            1,
            "ttf",
            "TrueType",
            "ABCDEE+Nudi 01 k",
            "F1",
            "WinAnsiEncoding",
            0,
        )
    ])

    result = pdf_service._detect_ocr_language(
        page,
        "",
        language_hint="tam",
    )

    assert result == "kan+eng"


def test_auto_language_detection_uses_unicode_script():
    page = make_page()

    result = pdf_service._detect_ocr_language(
        page,
        "தமிழ் இயற்பியல் மற்றும் அறிவியல்",
    )

    assert result == "tam+eng"


def test_invalid_language_is_rejected_before_pdf_open(monkeypatch):
    def unexpected_open(*args, **kwargs):
        pytest.fail("PDF should not be opened for an invalid language")

    monkeypatch.setattr(
        pdf_service.fitz,
        "open",
        unexpected_open,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported OCR language",
    ):
        pdf_service.extract_text_from_pdf(
            "test.pdf",
            language_hint="invalid",
        )


# --------------------------------------------------
# TESSERACT MODEL VALIDATION
# --------------------------------------------------

def test_available_ocr_languages_returns_installed_models(monkeypatch):
    monkeypatch.setattr(
        pdf_service.pytesseract,
        "get_languages",
        lambda config="": ["eng", "kan", "tam"],
    )

    result = pdf_service._available_ocr_languages(
        "kan+eng",
        strict=True,
    )

    assert result == "kan+eng"


def test_explicit_language_selection_fails_if_model_is_missing(monkeypatch):
    monkeypatch.setattr(
        pdf_service.pytesseract,
        "get_languages",
        lambda config="": ["eng", "kan"],
    )

    with pytest.raises(
        ValueError,
        match="Required OCR language model",
    ):
        pdf_service._available_ocr_languages(
            "tam+eng",
            strict=True,
        )


def test_automatic_detection_filters_missing_models(monkeypatch):
    monkeypatch.setattr(
        pdf_service.pytesseract,
        "get_languages",
        lambda config="": ["eng"],
    )

    result = pdf_service._available_ocr_languages(
        "tam+eng",
        strict=False,
    )

    assert result == "eng"


def test_missing_all_requested_models_raises_error(monkeypatch):
    monkeypatch.setattr(
        pdf_service.pytesseract,
        "get_languages",
        lambda config="": [],
    )

    with pytest.raises(
        ValueError,
        match="Required OCR language model",
    ):
        pdf_service._available_ocr_languages(
            "tam+eng",
        )


# --------------------------------------------------
# BLANK PAGE DETECTION
# --------------------------------------------------

def test_blank_page_detection_does_not_assume_blank_on_render_error():
    page = SimpleNamespace(
        get_pixmap=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("render failed")
        )
    )

    assert pdf_service._is_visually_blank(page) is False


def test_page_with_native_text_is_not_blank():
    page = SimpleNamespace(
        get_pixmap=lambda **kwargs: pytest.fail(
            "Rendering should not be necessary when native text exists"
        )
    )

    assert (
        pdf_service._is_visually_blank(
            page,
            native_text="This is a valid PDF page.",
        )
        is False
    )
