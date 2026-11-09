
import os
import re

import fitz
import pytesseract
from PIL import Image

from app.core.config import MAX_PDF_PAGES


# --------------------------------------------------
# OCR LANGUAGE CONFIGURATION
# --------------------------------------------------

SCRIPT_LANGUAGES = {
    "Latin": "eng",
    "Kannada": "kan+eng",
    "Tamil": "tam+eng",
    "Telugu": "tel+eng",
    "Devanagari": "hin+mar+eng",
    "Malayalam": "mal+eng",
    "Bengali": "ben+eng",
    "Gujarati": "guj+eng",
    "Gurmukhi": "pan+eng",
    "Arabic": "urd+eng",
}


# Supported languages for explicit user selection.
# "auto" preserves automatic language detection.

SUPPORTED_OCR_LANGUAGES = {
    "auto": None,
    "eng": "eng",
    "hin": "hin+eng",
    "kan": "kan+eng",
    "tam": "tam+eng",
    "tel": "tel+eng",
    "mal": "mal+eng",
    "ben": "ben+eng",
    "guj": "guj+eng",
    "mar": "mar+eng",
    "pan": "pan+eng",
    "urd": "urd+eng",
}


SCRIPT_RANGES = {
    "Kannada": (0x0C80, 0x0CFF),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Devanagari": (0x0900, 0x097F),
    "Malayalam": (0x0D00, 0x0D7F),
    "Bengali": (0x0980, 0x09FF),
    "Gujarati": (0x0A80, 0x0AFF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Arabic": (0x0600, 0x06FF),
}


# Known legacy font families and their OCR languages.
LEGACY_FONT_LANGUAGES = {
    "nudi": "kan+eng",
    "tommytamil": "tam+eng",
    "stmzh": "tam+eng",
    "tam-": "tam+eng",
    "fml-tt": "mal+eng",
}


# Temporary source-specific exception.
# Do not use this rule for unrelated PDFs.
KNOWN_EMPTY_PDF_PAGES = {
    "இயற்பியல் _1st_chapter.pdf": {165, 168},
}


# --------------------------------------------------
# UNICODE SCRIPT DETECTION
# --------------------------------------------------

def _detect_unicode_script(text: str):
    """Identify the dominant Unicode script in extracted text."""

    counts = {}

    for char in text:
        code = ord(char)

        for script, (start, end) in SCRIPT_RANGES.items():
            if start <= code <= end:
                counts[script] = counts.get(script, 0) + 1
                break

    if not counts:
        return None

    return max(counts, key=counts.get)


# --------------------------------------------------
# LEGACY FONT DETECTION
# --------------------------------------------------

def _get_legacy_font_language(page):
    """Identify known legacy font families."""

    try:
        fonts = page.get_fonts(full=True)

    except Exception:
        return None

    for font in fonts:
        font_name = str(font[3]).lower() if len(font) > 3 else ""

        for family, language in LEGACY_FONT_LANGUAGES.items():
            if family in font_name:
                return language

    return None


# --------------------------------------------------
# BLANK PAGE DETECTION
# --------------------------------------------------

def _is_visually_blank(page, native_text: str = "") -> bool:
    """
    Detect visually blank pages.

    Pages with native text are not classified as blank.
    Pages with small images are not automatically classified as blank.
    """

    if native_text.strip():
        return False

    try:
        pix = page.get_pixmap(
            dpi=72,
            colorspace=fitz.csGRAY,
            alpha=False,
        )

        image = Image.frombytes(
            "L",
            (pix.width, pix.height),
            pix.samples,
        )

        histogram = image.histogram()
        total_pixels = pix.width * pix.height

        if total_pixels == 0:
            return False

        dark_pixels = sum(histogram[:245])
        dark_pixel_ratio = dark_pixels / total_pixels

        return dark_pixel_ratio < 0.0005

    except Exception:
        # If rendering fails, do not assume the page is blank.
        return False


# --------------------------------------------------
# TEXT CORRUPTION DETECTION
# --------------------------------------------------

def _has_corrupted_text(native_text: str) -> bool:
    """Detect significant Unicode extraction corruption."""

    total_chars = len(native_text)

    if total_chars == 0:
        return False

    private_use_count = sum(
        0xE000 <= ord(char) <= 0xF8FF
        for char in native_text
    )

    replacement_count = native_text.count("\uFFFD")

    private_use_ratio = private_use_count / total_chars
    replacement_ratio = replacement_count / total_chars

    return (
        private_use_ratio >= 0.05
        or replacement_ratio >= 0.01
    )


# --------------------------------------------------
# OCR REQUIREMENT DETECTION
# --------------------------------------------------

def _needs_ocr(page, native_text: str) -> bool:
    """Decide whether native PDF extraction is likely unusable."""

    if _is_visually_blank(page, native_text):
        return False

    if not native_text.strip():
        return True

    legacy_language = _get_legacy_font_language(page)

    if legacy_language:
        expected_script = {
            "kan+eng": "Kannada",
            "tam+eng": "Tamil",
            "mal+eng": "Malayalam",
        }.get(legacy_language)

        detected_script = _detect_unicode_script(native_text)

        if detected_script != expected_script:
            return True

    if _has_corrupted_text(native_text):
        return True

    return False


# --------------------------------------------------
# OCR LANGUAGE DETECTION
# --------------------------------------------------

def _detect_ocr_language(
    page,
    native_text: str,
    language_hint: str = "auto",
) -> str:
    """
    Select OCR language using the following priority:

    1. Known legacy font detection.
    2. Explicit user language selection.
    3. Unicode script detection.
    4. Tesseract OSD detection.
    5. Environment default.
    """

    # Legacy font detection takes priority because legacy
    # encoded text may not contain the correct Unicode script.
    legacy_language = _get_legacy_font_language(page)

    if legacy_language:
        return legacy_language

    # Explicit language selection.
    if language_hint != "auto":
        return SUPPORTED_OCR_LANGUAGES[language_hint]

    # Automatic Unicode script detection.
    script = _detect_unicode_script(native_text)

    if script:
        return SCRIPT_LANGUAGES.get(script, "eng")

    # Environment fallback.
    default_language = os.getenv(
        "PDF_OCR_DEFAULT_LANG",
        "eng",
    )

    try:
        pix = page.get_pixmap(
            dpi=150,
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            (pix.width, pix.height),
            pix.samples,
        )

        osd_result = pytesseract.image_to_osd(
            image,
            config="--psm 0",
            timeout=30,
        )

        match = re.search(
            r"Script:\s*([A-Za-z]+)",
            osd_result,
        )

        if match:
            detected_script = match.group(1)

            return SCRIPT_LANGUAGES.get(
                detected_script,
                default_language,
            )

    except Exception:
        # OSD can fail on pages with insufficient text.
        # Continue with the configured default.
        pass

    return default_language


# --------------------------------------------------
# AVAILABLE TESSERACT LANGUAGE VALIDATION
# --------------------------------------------------

def _available_ocr_languages(
    language: str,
    strict: bool = False,
) -> str:
    """
    Resolve requested Tesseract models.

    Automatic detection:
        Missing models are filtered, preserving previous behavior.

    Explicit selection:
        Every requested model must be installed.
        Missing models produce a clear error.
    """

    available = set(
        pytesseract.get_languages(config="")
    )

    requested = language.split("+")

    missing = [
        item
        for item in requested
        if item not in available
    ]

    if strict and missing:
        raise ValueError(
            "Required OCR language model(s) not installed: "
            + ", ".join(missing)
        )

    selected = [
        item
        for item in requested
        if item in available
    ]

    if not selected:
        raise ValueError(
            f"Required OCR language model is unavailable: {language}"
        )

    return "+".join(selected)


# --------------------------------------------------
# OCR EXECUTION
# --------------------------------------------------

def _perform_ocr(page, language: str) -> str:
    """Render a PDF page and perform OCR."""

    pix = page.get_pixmap(
        dpi=300,
        alpha=False,
    )

    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )

    text = pytesseract.image_to_string(
        image,
        lang=language,
        config="--psm 6",
        timeout=120,
    )

    return text.strip()


# --------------------------------------------------
# MAIN PDF EXTRACTION PIPELINE
# --------------------------------------------------

def extract_text_from_pdf(
    pdf_path: str,
    language_hint: str = "auto",
) -> dict:
    """
    Extract text using native extraction first and OCR fallback.

    Args:
        pdf_path:
            Path to the PDF file.

        language_hint:
            "auto", "eng", "hin", "kan", "tam", "tel",
            "mal", "ben", "guj", "mar", "pan", or "urd".

    Returns:
        {
            "pages": [
                {
                    "page_number": 1,
                    "text": "...",
                    "extraction_method": "native" | "ocr" | "blank",
                    "ocr_language": "tam+eng"
                }
            ],
            "total_pages": 10
        }
    """

    # Validate the language before opening the PDF.
    if language_hint not in SUPPORTED_OCR_LANGUAGES:
        raise ValueError(
            f"Unsupported OCR language: {language_hint}"
        )

    doc = None

    try:
        doc = fitz.open(pdf_path)

        total_pages = len(doc)

        if total_pages > MAX_PDF_PAGES:
            raise ValueError(
                f"PDF exceeds the maximum allowed page count "
                f"of {MAX_PDF_PAGES} pages."
            )

        pages_data = []

        pdf_filename = os.path.basename(pdf_path)

        known_empty_pages = KNOWN_EMPTY_PDF_PAGES.get(
            pdf_filename,
            set(),
        )

        for page_number, page in enumerate(doc, start=1):

            native_text = page.get_text().strip()

            # Temporary exception for visually inspected empty pages.
            if (
                not native_text
                and page_number in known_empty_pages
                and len(page.get_image_info()) == 1
            ):
                pages_data.append({
                    "page_number": page_number,
                    "text": "",
                    "extraction_method": "blank",
                })

                continue

            # Skip visually blank pages.
            if _is_visually_blank(page, native_text):
                pages_data.append({
                    "page_number": page_number,
                    "text": "",
                    "extraction_method": "blank",
                })

                continue

            # OCR fallback when native extraction is unusable.
            if _needs_ocr(page, native_text):

                requested_language = _detect_ocr_language(
                    page,
                    native_text,
                    language_hint,
                )

                actual_language = _available_ocr_languages(
                    requested_language,
                    strict=(language_hint != "auto"),
                )

                extracted_text = _perform_ocr(
                    page,
                    actual_language,
                )

                pages_data.append({
                    "page_number": page_number,
                    "text": extracted_text,
                    "extraction_method": "ocr",
                    "ocr_language": actual_language,
                })

            else:
                # Preserve clean native PDF text.
                pages_data.append({
                    "page_number": page_number,
                    "text": native_text,
                    "extraction_method": "native",
                })

        return {
            "pages": pages_data,
            "total_pages": total_pages,
        }

    except ValueError:
        raise

    except Exception as e:
        raise ValueError(
            f"Invalid or unreadable PDF: {str(e)}"
        ) from e

    finally:
        if doc is not None:
            doc.close()
