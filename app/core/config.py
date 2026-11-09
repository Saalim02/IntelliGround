import os
from pathlib import Path

from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from the project .env file
load_dotenv(BASE_DIR / ".env")

# Storage root
STORAGE_DIR = BASE_DIR / "storage"

# Runtime storage folders
UPLOAD_DIR = STORAGE_DIR / "uploads"
VECTORSTORE_DIR = STORAGE_DIR / "vectorstore"
PAGE_IMAGE_DIR = STORAGE_DIR / "page_images"
DEBUG_OUTPUT_DIR = STORAGE_DIR / "debug_output"
SUMMARY_DIR = STORAGE_DIR / "summaries"
EVALUATION_DIR = STORAGE_DIR / "evaluations"

# Logs folder
LOG_DIR = BASE_DIR / "logs"

# Automatically create folders if missing
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Validation rules
ALLOWED_EXTENSIONS = [".pdf"]
ALLOWED_MIME_TYPES = ["application/pdf"]
MAX_FILE_SIZE_MB = 200
MAX_PDF_PAGES = 1400
# LLM evaluation configuration
# Can be overridden through the EVALUATION_MODEL environment variable.
EVALUATION_MODEL = os.getenv(
    "EVALUATION_MODEL",
    "gpt-4o-mini",
)

# Gnani Vachana API configuration. These are optional until voice features are
# configured; values are never returned to the frontend or written to logs.
GNANI_API_KEY = os.getenv("GNANI_API_KEY", "").strip()
GNANI_STT_MODEL = os.getenv("GNANI_STT_MODEL", "prisma-v3.0").strip()
# Provider-supported defaults keep Timbre usable when only GNANI_API_KEY is
# supplied.  Both values remain environment-overridable for deployments that
# select another verified voice.
GNANI_TTS_MODEL = os.getenv("GNANI_TTS_MODEL", "timbre-v2.5").strip()
GNANI_TTS_VOICE = os.getenv("GNANI_TTS_VOICE", "Kaveri").strip()
GNANI_STT_URL = os.getenv(
    "GNANI_STT_URL",
    "https://api.vachana.ai/stt/v3",
).strip()
GNANI_TTS_URL = os.getenv(
    "GNANI_TTS_URL",
    "https://api.vachana.ai/api/v1/tts/inference",
).strip()
GNANI_STT_WS_URL = os.getenv(
    "GNANI_STT_WS_URL",
    "wss://api.vachana.ai/stt/v3/stream",
).strip()
GNANI_TTS_WS_URL = os.getenv(
    "GNANI_TTS_WS_URL",
    "wss://api.vachana.ai/api/v1/tts",
).strip()
GNANI_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("GNANI_REQUEST_TIMEOUT_SECONDS", "60")
)
GNANI_MAX_AUDIO_BYTES = int(
    os.getenv("GNANI_MAX_AUDIO_BYTES", str(15 * 1024 * 1024))
)
# -----------------------------------
# APPLICATION URL CONFIGURATION
# -----------------------------------

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:8501",
).strip()

# -----------------------------------
# DEVELOPER ACCESS CONFIGURATION
# -----------------------------------

DEVELOPER_ACCESS_PASSWORD = os.getenv(
    "DEVELOPER_ACCESS_PASSWORD",
    "",
).strip()

# Local development can use one database-backed identity so all existing
# user-scoped storage, indexes, conversations, evaluations, and page-image
# paths continue to work without requiring a browser login. Authentication
# remains the default unless this flag is explicitly enabled.
LOCAL_DEV_AUTH_BYPASS = os.getenv(
    "LOCAL_DEV_AUTH_BYPASS",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
LOCAL_DEV_EMAIL = "localdev@intelliground.local"

# -----------------------------------
# REQUIRED ENVIRONMENT VALIDATION
# -----------------------------------
REQUIRED_ENV_VARS = (
    "OPENAI_API_KEY",
    "RAG_ACCESS_PASSWORD",
    "JWT_SECRET_KEY",
)


def validate_required_environment() -> None:
    """
    Validate required backend secrets without exposing
    their values in logs or error messages.
    """

    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name) or not os.getenv(name).strip()
    ]

    if missing:
        raise RuntimeError(
            "Required environment configuration is missing: "
            + ", ".join(missing)
        )
