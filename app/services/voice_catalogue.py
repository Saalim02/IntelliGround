"""Verified, cached Timbre v2.5 voice catalogue."""

from functools import lru_cache

from app.core.config import GNANI_TTS_MODEL
from app.services.gnani_prisma_service import VOICE_LANGUAGE_CODES

try:
    from gnani.tts import GnaniTTSClient, TIMBRE_V25_VOICES
except ImportError:
    GnaniTTSClient = None
    TIMBRE_V25_VOICES = frozenset()


# Official SDK 0.9.0 snapshot, used only when the SDK is not installed yet.
_SDK_V25_VOICE_SNAPSHOT = frozenset({
    "Nalini", "Bhavna", "Yashvi", "Urmila", "Jwala", "Chitra", "Ambuja",
    "Deepak", "Roopesh", "Vikrant", "Hemraj", "Jalaj", "Omkar", "Kaveri",
    "Trupti", "Devika", "Pranav", "Shlok", "Girish", "Asmita", "Trisha",
    "Brinda", "Vedika", "Noopur", "Suhana", "Lehara", "Lavanya", "Yukti",
    "Varuni", "Saanvi", "Kavin", "Reshma", "Riyaan", "Zahira", "Ishaan",
    "Kirra", "Dhruva", "Falak", "Veera", "Mehuli", "Zayan", "Poorvi",
})

# Metadata published in Gnani's Timbre v2.5 voice explorer. Gender is used
# only because the provider publishes it; it is never inferred from a name.
_VOICE_METADATA = {
    "Kaveri": ("Female", "English", "Confident", "A firm, confident, and composed voice that conveys professionalism.", "Customer support"),
    "Trupti": ("Female", "English", "Confident", "An upbeat, articulate, and mature voice that communicates with confidence and clarity.", "Business calls"),
    "Devika": ("Female", "English", "Authoritative", "An authoritative, confident, and professional voice for business conversations.", "Business calls"),
    "Pranav": ("Male", "English", "Warm", "A friendly, warm, and supportive voice that encourages meaningful interactions.", "Podcast & narration"),
    "Shlok": ("Male", "English", "Warm", "A warm, conversational, and friendly voice that feels natural and engaging.", "Conversational AI"),
    "Girish": ("Male", "English", "Warm", "A warm, expressive, and engaging voice that keeps conversations lively.", "Conversational AI"),
    "Nalini": ("Female", "Kannada", "Playful", "A friendly, playful, and cheerful voice that makes conversations enjoyable.", "Audiobook narration"),
    "Bhavna": ("Female", "Hindi", "Warm", "A friendly, warm, and supportive voice that creates a welcoming experience.", "Customer support"),
    "Yashvi": ("Female", "Hindi", "Calm", "A gentle, reassuring, and comforting voice that creates a sense of care.", "Healthcare reminders"),
    "Urmila": ("Female", "Hindi", "Professional", "A refined, composed, and formal voice suited for professional interactions.", "Business calls"),
    "Jwala": ("Female", "Hindi", "Playful", "A friendly, cheerful, and natural voice that feels easy to connect with.", "Conversational AI"),
    "Chitra": ("Female", "Gujarati", "Calm", "A mature, calm, and reassuring voice ideal for thoughtful interactions.", "Healthcare reminders"),
    "Ambuja": ("Female", "Hindi", "Professional", "A clear, composed, and professional voice suited for formal conversations.", "Business calls"),
    "Deepak": ("Male", "Hindi", "Authoritative", "A clear, measured, and confident voice that communicates with authority.", "Explainer video"),
    "Roopesh": ("Male", "Hindi", "Professional", "A grounded, professional, and knowledgeable voice that inspires trust.", "Business calls"),
    "Vikrant": ("Male", "Hindi", "Warm", "A warm, genuine, and trustworthy voice that builds confidence with listeners.", "Customer support"),
    "Hemraj": ("Male", "Kannada", "Energetic", "A confident, energetic, and youthful voice that captures attention.", "Sales & outreach"),
    "Jalaj": ("Male", "Telugu", "Energetic", "An expressive, dynamic, and engaging voice that brings conversations to life.", "Conversational AI"),
    "Omkar": ("Male", "Tamil", "Energetic", "An energetic, confident, and engaging voice that commands attention.", "Sales & outreach"),
    "Poorvi": ("Female", "Telugu", "Warm", "A natural, expressive, and versatile voice that adapts seamlessly to mixed-language conversations.", "Code-switching"),
    "Asmita": ("Female", "Tamil", "Professional", "A polished, refined, and smooth voice that delivers messages with elegance.", "Business calls"),
    "Trisha": ("Female", "Tamil", "Warm", "A warm, inviting, and friendly voice that creates an approachable experience.", "Customer support"),
    "Brinda": ("Female", "Tamil", "Professional", "A steady, grounded, and professional voice that inspires confidence.", "Business calls"),
    "Vedika": ("Female", "Tamil", "Professional", "A crisp, articulate, and precise voice that ensures clarity in every interaction.", "IVR & notifications"),
    "Noopur": ("Female", "Tamil", "Warm", "A warm, friendly, and natural voice that feels welcoming and genuine.", "Customer support"),
    "Suhana": ("Female", "Telugu", "Calm", "A calm, soothing, and confident voice that delivers reassurance with clarity.", "Healthcare reminders"),
    "Lehara": ("Female", "Telugu", "Calm", "A gentle, comforting, and trustworthy voice that puts people at ease.", "Healthcare reminders"),
    "Lavanya": ("Female", "Telugu", "Empathetic", "A soft, empathetic, and reassuring voice that builds trust with listeners.", "Healthcare reminders"),
    "Yukti": ("Female", "Telugu", "Warm", "A warm, welcoming, and graceful voice that leaves a lasting positive impression.", "Customer support"),
    "Varuni": ("Female", "Telugu", "Warm", "A clear, natural, and conversational voice that feels effortless and engaging.", "Conversational AI"),
    "Saanvi": ("Female", "Kannada", "Playful", "A joyful, lively, and cheerful voice that adds energy to every conversation.", "Audiobook narration"),
    "Kavin": ("Male", "Kannada", "Authoritative", "A firm, articulate, and authoritative voice suited for impactful communication.", "Business calls"),
    "Reshma": ("Female", "Malayalam", "Playful", "A bright, cheerful, and welcoming voice that leaves a positive impression.", "Sales & outreach"),
    "Riyaan": ("Male", "Malayalam", "Warm", "A friendly, relaxed, and natural voice that feels conversational and easygoing.", "Conversational AI"),
    "Zahira": ("Female", "Tamil", "Energetic", "An energetic, enthusiastic, and approachable voice that brings positivity to conversations.", "Sales & outreach"),
    "Ishaan": ("Male", "Marathi", "Professional", "A clear, articulate, and precise voice that delivers information effectively.", "IVR & notifications"),
    "Kirra": ("Female", "Bengali", "Calm", "A soft, gentle, and natural voice that feels calm and comforting.", "Healthcare reminders"),
    "Dhruva": ("Male", "Bengali", "Warm", "A casual, natural, and relatable voice that feels authentic and approachable.", "Conversational AI"),
    "Falak": ("Female", "Gujarati", "Energetic", "A youthful, clear, and approachable voice that sounds fresh and engaging.", "Sales & outreach"),
    "Veera": ("Male", "Gujarati", "Warm", "A friendly, youthful, and natural voice that feels approachable and authentic.", "Conversational AI"),
    "Mehuli": ("Female", "Punjabi", "Energetic", "An expressive, lively, and energetic voice full of enthusiasm.", "Sales & outreach"),
    "Zayan": ("Male", "Punjabi", "Empathetic", "A soft, caring, and empathetic voice that conveys warmth and compassion.", "Customer support"),
}


def _sdk_voice_ids() -> frozenset[str]:
    if GNANI_TTS_MODEL != "timbre-v2.5":
        return frozenset()
    if GnaniTTSClient is not None:
        return frozenset(GnaniTTSClient.supported_voices(model="timbre-v2.5"))
    return frozenset(TIMBRE_V25_VOICES) or _SDK_V25_VOICE_SNAPSHOT


def _language_name(code: str) -> str:
    from app.services.language_registry import LANGUAGE_NAMES

    return LANGUAGE_NAMES.get(code, code)


@lru_cache(maxsize=1)
def available_voices() -> tuple[dict, ...]:
    ids = _sdk_voice_ids()
    if len(ids) != 42:
        raise RuntimeError("Gnani Timbre v2.5 catalogue must contain exactly 42 voices.")
    languages = [{"code": code, "name": _language_name(code)} for code in VOICE_LANGUAGE_CODES]
    order = list(_VOICE_METADATA)
    return tuple({
        "id": voice_id,
        "name": voice_id,
        "model": "timbre-v2.5",
        "gender": _VOICE_METADATA[voice_id][0],
        "best_for": _VOICE_METADATA[voice_id][1],
        "style": _VOICE_METADATA[voice_id][2],
        "description": _VOICE_METADATA[voice_id][3],
        "use_case": _VOICE_METADATA[voice_id][4],
        "languages": languages,
    } for voice_id in sorted(ids, key=order.index))


def get_voice(voice_id: str) -> dict | None:
    return next((voice for voice in available_voices() if voice["id"] == voice_id), None)


def validate_voice_language(voice_id: str, language: str) -> dict:
    voice = get_voice(voice_id)
    if voice is None:
        raise ValueError("This voice is not available in the configured Gnani Timbre v2.5 catalogue.")
    if language not in VOICE_LANGUAGE_CODES:
        raise ValueError("This language does not have verified Gnani speech support.")
    return voice
