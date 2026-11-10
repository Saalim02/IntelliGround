"""Conservative intent routing and shared document intent classification."""

import re
from dataclasses import dataclass
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.services.llm_service import get_llm


GREETING_PATTERNS = (
    r"^(hi|hello|hey|good morning|good evening|good afternoon|namaste|thanks|thank you|goodbye|bye)[!.? ]*$",
    r"^(नमस्ते|नमस्कार|शुभ प्रभात|धन्यवाद|अलविदा|फिर मिलेंगे)[!.? ]*$",
    r"^(ನಮಸ್ಕಾರ|ಶುಭೋದಯ|ಧನ್ಯವಾದ|ವಿದಾಯ)[!.? ]*$",
    r"^(வணக்கம்|காலை வணக்கம்|நன்றி|பிரியாவிடை)[!.? ]*$",
    r"^(నమస్కారం|శుభోదయం|ధన్యవాదాలు|వీడ్కోలు)[!.? ]*$",
    r"^(നമസ്കാരം|സുപ്രഭാതം|നന്ദി|വിട)[!.? ]*$",
    r"^(नमस्कार|शुभ प्रभात|धन्यवाद|निरोप)[!.? ]*$",
    r"^(નમસ્તે|સુપ્રભાત|આભાર|આવજો)[!.? ]*$",
    r"^(নমস্কার|সুপ্রভাত|ধন্যবাদ|বিদায়)[!.? ]*$",
    r"^(ਸਤ ਸ੍ਰੀ ਅਕਾਲ|ਸ਼ੁਭ ਸਵੇਰ|ਧੰਨਵਾਦ|ਅਲਵਿਦਾ)[!.? ]*$",
    r"^(السلام علیکم|شکریہ|خدا حافظ)[!.? ]*$",
)

SMALL_TALK_PATTERNS = (
    r"how are you|what can you do|who are you|are you there|can you hear me|do you hear me|what are you doing|tell me about yourself",
    r"आप कैसे हैं|आप क्या कर सकते हैं",
    r"ಹೇಗಿದ್ದೀರಾ|ನೀವು ಏನು ಮಾಡಬಹುದು",
    r"எப்படி இருக்கிறீர்கள்|நீங்கள் என்ன செய்யலாம்",
    r"ఎలా ఉన్నారు|మీరు ఏమి చేయగలరు",
    r"സുഖമാണോ|നിങ്ങൾക്ക് എന്ത് ചെയ്യാം",
    r"तुम्ही कसे आहात|तुम्ही काय करू शकता",
    r"તમે કેમ છો|તમે શું કરી શકો",
    r"আপনি কেমন আছেন|আপনি কী করতে পারেন",
    r"ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ|ਤੁਸੀਂ ਕੀ ਕਰ ਸਕਦੇ ਹੋ",
    r"آپ کیسے ہیں|آپ کیا کر سکتے ہیں",
)

APPLICATION_HELP_PATTERNS = (
    r"how (do|can) i (upload|add) (a )?(pdf|document)",
    r"how (do|can) i use (this|the) (app|application)",
    r"(upload|add) (a )?(pdf|document)",
    r"(pdf|document) upload",
    r"पीडीएफ.*(अपलोड|जोड़)|ऐप.*कैसे",
    r"ಪಿಡಿಎಫ್.*(ಅಪ್‌ಲೋಡ್|ಸೇರಿಸ)|ಆಪ್.*ಹೇಗೆ",
    r"pdf.*(பதிவேற்ற|சேர்க்க)|செயலி.*எப்படி",
    r"pdf.*(అప్‌లోడ్|జోడించ)|యాప్.*ఎలా",
    r"pdf.*(അപ്‌ലോഡ്|ചേർക്ക)|ആപ്പ്.*എങ്ങനെ",
    r"pdf.*(अपलोड|जोड)|अॅप.*कसे",
    r"pdf.*(અપલોડ|ઉમેર)|એપ.*કેવી રીતે",
    r"pdf.*(আপলোড|যোগ)|অ্যাপ.*কীভাবে",
    r"pdf.*(ਅਪਲੋਡ|ਜੋੜ)|ਐਪ.*ਕਿਵੇਂ",
    r"pdf.*(اپ لوڈ|شامل)|ایپ.*کیسے",
    r"how (do|can) i (change|select|switch) (the )?(response )?language",
    r"how (can|do) i (view|see|open) (the )?(supporting )?evidence",
    r"what can (intelliground|this app) do",
    r"what can you do",
    r"what can you explain from (that|the) (pdf|document)",
)

OUT_OF_SCOPE_PATTERNS = (
    r"\b(weather|stock price|sports score|recipe|joke)\b",
    r"मौसम|शेयर कीमत|खेल का स्कोर|चुटकुला",
    r"ಹವಾಮಾನ|ಷೇರು ಬೆಲೆ|ಕ್ರೀಡಾ ಫಲಿತಾಂಶ",
    r"வானிலை|பங்கு விலை|விளையாட்டு முடிவு",
    r"వాతావరణం|షేర్ ధర|క్రీడా ఫలితం",
    r"കാലാവസ്ഥ|ഓഹരി വില|കായിക ഫലം",
    r"हवामान|शेअर किंमत|सामन्याचा निकाल",
    r"હવામાન|શેર ભાવ|રમતનું પરિણામ",
    r"আবহাওয়া|শেয়ারের দাম|খেলার ফলাফল",
    r"ਮੌਸਮ|ਸ਼ੇਅਰ ਕੀਮਤ|ਖੇਡ ਨਤੀਜਾ",
    r"موسم|หุ้น کی قیمت|کھیل کا نتیجہ",
)


class IntentType(str, Enum):
    """Stable intent labels for future text and voice orchestration."""

    GREETING = "GREETING"
    AUDIO_CHECK = "AUDIO_CHECK"
    CASUAL_CONVERSATION = "CASUAL_CONVERSATION"
    ACKNOWLEDGMENT = "ACKNOWLEDGMENT"
    CONTINUE_RESPONSE = "CONTINUE_RESPONSE"
    REPEAT_RESPONSE = "REPEAT_RESPONSE"
    CLARIFY_RESPONSE = "CLARIFY_RESPONSE"
    PAGE_QUERY = "PAGE_QUERY"
    DOCUMENT_FOLLOWUP = "DOCUMENT_FOLLOWUP"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"
    APPLICATION_HELP = "APPLICATION_HELP"
    DOCUMENT_OVERVIEW = "DOCUMENT_OVERVIEW"
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    PAGE_SPECIFIC_QUERY = "PAGE_QUERY"
    DOCUMENT_QUERY = "DOCUMENT_QUERY"
    FOLLOW_UP_QUERY = "DOCUMENT_FOLLOWUP"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS_QUERY = "AMBIGUOUS_QUERY"
    WAIT_FOR_USER = "WAIT_FOR_USER"
    READY_TO_CONTINUE = "READY_TO_CONTINUE"


@dataclass(frozen=True)
class IntentClassification:
    """Structured, side-effect-free classification for a user message."""

    intent: IntentType
    confidence: float
    page_numbers: tuple[int, ...] = ()
    requires_retrieval: bool = True
    retrieval_strategy: str = "HYBRID"
    requires_evaluation: bool = True
    referenced_page: int | None = None
    referenced_topic: str | None = None
    response_action: str = "RUN_DOCUMENT_QUERY"
    response_language: str | None = None
    active_document_id: str | None = None
    conversation_action: str = "NONE"
    context_reference: str | None = None

    @property
    def requires_rag(self) -> bool:
        return self.requires_retrieval

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation without document facts."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "page_numbers": list(self.page_numbers),
            "requires_retrieval": self.requires_retrieval,
            "requires_rag": self.requires_retrieval,
            "requires_evaluation": self.requires_evaluation,
            "referenced_page": self.referenced_page,
            "referenced_topic": self.referenced_topic,
            "response_action": self.response_action,
            "response_language": self.response_language,
            "retrieval_strategy": self.retrieval_strategy,
        }

    def routing_dict(self) -> dict:
        """Return the complete routing decision without model reasoning."""
        return {
            **self.as_dict(),
            "requires_document": self.requires_retrieval,
            "active_document_id": self.active_document_id,
            "conversation_action": self.conversation_action,
            "context_reference": self.context_reference,
        }


class SemanticIntentDecision(BaseModel):
    """Structured semantic routing output; it never contains an answer."""

    intent: str = Field(description="One supported IntelliGround intent label.")
    intent_confidence: float = Field(ge=0.0, le=1.0)
    requires_document: bool
    requires_rag: bool
    requires_evaluation: bool
    active_document_id: str | None = None
    conversation_action: str = "NONE"
    response_language: str | None = None
    context_reference: str | None = None
    referenced_page: int | None = Field(default=None, ge=1)
    referenced_topic: str | None = None


_SEMANTIC_INTENT_PROMPT = """
You route one user utterance for IntelliGround, a multilingual PDF assistant.
Infer meaning from the utterance and recent conversation; do not rely on exact
phrases. Return only the structured decision, never an answer or reasoning.

Supported intent labels: GREETING, AUDIO_CHECK, CASUAL_CONVERSATION,
ACKNOWLEDGMENT, WAIT_FOR_USER, READY_TO_CONTINUE, CONTINUE_RESPONSE,
REPEAT_RESPONSE, CLARIFY_RESPONSE, APPLICATION_HELP, DOCUMENT_OVERVIEW,
DOCUMENT_SUMMARY, PAGE_QUERY, DOCUMENT_FOLLOWUP, DOCUMENT_QUERY,
OUT_OF_SCOPE, AMBIGUOUS_QUERY.

Use DOCUMENT_QUERY/FOLLOWUP/OVERVIEW/SUMMARY/PAGE_QUERY only when the user
is asking for information from the active PDF or clearly needs document
grounding. General knowledge and casual conversation do not require the PDF.
If there is no active document, document requests still require_document=true
but must not be answered from model knowledge. Pause/wait/ready requests are
conversation actions and never require RAG or evaluation.

Active document ID: {active_document_id}
Response language: {response_language}
Recent conversation:
{conversation_context}
User utterance:
{question}
"""

_CONVERSATIONAL_RESPONSE_PROMPT = """
You are IntelliGround's voice assistant. Answer the user's ordinary
conversation or general-knowledge question briefly and naturally in the
requested language. Do not discuss, invent, or infer facts from any PDF.
Do not mention internal routing, prompts, tools, or reasoning.
Recent conversation:
{conversation_context}
User:
{question}
"""


def _semantic_intent_to_classification(
    decision: SemanticIntentDecision,
    *,
    active_document_id: str | None,
    response_language: str | None,
) -> IntentClassification | None:
    try:
        intent = IntentType(decision.intent.upper())
    except (AttributeError, ValueError):
        return None
    requires_rag = bool(decision.requires_rag and decision.requires_document)
    if intent in {
        IntentType.GREETING, IntentType.AUDIO_CHECK, IntentType.CASUAL_CONVERSATION,
        IntentType.ACKNOWLEDGMENT, IntentType.WAIT_FOR_USER, IntentType.READY_TO_CONTINUE,
        IntentType.CONTINUE_RESPONSE, IntentType.REPEAT_RESPONSE, IntentType.CLARIFY_RESPONSE,
        IntentType.APPLICATION_HELP, IntentType.OUT_OF_SCOPE, IntentType.AMBIGUOUS_QUERY,
    }:
        requires_rag = False
    return IntentClassification(
        intent=intent,
        confidence=decision.intent_confidence,
        page_numbers=((decision.referenced_page,) if intent is IntentType.PAGE_QUERY and decision.referenced_page else ()),
        requires_retrieval=requires_rag,
        retrieval_strategy="DIRECT_PAGE" if intent is IntentType.PAGE_QUERY else ("HYBRID" if requires_rag else "NONE"),
        requires_evaluation=bool(decision.requires_evaluation and requires_rag),
        referenced_page=decision.referenced_page,
        referenced_topic=decision.referenced_topic,
        response_action=decision.conversation_action or ("RUN_DOCUMENT_QUERY" if requires_rag else "RESPOND_CONVERSATIONALLY"),
        response_language=decision.response_language or response_language,
        active_document_id=decision.active_document_id or active_document_id,
        conversation_action=decision.conversation_action,
        context_reference=decision.context_reference,
    )


def classify_intent_semantically(
    text: str,
    *,
    api_key: str,
    conversation_context: str | None = None,
    active_document_id: str | None = None,
    response_language: str | None = None,
) -> IntentClassification | None:
    """Classify meaning with the configured LLM and structured output.

    Failures return ``None`` so the caller can use the conservative local
    classifier without making the voice path unavailable.
    """
    if not api_key or not str(api_key).strip():
        return None
    try:
        chain = (
            ChatPromptTemplate.from_template(_SEMANTIC_INTENT_PROMPT)
            | get_llm(str(api_key).strip()).with_structured_output(SemanticIntentDecision)
        )
        result = chain.invoke({
            "question": (text or "").strip(),
            "conversation_context": conversation_context or "(none)",
            "active_document_id": active_document_id or "(none)",
            "response_language": response_language or "en",
        })
        if not isinstance(result, SemanticIntentDecision):
            return None
        return _semantic_intent_to_classification(
            result,
            active_document_id=active_document_id,
            response_language=response_language,
        )
    except Exception:
        return None


def generate_conversational_response(
    text: str,
    *,
    api_key: str,
    conversation_context: str | None = None,
    response_language: str | None = None,
) -> str | None:
    """Generate a non-document response using the already configured LLM."""
    if not api_key or not str(api_key).strip():
        return None
    try:
        chain = (
            ChatPromptTemplate.from_template(_CONVERSATIONAL_RESPONSE_PROMPT)
            | get_llm(str(api_key).strip())
        )
        decision = chain.invoke({
            "question": (text or "").strip(),
            "conversation_context": conversation_context or "(none)",
            "response_language": response_language or "en",
        })
        content = getattr(decision, "content", decision)
        return str(content).strip() or None
    except Exception:
        return None


_PAGE_MARKERS = (
    "page", "pages", "p.",
    "पृष्ठ", "पेज", "ಪುಟ", "பக்கம்", "பக்க", "పేజీ", "പേജ്",
    "পৃষ্ঠা", "પાનું", "ਪੰਨਾ", "صفحہ",
)

_FOLLOW_UP_PATTERNS = (
    r"\b(tell me more|explain more|go deeper|what about that|what about it)\b",
    r"\bwhat about (the )?next page\b",
    r"\b(explain|describe)\s+(the\s+)?(second|third|next|that|this|it)\b",
    r"\b(which|what)\s+page\s+(mentions|contains|discusses)\s+(it|that|this)\b",
    r"\b(ಇದನ್ನು|ಅದನ್ನು|ಇದರ ಬಗ್ಗೆ|அதை|இதைக்|దాని గురించి|దాన్ని|उसके बारे में|उसे|તેના વિશે|اس کے بارے میں)\b",
)

_OVERVIEW_PATTERNS = (
    r"\bwhat is this (pdf|document) about\b",
    r"\bwhat(?:'s| is) in this (pdf|document)\b",
    r"\bwhat information does (this|the) (pdf|document) contain\b",
    r"\b(main topics|important points|key points|overview)\b",
    r"(यह|इस)\s+(पीडीएफ|दस्तावेज)\s+किस बारे में",
    r"(ಮುಖ್ಯ ವಿಷಯಗಳು|ಈ ಪಿಡಿಎಫ್|ಈ ದಾಖಲೆ).*(ಏನು|ಬಗ್ಗೆ)",
    r"(முக்கிய தலைப்புகள்|இந்த pdf|இந்த ஆவணம்).*(என்ன|பற்றி)",
    r"(ముఖ్య అంశాలు|ఈ pdf|ఈ పత్రం).*(ఏమిటి|గురించి)",
)

_SUMMARY_PATTERNS = (
    r"\b(summar(?:y|ize|ise)|brief summary|give me a summary)\b",
    r"\b(provide|give)\s+(a\s+)?(short\s+)?summary\b",
    r"\b(सारांश|संक्षेप में|ಸಾರಾಂಶ|சுருக்கம்|சுருக்கமாக|సారాంశం|ചുരുക്കം|সংক্ষিপ্তসার|સારાંશ|ਸੰਖੇਪ)\b",
)

_AUDIO_CHECK_PATTERNS = (
    r"^(can you hear me(?: clearly)?|do you hear me|are you listening)[!.? ]*$",
    r"^(क्या आप मुझे सुन सकते हैं|ನನ್ನ ಧ್ವನಿ ಕೇಳಿಸುತ್ತಿದೆಯೇ|என் குரல் கேட்கிறதா|నా మాట వినిపిస్తుందా)[!.? ]*$",
)
_ACKNOWLEDGMENT_PATTERNS = (
    r"^(thanks|thank you|okay|ok|that's good|that is good|good)[!.? ]*$",
    r"^(धन्यवाद|ठीक है|अच्छा|ಧನ್ಯವಾದ|ಸರಿ|நன்றி|சரி|ధన్యవాదాలు|సరే)[!.? ]*$",
)
_CONTINUE_PATTERNS = (
    r"\b(continue|keep going|continue speaking|go on)\b",
    r"(जारी रखें|ಮುಂದುವರಿಸಿ|தொடருங்கள்|కొనసాగించండి)",
)
_REPEAT_PATTERNS = (
    r"\b(explain that again|repeat that|say that again|repeat your answer)\b",
    r"(फिर से समझाइए|ಮತ್ತೆ ವಿವರಿಸಿ|மீண்டும் விளக்குங்கள்|మళ్లీ వివరించండి)",
)
_CLARIFY_PATTERNS = (
    r"^(what do you mean|which one|can you clarify|i don't understand)[!.? ]*$",
    r"(आपका मतलब क्या है|ನಿಮ್ಮ ಅರ್ಥವೇನು|நீங்கள் என்ன சொல்கிறீர்கள்|మీ ఉద్దేశం ఏమిటి)",
)
_WAIT_PATTERNS = (
    r"\b(wait|hold on|one moment|give me a second|take your time)\b",
    r"रुकिए|एक मिनट|समय लीजिए|ತಡೆಯಿರಿ|ಒಂದು ಕ್ಷಣ|காத்திருங்கள்|ஒரு நிமிடம்|ఆగండి|ఒక్క నిమిషం",
)
_READY_PATTERNS = (
    r"\b(i am ready|i'm ready|ready now|you can continue)\b",
    r"मैं तैयार हूँ|ನಾನು ಸಿದ್ಧ|நான் தயாராக இருக்கிறேன்|నేను సిద్ధంగా ఉన్నాను",
)


def _page_numbers(text: str) -> tuple[int, ...]:
    """Extract explicit page references only when a page marker is present."""
    numbers: list[int] = []
    marker_pattern = "|".join(re.escape(marker) for marker in _PAGE_MARKERS)
    for match in re.finditer(
        rf"(?:{marker_pattern})\s*(?:(?:number|no\.?|nos\.?)\s*)?((?:\d+)(?:\s*(?:,|and|&|و)\s*\d+)*)",
        text,
        flags=re.IGNORECASE,
    ):
        for raw_page in re.findall(r"\d+", match.group(1)):
            page = int(raw_page)
            if page > 0 and page not in numbers:
                numbers.append(page)
    return tuple(numbers)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_intent(
    text: str,
    *,
    conversation_context: str | None = None,
    response_language: str | None = None,
) -> IntentClassification:
    """Classify a message without retrieval, LLM calls, or document claims.

    Existing voice greeting and out-of-scope rules remain authoritative for
    those categories. Document-oriented rules are deliberately conservative:
    an uncertain factual question falls back to DOCUMENT_QUERY, while a
    context-free reference such as ``tell me more`` remains ambiguous.
    """
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    context_available = bool((conversation_context or "").strip())

    def non_rag(intent: IntentType, confidence: float, action: str) -> IntentClassification:
        return IntentClassification(
            intent, confidence, requires_retrieval=False,
            retrieval_strategy="NONE", requires_evaluation=False,
            response_action=action, response_language=response_language,
        )

    if _matches(_AUDIO_CHECK_PATTERNS, normalized):
        return non_rag(IntentType.AUDIO_CHECK, 0.99, "RESPOND_AUDIO_CHECK")
    if _matches(_ACKNOWLEDGMENT_PATTERNS, normalized):
        return non_rag(IntentType.ACKNOWLEDGMENT, 0.99, "RESPOND_ACKNOWLEDGMENT")
    if _matches(_CONTINUE_PATTERNS, normalized):
        return non_rag(IntentType.CONTINUE_RESPONSE, 0.98, "CONTINUE_LAST_RESPONSE")
    if _matches(_REPEAT_PATTERNS, normalized):
        return non_rag(IntentType.REPEAT_RESPONSE, 0.98, "REPEAT_LAST_RESPONSE")
    if _matches(_CLARIFY_PATTERNS, normalized):
        return non_rag(IntentType.CLARIFY_RESPONSE, 0.91, "REQUEST_CLARIFICATION")
    if _matches(_WAIT_PATTERNS, normalized):
        return IntentClassification(
            IntentType.WAIT_FOR_USER, 0.96, requires_retrieval=False,
            retrieval_strategy="NONE", requires_evaluation=False,
            response_action="PAUSE_RESPONSE", response_language=response_language,
            conversation_action="WAIT",
        )
    if _matches(_READY_PATTERNS, normalized):
        return IntentClassification(
            IntentType.READY_TO_CONTINUE, 0.96, requires_retrieval=False,
            retrieval_strategy="NONE", requires_evaluation=False,
            response_action="RESUME_LISTENING", response_language=response_language,
            conversation_action="RESUME",
        )

    voice_intent = classify_voice_intent(normalized)
    if voice_intent == "greeting":
        return non_rag(IntentType.GREETING, 0.99, "RESPOND_GREETING")
    if voice_intent == "application_help":
        return non_rag(IntentType.APPLICATION_HELP, 0.97, "RESPOND_APPLICATION_HELP")
    if voice_intent == "small_talk":
        # Preserve the legacy combined greeting/small-talk label for callers
        # that use it, while exposing the richer casual label for simple talk.
        legacy = not bool(re.fullmatch(r"how are you[!.? ]*", normalized))
        return non_rag(
            IntentType.GENERAL_CONVERSATION if legacy else IntentType.CASUAL_CONVERSATION,
            0.98, "RESPOND_CASUALLY",
        )
    if voice_intent == "out_of_scope":
        return non_rag(IntentType.OUT_OF_SCOPE, 0.96, "RESPOND_OUT_OF_SCOPE")

    pages = _page_numbers(normalized)
    if pages:
        return IntentClassification(IntentType.PAGE_QUERY, 0.97, pages, True, "DIRECT_PAGE", True, pages[0], None, "RUN_PAGE_QUERY", response_language)

    if _matches(_SUMMARY_PATTERNS, normalized):
        return IntentClassification(IntentType.DOCUMENT_SUMMARY, 0.95, (), True, "DOCUMENT_SUMMARY")

    if _matches(_OVERVIEW_PATTERNS, normalized):
        return IntentClassification(IntentType.DOCUMENT_OVERVIEW, 0.94, (), True, "DOCUMENT_OVERVIEW")

    if _matches(_FOLLOW_UP_PATTERNS, normalized):
        if context_available:
            return IntentClassification(IntentType.DOCUMENT_FOLLOWUP, 0.91, (), True, "HYBRID", True, None, None, "RUN_FOLLOWUP_QUERY", response_language)
        return non_rag(IntentType.AMBIGUOUS_QUERY, 0.88, "REQUEST_CLARIFICATION")

    # A bare deictic/reference question is ambiguous without prior context.
    if re.search(r"\b(that|this|it|there|the second one|the next topic)\b", normalized):
        if context_available:
            return IntentClassification(IntentType.DOCUMENT_FOLLOWUP, 0.84, (), True, "HYBRID", True, None, None, "RUN_FOLLOWUP_QUERY", response_language)
        return non_rag(IntentType.AMBIGUOUS_QUERY, 0.83, "REQUEST_CLARIFICATION")

    # Safe fallback for factual questions: the classifier does not assert that
    # the fact exists in the PDF and never fabricates evidence.
    return IntentClassification(IntentType.DOCUMENT_QUERY, 0.62, (), True, "HYBRID")


def classify_voice_intent(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in GREETING_PATTERNS):
        return "greeting"
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in APPLICATION_HELP_PATTERNS):
        return "application_help"
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in SMALL_TALK_PATTERNS):
        return "small_talk"
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"
    # Unknown factual requests remain document-bound by default.
    return "document"
