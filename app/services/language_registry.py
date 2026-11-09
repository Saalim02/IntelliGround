"""Authoritative application and provider language registry."""

LANGUAGE_REGISTRY = {
    "en": {"name": "English", "locale": "en-IN"},
    "kn": {"name": "Kannada", "locale": "kn-IN"},
    "hi": {"name": "Hindi", "locale": "hi-IN"},
    "ta": {"name": "Tamil", "locale": "ta-IN"},
    "te": {"name": "Telugu", "locale": "te-IN"},
    "ml": {"name": "Malayalam", "locale": "ml-IN"},
    "mr": {"name": "Marathi", "locale": "mr-IN"},
    "gu": {"name": "Gujarati", "locale": "gu-IN"},
    "bn": {"name": "Bengali", "locale": "bn-IN"},
    "pa": {"name": "Punjabi", "locale": "pa-IN"},
    # Retained in the application/text registry. Current verified Gnani voice
    # mappings do not include Urdu, so voice requests fail explicitly rather
    # than silently speaking English.
    "ur": {"name": "Urdu", "locale": "ur-IN"},
}

LANGUAGE_CODES = frozenset(LANGUAGE_REGISTRY)
LANGUAGE_NAMES = {code: item["name"] for code, item in LANGUAGE_REGISTRY.items()}

TEXT_MESSAGES = {
    "en": {"summary": "Grounded multilingual summary generated successfully.", "questions": "Grounded multilingual questions generated successfully.", "conversation": "Conversational response generated.", "success": "Answer generated successfully.", "review": "Answer generated, but additional human review is recommended.", "human_review": "Human review is required before this answer can be released.", "unknown": "An unrecognized evaluation decision was returned. Human review is required."},
    "hi": {"summary": "प्रमाणित बहुभाषी सारांश सफलतापूर्वक तैयार किया गया।", "questions": "प्रमाणित बहुभाषी प्रश्न सफलतापूर्वक तैयार किए गए।", "conversation": "बातचीत का उत्तर तैयार किया गया।", "success": "उत्तर सफलतापूर्वक तैयार किया गया।", "review": "उत्तर तैयार है, लेकिन अतिरिक्त मानवीय समीक्षा की सलाह दी जाती है।", "human_review": "यह उत्तर जारी करने से पहले मानवीय समीक्षा आवश्यक है।", "unknown": "अज्ञात मूल्यांकन निर्णय मिला। मानवीय समीक्षा आवश्यक है।"},
    "kn": {"summary": "ಆಧಾರಿತ ಬಹುಭಾಷಾ ಸಾರಾಂಶವನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ.", "questions": "ಆಧಾರಿತ ಬಹುಭಾಷಾ ಪ್ರಶ್ನೆಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ.", "conversation": "ಸಂಭಾಷಣೆಯ ಉತ್ತರವನ್ನು ರಚಿಸಲಾಗಿದೆ.", "success": "ಉತ್ತರವನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ.", "review": "ಉತ್ತರವನ್ನು ರಚಿಸಲಾಗಿದೆ, ಆದರೆ ಹೆಚ್ಚುವರಿ ಮಾನವ ಪರಿಶೀಲನೆ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ.", "human_review": "ಈ ಉತ್ತರವನ್ನು ಬಿಡುಗಡೆ ಮಾಡುವ ಮೊದಲು ಮಾನವ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ.", "unknown": "ಗುರುತಿಸಲಾಗದ ಮೌಲ್ಯಮಾಪನ ನಿರ್ಧಾರ ಬಂದಿದೆ. ಮಾನವ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ."},
    "ta": {"summary": "ஆதாரமுள்ள பன்மொழிச் சுருக்கம் வெற்றிகரமாக உருவாக்கப்பட்டது.", "questions": "ஆதாரமுள்ள பன்மொழிக் கேள்விகள் வெற்றிகரமாக உருவாக்கப்பட்டன.", "conversation": "உரையாடல் பதில் உருவாக்கப்பட்டது.", "success": "பதில் வெற்றிகரமாக உருவாக்கப்பட்டது.", "review": "பதில் உருவாக்கப்பட்டது; கூடுதல் மனித மதிப்பாய்வு பரிந்துரைக்கப்படுகிறது.", "human_review": "இந்தப் பதிலை வெளியிடுவதற்கு முன் மனித மதிப்பாய்வு தேவை.", "unknown": "அறியப்படாத மதிப்பீட்டு முடிவு கிடைத்தது. மனித மதிப்பாய்வு தேவை."},
    "te": {"summary": "ఆధారాలతో కూడిన బహుభాషా సారాంశం విజయవంతంగా రూపొందించబడింది.", "questions": "ఆధారాలతో కూడిన బహుభాషా ప్రశ్నలు విజయవంతంగా రూపొందించబడ్డాయి.", "conversation": "సంభాషణ సమాధానం రూపొందించబడింది.", "success": "సమాధానం విజయవంతంగా రూపొందించబడింది.", "review": "సమాధానం రూపొందించబడింది, కానీ అదనపు మానవ సమీక్ష సిఫార్సు చేయబడింది.", "human_review": "ఈ సమాధానాన్ని విడుదల చేయడానికి ముందు మానవ సమీక్ష అవసరం.", "unknown": "గుర్తించని మూల్యాంకన నిర్ణయం వచ్చింది. మానవ సమీక్ష అవసరం."},
    "ml": {"summary": "തെളിവ് അടിസ്ഥാനമാക്കിയ ബഹുഭാഷാ സംഗ്രഹം വിജയകരമായി തയ്യാറാക്കി.", "questions": "തെളിവ് അടിസ്ഥാനമാക്കിയ ബഹുഭാഷാ ചോദ്യങ്ങൾ വിജയകരമായി തയ്യാറാക്കി.", "conversation": "സംഭാഷണ മറുപടി തയ്യാറാക്കി.", "success": "മറുപടി വിജയകരമായി തയ്യാറാക്കി.", "review": "മറുപടി തയ്യാറാക്കി, എന്നാൽ അധിക മനുഷ്യപരിശോധന ശുപാർശ ചെയ്യുന്നു.", "human_review": "ഈ മറുപടി നൽകുന്നതിന് മുമ്പ് മനുഷ്യപരിശോധന ആവശ്യമാണ്.", "unknown": "തിരിച്ചറിയാത്ത മൂല്യനിർണയ തീരുമാനം ലഭിച്ചു. മനുഷ്യപരിശോധന ആവശ്യമാണ്."},
    "mr": {"summary": "पुराव्यावर आधारित बहुभाषिक सारांश यशस्वीपणे तयार करण्यात आला.", "questions": "पुराव्यावर आधारित बहुभाषिक प्रश्न यशस्वीपणे तयार करण्यात आले.", "conversation": "संवादाचे उत्तर तयार करण्यात आले.", "success": "उत्तर यशस्वीपणे तयार करण्यात आले.", "review": "उत्तर तयार करण्यात आले, परंतु अतिरिक्त मानवी पुनरावलोकनाची शिफारस आहे.", "human_review": "हे उत्तर देण्यापूर्वी मानवी पुनरावलोकन आवश्यक आहे.", "unknown": "अज्ञात मूल्यमापन निर्णय मिळाला. मानवी पुनरावलोकन आवश्यक आहे."},
    "gu": {"summary": "પુરાવા આધારિત બહુભાષી સારાંશ સફળતાપૂર્વક બનાવવામાં આવ્યો.", "questions": "પુરાવા આધારિત બહુભાષી પ્રશ્નો સફળતાપૂર્વક બનાવવામાં આવ્યા.", "conversation": "વાતચીતનો જવાબ બનાવવામાં આવ્યો.", "success": "જવાબ સફળતાપૂર્વક બનાવવામાં આવ્યો.", "review": "જવાબ બનાવવામાં આવ્યો છે, પરંતુ વધારાની માનવીય સમીક્ષા ભલામણ કરવામાં આવે છે.", "human_review": "આ જવાબ આપવા પહેલાં માનવીય સમીક્ષા જરૂરી છે.", "unknown": "અજાણ્યો મૂલ્યાંકન નિર્ણય મળ્યો. માનવીય સમીક્ષા જરૂરી છે."},
    "bn": {"summary": "প্রমাণভিত্তিক বহুভাষিক সারাংশ সফলভাবে তৈরি হয়েছে।", "questions": "প্রমাণভিত্তিক বহুভাষিক প্রশ্ন সফলভাবে তৈরি হয়েছে।", "conversation": "কথোপকথনের উত্তর তৈরি হয়েছে।", "success": "উত্তর সফলভাবে তৈরি হয়েছে।", "review": "উত্তর তৈরি হয়েছে, তবে অতিরিক্ত মানবীয় পর্যালোচনার পরামর্শ দেওয়া হচ্ছে।", "human_review": "এই উত্তর প্রকাশের আগে মানবীয় পর্যালোচনা প্রয়োজন।", "unknown": "অজ্ঞাত মূল্যায়ন সিদ্ধান্ত পাওয়া গেছে। মানবীয় পর্যালোচনা প্রয়োজন।"},
    "pa": {"summary": "ਸਬੂਤ-ਅਧਾਰਿਤ ਬਹੁਭਾਸ਼ੀ ਸਾਰ ਸਫਲਤਾਪੂਰਵਕ ਤਿਆਰ ਕੀਤਾ ਗਿਆ।", "questions": "ਸਬੂਤ-ਅਧਾਰਿਤ ਬਹੁਭਾਸ਼ੀ ਸਵਾਲ ਸਫਲਤਾਪੂਰਵਕ ਤਿਆਰ ਕੀਤੇ ਗਏ।", "conversation": "ਗੱਲਬਾਤ ਦਾ ਜਵਾਬ ਤਿਆਰ ਕੀਤਾ ਗਿਆ।", "success": "ਜਵਾਬ ਸਫਲਤਾਪੂਰਵਕ ਤਿਆਰ ਕੀਤਾ ਗਿਆ।", "review": "ਜਵਾਬ ਤਿਆਰ ਕੀਤਾ ਗਿਆ ਹੈ, ਪਰ ਵਾਧੂ ਮਨੁੱਖੀ ਸਮੀਖਿਆ ਦੀ ਸਿਫ਼ਾਰਸ਼ ਹੈ।", "human_review": "ਇਹ ਜਵਾਬ ਜਾਰੀ ਕਰਨ ਤੋਂ ਪਹਿਲਾਂ ਮਨੁੱਖੀ ਸਮੀਖਿਆ ਲੋੜੀਂਦੀ ਹੈ।", "unknown": "ਅਣਪਛਾਤਾ ਮੁਲਾਂਕਣ ਫੈਸਲਾ ਮਿਲਿਆ। ਮਨੁੱਖੀ ਸਮੀਖਿਆ ਲੋੜੀਂਦੀ ਹੈ."},
    "ur": {"summary": "ثبوت پر مبنی کثیر لسانی خلاصہ کامیابی سے تیار کیا گیا۔", "questions": "ثبوت پر مبنی کثیر لسانی سوالات کامیابی سے تیار کیے گئے۔", "conversation": "گفتگو کا جواب تیار کیا گیا۔", "success": "جواب کامیابی سے تیار کیا گیا۔", "review": "جواب تیار کیا گیا ہے، لیکن اضافی انسانی جائزے کی سفارش ہے۔", "human_review": "یہ جواب جاری کرنے سے پہلے انسانی جائزہ ضروری ہے۔", "unknown": "نامعلوم تشخیصی فیصلہ موصول ہوا۔ انسانی جائزہ ضروری ہے۔"},
}


def localized_text_message(code: str | None, key: str) -> str:
    language = (code or "en").strip().lower()
    return TEXT_MESSAGES.get(language, TEXT_MESSAGES["en"]).get(key, TEXT_MESSAGES["en"]["success"])

# These are script families, not language detectors.  Hindi and Marathi both
# use Devanagari, while Urdu uses an Arabic-derived script and Punjabi uses
# Gurmukhi.  Keeping this mapping here prevents each generator from inventing
# a different language/script policy.
LANGUAGE_SCRIPT_GROUPS = {
    "en": "latin",
    "hi": "devanagari",
    "mr": "devanagari",
    "bn": "bengali",
    "gu": "gujarati",
    "pa": "gurmukhi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "ur": "arabic",
}

SCRIPT_RANGES = {
    "latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
    "devanagari": ((0x0900, 0x097F),),
    "bengali": ((0x0980, 0x09FF),),
    "gujarati": ((0x0A80, 0x0AFF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
    "tamil": ((0x0B80, 0x0BFF),),
    "telugu": ((0x0C00, 0x0C7F),),
    "kannada": ((0x0C80, 0x0CFF),),
    "malayalam": ((0x0D00, 0x0D7F),),
    "arabic": ((0x0600, 0x06FF),),
}


def language_script_group(code: str) -> str:
    """Return the canonical script family for a registered text language."""
    return LANGUAGE_SCRIPT_GROUPS[validate_language(code)]


def is_language_supported_text(text: str, code: str) -> bool:
    """Validate generated text without confusing script with language.

    Latin technical terms and punctuation are neutral in Indic output.  The
    validator therefore requires meaningful target-script content rather than
    requiring every character to belong to that script.  Hindi/Marathi share
    a script, so this check deliberately validates script compatibility only;
    semantic language selection remains the provider's responsibility.
    """
    language = validate_language(code)
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    group = LANGUAGE_SCRIPT_GROUPS[language]
    ranges = SCRIPT_RANGES[group]

    def in_group(char: str) -> bool:
        point = ord(char)
        return any(start <= point <= end for start, end in ranges)

    target_count = sum(in_group(char) for char in letters)
    if group == "latin":
        return target_count >= 3 and target_count / len(letters) >= 0.40

    # Latin technical terms, acronyms, product names, dates, and numbers are
    # neutral in Indic output. Do not let a long ``API/PDF/NREGA`` token make
    # otherwise valid selected-language prose fail its script check. Other
    # non-target scripts remain disallowed, and target-script prose is still
    # required so an English-only answer cannot pass for an Indic language.
    neutral_latin = sum(
        (0x0041 <= ord(char) <= 0x005A) or (0x0061 <= ord(char) <= 0x007A)
        for char in letters
    )
    non_neutral_letters = len(letters) - neutral_latin
    if target_count == 0:
        return False
    if len(letters) <= 12:
        minimum = 1
    else:
        minimum = max(2, min(8, round(non_neutral_letters * 0.20)))
    return target_count >= minimum and (
        target_count / max(1, non_neutral_letters) >= 0.20
        or target_count >= 8
    )

# Provider-confirmed current Prisma/Timbre languages. Urdu is intentionally
# absent until the configured provider account/model confirms it.
GNANI_VOICE_LANGUAGE_CODES = {
    code: item["locale"] for code, item in LANGUAGE_REGISTRY.items() if code != "ur"
}

# A locale mapping is configuration, not proof that the provider account and
# selected model can actually transcribe or synthesize that language.  Keep
# this distinction explicit so callers never silently substitute English.
PROVIDER_CAPABILITIES = {
    "prisma_stt": {
        **{code: "unverified" for code in GNANI_VOICE_LANGUAGE_CODES},
        "ur": "unsupported",
    },
    "timbre_tts": {
        **{code: "unverified" for code in GNANI_VOICE_LANGUAGE_CODES},
        "ur": "unsupported",
    },
}


def provider_capability(provider: str, code: str | None) -> str:
    normalized = (code or "").strip().lower()
    return PROVIDER_CAPABILITIES.get(provider, {}).get(normalized, "unsupported")


def validate_language(code: str | None, *, voice: bool = False) -> str:
    normalized = (code or "").strip().lower()
    allowed = GNANI_VOICE_LANGUAGE_CODES if voice else LANGUAGE_REGISTRY
    if normalized not in allowed:
        scope = "voice" if voice else "text"
        raise ValueError(f"Unsupported {scope} language '{code}'.")
    return normalized
