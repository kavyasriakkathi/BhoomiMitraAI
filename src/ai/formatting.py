"""
BhoomiMitra AI — Multi-Intent Response Formatter & Authoritative Deduplicator

Optimizes WhatsApp responses when a farmer asks multiple questions in a single query.
Organizes answers into clear, farmer-friendly sections (Crop Advice, Weather, Shops, Market, Schemes, Escalation),
removes repetitive greetings and introductions, selects exactly ONE authoritative response per detected intent,
and compacts information for mobile readability across 13 Indian languages.
"""
import re
from typing import Dict, Optional, Tuple, List

# Section Header Emojis & Titles across 13 languages
SECTION_HEADERS_BY_LANG: Dict[str, Dict[str, str]] = {
    "te": {
        "crop_advice": "🌱 *పంట సలహా*",
        "weather": "🌡️ *వాతావరణ సమాచారం*",
        "shop": "🏬 *సమీప వ్యవసాయ దుకాణాలు*",
        "market": "📊 *మార్కెట్ ధరలు*",
        "schemes": "🏛️ *ప్రభుత్వ పథకాలు*",
        "escalation": "👨‍🌾 *వ్యవసాయ అధికారి సంప్రదింపు*",
    },
    "hi": {
        "crop_advice": "🌱 *फसल सलाह*",
        "weather": "🌡️ *मौसम जानकारी*",
        "shop": "🏬 *नजदीकी कृषि दुकानें*",
        "market": "📊 *मंडी भाव*",
        "schemes": "🏛️ *सरकारी योजनाएं*",
        "escalation": "👨‍🌾 *कृषि अधिकारी संपर्क*",
    },
    "en": {
        "crop_advice": "🌱 *Crop Advice*",
        "weather": "🌡️ *Weather Information*",
        "shop": "🏬 *Nearby Shops & Availability*",
        "market": "📊 *Market Prices*",
        "schemes": "🏛️ *Government Schemes*",
        "escalation": "👨‍🌾 *Krishi Officer Escalation*",
    },
    "ta": {
        "crop_advice": "🌱 *பயிர் ஆலோசனை*",
        "weather": "🌡️ *வானிலை தகவல்*",
        "shop": "🏬 *அருகிலுள்ள கடைகள்*",
        "market": "📊 *சந்தை விலைகள்*",
        "schemes": "🏛️ *அரசு திட்டங்கள்*",
        "escalation": "👨‍🌾 *வேளாண் அலுவலர் தொடர்பு*",
    },
    "kn": {
        "crop_advice": "🌱 *ಬೆಳೆ ಸಲಹೆ*",
        "weather": "🌡️ *ಹವಾಮಾನ ಮಾಹಿತಿ*",
        "shop": "🏬 *ಹತ್ತಿರದ ಅಂಗಡಿಗಳು*",
        "market": "📊 *ಮಾರುಕಟ್ಟೆ ದರಗಳು*",
        "schemes": "🏛️ *ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು*",
        "escalation": "👨‍🌾 *ಕೃಷಿ ಅಧಿಕಾರಿ ಸಂಪರ್ಕ*",
    },
    "ml": {
        "crop_advice": "🌱 *വിള ഉപദേശം*",
        "weather": "🌡️ *കാലാവസ്ഥാ വിവരം*",
        "shop": "🏬 *അടുത്തുള്ള കടകൾ*",
        "market": "📊 *വിപണി വിലകൾ*",
        "schemes": "🏛️ *സർക്കാർ പദ്ധതികൾ*",
        "escalation": "👨‍🌾 *കൃഷി ഓഫീസർ ബന്ധപ്പെടൽ*",
    },
    "mr": {
        "crop_advice": "🌱 *पीक सल्ला*",
        "weather": "🌡️ *हवामान माहिती*",
        "shop": "🏬 *जवळची कृषी दुकाने*",
        "market": "📊 *बाजारभाव*",
        "schemes": "🏛️ *शासकीय योजना*",
        "escalation": "👨‍🌾 *कृषी अधिकारी संपर्क*",
    },
    "bn": {
        "crop_advice": "🌱 *ফসল পরামর্শ*",
        "weather": "🌡️ *আবহাওয়ার তথ্য*",
        "shop": "🏬 *নিকটবর্তী দোকান*",
        "market": "📊 *বাজার দর*",
        "schemes": "🏛️ *সরকারি প্রকল্প*",
        "escalation": "👨‍🌾 *কৃষি কর্মকর্তা যোগাযোগ*",
    },
    "gu": {
        "crop_advice": "🌱 *પાકની સલાહ*",
        "weather": "🌡️ *હવામાન માહિતી*",
        "shop": "🏬 *નજીકની દુકાનો*",
        "market": "📊 *બજાર ભાવ*",
        "schemes": "🏛️ *સરકારી યોજનાઓ*",
        "escalation": "👨‍🌾 *કૃષિ અધિકારી સંપર્ક*",
    },
    "or": {
        "crop_advice": "🌱 *ଫସଲ ପରାମର୍ଶ*",
        "weather": "🌡️ *ପାଣିପାଗ ସୂଚନା*",
        "shop": "🏬 *ନିକଟସ୍ଥ ଦୋକାନ*",
        "market": "📊 *ବଜାର ଦର*",
        "schemes": "🏛️ *ସରକାରୀ ଯୋଜନା*",
        "escalation": "👨‍🌾 *କୃଷି ଅଧିକାରୀ ସମ୍ପର୍କ*",
    },
    "pa": {
        "crop_advice": "🌱 *ਫਸਲ ਸਲਾਹ*",
        "weather": "🌡️ *ਮੌਸਮ ਜਾਣਕਾਰੀ*",
        "shop": "🏬 *ਨੇੜਲੀਆਂ ਦੁਕਾਨਾਂ*",
        "market": "📊 *ਮੰਡੀ ਭਾਅ*",
        "schemes": "🏛️ *ਸਰਕਾਰੀ ਸਕੀਮਾਂ*",
        "escalation": "👨‍🌾 *ਖੇਤੀਬਾੜੀ ਅਧਿਕਾਰੀ ਸੰਪਰਕ*",
    },
    "as": {
        "crop_advice": "🌱 *শস্য পৰামৰ্শ*",
        "weather": "🌡️ *বতৰৰ তথ্য*",
        "shop": "🏬 *ওচৰৰ দোকান*",
        "market": "📊 *বজাৰ দৰ*",
        "schemes": "🏛️ *চৰকাৰী আঁচনি*",
        "escalation": "👨‍🌾 *কৃষি বিষয়া যোগাযোগ*",
    },
    "ur": {
        "crop_advice": "🌱 *فصل کا مشورہ*",
        "weather": "🌡️ *موسم کی معلومات*",
        "shop": "🏬 *قریبی دکانیں*",
        "market": "📊 *منڈی کے بھاؤ*",
        "schemes": "🏛️ *سرکاری اسکیمیں*",
        "escalation": "👨‍🌾 *زرعی افسر سے رابطہ*",
    },
}

SECTION_HEADERS_EN = SECTION_HEADERS_BY_LANG["en"]
SECTION_HEADERS_TE = SECTION_HEADERS_BY_LANG["te"]

# Unavailable/Fallback labels per section when requested intent has no active data
_UNAVAILABLE_LABELS_BY_LANG: Dict[str, Dict[str, str]] = {
    "te": {
        "weather": "ℹ️ ఈ ప్రాంతానికి ప్రస్తుతం వాతావరణ సమాచారం అందుబాటులో లేదు. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
        "shop": "ℹ️ ప్రస్తుతం ఈ ఉత్పత్తికి సమీప దుకాణాలు అందుబాటులో లేవు. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
        "market": "ℹ️ ఈ పంటకు మార్కెట్ ధరలు అందుబాటులో లేవు. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
        "schemes": "ℹ️ ప్రస్తుతం ఈ కేటగిరీలో పథకాలు అందుబాటులో లేవు. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
    },
    "hi": {
        "weather": "ℹ️ वर्तमान में मौसम की जानकारी उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।",
        "shop": "ℹ️ इस उत्पाद के लिए नजदीकी दुकानें उपलब्ध नहीं हैं। कृपया कुछ समय बाद पुनः प्रयास करें।",
        "market": "ℹ️ इस फसल के लिए मंडी भाव उपलब्ध नहीं हैं। कृपया कुछ समय बाद पुनः प्रयास करें।",
        "schemes": "ℹ️ वर्तमान में योजनाएं उपलब्ध नहीं हैं। कृपया कुछ समय बाद पुनः प्रयास करें।",
    },
    "en": {
        "weather": "ℹ️ Weather information is currently unavailable. Please try again after some time.",
        "shop": "ℹ️ Nearby shop information is currently unavailable. Please try again after some time.",
        "market": "ℹ️ Market price information is currently unavailable. Please try again after some time.",
        "schemes": "ℹ️ Government scheme information is currently unavailable. Please try again after some time.",
    },
    "ta": {
        "weather": "ℹ️ தற்போது வானிலை தகவல் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
        "shop": "ℹ️ தற்போது அருகிலுள்ள கடைகள் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
        "market": "ℹ️ இந்த பயிருக்கான சந்தை விலை கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
        "schemes": "ℹ️ தற்போது திட்டங்கள் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    },
    "kn": {
        "weather": "ℹ️ ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "shop": "ℹ️ ಪ್ರಸ್ತುತ ಹತ್ತಿರದ ಅಂಗಡಿಗಳು ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "market": "ℹ️ ಈ ಬೆಳೆಗೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "schemes": "ℹ️ ಪ್ರಸ್ತುತ ಯೋಜನೆಗಳು ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    },
    "ml": {
        "weather": "ℹ️ നിലവിൽ കാലാവസ്ഥാ വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
        "shop": "ℹ️ നിലവിൽ അടുത്തുള്ള കടകൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
        "market": "ℹ️ ഈ വിളയ്ക്കുള്ള വിപണി വില ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
        "schemes": "ℹ️ നിലവിൽ പദ്ധതികൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    },
    "mr": {
        "weather": "ℹ️ सध्या हवामानाची माहिती उपलब्ध नाही. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
        "shop": "ℹ️ या उत्पादनासाठी जवळची दुकाने उपलब्ध नाहीत. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
        "market": "ℹ️ या पिकासाठी बाजारभाव उपलब्ध नाहीत. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
        "schemes": "ℹ️ सध्या योजना उपलब्ध नाहीत. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
    },
    "bn": {
        "weather": "ℹ️ বর্তমানে আবহাওয়ার তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
        "shop": "ℹ️ নিকটবর্তী দোকান তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
        "market": "ℹ️ এই ফসলের জন্য বাজার দর উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
        "schemes": "ℹ️ বর্তমানে প্রকল্প তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
    },
    "gu": {
        "weather": "ℹ️ હાલમાં હવામાન માહિતી ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
        "shop": "ℹ️ નજીકની દુકાનો ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
        "market": "ℹ️ આ પાક માટે બજાર ભાવ ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
        "schemes": "ℹ️ હાલમાં યોજનાઓ ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
    },
    "or": {
        "weather": "ℹ️ ବର୍ତ୍ତମାନ ପାଣିପାଗ ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
        "shop": "ℹ️ ନିକଟସ୍ଥ ଦୋକାନ ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
        "market": "ℹ️ ଏହି ଫସଲ ପାଇଁ ବଜାର ଦର ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
        "schemes": "ℹ️ ବର୍ତ୍ତମାନ ଯୋଜନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
    },
    "pa": {
        "weather": "ℹ️ ਇਸ ਵੇਲੇ ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "shop": "ℹ️ ਨੇੜਲੀਆਂ ਦੁਕਾਨਾਂ ਉਪਲਬਧ ਨਹੀਂ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "market": "ℹ️ ਇਸ ਫਸਲ ਲਈ ਮੰਡੀ ਭਾਅ ਉਪਲਬਧ ਨਹੀਂ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "schemes": "ℹ️ ਇਸ ਵੇਲੇ ਸਕੀਮਾਂ ਉਪਲਬਧ ਨਹੀਂ ਹਨ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    },
    "as": {
        "weather": "ℹ️ বৰ্তমান বতৰৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
        "shop": "ℹ️ ওচৰৰ দোকানৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
        "market": "ℹ️ এই শস্যৰ বাবে বজাৰ দৰ উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
        "schemes": "ℹ️ বৰ্তমান আঁচনিৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
    },
    "ur": {
        "weather": "ℹ️ فی الوقت موسم کی معلومات دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
        "shop": "ℹ️ فی الوقت قریبی دکانیں دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
        "market": "ℹ️ اس فصل کے لیے منڈی کے بھاؤ دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
        "schemes": "ℹ️ فی الوقت اسکیمیں دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
    },
}

_UNAVAILABLE_LABELS_EN = _UNAVAILABLE_LABELS_BY_LANG["en"]
_UNAVAILABLE_LABELS_TE = _UNAVAILABLE_LABELS_BY_LANG["te"]


# Introductory filler patterns to strip from the beginning of text/sections
_INTRO_PATTERNS = [
    r"^(?:hello|hi|hey|namaste|namaskar|vanakkam|namaskara|greetings)(?:\s+(?:farmer|friend|brother|farmer brother|kisan))?[!,\.\s\-]+",
    r"^(?:నమస్తే|హలో|నమస్కారం|నమస్కారాలు|नमस्ते|नमस्कार|வணக்கம்|ನಮಸ್ಕಾರ)(?:\s+(?:రైతు సోదరా|రైతు మిత్రమా|రైతు అన్న|రైతు|మిత్రమా|అన్న|సోదరా|किसान भाई|शेतकरी))?[!,\.\s\-]+",
    r"^(?:i am|my name is)\s+bhoomimitra(?:\s+ai)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^నేను\s+భూమిమిత్ర(?:\s+ai)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^मैं\s+भूमिमित्र(?:\s+ai)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^here (?:is|are) the (?:details|answers|information)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^మీరు అడిగిన (?:సమాచారం|వివరాలు|సలహాలు)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^ఇక్కడ సమాచారం ఉంది(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^sure[,\s]+i can help you with (?:that|your questions)(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^ఖచ్చితంగా[,\s]+నేను మీకు సహాయం చేస్తాను(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^మీ పంటకు సంబంధించి[,\s]+(?:సమాచారం|వివరాలు)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
    r"^regarding your crop[,\s]+(?:information)?(?:[^\n\.\!\:]*[\:\.\!])?\s*",
]

_COMPILED_INTROS = [re.compile(p, re.IGNORECASE) for p in _INTRO_PATTERNS]

# Outro/trailing questions filler patterns to strip from the end of crop advice
_OUTRO_PATTERNS = [
    r"(?:మీకు ఇంకా ఏమైనా సహాయం కావాలా\??\s*|మీకు ఏమైనా సందేహాలు ఉంటే అడగండి[\.\?]?\s*|ఇంకా ఏదైనా సమాచారం కావాలంటే అడగండి[\.\?]?\s*|ధన్యవాదాలు[\.\!]?\s*|రైతే రాజు[\.\!]?\s*|శుభం[\.\!]?\s*)$",
    r"(?:आपको और कोई सहायता चाहिए\??\s*|धन्यवाद[\.\!]?\s*|जय जवान जय किसान[\.\!]?\s*)$",
    r"(?:do you need (?:any )?(?:further|more|other) assistance\??\s*|feel free to ask if you have (?:any )?questions[\.\?]?\s*|please let me know if you need anything else[\.\?]?\s*|let me know if you need anything else[\.\?]?\s*|thank you[\.\!]?\s*|thanks[\.\!]?\s*)$",
]

_COMPILED_OUTROS = [re.compile(p, re.IGNORECASE) for p in _OUTRO_PATTERNS]

# Keywords indicating pure agronomic/crop advice inquiry
_PURE_CROP_KEYWORDS_EN = [
    "spray", "disease", "pest", "fungus", "leaf", "rot", "spots", "dosage", "chemical",
    "pesticide", "bollworm", "alternaria", "control", "cure", "treatment", "prevent",
    "sowing", "stage", "cultivation", "water", "irrigate", "crop advice", "management",
    "how to", "symptoms", "deficiency", "fertilizer schedule", "what fertilizer", "which fertilizer",
    "how much fertilizer", "apply fertilizer", "blight", "blast", "rust", "wilt", "attack", "insects", "worms",
]

_PURE_CROP_KEYWORDS_TE = [
    "నివారణ", "తెగులు", "తెగుళ్ళు", "పురుగు", "పురుగులు", "ఆకు", "మచ్చలు", "మోతాదు",
    "పిచికారీ", "చికిత్స", "యాజమాన్యం", "సాగు", "లక్షణాలు", "ఎలా", "రోగం",
    "ఎరువుల మోతాదు", "మందు", "మందులు", "ఏం చేయాలి", "ఏమి చేయాలి", "రాలిపోవడం",
    "పచ్చదోమ", "తామర పురుగులు", "ఆల్టర్నేరియా", "అగ్గితెగులు", "ఎండిపోవడం", "పల్లాకు",
    "బూడిద తెగులు", "ఎరువు వాడాలి", "ఎరువులు వాడాలి", "పంట సలహా", "మందు పిచికారీ",
]

_PURE_BUY_PHRASES = [
    "where to buy", "where can i buy", "where i can buy", "shops near", "stores near",
    "dealer", "dealers", "buy urea", "buy dap", "buy pesticide", "buy seeds", "buy fertilizer",
    "కొనాలి", "ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "దుకాణం", "దుకాణాలు", "షాపు", "షాపులు",
    "दुकान", "कहाँ मिलेगा", "कहाँ से खरीदें", "दुकानें", "எங்கு கிடைக்கும்", "ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ",
]

_WEATHER_PHRASES_EN = [
    "weather", "forecast", "rain", "raining", "rainy", "temperature",
    "wind", "humidity", "climate", "degree", "hot", "cold", "will it rain",
]

_WEATHER_PHRASES_TE = [
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "కురుస్తుందా", "పడుతుందా",
    "ఉష్ణోగ్రత", "గాలి", "తేమ", "ఎండ", "చలి", "వాతావరణ అంచనా", "మంచు",
    "मौसम", "बारिश", "तापमान", "हवामान", "पाऊस", "வானிலை", "மழை", "ಹವಾಮಾನ", "ಮಳೆ",
]

_MARKET_PHRASES_EN = [
    "market price", "mandi price", "price of", "prices of", "quintal",
    "market rate", "mandi rate", "selling price", "rate per quintal",
]

_MARKET_PHRASES_TE = [
    "మార్కెట్ ధర", "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "క్వింటాల్", "క్వింటాలు",
    "ధర ఎంత", "రేటు ఎంత", "మార్కెట్లో", "అమ్ముకోవాలి", "గిట్టుబాటు ధర",
    "मंडी भाव", "बाजार भाव", "मंडी रेट", "भाव कितना", "சந்தை விலை", "மண்டி விலை", "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ", "ಮಂಡಿ ದರ",
]

_SCHEME_PHRASES_EN = [
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "kisan",
    "fasal bima", "insurance", "credit", "kcc", "solar pump", "government",
    "pm kisan", "rythu bandhu", "rythu bharosa", "kusum", "pmfby",
]

_SCHEME_PHRASES_TE = [
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "పంట బీమా", "యోజన",
    "కిసాన్", "ప్రభుత్వ", "రైతు బంధు", "రైతు భరోసా", "అర్హత", "ప్రయోజనాలు",
    "క్రెడిట్ కార్డ్", "సౌర పంప్", "ఆర్థిక సహాయం", "గ్రాంట్",
    "योजना", "सरकारी योजना", "सब्सिडी", "अनुदान", "திட்டம்", "மானியம்", "ಯೋಜನೆ",
]

_ESCALATION_PHRASES_EN = [
    "officer", "human", "expert", "scientist", "agent", "call", "talk", "escalation",
]

_ESCALATION_PHRASES_TE = [
    "అధికారి", "వ్యవసాయ అధికారి", "శాస్త్రవేత్త", "సంప్రదించండి", "హెల్ప్‌లైన్", "మాట్లాడాలి",
    "कृषि अधिकारी", "अधिकारी", "विशेषज्ञ", "அலுவலர்", "ಅಧಿಕಾರಿ",
]


def detect_user_intents(user_message: str) -> Dict[str, bool]:
    """
    Analyze farmer user message to detect all requested domains/intents.
    """
    msg_lower = user_message.lower()

    # 1. Weather Intent
    has_weather = any(p in msg_lower for p in _WEATHER_PHRASES_EN) or any(p in user_message for p in _WEATHER_PHRASES_TE)

    # 2. Shop Intent
    has_shop = any(p in msg_lower for p in _PURE_BUY_PHRASES) or any(p in user_message for p in ["ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "కొనాలి", "దుకాణం", "షాపు", "दुकान", "कहाँ मिलेगा", "எங்கு கிடைக்கும்", "ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ"])

    # 3. Market Intent
    has_market = (
        any(p in msg_lower for p in _MARKET_PHRASES_EN)
        or any(p in user_message for p in _MARKET_PHRASES_TE)
    )
    if not has_market:
        has_price_word = any(w in msg_lower for w in ["price", "rate", "mandi", "bhav", "dam", "vilai", "bele"]) or any(w in user_message for w in ["ధర", "రేటు", "भाव", "दर", "দাম", "விலை", "ಬೆಲೆ"])
        if has_price_word and not has_shop:
            has_market = True

    # 4. Schemes Intent
    has_schemes = any(p in msg_lower for p in _SCHEME_PHRASES_EN) or any(p in user_message for p in _SCHEME_PHRASES_TE)
    # 5. Escalation Intent
    has_escalation = any(p in msg_lower for p in _ESCALATION_PHRASES_EN) or any(p in user_message for p in _ESCALATION_PHRASES_TE)

    # 6. Crop Advice Intent
    has_agri_kw = any(kw in msg_lower for kw in _PURE_CROP_KEYWORDS_EN) or any(kw in user_message for kw in _PURE_CROP_KEYWORDS_TE)
    is_pure_non_crop = (has_shop or has_weather or has_market or has_schemes) and not has_agri_kw
    has_crop_advice = has_agri_kw and not (is_pure_non_crop and not has_agri_kw)

    if not any([has_weather, has_shop, has_market, has_schemes, has_escalation]):
        has_crop_advice = True

    return {
        "crop_advice": has_crop_advice,
        "weather": has_weather,
        "shop": has_shop,
        "market": has_market,
        "schemes": has_schemes,
        "escalation": has_escalation,
    }


def clean_introductions(text: str) -> str:
    """Remove repetitive introductory greetings or preamble from text."""
    if not text:
        return ""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _COMPILED_INTROS:
            new_text = pattern.sub("", cleaned).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned.strip()


def clean_outros(text: str) -> str:
    """Remove repetitive trailing questions or closings from text."""
    if not text:
        return ""
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        for pattern in _COMPILED_OUTROS:
            new_text = pattern.sub("", cleaned).strip()
            if new_text != cleaned:
                cleaned = new_text
                changed = True
    return cleaned.strip()


def clean_crop_advice_for_multi_intent(ai_text: str) -> str:
    """
    Clean speculative weather, market price, shop ads, scheme summaries, or refusal statements from primary AI text
    so it cleanly contains pure agronomic advisory when specialized structured enrichments exist.
    """
    if not ai_text:
        return ""

    refusal_markers = [
        "e-nam", "ఈ-నామ్", "ఈ - నామ్", "మార్కెట్ యార్డ్", "మార్కెట్ యార్డు",
        "market yard", "కేవలం వ్యవసాయం", "విషయాలపై మాత్రమే", "i can only help with farming",
        "only help with farming", "how can i help with your crops",
        "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో", "i'm sorry, i'm having trouble connecting",
    ]
    ai_lower = ai_text.lower()
    for marker in refusal_markers:
        if marker in ai_lower and len(ai_text.strip().split("\n")) <= 2:
            return ""

    speculative_sentence_markers = [
        "ధర", "ధరలు", "క్వింటాల్", "క్వింటాలు", "రేటు", "రేట్లు",
        "price", "prices", "mandi", "rate", "rates", "quintal", "मंडी भाव", "बाजार भाव",
        "వాతావరణం విషయానికి వస్తే", "వాతావరణం గురించి", "regarding weather",
        "as for the weather", "weather forecast shows", "వర్షం పడే అవకాశం",
        "will rain", "rain expected", "ఉష్ణోగ్రత", "weather is expected",
        "forecast", "degree", "weather", "मौसम", "बारिश",
        "సమీప డీలర్ల", "స్థానిక డీలర్ల", "దుకాణాల్లో దొరుకుతుంది", "దొరుకుతుంది",
        "you can buy from local", "available at nearby shops", "buy urea at", "where to buy",
        "లభిస్తుంది", "కొనుగోలు చేయవచ్చు", "లభ్యత", "దుకాణాల్లో", "డీలర్ల వద్ద", "दुकान",
        "ప్రభుత్వ పథకాలు", "పిఎం కిసాన్", "రైతు బంధు", "పథకం ద్వారా", "పథకం కింద",
        "government schemes", "pm kisan", "rythu bandhu", "subsidy is available",
        "scheme", "schemes", "yojana", "kisan samman", "सरकारी योजना",
    ]

    cleaned_paragraphs = []
    for para in ai_text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if para.startswith("⚠️ గమనిక:") or para.startswith("⚠️ Note:"):
            continue
        para = clean_outros(para)
        sentences = [s.strip() for s in re.split(r'(?<=[।\.\?\!])\s+', para) if s.strip()]
        valid_sentences = [
            s for s in sentences
            if not any(marker in s.lower() for marker in speculative_sentence_markers)
        ]
        if valid_sentences:
            cleaned_p = clean_outros(" ".join(valid_sentences))
            if cleaned_p:
                cleaned_paragraphs.append(cleaned_p)

    result = "\n".join(cleaned_paragraphs).strip()
    result = clean_introductions(result)
    result = clean_outros(result)
    return result


def decompose_assembled_response(assembled_text: str) -> Dict[str, str]:
    """
    Parse an assembled response string into discrete functional sections:
    - crop_advice (base AI text)
    - shop (🏬 ...)
    - market (📊 ...)
    - weather (🌡️ ...)
    - schemes (🏛️ ...)
    - escalation (👨‍🌾 ...)
    """
    sections: Dict[str, str] = {
        "crop_advice": "",
        "weather": "",
        "shop": "",
        "market": "",
        "schemes": "",
        "escalation": "",
    }

    if not assembled_text:
        return sections

    pattern = r"(?=(?:^|\n\n)(?:🏬|📊|🌡️|🌦️|🌤️|🏛️|👨‍🌾|⚠️))"
    chunks = re.split(pattern, assembled_text.strip())

    base_ai_parts = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk.startswith("🏬") or "Available Nearby Shops" in chunk or "సమీప వ్యవసాయ దుకాణాలు" in chunk or "Nearby Agricultural Shops" in chunk or "नजदीकी कृषि दुकानें" in chunk:
            sections["shop"] = chunk
        elif chunk.startswith("📊") or "Mandi Prices" in chunk or "మార్కెట్ ధరలు" in chunk or "మండి ధరలు" in chunk or "मंडी भाव" in chunk:
            sections["market"] = chunk
        elif chunk.startswith("⚠️") and ("మార్కెట్ ధర" in chunk or "market price" in chunk.lower() or "మండి" in chunk or "mandi" in chunk.lower() or "मंडी भाव" in chunk):
            sections["market"] = chunk
        elif chunk.startswith("🌡️") or chunk.startswith("🌦️") or chunk.startswith("🌤️") or "Weather Information" in chunk or "వాతావరణ సమాచారం" in chunk or "मौसम जानकारी" in chunk:
            sections["weather"] = chunk
        elif chunk.startswith("🏛️") or "Government Schemes" in chunk or "ప్రభుత్వ పథకాలు" in chunk or "सरकारी योजनाएं" in chunk:
            sections["schemes"] = chunk
        elif chunk.startswith("👨‍🌾") or "Escalation Ticket" in chunk or "సంప్రదింపు టికెట్" in chunk:
            sections["escalation"] = chunk
        else:
            base_ai_parts.append(chunk)

    if base_ai_parts:
        sections["crop_advice"] = "\n\n".join(base_ai_parts).strip()

    return sections


def compact_section(
    section_key: str,
    section_text: str,
    language: str = "en",
) -> str:
    """
    Compact a single section by removing internal redundant headers and formatting cleanly.
    """
    headers = SECTION_HEADERS_BY_LANG.get(language, SECTION_HEADERS_EN)
    section_header = headers.get(section_key, "")

    if not section_text:
        unavail = _UNAVAILABLE_LABELS_BY_LANG.get(language, _UNAVAILABLE_LABELS_EN)
        msg = unavail.get(section_key, "")
        return f"{section_header}:\n{msg}" if msg else ""

    lines = [line.strip() for line in section_text.split("\n") if line.strip()]

    if section_key == "crop_advice":
        cleaned = clean_introductions(section_text)
        cleaned = clean_crop_advice_for_multi_intent(cleaned)
        cleaned = clean_outros(cleaned)
        if not cleaned:
            return ""
        return f"{section_header}\n{cleaned}"

    elif section_key == "weather":
        body_lines = []
        for line in lines:
            if (line.startswith("🌡️ ") or line.startswith("🌦️ ")) and ("Weather Information" in line or "వాతావరణ సమాచారం" in line or "मौसम" in line):
                loc_match = re.search(r"\((.*?)\)", line)
                loc_str = f" ({loc_match.group(1)})" if loc_match else ""
                default_title = "వాతావరణ సమాచారం" if language == "te" else ("मौसम जानकारी" if language == "hi" else "Weather Information")
                section_header = f"🌡️ *{default_title}*{loc_str}"
                continue
            if line.startswith("📡"):
                continue
            body_lines.append(line)
        return f"{section_header}\n" + "\n".join(body_lines)

    elif section_key == "market":
        body_lines = []
        for line in lines:
            if line.startswith("📊") or line.startswith("⚠️"):
                section_header = line
                continue
            if line.startswith("📡"):
                continue
            body_lines.append(line)
        return f"{section_header}\n" + "\n".join(body_lines)

    elif section_key == "shop":
        body_lines = []
        for line in lines:
            if line.startswith("🏬"):
                continue
            if line.startswith("ℹ️") or "Find all shops at:" in line or "మరిన్ని దుకాణాల కోసం:" in line:
                continue
            body_lines.append(line)
        return f"{section_header}:\n" + "\n".join(body_lines)

    elif section_key == "schemes":
        body_lines = []
        for line in lines:
            if line.startswith("🏛️"):
                continue
            if line.startswith("⚠️") or "See all schemes at:" in line or "మరిన్ని పథకాల కోసం:" in line:
                continue
            body_lines.append(line)
        return f"{section_header}:\n" + "\n".join(body_lines)

    elif section_key == "escalation":
        body_lines = []
        for line in lines:
            if line.startswith("━━━━━━━━━━━━━━━━━━━━━━"):
                continue
            body_lines.append(line)
        return "\n".join(body_lines)

    return section_text


def format_multi_intent_response(
    assembled_text: str,
    user_message: str = "",
    language: str = "en",
) -> str:
    """
    Main entry point for WhatsApp response optimization and authoritative deduplication.

    - If single-intent: returns the response untouched/preserved for backwards compatibility.
    - If multi-intent (2+ domains detected):
      1. Detects which intents the farmer actually asked about.
      2. Strictly retains only ONE authoritative response per requested intent.
      3. Discards unrequested sections and generic AI summaries for domains covered by specialized modules.
      4. Cleans repetitive intros and outros.
      5. Organizes in logical order: Crop Advice -> Weather -> Shops -> Market -> Schemes -> Escalation.
    """
    if not assembled_text:
        return ""

    user_intents = detect_user_intents(user_message) if user_message else {}
    requested_intent_count = sum(1 for k, v in user_intents.items() if v)

    sections = decompose_assembled_response(assembled_text)

    active_enrichments = [
        k for k in ["weather", "shop", "market", "schemes", "escalation"]
        if sections[k] and sections[k].strip()
    ]

    has_crop = bool(sections["crop_advice"] and sections["crop_advice"].strip())
    has_requested_crop = user_intents.get("crop_advice", False)

    is_multi_intent = False
    if requested_intent_count >= 2:
        is_multi_intent = True
    elif len(active_enrichments) >= 2:
        is_multi_intent = True
    elif len(active_enrichments) == 1 and has_crop and has_requested_crop:
        is_multi_intent = True

    if not is_multi_intent:
        return assembled_text.strip()

    ordered_keys = ["crop_advice", "weather", "shop", "market", "schemes", "escalation"]
    formatted_blocks: List[str] = []

    for key in ordered_keys:
        if user_intents and not user_intents.get(key, False):
            continue

        section_raw = sections.get(key, "")
        compacted = compact_section(key, section_raw, language=language)
        if compacted and compacted.strip():
            formatted_blocks.append(compacted.strip())

    if not formatted_blocks:
        return assembled_text.strip()

    final_output = "\n\n".join(formatted_blocks).strip()
    return final_output
