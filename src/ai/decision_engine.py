"""
BhoomiMitra AI — AI Decision Engine & Orchestration Layer

The central decision-making brain of BhoomiMitra AI.
Responsible for:
1. Multi-lingual intent classification (English, Telugu, Tanglish, Mixed).
2. Authoritative module routing (Market, Weather, Schemes, Shops, Advisory, Vision).
3. Anti-hallucination and agricultural safety enforcement.
4. Factual data guarantees (real data is preserved exactly; LLM never guesses prices/weather/dosages).
5. Single-intent and multi-intent response assembly with localized fallbacks.
6. Guarding greetings against unnecessary expensive enrichments.
"""
import re
import unicodedata
from enum import Enum
from typing import List, Dict, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import Farmer, Conversation
from src.core.logging import logger
from src.ai.prompts import (
    get_fallback_response,
    get_market_fallback_response,
    get_weather_fallback_response,
    get_schemes_fallback_response,
    get_shops_fallback_response,
)


class FarmerIntent(str, Enum):
    GREETING = "greeting"
    CROP_ADVICE = "crop_advice"
    CROP_HEALTH = "crop_health"  # disease / pest
    FERTILIZER = "fertilizer_nutrient"  # fertilizer / nutrient advice
    IRRIGATION = "irrigation"
    WEATHER = "weather"
    MARKET_PRICE = "market_price"
    GOVERNMENT_SCHEMES = "government_schemes"
    SHOPS = "shops_input_availability"
    SOWING = "sowing"
    HARVESTING = "harvesting"
    REMINDERS = "reminders"
    STOCK_ALERT = "stock_alert"
    IMAGE_DIAGNOSIS = "image_diagnosis"
    GENERAL_FARMING = "general_farming"
    UNKNOWN = "unsupported_unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Inten# ─────────────────────────────────────────────────────────────────────────────
# Intent Keyword & Pattern Mappings (13 Languages + Transliterated)
# ─────────────────────────────────────────────────────────────────────────────

GREETING_PATTERNS = [
    r"\b(?:hello|hi|hey|hai|helo|namaste|namaskar|namaskaram|namaskaralu|vanakkam|namaskara|nomoshkar|nomoskar|kem\s*cho|sat\s*sri\s*akal|satsriakal|salaam|salam|aadab)\b",
    r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|greetings)\b",
    r"^(?:నమస్తే|నమస్కారం|నమస్కారాలు|హలో|హాయ్|नमस्ते|नमस्कार|வணக்கம்|ನಮಸ್ಕಾರ|നമസ്കാരം|নমস্কার|নমস্কাৰ|કેમ\s*છો|ਸਤਿ\s*ਸ੍ਰੀ\s*ਅਕਾਲ|ନମସ୍କାର|سلام|آداب)[\s\!,\.]*$",
    r"^(?:హాయ్|నమస్తే|హలో|నమస్కారం|नमस्ते|नमस्कार|வணக்கம்|ನಮಸ್ಕಾರ|നമസ്കാരം|নমস্কার|নমস্কাৰ|નમસ્તે|ਸਤਿ\s*ਸ੍ਰੀ\s*ਅਕਾਲ|ନମସ୍କାର|سلام)\s+(?:భూమిమిత్ర|bhoomimitra|भूमिमित्र|பூமிமித்ரா|ಭೂಮಿಮಿತ್ರ|ഭൂമിമിത്ര|ভূমিমিত্র|ভূমিমিত্ৰ|ભૂમિમિત્ર|ਭੂਮੀਮਿੱਤਰ|ଭୂମିମିତ୍ର|بھومی\s*مترا|రైతు\s*మిత్ర|రైతు\s*అన్న|రైతు|किसान|farmer)?[\s\!,\.]*$",
]

GREETING_TOKENS = {
    "hi", "hello", "hey", "namaste", "namaskar", "namaskaram", "vanakkam", "namaskara",
    "nomoshkar", "nomoskar", "salam", "salaam", "aadab", "kemcho",
    "హాయ్", "నమస్తే", "హలో", "నమస్కారం",
    "नमस्ते", "नमस्कार", "प्रणाम", "राम", "जय",
    "வணக்கம்",
    "ನಮಸ್ಕಾರ",
    "നമസ്കാരം",
    "নমস্কার", "নমস্কাৰ",
    "કેમ", "છો", "નમસ્તે",
    "ਸਤਿ", "ਸ੍ਰੀ", "ਅਕਾਲ",
    "ନମସ୍କାର",
    "سلام", "آداب",
}

GREETING_RESPONSES = {
    "te": "నమస్తే! నేను మీ భూమిమిత్ర AI వ్యవసాయ సహాయకుడిని. 🙏\n\nపంట సలహాలు, తెగుళ్ల నివారణ, ఎరువుల సమాచారం, మార్కెట్ ధరలు మరియు వాతావరణ అంచనా కోసం నన్ను అడగవచ్చు. మీకు ఏ విధంగా సహాయపడగలను?",
    "hi": "नमस्ते! मैं आपका भूमिमित्र AI कृषि सहायक हूँ। 🙏\n\nआप मुझसे फसल सलाह, कीट व रोग नियंत्रण, खाद की जानकारी, मंडी भाव और मौसम पूर्वानुमान के बारे में पूछ सकते हैं। मैं आपकी क्या मदद कर सकता हूँ?",
    "en": "Hello! I am your BhoomiMitra AI farming assistant. 🙏\n\nYou can ask me about crop advisory, pest/disease management, fertilizer recommendations, mandi prices, and weather forecasts. How can I assist you today?",
    "ta": "வணக்கம்! நான் உங்கள் பூமிமித்ரா AI விவசாய உதவியாளர். 🙏\n\nபயிர் ஆலோசனைகள், பூச்சி கட்டுப்பாடு, உர மேலாண்மை, சந்தை விலைகள் மற்றும் வானிலை தகவல்களுக்கு என்னை கேட்கலாம். உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಭೂಮಿಮಿತ್ರ AI ಕೃಷಿ ಸಹಾಯಕ. 🙏\n\nಬೆಳೆ ಸಲಹೆ, ಕೀಟ ಮತ್ತು ರೋಗ ನಿಯಂತ್ರಣ, ಗೊಬ್ಬರ ಮಾಹಿತಿ, ಮಾರುಕಟ್ಟೆ ದರಗಳು ಮತ್ತು ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಗಾಗಿ ನೀವು ನನ್ನನ್ನು ಕೇಳಬಹುದು. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
    "ml": "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ ഭൂമിമിത്ര AI കാർഷിക സഹായിയാണ്. 🙏\n\nവിള ഉപദേശങ്ങൾ, കീടനിയന്ത്രണം, വളപ്രയോഗം, വിപണി വിലകൾ, കാലാവസ്ഥാ പ്രവചനം എന്നിവയെക്കുറിച്ച് നിങ്ങൾക്ക് എന്നോട് ചോദിക്കാം. ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം?",
    "mr": "नमस्कार! मी आपला भूमिमित्र AI शेती सल्लागार आहे. 🙏\n\nपीक सल्ला, कीड व रोग नियंत्रण, खत व्यवस्थापन, बाजारभाव आणि हवामान अंदाजाबाबत आपण मला विचारू शकता. मी आपली काय मदत करू शकतो?",
    "bn": "নমস্কার! আমি আপনার ভূমিমিত্র AI কৃষি সহকারী। 🙏\n\nফসল পরামর্শ, কীটপতঙ্গ নিয়ন্ত্রণ, সারের তথ্য, বাজার দর এবং আবহাওয়ার পূর্বাভাসের জন্য আপনি আমাকে জিজ্ঞাসা করতে পারেন। আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    "gu": "નમસ્તે! હું તમારો ભૂમિમિત્ર AI કૃષિ સહાયક છું. 🙏\n\nપાક સલાહ, જીવાત નિયંત્રણ, ખાતરની માહિતી, બજાર ભાવ અને હવામાનની આગાહી વિશે તમે મને પૂછી શકો છો. હું તમને કેવી રીતે મદદ કરી શકું?",
    "or": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର ଭୂମିମିତ୍ର AI କୃଷି ସହାୟକ। 🙏\n\nଫସଲ ପରାମର୍ଶ, କୀଟ ନିୟନ୍ତ୍ରଣ, ସାର ସୂଚନା, ମଣ୍ଡି ଦର ଏବଂ ପାଣିପାଗ ପୂର୍ବାନୁମାନ ବିଷୟରେ ଆପଣ ମୋତେ ପଚାରିପାରିବେ। ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
    "pa": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ ਭੂਮੀਮਿੱਤਰ AI ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ ਹਾਂ। 🙏\n\nਫ਼ਸਲ ਸਲਾਹ, ਕੀਟ ਰੋਕਥਾਮ, ਖਾਦ ਜਾਣਕਾਰੀ, ਮੰਡੀ ਭਾਅ ਅਤੇ ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ ਲਈ ਤੁਸੀਂ ਮੈਨੂੰ ਪੁੱਛ ਸਕਦੇ ਹੋ। ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
    "as": "নমস্কাৰ! মই আপোনাৰ ভূমিমিত্ৰ AI কৃষি সহায়ক। 🙏\n\nশস্যৰ পৰামৰ্শ, পোক-পৰুৱা নিয়ント্ৰণ, সাৰৰ তথ্য, বজাৰ দৰ আৰু বতৰৰ আগজাননীৰ বিষয়ে আপুনি মোক সুধিব পাৰে। মই আপোনাক কেনেকৈ সহায় কৰিব পাৰোঁ?",
    "ur": "سلام! میں آپ کا بھومی مترا AI زرعی معاون ہوں۔ 🙏\n\nآپ مجھ سے فصل کے مشورے، کیڑوں اور بیماریوں کی روک تھام، کھاد کی معلومات، منڈی کے بھاؤ اور موسم کی پیشن گوئی کے بارے میں پوچھ سکتے ہیں। میں آپ کی کیا مدد کر سکتا ہوں?",
}


# ─────────────────────────────────────────────────────────────────────────────
# 11. Stock Alert Keywords (Proactive Input Restock Notifications)
# ─────────────────────────────────────────────────────────────────────────────

STOCK_ALERT_KEYWORDS_EN = [
    "alert me when", "notify me when", "tell me when", "when in stock", "comes in stock",
    "when urea", "stock alert", "set a urea stock alert", "set stock alert",
    "stop urea alert", "cancel urea stock alert", "cancel stock alert", "stop alert", "cancel alert",
    "my stock alerts", "show my alerts", "active alerts", "what alerts do i have",
    "my alerts", "list my alerts", "turn off alert", "disable alert",
]

STOCK_ALERT_KEYWORDS_TE = [
    "స్టాక్లోకి వస్తే", "స్టాక్ లోకి వస్తే", "స్టాక్ వస్తే", "దొరికితే నాకు చెప్పండి", "అలర్ట్ పెట్టండి",
    "అందుబాటులోకి వస్తే చెప్పండి", "స్టాక్ అలర్ట్", "నోటిఫికేషన్ పెట్టండి",
    "స్టాక్లోకి", "స్టాక్ లోకి",
    "అలర్ట్ ఆపండి", "అలర్ట్స్ ఆపండి", "స్టాక్ అలర్ట్ రద్దు", "నోటిఫికేషన్ ఆపండి",
    "అలర్ట్ వద్దు", "అలర్ట్ క్యాన్సిల్", "స్టాక్ అలర్ట్ ఆపండి",
    "నా అలర్ట్స్", "నా స్టాక్ అలర్ట్స్", "యాక్టివ్ అలర్ట్స్", "ఏ అలర్ట్స్ ఉన్నాయి",
    "నా అలర్ట్ లు", "అలర్ట్స్ లిస్ట్",
]

STOCK_ALERT_KEYWORDS_MULTILINGUAL = [
    "स्टॉक में आए तो बताना", "स्टॉक अलर्ट", "उपलब्ध होने पर बताएं",
    "अलर्ट बंद करें", "मेरे अलर्ट", "ஸ்டாக் வந்தால் சொல்லுங்கள்", "அலர்ட் வைக்கவும்",
]

STOCK_ALERT_KEYWORDS_TANGLISH = [
    "stock vasthe cheppandi", "stock vachaka cheppandi", "dorikithe cheppandi",
    "alert pettandi", "urea alert", "stock alert", "alert aapandi", "alert apandi",
    "alert vaddu", "alert cancel cheyandi", "naa alerts", "na alerts", "my alerts enti",
]

INTENT_KEYWORDS: Dict[FarmerIntent, Dict[str, List[str]]] = {
    FarmerIntent.STOCK_ALERT: {
        "en": STOCK_ALERT_KEYWORDS_EN,
        "te": STOCK_ALERT_KEYWORDS_TE,
        "multilingual": STOCK_ALERT_KEYWORDS_MULTILINGUAL,
        "transliterated": STOCK_ALERT_KEYWORDS_TANGLISH,
    },
    FarmerIntent.REMINDERS: {
        "en": ["remind me", "set reminder", "schedule reminder", "alert me", "reminder"],
        "te": ["గుర్తు చేయండి", "రిమైండర్", "షెడ్యూల్", "గుర్తుపెట్టుకో"],
        "hi": ["याद दिलाना", "रिमाइंडर", "याद दिलाएं", "अलर्ट करें", "शेड्यूल"],
        "ta": ["நினைவூட்டு", "நினைவூட்டல்", "ரிமைண்டர்"],
        "kn": ["ನೆನಪಿಸಿ", "ನೆನಪಿನಲ್ಲಿಡಿ", "ರಿಮೈಂಡರ್"],
        "ml": ["ഓർമ്മിപ്പിക്കുക", "ഓർമ്മപ്പെടുത്തൽ", "റിമൈൻഡർ"],
        "mr": ["आठवण करा", "आठवण द्या", "स्मरण", "रिमाइंडर"],
        "bn": ["মনে করিয়ে দাও", "মনে করাবেন", "রিমাইন্ডার"],
        "gu": ["યાદ અપાવો", "યાદ રાખજો", "રીમાઇન્ડર"],
        "pa": ["ਯਾਦ ਕਰਵਾਓ", "ਚੇਤੇ ਕਰਵਾਓ", "ਰਿਮਾਈਂਡർ"],
        "or": ["ମନେ ପକାଇ ଦିଅନ୍ତୁ", "ମନେ ରଖନ୍ତୁ", "ରିମାଇଣ୍ଡର"],
        "as": ["মনত পেলাই দিব", "মনত পেলাওক", "ৰিমাইণ্ডাৰ"],
        "ur": ["یاد دلائیں", "یاد دہانی", "ریمائنڈر"],
        "transliterated": ["remind cheyandi", "gurthu cheyandi", "yaad dilana", "yaad dilaye", "set reminder", "schedule reminder"],
    },
    FarmerIntent.MARKET_PRICE: {
        "en": [
            "market price", "mandi price", "mandi rate", "market rate", "selling price",
            "rate per quintal", "cotton price", "cotton rate", "paddy price", "chilli price",
            "tomato price", "price of", "prices of", "how much price", "how much rate",
            "market value", "mandi rates", "market prices",
        ],
        "te": [
            "మార్కెట్ ధర", "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "ధర ఎంత", "రేటు ఎంత",
            "క్వింటాల్", "క్వింటాలు", "అమ్ముకోవాలి", "గిట్టుబాటు ధర", "మార్కెట్లో", "మండిలో",
            "పత్తి ధర", "మిర్చి ధర", "వరి ధర", "టమాటా ధర", "రేట్లు", "ధరలు",
        ],
        "hi": [
            "मंडी भाव", "मंडी रेट", "बाजार भाव", "फसल का भाव", "दाम कितना", "भाव क्या है",
            "भाव कितना", "क्विंटल", "बिक्री मूल्य", "मंडी दर", "रेट क्या है", "मंडी में",
        ],
        "ta": [
            "சந்தை விலை", "மண்டி விலை", "விலை என்ன", "விற்பனை விலை", "குவிண்டால்",
            "சந்தை நிலவரம்", "ரேட் என்ன", "சந்தையில்",
        ],
        "kn": [
            "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ", "ಮಂಡಿ ದರ", "ಬೆಲೆ ಎಷ್ಟು", "ದರ ಎಷ್ಟು", "ಕ್ವಿಂಟಾಲ್",
            "ಮಾರುಕಟ್ಟೆ ದರ", "ಮಾರಾಟ ಬೆಲೆ", "ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ",
        ],
        "ml": [
            "വിപണി വില", "മാർക്കറ്റ് വില", "വില എത്ര", "ക്വിന്റൽ",
            "മാർക്കറ്റ് നിരക്ക്", "വിൽപ്പന വില", "മാർക്കറ്റിൽ",
        ],
        "mr": [
            "बाजारभाव", "मंडी भाव", "बाजार दर", "भाव किती", "दर किती", "क्विंटल",
            "हमीभाव", "विक्री दर", "मार्केट भाव",
        ],
        "bn": [
            "বাজার দর", "মান্ডি দর", "দাম কত", "দর কত", "কুইন্টাল",
            "বিক্রয় মূল্য", "বাজারের দাম", "মান্ডিতে",
        ],
        "gu": [
            "બજાર ભાવ", "મંડી ભાવ", "ભાવ શું છે", "ભાવ કેટલો", "ક્વિન્ટલ",
            "બજાર દર", "વેચાણ કિંમત", "માર્કેટ ભાવ",
        ],
        "pa": [
            "ਮੰਡੀ ਭਾਅ", "ਬਾਜ਼ਾਰ ਭਾਅ", "ਰੇਟ ਕੀ ਹੈ", "ਭਾਅ ਕਿੰਨਾ", "ਕੁਇੰਟਲ",
            "ਮੰਡੀ ਰੇਟ", "ਵੇਚ ਮੁੱਲ", "ਮੰਡੀ ਵਿੱਚ",
        ],
        "or": [
            "ମଣ୍ଡି ଦର", "ବଜାର ଦର", "ଦର କେତେ", "ମୂଲ୍ୟ କେତେ", "କ୍ୱିଣ୍ଟାଲ",
            "ବିକ୍ରି ମୂଲ୍ୟ", "ମଣ୍ଡିରେ",
        ],
        "as": [
            "বজাৰ দৰ", "মণ্ডী দৰ", "দাম কিমান", "দৰ কিমান", "কুইণ্টল",
            "বিক্ৰী মূল্য", "মণ্ডীত",
        ],
        "ur": [
            "منڈی کا بھاؤ", "مارکیٹ ریٹ", "قیمت کیا ہے", "بھاؤ کتنا", "فی کوئنٹل",
            "منڈی ریٹ", "منڈی میں",
        ],
        "transliterated": [
            "rate entha", "dhara entha", "rate entho", "dhara entho", "cotton rate", "patti rate",
            "patti dhara", "mirchi rate", "mirapa rate", "tomato rate", "tamata rate", "mandi rate",
            "market lo", "mandi lo", "market price", "mandi price", "bajar rate", "ammukovali",
            "eeroju rate", "today rate", "rate ela undi", "dhara ela undi", "per quintal rate",
            "lo cotton rate", "lo patti rate", "patti rate entha", "cotton rate entha",
            "bhav kitna", "bhav kya hai", "mandi bhav", "bajarbhav", "dara kete", "bele eshtu",
            "vilai enna", "dam koto", "mandi bha", "qeemat",
        ],
    },
    FarmerIntent.WEATHER: {
        "en": [
            "weather", "forecast", "rain", "raining", "rainy", "temperature", "humidity",
            "wind", "storm", "thunderstorm", "will it rain", "precipitation", "cloudy", "sunny",
            "climate", "degrees",
        ],
        "te": [
            "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "వానలు", "కురుస్తుందా",
            "కురుస్తుంది", "ఉష్ణోగ్రత", "తేమ", "ఎండ", "చలి", "వర్షం పడుతుందా", "వర్షం పడుతుంది",
            "వాతావరణ అంచనా", "మంచు", "తుఫాను", "జల్లులు", "మేఘాలు",
        ],
        "hi": [
            "मौसम", "मौसम कैसा", "बारिश", "वर्षा", "बारिश होगी", "तापमान", "हवा",
            "आंधी", "तूफान", "बादल", "मौसम का हाल", "ठंड", "गर्मी",
        ],
        "ta": [
            "வானிலை", "மழை", "மழை பெய்யுமா", "வெப்பநிலை", "காற்று", "புயல்",
            "மேகமூட்டம்", "வானிலை அறிக்கை",
        ],
        "kn": [
            "ಹವಾಮಾನ", "ಮಳೆ", "ಮಳೆ ಬರುತ್ತಾ", "ತಾಪಮಾನ", "ಗಾಳಿ", "ಮಳೆ ಮುನ್ಸೂಚನೆ", "ಮೋಡ",
        ],
        "ml": [
            "കാലാവസ്ഥ", "മഴ", "മഴ പെയ്യുമോ", "താപനില", "കാറ്റ്", "ഇടിമിന്നൽ", "കാലാവസ്ഥ പ്രവചനം",
        ],
        "mr": [
            "हवामान", "पाऊस", "पाऊस पडेल का", "तापमान", "वादळ", "गारपीट", "हवामान अंदाज",
        ],
        "bn": [
            "আবহাওয়া", "বৃষ্টি", "বৃষ্টি হবে কি", "তাপমাত্রা", "ঝড়", "আবহাওয়ার পূর্বাভাস", "মেঘ",
        ],
        "gu": [
            "હવામાન", "વરસાદ", "વરસાદ પડશે", "તાપમાન", "પવન", "વાવાઝોડું", "હવામાન આગાહી",
        ],
        "pa": [
            "ਮੌਸਮ", "ਮੀਂਹ", "ਬਰਸਾਤ", "ਮੀਂਹ ਪਵੇਗਾ", "ਤਾਪਮਾਨ", "ਹਵਾ", "ਤੂਫ਼ਾਨ", "ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ",
        ],
        "or": [
            "ପାଣିପାଗ", "ବର୍ଷା", "ବର୍ଷା ହେବ କି", "ତାପମାତ୍ରା", "ପବନ", "ଝଡ଼", "ପାଣିପାଗ ସୂଚନା",
        ],
        "as": [
            "বতৰ", "বৰষুণ", "বৰষুণ হ'ব নেকি", "তাপমাত্ৰা", "ধুমুহা", "বতৰৰ আগজাননী",
        ],
        "ur": [
            "موسم", "بارش", "بارش ہوگی", "درجہ حرارت", "طوفان", "موسم کا حال", "ہوا",
        ],
        "transliterated": [
            "varsham", "varsham paduthunda", "varsham vasthunda", "vana vasthunda",
            "vana paduthunda", "weather ela undi", "eeroju varsham", "repu varsham",
            "eppudu paduthundi", "rain paduthunda", "rain vasthada", "temperature entha",
            "cloudy ga undi", "varsham padtada", "varsham padtundha", "rain padtundha",
            "varsham eppudu", "varsham ela", "barish hogi", "barish kab", "mazha", "paus",
            "varsad", "bristi", "meeh", "barsha", "boroxun", "weather",
        ],
    },
    FarmerIntent.GOVERNMENT_SCHEMES: {
        "en": [
            "scheme", "schemes", "subsidy", "subsidies", "yojana", "pm kisan", "rythu bandhu",
            "rythu bharosa", "crop insurance", "fasal bima", "kcc", "kisan credit",
            "solar pump subsidy", "government assistance", "grant", "subsidized",
        ],
        "te": [
            "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "రైతు బంధు", "రైతు భరోసా", "పీఎం కిసాన్",
            "పంట బీమా", "రుణమాఫీ", "ప్రభుత్వ సహాయం", "ప్రభుత్వ", "అర్హత", "ప్రయోజనాలు",
            "ప్రభుత్వ పథకాలు", "రైతు పథకాలు",
        ],
        "hi": [
            "योजना", "सब्सिडी", "पीएम किसान", "फसल बीमा", "सरकारी योजना", "अनुदान",
            "ऋण माफी", "किसान क्रेडिट कार्ड", "केसीसी", "कृषि योजना", "सरकारी सहायता",
        ],
        "ta": [
            "திட்டம்", "மானியம்", "பிஎம் கிசான்", "பயிர் காப்பீடு", "அரசு திட்டம்", "விவசாய கடன்",
        ],
        "kn": [
            "ಯೋಜನೆ", "ಸಬ್ಸಿಡಿ", "ಪಿಎಂ ಕಿಸಾನ್", "ಬೆಳೆ ವಿಮೆ", "ಸರ್ಕಾರಿ ಯೋಜನೆ", "ರೈತ ಯೋಜನೆ", "ಸಾಲ ಮನ್ನಾ",
        ],
        "ml": [
            "പദ്ധതി", "സബ്‌സിഡി", "പിഎം കിസാൻ", "വിള ഇൻഷുറൻസ്", "സർക്കാർ പദ്ധതി", "കൃഷി ധനസഹായം",
        ],
        "mr": [
            "योजना", "अनुदान", "सब्सिडी", "पीएम किसान", "पीक विमा", "शासकीय योजना", "कर्जमाफी", "शेतकरी योजना",
        ],
        "bn": [
            "প্রকল্প", "ভর্তুকি", "যোজনা", "পিএম কিষাণ", "ফসল বীমা", "সরকারি অনুদান", "কৃষক বন্ধু",
        ],
        "gu": [
            "યોજના", "સબસિડી", "પીએમ કિસાન", "પાક વીમો", "સરકારી સહાય", "ખેડૂત યોજના", "ધિરાણ",
        ],
        "pa": [
            "ਸਕੀਮ", "ਸਬਸਿਡੀ", "ਯੋਜਨਾ", "ਪੀਐਮ ਕਿਸਾਨ", "ਫ਼ਸਲ ਬੀਮਾ", "ਸਰਕਾਰੀ ਸਹਾਇਤਾ", "ਕਰਜ਼ਾ ਮੁਆਫ਼ੀ",
        ],
        "or": [
            "ଯୋଜନା", "ସବସିଡି", "ପିଏମ କିଷାନ", "ଫସଲ ବୀମା", "ସରକାରୀ ସହାୟତା", "କାଳିଆ ଯୋଜନା",
        ],
        "as": [
            "আঁচনি", "ৰেহাই", "যোজনা", "পিএম কিষাণ", "শস্য বীমা", "চৰকাৰী সাহায্য",
        ],
        "ur": [
            "اسکیم", "سبسڈی", "سرکاری اسکیم", "پی ایم کسان", "فصل بیمہ", "قرض معافی", "امداد",
        ],
        "transliterated": [
            "pathakam", "pathakalu", "prabhutva pathakalu", "rythu pathakalu", "subsidilu",
            "subsidy", "pm kisan", "rythu bharosa", "rythu bandhu", "panta bheema",
            "kisan subsidy", "gov schemes", "sarakaru sahaym", "pathakam apply",
            "prabhutva", "pathakalu unnaya", "yojana", "anudan",
        ],
    },
    FarmerIntent.SHOPS: {
        "en": [
            "where to buy", "where can i buy", "where i can buy", "shops near", "stores near",
            "buy urea", "buy dap", "dealer", "dealers", "input availability", "available in shop",
            "fertilizer store", "pesticide store", "buy fertilizer", "buy pesticide", "buy seeds",
            "shops nearby", "store nearby", "agro agency",
        ],
        "te": [
            "ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "సమీప దుకాణాలు", "ఎరువుల దుకాణం", "మందుల షాపు",
            "కొనుగోలు", "దుకాణం", "షాపు", "లభిస్తుంది", "యూరియా దొరుకుతుందా", "డీలర్", "డీలర్లు",
            "దుకాణాలు", "షాపులు", "దొరికే చోటు",
        ],
        "hi": [
            "दुकान", "कहाँ मिलेगा", "कहाँ से खरीदें", "खाद की दुकान", "दवा की दुकान", "डीलर",
            "कहाँ उपलब्ध", "बीज भंडार", "कृषि सेवा केंद्र", "दुकान कहाँ",
        ],
        "ta": [
            "எங்கு கிடைக்கும்", "எங்கு வாங்கலாம்", "உரக்கடை", "மருந்துக்கடை", "அருகிலுள்ள கடை",
            "டீலர்", "விற்பனை நிலையம்", "கடை எங்கு",
        ],
        "kn": [
            "ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ", "ಎಲ್ಲಿ ಖರೀದಿಸಬೇಕು", "ಗೊಬ್ಬರದ ಅಂಗಡಿ", "ಔಷಧಿ ಅಂಗಡಿ", "ಹತ್ತಿರದ ಅಂಗಡಿ",
            "ಡೀಲರ್", "ಕೃಷಿ ಕೇಂದ್ರ", "ಅಂಗಡಿ ಎಲ್ಲಿ",
        ],
        "ml": [
            "എവിടെ ലഭിക്കും", "എവിടെ വാങ്ങാം", "വളക്കട", "കീടനാശിനി കട", "സമീപത്തെ കട",
            "ഡീലർ", "കട എവിടെ",
        ],
        "mr": [
            "कुठे मिळेल", "कुठून खरेदी करावे", "खताचे दुकान", "औषधाचे दुकान", "जवळचे दुकान",
            "कृषी केंद्र", "विक्रेता", "दुकान कुठे",
        ],
        "bn": [
            "কোথায় পাওয়া যাবে", "কোথায় কিনব", "সারের দোকান", "কীটনাশকের দোকান",
            "কাছের দোকান", "ডিলার", "দোকান কোথায়",
        ],
        "gu": [
            "ક્યાં મળશે", "ક્યાંથી ખરીદવું", "ખાતરની દુકાન", "દવાની દુકાન", "નજીકની દુકાન",
            "ડીલર", "એગ્રો સેન્ટર", "દુકાન ક્યાં",
        ],
        "pa": [
            "ਕਿੱਥੇ ਮਿਲੇਗਾ", "ਕਿੱਥੋਂ ਖਰੀਦੀਏ", "ਖਾਦ ਦੀ ਦੁਕਾਨ", "ਦਵਾਈ ਦੀ ਦੁਕਾਨ", "ਨੇੜਲੀ ਦੁਕਾਨ",
            "ਡੀਲਰ", "ਦੁਕਾਨ ਕਿੱਥੇ",
        ],
        "or": [
            "କେଉଁଠି ମିଳିବ", "କେଉଁଠୁ କିଣିବି", "ସାର ଦୋକାନ", "ଔଷଧ ଦୋକାନ", "ନିକଟସ୍ଥ ଦୋକାନ",
            "ଡିଲର", "ଦୋକାନ କେଉଁଠି",
        ],
        "as": [
            "ক'ত পোৱা যাব", "ক'ত কিনিব পাৰি", "সাৰৰ দোকান", "ঔষধৰ দোকান", "ওচৰৰ দোকান",
            "ডিলাৰ", "দোকান ক'ত",
        ],
        "ur": [
            "کہاں ملے گا", "کہاں سے خریدیں", "کھاد کی دکان", "دوا کی دکان", "قریبی دکان",
            "ڈیلر", "دکان کہاں",
        ],
        "transliterated": [
            "ekkada dorukuthundi", "ekkada konali", "shops ekkada", "shop ekkada", "urea ekkada",
            "dap ekkada", "seeds ekkada", "fertilizer shop", "pesticide shop", "near shops",
            "daggara shop", "konadaniki", "dorukuthunda", "shops daggara", "dealer daggara",
            "dorukutundi", "konachu", "kahan milega", "kuthe bhetel",
        ],
    },
    FarmerIntent.CROP_HEALTH: {
        "en": [
            "disease", "pest", "pests", "leaf spot", "yellowing", "yellow", "turning yellow",
            "yellow leaves", "yellow leaf", "wilting", "fungus",
            "insects", "bollworm", "aphids", "whitefly", "blight", "rot", "pesticide for",
            "cure", "symptoms", "worms", "bugs", "fungicide", "infestation", "stem borer",
            "leaf curl", "caterpillar", "blast", "rust", "alternaria", "powdery mildew",
        ],
        "te": [
            "తెగులు", "తెగుళ్ళు", "పురుగు", "పురుగులు", "ఆకుమచ్చ", "పసుపుగా", "రాలిపోవడం",
            "ముడత", "ఎండిపోవడం", "పచ్చదోమ", "తామర పురుగులు", "బూడిద తెగులు", "అగ్గితెగులు",
            "ఆల్టర్నేరియా", "నివారణ", "మందు పిచికారీ", "రోగం", "లక్షణాలు", "మచ్చలు", "పురుగుల",
        ],
        "hi": [
            "कीट", "रोग", "कीड़ा", "बीमारी", "सुंडी", "पत्ती पीली", "धब्बे", "फंगस",
            "इल्ली", "झुलसा", "माहू", "रोकथाम", "कीटनाशक छिड़काव", "रोग उपचार", "कीट उपचार", "रोग का उपचार",
        ],
        "ta": [
            "நோய்", "பூச்சி", "புழு", "இலைப்புள்ளி", "மஞ்சள் நிறம்", "பூஞ்சை",
            "கட்டுப்பாடு", "பூச்சிக்கொல்லி", "தாக்குதல்", "மருந்து தெளிப்பு",
        ],
        "kn": [
            "ರೋಗ", "ಕೀಟ", "ಹುಳು", "ಎಲೆ ಚುಕ್ಕೆ", "ಹಳದಿ", "ಶಿಲೀಂಧ್ರ",
            "ಬಾಧೆ", "ನಿಯಂತ್ರಣ", "ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ", "ಔಷಧ",
        ],
        "ml": [
            "രോഗം", "കീടം", "പുഴു", "ഇലപ്പുള്ളി", "മഞ്ഞളിപ്പ്", "കുമിൾ",
            "കീടനിയന്ത്രണം", "കീടനാശിനി പ്രയോഗം",
        ],
        "mr": [
            "रोग", "कीड", "अळी", "बोंडअळी", "पाने पिवळी", "करपा", "बुरशी",
            "तुडतुडे", "मावा", "नियंत्रण", "फवारणी औषध",
        ],
        "bn": [
            "রোগ", "পোকা", "কীটপতঙ্গ", "পাতায় দাগ", "হলুদ পাতা", "ছত্রাক",
            "পোকামাকড়", "ব্লাইট", "দমন", "কীটনাশক স্প্রে",
        ],
        "gu": [
            "રોગ", "જીવાત", "ઈયળ", "ગુલાબી ઈયળ", "પાંદડા પીળા", "ફૂગ",
            "નિયંત્રણ", "દવાનો છંટકાવ", "ઉપદ્રવ",
        ],
        "pa": [
            "ਬਿਮਾਰੀ", "ਕੀੜਾ", "ਸੁੰਡੀ", "ਗੁਲਾਬੀ ਸੁੰਡੀ", "ਪੀਲੇ ਪੱਤੇ", "ਉੱਲੀ",
            "ਰੋਕਥਾਮ", "ਕੀਟਨਾਸ਼ਕ ਸਪਰੇਅ", "ਤੇਲਾ",
        ],
        "or": [
            "ରୋଗ", "ପୋକ", "ପତ୍ର ଦାଗ", "ହଳଦିଆ", "କବକ",
            "ନିୟନ୍ତ୍ରଣ", "କୀଟନାଶକ ସ୍ପ୍ରେ", "ଚିକିତ୍ସା",
        ],
        "as": [
            "ৰোগ", "পোক", "পৰুৱা", "পাত হালধীয়া", "ভেঁকুৰ",
            "নিয়ন্ত্ৰণ", "কীটনাশক স্প্ৰে", "উপায়",
        ],
        "ur": [
            "بیماری", "کیڑا", "کیڑے", "سنڈی", "پتے پیلے", "فنگس",
            "روکتھام", "کیڑے مار دوا", "اسپرے", "بیماری کا علاج", "کیڑوں کا علاج",
        ],
        "transliterated": [
            "tegulu", "purugu", "purugulu", "aakulu pasupuga", "aaku machalu", "marutunnayi",
            "endipotundi", "mudatha", "dosa penu", "pacha doma", "kurchuku potundi",
            "panta rogamu", "pesticide spray", "cheda", "pulla rali", "pula rali",
            "purugula mandu", "tegulu mandu", "ralipothundi", "pasupuga marutunnayi",
            "keeda", "roga", "rogh", "poka", "puzhu", "poochi", "sundi",
        ],
    },
    FarmerIntent.FERTILIZER: {
        "en": [
            "fertilizer", "fertilizers", "nutrient", "npk", "urea application", "which fertilizer",
            "what fertilizer", "micronutrient", "zinc deficiency", "potash dose", "fertilizer schedule",
            "how much fertilizer", "apply fertilizer", "manure", "compost", "dosage of urea", "dap dose",
        ],
        "te": [
            "ఎరువు", "ఎరువులు", "ఏ ఎరువు వేయాలి", "ఎరువుల యాజమాన్యం", "పోషకాలు", "నత్రజని",
            "భాస్వరం", "పొటాష్", "సూక్ష్మ పోషకాలు", "ఎరువుల మోతాదు", "జింక్ లోపం", "ఎరువు వాడాలి",
            "బాస్వరం", "యూరియా మోతాదు",
        ],
        "hi": [
            "खाद", "उर्वरक", "यूरिया", "डीएपी", "पोटाश", "पोषक तत्व",
            "कितनी खाद", "एनपीके", "जिंक", "गोबर खाद", "खाद की मात्रा",
        ],
        "ta": [
            "உரம்", "உரங்கள்", "யூரியா", "டிஏபி", "பொட்டாஷ்",
            "ஊட்டச்சத்து", "எவ்வளவு உரம்", "உர அளவு", "என்.பி.கே",
        ],
        "kn": [
            "ಗೊಬ್ಬರ", "ರಸಗೊಬ್ಬರ", "ಯೂರಿಯಾ", "ಡಿಎಪಿ", "ಪೋಷಕಾಂಶ",
            "ಎಷ್ಟು ಗೊಬ್ಬರ", "ಗೊಬ್ಬರದ ಪ್ರಮಾಣ", "ಪೊಟ್ಯಾಷ್",
        ],
        "ml": [
            "വളം", "രാസവളം", "യൂറിയ", "ഡിഎപി", "പോഷകങ്ങൾ",
            "എത്ര വളം", "പൊട്ടാഷ്", "വളപ്രയോഗം",
        ],
        "mr": [
            "खत", "रासायनिक खत", "युरिया", "डीएपी", "पोटॅश",
            "पोषकतत्वे", "खताची मात्रा", "किती खत द्यावे", "शेणखत",
        ],
        "bn": [
            "সার", "রাসায়নিক সার", "ইউরিয়া", "ডিএপি", "পটাশ",
            "পুষ্টি উপাদান", "কত সার", "সারের মাত্রা",
        ],
        "gu": [
            "ખાતર", "રાસાયણિક ખાતર", "યૂરિયા", "ડીએપી", "પોટાશ",
            "પોષક તત્વો", "કેટલું ખાતર", "ખાતરનો ડોઝ",
        ],
        "pa": [
            "ਖਾਦ", "ਯੂਰੀਆ", "ਡੀਏਪੀ", "ਪੋਟਾਸ਼", "ਪੋਸ਼ਕ ਤੱਤ",
            "ਕਿੰਨੀ ਖਾਦ", "ਰੂੜੀ ਖਾਦ",
        ],
        "or": [
            "ସାର", "ରାସାୟନିକ ସାର", "ୟୁରିଆ", "ଡିଏପି", "ପୋଟାସ",
            "ପୋଷକ ତତ୍ତ୍ୱ", "କେତେ ସାର", "ସାର ପ୍ରୟୋଗ",
        ],
        "as": [
            "সাৰ", "ৰাসায়নিক সাৰ", "ইউৰিয়া", "ডিএপি", "পটাছ",
            "কি সাৰ", "সাৰৰ মাত্ৰা",
        ],
        "ur": [
            "کھاد", "یوریا", "ڈی اے پی", "پوٹاش", "غذائی اجزاء",
            "کتنی کھاد", "کھاد کا استعمال",
        ],
        "transliterated": [
            "e eruvu veyali", "eruvulu eppudu", "fertilizer eppudu", "npk ela veyali", "poshakalu",
            "eruvu dose", "micronutrients", "urea dose", "e eruvu vadali", "fertilizer schedule",
            "eruvu entha", "eruvulu ela", "fertilizer dose", "khad", "gobbara", "uram", "valam", "khat",
        ],
    },
    FarmerIntent.IRRIGATION: {
        "en": [
            "irrigation", "watering", "water schedule", "drip irrigation", "how much water",
            "when to water", "moisture", "flood irrigation", "sprinkler",
        ],
        "te": [
            "నీరు", "నీటి యాజమాన్యం", "నీరు పెట్టాలి", "తడి ఇవ్వాలి", "డ్రిప్", "బిందు సేద్యం",
            "ఎప్పుడు నీరు", "నీరు కట్టాలి", "నీటి తడి", "నీటి పారుదల",
        ],
        "hi": [
            "सिंचाई", "पानी देना", "पानी कब दें", "ड्रिप सिंचाई", "कितना पानी",
            "सिंचाई का समय", "फव्वारा",
        ],
        "ta": [
            "நீர்ப்பாசனம்", "தண்ணீர் பாய்ச்ச", "சொட்டு நீர்", "எப்போது தண்ணீர்", "பாசன வசதி",
        ],
        "kn": [
            "ನೀರಾವರಿ", "ನೀರುಣಿಸುವುದು", "ಹನಿ ನೀರಾವರಿ", "ಯಾವಾಗ ನೀರು", "ನೀರು ಕೊಡುವುದು",
        ],
        "ml": [
            "നനയ്ക്കൽ", "ജലസേചനം", "തുള്ളി നന", "എപ്പോൾ വെള്ളം", "നന",
        ],
        "mr": [
            "सिंचन", "पाणी देणे", "ठिबक सिंचन", "पाणी व्यवस्थापन", "कधी पाणी द्यावे", "तुषार",
        ],
        "bn": [
            "সেচ", "পানি দেওয়া", "ড্রিপ সেচ", "কখন পানি দেব", "জলসেচ",
        ],
        "gu": [
            "પિયત", "પાણી આપવું", "ટપક પદ્ધતિ", "ક્યારે પાણી આપવું", "સિંચાઈ",
        ],
        "pa": [
            "ਸਿੰਚਾਈ", "ਪਾਣੀ ਲਾਉਣਾ", "ਤੁਪਕਾ ਸਿੰਚਾਈ", "ਕਦੋਂ ਪਾਣੀ", "ਪਾਣੀ ਦੀ ਲੋੜ",
        ],
        "or": [
            "ଜଳସେଚନ", "ପାଣି ଦେବା", "ବୁନ୍ଦା ଜଳସେଚନ", "କେବେ ପାଣି",
        ],
        "as": [
            "জলসিঞ্চন", "পানী দিয়া", "টোপাল জলসিঞ্চন", "কেতিয়া পানী",
        ],
        "ur": [
            "آبپاشی", "پانی دینا", "ڈرپ آبپاشی", "کب پانی دیں", "پانی کی ضرورت",
        ],
        "transliterated": [
            "neeru eppudu pettali", "thadi eppudu ivvali", "neeti yajamanyam", "water eppudu pettali",
            "drip irrigation", "water kattachu", "thadulu eppudu", "neeru eppudu",
            "pani", "neeravari", "thanni", "vellam", "jol", "sinchan", "piyat",
        ],
    },
    FarmerIntent.SOWING: {
        "en": [
            "sowing", "seed rate", "seed treatment", "planting time", "how to sow", "seed spacing",
            "nursery", "germination", "transplanting", "sowing depth",
        ],
        "te": [
            "విత్తనాలు", "విత్తన శుద్ధి", "విత్తే సమయం", "నాట్లు", "నాటడం", "ఎప్పుడు విత్తాలి",
            "విత్తన మోతాదు", "మొలక", "నారుమడి",
        ],
        "hi": [
            "बुआई", "बीज उपचार", "बुवाई का समय", "बीज दर", "पौधरोपण", "अंकुरण", "कब बोएं", "बीज शोधन",
        ],
        "ta": [
            "விதைப்பு", "விதை நேர்த்தி", "விதைக்கும் நேரம்", "நாற்று நடுதல்", "முளைப்பு", "விதை அளவு",
        ],
        "kn": [
            "ಬಿತ್ತನೆ", "ಬೀಜೋಪಚಾರ", "ಬಿತ್ತನೆ ಸಮಯ", "ನಾಟಿ", "ಮೊಳಕೆ", "ಬೀಜದ ಪ್ರಮಾಣ",
        ],
        "ml": [
            "വിത്ത് വിതയ്ക്കൽ", "വിത്തുചികിത്സ", "നടീൽ സമയം", "മുളയ്ക്കൽ", "ഞാറുനടീൽ",
        ],
        "mr": [
            "पेरणी", "बीजप्रक्रिया", "पेरणीची वेळ", "बियाणे दर", "लागवड", "उगवण", "कधी पेरावे",
        ],
        "bn": [
            "বপন", "বীজ শোধন", "বপনের সময়", "চারা রোপণ", "অঙ্কুরোদগম", "বীজের হার",
        ],
        "gu": [
            "વાવણી", "બીજ માવજત", "વાવણીનો સમય", "ધરૂ રોપણ", "ઉગાવો", "બિયારણ દર",
        ],
        "pa": [
            "ਬਿਜਾਈ", "ਬੀਜ ਸੋਧ", "ਬਿਜਾਈ ਦਾ ਸਮਾਂ", "ਪਨੀਰੀ", "ਅੰਕੁਰਨ", "ਕਦੋਂ ਬੀਜੀਏ",
        ],
        "or": [
            "ବୁଣିବା", "ବିହନ ବିଶୋଧନ", "ବୁଣିବା ସମୟ", "ତଳି ରୋପଣ", "ଗଜା ହେବା",
        ],
        "as": [
            "বীজ সিঁচা", "বীজ শোধন", "ৰোপণৰ সময়", "পুলি ৰোপণ", "অঙ্কুৰণ",
        ],
        "ur": [
            "بوائی", "بیج کا علاج", "کاشت کا وقت", "پود کاری", "اگاؤ", "کب بوئیں",
        ],
        "transliterated": [
            "vithanalu eppudu veyali", "vithana shuddi", "sowing time", "seed rate entha",
            "natlu eppudu", "vithadam ela", "vithanala", "vithadam", "vithanalu", "vithana",
            "seeds eppudu", "buwai", "bittane", "perani", "vavani",
        ],
    },
    FarmerIntent.HARVESTING: {
        "en": [
            "harvesting", "harvest time", "when to harvest", "picking cotton", "harvest maturity",
            "threshing", "storage", "post harvest",
        ],
        "te": [
            "కోత", "కోత సమయం", "ఎప్పుడు కోయాలి", "పత్తి ఏరడం", "దిగుబడి", "నిల్వ", "కోయడం", "కోతలు",
        ],
        "hi": [
            "कटाई", "कटाई का समय", "फसल कटाई", "गहाई", "कब काटें", "भंडारण", "तुड़ाई",
        ],
        "ta": [
            "அறுவடை", "அறுவடை காலம்", "எப்போது அறுவடை", "பறித்தல்", "சேமிப்பு",
        ],
        "kn": [
            "ಕಟಾವು", "ಸುಗ್ಗಿ", "ಕಟಾವಿನ ಸಮಯ", "ಯಾವಾಗ ಕಟಾವು", "ಸಂಗ್ರಹಣೆ",
        ],
        "ml": [
            "വിളവെടുപ്പ്", "കൊയ്ത്ത്", "എപ്പോൾ കൊയ്യണം", "സംഭരണം",
        ],
        "mr": [
            "कापणी", "काढणी", "काढणीची वेळ", "वेचणी", "साठवणूक", "कधी काढावे",
        ],
        "bn": [
            "ফসল কাটা", "কাটার সময়", "তোলার সময়", "সংরক্ষণ", "মাড়াই",
        ],
        "gu": [
            "કાપણી", "લણણી", "કાપણીનો સમય", "વીણવું", "સંગ્રહ",
        ],
        "pa": [
            "ਵਾਢੀ", "ਕਟਾਈ", "ਵਾਢੀ ਦਾ ਸਮਾਂ", "ਚੁਗਾਈ", "ਸੰਭਾਲ",
        ],
        "or": [
            "ଅମଳ", "କାଟିବା ସମୟ", "ଫସଲ କଟା", "ସଂରକ୍ଷଣ",
        ],
        "as": [
            "শস্য চপোৱা", "কটাৰ সময়", "চপোৱাৰ সময়", "সংৰক্ষণ",
        ],
        "ur": [
            "کٹائی", "فصل کی کٹائی", "چنائی", "ذخیرہ اندوزی", "کب کاٹیں",
        ],
        "transliterated": [
            "kotha eppudu koyali", "harvesting time", "patti eppudu thiyyali", "kotha kosaru",
            "nilva cheyadam", "digubadi", "kotha samayam", "panta kotha", "kotha time", "koyadam",
            "katai", "katavu", "aruvadai", "koythu",
        ],
    },
}

# Module-level aliases for backward compatibility
MARKET_PRICE_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.MARKET_PRICE]["en"]
MARKET_PRICE_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.MARKET_PRICE]["te"]
MARKET_PRICE_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.MARKET_PRICE]["transliterated"]

WEATHER_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.WEATHER]["en"]
WEATHER_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.WEATHER]["te"]
WEATHER_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.WEATHER]["transliterated"]

SCHEMES_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.GOVERNMENT_SCHEMES]["en"]
SCHEMES_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.GOVERNMENT_SCHEMES]["te"]
SCHEMES_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.GOVERNMENT_SCHEMES]["transliterated"]

SHOPS_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.SHOPS]["en"]
SHOPS_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.SHOPS]["te"]
SHOPS_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.SHOPS]["transliterated"]

CROP_HEALTH_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.CROP_HEALTH]["en"]
CROP_HEALTH_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.CROP_HEALTH]["te"]
CROP_HEALTH_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.CROP_HEALTH]["transliterated"]

FERTILIZER_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.FERTILIZER]["en"]
FERTILIZER_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.FERTILIZER]["te"]
FERTILIZER_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.FERTILIZER]["transliterated"]

IRRIGATION_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.IRRIGATION]["en"]
IRRIGATION_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.IRRIGATION]["te"]
IRRIGATION_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.IRRIGATION]["transliterated"]

SOWING_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.SOWING]["en"]
SOWING_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.SOWING]["te"]
SOWING_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.SOWING]["transliterated"]

HARVESTING_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.HARVESTING]["en"]
HARVESTING_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.HARVESTING]["te"]
HARVESTING_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.HARVESTING]["transliterated"]

REMINDERS_KEYWORDS_EN = INTENT_KEYWORDS[FarmerIntent.REMINDERS]["en"]
REMINDERS_KEYWORDS_TE = INTENT_KEYWORDS[FarmerIntent.REMINDERS]["te"]
REMINDERS_KEYWORDS_TANGLISH = INTENT_KEYWORDS[FarmerIntent.REMINDERS]["transliterated"]

GENERAL_FARMING_WORDS = [
    # English
    "crop", "crops", "farm", "farming", "land", "field", "acre", "acres", "yield", "soil", "cotton", "paddy", "chilli", "tomato", "maize", "agriculture",
    # Telugu
    "పంట", "పొలం", "సాగు", "భూమి", "ఎకరం", "దిగుబడి", "నేల", "పత్తి", "వరి", "మిర్చి",
    # Hindi / Marathi
    "फसल", "खेती", "खेत", "किसान", "जमीन", "मिट्टी", "पैदावार", "कपास", "धान", "गेहूं", "शेती", "शेतकरी", "पीक",
    # Tamil
    "பயிர்", "விவசாயம்", "நிலம்", "வயல்", "மண்", "விவசாயி", "மகசூல்", "பருத்தி", "நெல்",
    # Kannada
    "ಬೆಳೆ", "ಕೃಷಿ", "ಜಮೀನು", "ಹೊಲ", "ಮಣ್ಣು", "ರೈತ", "ಇಳುವರಿ", "ಹತ್ತಿ", "ಭತ್ತ",
    # Malayalam
    "വിള", "കൃഷി", "നിലം", "പാടം", "മണ്ണ്", "കർഷകൻ", "വിളവ്", "നെല്ല്",
    # Bengali / Assamese
    "ফসল", "চাষ", "জমি", "মাটি", "কৃষক", "ফলন", "ধান", "খেতি",
    # Gujarati
    "પાક", "ખેતી", "જમીન", "ખેતર", "માટી", "ખેડૂત", "ઉત્પાદન", "કપાસ",
    # Punjabi
    "ਫ਼ਸਲ", "ਖੇਤੀ", "ਜ਼ਮੀਨ", "ਖੇਤ", "ਮਿੱਟੀ", "ਕਿਸਾਨ", "ਝਾੜ", "ਕਣਕ", "ਝੋਨਾ",
    # Odia
    "ଫସଲ", "ଚାଷ", "ଜମି", "ବିଲ", "ମାଟି", "କୃଷକ", "ଅମଳ", "ଧାନ",
    # Urdu
    "فصل", "کھیتی", "کھیت", "کسان", "زمین", "مٹی", "پیداوار", "کپاس",
]



GREETING_REPLIES = GREETING_RESPONSES

def _flatten_multilingual(intent: FarmerIntent) -> List[str]:
    kws = []
    for lang, lst in INTENT_KEYWORDS.get(intent, {}).items():
        if lang not in ("en", "te", "transliterated"):
            kws.extend(lst)
    return kws

MARKET_PRICE_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.MARKET_PRICE)
WEATHER_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.WEATHER)
SCHEMES_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.GOVERNMENT_SCHEMES)
SHOPS_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.SHOPS)
CROP_HEALTH_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.CROP_HEALTH)
FERTILIZER_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.FERTILIZER)
IRRIGATION_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.IRRIGATION)
SOWING_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.SOWING)
HARVESTING_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.HARVESTING)
REMINDERS_KEYWORDS_MULTILINGUAL = _flatten_multilingual(FarmerIntent.REMINDERS)


def _normalize_query_text(text: str) -> str:
    """
    Normalizes incoming queries for robust Indic and multilingual matching:
    1. Composes Unicode characters (NFC canonical composition).
    2. Strips invisible zero-width formatting codepoints (ZWNJ \u200c, ZWJ \u200d, BOM \ufeff).
    3. Collapses multiple whitespace to single space.
    """
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFC", text)
    cleaned = re.sub(r"[\u200b-\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _matches_intent_keywords(msg: str, intent: FarmerIntent) -> bool:
    """Check if lowercased message matches any keywords across all languages for a specific intent."""
    lang_map = INTENT_KEYWORDS.get(intent, {})
    for lang, keywords in lang_map.items():
        for kw in keywords:
            if _normalize_query_text(kw).lower() in msg:
                return True
    return False


class AIDecisionEngine:
    """
    Orchestration layer determining intents, data dependencies, and routing.
    """

    @staticmethod
    def is_greeting_only(user_message: str) -> bool:
        """
        Check whether the message is strictly a greeting/courtesy without any factual question.
        Guarantees greetings never trigger expensive external queries.
        """
        if not user_message or not user_message.strip():
            return False

        cleaned = _normalize_query_text(user_message)
        if not cleaned:
            return False

        msg = cleaned.lower()

        # Check against pure greeting patterns
        is_greeting_match = any(re.search(pat, msg, re.IGNORECASE) for pat in GREETING_PATTERNS)

        if not is_greeting_match:
            tokens = [t.strip(",.!?") for t in msg.split() if t.strip(",.!?")]
            if len(tokens) <= 3 and any(t in GREETING_TOKENS for t in tokens):
                is_greeting_match = True

        if not is_greeting_match:
            return False

        # If any domain keywords are present, it's not a pure greeting
        for intent in [
            FarmerIntent.STOCK_ALERT,
            FarmerIntent.MARKET_PRICE,
            FarmerIntent.WEATHER,
            FarmerIntent.GOVERNMENT_SCHEMES,
            FarmerIntent.SHOPS,
            FarmerIntent.CROP_HEALTH,
            FarmerIntent.FERTILIZER,
            FarmerIntent.IRRIGATION,
            FarmerIntent.SOWING,
            FarmerIntent.HARVESTING,
            FarmerIntent.REMINDERS,
        ]:
            if _matches_intent_keywords(msg, intent):
                return False

        return True

    @staticmethod
    def get_greeting_reply(language: str = "te") -> str:
        """Return a helpful, welcoming WhatsApp greeting response in the requested language."""
        return GREETING_RESPONSES.get(
            language,
            GREETING_RESPONSES.get("en", "Hello! I am your BhoomiMitra AI farming assistant. 🙏")
        )

    @classmethod
    def detect_all_intents(cls, user_message: str) -> List[FarmerIntent]:
        """
        Identify all intents present in the farmer's message across all 13 supported languages.
        Supports multi-intent queries (e.g. market prices + weather).
        """
        if not user_message or not user_message.strip():
            return [FarmerIntent.UNKNOWN]

        cleaned_query = _normalize_query_text(user_message)
        if not cleaned_query:
            return [FarmerIntent.UNKNOWN]

        msg = cleaned_query.lower()
        msg_original = cleaned_query

        # Pure greeting check
        if cls.is_greeting_only(msg_original):
            return [FarmerIntent.GREETING]

        detected: List[FarmerIntent] = []

        # 0. Stock Alerts (prioritized if user asks for stock availability alert/cancellation/listing)
        if _matches_intent_keywords(msg, FarmerIntent.STOCK_ALERT):
            detected.append(FarmerIntent.STOCK_ALERT)

        # 0.1. Reminders (prioritized if user asks to be reminded)
        if _matches_intent_keywords(msg, FarmerIntent.REMINDERS):
            detected.append(FarmerIntent.REMINDERS)

        # 1. Market Price
        if _matches_intent_keywords(msg, FarmerIntent.MARKET_PRICE):
            detected.append(FarmerIntent.MARKET_PRICE)

        # 2. Weather
        if _matches_intent_keywords(msg, FarmerIntent.WEATHER):
            detected.append(FarmerIntent.WEATHER)

        # 3. Government Schemes
        if _matches_intent_keywords(msg, FarmerIntent.GOVERNMENT_SCHEMES):
            detected.append(FarmerIntent.GOVERNMENT_SCHEMES)

        # 4. Shops / Input Availability
        if _matches_intent_keywords(msg, FarmerIntent.SHOPS):
            detected.append(FarmerIntent.SHOPS)

        # 5. Crop Health / Disease / Pest
        if _matches_intent_keywords(msg, FarmerIntent.CROP_HEALTH):
            detected.append(FarmerIntent.CROP_HEALTH)

        # 6. Fertilizer / Nutrients
        if _matches_intent_keywords(msg, FarmerIntent.FERTILIZER):
            detected.append(FarmerIntent.FERTILIZER)

        # 7. Irrigation
        if _matches_intent_keywords(msg, FarmerIntent.IRRIGATION):
            detected.append(FarmerIntent.IRRIGATION)

        # 8. Sowing
        if _matches_intent_keywords(msg, FarmerIntent.SOWING):
            detected.append(FarmerIntent.SOWING)

        # 9. Harvesting
        if _matches_intent_keywords(msg, FarmerIntent.HARVESTING):
            detected.append(FarmerIntent.HARVESTING)

        # Fallback: if no specific domain matched, identify as general farming or unknown
        if not detected:
            from src.rag.service import extract_crop_from_text
            mentioned_crop = extract_crop_from_text(msg_original)
            if mentioned_crop or any(w in msg for w in GENERAL_FARMING_WORDS):
                detected.append(FarmerIntent.GENERAL_FARMING)
            else:
                detected.append(FarmerIntent.UNKNOWN)

        return detected

    @classmethod
    def detect_primary_intent(cls, user_message: str) -> FarmerIntent:
        """Identify the single primary intent for a user message."""
        intents = cls.detect_all_intents(user_message)
        return intents[0] if intents else FarmerIntent.GENERAL_FARMING

    async def process_message(
        self,
        db: AsyncSession,
        farmer: Farmer,
        conversation: Conversation,
    ) -> str:
        """
        Main decision and orchestration pipeline.
        - Identifies intents
        - Bypasses LLM for pure greetings
        - Coordinates specialized services only when relevant
        - Enforces factual authoritativeness (never hallucinating numbers or prices)
        - Applies localized fallbacks when services produce no data
        - Formats and finalizes the outgoing response
        """
        user_message = conversation.user_message or ""
        logger.info(f"[DECISION ENGINE START] Farmer: {farmer.id} | Query: '{user_message}'")

        # Resolve language deterministically using centralized detector
        from src.language.detector import detect_language
        pref_fallback = getattr(farmer, "preferred_language", None) or "te"
        language = detect_language(user_message, fallback=pref_fallback)

        # 1. Pure Greeting Shortcut
        if self.is_greeting_only(user_message):
            logger.info(f"[DECISION ENGINE] Pure greeting detected. Returning instant greeting for farmer {farmer.id}.")
            reply = self.get_greeting_reply(language=language)
            conversation.ai_response = reply
            conversation.intent = FarmerIntent.GREETING.value
            db.add(conversation)
            await db.commit()
            return reply

        # 2. Detect all intents
        intents = self.detect_all_intents(user_message)
        primary_intent = intents[0]
        conversation.intent = primary_intent.value
        logger.info(f"[DECISION ENGINE] Detected intents: {[i.value for i in intents]} | Primary: {primary_intent.value}")

        has_market = FarmerIntent.MARKET_PRICE in intents
        has_weather = FarmerIntent.WEATHER in intents
        has_schemes = FarmerIntent.GOVERNMENT_SCHEMES in intents
        has_shops = FarmerIntent.SHOPS in intents
        has_stock_alert = FarmerIntent.STOCK_ALERT in intents
        has_crop_advice = any(i in intents for i in [
            FarmerIntent.CROP_ADVICE,
            FarmerIntent.CROP_HEALTH,
            FarmerIntent.FERTILIZER,
            FarmerIntent.IRRIGATION,
            FarmerIntent.SOWING,
            FarmerIntent.HARVESTING,
            FarmerIntent.REMINDERS,
            FarmerIntent.STOCK_ALERT,
            FarmerIntent.GENERAL_FARMING,
            FarmerIntent.UNKNOWN,
        ])

        # 3. AI Advisory Generation
        ai_response_text = ""
        from src.ai.repository import AIRepository
        from src.ai.service import AIService, _finalize_whatsapp_response
        from src.ai.schemas import AIGenerateRequest

        repo = AIRepository(db)
        ai_service = AIService(repo)
        request = AIGenerateRequest(
            farmer_id=farmer.id,
            conversation_id=conversation.id,
            message=user_message,
        )

        try:
            response = await ai_service.generate_ai_response(request)
            ai_response_text = response.response_text or ""
            logger.info(f"[DECISION ENGINE] Raw Gemini response ({len(ai_response_text)} chars)")
        except Exception as exc:
            logger.warning(f"[DECISION ENGINE] AI generation unavailable: {exc}. Deferring to specialized modules.")
            ai_response_text = ""

        # 4. Authoritative Module Routing (Only call enrichments when intent is relevant)
        # A.0. Stock Availability Alerts
        if has_stock_alert:
            try:
                from src.shops.stock_alerts import handle_stock_alert_query
                logger.info("[DECISION ENGINE] Routing to stock alerts service")
                ai_response_text = await handle_stock_alert_query(
                    db, user_message, ai_response_text, farmer, language=language
                )
            except Exception as alert_err:
                logger.warning(f"Stock alert handling warning: {alert_err}")

        # A. Shops / Input Availability
        if has_shops and not has_stock_alert:
            try:
                from src.shops.service import enrich_response_with_shops
                logger.info("[DECISION ENGINE] Routing to shops service")
                ai_response_text = await enrich_response_with_shops(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as err:
                logger.warning(f"Shops enrichment warning: {err}")

        # B. Market Price
        if has_market:
            try:
                from src.market.service import enrich_response_with_market_prices
                logger.info("[DECISION ENGINE] Routing to market price service")
                ai_response_text = await enrich_response_with_market_prices(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as mkt_err:
                logger.warning(f"Market enrichment warning: {mkt_err}")

        # C. Weather
        if has_weather:
            try:
                from src.weather.service import enrich_response_with_weather
                logger.info("[DECISION ENGINE] Routing to weather service")
                ai_response_text = await enrich_response_with_weather(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as weather_err:
                logger.warning(f"Weather enrichment warning: {weather_err}")

        # D. Government Schemes
        if has_schemes:
            try:
                from src.schemes.service import enrich_response_with_schemes
                logger.info("[DECISION ENGINE] Routing to schemes service")
                ai_response_text = await enrich_response_with_schemes(
                    db, user_message, ai_response_text, farmer
                )
            except Exception as scheme_err:
                logger.warning(f"Schemes enrichment warning: {scheme_err}")

        # E. Expert Escalation
        try:
            from src.escalation.service import enrich_response_with_escalation
            ai_response_text = await enrich_response_with_escalation(
                db, user_message, ai_response_text, farmer
            )
        except Exception as esc_err:
            logger.warning(f"Escalation enrichment warning: {esc_err}")

        # 5. Multi-Intent Response Formatting
        try:
            from src.ai.formatting import format_multi_intent_response
            ai_response_text = format_multi_intent_response(
                assembled_text=ai_response_text,
                user_message=user_message,
                language=language,
            )
        except Exception as fmt_err:
            logger.warning(f"Multi-intent formatting warning: {fmt_err}")

        # 6. Fallback Protection for Single Intent / Missing Responses
        ai_response_text = ai_response_text.strip() if ai_response_text else ""
        is_single_intent = len(intents) == 1

        if is_single_intent:
            if primary_intent == FarmerIntent.MARKET_PRICE and not any(k in ai_response_text for k in ["📊", "⚠️"]):
                ai_response_text = get_market_fallback_response(language)
            elif primary_intent == FarmerIntent.GOVERNMENT_SCHEMES and "🏛️" not in ai_response_text:
                ai_response_text = get_schemes_fallback_response(language)
            elif primary_intent == FarmerIntent.WEATHER and not any(w in ai_response_text for w in ["🌡️", "🌤️", "🌦️"]):
                ai_response_text = get_weather_fallback_response(language)
            elif primary_intent == FarmerIntent.SHOPS and "🏬" not in ai_response_text:
                ai_response_text = get_shops_fallback_response(language)
            elif primary_intent == FarmerIntent.STOCK_ALERT and not any(s in ai_response_text for s in ["🔔", "🏬", "✅", "ℹ️"]):
                ai_response_text = "🔔 యూరియా స్టాక్ అలర్ట్ యాక్టివ్ అయింది." if language == "te" else "🔔 Stock alert has been registered."
            elif not ai_response_text:
                ai_response_text = get_fallback_response(language)
        elif not ai_response_text:
            ai_response_text = get_fallback_response(language)

        # 7. Finalize WhatsApp Response (Length & block management)
        ai_response_text = _finalize_whatsapp_response(ai_response_text)

        logger.info(f"[DECISION ENGINE FINAL] Output chars: {len(ai_response_text)} | Preview: {ai_response_text[:120]}...")

        conversation.ai_response = ai_response_text
        db.add(conversation)
        await db.commit()

        return ai_response_text


_decision_engine_instance: Optional[AIDecisionEngine] = None


def get_decision_engine() -> AIDecisionEngine:
    """Return singleton instance of AIDecisionEngine."""
    global _decision_engine_instance
    if _decision_engine_instance is None:
        _decision_engine_instance = AIDecisionEngine()
    return _decision_engine_instance
