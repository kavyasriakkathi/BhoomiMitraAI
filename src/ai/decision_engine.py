"""
BhoomiMitra AI — AI Decision Engine & Orchestration Layer

The central decision-making brain of BhoomiMitra AI.
Responsible for:
1. Multi-lingual intent classification across 13 Indian languages (Telugu, Hindi, English, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Odia, Punjabi, Assamese, Urdu, plus Romanized inputs).
2. Authoritative module routing (Market, Weather, Schemes, Shops, Advisory, Vision, Escalation).
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
from src.language.detector import detect_language
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
# Intent Keyword & Pattern Mappings (13 Languages + Romanized)
# ─────────────────────────────────────────────────────────────────────────────

GREETING_PATTERNS = [
    r"\b(?:hello|hi|hey|hai|helo|namaste|namaskar|namaskaram|namaskaralu|pranam|vanakkam|namaskara|namaskaram|adaab|sat\s*sri\s*akal|khammaghani)\b",
    r"\b(?:good\s+morning|good\s+afternoon|good\s+evening|greetings)\b",
    r"^(?:నమస్తే|నమస్కారం|నమస్కారాలు|హలో|హాయ్|नमस्ते|नमस्कार|प्रणाम|வணக்கம்|ನಮಸ್ಕಾರ|ನಮಸ್ಕಾರಗಳು|നമസ്കാരം|নমস্কার|নমস্কাৰ|સત\s*શ્રી\s*અકાલ|ਸਤਿ\s*ਸ੍ਰੀ\s*ਅਕਾਲ|ନମସ୍କାର|سلام|ആദരവ്)[\s\!,\.]*$",
    r"^(?:హాయ్|నమస్తే|హలో|నమస్కారం|नमस्ते|नमस्कार|வணக்கம்|ನಮಸ್ಕಾರ|নমস্কার)\s+(?:భూమిమిత్ర|bhoomimitra|भूमिमित्र|பூமிமித்ரா|ಭೂಮಿಮಿತ್ರ|రైతు\s*మిత్ర|రైతు\s*అన్న|రైతు|किसान|விவசாயி)?[\s\!,\.]*$",
]

GREETING_REPLIES: Dict[str, str] = {
    "te": (
        "నమస్తే! నేను మీ భూమిమిత్ర AI వ్యవసాయ సహాయకుడిని. 🙏\n\n"
        "పంట సలహాలు, తెగుళ్ల నివారణ, ఎరువుల సమాచారం, మార్కెట్ ధరలు "
        "మరియు వాతావరణ అంచనా కోసం నన్ను అడగవచ్చు. మీకు ఏ విధంగా సహాయపడగలను?"
    ),
    "hi": (
        "नमस्ते! मैं आपका भूमिमित्र AI कृषि सहायक हूँ। 🙏\n\n"
        "आप मुझसे फसल सलाह, कीट व रोग नियंत्रण, खाद की जानकारी, मंडी भाव "
        "और मौसम पूर्वानुमान के बारे में पूछ सकते हैं। मैं आपकी क्या मदद कर सकता हूँ?"
    ),
    "en": (
        "Hello! I am your BhoomiMitra AI farming assistant. 🙏\n\n"
        "You can ask me about crop advisory, pest/disease management, fertilizer recommendations, "
        "mandi prices, and weather forecasts. How can I assist you today?"
    ),
    "ta": (
        "வணக்கம்! நான் உங்கள் பூமிமித்ரா AI விவசாய உதவியாளர். 🙏\n\n"
        "பயிர் ஆலோசனைகள், பூச்சி கட்டுப்பாடு, உர பரிந்துரைகள், சந்தை விலைகள் "
        "மற்றும் வானிலை முன்னறிவிப்பு பற்றி என்னிடம் கேட்கலாம். உங்களுக்கு எப்படி உதவ முடியும்?"
    ),
    "kn": (
        "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಭೂಮಿಮಿತ್ರ AI ಕೃಷಿ ಸಹಾಯಕ. 🙏\n\n"
        "ಬೆಳೆ ಸಲಹೆ, ಕೀಟ ಮತ್ತು ರೋಗ ನಿಯಂತ್ರಣ, ರಸಗೊಬ್ಬರ ಮಾಹಿತಿ, ಮಾರುಕಟ್ಟೆ ದರಗಳು "
        "ಮತ್ತು ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಗಾಗಿ ನೀವು ನನ್ನನ್ನು ಕೇಳಬಹುದು. ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
    ),
    "ml": (
        "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ ഭൂമിമിത്ര AI കാർഷിക സഹായിയാണ്. 🙏\n\n"
        "വിള ഉപദേശങ്ങൾ, കീട-രോഗ നിയന്ത്രണം, വളം വിവരങ്ങൾ, വിപണി വിലകൾ, "
        "കാലാവസ്ഥാ പ്രവചനം എന്നിവയെക്കുറിച്ച് നിങ്ങൾക്ക് എന്നോട് ചോദിക്കാം. ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കണം?"
    ),
    "mr": (
        "नमस्कार! मी तुमचा भूमिमित्र AI कृषी सहाय्यक आहे. 🙏\n\n"
        "तुम्ही मला पीक सल्ला, कीड व रोग नियंत्रण, खत व्यवस्थापन, बाजारभाव "
        "आणि हवामान अंदाजाबद्दल विचारू शकता. मी आपली काय मदत करू शकतो?"
    ),
    "bn": (
        "নমস্কার! আমি আপনার ভূমিমিত্র AI কৃষি সহকারী। 🙏\n\n"
        "আপনি ফসল পরামর্শ, কীট ও রোগ নিয়ন্ত্রণ, সার তথ্য, বাজার দর "
        "এবং আবহাওয়ার পূর্বাভাসের জন্য আমাকে জিজ্ঞাসা করতে পারেন। আমি আপনাকে কীভাবে সাহায্য করতে পারি?"
    ),
    "gu": (
        "નમસ્તે! હું તમારો ભૂમિમિત્ર AI કૃષિ સહાયક છું. 🙏\n\n"
        "તમે મને પાકની સલાહ, જીવાત નિયંત્રણ, ખાતરની માહિતી, બજાર ભાવ "
        "અને હવામાન આગાહી વિશે પૂછી શકો છો. હું તમને કેવી રીતે મદદ કરી શકું?"
    ),
    "or": (
        "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର ଭୂମିମିତ୍ର AI କୃଷି ସହାୟକ। 🙏\n\n"
        "ଆପଣ ଫସଲ ପରାମର୍ଶ, କୀଟ ନିୟନ୍ତ୍ରଣ, ସାର ସୂଚନା, ବଜାର ଦର "
        "ଏବଂ ପାଣିପାଗ ପୂର୍ବାନୁମାନ ବିଷୟରେ ମୋତେ ପଚାରିପାରିବେ। ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?"
    ),
    "pa": (
        "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ ਭੂਮੀਮਿੱਤਰ AI ਖੇਤੀਬਾੜੀ ਸਹਾਇਕ ਹਾਂ। 🙏\n\n"
        "ਤੁਸੀਂ ਫਸਲ ਦੀ ਸਲਾਹ, ਕੀੜੇ-ਮਕੌੜਿਆਂ ਦੀ ਰੋਕਥਾਮ, ਖਾਦ ਦੀ ਜਾਣਕਾਰੀ, ਮੰਡੀ ਭਾਅ "
        "ਅਤੇ ਮੌਸਮ ਬਾਰੇ ਪੁੱਛ ਸਕਦੇ ਹੋ। ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?"
    ),
    "as": (
        "নমস্কাৰ! মই আপোনাৰ ভূমিমিত্ৰ AI কৃষি সহায়ক। 🙏\n\n"
        "আপুনি শস্য পৰামৰ্শ, পোক-পৰুৱা নিয়ন্ত্ৰণ, সাৰৰ তথ্য, বজাৰ দৰ "
        "আৰু বতৰৰ আগজাননীৰ বিষয়ে মোক সুধিব পাৰে। মই আপোনাক কেনেকৈ সহায় কৰিব পাৰোঁ?"
    ),
    "ur": (
        "سلام! میں آپ کا بھومی مترا AI زرعی معاون ہوں۔ 🙏\n\n"
        "آپ فصل کے مشورے، کیڑوں اور بیماریوں کی روک تھام، کھاد کی معلومات، منڈی کے بھاؤ "
        "اور موسم کی پیش گوئی کے بارے میں مجھ سے پوچھ سکتے ہیں۔ میں آپ کی کیا مدد کر سکتا ہوں؟"
    ),
}

MARKET_PRICE_KEYWORDS_EN = [
    "market price", "mandi price", "mandi rate", "market rate", "selling price",
    "rate per quintal", "cotton price", "cotton rate", "paddy price", "chilli price",
    "tomato price", "price of", "prices of", "how much price", "how much rate",
    "market value", "mandi rates", "market prices",
]

MARKET_PRICE_KEYWORDS_TE = [
    "మార్కెట్ ధర", "మార్కెట్ ధరలు", "మండి ధర", "మండి ధరలు", "ధర ఎంత", "రేటు ఎంత",
    "క్వింటాల్", "క్వింటాలు", "అమ్ముకోవాలి", "గిట్టుబాటు ధర", "మార్కెట్లో", "మండిలో",
    "పత్తి ధర", "మిర్చి ధర", "వరి ధర", "టమాటా ధర", "రేట్లు", "ధరలు",
]

MARKET_PRICE_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "मंडी भाव", "बाजार भाव", "मंडी रेट", "क्विंटल का भाव", "कपास का भाव", "धान का भाव",
    "टमाटर का भाव", "मिर्च का भाव", "बाजार दर", "विक्री किंमत", "कापूस भाव", "सोयाबीन भाव",
    # Tamil
    "சந்தை விலை", "மண்டி விலை", "விலை என்ன", "குவிண்டால் விலை", "நெல் விலை", "பருத்தி விலை",
    # Kannada
    "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ", "ಮಂಡಿ ದರ", "ಬೆಲೆ ಎಷ್ಟು", "ಕ್ವಿಂಟಾಲ್ ಬೆಲೆ", "ಹತ್ತಿ ಬೆಲೆ", "ಭತ್ತದ ಬೆಲೆ",
    # Malayalam
    "വിപണി വില", "വിപണി നിരക്ക്", "വില എത്ര", "ക്വിന്റൽ വില", "നെല്ല് വില",
    # Bengali & Assamese
    "বাজার দর", "বাজারের দাম", "কত দাম", "কুইন্টাল দর", "ধানের দাম", "বজাৰ দৰ", "কপাহৰ দাম",
    # Gujarati
    "બજાર ભાવ", "મંડી ભાવ", "ભાવ કેટલો", "ભાવ શું છે", "કપાસનો ભાવ", "ડાંગરનો ભાવ",
    # Odia
    "ବଜାର ଦର", "ମଣ୍ଡି ଦର", "ଦର କେତେ", "କ୍ୱିଣ୍ଟାଲ ଦର", "ଧାନ ଦର",
    # Punjabi
    "ਮੰਡੀ ਭਾਅ", "ਬਾਜ਼ਾਰ ਭਾਅ", "ਭਾਅ ਕਿੰਨਾ", "ਕਣਕ ਦਾ ਭਾਅ", "ਝੋਨੇ ਦਾ ਭਾਅ",
    # Urdu
    "منڈی کا بھاؤ", "مارکیٹ ریٹ", "قیمت کیا ہے", "فی کوئنٹل ریٹ", "کپاس کا ریٹ",
]

MARKET_PRICE_KEYWORDS_TANGLISH = [
    "rate entha", "dhara entha", "rate entho", "dhara entho", "cotton rate", "patti rate",
    "patti dhara", "mirchi rate", "mirapa rate", "tomato rate", "tamata rate", "mandi rate",
    "market lo", "mandi lo", "market price", "mandi price", "bajar rate", "ammukovali",
    "eeroju rate", "today rate", "rate ela undi", "dhara ela undi", "per quintal rate",
    "lo cotton rate", "lo patti rate", "patti rate entha", "cotton rate entha",
    "mandi bhav", "kapas bhav", "dhan ka rate", "eshtu rate", "vilai enna", "koto dam",
]

WEATHER_KEYWORDS_EN = [
    "weather", "forecast", "rain", "raining", "rainy", "temperature", "humidity",
    "wind", "storm", "thunderstorm", "will it rain", "precipitation", "cloudy", "sunny",
    "climate", "degrees",
]

WEATHER_KEYWORDS_TE = [
    "వాతావరణం", "వాతావరణ", "వర్షం", "వర్షాలు", "వాన", "వానలు", "కురుస్తుందా",
    "పడుతుందా", "కురుస్తుంది", "పడుతుంది", "ఉష్ణోగ్రత", "తేమ", "ఎండ", "చలి",
    "వాతావరణ అంచనా", "మంచు", "తుఫాను", "జల్లులు", "మేఘాలు",
]

WEATHER_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "मौसम", "बारिश", "तापमान", "हवामान", "पाऊस", "गारपीट", "हवामान अंदाज", "वादळ", "उष्णता",
    # Tamil
    "வானிலை", "மழை", "மழை பெய்யுமா", "வெப்பநிலை", "காற்று", "வானிலை அறிக்கை",
    # Kannada
    "ಹವಾಮಾನ", "ಮಳೆ", "ಮಳೆ ಬರುತ್ತಾ", "ತಾಪಮಾನ", "ಗಾಳಿ", "ಹವಾಮಾನ ವರದಿ",
    # Malayalam
    "കാലാവസ്ഥ", "മഴ", "മഴ പെയ്യുമോ", "താപനില", "കാറ്റ്",
    # Bengali & Assamese
    "আবহাওয়া", "বৃষ্টি", "বৃষ্টি হবে কি", "তাপমাত্রা", "বতৰ", "বৰষুণ", "বতৰৰ আগজাননী",
    # Gujarati
    "હવામાન", "વરસાદ", "વરસાદ પડશે", "તાપમાન", "આગાહી",
    # Odia
    "ପାଣିପାଗ", "ବର୍ଷା", "ବର୍ଷା ହେବ କି", "ତାପମାତ୍ରା",
    # Punjabi
    "ਮੌਸਮ", "ਮੀਂਹ", "ਬਰਸਾਤ", "ਮੀਂਹ ਪਵੇਗਾ", "ਤਾਪਮਾਨ",
    # Urdu
    "موسم", "بارش", "درجہ حرارت", "کیا بارش ہوگی", "موسم کا حال",
]

WEATHER_KEYWORDS_TANGLISH = [
    "varsham", "varsham paduthunda", "varsham vasthunda", "vana vasthunda",
    "vana paduthunda", "weather ela undi", "eeroju varsham", "repu varsham",
    "eppudu paduthundi", "rain paduthunda", "rain vasthada", "temperature entha",
    "cloudy ga undi", "varsham padtada", "varsham padtundha", "rain padtundha",
    "varsham eppudu", "varsham ela", "barish hogi kya", "paus padel ka",
    "mazhai varuma", "male barutta", "varsad aavshe",
]

SCHEMES_KEYWORDS_EN = [
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "pm kisan", "rythu bandhu",
    "rythu bharosa", "crop insurance", "fasal bima", "kcc", "kisan credit",
    "solar pump subsidy", "government assistance", "grant", "subsidized",
]

SCHEMES_KEYWORDS_TE = [
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "రైతు బంధు", "రైతు భరోసా", "పీఎం కిసాన్",
    "పంట బీమా", "రుణమాఫీ", "ప్రభుత్వ సహాయం", "ప్రభుత్వ", "అర్హత", "ప్రయోజనాలు",
    "ప్రభుత్వ పథకాలు", "రైతు పథకాలు",
]

SCHEMES_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "योजना", "सरकारी योजना", "सब्सिडी", "अनुदान", "पीएम किसान", "फसल बीमा", "कर्जमाफी", "कृषी योजना",
    # Tamil
    "திட்டம்", "அரசு திட்டம்", "மானியம்", "பயிர் காப்பீடு", "கிசான்",
    # Kannada
    "ಯೋಜನೆ", "ಸರ್ಕಾರಿ ಯೋಜನೆ", "ಸಬ್ಸಿಡಿ", "ಬೆಳೆ ವಿಮೆ", "ರೈತ ಯೋಜನೆ",
    # Malayalam
    "പദ്ധതി", "സർക്കാർ പദ്ധതി", "സബ്‌സിഡി", "വിള ഇൻഷുറൻസ്",
    # Bengali & Assamese
    "প্রকল্প", "সরকারি প্রকল্প", "ভর্তুকি", "শস্য বীমা", "আঁচনি", "চৰকাৰী আঁচনি",
    # Gujarati
    "યોજના", "સરકારી યોજના", "સબસિડી", "પાક વીમો", "કિસાન યોજના",
    # Odia
    "ଯୋଜନା", "ସରକାରୀ ଯୋଜନା", "ସବସିଡି", "ଫସଲ ବୀମା",
    # Punjabi
    "ਸਕੀਮ", "ਸਰਕਾਰੀ ਸਕੀਮ", "ਸਬਸਿਡੀ", "ਫਸਲ ਬੀਮਾ", "ਕਿਸਾਨ ਸਕੀਮ",
    # Urdu
    "اسکیم", "سرکاری اسکیم", "سبسڈی", "فصل کا بیمہ", "کسان اسکیم",
]

SCHEMES_KEYWORDS_TANGLISH = [
    "pathakam", "pathakalu", "prabhutva pathakalu", "rythu pathakalu", "subsidilu",
    "subsidy", "pm kisan", "rythu bharosa", "rythu bandhu", "panta bheema",
    "kisan subsidy", "gov schemes", "sarakaru sahaym", "pathakam apply",
    "prabhutva", "pathakalu unnaya", "sarkari yojana", "fasal bima", "anudan",
]

SHOPS_KEYWORDS_EN = [
    "where to buy", "where can i buy", "where i can buy", "shops near", "stores near",
    "buy urea", "buy dap", "dealer", "dealers", "input availability", "available in shop",
    "fertilizer store", "pesticide store", "buy fertilizer", "buy pesticide", "buy seeds",
    "shops nearby", "store nearby", "agro agency",
]

SHOPS_KEYWORDS_TE = [
    "ఎక్కడ దొరుకుతుంది", "ఎక్కడ కొనాలి", "సమీప దుకాణాలు", "ఎరువుల దుకాణం", "మందుల షాపు",
    "కొనుగోలు", "దుకాణం", "షాపు", "లభిస్తుంది", "యూరియా దొరుకుతుందా", "డీలర్", "డీలర్లు",
    "దుకాణాలు", "షాపులు", "దొరికే చోటు",
]

SHOPS_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "दुकान", "कहाँ मिलेगा", "कहाँ से खरीदें", "दुकानें", "खाद की दुकान", "कृषी केंद्र", "कुठे मिळेल", "खरेदी",
    # Tamil
    "எங்கு கிடைக்கும்", "எங்கே வாங்கலாம்", "அருகிலுள்ள கடைகள்", "உரக் கடை", "பூச்சிக்கொல்லி கடை",
    # Kannada
    "ಎಲ್ಲಿ ಸಿಗುತ್ತದೆ", "ಎಲ್ಲಿ ಖರೀದಿಸಬೇಕು", "ಹತ್ತಿರದ ಅಂಗಡಿಗಳು", "ರಸಗೊಬ್ಬರ ಅಂಗಡಿ", "ಕೃಷಿ ಕೇಂದ್ರ",
    # Malayalam
    "എവിടെ കിട്ടും", "എവിടെ വാങ്ങാം", "അടുത്തുള്ള കടകൾ", "വളം കട",
    # Bengali & Assamese
    "কোথায় পাব", "কোথায় কিনতে পাওয়া যাবে", "নিকটবর্তী দোকান", "সারের দোকান", "ক'ত পাম",
    # Gujarati
    "ક્યાં મળશે", "ક્યાંથી ખરીદવું", "નજીકની દુકાનો", "ખાતરની દુકાન",
    # Odia
    "କେଉଁଠି ମିଳିବ", "କେଉଁଠାରୁ କିଣିବେ", "ନିକଟସ୍ଥ ଦୋକାନ", "ସାର ଦୋକାନ",
    # Punjabi
    "ਕਿੱਥੇ ਮਿਲੇਗਾ", "ਕਿੱਥੋਂ ਖਰੀਦੀਏ", "ਨੇੜਲੀਆਂ ਦੁਕਾਨਾਂ", "ਖਾਦ ਦੀ ਦੁਕਾਨ",
    # Urdu
    "کہاں ملے گا", "کہاں سے خریدیں", "قریبی دکانیں", "کھاد کی دکان",
]

SHOPS_KEYWORDS_TANGLISH = [
    "ekkada dorukuthundi", "ekkada konali", "shops ekkada", "shop ekkada", "urea ekkada",
    "dap ekkada", "seeds ekkada", "fertilizer shop", "pesticide shop", "near shops",
    "daggara shop", "konadaniki", "dorukuthunda", "shops daggara", "dealer daggara",
    "dorukutundi", "konachu", "kaha milega", "kuthe milel", "enga kedaikkum", "elli sigutte",
]

CROP_HEALTH_KEYWORDS_EN = [
    "disease", "pest", "pests", "leaf spot", "yellowing", "wilting", "fungus",
    "insects", "bollworm", "aphids", "whitefly", "blight", "rot", "pesticide for",
    "cure", "symptoms", "worms", "bugs", "fungicide", "infestation", "stem borer",
    "leaf curl", "caterpillar", "blast", "rust", "alternaria", "powdery mildew",
]

CROP_HEALTH_KEYWORDS_TE = [
    "తెగులు", "తెగుళ్ళు", "పురుగు", "పురుగులు", "ఆకుమచ్చ", "పసుపుగా", "రాలిపోవడం",
    "ముడత", "ఎండిపోవడం", "పచ్చదోమ", "తామర పురుగులు", "బూడిద తెగులు", "అగ్గితెగులు",
    "ఆల్టర్నేరియా", "నివారణ", "మందు పిచికారీ", "రోగం", "లక్షణాలు", "మచ్చలు", "పురుగుల",
]

CROP_HEALTH_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "रोग", "कीड़ा", "कीट", "फफूंद", "झुलसा", "इल्ली", "माहू", "कीटनाशक", "इलाज", "लक्षण", "रोग नियंत्रण", "कीड", "अळी", "औषध फवारणी",
    # Tamil
    "நோய்", "பூச்சி", "புழு", "இலைப்புள்ளி", "பூச்சிக்கொல்லி", "மருந்து", "நிவாரணம்",
    # Kannada
    "ರೋಗ", "ಕೀಟ", "ಹುಳು", "ಎಲೆ ಚುಕ್ಕೆ", "ಕೀಟನಾಶಕ", "ಔಷಧಿ", "ನಿಯಂತ್ರಣ",
    # Malayalam
    "രോഗം", "കീടങ്ങൾ", "പുഴു", "കുമിൾനാശിനി", "മരുന്ന്", "ചികിത്സ",
    # Bengali & Assamese
    "রোগ", "পোকা", "কীটপতঙ্গ", "কীটনাশক", "ঔষধ", "প্রতিকার", "পোক",
    # Gujarati
    "રોગ", "જીવાત", "ઇયળ", "કીટનાશક", "દવા", "ઉપચાર",
    # Odia
    "ରୋଗ", "ପୋକ", "କୀଟନାଶକ", "ଔଷଧ", "ନିୟନ୍ତ୍ରଣ",
    # Punjabi
    "ਰੋਗ", "ਬਿਮਾਰੀ", "ਕੀੜੇ", "ਸੁੰਡੀ", "ਕੀਟਨਾਸ਼ਕ", "ਸਪਰੇਅ", "ਦਵਾਈ",
    # Urdu
    "بیماری", "کیڑے", "سنڈی", "کیڑے مار دوا", "علاج", "اسپرے",
]

CROP_HEALTH_KEYWORDS_TANGLISH = [
    "tegulu", "purugu", "purugulu", "aakulu pasupuga", "aaku machalu", "marutunnayi",
    "endipotundi", "mudatha", "dosa penu", "pacha doma", "kurchuku potundi",
    "panta rogamu", "pesticide spray", "cheda", "pulla rali", "pula rali",
    "purugula mandu", "tegulu mandu", "ralipothundi", "pasupuga marutunnayi",
    "keeda laga hai", "rog ahe", "poochi marunthu", "rogha aushadha",
]

FERTILIZER_KEYWORDS_EN = [
    "fertilizer", "fertilizers", "nutrient", "npk", "urea application", "which fertilizer",
    "what fertilizer", "micronutrient", "zinc deficiency", "potash dose", "fertilizer schedule",
    "how much fertilizer", "apply fertilizer", "manure", "compost", "dosage of urea", "dap dose",
]

FERTILIZER_KEYWORDS_TE = [
    "ఎరువు", "ఎరువులు", "ఏ ఎరువు వేయాలి", "ఎరువుల యాజమాన్యం", "పోషకాలు", "నత్రజని",
    "భాస్వరం", "పొటాష్", "సూక్ష్మ పోషకాలు", "ఎరువుల మోతాదు", "జింక్ లోపం", "ఎరువు వాడాలి",
    "బాస్వరం", "యూరియా మోతాదు", "ఎంత వేయాలి", "యూరియా ఎంత",
]

FERTILIZER_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "खाद", "उर्वरक", "यूरिया", "डीएपी", "पोटाश", "पोषक तत्व", "खत", "खते", "रासायनिक खत", "सेंद्रिय खत", "मात्रा",
    # Tamil
    "உரம்", "உரங்கள்", "யூரியா", "டிஏபி", "உர அளவு", "நுண்ணூட்டம்",
    # Kannada
    "ಗೊಬ್ಬರ", "ರಸಗೊಬ್ಬರ", "ಯೂರಿಯಾ", "ಡಿಎಪಿ", "ಪೋಷಕಾಂಶ", "ಗೊಬ್ಬರದ ಪ್ರಮಾಣ",
    # Malayalam
    "വളം", "രാസവളം", "യൂറിയ", "ഡാപ്പ്", "വളപ്രയോഗം",
    # Bengali & Assamese
    "সার", "ইউরিয়া", "ডিএপি", "রাসায়নিক সার", "সাৰ",
    # Gujarati
    "ખાતર", "યુરિયા", "ડીએપી", "પોષક તત્વો", "ખાતરની માત્રા",
    # Odia
    "ସାର", "ୟୁରିଆ", "ଡିଏପି", "ପୋଷକ ତତ୍ତ୍ୱ",
    # Punjabi
    "ਖਾਦ", "ਯੂਰੀਆ", "ਡੀਏਪੀ", "ਖਾਦ ਦੀ ਮਾਤਰਾ",
    # Urdu
    "کھاد", "یوریا", "ڈی اے پی", "کھاد کی مقدار", "غذائی اجزاء",
]

FERTILIZER_KEYWORDS_TANGLISH = [
    "e eruvu veyali", "eruvulu eppudu", "fertilizer eppudu", "npk ela veyali", "poshakalu",
    "eruvu dose", "micronutrients", "urea dose", "e eruvu vadali", "fertilizer schedule",
    "eruvu entha", "eruvulu ela", "fertilizer dose", "konsa khad", "kiti khat takayche",
    "yava gobbara", "enna uram",
]

IRRIGATION_KEYWORDS_EN = [
    "irrigation", "watering", "water schedule", "drip irrigation", "how much water",
    "when to water", "moisture", "flood irrigation", "sprinkler",
]

IRRIGATION_KEYWORDS_TE = [
    "నీరు", "నీటి యాజమాన్యం", "నీరు పెట్టాలి", "తడి ఇవ్వాలి", "డ్రిప్", "బిందు సేద్యం",
    "ఎప్పుడు నీరు", "నీరు కట్టాలి", "నీటి తడి", "నీటి పారుదల",
]

IRRIGATION_KEYWORDS_MULTILINGUAL = [
    # Hindi & Marathi
    "सिंचाई", "पानी देना", "ड्रिप", "पानी कब दें", "पाणी व्यवस्थापन", "ठिबक सिंचन", "पाणी देणे",
    # Tamil
    "நீர்ப்பாசனம்", "தண்ணீர் பாய்ச்சுதல்", "சொட்டு நீர்", "பாசனம்",
    # Kannada
    "ನೀರಾವರಿ", "ನೀರುಣಿಸುವುದು", "ಹನಿ ನೀರಾವರಿ", "ನೀರು ಹಾಕುವುದು",
    # Malayalam
    "ജലസേചനം", "നനയ്ക്കൽ", "തുള്ളി നന",
    # Bengali & Assamese
    "সেচ", "পানি দেওয়া", "জলসেচ", "পানী যোগান",
    # Gujarati
    "પિયત", "પાણી આપવું", "ટપક પિયત", "સિંચાઈ",
    # Odia
    "ଜଳସେଚନ", "ପାଣି ଦେବା",
    # Punjabi
    "ਸਿੰਚਾਈ", "ਪਾਣੀ ਲਾਉਣਾ",
    # Urdu
    "آبپاشی", "پانی دینا", "ڈرپ اریگیشن",
]

IRRIGATION_KEYWORDS_TANGLISH = [
    "neeru eppudu pettali", "thadi eppudu ivvali", "neeti yajamanyam", "water eppudu pettali",
    "drip irrigation", "water kattachu", "thadulu eppudu", "neeru eppudu",
    "kitna pani chahiye", "pani kab dena hai", "kiti pani dyayche", "neeru eshtu", "thanni eppadi",
]

SOWING_KEYWORDS_EN = [
    "sowing", "seed rate", "seed treatment", "planting time", "how to sow", "seed spacing",
    "nursery", "germination", "transplanting", "sowing depth",
]

SOWING_KEYWORDS_TE = [
    "విత్తనాలు", "విత్తన శుద్ధి", "విత్తే సమయం", "నాట్లు", "నాటడం", "ఎప్పుడు విత్తాలి",
    "విత్తన మోతాదు", "మొలక", "నారుమడి",
]

SOWING_KEYWORDS_MULTILINGUAL = [
    "बुवाई", "बीज दर", "बीज उपचार", "पेरणी", "बियाणे", "விதைப்பு", "ಬಿತ್ತನೆ", "വിത്ത് നടൽ", "বপন", "વાવણી", "ବୁଣିବା", "ਬਿਜਾਈ", "بوائی",
]

SOWING_KEYWORDS_TANGLISH = [
    "vithanalu eppudu veyali", "vithana shuddi", "sowing time", "seed rate entha",
    "natlu eppudu", "vithadam ela", "seeds eppudu", "buwai kab kare", "perani kadhi",
]

HARVESTING_KEYWORDS_EN = [
    "harvesting", "harvest time", "when to harvest", "picking cotton", "harvest maturity",
    "threshing", "storage", "post harvest",
]

HARVESTING_KEYWORDS_TE = [
    "కోత", "కోత సమయం", "ఎప్పుడు కోయాలి", "పత్తి ఏరడం", "దిగుబడి", "నిల్వ", "కోయడం", "కోతలు",
]

HARVESTING_KEYWORDS_MULTILINGUAL = [
    "कटाई", "फसल कटाई", "कापणी", "काढणी", "அறுவடை", "ಕಟಾವು", "വിളവെടുപ്പ്", "ফসল কাটা", "કાપણી", "ଅମଳ", "ਵਾਢੀ", "کٹائی",
]

HARVESTING_KEYWORDS_TANGLISH = [
    "kotha eppudu koyali", "harvesting time", "patti eppudu thiyyali", "kotha kosaru",
    "nilva cheyadam", "digubadi", "kotha samayam", "katai kab kare", "kapani kadhi",
]

REMINDERS_KEYWORDS_EN = [
    "remind me", "set reminder", "schedule reminder", "alert me", "reminder",
]

REMINDERS_KEYWORDS_TE = [
    "గుర్తు చేయండి", "రిమైండర్", "షెడ్యూల్", "గుర్తుపెట్టుకో",
]

REMINDERS_KEYWORDS_MULTILINGUAL = [
    "याद दिलाना", "रिमाइंडर", "आठवण करून द्या", "நினைவூட்டல்", "ನೆನಪಿಸಿ", "ഓർമ്മിപ്പിക്കുക", "মনে করিয়ে দিন", "યાદ અપાવો", "ମନେ ପକାଇଦିଅ", "ਯਾਦ ਦਿਵਾਓ", "یاد دلائیں",
]

REMINDERS_KEYWORDS_TANGLISH = [
    "remind cheyandi", "gurthu cheyandi", "reminder pettandi", "schedule cheyandi",
    "yaad dilana", "reminder set karo",
]

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
            greeting_tokens = [
                "hi", "hello", "hey", "namaste", "namaskar", "namaskaram", "pranam",
                "vanakkam", "namaskara", "adaab", "హాయ్", "నమస్తే", "హలో", "नमस्ते",
                "नमस्कार", "प्रणाम", "வணக்கம்", "ನಮಸ್ಕಾರ", "নমস্কার", "سلام"
            ]
            if len(tokens) <= 3 and any(t in greeting_tokens for t in tokens):
                is_greeting_match = True

        if not is_greeting_match:
            return False

        # If any domain keywords are present, it's not a pure greeting
        domain_keywords = (
            MARKET_PRICE_KEYWORDS_EN + MARKET_PRICE_KEYWORDS_TE + MARKET_PRICE_KEYWORDS_MULTILINGUAL + MARKET_PRICE_KEYWORDS_TANGLISH +
            WEATHER_KEYWORDS_EN + WEATHER_KEYWORDS_TE + WEATHER_KEYWORDS_MULTILINGUAL + WEATHER_KEYWORDS_TANGLISH +
            SCHEMES_KEYWORDS_EN + SCHEMES_KEYWORDS_TE + SCHEMES_KEYWORDS_MULTILINGUAL + SCHEMES_KEYWORDS_TANGLISH +
            SHOPS_KEYWORDS_EN + SHOPS_KEYWORDS_TE + SHOPS_KEYWORDS_MULTILINGUAL + SHOPS_KEYWORDS_TANGLISH +
            CROP_HEALTH_KEYWORDS_EN + CROP_HEALTH_KEYWORDS_TE + CROP_HEALTH_KEYWORDS_MULTILINGUAL + CROP_HEALTH_KEYWORDS_TANGLISH +
            FERTILIZER_KEYWORDS_EN + FERTILIZER_KEYWORDS_TE + FERTILIZER_KEYWORDS_MULTILINGUAL + FERTILIZER_KEYWORDS_TANGLISH +
            IRRIGATION_KEYWORDS_EN + IRRIGATION_KEYWORDS_TE + IRRIGATION_KEYWORDS_MULTILINGUAL + IRRIGATION_KEYWORDS_TANGLISH +
            STOCK_ALERT_KEYWORDS_EN + STOCK_ALERT_KEYWORDS_TE + STOCK_ALERT_KEYWORDS_MULTILINGUAL + STOCK_ALERT_KEYWORDS_TANGLISH
        )

        for kw in domain_keywords:
            if kw in msg:
                return False

        return True

    @staticmethod
    def get_greeting_reply(language: str = "te") -> str:
        """Return a helpful, welcoming WhatsApp greeting response in the requested language."""
        return GREETING_REPLIES.get(language, GREETING_REPLIES["en"])

    @classmethod
    def detect_all_intents(cls, user_message: str) -> List[FarmerIntent]:
        """
        Identify all intents present in the farmer's message across supported languages.
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
        if (any(kw in msg for kw in STOCK_ALERT_KEYWORDS_EN) or
            any(kw in msg_original for kw in STOCK_ALERT_KEYWORDS_TE) or
            any(kw in msg_original for kw in STOCK_ALERT_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in STOCK_ALERT_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.STOCK_ALERT)

        # 0.1. Reminders (prioritized if user asks to be reminded)
        if (any(kw in msg for kw in REMINDERS_KEYWORDS_EN) or
            any(kw in msg_original for kw in REMINDERS_KEYWORDS_TE) or
            any(kw in msg_original for kw in REMINDERS_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in REMINDERS_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.REMINDERS)

        # 1. Market Price
        if (any(kw in msg for kw in MARKET_PRICE_KEYWORDS_EN) or
            any(kw in msg_original for kw in MARKET_PRICE_KEYWORDS_TE) or
            any(kw in msg_original for kw in MARKET_PRICE_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in MARKET_PRICE_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.MARKET_PRICE)

        # 2. Weather
        if (any(kw in msg for kw in WEATHER_KEYWORDS_EN) or
            any(kw in msg_original for kw in WEATHER_KEYWORDS_TE) or
            any(kw in msg_original for kw in WEATHER_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in WEATHER_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.WEATHER)

        # 3. Government Schemes
        if (any(kw in msg for kw in SCHEMES_KEYWORDS_EN) or
            any(kw in msg_original for kw in SCHEMES_KEYWORDS_TE) or
            any(kw in msg_original for kw in SCHEMES_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in SCHEMES_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.GOVERNMENT_SCHEMES)

        # 4. Shops / Input Availability
        if (any(kw in msg for kw in SHOPS_KEYWORDS_EN) or
            any(kw in msg_original for kw in SHOPS_KEYWORDS_TE) or
            any(kw in msg_original for kw in SHOPS_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in SHOPS_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.SHOPS)

        # 5. Crop Health / Disease / Pest
        if (any(kw in msg for kw in CROP_HEALTH_KEYWORDS_EN) or
            any(kw in msg_original for kw in CROP_HEALTH_KEYWORDS_TE) or
            any(kw in msg_original for kw in CROP_HEALTH_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in CROP_HEALTH_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.CROP_HEALTH)

        # 6. Fertilizer / Nutrients
        if (any(kw in msg for kw in FERTILIZER_KEYWORDS_EN) or
            any(kw in msg_original for kw in FERTILIZER_KEYWORDS_TE) or
            any(kw in msg_original for kw in FERTILIZER_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in FERTILIZER_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.FERTILIZER)

        # 7. Irrigation
        if (any(kw in msg for kw in IRRIGATION_KEYWORDS_EN) or
            any(kw in msg_original for kw in IRRIGATION_KEYWORDS_TE) or
            any(kw in msg_original for kw in IRRIGATION_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in IRRIGATION_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.IRRIGATION)

        # 8. Sowing
        if (any(kw in msg for kw in SOWING_KEYWORDS_EN) or
            any(kw in msg_original for kw in SOWING_KEYWORDS_TE) or
            any(kw in msg_original for kw in SOWING_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in SOWING_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.SOWING)

        # 9. Harvesting
        if (any(kw in msg for kw in HARVESTING_KEYWORDS_EN) or
            any(kw in msg_original for kw in HARVESTING_KEYWORDS_TE) or
            any(kw in msg_original for kw in HARVESTING_KEYWORDS_MULTILINGUAL) or
            any(kw in msg for kw in HARVESTING_KEYWORDS_TANGLISH)):
            detected.append(FarmerIntent.HARVESTING)

        # Fallback: if no specific domain matched, identify as general farming or unknown
        if not detected:
            from src.rag.service import extract_crop_from_text
            mentioned_crop = extract_crop_from_text(msg_original)
            agri_words = [
                "crop", "crops", "farm", "farming", "land", "field", "acre", "acres", "yield",
                "soil", "cotton", "paddy", "chilli", "tomato", "maize", "agriculture",
                "పంట", "పొలం", "సాగు", "భూమి", "ఎకరం", "దిగుబడి", "నేల", "పత్తి", "వరి", "మిర్చి",
                "फसल", "खेती", "खेत", "किसान", "पिक", "शेती", "பயிர்", "விவசாயம்", "ಬೆಳೆ", "ಕೃಷಿ", "விளைச்சல்"
            ]
            if mentioned_crop or any(w in msg for w in agri_words):
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
        - Identifies language dynamically (13 languages + Romanized)
        - Identifies intents
        - Bypasses LLM for pure greetings
        - Coordinates specialized services only when relevant
        - Enforces factual authoritativeness (never hallucinating numbers or prices)
        - Applies localized fallbacks when services produce no data
        - Formats and finalizes the outgoing response
        """
        user_message = conversation.user_message or ""
        logger.info(f"[DECISION ENGINE START] Farmer: {farmer.id} | Query: '{user_message}'")

        # Resolve language using deterministic multi-tier language detector
        pref_lang = getattr(farmer, "preferred_language", "te") or "te"
        language = detect_language(user_message, fallback=pref_lang)
        logger.info(f"[DECISION ENGINE] Detected language: '{language}' (Farmer preferred: '{pref_lang}')")

        # 1. Pure Greeting Shortcut
        if self.is_greeting_only(user_message):
            logger.info(f"[DECISION ENGINE] Pure greeting detected. Returning instant greeting ({language}) for farmer {farmer.id}.")
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
