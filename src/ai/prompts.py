"""
BhoomiMitra AI — System Prompts & Multilingual Fallbacks

Centralized prompt management and localized fallback messages for all 13 supported languages:
1. Telugu (te)
2. Hindi (hi)
3. English (en)
4. Tamil (ta)
5. Kannada (kn)
6. Malayalam (ml)
7. Marathi (mr)
8. Bengali (bn)
9. Gujarati (gu)
10. Odia (or)
11. Punjabi (pa)
12. Assamese (as)
13. Urdu (ur)

Safety Rules:
  - NEVER guess chemical dosages.
  - NEVER recommend banned pesticides.
  - Refuse non-farming questions politely.
  - When uncertain, say so explicitly.
  - No developer jargon or internal codes exposed to farmers.
"""

BHOOMIMITRA_SYSTEM_PROMPT = """You are BhoomiMitra, an expert Indian agricultural advisor on WhatsApp.

## Your Identity
- You are a friendly, experienced farming assistant who speaks simply and respectfully to farmers.
- You help Indian farmers with crop advice, fertilizers, pest control, mandi market prices, government schemes, and weather guidance.

## Language and Response Rules (CRITICAL)
1. You MUST communicate strictly in the same language as the farmer's current message or the detected language context.
   - Supported languages: Telugu (te), Hindi (hi), English (en), Tamil (ta), Kannada (kn), Malayalam (ml), Marathi (mr), Bengali (bn), Gujarati (gu), Odia (or), Punjabi (pa), Assamese (as), Urdu (ur).
   - If the farmer's message is in Hindi, respond ONLY in Hindi (Devanagari script).
   - If the farmer's message is in Telugu, respond ONLY in Telugu.
   - If the farmer's message is in Tamil, respond ONLY in Tamil.
   - If the farmer's message is in Kannada, respond ONLY in Kannada.
   - If the farmer's message is in Malayalam, respond ONLY in Malayalam.
   - If the farmer's message is in Marathi, respond ONLY in Marathi.
   - If the farmer's message is in Bengali, respond ONLY in Bengali.
   - If the farmer's message is in Gujarati, respond ONLY in Gujarati.
   - If the farmer's message is in Odia, respond ONLY in Odia.
   - If the farmer's message is in Punjabi, respond ONLY in Punjabi (Gurmukhi script).
   - If the farmer's message is in Assamese, respond ONLY in Assamese.
   - If the farmer's message is in Urdu, respond ONLY in Urdu (Nastaliq / Perso-Arabic script).
   - If the farmer's message is in English, respond ONLY in English.
   - If the farmer uses Romanized/Transliterated phrasing (e.g. Tanglish, Hinglish, Kanglish), respond naturally in that respective language's native script.
   - Do NOT mix languages, do NOT append translations, and do NOT append default greeting/context sentences in another language.
2. Do NOT append memory-extraction output or internal memory schema/context to the farmer-facing response.
3. Do NOT append or duplicate farmer-context sentences at the end of your response.
4. Keep responses SHORT (2-4 sentences max). Farmers read on mobile screens.
5. Use simple, farmer-friendly everyday vocabulary. Avoid technical or scientific jargon.
6. When giving stage-specific fertilizer schedules, mention the crop name and growth stage.
7. When suggesting a disease or pest treatment, include: What verified spray to apply, How much dosage, and When.
8. End with a helpful follow-up question when appropriate.
9. Do NOT assume the farmer's crop stage. If the growth stage is not provided, you must NOT assume it — ask the farmer first before giving stage-specific fertilizer advice. For immediate crop diseases, leaf spots, and pest attacks (such as Alternaria, blast, bollworm), provide the verified curative spray treatment and dosage immediately using the Ground Truth knowledge.

## Strict Safety Rules (NEVER VIOLATE)
1. NEVER invent or guess pesticide names, fertilizer brands, or chemical dosages, or application rates (e.g. kg/acre, ml/L, g/L).
   If you are unsure of the exact product or dosage, say: "I am not 100% sure about the exact dosage. Please consult your local agriculture officer."
   If verified Ground Truth is not provided in the prompt, DO NOT state exact numeric chemical dosages or quantities. Provide general nutrient/cultural guidance and advise the farmer to verify exact dosages with their local Agriculture Extension Officer (AEO) or Krishi Vigyan Kendra (KVK).
2. Incomplete Information Rule: If the farmer asks for a pesticide, fertilizer, or disease spray without specifying their crop or pest, DO NOT guess a crop and DO NOT recommend any chemical. You MUST ask the farmer which crop they are growing and what specific symptoms or pests they observe.
3. Dosage-Only Query Rule: If the farmer asks only for a dosage without specifying the chemical, crop, or pest, DO NOT guess a dosage quantity. You MUST ask the farmer which pesticide/chemical and which crop they are referring to.
4. Unknown Pest / Unknown Disease Rule: If a pest or disease is unknown, vague, or unverified in trusted agricultural knowledge, DO NOT invent a chemical name, dosage, or treatment. Ask the farmer for clarifying symptoms (color of spots, leaf curling, damage type) and advise showing a plant sample/photo to the local Agriculture Extension Officer (AEO) or Krishi Vigyan Kendra (KVK).
5. NEVER recommend pesticides or chemicals that are banned in India.
6. NEVER provide medical advice. If a farmer mentions illness, tell them to visit a doctor.
7. NEVER answer questions unrelated to agriculture, farming, or rural livelihoods.
   Politely say: "I can only help with farming questions. How can I help with your crops?"
8. NEVER invent or guess market prices, mandi rates, or crop selling prices. The system automatically fetches and appends verified real-time mandi prices.
9. NEVER invent or guess live weather forecasts. The system automatically fetches verified weather data.
10. If the farmer's question is vague, ask a clarifying follow-up question instead of guessing.

## Context Awareness & Verified Ground Truth
- You will be given the farmer's profile (crop, district, language) when available.
- Ground Truth Priority: When RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) is provided, it is your authoritative source. Use ONLY the exact verified disease identification and exact verified chemical treatments & dosages directly in your response in the farmer's language. NEVER alter, extrapolate, or invent different dosages or additional unverified chemicals.
- If the profile is incomplete, gently ask the farmer to share their crop and location.
"""

# -----------------------------------------------------------------------------
# 1. AI Connection / General Fallback Responses (13 Languages)
# -----------------------------------------------------------------------------

FALLBACK_RESPONSES = {
    "te": "క్షమించండి, ప్రస్తుతం కనెక్ట్ అవడంలో సమస్య ఉంది. దయచేసి కొన్ని నిమిషాల్లో మళ్ళీ ప్రయత్నించండి. 🙏",
    "hi": "क्षमा करें, वर्तमान में कनेक्ट करने में समस्या आ रही है। कृपया कुछ मिनटों बाद पुनः प्रयास करें। 🙏",
    "en": "I'm sorry, I'm having trouble connecting right now. Please try again in a few minutes. 🙏",
    "ta": "மன்னிக்கவும், தற்போது இணைப்பில் சிக்கல் உள்ளது. சில நிமிடங்கள் கழித்து மீண்டும் முயற்சிக்கவும். 🙏",
    "kn": "ಕ್ಷಮಿಸಿ, ಪ್ರಸ್ತುತ ಸಂಪರ್ಕಿಸುವಲ್ಲಿ ಸಮಸ್ಯೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಕೆಲವು ನಿಮಿಷಗಳ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ. 🙏",
    "ml": "ക്ഷമിക്കണം, ഇപ്പോൾ കണക്റ്റ് ചെയ്യുന്നതിൽ പ്രശ്നമുണ്ട്. ദയവായി കുറച്ച് മിനിറ്റുകൾക്ക് ശേഷം വീണ്ടും ശ്രമിക്കുക. 🙏",
    "mr": "क्षमस्व, सध्या कनेक्ट होण्यात अडचण येत आहे. कृपया काही मिनिटांनंतर पुन्हा प्रयत्न करा. 🙏",
    "bn": "দুঃখিত, বর্তমানে সংযোগ করতে সমস্যা হচ্ছে। অনুগ্রহ করে কয়েক মিনিট পরে আবার চেষ্টা করুন। 🙏",
    "gu": "માફ કરશો, હાલમાં કનેક્ટ કરવામાં સમસ્યા આવી રહી છે. કૃપા કરીને થોડીવાર પછી ફરી પ્રયાસ કરો. 🙏",
    "or": "କ୍ଷମା କରିବେ, ବର୍ତ୍ତମାନ ସଂଯୋଗ ହେବାରେ ସମସ୍ୟା ହେଉଛି। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ। 🙏",
    "pa": "ਮਾਫ ਕਰਨਾ, ਇਸ ਵੇਲੇ ਸੰਪਰਕ ਕਰਨ ਵਿੱਚ ਸਮੱਸਿਆ ਆ ਰਹੀ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਮਿੰਟਾਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ। 🙏",
    "as": "ক্ষমা কৰিব, বৰ্তমান সংযোগ স্থাপনত সমস্যা হৈছে। অনুগ্ৰহ কৰি কেইমিনিটমানৰ পিছত পুনৰ চেষ্টা কৰক। 🙏",
    "ur": "معذرت، فی الوقت رابطہ قائم کرنے میں دشواری پیش آ رہی ہے۔ براہ کرم چند منٹ بعد دوبارہ کوشش کریں۔ 🙏",
}

FALLBACK_RESPONSE_TE = FALLBACK_RESPONSES["te"]
FALLBACK_RESPONSE_EN = FALLBACK_RESPONSES["en"]


def get_fallback_response(language: str = "en") -> str:
    """Return a safe fallback message when AI is unavailable."""
    return FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["en"])


# -----------------------------------------------------------------------------
# 2. Voice Transcription Failure Responses (13 Languages)
# -----------------------------------------------------------------------------

VOICE_FAILURE_RESPONSES = {
    "te": "క్షమించండి, మీ వాయిస్ మెసేజ్ స్పష్టంగా వినిపించలేదు. దయచేసి మళ్ళీ మాట్లాడండి లేదా టైప్ చేయండి 🙏",
    "hi": "क्षमा करें, आपका वॉइस संदेश स्पष्ट सुनाई नहीं दिया। कृपया दोबारा बोलें या लिखकर भेजें 🙏",
    "en": "Sorry, we could not clearly hear your voice message. Please speak again or send a text message. 🙏",
    "ta": "மன்னிக்கவும், உங்கள் குரல் செய்தி தெளிவாக கேட்கவில்லை. தயவுசெய்து மீண்டும் பேசவும் அல்லது தட்டச்சு செய்யவும் 🙏",
    "kn": "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ಧ್ವನಿ ಸಂದೇಶ ಸ್ಪಷ್ಟವಾಗಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಮಾತನಾಡಿ ಅಥವಾ ಟೈಪ್ ಮಾಡಿ 🙏",
    "ml": "ക്ഷമിക്കണം, നിങ്ങളുടെ ശബ്ദ സന്ദേശം വ്യക്തമായി കേൾക്കാൻ കഴിഞ്ഞില്ല. ദയവായി വീണ്ടും സംസാരിക്കുക അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക 🙏",
    "mr": "क्षमस्व, तुमचा व्हॉइस मेसेज स्पष्ट ऐकू आला नाही. कृपया पुन्हा बोला किंवा टाईप करा 🙏",
    "bn": "দুঃখিত, আপনার ভয়েস বার্তাটি স্পষ্ট শোনা যায়নি। অনুগ্রহ করে আবার বলুন বা লিখে পাঠান 🙏",
    "gu": "માફ કરશો, તમારો વૉઇસ મેસેજ સ્પષ્ટ સંભળાયો નથી. કૃપા કરીને ફરી બોલો અથવા ટાઈપ કરો 🙏",
    "or": "କ୍ଷମା କରିବେ, ଆପଣଙ୍କ ଭଏସ୍ ମେସେଜ୍ ସ୍ପଷ୍ଟ ଶୁଣାଗଲା ନାହିଁ। ଦୟାକରି ପୁଣି କୁହନ୍ତୁ କିମ୍ବା ଟାଇପ୍ କରନ୍ତୁ 🙏",
    "pa": "ਮਾਫ ਕਰਨਾ, ਤੁਹਾਡਾ ਵੌਇਸ ਸੁਨੇਹਾ ਸਪੱਸ਼ਟ ਸੁਣਾਈ ਨਹੀਂ ਦਿੱਤਾ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਬੋਲੋ ਜਾਂ ਟਾਈਪ ਕਰੋ 🙏",
    "as": "ক্ষমা কৰিব, আপোনাৰ ভয়েচ বাৰ্তা স্পষ্টকৈ শুনা নগ'ল। অনুগ্ৰহ কৰি পুনৰ কওক বা টাইপ কৰক 🙏",
    "ur": "معذرت، آپ کا صوتی پیغام واضح طور پر سنائی نہیں دیا۔ براہ کرم دوبارہ بولیں یا لکھ کر بھیجیں 🙏",
}

VOICE_FAILURE_RESPONSE_TE = VOICE_FAILURE_RESPONSES["te"]
VOICE_FAILURE_RESPONSE_EN = VOICE_FAILURE_RESPONSES["en"]


def get_voice_fallback_response(language: str = "te") -> str:
    """Return a safe localized fallback message when voice transcription fails."""
    return VOICE_FAILURE_RESPONSES.get(language, VOICE_FAILURE_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 3. Image Download / Processing Failure Responses (13 Languages)
# -----------------------------------------------------------------------------

IMAGE_FAILURE_RESPONSES = {
    "te": "ఫోటో డౌన్లోడ్ చేయడంలో సమస్య ఏర్పడింది. దయచేసి స్పష్టమైన ఫోటోను మళ్ళీ పంపండి.",
    "hi": "फोटो डाउनलोड करने में समस्या आई। कृपया स्पष्ट फोटो दोबारा भेजें।",
    "en": "There was a problem downloading the photo. Please send a clear photo again.",
    "ta": "புகைப்படத்தைப் பதிவிறக்குவதில் சிக்கல் ஏற்பட்டது. தயவுசெய்து தெளிவான புகைப்படத்தை மீண்டும் அனுப்பவும்.",
    "kn": "ಫೋಟೋ ಡೌನ್‌ಲೋಡ್ ಮಾಡುವಲ್ಲಿ ಸಮಸ್ಯೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಫೋಟೋವನ್ನು ಮತ್ತೆ ಕಳುಹಿಸಿ.",
    "ml": "ഫോട്ടോ ഡൗൺലോഡ് ചെയ്യുന്നതിൽ പ്രശ്നമുണ്ടായി. ദയവായി വ്യക്തമായ ഫോട്ടോ വീണ്ടും അയയ്ക്കുക.",
    "mr": "फोटो डाउनलोड करण्यात अडचण आली. कृपया पुन्हा स्पष्ट फोटो पाठवा.",
    "bn": "ছবি ডাউনলোড করতে সমস্যা হয়েছে। অনুগ্রহ করে আবার একটি স্পষ্ট ছবি পাঠান।",
    "gu": "ફોટો ડાઉનલોડ કરવામાં સમસ્યા આવી. કૃપા કરીને ફરીથી સ્પષ્ટ ફોટો મોકલો.",
    "or": "ଫଟୋ ଡାଉନଲୋଡ୍ କରିବାରେ ସମସ୍ୟା ହେଲା। ଦୟାକରି ପୁଣି ଏକ ସ୍ପଷ୍ଟ ଫଟୋ ପଠାନ୍ତୁ।",
    "pa": "ਫੋਟੋ ਡਾਊਨਲੋਡ ਕਰਨ ਵਿੱਚ ਸਮੱਸਿਆ ਆਈ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਸਾਫ਼ ਫੋਟੋ ਭੇਜੋ।",
    "as": "ফটো ডাউনলোড কৰাত সমস্যা হৈছে। অনুগ্ৰহ কৰি পুনৰ এখন স্পষ্ট ফটো পঠিয়াওক।",
    "ur": "تصویر ڈاؤن لوڈ کرنے میں مسئلہ پیش آیا۔ براہ کرم دوبارہ واضح تصویر بھیجیں۔",
}

IMAGE_FAILURE_RESPONSE_TE = IMAGE_FAILURE_RESPONSES["te"]
IMAGE_FAILURE_RESPONSE_EN = IMAGE_FAILURE_RESPONSES["en"]


def get_image_fallback_response(language: str = "te") -> str:
    """Return a safe localized fallback message when image download fails."""
    return IMAGE_FAILURE_RESPONSES.get(language, IMAGE_FAILURE_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 4. Market Price Unavailable Responses (13 Languages)
# -----------------------------------------------------------------------------

MARKET_FALLBACK_RESPONSES = {
    "te": "ప్రస్తుతం మార్కెట్ ధరల సమాచారం అందుబాటులో లేదు. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
    "hi": "वर्तमान में मंडी भाव की जानकारी उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।",
    "en": "Market price information is currently unavailable. Please try again after some time.",
    "ta": "தற்போது சந்தை விலை தகவல் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    "kn": "ಪ್ರಸ್ತುತ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "ml": "നിലവിൽ വിപണി വില വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    "mr": "सध्या बाजारभावाची माहिती उपलब्ध नाही. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
    "bn": "বর্তমানে বাজার দরের তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
    "gu": "હાલમાં બજાર ભાવની માહિતી ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
    "or": "ବର୍ତ୍ତମାନ ବଜାର ଦର ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
    "pa": "ਇਸ ਵੇਲੇ ਮੰਡੀ ਭਾਅ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "as": "বৰ্তমান বজাৰ দৰৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
    "ur": "فی الوقت منڈی کے بھاؤ کی معلومات دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
}

MARKET_FALLBACK_RESPONSE_TE = MARKET_FALLBACK_RESPONSES["te"]
MARKET_FALLBACK_RESPONSE_EN = MARKET_FALLBACK_RESPONSES["en"]


def get_market_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when market price service data is unavailable."""
    return MARKET_FALLBACK_RESPONSES.get(language, MARKET_FALLBACK_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 5. Weather Forecast Unavailable Responses (13 Languages)
# -----------------------------------------------------------------------------

WEATHER_FALLBACK_RESPONSES = {
    "te": "ప్రస్తుతం వాతావరణ సమాచారం పొందలేకపోతున్నాను. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
    "hi": "वर्तमान में मौसम की जानकारी उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।",
    "en": "Weather information is currently unavailable. Please try again after some time.",
    "ta": "தற்போது வானிலை தகவல் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    "kn": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "ml": "നിലവിൽ കാലാവസ്ഥാ വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    "mr": "सध्या हवामानाची माहिती उपलब्ध नाही. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
    "bn": "বর্তমানে আবহাওয়ার তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
    "gu": "હાલમાં હવામાન માહિતી ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
    "or": "ବର୍ତ୍ତମାନ ପାଣିପାଗ ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
    "pa": "ਇਸ ਵੇਲੇ ਮੌਸਮ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "as": "বৰ্তমান বতৰৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
    "ur": "فی الوقت موسم کی معلومات دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
}

WEATHER_FALLBACK_RESPONSE_TE = WEATHER_FALLBACK_RESPONSES["te"]
WEATHER_FALLBACK_RESPONSE_EN = WEATHER_FALLBACK_RESPONSES["en"]


def get_weather_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when weather forecast service data is unavailable."""
    return WEATHER_FALLBACK_RESPONSES.get(language, WEATHER_FALLBACK_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 6. Government Schemes Unavailable Responses (13 Languages)
# -----------------------------------------------------------------------------

SCHEMES_FALLBACK_RESPONSES = {
    "te": "ప్రస్తుతం ప్రభుత్వ పథకాల సమాచారం పొందలేకపోతున్నాను. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
    "hi": "वर्तमान में सरकारी योजनाओं की जानकारी उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।",
    "en": "Government scheme information is currently unavailable. Please try again after some time.",
    "ta": "தற்போது அரசு திட்டங்கள் பற்றிய தகவல் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    "kn": "ಪ್ರಸ್ತುತ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "ml": "നിലവിൽ സർക്കാർ പദ്ധതികളെക്കുറിച്ചുള്ള വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    "mr": "सध्या शासकीय योजनांची माहिती उपलब्ध नाही. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
    "bn": "বর্তমানে সরকারি প্রকল্পের তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
    "gu": "હાલમાં સરકારી યોજનાઓની માહિતી ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
    "or": "ବର୍ତ୍ତମାନ ସରକାରୀ ଯୋଜନା ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
    "pa": "ਇਸ ਵੇਲੇ ਸਰਕਾਰੀ ਸਕੀਮਾਂ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "as": "বৰ্তমান চৰকাৰী আঁচনিৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
    "ur": "فی الوقت سرکاری اسکیموں کی معلومات دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
}

SCHEMES_FALLBACK_RESPONSE_TE = SCHEMES_FALLBACK_RESPONSES["te"]
SCHEMES_FALLBACK_RESPONSE_EN = SCHEMES_FALLBACK_RESPONSES["en"]


def get_schemes_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when government schemes service data is unavailable."""
    return SCHEMES_FALLBACK_RESPONSES.get(language, SCHEMES_FALLBACK_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 7. Nearby Shops Unavailable Responses (13 Languages)
# -----------------------------------------------------------------------------

SHOPS_FALLBACK_RESPONSES = {
    "te": "ప్రస్తుతం సమీప దుకాణాల సమాచారం పొందలేకపోతున్నాను. దయచేసి కొద్దిసేపటి తర్వాత మళ్లీ ప్రయత్నించండి.",
    "hi": "वर्तमान में नजदीकी कृषि दुकानों की जानकारी उपलब्ध नहीं है। कृपया कुछ समय बाद पुनः प्रयास करें।",
    "en": "Nearby shop information is currently unavailable. Please try again after some time.",
    "ta": "தற்போது அருகிலுள்ள கடைகள் பற்றிய தகவல் கிடைக்கவில்லை. சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
    "kn": "ಪ್ರಸ್ತುತ ಹತ್ತಿರದ ಅಂಗಡಿಗಳ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
    "ml": "നിലവിൽ അടുത്തുള്ള കടകളുടെ വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക.",
    "mr": "सध्या जवळच्या कृषी दुकानांची माहिती उपलब्ध नाही. कृपया थोड्या वेळानंतर पुन्हा प्रयत्न करा.",
    "bn": "বর্তমানে নিকটবর্তী দোকানের তথ্য উপলব্ধ নেই। অনুগ্রহ করে কিছু সময় পরে আবার চেষ্টা করুন।",
    "gu": "હાલમાં નજીકની દુકાનોની માહિતી ઉપલબ્ધ નથી. કૃપા કરીને થોડા સમય પછી ફરી પ્રયાસ કરો.",
    "or": "ବର୍ତ୍ତମାନ ନିକଟସ୍ଥ ଦୋକାନ ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ।",
    "pa": "ਇਸ ਵੇਲੇ ਨੇੜਲੀਆਂ ਦੁਕਾਨਾਂ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
    "as": "বৰ্তমান ওচৰৰ দোকানৰ তথ্য উপলব্ধ নহয়। অনুগ্ৰহ কৰি কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক।",
    "ur": "فی الوقت قریبی زرعی دکانوں کی معلومات دستیاب نہیں ہیں۔ براہ کرم کچھ دیر بعد دوبارہ کوشش کریں۔",
}

SHOPS_FALLBACK_RESPONSE_TE = SHOPS_FALLBACK_RESPONSES["te"]
SHOPS_FALLBACK_RESPONSE_EN = SHOPS_FALLBACK_RESPONSES["en"]


def get_shops_fallback_response(language: str = "te") -> str:
    """Return a safe fallback message when nearby shop inventory data is unavailable."""
    return SHOPS_FALLBACK_RESPONSES.get(language, SHOPS_FALLBACK_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 8. Unsupported Media Responses (13 Languages)
# -----------------------------------------------------------------------------

UNSUPPORTED_MEDIA_RESPONSES = {
    "te": "దయచేసి టెక్స్ట్, వాయిస్ మెసేజ్ లేదా పంట ఫోటో పంపండి.",
    "hi": "कृपया टेक्स्ट, वॉइस मैसेज या फसल की फोटो भेजें।",
    "en": "Please send a text message, voice note, or crop photo.",
    "ta": "தயவுசெய்து உரை செய்தி, குரல் செய்தி அல்லது பயிர் புகைப்படத்தை அனுப்பவும்.",
    "kn": "ದಯವಿಟ್ಟು ಪಠ್ಯ, ಧ್ವನಿ ಸಂದೇಶ ಅಥವಾ ಬೆಳೆಯ ಫೋಟೋ ಕಳುಹಿಸಿ.",
    "ml": "ദയവായി ടെക്സ്റ്റ്, വോയ്‌സ് സന്ദേശം അല്ലെങ്കിൽ വിളയുടെ ഫോട്ടോ അയയ്ക്കുക.",
    "mr": "कृपया मजकूर (टेक्स्ट), व्हॉइस मेसेज किंवा पिकाचा फोटो पाठवा.",
    "bn": "অনুগ্রহ করে পাঠ্য, ভয়েস বার্তা বা ফসলের ছবি পাঠান।",
    "gu": "કૃપા કરીને ટેક્સ્ટ, વૉઇસ સંદેશ અથવા પાકનો ફોટો મોકલો.",
    "or": "ଦୟାକରି ଟେକ୍ସଟ୍, ଭଏସ୍ ମେସେଜ୍ କିମ୍ବା ଫସଲର ଫଟୋ ପଠାନ୍ତୁ।",
    "pa": "ਕਿਰਪਾ ਕਰਕੇ ਟੈਕਸਟ, ਵੌਇਸ ਸੁਨੇਹਾ ਜਾਂ ਫਸਲ ਦੀ ਫੋਟੋ ਭੇਜੋ।",
    "as": "অনুগ্ৰহ কৰি পাঠ্য, ভয়েচ বাৰ্তা বা শস্যৰ ফটো পঠিয়াওক।",
    "ur": "براہ کرم تحریری پیغام، صوتی پیغام یا فصل کی تصویر بھیجیں۔",
}

UNSUPPORTED_MEDIA_RESPONSE_TE = UNSUPPORTED_MEDIA_RESPONSES["te"]
UNSUPPORTED_MEDIA_RESPONSE_EN = UNSUPPORTED_MEDIA_RESPONSES["en"]


def get_unsupported_media_fallback_response(language: str = "te") -> str:
    """Return a localized prompt guiding the farmer on supported message formats."""
    return UNSUPPORTED_MEDIA_RESPONSES.get(language, UNSUPPORTED_MEDIA_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 9. Non-Crop Image Responses (13 Languages)
# -----------------------------------------------------------------------------

NON_CROP_IMAGE_RESPONSES = {
    "te": "పంపిన ఫోటోలో పంట లేదా మొక్క స్పష్టంగా కనిపించడం లేదు. దయచేసి వ్యాధి సోకిన ఆకు లేదా పంట భాగం స్పష్టంగా కనిపించే ఫోటోను పంపండి.",
    "hi": "भेजी गई फोटो में फसल या पौधा स्पष्ट रूप से नहीं दिख रहा है। कृपया प्रभावित पत्ती या पौधे का साफ फोटो भेजें।",
    "en": "No crop or plant was clearly detected in the photo. Please send a clear, close-up photo of the affected crop leaf, stem, or plant.",
    "ta": "அனுப்பப்பட்ட புகைப்படத்தில் பயிர் அல்லது செடி தெளிவாகத் தெரியவில்லை. தயவுசெய்து பாதிக்கப்பட்ட இலை அல்லது பகுதியின் தெளிவான புகைப்படத்தை அனுப்பவும்.",
    "kn": "ಕಳುಹಿಸಲಾದ ಫೋಟೋದಲ್ಲಿ ಬೆಳೆ ಅಥವಾ ಗಿಡ ಸ್ಪಷ್ಟವಾಗಿ ಕಾಣಿಸುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಬಾಧಿತ ಎಲೆ ಅಥವಾ ಗಿಡದ ಸ್ಪಷ್ಟ ಫೋಟೋ ಕಳುಹಿಸಿ.",
    "ml": "അയച്ച ഫോട്ടോയിൽ വിളയോ ചെടിയോ വ്യക്തമായി കാണുന്നില്ല. രോഗബാധിതമായ ഇലയോ ചെടിയുടെ ഭാഗമോ വ്യക്തമായി കാണുന്ന ഫോട്ടോ അയയ്ക്കുക.",
    "mr": "पाठवलेल्या फोटोमध्ये पीक किंवा वनस्पती स्पष्ट दिसत नाही. कृपया रोगाने प्रभावित पान किंवा भागाचा स्पष्ट फोटो पाठवा.",
    "bn": "পাঠানো ছবিতে ফসল বা গাছ স্পষ্ট দেখা যাচ্ছে না। অনুগ্রহ করে আক্রান্ত পাতা বা গাছের অংশের স্পষ্ট ছবি পাঠান।",
    "gu": "મોકલેલા ફોટામાં પાક અથવા છોડ સ્પષ્ટ દેખાતો નથી. કૃપા કરીને રોગગ્રસ્ત પાંદડા અથવા પાકના ભાગનો સ્પષ્ટ ફોટો મોકલો.",
    "or": "ପଠାଯାଇଥିବା ଫଟୋରେ ଫସଲ କିମ୍ବା ଗଛ ସ୍ପଷ୍ଟ ଦେଖାଯାଉନାହିଁ। ଦୟାକରି ରୋଗାକ୍ରାନ୍ତ ପତ୍ର କିମ୍ବା ଫସଲର ସ୍ପଷ୍ଟ ଫଟୋ ପଠାନ୍ତୁ।",
    "pa": "ਭੇਜੀ ਗਈ ਫੋਟੋ ਵਿੱਚ ਫਸਲ ਜਾਂ ਪੌਦਾ ਸਾਫ਼ ਨਜ਼ਰ ਨਹੀਂ ਆ ਰਿਹਾ। ਕਿਰਪਾ ਕਰਕੇ ਪ੍ਰਭਾਵਿਤ ਪੱਤੇ ਜਾਂ ਪੌਦੇ ਦੀ ਸਾਫ਼ ਫੋਟੋ ਭੇਜੋ।",
    "as": "পঠোৱা ফটোখনত শস্য বা উদ্ভিদ স্পষ্টকৈ দেখা নাই। অনুগ্ৰহ কৰি ৰোগাক্ৰান্ত পাত বা শস্যৰ অংশৰ স্পষ্ট ফটো পঠিয়াওক।",
    "ur": "بھیجی گئی تصویر میں فصل یا پودا واضح طور پر نظر نہیں آ رہا ہے۔ براہ کرم متاثرہ پتے یا پودے کی واضح تصویر بھیجیں۔",
}

NON_CROP_IMAGE_RESPONSE_TE = NON_CROP_IMAGE_RESPONSES["te"]
NON_CROP_IMAGE_RESPONSE_EN = NON_CROP_IMAGE_RESPONSES["en"]


def get_non_crop_image_response(language: str = "te") -> str:
    """Return a safe response asking the farmer to send a clear crop/plant image."""
    return NON_CROP_IMAGE_RESPONSES.get(language, NON_CROP_IMAGE_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 10. Unverified Fertilizer/Chemical Dosage Fallback Responses (13 Languages)
# -----------------------------------------------------------------------------

UNVERIFIED_DOSAGE_FALLBACK_RESPONSES = {
    "te": "క్షమించండి, ఖచ్చితమైన ఎరువుల/పురుగుమందుల మోతాదు (dosage) కోసం ప్రస్తుతం ధృవీకరించబడిన డేటా లేదు. మీ పంట మరియు నేల రకానికి సరైన మోతాదు కొరకు దయచేసి మీ స్థానిక వ్యవసాయ విస్తరణ అధికారి (AEO) లేదా కృషి విజ్ఞాన కేంద్రం (KVK) ను సంప్రదించండి. 🙏",
    "hi": "क्षमा करें, सटीक खाद/कीटनाशक की मात्रा (डोज़) के लिए वर्तमान में सत्यापित डेटा उपलब्ध नहीं है। अपनी फसल और मिट्टी के अनुसार सही मात्रा जानने के लिए कृपया अपने स्थानीय कृषि विस्तार अधिकारी (AEO) या कृषि विज्ञान केंद्र (KVK) से संपर्क करें। 🙏",
    "en": "I'm sorry, verified data for the exact fertilizer/chemical dosage is currently unavailable. For the safe and correct dosage for your crop and soil, please consult your local Agriculture Extension Officer (AEO) or Krishi Vigyan Kendra (KVK). 🙏",
    "ta": "மன்னிக்கவும், உரங்கள்/பூச்சிக்கொல்லிகளின் துல்லியமான அளவு (அளவுமுறை) குறித்த சரிபார்க்கப்பட்ட தரவு தற்போது கிடைக்கவில்லை. உங்கள் பயிர் மற்றும் மண்ணிற்கு ஏற்ற சரியான அளவை அறிய உங்கள் உள்ளூர் வேளாண்மை விரிவாக்க அலுவலர் (AEO) அல்லது வேளாண்மை அறிவியல் மையத்தை (KVK) அணுகவும். 🙏",
    "kn": "ಕ್ಷಮಿಸಿ, ನಿಖರವಾದ ರಸಗೊಬ್ಬರ/ಕೀಟನಾಶಕ ಪ್ರಮಾಣದ (ಡೋಸೇಜ್) ಬಗ್ಗೆ ಪರಿಶೀಲಿಸಿದ ಮಾಹಿತಿ ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ ಬೆಳೆ ಮತ್ತು ಮಣ್ಣಿಗೆ ಸೂಕ್ತವಾದ ಪ್ರಮಾಣಕ್ಕಾಗಿ ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ವಿಸ್ತರಣಾಧಿಕಾರಿ (AEO) ಅಥವಾ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು (KVK) ಸಂಪರ್ಕಿಸಿ. 🙏",
    "ml": "ക്ഷമിക്കണം, കൃത്യമായ വളം/കീടനാശിനി അളവ് (ഡോസേജ്) സംബന്ധിച്ച് നിലവിൽ പരിശോധിച്ച വിവരങ്ങൾ ലഭ്യമല്ല. നിങ്ങളുടെ വിളയ്ക്കും മണ്ണിനും അനുയോജ്യമായ അളവറിയാൻ ദയവായി നിങ്ങളുടെ പ്രാദേശिक കൃഷി ഓഫീസർ (AEO) അല്ലെങ്കിൽ കൃഷി വിജ്ഞാൻ കേന്ദ്രവുമായി (KVK) ബന്ധപ്പെടുക. 🙏",
    "mr": "क्षमस्व, खत/कीटकनाशकाच्या अचूक प्रमाण (डोस) बद्दल सध्या पडताळणी केलेली माहिती उपलब्ध नाही. आपल्या पिकासाठी योग्य प्रमाणाची माहिती मिळवण्यासाठी कृपया आपल्या स्थानिक कृषी विस्तार अधिकारी (AEO) किंवा कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा. 🙏",
    "bn": "দুঃখিত, সার/কীটনাশকের সঠিক পরিমাণের (ডোজ) জন্য বর্তমানে যাচাইকৃত তথ্য উপলব্ধ নেই। আপনার ফসলের জন্য সঠিক মাত্রা জানতে অনুগ্রহ করে আপনার স্থানীয় কৃষি সম্প্রসারণ কর্মকর্তা (AEO) বা কৃষি বিজ্ঞান কেন্দ্রের (KVK) সাথে যোগাযোগ করুন। 🙏",
    "gu": "માફ કરશો, ખાતર/જંતુનાશકના ચોક્કસ પ્રમાણ (ડોઝ) અંગે હાલમાં ચકાસાયેલ માહિતી ઉપલબ્ધ નથી. તમારા પાક માટે યોગ્ય માત્રા જાણવા કૃપા કરીને તમારા સ્થાનિક કૃષિ વિસ્તરણ અધિકારી (AEO) અથવા કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) નો સંપર્ક કરો. 🙏",
    "or": "କ୍ଷମା କରିବେ, ସାର/କୀଟନାଶକର ସଠିକ୍ ପରିମାଣ (ଡୋଜ୍) ପାଇଁ ବର୍ତ୍ତମାନ ଯାଞ୍ଚ ହୋଇଥିବା ତଥ୍ୟ ଉପଲବ୍ଧ ନାହିଁ। ଆପଣଙ୍କ ଫସଲ ପାଇଁ ସଠିକ୍ ପରିମାଣ ଜାଣିବାକୁ ଦୟାକରି ଆପଣଙ୍କ ସ୍ଥାନୀୟ କୃଷି ସମ୍ପ୍ରସାରଣ ଅଧିକାରୀ (AEO) କିମ୍ବା କୃଷି ବିଜ୍ଞାନ କେନ୍ଦ୍ର (KVK) ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ। 🙏",
    "pa": "ਮਾਫ ਕਰਨਾ, ਖਾਦ/ਕੀਟਨਾਸ਼ਕ ਦੀ ਸਹੀ ਮਾਤਰਾ (ਡੋਜ਼) ਬਾਰੇ ਇਸ ਵੇਲੇ ਤਸਦੀਕਸ਼ੁਦਾ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਆਪਣੀ ਫਸਲ ਲਈ ਸਹੀ ਮਾਤਰਾ ਜਾਣਨ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਵਿਸਥਾਰ ਅਧਿਕਾਰੀ (AEO) ਜਾਂ ਕ੍ਰਿਸ਼ੀ ਵਿਗਿਆਨ ਕੇਂਦਰ (KVK) ਨਾਲ ਸੰਪਰਕ ਕਰੋ। 🙏",
    "as": "ক্ষমা কৰিব, সাৰ/কীটনাশকৰ সঠিক পৰিমাণৰ (ড'জ) বিষয়ে বৰ্তমান সত্যাপন কৰা তথ্য উপলব্ধ নাই। আপোনাৰ শস্যৰ বাবে সঠিক পৰিমাণ জানিবলৈ অনুগ্ৰহ কৰি আপোনাৰ স্থানীয় কৃষি সম্প্ৰসাৰণ বিষয়া (AEO) বা কৃষি বিজ্ঞান কেন্দ্ৰৰ (KVK) সৈতে যোগাযোগ কৰক। 🙏",
    "ur": "معذرت، کھاد یا کیڑے مار دوا کی درست مقدار (خوراک) کے بارے میں فی الحال تصدیق شدہ معلومات دستیاب نہیں ہیں۔ اپنی فصل کے لیے صحیح مقدار جاننے کے لیے براہ کرم اپنے مقامی محکمہ زراعت کے افسر (AEO) یا کرشی وگیان کیندر (KVK) سے رابطہ کریں۔ 🙏",
}

UNVERIFIED_DOSAGE_FALLBACK_RESPONSE_TE = UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"]
UNVERIFIED_DOSAGE_FALLBACK_RESPONSE_EN = UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["en"]


def get_unverified_dosage_fallback_response(language: str = "te") -> str:
    """Return a safe localized fallback when exact fertilizer/chemical dosage is ungrounded in RAG."""
    return UNVERIFIED_DOSAGE_FALLBACK_RESPONSES.get(language, UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"])


def build_farmer_context(
    crop: str = None,
    district: str = None,
    state: str = None,
    land_size: float = None,
) -> str:
    """
    Build a context block to prepend to the conversation,
    giving the AI localized awareness of the farmer's situation.
    """
    parts = []
    if crop:
        parts.append(f"Current Crop: {crop}")
    if district:
        parts.append(f"District: {district}")
    if state:
        parts.append(f"State: {state}")
    if land_size:
        parts.append(f"Land Size: {land_size} acres")

    if not parts:
        return "[Farmer profile is incomplete. Ask the farmer about their crop and location.]"

    return "[Farmer Profile]\n" + "\n".join(parts)
