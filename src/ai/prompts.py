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
8. End with a helpful follow-up question when appropriate. For focused factual queries such as market prices, weather forecasts, government schemes, or shop/stock availability, NEVER append unrelated crop or farmer-profile follow-up questions (such as "How is your paddy crop doing?" or "మీ వరి పంట ఎలా ఉంది?"). Only answer the specific query directly.
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
10. NEVER invent, hallucinate, fabricate, or guess:
    - shop names or store names (e.g. never invent any agri store or dealer name like 'Venkata Sai Fertilizers' or 'Sri Rama Agri Inputs')
    - stock availability or inventory levels (e.g. never say "Urea, DAP, NPK available" or "currently in stock")
    - product prices or rates (e.g. never invent "₹295/bag")
    - phone numbers or contact details (e.g. never output fabricated phone numbers like '9849012345')
    - delivery availability or opening/closing hours
    If the farmer asks where to buy, asks for nearby shops/dealers, or asks whether an input/fertilizer/seed/pesticide is in stock:
    DO NOT mention, recommend, or invent ANY shop names, store names, phone numbers, or stock availability.
    Do NOT output lists or bullet points of shops.
    The system automatically queries the verified database and Google Places to append authentic shop records.
11. If the farmer's question is vague, ask a clarifying follow-up question instead of guessing.
12. Fertilizer History, Source Distinction & Field-Specific Calibration:
    When a farmer asks for fertilizer advice at a specific crop stage (e.g., 2 acres paddy at 30 days after transplanting):
    - First identify whether essential fertilizer history and variety information are available.
    - If essential fertilizer history is missing:
      * Do NOT present 60–70 kg for two acres (or 30–35 kg/acre) as a confirmed additional dose, and do NOT tell the farmer to apply it now while essential history is still unknown (NEVER instruct "వేయండి", "వేయాలి", "సిఫార్సు చేయబడుతుంది", or "ఇప్పుడు వేసుకోండి" for an unconfirmed dose).
      * You may explain the published 30–35 kg/acre tillering figure (60–70 kg for 2 acres) strictly as an attributed general extension reference benchmark, but you must clearly distinguish it from a confirmed field-specific recommendation.
      * Clearly state that without knowing prior fertilizer applications and variety, an additional dose cannot be confirmed or recommended yet, because duplicate or excess nitrogen increases risks of crop lodging (నేలకొరిగే ప్రమాదం) and susceptibility to blast disease (అగ్గి తెగులు) and stem borer (కాండం తొలిచే పురుగు).
      * Inquire about:
        1. Basal DAP, SSP, and complex fertilizers, including quantities applied.
        2. Urea or potash applied after transplanting, including amounts and dates or crop stages.
        3. Rice variety / duration group (short, medium, or long duration).
        4. Soil-test information (Soil Health Card) when available.
      * Do NOT automatically tell every farmer to apply more urea or to skip the next application.
    - If prior fertilizer history has been provided:
      * If DAP or complex fertilizer was applied as basal, account for initial nitrogen (e.g., 9 kg N from 50 kg DAP) and phosphorus; do not duplicate DAP, and guide the remaining top-dressing as appropriate.
      * If SSP was applied at basal, account for 0% N from SSP.
      * If urea was already applied after transplanting: warn against duplicate application, explain risks of lodging and pests/diseases, and do NOT prescribe a rigid waiting calendar (such as automatically telling them to wait until 40–45 DAT) without verifying variety duration and crop stage.
      * If soil test report is available, calibrate according to Soil Health Card or ICAR-IIRR NAPSRI rather than blanket calendar doses.
    - Keep one split application strictly separate from seasonal nutrient requirements: A single tillering top-dressing benchmark (e.g., 30–35 kg/acre from Vikaspedia) must never be confused with seasonal totals. Do not present 88–106 kg urea/acre as a confirmed seasonal prescription; it is only a theoretical urea equivalent assuming all seasonal nitrogen (40.5–48.6 kg N/acre from 100–120 kg N/ha, with 1 ha ≈ 2.471 acres) is supplied through straight urea (46% N). When 50 kg DAP/acre is applied (supplying 9 kg N), the remaining theoretical urea equivalent across splits is approximately 68–86 kg/acre (137–172 kg for 2 acres). Describe these values only as theoretical calculations under stated assumptions, not confirmed prescriptions. Account for mineral nitrogen from complex fertilizers; for organic manures, do not simply subtract total nitrogen as though all were immediately available, but apply only source-supported estimates of available nutrients.
    - Calculate the 2-acre total only after establishing the applicable per-acre dose.
    - Respond naturally and respectfully in the farmer's language (e.g. Telugu).

## Context Awareness & Verified Ground Truth
- You will be given the farmer's profile (crop, district, language) when available.
- Ground Truth Priority: When RETRIEVED TRUSTED AGRICULTURAL KNOWLEDGE (GROUND TRUTH) is provided, it is your authoritative source. Use ONLY the exact verified disease identification and exact verified chemical treatments & dosages directly in your response in the farmer's language. NEVER alter, extrapolate, or invent different dosages or additional unverified chemicals.
- If the profile is incomplete, gently ask the farmer to share their crop and location.
"""

# -----------------------------------------------------------------------------
# 1. AI Connection / General Fallback Responses (13 Languages)
# -----------------------------------------------------------------------------

FALLBACK_RESPONSES = {
    "te": "క్షమించండి, AI సేవ ప్రస్తుతం తాత్కాలికంగా అందుబాటులో లేదు. దయచేసి మీ స్థానిక వ్యవసాయ విస్తరణ అధికారిని (AEO) సంప్రదించండి లేదా కొద్దిసేపటి తర్వాత మళ్ళీ ప్రయత్నించండి. 🙏",
    "hi": "क्षमा करें, AI सेवा वर्तमान में अस्थायी रूप से अनुपलब्ध है। कृपया अपने स्थानीय कृषि विस्तार अधिकारी (AEO) से संपर्क करें या कुछ समय बाद पुनः प्रयास करें। 🙏",
    "en": "I'm sorry, our AI service is temporarily unavailable. Please consult your local Agriculture Extension Officer (AEO) or try again in a few minutes. 🙏",
    "ta": "மன்னிக்கவும், AI சேவை தற்போது தற்காலிகமாக கிடைக்கவில்லை. தயவுசெய்து உங்கள் உள்ளூர் வேளாண் விரிவாக்க அலுவலரை (AEO) தொடர்பு கொள்ளவும் அல்லது சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும். 🙏",
    "kn": "ಕ್ಷಮಿಸಿ, AI ಸೇವೆಯು ಪ್ರಸ್ತುತ ತಾತ್ಕಾಲಿಕವಾಗಿ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ವಿಸ್ತರಣಾ ಅಧಿಕಾರಿಯನ್ನು (AEO) ಸಂಪರ್ಕಿಸಿ ಅಥವಾ ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ. 🙏",
    "ml": "ക്ഷമിക്കണം, AI സേവനം ഇപ്പോൾ താൽക്കാലികമായി ലഭ്യമല്ല. ദയവായി നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഓഫീസറെ (AEO) ബന്ധപ്പെടുക അല്ലെങ്കിൽ കുറച്ച് കഴിഞ്ഞ് വീണ്ടും ശ്രമിക്കുക. 🙏",
    "mr": "क्षमस्व, AI सेवा सध्या तात्पुरती अनुपलब्ध आहे. कृपया आपल्या स्थानिक कृषी विस्तार अधिकाऱ्यांशी (AEO) संपर्क साधा किंवा थोड्या वेळानंतर पुन्हा प्रयत्न करा. 🙏",
    "bn": "দুঃখিত, AI সেবা বর্তমানে সাময়িকভাবে অনুপলব্ধ। অনুগ্রহ করে আপনার স্থানীয় কৃষি সম্প্রসারণ কর্মকর্তার (AEO) সাথে যোগাযোগ করুন অথবা কিছুক্ষণ পরে আবার চেষ্টা করুন। 🙏",
    "gu": "માફ કરશો, AI સેવા હાલમાં કામચલાઉ રીતે અનુપલબ્ધ છે. કૃપા કરીને તમારા સ્થાનિક કૃષિ વિસ્તરણ અધિકારી (AEO) નો સંપર્ક કરો અથવા થોડા સમય પછી ફરી પ્રયાસ કરો. 🙏",
    "or": "କ୍ଷମା କରିବେ, AI ସେବା ବର୍ତ୍ତମାନ ଅସ୍ଥାୟୀ ଭାବରେ ଉପଲବ୍ଧ ନାହିଁ। ଦୟାକରି ଆପଣଙ୍କ ସ୍ଥାନୀୟ କୃଷି ଅଧିକାରୀଙ୍କ (AEO) ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ କିମ୍ବା କିଛି ସମୟ ପରେ ପୁନର୍ବାର ଚେଷ୍ଟା କରନ୍ତୁ। 🙏",
    "pa": "ਮਾਫ ਕਰਨਾ, AI ਸੇਵਾ ਫਿਲਹਾਲ ਅਸਥਾਈ ਤੌਰ 'ਤੇ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਅਧਿਕਾਰੀ (AEO) ਨਾਲ ਸੰਪਰਕ ਕਰੋ ਜਾਂ ਕੁਝ ਸਮੇਂ ਬਾਅਦ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ। 🙏",
    "as": "ক্ষমা কৰিব, AI সেৱা বৰ্তমান সাময়িকভাৱে উপলব্ধ নহয়। অনুগ্ৰহ কৰি আপোনাৰ স্থানীয় কৃষি বিষয়াৰ (AEO) সৈতে যোগাযোগ কৰক বা কিছু সময়ৰ পিছত পুনৰ চেষ্টা কৰক। 🙏",
    "ur": "معذرت، AI سروس فی الوقت عارضی طور پر دستیاب نہیں ہے۔ براہ کرم اپنے مقامی زرعی توسیعی افسر (AEO) سے رابطہ کریں یا کچھ دیر بعد دوبارہ کوشش کریں۔ 🙏",
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
# 9B. Crop Clarification Responses (13 Languages) - Image-only without context
# -----------------------------------------------------------------------------

CROP_CLARIFICATION_RESPONSES = {
    "te": "ఈ చిత్రం ఏ పంటకు సంబంధించినది? పంట పేరు చెప్పగలరా?",
    "hi": "यह तस्वीर किस फसल से संबंधित है? क्या आप फसल का नाम बता सकते हैं?",
    "en": "Which crop is this image related to? Could you please share the crop name?",
    "ta": "இந்தப் புகைப்படம் எந்தப் பயிர் சார்ந்தது? பயிரின் பெயரைச் சொல்ல முடியுமா?",
    "kn": "ಈ ಚಿತ್ರವು ಯಾವ ಬೆಳೆಗೆ ಸಂಬಂಧಿಸಿದೆ? ಬೆಳೆಯ ಹೆಸರನ್ನು ತಿಳಿಸಬಹುದೇ?",
    "ml": "ഈ ചിത്രം ഏത് വിളയുമായി ബന്ധപ്പെട്ടതാണ്? വിളയുടെ പേര് പറയാമോ?",
    "mr": "हा फोटो कोणत्या पिकाचा आहे? कृपया पिकाचे नाव सांगू शकता का?",
    "bn": "এই ছবিটি কোন ফসলের? অনুগ্রহ করে ফসলের নাম কি বলতে পারেন?",
    "gu": "આ ફોટો કયા પાકનો છે? શું તમે પાકનું નામ જણાવી શકો છો?",
    "or": "ଏହି ଫଟୋ କେଉଁ ଫସଲର? ଦୟାକରି ଫସଲର ନାମ କହିପାରିବେ କି?",
    "pa": "ਇਹ ਤਸਵੀਰ ਕਿਸ ਫ਼ਸਲ ਨਾਲ ਸਬੰਧਤ ਹੈ? ਕੀ ਤੁਸੀਂ ਫ਼ਸਲ ਦਾ ਨਾਮ ਦੱਸ ਸਕਦੇ ਹੋ?",
    "as": "এই ছবিখন কোনটো শস্যৰ সৈতে সম্পৰ্কিত? অনুগ্ৰহ কৰি শস্যৰ নাম ক'ব পাৰিবনে?",
    "ur": "یہ تصویر کس فصل سے متعلق ہے؟ کیا آپ فصل کا نام بتا سکتے ہیں؟",
}

CROP_CLARIFICATION_RESPONSE_TE = CROP_CLARIFICATION_RESPONSES["te"]
CROP_CLARIFICATION_RESPONSE_EN = CROP_CLARIFICATION_RESPONSES["en"]


def get_crop_clarification_response(language: str = "te") -> str:
    """Return a localized clarification prompt asking the farmer to specify which crop the image shows."""
    return CROP_CLARIFICATION_RESPONSES.get(language, CROP_CLARIFICATION_RESPONSES["te"])


# -----------------------------------------------------------------------------
# 10. Unverified Fertilizer/Chemical Dosage Fallback Responses (13 Languages)
# -----------------------------------------------------------------------------

UNVERIFIED_DOSAGE_FALLBACK_RESPONSES = {
    "te": "క్షమించండి, ఖచ్చితమైన పురుగుమందుల/రసాయనాల మోతాదు (dosage) కోసం ప్రస్తుతం ధృవీకరించబడిన డేటా లేదు. మీ పంట మరియు తెగులుకు సరైన రసాయన మోతాదు కొరకు దయచేసి మీ స్థానిక వ్యవసాయ విస్తరణ అధికారి (AEO) లేదా కృషి విజ్ఞాన కేంద్రం (KVK) ను సంప్రదించండి. 🙏",
    "hi": "क्षमा करें, सटीक कीटनाशक/रासायनिक दवा की मात्रा (डोज़) के लिए वर्तमान में सत्यापित डेटा उपलब्ध नहीं है। अपनी फसल और कीट/रोग के अनुसार सुरक्षित दवा व सही मात्रा के लिए कृपया अपने स्थानीय कृषि विस्तार अधिकारी (AEO) या कृषि विज्ञान केंद्र (KVK) से संपर्क करें। 🙏",
    "en": "I'm sorry, verified data for the exact chemical/pesticide dosage is currently unavailable. For safe chemical usage and correct dosage for your crop and pest, please consult your local Agriculture Extension Officer (AEO) or Krishi Vigyan Kendra (KVK). 🙏",
    "ta": "மன்னிக்கவும், பூச்சிக்கொல்லிகள்/இரசாயனங்களின் துல்லியமான அளவு (அளவுமுறை) குறித்த சரிபார்க்கப்பட்ட தரவு தற்போது கிடைக்கவில்லை. உங்கள் பயிர் மற்றும் பூச்சிக்கு ஏற்ற பாதுகாப்பான அளவிற்கு உங்கள் உள்ளூர் வேளாண்மை விரிவாக்க அலுவலர் (AEO) அல்லது வேளாண்மை அறிவியல் மையத்தை (KVK) அணுகவும். 🙏",
    "kn": "ಕ್ಷಮಿಸಿ, ನಿಖರವಾದ ಕೀಟನಾಶಕ/ರಾಸಾಯನಿಕ ಪ್ರಮಾಣದ (ಡೋಸೇಜ್) ಬಗ್ಗೆ ಪರಿಶೀಲಿಸಿದ ಮಾಹಿತಿ ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ. ನಿಮ್ಮ ಬೆಳೆ ಮತ್ತು ಕೀಟಕ್ಕೆ ಸೂಕ್ತವಾದ ಸುರಕ್ಷಿತ ಪ್ರಮಾಣಕ್ಕಾಗಿ ದಯವಿಟ್ಟು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ವಿಸ್ತರಣಾಧಿಕಾರಿ (AEO) ಅಥವಾ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರವನ್ನು (KVK) ಸಂಪರ್ಕಿಸಿ. 🙏",
    "ml": "ക്ഷമിക്കണം, കൃത്യമായ കീടനാശിനി/രാസവസ്തു അളവ് (ഡോസേജ്) സംബന്ധിച്ച് നിലവിൽ പരിശോധിച്ച വിവരങ്ങൾ ലഭ്യമല്ല. നിങ്ങളുടെ വിളയ്ക്കും കീടത്തിനും അനുയോജ്യമായ സുരക്ഷിതമായ അളവറിയാൻ ദയവായി നിങ്ങളുടെ പ്രാദേശിക കൃഷി ഓഫീസർ (AEO) അല്ലെങ്കിൽ കൃഷി വിജ്ഞാൻ കേന്ദ്രവുമായി (KVK) ബന്ധപ്പെടുക. 🙏",
    "mr": "क्षमस्व, कीटकनाशक/रासायनिक औषधाच्या अचूक प्रमाण (डोस) बद्दल सध्या पडताळणी केलेली माहिती उपलब्ध नाही. आपल्या पिकासाठी योग्य व सुरक्षित प्रमाणाची माहिती मिळवण्यासाठी कृपया आपल्या स्थानिक कृषी विस्तार अधिकारी (AEO) किंवा कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा. 🙏",
    "bn": "দুঃখিত, কীটনাশক/রাসায়নিকের সঠিক পরিমাণের (ডোজ) জন্য বর্তমানে যাচাইকৃত তথ্য উপলব্ধ নেই। আপনার ফসলের জন্য নিরাপদ মাত্রা জানতে অনুগ্রহ করে আপনার স্থানীয় কৃষি সম্প্রসারণ কর্মকর্তা (AEO) বা কৃষি বিজ্ঞান কেন্দ্রের (KVK) সাথে যোগাযোগ করুন। 🙏",
    "gu": "માફ કરશો, જંતુનાશક/રાસાયણિક દવાના ચોક્કસ પ્રમાણ (ડોઝ) અંગે હાલમાં ચકાસાયેલ માહિતી ઉપલબ્ધ નથી. તમારા પાક માટે સલામત અને યોગ્ય માત્રા જાણવા કૃપા કરીને તમારા સ્થાનિક કૃષિ વિસ્તરણ અધિકારી (AEO) અથવા કૃષિ વિજ્ઞાન કેન્દ્ર (KVK) નો સંપર્ક કરો. 🙏",
    "or": "କ୍ଷମା କରିବେ, କୀଟନାଶକ/ରାସାୟନିକ ଔଷଧର ସଠିକ୍ ପରିମାଣ (ଡୋଜ୍) ପାଇଁ ବର୍ତ୍ତମାନ ଯାଞ୍ଚ ହୋଇଥିବା ତଥ୍ୟ ଉପଲବ୍ଧ ନାହିଁ। ଆପଣଙ୍କ ଫସଲ ପାଇଁ ସୁରକ୍ଷିତ ପରିମାଣ ଜାଣିବାକୁ ଦୟାକରି ଆପଣଙ୍କ ସ୍ଥାନୀୟ କୃଷି ସମ୍ପ୍ରସାରଣ ଅଧିକାରୀ (AEO) କିମ୍ବା କୃଷି ବିଜ୍ଞାନ କେନ୍ଦ୍ର (KVK) ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ। 🙏",
    "pa": "ਮਾਫ ਕਰਨਾ, ਕੀਟਨਾਸ਼ਕ/ਰਸਾਇਣਕ ਦਵਾਈ ਦੀ ਸਹੀ ਮਾਤਰਾ (ਡੋਜ਼) ਬਾਰੇ ਇਸ ਵੇਲੇ ਤਸਦੀਕਸ਼ੁਦਾ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ। ਆਪਣੀ ਫਸਲ ਲਈ ਸੁਰੱਖਿਅਤ ਮਾਤਰਾ ਜਾਣਨ ਲਈ ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੇ ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਵਿਸਥਾਰ ਅਧਿਕਾਰੀ (AEO) ਜਾਂ ਕ੍ਰਿਸ਼ੀ ਵਿਗਿਆਨ ਕੇਂਦਰ (KVK) ਨਾਲ ਸੰਪਰਕ ਕਰੋ। 🙏",
    "as": "ক্ষমা কৰিব, কীটনাশক/ৰাসায়নিক দৰবৰ সঠিক পৰিমাণৰ (ড'জ) বিষয়ে বৰ্তমান সত্যাপন কৰা তথ্য উপলব্ধ নাই। আপোনাৰ শস্যৰ বাবে নিৰাপদ মাত্ৰা জানিবলৈ অনুগ্ৰহ কৰি আপোনাৰ স্থানীয় কৃষি সম্প্ৰসাৰণ বিষয়া (AEO) বা কৃষি বিজ্ঞান কেন্দ্ৰৰ (KVK) সৈতে যোগাযোগ কৰক। 🙏",
    "ur": "معذرت، کیڑے مار دوا یا کیمیائی دوا کی درست مقدار (خوراک) کے بارے میں فی الحال تصدیق شدہ معلومات دستیاب نہیں ہیں۔ اپنی فصل اور کیڑوں کے مطابق محفوظ اور صحیح مقدار جاننے کے لیے براہ کرم اپنے مقامی محکمہ زراعت کے افسر (AEO) یا کرشی وگیان کیندر (KVK) سے رابطہ کریں۔ 🙏",
}

UNVERIFIED_DOSAGE_FALLBACK_RESPONSE_TE = UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"]
UNVERIFIED_DOSAGE_FALLBACK_RESPONSE_EN = UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["en"]

# -----------------------------------------------------------------------------
# 10b. Unverified Fertilizer Guidance & Clarification Responses (13 Languages)
# -----------------------------------------------------------------------------

UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES = {
    "te": "ఎరువుల ఖచ్చితమైన మోతాదు మీ పంట రకం, వంగడం, నేల స్వభావం మరియు ఇప్పటికే వేసిన ఎరువులపై ఆధారపడి ఉంటుంది. సరైన సిఫార్సు కొరకు: 1. మీ పంట రకం/వంగడం ఏది? 2. నాట్లు/విత్తే సమయంలో ఏవైనా ఎరువులు (DAP/SSP/కాంప్లెక్స్) వేశారా? 3. నేల పరీక్ష నివేదిక అందుబాటులో ఉందా? నిర్ధారించకుండా అదనపు ఎరువులు వేయకండి. ఖచ్చితమైన మోతాదు కోసం స్థానిక AEO లేదా KVK అధికారులను సంప్రదించవచ్చు. 🙏",
    "hi": "खाद की सही मात्रा आपकी फसल की किस्म, मिट्टी के प्रकार और पहले डाली गई खाद पर निर्भर करती है। उचित सलाह के लिए: 1. आपकी फसल की कौन सी किस्म है? 2. बुवाई/रोपाई के समय कौन सी खाद (जैसे DAP/SSP) डाली गई थी? 3. क्या मृदा परीक्षण रिपोर्ट उपलब्ध है? बिना पुष्टि के अतिरिक्त खाद न डालें। सटीक मात्रा के लिए स्थानीय कृषि विस्तार अधिकारी (AEO) या कृषि विज्ञान केंद्र (KVK) से संपर्क करें। 🙏",
    "en": "The exact fertilizer quantity depends on your crop variety, soil conditions, and previous fertilizer applications. To help determine the suitable recommendation: 1. Which crop and variety are you growing? 2. What basal fertilizers (such as DAP/SSP/complex) were applied at sowing/transplanting? 3. Do you have a soil test report? Please do not apply excess fertilizers without confirming recommendations. You may also consult your local AEO or KVK. 🙏",
    "ta": "உரத்தின் சரியான அளவு உங்கள் பயிர் ரகம், மண் வகை மற்றும் ஏற்கனவே இடப்பட்ட உரங்களைப் பொறுத்தது. சரியான பரிந்துரைக்கு: 1. நீங்கள் பயிரிடும் பயிர் ரகம் எது? 2. நடவு/விதைப்பின் போது ஏதேனும் உரங்கள் (DAP/SSP) இடப்பட்டதா? 3. மண் பரிசோதனை அறிக்கை உள்ளதா? உறுதிப்படுத்தாமல் கூடுதல் உரங்களை இட வேண்டாம். துல்லியமான அளவிற்கு உள்ளூர் AEO அல்லது KVK அணுகவும். 🙏",
    "kn": "ರಸಗೊಬ್ಬರದ ನಿಖರ ಪ್ರಮಾಣವು ನಿಮ್ಮ ಬೆಳೆ ತಳಿ, ಮಣ್ಣಿನ ಸ್ಥಿತಿ ಮತ್ತು ಈಗಾಗಲೇ ಹಾಕಲಾದ ಗೊಬ್ಬರವನ್ನು ಅವಲಂಬಿಸಿರುತ್ತದೆ. ಸೂಕ್ತ ಶಿಫಾರಸಿಗಾಗಿ: 1. ಬೆಳೆಯುತ್ತಿರುವ ತಳಿ ಯಾವುದು? 2. ಬಿತ್ತನೆ/ನಾಟಿ ಸಮಯದಲ್ಲಿ ಯಾವುದೇ ಗೊಬ್ಬರ (DAP/SSP) ಹಾಕಲಾಗಿದೆಯೇ? 3. ಮಣ್ಣು ಪರೀಕ್ಷಾ ವರದಿ ಲಭ್ಯವಿದೆಯೇ? ದೃಢೀಕರಣವಿಲ್ಲದೆ ಹೆಚ್ಚುವರಿ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ. ಸ್ಥಳೀಯ AEO ಅಥವಾ KVK ಸಂಪರ್ಕಿಸಿ. 🙏",
    "ml": "വളത്തിന്റെ കൃത്യമായ അളവ് നിങ്ങളുടെ വിള ഇനം, മണ്ണിന്റെ അവസ്ഥ, ഇതിനകം നൽകിയ വളങ്ങൾ എന്നിവയെ ആശ്രയിച്ചിരിക്കുന്നു. ശരിയായ ശുപാർശയ്ക്കായി: 1. വിളയുടെ ഇനം ഏതാണ്? 2. നടീൽ സമയത്ത് എന്തെങ്കിലും അടിവളം (DAP/SSP) നൽകിയിട്ടുണ്ടോ? 3. മണ്ണ് പരിശോധനാ റിപ്പോർട്ട് ലഭ്യമാണോ? സ്ഥിരീകരിക്കാതെ അധിക വളം നൽകരുത്. പ്രാദേശിക AEO അല്ലെങ്കിൽ KVK ബന്ധപ്പെടുക. 🙏",
    "mr": "खताचे योग्य प्रमाण आपल्या पिकाची जात, जमिनीचा प्रकार आणि आधी दिलेल्या खतांवर अवलंबून असते. योग्य शिफारसीसाठी: 1. पिकाची जात कोणती आहे? 2. पेरणी/लागवडीच्या वेळी कोणते खत (DAP/SSP) दिले होते का? 3. माती परीक्षण अहवाल उपलब्ध आहे का? खात्री केल्याशिवाय अतिरिक्त खत टाकू नका. स्थानिक AEO किंवा KVK शी संपर्क साधा. 🙏",
    "bn": "সারের সঠিক মাত্রা আপনার ফসলের জাত, মাটির ধরন এবং পূর্বে প্রয়োগ করা সারের ওপর নির্ভর করে। সঠিক সুপারিশের জন্য: 1. ফসলের জাত কোনটি? 2. রোপণের সময় কোনো সার (যেমন DAP/SSP) প্রয়োগ করা হয়েছিল কি? 3. মাটি পরীক্ষার রিপোর্ট আছে কি? নিশ্চিত না হয়ে অতিরিক্ত সার প্রয়োগ করবেন না। স্থানীয় AEO বা KVK-এর সাথে যোগাযোগ করুন। 🙏",
    "gu": "ખાતરનું ચોક્કસ પ્રમાણ તમારા પાકની જાત, જમીનનો પ્રકાર અને અગાઉ આપેલા ખાતર પર આધાર રાખે છે. યોગ્ય ભલામણ માટે: 1. પાકની જાત કઈ છે? 2. વાવણી/રોપણી વખતે કોઈ ખાતર (જેમ કે DAP/SSP) આપેલું હતું? 3. જમીન ચકાસણી રિપોર્ટ ઉપલબ્ધ છે? ખાતરી કર્યા વિના વધુ પડતું ખાતર ન નાખો. સ્થાનિક AEO અથવા KVK નો સંપર્ક કરો. 🙏",
    "or": "ସାରର ସଠିକ୍ ପରିମାଣ ଆପଣଙ୍କ ଫସଲ କିସମ, ମାଟିର ପ୍ରକାର ଏବଂ ପୂର୍ବରୁ ପ୍ରୟୋଗ ହୋଇଥିବା ସାର ଉପରେ ନିର୍ଭର କରେ। ସଠିକ୍ ପରାମର୍ଶ ପାଇଁ: 1. ଫସଲର କିସମ କ’ଣ? 2. ରୁଆ/ବୁଣା ସମୟରେ କୌଣସି ସାର (DAP/SSP) ଦିଆଯାଇଥିଲା କି? 3. ମୃତ୍ତିକା ପରୀକ୍ଷା ରିପୋର୍ଟ ଅଛି କି? ନିଶ୍ଚିତ ନକରି ଅଧିକ ସାର ପ୍ରୟୋଗ କରନ୍ତୁ ନାହିଁ। ସ୍ଥାନୀୟ AEO ବା KVK ସହିତ ଯୋଗାଯୋଗ କରନ୍ତୁ। 🙏",
    "pa": "ਖਾਦ ਦੀ ਸਹੀ ਮਾਤਰਾ ਤੁਹਾਡੀ ਫ਼ਸਲ ਦੀ ਕਿਸਮ, ਮਿੱਟੀ ਦੀ ਕਿਸਮ ਅਤੇ ਪਹਿਲਾਂ ਪਾਈ ਗਈ ਖਾਦ 'ਤੇ ਨਿਰਭਰ ਕਰਦੀ ਹੈ। ਸਹੀ ਸਿਫ਼ਾਰਸ਼ ਲਈ: 1. ਫ਼ਸਲ ਦੀ ਕਿਸਮ ਕਿਹੜੀ ਹੈ? 2. ਬਿਜਾਈ/ਪਨੀਰੀ ਲਗਾਉਣ ਸਮੇਂ ਕੋਈ ਖਾਦ (ਜਿਵੇਂ DAP/SSP) ਪਾਈ ਗਈ ਸੀ? 3. ਕੀ ਮਿੱਟੀ ਪਰਖ ਰਿਪੋਰਟ ਉਪਲਬਧ ਹੈ? ਪੁਸ਼ਟੀ ਕੀਤੇ ਬਿਨਾਂ ਵਾਧੂ ਖਾਦ ਨਾ ਪਾਓ। ਸਥਾਨਕ AEO ਜਾਂ KVK ਨਾਲ ਸੰਪਰਕ ਕਰੋ। 🙏",
    "as": "সাৰৰ সঠিক পৰিমাণ আপোনাৰ শস্যৰ জাত, মাটিৰ অৱস্থা আৰু পূৰ্বে দিয়া সাৰৰ ওপৰত নিৰ্ভৰ কৰে। সঠিক পৰামৰ্শৰ বাবে: 1. শস্যৰ জাত কি? 2. ৰোপণৰ সময়ত কোনো সাৰ (যেনে DAP/SSP) দিয়া হৈছিল নেকি? 3. মাটি পৰীক্ষাৰ প্ৰতিবেদন আছে নেকি? নিশ্চিত নকৰাকৈ অতিৰিক্ত সাৰ নিদিব। স্থানীয় AEO বা KVK ৰ লগত যোগাযোগ কৰক। 🙏",
    "ur": "کھاد کی صحیح مقدار آپ کی فصل کی قسم، مٹی کی حالت اور پہلے سے ڈالی گئی کھاد پر منحصر ہے۔ مناسب رہنمائی کے لیے: 1. آپ کی فصل کی قسم کون سی ہے؟ 2. بوائی/ٹرانسپلانٹ کے وقت کیا کوئی کھاد (جیسے DAP/SSP) ڈالی گئی تھی؟ 3. کیا مٹی کے ٹیسٹ کی رپورٹ دستیاب ہے؟ تصدیق کے بغیر اضافی کھاد نہ ڈالیں۔ درست مقدار کے لیے مقامی AEO یا KVK سے رابطہ کریں۔ 🙏",
}

UNVERIFIED_FERTILIZER_FALLBACK_RESPONSE_TE = UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES["te"]
UNVERIFIED_FERTILIZER_FALLBACK_RESPONSE_EN = UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES["en"]


def get_unverified_dosage_fallback_response(
    language: str = "te",
    is_fertilizer: bool = False,
    fert_hist_info: dict = None,
) -> str:
    """Return a safe localized fallback when exact fertilizer/chemical dosage is ungrounded in RAG."""
    if is_fertilizer:
        return get_unverified_fertilizer_fallback_response(language, fert_hist_info=fert_hist_info)
    return UNVERIFIED_DOSAGE_FALLBACK_RESPONSES.get(language, UNVERIFIED_DOSAGE_FALLBACK_RESPONSES["te"])


def get_unverified_fertilizer_fallback_response(
    language: str = "te",
    fert_hist_info: dict = None,
) -> str:
    """Return a constructive clarifying fallback when fertilizer dosage is ungrounded in RAG, personalized to known history."""
    default_resp = UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES.get(language, UNVERIFIED_FERTILIZER_FALLBACK_RESPONSES["te"])
    if not fert_hist_info:
        return default_resp

    dap_neg = fert_hist_info.get("dap_negated", False)
    urea_neg = fert_hist_info.get("urea_negated", False)
    has_basal = fert_hist_info.get("has_basal_history", False)
    has_variety = fert_hist_info.get("has_variety_info", False)
    has_soil = fert_hist_info.get("has_soil_test_info", False)
    general_fert_neg = fert_hist_info.get("general_fert_negated", False)

    # If no specific details or negations were provided, return default standard response verbatim
    if not (dap_neg or urea_neg or has_basal or has_variety or has_soil or general_fert_neg):
        return default_resp

    if language == "te":
        questions = []
        if not has_variety:
            questions.append("మీ పంట రకం/వంగడం ఏది?")
        if dap_neg:
            questions.append("నాట్లు/విత్తే సమయంలో SSP లేదా ఇతర కాంప్లెక్స్ ఎరువులు ఏవైనా వేశారా?")
        elif not has_basal and not general_fert_neg:
            questions.append("నాట్లు/విత్తే సమయంలో ఏవైనా ఎరువులు (DAP/SSP/కాంప్లెక్స్) వేశారా?")
        if not has_soil:
            questions.append("నేల పరీక్ష నివేదిక అందుబాటులో ఉందా?")

        if questions:
            q_text = " ".join(f"{i+1}. {q}" for i, q in enumerate(questions))
            return (
                f"ఎరువుల ఖచ్చితమైన మోతాదు మీ పంట రకం, వంగడం, నేల స్వభావం మరియు ఇప్పటికే వేసిన ఎరువులపై ఆధారపడి ఉంటుంది. "
                f"సరైన సిఫార్సు కొరకు: {q_text} "
                f"నిర్ధారించకుండా అదనపు ఎరువులు వేయకండి. ఖచ్చితమైన మోతాదు కోసం స్థానిక AEO లేదా KVK అధికారులను సంప్రదించవచ్చు. 🙏"
            )
    elif language == "en":
        questions = []
        if not has_variety:
            questions.append("Which crop and variety are you growing?")
        if dap_neg:
            questions.append("Was SSP or another complex fertilizer applied at sowing/transplanting?")
        elif not has_basal and not general_fert_neg:
            questions.append("What basal fertilizers (such as DAP/SSP/complex) were applied at sowing/transplanting?")
        if not has_soil:
            questions.append("Do you have a soil test report?")

        if questions:
            q_text = " ".join(f"{i+1}. {q}" for i, q in enumerate(questions))
            return (
                f"The exact fertilizer quantity depends on your crop variety, soil conditions, and previous fertilizer applications. "
                f"To help determine the suitable recommendation: {q_text} "
                f"Please do not apply excess fertilizers without confirming recommendations. You may also consult your local AEO or KVK. 🙏"
            )

    return default_resp


def build_farmer_context(
    crop: str = None,
    district: str = None,
    state: str = None,
    land_size: float = None,
    gps_coordinates: dict = None,
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
    elif gps_coordinates and isinstance(gps_coordinates, dict):
        try:
            lat = float(gps_coordinates.get("latitude") or 0.0)
            lon = float(gps_coordinates.get("longitude") or 0.0)
            if lat != 0.0 and lon != 0.0:
                parts.append(f"GPS: ({round(lat, 6)}, {round(lon, 6)})")
        except (ValueError, TypeError):
            pass
    if state:
        parts.append(f"State: {state}")
    if land_size:
        parts.append(f"Land Size: {land_size} acres")

    if not parts:
        return "[Farmer profile is incomplete. Ask the farmer about their crop and location.]"

    return "[Farmer Profile]\n" + "\n".join(parts)
