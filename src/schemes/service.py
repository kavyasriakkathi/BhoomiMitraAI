from typing import List, Optional, Tuple
from uuid import UUID
from fastapi import HTTPException, status
from src.core.models import GovernmentScheme, SchemeApplication, Farmer, FarmerProfile
from src.schemes.repository import SchemeRepository
from src.farmers.repository import FarmerRepository
from src.schemes.schemas import (
    GovernmentSchemeCreate,
    GovernmentSchemeResponse,
    FarmerEligibilityResponse,
    SchemeEligibilityItem,
    SchemeApplicationCreate,
    SchemeApplicationResponse,
)


class SchemeService:
    def __init__(self, repository: SchemeRepository, farmer_repository: FarmerRepository):
        self.repository = repository
        self.farmer_repository = farmer_repository

    async def seed_defaults_if_empty(self) -> List[GovernmentSchemeResponse]:
        schemes = await self.repository.seed_default_schemes_if_empty()
        return [GovernmentSchemeResponse.model_validate(s) for s in schemes]

    async def list_schemes(self, state: Optional[str] = None, category: Optional[str] = None) -> List[GovernmentSchemeResponse]:
        # Ensure default schemes exist
        await self.repository.seed_default_schemes_if_empty()
        schemes = await self.repository.get_all_active(state=state, category=category)
        return [GovernmentSchemeResponse.model_validate(s) for s in schemes]

    async def create_scheme(self, data: GovernmentSchemeCreate) -> GovernmentSchemeResponse:
        existing = await self.repository.get_by_code(data.scheme_code)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Scheme with code '{data.scheme_code}' already exists."
            )

        scheme = GovernmentScheme(**data.model_dump())
        created = await self.repository.create_scheme(scheme)
        return GovernmentSchemeResponse.model_validate(created)

    async def evaluate_farmer_eligibility(self, farmer_id: UUID) -> FarmerEligibilityResponse:
        """
        AI Government Schemes Eligibility Engine
        Analyzes farmer's state, district, land size, and crop to determine match percentage,
        eligibility reasons, and recommended action steps with voice explanation.
        """
        farmer = await self.farmer_repository.get_by_id(farmer_id)
        if not farmer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Farmer with ID '{farmer_id}' not found."
            )

        # Seed defaults
        schemes = await self.repository.seed_default_schemes_if_empty()

        import inspect
        profile = getattr(farmer, "profile", None)
        if inspect.iscoroutine(profile):
            profile = await profile
        farmer_state = (profile.state if profile and isinstance(getattr(profile, "state", None), str) and profile.state else "Telangana").strip()
        farmer_district = (profile.district if profile and isinstance(getattr(profile, "district", None), str) and profile.district else "Jagtial").strip()
        farmer_land = profile.land_size_acres if profile and getattr(profile, "land_size_acres", None) is not None else 5.0
        farmer_name = profile.full_name if profile and getattr(profile, "full_name", None) else "Farmer"

        evaluated_items: List[SchemeEligibilityItem] = []
        eligible_count = 0

        for scheme in schemes:
            is_eligible = True
            reasons = []
            score = 100

            # 1. State matching check
            if scheme.state.lower() != "all india" and farmer_state.lower() not in scheme.state.lower():
                is_eligible = False
                score -= 40
                reasons.append(f"Scheme is specific to {scheme.state} state.")
            else:
                reasons.append(f"State '{farmer_state}' matches target region ({scheme.state}).")

            # 2. Land Size Check
            if farmer_land < scheme.min_land_acres:
                is_eligible = False
                score -= 30
                reasons.append(f"Requires minimum land of {scheme.min_land_acres} acres (You have {farmer_land} acres).")
            elif scheme.max_land_acres is not None and farmer_land > scheme.max_land_acres:
                is_eligible = False
                score -= 30
                reasons.append(f"Requires land size up to {scheme.max_land_acres} acres (You have {farmer_land} acres).")
            else:
                reasons.append(f"Land size ({farmer_land} acres) fits scheme land boundaries.")

            if is_eligible:
                eligible_count += 1
                recommended_action = f"Gather required documents ({scheme.required_documents}) and apply online or at Meeseva/CSC center."
                voice_exp = f"Namaste {farmer_name}! You are 100% eligible for {scheme.scheme_name}. Benefits include {scheme.benefits_summary}. Recommended action: {recommended_action}"
            else:
                recommended_action = f"Check alternative schemes or update land records."
                voice_exp = f"{scheme.scheme_name} is currently not matching your profile: {' '.join(reasons)}"

            evaluated_items.append(
                SchemeEligibilityItem(
                    scheme=GovernmentSchemeResponse.model_validate(scheme),
                    is_eligible=is_eligible,
                    match_score_percentage=max(score, 20),
                    eligibility_reason=" | ".join(reasons),
                    recommended_action=recommended_action,
                    voice_explanation=voice_exp
                )
            )

        # Sort eligible schemes first, then by match score
        evaluated_items.sort(key=lambda x: (x.is_eligible, x.match_score_percentage), reverse=True)

        return FarmerEligibilityResponse(
            farmer_id=farmer_id,
            farmer_name=farmer_name,
            state=farmer_state,
            district=farmer_district,
            land_size_acres=farmer_land,
            total_schemes_evaluated=len(schemes),
            eligible_schemes_count=eligible_count,
            schemes=evaluated_items
        )

    async def apply_for_scheme(self, data: SchemeApplicationCreate) -> SchemeApplicationResponse:
        farmer = await self.farmer_repository.get_by_id(data.farmer_id)
        if not farmer:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Farmer not found.")

        scheme = await self.repository.get_by_id(data.scheme_id)
        if not scheme:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Government scheme not found.")

        app = SchemeApplication(
            farmer_id=data.farmer_id,
            scheme_id=data.scheme_id,
            status="Applied",
            notes=data.notes or f"Application submitted via BhoomiMitra AI for {scheme.scheme_name}"
        )
        created = await self.repository.create_application(app)
        
        resp = SchemeApplicationResponse.model_validate(created)
        resp.scheme_name = scheme.scheme_name
        return resp

    async def get_farmer_applications(self, farmer_id: UUID) -> List[SchemeApplicationResponse]:
        apps = await self.repository.get_farmer_applications(farmer_id)
        results = []
        for a in apps:
            resp = SchemeApplicationResponse.model_validate(a)
            if a.scheme:
                resp.scheme_name = a.scheme.scheme_name
            results.append(resp)
        return results


# ---------------------------------------------------------------------------
# Intent Detection Keywords
# ---------------------------------------------------------------------------

_SCHEME_KEYWORDS_EN = {
    "scheme", "schemes", "subsidy", "subsidies", "yojana", "kisan",
    "fasal bima", "insurance", "credit", "kcc", "solar pump", "government",
    "apply", "eligible", "eligibility", "benefits", "pm kisan",
    "rythu bandhu", "rythu bharosa", "kusum", "pmfby", "financial assistance",
    "govt grant", "krishi yojana",
}
_SCHEME_KEYWORDS_TE = {
    "పథకం", "పథకాలు", "సబ్సిడీ", "సబ్సిడీలు", "పంట బీమా", "యోజన",
    "కిసాన్", "ప్రభుత్వ", "రైతు బంధు", "రైతు భరోసా", "అర్హత", "ప్రయోజనాలు",
    "క్రెడిట్ కార్డ్", "సౌర పంప్", "ఆర్థిక సహాయం", "గ్రాంట్", "రుణం",
    "పీఎం కిసాన్", "రైతు సంక్షేమం",
}

# Known crop nouns for future crop-specific prioritisation (not used for exclusion).
_CROP_KEYWORDS: dict = {
    "cotton": "Cotton",   "పత్తి": "Cotton",
    "rice": "Rice",       "వరి": "Rice",       "ధాన్యం": "Rice",
    "maize": "Maize",    "మొక్కజొన్న": "Maize",
    "wheat": "Wheat",    "గోధుమ": "Wheat",
    "chilli": "Chilli",  "మిర్చి": "Chilli",
    "tomato": "Tomato",  "టమాటా": "Tomato",
    "groundnut": "Groundnut", "వేరుశనగ": "Groundnut",
    "soybean": "Soybean", "సోయాబీన్": "Soybean",
    "turmeric": "Turmeric", "పసుపు": "Turmeric",
}

# ---------------------------------------------------------------------------
# Static WhatsApp Response Labels
# ---------------------------------------------------------------------------

_TE_LABELS = {
    "title":       "🏛️ మీకు వర్తించే ప్రభుత్వ పథకాలు ({count} పథకాలు)",
    "benefits":    "💰 ప్రయోజనాలు",
    "eligibility": "✅ అర్హత",
    "documents":   "📄 అవసరమైన పత్రాలు",
    "deadline":    "📅 దరఖాస్తు గడువు",
    "portal":      "🔗 అధికారిక పోర్టల్",
    "no_portal":   "సమీప మీసేవా / CSC కేంద్రాన్ని సంప్రదించండి",
    "disclaimer":  (
        "⚠️ గమనిక: దరఖాస్తు చేసే ముందు అధికారిక పోర్టల్‌లో "
        "తాజా వివరాలు మరియు అర్హత నిబంధనలు నిర్ధారించుకోండి. "
        "ప్రభుత్వ నిబంధనలు ఎప్పుడైనా మారవచ్చు."
    ),
    "more":        "మరిన్ని పథకాల కోసం: /schemes",
    "crop_note":   "(మీరు పేర్కొన్న పంట: {crop})",
    "unavailable": "ప్రస్తుతం ప్రభుత్వ పథకాల సమాచారం అందుబాటులో లేదు. దయచేసి సమీప రైతు వేదిక లేదా https://myscheme.gov.in పోర్టల్‌ని చూడండి.",
}
_EN_LABELS = {
    "title":       "🏛️ Government Schemes Available For You ({count} schemes)",
    "benefits":    "💰 Benefits",
    "eligibility": "✅ Eligibility",
    "documents":   "📄 Required Documents",
    "deadline":    "📅 Application Deadline",
    "portal":      "🔗 Official Portal",
    "no_portal":   "Contact your nearest Meeseva / CSC centre",
    "disclaimer":  (
        "⚠️ Note: Please verify scheme details, amounts and deadlines at the "
        "official portal before applying. Government rules may change at any time."
    ),
    "more":        "See all schemes at: /schemes",
    "crop_note":   "(Crop you mentioned: {crop})",
    "unavailable": "Government schemes information is currently unavailable. Please check https://myscheme.gov.in or your local agriculture office.",
}

_LABELS_BY_LANG = {
    "te": _TE_LABELS,
    "en": _EN_LABELS,
    "hi": {
        "title":       "🏛️ आपके लिए उपलब्ध सरकारी योजनाएं ({count} योजनाएं)",
        "benefits":    "💰 लाभ",
        "eligibility": "✅ पात्रता",
        "documents":   "📄 आवश्यक दस्तावेज",
        "deadline":    "📅 आवेदन की अंतिम तिथि",
        "portal":      "🔗 आधिकारिक पोर्टल",
        "no_portal":   "निकटतम सीएससी / ग्राहक सेवा केंद्र से संपर्क करें",
        "disclaimer":  "⚠️ नोट: कृपया आवेदन करने से पहले आधिकारिक पोर्टल पर विवरण की पुष्टि करें।",
        "more":        "सभी योजनाओं के लिए देखें: /schemes",
        "crop_note":   "(आपकी फसल: {crop})",
        "unavailable": "सरकारी योजनाओं की जानकारी वर्तमान में उपलब्ध नहीं है।",
    },
    "ta": {
        "title":       "🏛️ உங்களுக்கான அரசு திட்டங்கள் ({count} திட்டங்கள்)",
        "benefits":    "💰 நன்மைகள்",
        "eligibility": "✅ தகுதி",
        "documents":   "📄 தேவையான ஆவணங்கள்",
        "deadline":    "📅 விண்ணப்பிக்க கடைசி தேதி",
        "portal":      "🔗 அதிகாரப்பூர்வ இணையதளம்",
        "no_portal":   "அருகிலுள்ள இ-சேவை மையத்தை தொடர்பு கொள்ளவும்",
        "disclaimer":  "⚠️ குறிப்பு: விண்ணப்பிக்கும் முன் அதிகாரப்பூர்வ தளத்தில் சரிபார்க்கவும்.",
        "more":        "அனைத்து திட்டங்களுக்கும்: /schemes",
        "crop_note":   "(பயிர்: {crop})",
        "unavailable": "அரசு திட்டங்கள் தகவல் தற்போது கிடைக்கவில்லை.",
    },
    "kn": {
        "title":       "🏛️ ನಿಮಗಾಗಿ ಲಭ್ಯವಿರುವ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ({count} ಯೋಜನೆಗಳು)",
        "benefits":    "💰 ಪ್ರಯೋಜನಗಳು",
        "eligibility": "✅ ಅರ್ಹತೆ",
        "documents":   "📄 ಅಗತ್ಯ ದಾಖಲೆಗಳು",
        "deadline":    "📅 ಅರ್ಜಿ ಸಲ್ಲಿಸಲು ಕೊನೆಯ ದಿನಾಂಕ",
        "portal":      "🔗 ಅಧಿಕೃತ ಪೋರ್ಟಲ್",
        "no_portal":   "ಹತ್ತಿರದ ಸಿಎಸ್‌ಸಿ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ",
        "disclaimer":  "⚠️ ಗಮನಿಸಿ: ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ಮೊದಲು ಅಧಿಕೃತ ಪೋರ್ಟಲ್‌ನಲ್ಲಿ ಪರಿಶೀಲಿಸಿ.",
        "more":        "ಹೆಚ್ಚಿನ ಯೋಜನೆಗಳಿಗಾಗಿ: /schemes",
        "crop_note":   "(ಬೆಳೆ: {crop})",
        "unavailable": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಮಾಹಿತಿ ಪ್ರಸ್ತುತ ಲಭ್ಯವಿಲ್ಲ.",
    },
    "ml": {
        "title":       "🏛️ ലഭ്യമായ സർക്കാർ പദ്ധതികൾ ({count} എണ്ണം)",
        "benefits":    "💰 ആനുകൂല്യങ്ങൾ",
        "eligibility": "✅ യോഗ്യത",
        "documents":   "📄 ആവശ്യമായ രേഖകൾ",
        "deadline":    "📅 അവസാന തീയതി",
        "portal":      "🔗 ഔദ്യോഗിക പോർട്ടൽ",
        "no_portal":   "അക്ഷയ കേന്ദ്രവുമായി ബന്ധപ്പെടുക",
        "disclaimer":  "⚠️ ശ്രദ്ധിക്കുക: അപേക്ഷിക്കുന്നതിന് മുമ്പ് ഔദ്യോഗിക പോർട്ടലിൽ പരിശോധിക്കുക.",
        "more":        "കൂടുതൽ പദ്ധതികൾക്ക്: /schemes",
        "crop_note":   "(വിള: {crop})",
        "unavailable": "പദ്ധതി വിവരങ്ങൾ ലഭ്യമല്ല.",
    },
    "mr": {
        "title":       "🏛️ आपल्यासाठी उपलब्ध शासकीय योजना ({count} योजना)",
        "benefits":    "💰 फायदे",
        "eligibility": "✅ पात्रता",
        "documents":   "📄 आवश्यक कागदपत्रे",
        "deadline":    "📅 अर्जाची अंतिम मुदत",
        "portal":      "🔗 अधिकृत पोर्टल",
        "no_portal":   "जवळच्या सीएससी केंद्राशी संपर्क साधा",
        "disclaimer":  "⚠️ टीप: अर्ज करण्यापूर्वी अधिकृत पोर्टलवर माहिती तपासा.",
        "more":        "सर्व योजनांसाठी: /schemes",
        "crop_note":   "(पीक: {crop})",
        "unavailable": "शासकीय योजनांची माहिती सध्या उपलब्ध नाही.",
    },
    "bn": {
        "title":       "🏛️ আপনার জন্য সরকারি প্রকল্পসমূহ ({count} টি প্রকল্প)",
        "benefits":    "💰 সুবিধাসমূহ",
        "eligibility": "✅ যোগ্যতা",
        "documents":   "📄 প্রয়োজনীয় নথিপত্র",
        "deadline":    "📅 আবেদনের শেষ তারিখ",
        "portal":      "🔗 অফিসিয়াল পোর্টাল",
        "no_portal":   "নিকটবর্তী সিএসসি সেন্টারে যোগাযোগ করুন",
        "disclaimer":  "⚠️ দ্রষ্টব্য: আবেদনের পূর্বে অফিসিয়াল পোর্টালে যাচাই করুন।",
        "more":        "সকল প্রকল্পের জন্য: /schemes",
        "crop_note":   "(ফসল: {crop})",
        "unavailable": "সরকারি প্রকল্পের তথ্য বর্তমানে উপলব্ধ নেই।",
    },
    "gu": {
        "title":       "🏛️ તમારા માટે ઉપલબ્ધ સરકારી યોજનાઓ ({count} યોજનાઓ)",
        "benefits":    "💰 લાભો",
        "eligibility": "✅ પાત્રતા",
        "documents":   "📄 જરૂરી દસ્તાવેજો",
        "deadline":    "📅 અરજીની છેલ્લી તારીખ",
        "portal":      "🔗 સત્તાવાર પોર્ટલ",
        "no_portal":   "નજીકના સીએસસી કેન્દ્રનો સંપર્ક કરો",
        "disclaimer":  "⚠️ નોંધ: અરજી કરતા પહેલા સત્તાવાર પોર્ટલ પર ચકાસણી કરો.",
        "more":        "બધી યોજનાઓ માટે: /schemes",
        "crop_note":   "(પાક: {crop})",
        "unavailable": "સરકારી યોજનાઓની માહિતી ઉપલબ્ધ નથી.",
    },
    "or": {
        "title":       "🏛️ ଆପଣଙ୍କ ପାଇଁ ସରକାରୀ ଯୋଜନା ({count} ଟି ଯୋଜନା)",
        "benefits":    "💰 ଲାଭ",
        "eligibility": "✅ ଯୋଗ୍ୟତା",
        "documents":   "📄 ଆବଶ୍ୟକୀୟ କାଗଜପତ୍ର",
        "deadline":    "📅 ଆବେଦନର ଶେଷ ତାରିଖ",
        "portal":      "🔗 ସରକାରୀ ପୋର୍ଟାଲ",
        "no_portal":   "ନିକଟସ୍ଥ ଜନସେବା କେନ୍ଦ୍ର ସହ ଯୋଗାଯୋଗ କରନ୍ତୁ",
        "disclaimer":  "⚠️ ସୂଚନା: ଆବେଦନ ପୂର୍ବରୁ ସରକାରୀ ପୋର୍ଟାଲରେ ଯାଞ୍ଚ କରନ୍ତୁ।",
        "more":        "ସମସ୍ତ ଯୋଜନା ପାଇଁ: /schemes",
        "crop_note":   "(ଫସଲ: {crop})",
        "unavailable": "ସରକାରୀ ଯୋଜନା ସୂଚନା ଉପଲବ୍ଧ ନାହିଁ।",
    },
    "pa": {
        "title":       "🏛️ ਤੁਹਾਡੇ ਲਈ ਉਪਲਬਧ ਸਰਕਾਰੀ ਸਕੀਮਾਂ ({count} ਸਕੀਮਾਂ)",
        "benefits":    "💰 ਲਾਭ",
        "eligibility": "✅ ਯੋਗਤਾ",
        "documents":   "📄 ਲੋੜੀਂਦੇ ਦਸਤਾਵੇਜ਼",
        "deadline":    "📅 ਅਰਜ਼ੀ ਦੀ ਆਖਰੀ ਮਿਤੀ",
        "portal":      "🔗 ਅਧਿਕਾਰਤ ਪੋਰਟਲ",
        "no_portal":   "ਨੇੜਲੇ ਸੀਐਸਸੀ ਕੇਂਦਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ",
        "disclaimer":  "⚠️ ਨੋਟ: ਅਰਜ਼ੀ ਦੇਣ ਤੋਂ ਪਹਿਲਾਂ ਅਧਿਕਾਰਤ ਪੋਰਟਲ 'ਤੇ ਪੜਤਾਲ ਕਰੋ।",
        "more":        "ਸਾਰੀਆਂ ਸਕੀਮਾਂ ਲਈ: /schemes",
        "crop_note":   "(ਫਸਲ: {crop})",
        "unavailable": "ਸਰਕਾਰੀ ਸਕੀਮਾਂ ਦੀ ਜਾਣਕਾਰੀ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
    },
    "as": {
        "title":       "🏛️ আপোনাৰ বাবে চৰকাৰী আঁচনিসমূহ ({count} খন আঁচনি)",
        "benefits":    "💰 সুবিধাসমূহ",
        "eligibility": "✅ যোগ্যতা",
        "documents":   "📄 প্ৰয়োজনীয় নথিপত্ৰ",
        "deadline":    "📅 আবেদনৰ অন্তিম তাৰিখ",
        "portal":      "🔗 চৰকাৰী পৰ্টেল",
        "no_portal":   "ওচৰৰ চিএছচি কেন্দ্ৰৰ সৈতে যোগাযোগ কৰক",
        "disclaimer":  "⚠️ মন কৰিব: আবেদন কৰাৰ পূৰ্বে পৰ্টেলত পৰীক্ষা কৰক।",
        "more":        "সকলো আঁচনিৰ বাবে: /schemes",
        "crop_note":   "(শস্য: {crop})",
        "unavailable": "চৰকাৰী আঁচনিৰ তথ্য উপলব্ধ নহয়।",
    },
    "ur": {
        "title":       "🏛️ آپ کے لیے دستیاب سرکاری اسکیمیں ({count} اسکیمیں)",
        "benefits":    "💰 فوائد",
        "eligibility": "✅ اہلیت",
        "documents":   "📄 مطلوبہ دستاویزات",
        "deadline":    "📅 درخواست کی آخری تاریخ",
        "portal":      "🔗 سرکاری پورٹل",
        "no_portal":   "قریبی خدمت مرکز سے رابطہ کریں",
        "disclaimer":  "⚠️ نوٹ: درخواست دینے سے پہلے سرکاری پورٹل پر تصدیق کریں۔",
        "more":        "تمام اسکیموں کے لیے: /schemes",
        "crop_note":   "(فصل: {crop})",
        "unavailable": "سرکاری اسکیموں کی معلومات دستیاب نہیں ہیں۔",
    },
}


def _detect_scheme_intent(query_lower: str) -> bool:
    """Return True if the query contains a government-scheme keyword."""
    if any(kw in query_lower for kw in _SCHEME_KEYWORDS_EN):
        return True
    # Telugu keywords need exact substring match (not lowercased)
    return False


def _detect_crop_from_query(query_text: str) -> Optional[str]:
    """
    Return a normalised English crop name if the farmer mentions a known crop.
    Used for future prioritisation — never for scheme exclusion.
    """
    query_lower = query_text.lower()
    for kw, normalised in _CROP_KEYWORDS.items():
        if kw in query_lower or kw in query_text:
            return normalised
    return None


def _sort_schemes_by_crop_priority(
    schemes: list,
    mentioned_crop: Optional[str],
) -> list:
    """
    Sort schemes so that crop-specific matches appear first.
    "All Crops" schemes are always included — never excluded.
    Schemes matching the mentioned crop get a higher sort priority.
    """
    def priority(scheme) -> int:
        crop_type = (scheme.crop_type or "All Crops").lower()
        if crop_type == "all crops":
            return 1  # Second priority — always applicable
        if mentioned_crop and mentioned_crop.lower() in crop_type:
            return 0  # Highest priority — exact crop match
        return 2     # Lower priority — different specific crop

    return sorted(schemes, key=priority)


def _format_scheme_block(scheme, labels: dict, language: str) -> str:
    """Format a single scheme as a WhatsApp-friendly text block."""
    deadline_str = ""
    if scheme.application_deadline:
        try:
            from datetime import datetime, timezone
            deadline_dt = scheme.application_deadline
            is_past = False
            try:
                now = datetime.now(timezone.utc)
                if hasattr(deadline_dt, "tzinfo") and deadline_dt.tzinfo is not None:
                    is_past = deadline_dt < now
                else:
                    is_past = deadline_dt < datetime.utcnow()
            except Exception:
                is_past = False

            date_fmt = deadline_dt.strftime("%d %b %Y")
            if is_past:
                if language == "te":
                    deadline_str = f"{date_fmt} (గడువు ముగిసింది - తదుపరి సైకిల్ త్వరలో)"
                else:
                    deadline_str = f"{date_fmt} (Deadline Closed - Next cycle opening soon)"
            else:
                deadline_str = date_fmt
        except Exception:
            deadline_str = str(scheme.application_deadline)

    portal_str = scheme.official_portal_url or labels["no_portal"]

    lines = [
        f"\n*{scheme.scheme_name}*",
        f"{labels['benefits']}: {scheme.benefits_summary}",
        f"{labels['eligibility']}: {scheme.eligibility_criteria}",
        f"{labels['documents']}: {scheme.required_documents}",
    ]
    if deadline_str:
        lines.append(f"{labels['deadline']}: {deadline_str}")
    lines.append(f"{labels['portal']}: {portal_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline Integration Function — called from ai/service.py
# ---------------------------------------------------------------------------

async def enrich_response_with_schemes(
    db,
    query_text: str,
    ai_response: str,
    farmer,
) -> str:
    """
    Detect government-scheme intent in the farmer's query.
    If detected, append a formatted, safety-disclaimed scheme block to the AI response.

    Always returns the original ai_response unchanged if:
    - No scheme intent is detected
    - No schemes are found in the database
    - Any exception occurs
    """
    from src.core.logging import logger

    query_lower = query_text.lower()

    # Step 1: Detect intent
    from src.ai.decision_engine import SCHEMES_KEYWORDS_MULTILINGUAL, SCHEMES_KEYWORDS_TANGLISH
    has_intent = _detect_scheme_intent(query_lower)
    if not has_intent:
        has_intent = any(kw in query_text for kw in _SCHEME_KEYWORDS_TE)
    if not has_intent:
        has_intent = any(kw in query_text for kw in SCHEMES_KEYWORDS_MULTILINGUAL) or any(kw in query_lower for kw in SCHEMES_KEYWORDS_TANGLISH)
    if not has_intent:
        return ai_response

    # Step 2: Identify crop mentioned (for prioritisation only — not exclusion)
    mentioned_crop: Optional[str] = _detect_crop_from_query(query_text)

    # Step 3: Resolve farmer location for state filtering
    farmer_state: Optional[str] = None
    from src.language.detector import detect_language
    pref_lang = getattr(farmer, "preferred_language", "en") or "en"
    language = detect_language(query_text, fallback=pref_lang)
    labels = _LABELS_BY_LANG.get(language, _EN_LABELS if language == "en" else _TE_LABELS)

    try:
        import inspect
        from sqlalchemy import select
        from src.core.models import FarmerProfile
        if farmer and hasattr(farmer, "id"):
            profile_result = await db.execute(
                select(FarmerProfile).where(FarmerProfile.farmer_id == farmer.id)
            )
            if inspect.iscoroutine(profile_result):
                profile_result = await profile_result
            profile = profile_result.scalar_one_or_none() if hasattr(profile_result, "scalar_one_or_none") else None
            if inspect.iscoroutine(profile):
                profile = await profile
            if profile and isinstance(getattr(profile, "state", None), str) and profile.state:
                farmer_state = profile.state.strip()
    except Exception as loc_err:
        logger.warning(f"[SCHEMES ENRICH] Could not resolve farmer state: {loc_err}")

    # Step 4: Fetch matching schemes from DB
    try:
        repo = SchemeRepository(db)
        # Seed defaults if the table is empty (idempotent)
        await repo.seed_default_schemes_if_empty()
        schemes = await repo.get_all_active(state=farmer_state)
    except Exception as db_err:
        logger.warning(f"[SCHEMES ENRICH] DB query failed: {db_err}")
        return ai_response

    if not schemes:
        logger.info("[SCHEMES ENRICH] No active schemes found — returning original response.")
        return ai_response

    # Step 5: Sort — crop-specific matches first, then "All Crops", then others
    ranked = _sort_schemes_by_crop_priority(schemes, mentioned_crop)

    # Step 6: Cap at top 3
    top = ranked[:3]

    # Step 7: Format WhatsApp reply block
    labels = _TE_LABELS if language == "te" else _EN_LABELS
    header = labels["title"].format(count=len(top))
    if mentioned_crop:
        header += "\n" + labels["crop_note"].format(crop=mentioned_crop)

    scheme_blocks = [_format_scheme_block(s, labels, language) for s in top]
    footer_parts = [labels["disclaimer"]]
    if len(schemes) > 3:
        footer_parts.append(labels["more"])

    full_block = "\n".join([
        header,
        *scheme_blocks,
        "",
        "\n".join(footer_parts),
    ])

    logger.info(
        f"[SCHEMES ENRICH] Appending {len(top)} schemes "
        f"(crop hint: {mentioned_crop}, state: {farmer_state})."
    )
    return ai_response + "\n\n" + full_block
