"""
BhoomiMitra AI Decision Engine.
Performs deterministic, safety-first evaluations of farmer inquiries
before any generative AI processing or external service calls.
"""

from typing import Optional, Set
import re

from src.decision_engine.models import Decision, DecisionType, FarmerInput, RiskLevel
from src.decision_engine.validators import (
    detect_dosage_requests,
    detect_unknown_or_example_pesticides,
    has_unknown_or_example_pesticides,
    is_crop_missing,
    is_dosage_request,
    is_growth_stage_missing,
    is_problem_missing,
)

# Known crops for fallback context extraction
KNOWN_CROPS_EN: Set[str] = {
    "cotton", "paddy", "rice", "chilli", "chili", "chillies", "maize", "corn",
    "groundnut", "peanut", "tomato", "sugarcane", "wheat", "soybean", "soyabean",
    "pulses", "redgram", "blackgram", "greengram", "turmeric", "onion", "banana",
    "mango", "papaya", "brinjal", "bhendi", "okra"
}

KNOWN_CROPS_TE: Set[str] = {
    "పత్తి", "వరి", "మిర్చి", "మొక్కజొన్న", "వేరుశనగ", "టమోటా", "చెరకు", "గోధుమ",
    "సోయాబీన్", "కంది", "మినుము", "పెసర", "పసుపు", "ఉల్లి", "అరటి", "మామిడి",
    "బొప్పాయి", "వంకాయ", "బెండ"
}

# Common pest/disease/problem keywords
PROBLEM_KEYWORDS_EN: Set[str] = {
    "pest", "disease", "whitefly", "whiteflies", "bollworm", "aphid", "aphids",
    "thrip", "thrips", "caterpillar", "rot", "wilt", "yellowing", "fungus",
    "blast", "leaf spot", "virus", "dry", "dying", "spots", "attack", "eating",
    "damage", "mites", "borer", "rust", "blight", "mildew", "insects", "bugs",
    "weed", "weeds", "root rot", "stem borer"
}

PROBLEM_KEYWORDS_TE: Set[str] = {
    "తెల్లదోమ", "దోమ", "పురుగు", "తెగులు", "ఆకుమచ్చ", "ఎండు", "పచ్చదోమ",
    "తామరపురుగులు", "కాయతొలుచు", "లద్దెపురుగు", "వేరుకుళ్లు", "బూడిద", "తుప్పు",
    "నల్లి", "పల్లాకు", "రసంపీల్చే", "పసుపు", "రాలిపోతున్నాయి", "కలుపు",
    "కాండంతొలుచు", "ఆకుముడత"
}

# Stage-specific nutrient keywords
STAGE_SPECIFIC_KEYWORDS = [
    "fertilizer schedule", "fertilizer dose", "nutrient schedule",
    "when to apply urea", "when to apply dap", "stage fertilizer",
    "యూరియా ఎప్పుడు వేయాలి", "ఎరువుల మోతాదు", "ఎప్పుడు ఎరువు వేయాలి"
]


class DecisionEngine:
    """
    Deterministic Safety & Context Decision Engine for BhoomiMitra AI.
    Evaluates farmer inputs to ensure high safety standards and proper context
    before AI generation.
    """

    def is_telugu(self, text: str) -> bool:
        """Determines if the text contains Telugu script characters."""
        if not text:
            return False
        return any("\u0C00" <= ch <= "\u0C7F" for ch in text)

    def extract_crop(self, text: str) -> Optional[str]:
        """Attempts to identify a known crop mentioned in the raw message."""
        if not text:
            return None
        text_lower = text.lower()
        for crop in KNOWN_CROPS_EN:
            if re.search(r"\b" + re.escape(crop) + r"\b", text_lower):
                return crop.capitalize()
        for crop in KNOWN_CROPS_TE:
            if crop in text:
                return crop
        return None

    def extract_problem(self, text: str) -> Optional[str]:
        """Attempts to identify a known farming problem or symptom in the raw message."""
        if not text:
            return None
        text_lower = text.lower()
        for prob in PROBLEM_KEYWORDS_EN:
            if re.search(r"\b" + re.escape(prob) + r"\b", text_lower):
                return prob
        for prob in PROBLEM_KEYWORDS_TE:
            if prob in text:
                return prob
        return None

    def is_stage_dependent_query(self, text: str) -> bool:
        """Checks if the inquiry requires growth stage (e.g. stage-specific fertilizer schedule)."""
        if not text:
            return False
        text_lower = text.lower()
        return any(kw in text_lower or kw in text for kw in STAGE_SPECIFIC_KEYWORDS)

    def evaluate(self, farmer_input: FarmerInput) -> Decision:
        """
        Evaluates the input query and context against agricultural safety rules.
        Returns a deterministic Decision.
        """
        message = farmer_input.message or ""
        in_telugu = self.is_telugu(message)

        # -------------------------------------------------------------
        # RULE 1: UNKNOWN / EXAMPLE PESTICIDE
        # -------------------------------------------------------------
        if has_unknown_or_example_pesticides(message):
            detected = detect_unknown_or_example_pesticides(message)
            if in_telugu:
                response = (
                    "మీరు పేర్కొన్న రసాయనం/పురుగుమందు గుర్తింపు పొందినది కాదు. "
                    "పంట రక్షణ దృష్ట్యా తెలియని లేదా అనధికారిక మందులను వాడవద్దు. "
                    "దయచేసి సమీపంలోని వ్యవసాయ అధికారి లేదా రైతు భరోసా కేంద్రాన్ని (RBK) సంప్రదించండి."
                )
            else:
                response = (
                    "The mentioned pesticide/product is not recognized as a verified agricultural chemical. "
                    "To prevent crop damage, please avoid using unverified chemicals and consult your local "
                    "Agricultural Officer or Rythu Bharosa Kendra (RBK)."
                )
            return Decision(
                decision_type=DecisionType.SAFE_FALLBACK,
                risk_level=RiskLevel.HIGH,
                response=response,
                reasons=[f"Unknown or example pesticide detected: {', '.join(detected)}"],
                requires_human_review=True,
            )

        # -------------------------------------------------------------
        # RULE 2: DOSAGE REQUEST (Without verified source)
        # -------------------------------------------------------------
        if is_dosage_request(message):
            detected_dosage_terms = detect_dosage_requests(message)
            if in_telugu:
                response = (
                    "రసాయన లేదా పురుగుమందుల ఖచ్చితమైన మోతాదును ఇక్కడ నేరుగా నిర్ధారించలేము. "
                    "తప్పుడు మోతాదు పంటకు మరియు నేలకు నష్టం కలిగిస్తుంది. "
                    "ఖచ్చితమైన మరియు సురక్షితమైన మోతాదు కొరకు దయచేసి ఉత్పత్తి లేబుల్ చూడండి లేదా "
                    "స్థానిక వ్యవసాయ అధికారిని సంప్రదించండి."
                )
            else:
                response = (
                    "Exact pesticide or chemical dosage cannot be safely confirmed without verified product specifications. "
                    "Incorrect chemical dosage can harm crops and soil health. "
                    "Please refer to the official product label or consult your local Agricultural Extension Officer."
                )
            return Decision(
                decision_type=DecisionType.SAFE_FALLBACK,
                risk_level=RiskLevel.HIGH,
                response=response,
                reasons=[f"Dosage requested without verified source: {', '.join(detected_dosage_terms)}"],
                requires_human_review=True,
            )

        # -------------------------------------------------------------
        # RULE 2.5: NON-AGRICULTURAL QUERY
        # -------------------------------------------------------------
        from src.decision_engine.validators import is_non_agricultural
        if is_non_agricultural(message):
            if in_telugu:
                response = "భూమిమిత్ర కేవలం వ్యవసాయం, పంటల సంరక్షణ మరియు రైతు సంక్షేమం గురించిన ప్రశ్నలకు మాత్రమే సమాధానం ఇస్తుంది."
            else:
                response = "BhoomiMitra exclusively assists with agriculture, crop health, farming practices, and farmer welfare."
            return Decision(
                decision_type=DecisionType.SAFE_FALLBACK,
                risk_level=RiskLevel.LOW,
                response=response,
                reasons=["Non-agricultural question detected"],
                requires_human_review=False,
            )

        # Resolve effective crop & problem (from explicit field or extracted from message)
        effective_crop = farmer_input.crop or self.extract_crop(message)
        effective_problem = farmer_input.problem or self.extract_problem(message)

        # -------------------------------------------------------------
        # RULE 3: MISSING CROP
        # -------------------------------------------------------------
        if is_crop_missing(effective_crop):
            if in_telugu:
                response = "మీరు ఏ పంట సాగు చేస్తున్నారో దయచేసి తెలియజేయండి, తద్వారా మేము సరైన సలహా అందించగలము."
            else:
                response = "Please let us know which crop you are growing so we can provide accurate advice."
            return Decision(
                decision_type=DecisionType.ASK_CLARIFICATION,
                risk_level=RiskLevel.LOW,
                response=response,
                reasons=["Crop information is missing"],
                requires_human_review=False,
            )

        # -------------------------------------------------------------
        # RULE 4: MISSING GROWTH STAGE (Only for stage-dependent inquiries)
        # -------------------------------------------------------------
        if self.is_stage_dependent_query(message) and is_growth_stage_missing(farmer_input.growth_stage):
            if in_telugu:
                response = "సరైన ఎరువుల సిఫార్సు కొరకు, మీ పంట ప్రస్తుతం ఏ దశలో ఉందో (ఉదా: ఎదుగుదల దశ, పూత దశ, లేదా కాయ దశ) తెలియజేయండి."
            else:
                response = "To recommend the appropriate fertilizer schedule, please share your crop's current growth stage."
            return Decision(
                decision_type=DecisionType.ASK_CLARIFICATION,
                risk_level=RiskLevel.LOW,
                response=response,
                reasons=["Growth stage required for stage-specific nutrient inquiry"],
                requires_human_review=False,
            )

        # -------------------------------------------------------------
        # RULE 5: MISSING PROBLEM
        # -------------------------------------------------------------
        if is_problem_missing(effective_problem):
            if in_telugu:
                response = "మీ పంటలో కనిపిస్తున్న సమస్య లేదా తెగులు లక్షణాలను దయచేసి వివరించండి."
            else:
                response = "Could you please describe the specific problem or symptoms you are noticing on your crop?"
            return Decision(
                decision_type=DecisionType.ASK_CLARIFICATION,
                risk_level=RiskLevel.LOW,
                response=response,
                reasons=["Farming problem/symptoms not specified"],
                requires_human_review=False,
            )

        # -------------------------------------------------------------
        # RULE 6: NORMAL FARMING QUESTION
        # -------------------------------------------------------------
        return Decision(
            decision_type=DecisionType.ANSWER,
            risk_level=RiskLevel.LOW,
            response="",
            reasons=["Query passed safety validation with sufficient context"],
            requires_human_review=False,
        )
