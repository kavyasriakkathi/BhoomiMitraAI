# 🚀 BhoomiMitra AI — MVP Definition Document

> **Document Type:** Minimum Viable Product (MVP) Plan  
> **Audience:** Engineering Team, Founders, Initial Investors  
> **Objective:** Define the smallest, most impactful version of BhoomiMitra AI that can be tested with real farmers to validate the core hypothesis.

---

## 1. MVP Goal
To validate that farmers will use a WhatsApp-based AI assistant to ask agricultural questions and trust the AI's recommendations enough to act on them.

## 2. Target Users
**Pilot Group:** 100-200 progressive farmers in a single geographic region (e.g., Warangal district, Telangana).
- **Profile:** Owns a smartphone, uses WhatsApp daily, grows a common cash crop (e.g., Cotton or Chilli).

## 3. Core Problem
Farmers lack immediate, reliable, and localized advice for crop issues, forcing them to rely on biased agro-shop owners who upsell chemical pesticides.

---

## 4. MVP Features
*What we are building now.*

1. **WhatsApp Text & Voice Input:** Farmers can send text or voice notes.
2. **Dual-Language Support:** English + One local language (e.g., Telugu).
3. **Basic Onboarding:** Capture farmer location (District) and Crop.
4. **Q&A Advisory Engine:** AI answers questions on fertilizers, pests, and general crop care.
5. **Human Fallback (Wizard of Oz):** If AI confidence is low, the question is routed to a Telegram/Slack channel where a founder/expert types the reply, which is sent back to the farmer via the AI.

## 5. Features Not Included in MVP
*What we are explicitly NOT building yet (to save time).*

1. ❌ Image Upload / Computer Vision for disease detection (too risky/complex for MVP).
2. ❌ Real-time Market Prices integration.
3. ❌ Automated Proactive Reminders (Push notifications).
4. ❌ Agro Shop Dashboards and Nearby Shop routing.
5. ❌ Complex Database structures for Government Schemes.

---

## 6. User Journey
1. **Discovery:** Farmer receives a WhatsApp forward with a link to chat with "BhoomiMitra".
2. **Onboarding:** Farmer sends "Hi". AI asks for language, district, and current crop.
3. **Engagement:** Farmer sends a voice note: *"Na prathi chettu aakulu erraga marutunnayi, em cheyali?"* (My cotton leaves are turning red, what should I do?).
4. **Resolution:** AI translates, consults the LLM, translates back to Telugu, and replies with a text + voice note recommending a magnesium sulfate spray.
5. **Feedback:** AI asks if the advice was helpful (👍/👎).

## 7. Farmer WhatsApp Flow
- **Intent Recognition:** General chat vs. Agricultural query.
- **Context Injection:** AI automatically prepends the farmer's crop and district to the prompt.
- **Safety Gate:** AI refuses non-farming questions.

---

## 8. Technical Requirements
- **WhatsApp Provider:** Meta Cloud API (Free tier).
- **Backend:** FastAPI (Python).
- **AI Core:** Google Gemini 1.5 Flash (Fast, cheap, good regional language support).
- **Voice Services:** Google Cloud Speech-to-Text & Text-to-Speech.
- **Infrastructure:** Deployed on Render or Railway (Low cost, zero DevOps).

## 9. Database Requirements
*Keep it simple: PostgreSQL*
- `farmers`: phone_number, language, district, current_crop.
- `conversations`: farmer_id, message_type (voice/text), content, timestamp.
- `feedback`: conversation_id, rating.

## 10. AI Requirements
- **System Prompt:** Highly constrained prompt explicitly instructed to act as an Indian agronomist. 
- **Knowledge Base:** Small RAG (Retrieval-Augmented Generation) document containing standard practices for the 1-2 pilot crops (e.g., Cotton/Chilli) to prevent hallucination.

---

## 11. Testing Plan
1. **Internal Dogfooding (Week 1):** Team tests the bot with 100+ simulated farmer queries.
2. **Expert Verification (Week 2):** An actual agronomist reviews the AI's answers for safety and accuracy.
3. **Closed Beta (Week 3):** Onboard 10 friendly farmers. Monitor every chat manually.

## 12. Pilot Launch Strategy
- **Acquisition:** Partner with one local village Sarpanch or FPO (Farmer Producer Organization) to distribute the WhatsApp number.
- **Concierge Onboarding:** Call the first 50 farmers personally to explain how to use the voice feature.

## 13. Success Metrics (KPIs)
- **Activation Rate:** % of users who ask at least 1 farming question after onboarding. (Target: 40%)
- **Retention Rate:** % of users who return to ask a second question within 7 days. (Target: 20%)
- **Resolution Rate:** % of queries handled successfully by AI without human intervention. (Target: 70%)
- **Voice Usage:** % of queries sent via voice notes vs. text. (Hypothesis: >60% voice).

---

## 14. Timeline (4-Week Sprint)
- **Week 1:** Set up Meta API, FastAPI boilerplate, and Database.
- **Week 2:** Integrate Gemini, STT/TTS, and craft the System Prompt.
- **Week 3:** Build Human-fallback Slack integration, Internal Testing.
- **Week 4:** Pilot Launch with 100 farmers.

## 15. Future Expansion Plan (Post-MVP)
Once we prove farmers will actually use and trust a WhatsApp chatbot for advisory, we will raise seed funding or expand the team to build:
1. Crop Disease Image Recognition (The most requested feature).
2. Live Mandi price integrations.
3. Scalable Expert Escalation Dashboards.
