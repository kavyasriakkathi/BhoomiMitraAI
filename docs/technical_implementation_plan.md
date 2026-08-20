# 🛠️ BhoomiMitra AI — Technical Implementation Plan (MVP)

> **Document Type:** Technical Roadmap & Implementation Plan  
> **Audience:** Development Team, Engineering Leadership  
> **Objective:** Provide a step-by-step, actionable engineering roadmap to build and deploy the BhoomiMitra AI MVP in a 4-week sprint.

---

## 1. Development Phases & 2. Week-by-Week Implementation Plan

### Sprint 0: Environment Setup (Days 1-2)
- Repository setup, CI/CD scaffolding (GitHub Actions).
- Cloud Accounts creation (Render, Neon DB, Meta Developer Account).
- Tooling setup: Local Docker, Ruff (linter), Pytest.

### Sprint 1: Infrastructure & The Gateway (Week 1)
- **Goal:** Receive a WhatsApp message and save it to the database.
- **Tasks:**
  - Database schema creation (Alembic migrations for `farmers` and `conversations`).
  - FastAPI server setup with webhook endpoint.
  - Meta WhatsApp API validation and token handling.
  - Implement basic incoming message parser.

### Sprint 2: The Core AI & Voice Pipeline (Week 2)
- **Goal:** Process voice, get AI response, send voice back.
- **Tasks:**
  - Integrate Google Cloud STT & TTS APIs.
  - Integrate Gemini 1.5 Flash via LangChain.
  - Craft the System Prompt (The Agronomist Persona).
  - Connect the full pipeline: WhatsApp -> STT -> Prompt -> Gemini -> TTS -> WhatsApp.

### Sprint 3: State Management & Fallback (Week 3)
- **Goal:** Remember user context and handle failures safely.
- **Tasks:**
  - Implement Redis session memory (last 5 messages).
  - Build Farmer Onboarding state machine (capturing Language, Crop, District).
  - Implement "Wizard of Oz" Telegram/Slack webhook for human fallback on low confidence.
  - Security hardening (HMAC verification).

### Sprint 4: Testing, Polish & Deployment (Week 4)
- **Goal:** Ship a stable MVP for 100 farmers.
- **Tasks:**
  - E2E Testing with 5 test WhatsApp numbers.
  - Set up production logging (Datadog/Logtail) and monitoring.
  - Deploy to production (Render/Railway).
  - Final dry run and database wipe before Pilot Launch.

---

## 3. Technology Stack Selection
*Chosen for speed of iteration and low initial cost.*

- **Language/Framework:** Python 3.11 / FastAPI (Async native, perfect for I/O heavy AI tasks).
- **Database:** PostgreSQL (Hosted on Neon.tech for serverless auto-scaling and free tier).
- **Cache & State:** Redis (Hosted on Upstash or Render).
- **AI Core:** Google Gemini 1.5 Flash (Cheaper and faster than GPT-4; excellent at regional Indian languages).
- **Voice Processing:** Google Cloud Speech-to-Text & Text-to-Speech (Best support for Telugu/Hindi).
- **Deployment:** Render (PaaS - zero DevOps required for MVP).

---

## 4. Backend Setup Plan
- Structure as a Modular Monolith using standard MVC-like patterns.
- Pydantic for all data validation (incoming WhatsApp payloads, AI outputs).
- Dependency Injection for database sessions and configuration to make testing easier.

## 5. Database Implementation Plan
- Use **SQLAlchemy (Async)** as the ORM.
- **Alembic** for version-controlled schema migrations.
- MVP Tables: 
  1. `farmers` (id, phone, language, state, district, crop).
  2. `conversations` (id, farmer_id, message_id, text, type, timestamp).

## 6. WhatsApp Integration Plan
- Use the **Meta Cloud API** (avoiding on-premise WhatsApp Business API servers to save time).
- Create a single robust webhook (`POST /webhook/whatsapp`).
- Use a background task (`asyncio.create_task` or FastAPI `BackgroundTasks`) to process the payload immediately after returning `200 OK` to Meta to prevent timeout loops.

## 7. AI Integration Plan
- Use the official `google-generativeai` SDK.
- **System Prompt Design:** Constrain the AI strictly. *“You are BhoomiMitra. You speak to Indian farmers. Do not use complex jargon. If asked about non-farming topics, politely refuse. Never invent pesticide names.”*
- Append conversation history (from Redis) and farmer profile (from DB) to every API call.

## 8. Voice Processing Plan
- WhatsApp sends audio in `OGG/OPUS` format.
- **Pipeline:** Download from Meta CDN -> Send bytes directly to Google STT -> Receive Text -> AI Processing -> Send Text to Google TTS -> Receive MP3/OGG -> Upload to Meta CDN -> Send to Farmer.
- **Note:** Keep audio processing in memory where possible to avoid disk I/O bottlenecks.

## 9. Farmer Profile Implementation
- Implement a lightweight state machine.
- If a farmer's DB record has `crop == NULL`, the system intercepts the message and forces the onboarding flow ("Before we start, what crop are you growing?").
- Once the profile is complete, normal AI Q&A resumes.

---

## 10. Testing Strategy
- **Unit Tests:** Test the Prompt builder, Database CRUD, and Webhook signature validation locally without hitting APIs.
- **Integration Tests:** Use mocked responses for Meta and Gemini to test the full pipeline flow.
- **Manual QA:** Create a staging WhatsApp Business number for internal team dogfooding.

## 11. Deployment Plan
- Push to the `main` branch triggers a GitHub Action.
- GitHub Action runs Pytest and Ruff.
- If green, Render auto-deploys the FastAPI server.
- Environment variables (API keys, DB URLs) are stored securely in the Render dashboard.

---

## 12. Development Team Requirements
*For this 4-week sprint, a tiny, cross-functional team is ideal:*
- **1 Lead Backend/AI Engineer:** (Handles FastAPI, LLM integration, WhatsApp API).
- **1 Product Manager / Agronomist:** (Designs the system prompt, tests outputs, handles Slack fallback).

## 13. Cost Estimation (Monthly, for 200 MVP Farmers)
- **Compute (Render):** $15/mo (Basic tier).
- **Database (Neon):** $0/mo (Free tier fits MVP data).
- **Redis (Upstash):** $0/mo.
- **AI Token Cost (Gemini):** ~$10 - $20/mo (Highly dependent on usage).
- **Voice APIs (Google):** ~$20/mo.
- **WhatsApp API:** First 1,000 service conversations are free per month.
- **Total SaaS Cost:** **~$50/month**.

## 14. Risks and Solutions
| Risk | Impact | Solution |
|------|--------|----------|
| **WhatsApp Webhook Timeouts** | Meta resends messages, causing duplicate AI replies. | Offload processing to background tasks immediately. Deduplicate by logging Meta `message_id`. |
| **AI Hallucination** | Farmer receives wrong pesticide dosage. | Strict prompt engineering. Explicit instruction to trigger human fallback if uncertain. |
| **Voice Translation Errors** | Regional accents cause bad STT transcription. | Keep prompts short. Replay the transcribed text to the farmer ("Did you mean...") if STT confidence is low. |
| **High Latency** | Farmer waits >10 seconds for a reply. | Stream AI text first, send the voice note as a follow-up message a few seconds later. |
