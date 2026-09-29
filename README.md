# 🌾 BhoomiMitra AI

**Production-Grade AI WhatsApp & Web Farming Assistant for Indian Farmers**

[![Python Version](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](#license)
[![Tests](https://img.shields.io/badge/tests-1085%20passed-brightgreen.svg)](#testing)

BhoomiMitra AI is an intelligent agricultural assistant tailored for Indian farmers. It operates directly through **WhatsApp** (text, voice audio, crop leaf photos) and an accompanying **Web Dashboard**, providing personalized agronomic guidance grounded in official agricultural research (ICAR, PJTSAU, ANGRAU).

---

## 📌 Key Highlights

- **Zero-Hallucination Chemical Safety**: Refuses to invent pesticide dosages or chemical combinations. Grounded in authentic university Packages of Practices.
- **Multilingual Native Interaction**: Native support for **Telugu (తెలుగు)**, **Hindi (हिन्दी)**, and **English**, responding strictly in the farmer's language.
- **Multilingual Voice Pipeline**: Seamless voice-note ingestion and native speech responses across 13 Indian languages and dialects (including Tanglish).
- **Proactive Stock Siren**: Instant dual-channel alerts (phone-level audible FCM siren + direct WhatsApp notifications) when out-of-stock fertilizers or seeds are replenished at local dealers.
- **Multimodal Visual Diagnosis**: Vision-assisted crop disease detection with cautious non-definitive diagnostics.
- **Long-Term Farmer Memory**: Tracks soil type, acreage, crop cycles, disease history, and preferences across sessions.
- **Live Mandi & Market Prices**: Real-time mandi commodity price intelligence with local database fallback.
- **Localized Weather Forecasts**: 5-day / 3-hour hyperlocal weather forecasts via OpenWeatherMap with rain and climate alerts.
- **Hyperlocal Input Commerce**: Geo-spatial Haversine discovery of registered input shops, inventory stock checks, and purchase orders.
- **Government Subsidies & Schemes**: Automated eligibility evaluation for PM-Kisan, PMFBY crop insurance, and state schemes.
- **Human Expert Escalation**: Safe escalation to Agricultural Extension Officers (AEOs) for hazardous chemicals, complex diagnostics, or physical farm inspections.
- **Secure Digital Payments**: Integrated Razorpay checkout with cryptographic HMAC-SHA256 signature verification.

---

## 🏛️ System Architecture

```
                               ┌─────────────────────────────────────────┐
                               │   Farmer (WhatsApp / Web Dashboard)     │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │  Gateway & Security (Meta Graph API)    │
                               │  - HMAC-SHA256 Signature Verification   │
                               │  - Media & Voice Audio Ingestion        │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
 ┌───────────────────────────────┐     ┌─────────────────────────────────┐
 │   Farmer Long-Term Memory     │     │  Hybrid RAG Knowledge Engine    │
 │   - Soil, Farm, & Crop State  │────▶│  - ICAR / PJTSAU / ANGRAU Docs  │
 │   - Past Disease/Spray Logs   │     │  - Vector (768-d) + Keyword BM25│
 └───────────────────────────────┘     └────────────────┬────────────────┘
                                                        │
                                                        ▼
                               ┌─────────────────────────────────────────┐
                               │  AI Decision Engine (Google Gemini)     │
                               │  - 8-Step Agronomic Reasoning Engine    │
                               │  - Zero-Hallucination Safety Guardrails │
                               │  - Vision Multimodal Image Diagnosis    │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
 ┌───────────────────────────────┐     ┌─────────────────────────────────┐
 │  Enrichment & Commerce Engine │◀────│      Response Assembly Pipeline │
 │  - Mandi Prices & Weather     │     │  - Language Match Formatting    │
 │  - Shops, Schemes, Escalation │     │  - Multi-Turn Tool Enrichment   │
 └───────────────────────────────┘     └────────────────┬────────────────┘
                                                        │
                                                        ▼
                               ┌─────────────────────────────────────────┐
                               │     Outbound WhatsApp / Web Response    │
                               └─────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.12.10, FastAPI, Uvicorn, Pydantic v2 |
| **Database & ORM** | PostgreSQL (Production) / SQLite (Testing), SQLAlchemy 2.0 (Asyncpg / Aiosqlite), Alembic |
| **Caching & In-Memory** | Redis (Asyncio) |
| **AI / Large Language Models** | Google Gemini 3.6 Flash (`google-genai` SDK), Multimodal Vision |
| **Speech Processing (STT / TTS)** | Google Cloud Speech-to-Text v2, Sarvam AI, Google Cloud Text-to-Speech |
| **Push Notifications** | Firebase Cloud Messaging (FCM) High-Priority Stock Siren Alerts |
| **RAG & Search** | Custom Hybrid Vector Engine (768-dim embeddings) + Keyword Frequency Matching + Metadata Scoring |
| **PDF Extraction** | PyPDF 5.0+ with binary/compressed stream sanitization |
| **Messaging & Gateway** | Meta WhatsApp Cloud API (Graph API v20.0), Webhooks, HMAC-SHA256 Verification |
| **Payments** | Razorpay Payment Gateway (Cryptographic HMAC Verification) |
| **Web Frontend** | HTML5, Vanilla JavaScript, Responsive CSS3, Web Speech API |
| **Deployment** | Render (`render.yaml`), Docker-ready |

---

## 📂 Project Structure

```
BhoomiMitraAI/
├── docs/                      # Architectural and technical documentation
│   ├── ai_decision_engine.md
│   ├── api_architecture.md
│   ├── backend_architecture.md
│   ├── conversation_design.md
│   ├── database_design.md
│   ├── krishimitra_architecture.md
│   ├── mvp_definition.md
│   ├── security_architecture.md
│   ├── technical_implementation_plan.md
│   ├── vision_and_roadmap.md
│   └── voice_and_stock_siren.md
├── migrations/                # Alembic database migration scripts
│   └── versions/
├── scripts/                   # Utility & verification scripts
│   ├── rag_inspector.py       # RAG database inspection and diagnostic tool
│   ├── reindex_rag.py         # Knowledge document indexing and vector ingestion
│   └── seed_shops_data.py     # Retailer and inventory seeding
├── src/                       # Application source code
│   ├── advisory/              # Crop advisories and farmer guidance
│   ├── ai/                    # Gemini client, system prompts, AI decision engine
│   ├── auth/                  # JWT authentication, password hashing & RBAC
│   ├── common/                # Shared utilities and helpers
│   ├── conversation/          # Multi-turn interaction history & state
│   ├── core/                  # Database connections, base models, exception handlers
│   ├── crop_health/           # Disease diagnosis records and symptoms
│   ├── crops/                 # Crop lifecycle tracking per farm
│   ├── dashboard/             # Web application routes and legal page handlers
│   ├── escalation/            # Agricultural expert escalation & ticket workflows
│   ├── farmer_profiles/       # Farmer demographic and location state
│   ├── farmers/               # Farmer core identity and authentication
│   ├── farms/                 # Farm parcel management and soil parameters
│   ├── gateway/               # WhatsApp webhook, HMAC security, message processor
│   ├── inventory/             # Agri-retail product catalogs and stock levels
│   ├── language/              # Language detection heuristics & speech interfaces
│   ├── market/                # Live APMC mandi commodity prices & sync
│   ├── memory/                # Long-term dynamic farmer memory profiles
│   ├── orders/                # Cart & input purchase order requests
│   ├── payments/              # Razorpay checkout and webhook verification
│   ├── rag/                   # Document parsing, chunking, embeddings, hybrid search
│   ├── schemes/               # Government subsidies and eligibility engine
│   ├── shops/                 # Input retailer directory and Haversine geo-search
│   ├── weather/               # OpenWeatherMap forecast & rain alert service
│   ├── config.py              # Central Pydantic environment configuration
│   └── main.py                # FastAPI application factory & router registration
├── static/                    # Frontend assets & verified RAG knowledge files
│   ├── rag_docs/              # Clean agronomic guides (ICAR, PJTSAU, ANGRAU)
│   ├── app.js                 # Web dashboard interaction logic
│   ├── index.html             # Web dashboard landing page
│   ├── styles.css             # Unified application styles
│   └── voice.js               # Web Speech voice interface
├── templates/                 # Server-rendered HTML templates & legal compliance
│   ├── dashboard.html         # Farmer & retailer web portal
│   ├── data-deletion.html     # Facebook/Meta compliance data deletion page
│   ├── privacy-policy.html    # Privacy policy compliance page
│   └── terms.html             # Terms of service page
├── tests/                     # Comprehensive automated pytest suite (281 tests)
├── alembic.ini                # Alembic configuration
├── render.yaml                # Render cloud deployment configuration
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project overview and documentation
```

---

## 🗄️ Database Models & Tables

BhoomiMitra AI implements **19 production SQLAlchemy models**:

| Model Name | Table Name | Purpose |
| :--- | :--- | :--- |
| `Farmer` | `farmers` | Core farmer account, phone number, language preference |
| `FarmerProfile` | `farmer_profiles` | Name, state, district, current crop, land size |
| `Conversation` | `conversations` | Conversation logs, idempotency message IDs, intent tracking |
| `Farm` | `farms` | Farm parcels, soil type, irrigation type, GPS coordinates |
| `Crop` | `crops` | Crops planted per farm, variety, sowing dates, status |
| `CropHealth` | `crop_health` | Diagnostic records, symptoms, visual disease findings |
| `Advisory` | `advisories` | Agricultural advisories and alerts |
| `Expert` | `experts` | Registered agricultural scientists, AEOs, and extension officers |
| `UserAccount` | `user_accounts` | Web dashboard authentication, hashed passwords, and RBAC roles |
| `Shop` | `shops` | Input dealers, coordinates, delivery radius, license info |
| `Inventory` | `inventory` | Retail products, pricing, stock levels, packaging units |
| `OrderRequest` | `order_requests` | Farmer purchase requests, delivery address, and payment status |
| `GovernmentScheme` | `government_schemes` | Central & state schemes, criteria, benefits, deadlines |
| `SchemeApplication` | `scheme_applications` | Farmer scheme application status tracking |
| `MarketPrice` | `market_prices` | APMC mandi prices, modal rates, arrivals, and price records |
| `FarmerMemory` | `farmer_memory` | Long-term memory profile, soil state, and `expert_consultation_history` JSON tickets |
| `KnowledgeDocument`| `knowledge_documents` | RAG documents (ICAR, PJTSAU, university packages) |
| `KnowledgeChunk` | `knowledge_chunks` | Segmented clean text chunks for RAG indexing |
| `EmbeddingMetadata`| `embedding_metadata` | 768-dimension vector embeddings and metadata |

> **Note on Escalation Tickets**: Human expert escalation consultation tickets and audit history are persisted directly within `FarmerMemory.expert_consultation_history` JSON for seamless context retention across multi-turn exchanges.

---

## 🌐 Implemented API Modules & Endpoints

- **System**: `GET /health`
- **Authentication & RBAC (`/auth`)**: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `POST /auth/refresh`, `POST /auth/logout`
- **WhatsApp Webhook (`/webhook`)**: `GET /webhook/whatsapp`, `POST /webhook/whatsapp`, `GET /webhook/whatsapp/health`
- **AI Core (`/ai`)**: `POST /ai/generate`, `GET /ai/health`
- **RAG Knowledge Engine (`/rag`)**: `POST /rag/upload`, `POST /rag/query`, `POST /rag/rebuild`, `GET /rag/search`, `GET /rag/documents`, `DELETE /rag/document/{document_id}`
- **Farmer Memory (`/memory`)**: `GET /memory/{farmer_id}`, `PUT /memory/{farmer_id}`, `GET /memory/summary/{farmer_id}`, `GET /memory/voice/{farmer_id}`, `POST /memory/refresh`
- **Farmers (`/farmers`)**: Full CRUD (`GET`, `POST`, `PUT`, `DELETE`)
- **Farmer Profiles (`/farmer-profiles`)**: Full CRUD (`GET`, `POST`, `PUT`, `DELETE`)
- **Farms & Crops (`/farms`, `/crops`)**: Full CRUD on farm parcels and crop stages
- **Crop Health (`/crop-health`)**: Disease records, symptoms, and diagnoses
- **Advisories (`/advisories`)**: Agronomic advisories and broadcast alerts
- **Conversations (`/conversations`)**: Message history and logging
- **Agri Shops (`/shops`)**: CRUD, Haversine geo-search (`/shops/nearby`), product search (`/shops/search-product`)
- **Inventory Management (`/inventory`)**: CRUD, stock metrics (`/inventory/dashboard`), low-stock alerts (`/inventory/low-stock`), stock adjustments (`PATCH /inventory/{id}/stock`)
- **Order Requests (`/orders`)**: Ordering, farmer history (`/orders/farmer/{id}`), shop view (`/orders/shop/{id}`), status management
- **Payments & Razorpay (`/payments`)**: Order creation (`POST /payments/create-order`), cryptographic verification (`POST /payments/verify`), webhook handler (`POST /payments/webhook`)
- **Government Schemes (`/schemes`)**: Scheme listings, eligibility checks (`POST /schemes/evaluate`), application workflow (`POST /schemes/apply`)
- **Market Prices (`/market`)**: Commodity rates (`GET /market/prices`), commodities list (`GET /market/commodities`), APMC mandis (`GET /market/markets`), manual sync (`POST /market/refresh`)
- **Weather Forecast (`/weather`)**: 5-day / 3-hour forecasts (`GET /weather/forecast`)
- **Expert Escalation (`/escalation`)**: Active officers (`GET /escalation/experts`), ticket queue (`GET /escalation/tickets`), ticket status update (`PATCH /escalation/tickets/{id}/status`), farmer history (`GET /escalation/farmer/{id}/tickets`)
- **Web Dashboard & Legal**: `/`, `/dashboard`, `/farmer`, `/shop`, `/privacy-policy`, `/terms`, `/data-deletion`

---

## 🧠 AI & RAG Components

### 8-Step Agronomic Reasoning Engine
Every farmer message passes through a structured reasoning process:
1. **Intent Understanding**: Disease diagnosis, pest attack, fertilizer query, weather, schemes, market prices, or shop discovery.
2. **Context Memory Retrieval**: Ingests farmer land size, district, soil, and crop history.
3. **RAG Grounded Retrieval**: Matches verified ICAR/PJTSAU documents via hybrid vector + keyword search.
4. **Weather Integration**: Hyperlocal weather conditions and rain alerts.
5. **Image Diagnosis Integration**: Synthesizes vision findings with agronomic rules.
6. **Plain Language Formulation**: Removes unnecessary scientific jargon.
7. **Structured 5-Part Response**: Main Answer, Agronomic Reasoning, Actionable Steps, Verified Sources, and Confidence Rating.
8. **Clarification Protocol**: Requests missing details rather than guessing.

### Strict Pesticide Safety & Fallback Rule
- If an exact pesticide or chemical dosage is **not present** in the retrieved verified documents, the system **never invents** values.
- It explicitly indicates uncertainty and recommends consulting a local Agricultural Extension Officer (AEO) or Krishi Vigyan Kendra (KVK).

---

## 📲 WhatsApp Gateway & Webhook Security

- **Meta Cloud API (v20.0)**: Webhook endpoint receives incoming JSON payloads from WhatsApp.
- **HMAC-SHA256 Signature Verification**: Inbound requests are validated against `WHATSAPP_APP_SECRET` using `X-Hub-Signature-256`.
- **Media Pipeline**: Ingests voice notes and camera photos via Meta media download endpoints.
- **Idempotency**: `message_id` deduplication prevents reprocessing duplicate webhook deliveries.
---

## 🎙️ WhatsApp Voice Pipeline

BhoomiMitra AI offers an end-to-end voice interface for smallholder farmers:

```
WhatsApp voice note (.ogg/.opus)
       |
       v
Speech-to-Text (STT via Google Cloud / Sarvam AI / mock fallback)
       |
       v
Language Detection & Preference (13 Indian languages + Tanglish / code-switching)
       |
       v
Intent Classification (pure stock query, advisory, pest diagnosis, market price)
       |
       v
RAG / Farmer Context Retrieval (ICAR/PJTSAU Package of Practices + farm memory)
       |
       v
Gemini 3.6 Flash Agronomic Reasoning (Zero-hallucination dosage guardrails)
       |
       v
Final Grounded Answer Formulation (native script & phrasing)
       |
       +------------------------------+
       |                              |
       v                              v
Outbound WhatsApp Text        Text-to-Speech (TTS Audio Synthesis)
       |                              |
       |                              v
       +--------------------> Outbound WhatsApp Audio Message
```

---

## 🚨 Proactive Stock Siren

The **Stock Siren** alerts waiting farmers the instant critical agricultural inputs (Urea, DAP, seeds) are replenished at local registered dealers:

```
Dealer Inventory Transition (quantity updated > 0)
       |
       v
Trigger Stock Alert Event (scans active farmer alerts by product & district)
       |
       +------------------------------------------+
       |                                          |
       v                                          v
Phone-Level FCM Siren Alert            WhatsApp Chat Notification
(Audible alarm on Android devices)      (Direct WhatsApp restock message)
       |                                          |
       +--------------------+---------------------+
                            |
                            v
               Fail-Soft Isolated Delivery
```

### Channel Distinction: Phone-Level FCM Siren vs. WhatsApp Notifications

| Characteristic | Phone-Level FCM Siren | WhatsApp Stock Notification |
| :--- | :--- | :--- |
| **Channel** | Android Push Notification (`firebase-admin`) | Meta WhatsApp Cloud API (`Graph API v20.0`) |
| **Audio Alert** | **Audible siren tone** (`sound="stock_siren"`, `channel_id="stock_siren"`) | Standard incoming WhatsApp notification chime |
| **Priority** | Android High Priority (`priority="high"`, wake screen) | Background asynchronous message queue |
| **User Action** | Launches app directly to restocked dealer details (`OPEN_STOCK_SIREN`) | Direct chat buttons to place orders or call shop |
| **Data Payload** | Structured JSON payload (`product`, `qty`, `shop_id`, `district`) | Localized, human-readable message text |
| **Fault Isolation** | Soft-fails on missing/invalid tokens; never halts WhatsApp flow | Soft-fails independently; never blocks FCM siren |

---

## 🧪 Testing

BhoomiMitra AI includes a comprehensive regression and unit test suite covering end-to-end webhook flows, RAG extraction, dosage guardrails, shops, weather, market prices, escalation, language preservation, voice pipeline, and stock siren push alerts.

```powershell
# Run the complete test suite
pytest -v --tb=short
```

**Latest Test Results**:
```text
======================= 1085 passed, 772 warnings in 310.63s =======================
```

---

## ⚙️ Environment Setup & Local Installation

### Prerequisites
- Python 3.12.10
- PostgreSQL (or local SQLite for development)
- Google Gemini API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kavyasriakkathi/BhoomiMitraAI.git
   cd BhoomiMitraAI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and provide your credentials:
   ```ini
   APP_ENV=development
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/bhoomimitra
   REDIS_URL=redis://localhost:6379/0
   GOOGLE_GEMINI_API_KEY=your_gemini_api_key
   WHATSAPP_API_TOKEN=your_meta_whatsapp_token
   WHATSAPP_PHONE_NUMBER_ID=your_whatsapp_phone_number_id
   WHATSAPP_VERIFY_TOKEN=your_webhook_verify_token
   WHATSAPP_APP_SECRET=your_meta_app_secret
   DATA_GOV_API_KEY=your_data_gov_in_api_key
   OPENWEATHER_API_KEY=your_openweather_api_key
   JWT_SECRET_KEY=your_jwt_secret_key_min_32_chars
   ADMIN_REGISTRATION_KEY=your_admin_registration_passkey
   RAZORPAY_KEY_ID=your_razorpay_key_id
   RAZORPAY_KEY_SECRET=your_razorpay_key_secret
   ```

5. **Index the RAG Knowledge Base**:
   ```bash
   python scripts/reindex_rag.py
   ```

6. **Run the Development Server**:
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```
   Access the interactive API documentation at `http://localhost:8000/docs`.

---

## 🚀 Cloud Deployment (`render.yaml`)

The project includes configuration for deploying on Render:
- **Web Service**: FastAPI running under `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
- **Database**: Managed PostgreSQL instance with SSL support

---

## 🚧 Status of Modules

### ✅ Fully Implemented
- WhatsApp Gateway & HMAC Security
- AI Decision Engine & System Prompts (Google Gemini 3.6 Flash via `google-genai` SDK)
- WhatsApp Multilingual Voice Pipeline (STT, 13-language detection, Gemini reasoning, TTS audio reply)
- Proactive Stock Siren (Phone-level audible FCM push + direct WhatsApp notifications)
- Grounded RAG Pipeline & Reindexing
- Multimodal Crop Health Diagnosis
- Long-Term Farmer Memory & Voice State
- Hyperlocal Shop & Inventory Management
- APMC Mandi Live Market Prices & Commodity Normalization
- Hyperlocal Weather Forecasts & Rain Alerts
- Government Schemes Matching & Eligibility Engine
- Human Expert Escalation & Ticket Resolution Queue
- Razorpay Payment Gateway & Cryptographic Verification
- JWT Authentication & Role-Based Access Control (RBAC)
- Web Dashboard & Web Voice/Photo Scanner
- Meta Compliance Pages (Privacy Policy, Terms, Data Deletion)

### ⏳ Planned Features (Roadmap)
- **WhatsApp Interactive UI**: Native WhatsApp list pickers and quick reply buttons for scheme applications.
- **Automated Weather Push Alerts**: Scheduled cron notifications for pest risk alerts based on rainfall forecasts.

---

## ⚠️ Current Limitations

- Image diagnosis offers non-definitive indications and requires physical confirmation before critical chemical interventions.
- Chemical dosages are restricted to verified agricultural documents currently in the RAG index; queries for unindexed crops/chemicals will trigger safety fallbacks.

---

## 📄 License

Proprietary — All rights reserved. BhoomiMitra AI Team.
