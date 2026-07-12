# 🌾 KrishiMitra AI

**AI-powered WhatsApp Farming Assistant** — From crop planning to selling harvest.

## Vision

KrishiMitra AI helps Indian farmers through their entire farming journey via **WhatsApp only** — no apps, no websites. Farmers interact using **voice messages, text, and crop photos** in their local language.

## Supported Languages

- Telugu (తెలుగు)
- Hindi (हिन्दी)
- Kannada (ಕನ್ನಡ)
- Tamil (தமிழ்)
- English

## Architecture

```
Farmer (WhatsApp)
    │
    ▼
┌──────────────────┐
│  WhatsApp Gateway │  ← Meta Cloud API / Twilio
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Language Engine  │  ← STT, Translation, TTS
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│ Conversation Mgr  │  ← Session state, context memory
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│    AI Brain       │  ← LLM + confidence gating
└────────┬─────────┘
         │
    ▼
┌──────────────────────────────────────┐
│  Domain Services                      │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ │
│  │  Crop    │ │  Market  │ │ Expert │ │
│  │ Advisory │ │ Connect  │ │ Escal. │ │
│  └─────────┘ └──────────┘ └────────┘ │
└──────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+ / FastAPI |
| Database | PostgreSQL + Redis |
| AI/LLM | OpenAI GPT-4 / Google Gemini |
| Speech | Google Cloud STT / Whisper |
| Translation | Google Translate / IndicTrans2 |
| WhatsApp | Meta Cloud API |
| Infra | Docker, CI/CD |

## Project Structure

```
krishimitra-ai/
├── src/
│   ├── gateway/        # WhatsApp webhook & message routing
│   ├── language/        # STT, TTS, translation services
│   ├── conversation/    # Session management & state machine
│   ├── ai/              # LLM integration & confidence gating
│   ├── advisory/        # Crop planning, pest detection
│   ├── market/          # Mandi prices, buyer matching
│   ├── escalation/      # Expert routing & handoff
│   ├── farmers/         # Farmer profiles & history
│   └── common/          # Shared utilities, config, logging
├── tests/               # Unit & integration tests
├── docs/                # Documentation
├── scripts/             # Dev & deployment scripts
└── docker/              # Docker configs
```

## Design Principles

1. **Farmer-First** — Voice-in, voice-out. No typing required.
2. **Never Guess** — If AI confidence is low, ask follow-up questions or escalate to an expert.
3. **Local Language** — Every interaction in the farmer's preferred language.
4. **Modular** — Each service is independent and testable.
5. **Auditable** — Every conversation logged for quality review.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/krishimitra-ai.git
cd krishimitra-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment config
cp .env.example .env

# Run the server
uvicorn src.main:app --reload
```

## License

Proprietary — All rights reserved.
