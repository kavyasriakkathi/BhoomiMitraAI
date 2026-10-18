# 🌾 BhoomiMitra AI — MVP Demo & Operations Guide

## 1. System Overview & Architecture

BhoomiMitra AI is an AI-powered WhatsApp advisory system built for Indian farmers. It operates with strict ground-truth verification, multi-language support (Telugu, Hindi, English), and robust automated background data pipelines.

```
                   +----------------------------------+
                   |   Farmer via WhatsApp (Meta API) |
                   +-----------------+----------------+
                                     |
                                     v
                   +----------------------------------+
                   |  FastAPI Gateway (Render / Cloud)|
                   |  - In-flight Dedup Lock (UUID)   |
                   |  - 131056 Rate-Limit Suppression |
                   +-----------------+----------------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
    +-------------------+  +-------------------+  +-------------------+
    | Agmarknet Mandi   |  | RAG Engine        |  | Weather & Shops   |
    | Price Engine      |  | - Hybrid Search   |  | - OpenWeatherMap  |
    | - 767+ Live Rows  |  | - Dosage Safety   |  | - Google Places   |
    | - Auto sync: 7/13 |  | - Async Vector C. |  | - Stock Siren     |
    +-------------------+  +-------------------+  +-------------------+
              |                      |                      |
              +----------------------+----------------------+
                                     |
                                     v
                   +----------------------------------+
                   |  Synthesized Farmer Response     |
                   |  (Strict Language + Safety Gate) |
                   +----------------------------------+
```

---

## 2. Live Demo Script (Step-by-Step for WhatsApp)

When demonstrating the platform to stakeholders, investors, or pilot farmers, follow this curated 4-step sequence:

### Step 1: Telugu Mandi Price Lookup
- **Farmer Message:**
  > `మహబూబ్‌నగర్‌లో టమాటా ధర ఎంత?`
- **What BhoomiMitra Does:**
  1. Identifies Telugu intent for market prices.
  2. Normalizes `మహబూబ్‌నగర్` -> `Mahabubnagar` and `టమాటా` -> `Tomato`.
  3. Queries the live Agmarknet database (767 synced records).
  4. Returns the official mandi price (e.g. Mahabubnagar Rythu Bazar, ₹3,000/quintal) with modal, min, max, and arrival date in natural Telugu.

### Step 2: Multi-Language Cross-Verification (Hindi)
- **Farmer Message:**
  > `महबूबनगर में आज टमाटर का भाव कितना है?`
- **What BhoomiMitra Does:**
  1. Identifies Hindi intent.
  2. Retrieves the exact same canonical price.
  3. Formats entirely in Devanagari Hindi without any residual Telugu text or mixed characters.

### Step 3: Crop Diagnosis & Pesticide Safety Gate
- **Farmer Message (Turn 1):**
  > `పత్తి ఆకులు పసుపుగా మారి గోధుమ రంగు మచ్చలు ఉన్నాయి` *(Cotton leaves turning yellow with brown spots)*
- **What BhoomiMitra Does:**
  1. Recognizes symptoms of Alternaria leaf spot (*ఆల్టర్నేరియా ఆకుమచ్చ తెగులు*).
  2. **SAFETY GATE ACTIVATES**: Because diagnosis is preliminary, the system **strictly suppresses chemical dosages**.
  3. Politely asks the farmer for a photo or symptom confirmation.
- **Farmer Message (Turn 2):**
  > `అవును, మచ్చలు ఉన్నాయి కానీ ఫోటో ఇవ్వడం కుదరదు.` *(Yes, spots are present but cannot take photo)*
- **What BhoomiMitra Does:**
  1. Confirms Alternaria diagnosis from trusted baseline agronomy knowledge.
  2. Only then provides the ICAR/approved dosage: *Mancozeb 75% WP (2.5 - 3.0 g per liter of water)* with safe application guidance.

### Step 4: Weather & Voice Support
- Send a short 5-second Telugu voice note:
  > `ఈరోజు వర్షం పడుతుందా?` *(Will it rain today?)*
- **What BhoomiMitra Does:**
  1. Google Cloud STT transcribes Telugu voice audio.
  2. Fetches OpenWeather forecast for the farmer's registered GPS/district.
  3. Responds back with rain probability, temperature, and farming recommendations.

---

## 3. Background Ingestion & Automation Operations

### Mandi Sync Automation (macOS `launchd`)
- The scheduler runs twice daily at **07:00 IST** and **13:00 IST**.
- Uses an Indian resident connection to bypass Agmarknet geoblocks on foreign IPs.
- **Verify status:**
  ```bash
  launchctl list | grep bhoomimitra
  ```
- **Manual on-demand sync:**
  ```bash
  python3 scripts/sync_mandi_prices.py
  ```

---

## 4. Key Security & Operational Checklist

| Item | Status | Action Required |
|---|---|---|
| **Agmarknet Mandi Sync** | Active | Scheduled via `launchd` (07:00 & 13:00 IST) |
| **Meta 131056 Rate Limit Guard** | Active | Automatic 60s cooldown suppression |
| **In-Flight Lock** | Active | In-memory token lock prevents duplicate DB entries |
| **RAG Vector Cache** | Active | Static baseline embeddings cached; non-blocking async |
| **Full E2E Test Suite** | 100% Passed | **1,412 / 1,412 automated tests green** |
| **Admin Dashboard Seeding** | Complete | Login with `admin@bhoomimitra.ai` |
| **API Key Rotation** | Recommended | Rotate Data.gov.in key on portal when pilot expands |
