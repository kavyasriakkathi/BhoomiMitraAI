# 🎙️ Voice Pipeline & 🚨 Proactive Stock Siren Architecture

## 1. Multilingual WhatsApp Voice Pipeline

BhoomiMitra AI provides a native voice interface enabling smallholder farmers to communicate in their preferred regional language or dialect without needing to type.

### End-to-End Execution Flow

```
+------------------------+
|  WhatsApp Voice Note   | (.ogg / .opus audio from Meta Graph API)
+-----------+------------+
            |
            v
+------------------------+
|  Speech-to-Text (STT)  | Google Cloud Speech-to-Text / Sarvam AI (fail-soft fallback)
+-----------+------------+
            |
            v
+------------------------+
|  Language & Dialect    | Multi-language detector (13 Indian languages + Tanglish/code-mixed)
|  Preference Matcher    | Resolves farmer profile preference or speech cues
+-----------+------------+
            |
            v
+------------------------+
|  Intent Classification | Pure stock query, crop advisory, pest diagnosis, market price, etc.
+-----------+------------+
            |
            v
+------------------------+
|  RAG & Farmer Context  | ICAR/PJTSAU Package of Practices + Land size, soil type, crop history
+-----------+------------+
            |
            v
+------------------------+
|  AI Reasoning (Gemini) | Google Gemini 3.6 Flash via `google-genai` SDK
|                        | Strictly zero-hallucination chemical safety & dosage guardrails
+-----------+------------+
            |
            v
+------------------------+
| Final Grounded Answer  | Formatted in native script (Telugu, Hindi, etc.)
+-----------+------------+
            |
      +-----+------------------------+
      |                              |
      v                              v
+------------------+       +------------------------+
|  WhatsApp Text   |       | Text-to-Speech (TTS)   |
|  Response        |       | Generates native audio |
+------------------+       +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | Outbound WhatsApp Audio|
                           +------------------------+
```

### Key Technical Attributes
1. **Audio Ingestion**: WhatsApp webhook ingests binary audio payload with MIME validation (`audio/ogg; codecs=opus`, `audio/aac`, `audio/mp4`).
2. **Dialect & Tanglish Support**: Detects Romanized Telugu ("mandulu kavali", "panta samasya") alongside 13 scheduled Indian languages.
3. **Fail-Soft Graceful Degradation**: If external STT/TTS services experience latency or failure, the pipeline falls back gracefully to text delivery with clear error context, never dropping the farmer's session.

---

## 2. Proactive Stock Siren & Alert Engine

The **Stock Siren** addresses acute seasonal shortages of critical agricultural inputs (e.g., Urea, DAP, micronutrients, certified seeds). When input dealers update their inventory in the database, the backend triggers stock alert events across available notification paths (direct WhatsApp messaging within active conversation windows, template messages for re-engagement, and phone-level FCM alerts).

### End-to-End Workflow

```
+---------------------------------+
|     Dealer Inventory Restock    | Inventory transitioned: quantity > 0
+----------------+----------------+
                 |
                 v
+---------------------------------+
|   Trigger Stock Siren Event     | Scans active farmer alerts matching product & district
+----------------+----------------+
                 |
        +--------+--------------------------+
        |                                   |
        v                                   v
+-------------------------------+   +-------------------------------+
|   Phone-Level FCM Siren       |   |   WhatsApp Notification       |
|   (Android App Push)          |   |   (Direct Chat Message)       |
+-------------------------------+   +-------------------------------+
| * High-priority push channel  |   | * Meta Cloud API message      |
| * Custom 'stock_siren' tone   |   | * Full shop contact & address |
| * Wakes device with alert     |   | * Verified stock quantity     |
| * Click: OPEN_STOCK_SIREN     |   | * Standardized template       |
+---------------+---------------+   +---------------+---------------+
                |                                   |
                +---------------+-------------------+
                                |
                                v
+-------------------------------------------------------------------+
|                    Fail-Soft Isolation Guard                      |
| * FCM and WhatsApp dispatch pathways are fully decoupled          |
| * Missing FCM tokens or Firebase downtime never halts WhatsApp    |
| * WhatsApp API downtime or rate limits never block FCM delivery   |
+-------------------------------------------------------------------+
```

### Clarifying Notification Channels: FCM Siren vs. WhatsApp

| Feature | Phone-Level FCM Siren | WhatsApp Stock Notification |
| :--- | :--- | :--- |
| **Delivery Medium** | Native Android Push Notification (`Firebase Admin SDK`) | Meta WhatsApp Cloud API (`Graph API v20.0`) |
| **Audio / Siren** | **Custom audible siren tone** (`sound="stock_siren"`, `channel_id="stock_siren"`) | Default WhatsApp incoming notification tone |
| **Priority** | Android `priority="high"` with wake lock intent | Asynchronous WhatsApp message delivery, subject to Meta messaging policies and template approval |
| **Actionable Intent** | Launches mobile app directly into stock alert screen (`OPEN_STOCK_SIREN`) | Interactive chat options to order or call the shop dealer directly |
| **Payload Structure** | Machine-readable JSON data payload (product, district, shop_id, qty, brand) | Native-language human-readable text notification formatted with details |
| **Resilience** | Soft-fails with token invalidation cleanup (`deactivate_tokens_batch`) | Soft-fails with retry logging, isolated from FCM lifecycle |
