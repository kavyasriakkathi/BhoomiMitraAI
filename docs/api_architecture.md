# 🚀 BhoomiMitra AI — Complete API Architecture

> **Document Type:** Backend API Architecture & Technical Specification  
> **Audience:** Backend Engineers, DevOps, Frontend/Mobile Devs, PMs  
> **Objective:** Define a production-ready, highly scalable REST and Event-Driven API architecture for BhoomiMitra AI.

---

## 1. System-Wide Architecture Principles

### API Gateway & Load Balancing
- **Tech:** AWS API Gateway / Cloudflare + Nginx.
- **Purpose:** Acts as the single entry point. Handles SSL termination, WAF (Web Application Firewall), DDoS protection, and global rate limiting.
- **Load Balancing:** Round-robin across stateless FastAPI application servers (Auto-Scaling Groups).

### API Versioning
- **Strategy:** URI Versioning (`/api/v1/...`). Major breaking changes bump the version (e.g., `v2`). Non-breaking changes are handled continuously.

### Authentication & Authorization
- **Farmers (WhatsApp):** Implicit auth via Meta's WhatsApp Webhook. The system authenticates Meta's webhook via HMAC-SHA256 signature verification.
- **Web Dashboards (Experts, Admins, Shops):** OAuth 2.0 with JWT (JSON Web Tokens).
- **Authorization:** Role-Based Access Control (RBAC). Roles: `admin`, `expert`, `shop_owner`.

### Caching Strategy
- **Tech:** Redis.
- **Strategy:** 
  - *Short TTL (5-15 mins):* Real-time weather, nearby shop stock.
  - *Medium TTL (24 hours):* Market prices, government schemes.
  - *Permanent Cache:* Farmer profiles (invalidated on update).

### Event-Driven Architecture (Async)
- **Tech:** RabbitMQ / Apache Kafka.
- **Use Case:** Webhooks, AI processing (STT, TTS, Image Gen), and bulk Notifications (Emergency Alerts). Synchronous API calls are reserved for CRUD; Heavy AI tasks are enqueued.

### Monitoring & Logging
- **Logging:** Structured JSON logs via ELK stack (Elasticsearch, Logstash, Kibana) / Datadog.
- **Metrics:** Prometheus + Grafana (API latency, 5xx rates, Queue depths).
- **Tracing:** OpenTelemetry for tracing an API call through the LLM and back.

---

## 2. Global Request / Response Standards

**Base URL:** `https://api.bhoomimitra.ai/api/v1`

**Success Response (200 OK):**
```json
{
  "success": true,
  "data": { ... },
  "meta": { "timestamp": "2026-07-12T00:00:00Z" }
}
```

**Error Response (4xx / 5xx):**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Invalid crop age provided.",
    "details": { "crop_age": "Must be an integer." }
  }
}
```

---

## 3. Core API Specifications

*(Note: For brevity without losing detail, APIs are grouped into functional domains. All follow the global security and rate-limiting standards unless specified otherwise).*

### A. WhatsApp Webhook & Media Processing (Async / Event-Driven)

#### 1. WhatsApp Message Webhook (Receive)
- **Endpoint:** `POST /webhooks/whatsapp`
- **Request Body:** Standard Meta WhatsApp JSON payload.
- **Response:** `200 OK` (Ack immediately, process via RabbitMQ).
- **Auth:** HMAC-SHA256 signature (`X-Hub-Signature`).
- **Rate Limiting:** IP Whitelisted (Meta IPs only).
- **Security:** Replay attack prevention via Message IDs.

#### 2. Process Voice Message (Internal)
- **Endpoint:** `POST /internal/media/voice/process`
- **Request:** `{ "media_url": "...", "farmer_id": "uuid" }`
- **Response:** `{ "transcript": "...", "language": "te" }`

#### 3. Image Analysis (Internal)
- **Endpoint:** `POST /internal/media/image/analyze`
- **Request:** `{ "media_url": "...", "type": "disease|pest|soil" }`
- **Response:** `{ "classification": "Pink Bollworm", "confidence": 0.92 }`

---

### B. Farmer Management

#### 4. Register/Update Farmer
- **Endpoint:** `PUT /farmers/{phone_number}`
- **Auth:** Webhook / JWT (Admin).
- **Request:** `{ "language": "te", "district": "Warangal", "location": {"lat": 18.0, "lng": 79.5} }`
- **Validation:** Valid Indian phone number (regex).
- **Rate Limiting:** 5/min per IP.

#### 5. Farmer Profile Details
- **Endpoint:** `GET /farmers/{farmer_id}/profile`
- **Response:** Complete JSON profile (Crops, land size, soil type).

---

### C. AI Engine & Recommendations

#### 6. Chat Recommendation Engine (The Core AI Router)
- **Endpoint:** `POST /ai/chat`
- **Auth:** Internal (Triggered by queue worker).
- **Request:** `{ "farmer_id": "uuid", "message_text": "...", "intent": "fertilizer_req" }`
- **Response:** `{ "response_text": "...", "confidence": 0.88, "needs_expert": false }`
- **Validation:** Context payload must exist in Redis.

#### 7. Crop Recommendation API
- **Endpoint:** `GET /recommendations/crops`
- **Query Params:** `?district=Warangal&soil_type=Red&season=Kharif`
- **Response:** List of suitable crops with expected ROI.
- **Cache:** 24 Hours.

#### 8. Disease Detection & 9. Pest Detection APIs
- **Endpoint:** `POST /recommendations/diagnostics`
- **Request:** `{ "crop_id": "uuid", "image_url": "...", "symptoms_text": "..." }`
- **Response:** `{ "diagnosis": "Stem Borer", "treatments": [...] }`

#### 10. Fertilizer & 11. Pesticide Recommendation APIs
- **Endpoint:** `POST /recommendations/treatment`
- **Request:** `{ "farmer_crop_id": "uuid", "issue_type": "deficiency|pest", "target": "Nitrogen" }`
- **Response:** Chemical composition + local brand names + dosage calculation.
- **Security:** Strict cross-check with banned chemicals database.

---

### D. Commerce & Location APIs

#### 12. Urea Availability & 13. Agro Shop Search
- **Endpoint:** `GET /shops/nearby`
- **Query Params:** `?lat=18.0&lng=79.5&radius_km=15&product=Urea`
- **Response:** Array of shops, sorted by distance.
- **Cache:** 15 Minutes (Redis GEO query).

#### 14. Shop Inventory Update (For Shop Owners)
- **Endpoint:** `PUT /shops/{shop_id}/inventory/{product_id}`
- **Auth:** JWT (Role: `shop_owner`).
- **Request:** `{ "stock_quantity": 50, "price": 266.00 }`

---

### E. External Context Data

#### 15. Weather Forecast API
- **Endpoint:** `GET /weather/forecast`
- **Query Params:** `?district=Warangal`
- **Response:** 7-day forecast, severe weather flags.
- **Cache:** 1 Hour.

#### 16. Market Price API
- **Endpoint:** `GET /market-prices`
- **Query Params:** `?crop=Tomato&district=Warangal`
- **Response:** Latest Agmarknet data, modal price, min/max.

#### 17. Government Schemes & 18. Scheme Eligibility
- **Endpoint:** `POST /schemes/eligibility`
- **Request:** `{ "farmer_id": "uuid" }`
- **Response:** List of schemes farmer is mathematically eligible for.

---

### F. Alerts & Notifications

#### 19. Emergency Alert Dispatch (Heavy Rain, Pest Outbreak)
- **Endpoint:** `POST /notifications/emergency`
- **Auth:** JWT (Role: `admin`).
- **Request:** `{ "event_type": "flood", "affected_districts": ["Warangal"], "message_template_id": "T123" }`
- **Processing:** Enqueues thousands of jobs in RabbitMQ for staggered WhatsApp delivery.
- **Rate Limit:** Admins only.

#### 20. Reminder APIs (Harvest, Irrigation)
- **Endpoint:** `POST /notifications/reminders/schedule`
- **Request:** `{ "farmer_id": "uuid", "trigger_date": "2026-08-15", "type": "fertilizer_dose_2" }`

---

### G. Human Expertise & Support

#### 21. Expert Ticket Generation
- **Endpoint:** `POST /experts/tickets`
- **Auth:** Internal (Triggered by AI Low Confidence).
- **Request:** `{ "farmer_id": "uuid", "context": "...", "images": [...] }`

#### 22. Expert Reply
- **Endpoint:** `POST /experts/tickets/{ticket_id}/reply`
- **Auth:** JWT (Role: `expert`).
- **Request:** `{ "reply_text": "...", "internal_notes": "..." }`
- **Action:** Triggers translation -> TTS -> WhatsApp Webhook delivery to farmer.

#### 23. Nearby Agriculture Officer API
- **Endpoint:** `GET /officers/nearby`
- **Query Params:** `?lat=18.0&lng=79.5`
- **Response:** Government extension officer details for physical visits.

---

### H. Analytics, Feedback & Soil

#### 24. Feedback API
- **Endpoint:** `POST /feedback`
- **Request:** `{ "conversation_id": "uuid", "rating": 1, "comment": "..." }`
- **Action:** Updates AI Model weights (RLHF data pipeline).

#### 25. Soil Test Report Upload & Parsing
- **Endpoint:** `POST /soil-tests/upload`
- **Auth:** JWT (Role: `expert` or `admin`).
- **Request:** PDF/Image of soil health card.
- **Action:** OCR parsing, stores N-P-K, pH values to Farmer Profile.

#### 26. Admin Analytics APIs
- **Endpoint:** `GET /admin/analytics/dashboard`
- **Auth:** JWT (Role: `admin`).
- **Response:** Aggregated data on queries, escalation rates, active users.
- **Cache:** Refreshed every 15 minutes.

---

## 4. API Security & Reliability Rules

1. **Strict Input Validation:** All endpoints use Pydantic (FastAPI) for strict schema validation. Invalid types drop the request instantly.
2. **Idempotency:** All `POST` requests (like creating a ticket or sending a notification) require an `Idempotency-Key` header to prevent double-charging or double-sending due to network retries.
3. **Database Connection Pooling:** PgBouncer is used to prevent connection starvation during massive traffic spikes (e.g., during a widespread weather alert).
4. **Graceful Degradation:** If the LLM provider (e.g., OpenAI/Gemini) goes down, the API falls back to a deterministic rules engine (e.g., cached weather, basic FAQs) while pausing complex recommendations.
