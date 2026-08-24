# 🚀 BhoomiMitra AI — Complete API Architecture

> **Document Type:** Backend API Architecture & Technical Specification  
> **Audience:** Backend Engineers, DevOps, Frontend/Mobile Devs, PMs  
> **Objective:** Define a production-ready, highly scalable REST and asynchronous API architecture for BhoomiMitra AI.

---

## 1. System-Wide Architecture Principles

### API Gateway & Entry Point
- **Tech:** FastAPI ASGI application behind Uvicorn / Reverse Proxy (Nginx / Cloudflare / Render).
- **Purpose:** Acts as the entry point, terminating SSL, enforcing CORS, handling HMAC-SHA256 signature verification, and routing requests to modular endpoints.

### API Versioning & Routing Structure
- **Strategy:** Dedicated domain-prefixed routers registered on the central application instance (e.g. `/auth`, `/webhook`, `/ai`, `/rag`, `/market`, `/weather`, `/schemes`, `/shops`, `/inventory`, `/orders`, `/payments`, `/escalation`, `/memory`).

### Authentication & Authorization
- **Farmers (WhatsApp):** Implicit authentication via Meta WhatsApp Webhook with cryptographic HMAC-SHA256 signature verification (`X-Hub-Signature-256`) against `WHATSAPP_APP_SECRET`.
- **Web Dashboards (Experts, Admins, Shops):** JWT (JSON Web Tokens) with HTTP-only secure cookie support and bearer authorization headers.
- **Authorization:** Role-Based Access Control (RBAC) supporting `admin`, `expert`, and `shop_owner` roles via dependency injection (`require_admin`, `require_expert`).

### Caching Strategy
- **Tech:** Redis (Asyncio).
- **Strategy:** 
  - *Short TTL (30 mins):* Real-time weather forecasts (`weather:loc:...`).
  - *Medium TTL (6 hours):* Mandi market prices (`market:prices:...`).
  - *Dynamic Invalidation:* Farmer memory profiles and inventory stock counts.

### Asynchronous Processing Architecture
- **Tech:** Python `async`/`await` coroutines with SQLAlchemy 2.0 `asyncpg`/`aiosqlite` connection pooling and background task lifespans.
- **Use Case:** Non-blocking multi-turn AI response generation, simultaneous tool enrichments (Mandi, Weather, Schemes, Shops, Escalation), and automatic memory extraction.

---

## 2. Global Request / Response Standards

**Base URL:** `https://api.bhoomimitra.in` (Production) / `http://localhost:8000` (Local)

**Success Response (200 OK):**
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation successful"
}
```

**Error Response (4xx / 5xx):**
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid location or district provided.",
    "details": {}
  }
}
```

---

## 3. Implemented API Specifications

### A. Authentication & RBAC (`/auth`)

#### 1. Register User Account
- **Endpoint:** `POST /auth/register`
- **Request Body:** `{ "email": "officer@bhoomimitra.in", "password": "...", "role": "expert|shop_owner|admin", "admin_key": "..." }`
- **Response:** `201 Created` with user ID, email, and role.

#### 2. User Login
- **Endpoint:** `POST /auth/login`
- **Request Body:** `OAuth2PasswordRequestForm` (username/email + password)
- **Response:** JWT access token + Set-Cookie HTTP-only session.

#### 3. Current User Profile
- **Endpoint:** `GET /auth/me`
- **Auth:** JWT / Cookie (`require_user`).

#### 4. Token Refresh & Logout
- **Endpoints:** `POST /auth/refresh`, `POST /auth/logout`

---

### B. WhatsApp Webhook Gateway (`/webhook`)

#### 5. WhatsApp Webhook Verification
- **Endpoint:** `GET /webhook/whatsapp`
- **Query Params:** `hub.mode`, `hub.verify_token`, `hub.challenge`
- **Auth:** Validates against `WHATSAPP_VERIFY_TOKEN`.

#### 6. Inbound WhatsApp Message Receiver
- **Endpoint:** `POST /webhook/whatsapp`
- **Header:** `X-Hub-Signature-256` (HMAC-SHA256 verification against `WHATSAPP_APP_SECRET`).
- **Processing:** Async message deduplication via `message_id`, media/voice downloading, multi-turn AI decision pipeline with tool enrichment.

#### 7. Webhook Health
- **Endpoint:** `GET /webhook/whatsapp/health`

---

### C. AI Decision Engine & Grounded RAG (`/ai`, `/rag`)

#### 8. AI Message Generation
- **Endpoint:** `POST /ai/generate`
- **Request Body:** `{ "farmer_id": "uuid", "conversation_id": "uuid", "message": "..." }`
- **Response:** Grounded agronomic recommendation with confidence score.

#### 9. RAG Knowledge Documents & Search
- **Endpoints:**
  - `POST /rag/upload` (Upload and chunk ICAR/PJTSAU agricultural PDF/text guides)
  - `POST /rag/query` (Hybrid vector + keyword search over indexed knowledge chunks)
  - `POST /rag/rebuild` (Rebuild embeddings and metadata index)
  - `GET /rag/search` (Search verified knowledge base)
  - `GET /rag/documents` (List indexed agronomic source documents)
  - `DELETE /rag/document/{document_id}` (Delete document and associated chunks)

---

### D. Farmer Memory & Identity (`/memory`, `/farmers`, `/farmer-profiles`)

#### 10. Long-Term Dynamic Memory
- **Endpoints:**
  - `GET /memory/{farmer_id}` (Fetch structured memory: soil, farm size, crop cycles, preferences)
  - `PUT /memory/{farmer_id}` (Update memory profile)
  - `GET /memory/summary/{farmer_id}` (Formatted memory prompt for AI reasoning)
  - `GET /memory/voice/{farmer_id}` (Preferred voice, speed, and language configuration)
  - `POST /memory/refresh` (Sync memory from latest farmer profile state)

#### 11. Core Farmer & Profile CRUD
- **Endpoints:** Full CRUD on `/farmers` and `/farmer-profiles`.

---

### E. Live Market Prices & Mandis (`/market`)

#### 12. Mandi Price Intelligence
- **Endpoints:**
  - `GET /market/prices` (Query commodity market prices by `commodity`, `district`, `state`)
  - `GET /market/commodities` (List all standardized agricultural commodities)
  - `GET /market/markets` (List tracked APMC mandis and locations)
  - `POST /market/refresh` (Trigger asynchronous Agmarknet data synchronization)

---

### F. Weather Forecast & Climate Alerts (`/weather`)

#### 13. Hyperlocal Weather Forecast
- **Endpoint:** `GET /weather/forecast`
- **Query Params:** `latitude`, `longitude`, `district`, `state`
- **Response:** Current condition, temperature, humidity, wind speed, 5-day / 3-hour forecast slots, and rain alerts.

---

### G. Hyperlocal Agri Shops & Inventory (`/shops`, `/inventory`, `/orders`)

#### 14. Shop Directory & Geo-Search
- **Endpoints:**
  - `GET /shops/nearby` (Haversine spatial ranking by latitude/longitude or district)
  - `GET /shops/search-product` (Search input retailers stocking specific fertilizers, seeds, pesticides)
  - Full CRUD on `/shops/{shop_id}`

#### 15. Inventory Stock Management
- **Endpoints:**
  - `GET /inventory/dashboard` (Shopkeeper stock summary and valuation metrics)
  - `GET /inventory/low-stock` (Threshold low-stock warnings)
  - `PATCH /inventory/{inventory_id}/stock` (Quick stock quantity adjustment)
  - Full CRUD on `/inventory`

#### 16. Farmer Order Requests
- **Endpoints:**
  - `POST /orders` (Place purchase request)
  - `GET /orders/farmer/{farmer_id}` (Farmer order history)
  - `GET /orders/shop/{shop_id}` (Retailer pending orders)
  - `PATCH /orders/{order_id}/status` (Update fulfillment status: Pending, Confirmed, Delivered, Cancelled)

---

### H. Payments & Checkout (`/payments`)

#### 17. Razorpay Integration
- **Endpoints:**
  - `POST /payments/create-order` (Initiates Razorpay order for an existing `order_id`)
  - `POST /payments/verify` (Cryptographically verifies `razorpay_signature` via HMAC-SHA256)
  - `POST /payments/webhook` (Asynchronous payment capture and refund event webhook)

---

### I. Government Subsidies & Schemes (`/schemes`)

#### 18. Scheme Matching & Applications
- **Endpoints:**
  - `GET /schemes` (List available central & state agricultural schemes)
  - `POST /schemes/evaluate` (Evaluate farmer eligibility against land size, crop, state, and category)
  - `POST /schemes/apply` (Submit farmer scheme application)
  - `GET /schemes/applications/{farmer_id}` (Track farmer application status)

---

### J. Human Expert Escalation (`/escalation`)

#### 19. Extension Officer Ticket Queue & Resolution
- **Endpoints:**
  - `GET /escalation/experts` (List verified active agricultural scientists and AEOs)
  - `GET /escalation/tickets` (Queue of escalation tickets for Admin & Expert dashboards)
  - `PATCH /escalation/tickets/{ticket_id}/status` (Update status to Assigned / Resolved with agronomic notes)
  - `GET /escalation/farmer/{farmer_id}/tickets` (Farmer consultation ticket history)

> **Storage Architecture**: Escalation tickets are stored directly inside `FarmerMemory.expert_consultation_history` JSON for contextual retention and auditability.

---

## 4. API Security & Reliability Rules

1. **Strict Input Validation:** All endpoints use Pydantic v2 schemas for strong type enforcement.
2. **Cryptographic Webhook Verification:** Both WhatsApp (`X-Hub-Signature-256`) and Razorpay (`X-Razorpay-Signature`) webhooks are cryptographically authenticated before payload processing.
3. **RBAC Protection:** Sensitive management endpoints (inventory updates, ticket resolutions, admin tools) enforce role validation via dependency injection.
4. **Graceful Fallbacks & Zero Hallucination:** If external providers (OpenWeather, Agmarknet, Gemini) are unreachable, the system returns deterministic cached data or honest guidance without fabricating recommendations.
