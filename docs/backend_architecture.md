# 🏗️ BhoomiMitra AI — Backend Architecture

> **Document Type:** Backend Engineering & Architecture Specification  
> **Audience:** Backend Engineers, DevOps, System Architects  
> **Objective:** Define a highly scalable, fault-tolerant, cloud-native backend architecture for BhoomiMitra AI, designed to serve millions of farmers via WhatsApp.

---

## I. Architectural Paradigm & Technology Stack

### 1. Overall Backend Architecture & 3. Modular Monolith vs Microservices
- **Choice:** **Modular Monolith** initially, transitioning to **Microservices** at scale.
- **Justification:** For a startup, maintaining a distributed microservices architecture out of the gate introduces massive operational overhead (network latency, distributed tracing, complex deployments). A Modular Monolith (FastAPI) allows strict logical separation of domains (e.g., `src/weather/`, `src/ai/`) within a single deployable unit. Background workers (Celery/ARQ) handle heavy async tasks like AI processing.

### Technology Choices
- **Web Framework:** FastAPI (Python) — Fast, async-native, highly suited for AI integration.
- **17. Database Layer:** PostgreSQL 16 with PostGIS (for geospatial queries).
- **18. Cache Layer:** Redis Cluster (Sessions, caching API responses, rate limits).
- **20. Event Queue:** RabbitMQ or Redis Pub/Sub (for async webhooks and notifications).
- **21. File Storage:** Amazon S3 / Google Cloud Storage (for images and voice files).

---

## II. System Structure & Interactions

### 2. Folder Structure
```text
bhoomimitra-ai/
├── src/
│   ├── core/           # Config, DB connections, Security, Base models
│   ├── gateway/        # WhatsApp API endpoints (6)
│   ├── ai/             # Core reasoning & model orchestration (5, 32)
│   ├── farmers/        # Farmer profiles & state (7)
│   ├── commerce/       # Agro Shops & Inventory (8, 33)
│   ├── advisory/       # Crop, Disease, Pesticide Logic (9, 14, 15, 16)
│   ├── external/       # Weather, Market, Schemes APIs (10, 11, 12)
│   ├── notifications/  # Reminders, Alerts, Event routing (13, 34)
│   ├── analytics/      # Data aggregation for dashboards (35)
│   ├── media/          # Voice & Image processing pipelines (14, 15)
│   └── worker.py       # Celery/ARQ background job entrypoint
├── tests/              # Pytest unit & integration tests
├── docs/               # Architecture documents
└── docker-compose.yml  # Local dev environment
```

### Request Lifecycle
1. Meta sends a WhatsApp webhook to `/api/v1/webhook`.
2. **Gateway** verifies HMAC signature and enqueues the message to RabbitMQ. Webhook returns `200 OK` instantly to Meta.
3. **Background Job Processing (19)** picks up the message.
4. If Voice/Image, **Media Services (14, 15)** convert to text/labels.
5. **Farmer Service (7)** fetches profile and session context from DB/Redis.
6. **AI Service (5)** processes intent, interacting with **Weather (10)**, **Market (11)**, or **Commerce (8)** services as needed.
7. Output is generated, converted to speech (if needed), and routed back to Meta via the WhatsApp API.

---

## III. Core Services & Domains

### 4. Authentication Service
Manages OAuth2/JWT for web dashboards (Admins, Experts, Shops) and validates WhatsApp webhooks. Acts as middleware blocking unauthorized requests.

### 5. AI Service & 32. Model Orchestration
- **Architecture:** LangChain or LlamaIndex orchestration.
- **Multi-Provider Fallback (36):** Primary (Google Gemini 1.5 Pro). If timeout or 500 error occurs -> Fallback (OpenAI GPT-4o-mini). If both fail -> Deterministic Rules Engine or Human Expert Escalation.
- **Cost Optimization (37):** Heavy caching of identical queries (e.g., "Tomato price today"). Use smaller, faster models (GPT-3.5/Gemini Flash) for intent classification, reserving expensive models only for complex reasoning.

### 6. WhatsApp Service
The bridge to Meta's Cloud API. Handles text formatting, interactive buttons, and media upload/download via S3.

### 7. Farmer Service & 30. Offline Sync
- **Offline Strategy:** Since farmers might be in poor network areas, WhatsApp inherently retries sending messages. Our backend must ensure **Idempotency** (using Meta's message IDs) so if a delayed message arrives 3 times, we only process it once. Time-sensitive alerts (like weather) check the message timestamp and discard if stale.

### 8. Agro Shop, 11. Market Price & 33. Geospatial Services
- **Design:** Uses **PostGIS** `ST_DWithin` functions to instantly find the nearest Mandi or Agro Shop based on the farmer's stored coordinates. Caches spatial queries heavily in Redis.

### 9. Agriculture Expert Service
Provides an internal API for the web dashboard where experts can view low-confidence AI conversations, see crop images, and type manual responses which are then routed back to the farmer via the WhatsApp Service.

### 10. Weather & 12. Government Scheme Services
Proxy services that connect to third-party APIs (OpenWeatherMap, Data.gov.in). Implements Circuit Breakers (preventing our app from crashing if the external API is down) and aggressive Redis caching.

### 13. Notification Service & 34. Scheduling Architecture
- **Architecture:** Celery Beat or Cloud Scheduler.
- **Design:** A cron job runs every 15 minutes, scanning the `reminders` table. If a trigger condition is met (e.g., Crop is 45 days old), it enqueues a push task to RabbitMQ. 

### 14. Image Processing & 15. Voice Processing
Media is uploaded to S3. Presigned URLs are passed to external models (Google Vision / Whisper STT). This is entirely async to prevent blocking the main API threads.

### 16. Recommendation Engine
A deterministic Python logic engine (not an LLM) that calculates exact dosages for fertilizers and pesticides based on crop age and soil type, ensuring mathematical accuracy.

### 31. Multi-language Support Architecture
Every text string generated by the AI is tagged with a language code. The system maintains a localized template dictionary for standard messages (greetings, errors) and uses APIs (IndicTrans2) for dynamic AI translations before sending.

### 35. Analytics and Reporting Architecture
An async event bus emits every action (e.g., "Crop Recommended", "Price Checked") to a separate OLAP database (ClickHouse or a read-replica PostgreSQL) so complex dashboard queries don't slow down the main transactional database.

---

## IV. Operations, Resilience & Scaling

### 19. Background Job Processing & 20. Event Queue
All heavy lifting (LLM calls, Media conversion, Mass notifications) is done by worker nodes polling RabbitMQ. This ensures the web server never hangs.

### 22. Logging, 23. Monitoring, & 24. Error Handling
- **Logging:** JSON structured logs pushed to Datadog/ELK. Every log entry includes a `correlation_id` (the WhatsApp message ID) to trace a request across all services.
- **Monitoring:** Prometheus scrapes FastAPI metrics. Grafana dashboards visualize 5xx errors, P99 latency, and AI token usage.
- **Error Handling:** Global exception handlers in FastAPI return standardized error responses. Sentry captures all unhandled exceptions.

### 25. Configuration Management
Handled strictly via Environment Variables loaded into Pydantic `BaseSettings`. Secrets reside in AWS Secrets Manager, injected at runtime.

### 26. Testing Strategy
- **Unit Tests (Pytest):** Testing calculation logic in the Recommendation Engine.
- **Integration Tests:** Mocking Meta APIs and testing the full webhook lifecycle.
- **E2E Tests:** Automated WhatsApp test accounts sending actual messages in staging.

### 27. Scalability Strategy & 29. Deployment Strategy
- **Phase 1:** Docker containers deployed to a single heavy VM (Render/AWS EC2) with managed PostgreSQL.
- **Phase 2 (Growth):** ECS / Kubernetes (K8s) cluster. Web nodes auto-scale based on CPU usage. Worker nodes auto-scale based on RabbitMQ queue depth.
- **Deployment:** Blue/Green deployments via GitHub Actions to ensure zero-downtime updates.

### 28. High Availability (HA)
- **Database:** Multi-AZ deployment (Primary + Hot Standby).
- **Compute:** Spread across multiple availability zones.
- **Application Level:** Multi-AI provider fallback ensures the core service functions even if OpenAI/Google goes down.
