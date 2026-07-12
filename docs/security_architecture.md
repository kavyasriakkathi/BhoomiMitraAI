# 🛡️ BhoomiMitra AI — Security & Privacy Architecture

> **Document Type:** Security & Privacy Specification  
> **Audience:** Backend Engineers, Security Analysts, DevOps, Legal/Compliance  
> **Objective:** Define a production-ready, highly secure, and privacy-preserving architecture for an AI platform serving millions of Indian farmers.

---

## I. Identity & Access Management (IAM)

### 1. Authentication & 3. Farmer Identity Verification
- **Purpose:** Ensure only legitimate users and systems access BhoomiMitra AI.
- **Design:** Farmers are implicitly authenticated via WhatsApp (Phone Number binding). Dashboard users (Admins, Experts, Shops) use OAuth2 + OIDC.
- **Best Practices:** No passwords for farmers. Use WhatsApp's built-in cryptographic identity.
- **Implementation Strategy:** Verify Meta's webhook HMAC-SHA256 signatures for every incoming message. For dashboards, enforce MFA (Multi-Factor Authentication).

### 2. Authorization (RBAC) & 6. JWT Strategy
- **Purpose:** Restrict access based on user roles.
- **Design:** Role-Based Access Control (`admin`, `expert`, `shop_owner`).
- **Best Practices:** JWTs must be short-lived (15 minutes). Use HttpOnly, Secure, SameSite=Strict cookies for web dashboards, not LocalStorage.
- **Implementation Strategy:** FastAPI dependency injection to validate JWT scopes before controller execution. Refresh tokens are stored securely in Redis and revocable.

---

## II. Network & API Security

### 4. WhatsApp Security & 5. API Security
- **Purpose:** Prevent spoofing and unauthorized API access.
- **Design:** API Gateway validating payloads. Mutual TLS (mTLS) for internal microservices.
- **Best Practices:** IP whitelisting for Meta's webhook IPs. Prevent replay attacks using WhatsApp Message IDs.
- **Implementation Strategy:** Fast rejection at the Nginx/Cloudflare layer for non-Meta IPs hitting the webhook.

### 20. Rate Limiting & 21. DDoS Protection
- **Purpose:** Ensure system availability and prevent abuse.
- **Design:** Cloudflare WAF for volumetric DDoS. Redis-based rate limiting per user (e.g., 60 msgs/min per farmer).
- **Best Practices:** Implement CAPTCHA on web logins if under attack. Return `429 Too Many Requests` with `Retry-After` headers.
- **Implementation Strategy:** Token bucket algorithm in Redis for fine-grained API limiting.

---

## III. Data Security & Privacy

### 7. Encryption at Rest & 9. Database Security
- **Purpose:** Protect data if physical drives or DB backups are compromised.
- **Design:** AWS KMS/GCP Cloud KMS managed keys. AES-256 for all block storage (EBS/S3).
- **Best Practices:** Encrypt PII (Phone numbers, precise GPS coordinates) at the application layer before DB insertion.
- **Implementation Strategy:** PostgreSQL Transparent Data Encryption (TDE). Use SQLAlchemy `TypeDecorators` for app-layer PII encryption.

### 8. Encryption in Transit
- **Purpose:** Prevent man-in-the-middle (MITM) attacks.
- **Design:** TLS 1.3 only. HSTS enforced.
- **Best Practices:** Disable older protocols (TLS 1.1, 1.2).
- **Implementation Strategy:** Terminate SSL at the Cloudflare edge and API Gateway.

### 10. Image, 11. Voice Data Security & 22. Secure File Upload
- **Purpose:** Prevent malware uploads and protect farmer media.
- **Design:** S3 buckets with block public access. Media is accessed via short-lived pre-signed URLs.
- **Best Practices:** Never execute uploaded files. Strip EXIF metadata (GPS/Device info) from images upon upload.
- **Implementation Strategy:** Lambda function scans media for malware/MIME type mismatches before moving to the main S3 bucket.

### 12. Privacy Protection & 27. Compliance (Privacy-by-Design)
- **Purpose:** Uphold farmer trust and comply with India's DPDP (Digital Personal Data Protection) Act.
- **Design:** Data minimization. Only collect what is needed (Crop, Location, Issue). 
- **Best Practices:** Allow farmers to request data deletion via a simple WhatsApp command ("Delete my account").
- **Implementation Strategy:** Automated anonymization scripts that scrub PII from chat logs after 30 days, keeping only semantic data for AI training.

### 13. Data Retention Policy
- **Purpose:** Reduce liability and storage costs.
- **Design:** 
  - Voice/Images: Delete original media after 15 days (keep extracted text/analysis).
  - Chat Logs: Anonymize after 30 days.
  - Farmer Profiles: Keep until account deletion.

---

## IV. AI-Specific Security

### 17. AI Security, 18. Prompt Injection Protection & 19. Abuse Prevention
- **Purpose:** Prevent attackers from hijacking the LLM or extracting system prompts.
- **Design:** Strict input sanitization and dual-LLM evaluation (a smaller, cheaper LLM checks the user's input for malicious intents before passing to the main reasoning engine).
- **Best Practices:** Never trust user input. Sandbox the AI's output generation.
- **Implementation Strategy:** Use parameterized prompts. Implement a heuristic filter to block known injection phrases ("ignore previous instructions"). Cap daily AI interactions per farmer to prevent LLM resource exhaustion (Abuse prevention).

---

## V. Infrastructure & Operations

### 23. Secrets Management & 24. Environment Variables
- **Purpose:** Prevent credential leaks.
- **Design:** HashiCorp Vault or AWS Secrets Manager.
- **Best Practices:** Never commit `.env` files. Rotate DB passwords and API keys automatically every 30 days.
- **Implementation Strategy:** Applications fetch secrets at startup via IAM roles, keeping them strictly in memory.

### 25. Cloud Security
- **Purpose:** Secure the cloud boundary.
- **Design:** VPC with public/private subnets. Databases and Redis reside in private subnets with no internet gateway.
- **Implementation Strategy:** Bastion hosts / AWS Systems Manager for secure internal access. Security Groups restrict traffic to only necessary ports.

### 16. Audit Logs
- **Purpose:** Track "who did what and when" for compliance and forensics.
- **Design:** Immutable append-only log store (Amazon QLDB or S3 Object Lock).
- **Implementation Strategy:** Log all dashboard logins, PII access, and configuration changes.

---

## VI. Resilience & Incident Management

### 14. Backup Strategy & 15. Disaster Recovery (DR)
- **Purpose:** Recover from data loss or region-wide outages.
- **Design:** Active-Passive multi-region architecture.
- **Best Practices:** RPO (Recovery Point Objective) < 5 mins. RTO (Recovery Time Objective) < 1 hour.
- **Implementation Strategy:** PostgreSQL Point-in-Time Recovery (PITR) enabled. Continuous WAL archiving to S3. Cross-region automated snapshot replication.

### 26. Monitoring & Alerting
- **Purpose:** Detect anomalies in real-time.
- **Design:** Prometheus + Grafana for metrics; PagerDuty for alerts.
- **Best Practices:** Alert on high 5xx error rates, sudden traffic spikes, and AI API budget anomalies.

### 28. Security Incident Response & 29. Business Continuity Plan (BCP)
- **Purpose:** Standardize the response to breaches and catastrophic failures.
- **Design:** 
  - *Phase 1:* Containment (Isolate compromised systems, revoke tokens).
  - *Phase 2:* Eradication (Patch vulnerability).
  - *Phase 3:* Recovery (Restore from clean backups).
- **Implementation Strategy:** Annual tabletop exercises simulating ransomware or data breaches. Automated fallback to a static "System Maintenance" WhatsApp bot if the core AI engine fails.
