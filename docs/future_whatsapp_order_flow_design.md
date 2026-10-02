# Future Conversational WhatsApp Order Flow — Design Note

## 1. Executive Summary & Audit Context
During the BhoomiMitra WhatsApp & Revenue Audit, it was verified that conversational WhatsApp order placement does **not** currently exist in production. The current production database records (`order_requests` table) contain 0 orders, and all existing order endpoints are REST-based (implemented under `/api/v1/orders`), serving direct web API requests.

This document outlines the proposed architecture and state-machine design for conversational WhatsApp order placement in future releases. In accordance with safety rules:
- No conversational order creation logic is enabled in this release.
- Existing REST order endpoints (`/api/v1/orders`) remain active and unaffected.
- No fake or test orders have been inserted into production.
- No external payment integrations are connected until full multi-party validation is complete.

---

## 2. End-to-End Future Order Flow

The conversational WhatsApp order flow follows a sequential multi-turn state machine:

```
Farmer Inbound Request ("నాకు 2 బస్తాల యూరియా కావాలి")
       │
       ▼
[1. Product Identification]
   - Extract commodity / product name ("Urea")
   - Normalize brand / formulation if specified
       │
       ▼
[2. Location Grounding]
   - Retrieve farmer profile location (Village, Mandal, District)
   - Disambiguate if farmer requests delivery to a different locality (e.g. Korutla)
       │
       ▼
[3. Dealer Selection & Inventory Check]
   - Query verified inventory (`inventory` table) for active stock in the local district
   - Select authorized PACS / local dealer with verified stock
   - If multiple dealers, present top 2 nearest options with pricing
       │
       ▼
[4. Quantity & Unit Verification]
   - Parse requested quantity (e.g., 2 bags / 45kg bags)
   - Validate against minimum/maximum order limits and available dealer stock
       │
       ▼
[5. Price Confirmation & Breakdown]
   - Calculate total cost based on verified unit price (`unit_price * quantity`)
   - Include any delivery/pickup terms
   - Present itemized summary in farmer's preferred language (Telugu/English)
       │
       ▼
[6. Farmer Confirmation (Explicit Opt-In)]
   - Request unambiguous confirmation (e.g., "ఆర్డర్ ఖరారు చేయడానికి 'YES' లేదా 'సరే' అని టైప్ చేయండి")
   - Prevent accidental order creation on ambiguous input
       │
       ▼
[7. Order Creation & Persistence]
   - Generate unique Order ID (e.g., `ORD-YYYYMMDD-XXXX`)
   - Persist record in `order_requests` table with status `Pending`
   - Link `farmer_id`, `shop_id`, `items`, `total_amount`, and conversation reference
       │
       ▼
[8. Payment / Collection Arrangement]
   - Phase 1: Cash on Delivery (COD) / Pay on Pickup at PACS or dealer store
   - Phase 2: UPI deep-link / WhatsApp Pay integration with webhook verification
       │
       ▼
[9. Dealer Notification]
   - Trigger real-time alert to shop owner (via WhatsApp Template / SMS / Shop Dashboard)
   - Alert contains farmer name, phone number, items requested, and pickup/delivery window
       │
       ▼
[10. Order Status Updates & Tracking]
   - Shop owner updates status (`Accepted` -> `Ready for Pickup` -> `Completed`) via Dashboard or WhatsApp quick-reply
   - Automated WhatsApp outbound notifications dispatch to farmer at each status change
```

---

## 3. Interaction State Machine Specification

| State | Trigger | Inbound Example | System Action | Next State |
| :--- | :--- | :--- | :--- | :--- |
| **IDLE** | Farmer intent detected | "యూరియా బుక్ చేయాలి" | Detect buy intent, identify crop input | `AWAITING_DEALER` |
| **AWAITING_DEALER** | Locality matched | "కోరుట్ల PACS" | Check verified stock, quote unit price | `AWAITING_QUANTITY` |
| **AWAITING_QUANTITY** | Quantity input | "2 బస్తాలు" | Calculate total, format confirmation card | `AWAITING_CONFIRMATION` |
| **AWAITING_CONFIRMATION**| Explicit approval | "అవును / సరే" | Insert into `order_requests`, notify dealer | `ORDER_PLACED` |
| **ORDER_PLACED** | Order placed | — | Send order receipt & pickup instructions | `IDLE` |

---

## 4. Safety Guardrails & Anti-Fraud Rules
1. **Zero Phantom Inventory:** Never accept an order for products without active, verified inventory rows in `inventory` table.
2. **Double Confirmation:** Always require explicit affirmative confirmation before writing to the database.
3. **No Unauthenticated Payments:** Do not request card numbers, bank credentials, or OTPs over WhatsApp.
4. **Rate Limiting:** Maximum 3 open pending orders per farmer phone number to prevent spam/accidental duplicate orders.
5. **Human Fallback:** If farmer cancels or expresses confusion, seamlessly offer connection to the dealer's verified phone number or local AEO.

---

## 5. Implementation Status
- **Current State:** Architecture documented; REST order endpoints (`/api/v1/orders`) active; conversational WhatsApp ordering disabled.
- **Future Action:** Implement when dealer network onboarding and pilot inventory integrations are completed.
