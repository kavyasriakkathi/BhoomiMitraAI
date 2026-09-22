# BhoomiMitra AI — API Endpoints

## Authentication
- POST /auth/register
- POST /auth/login
- POST /auth/logout
- GET /auth/me

## Farmers
- POST /farmers
- GET /farmers
- GET /farmers/{farmer_id}
- PUT /farmers/{farmer_id}
- DELETE /farmers/{farmer_id}

## Farmer Profiles
- POST /farmer-profiles
- GET /farmer-profiles
- GET /farmer-profiles/farmer/{farmer_id}
- GET /farmer-profiles/{profile_id}
- PUT /farmer-profiles/{profile_id}
- DELETE /farmer-profiles/{profile_id}

## Farms
- POST /farms
- GET /farms
- GET /farms/farmer/{farmer_id}
- GET /farms/{farm_id}
- PUT /farms/{farm_id}
- DELETE /farms/{farm_id}

## Crops
- POST /crops
- GET /crops
- GET /crops/farm/{farm_id}
- GET /crops/{crop_id}
- PUT /crops/{crop_id}
- DELETE /crops/{crop_id}

## Crop Health
- POST /crop-health
- GET /crop-health
- GET /crop-health/crop/{crop_id}
- GET /crop-health/farmer/{farmer_id}
- GET /crop-health/{diagnosis_id}
- PUT /crop-health/{diagnosis_id}
- DELETE /crop-health/{diagnosis_id}

## Advisories
- POST /advisories
- GET /advisories
- GET /advisories/farmer/{farmer_id}
- GET /advisories/{advisory_id}
- PUT /advisories/{advisory_id}
- DELETE /advisories/{advisory_id}

## Shops
- POST /shops
- GET /shops
- GET /shops/farmer-search
- GET /shops/nearby
- GET /shops/search/location
- GET /shops/search/product
- GET /shops/{shop_id}
- PUT /shops/{shop_id}
- DELETE /shops/{shop_id}

## Inventory
- POST /inventory
- GET /inventory/search
- GET /inventory/shop/{shop_id}
- GET /inventory/dashboard/{shop_id}
- GET /inventory/dashboard/{shop_id}/low-stock
- GET /inventory/dashboard/{shop_id}/out-of-stock
- GET /inventory/{item_id}
- PUT /inventory/{item_id}
- DELETE /inventory/{item_id}
- PATCH /inventory/{item_id}/stock

## Orders
- POST /orders
- GET /orders/farmer/{farmer_id}
- GET /orders/shop/{shop_id}
- GET /orders/analytics/{shop_id}
- GET /orders/{order_id}
- PATCH /orders/{order_id}/status

## Payments
- POST /payments/create-order
- POST /payments/verify
- POST /payments/webhook

## Schemes
- GET /schemes
- POST /schemes
- GET /schemes/eligibility/{farmer_id}
- POST /schemes/apply
- GET /schemes/farmer-applications/{farmer_id}

## RAG
- POST /rag/upload
- POST /rag/rebuild
- GET /rag/search
- GET /rag/documents
- DELETE /rag/document/{id}
- POST /rag/query

## Market
- GET /market/prices
- POST /market/prices
- GET /market/prices/commodities

## Weather
- GET /weather/forecast

## Escalation
- GET /escalation/experts
- POST /escalation/experts
- GET /escalation/experts/{expert_id}
- PUT /escalation/experts/{expert_id}
- GET /escalation/tickets
- PATCH /escalation/tickets/{ticket_id}/status
- GET /escalation/tickets/farmer/{farmer_id}

## Analytics
- GET /analytics/summary
- GET /analytics/activity

## AI
- POST /ai/generate
- GET /ai/health
