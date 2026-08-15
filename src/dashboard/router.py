"""
BhoomiMitra AI — Web Dashboard Router

Serves the unified SaaS dashboard interface rendering Jinja2 HTML templates.
Routes:
  GET /           — Main BhoomiMitra Unified SaaS Dashboard
  GET /dashboard  — Alias for dashboard
  GET /farmer     — Alias for Farmer perspective
  GET /shop       — Alias for Shop Owner perspective
"""

import os
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

# Setup Jinja2 templates directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


@router.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
@router.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
@router.get("/farmer", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
@router.get("/shop", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
async def serve_unified_dashboard(request: Request):
    """
    Renders the BhoomiMitra Unified SaaS Application Dashboard.
    """
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "title": "BhoomiMitra AI — Unified Farmer & Agri-Shop SaaS Platform",
        }
    )


@router.get("/data-deletion", response_class=HTMLResponse, tags=["Legal"])
@router.get("/data-deletion.html", response_class=HTMLResponse, tags=["Legal"], include_in_schema=False)
async def serve_data_deletion_page(request: Request):
    """
    Renders the public Data Deletion Instructions for Meta/WhatsApp Business Compliance.
    """
    return templates.TemplateResponse(
        request=request,
        name="data-deletion.html",
    )

