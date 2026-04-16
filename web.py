"""FastAPI server for browsing stored Form D filings.

Read-only view over the SQLite DB that `python main.py run` writes to.
Start with: python main.py serve

Endpoints:
  GET  /                          -> HTML page (Grid.js table)
  GET  /api/filings               -> table data
  GET  /api/filings/{accession}   -> single filing + related persons
  GET  /api/stats                 -> stats strip
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from parser import Filing, RelatedPerson
from storage import FilingStore

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _load_db_path() -> str:
    """Resolve DB path: DB_PATH env > config storage.db_path > default."""
    load_dotenv()
    env_db = os.environ.get("DB_PATH")
    if env_db:
        return env_db
    config_path = Path(os.environ.get("FORM_D_WATCH_CONFIG") or "config.yaml")
    if config_path.exists():
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("storage") or {}).get("db_path", "form_d.sqlite")
    return "form_d.sqlite"


app = FastAPI(title="form-d-watch")


def _get_store() -> FilingStore:
    # Fresh connection per request -- SQLite handles many readers fine.
    return FilingStore(_load_db_path())


def _filing_row(f: Filing) -> dict[str, Any]:
    """Compact row for the table view."""
    return {
        "accession_number": f.accession_number,
        "form_type": f.form_type,
        "is_amendment": f.is_amendment,
        "amendment_count": f.amendment_count,
        "issuer_name": f.issuer_name,
        "cik": f.cik,
        "state": f.state,
        "country": f.country,
        "industry_group": f.industry_group,
        "sic": f.sic,
        "sic_description": f.sic_description,
        "total_offering_amount": f.total_offering_amount,
        "total_amount_sold": f.total_amount_sold,
        "date_of_first_sale": f.date_of_first_sale,
        "filed_at": f.filed_at.date().isoformat() if f.filed_at else None,
        "filing_url": f.filing_url,
    }


def _filing_detail(f: Filing) -> dict[str, Any]:
    """Full record for the modal detail view."""
    d = asdict(f)
    d["filed_at"] = f.filed_at.isoformat() if f.filed_at else None
    return d


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/filings")
def list_filings(
    since: Optional[str] = Query(None, description="YYYY-MM-DD"),
    country: Optional[str] = Query(None),
    min_size: Optional[int] = Query(None),
    limit: int = Query(1000, ge=1, le=5000),
):
    store = _get_store()
    try:
        filings = store.list_filings(
            since=since, country=country, min_size=min_size, limit=limit,
        )
        return {"count": len(filings), "filings": [_filing_row(f) for f in filings]}
    finally:
        store.close()


@app.get("/api/filings/{accession}")
def get_filing(accession: str):
    store = _get_store()
    try:
        f = store.get_filing(accession)
        if f is None:
            raise HTTPException(status_code=404, detail="Filing not found")
        return _filing_detail(f)
    finally:
        store.close()


@app.get("/api/stats")
def get_stats():
    store = _get_store()
    try:
        return store.stats()
    finally:
        store.close()
