"""HTTP client for SEC EDGAR.

Responsibilities:
- Enforce SEC's 10 req/sec rate limit (min 100 ms between requests).
- Send the required User-Agent on every request.
- Retry 429 / 5xx / connection errors with exponential backoff (tenacity),
  honoring Retry-After when EDGAR sends it.
- Expose the four endpoints form-d-watch actually uses; return raw
  bytes/dicts and let parser.py handle XML/JSON decoding.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


# --- endpoints ---------------------------------------------------------------
LATEST_FEED_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"

MIN_INTERVAL_SEC = 0.10  # 10 req/sec cap per SEC fair-access policy


class RateLimited(Exception):
    """EDGAR returned 429 after retries were exhausted."""


class EdgarClient:
    def __init__(self, user_agent: str, timeout: float = 30.0):
        if not user_agent or "@" not in user_agent:
            # EDGAR rejects requests without a UA that looks like a contact.
            raise ValueError(
                "EDGAR requires a User-Agent with a contact email, e.g. "
                "'Your Name your@email.com'. Set it in config.yaml or .env."
            )
        self.user_agent = user_agent
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._last_request_at = 0.0

    # --- core request with rate limit + retry -------------------------------
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_INTERVAL_SEC:
            time.sleep(MIN_INTERVAL_SEC - elapsed)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                RateLimited,
            )
        ),
    )
    def _get(self, url: str, params: dict[str, Any] | None = None, host: str | None = None) -> requests.Response:
        self._throttle()
        headers = {"Host": host} if host else {}
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        finally:
            self._last_request_at = time.monotonic()

        # Honor Retry-After on 429 before backing off via tenacity.
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(min(float(retry_after), 60.0))
                except ValueError:
                    pass
            log.warning("EDGAR 429 rate-limited on %s; retrying", url)
            raise RateLimited(f"429 from {url}")

        if 500 <= resp.status_code < 600:
            log.warning("EDGAR %s on %s; retrying", resp.status_code, url)
            # Treat 5xx like a transient network error so tenacity retries.
            raise requests.exceptions.ConnectionError(f"{resp.status_code} from {url}")

        resp.raise_for_status()
        return resp

    # --- public endpoints ---------------------------------------------------
    def get_latest_form_d(self, count: int = 100) -> bytes:
        """Atom feed of latest Form D / D/A filings across all issuers.

        Returns raw XML bytes. Used by the polling loop.
        """
        params = {
            "action": "getcurrent",
            "type": "D",
            "company": "",
            "dateb": "",
            "owner": "include",
            "count": count,
            "output": "atom",
        }
        resp = self._get(LATEST_FEED_URL, params=params, host="www.sec.gov")
        return resp.content

    def search_form_d(self, start_date: str, end_date: str, from_offset: int = 0) -> dict[str, Any]:
        """EDGAR full-text search for Form D filings in a date range.

        Dates are YYYY-MM-DD. Returns parsed JSON. Paginate with from_offset
        (step by 100 hits per page). Used by backfill.
        """
        params = {
            "q": "",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "forms": "D,D/A",
            "from": from_offset,
        }
        resp = self._get(FULL_TEXT_SEARCH_URL, params=params)
        return resp.json()

    def get_primary_doc_xml(self, cik: str | int, accession_number: str) -> bytes:
        """Fetch the Form D primary_doc.xml for a given filing.

        accession_number may be dashed ('0001765647-26-000001') or not.
        """
        cik_int = int(str(cik).lstrip("0") or "0")
        acc_nodash = accession_number.replace("-", "")
        url = f"{ARCHIVES_BASE}/{cik_int}/{acc_nodash}/primary_doc.xml"
        resp = self._get(url, host="www.sec.gov")
        return resp.content

    def get_submissions(self, cik: str | int) -> dict[str, Any]:
        """Fetch company submissions JSON (has SIC, sicDescription, addresses)."""
        cik_padded = str(cik).lstrip("0").zfill(10)
        url = f"{SUBMISSIONS_BASE}/CIK{cik_padded}.json"
        resp = self._get(url, host="data.sec.gov")
        return resp.json()
