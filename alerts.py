"""Alerting: console + optional Slack/Discord webhook.

Webhook failures are logged but never raised — polling must not die because
Slack returned a 502.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from parser import Filing

log = logging.getLogger("form_d_watch.alerts")

AlertKind = Literal["new", "amendment", "test"]


@dataclass
class AlertConfig:
    console: bool = True
    webhook_url: str = ""
    webhook_format: str = "slack"  # "slack" or "discord"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AlertConfig":
        if not d:
            return cls()
        return cls(
            console=bool(d.get("console", True)),
            webhook_url=(d.get("webhook_url") or "").strip(),
            webhook_format=(d.get("webhook_format") or "slack").lower(),
        )


def _fmt_money(amount: int | None) -> str:
    if amount is None:
        return "unknown"
    return f"${amount:,}"


def _headline(filing: Filing, kind: AlertKind) -> str:
    prefix = {
        "new": "[NEW FORM D]",
        "amendment": "[AMENDMENT → D/A]",
        "test": "[TEST ALERT]",
    }[kind]
    offer = _fmt_money(filing.total_offering_amount)
    loc = filing.state or filing.country or "?"
    return f"{prefix} {filing.issuer_name} — {offer} ({loc})"


def _render_console(filing: Filing, kind: AlertKind) -> str:
    lines = [
        _headline(filing, kind),
        f"  Accession     : {filing.accession_number}  ({filing.form_type})",
        f"  CIK           : {filing.cik}",
        f"  Address       : {', '.join(p for p in (filing.city, filing.state, filing.zip_code) if p) or '(unknown)'}",
        f"  Country       : {filing.country}",
        f"  Industry      : {filing.industry_group or '(unknown)'}"
        + (f"  [SIC {filing.sic}]" if filing.sic else ""),
        f"  Offering      : {_fmt_money(filing.total_offering_amount)}"
        f"  Sold: {_fmt_money(filing.total_amount_sold)}"
        f"  Remaining: {_fmt_money(filing.total_remaining)}",
        f"  First sale    : {filing.date_of_first_sale or 'unknown'}",
    ]
    if filing.related_persons:
        top = filing.related_persons[:5]
        people = "; ".join(f"{p.name} ({', '.join(p.relationships) or '—'})" for p in top)
        more = "" if len(filing.related_persons) <= 5 else f"  (+{len(filing.related_persons) - 5} more)"
        lines.append(f"  Related       : {people}{more}")
    if filing.filing_url:
        lines.append(f"  URL           : {filing.filing_url}")
    return "\n".join(lines)


# --- webhook payload builders ------------------------------------------------
def _related_persons_summary(filing: Filing, limit: int = 5) -> str:
    if not filing.related_persons:
        return "—"
    top = filing.related_persons[:limit]
    parts = [f"*{p.name}* ({', '.join(p.relationships) or '—'})" for p in top]
    if len(filing.related_persons) > limit:
        parts.append(f"_+{len(filing.related_persons) - limit} more_")
    return "\n".join(parts)


def _slack_payload(filing: Filing, kind: AlertKind) -> dict[str, Any]:
    headline = _headline(filing, kind)
    fields = [
        {"type": "mrkdwn", "text": f"*Offering*\n{_fmt_money(filing.total_offering_amount)}"},
        {"type": "mrkdwn", "text": f"*Sold*\n{_fmt_money(filing.total_amount_sold)}"},
        {"type": "mrkdwn", "text": f"*First sale*\n{filing.date_of_first_sale or 'unknown'}"},
        {"type": "mrkdwn", "text": f"*Industry*\n{filing.industry_group or '(unknown)'}"},
        {"type": "mrkdwn", "text": f"*Location*\n{filing.city or ''}, {filing.state or filing.country}"},
        {"type": "mrkdwn", "text": f"*CIK*\n{filing.cik}"},
    ]
    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": headline[:150]}},
        {"type": "section", "fields": fields},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Related persons*\n{_related_persons_summary(filing)}"},
        },
    ]
    if filing.filing_url:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"<{filing.filing_url}|View on EDGAR> · `{filing.accession_number}`"}
                ],
            }
        )
    return {"text": headline, "blocks": blocks}


def _discord_payload(filing: Filing, kind: AlertKind) -> dict[str, Any]:
    color = {"new": 0x2ECC71, "amendment": 0xF39C12, "test": 0x95A5A6}[kind]
    headline = _headline(filing, kind)
    fields = [
        {"name": "Offering", "value": _fmt_money(filing.total_offering_amount), "inline": True},
        {"name": "Sold", "value": _fmt_money(filing.total_amount_sold), "inline": True},
        {"name": "First sale", "value": filing.date_of_first_sale or "unknown", "inline": True},
        {"name": "Industry", "value": filing.industry_group or "(unknown)", "inline": True},
        {
            "name": "Location",
            "value": f"{filing.city or ''}, {filing.state or filing.country}".strip(", ") or "—",
            "inline": True,
        },
        {"name": "CIK", "value": filing.cik, "inline": True},
    ]
    if filing.related_persons:
        top = filing.related_persons[:5]
        val = "\n".join(f"**{p.name}** ({', '.join(p.relationships) or '—'})" for p in top)
        if len(filing.related_persons) > 5:
            val += f"\n*+{len(filing.related_persons) - 5} more*"
        fields.append({"name": "Related persons", "value": val[:1024], "inline": False})
    embed = {
        "title": headline[:256],
        "url": filing.filing_url or None,
        "color": color,
        "fields": fields,
        "footer": {"text": filing.accession_number},
    }
    return {"embeds": [embed]}


# --- Alerter -----------------------------------------------------------------
class Alerter:
    def __init__(self, cfg: AlertConfig, user_agent: str = "form-d-watch"):
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Content-Type": "application/json"})
        if cfg.webhook_url and cfg.webhook_format not in ("slack", "discord"):
            raise ValueError(f"Unknown webhook_format: {cfg.webhook_format!r}")

    def send(self, filing: Filing, kind: AlertKind = "new") -> None:
        if self.cfg.console:
            log.info("\n%s", _render_console(filing, kind))
        if self.cfg.webhook_url:
            self._post_webhook(filing, kind)

    # --- webhook with retry, never raises --------------------------------
    def _post_webhook(self, filing: Filing, kind: AlertKind) -> None:
        builder = _slack_payload if self.cfg.webhook_format == "slack" else _discord_payload
        payload = builder(filing, kind)
        try:
            self._post_with_retry(payload)
        except RetryError as e:
            log.warning("Webhook delivery failed after retries: %s", e)
        except Exception as e:  # never let alert failures kill the poller
            log.warning("Webhook delivery failed: %s", e)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
    )
    def _post_with_retry(self, payload: dict[str, Any]) -> None:
        resp = self._session.post(self.cfg.webhook_url, json=payload, timeout=10)
        # Slack returns 200 with body "ok"; Discord returns 204.
        if resp.status_code >= 500:
            raise requests.exceptions.ConnectionError(f"{resp.status_code} from webhook")
        if not resp.ok:
            # 4xx means the payload is bad -- don't retry, just log and move on.
            log.warning("Webhook returned %s: %s", resp.status_code, resp.text[:300])


# --- sample filing for `test-alert` -----------------------------------------
def sample_filing() -> Filing:
    """A synthetic but realistic Form D record for `form-d-watch test-alert`."""
    from parser import RelatedPerson

    return Filing(
        accession_number="0009999999-26-000001",
        form_type="D",
        is_amendment=False,
        cik="0009999999",
        issuer_name="Acme AI Labs, Inc.",
        entity_type="Corporation",
        year_of_inc="2023",
        jurisdiction_of_inc="DELAWARE",
        street1="123 Market St",
        city="San Francisco",
        state="CA",
        country="US",
        zip_code="94105",
        phone="415-555-0100",
        industry_group="Other Technology",
        sic="7372",
        sic_description="Prepackaged Software",
        total_offering_amount=25_000_000,
        total_amount_sold=18_000_000,
        total_remaining=7_000_000,
        date_of_first_sale="2026-04-01",
        minimum_investment=250_000,
        has_non_accredited=False,
        number_already_invested=12,
        related_persons=[
            RelatedPerson(name="Jane Founder", city="San Francisco", state_or_country="CA",
                          relationships=["Executive Officer", "Director"]),
            RelatedPerson(name="Sam Investor", city="Menlo Park", state_or_country="CA",
                          relationships=["Director"]),
        ],
        filing_url="https://www.sec.gov/Archives/edgar/data/9999999/000999999926000001/0009999999-26-000001-index.htm",
        primary_doc_url="https://www.sec.gov/Archives/edgar/data/9999999/000999999926000001/primary_doc.xml",
    )
