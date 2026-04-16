"""Typer CLI for form-d-watch.

Commands:
  run               start polling loop (interval from config)
  backfill --days   pull last N days via full-text search
  list    --since   query stored filings
  test-alert        send a sample alert
  stats             print counts by state, industry, size bucket

Both `run` and `backfill` go through the same process_pointer pipeline, so
the dedup / amendment / filter / alert behavior is identical.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv

from alerts import AlertConfig, Alerter, sample_filing
from edgar_client import EdgarClient
from filters import FilterConfig, FilterEvaluator
from parser import (
    FilingPointer,
    ParseError,
    parse_atom_feed,
    parse_full_text_search,
    parse_primary_doc,
    parse_submissions,
)
from storage import FilingStore

app = typer.Typer(
    add_completion=False,
    help="Monitor SEC EDGAR for Form D filings from US tech companies.",
)

log = logging.getLogger("form_d_watch")


# ----- shared app context ----------------------------------------------------
@dataclass
class AppContext:
    cfg: dict
    client: EdgarClient
    filter_eval: FilterEvaluator
    store: FilingStore
    alerter: Alerter

    @classmethod
    def from_config(cls, config_path: Path) -> "AppContext":
        # .env first so env vars can override YAML
        load_dotenv()
        if not config_path.exists():
            typer.echo(f"Config file not found: {config_path}", err=True)
            raise typer.Exit(code=2)
        with config_path.open() as f:
            cfg = yaml.safe_load(f) or {}

        user_agent = (
            os.environ.get("EDGAR_USER_AGENT")
            or (cfg.get("edgar") or {}).get("user_agent")
            or ""
        )

        alerts_cfg = dict(cfg.get("alerts") or {})
        env_webhook = os.environ.get("WEBHOOK_URL")
        if env_webhook:
            alerts_cfg["webhook_url"] = env_webhook

        db_path = (cfg.get("storage") or {}).get("db_path", "form_d.sqlite")

        try:
            client = EdgarClient(user_agent=user_agent)
        except ValueError as e:
            typer.echo(f"EDGAR client error: {e}", err=True)
            raise typer.Exit(code=2)

        return cls(
            cfg=cfg,
            client=client,
            filter_eval=FilterEvaluator(FilterConfig.from_dict(cfg.get("filters"))),
            store=FilingStore(db_path),
            alerter=Alerter(AlertConfig.from_dict(alerts_cfg), user_agent=user_agent),
        )


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers at INFO
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ----- pointer processing pipeline ------------------------------------------
def process_pointer(ctx: AppContext, pointer: FilingPointer, enrich_sic: bool = True) -> str:
    """Fetch, parse, filter, store, alert. Returns a short status tag.

    Status tags: skipped, fetch_error, parse_error, filtered,
                 inserted, updated, orphan_amendment.
    """
    if ctx.store.has_seen(pointer.accession_number):
        return "skipped"

    try:
        xml = ctx.client.get_primary_doc_xml(pointer.cik, pointer.accession_number)
    except Exception as e:
        log.warning("Fetch failed for %s: %s", pointer.accession_number, e)
        return "fetch_error"

    try:
        filing = parse_primary_doc(xml, pointer)
    except ParseError as e:
        log.warning("Parse failed for %s: %s", pointer.accession_number, e)
        return "parse_error"

    if enrich_sic:
        try:
            subs = ctx.client.get_submissions(pointer.cik)
            sic_info = parse_submissions(subs)
            filing.sic = sic_info["sic"]
            filing.sic_description = sic_info["sic_description"]
        except Exception as e:
            log.debug("SIC enrichment failed for cik=%s: %s", pointer.cik, e)

    decision = ctx.filter_eval.evaluate(filing)
    if not decision:
        log.debug(
            "Filtered: %s (%s: %s)",
            filing.issuer_name, decision.reason, decision.detail,
        )
        return "filtered"

    result = ctx.store.upsert_filing(filing)
    if result.action == "skipped":
        return "skipped"

    # Alert kind: amendment for D/A updates (including orphans), else new.
    if result.action in ("updated", "orphan_amendment"):
        kind: str = "amendment"
    else:
        kind = "new"
    ctx.alerter.send(result.filing or filing, kind=kind)  # type: ignore[arg-type]
    return result.action


def _bump(counters: dict[str, int], key: str) -> None:
    counters[key] = counters.get(key, 0) + 1


# ----- `run` -----------------------------------------------------------------
@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    once: bool = typer.Option(False, "--once", help="Run a single cycle and exit"),
) -> None:
    """Start the polling loop. Ctrl-C to stop cleanly."""
    _setup_logging(verbose)
    ctx = AppContext.from_config(config)
    interval_min = int((ctx.cfg.get("polling") or {}).get("interval_minutes", 30))

    shutdown = {"requested": False}

    def _handle(_signum, _frame):  # type: ignore[no-untyped-def]
        shutdown["requested"] = True
        log.info("Shutdown requested; finishing current cycle...")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    log.info("Starting poll loop (every %d min). Ctrl-C to stop.", interval_min)
    while not shutdown["requested"]:
        cycle_start = time.monotonic()
        try:
            _poll_once(ctx)
        except Exception:
            log.exception("Poll cycle crashed")

        if once or shutdown["requested"]:
            break

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, interval_min * 60 - elapsed)
        log.info("Cycle done in %.1fs. Next poll in %.0f min.", elapsed, sleep_for / 60)
        # Short sleeps so Ctrl-C is responsive.
        slept = 0.0
        while slept < sleep_for and not shutdown["requested"]:
            chunk = min(1.0, sleep_for - slept)
            time.sleep(chunk)
            slept += chunk

    ctx.store.close()
    log.info("Stopped.")


def _poll_once(ctx: AppContext) -> None:
    log.info("Polling EDGAR latest-filings feed...")
    try:
        atom = ctx.client.get_latest_form_d(count=100)
    except Exception as e:
        log.error("Failed to fetch Atom feed: %s", e)
        return
    try:
        pointers = parse_atom_feed(atom)
    except ParseError as e:
        log.error("Failed to parse Atom feed: %s", e)
        return

    log.info("Got %d Form D pointers", len(pointers))
    counters: dict[str, int] = {}
    for pointer in pointers:
        try:
            status = process_pointer(ctx, pointer)
        except Exception:
            log.exception("Unhandled error processing %s", pointer.accession_number)
            _bump(counters, "errors")
            continue
        _bump(counters, status)

    log.info(
        "Cycle summary: inserted=%d updated=%d orphans=%d filtered=%d "
        "skipped=%d fetch_err=%d parse_err=%d other_err=%d",
        counters.get("inserted", 0),
        counters.get("updated", 0),
        counters.get("orphan_amendment", 0),
        counters.get("filtered", 0),
        counters.get("skipped", 0),
        counters.get("fetch_error", 0),
        counters.get("parse_error", 0),
        counters.get("errors", 0),
    )


# ----- `backfill` ------------------------------------------------------------
@app.command()
def backfill(
    days: int = typer.Option(..., "--days", "-d", help="How many days back to pull"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_alerts: bool = typer.Option(
        True,  # default ON: silence alerts during backfill
        "--no-alerts/--alerts",
        help="Skip alerts during backfill (default). Use --alerts to send them.",
    ),
) -> None:
    """Pull Form D filings from the last N days via EDGAR full-text search."""
    _setup_logging(verbose)
    ctx = AppContext.from_config(config)

    # Swap in a silent alerter for the duration of the backfill if requested.
    if no_alerts:
        ctx.alerter = Alerter(
            AlertConfig(console=False, webhook_url="", webhook_format="slack"),
            user_agent=ctx.cfg.get("edgar", {}).get("user_agent", "form-d-watch"),
        )

    end = date.today()
    start = end - timedelta(days=days)
    log.info("Backfilling %s to %s", start, end)

    from_offset = 0
    total = 0
    counters: dict[str, int] = {}
    page_size = 100  # EDGAR full-text search returns up to 100 per page

    while True:
        try:
            payload = ctx.client.search_form_d(start.isoformat(), end.isoformat(), from_offset)
        except Exception:
            log.exception("Search failed at offset %d", from_offset)
            break

        raw_hits = payload.get("hits", {}).get("hits", []) or []
        pointers = parse_full_text_search(payload)
        if not pointers and not raw_hits:
            break

        log.info(
            "Page offset=%d: %d raw hits, %d Form D pointers",
            from_offset, len(raw_hits), len(pointers),
        )

        for pointer in pointers:
            try:
                status = process_pointer(ctx, pointer)
            except Exception:
                log.exception("Unhandled error processing %s", pointer.accession_number)
                _bump(counters, "errors")
                continue
            _bump(counters, status)
            total += 1

        if len(raw_hits) < page_size:
            break  # last page
        from_offset += page_size
        if from_offset >= 10_000:
            log.warning("Hit EDGAR full-text 10k cap; narrow the date range for more")
            break

    log.info("Backfill done. Processed %d pointers. Counters: %s", total, counters)
    ctx.store.close()


# ----- `list` ----------------------------------------------------------------
def _fmt_money(amount: Optional[int]) -> str:
    return "unknown" if amount is None else f"${amount:,}"


@app.command("list")
def list_cmd(
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD"),
    country: Optional[str] = typer.Option(None, "--country", help="Country code, e.g. US"),
    min_size: Optional[int] = typer.Option(None, "--min-size"),
    limit: int = typer.Option(50, "--limit"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Query stored filings."""
    _setup_logging(False)
    ctx = AppContext.from_config(config)
    filings = ctx.store.list_filings(
        since=since, country=country, min_size=min_size, limit=limit,
    )
    if not filings:
        typer.echo("No filings found.")
        return
    typer.echo(f"{'Filed':<12} {'Accession':<22} {'Amount':>15}  {'ST':<3} {'Form':<5} Issuer")
    for f in filings:
        filed = f.filed_at.date().isoformat() if f.filed_at else "—"
        form_label = f.form_type + ("*" if f.is_amendment else "")
        typer.echo(
            f"{filed:<12} {f.accession_number:<22} {_fmt_money(f.total_offering_amount):>15}  "
            f"{(f.state or '??'):<3} {form_label:<5} {f.issuer_name}"
        )
    ctx.store.close()


# ----- `test-alert` ----------------------------------------------------------
@app.command("test-alert")
def test_alert_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Send a sample alert (console + webhook if configured)."""
    _setup_logging(False)
    ctx = AppContext.from_config(config)
    ctx.alerter.send(sample_filing(), kind="test")
    typer.echo("Test alert sent.")
    ctx.store.close()


# ----- `serve` ---------------------------------------------------------------
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p"),
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file change (dev)"),
) -> None:
    """Launch the web UI (read-only browser over the stored filings)."""
    _setup_logging(False)
    # web.py reads this env var to locate the DB via the same config.
    os.environ["FORM_D_WATCH_CONFIG"] = str(config.resolve())
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "uvicorn not installed. Run: pip install -r requirements.txt",
            err=True,
        )
        raise typer.Exit(code=2)
    log.info("Serving web UI at http://%s:%d  (Ctrl-C to stop)", host, port)
    uvicorn.run("web:app", host=host, port=port, reload=reload)


# ----- `stats` ---------------------------------------------------------------
@app.command()
def stats(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
) -> None:
    """Print counts by state, industry, and size bucket."""
    _setup_logging(False)
    ctx = AppContext.from_config(config)
    data = ctx.store.stats()

    typer.echo(f"Total filings stored    : {data['total_filings']}")
    typer.echo(f"Amendments applied      : {data['total_amendments_applied']}")

    typer.echo("\nBy size bucket:")
    labels = {
        "unknown": "unknown",
        "under_1m": "< $1M",
        "b_1_5m": "$1M – $5M",
        "b_5_25m": "$5M – $25M",
        "b_25_100m": "$25M – $100M",
        "over_100m": "$100M+",
    }
    for key, label in labels.items():
        typer.echo(f"  {label:<14}: {data['by_size_bucket'].get(key, 0)}")

    typer.echo("\nTop states:")
    for k, v in list(data["by_state"].items())[:15]:
        typer.echo(f"  {k:<12}: {v}")

    typer.echo("\nTop industry groups:")
    for k, v in list(data["by_industry_group"].items())[:15]:
        typer.echo(f"  {k[:30]:<30}: {v}")

    ctx.store.close()


if __name__ == "__main__":
    app()
