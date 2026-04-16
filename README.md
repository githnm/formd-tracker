# form-d-watch

Monitor SEC EDGAR for new Form D filings from US tech companies and alert on
interesting ones. Form D is filed within 15 days of first sale; EDGAR publishes
immediately — this tool catches funding rounds before they hit the news.

<img width="1457" height="873" alt="image" src="https://github.com/user-attachments/assets/4e46fe90-719d-4e68-994a-5ca129d6b552" />

## Quick start

Five commands to a working web UI. The only thing you need to edit is your
name + email (SEC requires a real contact in the User-Agent header, or it
returns `403`).

```bash
git clone https://github.com/githnm/formd-tracker.git && cd formd-tracker
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env       # then edit .env: EDGAR_USER_AGENT="Your Name your@email.com"
python main.py backfill --days 7
python main.py serve       # open http://127.0.0.1:8000
```

That's it. To also poll for new filings every 30 min, open a second terminal
and run `python main.py run` from the same directory.

## Configure

`config.yaml` controls polling interval, filters, and alerts. Defaults are
sensible — you can usually leave it alone.

Optional: set `WEBHOOK_URL` (in `.env` or `config.yaml`) to deliver alerts to
Slack or Discord. Set `alerts.webhook_format` to `slack` or `discord`.

## Usage

Typical setup: two terminals, one polling, one serving the UI.

```bash
# Terminal 1 — seed the DB and start the poller
python main.py backfill --days 7    # pull recent filings (silent, no alerts)
python main.py run                   # poll every 30 min (Ctrl-C to stop)

# Terminal 2 — browse in your browser
python main.py serve                 # http://127.0.0.1:8000
```

Both processes read/write the same `form_d.sqlite`; SQLite handles this fine.

Other commands:

```bash
python main.py run --once                                 # single poll, useful for cron
python main.py list --since 2026-04-01 --min-size 5000000 # query from the CLI
python main.py stats                                      # counts by state/industry/size
python main.py test-alert                                 # verify console/webhook wiring
python main.py serve --host 0.0.0.0 --port 8080           # bind to LAN
```

`backfill` defaults to **no alerts** (so you don't blast a channel with 500
stale filings). Pass `--alerts` to send them anyway.

## Web UI

`python main.py serve` starts a FastAPI server with a single-page table view:

- Sortable, searchable, paginated table of every stored filing
- Stats strip: total filings, total raised, $100M+ rounds, top state, top industry
- Click any row's **View** to see full details + related persons + link to EDGAR
- Amendment counts shown inline (e.g. `D +3` for three D/A updates applied)

Read-only over the SQLite DB — safe to run alongside `run`. JSON endpoints at
`/api/filings`, `/api/filings/{accession}`, `/api/stats` if you want to script
against it.

The table includes **Site** and **LinkedIn** columns that open a Google search
for the issuer's name (Form D's XML doesn't actually include either URL).
One click jumps to results for `"Issuer Name" official site` or
`"Issuer Name" linkedin`.

## Filter modes

Set `filters.mode` in `config.yaml`:

- **`industry_group`** (default): strict — Form D's self-reported
  `industryGroupType` must be in `filters.industry_groups` (defaults to
  `Computers`, `Telecommunications`, `Other Technology`). High precision —
  drops the real-estate / PE-fund noise that loose keyword matching lets in.
- **`sic`**: strict — SIC code from the EDGAR submissions API must be in
  `filters.sic_codes`. Unknown SIC fails (common for brand-new issuers).
- **`keyword`**: loose — regex match across issuer name, industry group,
  and previous names. Word-boundary-prefix — so `ai` matches "AI Labs" and
  "Open AI" but not "Dairy" or "Mosaic". Higher recall, more false positives.

All modes additionally require country in `filters.countries` and
`total_offering_amount >= filters.min_offering_size`. **Unknown offering
amounts are NOT dropped** — they're still stored and alerted with "unknown".

## Amendments (Form D/A)

Form D amendments are updates to existing records, not new filings. A D/A has
its own accession number but references the same offering. This tool links
them via `(cik, date_of_first_sale)` and updates the original row in place,
tracking `amendment_count` and `latest_amendment_accession`. If a D/A arrives
for an offering we never saw (started watching too late), it's stored as its
own "orphan" row and alerted as an amendment.

## Project layout

```
form-d-watch/
├── main.py              # Typer CLI: run, backfill, list, test-alert, stats, serve
├── edgar_client.py      # HTTP (UA, 10 req/s rate limit, retry)
├── parser.py            # Atom feed + primary_doc.xml -> Filing
├── filters.py           # keyword / sic / industry_group
├── storage.py           # SQLite schema, upsert, amendment linking
├── alerts.py            # console + Slack/Discord webhook (opt-in)
├── web.py               # FastAPI: serves the web UI + JSON API
├── templates/
│   └── index.html       # Grid.js table + detail modal
├── config.yaml
├── .env.example
├── requirements.txt
└── README.md
```

## EDGAR endpoints used

| Endpoint | Purpose |
|---|---|
| `sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=D` (Atom) | `run` — polling feed |
| `efts.sec.gov/LATEST/search-index?forms=D,D/A` (JSON) | `backfill` — date-ranged search |
| `sec.gov/Archives/edgar/data/{cik}/{acc}/primary_doc.xml` | Filing detail |
| `data.sec.gov/submissions/CIK{padded}.json` | SIC enrichment |

Rate-limited client-side to 10 requests/second per SEC fair-access policy.
Retries with exponential backoff on `429` (honoring `Retry-After`), `5xx`,
and transient network errors.

## Database

SQLite, default at `./form_d.sqlite`. Schema in [storage.py](storage.py).
Three tables:

- `filings` — one row per offering (amendments update in place)
- `related_persons` — many per filing; replaced on amendment
- `seen_accessions` — dedup log for every accession ever processed

Safe to open with any SQLite client for ad-hoc queries.
