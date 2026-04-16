"""SQLite persistence for Form D filings.

Schema:
  filings              — one row per OFFERING (not per accession). PK is the
                         original D's accession_number; amendments update
                         this row in place per the user's spec.
  related_persons      — many-per-filing. Replaced on amendment (the latest
                         set is what matters for VC-frequency analysis).
  seen_accessions      — every accession we've ever processed (original or
                         amendment). Primary check for polling dedup.

Amendment linking:
  A D/A has its own accession number and does NOT share it with the original
  D. SEC's file_num IS shared, but it's not in primary_doc.xml. To avoid a
  second HTTP request per filing we match on (cik, date_of_first_sale),
  which together uniquely identify an offering. Fall back to most-recent
  non-amended D from the same CIK within 60 days if the date is missing.
  Truly unmatchable amendments are inserted as their own 'orphan' rows.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from parser import Filing, RelatedPerson

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
    accession_number        TEXT PRIMARY KEY,
    form_type               TEXT NOT NULL,
    is_amendment            INTEGER NOT NULL DEFAULT 0,
    amendment_count         INTEGER NOT NULL DEFAULT 0,
    latest_amendment_accession TEXT,
    latest_amended_at       TEXT,

    cik                     TEXT NOT NULL,
    issuer_name             TEXT NOT NULL,
    previous_names          TEXT,          -- JSON array
    entity_type             TEXT,
    year_of_inc             TEXT,
    jurisdiction_of_inc     TEXT,

    street1 TEXT, street2 TEXT, city TEXT,
    state TEXT, country TEXT NOT NULL,
    zip_code TEXT, phone TEXT,

    industry_group          TEXT,
    sic                     TEXT,
    sic_description         TEXT,

    total_offering_amount   INTEGER,
    total_amount_sold       INTEGER,
    total_remaining         INTEGER,
    date_of_first_sale      TEXT,
    minimum_investment      INTEGER,
    has_non_accredited      INTEGER,  -- 0 / 1 / NULL
    number_already_invested INTEGER,

    filing_url              TEXT,
    primary_doc_url         TEXT,
    filed_at                TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_filings_cik            ON filings(cik);
CREATE INDEX IF NOT EXISTS idx_filings_filed_at       ON filings(filed_at);
CREATE INDEX IF NOT EXISTS idx_filings_first_sale     ON filings(date_of_first_sale);
CREATE INDEX IF NOT EXISTS idx_filings_country_size   ON filings(country, total_offering_amount);
CREATE INDEX IF NOT EXISTS idx_filings_cik_firstsale  ON filings(cik, date_of_first_sale);

CREATE TABLE IF NOT EXISTS related_persons (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    filing_accession_number TEXT NOT NULL,
    name                    TEXT NOT NULL,
    street1 TEXT, street2 TEXT, city TEXT,
    state_or_country TEXT, zip_code TEXT,
    relationships           TEXT,          -- JSON array
    clarification           TEXT,
    FOREIGN KEY (filing_accession_number) REFERENCES filings(accession_number)
);
CREATE INDEX IF NOT EXISTS idx_related_filing ON related_persons(filing_accession_number);
CREATE INDEX IF NOT EXISTS idx_related_name   ON related_persons(name);

CREATE TABLE IF NOT EXISTS seen_accessions (
    accession_number        TEXT PRIMARY KEY,
    applied_to_accession    TEXT NOT NULL,
    form_type               TEXT NOT NULL,
    seen_at                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seen_applied ON seen_accessions(applied_to_accession);
"""


UpsertAction = Literal["inserted", "updated", "skipped", "orphan_amendment"]


@dataclass
class UpsertResult:
    action: UpsertAction
    primary_accession: str  # the row key the user should refer to
    filing: Filing | None = None  # the stored filing (post-merge for amendments)


# Fields that amendments are allowed to overwrite on the parent row.
# Identity fields (accession_number, cik, created_at) are not in this list.
_AMENDABLE_FIELDS = (
    "issuer_name",
    "previous_names",
    "entity_type",
    "year_of_inc",
    "jurisdiction_of_inc",
    "street1", "street2", "city", "state", "country", "zip_code", "phone",
    "industry_group", "sic", "sic_description",
    "total_offering_amount", "total_amount_sold", "total_remaining",
    "date_of_first_sale", "minimum_investment",
    "has_non_accredited", "number_already_invested",
    "filing_url", "primary_doc_url",
)


class FilingStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ----- basic lifecycle --------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FilingStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ----- dedup check ------------------------------------------------------
    def has_seen(self, accession_number: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_accessions WHERE accession_number = ?",
            (accession_number,),
        ).fetchone()
        return row is not None

    # ----- upsert -----------------------------------------------------------
    def upsert_filing(self, filing: Filing) -> UpsertResult:
        """Insert, update (for amendments), or skip.

        Dedup is the first thing we check — polling repeats are common.
        """
        if self.has_seen(filing.accession_number):
            return UpsertResult(action="skipped", primary_accession=filing.accession_number)

        now = datetime.now(timezone.utc).isoformat()

        if filing.form_type == "D/A" or filing.is_amendment:
            parent_acc = self._find_parent_for_amendment(filing)
            if parent_acc:
                return self._apply_amendment(parent_acc, filing, now)
            # Orphan: the original D was filed before we started watching.
            log.info(
                "Orphan amendment %s (cik=%s, first_sale=%s) — no parent found, inserting as own row",
                filing.accession_number, filing.cik, filing.date_of_first_sale,
            )
            self._insert_new(filing, now)
            return UpsertResult(action="orphan_amendment", primary_accession=filing.accession_number, filing=filing)

        self._insert_new(filing, now)
        return UpsertResult(action="inserted", primary_accession=filing.accession_number, filing=filing)

    # ----- internal: parent lookup for amendments --------------------------
    def _find_parent_for_amendment(self, filing: Filing) -> str | None:
        # Preferred match: same (CIK, date_of_first_sale). Tightest possible
        # identifier without fetching file_num from the index page.
        if filing.date_of_first_sale:
            row = self._conn.execute(
                """
                SELECT accession_number FROM filings
                WHERE cik = ? AND date_of_first_sale = ?
                ORDER BY COALESCE(filed_at, created_at) DESC
                LIMIT 1
                """,
                (filing.cik, filing.date_of_first_sale),
            ).fetchone()
            if row:
                return row["accession_number"]

        # Fallback: the most recent filing from the same CIK within 60 days.
        # Form D is filed within 15 days of first sale, so 60 days is a
        # comfortable window for amendments without pulling unrelated earlier
        # offerings from the same issuer.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=60)).date().isoformat()
        row = self._conn.execute(
            """
            SELECT accession_number FROM filings
            WHERE cik = ?
              AND (filed_at IS NULL OR substr(filed_at, 1, 10) >= ?)
            ORDER BY COALESCE(filed_at, created_at) DESC
            LIMIT 1
            """,
            (filing.cik, cutoff),
        ).fetchone()
        return row["accession_number"] if row else None

    # ----- internal: insert new row ----------------------------------------
    def _insert_new(self, filing: Filing, now: str) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO filings (
                    accession_number, form_type, is_amendment,
                    amendment_count, latest_amendment_accession, latest_amended_at,
                    cik, issuer_name, previous_names, entity_type, year_of_inc,
                    jurisdiction_of_inc,
                    street1, street2, city, state, country, zip_code, phone,
                    industry_group, sic, sic_description,
                    total_offering_amount, total_amount_sold, total_remaining,
                    date_of_first_sale, minimum_investment,
                    has_non_accredited, number_already_invested,
                    filing_url, primary_doc_url, filed_at,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filing.accession_number,
                    filing.form_type,
                    1 if filing.is_amendment else 0,
                    0,
                    None,
                    None,
                    filing.cik,
                    filing.issuer_name,
                    json.dumps(filing.previous_names),
                    filing.entity_type,
                    filing.year_of_inc,
                    filing.jurisdiction_of_inc,
                    filing.street1, filing.street2, filing.city,
                    filing.state, filing.country, filing.zip_code, filing.phone,
                    filing.industry_group, filing.sic, filing.sic_description,
                    filing.total_offering_amount,
                    filing.total_amount_sold,
                    filing.total_remaining,
                    filing.date_of_first_sale,
                    filing.minimum_investment,
                    None if filing.has_non_accredited is None else int(filing.has_non_accredited),
                    filing.number_already_invested,
                    filing.filing_url, filing.primary_doc_url,
                    filing.filed_at.isoformat() if filing.filed_at else None,
                    now, now,
                ),
            )
            self._replace_related_persons(filing.accession_number, filing.related_persons)
            cur.execute(
                "INSERT INTO seen_accessions (accession_number, applied_to_accession, form_type, seen_at) VALUES (?, ?, ?, ?)",
                (filing.accession_number, filing.accession_number, filing.form_type, now),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ----- internal: apply amendment to parent row -------------------------
    def _apply_amendment(self, parent_acc: str, amendment: Filing, now: str) -> UpsertResult:
        # Build UPDATE column list from _AMENDABLE_FIELDS but skip values that
        # are None on the amendment — amendments commonly omit fields that
        # haven't changed, and overwriting with NULL would destroy data.
        sets: list[str] = []
        values: list[Any] = []
        amend_dict = asdict(amendment)
        for field_name in _AMENDABLE_FIELDS:
            val = amend_dict.get(field_name)
            if val is None:
                continue
            if field_name == "previous_names":
                if not val:
                    continue
                val = json.dumps(val)
            sets.append(f"{field_name} = ?")
            values.append(val)

        # Always bump the amendment metadata.
        sets.extend([
            "amendment_count = amendment_count + 1",
            "latest_amendment_accession = ?",
            "latest_amended_at = ?",
            "updated_at = ?",
        ])
        values.extend([amendment.accession_number, now, now])
        values.append(parent_acc)

        cur = self._conn.cursor()
        try:
            cur.execute(
                f"UPDATE filings SET {', '.join(sets)} WHERE accession_number = ?",
                values,
            )
            # Replace related persons only if the amendment has them; otherwise
            # keep the existing list.
            if amendment.related_persons:
                self._replace_related_persons(parent_acc, amendment.related_persons)
            cur.execute(
                "INSERT INTO seen_accessions (accession_number, applied_to_accession, form_type, seen_at) VALUES (?, ?, ?, ?)",
                (amendment.accession_number, parent_acc, amendment.form_type, now),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        merged = self.get_filing(parent_acc)
        return UpsertResult(action="updated", primary_accession=parent_acc, filing=merged)

    def _replace_related_persons(self, accession: str, people: Iterable[RelatedPerson]) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM related_persons WHERE filing_accession_number = ?", (accession,))
        cur.executemany(
            """
            INSERT INTO related_persons (
                filing_accession_number, name, street1, street2, city,
                state_or_country, zip_code, relationships, clarification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    accession, p.name, p.street1, p.street2, p.city,
                    p.state_or_country, p.zip_code,
                    json.dumps(p.relationships), p.clarification,
                )
                for p in people
            ],
        )

    # ----- read-side --------------------------------------------------------
    def get_filing(self, accession_number: str) -> Filing | None:
        row = self._conn.execute(
            "SELECT * FROM filings WHERE accession_number = ?", (accession_number,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_filing(row)

    def list_filings(
        self,
        since: str | None = None,
        country: str | None = None,
        min_size: int | None = None,
        limit: int = 500,
    ) -> list[Filing]:
        """since: YYYY-MM-DD. Filters on filed_at; falls back to created_at."""
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("COALESCE(substr(filed_at, 1, 10), substr(created_at, 1, 10)) >= ?")
            params.append(since)
        if country:
            clauses.append("country = ?")
            params.append(country)
        if min_size is not None:
            clauses.append("(total_offering_amount IS NULL OR total_offering_amount >= ?)")
            params.append(min_size)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM filings {where} ORDER BY COALESCE(filed_at, created_at) DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_filing(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Counts by state, by industry_group, and by size bucket."""
        by_state = {
            r["state"] or "(unknown)": r["c"]
            for r in self._conn.execute(
                "SELECT state, COUNT(*) AS c FROM filings GROUP BY state ORDER BY c DESC"
            )
        }
        by_industry = {
            r["industry_group"] or "(unknown)": r["c"]
            for r in self._conn.execute(
                "SELECT industry_group, COUNT(*) AS c FROM filings GROUP BY industry_group ORDER BY c DESC"
            )
        }
        size_sql = """
        SELECT
          SUM(CASE WHEN total_offering_amount IS NULL THEN 1 ELSE 0 END) AS unknown,
          SUM(CASE WHEN total_offering_amount < 1000000 THEN 1 ELSE 0 END) AS under_1m,
          SUM(CASE WHEN total_offering_amount >= 1000000  AND total_offering_amount < 5000000   THEN 1 ELSE 0 END) AS b_1_5m,
          SUM(CASE WHEN total_offering_amount >= 5000000  AND total_offering_amount < 25000000  THEN 1 ELSE 0 END) AS b_5_25m,
          SUM(CASE WHEN total_offering_amount >= 25000000 AND total_offering_amount < 100000000 THEN 1 ELSE 0 END) AS b_25_100m,
          SUM(CASE WHEN total_offering_amount >= 100000000 THEN 1 ELSE 0 END) AS over_100m,
          COUNT(*) AS total
        FROM filings
        """
        size_row = self._conn.execute(size_sql).fetchone()
        by_size = {k: (size_row[k] or 0) for k in (
            "unknown", "under_1m", "b_1_5m", "b_5_25m", "b_25_100m", "over_100m", "total"
        )}
        amendment_total = self._conn.execute(
            "SELECT COALESCE(SUM(amendment_count), 0) AS c FROM filings"
        ).fetchone()["c"]
        total_raised = self._conn.execute(
            "SELECT COALESCE(SUM(total_offering_amount), 0) AS s FROM filings"
        ).fetchone()["s"]
        return {
            "by_state": by_state,
            "by_industry_group": by_industry,
            "by_size_bucket": by_size,
            "total_filings": by_size["total"],
            "total_amendments_applied": amendment_total,
            "total_raised_usd": total_raised,
        }

    # ----- helpers ----------------------------------------------------------
    def _row_to_filing(self, row: sqlite3.Row) -> Filing:
        people = [
            RelatedPerson(
                name=p["name"],
                street1=p["street1"], street2=p["street2"], city=p["city"],
                state_or_country=p["state_or_country"], zip_code=p["zip_code"],
                relationships=json.loads(p["relationships"] or "[]"),
                clarification=p["clarification"],
            )
            for p in self._conn.execute(
                "SELECT * FROM related_persons WHERE filing_accession_number = ? ORDER BY id",
                (row["accession_number"],),
            )
        ]
        filed_at = None
        if row["filed_at"]:
            try:
                filed_at = datetime.fromisoformat(row["filed_at"])
            except ValueError:
                filed_at = None
        return Filing(
            accession_number=row["accession_number"],
            form_type=row["form_type"],
            is_amendment=bool(row["is_amendment"]),
            cik=row["cik"],
            issuer_name=row["issuer_name"],
            amendment_count=row["amendment_count"] or 0,
            previous_names=json.loads(row["previous_names"] or "[]"),
            entity_type=row["entity_type"],
            year_of_inc=row["year_of_inc"],
            jurisdiction_of_inc=row["jurisdiction_of_inc"],
            street1=row["street1"], street2=row["street2"], city=row["city"],
            state=row["state"], country=row["country"],
            zip_code=row["zip_code"], phone=row["phone"],
            industry_group=row["industry_group"],
            sic=row["sic"], sic_description=row["sic_description"],
            total_offering_amount=row["total_offering_amount"],
            total_amount_sold=row["total_amount_sold"],
            total_remaining=row["total_remaining"],
            date_of_first_sale=row["date_of_first_sale"],
            minimum_investment=row["minimum_investment"],
            has_non_accredited=None if row["has_non_accredited"] is None else bool(row["has_non_accredited"]),
            number_already_invested=row["number_already_invested"],
            related_persons=people,
            filing_url=row["filing_url"] or "",
            primary_doc_url=row["primary_doc_url"] or "",
            filed_at=filed_at,
        )
