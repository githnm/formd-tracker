"""Parsers for EDGAR responses.

Three inputs, two output shapes:
  Atom feed + full-text search JSON  -> list[FilingPointer]
  primary_doc.xml + a FilingPointer  -> Filing

parse_submissions() pulls SIC / sicDescription from data.sec.gov since those
fields are NOT present in the Form D XML itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from lxml import etree

log = logging.getLogger(__name__)


class ParseError(Exception):
    """Malformed or unexpected EDGAR payload. Caller should log and skip."""


# US states + DC + territories. Form D's <stateOrCountry> uses these 2-letter
# codes for domestic issuers and different codes/descriptions for foreign ones.
US_STATE_CODES = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
        "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
        "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
        "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
        "WI", "WY", "DC", "PR", "VI", "GU", "AS", "MP",
    }
)


# --- data shapes -------------------------------------------------------------
@dataclass
class FilingPointer:
    """Lightweight filing reference produced by the polling / search layer."""

    accession_number: str
    cik: str
    form_type: str  # "D" or "D/A"
    company_name: str
    filed_at: datetime | None
    index_url: str


@dataclass
class RelatedPerson:
    name: str
    street1: str | None = None
    street2: str | None = None
    city: str | None = None
    state_or_country: str | None = None
    zip_code: str | None = None
    relationships: list[str] = field(default_factory=list)
    clarification: str | None = None


@dataclass
class Filing:
    # identity
    accession_number: str
    form_type: str
    is_amendment: bool
    cik: str
    issuer_name: str
    amendment_count: int = 0  # bumped by storage when D/A amendments land

    # issuer profile
    previous_names: list[str] = field(default_factory=list)
    entity_type: str | None = None
    year_of_inc: str | None = None
    jurisdiction_of_inc: str | None = None

    # address
    street1: str | None = None
    street2: str | None = None
    city: str | None = None
    state: str | None = None  # raw 2-letter code from Form D
    country: str = "US"  # derived: "US" if state is a US code, else description
    zip_code: str | None = None
    phone: str | None = None

    # industry
    industry_group: str | None = None  # self-reported in Form D
    sic: str | None = None  # enriched from submissions API; often absent
    sic_description: str | None = None

    # offering
    total_offering_amount: int | None = None  # None = unknown (don't skip)
    total_amount_sold: int | None = None
    total_remaining: int | None = None
    date_of_first_sale: str | None = None  # ISO YYYY-MM-DD
    minimum_investment: int | None = None
    has_non_accredited: bool | None = None
    number_already_invested: int | None = None

    # people
    related_persons: list[RelatedPerson] = field(default_factory=list)

    # URLs & timing
    filing_url: str = ""  # EDGAR index page (human-facing)
    primary_doc_url: str = ""  # primary_doc.xml location
    filed_at: datetime | None = None


# --- helpers -----------------------------------------------------------------
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_ACCESSION_RE = re.compile(r"accession-number=([\d-]+)")
_CIK_RE = re.compile(r"\((\d{7,10})\)")
_ARCHIVES_DATA_RE = re.compile(r"/Archives/edgar/data/(\d+)/([^/]+)/")


def _text(elem: etree._Element | None, path: str) -> str | None:
    """Find child at `path` and return stripped text, or None."""
    if elem is None:
        return None
    found = elem.find(path)
    if found is None or found.text is None:
        return None
    s = found.text.strip()
    return s or None


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def _strip_ns(root: etree._Element) -> etree._Element:
    """Remove XML namespaces in place so find() paths stay simple.

    Form D submissions are usually unnamespaced, but this is cheap insurance
    against a future schema change adding one.
    """
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _derive_country(state_code: str | None, description: str | None) -> str:
    if state_code and state_code.upper() in US_STATE_CODES:
        return "US"
    if description:
        return description
    return state_code or ""


# --- pointer parsers ---------------------------------------------------------
def parse_atom_feed(xml_bytes: bytes) -> list[FilingPointer]:
    """Parse the getcurrent Atom feed into FilingPointers.

    The feed carries MANY form types; we keep only D and D/A.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise ParseError(f"Atom feed XML invalid: {e}") from e

    pointers: list[FilingPointer] = []
    for entry in root.findall("a:entry", _ATOM_NS):
        term_el = entry.find("a:category", _ATOM_NS)
        form_type = term_el.get("term") if term_el is not None else None
        if form_type not in ("D", "D/A"):
            continue

        title = _text(entry, "a:title") or ""
        # "D - Beacon Biosignals, Inc. (0001765647) (Filer)"
        company_name = ""
        cik = ""
        m_cik = _CIK_RE.search(title)
        if m_cik:
            cik = m_cik.group(1)
            # company name is between "<form> - " and " (<cik>)"
            prefix = f"{form_type} - "
            if title.startswith(prefix):
                name_end = title.index(f"({cik})")
                company_name = title[len(prefix) : name_end].strip()

        id_text = _text(entry, "a:id") or ""
        m_acc = _ACCESSION_RE.search(id_text)
        accession = m_acc.group(1) if m_acc else ""

        link_el = entry.find("a:link", _ATOM_NS)
        index_url = link_el.get("href") if link_el is not None else ""

        updated_text = _text(entry, "a:updated")
        filed_at: datetime | None = None
        if updated_text:
            try:
                filed_at = datetime.fromisoformat(updated_text)
            except ValueError:
                log.debug("Unparseable Atom <updated>: %r", updated_text)

        if not accession or not cik:
            log.warning("Skipping Atom entry missing accession or CIK: %r", title)
            continue

        pointers.append(
            FilingPointer(
                accession_number=accession,
                cik=cik,
                form_type=form_type,
                company_name=company_name,
                filed_at=filed_at,
                index_url=index_url,
            )
        )
    return pointers


def parse_full_text_search(payload: dict[str, Any]) -> list[FilingPointer]:
    """Parse the efts.sec.gov search JSON into FilingPointers."""
    pointers: list[FilingPointer] = []
    hits = payload.get("hits", {}).get("hits", [])
    for hit in hits:
        src = hit.get("_source", {})
        form_type = src.get("form") or src.get("file_type")
        if form_type not in ("D", "D/A"):
            continue
        ciks = src.get("ciks") or []
        cik = ciks[0] if ciks else ""
        adsh = src.get("adsh") or ""
        display_names = src.get("display_names") or []
        company_name = display_names[0] if display_names else ""
        # "ACME CORP  (ACME)  (CIK 0001234567)" -> "ACME CORP"
        if company_name:
            company_name = re.split(r"\s{2,}\(", company_name)[0].strip()
        file_date = src.get("file_date")
        filed_at = None
        if file_date:
            try:
                filed_at = datetime.fromisoformat(file_date)
            except ValueError:
                pass
        # Build the index URL so callers have a human link even from search.
        index_url = ""
        if cik and adsh:
            cik_int = int(str(cik).lstrip("0") or "0")
            acc_nodash = adsh.replace("-", "")
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{adsh}-index.htm"

        if not adsh or not cik:
            continue

        pointers.append(
            FilingPointer(
                accession_number=adsh,
                cik=str(cik),
                form_type=form_type,
                company_name=company_name,
                filed_at=filed_at,
                index_url=index_url,
            )
        )
    return pointers


# --- primary_doc.xml parser --------------------------------------------------
def parse_primary_doc(xml_bytes: bytes, pointer: FilingPointer) -> Filing:
    """Parse a Form D primary_doc.xml and return a fully populated Filing.

    Missing elements are fine -- fields stay None/empty. Raises ParseError
    only for malformed XML or missing <primaryIssuer> (which means we don't
    even know who filed).
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        raise ParseError(f"primary_doc.xml invalid for {pointer.accession_number}: {e}") from e

    _strip_ns(root)
    issuer = root.find("primaryIssuer")
    if issuer is None:
        raise ParseError(f"primary_doc.xml missing <primaryIssuer> for {pointer.accession_number}")

    address = issuer.find("issuerAddress")
    state_code = _text(address, "stateOrCountry")
    state_desc = _text(address, "stateOrCountryDescription")

    offering = root.find("offeringData")
    industry_group = None
    is_amendment = False
    date_of_first_sale: str | None = None
    total_offering_amount: int | None = None
    total_amount_sold: int | None = None
    total_remaining: int | None = None
    min_invest: int | None = None
    has_non_accredited: bool | None = None
    number_already: int | None = None

    if offering is not None:
        industry_group = _text(offering, "industryGroup/industryGroupType")
        is_amendment = _bool(_text(offering, "typeOfFiling/newOrAmendment/isAmendment")) or False
        date_of_first_sale = _text(offering, "typeOfFiling/dateOfFirstSale/value")
        total_offering_amount = _int(_text(offering, "offeringSalesAmounts/totalOfferingAmount"))
        total_amount_sold = _int(_text(offering, "offeringSalesAmounts/totalAmountSold"))
        total_remaining = _int(_text(offering, "offeringSalesAmounts/totalRemaining"))
        min_invest = _int(_text(offering, "minimumInvestmentAccepted"))
        has_non_accredited = _bool(_text(offering, "investors/hasNonAccreditedInvestors"))
        number_already = _int(_text(offering, "investors/totalNumberAlreadyInvested"))

    # Fall back to the amendment signal from form_type if XML says false but
    # the Atom/search feed labeled the form D/A.
    if pointer.form_type == "D/A":
        is_amendment = True

    # year of incorporation is either "overFiveYears"/"withinFiveYears" flags
    # or an explicit <value>YYYY</value>.
    year_of_inc: str | None = None
    yoi = issuer.find("yearOfInc")
    if yoi is not None:
        if _bool(_text(yoi, "overFiveYears")):
            year_of_inc = "over_five_years"
        elif _bool(_text(yoi, "withinFiveYears")):
            val = _text(yoi, "value")
            year_of_inc = val or "within_five_years"
        else:
            year_of_inc = _text(yoi, "value")

    previous_names = [
        (el.text or "").strip()
        for el in issuer.findall("edgarPreviousNameList/previousName")
        if el.text
    ]

    related_persons: list[RelatedPerson] = []
    for rp in root.findall("relatedPersonsList/relatedPersonInfo"):
        name_el = rp.find("relatedPersonName")
        first = _text(name_el, "firstName") or ""
        middle = _text(name_el, "middleName") or ""
        last = _text(name_el, "lastName") or ""
        full_name = " ".join(p for p in (first, middle, last) if p).strip()
        if not full_name:
            continue
        rp_addr = rp.find("relatedPersonAddress")
        rels = [
            (el.text or "").strip()
            for el in rp.findall("relatedPersonRelationshipList/relationship")
            if el.text
        ]
        related_persons.append(
            RelatedPerson(
                name=full_name,
                street1=_text(rp_addr, "street1"),
                street2=_text(rp_addr, "street2"),
                city=_text(rp_addr, "city"),
                state_or_country=_text(rp_addr, "stateOrCountry"),
                zip_code=_text(rp_addr, "zipCode"),
                relationships=rels,
                clarification=_text(rp, "relationshipClarification"),
            )
        )

    # Build canonical primary_doc URL from the pointer.
    cik_int = int(pointer.cik.lstrip("0") or "0")
    acc_nodash = pointer.accession_number.replace("-", "")
    primary_doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/primary_doc.xml"
    )

    return Filing(
        accession_number=pointer.accession_number,
        form_type=pointer.form_type,
        is_amendment=is_amendment,
        cik=pointer.cik,
        issuer_name=_text(issuer, "entityName") or pointer.company_name,
        previous_names=previous_names,
        entity_type=_text(issuer, "entityType"),
        year_of_inc=year_of_inc,
        jurisdiction_of_inc=_text(issuer, "jurisdictionOfInc"),
        street1=_text(address, "street1"),
        street2=_text(address, "street2"),
        city=_text(address, "city"),
        state=state_code,
        country=_derive_country(state_code, state_desc),
        zip_code=_text(address, "zipCode"),
        phone=_text(issuer, "issuerPhoneNumber"),
        industry_group=industry_group,
        # sic/sic_description left None; enriched separately from submissions
        total_offering_amount=total_offering_amount,
        total_amount_sold=total_amount_sold,
        total_remaining=total_remaining,
        date_of_first_sale=date_of_first_sale,
        minimum_investment=min_invest,
        has_non_accredited=has_non_accredited,
        number_already_invested=number_already,
        related_persons=related_persons,
        filing_url=pointer.index_url,
        primary_doc_url=primary_doc_url,
        filed_at=pointer.filed_at,
    )


# --- submissions JSON --------------------------------------------------------
def parse_submissions(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract SIC / sicDescription from the submissions JSON.

    Both can be missing -- SEC docs are explicit that SIC is self-reported
    and often blank, particularly for brand-new issuers.
    """
    sic = payload.get("sic") or None
    if sic:
        sic = str(sic).strip() or None
    return {
        "sic": sic,
        "sic_description": (payload.get("sicDescription") or "").strip() or None,
    }
