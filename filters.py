"""Filter logic: does this Form D look like a US tech funding round?

Three modes, picked via config.yaml (filters.mode):

  keyword         — default. Regex across issuer name, industry_group, and
                    previous names. Word-boundary-prefix match so 'ai' hits
                    'AI Labs' but not 'Dairy' or 'Mosaic'.
  sic             — strict. SIC must be in the allow-list; unknown SIC fails.
  industry_group  — strict. Form D's self-reported industryGroupType must be
                    in the allow-list. Often more accurate than SIC for early
                    stage, since SEC blanks SIC on brand-new issuers.

All modes additionally require country in config.countries and
total_offering_amount >= min_offering_size. Unknown amounts pass through --
user rule: 'mark as unknown, don't skip'.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from parser import Filing

log = logging.getLogger(__name__)


# Form D's X0708 schema has a fixed set of industryGroupType values; these
# are the tech-coded ones used by the industry_group mode's default list.
DEFAULT_TECH_INDUSTRY_GROUPS: tuple[str, ...] = (
    "Computers",
    "Telecommunications",
    "Other Technology",
)


@dataclass
class FilterConfig:
    mode: str = "keyword"
    countries: list[str] = field(default_factory=lambda: ["US"])
    min_offering_size: int = 1_000_000
    keywords: list[str] = field(default_factory=list)
    sic_codes: list[str] = field(default_factory=list)
    industry_groups: list[str] = field(default_factory=lambda: list(DEFAULT_TECH_INDUSTRY_GROUPS))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FilterConfig":
        if not d:
            return cls()
        return cls(
            mode=d.get("mode", "keyword"),
            countries=list(d.get("countries") or ["US"]),
            min_offering_size=int(d.get("min_offering_size", 1_000_000)),
            keywords=list(d.get("keywords") or []),
            # SIC codes are strings to preserve leading zeros (e.g. "0900").
            sic_codes=[str(s) for s in (d.get("sic_codes") or [])],
            industry_groups=list(d.get("industry_groups") or DEFAULT_TECH_INDUSTRY_GROUPS),
        )


@dataclass
class Decision:
    passes: bool
    reason: str  # short tag, good for logs/metrics
    detail: str = ""  # human-readable context

    def __bool__(self) -> bool:
        return self.passes


def _build_keyword_pattern(keywords: list[str]) -> re.Pattern[str] | None:
    """Word-boundary-prefix match, case-insensitive.

    '\\bkw' rather than '\\bkw\\b' so short stems match longer words:
    'tech' -> 'technology', 'dev' -> 'developer'. Left boundary still
    blocks substring false-positives ('ai' in 'Dairy').
    """
    kws = [k.strip() for k in keywords if k and k.strip()]
    if not kws:
        return None
    parts = [re.escape(k) for k in kws]
    return re.compile(r"\b(?:" + "|".join(parts) + r")", re.IGNORECASE)


class FilterEvaluator:
    """Compiles the filter config once, then evaluates filings cheaply."""

    def __init__(self, cfg: FilterConfig):
        if cfg.mode not in ("keyword", "sic", "industry_group"):
            raise ValueError(f"Unknown filter mode: {cfg.mode!r}")
        self.cfg = cfg
        self._countries = {c.upper() for c in cfg.countries}
        self._sics = set(cfg.sic_codes)
        self._industry_groups = set(cfg.industry_groups)
        self._keyword_pattern = _build_keyword_pattern(cfg.keywords) if cfg.mode == "keyword" else None

    def evaluate(self, filing: Filing) -> Decision:
        # Country gate applies to every mode.
        if filing.country.upper() not in self._countries:
            return Decision(False, "country_mismatch", f"country={filing.country!r}")

        # Size gate -- unknown amounts pass by design.
        if filing.total_offering_amount is not None:
            if filing.total_offering_amount < self.cfg.min_offering_size:
                return Decision(
                    False,
                    "size_too_small",
                    f"${filing.total_offering_amount:,} < ${self.cfg.min_offering_size:,}",
                )

        # Tech gate -- mode-specific.
        if self.cfg.mode == "keyword":
            if self._keyword_pattern is None:
                return Decision(False, "no_keywords_configured", "")
            haystack_parts = [filing.issuer_name or "", filing.industry_group or ""]
            haystack_parts.extend(filing.previous_names)
            haystack = " | ".join(haystack_parts)
            m = self._keyword_pattern.search(haystack)
            if not m:
                return Decision(False, "no_keyword_match", f"name={filing.issuer_name!r}")
            return Decision(True, "keyword_match", f"matched {m.group(0)!r}")

        if self.cfg.mode == "sic":
            if not filing.sic:
                return Decision(False, "sic_unknown", "SIC not available from submissions API")
            if filing.sic not in self._sics:
                return Decision(False, "sic_mismatch", f"sic={filing.sic!r}")
            return Decision(True, "sic_match", f"sic={filing.sic!r}")

        # industry_group mode
        if not filing.industry_group:
            return Decision(False, "industry_group_unknown", "no industryGroupType in Form D")
        if filing.industry_group not in self._industry_groups:
            return Decision(
                False,
                "industry_group_mismatch",
                f"industry_group={filing.industry_group!r}",
            )
        return Decision(True, "industry_group_match", f"industry_group={filing.industry_group!r}")
