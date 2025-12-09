# panel_monitoring/app/rules/occupation_rules.py
import difflib
import re
from typing import Dict, Any
import structlog

log = structlog.get_logger(__name__)

HUMAN_JOB_KEYWORDS = {
    "housework",
    "housekeeping",
    "hotel",
    "cleaning",
    "cleaner",
    "janitor",
    "server",
    "cashier",
    "cook",
    "dishwasher",
    "warehouse",
    "delivery",
}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _has_fuzzy_job_word(text: str, cutoff: float = 0.8) -> bool:
    words = _tokenize_words(text)
    for w in words:
        # find a close match to any known job keyword
        if difflib.get_close_matches(w, HUMAN_JOB_KEYWORDS, n=1, cutoff=cutoff):
            return True
    return False


def _has_wage_pattern(text: str) -> bool:
    t = text.lower()

    # basic patterns that survive common typos
    patterns = [
        r"\b\d+\s*(usd|dollars?)\b",
        r"\$\s*\d+",
        r"\bper\s*hour\b",
        r"\b/hr\b",
        r"\b/hour\b",
    ]

    # quick-and-dirty fix for the `usd` → `use` typo case
    t = re.sub(r"\b(\d+)\s*use\b", r"\1 usd", t)

    return any(re.search(p, t) for p in patterns)


def looks_like_human_hourly_job(text: str) -> bool:
    if not text:
        return False

    t = text.lower()
    
    return _has_wage_pattern(text) and _has_fuzzy_job_word(text)


def apply_occupation_rules(
    normalized_signals: Dict[str, Any],
    *,
    event_text: str,
    event_id: str | None = None,
) -> Dict[str, Any]:
    """
    Post-process LLM signals with deterministic, rule-based overrides
    related to occupation / income text.
    """

    if looks_like_human_hourly_job(event_text):
        if normalized_signals.get("suspicious_signup"):
            log.info(
                "rules.apply_occupation_rules.downweight_suspicion_for_hourly_job",
                event_id=event_id,
                event_text=event_text,
            )
            normalized_signals["suspicious_signup"] = False
            normalized_signals["normal_signup"] = True
            normalized_signals["confidence"] = max(
                normalized_signals.get("confidence", 0.0), 0.75
            )
            normalized_signals["reason"] = (
                "Realistic human job with hourly wage; treating as normal."
            )

    return normalized_signals
