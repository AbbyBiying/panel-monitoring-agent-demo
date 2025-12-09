# panel_monitoring/app/rules/occupation_rules.py
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


def looks_like_human_hourly_job(text: str) -> bool:
    if not text:
        return False

    t = text.lower()
    has_wage_pattern = any(p in t for p in ["$", "usd", "per hour", "/hour", "hourly"])
    has_job_word = any(job in t for job in HUMAN_JOB_KEYWORDS)

    return has_wage_pattern and has_job_word


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
