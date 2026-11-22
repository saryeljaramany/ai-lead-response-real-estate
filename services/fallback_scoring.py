"""
services/fallback_scoring.py

Deterministic, rule-based lead scorer used as a last-resort fallback when
the AI scoring service is unavailable on the final retry attempt.

This module exposes a single public function, fallback_score, and the
module-level constant FALLBACK_REASONING_PREFIX that other modules can
use to identify scores produced by this service.

Design constraints
------------------
- Pure function: no network calls, no environment reads, no randomness,
  no clock access.
- Never raises for any lead-like input (uses getattr with None defaults
  throughout).
- Produces a LeadScore whose category, suggested_action, and priority_level
  are always consistent with the numeric score and the thresholds defined
  in models.lead.
- The reasoning field contains only fixed labels; no text derived from
  the lead input is ever included.
"""

import re

from models.lead import HOT_SCORE_MIN, WARM_SCORE_MIN, LeadScore

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

FALLBACK_REASONING_PREFIX = "Rule-based fallback (AI unavailable): "

# ---------------------------------------------------------------------------
# Internal helpers -- action / priority lookup
# ---------------------------------------------------------------------------

_ACTIONS = {
    "hot": "Contact the lead today and offer a viewing.",
    "warm": (
        "Follow up within two business days and clarify budget and timeline."
    ),
    "cold": (
        "Add to a nurture list and follow up when more details are available."
    ),
}

_PRIORITIES = {
    "hot": "urgent",
    "warm": "normal",
    "cold": "low",
}


def _category(score: int) -> str:
    """
    Derive the lead temperature category from a numeric score.

    Uses the same thresholds (WARM_SCORE_MIN, HOT_SCORE_MIN) as the
    LeadScore.derive_category model validator so the two always agree.

    Parameters
    ----------
    score : int
        Clamped integer score in the range [0, 100].

    Returns
    -------
    str
        One of "hot", "warm", or "cold".

    Raises
    ------
    None
        This function never raises an exception.
    """
    if score >= HOT_SCORE_MIN:
        return "hot"
    if score >= WARM_SCORE_MIN:
        return "warm"
    return "cold"


# ---------------------------------------------------------------------------
# Pre-compiled regular expressions
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d")

_URGENT_WORD_RE = re.compile(
    r"\b(?:immediate|immediately|asap|now|urgent|urgently)\b",
    re.IGNORECASE,
)

_URGENT_PHRASE_RE = re.compile(
    r"\b(?:this\s+week|this\s+month|next\s+week|next\s+month|few\s+weeks)\b",
    re.IGNORECASE,
)

_MEDIUM_PHRASE_RE = re.compile(
    r"\b(?:few\s+months|this\s+year|next\s+year|later\s+this\s+year)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"(?<!\.)\b(\d+)(?:\s*-\s*|\s*)(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)

_INTENT_WORD_RE = re.compile(
    r"\b(?:view|viewing|viewings|tour|visit|buy|buying|purchase|offer|"
    r"mortgage|preapproved|pre-approved|relocating|relocation|moving|"
    r"rent|renting|lease|schedule)\b",
    re.IGNORECASE,
)

_BROWSING_PHRASE_RE = re.compile(
    r"\b(?:just\s+looking|just\s+browsing|just\s+curious|no\s+rush|"
    r"someday|maybe\s+later|not\s+sure\s+yet|exploring\s+options)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Timeline classification helpers
# ---------------------------------------------------------------------------

def _classify_timeline(text: str) -> str:
    """
    Classify a timeline string as "urgent", "medium", or "other".

    All urgent criteria are checked in Pass 1 before any medium criteria
    in Pass 2, ensuring higher-priority urgency signals always win.

    Parameters
    ----------
    text : str
        The raw timeline string, stripped of surrounding whitespace.

    Returns
    -------
    str
        One of "urgent", "medium", or "other".

    Raises
    ------
    None
        This function never raises an exception.
    """
    # Pass 1: urgent checks (words, phrases, durations)
    if _URGENT_WORD_RE.search(text):
        return "urgent"

    if _URGENT_PHRASE_RE.search(text):
        return "urgent"

    for match in _DURATION_RE.finditer(text):
        n = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        if unit == "day" and n <= 90:
            return "urgent"
        if unit == "week" and n <= 12:
            return "urgent"
        if unit == "month" and n <= 3:
            return "urgent"

    # Pass 2: medium checks (phrases, durations)
    if _MEDIUM_PHRASE_RE.search(text):
        return "medium"

    for match in _DURATION_RE.finditer(text):
        n = int(match.group(1))
        unit = match.group(2).lower().rstrip("s")
        if unit == "month" and 4 <= n <= 12:
            return "medium"
        if unit == "year" and n <= 1:
            return "medium"

    return "other"


# ---------------------------------------------------------------------------
# Message scoring helpers
# ---------------------------------------------------------------------------

def _score_message(text: str) -> tuple[int, list[str]]:
    """
    Compute message-based score delta and collect applicable labels.

    Parameters
    ----------
    text : str
        The message field value, stripped of surrounding whitespace.

    Returns
    -------
    tuple[int, list[str]]
        A (delta, labels) pair where delta is the net integer score change
        (may be negative) and labels is a list of label strings to append
        to the reasoning.

    Raises
    ------
    None
        This function never raises an exception.
    """
    delta = 0
    labels: list[str] = []

    # Detailed message: >= 80 characters.
    if len(text) >= 80:
        delta += 5
        labels.append("detailed message")

    # Specific numbers: contains any ASCII digit.
    if _DIGIT_RE.search(text):
        delta += 5
        labels.append("specific numbers in message")

    # Purchase or viewing intent: whole-word match.
    if _INTENT_WORD_RE.search(text):
        delta += 10
        labels.append("purchase or viewing intent")

    # Browsing language: phrase match.
    if _BROWSING_PHRASE_RE.search(text):
        delta -= 15
        labels.append("browsing language")

    return delta, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fallback_score(lead: object) -> LeadScore:
    """
    Produce a deterministic, rule-based LeadScore for any lead-like input.

    Intended as a last-resort fallback when the AI scoring service is
    unavailable on the final retry. The result is always a valid LeadScore
    and never raises, regardless of the shape or attributes of lead.

    Parameters
    ----------
    lead : object
        An object whose attributes name, email, phone, message, property_type,
        preferred_location, budget, and timeline are read safely using
        getattr(lead, attr, None). Non-str values count as absent. The name,
        email, and phone fields are never read by the scorer.

    Returns
    -------
    LeadScore
        A fully populated LeadScore with:
        - score: integer in [0, 100].
        - category: derived from score using WARM_SCORE_MIN / HOT_SCORE_MIN.
        - reasoning: FALLBACK_REASONING_PREFIX followed by applied labels
          joined with ", ", or "no strong signals" when none applied.
        - suggested_action and priority_level: matched to category.

    Raises
    ------
    None
        This function never raises an exception for any input.
    """
    def _get(attr: str) -> str | None:
        val = getattr(lead, attr, None)
        if not isinstance(val, str):
            return None
        stripped = val.strip()
        return stripped if stripped else None

    points = 10  # base score
    labels: list[str] = []

    # --- Budget ---
    budget = _get("budget")
    if budget is not None:
        if _DIGIT_RE.search(budget):
            points += 20
            labels.append("budget stated")
        else:
            points += 5
            labels.append("budget mentioned")

    # --- Timeline ---
    timeline = _get("timeline")
    if timeline is not None:
        tclass = _classify_timeline(timeline)
        if tclass == "urgent":
            points += 25
            labels.append("urgent timeline")
        elif tclass == "medium":
            points += 12
            labels.append("medium-term timeline")
        else:
            points += 5
            labels.append("timeline given")

    # --- Preferred location (3+ characters after stripping) ---
    location = _get("preferred_location")
    if location is not None and len(location) >= 3:
        points += 10
        labels.append("location given")

    # --- Property type ---
    ptype = _get("property_type")
    if ptype is not None:
        points += 8
        labels.append("property type given")

    # --- Message ---
    msg_raw = getattr(lead, "message", None)
    msg = msg_raw.strip() if isinstance(msg_raw, str) else ""
    msg_delta, msg_labels = _score_message(msg)
    points += msg_delta
    labels.extend(msg_labels)

    # -----------------------------------------------------------------
    # Clamp to [0, 100] and derive category
    # -----------------------------------------------------------------
    score = max(0, min(100, points))
    cat = _category(score)

    reasoning_body = ", ".join(labels) if labels else "no strong signals"
    reasoning = FALLBACK_REASONING_PREFIX + reasoning_body

    return LeadScore(
        score=score,
        reasoning=reasoning,
        suggested_action=_ACTIONS[cat],
        priority_level=_PRIORITIES[cat],
    )
