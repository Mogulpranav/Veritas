"""
Confidence thresholds for Veritas.

The Critic replies with two signals:
    SUPPORTED:  YES | NO
    CONFIDENCE: LOW | MEDIUM | HIGH

CONFIDENCE is the Critic's confidence in its own SUPPORTED verdict. On its own it
does not say whether an answer is grounded ("NO / HIGH" means confidently
UNgrounded). So this module combines the two into one 0-5 grounding score:

    NO/HIGH=0   NO/MEDIUM=1   NO/LOW=2   YES/LOW=3   YES/MEDIUM=4   YES/HIGH=5

and lets you set a minimum score an answer must reach before it is shown.

The gate only ever downgrades an answer to a decline. It never lets through
something the Judge already rejected. min_grounding=0 turns the gate off.
"""

import re
from typing import Optional

DECLINE_TEXT = "I don't have enough grounded information to answer this confidently."

_SCORES = {
    (False, "HIGH"): 0,
    (False, "MEDIUM"): 1,
    (False, "LOW"): 2,
    (True, "LOW"): 3,
    (True, "MEDIUM"): 4,
    (True, "HIGH"): 5,
}


def parse_critique(raw: Optional[str]) -> dict:
    """Pull the Critic's verdict and confidence out of its raw text.

    Tolerates markdown bold, extra spaces and different casing. Anything it can't
    find comes back as None, and grounding_score is None unless BOTH were found.
    """
    raw = raw or ""
    m = re.search(r"\bSUPPORTED\b\W*(YES|NO)\b", raw, re.IGNORECASE)
    c = re.search(r"\bCONFIDENCE\b\W*(LOW|MEDIUM|HIGH)\b", raw, re.IGNORECASE)
    supported = (m.group(1).upper() == "YES") if m else None
    confidence = c.group(1).upper() if c else None
    score = _SCORES.get((supported, confidence))
    return {"supported": supported, "confidence": confidence, "grounding_score": score}


def gate(action: str, critique_raw: Optional[str], min_grounding: int = 0) -> dict:
    """Apply the minimum-grounding threshold to the Judge's action.

    Returns {"action", "gated", "reason", "grounding_score"}.
    """
    parsed = parse_critique(critique_raw)
    score = parsed["grounding_score"]
    out = {"action": action, "gated": False, "reason": "", "grounding_score": score}

    if min_grounding <= 0 or action == "REJECT":
        return out
    if score is None:
        out.update(action="REJECT", gated=True, reason="Critic output could not be parsed")
    elif score < min_grounding:
        out.update(
            action="REJECT",
            gated=True,
            reason=f"grounding score {score} is below the minimum of {min_grounding}",
        )
    return out
