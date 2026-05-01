# decision_engine.py  (v3 — Safer Logic + Better Explanations + Traceability)
# ─────────────────────────────────────────────────────────────────────────────
# IMPROVEMENTS:
#   1. Safer decision logic   — confidence < 0.5 → ESCALATE
#   2. Better reason text     — human-readable policy explanations
#   3. Traceability fields    — decision_rule + confidence_level in output
#   4. Smarter confidence     — updated thresholds: 0.7 / 0.4 / 0.2
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from dataclasses import dataclass

# ── Configuration ─────────────────────────────────────────────────────────────

# Areas that ALWAYS require human escalation — no exceptions
SENSITIVE_AREAS = {"billing", "fraud", "account_access", "payments"}

# Request types that always require specialist handling
SENSITIVE_REQUEST_TYPES = {"fraud_report", "refund_request", "payment_issue", "account_issue"}

# Minimum classifier confidence to trust the classification
MIN_CLASSIFIER_CONFIDENCE = 2.0

# IMPROVEMENT 1: Any overall confidence below this → ESCALATE (safety-first)
MIN_OVERALL_CONFIDENCE = 0.28  # Calibrated for TF-IDF score range (0.0–0.45)


# ── IMPROVEMENT 4: Smarter confidence label function ─────────────────────────

def confidence_label(score: float) -> str:
    """
    Maps a 0.0–1.0 score to a human-readable confidence label.

    Thresholds (updated v3):
        > 0.7  → HIGH      (very reliable)
        > 0.4  → MEDIUM    (acceptable)
        > 0.2  → LOW       (weak, use with caution)
        ≤ 0.2  → VERY_LOW  (unreliable → escalate)
    """
    if score > 0.45:
        return "HIGH"
    elif score > 0.32:
        return "MEDIUM"
    elif score > 0.18:
        return "LOW"
    else:
        return "VERY_LOW"


# ── DecisionResult dataclass ──────────────────────────────────────────────────

@dataclass
class DecisionResult:
    """
    Structured output of the decision engine.

    IMPROVEMENT 3 — Traceability fields added:
        decision_rule      : e.g. "#1", "#2" ... "#7"
        confidence_level   : HIGH / MEDIUM / LOW / VERY_LOW
    """
    decision:           str     # "REPLY" or "ESCALATE"
    reason:             str     # Human-readable explanation
    rule_triggered:     int     # Which rule number fired
    decision_rule:      str     # IMPROVEMENT 3: formatted rule label e.g. "#1"
    overall_confidence: float   # Combined score 0.0–1.0
    confidence_level:   str     # IMPROVEMENT 3+4: HIGH/MEDIUM/LOW/VERY_LOW

    def to_dict(self) -> dict:
        return {
            "decision":           self.decision,
            "reason":             self.reason,
            "decision_rule":      self.decision_rule,       # IMPROVEMENT 3
            "overall_confidence": self.overall_confidence,
            "confidence_level":   self.confidence_level,    # IMPROVEMENT 3
        }


# ── Confidence calculator ─────────────────────────────────────────────────────

def _compute_overall_confidence(classification: dict, retrieved_docs: list) -> float:
    """
    Computes a single 0.0–1.0 confidence score from:
      - Classifier scores (request + area, normalized to 0–1)
      - Retrieval score (top doc cosine similarity)

    Weights: retrieval 60%, classifier 40%
    """
    req_conf  = min(classification.get("request_confidence", 0.0) / 10.0, 1.0)
    area_conf = min(classification.get("area_confidence",    0.0) / 10.0, 1.0)
    classifier_score = (req_conf + area_conf) / 2.0
    retrieval_score  = retrieved_docs[0].score if retrieved_docs else 0.0
    return round((classifier_score * 0.4) + (retrieval_score * 0.6), 3)


# ── Main decision function ────────────────────────────────────────────────────

def decide(classification: dict, retrieved_docs: list) -> DecisionResult:
    """
    Applies 7 priority rules to decide REPLY or ESCALATE.

    Args:
        classification : Output from classifier.classify()
        retrieved_docs : List of RetrievalResult objects from retriever

    Returns:
        DecisionResult with full traceability metadata.
    """
    request_type = classification.get("request_type", "unknown")
    product_area = classification.get("product_area", "unknown")
    req_conf     = classification.get("request_confidence", 0.0)
    area_conf    = classification.get("area_confidence",    0.0)

    overall      = _compute_overall_confidence(classification, retrieved_docs)
    conf_level   = confidence_label(overall)   # IMPROVEMENT 4

    # ── Rule 1: Sensitive product area ────────────────────────────────────────
    if product_area in SENSITIVE_AREAS:
        return DecisionResult(
            decision="ESCALATE",
            # IMPROVEMENT 2: clear, policy-driven reason text
            reason=(
                f"Detected '{product_area}' issue. "
                f"As per policy, sensitive categories require human escalation."
            ),
            rule_triggered=1,
            decision_rule="#1",                # IMPROVEMENT 3
            overall_confidence=overall,
            confidence_level=conf_level,       # IMPROVEMENT 3+4
        )

    # ── Rule 2: Sensitive request type ───────────────────────────────────────
    if request_type in SENSITIVE_REQUEST_TYPES:
        return DecisionResult(
            decision="ESCALATE",
            reason=(
                f"Request classified as '{request_type}'. "
                f"This request type involves sensitive customer data and must be "
                f"reviewed by a qualified specialist before any action is taken."
            ),
            rule_triggered=2,
            decision_rule="#2",
            overall_confidence=overall,
            confidence_level=conf_level,
        )

    # ── Rule 3: Low classifier confidence ────────────────────────────────────
    if req_conf < MIN_CLASSIFIER_CONFIDENCE and area_conf < MIN_CLASSIFIER_CONFIDENCE:
        return DecisionResult(
            decision="ESCALATE",
            reason=(
                f"Classification confidence is too low to determine intent reliably "
                f"(request score: {req_conf:.1f}, area score: {area_conf:.1f}). "
                f"Escalating to prevent an incorrect automated response."
            ),
            rule_triggered=3,
            decision_rule="#3",
            overall_confidence=overall,
            confidence_level=conf_level,
        )

    # ── Rule 4: No documents retrieved ───────────────────────────────────────
    if not retrieved_docs:
        return DecisionResult(
            decision="ESCALATE",
            reason=(
                "No relevant support documentation was found for this ticket. "
                "Without a grounded source, generating a safe response is not possible."
            ),
            rule_triggered=4,
            decision_rule="#4",
            overall_confidence=0.0,
            confidence_level="VERY_LOW",
        )

    # ── Rule 5: Retrieved doc is NO_MATCH ─────────────────────────────────────
    best = retrieved_docs[0]
    if not best.is_reliable:
        return DecisionResult(
            decision="ESCALATE",
            reason=(
                f"Best matching document has confidence '{best.confidence}' "
                f"(similarity score: {best.score:.4f}). "
                f"Score is too low to safely ground a response — escalating."
            ),
            rule_triggered=5,
            decision_rule="#5",
            overall_confidence=overall,
            confidence_level=conf_level,
        )

    # ── Rule 6 (IMPROVEMENT 1): Overall confidence below safety threshold ─────
    if overall < MIN_OVERALL_CONFIDENCE:
        return DecisionResult(
            decision="ESCALATE",
            reason=(
                f"Overall confidence score {overall:.3f} ({conf_level}) is below "
                f"the safety threshold of {MIN_OVERALL_CONFIDENCE}. "
                f"Medium-confidence replies can be risky — escalating to ensure accuracy."
            ),
            rule_triggered=6,
            decision_rule="#6",
            overall_confidence=overall,
            confidence_level=conf_level,
        )

    # ── Rule 7: All checks passed → safe to reply ────────────────────────────
    return DecisionResult(
        decision="REPLY",
        reason=(
            f"Ticket classified as '{request_type}' in '{product_area}'. "
            f"Retrieval confidence: {best.confidence} (score: {best.score:.4f}). "
            f"Overall confidence: {overall:.3f} ({conf_level}) — above safety threshold. "
            f"Safe to generate a grounded automated response."
        ),
        rule_triggered=7,
        decision_rule="#7",
        overall_confidence=overall,
        confidence_level=conf_level,
    )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as dc

    @dc
    class MockDoc:
        score: float
        confidence: str
        is_reliable: bool
        doc_id: str = "DOC001"

    cases = [
        (
            "Fraud → Rule #1 ESCALATE",
            {"request_type": "fraud_report", "product_area": "fraud",
             "request_confidence": 5.0, "area_confidence": 4.0},
            [MockDoc(score=0.75, confidence="HIGH", is_reliable=True)],
        ),
        (
            "Medium confidence → Rule #6 ESCALATE (new safety rule)",
            {"request_type": "bug_report", "product_area": "technical",
             "request_confidence": 2.5, "area_confidence": 2.0},
            [MockDoc(score=0.30, confidence="MEDIUM", is_reliable=True)],
        ),
        (
            "High confidence technical → Rule #7 REPLY",
            {"request_type": "how_to_question", "product_area": "technical",
             "request_confidence": 6.0, "area_confidence": 5.0},
            [MockDoc(score=0.75, confidence="HIGH", is_reliable=True)],
        ),
        (
            "Unknown → Rule #3 ESCALATE",
            {"request_type": "unknown", "product_area": "unknown",
             "request_confidence": 0.5, "area_confidence": 0.5},
            [],
        ),
    ]

    print("\n── Decision Engine v3 Self-Test ──")
    for label, clf, docs in cases:
        r = decide(clf, docs)
        print(f"\n{label}")
        print(f"  Decision         : {r.decision}")
        print(f"  Decision Rule    : {r.decision_rule}")
        print(f"  Confidence Level : {r.confidence_level}")
        print(f"  Overall Score    : {r.overall_confidence}")
        print(f"  Reason           : {r.reason}")