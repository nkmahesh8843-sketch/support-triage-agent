# classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Classifies each support ticket into:
#   - request_type : what the customer wants (e.g. refund_request, bug_report)
#   - product_area : which domain it belongs to (e.g. billing, technical)
#
# Strategy: Keyword matching with confidence scoring.
# Each category has a weighted keyword list. The category with the highest
# cumulative score wins. If no category scores above a threshold → "unknown".
# ─────────────────────────────────────────────────────────────────────────────

from utils import clean_text

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD MAPS
# Format: { "category": [ ("keyword", weight), ... ] }
# Higher weight = stronger signal for that category.
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_TYPE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "refund_request": [
        ("refund", 3.0), ("money back", 3.0), ("reimburse", 2.0),
        ("charged twice", 2.5), ("overcharged", 2.5), ("cancel subscription", 1.5),
        ("get my money", 2.0), ("return payment", 2.0),
    ],
    "bug_report": [
        ("crash", 2.5), ("crashing", 2.5), ("error", 2.0), ("not working", 2.0),
        ("broken", 1.5), ("bug", 2.5), ("glitch", 2.0), ("issue", 1.0),
        ("error code", 2.5), ("fails", 1.5), ("failing", 1.5), ("freeze", 2.0),
        ("slow", 1.5), ("loading", 1.0), ("not loading", 2.0),
    ],
    "account_issue": [
        ("log in", 2.5), ("login", 2.5), ("cannot access", 2.5), ("locked out", 3.0),
        ("password", 2.0), ("account access", 3.0), ("sign in", 2.0),
        ("two factor", 2.0), ("2fa", 2.5), ("unauthorized access", 3.0),
        ("someone accessed", 3.0), ("delete account", 2.0),
    ],
    "fraud_report": [
        ("fraud", 4.0), ("fraudulent", 4.0), ("stolen", 3.0), ("unauthorized transaction", 4.0),
        ("did not make", 2.5), ("identity theft", 4.0), ("suspicious", 2.5),
        ("chargeback", 3.0), ("dispute", 2.0), ("credit card", 1.5),
    ],
    "payment_issue": [
        ("payment failed", 3.0), ("payment not working", 3.0), ("card declined", 3.0),
        ("transaction failed", 3.0), ("billing issue", 2.5), ("invoice", 2.0),
        ("charge", 1.5), ("debit", 1.5), ("payment method", 2.0),
    ],
    "how_to_question": [
        ("how do i", 3.0), ("how to", 2.5), ("how can i", 2.5), ("steps to", 2.0),
        ("guide", 1.5), ("tutorial", 1.5), ("help me", 1.0), ("instructions", 2.0),
        ("where do i", 2.0), ("what is", 1.0),
    ],
    "feature_request": [
        ("feature request", 4.0), ("new feature", 3.0), ("add support for", 3.0),
        ("would be great", 2.0), ("suggest", 2.0), ("request a feature", 3.0),
        ("wish", 1.5), ("can you add", 2.5), ("improvement", 1.5),
    ],
    "general_inquiry": [
        ("question", 1.5), ("inquiry", 1.5), ("information", 1.0), ("know about", 1.5),
        ("tell me", 1.0), ("what are", 1.0), ("plans", 1.0), ("pricing", 1.5),
    ],
}

PRODUCT_AREA_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "billing": [
        ("billing", 3.0), ("invoice", 2.5), ("subscription", 2.0), ("charged", 2.5),
        ("refund", 2.5), ("overcharged", 3.0), ("charged twice", 3.0),
        ("plan", 1.5), ("upgrade", 1.5), ("downgrade", 1.5), ("pricing", 2.0),
        ("receipt", 2.0), ("amount", 1.5),
    ],
    "payments": [
        ("payment", 3.0), ("card", 2.0), ("credit card", 2.5), ("debit", 2.5),
        ("transaction", 2.5), ("paypal", 2.5), ("payment method", 3.0),
        ("failed payment", 3.0), ("card declined", 3.0), ("chargeback", 3.0),
    ],
    "fraud": [
        ("fraud", 4.0), ("fraudulent", 4.0), ("unauthorized transaction", 4.0),
        ("identity theft", 4.0), ("stolen", 3.0), ("suspicious activity", 3.0),
        ("did not make", 2.5), ("dispute", 2.0),
    ],
    "account_access": [
        ("account", 2.0), ("login", 2.5), ("log in", 2.5), ("password", 2.5),
        ("locked out", 3.0), ("2fa", 3.0), ("two factor", 3.0), ("unauthorized access", 3.5),
        ("cannot access", 3.0), ("sign in", 2.5), ("delete account", 2.5),
        ("someone accessed", 3.5),
    ],
    "technical": [
        ("error", 2.0), ("crash", 2.5), ("bug", 2.5), ("slow", 2.0),
        ("not working", 2.0), ("api", 2.5), ("webhook", 2.5), ("integration", 2.0),
        ("error code", 3.0), ("sync", 2.0), ("mobile app", 2.5), ("performance", 2.0),
        ("notification", 1.5), ("export", 1.5), ("import", 1.5),
    ],
    "onboarding": [
        ("getting started", 3.0), ("new user", 2.5), ("setup", 2.0), ("onboard", 3.0),
        ("team member", 2.5), ("invite", 2.5), ("add user", 2.5), ("workspace", 2.0),
    ],
    "general": [
        ("contact", 1.5), ("support", 1.0), ("help", 1.0), ("question", 1.0),
        ("feature request", 2.0), ("feedback", 2.0), ("roadmap", 2.0),
    ],
}

# Minimum score to consider a classification confident
CONFIDENCE_THRESHOLD = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _score_text(text: str, keyword_map: dict[str, list[tuple[str, float]]]) -> tuple[str, float]:
    """
    Scores cleaned text against a keyword map.

    For each category, sums the weights of all matching keywords found in text.
    Returns the best (category, score) pair.

    Args:
        text:        Cleaned lowercase text to classify.
        keyword_map: Dict of category → list of (keyword, weight) pairs.

    Returns:
        Tuple of (best_category, best_score). Category is "unknown" if no match.
    """
    scores: dict[str, float] = {}

    for category, keywords in keyword_map.items():
        total_score = 0.0
        for keyword, weight in keywords:
            if keyword in text:
                total_score += weight
        if total_score > 0:
            scores[category] = total_score

    if not scores:
        return "unknown", 0.0

    best_category = max(scores, key=lambda k: scores[k])
    return best_category, scores[best_category]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def classify(issue_text: str) -> dict:
    """
    Classifies a support ticket text into request_type and product_area.

    Args:
        issue_text: Raw text of the customer's support ticket.

    Returns:
        A dict with keys:
            - request_type       : e.g. "bug_report", "refund_request", "unknown"
            - product_area       : e.g. "technical", "billing", "unknown"
            - request_confidence : float score for request_type classification
            - area_confidence    : float score for product_area classification
    """
    cleaned = clean_text(issue_text)

    request_type, req_confidence   = _score_text(cleaned, REQUEST_TYPE_KEYWORDS)
    product_area, area_confidence  = _score_text(cleaned, PRODUCT_AREA_KEYWORDS)

    return {
        "request_type":       request_type,
        "product_area":       product_area,
        "request_confidence": req_confidence,
        "area_confidence":    area_confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "I was charged twice for my subscription this month",
        "The app keeps crashing when I open the dashboard",
        "There are unauthorized transactions I did not make",
        "How do I export my data to CSV?",
        "I cannot log into my account, I think someone accessed it",
    ]
    print("\n── Classifier Self-Test ──")
    for text in samples:
        result = classify(text)
        print(f"\nTicket : {text}")
        print(f"  → request_type : {result['request_type']} (score: {result['request_confidence']:.1f})")
        print(f"  → product_area : {result['product_area']} (score: {result['area_confidence']:.1f})")