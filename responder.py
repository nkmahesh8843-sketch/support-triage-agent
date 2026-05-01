# responder.py  (v2 — Natural, Safe, Grounded Responses)
# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE: Responses are now:
#   ✅ Natural sounding (not robotic)
#   ✅ Grounded ONLY in retrieved docs (no hallucination)
#   ✅ Confidence-aware (HIGH vs MEDIUM vs LOW responses differ)
#   ✅ Empathetic opening lines
#   ✅ Area-specific escalation messages
# ─────────────────────────────────────────────────────────────────────────────

import random
from utils import clean_text

# ── Empathetic opening lines (varied so not robotic) ─────────────────────────
EMPATHY_OPENERS = [
    "Thanks for reaching out — happy to help with that.",
    "Thanks for contacting us — I've looked into this for you.",
    "Appreciate you getting in touch. Here's what I found:",
    "Thanks for your message. Let me help you with that.",
    "Got your message — here's the information you need:",
]

# ── Confidence-aware reply footers ───────────────────────────────────────────
FOOTER_HIGH = (
    "If this doesn't fully resolve your issue, reply and we'll dig deeper."
)
FOOTER_MEDIUM = (
    "If you need more specific help, reply with any additional details "
    "and our team will follow up."
)
FOOTER_LOW = (
    "This is our best available guidance, but for a more tailored answer "
    "please reply with more details or contact us directly."
)

# ── Area-specific escalation templates (natural, empathetic) ─────────────────
ESCALATION_TEMPLATES: dict[str, str] = {
    "billing": (
        "Thanks for getting in touch about your billing concern. "
        "Since billing issues require a closer look at your account details, "
        "I've flagged this for our billing team who will reach out within 1 business day. "
        "For anything urgent, you can email billing-support@company.com directly."
    ),
    "payments": (
        "Thanks for letting us know about this payment issue. "
        "To make sure it's handled accurately and securely, "
        "I've passed this to our payments team — they'll be in touch within 1 business day. "
        "Please don't retry the payment until you hear from them."
    ),
    "fraud": (
        "We take reports like this very seriously. "
        "I've immediately escalated this to our fraud investigation team, "
        "who will treat it as a high-priority case. "
        "Please avoid sharing any further account details until they contact you. "
        "Your security is our top priority."
    ),
    "account_access": (
        "Account security issues need careful handling — "
        "I've escalated this to our security team who will verify your identity "
        "and restore access safely. Expect a response within a few hours. "
        "If you suspect unauthorized access, use the Forgot Password option "
        "on the login page as an immediate precaution."
    ),
    "technical": (
        "It looks like your issue needs a closer look from our engineering team. "
        "I've escalated this and a specialist will follow up within 4 business hours. "
        "When they reach out, it'll help to have your device type, OS version, "
        "and any error codes or screenshots ready."
    ),
    "onboarding": (
        "Thanks for reaching out — your question has been passed to our "
        "onboarding team who can walk you through the setup in detail. "
        "They'll be in touch within 1 business day."
    ),
    "default": (
        "Thanks for getting in touch. Your request has been escalated to the "
        "right team and a specialist will contact you within 1 business day. "
        "We're sorry for any inconvenience and appreciate your patience."
    ),
}

MAX_SNIPPET_CHARS = 420


# ── Snippet extractor ─────────────────────────────────────────────────────────
def _extract_snippet(query: str, content: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    """
    Extracts the most query-relevant sentence(s) from document content.
    Scores sentences by keyword overlap with the query, then returns
    the top sentence plus the one after it for context.
    """
    stopwords = {
        "i", "my", "the", "a", "an", "is", "it", "to", "do", "how", "can",
        "me", "and", "or", "in", "on", "at", "for", "of", "with", "have",
        "was", "be", "this", "that", "are", "has", "not", "get",
    }
    query_words = set(clean_text(query).split()) - stopwords

    sentences = [s.strip() for s in content.replace(".\n", ". ").split(". ") if s.strip()]
    if not sentences:
        return content[:max_chars]

    scored = []
    for i, sentence in enumerate(sentences):
        words   = set(clean_text(sentence).split())
        overlap = len(query_words & words)
        scored.append((overlap, i, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_overlap, best_idx, best_sentence = scored[0]

    if best_overlap == 0:
        return content[:max_chars].strip()

    snippet = best_sentence
    if best_idx + 1 < len(sentences):
        combined = snippet + ". " + sentences[best_idx + 1]
        if len(combined) <= max_chars:
            snippet = combined

    return snippet[:max_chars].strip()


# ── Natural reply builder ─────────────────────────────────────────────────────
def _build_reply(
    issue_text: str,
    top_doc: object,        # RetrievalResult
    confidence_label: str,
) -> str:
    """
    Builds a natural, grounded reply using only the retrieved doc snippet.

    Structure:
      [Empathetic opener] + [Grounded doc content] + [Confidence-matched footer]
    """
    opener  = random.choice(EMPATHY_OPENERS)
    snippet = _extract_snippet(issue_text, top_doc.content)

    # Ensure snippet ends cleanly
    if snippet and snippet[-1] not in ".!?":
        snippet += "."

    # Pick footer based on how confident we are
    if confidence_label == "HIGH":
        footer = FOOTER_HIGH
    elif confidence_label == "MEDIUM":
        footer = FOOTER_MEDIUM
    else:
        footer = FOOTER_LOW

    # Compose final response — clean, readable, safe
    response = (
        f"{opener}\n\n"
        f"{snippet}\n\n"
        f"{footer}"
    )
    return response.strip()


# ── Main public function ──────────────────────────────────────────────────────
def generate_response(
    issue_text: str,
    decision: str,
    product_area: str,
    retrieved_docs: list,
    confidence_label: str = "LOW"
) -> str:
    """
    Generates the final customer-facing response.

    For ESCALATE → returns a warm, area-specific escalation message.
    For REPLY    → builds a natural, grounded response from top doc.

    Args:
        issue_text       : Original ticket text
        decision         : "REPLY" or "ESCALATE"
        product_area     : Classified product area (e.g. "technical")
        retrieved_docs   : List of RetrievalResult objects
        confidence_label : Overall confidence label from decision engine

    Returns:
        Final response string — safe to send to customer.
    """

    # ── Escalate path ─────────────────────────────────────────────────────────
    if decision == "ESCALATE":
        return ESCALATION_TEMPLATES.get(product_area, ESCALATION_TEMPLATES["default"])

    # ── Reply path ────────────────────────────────────────────────────────────
    if not retrieved_docs:
        return ESCALATION_TEMPLATES["default"]

    top_doc = retrieved_docs[0]
    if not top_doc.content:
        return ESCALATION_TEMPLATES["default"]

    return _build_reply(issue_text, top_doc, confidence_label)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class MockDoc:
        doc_id:  str
        title:   str
        content: str
        score:   float
        confidence: str
        is_reliable: bool
        product_area: str = "technical"

    doc = MockDoc(
        doc_id="DOC010", title="App Crashes and Error Codes",
        content=(
            "If the application crashes or shows an error code, first try: "
            "1) Clear app cache and cookies. 2) Restart the application. "
            "3) Check system status at status.company.com. "
            "Common error codes: ERR_002 means authentication failure — try re-logging in."
        ),
        score=0.72, confidence="HIGH", is_reliable=True,
    )

    print("\n── Responder v2 Self-Test ──")
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        print(f"\n[{conf} confidence REPLY]")
        r = generate_response(
            "I keep getting error code ERR_002",
            "REPLY", "technical", [doc], conf
        )
        print(r)

    print("\n[ESCALATE — fraud]")
    print(generate_response("unauthorized transactions", "ESCALATE", "fraud", []))