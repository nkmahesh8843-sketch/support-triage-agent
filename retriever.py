# retriever.py  (v2 — Enhanced TF-IDF + Confidence Scoring)
# ─────────────────────────────────────────────────────────────────────────────
# Uses advanced TF-IDF with:
#   - bigrams (1,2) for phrase matching
#   - query expansion (synonyms for common support terms)
#   - area boosting (docs in same area get a score boost)
#   - structured RetrievalResult with HIGH/MEDIUM/LOW/NO_MATCH confidence
#
# NOTE: Designed to auto-upgrade to sentence-transformers when internet
#       is available — see EMBEDDING_UPGRADE_NOTE at bottom of file.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, load_corpus

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.28
CONFIDENCE_MEDIUM = 0.14
CONFIDENCE_LOW    = 0.06

# ── Query synonym expansion ───────────────────────────────────────────────────
# Maps common user words → richer query terms for better matching
SYNONYMS: dict[str, list[str]] = {
    "crash":       ["crash", "crashing", "error", "broken", "not working", "fails"],
    "slow":        ["slow", "performance", "loading", "lagging", "timeout"],
    "login":       ["login", "log in", "sign in", "access", "password", "locked"],
    "charge":      ["charge", "charged", "billing", "invoice", "payment", "debit"],
    "fraud":       ["fraud", "unauthorized", "suspicious", "stolen", "identity"],
    "refund":      ["refund", "money back", "reimburse", "cancel", "dispute"],
    "export":      ["export", "download", "extract", "data", "csv"],
    "api":         ["api", "integration", "webhook", "endpoint", "developer"],
    "mobile":      ["mobile", "phone", "app", "android", "ios"],
    "team":        ["team", "member", "invite", "user", "colleague", "workspace"],
    "notification":["notification", "email", "alert", "message", "notify"],
}


# ── RetrievalResult dataclass ─────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """
    A retrieved document with full confidence metadata.

    score       : cosine similarity (0.0–1.0)
    confidence  : HIGH / MEDIUM / LOW / NO_MATCH
    is_reliable : True if score >= CONFIDENCE_LOW
    """
    doc_id:       str
    title:        str
    content:      str
    product_area: str
    score:        float
    confidence:   str
    is_reliable:  bool

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title,
            "content": self.content, "product_area": self.product_area,
            "score": self.score, "confidence": self.confidence,
            "is_reliable": self.is_reliable,
        }


def _score_to_confidence(score: float) -> tuple[str, bool]:
    if score >= CONFIDENCE_HIGH:     return "HIGH",     True
    elif score >= CONFIDENCE_MEDIUM: return "MEDIUM",   True
    elif score >= CONFIDENCE_LOW:    return "LOW",      True
    else:                            return "NO_MATCH", False


def _expand_query(query: str) -> str:
    """Expands query with synonyms to improve recall."""
    words  = set(clean_text(query).split())
    extras = []
    for key, synonyms in SYNONYMS.items():
        if key in words or any(w in words for w in synonyms[:2]):
            extras.extend(synonyms)
    if extras:
        return query + " " + " ".join(extras)
    return query


# ── Retriever class ───────────────────────────────────────────────────────────
class Retriever:
    """
    Enhanced TF-IDF retriever with:
      - bigram matching
      - query expansion via synonyms
      - area boosting
      - confidence scoring (HIGH/MEDIUM/LOW/NO_MATCH)
    """

    def __init__(self, corpus: list[dict], logger=None):
        self.corpus = corpus
        self.logger = logger
        self._log("Building enhanced TF-IDF index...")

        # Combine title (×2 for emphasis) + content for each doc
        self.doc_texts = [
            clean_text(doc["title"] + " " + doc["title"] + " " + doc["content"])
            for doc in corpus
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,    # dampens very frequent terms
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)
        self._log(f"Index ready — {len(corpus)} docs, vocab={len(self.vectorizer.vocabulary_)} terms.")

    def _vectorize_query(self, query: str) -> object:
        """Expand then vectorize a query."""
        expanded = _expand_query(clean_text(query))
        return self.vectorizer.transform([expanded])

    def retrieve(self, query: str, top_n: int = 2) -> list[RetrievalResult]:
        """Find top-N relevant documents for a query."""
        if not clean_text(query):
            return []
        q_vec  = self._vectorize_query(query)
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            conf, reliable = _score_to_confidence(score)
            doc = self.corpus[idx]
            results.append(RetrievalResult(
                doc_id=doc["doc_id"], title=doc["title"],
                content=doc["content"], product_area=doc["product_area"],
                score=round(score, 4), confidence=conf, is_reliable=reliable,
            ))
        return results

    def retrieve_by_area(self, query: str, product_area: str, top_n: int = 2) -> list[RetrievalResult]:
        """
        Retrieve with area boosting:
          - Score all docs normally
          - Apply 1.25x boost to docs in the same product area
          - Return top-N after boosting
        """
        if not clean_text(query):
            return []
        q_vec  = self._vectorize_query(query)
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten().copy()

        # Boost scores for docs in the same product area
        for i, doc in enumerate(self.corpus):
            if doc.get("product_area") == product_area:
                scores[i] *= 1.25

        top_idx = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_idx:
            score = float(scores[idx])
            # Un-boost score before reporting (report true similarity)
            true_score = score / 1.25 if self.corpus[idx].get("product_area") == product_area else score
            conf, reliable = _score_to_confidence(true_score)
            doc = self.corpus[idx]
            results.append(RetrievalResult(
                doc_id=doc["doc_id"], title=doc["title"],
                content=doc["content"], product_area=doc["product_area"],
                score=round(true_score, 4), confidence=conf, is_reliable=reliable,
            ))
        return results

    def _log(self, msg: str):
        if self.logger: self.logger.info(msg)
        else: print(f"  [Retriever] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING_UPGRADE_NOTE
# When running with internet access, replace the Retriever class with
# the sentence-transformers version in retriever_embeddings.py (included
# in the project). It's a drop-in replacement with identical API:
#   from retriever_embeddings import Retriever
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    corpus    = load_corpus("data/support_corpus.json")
    retriever = Retriever(corpus)
    tests = [
        ("I was billed twice for my subscription", "billing"),
        ("application crashes on my phone every time", "technical"),
        ("someone made transactions without my permission", "fraud"),
        ("how do I get my data out as a spreadsheet file", "technical"),
        ("cannot sign in forgot my password", "account_access"),
    ]
    print("\n── Retriever v2 Self-Test ──")
    for query, area in tests:
        print(f"\nQuery: {query}")
        for r in retriever.retrieve_by_area(query, area, top_n=2):
            print(f"  [{r.confidence:8s}] [{r.doc_id}] {r.title} (score: {r.score:.4f})")