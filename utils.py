# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities used across all modules.
# Handles: CSV I/O, logging configuration, and reusable helper functions.
# ─────────────────────────────────────────────────────────────────────────────

import csv
import json
import logging
import os
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str = "logs/log.txt") -> logging.Logger:
    """
    Creates a logger that writes to both terminal AND logs/log.txt.
    Each line includes timestamp, level, and message.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("TriageAgent")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # File handler — writes everything (DEBUG+) to log.txt
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Console handler — shows INFO+ in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info(f"NEW SESSION STARTED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CSV READER
# ─────────────────────────────────────────────────────────────────────────────

def read_tickets(csv_path: str) -> list[dict]:
    """
    Reads input CSV into a list of dicts.
    Required columns: ticket_id, issue_text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    tickets = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"ticket_id", "issue_text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV must have columns: {required}. Found: {set(reader.fieldnames or [])}")
        for row in reader:
            tickets.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})

    return tickets


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CSV WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_results(results: list[dict], output_path: str = "output/triage_results.csv") -> None:
    """
    Writes triage results to output CSV.
    Columns: ticket_id, request_type, product_area, decision, response, retrieved_doc_id
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["ticket_id", "request_type", "product_area", "decision", "response", "retrieved_doc_id"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CORPUS LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(corpus_path: str = "data/support_corpus.json") -> list[dict]:
    """
    Loads the support knowledge base from JSON.
    Expected fields per doc: doc_id, product_area, title, content
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEXT CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace in a string."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())