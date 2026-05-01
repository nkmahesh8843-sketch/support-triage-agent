# main.py  (v3 — Safer Logic + Traceability + Better Explanations)
# ─────────────────────────────────────────────────────────────────────────────
# Output CSV now includes:
#   decision_rule    → which rule fired (#1 – #7)
#   confidence_level → HIGH / MEDIUM / LOW / VERY_LOW
#   reason           → full human-readable policy explanation
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import os
import sys
import time

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from utils           import setup_logger, read_tickets, load_corpus
from classifier      import classify
from retriever       import Retriever
from decision_engine import decide
from responder       import generate_response


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent v3")
    parser.add_argument("--input",  default="data/support_issues.csv")
    parser.add_argument("--output", default="output/triage_results.csv")
    parser.add_argument("--corpus", default="data/support_corpus.json")
    parser.add_argument("--top-n",  type=int, default=2)
    return parser.parse_args()


def write_results(results: list[dict], output_path: str) -> None:
    """Writes results to CSV — includes all traceability columns."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "ticket_id",
        "request_type",
        "product_area",
        "decision",
        "decision_rule",       # IMPROVEMENT 3
        "confidence_level",    # IMPROVEMENT 3 + 4
        "overall_confidence",
        "reason",              # IMPROVEMENT 2
        "response",
        "retrieved_doc_id",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def process_ticket(ticket: dict, retriever: Retriever, top_n: int, logger) -> dict:
    """Full triage pipeline for one ticket."""
    ticket_id  = ticket["ticket_id"]
    issue_text = ticket["issue_text"]

    logger.debug(f"[{ticket_id}] Input: {issue_text[:80]}...")

    # 1. Classify
    clf          = classify(issue_text)
    request_type = clf["request_type"]
    product_area = clf["product_area"]

    # 2. Retrieve
    docs = retriever.retrieve_by_area(issue_text, product_area, top_n=top_n)
    doc_ids = ", ".join(d.doc_id for d in docs) if docs else "NONE"

    # 3. Decide (v3 — safer + traceable)
    dr = decide(clf, docs)

    logger.info(
        f"[{ticket_id}] {dr.decision:8s} | "
        f"Rule {dr.decision_rule} | "
        f"{dr.confidence_level} ({dr.overall_confidence:.3f}) | "
        f"{dr.reason}"
    )

    # 4. Respond
    response = generate_response(
        issue_text, dr.decision, product_area, docs, dr.confidence_level
    )

    return {
        "ticket_id":          ticket_id,
        "request_type":       request_type,
        "product_area":       product_area,
        "decision":           dr.decision,
        "decision_rule":      dr.decision_rule,        # IMPROVEMENT 3
        "confidence_level":   dr.confidence_level,     # IMPROVEMENT 3+4
        "overall_confidence": f"{dr.overall_confidence:.3f}",
        "reason":             dr.reason,               # IMPROVEMENT 2
        "response":           response,
        "retrieved_doc_id":   doc_ids,
    }


def main():
    args   = parse_args()
    logger = setup_logger()

    print("\n" + "═" * 62)
    print("   🤖  Support Triage Agent  v3  (Safety-First + Traceable)")
    print("═" * 62)
    print(f"   Input  : {args.input}")
    print(f"   Output : {args.output}")
    print("═" * 62 + "\n")

    # Load corpus
    try:
        corpus = load_corpus(args.corpus)
        logger.info(f"Corpus loaded: {len(corpus)} documents.")
    except FileNotFoundError as e:
        logger.error(str(e)); sys.exit(1)

    # Build retriever
    retriever = Retriever(corpus, logger=logger)

    # Load tickets
    try:
        tickets = read_tickets(args.input)
        logger.info(f"Loaded {len(tickets)} tickets.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e)); sys.exit(1)

    # Process all tickets
    results, reply_count, escalate_count = [], 0, 0
    start = time.time()

    for i, ticket in enumerate(tickets, 1):
        tid = ticket["ticket_id"]
        print(f"  [{i:02d}/{len(tickets):02d}] {tid}...", end=" ", flush=True)

        try:
            result = process_ticket(ticket, retriever, args.top_n, logger)
            results.append(result)

            d     = result["decision"]
            rule  = result["decision_rule"]
            level = result["confidence_level"]
            score = result["overall_confidence"]

            if d == "REPLY":
                print(f"✅ REPLY     | Rule {rule} | {level} ({score})")
                reply_count += 1
            else:
                print(f"⚠️  ESCALATE  | Rule {rule} | {level} ({score})")
                escalate_count += 1

        except Exception as e:
            logger.error(f"[{tid}] Unexpected error: {e}")
            results.append({
                "ticket_id": tid, "request_type": "unknown",
                "product_area": "unknown", "decision": "ESCALATE",
                "decision_rule": "#ERR", "confidence_level": "VERY_LOW",
                "overall_confidence": "0.000",
                "reason": f"Processing error: {e}. Escalated for safety.",
                "response": "We encountered an issue processing your request. A specialist will contact you shortly.",
                "retrieved_doc_id": "NONE",
            })
            escalate_count += 1
            print("❌ ERROR → ESCALATED")

    elapsed = time.time() - start

    # Write output
    try:
        write_results(results, args.output)
        logger.info(f"Results written to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output: {e}"); sys.exit(1)

    # Summary
    print("\n" + "═" * 62)
    print("   ✅  Complete")
    print("═" * 62)
    print(f"   Total tickets  : {len(results)}")
    print(f"   ✅ REPLY        : {reply_count}")
    print(f"   ⚠️  ESCALATE    : {escalate_count}")
    print(f"   ⏱  Time         : {elapsed:.2f}s")
    print(f"   📄 Output       : {args.output}")
    print(f"   📋 Logs         : logs/log.txt")
    print("═" * 62 + "\n")
    logger.info(
        f"Run complete — {len(results)} tickets | "
        f"{reply_count} REPLY | {escalate_count} ESCALATE | {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()