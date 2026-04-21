"""
Statistics for Claude Code LoCoMo QA judge results.

Reports accuracy by category and token/cost/latency statistics.

Usage:
    python stat_judge_result.py --input ./result/qa_results.csv
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Statistics for Claude Code QA results")
    parser.add_argument(
        "--input", default="./result/qa_results.csv",
        help="Path to judge result CSV",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    lines = process_qa_results(args.input)
    for line in lines:
        print(line)

    summary_path = os.path.join(os.path.dirname(args.input), "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


def process_qa_results(input_path: str) -> list[str]:
    correct = 0
    wrong = 0
    total_input_tokens = 0
    total_cache_creation = 0
    total_cache_read = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_cost = 0.0
    total_elapsed = 0.0
    total_turns = 0
    min_elapsed = None
    max_elapsed = None
    elapsed_rows = 0
    valid_rows = 0

    by_category: dict[str, dict] = {}

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "")
            if category == "5":
                continue

            valid_rows += 1

            result = row.get("result", "").strip().upper()
            if result == "CORRECT":
                correct += 1
            elif result == "WRONG":
                wrong += 1

            # Per-category stats
            if category not in by_category:
                by_category[category] = {"correct": 0, "wrong": 0, "total": 0}
            by_category[category]["total"] += 1
            if result == "CORRECT":
                by_category[category]["correct"] += 1
            elif result == "WRONG":
                by_category[category]["wrong"] += 1

            try:
                total_input_tokens += int(row.get("input_tokens", 0))
                total_cache_creation += int(row.get("cache_creation_input_tokens", 0))
                total_cache_read += int(row.get("cache_read_input_tokens", 0))
                total_output_tokens += int(row.get("output_tokens", 0))
                total_reasoning_tokens += int(row.get("reasoning_tokens", 0))
            except (ValueError, TypeError):
                pass

            try:
                total_cost += float(row.get("total_cost_usd", 0))
            except (ValueError, TypeError):
                pass

            try:
                total_turns += int(row.get("num_turns", 0))
            except (ValueError, TypeError):
                pass

            try:
                elapsed_raw = row.get("elapsed_seconds")
                if elapsed_raw is not None and str(elapsed_raw).strip():
                    elapsed = float(elapsed_raw)
                    total_elapsed += elapsed
                    min_elapsed = elapsed if min_elapsed is None else min(min_elapsed, elapsed)
                    max_elapsed = elapsed if max_elapsed is None else max(max_elapsed, elapsed)
                    elapsed_rows += 1
            except (ValueError, TypeError):
                pass

    total_graded = correct + wrong
    accuracy = correct / total_graded if total_graded > 0 else 0.0

    avg_input = total_input_tokens / valid_rows if valid_rows > 0 else 0.0
    avg_cache_creation = total_cache_creation / valid_rows if valid_rows > 0 else 0.0
    avg_cache_read = total_cache_read / valid_rows if valid_rows > 0 else 0.0
    avg_output = total_output_tokens / valid_rows if valid_rows > 0 else 0.0
    avg_reasoning = total_reasoning_tokens / valid_rows if valid_rows > 0 else 0.0
    avg_elapsed = total_elapsed / elapsed_rows if elapsed_rows > 0 else 0.0
    avg_turns = total_turns / valid_rows if valid_rows > 0 else 0.0
    avg_cost = total_cost / valid_rows if valid_rows > 0 else 0.0

    category_names = {
        "1": "single-hop",
        "2": "multi-hop",
        "3": "temporal",
        "4": "world-knowledge",
    }

    output = [
        "=== Claude Code LoCoMo QA Statistics (excluding category=5) ===",
        f"Total rows: {valid_rows:,}",
        f"Graded rows: {total_graded:,}",
        f"Correct: {correct:,}",
        f"Wrong: {wrong:,}",
        f"Accuracy: {accuracy:.2%}",
    ]

    if by_category:
        output.append("\nAccuracy by category:")
        for cat in sorted(by_category.keys()):
            stats = by_category[cat]
            cat_graded = stats["correct"] + stats["wrong"]
            cat_acc = stats["correct"] / cat_graded if cat_graded > 0 else 0.0
            name = category_names.get(cat, f"cat-{cat}")
            output.append(
                f"  {name:20s}: {stats['correct']}/{cat_graded} = {cat_acc:.2%}"
                f"  (total: {stats['total']})"
            )

    output.extend([
        f"\nLatency:",
        f"  Total elapsed: {total_elapsed:,.1f}s",
        f"  Avg elapsed: {avg_elapsed:,.1f}s",
        f"  Min elapsed: {(min_elapsed or 0.0):,.1f}s",
        f"  Max elapsed: {(max_elapsed or 0.0):,.1f}s",
        f"\nToken usage:",
        f"  Total input tokens: {total_input_tokens:,}",
        f"  Total cache creation: {total_cache_creation:,}",
        f"  Total cache read: {total_cache_read:,}",
        f"  Total output tokens: {total_output_tokens:,}",
        f"  Total reasoning tokens: {total_reasoning_tokens:,}",
        f"  Avg input tokens: {avg_input:,.0f}",
        f"  Avg cache creation: {avg_cache_creation:,.0f}",
        f"  Avg cache read: {avg_cache_read:,.0f}",
        f"  Avg output tokens: {avg_output:,.0f}",
        f"  Avg reasoning tokens: {avg_reasoning:,.0f}",
        f"\nCost:",
        f"  Total cost: ${total_cost:,.4f}",
        f"  Avg cost per question: ${avg_cost:,.4f}",
        f"\nTurns:",
        f"  Total turns: {total_turns:,}",
        f"  Avg turns per question: {avg_turns:,.1f}",
    ])

    return output


if __name__ == "__main__":
    main()
