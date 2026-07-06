"""
Hiver AI Challenge — End-to-End Pipeline
1. Load dataset
2. Generate AI replies using RAG (retrieves similar examples from dataset)
3. Evaluate each reply against expected (human) reply
4. Print per-example scores + aggregate report
5. Save full results to results.json

Usage:
    python run.py              # Run on full dataset (520 examples)
    python run.py --limit 10   # Run on first 10 examples (quick test)
    python run.py --model llama-3.1-8b-instant  # Use faster model (higher rate limits)
"""

import json
import argparse
import time
from datetime import datetime
from generator import generate_reply
from evaluator import evaluate_single, evaluate_batch


def load_dataset(path: str = "dataset.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(dataset: list[dict], model: str, limit: int = None) -> dict:
    """
    Run the full generate + evaluate pipeline.
    Returns a dict with all results and aggregate stats.
    """
    if limit:
        dataset = dataset[:limit]

    total = len(dataset)
    results = []

    print(f"\n{'='*60}")
    print(f"  Hiver AI Challenge — Email Reply Generator & Evaluator")
    print(f"  Model: {model}")
    print(f"  Dataset size: {total} examples")
    print(f"  Approach: RAG (retrieves similar emails + replies from dataset)")
    print(f"{'='*60}\n")

    start_time = time.time()

    for i, item in enumerate(dataset, 1):
        email = item["email"]
        expected = item["expected_reply"]

        print(f"[{i}/{total}] Generating reply...")
        generated = generate_reply(email, model=model)

        # Rate limit safety: Groq free tier
        # 70B: 30 RPM, 1,000/day | 8B: 30 RPM, 14,400/day
        # Each example needs 2 API calls (generate + judge)
        if i % 15 == 0:
            print(f"[{i}/{total}] Rate limit safety pause (2s)...")
            time.sleep(2)

        print(f"[{i}/{total}] Evaluating...")
        eval_result = evaluate_single(generated, expected, email)

        results.append({
            "email": email,
            "expected_reply": expected,
            "generated_reply": generated,
            "evaluation": eval_result
        })

        print(f"  → Semantic: {eval_result['semantic_similarity']:.1f} | "
              f"Style: {eval_result['style_match']:.1f} | "
              f"LLM Overall: {eval_result['llm_judge']['overall']:.1f} | "
              f"Final: {eval_result['final_score']:.1f}")
        print(f"  → Reasoning: {eval_result['llm_judge']['reasoning'][:100]}...")
        print()

    elapsed = time.time() - start_time

    # Aggregate stats
    aggregate = evaluate_batch([r["evaluation"] for r in results])

    return {
        "metadata": {
            "model": model,
            "dataset_size": total,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "approach": "RAG (Retrieval-Augmented Generation with few-shot prompting + style matching)"
        },
        "aggregate": aggregate,
        "per_example": results
    }


def print_report(report: dict):
    """Print a formatted summary report."""
    agg = report["aggregate"]
    meta = report["metadata"]

    print(f"\n{'='*60}")
    print(f"  FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Model: {meta['model']}")
    print(f"  Approach: {meta['approach']}")
    print(f"  Examples evaluated: {agg['count']}")
    print(f"  Time taken: {meta['elapsed_seconds']:.1f}s")
    print(f"{'='*60}")

    print(f"\n  📊 SEMANTIC SIMILARITY (BERTScore)")
    print(f"     Mean:  {agg['semantic_similarity']['mean']:.2f}")
    print(f"     Std:   {agg['semantic_similarity']['std']:.2f}")
    print(f"     Range: {agg['semantic_similarity']['min']:.2f} - {agg['semantic_similarity']['max']:.2f}")

    print(f"\n  🎨 STYLE MATCH (Action phrases + Opening similarity)")
    print(f"     Mean:  {agg['style_match']['mean']:.2f}")
    print(f"     Std:   {agg['style_match']['std']:.2f}")

    print(f"\n  🔗 COMBINED SEMANTIC (70% BERT + 30% Style)")
    print(f"     Mean:  {agg['combined_semantic']['mean']:.2f}")
    print(f"     Std:   {agg['combined_semantic']['std']:.2f}")

    print(f"\n  🤖 LLM JUDGE BREAKDOWN")
    for dim in ["tone", "actionability", "completeness", "empathy", "overall"]:
        mean = agg['llm_judge'][dim]['mean']
        std = agg['llm_judge'][dim]['std']
        print(f"     {dim.capitalize():15s} {mean:6.2f} ± {std:.2f}")

    print(f"\n  🏆 FINAL SCORE (25% Combined Semantic + 75% LLM Overall)")
    print(f"     Mean:  {agg['final_score']['mean']:.2f}")
    print(f"     Std:   {agg['final_score']['std']:.2f}")
    print(f"     Range: {agg['final_score']['min']:.2f} - {agg['final_score']['max']:.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Hiver AI Challenge — Email Reply Evaluator")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N examples (for quick testing)")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", 
                        help="Groq model to use (default: llama-3.3-70b-versatile)")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    args = parser.parse_args()

    dataset = load_dataset()
    report = run_pipeline(dataset, model=args.model, limit=args.limit)
    print_report(report)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Full results saved to {args.output}")


if __name__ == "__main__":
    main()