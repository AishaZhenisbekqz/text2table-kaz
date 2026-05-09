#!/usr/bin/env python3
"""
Evaluation script on the gold standard (n=1000).
Reports all 5 metrics with 95% bootstrap confidence intervals.

Usage:
    python scripts/evaluate.py \
        --adapter_path ./checkpoints/dynamic_best \
        --test_data ./assets/sample_data.json \
        --regime dynamic \
        --use_llm_judge
"""

import argparse
import json
import sys
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Text2TablePipeline
from src.evaluation.metrics import MetricSuite
from src.evaluation.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_ci(scores: List[float], B: int = 10000, alpha: float = 0.05):
    """Compute bootstrap 95% CI (percentile method). Efron & Tibshirani, 1993."""
    arr = np.array(scores)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(B)]
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(arr)), float(lower), float(upper)


def print_results_table(results: Dict):
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Gold Standard)")
    print("="*60)
    print(f"{'Metric':<22} {'Mean':>8} {'95% CI':>20}")
    print("-"*60)
    for metric, vals in results.items():
        if isinstance(vals, dict):
            print(f"{metric:<22} {vals['mean']:>8.4f}  [{vals['lower']:.4f}, {vals['upper']:.4f}]")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--regime", default="dynamic", choices=["static", "dynamic"])
    parser.add_argument("--use_llm_judge", action="store_true")
    parser.add_argument("--bootstrap_b", type=int, default=10000)
    parser.add_argument("--output", default="evaluation_results.json")
    args = parser.parse_args()

    # Load test data
    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Loaded {len(test_data)} test examples (regime={args.regime})")

    # Init pipeline
    pipeline = Text2TablePipeline.from_pretrained(
        base_model=args.base_model,
        lora_adapter=args.adapter_path,
        regime=args.regime,
    )

    metric_suite = MetricSuite()
    judge = LLMJudge() if args.use_llm_judge else None

    # Collect per-sample scores
    all_scores = {m: [] for m in ["coverage", "accuracy", "compression", "structure", "journalistic_value"]}

    for i, example in enumerate(test_data):
        if i % 50 == 0:
            logger.info(f"  Processing {i}/{len(test_data)}...")

        generated = pipeline(example["text"])
        result = metric_suite.evaluate(
            generated=generated,
            reference=example["table"],
            source_text=example["text"],
        )

        if judge:
            judge_scores = judge.evaluate(example["text"], example["table"], generated)
            result.journalistic_value = judge_scores.get("journalistic_value", 0.0)

        for m in all_scores:
            all_scores[m].append(getattr(result, m))

    # Bootstrap CIs
    final_results = {}
    for metric, scores in all_scores.items():
        mean, lower, upper = bootstrap_ci(scores, B=args.bootstrap_b)
        final_results[metric] = {"mean": mean, "lower": lower, "upper": upper}

    print_results_table(final_results)

    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
