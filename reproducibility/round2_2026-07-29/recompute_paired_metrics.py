#!/usr/bin/env python3
"""Recompute the CMES paired metric matrix from record-level outputs.

The five metric implementations are imported directly from the specified
Text2Table_Kaz repository. Statistical inference is implemented here because
the repository's current paired_bootstrap_test does not center the bootstrap
distribution under H0 and therefore is not suitable for a two-sided p-value.

Required inputs
---------------
benchmark:
    JSON or JSONL records with: id, text, table

static_predictions / dynamic_predictions:
    JSON or JSONL records with: id and generated_table (or generated/table).
    Journalistic Value must be supplied either as:
      - judge_runs: exactly three objects containing journalistic_value; or
      - journalistic_value: a precomputed mean, with --allow-precomputed-jv.

The script refuses duplicate IDs, missing pairs, missing gold tables, non-finite
scores, and an evaluation set whose size differs from --expected-n.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = (
    "coverage",
    "accuracy",
    "compression",
    "structure",
    "journalistic_value",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else payload.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path}: expected a JSON array or JSONL object records")
    return records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(records: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(records, 1):
        if "id" not in row:
            raise ValueError(f"{path}: record {position} has no id")
        record_id = str(row["id"])
        if record_id in indexed:
            raise ValueError(f"{path}: duplicate id {record_id!r}")
        indexed[record_id] = row
    return indexed


def generated_table(row: dict[str, Any], label: str, record_id: str) -> str:
    for key in ("generated_table", "generated", "table"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"{label}: id {record_id!r} has no generated table")


def journalistic_value(
    row: dict[str, Any],
    label: str,
    record_id: str,
    allow_precomputed: bool,
) -> tuple[float, list[float]]:
    runs = row.get("judge_runs")
    if isinstance(runs, list):
        if len(runs) != 3:
            raise ValueError(f"{label}: id {record_id!r} must have exactly 3 judge_runs")
        values = []
        for run_no, run in enumerate(runs, 1):
            if not isinstance(run, dict) or "journalistic_value" not in run:
                raise ValueError(
                    f"{label}: id {record_id!r}, judge run {run_no} lacks journalistic_value"
                )
            values.append(float(run["journalistic_value"]))
        return sum(values) / 3, values
    if allow_precomputed and "journalistic_value" in row:
        value = float(row["journalistic_value"])
        return value, [value]
    raise ValueError(
        f"{label}: id {record_id!r} lacks three raw judge runs; "
        "use --allow-precomputed-jv only when the raw logs were verified separately"
    )


def validate_score(value: float, label: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}: expected a finite score in [0, 1], got {value}")
    return value


def percentile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: list[float],
    b: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(b):
        sample_mean = sum(values[rng.randrange(n)] for _ in range(n)) / n
        means.append(sample_mean)
    means.sort()
    return sum(values) / n, percentile(means, 0.025), percentile(means, 0.975)


def paired_bootstrap(
    static: list[float],
    dynamic: list[float],
    b: int,
    seed: int,
) -> dict[str, float]:
    differences = [dynamic_value - static_value for static_value, dynamic_value in zip(static, dynamic)]
    n = len(differences)
    observed = sum(differences) / n

    rng_ci = random.Random(seed)
    boot_differences = []
    for _ in range(b):
        sample_mean = sum(differences[rng_ci.randrange(n)] for _ in range(n)) / n
        boot_differences.append(sample_mean)
    boot_differences.sort()

    centered = [value - observed for value in differences]
    rng_null = random.Random(seed)
    null_means = []
    for _ in range(b):
        sample_mean = sum(centered[rng_null.randrange(n)] for _ in range(n)) / n
        null_means.append(sample_mean)
    extreme = sum(abs(value) >= abs(observed) for value in null_means)

    return {
        "difference_dynamic_minus_static": observed,
        "difference_ci_lower": percentile(boot_differences, 0.025),
        "difference_ci_upper": percentile(boot_differences, 0.975),
        "p_value_raw": (extreme + 1) / (b + 1),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    total = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--static-predictions", type=Path, required=True)
    parser.add_argument("--dynamic-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--bootstrap-b", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-precomputed-jv", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_dir.resolve()))
    from src.evaluation.metrics import MetricSuite

    benchmark = index_unique(load_records(args.benchmark), args.benchmark)
    static = index_unique(load_records(args.static_predictions), args.static_predictions)
    dynamic = index_unique(load_records(args.dynamic_predictions), args.dynamic_predictions)

    id_sets = (set(benchmark), set(static), set(dynamic))
    if not id_sets[0] == id_sets[1] == id_sets[2]:
        raise ValueError(
            "Benchmark and prediction IDs differ: "
            f"benchmark={len(id_sets[0])}, static={len(id_sets[1])}, dynamic={len(id_sets[2])}"
        )
    if len(benchmark) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} paired records, found {len(benchmark)}")

    suite = MetricSuite()
    matrix_rows = []
    vectors = {
        regime: {metric: [] for metric in METRICS}
        for regime in ("static", "dynamic")
    }

    for record_id in sorted(benchmark, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
        gold = benchmark[record_id]
        source_text = gold.get("text")
        reference_table = gold.get("table")
        if not isinstance(source_text, str) or not source_text.strip():
            raise ValueError(f"benchmark: id {record_id!r} has no source text")
        if not isinstance(reference_table, str) or not reference_table.strip():
            raise ValueError(f"benchmark: id {record_id!r} has no gold reference table")

        row_out: dict[str, Any] = {"id": record_id}
        for regime, prediction_map in (("static", static), ("dynamic", dynamic)):
            prediction = prediction_map[record_id]
            generated = generated_table(prediction, regime, record_id)
            jv, jv_runs = journalistic_value(
                prediction,
                regime,
                record_id,
                args.allow_precomputed_jv,
            )
            result = suite.evaluate(
                generated=generated,
                reference=reference_table,
                source_text=source_text,
                journalistic_value=jv,
            )
            values = result.to_dict()
            for metric in METRICS:
                score = validate_score(float(values[metric]), f"{regime}/{record_id}/{metric}")
                vectors[regime][metric].append(score)
                row_out[f"{regime}_{metric}"] = score
            row_out[f"{regime}_jv_run_1"] = jv_runs[0]
            row_out[f"{regime}_jv_run_2"] = jv_runs[1] if len(jv_runs) == 3 else ""
            row_out[f"{regime}_jv_run_3"] = jv_runs[2] if len(jv_runs) == 3 else ""
        for metric in METRICS:
            row_out[f"difference_{metric}"] = (
                row_out[f"dynamic_{metric}"] - row_out[f"static_{metric}"]
            )
        matrix_rows.append(row_out)

    results: dict[str, dict[str, float]] = {}
    raw_p_values: dict[str, float] = {}
    for metric in METRICS:
        metric_seed = args.seed
        static_mean, static_lo, static_hi = bootstrap_mean_ci(
            vectors["static"][metric],
            args.bootstrap_b,
            metric_seed,
        )
        dynamic_mean, dynamic_lo, dynamic_hi = bootstrap_mean_ci(
            vectors["dynamic"][metric],
            args.bootstrap_b,
            metric_seed,
        )
        paired = paired_bootstrap(
            vectors["static"][metric],
            vectors["dynamic"][metric],
            args.bootstrap_b,
            metric_seed,
        )
        results[metric] = {
            "static_mean": static_mean,
            "static_ci_lower": static_lo,
            "static_ci_upper": static_hi,
            "dynamic_mean": dynamic_mean,
            "dynamic_ci_lower": dynamic_lo,
            "dynamic_ci_upper": dynamic_hi,
            **paired,
        }
        raw_p_values[metric] = paired["p_value_raw"]

    adjusted = holm_adjust(raw_p_values)
    for metric in METRICS:
        results[metric]["p_value_holm"] = adjusted[metric]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "paired_metric_matrix.csv"
    with matrix_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_rows[0].keys())
        writer.writeheader()
        writer.writerows(matrix_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "recomputed_from_record_level_outputs",
        "n_pairs": len(matrix_rows),
        "bootstrap_iterations": args.bootstrap_b,
        "base_seed": args.seed,
        "confidence_interval": "percentile bootstrap, 95%",
        "test": (
            "two-sided paired bootstrap on Dynamic-Static differences; "
            "differences centered under H0; add-one p-value"
        ),
        "multiple_testing": "Holm correction across five metrics",
        "metric_source_repo": str(args.repo_dir.resolve()),
        "metric_source_commit": git_commit(args.repo_dir),
        "inputs": {
            "benchmark": {
                "path": str(args.benchmark.resolve()),
                "sha256": file_sha256(args.benchmark),
            },
            "static_predictions": {
                "path": str(args.static_predictions.resolve()),
                "sha256": file_sha256(args.static_predictions),
            },
            "dynamic_predictions": {
                "path": str(args.dynamic_predictions.resolve()),
                "sha256": file_sha256(args.dynamic_predictions),
            },
        },
        "results": results,
    }
    result_path = args.output_dir / "paired_statistics.json"
    result_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(matrix_path)
    print(result_path)


if __name__ == "__main__":
    main()
