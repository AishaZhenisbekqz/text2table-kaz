#!/usr/bin/env python3
"""Generate Static or Dynamic predictions on the confirmed 1,000-record Gold Standard.

The script supports two explicitly selected protocols:

1. direct:
   Direct chat-template generation with the regime-specific training prompts.
2. pipeline:
   The public repository's seven-stage Text2TablePipeline, including
   self-consistency. The script explicitly propagates the selected regime to
   InsightGenerator because the current repository constructor does not do so.

The protocol is mandatory and is written into every record and the run
manifest. Run the script separately for Static and Dynamic models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}: invalid record at line {line_no}")
            records.append(row)
    if len({str(row["id"]) for row in records}) != len(records):
        raise ValueError(f"{path}: duplicate IDs")
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_direct_model(
    model_path: str,
    adapter_path: str | None,
    local_files_only: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=local_files_only,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=local_files_only,
        )
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def direct_generate(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    self_consistency_m: int,
    run_seed: int,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    completions = []
    for run_index in range(self_consistency_m):
        set_seed(run_seed + run_index)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
        with torch.no_grad():
            output = model.generate(input_ids, **kwargs)
        generated = output[0][input_ids.shape[1] :]
        completions.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return Counter(completions).most_common(1)[0][0]


def load_progress(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[str(row["id"])] = row
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--regime", choices=("static", "dynamic"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--generation-mode", choices=("direct", "pipeline"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--self-consistency-m", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.72)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.self_consistency_m < 1:
        raise ValueError("--self-consistency-m must be at least 1")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("Sampling temperature must be positive")

    benchmark = load_jsonl(args.benchmark)
    if len(benchmark) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} Gold records, found {len(benchmark)}")
    selected = benchmark[: args.limit] if args.limit else benchmark

    sys.path.insert(0, str(args.repo_dir.resolve()))
    set_seed(args.seed)

    if args.generation_mode == "direct":
        from src.training.dataset import DYNAMIC_SYSTEM, STATIC_SYSTEM, USER_TEMPLATE

        model, tokenizer = load_direct_model(
            args.model_path,
            args.adapter_path,
            args.local_files_only,
        )
        system_prompt = STATIC_SYSTEM if args.regime == "static" else DYNAMIC_SYSTEM
        pipeline = None
    else:
        from src.pipeline import Text2TablePipeline

        pipeline = Text2TablePipeline.from_pretrained(
            base_model=args.model_path,
            lora_adapter=args.adapter_path,
            regime=args.regime,
            self_consistency_m=args.self_consistency_m,
            theta=args.theta,
        )
        # Required correction: repository InsightGenerator otherwise remains dynamic.
        pipeline.insight_generator.regime = args.regime
        pipeline.insight_generator.temperature = args.temperature
        pipeline.insight_generator.max_new_tokens = args.max_new_tokens
        model = tokenizer = system_prompt = USER_TEMPLATE = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".progress.jsonl")
    latest = load_progress(progress_path)
    completed_ids = {
        record_id
        for record_id, row in latest.items()
        if row.get("status") == "ok"
        and row.get("regime") == args.regime
        and row.get("generation_mode") == args.generation_mode
        and row.get("model_path") == args.model_path
    }

    with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
        for position, gold in enumerate(selected, 1):
            record_id = str(gold["id"])
            if record_id in completed_ids:
                print(f"[{position}/{len(selected)}] {record_id}: already completed")
                continue

            record_seed = args.seed + position * 100
            try:
                if args.generation_mode == "direct":
                    generated = direct_generate(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=system_prompt,
                        user_prompt=USER_TEMPLATE.format(text=gold["text"]),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                        self_consistency_m=args.self_consistency_m,
                        run_seed=record_seed,
                    )
                else:
                    set_seed(record_seed)
                    generated = pipeline(gold["text"]).strip()
                if not generated:
                    raise RuntimeError("Model returned an empty string")
                result = {
                    "id": record_id,
                    "generated_table": generated,
                    "status": "ok",
                    "regime": args.regime,
                    "source_text_sha256": sha256_text(gold["text"]),
                    "model_path": args.model_path,
                    "adapter_path": args.adapter_path,
                    "generation_mode": args.generation_mode,
                    "seed_base": args.seed,
                    "seed_record": record_seed,
                    "temperature": args.temperature if args.do_sample or args.generation_mode == "pipeline" else None,
                    "do_sample": args.do_sample if args.generation_mode == "direct" else True,
                    "self_consistency_m": args.self_consistency_m,
                    "max_new_tokens": args.max_new_tokens,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                }
                print(f"[{position}/{len(selected)}] {record_id}: ok")
            except Exception as exc:
                result = {
                    "id": record_id,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "regime": args.regime,
                    "source_text_sha256": sha256_text(gold["text"]),
                    "model_path": args.model_path,
                    "adapter_path": args.adapter_path,
                    "generation_mode": args.generation_mode,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                }
                print(f"[{position}/{len(selected)}] {record_id}: ERROR {exc}")
                if args.fail_fast:
                    progress.write(json.dumps(result, ensure_ascii=False) + "\n")
                    progress.flush()
                    raise
            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.flush()

    latest = load_progress(progress_path)
    final_rows = [latest[str(row["id"])] for row in selected if str(row["id"]) in latest]
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok_count = sum(row.get("status") == "ok" for row in final_rows)
    error_count = sum(row.get("status") != "ok" for row in final_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if ok_count == len(selected) else "incomplete",
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_sha256": sha256_file(args.benchmark),
        "expected_records": len(selected),
        "output_records": len(final_rows),
        "ok_records": ok_count,
        "error_records": error_count,
        "regime": args.regime,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "generation_mode": args.generation_mode,
        "repo_dir": str(args.repo_dir.resolve()),
        "seed": args.seed,
        "temperature": args.temperature,
        "do_sample": args.do_sample if args.generation_mode == "direct" else True,
        "self_consistency_m": args.self_consistency_m,
        "max_new_tokens": args.max_new_tokens,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "regime_propagation_fix_applied": args.generation_mode == "pipeline",
        "environment": {
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(manifest_path)
    if error_count:
        raise SystemExit(f"Inference incomplete: {error_count} records failed")


if __name__ == "__main__":
    main()
