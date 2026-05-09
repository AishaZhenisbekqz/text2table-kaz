#!/usr/bin/env python3
"""
Single-article inference.

Usage:
    python scripts/infer.py \
        --adapter_path ./checkpoints/dynamic_best \
        --text "Қазақстан Республикасы..." \
        --regime dynamic

    # Or from file:
    python scripts/infer.py \
        --adapter_path ./checkpoints/dynamic_best \
        --input_file article.txt \
        --regime dynamic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import Text2TablePipeline


def main():
    parser = argparse.ArgumentParser(description="Text2Table_Kaz inference")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--regime", default="dynamic", choices=["static", "dynamic"])
    parser.add_argument("--text", type=str, default=None, help="Input text")
    parser.add_argument("--input_file", type=str, default=None, help="Input .txt file")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Please provide --text or --input_file")
        sys.exit(1)

    pipeline = Text2TablePipeline.from_pretrained(
        base_model=args.base_model,
        lora_adapter=args.adapter_path,
        regime=args.regime,
    )

    table = pipeline(text)
    print("\n" + "="*50)
    print(f"GENERATED TABLE ({args.regime.upper()} regime)")
    print("="*50)
    print(table)
    print("="*50)


if __name__ == "__main__":
    main()
