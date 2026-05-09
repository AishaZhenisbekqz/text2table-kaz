"""
Semi-automated annotation using the T³ pipeline.
Ospan et al. (2024), Section III-B-3.

T³ stages:
  (i)  Atomic SPO tuple extraction via dependency parsing + LLM verification
  (ii) Semantic clustering of co-referential tuples
  (iii) Schema-aware aggregation → Markdown table

Auto-labeling validated: κ_auto = 0.81 (static), 0.77 (dynamic)
"""

import json
from typing import List, Dict, Optional
from pathlib import Path


class T3Annotator:
    """
    Semi-automated labeler using the Text-Tuple-Table pipeline.
    Reference: Deng et al. (2024), EMNLP — doi:10.18653/v1/2024.emnlp-main.523
    """

    def __init__(self, pipeline, regime: str = "dynamic"):
        """
        Args:
            pipeline: Initialized Text2TablePipeline instance.
            regime:   "static" or "dynamic".
        """
        self.pipeline = pipeline
        self.regime = regime

    def annotate_batch(
        self,
        texts: List[str],
        output_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Generate table annotations for a batch of texts.

        Args:
            texts:       List of raw Kazakh text strings.
            output_path: If provided, save JSONL to this path.

        Returns:
            List of annotation dicts: {"text": ..., "table": ..., "regime": ...}
        """
        annotations = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Annotating {i}/{len(texts)}...")
            try:
                table = self.pipeline(text)
                if self._is_valid_table(table):
                    annotations.append({
                        "text": text,
                        "table": table,
                        "regime": self.regime,
                        "source": "t3_auto",
                    })
            except Exception as e:
                print(f"  Warning: annotation failed for sample {i}: {e}")
                continue

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for ann in annotations:
                    f.write(json.dumps(ann, ensure_ascii=False) + "\n")
            print(f"Saved {len(annotations)} annotations to {output_path}")

        return annotations

    def _is_valid_table(self, table: str) -> bool:
        """Validate that output is a well-formed Markdown table."""
        lines = [l.strip() for l in table.strip().split("\n") if l.strip()]
        pipe_lines = [l for l in lines if l.startswith("|")]
        if len(pipe_lines) < 3:   # header + separator + ≥1 row
            return False
        # Check separator row
        if not any(set(c) <= {"-", ":", " "} for c in pipe_lines[1].split("|")):
            return False
        return True
