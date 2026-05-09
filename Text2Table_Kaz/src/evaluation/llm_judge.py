"""
GPT-4 as judge evaluation protocol.
Ospan et al. (2024), Section III-G and Appendix A.

Bias mitigation:
  - Randomized example ordering
  - Temperature T=0.2
  - Self-consistency aggregation (m=3)
"""

import json
import logging
from typing import Dict, List, Optional
from statistics import mean

logger = logging.getLogger(__name__)

# Evaluation prompt from Appendix A of the paper
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for data journalism tasks. 
Compare the generated table to the reference table and source text.
Rate each dimension on a 0 to 1 scale (float, two decimal places).

Dimensions:
1) coverage: Are all key facts and numbers from the source text present in the generated table?
2) accuracy: Are all values factually correct? Penalize hallucinations heavily.
3) compression: Is the table concise? Penalize redundant or duplicate rows.
4) structure: Are columns consistent, headers meaningful, cell types homogeneous?
5) journalistic_value: Could a professional journalist use this table directly for reporting?

Return ONLY a JSON object with no additional text:
{"coverage": 0.85, "accuracy": 0.92, "compression": 0.97, "structure": 0.98, "journalistic_value": 0.90}"""

JUDGE_USER_TEMPLATE = """SOURCE TEXT:
{source_text}

REFERENCE TABLE:
{reference_table}

GENERATED TABLE:
{generated_table}

Evaluate the generated table on all 5 dimensions."""


class LLMJudge:
    """
    GPT-4 based evaluation following the LLM-as-a-Judge paradigm.
    Reference: Zheng et al. (2023), arXiv:2306.05685
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.2,
        m: int = 3,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.m = m
        self._setup_client(api_key)

    def _setup_client(self, api_key: Optional[str]):
        try:
            from openai import OpenAI
            import os
            key = api_key or os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=key) if key else None
        except ImportError:
            self.client = None
            logger.warning("openai package not installed. LLM judge unavailable.")

    def evaluate(
        self,
        source_text: str,
        reference_table: str,
        generated_table: str,
    ) -> Dict[str, float]:
        """
        Evaluate a single generated table.

        Args:
            source_text:       Original Kazakh article text.
            reference_table:   Gold-standard Markdown table.
            generated_table:   Model-generated Markdown table.

        Returns:
            Dict with 5 metric scores (mean over m=3 self-consistency samples).
        """
        if self.client is None:
            logger.warning("No OpenAI client. Returning placeholder scores.")
            return self._placeholder_scores()

        completions = []
        for _ in range(self.m):
            score = self._single_evaluation(source_text, reference_table, generated_table)
            if score is not None:
                completions.append(score)

        if not completions:
            return self._placeholder_scores()

        # Average across self-consistency samples
        keys = completions[0].keys()
        return {k: round(mean(c[k] for c in completions), 4) for k in keys}

    def _single_evaluation(
        self,
        source_text: str,
        reference_table: str,
        generated_table: str,
    ) -> Optional[Dict[str, float]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                        source_text=source_text[:1500],
                        reference_table=reference_table,
                        generated_table=generated_table,
                    )},
                ],
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM judge error: {e}")
            return None

    @staticmethod
    def _placeholder_scores() -> Dict[str, float]:
        return {
            "coverage": 0.0,
            "accuracy": 0.0,
            "compression": 0.0,
            "structure": 0.0,
            "journalistic_value": 0.0,
        }

    def batch_evaluate(
        self,
        examples: List[Dict],
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Evaluate a batch of examples."""
        results = []
        for i, ex in enumerate(examples):
            if show_progress and i % 10 == 0:
                print(f"  Evaluated {i}/{len(examples)}...")
            score = self.evaluate(
                source_text=ex["text"],
                reference_table=ex["table"],
                generated_table=ex["generated"],
            )
            results.append(score)
        return results
