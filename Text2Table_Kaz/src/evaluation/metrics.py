"""
Five evaluation metrics for text-to-table quality assessment.
Ospan et al. (2024), Section III-G and Appendix B.

Metrics (0–1 scale):
    1. Coverage      — proportion of key facts present
    2. Accuracy      — factual correspondence (anti-hallucination)
    3. Compression   — absence of redundant rows
    4. Structure     — schema consistency, header quality, type homogeneity
    5. Journalistic Value — usability for fact-based reporting
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EvaluationResult:
    coverage: float
    accuracy: float
    compression: float
    structure: float
    journalistic_value: float
    total: float = 0.0

    def __post_init__(self):
        self.total = (
            self.coverage + self.accuracy + self.compression +
            self.structure + self.journalistic_value
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "compression": round(self.compression, 4),
            "structure": round(self.structure, 4),
            "journalistic_value": round(self.journalistic_value, 4),
            "total": round(self.total, 4),
        }


def parse_markdown_table(markdown: str) -> Dict:
    """Parse a Markdown pipe table into headers + rows."""
    lines = [l.strip() for l in markdown.strip().split("\n") if l.strip()]
    table_lines = [l for l in lines if l.startswith("|")]
    if not table_lines:
        return {"headers": [], "rows": []}

    headers = [c.strip() for c in table_lines[0].split("|") if c.strip()]
    rows = []
    for line in table_lines[2:]:   # skip header separator
        cells = [c.strip() for c in line.split("|") if c.strip()]
        rows.append(cells)
    return {"headers": headers, "rows": rows}


class CoverageMetric:
    """
    Definition 1 (Appendix B):
        Coverage = |F_ref ∩ F_gen| / |F_ref|
    """

    def compute(self, generated: str, reference: str, source_text: str) -> float:
        ref_table = parse_markdown_table(reference)
        gen_table = parse_markdown_table(generated)

        ref_facts = self._extract_facts(ref_table)
        gen_facts = self._extract_facts(gen_table)

        if not ref_facts:
            return 1.0
        intersection = ref_facts & gen_facts
        return len(intersection) / len(ref_facts)

    def _extract_facts(self, table: Dict) -> set:
        facts = set()
        for row in table["rows"]:
            for cell in row:
                if cell and cell != "-":
                    # Normalize: lowercase, strip whitespace
                    facts.add(cell.lower().strip())
        return facts


class AccuracyMetric:
    """
    Definition 2 (Appendix B):
        Accuracy = (1/N) Σ 𝟙[v_i = v_i^ref]
    """

    def compute(self, generated: str, reference: str, source_text: str) -> float:
        gen_table = parse_markdown_table(generated)
        ref_table = parse_markdown_table(reference)

        gen_cells = self._flatten_cells(gen_table)
        ref_cells = self._flatten_cells(ref_table)

        if not gen_cells:
            return 0.0

        correct = sum(
            1 for g in gen_cells
            if any(self._cells_match(g, r) for r in ref_cells)
        )
        return correct / len(gen_cells)

    def _flatten_cells(self, table: Dict) -> List[str]:
        cells = []
        for row in table["rows"]:
            for cell in row:
                if cell and cell != "-":
                    cells.append(cell.lower().strip())
        return cells

    def _cells_match(self, a: str, b: str) -> bool:
        """Fuzzy match: exact or one is substring of other."""
        return a == b or a in b or b in a


class CompressionMetric:
    """
    Definition 3 (Appendix B):
        Compression = |R_unique| / |R_gen|
    """

    def compute(self, generated: str, reference: str = None, source_text: str = None) -> float:
        table = parse_markdown_table(generated)
        rows = table["rows"]
        if not rows:
            return 1.0
        unique_rows = {tuple(r) for r in rows}
        return len(unique_rows) / len(rows)


class StructureMetric:
    """
    Definition 4 (Appendix B):
        Structure = 0.4 * HeaderClarity + 0.3 * TypeHomogeneity + 0.3 * SchemaConsistency
    """

    def compute(self, generated: str, reference: str = None, source_text: str = None) -> float:
        table = parse_markdown_table(generated)

        header_clarity = self._header_clarity(table)
        type_homogeneity = self._type_homogeneity(table)
        schema_consistency = self._schema_consistency(table)

        return (
            0.4 * header_clarity +
            0.3 * type_homogeneity +
            0.3 * schema_consistency
        )

    def _header_clarity(self, table: Dict) -> float:
        headers = table["headers"]
        if not headers:
            return 0.0
        # Penalize empty or placeholder headers
        meaningful = sum(1 for h in headers if h and h != "-" and len(h) > 1)
        return meaningful / len(headers)

    def _type_homogeneity(self, table: Dict) -> float:
        """Check if cells within each column have consistent types."""
        if not table["rows"] or not table["headers"]:
            return 1.0
        n_cols = len(table["headers"])
        scores = []
        for col_idx in range(n_cols):
            col_cells = [
                row[col_idx] for row in table["rows"]
                if col_idx < len(row) and row[col_idx] != "-"
            ]
            if len(col_cells) < 2:
                scores.append(1.0)
                continue
            types = [self._infer_type(c) for c in col_cells]
            dominant_type = max(set(types), key=types.count)
            scores.append(types.count(dominant_type) / len(types))
        return sum(scores) / len(scores) if scores else 1.0

    def _schema_consistency(self, table: Dict) -> float:
        """All rows should have the same number of columns as the header."""
        if not table["headers"] or not table["rows"]:
            return 1.0
        expected = len(table["headers"])
        consistent = sum(1 for row in table["rows"] if len(row) == expected)
        return consistent / len(table["rows"])

    def _infer_type(self, cell: str) -> str:
        if re.match(r"^\d+(?:[.,]\d+)?%?$", cell.strip()):
            return "numeric"
        if re.match(r"^\d{4}$", cell.strip()):
            return "year"
        return "text"


class MetricSuite:
    """Compute all 5 metrics in one call."""

    def __init__(self):
        self.coverage = CoverageMetric()
        self.accuracy = AccuracyMetric()
        self.compression = CompressionMetric()
        self.structure = StructureMetric()

    def evaluate(
        self,
        generated: str,
        reference: str,
        source_text: str,
        journalistic_value: float = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            coverage=self.coverage.compute(generated, reference, source_text),
            accuracy=self.accuracy.compute(generated, reference, source_text),
            compression=self.compression.compute(generated),
            structure=self.structure.compute(generated),
            journalistic_value=journalistic_value if journalistic_value is not None else 0.0,
        )
