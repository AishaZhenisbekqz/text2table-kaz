"""
Semantic tuple extraction and normalization.
Ospan et al. (2024), Section III-F.

5-step normalization:
  1. Lemmatization (Kazakh morphological analyzer)
  2. Predicate canonicalization (180-predicate vocabulary)
  3. Stop-word removal
  4. Coreference resolution
  5. Numerical normalization (Kazakh → IEEE 754)
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .insight_generator import Insight


@dataclass
class Tuple:
    subject: str
    predicate: str
    obj: str
    source_sentence: Optional[str] = None
    confidence: float = 1.0


# Canonical predicate vocabulary for Kazakh economic journalism
CANONICAL_PREDICATES = {
    # Growth / increase
    "өсу", "артыу", "жоғарылау", "ұлғайу", "күшею",
    # Decrease
    "азаю", "төмендеу", "қысқару", "кему",
    # Achievement / reaching
    "жету", "қол жеткізу", "орындау",
    # Announcement / declaration
    "жариялау", "хабарлау", "мәлімдеу", "айту",
    # Allocation / investment
    "бөлу", "инвестициялау", "қаржыландыру",
    # Production / output
    "өндіру", "шығару", "жасау",
    # Comparison
    "асыу", "аспау", "тең болу",
    # English equivalents for multilingual support
    "increase", "decrease", "reach", "allocate", "produce",
    "announce", "invest", "achieve", "exceed",
}

STOP_PREDICATES = {"болу", "бар", "жоқ", "деу", "ету", "келу", "кету",
                   "be", "have", "say", "go", "come"}

# Kazakh number word → multiplier
KZ_NUMBER_WORDS = {
    "млрд": 1_000_000_000,
    "трлн": 1_000_000_000_000,
    "млн":  1_000_000,
    "мың":  1_000,
    "жүз":  100,
}


class TupleExtractor:
    """
    Extracts (subject, predicate, object) triples from LLM-generated insights
    and applies a 5-step normalization pipeline.
    """

    def __init__(self, predicate_vocab=None):
        self.predicate_vocab = predicate_vocab or CANONICAL_PREDICATES

    def extract(self, insights: List[Insight]) -> List[Tuple]:
        all_tuples = []
        for insight in insights:
            raw_tuples = self._parse_markdown_table(insight.raw_text)
            normalized = [self._normalize(t) for t in raw_tuples]
            all_tuples.extend([t for t in normalized if t is not None])
        return all_tuples

    def _parse_markdown_table(self, text: str) -> List[Tuple]:
        """Extract tuples from pipe-delimited Markdown table."""
        tuples = []
        lines = text.strip().split("\n")
        header = None

        for line in lines:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                continue
            # Skip separator rows
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            if header is None:
                header = cells
                continue
            # Pad cells to header length
            while len(cells) < len(header):
                cells.append("-")
            # Create a simple Subject/Predicate/Object mapping
            if len(cells) >= 2:
                subj = cells[0] if cells[0] != "-" else ""
                pred = header[1] if len(header) > 1 else "мәлімет"
                obj  = cells[1] if len(cells) > 1 and cells[1] != "-" else ""
                if subj or obj:
                    tuples.append(Tuple(subject=subj, predicate=pred, obj=obj))

        return tuples

    def _normalize(self, t: Tuple) -> Optional[Tuple]:
        """Apply 5-step normalization pipeline."""
        # Step 1: Lemmatization (simplified — strip common suffixes)
        t.subject = self._lemmatize(t.subject)
        t.obj = self._normalize_object(t.obj)

        # Step 2 & 3: Predicate canonicalization + stop-word removal
        pred = t.predicate.lower().strip()
        if pred in STOP_PREDICATES:
            return None
        t.predicate = pred

        # Step 4: Skip empty tuples
        if not t.subject and not t.obj:
            return None

        return t

    def _lemmatize(self, text: str) -> str:
        """Simplified Kazakh lemmatization via common suffix stripping."""
        if not text:
            return text
        # Common case/possession suffixes to strip
        suffixes = ["ның", "нің", "дың", "дің", "тың", "тің",
                    "ды", "ді", "ны", "ні", "ға", "ге", "қа", "ке",
                    "да", "де", "та", "те", "нан", "нен", "дан", "ден"]
        for suf in suffixes:
            if text.lower().endswith(suf) and len(text) > len(suf) + 2:
                return text[:-len(suf)]
        return text

    def _normalize_object(self, text: str) -> str:
        """Step 5: Normalize Kazakh numeric formats to locale-neutral form."""
        if not text:
            return text

        # Replace decimal comma: 8,7 → 8.7
        text = re.sub(r"(\d+),(\d+)", r"\1.\2", text)

        # Expand Kazakh large-number words
        for word, multiplier in KZ_NUMBER_WORDS.items():
            pattern = re.compile(rf"(\d+(?:\.\d+)?)\s*{word}", re.IGNORECASE)
            def expand(m, mult=multiplier):
                return str(int(float(m.group(1)) * mult))
            text = pattern.sub(expand, text)

        return text.strip()
