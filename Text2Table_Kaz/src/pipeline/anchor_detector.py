"""
Anchor segment detection — identifies sentences containing extractable
factual content (numbers, dates, named entities).
Implements the 3-stage hybrid approach from Ospan et al. (2024), Section III-C.
"""

import re
from typing import List, Set


class AnchorDetector:
    """
    Three-stage anchor detection:
      Stage 1 — Lexical-regex: numerical indicators, dates, percentages
      Stage 2 — Neural NER: Person, Org, Location, Quantity spans
      Stage 3 — Morphosyntactic filter: predicative verbs + cardinal numerals
    """

    # Stage 1 patterns (Kazakh numeric surface forms)
    PATTERNS = [
        re.compile(r"\d+[,.]?\d*\s*%"),                         # percentages: 8,7%
        re.compile(r"\d[\d\s]*(?:млрд|млн|мың|трлн)", re.I),    # large numbers
        re.compile(r"\d{4}\s*(?:жыл|ж\.?)"),                    # year expressions
        re.compile(r"(?:қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|"
                   r"шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)", re.I),
        re.compile(r"\d+\s*(?:млрд|млн)\.?\s*теңге", re.I),     # monetary values
        re.compile(r"\d+\.\d+"),                                  # decimal numbers
        re.compile(r"\b\d{1,3}(?:\s\d{3})+\b"),                  # space-separated large nums
    ]

    def __init__(self, use_ner: bool = False, ner_model=None):
        """
        Args:
            use_ner: Whether to use neural NER (requires fine-tuned Kazakh model).
            ner_model: Pre-loaded NER pipeline (optional).
        """
        self.use_ner = use_ner
        self.ner_model = ner_model

    def detect(self, sentences: List[str]) -> Set[int]:
        """
        Return indices of anchor sentences.

        Args:
            sentences: Segmented sentences.

        Returns:
            Set of integer indices identifying anchor sentences.
        """
        anchors = set()

        for i, sent in enumerate(sentences):
            # Stage 1: lexical patterns
            if self._lexical_match(sent):
                anchors.add(i)
                continue

            # Stage 2: NER (if available)
            if self.use_ner and self.ner_model is not None:
                if self._ner_match(sent):
                    anchors.add(i)
                    continue

        return anchors

    def _lexical_match(self, sentence: str) -> bool:
        return any(pat.search(sentence) for pat in self.PATTERNS)

    def _ner_match(self, sentence: str) -> bool:
        try:
            entities = self.ner_model(sentence)
            target_types = {"ORG", "PER", "LOC", "QUANTITY", "DATE", "PERCENT"}
            return any(e.get("entity_group", e.get("label", "")) in target_types
                       for e in entities)
        except Exception:
            return False
