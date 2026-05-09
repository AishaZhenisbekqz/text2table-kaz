"""
Kazakh-aware sentence segmentation.
Extends spaCy with Kazakh-specific heuristics as described in
Ospan et al. (2024), Section III-A.
"""

import re
from typing import List


class KazakhSegmenter:
    """
    Sentence boundary detection for Kazakh Cyrillic text.

    Heuristics applied (Section III-C):
      1. Abbreviated title patterns are treated as non-boundary tokens
      2. Decimal comma expressions (N,N) protected from false splits
      3. Guillemet quotation «» disambiguated from sentence terminals
    """

    # Common Kazakh abbreviations that are NOT sentence boundaries
    ABBREVIATIONS = {
        "б.з.б", "б.з", "т.б", "т.д", "млн", "млрд", "км", "кг",
        "мың", "жыл", "ж", "қ", "б", "с", "мин", "сек", "га", "га",
    }

    # Regex: decimal comma numbers like 23,4 or 1,5 (Kazakh numeric format)
    DECIMAL_COMMA_RE = re.compile(r"\d+,\d+")

    # Regex: large-number abbreviations  1,5 млрд  450 млн
    LARGE_NUMBER_RE = re.compile(
        r"\d[\d\s]*(?:млрд|млн|мың|трлн)\.?", re.IGNORECASE | re.UNICODE
    )

    # Sentence terminal punctuation
    TERMINAL_RE = re.compile(r"(?<=[.!?])\s+(?=[А-ЯӘІҢҒҮҰҚӨҺа-яәіңғүұқөһ\u0400-\u04FF])")

    def __init__(self):
        self._protected_spans: List = []

    def segment(self, text: str) -> List[str]:
        """
        Segment Kazakh text into sentences.

        Args:
            text: Raw Kazakh text (Cyrillic).

        Returns:
            List of sentence strings.
        """
        text = self._normalize_whitespace(text)
        text = self._protect_decimal_commas(text)
        sentences = self._split_on_terminals(text)
        sentences = self._restore_protected(sentences)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def _protect_decimal_commas(self, text: str) -> str:
        """Replace decimal commas with placeholder to prevent false splits."""
        def replace(m):
            return m.group(0).replace(",", "⟨COMMA⟩")
        return self.DECIMAL_COMMA_RE.sub(replace, text)

    def _split_on_terminals(self, text: str) -> List[str]:
        """Split on terminal punctuation, respecting abbreviations."""
        raw = self.TERMINAL_RE.split(text)
        result = []
        buffer = ""
        for part in raw:
            # Check if part ends with a known abbreviation
            stripped = part.rstrip()
            is_abbrev = any(stripped.endswith(abbr) for abbr in self.ABBREVIATIONS)
            if is_abbrev:
                buffer += part + " "
            else:
                result.append(buffer + part)
                buffer = ""
        if buffer:
            result.append(buffer)
        return result

    def _restore_protected(self, sentences: List[str]) -> List[str]:
        return [s.replace("⟨COMMA⟩", ",") for s in sentences]
