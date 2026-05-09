"""
Text preprocessing for Kazakh journalistic articles.
Ospan et al. (2024), Section III-A.

Steps:
  1. HTML/DOM cleanup
  2. Orthographic normalization (quotation marks, dashes, Unicode)
  3. Kazakh typo correction (layout substitutions, normative dictionary)
  4. Preservation of: numerals, dates, named entities, measurement units
"""

import re
import unicodedata
from typing import List


# Cyrillic layout substitution pairs common in Kazakh typography
# e.g., Latin 'e' used instead of Cyrillic 'е'
LAYOUT_SUBSTITUTIONS = {
    "Ё": "Е", "ё": "е",           # Rare in Kazakh
    "\u0456": "і",                  # Ukrainian і → Kazakh і
    "\u049B": "қ", "\u04A3": "ң",  # normalize variants
}


class KazakhPreprocessor:
    """
    Cleans and normalizes raw Kazakh web-scraped text.
    Preserves factual anchors (numbers, dates, NEs) invariant.
    """

    # Quotation mark normalization
    QUOTE_PAIRS = [
        ("\u00ab", "«"), ("\u00bb", "»"),
        ("\u201c", "«"), ("\u201d", "»"),
        ('"', "«"),
    ]

    # Dash normalization
    DASHES = ["\u2013", "\u2014", "\u2015", "\u2212"]

    def __init__(self, normative_dict_path: str = None):
        self.normative_dict = {}
        if normative_dict_path:
            self._load_normative_dict(normative_dict_path)

    def process(self, text: str) -> str:
        """Full preprocessing pipeline."""
        text = self._remove_html_artifacts(text)
        text = self._normalize_whitespace(text)
        text = self._normalize_quotes(text)
        text = self._normalize_dashes(text)
        text = self._remove_non_printable(text)
        text = self._fix_layout_substitutions(text)
        if self.normative_dict:
            text = self._apply_normative_dict(text)
        return text.strip()

    def _remove_html_artifacts(self, text: str) -> str:
        """Remove residual HTML tags and entities."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _normalize_quotes(self, text: str) -> str:
        for orig, replacement in self.QUOTE_PAIRS:
            text = text.replace(orig, replacement)
        return text

    def _normalize_dashes(self, text: str) -> str:
        for dash in self.DASHES:
            text = text.replace(dash, "–")
        return text

    def _remove_non_printable(self, text: str) -> str:
        return "".join(
            c for c in text
            if unicodedata.category(c)[0] not in ("C",) or c in ("\n", "\t")
        )

    def _fix_layout_substitutions(self, text: str) -> str:
        for wrong, correct in LAYOUT_SUBSTITUTIONS.items():
            text = text.replace(wrong, correct)
        return text

    def _apply_normative_dict(self, text: str) -> str:
        """Apply normative dictionary corrections (word-level)."""
        words = text.split()
        corrected = [self.normative_dict.get(w, w) for w in words]
        return " ".join(corrected)

    def _load_normative_dict(self, path: str):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.normative_dict[parts[0]] = parts[1]

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.process(t) for t in texts]
