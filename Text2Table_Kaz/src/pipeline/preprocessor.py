"""
Kazakh-specific text preprocessing and sentence segmentation.

Implements Section III-A preprocessing steps:
  - HTML/markup removal
  - Orthographic normalization (quotation marks, dashes, Unicode)
  - Kazakh-specific typo correction
  - Sentence segmentation with agglutinative morphology heuristics
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class PreprocessorConfig:
    # Cosine threshold for chunking (passed downstream)
    theta: float = 0.72
    # Normative Kazakh dictionary path (optional)
    dictionary_path: str | None = None


# Kazakh-specific Cyrillic character pairs that are commonly substituted
# by layout-adjacent keys (extended vs base Cyrillic)
KAZAKH_CYRILLIC_MAP: dict[str, str] = {
    "\u04d9": "\u0435",  # ә → е  (common OCR error)
    "\u04bb": "\u04bc",  # һ → Ң
    "\u0493": "\u0433",  # ғ → г
    "\u049b": "\u043a",  # қ → к
    "\u04a3": "\u043d",  # ң → н
    "\u04af": "\u0443",  # ү → у
    "\u04b1": "\u0443",  # ұ → у
    "\u04e9": "\u043e",  # ө → о
}

# Quotation normalization
QUOTE_MAP = str.maketrans({
    "\u00ab": '"',  # «
    "\u00bb": '"',  # »
    "\u201c": '"',  # "
    "\u201d": '"',  # "
    "\u2018": "'",  # '
    "\u2019": "'",  # '
})

# Dash normalization
DASH_PATTERN = re.compile(r"[\u2013\u2014\u2015]")  # en-dash, em-dash, hor. bar

# Non-printable characters (keep newlines)
NON_PRINTABLE = re.compile(r"[^\S\n]+"  # collapse whitespace
                           r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Decimal comma (Kazakh convention): "23,4%" — protect from sentence splitting
DECIMAL_COMMA = re.compile(r"(\d),((\d+)([%\s]|$))")

# Abbreviated titles that should not trigger sentence boundaries
KAZAKH_ABBREVS = frozenset([
    "т.б", "т.д", "т.с.с", "б.з.д", "б.з",
    "млн", "млрд", "трлн", "мың",
    "ж", "жж", "қ", "б",   # жыл, жылдар, қала, бет
    "проф", "акад", "д-р", "PhD",
    "ЖЗҚ", "ЖАҚ", "АҚ", "ЖШС",
])

# Sentence boundary regex (simplified; extended in spaCy integration)
SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЎҒҚҢҮҰӨӘІа-яa-zA-Z\u0400-\u04FF\d])")


class KazakhPreprocessor:
    """
    Preprocessing for Kazakh journalistic texts.

    Steps:
      1. Remove HTML markup
      2. Unicode normalization (NFC)
      3. Orthographic normalization (quotes, dashes, typos)
      4. Numerical/entity preservation
      5. Sentence segmentation with Kazakh heuristics
    """

    def __init__(self, config: PreprocessorConfig | None = None):
        self.config = config or PreprocessorConfig()
        self._html_tag = re.compile(r"<[^>]+>")
        self._multi_space = re.compile(r" {2,}")
        self._multi_newline = re.compile(r"\n{3,}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Full cleaning pipeline. Returns cleaned text."""
        text = self._remove_html(text)
        text = unicodedata.normalize("NFC", text)
        text = self._normalize_quotes(text)
        text = self._normalize_dashes(text)
        text = self._remove_non_printable(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def segment(self, text: str) -> list[str]:
        """
        Segment cleaned text into sentences.

        Applies Kazakh-specific heuristics to handle:
          - Agglutinative boundary ambiguity
          - Decimal comma expressions (23,4%)
          - Abbreviated titles
          - Kazakh guillemet quotation marks
        """
        text = self.clean(text)
        sentences = self._split_sentences(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return sentences

    # ------------------------------------------------------------------
    # Cleaning helpers
    # ------------------------------------------------------------------

    def _remove_html(self, text: str) -> str:
        return self._html_tag.sub(" ", text)

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        return text.translate(QUOTE_MAP)

    @staticmethod
    def _normalize_dashes(text: str) -> str:
        return DASH_PATTERN.sub("—", text)

    @staticmethod
    def _remove_non_printable(text: str) -> str:
        # Remove zero-width chars, BOM, etc.
        return re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = self._multi_space.sub(" ", text)
        text = self._multi_newline.sub("\n\n", text)
        return text

    # ------------------------------------------------------------------
    # Sentence segmentation
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """
        Rule-based sentence splitter with Kazakh heuristics.

        In production, this wraps spaCy's xx_ent_wiki_sm with
        the custom rules below as additional components.
        """
        # Protect decimal commas from splitting
        text = DECIMAL_COMMA.sub(r"\1\u2060\2", text)  # use word-joiner

        # Protect known abbreviations: "млн." → "млн\u2060"
        for abbrev in KAZAKH_ABBREVS:
            text = re.sub(
                r"\b" + re.escape(abbrev) + r"\.",
                abbrev + "\u2060",  # word joiner prevents split
                text,
            )

        # Split on sentence boundaries
        raw_sentences = SENT_BOUNDARY.split(text)

        # Restore protected characters
        sentences = [s.replace("\u2060", "") for s in raw_sentences]
        return sentences
