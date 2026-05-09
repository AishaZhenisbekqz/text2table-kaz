"""
Numerical and entity normalization for Kazakh text.

Implements Step 5 of the tuple normalization pipeline (Section III-B-5):
  - Parse Kazakh-format numbers (N,N format, large-number abbreviations)
  - Convert to IEEE 754 double-precision floating-point
  - Re-serialize in locale-neutral format
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


MULTIPLIERS: dict[str, float] = {
    "трлн": 1e12,
    "млрд": 1e9,
    "млн": 1e6,
    "мың": 1e3,
}

PERCENT_TOKENS = {"пайыз", "%"}

# ISO 8601 month names (Kazakh → number)
MONTH_MAP: dict[str, int] = {
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4,
    "мамыр": 5, "маусым": 6, "шілде": 7, "тамыз": 8,
    "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
}

# Kazakh → ISO year-quarter pattern: "III тоқсан 2024" → "2024-Q3"
QUARTER_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


@dataclass
class ParsedNumber:
    raw: str
    value: float
    unit: str       # "", "%", "млн", "млрд", "трлн", "мың"
    is_percent: bool

    def to_locale_neutral(self) -> str:
        """Serialize to locale-neutral string (no thousands separator, dot decimal)."""
        if self.is_percent:
            return f"{self.value:.1f}%"
        if self.value >= 1e9:
            return f"{self.value / 1e9:.2f} млрд"
        if self.value >= 1e6:
            return f"{self.value / 1e6:.2f} млн"
        if self.value >= 1e3 and self.unit == "мың":
            return f"{self.value / 1e3:.2f} мың"
        # Remove trailing zeros
        formatted = f"{self.value:.4f}".rstrip("0").rstrip(".")
        return formatted


class NumericalNormalizer:
    """
    Normalize Kazakh numerical surface forms to locale-neutral format.

    Handles:
      - Decimal comma: "23,4%" → 23.4%
      - Large abbreviations: "1,5 млрд" → 1_500_000_000
      - Ordinal temporal: "2024 жылы" → "2024"
      - Quarter expressions: "III тоқсан 2024" → "2024-Q3"
    """

    # Pattern: optional sign, digits, optional decimal (comma), optional unit
    _NUM_RE = re.compile(
        r"(?P<sign>[+-])?"
        r"(?P<int>\d[\d\s]*)"
        r"(?:[,\.](?P<dec>\d+))?"
        r"\s*"
        r"(?P<unit>трлн|млрд|млн|мың|пайыз|%)?",
        re.IGNORECASE | re.UNICODE,
    )

    _YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2})\s*(?:жыл[ы]?|ж\.?)?\b", re.UNICODE)
    _QUARTER_RE = re.compile(
        r"\b(I{1,3}V?|V?I{0,3})\s+тоқсан(?:\s+(20\d{2}))?\b", re.UNICODE
    )

    def normalize_number(self, text: str) -> Optional[ParsedNumber]:
        """
        Parse a Kazakh numerical expression into a ParsedNumber.

        Returns None if no numeric content found.
        """
        m = self._NUM_RE.search(text)
        if not m:
            return None

        sign = -1 if m.group("sign") == "-" else 1
        int_part = m.group("int").replace(" ", "")
        dec_part = m.group("dec") or ""
        unit = (m.group("unit") or "").lower().strip()

        try:
            value = float(f"{int_part}.{dec_part}" if dec_part else int_part)
        except ValueError:
            return None

        value *= sign
        is_percent = unit in PERCENT_TOKENS

        multiplier = MULTIPLIERS.get(unit, 1.0)
        if not is_percent:
            value *= multiplier

        return ParsedNumber(
            raw=m.group(),
            value=value,
            unit=unit,
            is_percent=is_percent,
        )

    def normalize_temporal(self, text: str) -> str:
        """
        Normalize Kazakh temporal expressions to ISO-like strings.

        Examples:
          "2024 жылы" → "2024"
          "III тоқсан 2024" → "2024-Q3"
          "қаңтар 2024" → "2024-01"
        """
        # Quarter: "III тоқсан 2024" → "2024-Q3"
        def replace_quarter(m):
            roman = m.group(1)
            year = m.group(2) or ""
            q = QUARTER_ROMAN.get(roman, 0)
            return f"{year}-Q{q}" if year else f"Q{q}"

        text = self._QUARTER_RE.sub(replace_quarter, text)

        # Month: "қаңтар 2024" → "2024-01"
        for kaz_month, month_num in MONTH_MAP.items():
            pattern = re.compile(
                rf"\b{kaz_month}\s+(20\d{{2}}|19\d{{2}})\b", re.IGNORECASE
            )
            text = pattern.sub(lambda m: f"{m.group(1)}-{month_num:02d}", text)

        # Year: "2024 жылы" → "2024"
        text = self._YEAR_RE.sub(lambda m: m.group(1), text)

        return text.strip()

    def normalize_cell(self, cell: str) -> str:
        """
        Auto-detect and normalize a table cell value.

        Tries numerical normalization first, then temporal.
        Returns original string if no pattern matches.
        """
        num = self.normalize_number(cell)
        if num:
            return num.to_locale_neutral()

        temporal = self.normalize_temporal(cell)
        if temporal != cell:
            return temporal

        return cell
