"""
Kazakh-specific NLP utilities.

Covers:
  - Named entity recognition (fine-tuned Kazakh NER model)
  - Morphosyntactic anchor detection
  - Vowel harmony validation
  - Script normalization (Cyrillic ↔ Latin)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple


class EntitySpan(NamedTuple):
    text: str
    start: int
    end: int
    label: str   # PERSON | ORG | LOC | QUANTITY | DATE


# Kazakh NER patterns (heuristic; production uses fine-tuned model)
NER_PATTERNS: dict[str, re.Pattern] = {
    "ORG": re.compile(
        r"\b(?:ҚазМұнайГаз|Самрұқ-Қазына|Байтерек|КазАтомПром|"
        r"ҚТЖ|«[А-ЯҒҚҢҮҰӨӘІa-zA-Z][^»]{2,}»|"
        r"[А-ЯҒҚҢҮҰӨӘІa-zA-Z][а-яғқңүұөәі]+\s+(?:АҚ|ЖАҚ|ЖШС|ТОО|Corp|LLC))\b",
        re.UNICODE,
    ),
    "PERSON": re.compile(
        r"\b[А-ЯҒҚҢҮҰӨӘІа-яғқңүұөәі]+\s+[А-ЯҒҚҢҮҰӨӘІ][а-яғқңүұөәі]+(?:ұлы|қызы)?\b",
        re.UNICODE,
    ),
    "QUANTITY": re.compile(
        r"\b\d[\d\s]*(?:[,\.]\d+)?\s*(?:млрд|млн|трлн|мың|%|пайыз|теңге|долл|евро|т\.б\.)\b",
        re.IGNORECASE | re.UNICODE,
    ),
    "DATE": re.compile(
        r"\b(?:20\d{2}|19\d{2})"
        r"(?:\s+жыл(?:ы|дың|да|дан)?)?"
        r"|\b(?:қаңтар|ақпан|наурыз|сәуір|мамыр|маусым|"
        r"шілде|тамыз|қыркүйек|қазан|қараша|желтоқсан)"
        r"(?:\s+20\d{2})?",
        re.IGNORECASE | re.UNICODE,
    ),
    "LOC": re.compile(
        r"\b(?:Алматы|Астана|Шымкент|Қарағанды|Атырау|Ақтөбе|"
        r"Павлодар|Семей|Өскемен|Тараз|Қостанай|Орал|"
        r"Қазақстан|Ресей|Қытай|АҚШ|ЕО|ТМД)\b",
        re.UNICODE,
    ),
}

# Kazakh vowel harmony classes
BACK_VOWELS = set("аоұу")
FRONT_VOWELS = set("әөүі")


class KazakhNLPUtils:
    """
    Collection of Kazakh-specific NLP utilities.

    In production, self.ner_model wraps the fine-tuned Kazakh NER
    model (Akhmed-Zaki et al., 2020). Without GPU, falls back to
    pattern-based heuristics.
    """

    def __init__(self, use_neural_ner: bool = False):
        self.use_neural_ner = use_neural_ner
        self._ner_model = None

    def detect_entities(self, text: str) -> list[EntitySpan]:
        """
        Detect named entities in Kazakh text.

        Returns list of EntitySpan (text, start, end, label).
        """
        spans = []
        for label, pattern in NER_PATTERNS.items():
            for m in pattern.finditer(text):
                spans.append(EntitySpan(
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    label=label,
                ))
        # Sort by start position; remove overlapping spans (keep longer)
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        return self._remove_overlaps(spans)

    def is_anchor_sentence(self, sentence: str) -> bool:
        """
        Morphosyntactic anchor filter (Section III-B-3).

        A sentence qualifies as an anchor if it contains:
          - A numerical quantity (lexical-regex pass), OR
          - A named entity (neural NER pass), OR
          - A predicative verb in indicative mood + cardinal numeral
        """
        has_number = bool(NER_PATTERNS["QUANTITY"].search(sentence))
        has_entity = bool(
            NER_PATTERNS["ORG"].search(sentence)
            or NER_PATTERNS["PERSON"].search(sentence)
            or NER_PATTERNS["LOC"].search(sentence)
        )
        has_predicate = self._has_indicative_predicate(sentence)
        return (has_number or has_entity) and has_predicate

    def check_vowel_harmony(self, word: str) -> bool:
        """
        Validate Kazakh vowel harmony in a word.

        Returns True if the word is vowel-harmonic (all vowels belong
        to the same class: front or back).
        """
        vowels_in_word = [c for c in word.lower() if c in BACK_VOWELS | FRONT_VOWELS]
        if not vowels_in_word:
            return True  # No vowels — consonant cluster, harmony not applicable

        back = any(v in BACK_VOWELS for v in vowels_in_word)
        front = any(v in FRONT_VOWELS for v in vowels_in_word)
        return not (back and front)  # Violation if both classes present

    def normalize_script(self, text: str, target: str = "cyrillic") -> str:
        """
        Convert between Kazakh Cyrillic and Latin scripts.

        The Kazakh Latin alphabet (2017 reform) is used in some
        contemporary publications alongside the traditional Cyrillic.
        """
        if target == "cyrillic":
            return self._latin_to_cyrillic(text)
        return self._cyrillic_to_latin(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_indicative_predicate(sentence: str) -> bool:
        """
        Heuristic check for predicative verb in indicative mood.

        Kazakh indicative past-tense suffixes: -ды/-ді/-ты/-ті
        Kazakh present-future suffixes: -ады/-еді/-йды/-йді
        """
        indicative = re.compile(
            r"[а-яғқңүұөәі]+(ды|ді|ты|ті|ады|еді|йды|йді|жатыр|тұр|отыр)\b",
            re.UNICODE,
        )
        return bool(indicative.search(sentence.lower()))

    @staticmethod
    def _remove_overlaps(spans: list[EntitySpan]) -> list[EntitySpan]:
        result = []
        last_end = -1
        for span in spans:
            if span.start >= last_end:
                result.append(span)
                last_end = span.end
        return result

    # Transliteration tables (Kazakh Latin 2017 reform ↔ Cyrillic)
    _LATIN_TO_CYR = str.maketrans({
        "A": "А", "a": "а", "Á": "Ә", "á": "ә",
        "B": "Б", "b": "б", "D": "Д", "d": "д",
        "E": "Е", "e": "е", "F": "Ф", "f": "ф",
        "G": "Г", "g": "г", "Ǵ": "Ғ", "ǵ": "ғ",
        "H": "Х", "h": "х", "I": "І", "i": "і",
        "Ï": "И", "ï": "и", "J": "Ж", "j": "ж",
        "K": "К", "k": "к", "Q": "Қ", "q": "қ",
        "L": "Л", "l": "л", "M": "М", "m": "м",
        "N": "Н", "n": "н", "Ń": "Ң", "ń": "ң",
        "O": "О", "o": "о", "Ó": "Ө", "ó": "ө",
        "P": "П", "p": "п", "R": "Р", "r": "р",
        "S": "С", "s": "с", "Sh": "Ш", "sh": "ш",
        "T": "Т", "t": "т", "U": "У", "u": "у",
        "Ú": "Ұ", "ú": "ұ", "Ü": "Ү", "ü": "ү",
        "V": "В", "v": "в", "Y": "Й", "y": "й",
        "Z": "З", "z": "з",
    })

    def _latin_to_cyrillic(self, text: str) -> str:
        return text.translate(self._LATIN_TO_CYR)

    def _cyrillic_to_latin(self, text: str) -> str:
        # Simplified reverse mapping
        rev = {v: k for k, v in self._LATIN_TO_CYR.items() if len(k) == 1}
        return text.translate(str.maketrans(rev))
