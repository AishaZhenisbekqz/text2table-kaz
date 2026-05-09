"""
Table assembly from normalized tuples.
Ospan et al. (2024), Section III-F.

Static regime:  5-column fixed schema
    Тақырып | Оқиға | Көрсеткіш | Кезең | Қосымша мәлімет

Dynamic regime: model-inferred schema
    Columns determined by predicate semantics of extracted tuples.
"""

from typing import List, Dict, Optional
from .tuple_extractor import Tuple

STATIC_COLUMNS = ["Тақырып", "Оқиға", "Көрсеткіш", "Кезең", "Қосымша мәлімет"]


class TableAssembler:
    """
    Assembles extracted tuples into a pipe-delimited Markdown table.

    For static regime:  maps tuples to the fixed 5-column schema.
    For dynamic regime: infers column structure from predicate semantics.
    """

    def __init__(self, regime: str = "dynamic"):
        assert regime in ("static", "dynamic"), "regime must be 'static' or 'dynamic'"
        self.regime = regime

    def assemble(self, tuples: List[Tuple]) -> str:
        """
        Assemble tuples into a Markdown table.

        Args:
            tuples: List of normalized (subject, predicate, object) triples.

        Returns:
            Pipe-delimited Markdown table string.
        """
        if not tuples:
            return "_Мәтіннен кесте жасауға болмады._"

        if self.regime == "static":
            return self._assemble_static(tuples)
        else:
            return self._assemble_dynamic(tuples)

    # ------------------------------------------------------------------ #
    #  Static regime                                                        #
    # ------------------------------------------------------------------ #

    def _assemble_static(self, tuples: List[Tuple]) -> str:
        """Map tuples to the 5-column static schema."""
        rows: Dict[str, Dict] = {}

        for t in tuples:
            subj = t.subject or "-"
            if subj not in rows:
                rows[subj] = {col: "-" for col in STATIC_COLUMNS}
                rows[subj]["Тақырып"] = subj

            pred_lower = t.predicate.lower()
            obj = t.obj or "-"

            # Heuristic column assignment
            if any(w in pred_lower for w in ["өсу", "азаю", "жету", "деңгей",
                                               "increase", "decrease", "reach"]):
                rows[subj]["Көрсеткіш"] = obj
            elif any(w in pred_lower for w in ["жыл", "тоқсан", "ай", "кезең",
                                                "year", "quarter", "period"]):
                rows[subj]["Кезең"] = obj
            elif any(w in pred_lower for w in ["жариялау", "мәлімдеу", "announce"]):
                rows[subj]["Оқиға"] = obj
            else:
                if rows[subj]["Оқиға"] == "-":
                    rows[subj]["Оқиға"] = obj
                else:
                    rows[subj]["Қосымша мәлімет"] = obj

        return self._render_markdown(STATIC_COLUMNS, list(rows.values()))

    # ------------------------------------------------------------------ #
    #  Dynamic regime                                                       #
    # ------------------------------------------------------------------ #

    def _assemble_dynamic(self, tuples: List[Tuple]) -> str:
        """Infer schema from predicate semantics."""
        # Collect unique predicates as columns
        predicates = list(dict.fromkeys(t.predicate for t in tuples if t.predicate))

        # Limit to 8 columns (paper constraint)
        if len(predicates) > 7:
            predicates = predicates[:7]

        columns = ["Субъект"] + predicates

        # Group by subject
        rows: Dict[str, Dict] = {}
        for t in tuples:
            subj = t.subject or "-"
            if subj not in rows:
                rows[subj] = {col: "-" for col in columns}
                rows[subj]["Субъект"] = subj
            if t.predicate in rows[subj]:
                rows[subj][t.predicate] = t.obj or "-"

        return self._render_markdown(columns, list(rows.values()))

    # ------------------------------------------------------------------ #
    #  Rendering                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _render_markdown(columns: List[str], rows: List[Dict]) -> str:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines = [header, separator]

        for row in rows:
            cells = [str(row.get(col, "-")) for col in columns]
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)
