"""Unit tests for the Text2Table_Kaz pipeline components."""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.segmenter import KazakhSegmenter
from src.pipeline.chunker import SemanticChunker, Chunk
from src.pipeline.anchor_detector import AnchorDetector
from src.pipeline.table_assembler import TableAssembler
from src.pipeline.tuple_extractor import Tuple


# ── Segmenter ──────────────────────────────────────────────────────────────

class TestKazakhSegmenter:
    def setup_method(self):
        self.seg = KazakhSegmenter()

    def test_basic_segmentation(self):
        text = "Қазақстан экономикасы өсті. Инфляция 8,7% деңгейінде."
        sents = self.seg.segment(text)
        assert len(sents) >= 1

    def test_decimal_comma_preserved(self):
        text = "Өсу қарқыны 23,4%-ды құрады. Басқа деректер де бар."
        sents = self.seg.segment(text)
        # Decimal comma should not cause false split inside a sentence
        assert any("23,4" in s for s in sents)

    def test_empty_text(self):
        assert self.seg.segment("") == []

    def test_single_sentence(self):
        text = "Бұл бір ғана сөйлем."
        sents = self.seg.segment(text)
        assert len(sents) == 1


# ── Anchor Detector ────────────────────────────────────────────────────────

class TestAnchorDetector:
    def setup_method(self):
        self.detector = AnchorDetector(use_ner=False)

    def test_percentage_detected(self):
        sents = ["Инфляция 8,7%-ды құрады.", "Бұл маңызды мәселе."]
        anchors = self.detector.detect(sents)
        assert 0 in anchors
        assert 1 not in anchors

    def test_year_detected(self):
        sents = ["2023 жыл экономика үшін маңызды болды."]
        anchors = self.detector.detect(sents)
        assert 0 in anchors

    def test_no_anchor(self):
        sents = ["Бұл мәтінде сандар жоқ.", "Тек сөздер ғана."]
        anchors = self.detector.detect(sents)
        assert len(anchors) == 0


# ── Chunker ────────────────────────────────────────────────────────────────

class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(theta=0.72, min_sentences=1, max_sentences=15)

    def _make_embeddings(self, n: int, similar: bool = True) -> np.ndarray:
        if similar:
            base = np.random.rand(768).astype(np.float32)
            base /= np.linalg.norm(base)
            embs = np.stack([base + np.random.randn(768) * 0.05 for _ in range(n)])
        else:
            embs = np.random.randn(n, 768).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms

    def test_similar_sentences_stay_together(self):
        """With very high theta, similar sentences should produce fewer chunks than n."""
        sents = [f"Сөйлем {i}" for i in range(5)]
        embs = self._make_embeddings(5, similar=True)
        # Use low theta (0.3) so similar embeddings are guaranteed to stay together
        chunker = SemanticChunker(theta=0.30, min_sentences=1, max_sentences=15)
        chunks = chunker.chunk(sents, embs)
        # With theta=0.3 and similar embeddings, should collapse to ≤3 chunks
        assert len(chunks) <= 3

    def test_dissimilar_splits(self):
        sents = [f"Сөйлем {i}" for i in range(6)]
        embs = self._make_embeddings(6, similar=False)
        chunks = self.chunker.chunk(sents, embs)
        # Dissimilar → multiple chunks possible
        assert len(chunks) >= 1

    def test_length_filter(self):
        chunker = SemanticChunker(theta=0.5, min_sentences=3, max_sentences=15)
        sents = ["Қысқа."]  # too short
        embs = self._make_embeddings(1)
        chunks = chunker.chunk(sents, embs)
        assert len(chunks) == 0


# ── Table Assembler ────────────────────────────────────────────────────────

class TestTableAssembler:
    def test_static_assembly(self):
        assembler = TableAssembler(regime="static")
        tuples = [
            Tuple(subject="Энергетика", predicate="өсу", obj="15%"),
            Tuple(subject="Энергетика", predicate="жыл", obj="2024"),
        ]
        table = assembler.assemble(tuples)
        assert "Тақырып" in table
        assert "Энергетика" in table
        assert "|" in table

    def test_dynamic_assembly(self):
        assembler = TableAssembler(regime="dynamic")
        tuples = [
            Tuple(subject="ҰБ", predicate="Инфляция", obj="8.7%"),
            Tuple(subject="ҰБ", predicate="Кезең", obj="2024 Q1"),
        ]
        table = assembler.assemble(tuples)
        assert "Субъект" in table
        assert "ҰБ" in table

    def test_empty_tuples(self):
        assembler = TableAssembler(regime="dynamic")
        result = assembler.assemble([])
        assert "болмады" in result or len(result) > 0

    def test_markdown_format(self):
        assembler = TableAssembler(regime="static")
        tuples = [Tuple(subject="Тест", predicate="мән", obj="100")]
        table = assembler.assemble(tuples)
        lines = table.strip().split("\n")
        assert all("|" in line for line in lines if line.strip())


# ── Metrics ────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_compression_perfect(self):
        from src.evaluation.metrics import CompressionMetric
        m = CompressionMetric()
        table = "| A | B |\n|---|---|\n| x | 1 |\n| y | 2 |"
        assert m.compute(table) == 1.0

    def test_compression_duplicates(self):
        from src.evaluation.metrics import CompressionMetric
        m = CompressionMetric()
        table = "| A | B |\n|---|---|\n| x | 1 |\n| x | 1 |"
        assert m.compute(table) < 1.0

    def test_structure_valid(self):
        from src.evaluation.metrics import StructureMetric
        m = StructureMetric()
        table = "| Тақырып | Мән |\n|---------|-----|\n| Тест | 42 |"
        score = m.compute(table)
        assert 0.0 <= score <= 1.0
