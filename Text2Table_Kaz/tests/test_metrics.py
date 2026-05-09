"""Tests for evaluation metrics."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    CoverageMetric, AccuracyMetric, CompressionMetric,
    StructureMetric, MetricSuite, EvaluationResult
)


REF = "| Тақырып | Мән |\n|---------|-----|\n| Инфляция | 8.7% |"
GEN_GOOD = "| Тақырып | Мән |\n|---------|-----|\n| Инфляция | 8.7% |"
GEN_PARTIAL = "| Тақырып | Мән |\n|---------|-----|\n| Инфляция | 9% |"
SOURCE = "Инфляция деңгейі 8,7%-ды құрады."


class TestCoverage:
    def test_perfect_coverage(self):
        m = CoverageMetric()
        assert m.compute(GEN_GOOD, REF, SOURCE) == 1.0

    def test_partial_coverage(self):
        m = CoverageMetric()
        score = m.compute(GEN_PARTIAL, REF, SOURCE)
        assert 0.0 <= score <= 1.0


class TestAccuracy:
    def test_perfect_accuracy(self):
        m = AccuracyMetric()
        assert m.compute(GEN_GOOD, REF, SOURCE) >= 0.8

    def test_inaccurate(self):
        m = AccuracyMetric()
        gen_wrong = "| Тақырып | Мән |\n|---------|-----|\n| Валюта | 500 |"
        score = m.compute(gen_wrong, REF, SOURCE)
        assert score < 0.5


class TestMetricSuite:
    def test_full_suite(self):
        suite = MetricSuite()
        result = suite.evaluate(GEN_GOOD, REF, SOURCE, journalistic_value=0.9)
        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.coverage <= 1.0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.compression <= 1.0
        assert 0.0 <= result.structure <= 1.0
        assert result.total > 0.0

    def test_to_dict(self):
        suite = MetricSuite()
        result = suite.evaluate(GEN_GOOD, REF, SOURCE)
        d = result.to_dict()
        assert set(d.keys()) == {"coverage", "accuracy", "compression",
                                  "structure", "journalistic_value", "total"}
