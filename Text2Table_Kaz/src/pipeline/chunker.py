"""
Cosine-threshold semantic chunking.
Ospan et al. (2024), Equations (3)–(4), Section III-D.

Boundary declared at position i when:
    sim(s_i, s_{i+1}) < θ    (θ = 0.72, grid-searched over {0.60..0.80})

Chunk length filter: 3 ≤ |c| ≤ 15 sentences.
"""

import numpy as np
from typing import List, Set
from dataclasses import dataclass


@dataclass
class Chunk:
    sentences: List[str]
    start_idx: int
    end_idx: int
    has_anchor: bool = False


class SemanticChunker:
    """
    Splits a document into thematically coherent chunks using
    cosine similarity between consecutive sentence embeddings.
    """

    def __init__(
        self,
        theta: float = 0.72,
        min_sentences: int = 3,
        max_sentences: int = 15,
    ):
        """
        Args:
            theta: Similarity threshold. Boundary when sim < theta.
                   Calibrated via grid search (Section III-D).
            min_sentences: Minimum chunk size (ensures context sufficiency).
            max_sentences: Maximum chunk size (prevents LLM context overflow).
        """
        self.theta = theta
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def chunk(
        self,
        sentences: List[str],
        embeddings: np.ndarray,
        anchors: Set[int] = None,
    ) -> List[Chunk]:
        """
        Segment sentences into semantic chunks.

        Args:
            sentences:  List of sentence strings.
            embeddings: (n, 768) L2-normalized embeddings.
            anchors:    Set of anchor sentence indices.

        Returns:
            List of Chunk objects (length-filtered).
        """
        if len(sentences) == 0:
            return []

        anchors = anchors or set()
        boundaries = self._detect_boundaries(embeddings)
        raw_chunks = self._split_at_boundaries(sentences, boundaries, anchors)
        filtered = self._apply_length_filter(raw_chunks)
        return filtered

    def _detect_boundaries(self, embeddings: np.ndarray) -> List[int]:
        """Return indices i where a new chunk should begin after sentence i."""
        boundaries = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            if sim < self.theta:
                boundaries.append(i)
        return boundaries

    def _split_at_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        anchors: Set[int],
    ) -> List[Chunk]:
        chunks = []
        start = 0
        boundary_set = set(boundaries)

        for i in range(len(sentences)):
            if i in boundary_set or i == len(sentences) - 1:
                end = i + 1
                chunk_sents = sentences[start:end]
                has_anchor = any(j in anchors for j in range(start, end))
                chunks.append(Chunk(
                    sentences=chunk_sents,
                    start_idx=start,
                    end_idx=end - 1,
                    has_anchor=has_anchor,
                ))
                start = end
        return chunks

    def _apply_length_filter(self, chunks: List[Chunk]) -> List[Chunk]:
        """Keep only chunks with anchor content and valid length."""
        filtered = []
        for chunk in chunks:
            n = len(chunk.sentences)
            if n < self.min_sentences:
                continue
            if n > self.max_sentences:
                # Split oversized chunks at midpoint
                mid = n // 2
                filtered.append(Chunk(
                    sentences=chunk.sentences[:mid],
                    start_idx=chunk.start_idx,
                    end_idx=chunk.start_idx + mid - 1,
                    has_anchor=chunk.has_anchor,
                ))
                filtered.append(Chunk(
                    sentences=chunk.sentences[mid:],
                    start_idx=chunk.start_idx + mid,
                    end_idx=chunk.end_idx,
                    has_anchor=chunk.has_anchor,
                ))
            else:
                filtered.append(chunk)
        return filtered
