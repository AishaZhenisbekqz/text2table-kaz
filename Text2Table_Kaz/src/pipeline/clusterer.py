"""
Thematic clustering of semantic chunks via k-Means.
Ospan et al. (2024), Equations (5)–(7), Section III-E.

Optimal k* determined by elbow on inertia curve:
    k* = min{k : |I(k) - I(k+1)| < δ · I(1)},  δ = 0.05
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from .chunker import Chunk


@dataclass
class ThematicCluster:
    cluster_id: int
    chunks: List[Chunk] = field(default_factory=list)
    centroid: np.ndarray = None

    @property
    def text(self) -> str:
        """Concatenate all sentences in cluster."""
        sentences = []
        for chunk in self.chunks:
            sentences.extend(chunk.sentences)
        return " ".join(sentences)


class ThematicClusterer:
    """
    Groups semantically similar chunks into thematic clusters.
    Each chunk is represented by mean-pooled sentence embeddings (Eq. 5).
    """

    def __init__(
        self,
        delta: float = 0.05,
        max_k: int = 10,
        min_k: int = 2,
        random_state: int = 42,
    ):
        """
        Args:
            delta: Elbow sensitivity (Equation 7).
            max_k: Maximum number of clusters to consider.
            min_k: Minimum number of clusters.
            random_state: For k-means++ reproducibility.
        """
        self.delta = delta
        self.max_k = max_k
        self.min_k = min_k
        self.random_state = random_state

    def cluster(
        self,
        chunks: List[Chunk],
        sentence_embeddings: np.ndarray,
    ) -> List[ThematicCluster]:
        """
        Cluster chunks into thematic groups.

        Args:
            chunks: List of Chunk objects.
            sentence_embeddings: (n_sentences, 768) embeddings for all sentences.

        Returns:
            List of ThematicCluster objects.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        if len(chunks) == 0:
            return []

        # Compute chunk-level pooled vectors (Equation 5)
        chunk_vectors = self._pool_chunk_vectors(chunks, sentence_embeddings)

        if len(chunks) <= self.min_k:
            # Too few chunks to cluster meaningfully
            return [ThematicCluster(cluster_id=0, chunks=chunks,
                                    centroid=chunk_vectors.mean(axis=0))]

        k_star = self._find_optimal_k(chunk_vectors)
        labels = self._fit_kmeans(chunk_vectors, k_star)

        clusters = self._build_clusters(chunks, labels, chunk_vectors, k_star)
        return clusters

    def _pool_chunk_vectors(
        self, chunks: List[Chunk], sent_embs: np.ndarray
    ) -> np.ndarray:
        """Mean-pool sentence embeddings per chunk (Equation 5)."""
        vectors = []
        idx = 0
        for chunk in chunks:
            n = len(chunk.sentences)
            v = sent_embs[chunk.start_idx:chunk.end_idx + 1].mean(axis=0)
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    def _find_optimal_k(self, vectors: np.ndarray) -> int:
        """Elbow method on inertia curve (Equations 6–7)."""
        k_range = range(self.min_k, min(self.max_k + 1, len(vectors)))
        inertias = {}
        for k in k_range:
            km = KMeans(n_clusters=k, init="k-means++",
                        random_state=self.random_state, n_init=10)
            km.fit(vectors)
            inertias[k] = km.inertia_

        baseline = inertias[self.min_k]
        prev_k = self.min_k
        for k in sorted(inertias.keys())[1:]:
            drop = abs(inertias[prev_k] - inertias[k])
            if drop < self.delta * baseline:
                return prev_k
            prev_k = k
        return prev_k

    def _fit_kmeans(self, vectors: np.ndarray, k: int) -> np.ndarray:
        km = KMeans(n_clusters=k, init="k-means++",
                    random_state=self.random_state, n_init=10)
        return km.fit_predict(vectors)

    def _build_clusters(
        self,
        chunks: List[Chunk],
        labels: np.ndarray,
        vectors: np.ndarray,
        k: int,
    ) -> List[ThematicCluster]:
        cluster_map: Dict[int, ThematicCluster] = {}
        for i, (chunk, label) in enumerate(zip(chunks, labels)):
            if label not in cluster_map:
                cluster_map[label] = ThematicCluster(cluster_id=int(label))
            cluster_map[label].chunks.append(chunk)

        # Compute centroids
        for label, cluster in cluster_map.items():
            idxs = [i for i, l in enumerate(labels) if l == label]
            cluster.centroid = vectors[idxs].mean(axis=0)

        return list(cluster_map.values())
