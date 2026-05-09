"""
Multilingual sentence embeddings using paraphrase-multilingual-mpnet-base-v2.
Ospan et al. (2024), Equation (2): e_i = f_embed(s_i; Φ)
"""

import numpy as np
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


class SentenceEmbedder:
    """
    Encodes sentences into 768-dim L2-normalized embeddings.

    Mean pooling over final-layer hidden states → L2 normalization
    onto the unit hypersphere S^767 (ensures cosine sim = dot product).
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = 768

    def encode(self, sentences: List[str]) -> np.ndarray:
        """
        Encode a list of sentences.

        Args:
            sentences: List of sentence strings.

        Returns:
            np.ndarray of shape (n, 768), L2-normalized.
        """
        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,   # L2-normalize → cosine = dot product
            batch_size=64,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalized vectors.
        Since embeddings are on the unit hypersphere: sim = a · b
        (Equation 3 in paper).
        """
        return float(np.dot(a, b))
