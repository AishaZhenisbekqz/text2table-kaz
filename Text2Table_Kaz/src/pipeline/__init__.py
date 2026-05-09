from .segmenter import KazakhSegmenter
from .embedder import SentenceEmbedder
from .anchor_detector import AnchorDetector
from .chunker import SemanticChunker
from .clusterer import ThematicClusterer
from .insight_generator import InsightGenerator
from .tuple_extractor import TupleExtractor
from .table_assembler import TableAssembler

class Text2TablePipeline:
    """
    End-to-end Kazakh text-to-table generation pipeline.
    Implements the 7-stage architecture from:
      Ospan et al., IEEE Access 2024 — doi:10.1109/ACCESS.2024.0429000
    """

    def __init__(
        self,
        base_model: str,
        lora_adapter: str = None,
        regime: str = "dynamic",
        device: str = "auto",
        theta: float = 0.72,
        kmeans_delta: float = 0.05,
        self_consistency_m: int = 5,
    ):
        self.regime = regime
        self.segmenter = KazakhSegmenter()
        self.embedder = SentenceEmbedder()
        self.anchor_detector = AnchorDetector()
        self.chunker = SemanticChunker(theta=theta)
        self.clusterer = ThematicClusterer(delta=kmeans_delta)
        self.insight_generator = InsightGenerator(
            base_model=base_model,
            lora_adapter=lora_adapter,
            device=device,
            m=self_consistency_m,
        )
        self.tuple_extractor = TupleExtractor()
        self.assembler = TableAssembler(regime=regime)

    @classmethod
    def from_pretrained(cls, base_model: str, lora_adapter: str, regime: str = "dynamic", **kwargs):
        return cls(base_model=base_model, lora_adapter=lora_adapter, regime=regime, **kwargs)

    def __call__(self, text: str) -> str:
        sentences = self.segmenter.segment(text)
        embeddings = self.embedder.encode(sentences)
        anchors = self.anchor_detector.detect(sentences)
        chunks = self.chunker.chunk(sentences, embeddings, anchors)
        clusters = self.clusterer.cluster(chunks, embeddings)
        insights = self.insight_generator.generate(clusters)
        tuples = self.tuple_extractor.extract(insights)
        return self.assembler.assemble(tuples)
