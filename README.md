# Text2Table_Kaz 🇰🇿

**Dual-Regime Text-to-Table Generation for the Kazakh Language Using Parameter-Efficient Large Language Models**

[![Paper](https://img.shields.io/badge/IEEE%20Access-2024-blue)](https://doi.org/10.1109/ACCESS.2024.0429000)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Adapters-orange)](https://huggingface.co)

> *Ospan A., Mansurova M., Sailau A., Sarsembayeva T., Mosavi A.*
> Al-Farabi Kazakh National University · Obuda University
> Supported by Grant BR24993001 (Committee of Science, MES RK)

---

## Overview

This repository contains the full implementation of a **morphology-aware, dual-regime framework** for converting Kazakh-language journalistic texts into structured relational tables.

| Regime | Schema | Description |
|--------|--------|-------------|
| **Static** | Fixed 5-column | `Sector · Event · Indicator · Period · Additional` |
| **Dynamic** | Inductively inferred | Model determines optimal column set per document |

### Key Results

| Metric | Static | Dynamic |
|--------|--------|---------|
| Coverage | 0.692 | **0.718** |
| Accuracy | 0.740 | **0.761** |
| Compression | **0.969** | 0.954 |
| Structure | **0.989** | 0.976 |
| Journalistic Value | 0.915 | **0.932** |
| **Total (/5)** | 4.305 | **4.341** |

### Generalization Gap Comparison

| Model | Params | Δ_gap (epoch 3) | Val Loss |
|-------|--------|-----------------|----------|
| **Qwen3.5-4B (ours)** | **4B** | **−0.064** | **0.482** |
| Qwen3-14B | 14B | 0.080 | 0.906 |
| Llama-3.1-8B | 8B | 0.101 | 0.899 |

Our 4B adapter achieves **comparable validation loss** to 8–14B baselines with ~3× fewer parameters.

---

## Pipeline Architecture

```
Input Text (Kazakh)
       │
       ▼
┌─────────────────────┐
│  1. Sentence Split  │  Kazakh-aware boundary detection
│     + Embedding     │  paraphrase-multilingual-mpnet-base-v2
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  2. Anchor Filter   │  Lexical-regex + NER + Morphosyntax
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  3. Cosine Chunking │  θ = 0.72, chunks: 3–15 sentences
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  4. k-Means Cluster │  Elbow method, δ = 0.05
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  5. CoT + Self-     │  m=5 completions, majority vote
│     Consistency     │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  6. Tuple Extraction│  Dep. parse + LLM verification
│     & Normalization │  Lemma · Predicate · Coref · Num
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  7. Table Assembly  │  Static (5-col) or Dynamic schema
└─────────────────────┘
```

---

## Repository Structure

```
Text2Table_Kaz/
├── src/
│   ├── pipeline/
│   │   ├── segmenter.py          # Kazakh sentence segmentation
│   │   ├── embedder.py           # Multilingual sentence embeddings
│   │   ├── anchor_detector.py    # Factual anchor identification
│   │   ├── chunker.py            # Cosine-threshold semantic chunking
│   │   ├── clusterer.py          # k-Means thematic clustering
│   │   ├── insight_generator.py  # CoT + self-consistency prompting
│   │   ├── tuple_extractor.py    # SPO tuple extraction & normalization
│   │   └── table_assembler.py    # Markdown table assembly
│   ├── training/
│   │   ├── lora_trainer.py       # LoRA fine-tuning (Qwen3.5-4B)
│   │   ├── dataset.py            # Dataset loading & prompt formatting
│   │   └── callbacks.py          # Training callbacks & early stopping
│   ├── evaluation/
│   │   ├── metrics.py            # 5 evaluation metrics
│   │   ├── llm_judge.py          # GPT-4 as judge protocol
│   │   └── human_eval.py         # Human evaluation utilities
│   └── data/
│       ├── crawler.py            # Egemen Qazaqstan async crawler
│       ├── preprocessor.py       # Text cleaning & normalization
│       └── annotation.py         # T³ semi-automated annotation
├── configs/
│   ├── static_regime.yaml
│   ├── dynamic_regime.yaml
│   └── pipeline.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── crawl_data.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pipeline_demo.ipynb
│   └── 03_results_analysis.ipynb
├── tests/
├── docker/
├── assets/
│   └── sample_data.json
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Text2Table_Kaz.git
cd Text2Table_Kaz
pip install -r requirements.txt
```

**Docker (exact replication):**
```bash
docker-compose -f docker/docker-compose.yml up
```

---

## Quick Start

### Inference

```python
from src.pipeline import Text2TablePipeline

pipeline = Text2TablePipeline.from_pretrained(
    base_model="Qwen/Qwen3.5-4B",
    lora_adapter="AishaSailau/qwen3.5-text2table-dynamic",
    regime="dynamic"
)

text = """
Қазақстан Республикасының Ұлттық Банкі 2024 жылдың бірінші тоқсанында
инфляция деңгейі 8,7%-ға дейін төмендегенін хабарлады. Бұл көрсеткіш
алдыңғы жылдың сәйкес кезеңімен салыстырғанда 2,3 пайыздық тармаққа азайды.
"""

table = pipeline(text)
print(table)
```

### Training

```bash
# Dynamic regime (recommended)
python scripts/train.py --config configs/dynamic_regime.yaml --seed 42

# Static regime
python scripts/train.py --config configs/static_regime.yaml --seed 42
```

### Evaluation

```bash
python scripts/evaluate.py \
    --adapter_path ./checkpoints/dynamic_best \
    --test_data ./assets/sample_data.json \
    --regime dynamic \
    --use_llm_judge
```

---

## Pre-trained Adapters

| Adapter | Regime | Val Loss | Link |
|---------|--------|----------|------|
| `text2table-kaz-static` | Static | 0.524 | [🤗 Hugging Face](https://huggingface.co/AishaSailau/qwen3.5-text2table-static) |
| `text2table-kaz-dynamic` | Dynamic | 0.482 | [🤗 Hugging Face](https://huggingface.co/AishaSailau/qwen3.5-text2table-dynamic) |

---

## Dataset

| Split | Instances | Description |
|-------|-----------|-------------|
| Train static | 17,500 | Fixed 5-column schema |
| Train dynamic | 17,500 | Text-derived schema |
| Validation | 3,000 | 1,500 per regime |
| Gold Standard | 1,000 | Stratified, human-validated |
| Public sample | 100 | Anonymized, in `assets/` |

Full corpus: 35,000 instances from *Egemen Qazaqstan* (2017–2025).
IAA: κ = 0.84 (static), κ = 0.79 (dynamic).

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Base model | Qwen3.5-4B |
| LoRA rank *r* | 32 |
| LoRA alpha *α* | 64 |
| Trainable params | ~2% |
| Learning rate | 1e-4 (cosine) |
| Warmup steps | 200 |
| Batch size | 32 |
| Precision | bfloat16 |
| Epochs | 3 (patience=2) |
| Hardware | 1× NVIDIA A100 80GB |

---

## License

Apache 2.0. Supported by Grant **BR24993001** (MES RK).
