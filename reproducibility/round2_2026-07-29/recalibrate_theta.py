import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_tuning_corpus(args) -> list[dict]:
    if args.tuning_corpus:
        rows = load_jsonl(args.tuning_corpus)
        id_field = "id"
    elif args.tuning_corpus_hf:
        from datasets import load_dataset

        token = os.environ.get("HF_TOKEN")
        ds = load_dataset(args.tuning_corpus_hf, token=token)

        if args.tuning_corpus_hf_split:
            split_name = args.tuning_corpus_hf_split
        else:
            available = list(ds.keys())
            raise SystemExit(
                f"Укажите --tuning-corpus-hf-split. Доступные сплиты: {available}"
            )

        split_data = ds[split_name]
        text_field = args.tuning_corpus_hf_text_field
        id_field_name = args.tuning_corpus_hf_id_field

        rows = []
        for i, row in enumerate(split_data):
            row_id = str(row[id_field_name]) if id_field_name and id_field_name in row else str(i)
            rows.append({"id": row_id, "text": row[text_field]})
        id_field = "id"
    else:
        raise SystemExit("Укажите либо --tuning-corpus, либо --tuning-corpus-hf.")

    if args.tuning_sample_n:
        gold_rows = load_jsonl(args.gold_benchmark)
        gold_ids = {str(r["id"]) for r in gold_rows}
        gold_texts = {r["text"] for r in gold_rows}

        pre_filter_n = len(rows)
        rows = [r for r in rows if r["id"] not in gold_ids and r["text"] not in gold_texts]
        excluded = pre_filter_n - len(rows)
        if excluded:
            print(f"Исключено {excluded} документов из-за пересечения с Gold-1000 "
                  f"(до подвыборки).")

        if len(rows) < args.tuning_sample_n:
            raise SystemExit(
                f"После исключения пересечений с Gold-1000 осталось только "
                f"{len(rows)} документов, нужно {args.tuning_sample_n}."
            )

        rng = random.Random(args.tuning_sample_seed)
        rows = rng.sample(rows, args.tuning_sample_n)
        print(f"ОТКЛОНЕНИЕ ОТ ОРИГИНАЛЬНОЙ МЕТОДОЛОГИИ (задокументировать в Methods): "
              f"исходная 500-документная calibration-выборка авторов недоступна "
              f"отдельно (в {args.tuning_corpus_hf or args.tuning_corpus} есть только "
              f"общий train-сплит). Использована новая случайная выборка "
              f"{args.tuning_sample_n} документов (seed={args.tuning_sample_seed}) "
              f"после исключения пересечений с Gold-1000.")

    return rows


def harmonic_score(
    sentences_by_chunk: list[list[np.ndarray]],
    boundary_embeddings: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    coherences = []
    for chunk_embeddings in sentences_by_chunk:
        if len(chunk_embeddings) < 2:
            continue
        sims = []
        for i in range(len(chunk_embeddings)):
            for j in range(i + 1, len(chunk_embeddings)):
                sims.append(float(np.dot(chunk_embeddings[i], chunk_embeddings[j])))
        coherences.append(np.mean(sims))

    separations = []
    for last_emb, first_emb in boundary_embeddings:
        sim = float(np.dot(last_emb, first_emb))
        separations.append(1.0 - sim)

    coherence = float(np.mean(coherences)) if coherences else None
    separation = float(np.mean(separations)) if separations else None

    if coherence is None or separation is None or (coherence + separation) == 0:
        return coherence, separation, None

    harmonic = 2 * coherence * separation / (coherence + separation)
    return coherence, separation, harmonic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--tuning-corpus", default=None,
                         help="Локальный JSONL с полями id/text.")
    parser.add_argument("--tuning-corpus-hf", default=None,
                         help="Имя датасета на Hugging Face, например AishaSailau/train_table.")
    parser.add_argument("--tuning-corpus-hf-split", default=None)
    parser.add_argument("--tuning-corpus-hf-text-field", default="text")
    parser.add_argument("--tuning-corpus-hf-id-field", default="url",
                         help="Поле-идентификатор в HF датасете (train_table не имеет 'id', но имеет 'url', "
                              "и партиционирование в статье делалось по нормализованному URL).")
    parser.add_argument("--tuning-sample-n", type=int, default=None,
                         help="Если задано, взять случайную подвыборку этого размера из загруженного "
                              "сплита ПОСЛЕ исключения пересечений с --gold-benchmark. Нужно, когда "
                              "отдельного calibration-сплита нет (например, train_table имеет только "
                              "один сплит 'train' с 10000 строк).")
    parser.add_argument("--tuning-sample-seed", type=int, default=42)
    parser.add_argument("--gold-benchmark", required=True,
                         help="benchmark_gold_1000.jsonl -- используется ТОЛЬКО для проверки на пересечение ID/текстов, не для калибровки.")
    parser.add_argument("--theta-grid", type=float, nargs="+",
                         default=[0.60, 0.65, 0.70, 0.72, 0.75, 0.80])
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=15)
    parser.add_argument("--output", default="theta_recalibration_report.json")
    args = parser.parse_args()

    sys.path.insert(0, args.repo_dir)
    from src.pipeline.segmenter import KazakhSegmenter
    from src.pipeline.embedder import SentenceEmbedder
    from src.pipeline.chunker import SemanticChunker

    tuning_rows = load_tuning_corpus(args)
    gold_rows = load_jsonl(args.gold_benchmark)

    tuning_ids = {str(r.get("id", i)) for i, r in enumerate(tuning_rows)}
    gold_ids = {str(r["id"]) for r in gold_rows}
    tuning_texts = {r["text"] for r in tuning_rows}
    gold_texts = {r["text"] for r in gold_rows}

    id_overlap = tuning_ids & gold_ids
    text_overlap = tuning_texts & gold_texts
    if id_overlap or text_overlap:
        print(f"ОСТАНОВКА: обнаружено пересечение с Gold-1000.")
        print(f"  Пересекающихся ID: {len(id_overlap)}")
        print(f"  Пересекающихся текстов (дословно): {len(text_overlap)}")
        print("Калибровка на пересекающихся данных недействительна (data leakage).")
        sys.exit(1)
    print(f"Проверка на пересечение пройдена: 0 общих ID, 0 общих текстов.")
    print(f"Tuning-корпус: {len(tuning_rows)} документов. Gold-1000: {len(gold_rows)} документов (не используется для калибровки).")
    if len(tuning_rows) != 500:
        print(f"ПРЕДУПРЕЖДЕНИЕ: в рукописи калибровка проводилась на 500 документах, "
              f"у вас {len(tuning_rows)}. Если это не тот сплит -- проверьте "
              f"--tuning-corpus-hf-split перед тем как использовать эти результаты в статье.")

    segmenter = KazakhSegmenter()
    embedder = SentenceEmbedder()

    per_doc_sentences = []
    per_doc_embeddings = []
    for row in tuning_rows:
        sentences = segmenter.segment(row["text"])
        if len(sentences) < 2:
            continue
        embeddings = np.asarray(embedder.encode(sentences))
        per_doc_sentences.append(sentences)
        per_doc_embeddings.append(embeddings)

    print(f"Документов с >=2 предложениями: {len(per_doc_sentences)}")

    results = []
    for theta in args.theta_grid:
        chunker = SemanticChunker(
            theta=theta,
            min_sentences=args.min_sentences,
            max_sentences=args.max_sentences,
        )
        n_chunks_per_doc = []
        n_docs_with_zero_chunks = 0
        chunk_sizes = []
        all_chunk_embeddings = []       # список списков embeddings по chunk
        all_boundary_pairs = []         # список (last_emb, first_emb) для соседних chunks

        for sentences, embeddings in zip(per_doc_sentences, per_doc_embeddings):
            chunks = chunker.chunk(sentences, embeddings, anchors=set())
            n_chunks_per_doc.append(len(chunks))
            if len(chunks) == 0:
                n_docs_with_zero_chunks += 1
            chunk_sizes.extend(len(c.sentences) for c in chunks)

            for c in chunks:
                chunk_embs = [embeddings[i] for i in range(c.start_idx, c.end_idx + 1)]
                all_chunk_embeddings.append(chunk_embs)

            for k in range(len(chunks) - 1):
                last_emb = embeddings[chunks[k].end_idx]
                first_emb = embeddings[chunks[k + 1].start_idx]
                all_boundary_pairs.append((last_emb, first_emb))

        coherence, separation, harmonic = harmonic_score(all_chunk_embeddings, all_boundary_pairs)

        n_docs = len(per_doc_sentences)
        results.append({
            "theta": theta,
            "n_docs": n_docs,
            "pct_docs_zero_chunks": n_docs_with_zero_chunks / n_docs if n_docs else None,
            "mean_chunks_per_doc": float(np.mean(n_chunks_per_doc)) if n_chunks_per_doc else None,
            "mean_chunk_size": float(np.mean(chunk_sizes)) if chunk_sizes else None,
            "median_chunk_size": float(np.median(chunk_sizes)) if chunk_sizes else None,
            "total_chunks": len(chunk_sizes),
            "intra_chunk_coherence": coherence,
            "inter_chunk_separation": separation,
            "harmonic_mean": harmonic,
        })
        print(f"theta={theta:.2f}: "
              f"zero_chunks={n_docs_with_zero_chunks}/{n_docs} ({n_docs_with_zero_chunks / n_docs:.1%}), "
              f"coherence={coherence}, separation={separation}, harmonic_mean={harmonic}")

    valid_results = [r for r in results if r["harmonic_mean"] is not None]
    if not valid_results:
        best = None
        print("\nНИ ОДНО theta не дало вычислимого harmonic_mean (недостаточно chunks/границ). "
              "Нужно смотреть diagnostics вручную.")
    else:
        best = max(valid_results, key=lambda r: r["harmonic_mean"])
        print(f"\nЛУЧШЕЕ theta по критерию рукописи (max harmonic mean): "
              f"theta={best['theta']}, harmonic_mean={best['harmonic_mean']:.4f}")

    report = {
        "tuning_corpus": args.tuning_corpus or f"hf:{args.tuning_corpus_hf}:{args.tuning_corpus_hf_split}",
        "tuning_corpus_n_documents": len(tuning_rows),
        "tuning_sample_n": args.tuning_sample_n,
        "tuning_sample_seed": args.tuning_sample_seed,
        "methodology_deviation": (
            "Original 500-document calibration split from Ospan et al. (2024) is not "
            "separately available; AishaSailau/train_table exposes only a single "
            "'train' split of 10,000 rows. A fresh random sample was drawn instead "
            "(seed as recorded above), after excluding any rows overlapping "
            "benchmark_gold_1000.jsonl by id/url or by exact text. This must be "
            "stated explicitly in Methods / response letter."
        ) if args.tuning_sample_n else None,
        "gold_benchmark_checked_against": args.gold_benchmark,
        "id_overlap_with_gold": len(id_overlap),
        "text_overlap_with_gold": len(text_overlap),
        "original_theta_from_paper": 0.72,
        "theta_grid": args.theta_grid,
        "selection_criterion": (
            "Maximize harmonic mean of intra-chunk coherence and inter-chunk "
            "separation, per manuscript Section 3.2 / Eq. (4). Operationalization: "
            "coherence = mean pairwise cosine similarity of sentence embeddings "
            "within each chunk (chunks with >=2 sentences), averaged over chunks; "
            "separation = mean (1 - cosine similarity) between the last sentence "
            "embedding of a chunk and the first sentence embedding of the next "
            "chunk, averaged over all adjacent chunk boundaries. This "
            "operationalization is NOT verbatim from the manuscript (which gives "
            "only a verbal description) -- verify against original authors' code "
            "if a more precise formula exists."
        ),
        "results_per_theta": results,
        "recommended_theta": best["theta"] if best else None,
        "recommended_theta_harmonic_mean": best["harmonic_mean"] if best else None,
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт записан: {args.output}")
    if best and best["theta"] != 0.72:
        print(f"ВНИМАНИЕ: рекомендованное theta ({best['theta']}) отличается от значения в "
              f"рукописи (0.72). Это отклонение нужно явно задокументировать в Methods и "
              f"response letter, с обоснованием (version drift эмбеддинг-модели/библиотек).")


if __name__ == "__main__":
    main()

