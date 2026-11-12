"""與 modernBert-fp 同名入口：Top-5 排名與 full-valid micro F1；訓練亦共用這裡。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tqdm.auto import tqdm

from config import write_json
from inference import (add_evaluation_arguments, encode_corpus, evaluation_config,
                       evaluation_context, load_model)


def score_predictions(relevance, predictions):
    tp = fp = fn = 0
    per_query = {}
    for qid, positives in relevance.items():
        gold, predicted = set(positives), set(predictions.get(qid, []))
        a, b, c = len(gold & predicted), len(predicted - gold), len(gold - predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        per_query[qid] = {"tp": a, "fp": b, "fn": c,
                          "predictions": predictions.get(qid, []), "positives": sorted(gold)}
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "num_queries": len(relevance),
            "missing_predictions": sum(not predictions.get(q) for q in relevance)}, per_query


def top_predictions(qid, ids, scores, top_k=5):
    if len(ids) != len(scores):
        raise ValueError("Candidate ID and score counts disagree")
    if any(not math.isfinite(float(score)) for score in scores):
        raise FloatingPointError("Non-finite retrieval similarity")
    return [pid for pid, score in sorted(((pid, float(s)) for pid, s in zip(ids, scores) if pid != qid),
                                        key=lambda item: (-item[1], item[0]))[:top_k]]


def rank_embeddings(packet, config, relevance, output_prefix, position=0):
    candidates, ids = packet["candidate_ids"], packet["vector_ids"]
    vectors = packet["vectors"].float().cpu()
    if not candidates or len(ids) != len(set(ids)) or len(candidates) != len(set(candidates)):
        raise ValueError("Empty/duplicate embedding IDs")
    if vectors.ndim != 2 or len(vectors) != len(ids) or not torch.isfinite(vectors).all():
        raise ValueError("Invalid embedding vectors")
    locations = {pid: i for i, pid in enumerate(ids)}
    if not set(candidates) <= locations.keys() or not set(relevance) <= locations.keys():
        raise ValueError("Embeddings omit candidate/query documents")
    if {p for ps in relevance.values() for p in ps} - set(candidates):
        raise ValueError("Candidate embeddings omit relevant documents")
    candidate_vectors = vectors[[locations[pid] for pid in candidates]]
    candidate_locations = {pid: i for i, pid in enumerate(candidates)}
    predictions, lines = {}, []
    for qid in tqdm(sorted(relevance), desc=f"{Path(output_prefix).name} | Ranking valid queries",
                    unit="query", position=position, dynamic_ncols=True, disable=False):
        scores = torch.mv(candidate_vectors, vectors[locations[qid]]).tolist()
        predictions[qid] = top_predictions(qid, candidates, scores, config["retrieval_top_k"])
        for rank, pid in enumerate(predictions[qid], 1):
            lines.append(f"{qid} Q0 {pid} {rank} {scores[candidate_locations[pid]]:.10g} bm25_epoch_resample")
    metrics, per_query = score_predictions(relevance, predictions)
    metrics.update({"candidate_count": len(candidates), "mode": config["mode"],
                    "top_k": config["retrieval_top_k"], "similarity_dtype": "float32"})
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(prefix.with_suffix(".json"), {"metrics": metrics, "per_query": per_query})
    return metrics


def evaluate_model_retrieval(model, tokenizer, config, relevance, candidate_ids, output_prefix, *, progress_position=0):
    packet = encode_corpus(model, tokenizer, config, relevance, candidate_ids,
                           Path(output_prefix).name, progress_position)
    return rank_embeddings(packet, config, relevance, output_prefix, progress_position)


def evaluate_checkpoint(config, checkpoint="best", legacy=False, destination=None, embeddings=None):
    checkpoint, bundle, metadata = evaluation_context(config, checkpoint, legacy)
    prefix = Path(destination) if destination else Path(config["output_dir"]) / "standalone" / checkpoint.name
    candidates = bundle.get("candidate_ids", bundle["manifest"]["candidate_ids"])
    if embeddings:
        packet = torch.load(embeddings, map_location="cpu", weights_only=True)
        if packet.get("metadata") != metadata or packet["candidate_ids"] != sorted(candidates):
            raise ValueError("Embedding checkpoint/config/data hash mismatch; re-run inference.py")
        result = rank_embeddings(packet, config, bundle["valid_labels"], prefix)
    else:
        model, tokenizer = load_model(config, checkpoint, legacy)
        result = evaluate_model_retrieval(model, tokenizer, config, bundle["valid_labels"], candidates, prefix)
    write_json(prefix.with_suffix(".manifest.json"), {**metadata, "legacy": legacy, "metrics": result})
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument("--embeddings", help="Optional inference.py output; checked against checkpoint/data/config")
    args = parser.parse_args()
    config = evaluation_config(parser, args)
    return evaluate_checkpoint(config, args.checkpoint, args.legacy, args.destination, args.embeddings)


if __name__ == "__main__":
    main()
