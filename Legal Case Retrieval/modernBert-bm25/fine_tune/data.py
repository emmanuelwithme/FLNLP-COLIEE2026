"""資料準備、BM25 epoch 重抽、Dataset 與 collator。沒有模型動態負採樣。"""
from __future__ import annotations

import hashlib
import copy
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import (DEBUG_TRAIN_PAIRS, DEBUG_VALID_QUERIES, DEBUG_CANDIDATES,
                    digest, read_json, write_json)


def case_id(value):
    value = str(value).removesuffix(".txt")
    if not value.isdigit():
        raise ValueError(f"Invalid case ID: {value!r}")
    return value.zfill(6)


def labels(path):
    raw = read_json(path)
    result = {}
    for q, positives in raw.items():
        qid = case_id(q)
        if qid in result:
            raise ValueError(f"Duplicate normalized query: {qid}")
        result[qid] = sorted({case_id(p) for p in positives})
        if not result[qid] or qid in result[qid]:
            raise ValueError(f"Empty or self-positive label: {qid}")
    return result, sum(len(v) for v in raw.values())


def query_ids(path):
    ids = [case_id(line.split()[0]) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate qids in {path}")
    return sorted(ids)


def build_pools(path, positives, top_k=100, negatives=15):
    ranked = {q: [] for q in positives}
    rank_documents = {q: {} for q in positives}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"Malformed TREC row {path}:{number}")
            qid, pid, rank = case_id(parts[0]), case_id(parts[2]), int(parts[3])
            if rank < 1:
                raise ValueError(f"Invalid rank at {path}:{number}")
            if qid in ranked and rank <= top_k:
                if rank in rank_documents[qid] and rank_documents[qid][rank] != pid:
                    raise ValueError(f"Conflicting documents at rank {rank} for {qid}: {path}:{number}")
                rank_documents[qid][rank] = pid
                ranked[qid].append((rank, pid))
    pools = {}
    for qid, values in sorted(ranked.items()):
        seen = set(positives[qid]) | {qid}
        pool = []
        for _, pid in sorted(values):
            if pid not in seen:
                pool.append(pid)
                seen.add(pid)
        if len(pool) < negatives:
            raise ValueError(f"{qid}: only {len(pool)} legal negatives in raw top-{top_k}; need {negatives}")
        pools[qid] = pool
    return pools


def sample_epoch(positives, pools, epoch_index, seed=289, negatives=15, pairs=None):
    pairs = pairs if pairs is not None else [(q, p) for q in sorted(positives) for p in positives[q]]
    rows = []
    for qid, pid in pairs:
        sample_seed = int(digest([seed, epoch_index, qid, pid]), 16)
        chosen = random.Random(sample_seed).sample(pools[qid], negatives)
        rows.append({"query_id": qid, "positive_id": pid, "negative_ids": chosen})
    audit_samples(rows, positives, pools, negatives)
    return rows


def audit_samples(rows, positives, pools, negatives=15):
    for row in rows:
        q, p, ns = row["query_id"], row["positive_id"], row["negative_ids"]
        if p not in positives[q] or len(ns) != negatives or len(set(ns)) != negatives:
            raise ValueError(f"Invalid positive/count/duplicate in {q}/{p}")
        if q in ns or set(ns) & set(positives[q]) or not set(ns) <= set(pools[q]):
            raise ValueError(f"Self/positive/out-of-pool negative in {q}/{p}")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(config, write=True):
    doc_dir = Path(config["document_dir"])
    files = sorted(doc_dir.glob("*.txt"))
    candidates = [case_id(p.stem) for p in files]
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("Candidate corpus is empty or has duplicate normalized IDs")
    for p in files:
        if p.name != case_id(p.stem) + ".txt":
            raise ValueError(f"Document filenames must use six-digit IDs: {p}")
    train, raw_train = labels(config["train_labels"])
    valid, raw_valid = labels(config["valid_labels"])
    if set(train) & set(valid):
        raise ValueError("Train/valid query overlap")
    for split, rels in (("train", train), ("valid", valid)):
        if set(query_ids(config[f"{split}_qids"])) != set(rels):
            raise ValueError(f"{split}: qid and label keys disagree")
        required = set(rels) | {p for ps in rels.values() for p in ps}
        if required - set(candidates):
            raise ValueError(f"{split}: missing documents {sorted(required - set(candidates))[:10]}")
    train_pools = build_pools(config["bm25_train"], train, config["top_k"], config["negatives"])
    valid_pools = build_pools(config["bm25_valid"], valid, config["top_k"], config["negatives"])
    required_neg = {p for pool in [*train_pools.values(), *valid_pools.values()] for p in pool}
    if required_neg - set(candidates):
        raise ValueError(f"Missing BM25 documents: {sorted(required_neg - set(candidates))[:10]}")
    counts = {"train_queries": len(train), "valid_queries": len(valid),
              "train_pairs": sum(map(len, train.values())), "valid_pairs": sum(map(len, valid.values())),
              "candidates": len(candidates)}
    if counts != config["expected_counts"]:
        raise ValueError(f"Dataset differs from expected snapshot: {counts}")
    file_hashes = {k: sha256_file(config[k]) for k in
                   ("train_labels", "valid_labels", "train_qids", "valid_qids", "bm25_train", "bm25_valid")}
    document_hashes = {p.stem: sha256_file(p) for p in files}
    manifest = {"counts": counts, "raw_train_links": raw_train, "raw_valid_links": raw_valid,
                "source_hashes": file_hashes, "document_hashes": document_hashes,
                "candidate_ids": candidates, "train_pool_range": [min(map(len, train_pools.values())), max(map(len, train_pools.values()))],
                "valid_pool_range": [min(map(len, valid_pools.values())), max(map(len, valid_pools.values()))],
                "protocol": "2025 processed; no scope/content filtering; raw BM25 top100; exclude self and positives"}
    manifest["data_hash"] = digest(manifest)
    bundle = {"manifest": manifest, "train_labels": train, "valid_labels": valid,
              "train_pools": train_pools, "valid_pools": valid_pools}
    if write:
        folder = Path(config["output_dir"]) / "prepared"
        previous = folder / "data_manifest.json"
        if previous.exists() and read_json(previous)["data_hash"] != manifest["data_hash"]:
            raise ValueError("Prepared dataset changed; use a new output directory")
        artifacts = {previous: manifest}
        for key in ("train_pools", "valid_pools"):
            artifacts[folder / f"{key}.json"] = bundle[key]
        artifacts[folder / "fixed_valid.json"] = sample_epoch(valid, valid_pools, -1, config["sampler_seed"])
        # Validate every existing artifact before writing any file. Preparing data
        # with another seed must not silently replace a run's frozen valid set.
        for path, value in artifacts.items():
            if path.exists() and read_json(path) != value:
                raise ValueError(f"Prepared artifact changed: {path}; use a new output directory")
        for path, value in artifacts.items():
            write_json(path, value)
    return bundle


def select_debug_data(config, bundle):
    """只縮小真實資料；負樣本仍取自原始完整 BM25 pools。"""
    result = copy.deepcopy(bundle)
    pairs = [(q, p) for q in sorted(result["train_labels"]) for p in result["train_labels"][q]]
    count = config.get("debug_train_pairs", DEBUG_TRAIN_PAIRS)
    # 舊版 debug checkpoint 的範圍保持 4/256；新 run 明確保存擴大後的範圍。
    valid_count = config.get("debug_valid_queries", 4 if config["mode"] == "debug" else DEBUG_VALID_QUERIES)
    candidate_count = config.get("debug_candidates", 256 if config["mode"] == "debug" else DEBUG_CANDIDATES)
    if count > len(pairs):
        raise ValueError(f"Debug requests {count} pairs, but only {len(pairs)} exist")
    result["pairs"] = pairs[:count]
    result["valid_labels"] = {q: result["valid_labels"][q] for q in sorted(result["valid_labels"])[:valid_count]}
    required = set(result["valid_labels"]) | {p for ps in result["valid_labels"].values() for p in ps}
    if len(required) > candidate_count:
        raise ValueError(f"Selected debug queries and positives exceed {candidate_count} candidates")
    for pid in result["manifest"]["candidate_ids"]:
        if len(required) >= candidate_count:
            break
        required.add(pid)
    if len(required) != candidate_count or len(result["valid_labels"]) != valid_count:
        raise ValueError(f"Debug requires {valid_count} valid queries and {candidate_count} real candidates")
    result["candidate_ids"] = sorted(required)
    return result


class ContrastiveDataset(Dataset):
    """一列是一個 (query, positive, 15 negatives)；每 epoch 替換 rows。"""
    def __init__(self, rows, document_dir):
        self.rows = rows
        self.document_dir = Path(document_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        def text(pid):
            return (self.document_dir / f"{pid}.txt").read_text(encoding="utf-8").strip()

        return {"query_text": text(row["query_id"]), "positive_text": text(row["positive_id"]),
                "negative_texts": [text(pid) for pid in row["negative_ids"]], "sample_ids": row}


class ContrastiveCollator:
    """與 modernBert-fp 相同的 query / positive / negatives 合併 padding。"""
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, batch):
        size = len(batch)
        if any(len(row["negative_texts"]) != 15 for row in batch):
            raise ValueError("Each pair requires 15 negatives")
        texts = ([row["query_text"] for row in batch] + [row["positive_text"] for row in batch]
                 + [text for row in batch for text in row["negative_texts"]])
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        parts = [{k: v[start:end] for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
                 for start, end in ((0, size), (size, 2 * size), (2 * size, len(texts)))]
        return dict(zip(("anchor_input", "positive_input", "negative_input"), parts),
                    labels=torch.zeros(size, dtype=torch.long),
                    sample_ids=[row["sample_ids"] for row in batch if "sample_ids" in row])
