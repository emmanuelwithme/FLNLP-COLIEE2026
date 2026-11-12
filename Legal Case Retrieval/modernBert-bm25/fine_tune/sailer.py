"""Distinct-query logical batches, label mask and Trainer GradCache extension."""
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import TrainerCallback

from config import digest, read_json, write_json
from fine_tune.data import ContrastiveCollator, sample_epoch
from fine_tune.gradcache import cached_backward, masked_inbatch_loss
from fine_tune.fine_tune import BM25NegativeSamplingTrainer
from fine_tune.batch_grouping import SHUFFLE_REPAIR, LEGACY_DEGREE, shuffle_repair_rows, legacy_degree_rows


def prepare_label_mask(config, bundle, tokenizer):
    """Fingerprint exactly the stripped/truncated model input, all corpus IDs.

    Known positives stay in the dataset; other positives and their input
    aliases are removed from each loss denominator. Empty gold/query is an
    explicit preflight error, never silently dropped from validation.
    """
    path = Path(config["output_dir"]) / "prepared/label_mask_manifest.json"
    identity = {"version": 1, "data_hash": bundle["manifest"]["data_hash"],
                "model_name": config["model_name"], "model_revision": config["model_revision"],
                "max_length": config["max_length"], "rule": "stripped-text + truncated-token identity"}
    # Reuse the verified full-corpus fingerprints for new debug runs. Each run
    # still saves its own manifest; cached identity is tied to the data hash.
    root = Path(config["output_dir"])
    if not path.exists() and "_debug" in root.parts:
        ancestor = Path(*root.parts[:root.parts.index("_debug")]) / "prepared/label_mask_manifest.json"
        if ancestor.exists():
            candidate = read_json(ancestor)
            if candidate.get("identity") == identity and candidate.get("hash") == digest({k: v for k, v in candidate.items() if k != "hash"}):
                write_json(path, candidate)
    if path.exists():
        saved = read_json(path)
        if saved.get("identity") != identity or saved.get("hash") != digest({k: v for k, v in saved.items() if k != "hash"}):
            raise ValueError("Label-mask manifest/config/hash changed")
        return saved
    entries = {}
    ids = bundle["manifest"]["candidate_ids"]
    for start in tqdm(range(0, len(ids), 16), desc="Preflight | Token fingerprints", unit="batch", dynamic_ncols=True):
        group = ids[start:start + 16]
        texts = [(Path(config["document_dir"]) / f"{pid}.txt").read_text(encoding="utf-8").strip() for pid in group]
        tokens = tokenizer(texts, truncation=True, max_length=config["max_length"], padding=False)["input_ids"]
        for pid, text, token_ids in zip(group, texts, tokens):
            entries[pid] = {"text_hash": digest(text), "token_hash": digest(token_ids), "valid": bool(text)}
    positives = {**bundle["train_labels"], **bundle["valid_labels"]}
    required = set(positives) | {p for ps in positives.values() for p in ps}
    invalid = sorted(pid for pid in required if not entries[pid]["valid"])
    if invalid:
        raise ValueError(f"Empty query/positive documents: {invalid}; do not silently drop gold")
    result = {"identity": identity, "documents": entries, "known_positives": positives,
              "invalid_candidates": sorted(pid for pid in ids if not entries[pid]["valid"])}
    result["hash"] = digest(result)
    write_json(path, result)
    return result


def label_mask(rows, manifest):
    """Per-query canonicalization; current positive always owns its column.

    Mask all other known positive IDs/input aliases, self IDs/input aliases,
    invalid corpus IDs/empty text, and duplicate IDs/token identities. A gold
    alias is never relabelled as a negative. Remaining unlabelled relevant
    cases cannot be identified without additional annotations.
    """
    if len(rows) != 4 or len({r["query_id"] for r in rows}) != 4:
        raise ValueError("A logical batch requires 4 distinct query IDs")
    documents = manifest["documents"]
    passages = [p for row in rows for p in [row["positive_id"], *row["negative_ids"]]]
    if any(len(r["negative_ids"]) != 15 for r in rows):
        raise ValueError("Each query requires 1 positive and 15 sampled negatives")
    allowed = torch.zeros((4, 64), dtype=torch.bool)
    targets = torch.arange(4, dtype=torch.long) * 16
    stats = Counter()
    for index, row in enumerate(rows):
        qid, positive = row["query_id"], row["positive_id"]
        target = int(targets[index])
        gold = set(manifest["known_positives"].get(qid, []))
        if positive not in gold or positive == qid or not documents.get(positive, {}).get("valid", False):
            raise ValueError(f"Invalid designated positive: {qid}/{positive}")
        gold_keys = {documents[p]["token_hash"] for p in gold}
        query_key = documents[qid]["token_hash"]
        # Reserve the target FIRST, even if a duplicate appears earlier.
        seen = {documents[positive]["token_hash"]}
        for column, pid in enumerate(passages):
            if column == target:
                allowed[index, column] = True
                continue
            item = documents.get(pid)
            if not item or not item["valid"]:
                stats["invalid"] += 1
            elif pid == qid or item["token_hash"] == query_key:
                stats["self_or_alias"] += 1
            elif pid in gold or item["token_hash"] in gold_keys:
                stats["known_positive_or_alias"] += 1
            elif item["token_hash"] in seen:
                stats["duplicate"] += 1
            else:
                allowed[index, column] = True
                seen.add(item["token_hash"])
    if (allowed.sum(1) < 2).any():
        raise ValueError("Label mask leaves a query without legal negatives")
    stats.update({"allowed": int(allowed.sum()), "masked": int((~allowed).sum()),
                  "queries": 4, "logical_batches": 1})
    return allowed, targets, dict(stats)


def logical_epoch_rows(labels, pools, epoch, seed, pairs=None, strategy=SHUFFLE_REPAIR):
    """Negative draws stay independent of grouping/padding/repair RNG streams."""
    original = sample_epoch(labels, pools, epoch, seed, pairs=pairs)
    if strategy == SHUFFLE_REPAIR:
        return shuffle_repair_rows(original, epoch, seed)
    if strategy == LEGACY_DEGREE:
        return legacy_degree_rows(original, epoch, seed)
    raise ValueError(f"Unknown batch grouping strategy: {strategy}")


def debug_pairs(labels, count):
    """Spread a small debug profile across queries, preserving distinct-q batches."""
    result = []
    for offset in range(max(map(len, labels.values()))):
        for q in sorted(labels):
            if offset < len(labels[q]):
                result.append((q, labels[q][offset]))
                if len(result) == count:
                    return result
    raise ValueError("Insufficient debug training pairs")


class LogicalBatchDataset(Dataset):
    def __init__(self, rows, document_dir):
        self.rows, self.document_dir = rows, Path(document_dir)

    def __len__(self):
        return len(self.rows) // 4

    def __getitem__(self, index):
        rows = self.rows[index * 4:(index + 1) * 4]
        ids = [r["query_id"] for r in rows] + [p for r in rows for p in [r["positive_id"], *r["negative_ids"]]]
        # Read repeated IDs once. Input-alias deduplication below also avoids
        # repeated tokenization/encoder work for different IDs with same tokens.
        texts = {p: (self.document_dir / f"{p}.txt").read_text(encoding="utf-8").strip() for p in set(ids)}
        return {"logical_rows": rows, "document_ids": ids, "texts": [texts[p] for p in ids]}


class SailerCollator:
    def __init__(self, tokenizer, config, manifest):
        self.tokenizer, self.config, self.manifest = tokenizer, config, manifest
        self.pairwise = ContrastiveCollator(tokenizer, config["max_length"])
        if hasattr(tokenizer, "deprecation_warnings"):
            # We already tokenize the WHOLE logical batch in one fast call;
            # the subsequent pads only partition it into encoder chunks.
            tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def __call__(self, batch):
        if "logical_rows" not in batch[0]:
            return self.pairwise(batch)  # unchanged fixed sampled-validation loss
        if len(batch) != 1:
            raise ValueError("Physical DataLoader batch_size must be 1 logical-batch container")
        rows, texts = batch[0]["logical_rows"], batch[0]["texts"]
        allowed, targets, stats = label_mask(rows, self.manifest)
        mapping = list(range(len(texts)))
        if self.config.get("deduplicate_inputs", False):
            ids = batch[0]["document_ids"]
            unique, seen, mapping = [], {}, []
            for pid, text in zip(ids, texts):
                key = self.manifest["documents"][pid]["token_hash"]
                if key not in seen:
                    seen[key] = len(unique)
                    unique.append(text)
                mapping.append(seen[key])
            texts = unique
        chunks = []
        size = self.config["gradcache_chunk_size"]
        # One fast-tokenizer call rather than 34 calls for the old chunk=2.
        tokens = self.tokenizer(texts, padding=False, truncation=True, max_length=self.config["max_length"])
        for start in range(0, len(texts), size):
            items = {k: v[start:start + size] for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
            padded = self.tokenizer.pad(items, padding=True, return_tensors="pt")
            chunks.append(dict(padded))
        return {"chunks": chunks, "allowed": allowed, "labels": targets,
                "representation_indices": torch.tensor(mapping, dtype=torch.long),
                "sample_ids": rows, "mask_stats": stats}


class LabelMaskAudit(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        planned = Counter(digest(trainer.train_dataset.rows[i:i + 4])
                          for i in range(0, len(trainer.train_dataset.rows), 4))
        received = Counter(trainer.epoch_seen_groups)
        if planned != received:
            raise ValueError("Logical groups received by the model differ from planned grouping")
        write_json(trainer.run_dir / "sampling" / f"epoch_{int(state.epoch):02d}_groups_received.json",
                   {"logical_batches": sum(received.values()), "matches_planned": True,
                    "group_sample_hash": digest(sorted(received.elements()))})
        stats = dict(self.trainer.epoch_mask_stats)
        self.trainer.log({f"inbatch/epoch_{key}": value for key, value in stats.items()})
        write_json(self.trainer.run_dir / "sampling" / f"epoch_{int(state.epoch):02d}_label_mask.json", stats)


class SailerGradCacheTrainer(BM25NegativeSamplingTrainer):
    def __init__(self, *args, label_manifest, **kwargs):
        self.label_manifest = label_manifest
        super().__init__(*args, **kwargs)
        if self.config.get("deduplicate_inputs", False):
            # Sharing an input representation is equivalent only for this
            # deterministic encoder/projector, not independent dropout masks.
            stochastic = [name for name, module in self.model.named_modules()
                          if isinstance(module, torch.nn.Dropout) and module.p != 0]
            cfg = self.model.encoder.config
            if stochastic or any(getattr(cfg, name, 0) != 0 for name in ("attention_dropout", "embedding_dropout", "mlp_dropout")):
                raise ValueError("Input deduplication requires zero encoder/projector dropout")
        self.add_callback(LabelMaskAudit(self))

    def resample_epoch(self, epoch):
        rows, packing = logical_epoch_rows(self.bundle["train_labels"], self.bundle["train_pools"], epoch,
                                           self.config["sampler_seed"], self.bundle.get("pairs"),
                                           self.config.get("batch_grouping", LEGACY_DEGREE))
        path = self.run_dir / "sampling" / f"epoch_{epoch + 1:02d}_train.json"
        if path.exists() and read_json(path) != rows:
            raise ValueError("Stored logical epoch samples disagree")
        write_json(path, rows)
        self.train_dataset.rows = rows
        self.epoch_seen_samples, self.epoch_mask_stats = [], Counter()
        self.epoch_seen_groups = []
        current, cumulative = defaultdict(set), defaultdict(set)
        for r in rows:
            current[r["query_id"]].update(r["negative_ids"])
        for previous in range(epoch + 1):
            for r in read_json(self.run_dir / "sampling" / f"epoch_{previous + 1:02d}_train.json"):
                cumulative[r["query_id"]].update(r["negative_ids"])
        denominator = sum(len(self.bundle["train_pools"][q]) for q in current)
        metrics = {"sampling/epoch_pool_coverage": sum(map(len, current.values())) / denominator,
                   "sampling/cumulative_pool_coverage": sum(map(len, cumulative.values())) / denominator,
                   "sampling/self_negative_count": 0, "sampling/positive_leak_count": 0,
                   "sampling/padded_pairs": packing["padded_pairs"]}
        write_json(path.with_suffix(".audit.json"), {"epoch": epoch + 1, "rows": len(rows),
                   "sample_hash": digest(rows), "label_mask_hash": self.label_manifest["hash"], **packing, **metrics})
        self.log(metrics)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.args.gradient_accumulation_steps != 1 or self.accelerator.num_processes != 1:
            raise RuntimeError("GradCache protocol requires one GPU and accumulation=1")
        model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        rows = inputs["sample_ids"]
        self.step_samples = rows
        self.epoch_seen_samples.extend(rows)
        self.epoch_seen_groups.append(digest(rows))
        self.observed_max_length = max(getattr(self, "observed_max_length", 0),
                                       max(c["input_ids"].shape[1] for c in inputs["chunks"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        loss = cached_backward(model.encode, inputs["chunks"],
            lambda reps: masked_inbatch_loss(reps[inputs["representation_indices"]], model.temperature,
                                            inputs["allowed"], inputs["labels"])[0],
            self.accelerator.backward, lambda: torch.autocast("cuda", dtype=torch.float16),
            checkpointing_model=model.encoder)
        self.epoch_mask_stats.update(inputs["mask_stats"])
        self.monitor_metrics = {**getattr(self, "monitor_metrics", {}),
            "inbatch/allowed_negatives_mean": (inputs["allowed"].sum().item() - 4) / 4,
            "inbatch/masked_columns": inputs["mask_stats"]["masked"],
            "gradcache/chunk_documents": self.config["gradcache_chunk_size"],
            "gradcache/unique_documents": sum(len(c["input_ids"]) for c in inputs["chunks"]),
            "gradcache/deduplicated_documents": 68 - sum(len(c["input_ids"]) for c in inputs["chunks"]),
            "gradcache/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gradcache/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        return loss

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        write_json(Path(output_dir or self.args.output_dir) / "gradcache_state.json",
                   {"label_mask_hash": self.label_manifest["hash"], "logical_batch_size": 4,
                    "chunk_documents": self.config["gradcache_chunk_size"],
                    "deduplicate_inputs": self.config.get("deduplicate_inputs", False),
                    "gradient_checkpointing": self.config["gradient_checkpointing"],
                    "batch_grouping": self.config.get("batch_grouping", LEGACY_DEGREE),
                    "epoch_mask_stats": dict(self.epoch_mask_stats)})
