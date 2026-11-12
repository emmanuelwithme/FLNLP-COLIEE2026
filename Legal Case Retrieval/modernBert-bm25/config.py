"""本實驗唯一設定來源；不讀取根目錄 .env，也不依賴 modernBert-fp。"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs/coliee2025.json"
IMPLEMENTATION = "modernBert-fp-bm25-v2"
DEBUG_TRAIN_PAIRS = 100
DEBUG_VALID_QUERIES = 20
DEBUG_CANDIDATES = 1000
DEBUG_FIELDS = {"debug_train_pairs", "debug_valid_queries", "debug_candidates"}
SAILER_FIELDS = {"training_method", "logical_batch_size", "gradcache_chunk_size", "deduplicate_inputs", "batch_grouping"}
SAILER_IMPLEMENTATION = "modernBert-sailer-gradcache-v3"
SAILER_LEGACY_IMPLEMENTATION = "modernBert-sailer-gradcache-v2"
PATH_KEYS = ("document_dir", "train_qids", "valid_qids", "train_labels", "valid_labels",
             "bm25_train", "bm25_valid", "output_dir")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def implementation_id(config):
    if config.get("training_method") != "sailer_gradcache":
        return IMPLEMENTATION
    return (SAILER_IMPLEMENTATION if config.get("batch_grouping", "legacy_degree_v1") == "shuffle_repair_v1"
            else SAILER_LEGACY_IMPLEMENTATION)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def default_report_directory(kind):
    """Keep new diagnostic runs separate from committed historical evidence."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return REPO_ROOT / "reports" / "modernbert-bm25" / kind / run_id


def load_config(path=DEFAULT_CONFIG, output_dir=None):
    config = read_json(path)
    required = set(read_json(DEFAULT_CONFIG))
    if not isinstance(config, dict) or required - set(config) or set(config) - required - DEBUG_FIELDS - SAILER_FIELDS:
        raise ValueError(f"Config fields must match {DEFAULT_CONFIG.name}; missing/unknown fields are not allowed")
    if config["version"] != 2 or config["year"] != 2025:
        raise ValueError("This implementation requires config version 2 and COLIEE 2025.")
    if config["model_name"] != "answerdotai/ModernBERT-base":
        raise ValueError("Initialize from original answerdotai/ModernBERT-base, not an FP checkpoint.")
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    for key in PATH_KEYS:
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path string")
        p = Path(config[key])
        config[key] = str((REPO_ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if any(config[key] != value for key, value in
           {"batch_size": 1, "negatives": 15, "top_k": 100, "max_length": 4096, "retrieval_top_k": 5}.items()):
        raise ValueError("Controlled protocol: batch_size=1, top_k=100, negatives=15, max_length=4096, retrieval_top_k=5.")
    if config["mode"] not in ("formal", "debug"):
        raise ValueError("mode must be formal or debug")
    if config.get("training_method", "pairwise") not in ("pairwise", "sailer_gradcache"):
        raise ValueError("Unknown training_method")
    if config.get("training_method") == "sailer_gradcache":
        if config.get("logical_batch_size") != 4 or type(config.get("logical_batch_size")) is not int:
            raise ValueError("SAILER GradCache requires logical_batch_size=4")
        chunk = config.get("gradcache_chunk_size")
        if type(chunk) is not int or not 1 <= chunk <= 16:
            raise ValueError("gradcache_chunk_size must be an integer in [1,16]; benchmark memory before increasing it")
        if type(config.get("deduplicate_inputs", False)) is not bool:
            raise ValueError("deduplicate_inputs must be a JSON boolean")
        if config.get("batch_grouping", "legacy_degree_v1") not in ("shuffle_repair_v1", "legacy_degree_v1"):
            raise ValueError("Unknown batch_grouping strategy")
        if config["gradient_accumulation_steps"] != 1:
            raise ValueError("A logical batch already contains 4 queries; gradient_accumulation_steps must be 1")
    elif SAILER_FIELDS & (set(config) - {"training_method"}):
        raise ValueError("logical_batch_size/gradcache_chunk_size require sailer_gradcache")
    for key in ("fp16", "local_files_only", "gradient_checkpointing"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean, not a string or number")
    if not config["fp16"] or config["attention_implementation"] != "flash_attention_2":
        raise ValueError("This experiment uses fp16=true and attention_implementation=flash_attention_2")
    for key in ("batch_size", "eval_batch_size", "retrieval_batch_size", "gradient_accumulation_steps",
                "num_train_epochs", "logging_steps", "early_stopping_patience", "save_total_limit",
                "max_length", "top_k", "negatives", "retrieval_top_k"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "sampler_seed", "dataloader_num_workers"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("learning_rate", "temperature_lr", "initial_temperature", "temperature_min", "temperature_max"):
        if not finite_number(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not config["temperature_min"] < config["temperature_max"] or not config["temperature_min"] <= config["initial_temperature"] <= config["temperature_max"]:
        raise ValueError("initial_temperature must be within the positive temperature bounds")
    if (not finite_number(config["warmup_ratio"]) or not 0 <= config["warmup_ratio"] <= 1
            or not finite_number(config["weight_decay"]) or config["weight_decay"] < 0):
        raise ValueError("warmup_ratio must be in [0,1]; weight_decay must be finite and nonnegative")
    if not isinstance(config["model_revision"], str) or not config["model_revision"].strip():
        raise ValueError("model_revision must be explicitly pinned")
    counts = config["expected_counts"]
    if (not isinstance(counts, dict) or set(counts) != {"train_queries", "valid_queries", "train_pairs", "valid_pairs", "candidates"}
            or any(type(n) is not int or n <= 0 for n in counts.values())):
        raise ValueError("expected_counts must contain positive integer query/pair/candidate counts")
    if "debug_train_pairs" in config:
        count = config["debug_train_pairs"]
        if config["mode"] != "debug" or type(count) is not int or not 8 <= count <= counts["train_pairs"]:
            raise ValueError("debug_train_pairs requires debug mode and 8 <= count <= the formal train-pair count")
    for key, limit in (("debug_valid_queries", counts["valid_queries"]), ("debug_candidates", counts["candidates"])):
        if key in config and (config["mode"] != "debug" or type(config[key]) is not int or not 1 <= config[key] <= limit):
            raise ValueError(f"{key} requires debug mode and a positive integer within the formal data count")
    return config


def small_run_config(formal, output_dir, train_pairs=DEBUG_TRAIN_PAIRS,
                     valid_queries=DEBUG_VALID_QUERIES, candidates=DEBUG_CANDIDATES):
    """Same model and hyperparameters; only reduce data/epochs/logging for a short run."""
    if type(train_pairs) is not int or not 8 <= train_pairs <= formal["expected_counts"]["train_pairs"]:
        raise ValueError("Debug train-pair count must be between 8 and the formal train-pair count")
    for count, limit in ((valid_queries, formal["expected_counts"]["valid_queries"]),
                         (candidates, formal["expected_counts"]["candidates"])):
        if type(count) is not int or not 1 <= count <= limit:
            raise ValueError("Debug query/candidate counts must be positive and within the formal data counts")
    return dict(formal, mode="debug", output_dir=str(Path(output_dir).resolve()),
                num_train_epochs=2, logging_steps=1, debug_train_pairs=train_pairs,
                debug_valid_queries=valid_queries, debug_candidates=candidates)


def debug_run_config(formal, resume=None, train_pairs=None):
    root = Path(formal["output_dir"]) / "_debug"
    if resume:
        folder = Path(resume).resolve().parent
        if not folder.is_relative_to(root.resolve()):
            raise ValueError("--debug resume checkpoint must be inside this output's _debug directory")
        saved_path = folder / "run_config.json"
        saved = load_config(saved_path)
        count = saved.get("debug_train_pairs", 8)
        if train_pairs is not None and train_pairs != count:
            raise ValueError("Cannot change debug_train_pairs while resuming")
        profile = small_run_config(formal, folder, train_pairs=count,
                                   valid_queries=saved.get("debug_valid_queries", 4),
                                   candidates=saved.get("debug_candidates", 256))
        # 舊 debug 未存這兩欄，續訓仍須維持原 config 與 4/256 範圍。
        for key in ("debug_valid_queries", "debug_candidates"):
            if key not in saved:
                profile.pop(key)
        return profile
    else:
        folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return small_run_config(formal, folder, train_pairs=DEBUG_TRAIN_PAIRS if train_pairs is None else train_pairs)
