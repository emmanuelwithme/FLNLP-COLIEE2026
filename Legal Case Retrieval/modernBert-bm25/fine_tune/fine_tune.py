"""ModernBERT BM25 微調入口：依 modernBert-fp 的 Dataset → Trainer → 評估流程。"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer import DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import ProgressCallback
from tqdm.auto import tqdm

from config import (DEFAULT_CONFIG, DEBUG_TRAIN_PAIRS, IMPLEMENTATION, PROJECT_DIR, debug_run_config, implementation_id,
                    REPO_ROOT, digest, load_config, read_json, write_json)
from fine_tune.data import (ContrastiveCollator, ContrastiveDataset, prepare,
                            sample_epoch, select_debug_data, sha256_file)
from fine_tune.modernbert_contrastive_model import ModernBERTContrastive
from similarity_and_rank import evaluate_model_retrieval

# True：真實 ModernBERT-base、少量資料、連續兩輪。指令 --debug/--no-debug 優先。
DEBUG = False


class ConsoleProgressCallback(ProgressCallback):
    """一個總訓練 steps 進度條，標示 epoch；評估另顯示 batch/文件進度。"""
    @staticmethod
    def training_description(args, state):
        total = math.ceil(args.num_train_epochs)
        return f"Training | epoch {min(int(state.epoch or 0) + 1, total)}/{total}"

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, initial=state.global_step,
                desc=self.training_description(args, state), unit="step", dynamic_ncols=True, disable=False)
        self.current_step = state.global_step

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.set_description(self.training_description(args, state))

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and eval_dataloader is not None:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader),
                    desc=f"Epoch {int(state.epoch or 0)} | Sampled valid loss", unit="batch",
                    position=1 if self.training_bar is not None else 0,
                    leave=self.training_bar is None, dynamic_ncols=True, disable=False)
            self.prediction_bar.update(1)

    def close_prediction_bar(self):
        if self.prediction_bar is not None:
            self.prediction_bar.close()
            self.prediction_bar = None

    def close(self):
        self.close_prediction_bar()
        if self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None


class EpochSamplingCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        # persistent_workers=False：新 iterator 的 workers 會取得更新後 rows。
        self.trainer.resample_epoch(int(state.epoch or 0))

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        order = lambda row: (row["query_id"], row["positive_id"])
        expected = sorted(trainer.train_dataset.rows, key=order)
        received = sorted(trainer.epoch_seen_samples, key=order)
        if received != expected:
            raise ValueError("Training did not receive exactly this epoch's resampled pairs/negatives")
        write_json(trainer.run_dir / "sampling" / f"epoch_{int(state.epoch):02d}_received.json",
                   {"rows": len(received), "sample_hash": digest(received), "matches_planned": True})


class TrainingMonitor(TrainerCallback):
    """只觀察 AMP 結果；不檢查／修改梯度，不取代 Trainer 的 scaler.step。"""
    def __init__(self, trainer):
        self.trainer = trainer
        self.attempts = self.updates = self.skipped = self.consecutive = 0
        self.scale_before = None

    def snapshot(self):
        return {"attempts": self.attempts, "updates": self.updates,
                "skipped": self.skipped, "consecutive": self.consecutive}

    def restore(self, path):
        saved = read_json(path)
        if (set(saved) != set(self.snapshot()) or any(type(n) is not int or n < 0 for n in saved.values())
                or saved["attempts"] != saved["updates"] + saved["skipped"]
                or saved["consecutive"] > saved["skipped"]):
            raise ValueError("Invalid AMP monitor state")
        for name, value in saved.items():
            setattr(self, name, value)

    @staticmethod
    def check_weights(model):
        checks = [torch.isfinite(p.detach()).all() for p in model.parameters()]
        if not torch.stack(checks).all():
            raise FloatingPointError("Non-finite model weights")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.attempts != state.global_step:
            raise ValueError("AMP monitor does not match checkpoint global_step")
        self.check_weights(model)

    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer.step_samples = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        scaler = self.trainer.accelerator.scaler
        self.scale_before = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else None

    def on_optimizer_step(self, args, state, control, **kwargs):
        scaler = self.trainer.accelerator.scaler
        after = float(scaler.get_scale()) if self.scale_before is not None else None
        self.attempts += 1
        # fused AdamW 可在 kernel 內跳過更新；不能只相信 step_was_skipped。
        if after is not None and after < self.scale_before:
            self.skipped += 1
            self.consecutive += 1
            tqdm.write(f"[AMP] skipped update {state.global_step + 1}: scale {self.scale_before} -> {after}")
            self.check_weights(kwargs["model"])
            if self.consecutive >= 8:
                raise FloatingPointError("8 consecutive AMP overflows; inspect loss/data before restarting")
        else:
            self.updates += 1
            self.consecutive = 0
        self.trainer.monitor_metrics = {**getattr(self.trainer, "monitor_metrics", {}),
                                        "amp/scale": after or 1.0, "amp/skipped_updates": self.skipped,
                                        "amp/updates_applied": self.updates}

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.check_weights(model)


class BM25NegativeSamplingTrainer(Trainer):
    """沿用 HF 訓練迴圈；只擴充抽樣、optimizer groups、評估及完整模型保存。"""
    def __init__(self, *args, config, bundle, **kwargs):
        self.config, self.bundle = config, bundle
        self.run_dir = Path(config["output_dir"])
        super().__init__(*args, **kwargs)
        self.remove_callback(DEFAULT_PROGRESS_CALLBACK)
        self.progress = ConsoleProgressCallback()
        self.monitor = TrainingMonitor(self)
        self.add_callback(self.progress)
        self.add_callback(EpochSamplingCallback(self))
        self.add_callback(self.monitor)

    def log(self, logs, start_time=None):
        if "loss" in logs:
            logs = {**logs, **getattr(self, "monitor_metrics", {}),
                    "model/temperature": float(self.model.temperature.detach())}
            if self.optimizer is not None:
                logs.update({f"optimizer/lr_{i}": group["lr"] for i, group in enumerate(self.optimizer.param_groups)})
        return super().log(logs, start_time)

    def resample_epoch(self, epoch):
        rows = sample_epoch(self.bundle["train_labels"], self.bundle["train_pools"], epoch,
                            self.config["sampler_seed"], pairs=self.bundle.get("pairs"))
        path = self.run_dir / "sampling" / f"epoch_{epoch + 1:02d}_train.json"
        if path.exists() and read_json(path) != rows:
            raise ValueError(f"Stored epoch samples disagree: {path}")
        write_json(path, rows)
        self.train_dataset.rows = rows
        self.epoch_seen_samples = []
        current, cumulative = {}, {}
        for row in rows:
            current.setdefault(row["query_id"], set()).update(row["negative_ids"])
        for previous in range(epoch + 1):
            for row in read_json(self.run_dir / "sampling" / f"epoch_{previous + 1:02d}_train.json"):
                cumulative.setdefault(row["query_id"], set()).update(row["negative_ids"])
        denominator = sum(len(self.bundle["train_pools"][q]) for q in current)
        metrics = {"sampling/epoch_pool_coverage": sum(map(len, current.values())) / denominator,
                   "sampling/cumulative_pool_coverage": sum(map(len, cumulative.values())) / denominator,
                   "sampling/self_negative_count": 0, "sampling/positive_leak_count": 0}
        write_json(path.with_suffix(".audit.json"), {"epoch": epoch + 1, "rows": len(rows),
                                                   "sample_hash": digest(rows), **metrics})
        self.log(metrics)

    def create_optimizer(self):
        if self.optimizer is None:
            # 與 modernBert-fp 相同的三組 AdamW 參數與 temperature 專屬 LR。
            decay, no_decay, temperature = [], [], []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name == "log_temperature":
                    temperature.append(parameter)
                elif any(part in name for part in ("bias", "LayerNorm.weight", "layer_norm.weight")):
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
            self.optimizer = torch.optim.AdamW([
                {"params": decay, "lr": self.config["learning_rate"], "weight_decay": self.config["weight_decay"]},
                {"params": no_decay, "lr": self.config["learning_rate"], "weight_decay": 0.0},
                {"params": temperature, "lr": self.config["temperature_lr"], "weight_decay": 0.0}],
                betas=(0.9, 0.999), eps=1e-8, fused=True)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        samples = inputs.pop("sample_ids", [])
        if model.training:
            self.step_samples = getattr(self, "step_samples", []) + samples
            if not hasattr(self, "epoch_seen_samples"):
                self.epoch_seen_samples = []
            self.epoch_seen_samples.extend(samples)
            self.observed_max_length = max(getattr(self, "observed_max_length", 0),
                                          inputs["anchor_input"]["input_ids"].shape[-1])
        else:
            self.last_eval_samples = samples
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite forward loss; step={self.state.global_step}; samples={samples}")
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        try:
            output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        finally:
            self.progress.close_prediction_bar()
        epoch = float(self.state.epoch or 0)
        metrics = evaluate_model_retrieval(self.model, self.processing_class, self.config,
            self.bundle["valid_labels"], self.bundle.get("candidate_ids", self.bundle["manifest"]["candidate_ids"]),
            self.run_dir / "rankings" / f"epoch_{int(epoch):02d}",
            progress_position=1 if self.progress.training_bar is not None else 0)
        for key in ("f1", "precision", "recall", "tp", "fp", "fn", "num_queries", "candidate_count", "missing_predictions"):
            output.metrics[f"{metric_key_prefix}_global_{key}"] = metrics[key]
        path = self.run_dir / "metrics_history.json"
        history = read_json(path) if path.exists() else []
        baseline = next((row["eval_global_f1"] for row in history if row["epoch"] == 0), metrics["f1"])
        output.metrics[f"{metric_key_prefix}_improvement_vs_epoch0"] = metrics["f1"] - baseline
        history = [row for row in history if row["epoch"] != epoch]
        history.append({"epoch": epoch, "step": self.state.global_step, "mode": self.config["mode"], **output.metrics})
        write_json(path, sorted(history, key=lambda row: row["epoch"]))
        tqdm.write(f"Epoch {epoch:g} valid F1@5={metrics['f1']:.8f}; queries={metrics['num_queries']}")
        # 在 on_evaluate 之前加入 F1，讓 early stopping 與 best checkpoint 使用同一值。
        return output

    def _save(self, output_dir=None, state_dict=None):
        folder = Path(output_dir or self.args.output_dir)
        self.model.save_pretrained(folder, state_dict)
        self.processing_class.save_pretrained(folder)
        torch.save(self.args, folder / "training_args.bin")
        write_json(folder / "amp_monitor.json", self.monitor.snapshot())


def validate_resume(config, manifest, checkpoint):
    root, checkpoint = Path(config["output_dir"]), Path(checkpoint).resolve()
    if not (root / "run_config.json").is_file():
        raise ValueError("Resume run_config.json is missing; use the checkpoint's original config/output directory")
    if read_json(root / "run_config.json") != config:
        raise ValueError("Resume configuration changed")
    original = read_json(root / "run_manifest.json")
    implementation = implementation_id(config)
    if original.get("implementation") != implementation:
        raise ValueError("Resume requires a checkpoint from this refactored implementation")
    if original["data_hash"] != manifest["data_hash"]:
        raise ValueError("Resume data hash mismatch")
    if checkpoint.parent != root.resolve():
        raise ValueError("Resume checkpoint must be a direct child of output_dir")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth",
                "model.safetensors", "config.json", "contrastive_config.json", "tokenizer_config.json", "amp_monitor.json")
    if config.get("training_method") == "sailer_gradcache":
        required += ("gradcache_state.json",)
    for name in required:
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete checkpoint: missing {name}")
    if config.get("training_method") == "sailer_gradcache":
        cached = read_json(checkpoint / "gradcache_state.json")
        if cached.get("batch_grouping", "legacy_degree_v1") != config.get("batch_grouping", "legacy_degree_v1"):
            raise ValueError("Resume batch grouping strategy mismatch")
    state = read_json(checkpoint / "trainer_state.json")
    if float(state["epoch"]) != int(state["epoch"]):
        raise ValueError("Resume requires an epoch-boundary checkpoint")
    if state["epoch"] >= config["num_train_epochs"]:
        raise ValueError("This run already completed its configured epochs")
    if any(int(p.name.split("-")[-1]) > state["global_step"] for p in root.glob("checkpoint-*")
           if p.is_dir() and p.name.split("-")[-1].isdigit()):
        raise ValueError("Later checkpoints exist; explicitly resume the latest one")
    monitor = TrainingMonitor(None)
    monitor.restore(checkpoint / "amp_monitor.json")
    if monitor.attempts != state["global_step"]:
        raise ValueError("AMP state/global_step mismatch")
    if not any(row["epoch"] == 0 for row in read_json(root / "metrics_history.json")):
        raise ValueError("Missing original epoch0 history")
    return checkpoint


def environment_info():
    return {"versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "flash-attn", "tensorboard")},
            "gpu": torch.cuda.get_device_name(),
            "reference_source_sha256": {name: sha256_file(REPO_ROOT / "Legal Case Retrieval/modernBert-fp" / name)
                for name in ("fine_tune/fine_tune.py", "fine_tune/modernbert_contrastive_model.py")
                if (REPO_ROOT / "Legal Case Retrieval/modernBert-fp" / name).is_file()},
            "python_source_sha256": {str(p.relative_to(PROJECT_DIR)): sha256_file(p)
                                     for p in sorted(PROJECT_DIR.rglob("*.py"))}}


def run_training(config, resume=None, callbacks=()):
    """正式與 debug 唯一訓練流程。callbacks 僅供整合測試模擬中斷。"""
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use one CUDA GPU with FP16 and Flash Attention 2")
    root = Path(config["output_dir"])
    if not resume and ((root / "run_config.json").exists() or list(root.glob("checkpoint-*"))
                       or list((root / "tb").glob("**/events.out.tfevents.*"))):
        raise ValueError("Output contains a prior run: choose a new output-dir or explicit resume")
    bundle = prepare(config, write=False)
    sailer = config.get("training_method") == "sailer_gradcache"
    implementation = implementation_id(config)
    if resume:
        resume = validate_resume(config, bundle["manifest"], resume)
    else:
        prepare(config)
        write_json(root / "run_config.json", config)
        write_json(root / "run_manifest.json", {"implementation": implementation,
            "data_hash": bundle["manifest"]["data_hash"], "environment": environment_info(),
            "model_name": config["model_name"], "model_revision": config["model_revision"],
            "note": "Current 2025 snapshot; self negatives corrected; not a resampling-only historical ablation."})
    if config["mode"] == "debug":
        bundle = select_debug_data(config, bundle)
        if sailer:
            from fine_tune.sailer import debug_pairs
            bundle["pairs"] = debug_pairs(bundle["train_labels"], config["debug_train_pairs"])
    write_json(root / "evaluation_protocol.json", {"mode": config["mode"],
        "valid_query_ids": sorted(bundle["valid_labels"]),
        "candidate_ids": bundle.get("candidate_ids", bundle["manifest"]["candidate_ids"]),
        "train_pairs": bundle.get("pairs", [(q, p) for q in sorted(bundle["train_labels"]) for p in bundle["train_labels"][q]])})
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["model_revision"],
                                             local_files_only=config["local_files_only"])
    tokenizer.model_max_length = config["max_length"]
    label_manifest = None
    if sailer:
        from fine_tune.sailer import (SailerGradCacheTrainer, SailerCollator, LogicalBatchDataset,
                                     logical_epoch_rows, prepare_label_mask)
        # Fingerprints always use the complete corpus and both original label
        # splits, even when debug selects only a few evaluation queries.
        label_manifest = prepare_label_mask(config, prepare(config, write=False), tokenizer)
        if resume:
            cached_state = read_json(resume / "gradcache_state.json")
            if cached_state["label_mask_hash"] != label_manifest["hash"]:
                raise ValueError("Resume label-mask manifest changed")
    model = (ModernBERTContrastive.from_checkpoint(resume, config["attention_implementation"])
             if resume else ModernBERTContrastive.from_base(config))
    if resume:
        model.enable_checkpointing(config["gradient_checkpointing"])
    if sailer:
        # This installed ModernBERT/torch.compile combination uses numerically
        # different no-grad and grad forward paths (real-base gradient test).
        # Disable only the optional reference compiler, retaining FlashAttn 2
        # and gradient checkpointing, so cached/replayed representations match.
        model.encoder.config.reference_compile = False
    model.to("cuda", dtype=torch.float32)
    epoch = int(read_json(resume / "trainer_state.json")["epoch"]) if resume else 0
    rows = sample_epoch(bundle["train_labels"], bundle["train_pools"], epoch,
                        config["sampler_seed"], pairs=bundle.get("pairs"))
    if sailer:
        rows, _ = logical_epoch_rows(bundle["train_labels"], bundle["train_pools"], epoch,
                                     config["sampler_seed"], bundle.get("pairs"),
                                     config.get("batch_grouping", "legacy_degree_v1"))
    fixed_valid = sample_epoch(bundle["valid_labels"], bundle["valid_pools"], -1, config["sampler_seed"])
    write_json(root / "fixed_valid_used.json", fixed_valid)
    args = TrainingArguments(output_dir=str(root), logging_dir=str(root / "tb"),
        dataloader_num_workers=config["dataloader_num_workers"], dataloader_persistent_workers=False,
        per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"], fp16=True, bf16=False, tf32=False,
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        num_train_epochs=config["num_train_epochs"], warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear", optim="adamw_torch_fused", logging_steps=config["logging_steps"],
        logging_first_step=True, eval_strategy="epoch", save_strategy="epoch", eval_on_start=False,
        save_total_limit=config["save_total_limit"], load_best_model_at_end=True,
        metric_for_best_model="eval_global_f1", greater_is_better=True, remove_unused_columns=False,
        report_to=["tensorboard"], prediction_loss_only=True, disable_tqdm=False,
        seed=config["seed"], data_seed=config["seed"], label_names=["labels"],
        restore_callback_states_from_checkpoint=True)
    trainer_class = SailerGradCacheTrainer if sailer else BM25NegativeSamplingTrainer
    train_dataset = (LogicalBatchDataset if sailer else ContrastiveDataset)(rows, config["document_dir"])
    collator = SailerCollator(tokenizer, config, label_manifest) if sailer else ContrastiveCollator(tokenizer, config["max_length"])
    trainer = trainer_class(model=model, args=args, processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=ContrastiveDataset(fixed_valid, config["document_dir"]),
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"],
                                        early_stopping_threshold=0.0), *callbacks], config=config, bundle=bundle,
        **({"label_manifest": label_manifest} if sailer else {}))
    if resume:
        trainer.monitor.restore(resume / "amp_monitor.json")
    try:
        if not resume:
            tqdm.write("Epoch 0: sampled valid loss -> corpus encoding -> full-valid F1 (diagnostic only).")
            trainer.state.epoch = 0.0
            trainer.evaluate()
            if trainer.state.best_metric is not None:
                raise AssertionError("Epoch0 must not establish best_metric")
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        history = read_json(root / "metrics_history.json")
        write_json(root / "summary.json", {"mode": config["mode"], "implementation": implementation,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_valid_f1": trainer.state.best_metric, "last_epoch": trainer.state.epoch,
            "global_step": trainer.state.global_step, "epoch0_f1": history[0]["eval_global_f1"],
            "improvement_vs_epoch0": trainer.state.best_metric - history[0]["eval_global_f1"],
            "observed_max_length": getattr(trainer, "observed_max_length", None),
            "amp": trainer.monitor.snapshot(), "data_hash": bundle["manifest"]["data_hash"]})
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error),
            "step": trainer.state.global_step, "epoch": trainer.state.epoch,
            "train_samples": getattr(trainer, "step_samples", []),
            "valid_samples": getattr(trainer, "last_eval_samples", []), "traceback": traceback.format_exc()})
        raise
    finally:
        trainer.progress.close()
    return read_json(root / "summary.json")


def main(argv=None, debug=DEBUG, default_config=DEFAULT_CONFIG):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=debug)
    parser.add_argument("--debug-train-pairs", type=int, help=f"Debug pairs, default {DEBUG_TRAIN_PAIRS} (maximum 5421)")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.output_dir)
    if config["mode"] != "formal":
        parser.error("Use configs/coliee2025.json with --debug or --no-debug, not a saved debug config")
    if args.debug:
        config = debug_run_config(config, args.resume_from_checkpoint, args.debug_train_pairs)
    elif args.debug_train_pairs is not None:
        parser.error("--debug-train-pairs requires --debug")
    print(f"DEBUG={args.debug}; output={config['output_dir']}", flush=True)
    if config.get("training_method") == "sailer_gradcache":
        print(f"SAILER-style: 4 distinct queries x 64 passages; label mask; GradCache chunk={config['gradcache_chunk_size']} documents; all positive pairs.", flush=True)
        print(f"Efficiency: input deduplication={config.get('deduplicate_inputs', False)}; gradient checkpointing={config['gradient_checkpointing']}", flush=True)
        print(f"Batch grouping: {config.get('batch_grouping', 'legacy_degree_v1')}", flush=True)
    print(f"Debug: real ModernBERT-base, {config['debug_train_pairs']} train pairs / "
          f"{config.get('debug_valid_queries', 4)} valid queries / "
          f"{config.get('debug_candidates', 256)} candidates; same training flow."
          if args.debug else "Formal: 1341 train queries / 336 valid queries / 7349 candidates.", flush=True)
    return run_training(config, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
