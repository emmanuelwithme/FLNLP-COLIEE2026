"""取自 modernBert-fp 的 CLS/projector/cosine InfoNCE；訓練與推論共用唯一模型。"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import ModernBertConfig, ModernBertModel

from config import read_json, write_json


class ModernBERTContrastive(nn.Module):
    def __init__(self, encoder, initial_temperature=0.55555, temperature_min=0.001, temperature_max=2.0):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.projector = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature), dtype=torch.float32))
        self.temperature_min, self.temperature_max = temperature_min, temperature_max
        self.encoder.config.use_cache = False

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    @classmethod
    def from_base(cls, config):
        encoder = ModernBertModel.from_pretrained(
            config["model_name"], revision=config["model_revision"],
            local_files_only=config["local_files_only"],
            attn_implementation=config["attention_implementation"], torch_dtype=torch.float32,
            device_map={"": "cuda"})
        model = cls(encoder, config["initial_temperature"], config["temperature_min"], config["temperature_max"])
        model.enable_checkpointing(config["gradient_checkpointing"])
        return model

    def enable_checkpointing(self, enabled):
        if enabled:
            self.encoder.enable_input_require_grads()
            self.encoder.gradient_checkpointing_enable()

    def encode(self, inputs):
        cls = self.encoder(**inputs).last_hidden_state[:, 0, :]
        return nn.functional.normalize(self.projector(cls), p=2, dim=-1)

    def forward(self, anchor_input, positive_input, negative_input, labels=None):
        bsz = anchor_input["input_ids"].shape[0]
        merged = {k: torch.cat([anchor_input[k], positive_input[k], negative_input[k]], dim=0)
                  for k in ("input_ids", "attention_mask")}
        vectors = self.encode(merged)
        anchors, positives = vectors[:bsz], vectors[bsz:2 * bsz]
        negatives = vectors[2 * bsz:].reshape(bsz, -1, vectors.shape[-1])
        pos = nn.functional.cosine_similarity(anchors, positives, dim=-1).unsqueeze(1)
        neg = nn.functional.cosine_similarity(anchors.unsqueeze(1), negatives, dim=-1)
        logits = torch.cat([pos, neg], dim=1) / self.temperature
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, folder, state_dict=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(folder)
        write_json(folder / "contrastive_config.json", {"architecture": "CLS-Linear-ReLU-Linear-L2",
                   "temperature_min": self.temperature_min, "temperature_max": self.temperature_max,
                   # ModernBertConfig.to_dict() deliberately omits this flag.
                   "reference_compile": self.encoder.config.reference_compile})
        weights = self.state_dict() if state_dict is None else state_dict
        save_file({k: v.detach().cpu().contiguous() for k, v in weights.items()},
                  str(folder / "model.safetensors"), metadata={"format": "pt"})

    @classmethod
    def from_checkpoint(cls, folder, attention="flash_attention_2", legacy_config=None):
        folder = Path(folder)
        # Legacy 0911 checkpoints sometimes have no config; use original base config only when requested.
        if (folder / "config.json").exists():
            encoder_config = ModernBertConfig.from_pretrained(folder, local_files_only=True)
        elif legacy_config is not None:
            encoder_config = ModernBertConfig.from_pretrained(legacy_config["model_name"],
                revision=legacy_config["model_revision"], local_files_only=legacy_config["local_files_only"])
        else:
            raise ValueError("Missing config.json; use explicit --legacy for a historical 0911 checkpoint")
        encoder_config._attn_implementation = attention
        settings = read_json(folder / "contrastive_config.json") if (folder / "contrastive_config.json").exists() else {}
        if "reference_compile" in settings:
            encoder_config.reference_compile = settings["reference_compile"]
        model = cls(ModernBertModel(encoder_config), temperature_min=settings.get("temperature_min", 0.001),
                    temperature_max=settings.get("temperature_max", 2.0))
        path = folder / "model.safetensors"
        weights = load_file(str(path)) if path.exists() else torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        return model
