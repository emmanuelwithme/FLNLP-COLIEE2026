import os
import sys
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import (
    PreTrainedModel,
    ModernBertModel,
    ModernBertConfig,
    AutoTokenizer,
    WEIGHTS_NAME,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.device import get_device
from repo_config import get_env_path


class ContrastiveConfig(ModernBertConfig):
    """
    ModernBert 的對比式設定，僅新增溫度相關參數，其餘完全沿用 backbone 的欄位。
    """

    model_type = "modernbert"

    def __init__(
        self,
        temperature_init: float = 0.55555,
        temperature_min: float = 1e-3,
        temperature_max: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature_init = float(temperature_init)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)


class ModernBERTContrastive(PreTrainedModel):
    """
    將 modernBert-fp/fine_tune.py 中使用的 encoder + projector + 溫度參數
    打包成 PreTrainedModel，方便 Trainer / inference 以 .from_pretrained() 載入。
    """

    config_class = ContrastiveConfig
    base_model_prefix = "encoder."

    @property
    def temperature(self) -> torch.Tensor:
        temp = torch.exp(self.log_temperature)
        return temp.clamp(self.temperature_min, self.temperature_max)

    def __init__(self, config: ContrastiveConfig):
        super().__init__(config)

        # encoder 延後初始化，避免在 __init__ 阶段就占用顯存
        self.encoder: Optional[ModernBertModel] = None

        hidden_dim = config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.temperature_min = getattr(config, "temperature_min", 1e-3)
        self.temperature_max = getattr(config, "temperature_max", 2.0)
        init_temp = getattr(config, "temperature_init", 0.55555)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(init_temp)), dtype=torch.float32)
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        encoder_model_name_or_path: Optional[str] = None,
        encoder_kwargs: Dict = None,
        **kwargs,
    ):
        # 1) 擷取要傳給 encoder 的 kwargs
        hf_loading_args = ["device_map", "torch_dtype", "attn_implementation", "trust_remote_code"]
        if encoder_kwargs is None:
            encoder_kwargs = {}
        for k in hf_loading_args:
            if k in kwargs:
                encoder_kwargs[k] = kwargs.pop(k)

        # 2) 載入 config，若 checkpoint 內沒有 config，退回使用 encoder_model_name_or_path
        config = kwargs.pop("config", None)
        config_kwargs = dict(kwargs)
        last_error = None
        if config is None:
            for cfg_path in (pretrained_model_name_or_path, encoder_model_name_or_path):
                if cfg_path is None:
                    continue
                try:
                    config = cls.config_class.from_pretrained(cfg_path, **config_kwargs)
                    break
                except Exception as err:
                    last_error = err
                    continue
            if config is None:
                raise last_error if last_error is not None else ValueError("無法載入 ModernBERT config")

        # 3) 建立空模型（encoder 稍後填入）
        model = cls(config)

        # 4) 讀取 state dict（優先 safetensors）
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        bin_path = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        if os.path.exists(safetensors_path):
            state_dict = safe_load(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"在 {pretrained_model_name_or_path} 下既沒找到 model.safetensors，也沒找到 {WEIGHTS_NAME}"
            )

        # 5) 先載入 encoder 權重，再掛上 projector / log_temperature
        prefix = cls.base_model_prefix
        encoder_sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        model.encoder, loading_info = ModernBertModel.from_pretrained(
            pretrained_model_name_or_path=None,
            config=config,
            state_dict=encoder_sd,
            output_loading_info=True,
            **encoder_kwargs,
        )
        # 與 fine_tune.py 保持一致的訓練設定
        try:
            model.encoder.config.use_cache = False
            model.encoder.enable_input_require_grads()
            model.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

        # 印出 encoder 的載入狀態
        print("encoder missing keys (随机初始化的层)：")
        print(loading_info.get("missing_keys", []))
        print("\nencoder unexpected keys (checkpoint 多余的权重)：")
        print(loading_info.get("unexpected_keys", []))
        print("\nencoder error messages：")
        print(loading_info.get("error_msgs", []))

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        print("❗Missing keys (没在 checkpoint 里找到，只能随机初始化)：")
        for k in missing_keys:
            print("   ", k)
        print("✅Unexpected keys (checkpoint 里多的，模型里没用到)：")
        for k in unexpected_keys:
            print("   ", k)

        print(f"[from_pretrained] log_temperature = {model.log_temperature.item():.6f}")
        print(f"[from_pretrained] temperature     = {model.temperature.item():.6f}")
        return model

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**input_batch)
        cls_hidden = output.last_hidden_state[:, 0, :]
        proj_vec = self.projector(cls_hidden)
        proj_vec = torch.nn.functional.normalize(proj_vec, p=2, dim=-1)
        return proj_vec

    def encode_in_chunks(self, negatives_input: Dict[str, torch.Tensor], chunk_size: int) -> torch.Tensor:
        ids = negatives_input["input_ids"]
        attn = negatives_input["attention_mask"]
        all_vecs = []
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            out = self.encoder(input_ids=ids[start:end], attention_mask=attn[start:end])
            cls = out.last_hidden_state[:, 0, :]
            proj = self.projector(cls)
            all_vecs.append(torch.nn.functional.normalize(proj, p=2, dim=-1))
        return torch.cat(all_vecs, dim=0)

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        anchor_vec = self.encode(anchor_input)
        positive_vec = self.encode(positive_input)
        negative_vec = self.encode(negative_input)

        bsz = anchor_vec.size(0)
        neg_count = negative_vec.size(0) // bsz
        negative_vec = negative_vec.view(bsz, neg_count, -1)

        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================
# 以下示範：如何一行載入預先存好的完整 checkpoint
# ============================================
if __name__ == "__main__":
    device = get_device()

    checkpoint_dir = get_env_path("TASK1_FP_EXAMPLE_CHECKPOINT_DIR", required=True)
    base_encoder_dir = get_env_path("TASK1_BASE_ENCODER_DIR", required=True)
    assert checkpoint_dir is not None
    assert base_encoder_dir is not None

    # 1) 先載 tokenizer（checkpoint 內已有 tokenizer 檔）
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)

    # 2) 一行載入：包含 encoder + projector head
    model = ModernBERTContrastive.from_pretrained(
        str(checkpoint_dir),
        encoder_model_name_or_path=str(base_encoder_dir),
        encoder_kwargs={
            "device_map": device,
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        },
    )
    model = model.to(device)
    model = model.half()  # projector 也轉成 fp16
    model = model.eval()

    # 3) 演示：用 model.encode() 生成某一句话的 embedding
    text = "這是一段用來測試的文字。"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        embedding = model.encode({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask})
    print("Embedding shape:", embedding.shape)  # (1, hidden_size)
    print(f"log_temperature = {model.log_temperature.item():.6f}")
    print(f"temperature     = {model.temperature.item():.6f}")
