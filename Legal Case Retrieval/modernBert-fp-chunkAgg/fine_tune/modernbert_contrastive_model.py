from __future__ import annotations

import os
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import ModernBertModel

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env_utils import load_chunkagg_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_dotenv(__file__)

# 中文註解：TASK1_MAX_DOCUMENT_CHUNKS 控制文件編碼時最多保留幾個 chunk。
MAX_DOCUMENT_CHUNKS = max(1, int(os.getenv("TASK1_MAX_DOCUMENT_CHUNKS", "3")))
# 中文註解：TASK1_DOCUMENT_CHUNK_LENGTH 控制每個 chunk 的 token 上限，含 special tokens。
DOCUMENT_CHUNK_LENGTH = max(8, int(os.getenv("TASK1_DOCUMENT_CHUNK_LENGTH", "4096")))
# 中文註解：TASK1_CHUNK_MICROBATCH_SIZE 控制 encoder 每次前向送幾個 chunk，避免 OOM。
CHUNK_MICROBATCH_SIZE = max(1, int(os.getenv("TASK1_CHUNK_MICROBATCH_SIZE", "1")))


class ChunkFusionBlock(nn.Module):
    """中文註解：用 1 層 pre-norm transformer block 融合 1~3 個 chunk 向量。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        normed_states = self.attn_norm(hidden_states)
        attn_output, _ = self.self_attn(
            normed_states,
            normed_states,
            normed_states,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = residual + self.attn_dropout(attn_output)

        residual = hidden_states
        normed_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(normed_states)
        return hidden_states


class ModernBERTContrastive(nn.Module):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        temperature: float = 0.55555,
        max_chunks: int = MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size: int = CHUNK_MICROBATCH_SIZE,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        dtype = torch.float32
        encoder_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "device_map": {"": str(device)},
        }
        if device.type == "cuda":
            encoder_kwargs["attn_implementation"] = "flash_attention_2"
        self.encoder = ModernBertModel.from_pretrained(model_name, **encoder_kwargs)
        self.encoder.to(device=device, dtype=dtype)
        self.encoder.config.use_cache = False

        hidden_dim = self.encoder.config.hidden_size
        self.hidden_dim = hidden_dim
        self.max_chunks = max_chunks
        self.chunk_microbatch_size = max(1, int(chunk_microbatch_size))
        self.supports_chunked_documents = True

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 中文註解：聚合 1~3 個 chunk 的 learnable [DOC] token 與位置向量。
        self.doc_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.chunk_pos_emb = nn.Parameter(torch.zeros(1, max_chunks + 1, hidden_dim))
        num_heads = getattr(self.encoder.config, "num_attention_heads", 12)
        self.chunk_fusion = ChunkFusionBlock(hidden_dim, num_heads=num_heads, dropout=fusion_dropout)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(temperature)), dtype=torch.float32)
        )
        self.temperature_min = 1e-3
        self.temperature_max = 2.0

        self._init_chunk_agg_parameters()

    def _init_chunk_agg_parameters(self):
        nn.init.normal_(self.doc_token, mean=0.0, std=0.02)
        nn.init.normal_(self.chunk_pos_emb, mean=0.0, std=0.02)
        for module in self.chunk_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def load_checkpoint(self, checkpoint_dir: str | Path):
        checkpoint_dir = Path(checkpoint_dir)
        safetensors_path = checkpoint_dir / "model.safetensors"
        bin_path = checkpoint_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            state_dict = safe_load(str(safetensors_path), device="cpu")
        elif bin_path.exists():
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"在 {checkpoint_dir} 下找不到 model.safetensors 或 pytorch_model.bin"
            )

        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print("⚠️ Missing keys:", missing_keys)
        if unexpected_keys:
            print("⚠️ Unexpected keys:", unexpected_keys)
        return self

    def _encode_projected_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_count, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * chunk_count, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * chunk_count, seq_len)
        flat_chunk_mask = chunk_mask.view(batch_size * chunk_count).bool() & flat_attention_mask.any(dim=1)
        valid_indices = flat_chunk_mask.nonzero(as_tuple=False).flatten()

        flat_embeddings = self.projector[0].weight.new_zeros(
            batch_size * chunk_count,
            self.hidden_dim,
        )
        if valid_indices.numel() == 0:
            return flat_embeddings.view(batch_size, chunk_count, self.hidden_dim)

        for start in range(0, valid_indices.numel(), self.chunk_microbatch_size):
            batch_indices = valid_indices[start : start + self.chunk_microbatch_size]
            encoder_outputs = self.encoder(
                input_ids=flat_input_ids.index_select(0, batch_indices),
                attention_mask=flat_attention_mask.index_select(0, batch_indices),
            )
            cls_vectors = encoder_outputs.last_hidden_state[:, 0, :]
            projected_vectors = self.projector(cls_vectors).to(dtype=flat_embeddings.dtype)
            flat_embeddings.index_copy_(0, batch_indices, projected_vectors)

        return flat_embeddings.view(batch_size, chunk_count, self.hidden_dim)

    def encode_document(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        chunk_mask = input_batch.get("chunk_mask")

        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            if chunk_mask is None:
                chunk_mask = attention_mask.any(dim=-1).long()
            else:
                chunk_mask = chunk_mask.view(-1, 1)
        elif input_ids.dim() == 3:
            if chunk_mask is None:
                chunk_mask = attention_mask.any(dim=-1).long()
        else:
            raise ValueError(f"Unsupported input_ids ndim: {input_ids.dim()}")

        chunk_mask = (
            chunk_mask.to(device=input_ids.device, dtype=torch.long)
            * attention_mask.any(dim=-1).to(device=input_ids.device, dtype=torch.long)
        )
        chunk_vectors = self._encode_projected_chunks(input_ids, attention_mask, chunk_mask)

        batch_size, chunk_count, _ = chunk_vectors.shape
        doc_tokens = self.doc_token.expand(batch_size, -1, -1)
        fusion_inputs = torch.cat([doc_tokens, chunk_vectors], dim=1)
        fusion_inputs = fusion_inputs + self.chunk_pos_emb[:, : chunk_count + 1, :]
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device),
                ~chunk_mask.bool(),
            ],
            dim=1,
        )
        fused_states = self.chunk_fusion(fusion_inputs, padding_mask=padding_mask)
        document_vectors = fused_states[:, 0, :]
        return torch.nn.functional.normalize(document_vectors, p=2, dim=-1)

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_document(input_batch)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Inference loader only; use encode()/encode_document().")
