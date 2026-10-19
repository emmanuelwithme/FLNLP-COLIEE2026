from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load
from transformers import ModernBertModel

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env_utils import load_chunkagg_v6_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_v6_dotenv(__file__)

MAX_DOCUMENT_CHUNKS = max(1, int(os.getenv("TASK1_CHUNKAGG_V6_MAX_DOCUMENT_CHUNKS", "3")))
DOCUMENT_CHUNK_LENGTH = max(8, int(os.getenv("TASK1_CHUNKAGG_V6_DOCUMENT_CHUNK_LENGTH", "4096")))
CHUNK_MICROBATCH_SIZE = max(1, int(os.getenv("TASK1_CHUNKAGG_V6_CHUNK_MICROBATCH_SIZE", "1")))


class ChunkFusionBlock(nn.Module):
    """One pre-norm transformer block over ``[DOC] + chunk vectors``."""

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
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed_states = self.attn_norm(hidden_states)
        attn_output, attn_weights = self.self_attn(
            normed_states,
            normed_states,
            normed_states,
            key_padding_mask=padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden_states = residual + self.attn_dropout(attn_output)

        residual = hidden_states
        hidden_states = residual + self.ffn(self.ffn_norm(hidden_states))
        if return_attention:
            return hidden_states, attn_weights
        return hidden_states


class ResidualProjector(nn.Module):
    """Preserve the continued-pretraining representation at initialization."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.fc2(F.gelu(self.fc1(hidden_states)))
        return self.output_norm(hidden_states + residual)


class ModernBERTContrastive(nn.Module):
    """ModernBERT document retriever with end-to-end chunk aggregation.

    The stable base representation is the masked mean of individually normalized
    projected chunk vectors. A four-token fusion transformer learns a residual
    correction. The residual projection starts at zero, so random fusion
    parameters cannot overwrite chunk evidence at initialization.
    """

    # v6 intentionally keeps the v5 network/loss but uses a new checkpoint
    # contract so no earlier experiment can be resumed accidentally.
    ARCHITECTURE_VERSION = 6

    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        temperature: float = 0.55555,
        max_chunks: int = MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size: int = CHUNK_MICROBATCH_SIZE,
        fusion_dropout: float = 0.1,
        temperature_min: float = 0.05,
        temperature_max: float = 2.0,
        evidence_loss_weight: float = 0.2,
        evidence_pool_temperature: float = 0.1,
        fusion_residual_max_norm: float = 0.5,
        fusion_residual_penalty_weight: float = 0.01,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be > 0, got {max_chunks}")
        if not 0 < temperature_min < temperature_max:
            raise ValueError(
                "temperature bounds must satisfy 0 < min < max, got "
                f"{temperature_min} and {temperature_max}"
            )
        if not temperature_min <= temperature <= temperature_max:
            raise ValueError(
                "initial temperature must be inside the configured bounds, got "
                f"temperature={temperature}, bounds=[{temperature_min}, {temperature_max}]"
            )
        if evidence_loss_weight < 0:
            raise ValueError("evidence_loss_weight must be >= 0")
        if evidence_pool_temperature <= 0:
            raise ValueError("evidence_pool_temperature must be > 0")
        if fusion_residual_max_norm <= 0:
            raise ValueError("fusion_residual_max_norm must be > 0")
        if fusion_residual_penalty_weight < 0:
            raise ValueError("fusion_residual_penalty_weight must be >= 0")

        device = torch.device(device)
        encoder_kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }
        if device.type == "cuda":
            encoder_kwargs["device_map"] = {"": str(device)}
            if importlib.util.find_spec("flash_attn") is not None:
                encoder_kwargs["attn_implementation"] = "flash_attention_2"

        self.encoder = ModernBertModel.from_pretrained(model_name, **encoder_kwargs)
        if device.type != "cuda":
            self.encoder.to(device=device, dtype=torch.float32)
        self.encoder.config.use_cache = False

        hidden_dim = int(self.encoder.config.hidden_size)
        num_heads = int(getattr(self.encoder.config, "num_attention_heads", 12))
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden size {hidden_dim} must be divisible by attention heads {num_heads}"
            )

        self.hidden_dim = hidden_dim
        self.max_chunks = int(max_chunks)
        self.chunk_microbatch_size = max(1, int(chunk_microbatch_size))
        self.supports_chunked_documents = True

        self.projector = ResidualProjector(hidden_dim)
        self.doc_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.chunk_pos_emb = nn.Parameter(torch.zeros(1, self.max_chunks + 1, hidden_dim))
        self.chunk_fusion = ChunkFusionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=fusion_dropout,
        )
        self.fusion_output_norm = nn.LayerNorm(hidden_dim)
        self.fusion_output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(temperature)), dtype=torch.float32)
        )
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.evidence_loss_weight = float(evidence_loss_weight)
        self.evidence_pool_temperature = float(evidence_pool_temperature)
        self.fusion_residual_penalty_weight = float(fusion_residual_penalty_weight)
        self.register_buffer(
            "fusion_residual_max_norm",
            torch.tensor(float(fusion_residual_max_norm), dtype=torch.float32),
        )
        self.register_buffer(
            "chunkagg_architecture_version",
            torch.tensor(self.ARCHITECTURE_VERSION, dtype=torch.int64),
        )

        self._init_chunk_aggregation_parameters()
        self.encoder.enable_input_require_grads()
        self.encoder.gradient_checkpointing_enable()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)

    @torch.no_grad()
    def clamp_temperature_parameter_(self) -> None:
        """Clamp the parameter itself so gradients can move it away from a bound."""
        self.log_temperature.clamp_(
            min=math.log(self.temperature_min),
            max=math.log(self.temperature_max),
        )

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def _init_chunk_aggregation_parameters(self) -> None:
        # [DOC] is a learned bias on top of the document mean, not a standalone
        # content-free token that can become a shortcut.
        nn.init.zeros_(self.doc_token)
        nn.init.normal_(self.chunk_pos_emb, mean=0.0, std=0.02)
        for module in self.chunk_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.ones_(self.fusion_output_norm.weight)
        nn.init.zeros_(self.fusion_output_norm.bias)
        nn.init.zeros_(self.fusion_output.weight)

    def _soft_cap_fusion_residual(
        self,
        raw_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Smoothly keep the residual norm below the configured maximum.

        ``r / (1 + ||r|| / M)`` is identity-like around zero, so the
        zero-initialized output projection receives gradients immediately, and
        its resulting norm is always strictly below ``M``.
        """
        raw_norm = raw_residual.float().norm(dim=-1, keepdim=True)
        max_norm = self.fusion_residual_max_norm.to(
            device=raw_residual.device,
            dtype=raw_norm.dtype,
        )
        scale = (1.0 + raw_norm / max_norm).reciprocal().to(raw_residual.dtype)
        return raw_residual * scale, raw_norm.squeeze(-1)

    @staticmethod
    def _assert_nonempty_training_documents(
        name: str,
        input_batch: Dict[str, torch.Tensor],
    ) -> None:
        attention_mask = input_batch["attention_mask"]
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)
        valid_chunks = attention_mask.bool().any(dim=-1)
        chunk_mask = input_batch.get("chunk_mask")
        if chunk_mask is not None:
            if chunk_mask.dim() == 1:
                chunk_mask = chunk_mask.unsqueeze(1)
            valid_chunks = valid_chunks & chunk_mask.to(
                device=valid_chunks.device,
                dtype=torch.bool,
            )
        invalid_rows = (~valid_chunks.any(dim=-1)).nonzero(as_tuple=False).flatten()
        if invalid_rows.numel() > 0:
            raise ValueError(
                f"{name} contains documents with no valid chunks at rows "
                f"{invalid_rows.detach().cpu().tolist()}"
            )

    def chunk_aggregation_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield "doc_token", self.doc_token
        yield "chunk_pos_emb", self.chunk_pos_emb
        for name, parameter in self.chunk_fusion.named_parameters():
            yield f"chunk_fusion.{name}", parameter
        for name, parameter in self.fusion_output_norm.named_parameters():
            yield f"fusion_output_norm.{name}", parameter
        for name, parameter in self.fusion_output.named_parameters():
            yield f"fusion_output.{name}", parameter

    def added_parameter_count(self) -> int:
        return sum(parameter.numel() for _, parameter in self.chunk_aggregation_named_parameters())

    def load_checkpoint(self, checkpoint_dir: str | Path) -> "ModernBERTContrastive":
        checkpoint_dir = Path(checkpoint_dir)
        safetensors_path = checkpoint_dir / "model.safetensors"
        bin_path = checkpoint_dir / "pytorch_model.bin"
        if safetensors_path.exists():
            state_dict = safe_load(str(safetensors_path), device="cpu")
        elif bin_path.exists():
            try:
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found under {checkpoint_dir}"
            )

        saved_version = state_dict.get("chunkagg_architecture_version")
        if saved_version is None or int(saved_version.item()) != self.ARCHITECTURE_VERSION:
            found = "legacy/unversioned" if saved_version is None else int(saved_version.item())
            raise RuntimeError(
                "Incompatible chunkAgg checkpoint architecture: "
                f"checkpoint={found}, expected={self.ARCHITECTURE_VERSION}. "
                "This refactored model must be trained from a new output directory."
            )

        self.load_state_dict(state_dict, strict=True)
        return self

    def _normalize_document_batch(
        self,
        input_batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        chunk_mask = input_batch.get("chunk_mask")

        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            chunk_mask = (
                attention_mask.any(dim=-1).long()
                if chunk_mask is None
                else chunk_mask.reshape(-1, 1)
            )
        elif input_ids.dim() == 3:
            if chunk_mask is None:
                chunk_mask = attention_mask.any(dim=-1).long()
        else:
            raise ValueError(f"input_ids must be rank 2 or 3, got shape {tuple(input_ids.shape)}")

        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have the same shape, got "
                f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}"
            )
        if input_ids.size(1) > self.max_chunks:
            raise ValueError(
                f"batch has {input_ids.size(1)} chunks but model max_chunks={self.max_chunks}"
            )
        if chunk_mask.shape != input_ids.shape[:2]:
            raise ValueError(
                f"chunk_mask must have shape {tuple(input_ids.shape[:2])}, got {tuple(chunk_mask.shape)}"
            )

        valid_token_mask = attention_mask.any(dim=-1)
        chunk_mask = chunk_mask.to(device=input_ids.device, dtype=torch.bool) & valid_token_mask
        return input_ids, attention_mask, chunk_mask

    def _encode_projected_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_count, seq_len = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * chunk_count, seq_len)
        flat_attention_mask = attention_mask.reshape(batch_size * chunk_count, seq_len)
        valid_indices = chunk_mask.reshape(-1).nonzero(as_tuple=False).flatten()

        projector_parameter = next(self.projector.parameters())
        flat_embeddings = projector_parameter.new_zeros(
            batch_size * chunk_count,
            self.hidden_dim,
        )
        for start in range(0, valid_indices.numel(), self.chunk_microbatch_size):
            batch_indices = valid_indices[start : start + self.chunk_microbatch_size]
            encoder_outputs = self.encoder(
                input_ids=flat_input_ids.index_select(0, batch_indices),
                attention_mask=flat_attention_mask.index_select(0, batch_indices),
            )
            projected_vectors = self.projector(encoder_outputs.last_hidden_state[:, 0, :])
            projected_vectors = projected_vectors.to(dtype=flat_embeddings.dtype)
            flat_embeddings.index_copy_(0, batch_indices, projected_vectors)

        return flat_embeddings.view(batch_size, chunk_count, self.hidden_dim)

    def _encode_document_outputs(
        self,
        input_batch: Dict[str, torch.Tensor],
        *,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, chunk_mask = self._normalize_document_batch(input_batch)
        chunk_vectors = self._encode_projected_chunks(input_ids, attention_mask, chunk_mask)
        chunk_vectors = F.normalize(chunk_vectors, p=2, dim=-1)
        chunk_vectors = chunk_vectors * chunk_mask.unsqueeze(-1).to(chunk_vectors.dtype)

        chunk_count = chunk_vectors.size(1)
        denominator = chunk_mask.sum(dim=1, keepdim=True).clamp_min(1).to(chunk_vectors.dtype)
        mean_chunk = chunk_vectors.sum(dim=1) / denominator
        has_chunks = chunk_mask.any(dim=1, keepdim=True)
        empty_fallback = F.normalize(self.doc_token[:, 0, :], p=2, dim=-1).expand_as(mean_chunk)
        normalized_mean = F.normalize(mean_chunk, p=2, dim=-1)
        base_vector = torch.where(has_chunks, normalized_mean, empty_fallback)

        first_chunk = chunk_vectors[:, 0, :]
        first_vector = torch.where(
            chunk_mask[:, :1],
            first_chunk,
            empty_fallback,
        )

        batch_size = input_ids.size(0)
        doc_tokens = base_vector.unsqueeze(1) + self.doc_token
        fusion_inputs = torch.cat([doc_tokens, chunk_vectors], dim=1)
        fusion_inputs = fusion_inputs + self.chunk_pos_emb[:, : chunk_count + 1, :]
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device),
                ~chunk_mask,
            ],
            dim=1,
        )
        fusion_result = self.chunk_fusion(
            fusion_inputs,
            padding_mask=padding_mask,
            return_attention=return_attention,
        )
        if return_attention:
            fused_states, attention_weights = fusion_result
        else:
            fused_states = fusion_result
            attention_weights = None
        fused_doc = fused_states[:, 0, :]
        raw_fusion_residual = self.fusion_output(self.fusion_output_norm(fused_doc))
        fusion_residual, raw_fusion_residual_norm = self._soft_cap_fusion_residual(
            raw_fusion_residual
        )
        document_vector = F.normalize(base_vector + fusion_residual, p=2, dim=-1)
        fusion_residual_norm = fusion_residual.float().norm(dim=-1)
        residual_cap = self.fusion_residual_max_norm.to(
            device=fusion_residual_norm.device,
            dtype=fusion_residual_norm.dtype,
        )

        outputs = {
            "fused": document_vector,
            "mean": base_vector,
            "first": F.normalize(first_vector, p=2, dim=-1),
            "chunks": chunk_vectors,
            "chunk_mask": chunk_mask,
            "chunk_mean_pre_norm": mean_chunk.norm(dim=-1),
            "fusion_raw_residual_norm": raw_fusion_residual_norm,
            "fusion_residual_norm": fusion_residual_norm,
            "fusion_residual_cap_utilization": fusion_residual_norm / residual_cap,
            "base_fused_cosine": (base_vector.float() * document_vector.float()).sum(dim=-1),
        }
        if attention_weights is not None:
            doc_attention = attention_weights[:, :, 0, :].float()
            probabilities = doc_attention.clamp_min(1e-12)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1)
            valid_token_count = (~padding_mask).sum(dim=-1).float()
            max_entropy = valid_token_count.log().clamp_min(1e-12).unsqueeze(1)
            normalized_entropy = torch.where(
                valid_token_count.unsqueeze(1) > 1,
                entropy / max_entropy,
                torch.zeros_like(entropy),
            )
            outputs["fusion_attention_entropy"] = normalized_entropy.mean(dim=1)
            outputs["fusion_attention_self_mass"] = doc_attention[:, :, 0].mean(dim=1)
            outputs["fusion_attention_chunk_mass"] = doc_attention[:, :, 1:].sum(dim=-1).mean(dim=1)
        return outputs

    def encode_document(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode_document_outputs(input_batch)["fused"]

    def encode_variants(self, input_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self._encode_document_outputs(input_batch)
        return {name: outputs[name] for name in ("fused", "mean", "first")}

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_document(input_batch)

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        negative_counts: Optional[torch.Tensor] = None,
        cached_negative_vectors: Optional[torch.Tensor] = None,
        cached_negative_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        self._assert_nonempty_training_documents("anchor_input", anchor_input)
        self._assert_nonempty_training_documents("positive_input", positive_input)
        self._assert_nonempty_training_documents("negative_input", negative_input)
        batch_size = anchor_input["input_ids"].size(0)
        positive_total = positive_input["input_ids"].size(0)
        negative_total = negative_input["input_ids"].size(0)
        if positive_total != batch_size:
            raise ValueError(
                "single-positive InfoNCE requires exactly one positive per query: "
                f"queries={batch_size}, positives={positive_total}"
            )

        if negative_counts is None:
            if batch_size == 0 or negative_total % batch_size != 0:
                raise ValueError(
                    "negative_counts is required when negatives cannot be split evenly per query"
                )
            negative_counts_list = [negative_total // batch_size] * batch_size
        else:
            negative_counts_list = [int(value) for value in negative_counts.detach().cpu().tolist()]
        if len(negative_counts_list) != batch_size:
            raise ValueError(
                f"negative_counts has {len(negative_counts_list)} rows for batch size {batch_size}"
            )
        if any(count <= 0 for count in negative_counts_list):
            raise ValueError(f"every query needs at least one negative: {negative_counts_list}")
        if sum(negative_counts_list) != negative_total:
            raise ValueError(
                f"negative_counts sum to {sum(negative_counts_list)}, but batch has {negative_total} negatives"
            )

        merged_batch = {
            key: torch.cat(
                [anchor_input[key], positive_input[key], negative_input[key]],
                dim=0,
            )
            for key in ("input_ids", "attention_mask", "chunk_mask")
        }
        encoded = self._encode_document_outputs(merged_batch, return_attention=True)
        all_vectors = encoded["fused"]
        anchor_vectors = all_vectors[:batch_size]
        positive_vectors = all_vectors[batch_size : 2 * batch_size]
        negative_vectors = all_vectors[2 * batch_size :]

        all_chunks = encoded["chunks"]
        all_chunk_masks = encoded["chunk_mask"]
        positive_chunks = all_chunks[batch_size : 2 * batch_size]
        positive_chunk_masks = all_chunk_masks[batch_size : 2 * batch_size]
        negative_chunks = all_chunks[2 * batch_size :]
        negative_chunk_masks = all_chunk_masks[2 * batch_size :]

        if cached_negative_vectors is not None:
            if cached_negative_vectors.dim() != 3:
                raise ValueError("cached_negative_vectors must have shape [B, K, D]")
            if cached_negative_vectors.shape[0] != batch_size:
                raise ValueError("cached negative batch size does not match anchor batch")
            if cached_negative_vectors.shape[2] != self.hidden_dim:
                raise ValueError("cached negative embedding dimension does not match model")
            if cached_negative_mask is None:
                cached_negative_mask = torch.ones(
                    cached_negative_vectors.shape[:2],
                    dtype=torch.bool,
                    device=cached_negative_vectors.device,
                )
            else:
                cached_negative_mask = cached_negative_mask.to(
                    device=cached_negative_vectors.device,
                    dtype=torch.bool,
                )

        temperature = self.temperature
        rows = []
        evidence_rows = []
        positive_scores = []
        hardest_negative_scores = []
        positive_evidence_scores = []
        evidence_dilution = []
        negative_cursor = 0
        for row_index, negative_count in enumerate(negative_counts_list):
            row_anchor = anchor_vectors[row_index]
            positive_score = torch.dot(row_anchor, positive_vectors[row_index]).unsqueeze(0)
            row_negatives = negative_vectors[
                negative_cursor : negative_cursor + negative_count
            ]
            row_negative_chunks = negative_chunks[
                negative_cursor : negative_cursor + negative_count
            ]
            row_negative_chunk_masks = negative_chunk_masks[
                negative_cursor : negative_cursor + negative_count
            ]
            negative_cursor += negative_count
            negative_scores = torch.matmul(row_negatives, row_anchor)
            if cached_negative_vectors is not None:
                row_cache_mask = cached_negative_mask[row_index]
                row_cached = cached_negative_vectors[row_index][row_cache_mask]
                if row_cached.numel() > 0:
                    cached_scores = torch.matmul(row_cached, row_anchor)
                    negative_scores = torch.cat([negative_scores, cached_scores], dim=0)
            rows.append(torch.cat([positive_score, negative_scores], dim=0) / temperature)

            def smooth_chunk_score(
                chunks: torch.Tensor,
                masks: torch.Tensor,
                fallback_vectors: torch.Tensor,
            ) -> torch.Tensor:
                raw_scores = torch.matmul(chunks, row_anchor)
                scaled_scores = raw_scores / self.evidence_pool_temperature
                scaled_scores = scaled_scores.masked_fill(~masks, -1e4)
                chunk_counts = masks.sum(dim=-1).clamp_min(1).to(raw_scores.dtype)
                pooled = self.evidence_pool_temperature * (
                    torch.logsumexp(scaled_scores, dim=-1) - chunk_counts.log()
                )
                fallback_scores = torch.matmul(fallback_vectors, row_anchor)
                return torch.where(masks.any(dim=-1), pooled, fallback_scores)

            pos_evidence = smooth_chunk_score(
                positive_chunks[row_index : row_index + 1],
                positive_chunk_masks[row_index : row_index + 1],
                positive_vectors[row_index : row_index + 1],
            )
            neg_evidence = smooth_chunk_score(
                row_negative_chunks,
                row_negative_chunk_masks,
                row_negatives,
            )
            evidence_rows.append(
                torch.cat([pos_evidence, neg_evidence], dim=0) / temperature
            )

            positive_chunk_raw = torch.matmul(positive_chunks[row_index], row_anchor)
            positive_chunk_raw = positive_chunk_raw.masked_fill(
                ~positive_chunk_masks[row_index],
                -1e4,
            )
            max_positive_chunk = torch.where(
                positive_chunk_masks[row_index].any(),
                positive_chunk_raw.max(),
                positive_score.squeeze(0),
            )
            positive_scores.append(positive_score.squeeze(0))
            hardest_negative_scores.append(negative_scores.max())
            positive_evidence_scores.append(pos_evidence.squeeze(0))
            evidence_dilution.append(max_positive_chunk - positive_score.squeeze(0))

        max_width = max(row.numel() for row in rows)
        logits = rows[0].new_full((batch_size, max_width), -1e4)
        for row_index, row in enumerate(rows):
            logits[row_index, : row.numel()] = row

        if labels is None:
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        else:
            labels = labels.to(device=logits.device, dtype=torch.long)
        document_loss = F.cross_entropy(logits, labels)

        max_evidence_width = max(row.numel() for row in evidence_rows)
        evidence_logits = evidence_rows[0].new_full(
            (batch_size, max_evidence_width),
            -1e4,
        )
        for row_index, row in enumerate(evidence_rows):
            evidence_logits[row_index, : row.numel()] = row
        evidence_loss = F.cross_entropy(evidence_logits, labels)
        max_residual_norm = self.fusion_residual_max_norm.to(
            device=encoded["fusion_raw_residual_norm"].device,
            dtype=encoded["fusion_raw_residual_norm"].dtype,
        )
        fusion_residual_excess = F.relu(
            encoded["fusion_raw_residual_norm"] - max_residual_norm
        )
        fusion_residual_penalty = fusion_residual_excess.square().mean()
        loss = (
            document_loss
            + self.evidence_loss_weight * evidence_loss
            + self.fusion_residual_penalty_weight * fusion_residual_penalty
        )

        diagnostics = {
            "loss_document": document_loss.detach(),
            "loss_evidence": evidence_loss.detach(),
            "loss_fusion_residual_penalty": fusion_residual_penalty.detach(),
            "positive_score": torch.stack(positive_scores).mean().detach(),
            "hardest_negative_score": torch.stack(hardest_negative_scores).mean().detach(),
            "score_margin": (
                torch.stack(positive_scores) - torch.stack(hardest_negative_scores)
            ).mean().detach(),
            "positive_evidence_score": torch.stack(positive_evidence_scores).mean().detach(),
            "evidence_dilution_index": torch.stack(evidence_dilution).mean().detach(),
            "chunk_mean_pre_norm": encoded["chunk_mean_pre_norm"].mean().detach(),
            "fusion_raw_residual_norm": encoded["fusion_raw_residual_norm"].mean().detach(),
            "fusion_residual_norm": encoded["fusion_residual_norm"].mean().detach(),
            "fusion_residual_cap_utilization": encoded[
                "fusion_residual_cap_utilization"
            ].mean().detach(),
            "base_fused_cosine": encoded["base_fused_cosine"].mean().detach(),
            "fusion_attention_entropy": encoded["fusion_attention_entropy"].mean().detach(),
            "fusion_attention_self_mass": encoded[
                "fusion_attention_self_mass"
            ].mean().detach(),
            "fusion_attention_chunk_mass": encoded[
                "fusion_attention_chunk_mass"
            ].mean().detach(),
            "cached_negative_count": torch.tensor(
                0.0
                if cached_negative_mask is None
                else float(cached_negative_mask.sum(dim=1).float().mean().item()),
                device=loss.device,
            ),
        }
        return {
            "loss": loss,
            "logits": logits,
            "_diagnostics": diagnostics,
            "_cache_positive_vectors": positive_vectors.detach(),
        }
