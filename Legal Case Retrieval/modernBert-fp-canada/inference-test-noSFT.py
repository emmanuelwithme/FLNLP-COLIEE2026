from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import get_env_path

CKPT_DIR = get_env_path("TASK1_CANADA_BASE_ENCODER_DIR", required=True)
assert CKPT_DIR is not None
MAX_LENGTH = 4096

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    dtype = torch.float32
    print("使用 CPU")

if not CKPT_DIR.exists():
    raise FileNotFoundError(f"找不到 continued pretraining checkpoint: {CKPT_DIR}")

# 初始化 tokenizer / model（使用續訓的 ModernBERT checkpoint）
tokenizer = AutoTokenizer.from_pretrained(
    str(CKPT_DIR),
    model_max_length=MAX_LENGTH,
    use_fast=True,
    trust_remote_code=True,
)
# 避免長文本警告，實際長度在 encode 時控制
tokenizer.model_max_length = 1_000_000_000

model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True, "device_map": {"": str(device)}}
if device.type == "cuda":
    model_kwargs["attn_implementation"] = "flash_attention_2"

model = AutoModel.from_pretrained(str(CKPT_DIR), **model_kwargs)
model = model.eval()

def get_embeddings(texts, batch_size: int = 8):
    """Generate CLS embeddings for a list of texts in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # 取 CLS 表示，轉成 float32 方便後續在 CPU 做相似度計算
            embeddings = outputs.last_hidden_state[:, 0, :].float()
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# 準備輸入句子（示例）
sentences = [
    "The defendant stole a vehicle and drove away.",
    "The Court of Appeal held that the defendant was guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft.",
    "The defendant was found guilty of theft by the Court of Appeal.",
    "The Court of Appeal found the defendant guilty of theft.",
]

print(f"Generating embeddings for {len(sentences)} sentences using {CKPT_DIR} ...")
sent_embeddings = get_embeddings(sentences)
print(f"embeddings shape: {sent_embeddings.shape}")

# 計算並比較每對句子的相似度
print("Similarity Comparison:")
for i in range(len(sent_embeddings)):
    for j in range(i + 1, len(sent_embeddings)):
        similarity = cosine_similarity(sent_embeddings[i].unsqueeze(0), sent_embeddings[j].unsqueeze(0))
        print(f"Sentence {i} vs Sentence {j}:")
        print(f"{sentences[i]}")
        print(f"{sentences[j]}")
        print(f"Cosine Similarity: {similarity.item():.4f}")
        if similarity > 0.8:
            print("Result: Highly Similar")
        elif similarity > 0.6:
            print("Result: Moderately Similar")
        else:
            print("Result: Not Very Similar")
        print()
