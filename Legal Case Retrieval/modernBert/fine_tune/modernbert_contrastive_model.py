import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Optional
from safetensors.torch import load_file as safe_load
from transformers import (
    PreTrainedModel,
    ModernBertModel,
    ModernBertConfig,
    AutoTokenizer,
    WEIGHTS_NAME,               # = "pytorch_model.bin"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.device import get_device
from repo_config import get_env_path

class ContrastiveConfig(ModernBertConfig):
    """
    继承 ModernBertConfig，不手动重写任何 backbone 的字段。
    只在此基础上新增 temperature。projector head 的维度
    将直接使用 config.hidden_size。
    """
    model_type = "modernbert"

    def __init__(
        self,
        # temperature: float = 1.0,
        **kwargs,  # 其余所有 ModernBertConfig 的字段都会被自动接收
    ):
        super().__init__(**kwargs)
        # 只新增 temperature 参数，projector_hidden_size 直接用 hidden_size
        # self.temperature = temperature


class ModernBERTContrastive(PreTrainedModel):
    """
    Contrastive 模型：backbone 用 ModernBertModel，取 CLS 向量后
    再过一个两层 Linear+ReLU 的 projector head，最后做 L2 归一化
    并配合 temperature 计算 InfoNCE loss。
    只要执行 .from_pretrained(checkpoint_dir)，就能一次性加载
    encoder + projector head 的所有权重。
    """
    config_class = ContrastiveConfig
    base_model_prefix = "encoder."  # 告訴HF模型參數權重：encoder.* 前綴對應 ModernBertModel backbone
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def __init__(self, config: ContrastiveConfig):
        super().__init__(config)

        # 1) encoder 延後初始化
        self.encoder = None

        # 2) Projection head：输入与输出维度都设为 config.hidden_size
        hidden_dim = config.hidden_size
        # 这里不用写死 768，以后只要 backbone hidden_size 改了，自动跟着改
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3) 温度系数
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # 4) 与训练时保持一致：关闭 use_cache，启用梯度检查点
        # self.encoder.config.use_cache = False
        # self.encoder.enable_input_require_grads()
        # self.encoder.gradient_checkpointing_enable()

        # 5) 初始化权重（确保 backbone + head 都初始化）
        # self.init_weights()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        encoder_kwargs: Dict = None,
        **kwargs
    ):
        # —— 1) 把用户传给 encoder 的那些 kwargs 拿出来
        hf_loading_args = ["device_map", "torch_dtype", "attn_implementation"]
        if encoder_kwargs is None:
            encoder_kwargs = {}
        for k in hf_loading_args:
            if k in kwargs:
                encoder_kwargs[k] = kwargs.pop(k)

        # —— 2) 加载 config（剩余 kwargs 都给 ContrastiveConfig）
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )

        # —— 3) 建实例
        model = cls(config)

        # —— 4) 再把 encoder + projector head + log_temperature 的权重一次性 load 进来
        # 优先找 safetensors，否则找 pytorch_model.bin
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        bin_path        = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        if os.path.exists(safetensors_path):
            # 需要安装 safetensors: pip install safetensors
            state_dict = safe_load(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"在 {pretrained_model_name_or_path} 下既没找到 model.safetensors，也没找到 {WEIGHTS_NAME}"
            )
        
        # 只挑 encoder 部分，並去掉 "encoder." 前綴
        prefix = cls.base_model_prefix
        encoder_sd = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        
        # —— 5) load backbone
        model.encoder, loading_info = ModernBertModel.from_pretrained(
            pretrained_model_name_or_path=None, # 不再從路徑讀檔
            config=config,
            state_dict=encoder_sd, # 上面剝 prefix 後的那份
            output_loading_info=True, #檢查模型權重哪些沒有正常載入被隨機初始化
            **encoder_kwargs
        )

        # 6) 查看 loading_info
        print("encoder missing keys (随机初始化的层)：")
        print(loading_info["missing_keys"])
        print("\nencoder unexpected keys (checkpoint 多余的权重)：")
        print(loading_info["unexpected_keys"])
        print("\nencoder error messages：")
        print(loading_info["error_msgs"])


        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        # 7) 打印出来
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
        """
        给定由 tokenizer 返回的 input_batch（包含 input_ids, attention_mask），
        先经过 backbone（ModernBERT），取 CLS 隐藏向量，再跑 projector，
        最后做 L2 归一化，返回形状 (batch_size, hidden_size) 的 embedding。
        """
        output = self.encoder(**input_batch)
        cls_hidden = output.last_hidden_state[:, 0, :]  # (bsz, hidden_size)
        proj_vec = self.projector(cls_hidden)           # (bsz, hidden_size)
        proj_vec = torch.nn.functional.normalize(proj_vec, p=2, dim=-1)
        return proj_vec

    def forward(
        self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Contrastive 训练时的 forward：
          - anchor_input, positive_input, negative_input: 都是 tokenizer 返回的 dict
          - labels: LongTensor，shape = (batch_size,)，通常都设为 0，表示第 0 列是正样本
        返回包含 "loss" 与 "logits" 的字典：
          - logits.shape = (batch_size, 1 + neg_count)
          - loss = CrossEntropy(logits, labels)
        """
        anchor_vec   = self.encode(anchor_input)       # (bsz, H)
        positive_vec = self.encode(positive_input)     # (bsz, H)
        negative_vec = self.encode(negative_input)     # (bsz * neg_count, H)

        bsz = anchor_vec.size(0)
        # negative_vec 的前提是形状 (bsz*neg_count, H)
        neg_count = negative_vec.size(0) // bsz
        negative_vec = negative_vec.view(bsz, neg_count, -1)  # (bsz, neg_count, H)

        # 计算正样本相似度 (bsz, 1)
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        # 计算负样本相似度 (bsz, neg_count)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)

        # 拼成 (bsz, 1+neg_count)，再除以温度
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}


# ============================================
# 以下示范：如何一行载入预先存好的完整 checkpoint
# ============================================
if __name__ == "__main__":
    device = get_device()

    checkpoint_dir = get_env_path("TASK1_LEGACY_EXAMPLE_CHECKPOINT_DIR", required=True)
    assert checkpoint_dir is not None

    # 1) 载入 tokenizer（该目录下已经包含 tokenizer.json、tokenizer_config.json）
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    # 2) 一行载入：包含 encoder + projector head
    model = ModernBERTContrastive.from_pretrained(
        str(checkpoint_dir),
        encoder_kwargs={"device_map": device, "torch_dtype": torch.float16, "attn_implementation": "flash_attention_2"},
    )
    model = model.to(device)
    model = model.half() #把projector的精度也轉成torch.float16(ModernBert backbone在from_pretrained()就指定載入是float16)
    model = model.eval()

    # 3) 演示：用 model.encode() 生成某一句话的 embedding
    text = "這是一段用來測試的文字。"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096
    ).to(device)

    with torch.no_grad():
        embedding = model.encode({"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask})
    print("Embedding shape:", embedding.shape)  # (1, hidden_size)
    # print("Embedding (L2-normalized):", embedding)
    print(f"log_temperature = {model.log_temperature.item():.6f}")
    print(f"temperature     = {model.temperature.item():.6f}")
