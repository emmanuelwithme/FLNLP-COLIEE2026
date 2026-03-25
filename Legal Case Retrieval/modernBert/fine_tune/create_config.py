#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
from modernbert_contrastive_model import ModernBERTContrastive, ContrastiveConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import get_env_path

def convert_all_checkpoints(root_dir: str):
    """
    遍历 root_dir 下所有以 "checkpoint-" 开头的子文件夹，
    对每个 checkpoint 目录执行以下操作：
      1) 用 ContrastiveConfig.from_pretrained 读取旧版本的 config
      2) 用 ModernBERTContrastive.from_pretrained 加载模型权重（backbone + projector）
      3) 保存为 Hugging Face 标准格式：model.save_pretrained & tokenizer.save_pretrained
    """
    for folder in os.listdir(root_dir):
        if not folder.startswith("checkpoint-"):
            continue

        ckpt_dir = os.path.join(root_dir, folder)
        if not os.path.isdir(ckpt_dir):
            continue

        print(f"\n>>> 处理目录: {ckpt_dir}")

        # 1) 读回旧的 config；如果那里本身就是 HF 格式，也会直接加载
        try:
            config = ContrastiveConfig.from_pretrained(ckpt_dir)
        except Exception as e:
            print(f"  ❌ 无法从 {ckpt_dir} 加载 ContrastiveConfig: {e}")
            continue

        # 2) 用 config 加载模型权重（encoder + projector head）
        try:
            model = ModernBERTContrastive.from_pretrained(ckpt_dir, config=config)
        except Exception as e:
            print(f"  ❌ 无法从 {ckpt_dir} 加载 ModernBERTContrastive: {e}")
            continue

        # 3) 保存为 HF 标准格式
        try:
            model.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"  ❌ 保存模型到 {ckpt_dir} 失败: {e}")
            continue

        # 4) 也把 tokenizer.json/tokenizer_config.json 保证保存到该目录
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"  ❌ 保存 tokenizer 到 {ckpt_dir} 失败: {e}")
            continue

        print(f"  ✅ 已成功转换并保存：{ckpt_dir}")


if __name__ == "__main__":
    root_dir = get_env_path("TASK1_LEGACY_CONVERT_ROOT_DIR", required=True)
    assert root_dir is not None
    convert_all_checkpoints(str(root_dir))
# 这样 old_ckpt 文件夹下就会有：
#   ├ config.json         ← 包含所有 ModernBERT-base + projector_hidden_size + temperature
#   ├ pytorch_model.bin   ← encoder + projector 权重
#   ├ tokenizer.json
#   └ tokenizer_config.json
