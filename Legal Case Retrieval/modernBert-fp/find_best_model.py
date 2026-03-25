#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

def find_best_checkpoint(checkpoint_root: str, metric: str, mode: str) -> str:
    """
    遍歷 checkpoint_root 底下所有以 "checkpoint-" 開頭的子資料夾，
    在每個子資料夾的 trainer_state.json 裡，只比對「該 checkpoint 當下 global_step 所記錄的 eval 指標值」，
    最後選出指定 metric 表現最好的那個 checkpoint 資料夾。

    參數：
    - checkpoint_root: str, 存放所有 checkpoint 子資料夾 (例如 "./models/modernBERT_contrastive")
    - metric: str, 要比較的指標名稱 (trainer_state.json 的 log_history 裡的 key)，
              例如 "eval_loss", "eval_acc1", "eval_acc5"
    - mode: str, 指定該 metric 是否「越小越好」（"min"）或「越大越好」（"max"）

    回傳：
    - best_checkpoint: str, 最佳 checkpoint 所在的資料夾路徑
    """
    if mode not in ("min", "max"):
        raise ValueError("mode 只能是 'min' 或 'max'")

    best_checkpoint = None
    best_value = None
    greater_is_better = (mode == "max")

    for folder in os.listdir(checkpoint_root):
        if not folder.startswith("checkpoint-"):
            continue

        ckpt_dir = os.path.join(checkpoint_root, folder)
        state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue

        # 讀取該 checkpoint 底下的 trainer_state.json
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            # 若檔案讀取失敗就跳過
            continue

        # 先取得當下這個 checkpoint 的 global_step
        cur_step = state.get("global_step")
        if cur_step is None:
            # 沒有 global_step 就跳過
            continue

        # 在 log_history 裡找到 step == cur_step 的那筆記錄，並取出 metric
        value = None
        for record in state.get("log_history", []):
            if record.get("step") == cur_step and metric in record:
                value = record[metric]
                break

        if value is None:
            # 如果在這個 checkpoint 裡面，找不到 step == cur_step 且有該 metric 的 entry，就跳過
            # （比如這個 checkpoint 只存了 train-loss 而沒有 eval 指標）
            continue

        # 跟全域 best_value 比較
        if best_value is None:
            best_value = value
            best_checkpoint = ckpt_dir
        else:
            if greater_is_better:
                if value > best_value:
                    best_value = value
                    best_checkpoint = ckpt_dir
            else:
                if value < best_value:
                    best_value = value
                    best_checkpoint = ckpt_dir

    if best_checkpoint is None:
        raise ValueError(f"在路徑 {checkpoint_root} 底下，找不到任何包含「{metric}」的 checkpoint。")

    return best_checkpoint, best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="從多個 checkpoint 資料夾中，僅以該 checkpoint 對應的 global_step 所記錄的 eval 指標來比較，挑出最好的 checkpoint 路徑"
    )
    parser.add_argument(
        "--root", "-r",
        type=str,
        required=True,
        help="checkpoint 的主目錄 (例如 './models/modernBERT_contrastive')"
    )
    parser.add_argument(
        "--metric", "-m",
        type=str,
        required=True,
        help="要比較的指標名稱 (如 'eval_loss', 'eval_acc1', 'eval_acc5')"
    )
    parser.add_argument(
        "--mode", "-o",
        type=str,
        choices=["min", "max"],
        required=True,
        help="指定該指標是『越小越好』(min) 還是『越大越好』(max)"
    )
    args = parser.parse_args()

    best_ckpt, best_value = find_best_checkpoint(args.root, args.metric, args.mode)
    print(f"best_ckpt: {best_ckpt}, best_value: {best_value}")
