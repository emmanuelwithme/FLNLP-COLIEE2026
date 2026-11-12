"""參照 modernBert-fp：只比較直接子目錄 checkpoint 當下的 valid 指標。"""
import argparse
import math
from pathlib import Path

from config import load_config, read_json


def find_best_checkpoint(checkpoint_root, metric="eval_global_f1", mode="max"):
    if mode not in ("min", "max"):
        raise ValueError("mode must be min or max")
    checkpoints = sorted((p for p in Path(checkpoint_root).glob("checkpoint-*")
                          if p.is_dir() and p.name.split("-")[-1].isdigit()),
                         key=lambda p: int(p.name.split("-")[-1]))
    best, value = None, None
    for checkpoint in checkpoints:
        if not (checkpoint / "trainer_state.json").is_file():
            continue
        state = read_json(checkpoint / "trainer_state.json")
        rows = [row for row in state.get("log_history", [])
                if row.get("step") == state["global_step"] and row.get("epoch", 0) >= 1 and metric in row]
        if not rows:
            continue
        current = rows[-1][metric]
        if not math.isfinite(current):
            raise ValueError(f"Non-finite checkpoint metric: {checkpoint}")
        if value is None or (current > value if mode == "max" else current < value):
            best, value = checkpoint, current
    if best is None:
        raise ValueError(f"No trained checkpoint with {metric} in {checkpoint_root}")
    return str(best.resolve()), value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--run-dir", default=load_config()["output_dir"])
    parser.add_argument("--metric", default="eval_global_f1")
    parser.add_argument("--mode", choices=("min", "max"), default="max")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    if args.history:
        print("epoch  step  sampled_valid_loss  valid_F1  delta_vs_epoch0")
        for row in read_json(Path(args.root) / "metrics_history.json"):
            print(f"{row['epoch']:g} {row['step']:6d} {row['eval_loss']:.6f} "
                  f"{row['eval_global_f1']:.8f} {row['eval_improvement_vs_epoch0']:+.8f}")
    print(find_best_checkpoint(args.root, args.metric, args.mode))


if __name__ == "__main__":
    main()
