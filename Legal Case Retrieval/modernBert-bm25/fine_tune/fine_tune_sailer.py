"""4 distinct queries × 64 passages, label masking and memory-bounded GradCache."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_tune.fine_tune import main
from config import PROJECT_DIR

if __name__ == "__main__":
    main(default_config=PROJECT_DIR / "configs/coliee2025_sailer_gradcache.json")
