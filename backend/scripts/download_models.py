"""Pre-download Hugging Face weights and warm up ModelRegistry (run from backend/)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.model_registry import ModelRegistry

if __name__ == "__main__":
    status = ModelRegistry.get_instance().load_all()
    for name, msg in status.items():
        print(f"  {name}: {msg}")
