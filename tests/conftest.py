import sys
from pathlib import Path

# Make the repo root and tests dir importable without installation
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
