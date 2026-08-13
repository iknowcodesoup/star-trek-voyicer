import sys
from pathlib import Path

# src/ modules import each other flatly ("from resample import ..."), the same
# way main.py sets them up. Tests need the same path for those imports to work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
