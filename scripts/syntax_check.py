from __future__ import annotations

import compileall
from pathlib import Path

root = Path(__file__).resolve().parents[1]
ok = compileall.compile_dir(root / "experiments", quiet=1)
if not ok:
    raise SystemExit("Python syntax check failed.")
print("All experiment scripts passed Python syntax compilation.")
