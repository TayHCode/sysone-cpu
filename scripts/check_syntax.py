"""Check repository Python syntax without creating bytecode files."""
import ast
from pathlib import Path

for folder in ("scripts", "benchmarks", "tests"):
    for path in sorted(Path(folder).glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax checked.")
