"""GROMACS validation export (CLI wrapper)."""

from __future__ import annotations

import sys
from pathlib import Path

from amphiquantic.validation.gromacs_export import export_all


if __name__ == "__main__":
    aqtop = Path(sys.argv[1])
    out = Path(sys.argv[2])
    paths = export_all(aqtop, out)
    for name, path in paths.items():
        print(f"Wrote {path} ({name})")
