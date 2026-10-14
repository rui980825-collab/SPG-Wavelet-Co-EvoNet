"""Path-only launcher for the preserved source entrypoints."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", help="Released Python entrypoint, relative to repository root")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entry = (ROOT / args.entrypoint).resolve()
    if not entry.is_relative_to(ROOT) or not entry.is_file() or entry.suffix != ".py":
        parser.error("entrypoint must be a Python file inside this release")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry)] + args.arguments
    runpy.run_path(str(entry), run_name="__main__")

if __name__ == "__main__":
    main()
