"""Change only input/output paths in a copy of an existing synthesis YAML."""
import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    for flag in ("generator-ckpt", "proto-bank-path", "patch-bank-root", "background-pool-root", "out-root"):
        parser.add_argument("--" + flag)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve() or output.exists():
        parser.error("output must be a new file, distinct from the original config")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    for key in ("generator_ckpt", "proto_bank_path", "patch_bank_root", "background_pool_root", "out_root"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

if __name__ == "__main__":
    main()
