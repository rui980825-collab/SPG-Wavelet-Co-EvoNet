import argparse
from pathlib import Path
import yaml

from coevonet.engine.train_generator import train_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/coevonet.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_generator(cfg, cfg_path)


if __name__ == "__main__":
    main()
