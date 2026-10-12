"""Configuration inspection only. Inference belongs to Phase 8."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from zaid_lm.config import load_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a proposed configuration; no training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        overrides = {
            key: value
            for key, value in {"device": args.device, "seed": args.seed}.items()
            if value is not None
        }
        config = replace(config, **overrides)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "proposal_only", "config": config.to_dict()}, indent=2))
    return 0
