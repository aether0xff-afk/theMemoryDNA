from __future__ import annotations

import argparse
from pathlib import Path

from memorydna.memory_study import PRESETS, run_component


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dual epigenetic-memory study")
    parser.add_argument("--component", choices=("anchor", "fine", "patterns", "switch"), required=True)
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_component(args.component, args.output, args.preset, args.seed)
    print(f"Wrote {args.component} results to {args.output}")


if __name__ == "__main__":
    main()
