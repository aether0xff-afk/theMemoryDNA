from pathlib import Path

from memorydna.slow_reference import run_slow_reference


if __name__ == "__main__":
    run_slow_reference(Path("results/slow-reference"), seed=7, environment_replicates=6)
    print("Wrote results/slow-reference")
