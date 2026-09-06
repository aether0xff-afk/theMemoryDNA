from pathlib import Path

from memorydna.paired_pattern import run_paired_pattern


if __name__ == "__main__":
    run_paired_pattern(Path("results/paired-pattern"), seed=7, population=160, generations=500, replicates=20)
    print("Wrote results/paired-pattern")
