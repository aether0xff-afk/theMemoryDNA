from pathlib import Path

from memorydna.reference_memory import run_reference_benchmark


if __name__ == "__main__":
    run_reference_benchmark(Path("results/reference-memory"), seed=7, sequence_replicates=8)
    print("Wrote results/reference-memory")
