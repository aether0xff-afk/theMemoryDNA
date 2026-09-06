from pathlib import Path
import sys

# Allow `python run_experiments.py` directly from a fresh clone. An editable
# install is still recommended for development/tests.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from memorydna.experiments import main


if __name__ == "__main__":
    main()
