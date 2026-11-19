import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the micromobility injury severity models and run the manuscript comparisons."
    )
    parser.add_argument(
        "workbooks",
        type=Path,
        help="Directory containing cyclist2022.xlsx through cyclist2024.xlsx and escooter2022.xlsx through escooter2024.xlsx",
    )
    folder = parser.parse_args().workbooks.resolve()
    code = Path(__file__).resolve().parent
    subprocess.run([sys.executable, code / "prepare_data.py", folder], check=True)
    for script in ("fit_models.py", "statistical_tests.py", "cross_mode_test.py"):
        subprocess.run([sys.executable, code / script], check=True)


if __name__ == "__main__":
    main()
