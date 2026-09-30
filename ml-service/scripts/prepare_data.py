from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.pipeline import run_preprocessing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the cleaned PV dataset into ML-ready engineering features and chronological splits.")
    parser.add_argument("--input", default="data/processed/clean_pv_data.csv", help="Path to the cleaned raw CSV dataset.")
    parser.add_argument("--output", default="data/processed", help="Directory for ML-ready feature and split artifacts.")
    args = parser.parse_args()

    input_path = project_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    result = run_preprocessing_pipeline(input_path, output_dir)
    print("Generated preprocessing artifacts:")
    for name, value in result.items():
        print(f"- {name}: {value}")


if __name__ == "__main__":
    main()
